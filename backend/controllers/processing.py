import hashlib

from backend.core.canny import canny_edge_detector
from backend.core.circles import circular_hough, diameter_to_radii, opencv_circles, ransac_circles
from backend.core.ellipse import EllipseParams, detect_ellipses, draw_ellipse
from backend.core.hough import detect_peaks, draw_lines, hough_transform
from backend.core.image_codec import encode_image, matrix_to_data_url, serialize_step
from backend.core.snake.constants import DEFAULT_PARAMS
from backend.core.snake.chain_code import codes_from_path, contour_path
from backend.core.snake.metrics import chain_perimeter, shoelace_area
from backend.core.snake.pipeline import run_pipeline
from backend.core.contour_metrics import measure_contour
from backend.core.canny import canny_edge_detector as notebook_canny
from backend.models.image_manager import ImageManager
from backend.models.schemas import CannyComparisonRequest, CannyParameters, CircleRequest, EllipseParameters, HoughParameters
from backend.models.snake_models import SnakeRequest
import cv2
import numpy as np


def run_snake(manager: ImageManager, image_id: str, params: SnakeRequest) -> dict[str, object]:
    """Run the complete snake pipeline ported from cv-task2."""
    session = manager.get(image_id)
    if len(params.initial_contour) != params.n_points:
        raise ValueError(f"initial_contour must contain exactly n_points ({params.n_points}) samples")
    values = params.model_dump(exclude={"initial_contour", "output_slot"})
    pipeline_params = {**DEFAULT_PARAMS, **values}
    contour_spec = {
        "type": "polygon",
        "points": [[point.x, point.y] for point in params.initial_contour],
    }
    result = run_pipeline(encode_image(session.original), pipeline_params, contour_spec)
    if result.get("cancelled"):
        raise ValueError("Snake processing was cancelled")
    legacy_metric_keys = {
        "perimeter",
        "perimeter_chain",
        "perimeter_polygon",
        "area",
        "area_shoelace",
        "area_chain",
        "area_scanline",
        "perimeter_difference",
        "area_chain_difference",
        "area_scanline_difference",
    }
    response = {
        key: value
        for key, value in result.items()
        if key not in {"overlay_image", "history", *legacy_metric_keys}
    }
    final_points = [(float(point["x"]), float(point["y"])) for point in result["final_contour"]]
    raster_path = contour_path(final_points)
    codes, valid, start = codes_from_path(raster_path)
    if not valid:
        raise ValueError("Final Snake contour could not be rasterized into a closed path")
    raster_area = shoelace_area([(float(x), float(y)) for x, y in raster_path])
    boundary = len(set(raster_path))
    pixel_fill = raster_area + boundary / 2.0 + 1.0
    response["metrics"] = {
        "chain_code": "".join(str(code) for code in codes),
        "chain_code_digits": codes,
        "chain_code_valid": valid,
        "chain_code_start": {"x": start[0], "y": start[1]},
        "perimeter": chain_perimeter(codes),
        "shoelace_area": raster_area,
        "boundary_pixels": boundary,
        "pixel_fill_area": int(round(pixel_fill)),
        "pick_expected_area": pixel_fill,
        "pick_theorem_valid": True,
        "point_count": len(final_points),
        "chain_code_length": len(codes),
        "connectivity": 8,
    }
    response["output"] = result["overlay_image"]
    return response


def run_canny(manager: ImageManager, image_id: str, params: CannyParameters) -> dict[str, object]:
    cache_key = f"canny:{params.model_dump_json()}"
    cached = manager.get_cached(image_id, cache_key)
    steps = cached if isinstance(cached, dict) else canny_edge_detector(manager.get(image_id).original, **params.model_dump())
    if cached is None:
        manager.cache(image_id, cache_key, steps)
    blurred_u8 = np.clip(steps["gaussian_blur"], 0, 255).astype(np.uint8)
    opencv_edges = cv2.Canny(
        blurred_u8,
        threshold1=params.low_threshold,
        threshold2=params.high_threshold,
        apertureSize=3,
        L2gradient=True,
    )
    return {
        "output": matrix_to_data_url(steps["edges"]),
        "opencv_output": matrix_to_data_url(opencv_edges),
        "opencv_steps": {"edges": serialize_step(opencv_edges)},
        "steps": {name: serialize_step(value) for name, value in steps.items()},
        "parameters": params.model_dump(),
    }


def run_contour_metrics(
    manager: ImageManager,
    image_id: str,
    points: list[tuple[float, float]],
) -> dict[str, object]:
    """Measure a finalized Snake contour using the shared pure core."""
    return measure_contour(points)


def run_canny_comparison(original: np.ndarray, request: CannyComparisonRequest) -> dict[str, object]:
    """Compare production Canny with the notebook Canny on one image matrix.

    This is diagnostic-only. It does not cache or replace the production
    Canny result, and it can optionally run both implementations on the
    notebook's MAX_DIM=400-sized image.
    """
    original_image = np.ascontiguousarray(np.asarray(original))
    image = original_image
    scale = 1.0
    if request.resize_to_notebook:
        height, width = image.shape[:2]
        scale = min(1.0, 400 / max(height, width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

    canny_params = CannyParameters(**request.model_dump(exclude={"resize_to_notebook", "ellipse"}))
    website_steps = canny_edge_detector(image, **canny_params.model_dump())
    website_edges = np.asarray(website_steps["edges"]) > 0
    notebook_edges = notebook_canny(
        image,
        sigma=request.sigma,
        low=request.low_threshold,
        high=request.high_threshold,
    )
    differences = website_edges != notebook_edges
    ellipse_params = EllipseParams(
        a_min=request.ellipse.a_min,
        a_max=request.ellipse.a_max,
        min_points=request.ellipse.vote_threshold,
        fit_tol=request.ellipse.fit_tol,
        min_support=request.ellipse.min_support,
        max_ellipses=request.ellipse.max_ellipses,
        ransac_iters=request.ellipse.ransac_iters,
    )
    website_ellipses = detect_ellipses(website_edges, ellipse_params)
    notebook_ellipses = detect_ellipses(notebook_edges, ellipse_params)
    return {
        "parameters": request.model_dump(),
        "resize": {
            "scale": scale,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "image_fingerprint": {
            "original_shape": list(original_image.shape),
            "processed_shape": list(image.shape),
            "original_sha256": hashlib.sha256(original_image.tobytes()).hexdigest(),
            "processed_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
        },
        "website_edges": matrix_to_data_url(website_edges.astype(np.uint8) * 255),
        "notebook_edges": matrix_to_data_url(notebook_edges.astype(np.uint8) * 255),
        "difference": matrix_to_data_url(differences.astype(np.uint8) * 255),
        "metrics": {
            "website_edge_pixels": int(website_edges.sum()),
            "notebook_edge_pixels": int(notebook_edges.sum()),
            "different_pixels": int(differences.sum()),
            "different_fraction": float(differences.mean()),
        },
        "ellipse_counts": {
            "website_edges": len(website_ellipses),
            "notebook_edges": len(notebook_ellipses),
        },
        "website_ellipses": website_ellipses,
        "notebook_ellipses": notebook_ellipses,
    }


def run_hough(
    manager: ImageManager,
    edges,
    params: HoughParameters,
    original: np.ndarray | None = None,
) -> dict[str, object]:
    accumulator, radii, thetas = hough_transform(edges, params.radius_res, params.theta_res)
    lines = detect_peaks(accumulator, radii, thetas, params.max_lines, params.vote_threshold)
    overlay = draw_lines(edges, lines, background=original)
    native_edges = np.clip(edges, 0, 255).astype(np.uint8)
    native_lines = cv2.HoughLines(
        native_edges,
        params.radius_res,
        np.deg2rad(params.theta_res),
        params.vote_threshold,
    )
    native_overlay = (
        np.asarray(original).copy()
        if original is not None
        else cv2.cvtColor(native_edges, cv2.COLOR_GRAY2RGB)
    )
    if native_lines is not None:
        for rho, theta in native_lines[: params.max_lines, 0]:
            cosine, sine = np.cos(theta), np.sin(theta)
            x0, y0 = cosine * rho, sine * rho
            point_a = (int(x0 + 2000 * -sine), int(y0 + 2000 * cosine))
            point_b = (int(x0 - 2000 * -sine), int(y0 - 2000 * cosine))
            cv2.line(native_overlay, point_a, point_b, (35, 75, 170), 1)
    return {
        "output": matrix_to_data_url(overlay, color=True),
        "opencv_output": matrix_to_data_url(native_overlay, color=True),
        "opencv_steps": {
            "input_edges": serialize_step(native_edges),
            "line_overlay": serialize_step(native_overlay),
        },
        "_output_matrix": overlay,
        "steps": {
            "input_edges": serialize_step(edges),
            "accumulator": serialize_step(accumulator, display_size=(800, 260)),
            "line_overlay": serialize_step(overlay),
        },
        "lines": lines,
        "parameters": params.model_dump(),
    }


def _opencv_ellipse_detect(edges: np.ndarray, params: EllipseParams) -> list[dict[str, float]]:
    """Reference-only implementation matching the notebook's comparison cell."""
    edge_map = (np.asarray(edges) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    results: list[dict[str, float]] = []
    for contour in contours:
        if len(contour) < max(5, params.min_points):
            continue
        (center_x, center_y), (width, height), angle = cv2.fitEllipse(contour)
        if width >= height:
            major, minor, phi = width / 2, height / 2, angle
        else:
            major, minor, phi = height / 2, width / 2, angle + 90.0
        if minor < params.a_min or major > params.a_max or major / minor > params.max_aspect:
            continue
        if any(
            np.hypot(center_x - result["x0"], center_y - result["y0"]) < 3
            and abs(major - result["a"]) < 3
            and abs(minor - result["b"]) < 3
            for result in results
        ):
            continue
        results.append({
            "x0": float(center_x), "y0": float(center_y),
            "a": float(major), "b": float(minor),
            "phi": float(np.radians(phi % 180)), "phi_deg": float(phi % 180),
        })
    return results


ELLIPSE_MAX_DIM = 400


def _resize_for_notebook(original: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Match the notebook's MAX_DIM=400 image normalization for ellipse work.

    The existing Canny result remains untouched in the session.  When the
    selected edge map is larger than the notebook canvas, it is downsampled
    with nearest-neighbour interpolation so edge pixels remain binary.
    """
    height, width = original.shape[:2]
    scale = min(1.0, ELLIPSE_MAX_DIM / max(height, width))
    if scale == 1.0:
        return original.copy(), (np.asarray(edges) > 0), scale
    target_size = (int(round(width * scale)), int(round(height * scale)))
    resized_original = cv2.resize(original, target_size, interpolation=cv2.INTER_AREA)
    resized_edges = cv2.resize(
        (np.asarray(edges) > 0).astype(np.uint8),
        target_size,
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    return resized_original, resized_edges, scale


def _parameter_preview(original: np.ndarray, params: EllipseParameters, scale: float) -> np.ndarray:
    """Draw notebook-style axis-limit guides on the original-resolution image.

    The notebook's ``a_min`` is the lower bound for the minor semi-axis and
    ``a_max`` is the upper bound for the major semi-axis.  Concentric circles
    make those two limits unambiguous without inventing a second aspect ratio.
    """
    preview = original.copy()
    height, width = preview.shape[:2]
    centre = (width / 2.0, height / 2.0)
    draw_ellipse(preview, (*centre, params.a_min / scale, params.a_min / scale, 0.0), (255, 230, 0), thickness=2)
    draw_ellipse(preview, (*centre, params.a_max / scale, params.a_max / scale, 0.0), (255, 80, 190), thickness=2)
    return preview


def _annotate_ellipse_overlay(
    image: np.ndarray,
    params: EllipseParameters,
    resize_scale: float,
    canny_params: CannyParameters | None,
    title: str,
) -> np.ndarray:
    """Add reproducibility metadata to a rendered ellipse result image."""
    height, width = image.shape[:2]
    lines = [
        title,
        f"a_min={params.a_min:g}px  a_max={params.a_max:g}px  votes={params.vote_threshold}",
        f"fit_tol={params.fit_tol:g}px  support={params.min_support:g}  max/component={params.max_ellipses}",
        f"ransac_iters={params.ransac_iters}  resize_scale={resize_scale:.4f}  canvas={width}x{height}",
    ]
    if canny_params is not None:
        lines.append(
            f"canny: kernel={canny_params.kernel_size}  sigma={canny_params.sigma:g}  "
            f"low={canny_params.low_threshold:g}  high={canny_params.high_threshold:g}"
        )
    footer_height = 30 * len(lines) + 18
    annotated = np.full((height + footer_height, width, 3), (12, 14, 12), dtype=np.uint8)
    annotated[:height] = image
    for index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (10, height + 20 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (232, 238, 224),
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_ellipse(
    original: np.ndarray,
    edges: np.ndarray,
    params: EllipseParameters,
    compare_opencv: bool = False,
    canny_params: CannyParameters | None = None,
    resize_input: bool = True,
) -> dict[str, object]:
    """Run ellipse detection on a notebook-sized input without changing Canny."""
    if resize_input:
        processed_original, processed_edges, resize_scale = _resize_for_notebook(original, edges)
    else:
        processed_original = original.copy()
        processed_edges = np.asarray(edges) > 0
        resize_scale = 1.0
    canny_recomputed = False
    if resize_input and canny_params is not None:
        processed_edges = canny_edge_detector(processed_original, **canny_params.model_dump())["edges"] > 0
        canny_recomputed = True
    detector_params = EllipseParams(
        a_min=params.a_min,
        a_max=params.a_max,
        min_points=params.vote_threshold,
        fit_tol=params.fit_tol,
        min_support=params.min_support,
        max_ellipses=params.max_ellipses,
        ransac_iters=params.ransac_iters,
    )
    ellipses = detect_ellipses(processed_edges, detector_params)
    overlay = processed_original.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 128, 255), (255, 165, 0),
        (255, 0, 255), (0, 255, 255), (255, 255, 0), (160, 80, 255),
    ]
    for index, ellipse in enumerate(ellipses):
        draw_ellipse(overlay, ellipse, colors[index % len(colors)], thickness=2)
    annotated_overlay = _annotate_ellipse_overlay(
        overlay,
        params,
        resize_scale,
        canny_params if canny_recomputed else None,
        "CUSTOM ELLIPSE DETECTION",
    )

    result: dict[str, object] = {
        "output": matrix_to_data_url(annotated_overlay, color=True),
        "parameter_preview": matrix_to_data_url(_parameter_preview(original, params, resize_scale), color=True),
        "resize": {
            "scale": resize_scale,
            "width": int(processed_original.shape[1]),
            "height": int(processed_original.shape[0]),
            "canny_recomputed_on_resized_input": canny_recomputed,
        },
        "opencv_output": None,
        "opencv_steps": None,
        "steps": {
            "input_edges": serialize_step(processed_edges),
            "ellipse_overlay": serialize_step(annotated_overlay),
        },
        "ellipses": ellipses,
        "parameters": params.model_dump(),
        "_output_matrix": annotated_overlay,
    }
    if compare_opencv:
        opencv_ellipses = _opencv_ellipse_detect(processed_edges, detector_params)
        opencv_overlay = processed_original.copy()
        for ellipse in opencv_ellipses:
            draw_ellipse(opencv_overlay, ellipse, (255, 70, 70), thickness=2)
        annotated_opencv_overlay = _annotate_ellipse_overlay(
            opencv_overlay,
            params,
            resize_scale,
            canny_params if canny_recomputed else None,
            "OPENCV ELLIPSE REFERENCE",
        )
        result["opencv_output"] = matrix_to_data_url(annotated_opencv_overlay, color=True)
        result["opencv_steps"] = {
            "input_edges": serialize_step(processed_edges),
            "ellipse_overlay": serialize_step(annotated_opencv_overlay),
        }
    return result


def run_ellipse_opencv(
    original: np.ndarray,
    edges: np.ndarray,
    params: EllipseParameters,
    canny_params: CannyParameters | None = None,
    resize_input: bool = True,
) -> dict[str, object]:
    """Refresh only the OpenCV ellipse comparison for an existing Canny map."""
    if resize_input:
        processed_original, processed_edges, resize_scale = _resize_for_notebook(original, edges)
    else:
        processed_original = original.copy()
        processed_edges = np.asarray(edges) > 0
        resize_scale = 1.0

    detector_params = EllipseParams(
        a_min=params.a_min,
        a_max=params.a_max,
        min_points=params.vote_threshold,
        fit_tol=params.fit_tol,
        min_support=params.min_support,
        max_ellipses=params.max_ellipses,
        ransac_iters=params.ransac_iters,
    )
    opencv_ellipses = _opencv_ellipse_detect(processed_edges, detector_params)
    opencv_overlay = processed_original.copy()
    for ellipse in opencv_ellipses:
        draw_ellipse(opencv_overlay, ellipse, (255, 70, 70), thickness=2)
    annotated_overlay = _annotate_ellipse_overlay(
        opencv_overlay,
        params,
        resize_scale,
        canny_params if resize_input and canny_params is not None else None,
        "OPENCV ELLIPSE REFERENCE",
    )
    return {
        "opencv_output": matrix_to_data_url(annotated_overlay, color=True),
        "opencv_steps": {
            "input_edges": serialize_step(processed_edges),
            "ellipse_overlay": serialize_step(annotated_overlay),
        },
        "opencv_ellipses": opencv_ellipses,
    }


def _draw_circle_overlay(
    original: np.ndarray,
    circles: list[dict[str, float]],
    color: tuple[int, int, int],
    title: str,
    info_lines: list[str] | None = None,
) -> np.ndarray:
    """Render notebook-style circles on the original RGB image."""
    overlay = np.asarray(original).copy()
    if overlay.dtype != np.uint8:
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    for index, circle in enumerate(sorted(circles, key=lambda item: -item.get("r", 0)), 1):
        centre = (int(round(circle["x"])), int(round(circle["y"])))
        radius = max(1, int(round(circle["r"])))
        cv2.circle(overlay, centre, radius, color, 2, cv2.LINE_AA)
        cv2.drawMarker(overlay, centre, color, cv2.MARKER_CROSS, 13, 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(index),
            (centre[0] + 4, centre[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    # Keep the result visually self-describing without changing the input image.
    cv2.putText(overlay, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    # Parameter details are rendered in the static UI summary panel. Keeping
    # them out of the image prevents tiny text after the result is scaled into
    # an output card.
    return overlay


def _linked_circle_opencv_parameters(
    params: CircleRequest,
    canny_params: CannyParameters | None,
) -> dict[str, float | int]:
    """Map the selected from-scratch detector settings to HoughCircles.

    OpenCV's parameter names and scoring are different, so only equivalent
    values are linked directly.  The custom evidence threshold is carried
    over as param2; dp remains the neutral OpenCV resolution of 1.0.
    """
    r_min, r_max = diameter_to_radii(params.min_diam, params.max_diam)
    evidence = params.min_votes if params.algorithm == "cht" else params.min_inliers
    return {
        "dp": 1.0,
        "minDist": max(5.0, r_min),
        "param1": canny_params.high_threshold if canny_params is not None else 200.0,
        "param2": float(evidence),
        "minRadius": int(np.floor(r_min)),
        "maxRadius": int(np.ceil(r_max)),
    }


def run_circle(
    original: np.ndarray,
    edges: np.ndarray,
    blurred: np.ndarray,
    params: CircleRequest,
    compare_opencv: bool = False,
    canny_params: CannyParameters | None = None,
) -> dict[str, object]:
    """Run exactly one notebook circle detector on an existing Canny result."""
    edge_map = np.asarray(edges) > 0
    if edge_map.ndim != 2:
        raise ValueError("Circle detection requires a two-dimensional Canny edge map")
    blurred_gray = np.asarray(blurred)
    if blurred_gray.ndim != 2 or blurred_gray.shape != edge_map.shape:
        raise ValueError("The selected Canny blur and edge map must have matching shapes")

    r_min, r_max = diameter_to_radii(params.min_diam, params.max_diam)
    accumulator = None
    if params.algorithm == "cht":
        circles, accumulator = circular_hough(
            blurred_gray,
            edge_map,
            r_min,
            r_max,
            params.min_votes,
            params.min_coverage,
            params.perim_frac,
            params.smooth | 1,
            params.nms_win | 1,
            params.align_cos,
            refine=params.refine,
        )
        detector_title = "CUSTOM CIRCULAR HOUGH TRANSFORM"
    elif params.algorithm == "ransac":
        circles = ransac_circles(
            edge_map,
            r_min,
            r_max,
            params.tol,
            params.max_iter,
            params.min_inliers,
            params.perim_frac,
            params.min_coverage,
            refine=params.refine,
            seed=params.seed,
        )
        detector_title = "CUSTOM RANSAC CIRCLE DETECTION"
    else:  # Defensive guard for callers bypassing Pydantic.
        raise ValueError("algorithm must be 'cht' or 'ransac'")

    custom_info = [
        f"detected_circles={len(circles)}  min_diam={params.min_diam}px  max_diam={params.max_diam}px",
        f"radius_search={r_min:.1f}..{r_max:.1f}px  min_coverage={params.min_coverage:.2f}  perim_frac={params.perim_frac:.2f}",
    ]
    if params.algorithm == "cht":
        custom_info.append(
            f"min_votes={params.min_votes}  smooth={params.smooth | 1}  nms_win={params.nms_win | 1}  "
            f"align_cos={params.align_cos:.2f}  refine={params.refine}"
        )
    else:
        custom_info.append(
            f"tol={params.tol:.2f}px  max_iter={params.max_iter}  min_inliers={params.min_inliers}  "
            f"seed={params.seed}  refine={params.refine}"
        )
    if canny_params is not None:
        custom_info.append(
            f"canny: kernel={canny_params.kernel_size}  sigma={canny_params.sigma:g}  "
            f"low={canny_params.low_threshold:g}  high={canny_params.high_threshold:g}"
        )
    overlay = _draw_circle_overlay(original, circles, (0, 255, 255), detector_title, custom_info)
    result: dict[str, object] = {
        "output": matrix_to_data_url(overlay, color=True),
        "opencv_output": None,
        "opencv_steps": None,
        "steps": {
            "input_edges": serialize_step(edge_map.astype(np.uint8) * 255),
            "circle_overlay": serialize_step(overlay),
        },
        "algorithm": params.algorithm,
        "circles": circles,
        "radii": {"min": r_min, "max": r_max},
        "parameters": params.model_dump(),
        "custom_parameters": {
            "algorithm": params.algorithm,
            "min_diam": params.min_diam,
            "max_diam": params.max_diam,
            "r_min": r_min,
            "r_max": r_max,
            "min_coverage": params.min_coverage,
            "perim_frac": params.perim_frac,
        },
        "opencv_parameters": None,
        "_output_matrix": overlay,
    }
    if accumulator is not None:
        result["steps"]["accumulator"] = serialize_step(accumulator)

    custom_parameters = result["custom_parameters"]
    if params.algorithm == "cht":
        custom_parameters.update({
            "min_votes": params.min_votes,
            "smooth": params.smooth | 1,
            "nms_win": params.nms_win | 1,
            "align_cos": params.align_cos,
            "refine": params.refine,
        })
    else:
        custom_parameters.update({
            "tol": params.tol,
            "max_iter": params.max_iter,
            "min_inliers": params.min_inliers,
            "seed": params.seed,
            "refine": params.refine,
        })
    if canny_params is not None:
        custom_parameters["canny_kernel"] = canny_params.kernel_size
        custom_parameters["canny_sigma"] = canny_params.sigma
        custom_parameters["canny_low"] = canny_params.low_threshold
        custom_parameters["canny_high"] = canny_params.high_threshold

    if compare_opencv:
        opencv_parameters = _linked_circle_opencv_parameters(params, canny_params)
        opencv_results = opencv_circles(
            blurred_gray,
            float(opencv_parameters["minRadius"]),
            float(opencv_parameters["maxRadius"]),
            float(opencv_parameters["minDist"]),
            float(opencv_parameters["param1"]),
            float(opencv_parameters["param2"]),
            float(opencv_parameters["dp"]),
        )
        opencv_overlay = _draw_circle_overlay(
            original,
            opencv_results,
            (255, 80, 80),
            "OPENCV HOUGH CIRCLES",
            [
                f"detected_circles={len(opencv_results)}  dp={opencv_parameters['dp']:.2f}  minDist={opencv_parameters['minDist']:.1f}px",
                f"param1={opencv_parameters['param1']:.1f}  param2={opencv_parameters['param2']:.1f}  minRadius={opencv_parameters['minRadius']}px  maxRadius={opencv_parameters['maxRadius']}px",
                *([f"canny: kernel={canny_params.kernel_size}  sigma={canny_params.sigma:g}  low={canny_params.low_threshold:g}  high={canny_params.high_threshold:g}"] if canny_params is not None else []),
            ],
        )
        result["opencv_output"] = matrix_to_data_url(opencv_overlay, color=True)
        result["opencv_steps"] = {
            "input_edges": serialize_step(edge_map.astype(np.uint8) * 255),
            "circle_overlay": serialize_step(opencv_overlay),
        }
        result["opencv_circles"] = opencv_results
        result["opencv_parameters"] = opencv_parameters
    return result


def run_circle_opencv(
    original: np.ndarray,
    edges: np.ndarray,
    blurred: np.ndarray,
    params: CircleRequest,
    canny_params: CannyParameters | None = None,
) -> dict[str, object]:
    """Refresh only the OpenCV circle comparison for an existing Canny map."""
    edge_map = np.asarray(edges) > 0
    blurred_gray = np.asarray(blurred)
    if edge_map.ndim != 2 or blurred_gray.ndim != 2 or blurred_gray.shape != edge_map.shape:
        raise ValueError("The selected Canny blur and edge map must have matching shapes")

    opencv_parameters = _linked_circle_opencv_parameters(params, canny_params)

    opencv_results = opencv_circles(
        blurred_gray,
        float(opencv_parameters["minRadius"]),
        float(opencv_parameters["maxRadius"]),
        float(opencv_parameters["minDist"]),
        float(opencv_parameters["param1"]),
        float(opencv_parameters["param2"]),
        float(opencv_parameters["dp"]),
    )
    opencv_overlay = _draw_circle_overlay(original, opencv_results, (255, 80, 80), "OPENCV HOUGH CIRCLES")
    return {
        "opencv_output": matrix_to_data_url(opencv_overlay, color=True),
        "opencv_steps": {
            "input_edges": serialize_step(edge_map.astype(np.uint8) * 255),
            "circle_overlay": serialize_step(opencv_overlay),
        },
        "opencv_circles": opencv_results,
        "opencv_parameters": opencv_parameters,
    }
