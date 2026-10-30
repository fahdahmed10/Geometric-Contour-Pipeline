import hashlib

from backend.core.canny import canny_edge_detector
from backend.core.ellipse import EllipseParams, detect_ellipses, draw_ellipse
from backend.core.hough import detect_peaks, draw_lines, hough_transform
from backend.core.image_codec import matrix_to_data_url, serialize_step
from backend.core.notebook_canny import canny as notebook_canny
from backend.models.image_manager import ImageManager
from backend.models.schemas import CannyComparisonRequest, CannyParameters, EllipseParameters, HoughParameters
import cv2
import numpy as np


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


def run_hough(manager: ImageManager, edges, params: HoughParameters) -> dict[str, object]:
    accumulator, radii, thetas = hough_transform(edges, params.radius_res, params.theta_res)
    lines = detect_peaks(accumulator, radii, thetas, params.max_lines, params.vote_threshold)
    overlay = draw_lines(edges, lines)
    native_edges = np.clip(edges, 0, 255).astype(np.uint8)
    native_lines = cv2.HoughLines(
        native_edges,
        params.radius_res,
        np.deg2rad(params.theta_res),
        params.vote_threshold,
    )
    native_overlay = cv2.cvtColor(native_edges, cv2.COLOR_GRAY2RGB)
    if native_lines is not None:
        for rho, theta in native_lines[: params.max_lines, 0]:
            cosine, sine = np.cos(theta), np.sin(theta)
            x0, y0 = cosine * rho, sine * rho
            point_a = (int(x0 + 2000 * -sine), int(y0 + 2000 * cosine))
            point_b = (int(x0 - 2000 * -sine), int(y0 - 2000 * cosine))
            cv2.line(native_overlay, point_a, point_b, (79, 70, 229), 1)
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
            "accumulator": serialize_step(accumulator),
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
    footer_height = 22 * len(lines) + 12
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
