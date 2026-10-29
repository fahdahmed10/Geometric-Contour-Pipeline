from backend.core.canny import canny_edge_detector
from backend.core.hough import detect_peaks, draw_lines, hough_transform
from backend.core.image_codec import matrix_to_data_url, serialize_step
from backend.models.image_manager import ImageManager
from backend.models.schemas import CannyParameters, HoughParameters
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
