from backend.core.canny import canny_edge_detector
from backend.core.hough import detect_peaks, draw_lines, hough_transform
from backend.core.image_codec import matrix_to_data_url, serialize_step
from backend.models.image_manager import ImageManager
from backend.models.schemas import CannyParameters, HoughParameters


def run_canny(manager: ImageManager, image_id: str, params: CannyParameters) -> dict[str, object]:
    cache_key = f"canny:{params.model_dump_json()}"
    cached = manager.get_cached(image_id, cache_key)
    steps = cached if isinstance(cached, dict) else canny_edge_detector(manager.get(image_id).original, **params.model_dump())
    if cached is None:
        manager.cache(image_id, cache_key, steps)
    return {
        "output": matrix_to_data_url(steps["edges"]),
        "steps": {name: serialize_step(value) for name, value in steps.items()},
        "parameters": params.model_dump(),
    }


def run_hough(manager: ImageManager, edges, params: HoughParameters) -> dict[str, object]:
    accumulator, radii, thetas = hough_transform(edges, params.radius_res, params.theta_res)
    lines = detect_peaks(accumulator, radii, thetas, params.max_lines, params.vote_threshold)
    overlay = draw_lines(edges, lines)
    return {
        "output": matrix_to_data_url(overlay, color=True),
        "_output_matrix": overlay,
        "steps": {
            "input_edges": serialize_step(edges),
            "accumulator": serialize_step(accumulator),
            "line_overlay": serialize_step(overlay),
        },
        "lines": lines,
        "parameters": params.model_dump(),
    }
