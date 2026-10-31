# Manual implementation. No external CV library used.
"""Orchestrate the hand-written grayscale, smoothing, energy, contour and metric stages."""
from array import array
from base64 import b64encode
from math import hypot
from time import perf_counter
from .chain_code import contour_path, codes_from_path, normalized_first_difference
from .constants import LARGE_IMAGE_PIXELS, MAX_FRAMES, SOBEL_X, SOBEL_Y
from .contour import average_spacing, clamp_contour, distinct_count, make_contour, resample_uniform
from .decode import decode_rgba
from .energy import external_energy
from .gaussian import gaussian_kernel, separable_blur
from .gradient import sobel
from .greedy import evolve
from .grayscale import to_grayscale
from .metrics import chain_area, chain_perimeter, polygon_perimeter, scanline_area, shoelace_area
from .png import encode_png
from .render import gray_rgba, overlay, stretch_u8


def _data_url(width: int, height: int, rgba, cancel=None) -> str:
    """Encode an RGBA thumbnail as a PNG data URL."""
    return "data:image/png;base64," + b64encode(encode_png(width, height, rgba, cancel)).decode("ascii")


def _summary(values) -> dict:
    """Summarize a flat scalar plane with its minimum, maximum and mean."""
    if not values: return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(min(values)), "max": float(max(values)), "mean": float(sum(values) / len(values))}


def _even_frames(frames: list[dict]) -> list[dict]:
    """Keep at most MAX_FRAMES approximately even frames, including first and last."""
    if len(frames) <= MAX_FRAMES: return frames
    indices = [round(i * (len(frames) - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
    return [frames[i] for i in indices]


def _run_pipeline(content: bytes, params: dict, contour_spec: dict, cancel=None) -> dict:
    """Run every manually implemented stage and return the Snake API response plus process log."""
    timings: dict[str, float] = {}; log: list[dict] = []; warnings: list[str] = []
    start = perf_counter()
    def stage(name: str, description: str, numbers: dict, began: float, thumbnail: str | None = None) -> None:
        duration = (perf_counter() - began) * 1000.0
        timings[name] = duration
        entry = {"name": name, "description": description, "numbers": numbers, "durationMs": duration}
        if thumbnail is not None: entry["thumbnail"] = thumbnail
        log.append(entry)

    decode_began = perf_counter()
    width, height, rgba = decode_rgba(content)
    if width * height > LARGE_IMAGE_PIXELS: warnings.append("Large image: processing may take a while")
    if cancel is not None and cancel.is_set(): return {"cancelled": True}
    began = perf_counter(); gray = to_grayscale(width, height, rgba, cancel)
    if cancel is not None and cancel.is_set(): return {"cancelled": True}
    gray_image = _data_url(width, height, gray_rgba(width, height, stretch_u8(gray, width=width, cancel=cancel), cancel), cancel)
    gray_stats = _summary(gray)
    stage("Input", "Decode the image file into RGBA bytes. Alpha is ignored by the luminance step.", {"width": width, "height": height, "channels": 4}, decode_began)
    began = perf_counter()
    stage("Grayscale", "Compute gray = 0.299R + 0.587G + 0.114B per pixel.", gray_stats, began, gray_image)

    began = perf_counter(); sigma = float(params["gaussian_sigma"]); kernel = gaussian_kernel(sigma)
    smooth = separable_blur(width, height, gray, kernel, cancel)
    if cancel is not None and cancel.is_set(): return {"cancelled": True}
    smooth_image = _data_url(width, height, gray_rgba(width, height, stretch_u8(smooth, width=width, cancel=cancel), cancel), cancel)
    stage("Gaussian", "Separable horizontal then vertical Gaussian convolution; border pixels are replicated.", {"sigma": sigma, "kernel": kernel, "kernelSize": len(kernel), "borderPolicy": "replicate"}, began, smooth_image)

    began = perf_counter(); gx, gy, magnitude, magnitude2, max_gradient, mean_gradient = sobel(width, height, smooth, cancel)
    if cancel is not None and cancel.is_set(): return {"cancelled": True}
    gradient_image = _data_url(width, height, gray_rgba(width, height, stretch_u8(magnitude, width=width, cancel=cancel), cancel), cancel)
    stage("Sobel", "Manual 3×3 Sobel convolution for horizontal/vertical derivatives and gradient magnitude; edges are replicated at image borders.", {"kernelX": SOBEL_X, "kernelY": SOBEL_Y, "maxMagnitude": max_gradient, "meanMagnitude": mean_gradient, "borderPolicy": "replicate"}, began, gradient_image)

    began = perf_counter(); energy, max_magnitude_squared = external_energy(magnitude2, width, cancel)
    if cancel is not None and cancel.is_set(): return {"cancelled": True}
    energy_image = _data_url(width, height, gray_rgba(width, height, stretch_u8(energy, width=width, cancel=cancel), cancel), cancel)
    stage("Energy", "Eimage = −|∇I|² normalized by maximum squared magnitude into [−1, 0].", {"maxMagnitudeSquared": max_magnitude_squared, "normalization": "negative magnitude squared divided by global maximum; zero map if maximum is zero"}, began, energy_image)

    began = perf_counter(); initial_raw = make_contour(contour_spec, int(params["n_points"]))
    initial = clamp_contour(initial_raw, width, height)
    initial = resample_uniform(initial, int(params["n_points"]))
    spacing = average_spacing(initial)
    stage("Initial contour", "Create the selected shape, clamp to image bounds, then sample uniformly by closed arc length.", {"type": contour_spec.get("type", "polygon"), "pointCount": len(initial), "meanSpacing": spacing, "clampedToImageBounds": True}, began)

    began = perf_counter(); evolved = evolve(width, height, energy, magnitude, initial, params, cancel)
    if evolved.get("cancelled"): return {"cancelled": True}
    final = evolved["points"]
    stage("Greedy loop", "Minimize independently min-max-normalized continuity, curvature, and image energy terms; update in order using the already-updated predecessor.", {"alpha": params["alpha"], "beta": params["beta"], "gamma": params["gamma"], "windowSize": params["window_size"], "iterationsRun": len(evolved["frames"]), "stopReason": evolved["stop_reason"], "finalMovedPoints": evolved["moved"], "firstTotalEnergy": evolved["energies"][0] if evolved["energies"] else 0.0, "lastTotalEnergy": evolved["energies"][-1] if evolved["energies"] else 0.0, "energyPerIteration": evolved["energies"], "movementThreshold": params["movement_threshold"], "cornerRelaxation": params["corner_relaxation"], "cornerThreshold": params["corner_threshold"], "resampleEveryK": params["resample_every_k"]}, began)

    began = perf_counter(); path = contour_path(final); codes, valid, start_point = codes_from_path(path)
    chain = "".join(str(code) for code in codes)
    stage("Chain code", "Bresenham-rasterize the closed contour and encode 8-connected Freeman directions (0 east; y increases down).", {"start": {"x": start_point[0], "y": start_point[1]}, "length": len(codes), "first40": codes[:40], "valid": valid}, began)

    began = perf_counter(); p_chain = chain_perimeter(codes); p_polygon = polygon_perimeter(final)
    a_shoe = shoelace_area(final); a_chain = chain_area(path, codes); a_scan = scanline_area(width, height, final, cancel)
    if cancel is not None and cancel.is_set(): return {"cancelled": True}
    metrics = {"perimeter": p_chain, "perimeter_chain": p_chain, "perimeter_polygon": p_polygon, "area": a_shoe, "area_shoelace": a_shoe, "area_chain": a_chain, "area_scanline": a_scan, "perimeter_difference": abs(p_chain - p_polygon), "area_chain_difference": abs(a_shoe - a_chain), "area_scanline_difference": abs(a_shoe - a_scan), "point_count": len(final), "chain_code_length": len(codes)}
    stage("Metrics", "Chain perimeter weights orthogonal/diagonal steps by 1/√2; polygon and discrete Green area are cross-checks against shoelace and even-odd scanline count.", metrics, began)

    began = perf_counter(); final_overlay = _data_url(
        width,
        height,
        overlay(
            width,
            height,
            rgba,
            initial,
            final,
            int(params["initial_thickness"]),
            int(params["final_thickness"]),
            bool(params["show_vertices"]),
        ),
        cancel,
    )
    frame_list = _even_frames(evolved["frames"])
    history = [{"iteration": frame["iteration"], "contour": frame["contour"], "energy": frame["energy"], "movedPoints": frame["movedPoints"], "averageDistance": frame["averageDistance"]} for frame in frame_list]
    stage("Rendering", "Draw the initial and final contours, final vertices, and encode PNG thumbnails with a manual PNG writer.", {"storedFrames": len(history), "totalFrames": len(evolved["frames"]), "frameThinning": len(history) != len(evolved["frames"])}, began, final_overlay)
    began = perf_counter()
    stage("Timing", "Elapsed wall-clock time recorded for each preceding pipeline stage.", dict(timings), began)
    total_ms = (perf_counter() - start) * 1000.0
    return {"final_contour": [{"x": x, "y": y} for x, y in final], "initial_contour": [{"x": x, "y": y} for x, y in initial], "history": history, "frames": history, "iterations_run": len(evolved["frames"]), "converged": evolved["stop_reason"] == "converged", "stop_reason": evolved["stop_reason"], "overlay_image": final_overlay, "energy_map_image": energy_image, "grayscale_image": gray_image, "smoothed_image": smooth_image, "gradient_magnitude_image": gradient_image, "chain_code": chain, "chain_code_digits": codes, "chain_code_first_difference": normalized_first_difference(codes), "chain_code_start": {"x": start_point[0], "y": start_point[1]}, "chain_code_valid": valid, "connectivity": 8, **metrics, "stats": {"final_energy": evolved["energies"][-1] if evolved["energies"] else 0.0, "iterations": len(evolved["frames"]), "points": len(final)}, "energy_per_iteration": evolved["energies"], "processLog": log, "warnings": warnings, "stage_timings_ms": timings, "processing_time_ms": total_ms, "image_width": width, "image_height": height}


def run_pipeline(content: bytes, params: dict, contour_spec: dict, cancel=None) -> dict:
    """Return a cancellation result when any row- or iteration-level poll observes disconnect."""
    try:
        return _run_pipeline(content, params, contour_spec, cancel)
    except InterruptedError:
        return {"cancelled": True}
