"""Notebook-derived circle detectors.

This module contains the two circle detectors from the circle notebook.  It
deliberately does not contain an edge detector: callers provide the selected
edge map and the Gaussian-blurred grayscale matrix produced by ContourLab's
existing Canny stage.
"""

from __future__ import annotations

import cv2
import numpy as np


def conv2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D correlation with edge padding, using shifted array sums."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    height, width = img.shape
    padded = np.pad(img.astype(np.float32), ((ph, ph), (pw, pw)), mode="edge")
    output = np.zeros((height, width), np.float32)
    for row in range(kh):
        for column in range(kw):
            if kernel[row, column] != 0:
                output += kernel[row, column] * padded[row : row + height, column : column + width]
    return output


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
SOBEL_Y = SOBEL_X.T.copy()


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return conv2d_same(gray, SOBEL_X), conv2d_same(gray, SOBEL_Y)


def max_filter(img: np.ndarray, size: int) -> np.ndarray:
    radius = size // 2
    height, width = img.shape
    padded = np.pad(img, radius, mode="constant", constant_values=-np.inf)
    output = np.full((height, width), -np.inf, np.float32)
    for row in range(size):
        for column in range(size):
            output = np.maximum(output, padded[row : row + height, column : column + width])
    return output


def angular_coverage(dx: np.ndarray, dy: np.ndarray, n_bins: int = 36) -> float:
    """Fraction of the 360° perimeter covered by supporting points."""
    if len(dx) == 0:
        return 0.0
    angles = np.arctan2(dy, dx)
    bins = ((angles + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    return float(np.unique(bins).size / n_bins)


def fit_circle_lsq(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Algebraic (Kasa) least-squares circle fit."""
    matrix = np.column_stack([x, y, np.ones_like(x)])
    solution, *_ = np.linalg.lstsq(matrix, -(x**2 + y**2), rcond=None)
    center_x, center_y = -solution[0] / 2.0, -solution[1] / 2.0
    radius_squared = center_x * center_x + center_y * center_y - solution[2]
    return (float(center_x), float(center_y), float(np.sqrt(radius_squared))) if radius_squared > 0 else None


def nms_circles(
    circles: list[dict[str, float]],
    center_frac: float = 0.5,
    r_frac: float = 0.25,
    min_center: float = 4.0,
    min_dr: float = 3.0,
) -> list[dict[str, float]]:
    """Suppress duplicate detections while preserving concentric circles."""
    kept: list[dict[str, float]] = []
    for circle in sorted(circles, key=lambda item: -item["score"]):
        duplicate = False
        for existing in kept:
            radius_min = min(circle["r"], existing["r"])
            if (
                np.hypot(circle["x"] - existing["x"], circle["y"] - existing["y"])
                < max(min_center, center_frac * radius_min)
                and abs(circle["r"] - existing["r"]) < max(min_dr, r_frac * radius_min)
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(circle)
    return kept


def circular_hough(
    gray_blur: np.ndarray,
    edges: np.ndarray,
    r_min: float,
    r_max: float,
    min_votes: int,
    min_coverage: float = 0.5,
    perim_frac: float = 0.0,
    smooth: int = 3,
    nms_win: int = 7,
    align_cos: float = 0.7,
    max_candidates: int = 200,
    refine: bool = True,
) -> tuple[list[dict[str, float]], np.ndarray]:
    """Circular Hough Transform from scratch using gradient-direction votes."""
    height, width = edges.shape
    y_indices, x_indices = np.nonzero(edges)

    gradient_x, gradient_y = sobel_gradients(gray_blur.astype(np.float32))
    edge_gx, edge_gy = gradient_x[y_indices, x_indices], gradient_y[y_indices, x_indices]
    magnitude = np.hypot(edge_gx, edge_gy)
    valid = magnitude > 1e-6
    x_indices, y_indices = x_indices[valid], y_indices[valid]
    edge_gx, edge_gy, magnitude = edge_gx[valid], edge_gy[valid], magnitude[valid]
    if len(x_indices) < 3:
        return [], np.zeros((height, width), np.float32)

    normal_x, normal_y = edge_gx / magnitude, edge_gy / magnitude
    x_points, y_points = x_indices.astype(np.float64), y_indices.astype(np.float64)

    accumulator = np.zeros(height * width, np.float64)
    radii = np.arange(int(np.floor(r_min)), int(np.ceil(r_max)) + 1)
    per_chunk = max(1, int(4_000_000 // (2 * len(x_points))))
    for start in range(0, len(radii), per_chunk):
        votes: list[np.ndarray] = []
        for radius in radii[start : start + per_chunk]:
            for sign in (1.0, -1.0):
                center_x = np.rint(x_points + sign * radius * normal_x).astype(np.int64)
                center_y = np.rint(y_points + sign * radius * normal_y).astype(np.int64)
                in_bounds = (center_x >= 0) & (center_x < width) & (center_y >= 0) & (center_y < height)
                votes.append(center_y[in_bounds] * width + center_x[in_bounds])
        if votes:
            accumulator += np.bincount(np.concatenate(votes), minlength=height * width)
    accumulator = accumulator.reshape(height, width).astype(np.float32)
    accumulator_smoothed = conv2d_same(accumulator, np.ones((smooth, smooth), np.float32))

    is_peak = (accumulator_smoothed >= max_filter(accumulator_smoothed, nms_win)) & (
        accumulator_smoothed >= 0.5 * min_votes
    )
    peak_y, peak_x = np.nonzero(is_peak)
    order = np.argsort(-accumulator_smoothed[peak_y, peak_x])[:max_candidates]

    bins = np.arange(np.floor(r_min) - 0.5, np.ceil(r_max) + 1.5, 1.0)
    radius_centers = 0.5 * (bins[:-1] + bins[1:])
    found: list[dict[str, float]] = []
    for peak_index in order:
        center_x, center_y = float(peak_x[peak_index]), float(peak_y[peak_index])
        dx, dy = x_points - center_x, y_points - center_y
        distances = np.hypot(dx, dy)
        radial_alignment = np.abs(dx * normal_x + dy * normal_y) / np.maximum(distances, 1e-9) >= align_cos
        selected = (distances >= r_min) & (distances <= r_max) & radial_alignment
        if selected.sum() < min_votes:
            continue
        selected_distances = distances[selected]
        histogram, _ = np.histogram(selected_distances, bins=bins)
        smoothed_histogram = np.convolve(histogram, np.ones(3), mode="same")
        for radius_index in range(len(smoothed_histogram)):
            lo, hi = max(0, radius_index - 2), min(len(smoothed_histogram), radius_index + 3)
            if (
                smoothed_histogram[radius_index] < min_votes
                or smoothed_histogram[radius_index] < smoothed_histogram[lo:hi].max()
                or (radius_index > 0 and smoothed_histogram[radius_index] == smoothed_histogram[radius_index - 1])
            ):
                continue
            radius_points = np.abs(selected_distances - radius_centers[radius_index]) <= 1.5
            if not radius_points.any():
                continue
            radius_estimate = float(selected_distances[radius_points].mean())
            tolerance = 1.5 + 0.02 * radius_estimate
            support = selected & (np.abs(distances - radius_estimate) <= tolerance)
            refined_x, refined_y, refined_radius = center_x, center_y, radius_estimate
            if refine and support.sum() >= 3:
                for _ in range(2):
                    fit = fit_circle_lsq(x_points[support], y_points[support])
                    if fit is None or not (r_min <= fit[2] <= r_max):
                        break
                    refined_x, refined_y, refined_radius = fit
                    dx, dy = x_points - refined_x, y_points - refined_y
                    distances = np.hypot(dx, dy)
                    radial_alignment = np.abs(dx * normal_x + dy * normal_y) / np.maximum(distances, 1e-9) >= align_cos
                    support = radial_alignment & (np.abs(distances - refined_radius) <= 1.5 + 0.02 * refined_radius)
            support_count = int(support.sum())
            coverage = angular_coverage(dx[support], dy[support])
            if (
                support_count >= max(min_votes, perim_frac * 2.0 * np.pi * refined_radius)
                and coverage >= min_coverage
                and r_min <= refined_radius <= r_max
            ):
                found.append(
                    {
                        "x": float(refined_x),
                        "y": float(refined_y),
                        "r": float(refined_radius),
                        "score": float(support_count),
                        "coverage": float(coverage),
                    }
                )
    return nms_circles(found), accumulator_smoothed


def ransac_circles(
    edges: np.ndarray,
    r_min: float,
    r_max: float,
    tol: float,
    max_iter: int,
    min_inliers: int,
    perim_frac: float = 0.4,
    min_coverage: float = 0.5,
    batch: int = 128,
    max_circles: int = 30,
    refine: bool = True,
    seed: int = 0,
) -> list[dict[str, float]]:
    """RANSAC from scratch: random 3-point fits with iterative inlier removal."""
    rng = np.random.default_rng(seed)
    y_indices, x_indices = np.nonzero(edges)
    points = np.column_stack([x_indices, y_indices]).astype(np.float64)
    circles: list[dict[str, float]] = []

    for _ in range(max_circles):
        point_count = len(points)
        if point_count < max(min_inliers, 3):
            break
        x_points, y_points = points[:, 0], points[:, 1]
        best: tuple[int, float, float, float] | None = None
        drawn = 0
        while drawn < max_iter:
            batch_size = min(batch, max_iter - drawn, max(8, int(4_000_000 // point_count)))
            drawn += batch_size
            indices = rng.integers(0, point_count, size=(batch_size, 3))
            p1, p2, p3 = points[indices[:, 0]], points[indices[:, 1]], points[indices[:, 2]]
            a11, a12 = 2 * (p2[:, 0] - p1[:, 0]), 2 * (p2[:, 1] - p1[:, 1])
            a21, a22 = 2 * (p3[:, 0] - p1[:, 0]), 2 * (p3[:, 1] - p1[:, 1])
            b1 = (p2**2).sum(1) - (p1**2).sum(1)
            b2 = (p3**2).sum(1) - (p1**2).sum(1)
            determinant = a11 * a22 - a12 * a21
            valid = (
                (np.abs(determinant) > 8.0)
                & (indices[:, 0] != indices[:, 1])
                & (indices[:, 0] != indices[:, 2])
                & (indices[:, 1] != indices[:, 2])
            )
            safe_determinant = np.where(valid, determinant, 1.0)
            center_x = (b1 * a22 - a12 * b2) / safe_determinant
            center_y = (a11 * b2 - b1 * a21) / safe_determinant
            radius = np.hypot(p1[:, 0] - center_x, p1[:, 1] - center_y)
            valid &= (radius >= r_min) & (radius <= r_max)
            if not valid.any():
                continue
            center_x, center_y, radius = center_x[valid], center_y[valid], radius[valid]
            distances = np.abs(
                np.hypot(x_points[None, :] - center_x[:, None], y_points[None, :] - center_y[:, None])
                - radius[:, None]
            )
            counts = (distances < tol).sum(1)
            best_index = int(np.argmax(counts))
            if best is None or counts[best_index] > best[0]:
                best = (
                    int(counts[best_index]),
                    float(center_x[best_index]),
                    float(center_y[best_index]),
                    float(radius[best_index]),
                )
        if best is None:
            break

        _, center_x, center_y, radius = best
        inlier_mask = np.abs(np.hypot(x_points - center_x, y_points - center_y) - radius) < tol
        if refine and inlier_mask.sum() >= 3:
            fit = fit_circle_lsq(x_points[inlier_mask], y_points[inlier_mask])
            if fit is not None and r_min <= fit[2] <= r_max:
                center_x, center_y, radius = fit
                inlier_mask = np.abs(np.hypot(x_points - center_x, y_points - center_y) - radius) < tol
        inlier_count = int(inlier_mask.sum())
        coverage = angular_coverage(x_points[inlier_mask] - center_x, y_points[inlier_mask] - center_y)
        if inlier_count < max(min_inliers, perim_frac * 2 * np.pi * radius) or coverage < min_coverage:
            break
        circles.append(
            {
                "x": float(center_x),
                "y": float(center_y),
                "r": float(radius),
                "score": float(inlier_count),
                "coverage": float(coverage),
            }
        )
        points = points[~inlier_mask]
    return circles


def diameter_to_radii(min_diam: float, max_diam: float) -> tuple[float, float]:
    if min_diam > max_diam:
        min_diam, max_diam = max_diam, min_diam
    return 0.85 * min_diam / 2.0, 1.15 * max_diam / 2.0


def opencv_circles(
    gray_blur: np.ndarray,
    r_min: float,
    r_max: float,
    min_dist: float,
    param1: float,
    param2: float,
    dp: float = 1.0,
) -> list[dict[str, float]]:
    """The notebook's opt-in OpenCV comparison implementation."""
    blurred = np.clip(np.rint(gray_blur), 0, 255).astype(np.uint8)
    detected = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=float(dp),
        minDist=float(min_dist),
        param1=float(param1),
        param2=float(param2),
        minRadius=int(np.floor(r_min)),
        maxRadius=int(np.ceil(r_max)),
    )
    if detected is None:
        return []
    return [{"x": float(x), "y": float(y), "r": float(radius)} for x, y, radius in detected[0]]
