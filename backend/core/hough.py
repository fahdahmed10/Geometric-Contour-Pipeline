import numpy as np
import cv2


def hough_transform(edges: np.ndarray, radius_res: float, theta_res: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = edges.shape
    max_radius = np.sqrt(height**2 + width**2)
    radii = np.arange(-max_radius, max_radius + radius_res, radius_res)
    thetas = np.arange(0, 180, theta_res)
    accumulator = np.zeros((len(radii), len(thetas)), dtype=np.int32)
    cosine, sine = np.cos(np.deg2rad(thetas)), np.sin(np.deg2rad(thetas))
    for y, x in zip(*np.nonzero(edges)):
        indices = np.rint((x * cosine + y * sine + max_radius) / radius_res).astype(int)
        valid = (indices >= 0) & (indices < len(radii))
        accumulator[indices[valid], np.nonzero(valid)[0]] += 1
    return accumulator, radii, thetas


def detect_peaks(
    accumulator: np.ndarray,
    radii: np.ndarray,
    thetas: np.ndarray,
    max_lines: int,
    threshold: int,
    neighborhood: int = 1,
) -> list[dict[str, float]]:
    """Select local accumulator maxima using the notebook's suppression rule."""
    if neighborhood < 0:
        raise ValueError("neighborhood must be non-negative")
    candidates: list[tuple[int, int, int]] = []
    radius_count, theta_count = accumulator.shape
    for radius_index, theta_index in np.argwhere(accumulator >= threshold):
        vote = int(accumulator[radius_index, theta_index])
        is_peak = True
        for radius_offset in range(-neighborhood, neighborhood + 1):
            for theta_offset in range(-neighborhood, neighborhood + 1):
                neighbor_radius = radius_index + radius_offset
                neighbor_theta = theta_index + theta_offset
                if (
                    0 <= neighbor_radius < radius_count
                    and 0 <= neighbor_theta < theta_count
                    and accumulator[neighbor_radius, neighbor_theta] > vote
                ):
                    is_peak = False
                    break
            if not is_peak:
                break
        if is_peak:
            candidates.append((vote, int(radius_index), int(theta_index)))
    candidates.sort(reverse=True)
    lines: list[dict[str, float]] = []
    for vote, radius_index, theta_index in candidates[:max_lines]:
        radius, theta = float(radii[radius_index]), float(thetas[theta_index])
        lines.append({"rho": radius, "theta": theta, "votes": vote})
    return lines


def draw_lines(
    edges: np.ndarray,
    lines: list[dict[str, float]],
    background: np.ndarray | None = None,
) -> np.ndarray:
    if background is None:
        canvas = np.zeros((*edges.shape, 3), dtype=np.uint8)
        canvas[edges > 0] = (220, 245, 106)
    else:
        canvas = np.asarray(background).copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    height, width = edges.shape
    for line in lines:
        theta = np.deg2rad(line["theta"])
        rho = line["rho"]
        cosine, sine = np.cos(theta), np.sin(theta)
        x0, y0 = cosine * rho, sine * rho
        point_a = (int(x0 + 2000 * -sine), int(y0 + 2000 * cosine))
        point_b = (int(x0 - 2000 * -sine), int(y0 - 2000 * cosine))
        cv2.clipLine((0, 0, width, height), point_a, point_b)
        cv2.line(canvas, point_a, point_b, (255, 110, 82), 1)
    return canvas
