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


def detect_peaks(accumulator: np.ndarray, radii: np.ndarray, thetas: np.ndarray, max_lines: int, threshold: int) -> list[dict[str, float]]:
    candidates = np.argwhere(accumulator >= threshold)
    ranked = sorted(candidates, key=lambda point: int(accumulator[tuple(point)]), reverse=True)
    lines: list[dict[str, float]] = []
    for radius_index, theta_index in ranked:
        radius, theta = float(radii[radius_index]), float(thetas[theta_index])
        if all(abs(radius - line["rho"]) > 10 or abs(theta - line["theta"]) > 10 for line in lines):
            lines.append({"rho": radius, "theta": theta, "votes": int(accumulator[radius_index, theta_index])})
        if len(lines) >= max_lines:
            break
    return lines


def draw_lines(edges: np.ndarray, lines: list[dict[str, float]]) -> np.ndarray:
    canvas = np.zeros((*edges.shape, 3), dtype=np.uint8)
    canvas[edges > 0] = (220, 245, 106)
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
