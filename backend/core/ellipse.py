"""Custom ellipse detection ported from the ellipse notebook.

The detector intentionally starts with an edge map supplied by the caller.  The
notebook also contains a standalone Canny implementation, but the application
already owns that stage in :mod:`backend.core.canny`, so it is not duplicated
here.
"""

from dataclasses import dataclass

import numpy as np


def _shift(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return a shifted array with zeros outside the image."""
    height, width = array.shape
    padded = np.pad(array, 1)
    return padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Find 8-connected foreground components as ``(x, y)`` point arrays."""
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    padded_width = padded.shape[1]
    flat = bytearray(padded.ravel().tobytes())
    offsets = (-padded_width - 1, -padded_width, -padded_width + 1, -1, 1,
               padded_width - 1, padded_width, padded_width + 1)
    components = []
    for start in np.flatnonzero(padded.ravel()).tolist():
        if not flat[start]:
            continue
        flat[start] = 0
        stack, members = [start], []
        while stack:
            point = stack.pop()
            members.append(point)
            for offset in offsets:
                neighbour = point + offset
                if flat[neighbour]:
                    flat[neighbour] = 0
                    stack.append(neighbour)
        ys, xs = np.divmod(np.asarray(members), padded_width)
        components.append(np.column_stack([xs - 1, ys - 1]).astype(np.float64))
    return components


def _conic_to_geometric(A, B, C, D, E, F, mean_x, mean_y, scale):
    """Convert normalized conic coefficients to ``(x0, y0, a, b, phi)``."""
    determinant = A * C - B * B / 4.0
    if not np.isfinite(determinant) or abs(determinant) < 1e-14:
        return None
    quadratic = np.array([[A, B / 2.0], [B / 2.0, C]])
    try:
        centre = np.linalg.solve(2.0 * quadratic, np.array([-D, -E]))
    except np.linalg.LinAlgError:
        return None
    centred_constant = F + 0.5 * (D * centre[0] + E * centre[1])
    eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    if np.any(np.abs(eigenvalues) < 1e-14):
        return None
    radii = -centred_constant / eigenvalues
    if np.any(radii <= 0):
        return None
    axes = np.sqrt(radii)
    major_index = int(np.argmax(axes))
    major, minor = axes[major_index], axes[1 - major_index]
    direction = eigenvectors[:, major_index]
    phi = float(np.arctan2(direction[1], direction[0]) % np.pi)
    return (centre[0] * scale + mean_x, centre[1] * scale + mean_y,
            major * scale, minor * scale, phi)


def fit_ellipse(points: np.ndarray):
    """Fit an ellipse with the notebook's stable direct least-squares method."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 5:
        return None
    x, y = points[:, 0], points[:, 1]
    mean_x, mean_y = x.mean(), y.mean()
    scale = np.sqrt(np.mean((x - mean_x) ** 2 + (y - mean_y) ** 2))
    if scale < 1e-9:
        return None
    normalized_x, normalized_y = (x - mean_x) / scale, (y - mean_y) / scale
    quadratic_design = np.column_stack([normalized_x * normalized_x,
                                         normalized_x * normalized_y,
                                         normalized_y * normalized_y])
    linear_design = np.column_stack([normalized_x, normalized_y, np.ones_like(normalized_x)])
    S1 = quadratic_design.T @ quadratic_design
    S2 = quadratic_design.T @ linear_design
    S3 = linear_design.T @ linear_design
    try:
        transform = -np.linalg.solve(S3, S2.T)
        reduced = S1 + S2 @ transform
        reduced = np.vstack([reduced[2] / 2.0, -reduced[1], reduced[0] / 2.0])
        _, vectors = np.linalg.eig(reduced)
    except np.linalg.LinAlgError:
        return None
    vectors = np.real(vectors)
    conic_condition = 4.0 * vectors[0] * vectors[2] - vectors[1] ** 2
    valid = np.flatnonzero(conic_condition > 0)
    if len(valid) == 0:
        return None
    quadratic_coefficients = vectors[:, valid[np.argmax(conic_condition[valid])]]
    linear_coefficients = transform @ quadratic_coefficients
    A, B, C, D, E, F = np.concatenate([quadratic_coefficients, linear_coefficients])
    return _conic_to_geometric(A, B, C, D, E, F, mean_x, mean_y, scale)


def ellipse_distance(points: np.ndarray, ellipse) -> np.ndarray:
    """First-order (Sampson) approximation of geometric distance in pixels."""
    x0, y0, a, b, phi = ellipse
    cosine, sine = np.cos(phi), np.sin(phi)
    dx, dy = points[:, 0] - x0, points[:, 1] - y0
    u, v = dx * cosine + dy * sine, -dx * sine + dy * cosine
    residual = (u / a) ** 2 + (v / b) ** 2 - 1.0
    gradient = np.sqrt((2 * u / a ** 2) ** 2 + (2 * v / b ** 2) ** 2) + 1e-12
    return np.abs(residual) / gradient


def ellipse_perimeter(a: float, b: float) -> float:
    """Ramanujan's perimeter approximation used by the notebook."""
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


@dataclass
class EllipseParams:
    a_min: float = 10.0
    a_max: float = 150.0
    min_points: int = 40
    fit_tol: float = 2.0
    min_support: float = 0.5
    max_ellipses: int = 3
    ransac_iters: int = 300
    sample_size: int = 6
    max_aspect: float = 10.0
    seed: int = 0


def _valid(ellipse, params: EllipseParams) -> bool:
    if ellipse is None or not np.all(np.isfinite(ellipse)):
        return False
    _, _, major, minor, _ = ellipse
    return minor >= params.a_min and major <= params.a_max and (major / minor) <= params.max_aspect


def _refine(points, ellipse, params, iterations=3):
    for _ in range(iterations):
        inliers = ellipse_distance(points, ellipse) < params.fit_tol
        if inliers.sum() < 5:
            break
        refined = fit_ellipse(points[inliers])
        if not _valid(refined, params):
            break
        ellipse = refined
    return ellipse, ellipse_distance(points, ellipse) < params.fit_tol


def _best_ellipse(points, params: EllipseParams, rng: np.random.Generator):
    count = len(points)
    best, best_count = None, 0
    hypotheses = [fit_ellipse(points)]
    for iteration in range(params.ransac_iters + 1):
        if iteration == 0:
            ellipse = hypotheses[0]
        else:
            seed_point = points[rng.integers(count)]
            distances = np.hypot(points[:, 0] - seed_point[0], points[:, 1] - seed_point[1])
            pool = np.flatnonzero(distances <= 2.0 * params.a_max)
            if len(pool) < params.sample_size:
                continue
            ellipse = fit_ellipse(points[rng.choice(pool, params.sample_size, replace=False)])
        if not _valid(ellipse, params):
            continue
        inlier_count = int((ellipse_distance(points, ellipse) < params.fit_tol).sum())
        if inlier_count > best_count:
            best, best_count = ellipse, inlier_count
            if inlier_count >= 0.95 * count:
                break
    if best is None:
        return None, None
    return _refine(points, best, params)


def detect_ellipses(edges: np.ndarray, params: EllipseParams | None = None) -> list[dict[str, float]]:
    """Detect ellipses in an existing boolean/0-255 edge map."""
    params = params or EllipseParams()
    edge_map = np.asarray(edges) > 0
    rng = np.random.default_rng(params.seed)
    results = []
    for component in connected_components(edge_map):
        remaining = component
        for _ in range(params.max_ellipses):
            if len(remaining) < max(params.min_points, 5):
                break
            ellipse, inliers = _best_ellipse(remaining, params, rng)
            if ellipse is None:
                break
            inlier_count = int(inliers.sum())
            support = inlier_count / ellipse_perimeter(ellipse[2], ellipse[3])
            if inlier_count < params.min_points or support < params.min_support:
                break
            rms = float(np.sqrt(np.mean(ellipse_distance(remaining[inliers], ellipse) ** 2)))
            results.append({
                "x0": float(ellipse[0]), "y0": float(ellipse[1]),
                "a": float(ellipse[2]), "b": float(ellipse[3]),
                "phi": float(ellipse[4]), "phi_deg": float(np.degrees(ellipse[4])),
                "n_inliers": inlier_count, "support": float(min(support, 1.0)), "rms": rms,
            })
            remaining = remaining[ellipse_distance(remaining, ellipse) >= 1.5 * params.fit_tol]
    results.sort(key=lambda result: -result["support"])
    return results


def ellipse_points(x0, y0, a, b, phi, n=360):
    """Generate points from the notebook's explicit parametric ellipse equation."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = x0 + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
    y = y0 + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
    return x, y


def draw_ellipse(image: np.ndarray, ellipse, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Rasterize an ellipse into an RGB image without OpenCV drawing helpers."""
    if isinstance(ellipse, dict):
        ellipse = (ellipse["x0"], ellipse["y0"], ellipse["a"], ellipse["b"], ellipse["phi"])
    x0, y0, a, b, phi = ellipse
    height, width = image.shape[:2]
    count = max(180, int(4 * ellipse_perimeter(a, b)))
    x, y = ellipse_points(x0, y0, a, b, phi, count)
    x_indices, y_indices = np.round(x).astype(int), np.round(y).astype(int)
    radius = thickness // 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = y_indices + dy, x_indices + dx
            valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            image[yy[valid], xx[valid]] = color
    return image
