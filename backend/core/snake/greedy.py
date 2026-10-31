# Manual implementation. No external CV library used.
"""Greedy active contour using Econtour = alpha Eelastic + beta Esmooth + gamma Eimage."""
from math import hypot
from .contour import average_spacing, clamp_contour, resample_uniform


def normalize_window(values: list[float]) -> list[float]:
    """Min-max normalize candidate terms independently with a zero-range guard."""
    low, high = min(values), max(values)
    if high - low <= 1e-12: return [0.0] * len(values)
    inverse = 1.0 / (high - low)
    return [(value - low) * inverse for value in values]


def relax_corners(points: list[tuple[float, float]], beta_values: list[float], gradient: list[float], width: int, threshold: float) -> list[float]:
    """Set beta to zero at local curvature maxima with high image gradient."""
    count = len(points); curvature = [0.0] * count
    for i, point in enumerate(points):
        prev, nxt = points[(i - 1) % count], points[(i + 1) % count]
        dx = prev[0] - 2 * point[0] + nxt[0]
        dy = prev[1] - 2 * point[1] + nxt[1]
        curvature[i] = dx * dx + dy * dy
    high_gradient = max(gradient, default=0.0) * 0.5
    adjusted = beta_values[:]
    for i, value in enumerate(curvature):
        if value >= threshold and value >= curvature[(i - 1) % count] and value >= curvature[(i + 1) % count]:
            x, y = points[i]; index = int(round(y)) * width + int(round(x))
            if 0 <= index < len(gradient) and gradient[index] >= high_gradient: adjusted[i] = 0.0
    return adjusted


def evolve(width: int, height: int, energy, gradient, points: list[tuple[float, float]], params: dict, cancel=None) -> dict:
    """Greedy updates use current updated predecessor, average-distance continuity, and motion-count stopping."""
    alpha, beta, gamma = float(params["alpha"]), float(params["beta"]), float(params["gamma"])
    window = int(params["window_size"]); radius = window // 2
    max_iterations = int(params["max_iterations"]); movement_threshold = int(params["movement_threshold"])
    count = len(points); frames = []; energies = []; moved_history = []
    betas = [beta] * count; stop_reason = "max_iterations"; moved = 0
    for iteration in range(1, max_iterations + 1):
        if cancel is not None and cancel.is_set(): return {"cancelled": True}
        mean_distance = average_spacing(points)
        updated = points[:]; total_selected = 0.0; moved = 0
        for i in range(count):
            if cancel is not None and cancel.is_set(): return {"cancelled": True}
            prev = updated[(i - 1) % count]
            nxt = updated[0] if i == count - 1 else points[i + 1]
            x0, y0 = points[i]
            candidates = []; cont = []; curv = []; imgs = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x, y = int(round(x0)) + dx, int(round(y0)) + dy
                    if x < 0 or y < 0 or x >= width or y >= height: continue
                    dist = hypot(x - prev[0], y - prev[1])
                    cdx = prev[0] - 2 * x + nxt[0]; cdy = prev[1] - 2 * y + nxt[1]
                    candidates.append((float(x), float(y)))
                    cont.append((mean_distance - dist) ** 2)
                    curv.append(cdx * cdx + cdy * cdy)
                    imgs.append(float(energy[y * width + x]))
            if not candidates: continue
            cn, kn, en = normalize_window(cont), normalize_window(curv), normalize_window(imgs)
            best = 0; best_energy = float("inf")
            for j in range(len(candidates)):
                value = alpha * cn[j] + betas[i] * kn[j] + gamma * en[j]
                if value < best_energy: best_energy, best = value, j
            updated[i] = candidates[best]; total_selected += best_energy
            if hypot(updated[i][0] - x0, updated[i][1] - y0) > 1e-9: moved += 1
        points = updated
        if bool(params["corner_relaxation"]):
            betas = relax_corners(points, [beta] * count, gradient, width, float(params["corner_threshold"]))
        if int(params["resample_every_k"]) > 0 and iteration % int(params["resample_every_k"]) == 0:
            points = clamp_contour(resample_uniform(points, count), width, height)
        frames.append({"iteration": iteration, "contour": [{"x": x, "y": y} for x, y in points], "energy": total_selected / max(1, count), "movedPoints": moved, "averageDistance": average_spacing(points)})
        energies.append(total_selected / max(1, count)); moved_history.append(moved)
        if moved < movement_threshold:
            stop_reason = "converged"; break
    return {"cancelled": False, "points": points, "frames": frames, "energies": energies, "moved": moved, "stop_reason": stop_reason}
