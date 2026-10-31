# Manual implementation. No external CV library used.
"""Contour creation, clamping, closed lengths, and uniform arc-length sampling."""
from math import cos, hypot, pi, sin


def clamp_contour(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    """Clamp vertices to image coordinate bounds [0,w-1] by [0,h-1]."""
    return [(min(width - 1.0, max(0.0, x)), min(height - 1.0, max(0.0, y))) for x, y in points]


def make_contour(spec: dict, n: int) -> list[tuple[float, float]]:
    """Create polygon, circle, or ellipse vertices using parametric coordinates."""
    kind = spec.get("type", "polygon")
    if kind == "polygon":
        return [(float(x), float(y)) for x, y in spec.get("points", [])]
    center = spec.get("center", [0, 0])
    cx, cy = float(center[0]), float(center[1])
    if kind == "circle": rx = ry = float(spec.get("radius", 0))
    elif kind == "ellipse": rx, ry = float(spec.get("rx", 0)), float(spec.get("ry", 0))
    else: raise ValueError("initial_contour.type must be polygon, circle, or ellipse")
    return [(cx + rx * cos(2 * pi * i / n), cy + ry * sin(2 * pi * i / n)) for i in range(n)]


def distinct_count(points: list[tuple[float, float]]) -> int:
    """Count distinct coordinates for polygon validation."""
    return len(set((round(x, 6), round(y, 6)) for x, y in points))


def average_spacing(points: list[tuple[float, float]]) -> float:
    """Compute closed contour mean edge distance d-bar."""
    if not points: return 0.0
    return sum(hypot(points[(i + 1) % len(points)][0] - p[0], points[(i + 1) % len(points)][1] - p[1]) for i, p in enumerate(points)) / len(points)


def resample_uniform(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    """Resample a closed polygon at equal cumulative arc-length intervals."""
    if len(points) < 2 or count <= 0: return points[:]
    lengths: list[float] = []
    total = 0.0
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        length = hypot(nxt[0] - point[0], nxt[1] - point[1])
        lengths.append(length); total += length
    if total <= 0: return [points[0]] * count
    output: list[tuple[float, float]] = []
    edge = 0; walked = 0.0
    for sample in range(count):
        target = total * sample / count
        while edge < len(points) - 1 and walked + lengths[edge] < target:
            walked += lengths[edge]; edge += 1
        segment = lengths[edge]
        fraction = 0.0 if segment == 0 else (target - walked) / segment
        a, b = points[edge], points[(edge + 1) % len(points)]
        output.append((a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction))
    return output
