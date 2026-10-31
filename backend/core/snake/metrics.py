# Manual implementation. No external CV library used.
"""Manual contour perimeter and area measures with independent cross-checks."""
from math import ceil, hypot
from .constants import DIRECTIONS, SQRT2


def polygon_perimeter(points: list[tuple[float, float]]) -> float:
    """Sum Euclidean edge lengths around the closed polygon."""
    return sum(hypot(points[(i + 1) % len(points)][0] - p[0], points[(i + 1) % len(points)][1] - p[1]) for i, p in enumerate(points))


def chain_perimeter(codes: list[int]) -> float:
    """Count orthogonal Freeman steps as one and diagonal steps as sqrt two."""
    return sum(SQRT2 if code % 2 else 1.0 for code in codes)


def shoelace_area(points: list[tuple[float, float]]) -> float:
    """Compute absolute polygon area by the shoelace formula."""
    total = 0.0
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]; total += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(total) * 0.5


def chain_area(path: list[tuple[int, int]], codes: list[int]) -> float:
    """Compute discrete Green area from integer chain path cross products."""
    total = 0
    for i, (x, y) in enumerate(path):
        dx, dy = DIRECTIONS[codes[i]]; total += x * (y + dy) - (x + dx) * y
    return abs(total) * 0.5


def scanline_area(width: int, height: int, polygon: list[tuple[float, float]], cancel=None) -> int:
    """Count pixel-center samples inside the polygon using even-odd scanlines."""
    count = 0; n = len(polygon)
    for y in range(height):
        if cancel is not None and cancel.is_set(): break
        scan_y = y + 0.5
        crossings = []
        for i, (x1, y1) in enumerate(polygon):
            x2, y2 = polygon[(i + 1) % n]
            if (y1 > scan_y) != (y2 > scan_y):
                crossings.append((x2 - x1) * (scan_y - y1) / (y2 - y1) + x1)
        crossings.sort()
        for i in range(0, len(crossings) - 1, 2):
            left = max(0, ceil(crossings[i] - 0.5))
            right = min(width - 1, ceil(crossings[i + 1] - 0.5) - 1)
            if right >= left: count += right - left + 1
    return count
