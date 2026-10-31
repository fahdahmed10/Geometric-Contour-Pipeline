# Manual implementation. No external CV library used.
"""Bresenham path and Freeman chain code (0 east; image y grows downward)."""
from .constants import DIRECTIONS


def bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Rasterize an integer segment using the hand-written Bresenham recurrence."""
    points = []; dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1; error = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1: break
        twice = 2 * error
        if twice >= dy: error += dy; x0 += sx
        if twice <= dx: error += dx; y0 += sy
    return points


def contour_path(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Join rounded contour vertices with Bresenham and remove adjacent duplicates."""
    path: list[tuple[int, int]] = []
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        segment = bresenham(round(point[0]), round(point[1]), round(nxt[0]), round(nxt[1]))
        for pixel in segment:
            if not path or path[-1] != pixel: path.append(pixel)
    if len(path) > 1 and path[-1] == path[0]: path.pop()
    return path


def codes_from_path(path: list[tuple[int, int]]) -> tuple[list[int], bool, tuple[int, int]]:
    """Map each closed 8-neighbor step to Freeman direction and validate closure."""
    if not path: return [], False, (0, 0)
    lookup = {delta: i for i, delta in enumerate(DIRECTIONS)}
    codes = []
    for i, point in enumerate(path):
        nxt = path[(i + 1) % len(path)]; delta = (nxt[0] - point[0], nxt[1] - point[1])
        if delta not in lookup: return codes, False, path[0]
        codes.append(lookup[delta])
    x, y = path[0]
    for code in codes:
        dx, dy = DIRECTIONS[code]; x += dx; y += dy
    return codes, (x, y) == path[0], path[0]


def normalized_first_difference(codes: list[int]) -> list[int]:
    """Return modulo-eight first differences for rotation-invariant direction changes."""
    if not codes: return []
    return [(codes[(i + 1) % len(codes)] - code) % 8 for i, code in enumerate(codes)]
