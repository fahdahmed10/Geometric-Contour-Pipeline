"""Chain-code, lattice area, and perimeter measurements for closed contours."""

from math import isclose

from backend.core.snake.chain_code import codes_from_path, contour_path
from backend.core.snake.constants import DIRECTIONS
from backend.core.snake.metrics import chain_perimeter, shoelace_area


def boundary_pixels(path: list[tuple[int, int]]) -> int:
    """Return the number of unique lattice pixels visited by the contour."""
    return len(set(path))


def _point_on_segment(x: int, y: int, start: tuple[int, int], end: tuple[int, int]) -> bool:
    sx, sy = start
    ex, ey = end
    cross = (x - sx) * (ey - sy) - (y - sy) * (ex - sx)
    if cross != 0:
        return False
    return min(sx, ex) <= x <= max(sx, ex) and min(sy, ey) <= y <= max(sy, ey)


def _inside_or_boundary(x: int, y: int, polygon: list[tuple[int, int]]) -> bool:
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(x, y, start, end):
            return True
        sx, sy = start
        ex, ey = end
        crosses = (sy > y) != (ey > y)
        if crosses:
            intersection_x = sx + (y - sy) * (ex - sx) / (ey - sy)
            if intersection_x > x:
                inside = not inside
    return inside


def pixel_fill_area(path: list[tuple[int, int]]) -> int:
    """Count lattice pixels inside or on a closed lattice contour."""
    if not path:
        return 0
    min_x = min(point[0] for point in path)
    max_x = max(point[0] for point in path)
    min_y = min(point[1] for point in path)
    max_y = max(point[1] for point in path)
    return sum(
        _inside_or_boundary(x, y, path)
        for y in range(min_y, max_y + 1)
        for x in range(min_x, max_x + 1)
    )


def measure_contour(points: list[tuple[float, float]]) -> dict[str, object]:
    """Measure a contour and assert the lattice Pick-theorem relationship."""
    if len(points) < 3:
        raise ValueError("A contour requires at least three points")
    path = contour_path(points)
    codes, valid, start = codes_from_path(path)
    if not valid:
        raise ValueError("Contour rasterization did not produce a closed 8-connected path")
    perimeter = chain_perimeter(codes)
    input_area = shoelace_area(points)
    area = shoelace_area([(float(x), float(y)) for x, y in path])
    boundary = boundary_pixels(path)
    fill = pixel_fill_area(path)
    pick_expected = area + boundary / 2.0 + 1.0
    if not isclose(fill, pick_expected, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Pixel-fill area violated Pick's theorem: "
            f"fill={fill}, expected={pick_expected}, boundary={boundary}"
        )
    return {
        "chain_code": "".join(str(code) for code in codes),
        "chain_code_digits": codes,
        "chain_code_valid": valid,
        "chain_code_start": {"x": start[0], "y": start[1]},
        "perimeter": perimeter,
        "shoelace_area": area,
        "input_polygon_area": input_area,
        "boundary_pixels": boundary,
        "pixel_fill_area": fill,
        "pick_expected_area": pick_expected,
        "pick_theorem_valid": True,
        "point_count": len(points),
        "chain_code_length": len(codes),
        "connectivity": 8,
        "directions": DIRECTIONS,
    }
