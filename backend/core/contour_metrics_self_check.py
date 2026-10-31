"""Deterministic checks for Person 4 contour measurements."""

from math import cos, isclose, pi, sin, sqrt

from backend.core.contour_metrics import measure_contour


def run() -> None:
    side = 10
    square = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]
    square_result = measure_contour(square)
    assert square_result["perimeter"] == 40.0
    assert square_result["shoelace_area"] == 100.0
    assert square_result["pixel_fill_area"] == 121
    assert square_result["boundary_pixels"] == 40
    assert square_result["pick_theorem_valid"]

    diamond_side = 10
    diamond = [(0.0, 0.0), (diamond_side, -diamond_side), (2 * diamond_side, 0.0), (diamond_side, diamond_side)]
    diamond_result = measure_contour(diamond)
    assert all(code % 2 == 1 for code in diamond_result["chain_code_digits"])
    assert isclose(diamond_result["perimeter"], 4 * diamond_side * sqrt(2), rel_tol=0.0, abs_tol=1e-9)

    reversed_result = measure_contour(list(reversed(square)))
    for key in ("perimeter", "shoelace_area", "boundary_pixels", "pixel_fill_area"):
        assert reversed_result[key] == square_result[key]

    radius = 60
    circle = []
    for index in range(720):
        point = (
            float(round(radius * cos(2 * pi * index / 720))),
            float(round(radius * sin(2 * pi * index / 720))),
        )
        if not circle or point != circle[-1]:
            circle.append(point)
    circle_result = measure_contour(circle)
    assert abs(circle_result["pixel_fill_area"] - pi * radius**2) / (pi * radius**2) <= 0.02
    assert circle_result["perimeter"] >= 2 * pi * radius
    print("contour_metrics_self_check: PASS")


if __name__ == "__main__":
    run()
