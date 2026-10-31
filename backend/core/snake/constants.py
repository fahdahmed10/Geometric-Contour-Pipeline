# Manual implementation. No external CV library used.
"""Constants for the manually implemented snake pipeline."""
from math import sqrt
from types import MappingProxyType

SOBEL_X: tuple[tuple[int, ...], ...] = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
SOBEL_Y: tuple[tuple[int, ...], ...] = ((-1, -2, -1), (0, 0, 0), (1, 2, 1))
OVERLAY_INITIAL: tuple[int, int, int, int] = (244, 190, 96, 255)
OVERLAY_FINAL: tuple[int, int, int, int] = (125, 211, 192, 255)
OVERLAY_VERTEX: tuple[int, int, int, int] = (244, 246, 250, 255)
DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))
DEFAULT_PARAMS = MappingProxyType({
    "alpha": 1.0, "beta": 1.0, "gamma": 1.2, "n_points": 20,
    "window_size": 5, "max_iterations": 200, "movement_threshold": 1,
    "gaussian_sigma": 1.0, "corner_relaxation": True,
    "corner_threshold": 0.1, "resample_every_k": 0,
    "initial_thickness": 2, "final_thickness": 3, "show_vertices": False,
})
SQRT2: float = sqrt(2.0)
MAX_FRAMES: int = 200
LARGE_IMAGE_PIXELS: int = 2_000_000
