# Manual implementation. No external CV library used.
"""Explicit Sobel gradients and magnitude maps."""
from array import array
from math import sqrt
from .constants import SOBEL_X, SOBEL_Y


def sobel(width: int, height: int, image: array, cancel=None) -> tuple[array, array, array, array, float, float]:
    """Convolve with named 3x3 Sobel kernels and compute magnitude and square."""
    gx = array("f", [0.0]) * (width * height)
    gy = array("f", [0.0]) * (width * height)
    magnitude = array("f", [0.0]) * (width * height)
    magnitude2 = array("f", [0.0]) * (width * height)
    max_mag = 0.0
    sum_mag = 0.0
    for y in range(height):
        if cancel is not None and cancel.is_set(): break
        for x in range(width):
            sx = sy = 0.0
            for ky in range(-1, 2):
                yy = min(height - 1, max(0, y + ky))
                row = yy * width
                for kx in range(-1, 2):
                    xx = min(width - 1, max(0, x + kx))
                    value = image[row + xx]
                    sx += value * SOBEL_X[ky + 1][kx + 1]
                    sy += value * SOBEL_Y[ky + 1][kx + 1]
            index = y * width + x
            square = sx * sx + sy * sy
            mag = sqrt(square)
            gx[index], gy[index] = sx, sy
            magnitude[index], magnitude2[index] = mag, square
            sum_mag += mag
            if mag > max_mag: max_mag = mag
    return gx, gy, magnitude, magnitude2, max_mag, sum_mag / max(1, width * height)
