# Manual implementation. No external CV library used.
"""Hand-built Gaussian kernel and separable convolution."""
from array import array
from math import ceil, exp


def gaussian_kernel(sigma: float) -> list[float]:
    """Build normalized 1D Gaussian weights with radius ceil(3 sigma)."""
    if sigma <= 0:
        return [1.0]
    radius = ceil(3.0 * sigma)
    denominator = 2.0 * sigma * sigma
    values = [exp(-(x * x) / denominator) for x in range(-radius, radius + 1)]
    total = sum(values)
    return [value / total for value in values]


def separable_blur(width: int, height: int, source: array, kernel: list[float], cancel=None) -> array:
    """Apply horizontal then vertical convolution with replicated edge pixels."""
    if len(kernel) == 1:
        return array("f", source)
    radius = len(kernel) // 2
    temp = array("f", [0.0]) * (width * height)
    output = array("f", [0.0]) * (width * height)
    for y in range(height):
        if cancel is not None and cancel.is_set():
            return output
        row = y * width
        for x in range(width):
            total = 0.0
            for k, weight in enumerate(kernel):
                sx = x + k - radius
                if sx < 0: sx = 0
                elif sx >= width: sx = width - 1
                total += source[row + sx] * weight
            temp[row + x] = total
    for y in range(height):
        if cancel is not None and cancel.is_set():
            return output
        row = y * width
        for x in range(width):
            total = 0.0
            for k, weight in enumerate(kernel):
                sy = y + k - radius
                if sy < 0: sy = 0
                elif sy >= height: sy = height - 1
                total += temp[sy * width + x] * weight
            output[row + x] = total
    return output
