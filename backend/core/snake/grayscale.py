# Manual implementation. No external CV library used.
"""Manual RGB luminance conversion."""
from array import array


def to_grayscale(width: int, height: int, rgba: bytes, cancel=None) -> array:
    """Compute gray = .299R + .587G + .114B, ignoring alpha."""
    count = width * height
    gray = array("f", [0.0]) * count
    for y in range(height):
        if cancel is not None and cancel.is_set(): break
        source_row = y * width * 4; target_row = y * width
        for x in range(width):
            j = source_row + x * 4
            gray[target_row + x] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]
    return gray
