# Manual implementation. No external CV library used.
"""Normalized external image energy Eimage = -|gradient| squared."""
from array import array


def external_energy(magnitude_squared: array, width: int | None = None, cancel=None) -> tuple[array, float]:
    """Normalize negative squared gradient to [-1, 0], guarding a zero maximum."""
    maximum = max(magnitude_squared, default=0.0)
    if maximum <= 0.0:
        return array("f", [0.0]) * len(magnitude_squared), maximum
    output = array("f", [0.0]) * len(magnitude_squared)
    row_width = width or len(magnitude_squared)
    for start in range(0, len(magnitude_squared), row_width):
        if cancel is not None and cancel.is_set(): break
        end = min(len(magnitude_squared), start + row_width)
        for index in range(start, end): output[index] = -magnitude_squared[index] / maximum
    return output, maximum
