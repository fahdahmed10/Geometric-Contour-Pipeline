# Manual implementation. No external CV library used.
"""Manual scalar image scaling and RGBA rasterization primitives."""
from array import array
from .constants import OVERLAY_FINAL, OVERLAY_INITIAL, OVERLAY_VERTEX


def stretch_u8(values, low=None, high=None, width=None, cancel=None) -> bytearray:
    """Linearly min-max stretch values to byte intensity with a flat-map guard."""
    if not values: return bytearray()
    lo = min(values) if low is None else low; hi = max(values) if high is None else high
    if hi <= lo: return bytearray(len(values))
    scale = 255.0 / (hi - lo)
    output = bytearray(len(values)); row_width = width or len(values)
    for start in range(0, len(values), row_width):
        if cancel is not None and cancel.is_set(): break
        end = min(len(values), start + row_width)
        for i in range(start, end): output[i] = max(0, min(255, int((values[i] - lo) * scale + 0.5)))
    return output


def gray_rgba(width: int, height: int, values, cancel=None) -> bytearray:
    """Expand a grayscale byte plane to opaque RGBA."""
    output = bytearray(width * height * 4)
    for y in range(height):
        if cancel is not None and cancel.is_set(): break
        row = y * width
        for x in range(width):
            out = (row + x) * 4; value = values[row + x]
            output[out] = output[out + 1] = output[out + 2] = value; output[out + 3] = 255
    return output


def _pixel(canvas: bytearray, width: int, height: int, x: int, y: int, color) -> None:
    """Write one in-bounds RGBA pixel."""
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 4; canvas[offset:offset + 4] = bytes(color)


def draw_line(canvas: bytearray, width: int, height: int, start, end, color, thickness: int = 1) -> None:
    """Rasterize a line using integer Bresenham steps."""
    x0, y0 = round(start[0]), round(start[1]); x1, y1 = round(end[0]), round(end[1])
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1; dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1; error = dx + dy
    while True:
        draw_dot(canvas, width, height, (x0, y0), max(0, thickness - 1), color)
        if x0 == x1 and y0 == y1: break
        twice = 2 * error
        if twice >= dy: error += dy; x0 += sx
        if twice <= dx: error += dx; y0 += sy


def draw_dot(canvas: bytearray, width: int, height: int, center, radius: int, color) -> None:
    """Draw a filled circular vertex with a manual integer disk loop."""
    cx, cy = round(center[0]), round(center[1]); rr = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= rr: _pixel(canvas, width, height, cx + dx, cy + dy, color)


def overlay(
    width: int,
    height: int,
    rgba: bytes,
    initial,
    final,
    initial_thickness: int = 2,
    final_thickness: int = 3,
    show_vertices: bool = False,
) -> bytearray:
    """Copy input RGBA then draw initial/final contours and final vertex dots."""
    canvas = bytearray(rgba)
    for i, point in enumerate(initial):
        draw_line(canvas, width, height, point, initial[(i + 1) % len(initial)], OVERLAY_INITIAL, initial_thickness)
    for i, point in enumerate(final):
        draw_line(canvas, width, height, point, final[(i + 1) % len(final)], OVERLAY_FINAL, final_thickness)
    if show_vertices:
        for point in final:
            draw_dot(canvas, width, height, point, max(1, final_thickness), OVERLAY_VERTEX)
    return canvas
