# Manual implementation. No external CV library used.
"""Plain-Python assertions for deterministic math and image encoding behavior."""
import math
import struct
import zlib
from array import array
from .chain_code import bresenham, codes_from_path, contour_path
from .contour import resample_uniform
from .constants import DEFAULT_PARAMS
from .gaussian import gaussian_kernel
from .gradient import sobel
from .greedy import evolve
from .metrics import chain_perimeter, shoelace_area
from .png import PNG_SIGNATURE, encode_png


def run() -> None:
    """Assert the requested geometry, Gaussian, PNG, and synthetic-circle checks."""
    side = 12
    square = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]
    assert shoelace_area(square) == side * side
    path = contour_path(square)
    codes, valid, start = codes_from_path(path)
    point = start
    from .constants import DIRECTIONS
    for code in codes: point = (point[0] + DIRECTIONS[code][0], point[1] + DIRECTIONS[code][1])
    assert valid and point == start and chain_perimeter(codes) == 4 * side
    diagonal = bresenham(0, 0, 9, 9)
    diagonal_codes = []
    for first, second in zip(diagonal, diagonal[1:]):
        delta = (second[0] - first[0], second[1] - first[1]); diagonal_codes.append(DIRECTIONS.index(delta))
    assert diagonal_codes and all(code in (1, 3, 5, 7) for code in diagonal_codes)
    kernel = gaussian_kernel(1.3)
    assert abs(sum(kernel) - 1.0) < 1e-9
    width = height = 2; rgba = bytes(range(width * height * 4)); encoded = encode_png(width, height, rgba)
    assert encoded.startswith(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE); compressed = b""
    while offset < len(encoded):
        length = struct.unpack(">I", encoded[offset:offset + 4])[0]; kind = encoded[offset + 4:offset + 8]
        payload = encoded[offset + 8:offset + 8 + length]
        if kind == b"IDAT": compressed += payload
        offset += 12 + length
    assert len(zlib.decompress(compressed)) == height * (1 + width * 4)
    width = height = 64; target = 19.0; gray = array("f", [0.0]) * (width * height)
    for y in range(height):
        for x in range(width):
            if (x - 32) ** 2 + (y - 32) ** 2 <= target * target: gray[y * width + x] = 255.0
    _, _, mag, mag2, _, _ = sobel(width, height, gray)
    from .energy import external_energy
    energy, _ = external_energy(mag2)
    snake = [(32 + 29 * math.cos(2 * math.pi * i / 64), 32 + 29 * math.sin(2 * math.pi * i / 64)) for i in range(64)]
    params = {**DEFAULT_PARAMS, "alpha": 0.2, "beta": 0.1, "gamma": 3.0, "n_points": 64, "window_size": 5, "max_iterations": 100, "movement_threshold": 0, "gaussian_sigma": 0.0, "corner_relaxation": False}
    result = evolve(width, height, energy, mag, snake, params)
    final = result["points"]
    mean_radius = sum(math.hypot(x - 32, y - 32) for x, y in final) / len(final)
    assert abs(mean_radius - target) <= 2.0, f"circle radius {mean_radius:.3f} not within 2 px of {target}"
    print(f"self_check: PASS (circle mean radius={mean_radius:.3f}px; square area/perimeter; closed Freeman path; diagonal Bresenham; Gaussian sum; PNG IDAT length)")


if __name__ == "__main__": run()

