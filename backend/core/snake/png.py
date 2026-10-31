# Manual implementation. No external CV library used.
"""Minimal PNG writer using only struct and zlib."""
import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    """Pack a PNG chunk with a CRC over type and data."""
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)


def encode_png(width: int, height: int, rgba: bytes | bytearray, cancel=None) -> bytes:
    """Encode RGBA scanlines using filter type zero and zlib-compressed IDAT."""
    if width <= 0 or height <= 0 or len(rgba) != width * height * 4: raise ValueError("Invalid RGBA image dimensions")
    stride = width * 4; raw = bytearray((stride + 1) * height)
    for y in range(height):
        if cancel is not None and cancel.is_set(): raise InterruptedError("Snake request cancelled")
        destination = y * (stride + 1); raw[destination] = 0
        raw[destination + 1:destination + 1 + stride] = rgba[y * stride:(y + 1) * stride]
    ihdr = struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0)
    return PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")
