# Manual implementation. No external CV library used.
"""Decode uploaded image bytes into RGBA; Pillow is used only for decoding."""
from io import BytesIO
from PIL import Image


def decode_rgba(content: bytes) -> tuple[int, int, bytes]:
    """Decode the file container to RGBA bytes; no image processing occurs here."""
    with Image.open(BytesIO(content)) as image:
        rgba = image.convert("RGBA")
        return rgba.width, rgba.height, rgba.tobytes()
