import base64
from io import BytesIO

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a supported image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def matrix_to_data_url(matrix: np.ndarray, *, color: bool = False) -> str:
    array = np.asarray(matrix)
    if array.size == 0:
        raise ValueError("Cannot encode an empty image")
    if array.dtype != np.uint8:
        min_value, max_value = float(array.min()), float(array.max())
        if max_value > min_value:
            array = ((array - min_value) * 255 / (max_value - min_value)).astype(np.uint8)
        else:
            array = np.zeros(array.shape, dtype=np.uint8)
    if color:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", array)
    if not success:
        raise ValueError("Unable to encode processing output")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def serialize_step(matrix: np.ndarray, display_size: tuple[int, int] | None = None) -> dict[str, object]:
    array = np.asarray(matrix)
    display = array
    if display_size is not None:
        display = cv2.resize(array, display_size, interpolation=cv2.INTER_NEAREST)
    return {
        "image": matrix_to_data_url(display),
        "shape": list(array.shape),
        "min": float(array.min()) if array.size else 0,
        "max": float(array.max()) if array.size else 0,
    }


def encode_image(matrix: np.ndarray) -> bytes:
    """Encode an image matrix for a pure-Python pipeline that accepts image bytes."""
    array = np.asarray(matrix)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2RGBA)
    success, buffer = cv2.imencode(".png", array)
    if not success:
        raise ValueError("Unable to encode processing input")
    return buffer.tobytes()


def draw_contour(image: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    """Draw a closed contour for presentation without putting drawing in the algorithm."""
    import cv2

    output = np.asarray(image).copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    output = np.clip(output, 0, 255).astype(np.uint8)
    vertices = np.rint(np.asarray(points)).astype(np.int32).reshape((-1, 1, 2))
    if len(vertices) >= 3:
        cv2.polylines(output, [vertices], True, (0, 255, 180), 2, cv2.LINE_AA)
    return output
