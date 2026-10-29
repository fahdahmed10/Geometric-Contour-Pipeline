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


def serialize_step(matrix: np.ndarray) -> dict[str, object]:
    array = np.asarray(matrix)
    return {
        "image": matrix_to_data_url(array),
        "shape": list(array.shape),
        "min": float(array.min()) if array.size else 0,
        "max": float(array.max()) if array.size else 0,
    }
