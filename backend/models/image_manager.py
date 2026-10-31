from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

import numpy as np


@dataclass
class ImageSession:
    original: np.ndarray
    cache: dict[str, object] = field(default_factory=dict)


class ImageManager:
    """Owns uploaded matrices and reusable intermediate results for one process."""

    def __init__(self) -> None:
        self._sessions: dict[str, ImageSession] = {}
        self._lock = RLock()

    def create(self, image: np.ndarray) -> str:
        image_id = str(uuid4())
        with self._lock:
            self._sessions[image_id] = ImageSession(original=image.copy())
        return image_id

    def get(self, image_id: str) -> ImageSession:
        with self._lock:
            try:
                return self._sessions[image_id]
            except KeyError as exc:
                raise KeyError("Image session was not found or has expired") from exc

    def get_cached(self, image_id: str, key: str) -> object | None:
        return self.get(image_id).cache.get(key)

    def cache(self, image_id: str, key: str, value: object) -> object:
        with self._lock:
            self.get(image_id).cache[key] = value
        return value


image_manager = ImageManager()
