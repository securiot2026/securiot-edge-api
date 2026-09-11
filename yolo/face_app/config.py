from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import FaceAppError


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise FaceAppError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < minimum:
        raise FaceAppError(f"{name} must be at least {minimum}, got {value}.")
    return value


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise FaceAppError(f"{name} must be a number, got {raw!r}.") from exc
    if value < minimum:
        raise FaceAppError(f"{name} must be at least {minimum}, got {value}.")
    return value


@dataclass(frozen=True)
class Settings:
    model_path: Path
    data_dir: Path
    camera_index: int
    detector_confidence: float
    detector_input_size: int
    similarity_threshold: float
    photo_count: int
    minimum_embeddings: int
    blur_threshold: float
    minimum_face_size: int
    capture_cooldown: float
    crop_margin: float
    frame_stride: int

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            model_path=Path(
                os.getenv("FACE_YOLO_MODEL", "models/yolov8n-face.pt")
            ),
            data_dir=Path(os.getenv("FACE_DATA_DIR", "data")),
            camera_index=_env_int("FACE_CAMERA_INDEX", 0),
            detector_confidence=_env_float("FACE_DETECTOR_CONFIDENCE", 0.5),
            detector_input_size=_env_int("FACE_DETECTOR_INPUT_SIZE", 640, 32),
            similarity_threshold=_env_float("FACE_SIMILARITY_THRESHOLD", 0.45),
            photo_count=_env_int("FACE_PHOTO_COUNT", 30, 1),
            minimum_embeddings=_env_int("FACE_MINIMUM_EMBEDDINGS", 10, 1),
            blur_threshold=_env_float("FACE_BLUR_THRESHOLD", 100.0),
            minimum_face_size=_env_int("FACE_MINIMUM_FACE_SIZE", 80, 1),
            capture_cooldown=_env_float("FACE_CAPTURE_COOLDOWN", 0.35),
            crop_margin=_env_float("FACE_CROP_MARGIN", 0.2),
            frame_stride=_env_int("FACE_FRAME_STRIDE", 1, 1),
        )
        if not 0.0 <= settings.detector_confidence <= 1.0:
            raise FaceAppError(
                "FACE_DETECTOR_CONFIDENCE must be between 0 and 1."
            )
        if not -1.0 <= settings.similarity_threshold <= 1.0:
            raise FaceAppError(
                "FACE_SIMILARITY_THRESHOLD must be between -1 and 1."
            )
        if settings.minimum_embeddings > settings.photo_count:
            raise FaceAppError(
                "FACE_MINIMUM_EMBEDDINGS cannot exceed FACE_PHOTO_COUNT."
            )
        return settings
