from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.vision.errors import VisionError


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise VisionError(f"Invalid boolean value: {value!r}.")


@dataclass(frozen=True)
class VisionSettings:
    enabled: bool
    yolo_model_path: Path
    yolo_confidence: float
    yolo_iou: float
    yolo_input_size: int
    yolo_max_detections: int
    yolo_class_id: int
    yolo_device: str
    arcface_model_path: Path
    arcface_device: str
    identity_dir: Path
    similarity_threshold: float
    crop_margin: float
    minimum_enrollment_images: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> VisionSettings:
        settings = cls(
            enabled=_as_bool(config.get("FACE_RECOGNITION_ENABLED", False)),
            yolo_model_path=Path(
                str(config.get("FACE_YOLO_MODEL_PATH", "models/yolov8n-face.pt"))
            ).expanduser(),
            yolo_confidence=float(config.get("FACE_YOLO_CONFIDENCE_THRESHOLD", 0.5)),
            yolo_iou=float(config.get("FACE_YOLO_IOU_THRESHOLD", 0.45)),
            yolo_input_size=int(config.get("FACE_YOLO_INPUT_SIZE", 640)),
            yolo_max_detections=int(config.get("FACE_YOLO_MAX_DETECTIONS", 10)),
            yolo_class_id=int(config.get("FACE_YOLO_CLASS_ID", 0)),
            yolo_device=str(config.get("FACE_YOLO_DEVICE", "cpu")).strip().lower(),
            arcface_model_path=Path(
                str(config.get("ARCFACE_MODEL_PATH", "models/w600k_r50.onnx"))
            ).expanduser(),
            arcface_device=str(config.get("ARCFACE_DEVICE", "cpu")).strip().lower(),
            identity_dir=Path(
                str(config.get("FACE_IDENTITY_DIR", "data/faces"))
            ).expanduser(),
            similarity_threshold=float(config.get("FACE_SIMILARITY_THRESHOLD", 0.45)),
            crop_margin=float(config.get("FACE_CROP_MARGIN", 0.2)),
            minimum_enrollment_images=int(
                config.get("FACE_MINIMUM_ENROLLMENT_IMAGES", 3)
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        for name, value in (
            ("FACE_YOLO_CONFIDENCE_THRESHOLD", self.yolo_confidence),
            ("FACE_YOLO_IOU_THRESHOLD", self.yolo_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise VisionError(f"{name} must be between 0 and 1.")
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise VisionError("FACE_SIMILARITY_THRESHOLD must be between -1 and 1.")
        if self.crop_margin < 0.0:
            raise VisionError("FACE_CROP_MARGIN cannot be negative.")
        if self.yolo_input_size < 32:
            raise VisionError("FACE_YOLO_INPUT_SIZE must be at least 32.")
        if self.yolo_max_detections < 1:
            raise VisionError("FACE_YOLO_MAX_DETECTIONS must be at least 1.")
        if self.yolo_class_id < 0:
            raise VisionError("FACE_YOLO_CLASS_ID cannot be negative.")
        if self.minimum_enrollment_images < 1:
            raise VisionError("FACE_MINIMUM_ENROLLMENT_IMAGES must be at least 1.")
        if not self.yolo_device:
            raise VisionError("FACE_YOLO_DEVICE cannot be empty.")
        if self.arcface_device not in {"auto", "cpu", "cuda", "coreml"}:
            raise VisionError(
                "ARCFACE_DEVICE must be one of: auto, cpu, cuda, coreml."
            )
