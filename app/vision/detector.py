from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.vision.errors import VisionError
from app.vision.image import BBox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceDetection:
    bbox: BBox
    confidence: float


def resolve_yolo_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError as exc:
        raise VisionError("PyTorch is required by the YOLO face detector.") from exc
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YoloFaceDetector:
    """Face-only Ultralytics YOLO detector."""

    def __init__(
        self,
        model_path: Path,
        confidence: float,
        iou: float,
        input_size: int,
        max_detections: int,
        class_id: int,
        device: str,
        model: Any | None = None,
    ) -> None:
        if model is None:
            if model_path.suffix.lower() != ".pt" or not model_path.is_file():
                raise VisionError(
                    "Face-specific YOLO weights were not found at "
                    f"{model_path}. Set FACE_YOLO_MODEL_PATH to an "
                    "Ultralytics-compatible face detector; stock COCO weights "
                    "do not detect faces."
                )
            try:
                from ultralytics import YOLO

                model = YOLO(str(model_path))
            except Exception as exc:
                raise VisionError(
                    f"Could not load YOLO face weights from {model_path}: {exc}"
                ) from exc
        self._model = model
        self.confidence = confidence
        self.iou = iou
        self.input_size = input_size
        self.max_detections = max_detections
        self.class_id = class_id
        self.device = resolve_yolo_device(device)
        self._prediction_lock = threading.Lock()

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if frame.size == 0:
            return []
        try:
            return self._predict(frame)
        except Exception as exc:
            if self.device == "cpu":
                raise VisionError(f"YOLO face inference failed on CPU: {exc}") from exc
            failed_device = self.device
            self.device = "cpu"
            logger.warning(
                "YOLO face inference failed on %s; retrying on CPU: %s",
                failed_device,
                exc,
            )
            try:
                return self._predict(frame)
            except Exception as cpu_exc:
                raise VisionError(
                    f"YOLO face inference also failed on CPU: {cpu_exc}"
                ) from cpu_exc

    def _predict(self, frame: np.ndarray) -> list[FaceDetection]:
        with self._prediction_lock:
            results = self._model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.input_size,
                max_det=self.max_detections,
                classes=[self.class_id],
                device=self.device,
                verbose=False,
            )
        if not results or results[0].boxes is None:
            return []

        height, width = frame.shape[:2]
        boxes = _as_numpy(results[0].boxes.xyxy)
        confidences = _as_numpy(results[0].boxes.conf)
        detections = []
        for coordinates, confidence in zip(boxes, confidences, strict=True):
            x1, y1, x2, y2 = (
                round(float(value)) for value in coordinates.tolist()
            )
            clipped = (
                max(0, x1),
                max(0, y1),
                min(width, x2),
                min(height, y2),
            )
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                detections.append(FaceDetection(clipped, float(confidence)))
        return detections
