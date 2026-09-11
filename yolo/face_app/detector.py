from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import FaceAppError
from .vision import BBox


@dataclass(frozen=True)
class Detection:
    box: BBox
    confidence: float


def preferred_torch_device() -> str:
    try:
        import torch
    except ImportError as exc:
        raise FaceAppError(
            "PyTorch is unavailable. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloFaceDetector:
    def __init__(
        self, model_path: Path, confidence: float, input_size: int
    ) -> None:
        if model_path.suffix.lower() != ".pt" or not model_path.is_file():
            raise FaceAppError(
                f"Face-specific YOLO weights were not found at {model_path}. "
                "Place an Ultralytics-compatible face detector .pt there or set "
                "FACE_YOLO_MODEL/--model. Stock COCO weights such as yolov8n.pt "
                "do not detect faces."
            )
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(model_path))
        except Exception as exc:
            raise FaceAppError(
                f"Could not load YOLO face weights from {model_path}: {exc}"
            ) from exc
        self.confidence = confidence
        self.input_size = input_size
        self.device = preferred_torch_device()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        try:
            return self._predict(frame)
        except Exception as exc:
            if self.device == "cpu":
                raise FaceAppError(
                    f"YOLO face inference failed on CPU: {exc}"
                ) from exc
            failed_device = self.device
            self.device = "cpu"
            print(
                f"Warning: YOLO inference failed on {failed_device}; "
                f"retrying on CPU: {exc}",
                file=sys.stderr,
            )
            try:
                return self._predict(frame)
            except Exception as cpu_exc:
                raise FaceAppError(
                    f"YOLO face inference also failed on CPU: {cpu_exc}"
                ) from cpu_exc

    def _predict(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []
        height, width = frame.shape[:2]
        boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        confidences = results[0].boxes.conf.detach().cpu().numpy()
        detections: list[Detection] = []
        for coordinates, confidence in zip(boxes, confidences, strict=True):
            x1, y1, x2, y2 = (
                int(round(value)) for value in coordinates.tolist()
            )
            clipped = (
                max(0, x1),
                max(0, y1),
                min(width, x2),
                min(height, y2),
            )
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                detections.append(Detection(clipped, float(confidence)))
        return detections
