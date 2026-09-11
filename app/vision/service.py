from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.vision.detector import YoloFaceDetector
from app.vision.embedder import ArcFaceEmbedder
from app.vision.errors import VisionError
from app.vision.image import BBox, expanded_square_crop
from app.vision.storage import IdentityDatabase


@dataclass(frozen=True)
class FaceRecognition:
    bbox: BBox
    detection_confidence: float
    identity: str
    similarity: float | None
    known: bool

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "detection_confidence": self.detection_confidence,
            "identity": self.identity,
            "similarity": self.similarity,
            "known": self.known,
        }


class FaceRecognitionService:
    def __init__(
        self,
        detector: YoloFaceDetector,
        embedder: ArcFaceEmbedder,
        database: IdentityDatabase,
        similarity_threshold: float,
        crop_margin: float,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.database = database
        self.similarity_threshold = similarity_threshold
        self.crop_margin = crop_margin

    def recognize_path(self, image_path: str | Path) -> list[FaceRecognition]:
        import cv2

        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise VisionError(f"OpenCV could not decode image {image_path}.")
        return self.recognize(frame)

    def recognize(self, frame: np.ndarray) -> list[FaceRecognition]:
        recognitions = []
        for detection in self.detector.detect(frame):
            crop = expanded_square_crop(frame, detection.bbox, self.crop_margin)
            embedding = self.embedder.embed(crop)
            name, similarity = self.database.match(embedding)
            known = (
                name is not None
                and similarity is not None
                and similarity >= self.similarity_threshold
            )
            recognitions.append(
                FaceRecognition(
                    bbox=detection.bbox,
                    detection_confidence=detection.confidence,
                    identity=name if known else "unknown",
                    similarity=similarity,
                    known=known,
                )
            )
        return recognitions
