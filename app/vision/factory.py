from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any

from app.vision.detector import YoloFaceDetector
from app.vision.embedder import ArcFaceEmbedder
from app.vision.service import FaceRecognitionService
from app.vision.settings import VisionSettings
from app.vision.storage import FaceIdentityStorage

logger = logging.getLogger(__name__)

_recognizer_cache: dict[VisionSettings, FaceRecognitionService] = {}
_cache_lock = threading.Lock()


def get_face_recognizer(config: Mapping[str, Any]) -> FaceRecognitionService:
    settings = VisionSettings.from_mapping(config)
    with _cache_lock:
        recognizer = _recognizer_cache.get(settings)
        if recognizer is None:
            recognizer = _build_face_recognizer(settings)
            _recognizer_cache[settings] = recognizer
        return recognizer


def _build_face_recognizer(settings: VisionSettings) -> FaceRecognitionService:
    detector = YoloFaceDetector(
        model_path=settings.yolo_model_path,
        confidence=settings.yolo_confidence,
        iou=settings.yolo_iou,
        input_size=settings.yolo_input_size,
        max_detections=settings.yolo_max_detections,
        class_id=settings.yolo_class_id,
        device=settings.yolo_device,
    )
    embedder = ArcFaceEmbedder(
        model_path=settings.arcface_model_path,
        device=settings.arcface_device,
    )
    database, warnings = FaceIdentityStorage(settings.identity_dir).load()
    for warning in warnings:
        logger.warning(warning)
    return FaceRecognitionService(
        detector=detector,
        embedder=embedder,
        database=database,
        similarity_threshold=settings.similarity_threshold,
        crop_margin=settings.crop_margin,
    )


def reset_face_recognizer_cache() -> None:
    """Clear cached model instances after enrollment or in tests."""
    with _cache_lock:
        _recognizer_cache.clear()
