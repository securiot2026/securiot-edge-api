"""Isolated YOLO face-detection and ArcFace recognition module."""

from app.vision.errors import VisionError
from app.vision.factory import get_face_recognizer
from app.vision.service import FaceRecognition, FaceRecognitionService
from app.vision.settings import VisionSettings

__all__ = [
    "FaceRecognition",
    "FaceRecognitionService",
    "VisionError",
    "VisionSettings",
    "get_face_recognizer",
]
