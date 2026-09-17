from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.vision.detector import YoloFaceDetector
from app.vision.embedder import ArcFaceEmbedder
from app.vision.errors import VisionError
from app.vision.factory import reset_face_recognizer_cache
from app.vision.image import expanded_square_crop, laplacian_variance
from app.vision.settings import VisionSettings
from app.vision.storage import FaceIdentityStorage, validate_display_name


@dataclass(frozen=True)
class EnrollmentResult:
    identity_path: Path
    accepted_images: int
    skipped_images: tuple[str, ...]


def validate_capture_quality(
    image_paths: list[Path],
    settings: VisionSettings,
) -> tuple[list[Path], list[str]]:
    import cv2

    detector = YoloFaceDetector(
        model_path=settings.yolo_model_path,
        confidence=settings.yolo_confidence,
        iou=settings.yolo_iou,
        input_size=settings.yolo_input_size,
        max_detections=settings.yolo_max_detections,
        class_id=settings.yolo_class_id,
        device=settings.yolo_device,
    )

    valid_paths = []
    skipped = []
    for image_path in image_paths:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            skipped.append(f"{image_path.name}: unreadable image")
            continue

        detections = detector.detect(frame)
        if len(detections) == 0:
            skipped.append(f"{image_path.name}: No face detected")
            continue
        if len(detections) > 1:
            skipped.append(f"{image_path.name}: Exactly one face is required")
            continue

        x1, y1, x2, y2 = detections[0].bbox
        if min(x2 - x1, y2 - y1) < settings.minimum_face_size:
            skipped.append(f"{image_path.name}: Move closer to the camera")
            continue

        crop = expanded_square_crop(frame, detections[0].bbox, settings.crop_margin)
        if laplacian_variance(crop) < settings.blur_threshold:
            skipped.append(f"{image_path.name}: Image is too blurry; hold still")
            continue

        valid_paths.append(image_path)

    return valid_paths, skipped


def enroll_identity(
    name: str,
    image_paths: list[Path],
    settings: VisionSettings,
) -> EnrollmentResult:
    import cv2

    display_name = validate_display_name(name)
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

    embeddings = []
    skipped = []
    for image_path in image_paths:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            skipped.append(f"{image_path}: unreadable image")
            continue
        detections = detector.detect(frame)
        if len(detections) != 1:
            skipped.append(
                f"{image_path}: expected one face, detected {len(detections)}"
            )
            continue
        crop = expanded_square_crop(
            frame, detections[0].bbox, settings.crop_margin
        )
        embeddings.append(embedder.embed(crop))

    if len(embeddings) < settings.minimum_enrollment_images:
        raise VisionError(
            f"Only {len(embeddings)} valid enrollment images were found; "
            f"at least {settings.minimum_enrollment_images} are required."
        )

    identity_path = FaceIdentityStorage(settings.identity_dir).save(
        display_name, np.vstack(embeddings)
    )
    reset_face_recognizer_cache()
    return EnrollmentResult(identity_path, len(embeddings), tuple(skipped))
