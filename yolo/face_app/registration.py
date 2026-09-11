from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .detector import YoloFaceDetector
from .embedder import ArcFaceEmbedder
from .errors import FaceAppError
from .storage import FaceStorage, validate_display_name
from .vision import (
    draw_detection,
    expanded_square_crop,
    laplacian_variance,
    open_camera,
    put_status,
    show,
)

PROMPTS = (
    "Look straight at the camera",
    "Turn slightly to your left",
    "Turn slightly to your right",
    "Tilt your chin slightly up",
    "Tilt your chin slightly down",
    "Smile naturally",
)


@dataclass(frozen=True)
class RegistrationResult:
    completed: bool
    accepted_photos: int
    embeddings: int
    capture_dir: Path


def register_user(name: str, settings: Settings) -> RegistrationResult:
    import cv2

    display_name = validate_display_name(name)
    detector = YoloFaceDetector(
        settings.model_path,
        settings.detector_confidence,
        settings.detector_input_size,
    )
    storage = FaceStorage(settings.data_dir)
    capture_dir = storage.create_capture_dir(display_name)
    camera = open_camera(settings.camera_index)
    accepted_paths: list[Path] = []
    last_capture = 0.0
    rejection = "Center one face in the frame"
    cancelled = False
    read_failed = False
    try:
        while len(accepted_paths) < settings.photo_count:
            ok, frame = camera.read()
            if not ok or frame is None:
                read_failed = True
                break
            detections = detector.detect(frame)
            rejection = _rejection_reason(frame, detections, settings)
            now = time.monotonic()
            if (
                rejection is None
                and now - last_capture >= settings.capture_cooldown
            ):
                photo_path = (
                    capture_dir / f"photo-{len(accepted_paths) + 1:03d}.jpg"
                )
                if not cv2.imwrite(str(photo_path), frame):
                    raise FaceAppError(
                        f"OpenCV could not write captured photo to {photo_path}."
                    )
                accepted_paths.append(photo_path)
                last_capture = now
                rejection = "Captured"
            display = frame.copy()
            for detection in detections:
                draw_detection(
                    display,
                    detection.box,
                    f"face {detection.confidence:.2f}",
                    (0, 200, 255),
                )
            prompt_index = min(
                len(PROMPTS) - 1,
                len(accepted_paths) * len(PROMPTS) // settings.photo_count,
            )
            put_status(
                display,
                [
                    f"Registration: {display_name}",
                    f"Valid photos: {len(accepted_paths)}/{settings.photo_count}",
                    PROMPTS[prompt_index],
                    rejection or "Hold still",
                    "Press q to cancel",
                ],
                (30, 230, 30),
            )
            show("Face registration", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cancelled = True
                break
    finally:
        camera.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if read_failed:
        raise FaceAppError(
            "Camera stopped returning frames. "
            f"{len(accepted_paths)} partial photos remain in {capture_dir}."
        )
    if cancelled or len(accepted_paths) < settings.photo_count:
        return RegistrationResult(False, len(accepted_paths), 0, capture_dir)

    embedder = ArcFaceEmbedder()
    embeddings: list[np.ndarray] = []
    for photo_path in accepted_paths:
        image = cv2.imread(str(photo_path))
        if image is None:
            print(
                f"Warning: skipping unreadable photo {photo_path}",
                file=sys.stderr,
            )
            continue
        detections = detector.detect(image)
        if len(detections) != 1:
            print(
                f"Warning: skipping {photo_path}; YOLO found "
                f"{len(detections)} faces during processing.",
                file=sys.stderr,
            )
            continue
        crop = expanded_square_crop(
            image, detections[0].box, settings.crop_margin
        )
        try:
            embeddings.append(embedder.embed(crop))
        except (FaceAppError, ValueError) as exc:
            print(
                f"Warning: skipping embedding for {photo_path}: {exc}",
                file=sys.stderr,
            )
    if len(embeddings) < settings.minimum_embeddings:
        raise FaceAppError(
            f"Only {len(embeddings)} valid embeddings were produced; at least "
            f"{settings.minimum_embeddings} are required. Photos remain in "
            f"{capture_dir}; the identity database was not changed."
        )
    storage.save_identity(display_name, np.vstack(embeddings))
    return RegistrationResult(
        True, len(accepted_paths), len(embeddings), capture_dir
    )


def _rejection_reason(
    frame: np.ndarray, detections: list, settings: Settings
) -> str | None:
    if not detections:
        return "No face detected"
    if len(detections) != 1:
        return "Exactly one face is required"
    x1, y1, x2, y2 = detections[0].box
    if min(x2 - x1, y2 - y1) < settings.minimum_face_size:
        return "Move closer to the camera"
    crop = expanded_square_crop(
        frame, detections[0].box, settings.crop_margin
    )
    if laplacian_variance(crop) < settings.blur_threshold:
        return "Image is too blurry; hold still"
    return None
