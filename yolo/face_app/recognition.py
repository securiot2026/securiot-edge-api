from __future__ import annotations

import sys

import numpy as np

from .config import Settings
from .detector import Detection, YoloFaceDetector
from .embedder import ArcFaceEmbedder, normalize
from .errors import FaceAppError
from .storage import FaceStorage, IdentityDatabase
from .viewport import SmoothFaceViewport
from .vision import (
    draw_detection,
    expanded_square_crop,
    open_camera,
    put_status,
    show,
)


def match_embedding(
    embedding: np.ndarray, database: IdentityDatabase
) -> tuple[str, float]:
    if not database.names or database.centroids.size == 0:
        raise ValueError("Identity database is empty.")
    similarities = database.centroids @ normalize(embedding)
    best_index = int(np.argmax(similarities))
    return database.names[best_index], float(similarities[best_index])


def known_face_boxes(
    annotations: list[tuple[Detection, str, float, bool]],
) -> list[tuple[int, int, int, int]]:
    return [
        detection.box for detection, _, _, known in annotations if known
    ]


def recognize(settings: Settings) -> None:
    import cv2

    storage = FaceStorage(settings.data_dir)
    database, warnings = storage.load_database()
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if not database.names:
        raise FaceAppError(
            "No valid registered users were found. Run: "
            "python main.py register --name 'Name'"
        )
    detector = YoloFaceDetector(
        settings.model_path,
        settings.detector_confidence,
        settings.detector_input_size,
    )
    embedder = ArcFaceEmbedder()
    camera = open_camera(settings.camera_index)
    frame_number = 0
    annotations: list[tuple[Detection, str, float, bool]] = []
    viewport = SmoothFaceViewport()
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise FaceAppError(
                    "Camera stopped returning frames. Check the connection "
                    "and camera permissions."
                )
            if frame_number % settings.frame_stride == 0:
                annotations = []
                for detection in detector.detect(frame):
                    crop = expanded_square_crop(
                        frame, detection.box, settings.crop_margin
                    )
                    try:
                        name, similarity = match_embedding(
                            embedder.embed(crop), database
                        )
                    except (FaceAppError, ValueError) as exc:
                        print(
                            f"Warning: skipping one face: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    known = similarity >= settings.similarity_threshold
                    annotations.append(
                        (
                            detection,
                            name if known else "DESCONOCIDO",
                            similarity,
                            known,
                        )
                    )
                viewport.update(frame.shape, known_face_boxes(annotations))
            frame_number += 1
            display = frame.copy()
            for detection, name, similarity, known in annotations:
                color = (40, 210, 40) if known else (40, 40, 230)
                draw_detection(
                    display,
                    detection.box,
                    f"{name} {similarity:.2f}",
                    color,
                )
            display = viewport.apply(display)
            put_status(
                display,
                [
                    f"Threshold: {settings.similarity_threshold:.2f}",
                    "Press q to quit",
                ],
                (0, 220, 255),
            )
            show("Face recognition", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
    finally:
        camera.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
