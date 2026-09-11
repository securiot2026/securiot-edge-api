from __future__ import annotations

from typing import Any

import numpy as np

from .errors import FaceAppError

BBox = tuple[int, int, int, int]


def expanded_square_crop(
    frame: np.ndarray, box: BBox, margin: float
) -> np.ndarray:
    """Return a clipped square crop around a YOLO box."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = max(0, int(round(center_x - side / 2.0)))
    top = max(0, int(round(center_y - side / 2.0)))
    right = min(width, int(round(center_x + side / 2.0)))
    bottom = min(height, int(round(center_y + side / 2.0)))
    if right <= left or bottom <= top:
        return frame[0:0, 0:0]
    return frame[top:bottom, left:right].copy()


def laplacian_variance(image: np.ndarray) -> float:
    import cv2

    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def put_status(
    frame: np.ndarray, lines: list[str], color: tuple[int, int, int]
) -> None:
    import cv2

    for index, text in enumerate(lines):
        y = 30 + index * 28
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_detection(
    frame: np.ndarray,
    box: BBox,
    label: str,
    color: tuple[int, int, int],
) -> None:
    import cv2

    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_y = min(frame.shape[0] - 8, y2 + 24)
    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def show(window: str, frame: np.ndarray) -> None:
    import cv2

    try:
        cv2.imshow(window, frame)
    except cv2.error as exc:
        raise FaceAppError(
            "OpenCV could not open a GUI window. Run in a graphical desktop "
            "session and install the GUI-capable opencv-contrib-python from "
            "requirements.txt."
        ) from exc


def open_camera(index: int) -> Any:
    import cv2

    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise FaceAppError(
            f"Could not open camera index {index}. Check camera permissions, "
            "connection, and FACE_CAMERA_INDEX."
        )
    return camera
