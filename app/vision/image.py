from __future__ import annotations

import numpy as np

from app.vision.errors import VisionError

BBox = tuple[int, int, int, int]


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise VisionError("ArcFace produced a zero or non-finite embedding.")
    return value / norm


def expanded_square_crop(frame: np.ndarray, box: BBox, margin: float) -> np.ndarray:
    """Return a clipped square crop around a YOLO face box."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = max(0, round(center_x - side / 2.0))
    top = max(0, round(center_y - side / 2.0))
    right = min(width, round(center_x + side / 2.0))
    bottom = min(height, round(center_y + side / 2.0))
    if right <= left or bottom <= top:
        return frame[0:0, 0:0]
    return frame[top:bottom, left:right].copy()
