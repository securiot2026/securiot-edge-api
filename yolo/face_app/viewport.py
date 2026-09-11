from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .vision import BBox


def crop_and_resize(frame: np.ndarray, box: BBox) -> np.ndarray:
    """Crop a bounded region and resize it to the original frame dimensions."""
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return frame.copy()
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


class SmoothFaceViewport:
    def __init__(
        self,
        smoothing: float = 0.18,
        face_fraction: float = 0.33,
        max_zoom: float = 2.0,
        loss_grace: int = 5,
    ) -> None:
        self.smoothing = smoothing
        self.face_fraction = face_fraction
        self.max_zoom = max_zoom
        self.loss_grace = loss_grace
        self._size = (0, 0)
        self._center = (0.0, 0.0)
        self._crop_width = 0.0
        self._target_center: tuple[float, float] | None = None
        self._losses = 0

    @property
    def crop_box(self) -> BBox:
        height, width = self._size
        if not width or not height:
            return (0, 0, 0, 0)
        crop_width = min(width, max(1, int(round(self._crop_width))))
        crop_height = min(
            height, max(1, int(round(crop_width * height / width)))
        )
        center_x, center_y = self._center
        left = min(
            width - crop_width,
            max(0, int(round(center_x - crop_width / 2))),
        )
        top = min(
            height - crop_height,
            max(0, int(round(center_y - crop_height / 2))),
        )
        return (left, top, left + crop_width, top + crop_height)

    def reset(self, frame_shape: Sequence[int]) -> None:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        self._size = (height, width)
        self._center = (width / 2, height / 2)
        self._crop_width = float(width)
        self._target_center = None
        self._losses = 0

    def update(
        self, frame_shape: Sequence[int], boxes: Sequence[BBox]
    ) -> BBox | None:
        size = (int(frame_shape[0]), int(frame_shape[1]))
        if size != self._size:
            self.reset(frame_shape)
            return None

        valid = [
            box for box in boxes if box[2] > box[0] and box[3] > box[1]
        ]
        if valid:
            if self._target_center is None:
                selected = max(
                    valid,
                    key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
                )
            else:
                tx, ty = self._target_center
                selected = min(
                    valid,
                    key=lambda box: (
                        ((box[0] + box[2]) / 2 - tx) ** 2
                        + ((box[1] + box[3]) / 2 - ty) ** 2
                    ),
                )
            center = (
                (selected[0] + selected[2]) / 2,
                (selected[1] + selected[3]) / 2,
            )
            face_size = max(
                selected[2] - selected[0], selected[3] - selected[1]
            )
            height, width = self._size
            zoom = min(
                self.max_zoom,
                max(1.0, self.face_fraction * min(width, height) / face_size),
            )
            target_width = width / zoom
            self._target_center = center
            self._losses = 0
        else:
            self._losses += 1
            if self._losses <= self.loss_grace:
                return None
            height, width = self._size
            center = (width / 2, height / 2)
            target_width = float(width)
            self._target_center = None
            selected = None

        alpha = self.smoothing
        self._center = tuple(
            current + alpha * (target - current)
            for current, target in zip(self._center, center)
        )
        self._crop_width += alpha * (target_width - self._crop_width)
        return selected

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] != self._size:
            self.reset(frame.shape)
        return crop_and_resize(frame, self.crop_box)
