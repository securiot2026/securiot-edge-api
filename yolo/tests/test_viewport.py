import unittest

import numpy as np

from face_app.detector import Detection
from face_app.recognition import known_face_boxes
from face_app.viewport import SmoothFaceViewport, crop_and_resize


class SmoothFaceViewportTests(unittest.TestCase):
    shape = (100, 200, 3)

    def test_starts_full_then_converges_without_exceeding_zoom_cap(self) -> None:
        viewport = SmoothFaceViewport(smoothing=0.2)
        viewport.update(self.shape, [(95, 45, 105, 55)])
        self.assertEqual(viewport.crop_box, (0, 0, 200, 100))
        for _ in range(100):
            viewport.update(self.shape, [(95, 45, 105, 55)])
        left, top, right, bottom = viewport.crop_box
        self.assertEqual((right - left, bottom - top), (100, 50))

    def test_center_and_size_smoothing_reduce_jitter(self) -> None:
        viewport = SmoothFaceViewport(smoothing=0.2)
        viewport.update(self.shape, [])
        viewport.update(self.shape, [(20, 30, 60, 70)])
        first_left = viewport.crop_box[0]
        viewport.update(self.shape, [(140, 30, 180, 70)])
        self.assertLess(viewport.crop_box[0] - first_left, 120)
        self.assertGreater(viewport.crop_box[2] - viewport.crop_box[0], 100)

    def test_crop_stays_bounded_and_preserves_aspect_at_edges(self) -> None:
        viewport = SmoothFaceViewport(smoothing=1.0)
        viewport.update(self.shape, [])
        viewport.update(self.shape, [(0, 0, 10, 10)])
        self.assertEqual(viewport.crop_box, (0, 0, 100, 50))
        left, top, right, bottom = viewport.crop_box
        self.assertEqual((right - left) / (bottom - top), 2.0)

    def test_loss_grace_holds_then_smoothly_returns_to_full_frame(self) -> None:
        viewport = SmoothFaceViewport(smoothing=0.5, loss_grace=5)
        viewport.update(self.shape, [])
        viewport.update(self.shape, [(90, 40, 110, 60)])
        held = viewport.crop_box
        for _ in range(5):
            viewport.update(self.shape, [])
        self.assertEqual(viewport.crop_box, held)
        viewport.update(self.shape, [])
        self.assertGreater(
            viewport.crop_box[2] - viewport.crop_box[0], held[2] - held[0]
        )

    def test_selection_uses_largest_then_nearest_until_continuity_clears(
        self,
    ) -> None:
        small = (10, 10, 30, 30)
        large = (120, 20, 180, 80)
        viewport = SmoothFaceViewport(loss_grace=1)
        viewport.update(self.shape, [])
        self.assertEqual(viewport.update(self.shape, [small, large]), large)
        near = (125, 25, 145, 45)
        farther_but_larger = (0, 0, 80, 80)
        self.assertEqual(
            viewport.update(self.shape, [farther_but_larger, near]), near
        )
        viewport.update(self.shape, [])
        viewport.update(self.shape, [])
        selected = viewport.update(
            self.shape, [near, farther_but_larger]
        )
        self.assertEqual(selected, farther_but_larger)

    def test_crop_resize_keeps_output_dimensions_and_resets_on_size_change(
        self,
    ) -> None:
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(self.shape)
        self.assertEqual(
            crop_and_resize(frame, (50, 25, 150, 75)).shape, self.shape
        )
        viewport = SmoothFaceViewport(smoothing=1.0)
        viewport.update(self.shape, [])
        viewport.update(self.shape, [(90, 40, 110, 60)])
        resized = np.zeros((80, 120, 3), dtype=np.uint8)
        self.assertEqual(viewport.apply(resized).shape, resized.shape)
        self.assertEqual(viewport.crop_box, (0, 0, 120, 80))

    def test_unknown_annotations_are_excluded_from_viewport_targets(self) -> None:
        known = Detection((10, 10, 30, 30), 0.9)
        unknown = Detection((100, 10, 180, 90), 0.9)
        annotations = [
            (known, "Ada", 0.8, True),
            (unknown, "DESCONOCIDO", 0.2, False),
        ]
        self.assertEqual(known_face_boxes(annotations), [known.box])


if __name__ == "__main__":
    unittest.main()
