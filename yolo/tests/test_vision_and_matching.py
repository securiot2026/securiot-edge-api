import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from face_app.recognition import match_embedding
from face_app.storage import IdentityDatabase
from face_app.vision import expanded_square_crop, laplacian_variance


class VisionAndMatchingTests(unittest.TestCase):
    def test_square_crop_clips_to_frame(self) -> None:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        crop = expanded_square_crop(frame, (0, 10, 30, 50), margin=0.25)
        self.assertGreater(crop.size, 0)
        self.assertLessEqual(crop.shape[0], 80)
        self.assertLessEqual(crop.shape[1], 100)

    def test_laplacian_variance_uses_grayscale_variance(self) -> None:
        fake_cv2 = types.SimpleNamespace(
            COLOR_BGR2GRAY=1,
            CV_64F=2,
            cvtColor=lambda image, code: image[:, :, 0],
            Laplacian=lambda gray, depth: gray.astype(float),
        )
        image = np.array([[[0], [10]], [[20], [30]]], dtype=np.uint8)
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            self.assertEqual(laplacian_variance(image), 125.0)

    def test_matching_is_vectorized_cosine_similarity(self) -> None:
        database = IdentityDatabase(
            ("Ada", "Grace"),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        name, similarity = match_embedding(np.array([0.1, 0.9]), database)
        self.assertEqual(name, "Grace")
        self.assertGreater(similarity, 0.99)


if __name__ == "__main__":
    unittest.main()
