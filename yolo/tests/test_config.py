import os
import unittest
from unittest.mock import patch

from face_app.config import Settings
from face_app.errors import FaceAppError


class SettingsTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.photo_count, 30)
        self.assertEqual(settings.frame_stride, 1)
        self.assertEqual(settings.similarity_threshold, 0.45)

    def test_minimum_embeddings_cannot_exceed_photo_count(self) -> None:
        environment = {
            "FACE_PHOTO_COUNT": "5",
            "FACE_MINIMUM_EMBEDDINGS": "6",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(FaceAppError, "cannot exceed"):
                Settings.from_env()

    def test_invalid_numeric_environment_value_is_actionable(self) -> None:
        with patch.dict(
            os.environ, {"FACE_FRAME_STRIDE": "fast"}, clear=True
        ):
            with self.assertRaisesRegex(
                FaceAppError, "FACE_FRAME_STRIDE must be an integer"
            ):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
