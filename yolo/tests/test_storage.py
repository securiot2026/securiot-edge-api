import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_app.errors import FaceAppError
from face_app.storage import FaceStorage, user_slug, validate_display_name


class StorageTests(unittest.TestCase):
    def test_slug_is_safe_and_stable(self) -> None:
        self.assertEqual(user_slug("  José / García  "), "jose-garcia")
        with self.assertRaises(FaceAppError):
            user_slug("/../")

    def test_identity_round_trip_normalizes_data(self) -> None:
        embeddings = np.array(
            [[3.0, 0.0], [1.0, 1.0]], dtype=np.float32
        )
        with tempfile.TemporaryDirectory() as directory:
            storage = FaceStorage(Path(directory))
            storage.save_identity("Ada Lovelace", embeddings)
            database, warnings = storage.load_database()
        self.assertEqual(database.names, ("Ada Lovelace",))
        self.assertEqual(warnings, [])
        self.assertAlmostEqual(
            float(np.linalg.norm(database.centroids[0])), 1.0, places=6
        )

    def test_corrupt_identity_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad_file = Path(directory) / "users" / "bad" / "identity.npz"
            bad_file.parent.mkdir(parents=True)
            bad_file.write_text("not an npz", encoding="utf-8")
            database, warnings = FaceStorage(Path(directory)).load_database()
        self.assertEqual(database.names, ())
        self.assertEqual(len(warnings), 1)

    def test_name_rejects_control_characters(self) -> None:
        with self.assertRaises(FaceAppError):
            validate_display_name("Ada\nLovelace")


if __name__ == "__main__":
    unittest.main()
