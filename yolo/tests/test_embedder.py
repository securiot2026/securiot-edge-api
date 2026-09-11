import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2  # noqa: F401 - keep the native extension loaded while patching modules
import numpy as np

from face_app.embedder import ArcFaceEmbedder
from face_app.errors import FaceAppError


class EmbedderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Mock(taskname="recognition", input_size=(112, 112))
        self.model.get_feat.return_value = np.array(
            [[3.0, 4.0]], dtype=np.float32
        )
        self.get_model = Mock(return_value=self.model)
        self.model_dir = Path("models") / "buffalo_l"
        self.ensure_available = Mock(return_value=str(self.model_dir))
        modules = {
            "insightface": SimpleNamespace(),
            "insightface.model_zoo": SimpleNamespace(get_model=self.get_model),
            "insightface.utils.storage": SimpleNamespace(
                ensure_available=self.ensure_available
            ),
        }
        for patcher in (
            patch.dict(sys.modules, modules),
            patch("pathlib.Path.is_file", return_value=True),
            patch(
                "face_app.embedder.preferred_onnx_providers",
                return_value=["CPUExecutionProvider"],
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_loads_only_explicit_recognition_model(self) -> None:
        ArcFaceEmbedder()

        self.ensure_available.assert_called_once_with("models", "buffalo_l")
        self.get_model.assert_called_once_with(
            str(self.model_dir / "w600k_r50.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.model.prepare.assert_called_once_with(ctx_id=0)

    def test_missing_recognition_file_reports_path(self) -> None:
        with patch("pathlib.Path.is_file", return_value=False):
            with self.assertRaisesRegex(FaceAppError, "w600k_r50.onnx"):
                ArcFaceEmbedder()
        self.get_model.assert_not_called()

    def test_rejects_missing_or_wrong_model_type(self) -> None:
        for model in (None, Mock(taskname="detection")):
            with self.subTest(model=model):
                self.get_model.return_value = model
                with self.assertRaisesRegex(FaceAppError, "recognition"):
                    ArcFaceEmbedder()
        self.assertEqual(self.get_model.call_count, 2)

    def test_download_error_is_actionable(self) -> None:
        self.ensure_available.side_effect = OSError("download unavailable")
        with self.assertRaisesRegex(FaceAppError, "download unavailable"):
            ArcFaceEmbedder()

    def test_accelerator_initialization_retries_on_cpu(self) -> None:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.get_model.side_effect = [
            RuntimeError("accelerator unavailable"),
            self.model,
        ]
        with patch(
            "face_app.embedder.preferred_onnx_providers",
            return_value=providers,
        ):
            embedder = ArcFaceEmbedder()
        model_path = str(self.model_dir / "w600k_r50.onnx")
        self.assertEqual(
            self.get_model.call_args_list,
            [
                call(model_path, providers=providers),
                call(model_path, providers=["CPUExecutionProvider"]),
            ],
        )
        self.assertEqual(embedder.providers, ["CPUExecutionProvider"])
        self.model.prepare.assert_called_once_with(ctx_id=0)

    def test_preserves_bgr_input_and_normalizes_output(self) -> None:
        crop = np.full((80, 100, 3), (10, 20, 30), dtype=np.uint8)
        result = ArcFaceEmbedder().embed(crop)
        resized = self.model.get_feat.call_args.args[0]
        self.assertEqual(resized.shape, (112, 112, 3))
        np.testing.assert_array_equal(resized[0, 0], [10, 20, 30])
        np.testing.assert_allclose(result, [0.6, 0.8])

    def test_accelerator_inference_retries_on_cpu(self) -> None:
        accelerated_model = Mock(taskname="recognition", input_size=(112, 112))
        accelerated_model.get_feat.side_effect = RuntimeError(
            "accelerator inference failed"
        )
        self.get_model.side_effect = [accelerated_model, self.model]
        with patch(
            "face_app.embedder.preferred_onnx_providers",
            return_value=[
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
        ):
            embedder = ArcFaceEmbedder()
        result = embedder.embed(
            np.ones((112, 112, 3), dtype=np.uint8)
        )
        np.testing.assert_allclose(result, [0.6, 0.8])
        self.assertEqual(embedder.providers, ["CPUExecutionProvider"])
        self.assertEqual(self.get_model.call_count, 2)

    def test_cpu_initialization_failure_is_not_retried(self) -> None:
        self.get_model.side_effect = RuntimeError("invalid ONNX")
        with self.assertRaisesRegex(FaceAppError, "invalid ONNX"):
            ArcFaceEmbedder()
        self.assertEqual(self.get_model.call_count, 1)

    def test_cpu_inference_failure_is_not_retried(self) -> None:
        self.model.get_feat.side_effect = RuntimeError("CPU unavailable")
        embedder = ArcFaceEmbedder()
        with self.assertRaisesRegex(FaceAppError, "inference failed on CPU"):
            embedder.embed(np.ones((112, 112, 3), dtype=np.uint8))
        self.assertEqual(self.get_model.call_count, 1)
        self.model.get_feat.assert_called_once()

    def test_empty_crop_is_rejected_before_inference(self) -> None:
        embedder = ArcFaceEmbedder()
        with self.assertRaisesRegex(ValueError, "empty face crop"):
            embedder.embed(np.empty((0, 0, 3), dtype=np.uint8))
        self.model.get_feat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
