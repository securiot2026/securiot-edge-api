from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.vision.detector import FaceDetection, YoloFaceDetector
from app.vision.embedder import ArcFaceEmbedder, select_onnx_providers
from app.vision.errors import VisionError
from app.vision.service import FaceRecognitionService
from app.vision.settings import VisionSettings
from app.vision.storage import FaceIdentityStorage, IdentityDatabase


def _settings(**overrides):
    config = {
        "FACE_RECOGNITION_ENABLED": True,
        "FACE_YOLO_MODEL_PATH": "models/face.pt",
        "ARCFACE_MODEL_PATH": "models/arcface.onnx",
    }
    config.update(overrides)
    return VisionSettings.from_mapping(config)


def test_vision_settings_have_cpu_defaults_and_validate_thresholds():
    settings = _settings()

    assert settings.yolo_device == "cpu"
    assert settings.arcface_device == "cpu"

    with pytest.raises(VisionError, match="between 0 and 1"):
        _settings(FACE_YOLO_CONFIDENCE_THRESHOLD=1.1)


def test_yolo_face_detector_passes_configuration_and_clips_boxes():
    model = Mock()
    model.predict.return_value = [
        SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=np.array([[-5.0, 10.0, 110.0, 90.0]], dtype=np.float32),
                conf=np.array([0.91], dtype=np.float32),
            )
        )
    ]
    detector = YoloFaceDetector(
        model_path=Path("unused.pt"),
        confidence=0.6,
        iou=0.4,
        input_size=320,
        max_detections=4,
        class_id=0,
        device="cpu",
        model=model,
    )
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert detections == [FaceDetection((0, 10, 100, 80), pytest.approx(0.91))]
    model.predict.assert_called_once_with(
        source=frame,
        conf=0.6,
        iou=0.4,
        imgsz=320,
        max_det=4,
        classes=[0],
        device="cpu",
        verbose=False,
    )


def test_arcface_embedder_runs_onnx_and_normalizes_output(tmp_path):
    model_path = tmp_path / "arcface.onnx"
    model_path.write_bytes(b"model")
    session = Mock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="input", shape=[1, 3, 112, 112])
    ]
    session.get_outputs.return_value = [SimpleNamespace(name="embedding")]
    session.run.return_value = [np.array([[3.0, 4.0]], dtype=np.float32)]
    factory = Mock(return_value=session)
    embedder = ArcFaceEmbedder(
        model_path=model_path,
        device="cpu",
        session_factory=factory,
        available_providers=["CPUExecutionProvider"],
    )

    embedding = embedder.embed(np.full((80, 100, 3), 20, dtype=np.uint8))

    np.testing.assert_allclose(embedding, [0.6, 0.8])
    tensor = session.run.call_args.args[1]["input"]
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32


def test_arcface_cuda_requires_gpu_runtime():
    with pytest.raises(VisionError, match="onnxruntime-gpu"):
        select_onnx_providers("cuda", ["CPUExecutionProvider"])


def test_arcface_initialization_falls_back_to_cpu(tmp_path):
    model_path = tmp_path / "arcface.onnx"
    model_path.write_bytes(b"model")
    session = Mock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="input", shape=[1, 3, 112, 112])
    ]
    session.get_outputs.return_value = [SimpleNamespace(name="embedding")]
    factory = Mock(side_effect=[RuntimeError("CUDA unavailable"), session])

    embedder = ArcFaceEmbedder(
        model_path=model_path,
        device="auto",
        session_factory=factory,
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert embedder.providers == ["CPUExecutionProvider"]
    assert factory.call_args_list[1].kwargs["providers"] == ["CPUExecutionProvider"]


def test_identity_storage_round_trip_and_cosine_match(tmp_path):
    storage = FaceIdentityStorage(tmp_path)
    storage.save(
        "Ada Lovelace",
        np.array([[3.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )

    database, warnings = storage.load()
    name, similarity = database.match(np.array([1.0, 0.0], dtype=np.float32))

    assert warnings == []
    assert name == "Ada Lovelace"
    assert similarity is not None and similarity > 0.9


def test_service_marks_matches_above_threshold_as_known():
    detector = Mock()
    detector.detect.return_value = [FaceDetection((10, 10, 50, 50), 0.9)]
    embedder = Mock()
    embedder.embed.return_value = np.array([1.0, 0.0], dtype=np.float32)
    database = IdentityDatabase(
        ("Ada",), np.array([[1.0, 0.0]], dtype=np.float32)
    )
    service = FaceRecognitionService(detector, embedder, database, 0.45, 0.2)

    recognitions = service.recognize(np.zeros((80, 80, 3), dtype=np.uint8))

    assert len(recognitions) == 1
    assert recognitions[0].identity == "Ada"
    assert recognitions[0].known is True
    assert recognitions[0].similarity == pytest.approx(1.0)


def test_service_returns_unknown_when_no_identities_are_enrolled():
    detector = Mock()
    detector.detect.return_value = [FaceDetection((10, 10, 50, 50), 0.9)]
    embedder = Mock()
    embedder.embed.return_value = np.array([1.0, 0.0], dtype=np.float32)
    database = IdentityDatabase((), np.empty((0, 0), dtype=np.float32))
    service = FaceRecognitionService(detector, embedder, database, 0.45, 0.2)

    recognition = service.recognize(np.zeros((80, 80, 3), dtype=np.uint8))[0]

    assert recognition.identity == "unknown"
    assert recognition.similarity is None
    assert recognition.known is False
