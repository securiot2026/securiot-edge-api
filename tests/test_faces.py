import io
import os

import pytest
from PIL import Image

import app.faces as faces_module
from app import create_app
from app.vision.enrollment import EnrollmentResult
from app.vision.errors import VisionError


class FakeConfig:
    EDGE_DB_PATH = ":memory:"
    DEVICE_SHARED_SECRET = "testsecret"
    EDGE_ADMIN_API_KEY = "adminsecret"
    CLOUD_API_URL = "http://cloud.example"
    CLOUD_DEVICE_API_KEY = "cloudkey"
    RELAY_INTERVAL_SECONDS = 3600

    DETECTION_BACKEND = "mock"
    YOLO_MODEL_PATH = "yolov8n.pt"
    ALLOWED_OBJECT_CLASS = "backpack"
    DETECTION_CONFIDENCE_THRESHOLD = 0.5
    DETECTION_DEBOUNCE_COUNT = 2
    DOOR_ACTION_COOLDOWN_SECONDS = 30
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024
    FACE_ENROLLMENT_MAX_CONTENT_LENGTH = 10 * 1024 * 1024

    FACE_RECOGNITION_ENABLED = False
    FACE_YOLO_MODEL_PATH = "models/yolov8n-face.pt"
    FACE_YOLO_CONFIDENCE_THRESHOLD = 0.5
    FACE_YOLO_IOU_THRESHOLD = 0.45
    FACE_YOLO_INPUT_SIZE = 640
    FACE_YOLO_MAX_DETECTIONS = 10
    FACE_YOLO_CLASS_ID = 0
    FACE_YOLO_DEVICE = "cpu"
    ARCFACE_MODEL_PATH = "models/w600k_r50.onnx"
    ARCFACE_DEVICE = "cpu"
    FACE_IDENTITY_DIR = "data/faces"
    FACE_SIMILARITY_THRESHOLD = 0.45
    FACE_CROP_MARGIN = 0.2
    FACE_MINIMUM_ENROLLMENT_IMAGES = 3


@pytest.fixture
def app():
    return create_app(config_object=FakeConfig, start_relay=False)


@pytest.fixture
def client(app):
    return app.test_client()


def _fixture_bytes():
    output = io.BytesIO()
    Image.new("RGB", (640, 480), (30, 30, 30)).save(output, format="JPEG")
    return output.getvalue()


def _post_faces(client, name="Ada Lovelace", count=3, headers=None):
    headers = headers if headers is not None else {"X-Admin-Key": "adminsecret"}
    data = {"name": name}
    files = [
        (io.BytesIO(_fixture_bytes()), f"face-{i}.jpg") for i in range(count)
    ]
    data["images"] = files
    return client.post(
        "/faces", data=data, headers=headers, content_type="multipart/form-data"
    )


def test_faces_without_admin_key_returns_401_before_touching_files(app, client, monkeypatch):
    called = {"mkdtemp": False}
    original_mkdtemp = __import__("tempfile").mkdtemp

    def spy_mkdtemp(*a, **k):
        called["mkdtemp"] = True
        return original_mkdtemp(*a, **k)

    monkeypatch.setattr("app.faces.tempfile.mkdtemp", spy_mkdtemp)

    with app.app_context():
        response = _post_faces(client, headers={})

    assert response.status_code == 401
    assert called["mkdtemp"] is False


def test_faces_with_wrong_admin_key_returns_401(app, client):
    with app.app_context():
        response = _post_faces(client, headers={"X-Admin-Key": "wrong"})

    assert response.status_code == 401


def test_successful_enrollment_returns_201_and_cleans_up_temp_dir(app, client, monkeypatch):
    recorded_path = {}
    original_mkdtemp = __import__("tempfile").mkdtemp

    def recording_mkdtemp(*a, **k):
        path = original_mkdtemp(*a, **k)
        recorded_path["path"] = path
        return path

    monkeypatch.setattr("app.faces.tempfile.mkdtemp", recording_mkdtemp)
    monkeypatch.setattr(
        faces_module, "validate_capture_quality", lambda paths, settings: (paths, [])
    )

    fake_result = EnrollmentResult(
        identity_path="data/faces/ada-lovelace/identity.npz",
        accepted_images=3,
        skipped_images=(),
    )
    monkeypatch.setattr(
        faces_module, "enroll_identity", lambda name, paths, settings: fake_result
    )

    with app.app_context():
        response = _post_faces(client)

    assert response.status_code == 201
    body = response.get_json()
    assert body["accepted_images"] == 3
    assert body["skipped_images"] == []
    assert body["name"] == "Ada Lovelace"
    assert not os.path.exists(recorded_path["path"])


def test_vision_error_returns_400_and_cleans_up(app, client, monkeypatch):
    recorded_path = {}
    original_mkdtemp = __import__("tempfile").mkdtemp

    def recording_mkdtemp(*a, **k):
        path = original_mkdtemp(*a, **k)
        recorded_path["path"] = path
        return path

    monkeypatch.setattr("app.faces.tempfile.mkdtemp", recording_mkdtemp)
    monkeypatch.setattr(
        faces_module, "validate_capture_quality", lambda paths, settings: (paths, [])
    )

    def raise_vision_error(name, paths, settings):
        raise VisionError("Only 1 valid enrollment images were found")

    monkeypatch.setattr(faces_module, "enroll_identity", raise_vision_error)

    with app.app_context():
        response = _post_faces(client)

    assert response.status_code == 400
    assert "error" in response.get_json()
    assert not os.path.exists(recorded_path["path"])


def test_vision_error_400_includes_quality_gate_reasons_and_cleans_up(app, client, monkeypatch):
    recorded_path = {}
    original_mkdtemp = __import__("tempfile").mkdtemp

    def recording_mkdtemp(*a, **k):
        path = original_mkdtemp(*a, **k)
        recorded_path["path"] = path
        return path

    monkeypatch.setattr("app.faces.tempfile.mkdtemp", recording_mkdtemp)

    reasons = [
        "face-1.jpg: Image is too blurry; hold still",
        "face-2.jpg: No face detected",
    ]
    monkeypatch.setattr(
        faces_module,
        "validate_capture_quality",
        lambda paths, settings: (paths[:1], list(reasons)),
    )

    def raise_vision_error(name, paths, settings):
        raise VisionError("Only 1 valid enrollment images were found; at least 3 are required.")

    monkeypatch.setattr(faces_module, "enroll_identity", raise_vision_error)

    with app.app_context():
        response = _post_faces(client)

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"].startswith("Only 1 valid")
    assert body["skipped_images"] == reasons
    assert not os.path.exists(recorded_path["path"])


def test_vision_error_400_without_quality_skips_has_empty_skipped_images(app, client, monkeypatch):
    monkeypatch.setattr(
        faces_module, "validate_capture_quality", lambda paths, settings: (paths, [])
    )

    def raise_vision_error(name, paths, settings):
        raise VisionError("Display name is invalid")

    monkeypatch.setattr(faces_module, "enroll_identity", raise_vision_error)

    with app.app_context():
        response = _post_faces(client, name="!!!")

    assert response.status_code == 400
    body = response.get_json()
    assert "error" in body
    assert body["skipped_images"] == []


def test_quality_gate_skips_are_excluded_from_enrollment_and_reported(app, client, monkeypatch):
    captured_args = {}

    def fake_validate_capture_quality(paths, settings):
        # Exclude the first path as a quality-gate rejection; pass the rest through.
        return paths[1:], [f"{paths[0].name}: Image is too blurry; hold still"]

    def fake_enroll_identity(name, paths, settings):
        captured_args["paths"] = paths
        return EnrollmentResult(
            identity_path="data/faces/ada-lovelace/identity.npz",
            accepted_images=len(paths),
            skipped_images=(),
        )

    monkeypatch.setattr(
        faces_module, "validate_capture_quality", fake_validate_capture_quality
    )
    monkeypatch.setattr(faces_module, "enroll_identity", fake_enroll_identity)

    with app.app_context():
        response = _post_faces(client, count=3)

    assert response.status_code == 201
    body = response.get_json()
    assert any("too blurry" in reason for reason in body["skipped_images"])
    assert len(captured_args["paths"]) == 2


def test_no_images_returns_400(app, client):
    with app.app_context():
        response = client.post(
            "/faces",
            data={"name": "Nobody"},
            headers={"X-Admin-Key": "adminsecret"},
            content_type="multipart/form-data",
        )

    assert response.status_code == 400
