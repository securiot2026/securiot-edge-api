import io

import pytest
from PIL import Image

from app import create_app
from app.models import Reading
from app.relay import relay_cycle
from app.vision import VisionError
from app.vision.service import FaceRecognition


class FakeConfig:
    EDGE_DB_PATH = ":memory:"
    DEVICE_SHARED_SECRET = "testsecret"
    CLOUD_API_URL = "http://cloud.example"
    CLOUD_DEVICE_API_KEY = "cloudkey"
    RELAY_INTERVAL_SECONDS = 3600

    DETECTION_BACKEND = "mock"
    YOLO_MODEL_PATH = "yolov8n.pt"
    ALLOWED_OBJECT_CLASS = "backpack"
    DETECTION_CONFIDENCE_THRESHOLD = 0.5
    DETECTION_DEBOUNCE_COUNT = 2
    DOOR_ACTION_COOLDOWN_SECONDS = 30


@pytest.fixture
def app():
    return create_app(config_object=FakeConfig, start_relay=False)


@pytest.fixture
def client(app):
    return app.test_client()


def _fixture_bytes(name):
    output = io.BytesIO()
    Image.new("RGB", (640, 480), (30, 30, 30)).save(output, format="JPEG")
    return output.getvalue()


def _post_frame(client, fixture_name, device_id, zone_id="zone-1", headers=None):
    headers = headers if headers is not None else {"X-Device-Key": "testsecret"}
    data = {
        "device_id": device_id,
        "zone_id": zone_id,
        "frame": (io.BytesIO(_fixture_bytes(fixture_name)), fixture_name),
    }
    return client.post(
        "/frames", data=data, headers=headers, content_type="multipart/form-data"
    )


def test_frames_without_device_key_returns_401(app, client):
    with app.app_context():
        response = _post_frame(
            client, "person_left.jpg", device_id="dev-401", headers={}
        )

        assert response.status_code == 401


def test_confirmed_person_detection_buffers_readings_and_returns_servo_command(
    app, client
):
    with app.app_context():
        first = _post_frame(client, "person_left.jpg", device_id="dev-person")
        assert first.status_code == 200
        assert first.get_json()["door_action"] is None

        second = _post_frame(client, "person_left.jpg", device_id="dev-person")

        assert second.status_code == 200
        body = second.get_json()
        assert body["pan_delta"] is not None
        assert body["tilt_delta"] is not None
        assert body["door_action"] == "lock"
        assert body["alert"] is True

        readings = list(Reading.select().where(Reading.device_id == "dev-person"))
        assert len(readings) == 2

        sensor_types = {r.sensor_type for r in readings}
        assert sensor_types == {"camera_detection", "door_contact"}

        door_reading = next(r for r in readings if r.sensor_type == "door_contact")
        assert door_reading.value == "open"

        for reading in readings:
            assert reading.synced is False


def test_door_action_cooldown_suppresses_repeat_lock_but_keeps_tracking(app, client):
    with app.app_context():
        # Build the streak up to the debounce threshold (frame 2 escalates
        # and fires the door lock, starting the cooldown window).
        _post_frame(client, "person_left.jpg", device_id="dev-cooldown")
        triggering = _post_frame(client, "person_left.jpg", device_id="dev-cooldown")
        assert triggering.get_json()["door_action"] == "lock"

        # A further frame while the person remains in view, still within
        # the cooldown window: pan/tilt keeps tracking, but the door does
        # not re-lock and no second door_contact Reading is created.
        continued = _post_frame(client, "person_left.jpg", device_id="dev-cooldown")

        assert continued.status_code == 200
        body = continued.get_json()
        assert body["pan_delta"] is not None
        assert body["tilt_delta"] is not None
        assert body["door_action"] is None

        readings = list(Reading.select().where(Reading.device_id == "dev-cooldown"))
        door_readings = [r for r in readings if r.sensor_type == "door_contact"]
        camera_readings = [r for r in readings if r.sensor_type == "camera_detection"]

        assert len(door_readings) == 1
        assert len(camera_readings) == 2


def test_empty_frame_returns_no_command_and_creates_no_readings(app, client):
    with app.app_context():
        response = _post_frame(client, "empty.jpg", device_id="dev-empty")

        assert response.status_code == 200
        body = response.get_json()
        assert body["pan_delta"] is None
        assert body["tilt_delta"] is None
        assert body["door_action"] is None
        assert body["alert"] is False
        assert "faces" not in body

        readings = list(Reading.select().where(Reading.device_id == "dev-empty"))
        assert len(readings) == 0


class FakeCloudResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_frames_readings_are_picked_up_by_the_unmodified_relay_cycle(
    app, client, monkeypatch
):
    with app.app_context():
        _post_frame(client, "person_left.jpg", device_id="dev-relay")
        _post_frame(client, "person_left.jpg", device_id="dev-relay")

        readings = list(Reading.select().where(Reading.device_id == "dev-relay"))
        assert len(readings) == 2
        assert all(r.synced is False for r in readings)

        monkeypatch.setattr(
            "app.relay.requests.post", lambda *a, **k: FakeCloudResponse(201)
        )
        relay_cycle()

        refreshed = list(Reading.select().where(Reading.device_id == "dev-relay"))
        assert len(refreshed) == 2
        assert all(r.synced is True for r in refreshed)


def test_frames_includes_face_recognition_when_enabled(monkeypatch):
    class FaceConfig(FakeConfig):
        FACE_RECOGNITION_ENABLED = True

    recognizer = type(
        "Recognizer",
        (),
        {
            "recognize_path": lambda self, path: [
                FaceRecognition((10, 20, 60, 80), 0.92, "Ada", 0.81, True)
            ]
        },
    )()
    monkeypatch.setattr("app.frames.get_face_recognizer", lambda config: recognizer)
    app = create_app(config_object=FaceConfig, start_relay=False)
    client = app.test_client()

    with app.app_context():
        response = _post_frame(client, "empty.jpg", device_id="dev-face")

    assert response.status_code == 200
    assert response.get_json()["faces"] == [
        {
            "bbox": [10, 20, 60, 80],
            "detection_confidence": 0.92,
            "identity": "Ada",
            "similarity": 0.81,
            "known": True,
        }
    ]


def test_frames_returns_503_when_enabled_recognition_is_unavailable(monkeypatch):
    class FaceConfig(FakeConfig):
        FACE_RECOGNITION_ENABLED = True

    def unavailable(config):
        raise VisionError("missing local model")

    monkeypatch.setattr("app.frames.get_face_recognizer", unavailable)
    app = create_app(config_object=FaceConfig, start_relay=False)
    client = app.test_client()

    with app.app_context():
        response = _post_frame(client, "empty.jpg", device_id="dev-face-error")

    assert response.status_code == 503
    assert response.get_json() == {"error": "face recognition unavailable"}
