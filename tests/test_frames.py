import datetime
import io
import os

import pytest

from app import create_app
from app.models import Reading

FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures", "sample_frames"
)


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
    with open(os.path.join(FIXTURES_DIR, name), "rb") as fh:
        return fh.read()


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

        readings = list(Reading.select().where(Reading.device_id == "dev-empty"))
        assert len(readings) == 0
