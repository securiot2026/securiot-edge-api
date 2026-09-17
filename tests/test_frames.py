import io

import pytest
from PIL import Image

from app import create_app
from app.commands import _manual_command_delivered, _pending_commands
from app.models import DeviceSetting, Reading
from app.relay import relay_cycle
from app.vision import VisionError
from app.vision.service import FaceRecognition


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
    COMMAND_EXPIRY_SECONDS = 8


@pytest.fixture(autouse=True)
def _clear_command_state():
    _pending_commands.clear()
    _manual_command_delivered.clear()
    yield
    _pending_commands.clear()
    _manual_command_delivered.clear()


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


def test_per_device_override_enables_recognition_over_global_default(monkeypatch):
    class FaceConfig(FakeConfig):
        FACE_RECOGNITION_ENABLED = False

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
        DeviceSetting.create(device_id="dev-override-on", face_recognition_enabled=True)
        response = _post_frame(client, "empty.jpg", device_id="dev-override-on")

    assert response.status_code == 200
    assert "faces" in response.get_json()


def test_per_device_override_disables_recognition_over_global_default(monkeypatch):
    class FaceConfig(FakeConfig):
        FACE_RECOGNITION_ENABLED = True

    app = create_app(config_object=FaceConfig, start_relay=False)
    client = app.test_client()

    with app.app_context():
        DeviceSetting.create(device_id="dev-override-off", face_recognition_enabled=False)
        response = _post_frame(client, "empty.jpg", device_id="dev-override-off")

    assert response.status_code == 200
    assert "faces" not in response.get_json()


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


def test_manual_command_delivered_even_on_non_qualifying_frame(app, client):
    with app.app_context():
        import datetime

        _pending_commands["dev-manual-empty"] = {
            "light": {
                "action": {"on": True},
                "issued_at": datetime.datetime.utcnow(),
            }
        }

        response = _post_frame(client, "empty.jpg", device_id="dev-manual-empty")

        assert response.status_code == 200
        body = response.get_json()
        assert body["manual_command"]["actuator"] == "light"
        assert body["manual_command"]["action"] == {"on": True}
        # Delivered once: the pending slot is now empty for this device.
        assert "dev-manual-empty" not in _pending_commands or not _pending_commands[
            "dev-manual-empty"
        ]


def test_manual_command_expires_silently_after_ttl(app, client):
    with app.app_context():
        import datetime

        stale_issued_at = datetime.datetime.utcnow() - datetime.timedelta(seconds=30)
        _pending_commands["dev-manual-expired"] = {
            "buzzer": {"action": {"on": True}, "issued_at": stale_issued_at}
        }

        response = _post_frame(client, "empty.jpg", device_id="dev-manual-expired")

        assert response.status_code == 200
        assert response.get_json()["manual_command"] is None
        assert not _pending_commands.get("dev-manual-expired")


def test_manual_pan_tilt_command_overrides_automatic_tracking_delta(app, client):
    with app.app_context():
        import datetime

        # Build an active tracking episode first (matches the existing
        # cooldown test's setup), then queue a manual pan/tilt command for
        # the next poll — it must prevail over that poll's own automatic
        # tracking delta, not blend with it.
        _post_frame(client, "person_left.jpg", device_id="dev-manual-override")
        _post_frame(client, "person_left.jpg", device_id="dev-manual-override")

        _pending_commands["dev-manual-override"] = {
            "pan_tilt": {
                "action": {"pan_delta": -5.0, "tilt_delta": 2.0},
                "issued_at": datetime.datetime.utcnow(),
            }
        }

        response = _post_frame(
            client, "person_left.jpg", device_id="dev-manual-override"
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["manual_command"]["actuator"] == "pan_tilt"
        assert body["manual_command"]["action"] == {
            "pan_delta": -5.0,
            "tilt_delta": 2.0,
        }
        # The automatic delta is suppressed this cycle — firmware applies
        # the manual command instead, never both.
        assert body["pan_delta"] is None
        assert body["tilt_delta"] is None


def test_manual_command_on_other_actuator_does_not_suppress_automatic_tracking(
    app, client
):
    with app.app_context():
        import datetime

        _post_frame(client, "person_left.jpg", device_id="dev-manual-light")
        _post_frame(client, "person_left.jpg", device_id="dev-manual-light")

        _pending_commands["dev-manual-light"] = {
            "light": {
                "action": {"on": True},
                "issued_at": datetime.datetime.utcnow(),
            }
        }

        response = _post_frame(
            client, "person_left.jpg", device_id="dev-manual-light"
        )

        body = response.get_json()
        assert body["manual_command"]["actuator"] == "light"
        assert body["pan_delta"] is not None
        assert body["tilt_delta"] is not None


def test_manual_command_suppresses_automatic_door_action_during_cooldown(
    app, client
):
    with app.app_context():
        import datetime

        # Directly stamp the internal suppression-state dict, per
        # 07-CONTEXT.md's D-R2: CTRL-06 is verified at the unit level
        # against app/commands.py's suppression check, not via a literal
        # end-to-end POST through /commands with a "door" actuator value
        # (the public actuator enum never accepts "door").
        _manual_command_delivered["dev-manual-door"] = {
            "door": datetime.datetime.utcnow()
        }

        _post_frame(client, "person_left.jpg", device_id="dev-manual-door")
        triggering = _post_frame(
            client, "person_left.jpg", device_id="dev-manual-door"
        )

        assert triggering.status_code == 200
        assert triggering.get_json()["door_action"] is None

        readings = list(
            Reading.select().where(Reading.device_id == "dev-manual-door")
        )
        door_readings = [r for r in readings if r.sensor_type == "door_contact"]
        assert door_readings == []
