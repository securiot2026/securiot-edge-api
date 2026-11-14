import pytest

from app import create_app, occupancy


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
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024
    FACE_ENROLLMENT_MAX_CONTENT_LENGTH = 10 * 1024 * 1024

    FACE_RECOGNITION_ENABLED = False


@pytest.fixture
def app():
    return create_app(config_object=FakeConfig, start_relay=False)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def _clear_occupancy_state():
    occupancy._counts.clear()


def test_occupancy_defaults_to_zero_for_unknown_device(app, client):
    response = client.get(
        "/occupancy/dev-1", headers={"X-Admin-Key": "adminsecret"}
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["device_id"] == "dev-1"
    assert body["count"] == 0
    assert "date" in body


def test_occupancy_reflects_recorded_entries(app, client):
    import datetime

    now = datetime.datetime.utcnow()
    occupancy.record_entry_and_get_count("dev-1", now)
    occupancy.record_entry_and_get_count("dev-1", now)

    response = client.get(
        "/occupancy/dev-1", headers={"X-Admin-Key": "adminsecret"}
    )
    assert response.get_json()["count"] == 2


def test_occupancy_requires_admin_key(app, client):
    response = client.get("/occupancy/dev-1")
    assert response.status_code == 401
