import pytest

from app import create_app
from app.models import DeviceSetting


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


@pytest.fixture
def app():
    return create_app(config_object=FakeConfig, start_relay=False)


@pytest.fixture
def client(app):
    return app.test_client()


def test_get_with_no_row_returns_global_default(app, client):
    with app.app_context():
        response = client.get(
            "/devices/dev-1/face-recognition", headers={"X-Admin-Key": "adminsecret"}
        )

    assert response.status_code == 200
    assert response.get_json()["face_recognition_enabled"] is False


def test_patch_persists_override_and_get_reflects_it(app, client):
    with app.app_context():
        patch_response = client.patch(
            "/devices/dev-1/face-recognition",
            json={"enabled": True},
            headers={"X-Admin-Key": "adminsecret"},
        )
        assert patch_response.status_code == 200

        get_response = client.get(
            "/devices/dev-1/face-recognition", headers={"X-Admin-Key": "adminsecret"}
        )

    assert get_response.get_json()["face_recognition_enabled"] is True


def test_patch_without_admin_key_returns_401_and_creates_no_row(app, client):
    with app.app_context():
        response = client.patch(
            "/devices/dev-1/face-recognition", json={"enabled": True}, headers={}
        )

        assert response.status_code == 401
        assert DeviceSetting.select().count() == 0


def test_patch_with_non_boolean_enabled_returns_400(app, client):
    with app.app_context():
        for bad_value in ["true", 1]:
            response = client.patch(
                "/devices/dev-1/face-recognition",
                json={"enabled": bad_value},
                headers={"X-Admin-Key": "adminsecret"},
            )
            assert response.status_code == 400


def test_two_patches_for_same_device_update_one_row(app, client):
    with app.app_context():
        client.patch(
            "/devices/dev-1/face-recognition",
            json={"enabled": True},
            headers={"X-Admin-Key": "adminsecret"},
        )
        client.patch(
            "/devices/dev-1/face-recognition",
            json={"enabled": False},
            headers={"X-Admin-Key": "adminsecret"},
        )

        assert DeviceSetting.select().count() == 1
        assert DeviceSetting.get(DeviceSetting.device_id == "dev-1").face_recognition_enabled is False
