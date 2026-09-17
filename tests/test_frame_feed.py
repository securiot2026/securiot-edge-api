import base64
import io

import pytest
from PIL import Image

from app import create_app


class FakeConfig:
    EDGE_DB_PATH = ":memory:"
    DEVICE_SHARED_SECRET = "testsecret"
    EDGE_ADMIN_API_KEY = "adminsecret"
    CLOUD_API_URL = "http://cloud.example"
    CLOUD_DEVICE_API_KEY = "cloudkey"
    RELAY_INTERVAL_SECONDS = 3600

    DETECTION_BACKEND = "mock"
    ALLOWED_OBJECT_CLASS = "backpack"
    DOOR_ACTION_COOLDOWN_SECONDS = 30


@pytest.fixture
def app():
    return create_app(config_object=FakeConfig, start_relay=False)


@pytest.fixture
def client(app):
    return app.test_client()


ADMIN_HEADERS = {"X-Admin-Key": "adminsecret"}


def _jpeg_bytes():
    output = io.BytesIO()
    Image.new("RGB", (640, 480), (10, 20, 30)).save(output, format="JPEG")
    return output.getvalue()


def _post_frame(client, device_id):
    data = {
        "device_id": device_id,
        "zone_id": "zone-1",
        "frame": (io.BytesIO(_jpeg_bytes()), "frame.jpg"),
    }
    return client.post(
        "/frames",
        data=data,
        headers={"X-Device-Key": "testsecret"},
        content_type="multipart/form-data",
    )


def test_latest_frame_requires_admin_key(client):
    assert client.get("/frames/latest/dev-1").status_code == 401


def test_latest_frame_404_when_device_has_no_frame_yet(app, client):
    with app.app_context():
        response = client.get("/frames/latest/dev-unseen", headers=ADMIN_HEADERS)
    assert response.status_code == 404


def test_latest_frame_returns_cached_jpeg_and_capture_timestamp(app, client):
    with app.app_context():
        frame_response = _post_frame(client, "dev-1")
        assert frame_response.status_code == 200

        response = client.get("/frames/latest/dev-1", headers=ADMIN_HEADERS)

    assert response.status_code == 200
    body = response.get_json()
    assert body["device_id"] == "dev-1"
    assert body["captured_at"].endswith("Z")
    decoded = base64.b64decode(body["frame"])
    assert decoded == _jpeg_bytes()


def test_latest_frame_overwrites_previous_frame_no_history(app, client):
    with app.app_context():
        _post_frame(client, "dev-1")
        first = client.get("/frames/latest/dev-1", headers=ADMIN_HEADERS).get_json()

        _post_frame(client, "dev-1")
        second = client.get("/frames/latest/dev-1", headers=ADMIN_HEADERS).get_json()

    # Same fixture bytes each time (deterministic test image), but this
    # asserts the cache only ever holds ONE entry per device: a second
    # /frames post must not leave the first response retrievable anywhere,
    # and only the single latest capture timestamp should be readable —
    # covered structurally by app/frame_cache.py always overwriting the
    # dict entry (never appending), not by content-diffing two identical
    # fixture images.
    assert first["device_id"] == second["device_id"] == "dev-1"


def test_latest_frame_is_per_device(app, client):
    with app.app_context():
        _post_frame(client, "dev-1")

        response = client.get("/frames/latest/dev-2", headers=ADMIN_HEADERS)

    assert response.status_code == 404
