import datetime
import io

import pytest
import requests
from PIL import Image

from app import create_app, heartbeat
from app.heartbeat import heartbeat_cycle, record_seen
from app.models import Reading


class FakeConfig:
    EDGE_DB_PATH = ":memory:"
    DEVICE_SHARED_SECRET = "testsecret"
    EDGE_ADMIN_API_KEY = "adminsecret"
    CLOUD_API_URL = "http://cloud.example"
    CLOUD_DEVICE_API_KEY = "cloudkey"
    RELAY_INTERVAL_SECONDS = 3600
    HEARTBEAT_INTERVAL_SECONDS = 30

    DETECTION_BACKEND = "mock"
    YOLO_MODEL_PATH = "yolov8n.pt"
    ALLOWED_OBJECT_CLASS = "backpack"
    DETECTION_CONFIDENCE_THRESHOLD = 0.5
    DETECTION_DEBOUNCE_COUNT = 2
    DOOR_ACTION_COOLDOWN_SECONDS = 30
    COMMAND_EXPIRY_SECONDS = 8


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _reset_heartbeat_state():
    heartbeat.reset()
    yield
    heartbeat.reset()


@pytest.fixture
def app():
    return create_app(config_object=FakeConfig, start_relay=False)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def posts(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(201)

    monkeypatch.setattr("app.heartbeat.requests.post", fake_post)
    return calls


def _jpeg(valid=True):
    if not valid:
        return b"not an image"
    output = io.BytesIO()
    Image.new("RGB", (640, 480), (30, 30, 30)).save(output, format="JPEG")
    return output.getvalue()


def _post_frame(client, device_id="dev-hb", zone_id="zone-hb", name="empty.jpg",
                headers=None, valid=True):
    headers = headers if headers is not None else {"X-Device-Key": "testsecret"}
    return client.post(
        "/frames",
        data={
            "device_id": device_id,
            "zone_id": zone_id,
            "frame": (io.BytesIO(_jpeg(valid)), name),
        },
        headers=headers,
        content_type="multipart/form-data",
    )


T0 = datetime.datetime(2026, 10, 30, 12, 0, 0)


def test_frame_then_cycle_posts_one_heartbeat(app, client, posts):
    with app.app_context():
        assert _post_frame(client).status_code == 200

        heartbeat_cycle()

        assert len(posts) == 1
        call = posts[0]
        assert call["url"] == "http://cloud.example/api/v1/telemetry"
        assert call["headers"] == {"X-Device-Key": "cloudkey"}
        assert call["timeout"] == 10
        body = call["json"]
        assert body["sensor_type"] == "heartbeat"
        assert body["device_id"] == "dev-hb"
        assert body["zone_id"] == "zone-hb"
        assert body["value"] == "alive"
        assert body["reading_id"]
        assert body["recorded_at"].endswith("Z")


def test_empty_scene_creates_heartbeat_and_no_reading(app, client, posts):
    with app.app_context():
        assert _post_frame(client, name="empty.jpg").status_code == 200

        heartbeat_cycle()

        assert len(posts) == 1
        assert Reading.select().count() == 0


def test_second_cycle_without_new_frame_posts_nothing(app, client, posts):
    with app.app_context():
        _post_frame(client)
        heartbeat_cycle()
        heartbeat_cycle()

        assert len(posts) == 1


def test_new_frame_inside_interval_waits_until_interval_elapsed(app, posts):
    with app.app_context():
        record_seen("dev-1", "zone-1", T0)
        heartbeat_cycle(now=T0)
        assert len(posts) == 1

        record_seen("dev-1", "zone-1", T0 + datetime.timedelta(seconds=5))
        heartbeat_cycle(now=T0 + datetime.timedelta(seconds=10))
        assert len(posts) == 1

        heartbeat_cycle(now=T0 + datetime.timedelta(seconds=31))
        assert len(posts) == 2
        assert posts[1]["json"]["recorded_at"] == "2026-10-30T12:00:05Z"


def test_cloud_exception_keeps_device_due(app, monkeypatch):
    attempts = []

    def failing_post(*args, **kwargs):
        attempts.append(1)
        raise requests.ConnectionError("cloud unreachable")

    monkeypatch.setattr("app.heartbeat.requests.post", failing_post)
    with app.app_context():
        record_seen("dev-1", "zone-1", T0)

        heartbeat_cycle(now=T0)
        heartbeat_cycle(now=T0 + datetime.timedelta(seconds=1))

        assert len(attempts) == 2


def test_cloud_non_2xx_keeps_device_due(app, monkeypatch):
    attempts = []

    def rejecting_post(*args, **kwargs):
        attempts.append(1)
        return FakeResponse(500)

    monkeypatch.setattr("app.heartbeat.requests.post", rejecting_post)
    with app.app_context():
        record_seen("dev-1", "zone-1", T0)

        heartbeat_cycle(now=T0)
        heartbeat_cycle(now=T0 + datetime.timedelta(seconds=1))

        assert len(attempts) == 2


def test_wrong_device_key_records_nothing(app, client, posts):
    with app.app_context():
        response = _post_frame(client, headers={"X-Device-Key": "wrong"})
        assert response.status_code == 401

        heartbeat_cycle()

        assert posts == []


def test_undecodable_image_still_records_last_seen(app, client, posts):
    with app.app_context():
        response = _post_frame(client, valid=False)
        assert response.status_code == 400

        heartbeat_cycle()

        assert len(posts) == 1
        assert posts[0]["json"]["device_id"] == "dev-hb"


def test_devices_are_tracked_independently(app, posts):
    with app.app_context():
        record_seen("dev-a", "zone-a", T0)
        record_seen("dev-b", "zone-b", T0)

        heartbeat_cycle(now=T0)

        assert sorted(call["json"]["device_id"] for call in posts) == ["dev-a", "dev-b"]


def test_interval_defaults_to_30_when_config_omits_it(app, posts):
    app.config.pop("HEARTBEAT_INTERVAL_SECONDS", None)
    with app.app_context():
        record_seen("dev-1", "zone-1", T0)
        heartbeat_cycle(now=T0)
        record_seen("dev-1", "zone-1", T0 + datetime.timedelta(seconds=20))
        heartbeat_cycle(now=T0 + datetime.timedelta(seconds=29))
        assert len(posts) == 1
        heartbeat_cycle(now=T0 + datetime.timedelta(seconds=30))
        assert len(posts) == 2
