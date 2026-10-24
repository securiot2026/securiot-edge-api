import io
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app import create_app
from app.local_console import CameraBridge, create_console_blueprint
from app.vision.errors import VisionError


@pytest.fixture
def console(tmp_path):
    class Config:
        EDGE_DB_PATH = ":memory:"
        DEVICE_SHARED_SECRET = "server-only-device-secret"
        FACE_RECOGNITION_ENABLED = True
        FACE_IDENTITY_DIR = str(tmp_path / "identities")
        MAX_CONTENT_LENGTH = 2 * 1024 * 1024

    app = create_app(Config, start_relay=False)
    bridge = CameraBridge("http://192.168.1.2", "http://127.0.0.1:5050", Config.DEVICE_SHARED_SECRET)
    app.register_blueprint(create_console_blueprint(bridge, app.config))
    client = app.test_client()
    page = client.get("/")
    token = re.search(r'<style nonce="([^"]+)"', page.text).group(1)
    return SimpleNamespace(app=app, client=client, bridge=bridge, token=token, directory=tmp_path)


def post(console, data):
    return console.client.post(
        "/console/enroll", json=data,
        headers={"X-Console-Token": console.token},
    )


def test_page_is_private_and_does_not_disclose_device_key(console):
    response = console.client.get("/")
    assert response.status_code == 200
    assert "Register a face" in response.text
    assert console.bridge.device_key not in response.text
    assert response.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_console_rejects_remote_access_dns_rebinding_and_missing_token(console):
    assert console.client.get("/", environ_base={"REMOTE_ADDR": "192.168.1.8"}).status_code == 403
    assert console.client.get("/", headers={"Host": "evil.example"}).status_code == 403
    assert console.client.post("/console/enroll", json={"name": "Ada", "consent": True}).status_code == 403
    response = console.client.post(
        "/console/enroll", json={"name": "Ada", "consent": True},
        headers={"X-Console-Token": console.token, "Origin": "https://evil.example"},
    )
    assert response.status_code == 403


@pytest.mark.parametrize("data", [
    {"name": "Ada", "consent": False},
    {"name": "", "consent": True},
    {"name": "\nInvalid", "consent": True},
    ["not", "an", "object"],
])
def test_enrollment_requires_valid_name_and_explicit_consent(console, data):
    assert post(console, data).status_code == 400
    assert not console.bridge.paused.is_set()


def test_enrollment_uses_fresh_temporary_images_and_resumes_bridge(console, monkeypatch):
    image = io.BytesIO()
    Image.new("RGB", (640, 480)).save(image, format="JPEG")
    monkeypatch.setattr("app.local_console.camera_snapshot", lambda url: (image.getvalue(), (640, 480)))
    monkeypatch.setattr("app.local_console.time.sleep", lambda seconds: None)
    captured = []

    def enroll(name, paths, settings):
        assert name == "Ada"
        assert len(paths) == 5
        assert all(path.is_file() for path in paths)
        assert console.bridge.paused.is_set()
        captured.extend(paths)
        return SimpleNamespace(accepted_images=5, skipped_images=())

    monkeypatch.setattr("app.local_console.enroll_identity", enroll)
    response = post(console, {"name": "Ada", "consent": True})
    assert response.status_code == 201
    assert response.json == {"name": "Ada", "accepted_images": 5, "skipped_images": 0}
    assert all(not path.exists() for path in captured)
    assert not console.bridge.paused.is_set()
    assert not console.bridge.processing_lock.locked()
    assert not console.bridge.enrollment_lock.locked()


def test_invalid_capture_does_not_save_an_identity(console, monkeypatch):
    monkeypatch.setattr("app.local_console.camera_snapshot", lambda url: (_ for _ in ()).throw(VisionError("No face")))
    assert post(console, {"name": "Ada", "consent": True}).status_code == 422
    assert not (console.directory / "identities").exists()
    assert not console.bridge.paused.is_set()
    assert not console.bridge.enrollment_lock.locked()


def test_existing_identity_requires_replacement_confirmation(console):
    path = console.directory / "identities" / "ada" / "identity.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing identity, not overwritten")
    response = post(console, {"name": "Ada", "consent": True})
    assert response.status_code == 409
    assert path.read_bytes() == b"existing identity, not overwritten"
    assert not console.bridge.paused.is_set()


def test_cached_frame_and_status(console):
    assert console.client.get("/console/frame.jpg").status_code == 503
    console.bridge.image = b"jpeg"
    response = console.client.get("/console/frame.jpg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert console.client.get("/console/status").json["frames_processed"] == 0
    assert console.client.get("/console/identities").json == {"names": []}


def test_bridge_reuses_authenticated_frames_contract(monkeypatch):
    bridge = CameraBridge("http://camera", "http://127.0.0.1:5050", "private-key")
    monkeypatch.setattr("app.local_console.camera_snapshot", lambda url: (b"jpeg", (640, 480)))
    session = MagicMock()
    session.__enter__.return_value = session
    session.post.return_value.json.return_value = {"faces": [], "alert": False}
    monkeypatch.setattr("app.local_console.requests.Session", lambda: session)
    bridge.process_once()
    call = session.post.call_args
    assert call.args == ("http://127.0.0.1:5050/frames",)
    assert call.kwargs["headers"] == {"X-Device-Key": "private-key"}
    assert call.kwargs["data"]["frame_width"] == 640
    assert bridge.snapshot()["frames_processed"] == 1
    assert bridge.snapshot()["connected"] is True
