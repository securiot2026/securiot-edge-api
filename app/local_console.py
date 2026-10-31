"""Loopback-only camera console; reuses the existing /frames API contract."""

import copy
import io
import secrets
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from flask import Blueprint, Response, abort, jsonify, render_template, request
from PIL import Image

from app.vision.enrollment import enroll_identity
from app.vision.errors import VisionError
from app.vision.settings import VisionSettings
from app.vision.storage import FaceIdentityStorage, identity_slug, validate_display_name


def camera_snapshot(camera_url):
    """Fetch one bounded, decodable JPEG, without using environment proxies."""
    with requests.Session() as session:
        session.trust_env = False
        with session.get(
            camera_url + "/capture", timeout=(3, 5), stream=True,
            allow_redirects=False,
        ) as response:
            response.raise_for_status()
            data = response.raw.read(2 * 1024 * 1024 + 1, decode_content=True)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Camera image exceeds the 2 MB limit.")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or image.width > 4096 or image.height > 4096:
            raise ValueError("Camera must return a JPEG no larger than 4096 × 4096.")
        size = image.size
        image.verify()
    return data, size


class CameraBridge:
    """Pull snapshots without consuming the ESP32's single MJPEG connection."""

    def __init__(self, camera_url, edge_url, device_key, interval=0.5):
        self.camera_url = camera_url.rstrip("/")
        self.edge_url = edge_url.rstrip("/")
        self.device_key = device_key
        self.interval = interval
        self.lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.enrollment_lock = threading.Lock()
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self.image = None
        self.state = {
            "frames_processed": 0, "last_frame_at": None,
            "inference_ms": None, "result": {}, "error": None,
        }

    def snapshot(self):
        with self.lock:
            state = copy.deepcopy(self.state)
        last = state["last_frame_at"]
        state["connected"] = last is not None and time.time() - last < 10
        state["enrolling"] = self.paused.is_set()
        return state

    def process_once(self):
        with self.processing_lock:
            if self.paused.is_set():
                return
            image, (width, height) = camera_snapshot(self.camera_url)
            with self.lock:
                self.image = image
            started = time.monotonic()
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(
                    self.edge_url + "/frames",
                    headers={"X-Device-Key": self.device_key},
                    data={"device_id": "esp32cam-01", "zone_id": "zone-01",
                          "frame_width": width, "frame_height": height},
                    files={"frame": ("frame.jpg", image, "image/jpeg")},
                    timeout=(3, 45), allow_redirects=False,
                )
                response.raise_for_status()
                result = response.json()
            with self.lock:
                self.state.update(
                    frames_processed=self.state["frames_processed"] + 1,
                    last_frame_at=time.time(), result=result, error=None,
                    inference_ms=round((time.monotonic() - started) * 1000),
                )

    def run(self):
        while not self.stopped.is_set():
            started = time.monotonic()
            try:
                self.process_once()
            except (requests.RequestException, ValueError, OSError):
                with self.lock:
                    self.state["error"] = (
                        "Camera or inference unavailable. Check the hotspot and Edge logs."
                    )
            self.stopped.wait(max(0.1, self.interval - (time.monotonic() - started)))


def create_console_blueprint(bridge, config):
    bp = Blueprint("local_console", __name__)
    token = secrets.token_urlsafe(32)
    settings = VisionSettings.from_mapping(config)
    camera = urlsplit(bridge.camera_url)
    stream_url = f"{camera.scheme}://{camera.hostname}:81/stream"

    @bp.before_request
    def protect_console():
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            abort(403)
        if request.host.split(":")[0] not in {"127.0.0.1", "localhost"}:
            abort(403)
        if request.method == "POST":
            if not secrets.compare_digest(
                request.headers.get("X-Console-Token", ""), token
            ):
                abort(403)
            origin = request.headers.get("Origin")
            if origin and origin != request.host_url.rstrip("/"):
                abort(403)

    @bp.after_request
    def private_response(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            f"default-src 'self'; script-src 'nonce-{token}'; "
            f"style-src 'nonce-{token}'; img-src 'self' data: {camera.scheme}://"
            f"{camera.hostname}:81; connect-src 'self'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'; object-src 'none'"
        )
        return response

    @bp.get("/")
    def index():
        return render_template(
            "local_console.html", token=token, stream_url=stream_url,
            camera_url=bridge.camera_url,
        )

    @bp.get("/console/status")
    def status():
        return jsonify(bridge.snapshot())

    @bp.get("/console/frame.jpg")
    def frame():
        with bridge.lock:
            image = bridge.image
        if image is None:
            return jsonify(error="Waiting for the first camera frame."), 503
        return Response(image, mimetype="image/jpeg")

    @bp.get("/console/identities")
    def identities():
        database, _ = FaceIdentityStorage(settings.identity_dir).load()
        return jsonify(names=list(database.names))

    @bp.post("/console/enroll")
    def enroll():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify(error="Enrollment requires a JSON object."), 400
        if data.get("consent") is not True:
            return jsonify(error="Explicit consent is required before enrollment."), 400
        try:
            name = validate_display_name(data.get("name", ""))
            destination = settings.identity_dir / identity_slug(name) / "identity.npz"
        except (VisionError, TypeError, AttributeError) as exc:
            return jsonify(error=str(exc)), 400
        if not settings.enabled:
            return jsonify(error="Face recognition is disabled."), 503
        if not bridge.enrollment_lock.acquire(blocking=False):
            return jsonify(error="An enrollment is already running."), 409
        bridge.paused.set()
        try:
            if destination.exists() and data.get("replace") is not True:
                return jsonify(error="This name exists. Confirm replacement first."), 409
            if not bridge.processing_lock.acquire(timeout=50):
                return jsonify(error="Inference is busy. Try again shortly."), 503
            try:
                with tempfile.TemporaryDirectory(prefix="securiot-enroll-") as directory:
                    paths = []
                    for index in range(settings.minimum_enrollment_images + 2):
                        image, _ = camera_snapshot(bridge.camera_url)
                        path = Path(directory) / f"sample-{index}.jpg"
                        path.write_bytes(image)
                        paths.append(path)
                        time.sleep(0.45)
                    result = enroll_identity(name, paths, settings)
                return jsonify(
                    name=name, accepted_images=result.accepted_images,
                    skipped_images=len(result.skipped_images),
                ), 201
            finally:
                bridge.processing_lock.release()
        except VisionError as exc:
            return jsonify(error=str(exc)), 422
        except (requests.RequestException, OSError, ValueError):
            return jsonify(error="Capture failed. No new identity was saved."), 503
        finally:
            bridge.paused.clear()
            bridge.enrollment_lock.release()

    return bp
