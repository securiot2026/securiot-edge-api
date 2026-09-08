import datetime
import json
import os
import tempfile
import uuid

from flask import Blueprint, current_app, jsonify, request

from app import debounce
from app.detection import DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH, get_detector
from app.pan_tilt import compute_pan_tilt
from app.reading_buffer import buffer_reading

frames_bp = Blueprint("frames", __name__)

# device_id -> datetime of the last door_action fired. Module-level and
# in-memory, matching app/debounce.py's accepted MVP limitation (resets on
# restart, single-prototype-device scope).
_last_door_action = {}


def _empty_response():
    return jsonify(
        {
            "pan_delta": None,
            "tilt_delta": None,
            "door_action": None,
            "alert": False,
        }
    )


@frames_bp.route("/frames", methods=["POST"])
def frames():
    device_key = request.headers.get("X-Device-Key")
    if not device_key or device_key != current_app.config["DEVICE_SHARED_SECRET"]:
        return jsonify({"error": "unauthorized"}), 401

    device_id = request.form.get("device_id")
    zone_id = request.form.get("zone_id")
    frame_file = request.files.get("frame")

    # Preserve the uploaded filename on disk (not a random tempfile name):
    # MockDetector's test-only filename convention (used by automated tests
    # and scripts/run_test_folder.py) inspects the basename, not the bytes.
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, frame_file.filename or "frame.jpg")
    try:
        frame_file.save(tmp_path)
        detector = get_detector(current_app.config)
        try:
            detections = detector.detect(tmp_path)
        except Exception:
            return jsonify({"error": "invalid image data"}), 400
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        os.rmdir(tmp_dir)

    allowed_object_class = current_app.config.get("ALLOWED_OBJECT_CLASS", "backpack")
    qualifying = [
        d for d in detections if d["class_name"] in ("person", allowed_object_class)
    ]
    top_detection = (
        max(qualifying, key=lambda d: d["confidence"]) if qualifying else None
    )

    result = debounce.record(device_id, qualifying=bool(top_detection))
    if not result["escalate"]:
        return _empty_response(), 200

    pan_delta = tilt_delta = None
    if top_detection["class_name"] == "person":
        frame_width = int(request.form.get("frame_width", DEFAULT_FRAME_WIDTH))
        frame_height = int(request.form.get("frame_height", DEFAULT_FRAME_HEIGHT))
        delta = compute_pan_tilt(top_detection["bbox"], frame_width, frame_height)
        pan_delta = delta["pan_delta"]
        tilt_delta = delta["tilt_delta"]

    recorded_at = datetime.datetime.utcnow()
    buffer_reading(
        reading_id=str(uuid.uuid4()),
        device_id=device_id,
        zone_id=zone_id,
        sensor_type="camera_detection",
        value=json.dumps(
            {
                "class_name": top_detection["class_name"],
                "confidence": top_detection["confidence"],
            }
        ),
        recorded_at=recorded_at,
    )

    # The door only "closes" once per detection episode, not once per frame:
    # while a person/object stays confirmed in view, every escalating frame
    # keeps buffering a camera_detection reading (history) and keeps
    # returning live pan/tilt (continuous tracking), but the door_contact
    # Reading (and the door_action itself) only fires again once the
    # cooldown window has elapsed for this device.
    cooldown_seconds = current_app.config.get("DOOR_ACTION_COOLDOWN_SECONDS", 30)
    last_fired = _last_door_action.get(device_id)
    within_cooldown = (
        last_fired is not None
        and (recorded_at - last_fired).total_seconds() < cooldown_seconds
    )

    door_action = None
    if not within_cooldown:
        buffer_reading(
            reading_id=str(uuid.uuid4()),
            device_id=device_id,
            zone_id=zone_id,
            sensor_type="door_contact",
            value="open",
            recorded_at=recorded_at,
        )
        _last_door_action[device_id] = recorded_at
        door_action = "lock"

    return (
        jsonify(
            {
                "pan_delta": pan_delta,
                "tilt_delta": tilt_delta,
                "door_action": door_action,
                "alert": True,
            }
        ),
        200,
    )
