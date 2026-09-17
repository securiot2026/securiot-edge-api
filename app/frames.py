import datetime
import json
import os
import tempfile
import uuid

from flask import Blueprint, current_app, jsonify, request
from PIL import Image, UnidentifiedImageError

from app import debounce, occupancy
from app.commands import manual_suppression_active, pop_deliverable_command
from app.detection import DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH, get_detector
from app.device_settings import get_face_recognition_enabled
from app.pan_tilt import compute_pan_tilt
from app.reading_buffer import buffer_reading
from app.rules_engine import dispatch, evaluate
from app.vision import VisionError, get_face_recognizer

frames_bp = Blueprint("frames", __name__)

# device_id -> bool, True once app/debounce.py has confirmed a detection
# episode for this device and until a non-qualifying frame ends it. This
# lets pan/tilt tracking and reading history continue on every frame of a
# confirmed episode, not just the single frame where debounce.record()
# reports escalate=True (debounce only reports that once per episode, by
# design; see app/debounce.py).
_active_episode = {}


def _response_payload(
    pan_delta,
    tilt_delta,
    door_action,
    alert,
    faces=None,
    manual_command=None,
    rule_command=None,
):
    payload = {
        "pan_delta": pan_delta,
        "tilt_delta": tilt_delta,
        "door_action": door_action,
        "alert": alert,
        "manual_command": manual_command,
        "rule_command": rule_command,
    }
    if faces is not None:
        payload["faces"] = [face.to_dict() for face in faces]
    return payload


def _empty_response(
    faces=None, manual_command=None, rule_command=None, alert=False, door_action=None
):
    return jsonify(
        _response_payload(
            None, None, door_action, alert, faces, manual_command, rule_command
        )
    )


@frames_bp.route("/frames", methods=["POST"])
def frames():
    device_key = request.headers.get("X-Device-Key")
    if not device_key or device_key != current_app.config["DEVICE_SHARED_SECRET"]:
        return jsonify({"error": "unauthorized"}), 401

    device_id = request.form.get("device_id")
    zone_id = request.form.get("zone_id")
    frame_file = request.files.get("frame")

    now = datetime.datetime.utcnow()

    # Preserve the uploaded filename on disk (not a random tempfile name):
    # MockDetector's test-only filename convention (used by automated tests
    # and scripts/run_test_folder.py) inspects the basename, not the bytes.
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, frame_file.filename or "frame.jpg")
    face_recognitions = None
    detections = []
    try:
        frame_file.save(tmp_path)

        # T-04-06 (DoS): reject undecodable image data with 400 before it
        # ever reaches a CPU-bound YOLO inference call.
        try:
            with Image.open(tmp_path) as img:
                img.verify()
        except (OSError, UnidentifiedImageError):
            return jsonify({"error": "invalid image data"}), 400

        detector = get_detector(current_app.config)
        try:
            detections = detector.detect(tmp_path)
        except Exception:
            return jsonify({"error": "invalid image data"}), 400

        if get_face_recognition_enabled(device_id):
            try:
                face_recognitions = get_face_recognizer(
                    current_app.config
                ).recognize_path(tmp_path)
            except VisionError as exc:
                current_app.logger.error("Face recognition unavailable: %s", exc)
                return (
                    jsonify({"error": "face recognition unavailable"}),
                    503,
                )

        # RULE-07/08: weapon and fire are severity conditions, evaluated on
        # every frame independent of the person/object presence debounce
        # below — a visible knife or fire is not something worth waiting
        # two frames to react to. image_path must be read here, before the
        # `finally` block below deletes it.
        severity_matches = evaluate(
            device_id,
            now,
            ("weapon", "fire"),
            detections=detections,
            image_path=tmp_path,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        os.rmdir(tmp_dir)

    # Manual commands are delivered on the very next /frames poll regardless
    # of detection state (CTRL-04) — pop before any early-return path so an
    # operator's pan/tilt/light/buzzer command is never gated on a person
    # being in view.
    manual_command = pop_deliverable_command(device_id, now)

    allowed_object_class = current_app.config.get("ALLOWED_OBJECT_CLASS", "backpack")
    qualifying = [
        d for d in detections if d["class_name"] in ("person", allowed_object_class)
    ]
    top_detection = (
        max(qualifying, key=lambda d: d["confidence"]) if qualifying else None
    )
    qualifying_frame = bool(top_detection)
    result = debounce.record(device_id, qualifying=qualifying_frame)

    object_matches = []
    if qualifying_frame:
        if result["escalate"]:
            _active_episode[device_id] = True
            occupancy.record_entry_and_get_count(device_id, now)

        if _active_episode.get(device_id, False):
            object_matches = evaluate(
                device_id,
                now,
                ("object_class", "occupancy"),
                detections=detections,
                occupancy_count=occupancy.get_count(device_id, now),
            )
    else:
        _active_episode[device_id] = False

    matched_rules = sorted(
        severity_matches + object_matches, key=lambda r: r.priority
    )

    if not matched_rules and not qualifying_frame:
        # Nothing matched and nothing is in view: identical to the old
        # early-return shape, still honoring any manual command/pending
        # weapon-fire dispatch (there is none here, matched_rules is empty).
        return _empty_response(face_recognitions, manual_command), 200

    # CTRL-06: a manual command delivered for the "door" actuator (test-only
    # public-API-inaccessible value, see app/commands.py) suppresses the
    # rules engine's automatic door_lock action for the manual cooldown
    # window — checked before dispatch so a suppressed rule never buffers a
    # door_contact Reading or consumes its own per-rule cooldown.
    suppress_door = manual_suppression_active(
        device_id,
        "door",
        now,
        current_app.config.get("DOOR_ACTION_COOLDOWN_SECONDS", 30),
    )
    dispatch_result = dispatch(
        device_id,
        zone_id,
        matched_rules,
        now,
        manual_command=manual_command,
        suppress_door=suppress_door,
    )
    door_action = dispatch_result["door_action"]
    rule_command = dispatch_result["rule_command"]

    if not qualifying_frame:
        # Weapon/fire matched on a frame with no person/object in view:
        # still dispatch (door lock / alert / rule_command above), but there
        # is no pan/tilt tracking or camera_detection Reading to buffer.
        return (
            _empty_response(
                face_recognitions,
                manual_command,
                rule_command,
                alert=bool(matched_rules),
                door_action=door_action,
            ),
            200,
        )

    if not _active_episode.get(device_id, False):
        return _empty_response(face_recognitions, manual_command, rule_command), 200

    pan_delta = tilt_delta = None
    # A manual pan/tilt command prevails over the same-cycle automatic
    # tracking delta (CTRL-06) — never blend the two into a combined/jittery
    # servo move. The firmware applies whichever one is present.
    if manual_command is None or manual_command["actuator"] != "pan_tilt":
        if top_detection["class_name"] == "person":
            frame_width = int(request.form.get("frame_width", DEFAULT_FRAME_WIDTH))
            frame_height = int(request.form.get("frame_height", DEFAULT_FRAME_HEIGHT))
            delta = compute_pan_tilt(top_detection["bbox"], frame_width, frame_height)
            pan_delta = delta["pan_delta"]
            tilt_delta = delta["tilt_delta"]

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
        recorded_at=now,
    )

    return (
        jsonify(
            _response_payload(
                pan_delta,
                tilt_delta,
                door_action,
                True,
                face_recognitions,
                manual_command,
                rule_command,
            )
        ),
        200,
    )
