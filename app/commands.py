import datetime

from flask import Blueprint, current_app, jsonify, request

from app.auth import require_admin_key

commands_bp = Blueprint("commands", __name__)

ACTUATORS = ("pan_tilt", "light", "buzzer")

# device_id -> {actuator: {"action": dict, "issued_at": datetime}}. Module-level
# and in-memory, matching app/frames.py's _last_door_action accepted MVP
# limitation (resets on restart, single-prototype-device scope). One pending
# command slot per actuator per device: a new command for the same actuator
# replaces, does not stack.
_pending_commands = {}

# device_id -> {actuator: datetime of last manual delivery to /frames}. Drives
# CTRL-06 cooldown suppression of the automatic door-lock action. Only ever
# populated for "pan_tilt"/"light"/"buzzer" by real command delivery; the
# "door" key is written only by tests, directly, to exercise the suppression
# check in isolation (see app/frames.py's cooldown block and Open Question 1
# in 07-RESEARCH.md — "door" is not a value accepted by create_command()).
_manual_command_delivered = {}


def _validate_action(actuator, action):
    if actuator == "pan_tilt":
        if (
            not isinstance(action, dict)
            or not isinstance(action.get("pan_delta"), (int, float))
            or not isinstance(action.get("tilt_delta"), (int, float))
        ):
            return "action must be {'pan_delta': number, 'tilt_delta': number} for actuator 'pan_tilt'"
    elif actuator in ("light", "buzzer"):
        if not isinstance(action, dict) or not isinstance(action.get("on"), bool):
            return f"action must be {{'on': boolean}} for actuator '{actuator}'"
    return None


@commands_bp.route("/commands", methods=["POST"])
def create_command():
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    payload = request.get_json(silent=True) or {}
    device_id = payload.get("device_id")
    actuator = payload.get("actuator")
    action = payload.get("action")

    if not device_id or not isinstance(device_id, str):
        return jsonify({"error": "device_id (string) is required"}), 400
    if actuator not in ACTUATORS:
        return (
            jsonify({"error": f"actuator must be one of {list(ACTUATORS)}"}),
            400,
        )
    validation_error = _validate_action(actuator, action)
    if validation_error is not None:
        return jsonify({"error": validation_error}), 400

    issued_at = datetime.datetime.utcnow()
    device_commands = _pending_commands.setdefault(device_id, {})
    device_commands[actuator] = {"action": action, "issued_at": issued_at}

    return (
        jsonify(
            {
                "device_id": device_id,
                "actuator": actuator,
                "action": action,
                "issued_at": issued_at.isoformat() + "Z",
            }
        ),
        201,
    )


def pop_deliverable_command(device_id, now):
    """Pop and return the single deliverable pending command for a device, if
    any: the most recently issued command across actuators that has not yet
    expired. Expired commands are discarded silently (never delivered late).
    Returns None if no command is pending or the only pending one(s) expired.
    """
    device_commands = _pending_commands.get(device_id)
    if not device_commands:
        return None

    expiry_seconds = current_app.config.get("COMMAND_EXPIRY_SECONDS", 8)

    live = {
        actuator: entry
        for actuator, entry in device_commands.items()
        if (now - entry["issued_at"]).total_seconds() < expiry_seconds
    }
    _pending_commands[device_id] = live
    if not live:
        return None

    actuator, entry = max(live.items(), key=lambda item: item[1]["issued_at"])
    del _pending_commands[device_id][actuator]

    _manual_command_delivered.setdefault(device_id, {})[actuator] = now

    return {
        "actuator": actuator,
        "action": entry["action"],
        "issued_at": entry["issued_at"].isoformat() + "Z",
    }


def manual_suppression_active(device_id, actuator, now, cooldown_seconds):
    """True while a manual command delivered for `actuator` on `device_id`
    is still within its cooldown window — used to suppress the matching
    automatic action for that same actuator (CTRL-06)."""
    delivered_at = _manual_command_delivered.get(device_id, {}).get(actuator)
    return (
        delivered_at is not None
        and (now - delivered_at).total_seconds() < cooldown_seconds
    )
