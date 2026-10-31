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

# How long a rule effect queued by an explicit dry-run (execute true) keeps the
# light/buzzer on at the device. Shared contract with the firmware (12-03).
DRY_RUN_EFFECT_DURATION_SECONDS = 5


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


def enqueue_rule_effects(device_id, result, now):
    """Internal (dry-run execute only): turn a rules_engine.dispatch() result
    into pending manual-queue entries and return the queued actuator names.

    Entries are stored exactly like create_command does, so expiry, delivery
    and status keep working. "door_lock" is deliberately NOT in the public
    ACTUATORS tuple: this helper is its only producer and is reachable only
    from the admin-gated dry-run route.
    """
    queued = []
    device_commands = _pending_commands.setdefault(device_id, {})

    rule_command = result.get("rule_command")
    if rule_command and rule_command.get("actuator") in ("light", "buzzer"):
        actuator = rule_command["actuator"]
        action = {"on": True, **(rule_command.get("action") or {})}
        action["duration_s"] = DRY_RUN_EFFECT_DURATION_SECONDS
        device_commands[actuator] = {"action": action, "issued_at": now}
        queued.append(actuator)

    if result.get("door_action") == "lock":
        device_commands["door_lock"] = {"action": {}, "issued_at": now}
        queued.append("door_lock")

    return queued


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


# How long a delivered command still reads back as "delivered" to a client
# polling for status (CTRL-05), instead of dropping to "none". Deliberately
# short and separate from the door-cooldown window: this is UI feedback
# freshness, not an actuation-suppression concern.
_STATUS_RECENT_SECONDS = 15


def command_status(device_id, actuator, now):
    """Status of the most recent command issued for `actuator` on
    `device_id`, for CTRL-05's pending -> entregado feedback:
    - "pending": still queued, not yet popped by a /frames poll.
    - "delivered": popped (and, if pan_tilt/light/buzzer, applied) within
      the last _STATUS_RECENT_SECONDS.
    - "none": no command ever issued, or it expired/aged out of the
      "delivered" window without ever being marked delivered.
    """
    device_commands = _pending_commands.get(device_id, {})
    entry = device_commands.get(actuator)
    if entry is not None:
        expiry_seconds = current_app.config.get("COMMAND_EXPIRY_SECONDS", 8)
        if (now - entry["issued_at"]).total_seconds() < expiry_seconds:
            return "pending"

    delivered_at = _manual_command_delivered.get(device_id, {}).get(actuator)
    if (
        delivered_at is not None
        and (now - delivered_at).total_seconds() < _STATUS_RECENT_SECONDS
    ):
        return "delivered"

    return "none"


@commands_bp.route("/commands/<device_id>/status", methods=["GET"])
def get_command_status(device_id):
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    actuator = request.args.get("actuator")
    if actuator not in ACTUATORS:
        return (
            jsonify({"error": f"actuator must be one of {list(ACTUATORS)}"}),
            400,
        )

    now = datetime.datetime.utcnow()
    return jsonify(
        {
            "device_id": device_id,
            "actuator": actuator,
            "status": command_status(device_id, actuator, now),
        }
    )
