import datetime

from flask import Blueprint, current_app, jsonify, request

from app.auth import require_admin_key
from app.models import DeviceSetting

device_settings_bp = Blueprint("device_settings", __name__)


def get_face_recognition_enabled(device_id):
    setting = DeviceSetting.get_or_none(DeviceSetting.device_id == device_id)
    if setting is not None and setting.face_recognition_enabled is not None:
        return setting.face_recognition_enabled
    return current_app.config.get("FACE_RECOGNITION_ENABLED", False)


@device_settings_bp.route("/devices/<device_id>/face-recognition", methods=["GET"])
def get_face_recognition(device_id):
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    return (
        jsonify(
            {
                "device_id": device_id,
                "face_recognition_enabled": get_face_recognition_enabled(device_id),
            }
        ),
        200,
    )


@device_settings_bp.route("/devices/<device_id>/face-recognition", methods=["PATCH"])
def patch_face_recognition(device_id):
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    payload = request.get_json(silent=True) or {}
    if "enabled" not in payload or not isinstance(payload["enabled"], bool):
        return jsonify({"error": "enabled (boolean) is required"}), 400

    setting, _ = DeviceSetting.get_or_create(device_id=device_id)
    setting.face_recognition_enabled = payload["enabled"]
    setting.updated_at = datetime.datetime.utcnow()
    setting.save()

    return (
        jsonify(
            {
                "device_id": device_id,
                "face_recognition_enabled": get_face_recognition_enabled(device_id),
            }
        ),
        200,
    )
