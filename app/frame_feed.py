import base64

from flask import Blueprint, jsonify

from app.auth import require_admin_key
from app.frame_cache import get_latest_frame

frame_feed_bp = Blueprint("frame_feed", __name__)


@frame_feed_bp.route("/frames/latest/<device_id>", methods=["GET"])
def latest_frame(device_id):
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    cached = get_latest_frame(device_id)
    if cached is None:
        return jsonify({"error": "no frame captured yet for this device"}), 404

    return (
        jsonify(
            {
                "device_id": device_id,
                "frame": base64.b64encode(cached["jpeg_bytes"]).decode("ascii"),
                "captured_at": cached["captured_at"].isoformat() + "Z",
            }
        ),
        200,
    )
