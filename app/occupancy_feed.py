import datetime

from flask import Blueprint, jsonify

from app import occupancy
from app.auth import require_admin_key

occupancy_feed_bp = Blueprint("occupancy_feed", __name__)


@occupancy_feed_bp.route("/occupancy/<device_id>", methods=["GET"])
def get_occupancy(device_id):
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    now = datetime.datetime.utcnow()
    return jsonify(
        {
            "device_id": device_id,
            "count": occupancy.get_count(device_id, now),
            "date": now.date().isoformat(),
        }
    )
