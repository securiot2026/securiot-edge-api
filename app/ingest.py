import datetime

from flask import Blueprint, current_app, jsonify, request

from app.models import Reading

ingest_bp = Blueprint("ingest", __name__)

REQUIRED_FIELDS = (
    "reading_id",
    "device_id",
    "zone_id",
    "sensor_type",
    "value",
    "recorded_at",
)


def _parse_recorded_at(raw):
    return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))


@ingest_bp.route("/ingest", methods=["POST"])
def ingest():
    device_key = request.headers.get("X-Device-Key")
    if not device_key or device_key != current_app.config["DEVICE_SHARED_SECRET"]:
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    missing = [field for field in REQUIRED_FIELDS if not payload.get(field)]
    if missing:
        return jsonify({"error": "missing fields", "fields": missing}), 400

    try:
        recorded_at = _parse_recorded_at(payload["recorded_at"])
    except ValueError:
        return jsonify({"error": "invalid recorded_at"}), 400

    Reading.insert(
        reading_id=payload["reading_id"],
        device_id=payload["device_id"],
        zone_id=payload["zone_id"],
        sensor_type=payload["sensor_type"],
        value=payload["value"],
        recorded_at=recorded_at,
        synced=False,
    ).on_conflict_ignore().execute()

    return jsonify({"status": "buffered"}), 201
