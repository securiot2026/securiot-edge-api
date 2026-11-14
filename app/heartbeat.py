"""Device last-seen tracking and rate-limited heartbeat push to Cloud (D-06).

Every authenticated /frames request calls record_seen(), even when the scene is
empty or the image is undecodable: the device proved it is alive. A scheduled
heartbeat_cycle() then pushes at most one small heartbeat per device per
HEARTBEAT_INTERVAL_SECONDS through Cloud's existing authenticated telemetry
route. Heartbeats are never written to the Reading table and never go through
buffer_reading, so detection-only Reading counts stay untouched.

Module-level, lock-protected dict, same MVP pattern as app/frame_cache.py:
resets on restart.
"""

import datetime
import logging
import threading
import uuid

import requests
from flask import current_app

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30

_lock = threading.Lock()
# device_id -> {"zone_id", "seen_at", "reported_at", "last_push_at"}
_devices = {}


def reset():
    with _lock:
        _devices.clear()


def record_seen(device_id, zone_id, now):
    """Refresh the in-memory last-seen record. now is a UTC datetime."""
    with _lock:
        entry = _devices.get(device_id)
        if entry is None:
            _devices[device_id] = {
                "zone_id": zone_id,
                "seen_at": now,
                "reported_at": None,
                "last_push_at": None,
            }
            return
        entry["zone_id"] = zone_id
        entry["seen_at"] = now


def _utc_z(moment):
    """ISO8601 with an explicit Z suffix so Cloud never parses it as local time."""
    if moment.tzinfo is not None:
        moment = moment.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return moment.isoformat(timespec="seconds") + "Z"


def heartbeat_cycle(now=None):
    """Push one heartbeat for each device that has new activity and whose last
    successful push is older than the interval. A failed push leaves the device
    due so the next cycle retries."""
    now = now or datetime.datetime.utcnow()
    cloud_api_url = current_app.config["CLOUD_API_URL"]
    device_api_key = current_app.config["CLOUD_DEVICE_API_KEY"]
    interval = datetime.timedelta(
        seconds=current_app.config.get(
            "HEARTBEAT_INTERVAL_SECONDS", DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        )
    )

    with _lock:
        due = [
            (device_id, entry["zone_id"], entry["seen_at"])
            for device_id, entry in _devices.items()
            if (entry["reported_at"] is None or entry["seen_at"] > entry["reported_at"])
            and (entry["last_push_at"] is None or now - entry["last_push_at"] >= interval)
        ]

    for device_id, zone_id, seen_at in due:
        body = {
            "reading_id": str(uuid.uuid4()),
            "device_id": device_id,
            "zone_id": zone_id,
            "sensor_type": "heartbeat",
            "value": "alive",
            "recorded_at": _utc_z(seen_at),
        }
        try:
            response = requests.post(
                f"{cloud_api_url}/api/v1/telemetry",
                json=body,
                headers={"X-Device-Key": device_api_key},
                timeout=10,
            )
        except requests.RequestException:
            logger.warning("heartbeat: connection error for device %s", device_id)
            continue

        if 200 <= response.status_code < 300:
            with _lock:
                entry = _devices.get(device_id)
                if entry is not None:
                    # Mark only the snapshot we sent: a frame that arrived while
                    # the request was in flight stays due for the next interval.
                    entry["reported_at"] = seen_at
                    entry["last_push_at"] = now
        else:
            logger.warning(
                "heartbeat: cloud API returned %s for device %s",
                response.status_code,
                device_id,
            )
