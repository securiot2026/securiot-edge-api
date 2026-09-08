"""Synthetic device simulator that stands in for the physical ESP32 through Phase 4.

Periodically POSTs door-contact readings to a running Edge API's /ingest
endpoint, exercising the same shared-secret auth and idempotency (unique
reading_id per POST) path the real device relay relies on.
"""
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import requests

EDGE_API_URL = os.environ.get("EDGE_API_URL", "http://localhost:5001")
DEVICE_SHARED_SECRET = os.environ.get("DEVICE_SHARED_SECRET", "")
DEVICE_ID = os.environ.get("DEVICE_ID", "dev-1")
ZONE_ID = os.environ.get("ZONE_ID", "zone-1")
INTERVAL_SECONDS = float(os.environ.get("INTERVAL_SECONDS", "5"))

INGEST_URL = f"{EDGE_API_URL.rstrip('/')}/ingest"


def build_reading(is_open: bool) -> dict:
    return {
        "reading_id": str(uuid.uuid4()),
        "device_id": DEVICE_ID,
        "zone_id": ZONE_ID,
        "sensor_type": "door_contact",
        "value": "open" if is_open else "closed",
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def post_reading(reading: dict) -> None:
    headers = {"X-Device-Key": DEVICE_SHARED_SECRET, "Content-Type": "application/json"}
    try:
        response = requests.post(INGEST_URL, json=reading, headers=headers, timeout=5)
        print(
            f"POST {INGEST_URL} value={reading['value']} reading_id={reading['reading_id']} "
            f"-> {response.status_code}",
            flush=True,
        )
    except requests.RequestException as exc:
        print(f"POST {INGEST_URL} value={reading['value']} -> ERROR: {exc}", flush=True)


def main() -> None:
    print(
        f"Starting device simulator: device_id={DEVICE_ID} zone_id={ZONE_ID} "
        f"target={INGEST_URL} interval={INTERVAL_SECONDS}s",
        flush=True,
    )
    is_open = True
    try:
        while True:
            post_reading(build_reading(is_open))
            is_open = not is_open
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("Simulator stopped.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
