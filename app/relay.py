import datetime
import logging

import requests
from flask import current_app

from app.models import Reading

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 300


def _backoff_seconds(sync_attempts):
    return min(2 ** sync_attempts, MAX_BACKOFF_SECONDS)


def relay_cycle():
    now = datetime.datetime.utcnow()
    cloud_api_url = current_app.config["CLOUD_API_URL"]
    device_api_key = current_app.config["CLOUD_DEVICE_API_KEY"]

    pending = Reading.select().where(
        (Reading.synced == False)  # noqa: E712
        & (Reading.next_attempt_at.is_null() | (Reading.next_attempt_at <= now))
    )

    for reading in pending:
        body = {
            "reading_id": reading.reading_id,
            "device_id": reading.device_id,
            "zone_id": reading.zone_id,
            "sensor_type": reading.sensor_type,
            "value": reading.value,
            "recorded_at": reading.recorded_at.isoformat(),
        }
        try:
            response = requests.post(
                f"{cloud_api_url}/api/v1/telemetry",
                json=body,
                headers={"X-Device-Key": device_api_key},
                timeout=10,
            )
        except requests.RequestException:
            logger.warning(
                "relay: connection error forwarding reading %s", reading.reading_id
            )
            reading.sync_attempts += 1
            reading.next_attempt_at = now + datetime.timedelta(
                seconds=_backoff_seconds(reading.sync_attempts)
            )
            reading.save()
            continue

        if 200 <= response.status_code < 300:
            reading.synced = True
            reading.save()
        else:
            logger.warning(
                "relay: cloud API returned %s for reading %s",
                response.status_code,
                reading.reading_id,
            )
            reading.sync_attempts += 1
            reading.next_attempt_at = now + datetime.timedelta(
                seconds=_backoff_seconds(reading.sync_attempts)
            )
            reading.save()
