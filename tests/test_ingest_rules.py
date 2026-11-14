import datetime
import json

import pytest

from app import create_app
from app.models import Reading, Rule


class FakeConfig:
    EDGE_DB_PATH = ":memory:"
    DEVICE_SHARED_SECRET = "testsecret"
    EDGE_ADMIN_API_KEY = "adminsecret"
    CLOUD_API_URL = "http://cloud.example"
    CLOUD_DEVICE_API_KEY = "cloudkey"
    RELAY_INTERVAL_SECONDS = 3600
    DETECTION_BACKEND = "mock"
    ALLOWED_OBJECT_CLASS = "backpack"
    DOOR_ACTION_COOLDOWN_SECONDS = 30


@pytest.fixture
def app():
    return create_app(config_object=FakeConfig, start_relay=False)


@pytest.fixture
def client(app):
    return app.test_client()


def _ingest(client, **overrides):
    payload = {
        "reading_id": "r-1",
        "device_id": "dev-sensor",
        "zone_id": "zone-1",
        "sensor_type": "temperature",
        "value": "10",
        "recorded_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    payload.update(overrides)
    return client.post(
        "/ingest",
        json=payload,
        headers={"X-Device-Key": "testsecret"},
    )


def test_sensor_threshold_rule_fires_alert_on_ingested_reading(app, client):
    with app.app_context():
        Rule.create(
            device_id=None,
            name="Temp too high",
            condition_type="sensor_threshold",
            operator=">",
            sensor_type="temperature",
            match_value="35",
            actions=json.dumps([{"type": "alert", "severity": "high"}]),
            enabled=True,
            priority=5,
            cooldown_seconds=60,
        )

        response = _ingest(client, reading_id="r-hot", value="42")
        assert response.status_code == 201

        alerts = list(
            Reading.select().where(
                Reading.device_id == "dev-sensor",
                Reading.sensor_type == "rule_alert",
            )
        )
        assert len(alerts) == 1
        assert json.loads(alerts[0].value)["severity"] == "high"


def test_sensor_threshold_rule_does_not_fire_below_threshold(app, client):
    with app.app_context():
        Rule.create(
            device_id=None,
            name="Temp too high",
            condition_type="sensor_threshold",
            operator=">",
            sensor_type="temperature",
            match_value="35",
            actions=json.dumps([{"type": "alert", "severity": "high"}]),
            enabled=True,
            priority=5,
            cooldown_seconds=60,
        )

        response = _ingest(client, reading_id="r-cold", value="20")
        assert response.status_code == 201

        alerts = list(
            Reading.select().where(Reading.sensor_type == "rule_alert")
        )
        assert alerts == []


def test_sensor_threshold_rule_ignores_unrelated_sensor_type(app, client):
    with app.app_context():
        Rule.create(
            device_id=None,
            name="Temp too high",
            condition_type="sensor_threshold",
            operator=">",
            sensor_type="temperature",
            match_value="35",
            actions=json.dumps([{"type": "alert", "severity": "high"}]),
            enabled=True,
            priority=5,
            cooldown_seconds=60,
        )

        response = _ingest(
            client, reading_id="r-other", sensor_type="humidity", value="99"
        )
        assert response.status_code == 201

        alerts = list(
            Reading.select().where(Reading.sensor_type == "rule_alert")
        )
        assert alerts == []
