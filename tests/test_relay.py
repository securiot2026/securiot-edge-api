import datetime

import pytest
import requests

from app import create_app
from app.models import Reading
from app.relay import relay_cycle


class FakeConfig:
    EDGE_DB_PATH = ":memory:"
    DEVICE_SHARED_SECRET = "testsecret"
    CLOUD_API_URL = "http://cloud.example"
    CLOUD_DEVICE_API_KEY = "cloudkey"
    RELAY_INTERVAL_SECONDS = 3600


@pytest.fixture
def app():
    return create_app(config_object=FakeConfig, start_relay=False)


def make_reading(**overrides):
    defaults = dict(
        reading_id="r-1",
        device_id="dev-1",
        zone_id="zone-1",
        sensor_type="door_contact",
        value="open",
        recorded_at=datetime.datetime(2026, 9, 7, tzinfo=datetime.timezone.utc),
    )
    defaults.update(overrides)
    return Reading.create(**defaults)


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_relay_marks_synced_on_success(app, monkeypatch):
    with app.app_context():
        reading = make_reading()

        monkeypatch.setattr(
            "app.relay.requests.post", lambda *a, **k: FakeResponse(201)
        )
        relay_cycle()

        assert Reading.get_by_id(reading.id).synced is True


def test_relay_leaves_unsynced_on_connection_error(app, monkeypatch):
    with app.app_context():
        reading = make_reading(reading_id="r-2")

        def raise_connection_error(*a, **k):
            raise requests.ConnectionError("cloud unreachable")

        monkeypatch.setattr("app.relay.requests.post", raise_connection_error)

        relay_cycle()

        refreshed = Reading.get_by_id(reading.id)
        assert refreshed.synced is False
        assert refreshed.sync_attempts == 1
        assert refreshed.next_attempt_at is not None


def test_relay_does_not_resend_already_synced_row(app, monkeypatch):
    with app.app_context():
        reading = make_reading(reading_id="r-3", synced=True)

        calls = []
        monkeypatch.setattr(
            "app.relay.requests.post",
            lambda *a, **k: calls.append(1) or FakeResponse(201),
        )

        relay_cycle()

        assert calls == []
        assert Reading.get_by_id(reading.id).synced is True


def test_relay_skips_row_within_backoff_window(app, monkeypatch):
    with app.app_context():
        future = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
        reading = make_reading(
            reading_id="r-4", sync_attempts=2, next_attempt_at=future
        )

        calls = []
        monkeypatch.setattr(
            "app.relay.requests.post",
            lambda *a, **k: calls.append(1) or FakeResponse(201),
        )

        relay_cycle()

        assert calls == []
        refreshed = Reading.get_by_id(reading.id)
        assert refreshed.synced is False
        assert refreshed.sync_attempts == 2
