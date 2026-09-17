import json

import pytest

from app import create_app
from app.models import Rule


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


ADMIN_HEADERS = {"X-Admin-Key": "adminsecret"}


def test_rules_endpoints_require_admin_key(client):
    assert client.get("/rules").status_code == 401
    assert client.post("/rules", json={}).status_code == 401
    assert client.delete("/rules/1").status_code == 401


def test_list_rules_returns_seeded_migrated_rules(app, client):
    with app.app_context():
        response = client.get("/rules", headers=ADMIN_HEADERS)
        assert response.status_code == 200
        names = {r["name"] for r in response.get_json()}
        assert "Legacy door lock — person (migrated)" in names


def test_create_rule_rejects_invalid_condition_type(client):
    response = client.post(
        "/rules",
        json={
            "name": "Bad rule",
            "condition_type": "not-a-real-type",
            "actions": [{"type": "alert"}],
        },
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 400


def test_create_rule_requires_operator_and_sensor_type_for_sensor_threshold(client):
    response = client.post(
        "/rules",
        json={
            "name": "Temp too high",
            "condition_type": "sensor_threshold",
            "match_value": "40",
            "actions": [{"type": "alert", "severity": "high"}],
        },
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 400
    assert "operator" in json.dumps(response.get_json())


def test_create_update_delete_rule_round_trip(app, client):
    with app.app_context():
        created = client.post(
            "/rules",
            json={
                "name": "Temp too high",
                "condition_type": "sensor_threshold",
                "operator": ">",
                "sensor_type": "temperature",
                "match_value": "40",
                "actions": [{"type": "alert", "severity": "high"}],
                "priority": 5,
            },
            headers=ADMIN_HEADERS,
        )
        assert created.status_code == 201
        rule_id = created.get_json()["id"]

        updated = client.patch(
            f"/rules/{rule_id}",
            json={"enabled": False},
            headers=ADMIN_HEADERS,
        )
        assert updated.status_code == 200
        assert updated.get_json()["enabled"] is False

        deleted = client.delete(f"/rules/{rule_id}", headers=ADMIN_HEADERS)
        assert deleted.status_code == 204
        assert Rule.get_or_none(Rule.id == rule_id) is None


def test_update_missing_rule_returns_404(client):
    response = client.patch(
        "/rules/999999", json={"enabled": False}, headers=ADMIN_HEADERS
    )
    assert response.status_code == 404


def test_dry_run_requires_admin_key(client):
    assert client.post("/rules/1/dry-run").status_code == 401


def test_dry_run_missing_rule_returns_404(client):
    response = client.post("/rules/999999/dry-run", headers=ADMIN_HEADERS)
    assert response.status_code == 404


def test_dry_run_fires_alert_action_and_updates_last_triggered(app, client):
    with app.app_context():
        created = client.post(
            "/rules",
            json={
                "name": "Manual test alert",
                "condition_type": "sensor_threshold",
                "operator": ">",
                "sensor_type": "temperature",
                "match_value": "40",
                "actions": [{"type": "alert", "severity": "high"}],
            },
            headers=ADMIN_HEADERS,
        )
        rule_id = created.get_json()["id"]
        assert created.get_json()["last_triggered_at"] is None

        response = client.post(f"/rules/{rule_id}/dry-run", headers=ADMIN_HEADERS)
        assert response.status_code == 200
        body = response.get_json()
        assert body["rule"]["last_triggered_at"] is not None
        assert Rule.get_by_id(rule_id).last_triggered_at is not None


def test_dry_run_rejects_disabled_rule(app, client):
    with app.app_context():
        created = client.post(
            "/rules",
            json={
                "name": "Disabled rule",
                "condition_type": "object_class",
                "match_value": "person",
                "actions": [{"type": "door_lock"}],
                "enabled": False,
            },
            headers=ADMIN_HEADERS,
        )
        rule_id = created.get_json()["id"]

        response = client.post(f"/rules/{rule_id}/dry-run", headers=ADMIN_HEADERS)
        assert response.status_code == 400
