import datetime
import json
import uuid

import pytest

from app import create_app
from app.commands import _manual_command_delivered, _pending_commands, pop_deliverable_command
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
    COMMAND_EXPIRY_SECONDS = 8


@pytest.fixture(autouse=True)
def _clear_command_state():
    _pending_commands.clear()
    _manual_command_delivered.clear()
    yield
    _pending_commands.clear()
    _manual_command_delivered.clear()


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


def _create_rule(client, **overrides):
    payload = {
        "name": "Manual test alert",
        "condition_type": "sensor_threshold",
        "operator": ">",
        "sensor_type": "temperature",
        "match_value": "40",
        "actions": [{"type": "alert", "severity": "high"}],
    }
    payload.update(overrides)
    created = client.post("/rules", json=payload, headers=ADMIN_HEADERS)
    assert created.status_code == 201
    return created.get_json()["id"]


def _dry_run(client, rule_id, body=None):
    return client.post(
        f"/rules/{rule_id}/dry-run",
        json=body if body is not None else {},
        headers=ADMIN_HEADERS,
    )


def _seed_reading(device_id, zone_id):
    Reading.create(
        reading_id=f"seed-{device_id}-{zone_id}-{uuid.uuid4()}",
        device_id=device_id,
        zone_id=zone_id,
        sensor_type="occupancy",
        value="1",
        recorded_at=datetime.datetime.utcnow(),
    )


def test_dry_run_alert_buffers_marked_reading_with_real_zone(app, client):
    with app.app_context():
        rule_id = _create_rule(client)

        response = _dry_run(client, rule_id, {"device_id": "dev-1", "zone_id": "zone-9"})
        assert response.status_code == 200
        body = response.get_json()
        assert body["dry_run"] is True
        assert body["device_id"] == "dev-1"
        assert body["zone_id"] == "zone-9"
        assert "rule" in body and "result" in body

        reading = Reading.get(
            (Reading.device_id == "dev-1") & (Reading.sensor_type == "rule_alert")
        )
        assert reading.zone_id == "zone-9"
        assert reading.synced is False
        value = json.loads(reading.value)
        assert value["severity"] == "high"
        assert value["message"].startswith("[Dry-run] ")
        assert value["dry_run"] is True


def test_dry_run_door_lock_buffers_door_contact_reading(app, client):
    with app.app_context():
        rule_id = _create_rule(
            client,
            name="Door test",
            condition_type="object_class",
            operator=None,
            sensor_type=None,
            match_value="person",
            actions=[{"type": "door_lock"}],
        )

        response = _dry_run(client, rule_id, {"device_id": "dev-1", "zone_id": "zone-9"})
        assert response.status_code == 200
        assert response.get_json()["result"]["door_action"] == "lock"

        reading = Reading.get(
            (Reading.device_id == "dev-1") & (Reading.sensor_type == "door_contact")
        )
        assert reading.zone_id == "zone-9"
        assert reading.value == "open"


def test_dry_run_leaves_last_triggered_at_none_for_never_fired_rule(app, client):
    with app.app_context():
        rule_id = _create_rule(client)

        response = _dry_run(client, rule_id, {"device_id": "dev-1", "zone_id": "zone-9"})
        assert response.status_code == 200
        assert response.get_json()["rule"]["last_triggered_at"] is None
        assert Rule.get_by_id(rule_id).last_triggered_at is None


def test_dry_run_keeps_previous_last_triggered_at_for_fired_rule(app, client):
    with app.app_context():
        rule_id = _create_rule(client)
        previous = datetime.datetime(2026, 1, 2, 3, 4, 5)
        rule = Rule.get_by_id(rule_id)
        rule.last_triggered_at = previous
        rule.save()

        response = _dry_run(client, rule_id, {"device_id": "dev-1", "zone_id": "zone-9"})
        assert response.status_code == 200
        assert response.get_json()["rule"]["last_triggered_at"] == previous.isoformat() + "Z"
        assert Rule.get_by_id(rule_id).last_triggered_at == previous


def test_dry_run_bypasses_cooldown(app, client):
    with app.app_context():
        rule_id = _create_rule(client, cooldown_seconds=3600)
        rule = Rule.get_by_id(rule_id)
        rule.last_triggered_at = datetime.datetime.utcnow()
        rule.save()

        response = _dry_run(client, rule_id, {"device_id": "dev-1", "zone_id": "zone-9"})
        assert response.status_code == 200
        assert (
            Reading.select()
            .where(
                (Reading.device_id == "dev-1") & (Reading.sensor_type == "rule_alert")
            )
            .count()
            == 1
        )


def test_dry_run_takes_zone_from_latest_reading_when_body_has_none(app, client):
    with app.app_context():
        rule_id = _create_rule(client)
        _seed_reading("dev-1", "zone-old")
        _seed_reading("dev-1", "zone-new")

        response = _dry_run(client, rule_id, {"device_id": "dev-1"})
        assert response.status_code == 200
        assert response.get_json()["zone_id"] == "zone-new"
        reading = Reading.get(
            (Reading.device_id == "dev-1") & (Reading.sensor_type == "rule_alert")
        )
        assert reading.zone_id == "zone-new"


def test_dry_run_400_when_zone_unresolvable(app, client):
    with app.app_context():
        rule_id = _create_rule(client)

        response = _dry_run(client, rule_id, {"device_id": "dev-1"})
        assert response.status_code == 400
        assert "zone_id" in response.get_json()["error"]
        assert Reading.select().count() == 0


def test_dry_run_uses_rule_device_id_when_body_omits_it(app, client):
    with app.app_context():
        rule_id = _create_rule(client, device_id="dev-7")

        response = _dry_run(client, rule_id, {"zone_id": "zone-3"})
        assert response.status_code == 200
        assert response.get_json()["device_id"] == "dev-7"


def test_dry_run_400_when_device_unresolvable(app, client):
    with app.app_context():
        rule_id = _create_rule(client)  # device_id null: applies to every device

        response = _dry_run(client, rule_id, {"zone_id": "zone-3"})
        assert response.status_code == 400
        assert "device_id" in response.get_json()["error"]


def test_dry_run_400_when_body_device_differs_from_rule_device(app, client):
    with app.app_context():
        rule_id = _create_rule(client, device_id="dev-7")

        response = _dry_run(client, rule_id, {"device_id": "dev-8", "zone_id": "zone-3"})
        assert response.status_code == 400
        assert Reading.select().count() == 0


@pytest.mark.parametrize(
    "body",
    [
        {"device_id": "", "zone_id": "zone-1"},
        {"device_id": "   ", "zone_id": "zone-1"},
        {"device_id": 5, "zone_id": "zone-1"},
        {"device_id": "dev-1", "zone_id": ""},
        {"device_id": "dev-1", "zone_id": "  "},
        {"device_id": "dev-1", "zone_id": 7},
    ],
)
def test_dry_run_rejects_blank_or_non_string_identity(app, client, body):
    with app.app_context():
        rule_id = _create_rule(client)

        response = _dry_run(client, rule_id, body)
        assert response.status_code == 400
        assert Reading.select().count() == 0


def test_dry_run_without_body_is_400_not_a_crash(app, client):
    with app.app_context():
        rule_id = _create_rule(client)

        response = client.post(f"/rules/{rule_id}/dry-run", headers=ADMIN_HEADERS)
        assert response.status_code == 400


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


IDENT = {"device_id": "dev-1", "zone_id": "zone-9"}


def test_dry_run_default_does_not_enqueue_commands(app, client):
    with app.app_context():
        rule_id = _create_rule(
            client,
            name="Light test",
            actions=[{"type": "light", "color": "red"}],
        )

        for body in (dict(IDENT), {**IDENT, "execute": False}):
            response = _dry_run(client, rule_id, body)
            assert response.status_code == 200
            data = response.get_json()
            assert data["executed"] is False
            assert data["queued"] == []
        assert _pending_commands.get("dev-1", {}) == {}


def test_dry_run_execute_light_queues_light_command(app, client):
    with app.app_context():
        rule_id = _create_rule(
            client,
            name="Light test",
            actions=[{"type": "light", "color": "red"}],
        )

        response = _dry_run(client, rule_id, {**IDENT, "execute": True})
        assert response.status_code == 200
        data = response.get_json()
        assert data["executed"] is True
        assert data["queued"] == ["light"]
        entry = _pending_commands["dev-1"]["light"]
        assert entry["action"] == {"on": True, "color": "red", "duration_s": 5}


def test_dry_run_execute_buzzer_queues_buzzer_command(app, client):
    with app.app_context():
        rule_id = _create_rule(
            client, name="Buzzer test", actions=[{"type": "buzzer"}]
        )

        response = _dry_run(client, rule_id, {**IDENT, "execute": True})
        assert response.status_code == 200
        assert response.get_json()["queued"] == ["buzzer"]
        assert _pending_commands["dev-1"]["buzzer"]["action"] == {
            "on": True,
            "duration_s": 5,
        }


def test_dry_run_execute_door_lock_queues_door_lock_command(app, client):
    with app.app_context():
        rule_id = _create_rule(
            client,
            name="Door test",
            condition_type="object_class",
            operator=None,
            sensor_type=None,
            match_value="person",
            actions=[{"type": "door_lock"}],
        )

        response = _dry_run(client, rule_id, {**IDENT, "execute": True})
        assert response.status_code == 200
        assert response.get_json()["queued"] == ["door_lock"]
        assert _pending_commands["dev-1"]["door_lock"]["action"] == {}


def test_dry_run_queued_command_is_deliverable_like_manual_command(app, client):
    with app.app_context():
        rule_id = _create_rule(
            client, name="Buzzer test", actions=[{"type": "buzzer"}]
        )
        _dry_run(client, rule_id, {**IDENT, "execute": True})

        delivered = pop_deliverable_command("dev-1", datetime.datetime.utcnow())
        assert delivered["actuator"] == "buzzer"
        assert delivered["action"] == {"on": True, "duration_s": 5}
        assert "issued_at" in delivered


@pytest.mark.parametrize("execute", ["true", 1, 0, "yes", [], {}])
def test_dry_run_non_boolean_execute_is_400_and_queues_nothing(app, client, execute):
    with app.app_context():
        rule_id = _create_rule(
            client, name="Buzzer test", actions=[{"type": "buzzer"}]
        )

        response = _dry_run(client, rule_id, {**IDENT, "execute": execute})
        assert response.status_code == 400
        assert "execute" in response.get_json()["error"]
        assert _pending_commands.get("dev-1", {}) == {}
        assert Reading.select().count() == 0


def test_dry_run_execute_alert_only_rule_queues_nothing(app, client):
    with app.app_context():
        rule_id = _create_rule(client)

        response = _dry_run(client, rule_id, {**IDENT, "execute": True})
        assert response.status_code == 200
        data = response.get_json()
        assert data["executed"] is True
        assert data["queued"] == []
        assert _pending_commands.get("dev-1", {}) == {}


def test_public_commands_endpoint_still_rejects_door_lock(client):
    response = client.post(
        "/commands",
        json={"device_id": "dev-1", "actuator": "door_lock", "action": {}},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 400
    assert _pending_commands.get("dev-1", {}) == {}
