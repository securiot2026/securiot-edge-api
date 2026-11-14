import io
import json

import pytest
from PIL import Image

from app import create_app
from app.commands import _manual_command_delivered, _pending_commands
from app.models import Rule


class FakeConfig:
    EDGE_DB_PATH = ":memory:"
    DEVICE_SHARED_SECRET = "testsecret"
    EDGE_ADMIN_API_KEY = "adminsecret"
    CLOUD_API_URL = "http://cloud.example"
    CLOUD_DEVICE_API_KEY = "cloudkey"
    RELAY_INTERVAL_SECONDS = 3600

    DETECTION_BACKEND = "mock"
    YOLO_MODEL_PATH = "yolov8n.pt"
    ALLOWED_OBJECT_CLASS = "backpack"
    DETECTION_CONFIDENCE_THRESHOLD = 0.5
    DETECTION_DEBOUNCE_COUNT = 2
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


def _solid_color_bytes(color):
    output = io.BytesIO()
    Image.new("RGB", (640, 480), color).save(output, format="JPEG")
    return output.getvalue()


def _post_frame(client, fixture_name, device_id, zone_id="zone-1", color=(30, 30, 30)):
    data = {
        "device_id": device_id,
        "zone_id": zone_id,
        "frame": (io.BytesIO(_solid_color_bytes(color)), fixture_name),
    }
    return client.post(
        "/frames",
        data=data,
        headers={"X-Device-Key": "testsecret"},
        content_type="multipart/form-data",
    )


def test_seed_migrates_legacy_door_rule_on_first_boot(app):
    with app.app_context():
        rules = list(Rule.select())
        names = {r.name for r in rules}
        assert "Legacy door lock — person (migrated)" in names
        assert "Legacy door lock — backpack (migrated)" in names
        assert all(r.enabled for r in rules)


def test_seed_only_skips_when_table_is_non_empty(app):
    with app.app_context():
        from app.rules_engine import seed_default_rules_if_empty

        # Already seeded on app boot (table non-empty) — calling again is a
        # no-op, an operator's edits to the seeded rows are never clobbered.
        Rule.update(name="Operator-renamed rule").where(
            Rule.name.contains("migrated")
        ).execute()
        seed_default_rules_if_empty(
            allowed_object_class="backpack", cooldown_seconds=30
        )
        renamed = Rule.select().where(Rule.name == "Operator-renamed rule")
        assert renamed.count() == 2

        # Documented MVP simplification: an operator deleting *every* rule
        # is indistinguishable from a never-seeded table, so the next boot
        # does reseed the migrated-equivalent rules (RULE-12) rather than
        # leaving the device with zero rules and no door lock at all.
        Rule.delete().execute()
        seed_default_rules_if_empty(
            allowed_object_class="backpack", cooldown_seconds=30
        )
        assert Rule.select().count() == 2


def test_weapon_condition_fires_on_knife_class_regardless_of_debounce(app, client):
    with app.app_context():
        Rule.delete().execute()  # isolate from the seeded person/backpack rules
        Rule.create(
            device_id=None,
            name="Weapon alert",
            condition_type="weapon",
            actions=json.dumps([{"type": "alert", "severity": "critical"}]),
            enabled=True,
            priority=10,
            cooldown_seconds=30,
        )

        # A single frame is enough — weapon detection is not debounced.
        response = _post_frame(client, "knife.jpg", device_id="dev-weapon")

        assert response.status_code == 200
        body = response.get_json()
        assert body["alert"] is True

        from app.models import Reading

        alerts = list(
            Reading.select().where(
                Reading.device_id == "dev-weapon", Reading.sensor_type == "rule_alert"
            )
        )
        assert len(alerts) == 1
        payload = json.loads(alerts[0].value)
        assert payload["severity"] == "critical"
        assert payload["rule_name"] == "Weapon alert"


def test_fire_condition_fires_on_bright_red_dominant_frame(app, client):
    with app.app_context():
        Rule.delete().execute()
        Rule.create(
            device_id=None,
            name="Fire lockdown",
            condition_type="fire",
            actions=json.dumps([{"type": "door_lock"}]),
            enabled=True,
            priority=5,
            cooldown_seconds=30,
        )

        response = _post_frame(
            client, "empty.jpg", device_id="dev-fire", color=(255, 140, 60)
        )

        assert response.status_code == 200
        assert response.get_json()["door_action"] == "lock"


def test_fire_condition_does_not_fire_on_neutral_frame(app, client):
    with app.app_context():
        Rule.delete().execute()
        Rule.create(
            device_id=None,
            name="Fire lockdown",
            condition_type="fire",
            actions=json.dumps([{"type": "door_lock"}]),
            enabled=True,
            priority=5,
            cooldown_seconds=30,
        )

        response = _post_frame(client, "empty.jpg", device_id="dev-nofire")

        assert response.get_json()["door_action"] is None


def test_occupancy_rule_fires_only_once_daily_limit_reached(app, client):
    with app.app_context():
        Rule.delete().execute()
        Rule.create(
            device_id=None,
            name="Occupancy lock",
            condition_type="occupancy",
            match_value="2",
            actions=json.dumps([{"type": "door_lock"}]),
            enabled=True,
            priority=5,
            cooldown_seconds=30,
        )

        # Two confirmed episodes (2 frames each, DETECTION_DEBOUNCE_COUNT=2)
        # for the same device: first episode brings the count to 1, not
        # enough; a gap frame ends the episode; the second episode brings it
        # to 2, reaching the configured daily limit.
        _post_frame(client, "person_left.jpg", device_id="dev-occupancy")
        first_episode = _post_frame(
            client, "person_left.jpg", device_id="dev-occupancy"
        )
        assert first_episode.get_json()["door_action"] is None

        _post_frame(client, "empty.jpg", device_id="dev-occupancy")  # ends episode

        _post_frame(client, "person_left.jpg", device_id="dev-occupancy")
        second_episode = _post_frame(
            client, "person_left.jpg", device_id="dev-occupancy"
        )
        assert second_episode.get_json()["door_action"] == "lock"


def test_priority_most_restrictive_wins_on_same_actuator(app, client):
    with app.app_context():
        Rule.delete().execute()
        # Two rules both target "light" for the same weapon condition;
        # priority=1 (more restrictive) must win over priority=50.
        Rule.create(
            device_id=None,
            name="Weapon: red light (restrictive)",
            condition_type="weapon",
            actions=json.dumps([{"type": "light", "color": "red"}]),
            enabled=True,
            priority=1,
            cooldown_seconds=30,
        )
        Rule.create(
            device_id=None,
            name="Weapon: yellow light (lenient)",
            condition_type="weapon",
            actions=json.dumps([{"type": "light", "color": "yellow"}]),
            enabled=True,
            priority=50,
            cooldown_seconds=30,
        )

        response = _post_frame(client, "knife.jpg", device_id="dev-priority")

        assert response.get_json()["rule_command"] == {
            "actuator": "light",
            "action": {"color": "red"},
            "rule": "Weapon: red light (restrictive)",
        }


def test_disabled_rule_never_matches(app, client):
    with app.app_context():
        Rule.delete().execute()
        Rule.create(
            device_id=None,
            name="Disabled weapon rule",
            condition_type="weapon",
            actions=json.dumps([{"type": "door_lock"}]),
            enabled=False,
            priority=1,
            cooldown_seconds=30,
        )

        response = _post_frame(client, "knife.jpg", device_id="dev-disabled")

        assert response.get_json()["door_action"] is None


def test_manual_command_suppresses_rule_command_on_same_actuator(app, client):
    with app.app_context():
        import datetime

        Rule.delete().execute()
        Rule.create(
            device_id=None,
            name="Weapon: light",
            condition_type="weapon",
            actions=json.dumps([{"type": "light", "color": "red"}]),
            enabled=True,
            priority=1,
            cooldown_seconds=30,
        )
        _pending_commands["dev-manual-light"] = {
            "light": {
                "action": {"on": True},
                "issued_at": datetime.datetime.utcnow(),
            }
        }

        response = _post_frame(client, "knife.jpg", device_id="dev-manual-light")

        body = response.get_json()
        assert body["manual_command"]["actuator"] == "light"
        assert body["rule_command"] is None
