import pytest

from app import create_app
from app.commands import _manual_command_delivered, _pending_commands


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
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024
    FACE_ENROLLMENT_MAX_CONTENT_LENGTH = 10 * 1024 * 1024

    FACE_RECOGNITION_ENABLED = False


@pytest.fixture
def app():
    return create_app(config_object=FakeConfig, start_relay=False)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def _clear_command_state():
    _pending_commands.clear()
    _manual_command_delivered.clear()
    yield
    _pending_commands.clear()
    _manual_command_delivered.clear()


def test_create_command_without_admin_key_returns_401(app, client):
    with app.app_context():
        response = client.post(
            "/commands",
            json={
                "device_id": "dev-1",
                "actuator": "light",
                "action": {"on": True},
            },
        )
        assert response.status_code == 401


def test_create_command_rejects_unknown_actuator(app, client):
    with app.app_context():
        response = client.post(
            "/commands",
            json={"device_id": "dev-1", "actuator": "sprinkler", "action": {}},
            headers={"X-Admin-Key": "adminsecret"},
        )
        assert response.status_code == 400


def test_create_command_rejects_malformed_pan_tilt_action(app, client):
    with app.app_context():
        response = client.post(
            "/commands",
            json={"device_id": "dev-1", "actuator": "pan_tilt", "action": {}},
            headers={"X-Admin-Key": "adminsecret"},
        )
        assert response.status_code == 400


def test_create_command_accepts_valid_light_command(app, client):
    with app.app_context():
        response = client.post(
            "/commands",
            json={"device_id": "dev-1", "actuator": "light", "action": {"on": True}},
            headers={"X-Admin-Key": "adminsecret"},
        )
        assert response.status_code == 201
        body = response.get_json()
        assert body["device_id"] == "dev-1"
        assert body["actuator"] == "light"
        assert body["action"] == {"on": True}


def test_second_command_for_same_actuator_replaces_first(app, client):
    with app.app_context():
        client.post(
            "/commands",
            json={"device_id": "dev-1", "actuator": "buzzer", "action": {"on": True}},
            headers={"X-Admin-Key": "adminsecret"},
        )
        client.post(
            "/commands",
            json={
                "device_id": "dev-1",
                "actuator": "buzzer",
                "action": {"on": False},
            },
            headers={"X-Admin-Key": "adminsecret"},
        )

        assert len(_pending_commands["dev-1"]) == 1
        assert _pending_commands["dev-1"]["buzzer"]["action"] == {"on": False}


def test_status_is_none_when_no_command_ever_issued(app, client):
    response = client.get(
        "/commands/dev-1/status?actuator=light",
        headers={"X-Admin-Key": "adminsecret"},
    )
    assert response.status_code == 200
    assert response.get_json()["status"] == "none"


def test_status_is_pending_after_create_before_delivery(app, client):
    client.post(
        "/commands",
        json={"device_id": "dev-1", "actuator": "buzzer", "action": {"on": True}},
        headers={"X-Admin-Key": "adminsecret"},
    )
    response = client.get(
        "/commands/dev-1/status?actuator=buzzer",
        headers={"X-Admin-Key": "adminsecret"},
    )
    assert response.get_json()["status"] == "pending"


def test_status_is_delivered_after_frames_pops_it(app, client):
    from app.commands import pop_deliverable_command
    import datetime

    client.post(
        "/commands",
        json={"device_id": "dev-1", "actuator": "buzzer", "action": {"on": True}},
        headers={"X-Admin-Key": "adminsecret"},
    )
    with app.app_context():
        pop_deliverable_command("dev-1", datetime.datetime.utcnow())

    response = client.get(
        "/commands/dev-1/status?actuator=buzzer",
        headers={"X-Admin-Key": "adminsecret"},
    )
    assert response.get_json()["status"] == "delivered"


def test_status_rejects_unknown_actuator(app, client):
    response = client.get(
        "/commands/dev-1/status?actuator=siren",
        headers={"X-Admin-Key": "adminsecret"},
    )
    assert response.status_code == 400


def test_status_requires_admin_key(app, client):
    response = client.get("/commands/dev-1/status?actuator=light")
    assert response.status_code == 401


ADMIN = {"X-Admin-Key": "adminsecret"}


def _post_door(client, action, device_id="dev-1", headers=ADMIN):
    return client.post(
        "/commands",
        json={"device_id": device_id, "actuator": "door", "action": action},
        headers=headers,
    )


@pytest.mark.parametrize("state", ["open", "close"])
def test_create_command_accepts_valid_door_command(app, client, state):
    response = _post_door(client, {"state": state})
    assert response.status_code == 201
    body = response.get_json()
    assert body["actuator"] == "door"
    assert body["action"] == {"state": state}
    assert _pending_commands["dev-1"]["door"]["action"] == {"state": state}


def test_door_command_without_admin_key_returns_401(app, client):
    response = _post_door(client, {"state": "open"}, headers={})
    assert response.status_code == 401
    assert _pending_commands == {}


@pytest.mark.parametrize(
    "action",
    [
        None,
        [],
        ["open"],
        "open",
        {},
        {"state": "ajar"},
        {"state": "OPEN"},
        {"state": 1},
        {"state": None},
    ],
)
def test_door_command_rejects_invalid_action(app, client, action):
    payload = {"device_id": "dev-1", "actuator": "door"}
    if action is not None:
        payload["action"] = action
    response = client.post("/commands", json=payload, headers=ADMIN)
    assert response.status_code == 400
    assert _pending_commands == {}


def test_door_command_drops_extra_keys(app, client):
    response = _post_door(client, {"state": "open", "force": True, "x": 1})
    assert response.status_code == 201
    assert response.get_json()["action"] == {"state": "open"}
    assert _pending_commands["dev-1"]["door"]["action"] == {"state": "open"}


def test_door_command_delivery_returns_it_and_arms_suppression(app, client):
    from app.commands import manual_suppression_active, pop_deliverable_command
    import datetime

    _post_door(client, {"state": "open"})
    with app.app_context():
        now = datetime.datetime.utcnow()
        delivered = pop_deliverable_command("dev-1", now)
        assert delivered["actuator"] == "door"
        assert delivered["action"] == {"state": "open"}
        assert manual_suppression_active("dev-1", "door", now, 30) is True


def test_door_status_pending_then_delivered(app, client):
    from app.commands import pop_deliverable_command
    import datetime

    _post_door(client, {"state": "close"})
    pending = client.get("/commands/dev-1/status?actuator=door", headers=ADMIN)
    assert pending.status_code == 200
    assert pending.get_json()["status"] == "pending"

    with app.app_context():
        pop_deliverable_command("dev-1", datetime.datetime.utcnow())
    delivered = client.get("/commands/dev-1/status?actuator=door", headers=ADMIN)
    assert delivered.get_json()["status"] == "delivered"


def test_status_unknown_actuator_error_lists_door(app, client):
    response = client.get("/commands/dev-1/status?actuator=siren", headers=ADMIN)
    assert response.status_code == 400
    assert "door" in response.get_json()["error"]


def test_second_door_command_replaces_first(app, client):
    from app.commands import pop_deliverable_command
    import datetime

    _post_door(client, {"state": "open"})
    _post_door(client, {"state": "close"})
    with app.app_context():
        now = datetime.datetime.utcnow()
        delivered = pop_deliverable_command("dev-1", now)
        assert delivered["action"] == {"state": "close"}
        assert pop_deliverable_command("dev-1", now) is None
