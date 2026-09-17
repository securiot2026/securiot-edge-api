"""Operator-facing CRUD for Rule rows (RULE-01..04's backend contract —
Web App UI itself is Phase 10; this phase only needs the REST surface to
exist and work, admin-key gated like the other operator endpoints added in
Phase 06.1/7 (POST /faces, /devices/<id>/face-recognition, POST /commands).
"""

import datetime
import json

from flask import Blueprint, current_app, jsonify, request

from app.auth import require_admin_key
from app.models import Rule
from app.rules_engine import dispatch

rules_bp = Blueprint("rules", __name__)

CONDITION_TYPES = ("object_class", "weapon", "fire", "occupancy", "sensor_threshold")
OPERATORS = (">", "<", "==")


def _rule_to_dict(rule):
    return {
        "id": rule.id,
        "device_id": rule.device_id,
        "name": rule.name,
        "condition_type": rule.condition_type,
        "operator": rule.operator,
        "sensor_type": rule.sensor_type,
        "match_value": rule.match_value,
        "actions": json.loads(rule.actions),
        "enabled": rule.enabled,
        "priority": rule.priority,
        "cooldown_seconds": rule.cooldown_seconds,
        "last_triggered_at": (
            rule.last_triggered_at.isoformat() + "Z"
            if rule.last_triggered_at
            else None
        ),
    }


def _validate_payload(payload, *, partial):
    errors = []

    def require(field, predicate, message):
        if field in payload or not partial:
            if not predicate(payload.get(field)):
                errors.append(message)

    if not partial or "name" in payload:
        if not isinstance(payload.get("name"), str) or not payload.get("name", "").strip():
            errors.append("name (non-empty string) is required")

    if not partial or "condition_type" in payload:
        if payload.get("condition_type") not in CONDITION_TYPES:
            errors.append(f"condition_type must be one of {list(CONDITION_TYPES)}")

    if not partial or "actions" in payload:
        actions = payload.get("actions")
        if (
            not isinstance(actions, list)
            or not actions
            or not all(isinstance(a, dict) and a.get("type") for a in actions)
        ):
            errors.append("actions must be a non-empty list of {'type': ...} objects")

    condition_type = payload.get("condition_type")
    if condition_type == "sensor_threshold":
        if payload.get("operator") not in OPERATORS:
            errors.append(f"operator must be one of {list(OPERATORS)} for sensor_threshold rules")
        if not payload.get("sensor_type"):
            errors.append("sensor_type is required for sensor_threshold rules")

    return errors


@rules_bp.route("/rules", methods=["GET"])
def list_rules():
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    device_id = request.args.get("device_id")
    query = Rule.select()
    rules = [
        r for r in query if device_id is None or r.device_id in (None, device_id)
    ]
    return jsonify([_rule_to_dict(r) for r in rules]), 200


@rules_bp.route("/rules", methods=["POST"])
def create_rule():
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    payload = request.get_json(silent=True) or {}
    errors = _validate_payload(payload, partial=False)
    if errors:
        return jsonify({"error": "invalid rule", "details": errors}), 400

    rule = Rule.create(
        device_id=payload.get("device_id"),
        name=payload["name"].strip(),
        condition_type=payload["condition_type"],
        operator=payload.get("operator"),
        sensor_type=payload.get("sensor_type"),
        match_value=(
            str(payload["match_value"]) if payload.get("match_value") is not None else None
        ),
        actions=json.dumps(payload["actions"]),
        enabled=bool(payload.get("enabled", True)),
        priority=int(payload.get("priority", 100)),
        cooldown_seconds=int(payload.get("cooldown_seconds", 30)),
    )
    return jsonify(_rule_to_dict(rule)), 201


@rules_bp.route("/rules/<int:rule_id>", methods=["PATCH"])
def update_rule(rule_id):
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    rule = Rule.get_or_none(Rule.id == rule_id)
    if rule is None:
        return jsonify({"error": "rule not found"}), 404

    payload = request.get_json(silent=True) or {}
    errors = _validate_payload(payload, partial=True)
    if errors:
        return jsonify({"error": "invalid rule", "details": errors}), 400

    for field in (
        "device_id",
        "name",
        "condition_type",
        "operator",
        "sensor_type",
        "priority",
        "cooldown_seconds",
        "enabled",
    ):
        if field in payload:
            setattr(rule, field, payload[field])
    if "match_value" in payload:
        rule.match_value = (
            str(payload["match_value"]) if payload["match_value"] is not None else None
        )
    if "actions" in payload:
        rule.actions = json.dumps(payload["actions"])

    rule.save()
    return jsonify(_rule_to_dict(rule)), 200


@rules_bp.route("/rules/<int:rule_id>", methods=["DELETE"])
def delete_rule(rule_id):
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    rule = Rule.get_or_none(Rule.id == rule_id)
    if rule is None:
        return jsonify({"error": "rule not found"}), 404

    rule.delete_instance()
    return "", 204


@rules_bp.route("/rules/<int:rule_id>/dry-run", methods=["POST"])
def dry_run_rule(rule_id):
    """RULE-11: let the operator test a rule's configured action without
    waiting for a real detection. Reuses app/rules_engine.py's dispatch() —
    the same single action-dispatch path /frames and /ingest already go
    through — so a dry-run genuinely exercises the configured action
    (door_lock / light / buzzer / alert), not a simulated summary of it.
    Cooldown is intentionally bypassed (an explicit manual test), but the
    rule's last_triggered_at IS updated, same as a real trigger."""
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    rule = Rule.get_or_none(Rule.id == rule_id)
    if rule is None:
        return jsonify({"error": "rule not found"}), 404
    if not rule.enabled:
        return jsonify({"error": "cannot dry-run a disabled rule"}), 400

    now = datetime.datetime.utcnow()
    result = dispatch(
        rule.device_id,
        None,
        [rule],
        now,
        manual_command=None,
        suppress_door=False,
    )
    current_app.logger.info("Dry-run fired rule %s (%s)", rule.id, rule.name)
    return jsonify({"rule": _rule_to_dict(rule), "result": result}), 200
