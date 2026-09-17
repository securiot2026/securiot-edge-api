"""The condition->action rules engine (RULE-05/06/07/08/09/12).

Replaces the hardcoded door_contact_open rule that used to live directly in
app/frames.py. Rules are stored in the Rule Peewee model (app/models.py) and
evaluated fresh on every call — no caching, since the SQLite table is local
to this Edge instance and cheap to query at MVP scale.

Conflict resolution (most-restrictive-wins): callers pass already-matched
rules, sorted ascending by `priority` (lower number = higher priority), into
resolve_actuator_actions(). For a given actuator name, the first rule in
that order wins — i.e. the lowest-priority-number rule targeting that
actuator. This is the locked answer to the previously-open "rule conflict
resolution" design question (see STATE.md Blockers/Concerns, 2026-09-08).
"""

import json
import uuid

from flask import current_app

from app.fire_detection import detect_fire
from app.models import Rule
from app.reading_buffer import buffer_reading

WEAPON_CLASS_NAME = "knife"


def _condition_matches(rule, *, detections, image_path, occupancy_count):
    condition_type = rule.condition_type

    if condition_type == "object_class":
        target = rule.match_value
        return any(d["class_name"] == target for d in detections)

    if condition_type == "weapon":
        return any(d["class_name"] == WEAPON_CLASS_NAME for d in detections)

    if condition_type == "fire":
        if image_path is None:
            return False
        return detect_fire(
            image_path,
            min_fire_pixel_ratio=current_app.config.get("FIRE_MIN_PIXEL_RATIO", 0.12),
            min_brightness=current_app.config.get("FIRE_MIN_BRIGHTNESS", 140.0),
        )

    if condition_type == "occupancy":
        if occupancy_count is None or rule.match_value is None:
            return False
        try:
            limit = int(rule.match_value)
        except ValueError:
            return False
        return occupancy_count >= limit

    if condition_type == "sensor_threshold":
        # Evaluated separately via evaluate_sensor_threshold() below, which
        # has the sensor_type/value context this generic detections-based
        # matcher does not receive.
        return False

    return False


def _cooldown_active(rule, now):
    if rule.last_triggered_at is None:
        return False
    return (now - rule.last_triggered_at).total_seconds() < rule.cooldown_seconds


def _candidate_rules(device_id, condition_types):
    query = Rule.select().where(
        (Rule.enabled == True) & (Rule.condition_type.in_(condition_types))  # noqa: E712
    )
    for rule in query:
        if rule.device_id is not None and rule.device_id != device_id:
            continue
        yield rule


def evaluate(device_id, now, condition_types, *, detections=None, image_path=None, occupancy_count=None):
    """Return matched, not-in-cooldown rules for the given condition types,
    sorted ascending by priority (lowest number first)."""
    detections = detections or []
    matched = [
        rule
        for rule in _candidate_rules(device_id, condition_types)
        if not _cooldown_active(rule, now)
        and _condition_matches(
            rule,
            detections=detections,
            image_path=image_path,
            occupancy_count=occupancy_count,
        )
    ]
    matched.sort(key=lambda r: r.priority)
    return matched


def evaluate_sensor_threshold(device_id, now, sensor_type, sensor_value):
    """RULE-05: threshold/equality conditions against an arbitrary ingested
    sensor reading (not camera-derived), evaluated from app/ingest.py."""
    try:
        observed = float(sensor_value)
    except (TypeError, ValueError):
        return []

    matched = []
    for rule in _candidate_rules(device_id, ("sensor_threshold",)):
        if rule.sensor_type != sensor_type or rule.match_value is None:
            continue
        if _cooldown_active(rule, now):
            continue
        try:
            threshold = float(rule.match_value)
        except ValueError:
            continue

        op = rule.operator
        matches = (
            (op == ">" and observed > threshold)
            or (op == "<" and observed < threshold)
            or (op == "==" and observed == threshold)
        )
        if matches:
            matched.append(rule)

    matched.sort(key=lambda r: r.priority)
    return matched


def mark_triggered(rules, now):
    for rule in rules:
        rule.last_triggered_at = now
        rule.save()


def resolve_actuator_actions(matched_rules):
    """most-restrictive-wins: matched_rules must already be priority-sorted
    ascending. Returns {actuator_type: {"rule": Rule, "action": dict}} —
    first rule seen per actuator type wins."""
    resolved = {}
    for rule in matched_rules:
        for action in json.loads(rule.actions):
            actuator = action.get("type")
            if actuator and actuator not in resolved:
                resolved[actuator] = {"rule": rule, "action": action}
    return resolved


def dispatch(
    device_id, zone_id, matched_rules, now, *, manual_command=None, suppress_door=False
):
    """Single, shared action-dispatch path (RULE-12: exactly one place that
    turns a matched rule into an effect — frames.py and ingest.py both call
    this instead of each having their own copy of the same logic).

    Returns {"door_action": "lock" | None, "rule_command": dict | None}.
    `rule_command` mirrors app/commands.py's manual_command shape
    ({"actuator", "action"}) for light/buzzer actions, but is delivered
    synchronously in this same response rather than queued — a manual
    operator command for the same actuator this cycle always wins (CTRL-06
    precedent: manual suppresses automatic).
    """
    resolved = resolve_actuator_actions(matched_rules)
    result = {"door_action": None, "rule_command": None}
    fired_rules = []

    for actuator, entry in resolved.items():
        rule = entry["rule"]
        action = entry["action"]

        if actuator == "door_lock":
            if suppress_door:
                continue
            fired_rules.append(rule)
            buffer_reading(
                reading_id=str(uuid.uuid4()),
                device_id=device_id,
                zone_id=zone_id,
                sensor_type="door_contact",
                value="open",
                recorded_at=now,
            )
            result["door_action"] = "lock"

        elif actuator in ("light", "buzzer"):
            manual_conflict = (
                manual_command is not None
                and manual_command.get("actuator") == actuator
            )
            if not manual_conflict:
                fired_rules.append(rule)
                result["rule_command"] = {
                    "actuator": actuator,
                    "action": {k: v for k, v in action.items() if k != "type"},
                    "rule": rule.name,
                }

        elif actuator == "alert":
            fired_rules.append(rule)
            buffer_reading(
                reading_id=str(uuid.uuid4()),
                device_id=device_id,
                zone_id=zone_id,
                sensor_type="rule_alert",
                value=json.dumps(
                    {
                        "rule_name": rule.name,
                        "severity": action.get("severity", "medium"),
                        "message": action.get(
                            "message", f"Rule '{rule.name}' triggered"
                        ),
                    }
                ),
                recorded_at=now,
            )

    mark_triggered(fired_rules, now)
    return result


def seed_default_rules_if_empty(allowed_object_class, cooldown_seconds):
    """RULE-12: migrate the old hardcoded door_contact_open behavior into
    two equivalent rules in the new engine (person, and the configured
    ALLOWED_OBJECT_CLASS — the same two classes the old hardcoded check
    accepted). Runs once, only when the Rule table is empty, so an
    operator's later edits/deletes are never overwritten on restart."""
    if Rule.select().count() > 0:
        return

    door_lock_action = json.dumps([{"type": "door_lock"}])
    Rule.create(
        device_id=None,
        name="Legacy door lock — person (migrated)",
        condition_type="object_class",
        match_value="person",
        actions=door_lock_action,
        enabled=True,
        priority=100,
        cooldown_seconds=cooldown_seconds,
    )
    if allowed_object_class and allowed_object_class != "person":
        Rule.create(
            device_id=None,
            name=f"Legacy door lock — {allowed_object_class} (migrated)",
            condition_type="object_class",
            match_value=allowed_object_class,
            actions=door_lock_action,
            enabled=True,
            priority=100,
            cooldown_seconds=cooldown_seconds,
        )
