"""In-memory per-device consecutive-qualifying-frame streak tracker.

Turns raw per-frame detections into a confirmed detection event, per
PITFALLS.md Pitfall 7 (avoid false-positive-driven alerts from a single
stray detection).

State lives in a module-level dict and resets on process restart. This
is an accepted MVP limitation for a single-prototype device, not a
correctness bug for the academic scope, and is not to be "fixed" with a
database table in this plan.

record() only reports that the streak reached the threshold on this
call, exactly once per escalation (not on every subsequent qualifying
frame) - the caller is responsible for any cooldown before allowing a
new escalation for the same device.
"""

from app.config import Config

_streaks = {}


def record(device_id, qualifying, threshold=None):
    if threshold is None:
        threshold = Config.DETECTION_DEBOUNCE_COUNT

    if not qualifying:
        _streaks[device_id] = 0
        return {"escalate": False}

    count = _streaks.get(device_id, 0) + 1
    _streaks[device_id] = count

    return {"escalate": count == threshold}
