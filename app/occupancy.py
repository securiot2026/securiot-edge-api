"""Per-device daily occupancy counter for RULE-09.

The detection pipeline only reports presence, not direction (no entry/exit
vector), so "occupancy" here is a pragmatic MVP scoping: one increment per
confirmed detection episode (the same debounce-escalation moment that used
to fire the hardcoded door lock), reset at UTC midnight. This is a locked,
documented simplification, not an attempt at true in/out tracking.

State lives in a module-level dict and resets on process restart, matching
the same accepted MVP limitation already used by app/debounce.py and
app/commands.py.
"""

_counts = {}  # device_id -> (date, count)


def record_entry_and_get_count(device_id, now):
    today = now.date()
    date, count = _counts.get(device_id, (today, 0))
    if date != today:
        count = 0
    count += 1
    _counts[device_id] = (today, count)
    return count


def get_count(device_id, now):
    today = now.date()
    date, count = _counts.get(device_id, (today, 0))
    return count if date == today else 0
