"""In-memory latest-frame cache — FEED-03: no history, one frame per device.

Module-level dict, same MVP pattern as app/debounce.py and app/commands.py:
resets on restart, single-prototype-device scope. Each store() call
overwrites the previous entry for that device_id; nothing is ever appended,
so no history can accumulate here even under sustained load.
"""

_latest_frames = {}


def store_latest_frame(device_id, jpeg_bytes, captured_at):
    """Overwrite the cached frame for device_id. captured_at is a
    datetime.datetime (UTC) — the moment the Edge API received this frame,
    used as a synchronous-capture proxy since the firmware has no synced
    RTC to stamp frames with its own clock (see FEED-04 decision in
    09-SUMMARY.md)."""
    _latest_frames[device_id] = {
        "jpeg_bytes": jpeg_bytes,
        "captured_at": captured_at,
    }


def get_latest_frame(device_id):
    """Returns {"jpeg_bytes": bytes, "captured_at": datetime} or None."""
    return _latest_frames.get(device_id)
