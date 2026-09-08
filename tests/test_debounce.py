from app.debounce import record


def test_escalates_only_on_nth_consecutive_qualifying_frame():
    first = record("device-a", qualifying=True, threshold=2)
    second = record("device-a", qualifying=True, threshold=2)

    assert first["escalate"] is False
    assert second["escalate"] is True


def test_does_not_re_escalate_on_further_qualifying_frames():
    record("device-b", qualifying=True, threshold=2)
    record("device-b", qualifying=True, threshold=2)
    third = record("device-b", qualifying=True, threshold=2)

    assert third["escalate"] is False


def test_non_qualifying_frame_resets_the_streak():
    record("device-c", qualifying=True, threshold=2)
    reset = record("device-c", qualifying=False, threshold=2)
    next_first = record("device-c", qualifying=True, threshold=2)

    assert reset["escalate"] is False
    assert next_first["escalate"] is False


def test_streaks_are_scoped_per_device():
    record("device-d1", qualifying=True, threshold=2)
    record("device-d1", qualifying=True, threshold=2)
    device_2_first = record("device-d2", qualifying=True, threshold=2)

    assert device_2_first["escalate"] is False


def test_uses_configured_debounce_count_by_default():
    from app.config import Config

    original = Config.DETECTION_DEBOUNCE_COUNT
    Config.DETECTION_DEBOUNCE_COUNT = 1
    try:
        result = record("device-e", qualifying=True)
    finally:
        Config.DETECTION_DEBOUNCE_COUNT = original

    assert result["escalate"] is True
