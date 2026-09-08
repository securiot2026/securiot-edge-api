from app.models import Reading


def buffer_reading(reading_id, device_id, zone_id, sensor_type, value, recorded_at):
    Reading.insert(
        reading_id=reading_id,
        device_id=device_id,
        zone_id=zone_id,
        sensor_type=sensor_type,
        value=value,
        recorded_at=recorded_at,
        synced=False,
    ).on_conflict_ignore().execute()
