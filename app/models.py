import datetime

from peewee import (
    BooleanField,
    CharField,
    DatabaseProxy,
    DateTimeField,
    IntegerField,
    Model,
)

database_proxy = DatabaseProxy()


class BaseModel(Model):
    class Meta:
        database = database_proxy


class DeviceSetting(BaseModel):
    device_id = CharField(unique=True)
    face_recognition_enabled = BooleanField(null=True)
    updated_at = DateTimeField(default=datetime.datetime.utcnow)


class Rule(BaseModel):
    # None (null) = applies to every device this Edge instance serves.
    device_id = CharField(null=True)
    name = CharField()
    # object_class | weapon | fire | occupancy | sensor_threshold
    condition_type = CharField()
    # Used by sensor_threshold only: '>' | '<' | '=='
    operator = CharField(null=True)
    # sensor_threshold only: which /ingest sensor_type this rule watches.
    sensor_type = CharField(null=True)
    # object_class: target class name. occupancy: daily limit (stringified
    # int). sensor_threshold: threshold value (stringified number).
    match_value = CharField(null=True)
    # JSON list of action dicts, e.g. [{"type": "door_lock"}] or
    # [{"type": "light", "color": "red"}] or [{"type": "alert", "severity": "high"}].
    actions = CharField()
    enabled = BooleanField(default=True)
    # Lower number = higher priority; wins conflicts on the same actuator.
    priority = IntegerField(default=100)
    cooldown_seconds = IntegerField(default=30)
    last_triggered_at = DateTimeField(null=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)


class Reading(BaseModel):
    reading_id = CharField(unique=True)
    device_id = CharField()
    zone_id = CharField()
    sensor_type = CharField()
    value = CharField()
    recorded_at = DateTimeField()
    synced = BooleanField(default=False)
    sync_attempts = IntegerField(default=0)
    next_attempt_at = DateTimeField(null=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)
