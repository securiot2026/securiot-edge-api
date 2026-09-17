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
