from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from peewee import SqliteDatabase

from app.commands import commands_bp
from app.config import Config
from app.device_settings import device_settings_bp
from app.faces import faces_bp
from app.frame_feed import frame_feed_bp
from app.frames import frames_bp
from app.heartbeat import heartbeat_cycle
from app.ingest import ingest_bp
from app.models import DeviceSetting, Reading, Rule, database_proxy
from app.occupancy_feed import occupancy_feed_bp
from app.relay import relay_cycle
from app.rules_admin import rules_bp
from app.rules_engine import seed_default_rules_if_empty


def create_app(config_object=Config, start_relay=True):
    app = Flask(__name__)
    app.config.from_object(config_object)

    database = SqliteDatabase(app.config["EDGE_DB_PATH"])
    database_proxy.initialize(database)
    database.connect(reuse_if_open=True)
    database.create_tables([Reading, DeviceSetting, Rule])

    # RULE-12: seed the migrated-equivalent rules once, only on a fresh
    # (empty) Rule table — never overwrites operator edits on restart.
    with app.app_context():
        seed_default_rules_if_empty(
            allowed_object_class=app.config.get("ALLOWED_OBJECT_CLASS", "backpack"),
            cooldown_seconds=app.config.get("DOOR_ACTION_COOLDOWN_SECONDS", 30),
        )

    app.register_blueprint(ingest_bp)
    app.register_blueprint(frames_bp)
    app.register_blueprint(faces_bp)
    app.register_blueprint(device_settings_bp)
    app.register_blueprint(commands_bp)
    app.register_blueprint(rules_bp)
    app.register_blueprint(frame_feed_bp)
    app.register_blueprint(occupancy_feed_bp)

    if start_relay:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            lambda: _run_relay_cycle(app),
            "interval",
            seconds=app.config["RELAY_INTERVAL_SECONDS"],
            id="relay_cycle",
        )
        scheduler.start()
        app.extensions["relay_scheduler"] = scheduler

    return app


def _run_relay_cycle(app):
    with app.app_context():
        # Independent try/excepts: a heartbeat failure must never block the
        # relay and vice versa.
        try:
            relay_cycle()
        except Exception:
            app.logger.exception("relay cycle failed")
        try:
            heartbeat_cycle()
        except Exception:
            app.logger.exception("heartbeat cycle failed")
