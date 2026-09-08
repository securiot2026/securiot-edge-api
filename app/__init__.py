from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from peewee import SqliteDatabase

from app.config import Config
from app.ingest import ingest_bp
from app.models import Reading, database_proxy
from app.relay import relay_cycle


def create_app(config_object=Config, start_relay=True):
    app = Flask(__name__)
    app.config.from_object(config_object)

    database = SqliteDatabase(app.config["EDGE_DB_PATH"])
    database_proxy.initialize(database)
    database.connect(reuse_if_open=True)
    database.create_tables([Reading])

    app.register_blueprint(ingest_bp)

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
        relay_cycle()
