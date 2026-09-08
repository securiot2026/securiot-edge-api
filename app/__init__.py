from flask import Flask
from peewee import SqliteDatabase

from app.config import Config
from app.ingest import ingest_bp
from app.models import Reading, database_proxy


def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)

    database = SqliteDatabase(app.config["EDGE_DB_PATH"])
    database_proxy.initialize(database)
    database.connect(reuse_if_open=True)
    database.create_tables([Reading])

    app.register_blueprint(ingest_bp)

    return app
