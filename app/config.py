import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    EDGE_DB_PATH = os.environ.get("EDGE_DB_PATH", "edge.db")
    DEVICE_SHARED_SECRET = os.environ.get("DEVICE_SHARED_SECRET", "")
    CLOUD_API_URL = os.environ.get("CLOUD_API_URL", "http://localhost:5000")
    CLOUD_DEVICE_API_KEY = os.environ.get("CLOUD_DEVICE_API_KEY", "")
    RELAY_INTERVAL_SECONDS = int(os.environ.get("RELAY_INTERVAL_SECONDS", "10"))

    DETECTION_BACKEND = os.environ.get("DETECTION_BACKEND", "mock")
    YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
    ALLOWED_OBJECT_CLASS = os.environ.get("ALLOWED_OBJECT_CLASS", "backpack")
    DETECTION_CONFIDENCE_THRESHOLD = float(
        os.environ.get("DETECTION_CONFIDENCE_THRESHOLD", "0.5")
    )
    DETECTION_DEBOUNCE_COUNT = int(os.environ.get("DETECTION_DEBOUNCE_COUNT", "2"))
    DOOR_ACTION_COOLDOWN_SECONDS = int(
        os.environ.get("DOOR_ACTION_COOLDOWN_SECONDS", "30")
    )
