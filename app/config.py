import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name, default=False):
    raw = os.environ.get(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


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
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", str(2 * 1024 * 1024)))

    FACE_RECOGNITION_ENABLED = _env_bool("FACE_RECOGNITION_ENABLED", False)
    FACE_YOLO_MODEL_PATH = os.environ.get(
        "FACE_YOLO_MODEL_PATH", "models/yolov8n-face.pt"
    )
    FACE_YOLO_CONFIDENCE_THRESHOLD = float(
        os.environ.get("FACE_YOLO_CONFIDENCE_THRESHOLD", "0.5")
    )
    FACE_YOLO_IOU_THRESHOLD = float(
        os.environ.get("FACE_YOLO_IOU_THRESHOLD", "0.45")
    )
    FACE_YOLO_INPUT_SIZE = int(os.environ.get("FACE_YOLO_INPUT_SIZE", "640"))
    FACE_YOLO_MAX_DETECTIONS = int(
        os.environ.get("FACE_YOLO_MAX_DETECTIONS", "10")
    )
    FACE_YOLO_CLASS_ID = int(os.environ.get("FACE_YOLO_CLASS_ID", "0"))
    FACE_YOLO_DEVICE = os.environ.get("FACE_YOLO_DEVICE", "cpu")

    ARCFACE_MODEL_PATH = os.environ.get(
        "ARCFACE_MODEL_PATH", "models/w600k_r50.onnx"
    )
    ARCFACE_DEVICE = os.environ.get("ARCFACE_DEVICE", "cpu")
    FACE_IDENTITY_DIR = os.environ.get("FACE_IDENTITY_DIR", "data/faces")
    FACE_SIMILARITY_THRESHOLD = float(
        os.environ.get("FACE_SIMILARITY_THRESHOLD", "0.45")
    )
    FACE_CROP_MARGIN = float(os.environ.get("FACE_CROP_MARGIN", "0.2"))
    FACE_MINIMUM_ENROLLMENT_IMAGES = int(
        os.environ.get("FACE_MINIMUM_ENROLLMENT_IMAGES", "3")
    )
