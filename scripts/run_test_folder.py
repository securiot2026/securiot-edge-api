"""End-to-end test-image-folder harness for the /frames pipeline.

Boots a test Edge API app (in-memory SQLite, no background relay), then
POSTs every image in tests/fixtures/sample_frames/ (in filename order) to
/frames as a single simulated device, printing a one-line summary per image.

This proves the full frame -> detection -> debounce -> telemetry -> relay
path end to end without a camera or a real ESP32-CAM device. Run via:

    python scripts/run_test_folder.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import create_app  # noqa: E402
from app.models import Reading  # noqa: E402

FIXTURES_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "sample_frames")
DEVICE_ID = "test-folder-device"
ZONE_ID = "test-folder-zone"


class HarnessConfig:
    EDGE_DB_PATH = ":memory:"
    DEVICE_SHARED_SECRET = "harness-secret"
    CLOUD_API_URL = "http://cloud.example"
    CLOUD_DEVICE_API_KEY = "harness-cloud-key"
    RELAY_INTERVAL_SECONDS = 3600

    DETECTION_BACKEND = "mock"
    YOLO_MODEL_PATH = "yolov8n.pt"
    ALLOWED_OBJECT_CLASS = "backpack"
    DETECTION_CONFIDENCE_THRESHOLD = 0.5
    DETECTION_DEBOUNCE_COUNT = 2
    DOOR_ACTION_COOLDOWN_SECONDS = 30


def run():
    app = create_app(config_object=HarnessConfig, start_relay=False)
    client = app.test_client()

    with app.app_context():
        image_names = sorted(os.listdir(FIXTURES_DIR))
        for image_name in image_names:
            image_path = os.path.join(FIXTURES_DIR, image_name)
            with open(image_path, "rb") as frame_fh:
                response = client.post(
                    "/frames",
                    data={
                        "device_id": DEVICE_ID,
                        "zone_id": ZONE_ID,
                        "frame": (frame_fh, image_name),
                    },
                    headers={"X-Device-Key": HarnessConfig.DEVICE_SHARED_SECRET},
                    content_type="multipart/form-data",
                )

            body = response.get_json()
            reading_count = Reading.select().count()
            print(
                f"{image_name}: pan_delta={body['pan_delta']} "
                f"door_action={body['door_action']} alert={body['alert']} "
                f"(readings so far: {reading_count})"
            )


if __name__ == "__main__":
    run()
