"""Manual smoke test for the real YOLOv8n backend.

Not part of the automated pytest suite: this script needs network access
to download yolov8n.pt on first run (Ultralytics fetches it from GitHub
release assets), so it cannot run in an offline CI/sandbox environment.
Run it manually against a real photo containing a person, e.g.:

    python scripts/verify_real_yolo.py path/to/photo.jpg

Requires `ultralytics` installed (see requirements.txt).
"""

import sys

from app.config import Config
from app.detection import get_detector


def main():
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <path-to-image>")
        sys.exit(1)

    image_path = sys.argv[1]

    config = {
        "DETECTION_BACKEND": "yolo",
        "YOLO_MODEL_PATH": Config.YOLO_MODEL_PATH,
        "ALLOWED_OBJECT_CLASS": Config.ALLOWED_OBJECT_CLASS,
        "DETECTION_CONFIDENCE_THRESHOLD": Config.DETECTION_CONFIDENCE_THRESHOLD,
    }

    detector = get_detector(config)
    detections = detector.detect(image_path)

    if not detections:
        print("No detections above the confidence threshold.")
        return

    for detection in detections:
        print(
            f"{detection['class_name']}: confidence={detection['confidence']:.2f} "
            f"bbox={detection['bbox']}"
        )


if __name__ == "__main__":
    main()
