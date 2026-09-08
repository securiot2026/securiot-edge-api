"""Detection backends for the Edge API.

A `Detection` is a plain dict: {"class_name": str, "confidence": float,
"bbox": (x1, y1, x2, y2)} with bbox in pixel coordinates (x1 < x2, y1 < y2).

Default frame size assumed by MockDetector's filename-convention bounding
boxes and by scripts/generate_test_images.py.
"""

import os

DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480


class MockDetector:
    """Filename-convention detector: inspects the path, not image bytes.

    Used by automated tests so they never require a real camera, network
    access, or downloaded YOLO model weights.
    """

    def __init__(self, allowed_object_class="backpack"):
        self.allowed_object_class = allowed_object_class

    def detect(self, image_path):
        filename = os.path.basename(image_path).lower()

        if "person" in filename:
            if "left" in filename:
                bbox = (40, 120, 240, 460)
            elif "right" in filename:
                bbox = (400, 120, 600, 460)
            else:
                bbox = (270, 120, 470, 460)
            return [{"class_name": "person", "confidence": 0.9, "bbox": bbox}]

        if self.allowed_object_class in filename:
            bbox = (270, 190, 370, 290)
            return [
                {
                    "class_name": self.allowed_object_class,
                    "confidence": 0.9,
                    "bbox": bbox,
                }
            ]

        return []


def get_detector(config):
    if config["DETECTION_BACKEND"] == "mock":
        return MockDetector(allowed_object_class=config.get("ALLOWED_OBJECT_CLASS", "backpack"))
    raise NotImplementedError("yolo backend implemented in Task 2")
