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


class YoloDetector:
    """Real YOLOv8n backend (Ultralytics, CPU inference).

    The loaded model is not filtered by `classes=[...]` at call time so the
    same model instance can be reused across configs on a live device;
    filtering happens here on the parsed result instead.
    """

    def __init__(self, model, allowed_object_class, confidence_threshold):
        self.model = model
        self.allowed_object_class = allowed_object_class
        self.confidence_threshold = confidence_threshold

    def detect(self, image_path):
        results = self.model(image_path)
        detections = []
        for result in results:
            for bbox, conf, cls in zip(
                result.boxes.xyxy, result.boxes.conf, result.boxes.cls
            ):
                confidence = float(conf)
                if confidence < self.confidence_threshold:
                    continue

                class_id = int(cls)
                class_name = self.model.names[class_id]
                if class_name not in ("person", self.allowed_object_class):
                    continue

                x1, y1, x2, y2 = (float(v) for v in bbox)
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                    }
                )
        return detections


_yolo_model_cache = {}


def get_detector(config):
    if config["DETECTION_BACKEND"] == "mock":
        return MockDetector(allowed_object_class=config.get("ALLOWED_OBJECT_CLASS", "backpack"))

    if config["DETECTION_BACKEND"] == "yolo":
        from ultralytics import YOLO

        model_path = config.get("YOLO_MODEL_PATH", "yolov8n.pt")
        model = _yolo_model_cache.get(model_path)
        if model is None:
            model = YOLO(model_path)
            _yolo_model_cache[model_path] = model

        return YoloDetector(
            model=model,
            allowed_object_class=config.get("ALLOWED_OBJECT_CLASS", "backpack"),
            confidence_threshold=config.get("DETECTION_CONFIDENCE_THRESHOLD", 0.5),
        )

    raise NotImplementedError(f"unknown DETECTION_BACKEND: {config['DETECTION_BACKEND']!r}")
