from app.detection import DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH, get_detector
from app.pan_tilt import compute_pan_tilt

MOCK_CONFIG = {
    "DETECTION_BACKEND": "mock",
    "ALLOWED_OBJECT_CLASS": "backpack",
}

YOLO_CONFIG = {
    "DETECTION_BACKEND": "yolo",
    "YOLO_MODEL_PATH": "yolov8n.pt",
    "ALLOWED_OBJECT_CLASS": "backpack",
    "DETECTION_CONFIDENCE_THRESHOLD": 0.5,
}


def test_mock_detector_finds_person_left_and_points_camera_left():
    detector = get_detector(MOCK_CONFIG)

    detections = detector.detect("tests/fixtures/person_left.jpg")

    assert len(detections) == 1
    detection = detections[0]
    assert detection["class_name"] == "person"

    delta = compute_pan_tilt(
        detection["bbox"], DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT
    )
    assert delta["pan_delta"] < 0


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    """Stand-in for ultralytics.YOLO: no real model load, no network access."""

    names = {0: "person", 1: "bicycle", 24: "backpack"}

    def __init__(self, model_path):
        self.model_path = model_path

    def __call__(self, image_path):
        return [
            FakeResult(
                FakeBoxes(
                    xyxy=[
                        (10.0, 10.0, 100.0, 100.0),
                        (200.0, 200.0, 300.0, 300.0),
                        (270.0, 190.0, 370.0, 290.0),
                        (5.0, 5.0, 50.0, 50.0),
                    ],
                    conf=[0.9, 0.8, 0.95, 0.3],
                    cls=[0, 1, 24, 0],
                )
            )
        ]


def test_yolo_detector_parses_results_filters_by_class_and_confidence(monkeypatch):
    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)

    detector = get_detector(YOLO_CONFIG)
    detections = detector.detect("tests/fixtures/whatever.jpg")

    assert len(detections) == 2
    class_names = {d["class_name"] for d in detections}
    assert class_names == {"person", "backpack"}
    for detection in detections:
        assert detection["confidence"] >= 0.5


def test_get_detector_dispatches_mock_vs_yolo(monkeypatch):
    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)

    mock_detector = get_detector(MOCK_CONFIG)
    yolo_detector = get_detector(YOLO_CONFIG)

    assert mock_detector.__class__.__name__ == "MockDetector"
    assert yolo_detector.__class__.__name__ == "YoloDetector"
