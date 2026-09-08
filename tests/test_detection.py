from app.detection import DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH, get_detector
from app.pan_tilt import compute_pan_tilt

MOCK_CONFIG = {
    "DETECTION_BACKEND": "mock",
    "ALLOWED_OBJECT_CLASS": "backpack",
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
