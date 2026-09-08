"""Pan/tilt delta calculation from a detection bounding box.

Sign convention: negative pan_delta means the target is left of frame
center (camera should pan left to re-center it); negative tilt_delta
means the target is above frame center (camera should tilt up).
Both values are normalized to [-1.0, 1.0] and clamped to that range.
This is a simple proportional calculation, not a PID controller.
"""


def compute_pan_tilt(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    pan_delta = (bbox_center_x - frame_center_x) / (frame_width / 2)
    tilt_delta = (bbox_center_y - frame_center_y) / (frame_height / 2)

    pan_delta = max(-1.0, min(1.0, pan_delta))
    tilt_delta = max(-1.0, min(1.0, tilt_delta))

    return {"pan_delta": pan_delta, "tilt_delta": tilt_delta}
