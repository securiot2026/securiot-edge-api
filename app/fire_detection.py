"""Heuristic fire condition for RULE-08: color/brightness thresholding over
the captured frame, no trained model and no new sensor.

Not a general-purpose fire detector — a deliberately simple, explainable
heuristic scoped to the academic MVP: a frame "looks like fire" when a large
enough share of its pixels are bright and red-dominant (red clearly above
both green and blue).
"""

import numpy as np
from PIL import Image

DEFAULT_MIN_FIRE_PIXEL_RATIO = 0.12
DEFAULT_MIN_BRIGHTNESS = 140.0


def detect_fire(
    image_path,
    min_fire_pixel_ratio=DEFAULT_MIN_FIRE_PIXEL_RATIO,
    min_brightness=DEFAULT_MIN_BRIGHTNESS,
):
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)

    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    brightness = (red + green + blue) / 3.0

    fire_mask = (
        (red > 150)
        & (red > green)
        & (green > blue)
        & ((red - blue) > 40)
        & (brightness > min_brightness)
    )

    ratio = float(np.count_nonzero(fire_mask)) / fire_mask.size
    return ratio >= min_fire_pixel_ratio
