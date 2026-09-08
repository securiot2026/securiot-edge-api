"""Generate synthetic placeholder images used to drive MockDetector's
filename convention and the end-to-end harness built in 04-02.

These are simple solid-background images with a labeled rectangle drawn
on them; they are NOT expected to be detected correctly by the *real*
YOLO backend, since synthetic shapes do not resemble real-world training
data. Real-model accuracy validation is Juan's manual step, run via
scripts/verify_real_yolo.py against a real photo.
"""

import os
import sys

from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.detection import DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "sample_frames")

BACKGROUND_COLOR = (30, 30, 30)
RECTANGLE_COLOR = (200, 200, 200)
ALLOWED_OBJECT_CLASS = "backpack"


def _blank_image():
    return Image.new("RGB", (DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT), BACKGROUND_COLOR)


def _draw_labeled_rectangle(image, bbox, label):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=RECTANGLE_COLOR, width=4)
    draw.text((bbox[0], max(bbox[1] - 15, 0)), label, fill=RECTANGLE_COLOR)


def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    person_left = _blank_image()
    _draw_labeled_rectangle(person_left, (40, 120, 240, 460), "person")
    person_left.save(os.path.join(OUTPUT_DIR, "person_left.jpg"))

    person_center = _blank_image()
    _draw_labeled_rectangle(person_center, (270, 120, 470, 460), "person")
    person_center.save(os.path.join(OUTPUT_DIR, "person_center.jpg"))

    object_center = _blank_image()
    _draw_labeled_rectangle(
        object_center, (270, 190, 370, 290), ALLOWED_OBJECT_CLASS
    )
    object_center.save(os.path.join(OUTPUT_DIR, "object_center.jpg"))

    empty = _blank_image()
    empty.save(os.path.join(OUTPUT_DIR, "empty.jpg"))


if __name__ == "__main__":
    generate()
    print(f"Generated fixture images in {OUTPUT_DIR}")
