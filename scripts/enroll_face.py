"""Enroll one identity from local images for ArcFace recognition.

Model paths, devices, thresholds, and the identity directory come from the
same environment variables used by the Edge API. No biometric data is sent
over HTTP.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.config import Config  # noqa: E402
from app.vision.enrollment import enroll_identity  # noqa: E402
from app.vision.errors import VisionError  # noqa: E402
from app.vision.settings import VisionSettings  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create or replace an ArcFace identity from face images."
    )
    parser.add_argument("--name", required=True, help="Display name to enroll.")
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Image paths. Each image must contain exactly one detected face.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settings = VisionSettings.from_mapping(vars(Config))
    try:
        result = enroll_identity(args.name, args.images, settings)
    except VisionError as exc:
        print(f"Enrollment failed: {exc}", file=sys.stderr)
        return 1

    for warning in result.skipped_images:
        print(f"Skipped: {warning}", file=sys.stderr)
    print(
        f"Enrolled {args.name!r} from {result.accepted_images} images at "
        f"{result.identity_path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
