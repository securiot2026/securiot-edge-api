from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from face_app.config import Settings
from face_app.errors import FaceAppError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local YOLO and ArcFace face-recognition prototype"
    )
    parser.add_argument(
        "--model", type=Path, help="Path to face-specific Ultralytics YOLO .pt weights"
    )
    parser.add_argument("--camera", type=int, help="OpenCV camera index")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser(
        "register", help="Capture photos and register one identity"
    )
    register_parser.add_argument(
        "--name", help="Display name; prompted securely if omitted"
    )
    recognize_parser = subparsers.add_parser(
        "recognize", help="Recognize registered faces from the camera"
    )
    recognize_parser.add_argument(
        "--threshold", type=float, help="Cosine similarity threshold (-1 to 1)"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        settings = Settings.from_env()
        if args.model is not None:
            settings = replace(settings, model_path=args.model)
        if args.camera is not None:
            if args.camera < 0:
                raise FaceAppError("--camera must be zero or greater.")
            settings = replace(settings, camera_index=args.camera)
        if args.command == "register":
            from face_app.registration import register_user

            name = args.name if args.name is not None else input("Name to register: ")
            result = register_user(name, settings)
            if not result.completed:
                print(
                    f"Registration cancelled. {result.accepted_photos} partial "
                    "photos remain in "
                    f"{result.capture_dir}; the identity database was not changed."
                )
                return 1
            print(
                f"Registered {name!r} with {result.embeddings} embeddings. "
                f"Photos: {result.capture_dir}"
            )
            return 0
        if args.command == "recognize":
            if args.threshold is not None and not -1.0 <= args.threshold <= 1.0:
                raise FaceAppError("--threshold must be between -1 and 1.")
            if args.threshold is not None:
                settings = replace(
                    settings, similarity_threshold=args.threshold
                )
            from face_app.recognition import recognize

            recognize(settings)
            return 0
        raise FaceAppError(f"Unsupported command: {args.command}")
    except (FaceAppError, EOFError, KeyboardInterrupt) as exc:
        message = (
            "Interrupted by user."
            if isinstance(exc, KeyboardInterrupt)
            else str(exc)
        )
        print(f"Error: {message}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(
            f"Error: missing Python dependency ({exc}). Install with: "
            "python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
