import os
import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from app.auth import require_admin_key
from app.vision.enrollment import enroll_identity, validate_capture_quality
from app.vision.errors import VisionError
from app.vision.settings import VisionSettings

faces_bp = Blueprint("faces", __name__)


def _unique_path(directory, filename):
    candidate = os.path.join(directory, filename)
    if not os.path.exists(candidate):
        return candidate
    stem, ext = os.path.splitext(filename)
    counter = 1
    while True:
        candidate = os.path.join(directory, f"{stem}-{counter}{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


@faces_bp.route("/faces", methods=["POST"])
def enroll_face():
    unauthorized = require_admin_key()
    if unauthorized is not None:
        return unauthorized

    request.max_content_length = current_app.config["FACE_ENROLLMENT_MAX_CONTENT_LENGTH"]

    name = request.form.get("name", "")
    images = request.files.getlist("images")
    if not images:
        return jsonify({"error": "at least one image is required"}), 400

    tmp_dir = tempfile.mkdtemp()
    temp_paths = []
    try:
        for image in images:
            filename = secure_filename(image.filename or "upload.jpg")
            tmp_path = _unique_path(tmp_dir, filename)
            image.save(tmp_path)
            temp_paths.append(tmp_path)

            try:
                with Image.open(tmp_path) as img:
                    img.verify()
            except (OSError, UnidentifiedImageError):
                return jsonify({"error": "invalid image data"}), 400

        settings = VisionSettings.from_mapping(current_app.config)
        quality_valid_paths, quality_skipped = validate_capture_quality(
            [Path(p) for p in temp_paths], settings
        )
        try:
            result = enroll_identity(name, quality_valid_paths, settings)
        except VisionError as exc:
            return jsonify({"error": str(exc)}), 400

        return (
            jsonify(
                {
                    "identity_path": str(result.identity_path),
                    "name": name,
                    "accepted_images": result.accepted_images,
                    "skipped_images": quality_skipped + list(result.skipped_images),
                }
            ),
            201,
        )
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
