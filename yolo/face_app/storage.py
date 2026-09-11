from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .embedder import normalize
from .errors import FaceAppError


@dataclass(frozen=True)
class IdentityDatabase:
    names: tuple[str, ...]
    centroids: np.ndarray


def validate_display_name(name: str) -> str:
    if any(ord(char) < 32 for char in name):
        raise FaceAppError("Name must contain 1-80 printable characters.")
    cleaned = " ".join(name.strip().split())
    if not cleaned or len(cleaned) > 80:
        raise FaceAppError("Name must contain 1-80 printable characters.")
    return cleaned


def user_slug(name: str) -> str:
    display_name = validate_display_name(name)
    ascii_name = (
        unicodedata.normalize("NFKD", display_name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_name).strip("-").lower()
    if not slug:
        raise FaceAppError(
            "Name must contain at least one letter or number that can be used "
            "as a folder name."
        )
    return slug[:64]


class FaceStorage:
    def __init__(self, data_dir: Path) -> None:
        self.users_dir = data_dir / "users"
        try:
            self.users_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=self.users_dir, prefix=".write-test-"
            ):
                pass
        except OSError as exc:
            raise FaceAppError(
                f"Data directory is not writable ({self.users_dir}): {exc}"
            ) from exc

    def user_dir(self, display_name: str) -> Path:
        return self.users_dir / user_slug(display_name)

    def create_capture_dir(self, display_name: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.user_dir(display_name) / "photos" / stamp
        try:
            path.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise FaceAppError(
                f"Could not create capture directory {path}: {exc}"
            ) from exc
        return path

    def save_identity(self, display_name: str, embeddings: np.ndarray) -> Path:
        display_name = validate_display_name(display_name)
        matrix = np.asarray(
            [normalize(row) for row in embeddings], dtype=np.float32
        )
        if matrix.ndim != 2 or len(matrix) == 0:
            raise FaceAppError("Cannot save an identity without embeddings.")
        centroid = normalize(matrix.mean(axis=0))
        destination = self.user_dir(display_name) / "identity.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=destination.parent,
                prefix=".identity-",
                suffix=".npz",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    display_name=display_name,
                    embeddings=matrix,
                    centroid=centroid,
                )
            os.replace(temporary, destination)
        except OSError as exc:
            if "temporary" in locals():
                temporary.unlink(missing_ok=True)
            raise FaceAppError(
                f"Could not save identity database {destination}: {exc}"
            ) from exc
        return destination

    def load_database(self) -> tuple[IdentityDatabase, list[str]]:
        names: list[str] = []
        centroids: list[np.ndarray] = []
        warnings: list[str] = []
        for path in sorted(self.users_dir.glob("*/identity.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    display_name = str(data["display_name"].item())
                    centroid = normalize(data["centroid"])
                    embeddings = np.asarray(data["embeddings"])
                    if (
                        embeddings.ndim != 2
                        or embeddings.shape[1] != centroid.shape[0]
                        or len(embeddings) == 0
                    ):
                        raise ValueError("invalid embedding matrix shape")
                names.append(validate_display_name(display_name))
                centroids.append(centroid)
            except Exception as exc:
                warnings.append(f"Ignoring corrupt identity file {path}: {exc}")
        matrix = (
            np.vstack(centroids).astype(np.float32)
            if centroids
            else np.empty((0, 0), dtype=np.float32)
        )
        return IdentityDatabase(tuple(names), matrix), warnings
