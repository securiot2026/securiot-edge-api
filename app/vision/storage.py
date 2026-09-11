from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.vision.errors import VisionError
from app.vision.image import normalize


def validate_display_name(name: str) -> str:
    if any(ord(character) < 32 for character in name):
        raise VisionError("Name must contain 1-80 printable characters.")
    cleaned = " ".join(name.strip().split())
    if not cleaned or len(cleaned) > 80:
        raise VisionError("Name must contain 1-80 printable characters.")
    return cleaned


def identity_slug(name: str) -> str:
    display_name = validate_display_name(name)
    ascii_name = (
        unicodedata.normalize("NFKD", display_name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_name).strip("-").lower()
    if not slug:
        raise VisionError(
            "Name must contain a letter or number usable as a directory name."
        )
    return slug[:64]


@dataclass(frozen=True)
class IdentityDatabase:
    names: tuple[str, ...]
    centroids: np.ndarray

    def match(self, embedding: np.ndarray) -> tuple[str | None, float | None]:
        if not self.names or self.centroids.size == 0:
            return None, None
        normalized = normalize(embedding)
        if self.centroids.shape[1] != normalized.shape[0]:
            raise VisionError(
                "ArcFace embedding size does not match the enrolled identity database."
            )
        similarities = self.centroids @ normalized
        best_index = int(np.argmax(similarities))
        return self.names[best_index], float(similarities[best_index])


class FaceIdentityStorage:
    def __init__(self, identity_dir: Path) -> None:
        self.identity_dir = identity_dir

    def save(self, display_name: str, embeddings: np.ndarray) -> Path:
        display_name = validate_display_name(display_name)
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise VisionError("Cannot save an identity without embeddings.")
        normalized = np.vstack([normalize(row) for row in matrix]).astype(np.float32)
        centroid = normalize(normalized.mean(axis=0))
        destination = self.identity_dir / identity_slug(display_name) / "identity.npz"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VisionError(
                f"Could not create identity directory {destination.parent}: {exc}"
            ) from exc
        temporary: Path | None = None
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
                    embeddings=normalized,
                    centroid=centroid,
                )
            os.replace(temporary, destination)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise VisionError(
                f"Could not save identity database {destination}: {exc}"
            ) from exc
        return destination

    def load(self) -> tuple[IdentityDatabase, list[str]]:
        names = []
        centroids = []
        warnings = []
        for path in sorted(self.identity_dir.glob("*/identity.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    display_name = validate_display_name(
                        str(data["display_name"].item())
                    )
                    centroid = normalize(data["centroid"])
                    embeddings = np.asarray(data["embeddings"])
                    if (
                        embeddings.ndim != 2
                        or embeddings.shape[0] == 0
                        or embeddings.shape[1] != centroid.shape[0]
                    ):
                        raise ValueError("invalid embedding matrix shape")
                names.append(display_name)
                centroids.append(centroid)
            except (EOFError, KeyError, OSError, ValueError, VisionError) as exc:
                warnings.append(f"Ignoring corrupt identity file {path}: {exc}")
        matrix = (
            np.vstack(centroids).astype(np.float32)
            if centroids
            else np.empty((0, 0), dtype=np.float32)
        )
        return IdentityDatabase(tuple(names), matrix), warnings
