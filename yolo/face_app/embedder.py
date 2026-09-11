from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .errors import FaceAppError


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding has zero or non-finite norm.")
    return value / norm


def preferred_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise FaceAppError(
            "ONNX Runtime is unavailable. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    available = set(ort.get_available_providers())
    providers = [
        name
        for name in ("CUDAExecutionProvider", "CoreMLExecutionProvider")
        if name in available
    ]
    if "CPUExecutionProvider" not in available:
        raise FaceAppError(
            "ONNX Runtime does not expose CPUExecutionProvider; reinstall onnxruntime."
        )
    return [*providers, "CPUExecutionProvider"]


class ArcFaceEmbedder:
    """Recognition-only InsightFace wrapper without a second face detector."""

    def __init__(self) -> None:
        self.providers = preferred_onnx_providers()
        try:
            self._load(self.providers)
        except FaceAppError as exc:
            if self.providers == ["CPUExecutionProvider"]:
                raise
            print(
                f"Warning: ArcFace initialization failed with {self.providers}; "
                f"retrying on CPU: {exc}",
                file=sys.stderr,
            )
            self.providers = ["CPUExecutionProvider"]
            self._load(self.providers)

    def _load(self, providers: list[str]) -> None:
        try:
            from insightface.model_zoo import get_model
            from insightface.utils.storage import ensure_available

            model_dir = Path(ensure_available("models", "buffalo_l")).expanduser()
            model_path = model_dir / "w600k_r50.onnx"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"ArcFace recognition weights were not found at {model_path}."
                )
            model = get_model(str(model_path), providers=providers)
            if model is None or model.taskname != "recognition":
                raise RuntimeError(
                    f"{model_path} did not expose an ArcFace recognition model."
                )
            model.prepare(ctx_id=0)
            self._model = model
        except Exception as exc:
            raise FaceAppError(
                "Could not initialize InsightFace buffalo_l recognition model. "
                "Check internet access for the first model download, disk "
                f"permissions, and ONNX Runtime installation: {exc}"
            ) from exc

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop.size == 0:
            raise ValueError("Cannot embed an empty face crop.")
        try:
            return self._get_embedding(face_crop)
        except Exception as exc:
            if self.providers == ["CPUExecutionProvider"]:
                raise FaceAppError(
                    f"ArcFace inference failed on CPU: {exc}"
                ) from exc
            print(
                f"Warning: ArcFace inference failed with {self.providers}; "
                f"retrying on CPU: {exc}",
                file=sys.stderr,
            )
            self.providers = ["CPUExecutionProvider"]
            self._load(self.providers)
            try:
                return self._get_embedding(face_crop)
            except Exception as cpu_exc:
                raise FaceAppError(
                    f"ArcFace inference also failed on CPU: {cpu_exc}"
                ) from cpu_exc

    def _get_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        import cv2

        input_size = tuple(int(value) for value in self._model.input_size)
        resized = cv2.resize(face_crop, input_size, interpolation=cv2.INTER_AREA)
        features = self._model.get_feat(resized)
        if features is None or len(features) != 1:
            raise ValueError("ArcFace returned an unexpected embedding batch.")
        return normalize(features[0])
