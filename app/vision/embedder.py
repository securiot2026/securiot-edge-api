from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.vision.errors import VisionError
from app.vision.image import normalize

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"


def select_onnx_providers(device: str, available: Sequence[str]) -> list[str]:
    available_set = set(available)
    if CPU_PROVIDER not in available_set:
        raise VisionError(
            "ONNX Runtime does not expose CPUExecutionProvider; reinstall onnxruntime."
        )
    if device == "cpu":
        return [CPU_PROVIDER]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available_set:
            raise VisionError(
                "ARCFACE_DEVICE=cuda requires onnxruntime-gpu and a compatible "
                "CUDA runtime."
            )
        return ["CUDAExecutionProvider", CPU_PROVIDER]
    if device == "coreml":
        if "CoreMLExecutionProvider" not in available_set:
            raise VisionError(
                "ARCFACE_DEVICE=coreml is not available in this ONNX Runtime build."
            )
        return ["CoreMLExecutionProvider", CPU_PROVIDER]
    return [
        provider
        for provider in (
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            CPU_PROVIDER,
        )
        if provider in available_set
    ]


class ArcFaceEmbedder:
    """ArcFace ONNX inference without invoking a second face detector."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        session_factory: Callable[..., Any] | None = None,
        available_providers: Sequence[str] | None = None,
    ) -> None:
        if model_path.suffix.lower() != ".onnx" or not model_path.is_file():
            raise VisionError(
                f"ArcFace ONNX weights were not found at {model_path}. "
                "Set ARCFACE_MODEL_PATH to a compatible recognition model."
            )
        if session_factory is None or available_providers is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise VisionError(
                    "ONNX Runtime is required by the ArcFace embedder."
                ) from exc
            session_factory = session_factory or ort.InferenceSession
            available_providers = available_providers or ort.get_available_providers()

        self._model_path = model_path
        self._session_factory = session_factory
        self.providers = select_onnx_providers(device, available_providers)
        try:
            self._load_session(self.providers)
        except VisionError as exc:
            if self.providers == [CPU_PROVIDER]:
                raise
            logger.warning(
                "ArcFace initialization failed with %s; retrying on CPU: %s",
                self.providers,
                exc,
            )
            self.providers = [CPU_PROVIDER]
            self._load_session(self.providers)

    def _load_session(self, providers: list[str]) -> None:
        try:
            session = self._session_factory(str(self._model_path), providers=providers)
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or not outputs:
                raise ValueError("expected one input and at least one output")
            input_shape = inputs[0].shape
            if len(input_shape) != 4:
                raise ValueError(f"expected NCHW input, got {input_shape!r}")
            if isinstance(input_shape[1], int) and input_shape[1] != 3:
                raise ValueError(f"expected three color channels, got {input_shape!r}")
            self._session = session
            self._input_name = inputs[0].name
            self._output_name = outputs[0].name
            self._input_height = _fixed_dimension(input_shape[2], 112)
            self._input_width = _fixed_dimension(input_shape[3], 112)
        except Exception as exc:
            raise VisionError(
                f"Could not initialize ArcFace model {self._model_path}: {exc}"
            ) from exc

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop.size == 0:
            raise VisionError("Cannot embed an empty face crop.")
        tensor = self._preprocess(face_crop)
        try:
            return self._run(tensor)
        except Exception as exc:
            if self.providers == [CPU_PROVIDER]:
                raise VisionError(f"ArcFace inference failed on CPU: {exc}") from exc
            logger.warning(
                "ArcFace inference failed with %s; retrying on CPU: %s",
                self.providers,
                exc,
            )
            self.providers = [CPU_PROVIDER]
            self._load_session(self.providers)
            try:
                return self._run(tensor)
            except Exception as cpu_exc:
                raise VisionError(
                    f"ArcFace inference also failed on CPU: {cpu_exc}"
                ) from cpu_exc

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        import cv2

        tensor = cv2.dnn.blobFromImage(
            face_crop,
            scalefactor=1.0 / 127.5,
            size=(self._input_width, self._input_height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return np.asarray(tensor, dtype=np.float32)

    def _run(self, tensor: np.ndarray) -> np.ndarray:
        outputs = self._session.run(
            [self._output_name], {self._input_name: tensor}
        )
        if not outputs:
            raise ValueError("ArcFace returned no outputs")
        embedding = np.asarray(outputs[0], dtype=np.float32)
        if embedding.size == 0:
            raise ValueError("ArcFace returned an empty embedding")
        return normalize(embedding.reshape(-1))


def _fixed_dimension(value: Any, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default
