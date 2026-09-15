"""Qwen3-ASR engine adapter for primary and secondary ASR paths.

The third-party SDK is intentionally isolated here.  The rest of the
pipeline consumes the same ``ASREngine`` contract as Whisper and FunASR.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .base import (
    ASRDependencyError,
    ASREngine,
    ASRModelError,
    TranscriptionSegment,
    WordTimestamp,
)
from .model_paths import (
    DEFAULT_QWEN_MODEL,
    default_qwen_model_path,
    internal_language_name,
    qwen_language_name,
    review_model_cache_dir,
)
from .optional_adapters import _adapt_qwen_segments
from .review_engines import ReviewEngineUnavailableError


def qwen_model_cache_dir() -> Path:
    """Backward-compatible alias for the shared review model directory."""
    return review_model_cache_dir()


def qwen_model_path_ready(model_path: str | Path | None) -> bool:
    path = Path(model_path).expanduser() if model_path else default_qwen_model_path()
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    return any(
        item.is_file() and item.suffix in {".safetensors", ".bin", ".pt", ".pth"}
        for item in path.rglob("*")
    )


class QwenASREngine(ASREngine):
    """Qwen3-ASR adapter with the project's segment/word timestamp contract."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str = "auto",
        language: str | None = None,
    ) -> None:
        self._model_path = str(model_path or default_qwen_model_path())
        self._device = device
        self._language = language
        self._model: Any = None

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def model_name(self) -> str:
        return Path(self._model_path).name or DEFAULT_QWEN_MODEL

    @property
    def model_path(self) -> str:
        return str(Path(self._model_path).expanduser())

    def availability(self) -> dict[str, Any]:
        path = Path(self.model_path)
        if not path.is_dir():
            return {
                "status": "unavailable",
                "reason": "model_path_missing",
                "model": str(path),
            }
        if not qwen_model_path_ready(path):
            return {
                "status": "unavailable",
                "reason": "model_snapshot_incomplete",
                "model": str(path),
            }
        if importlib.util.find_spec("qwen_asr") is None:
            return {
                "status": "unavailable",
                "reason": "qwen_asr_not_installed",
                "model": str(path),
            }
        return {"status": "ready", "model": str(path)}

    def load_model(self) -> None:
        if self._model is not None:
            return
        path = Path(self.model_path)
        if not qwen_model_path_ready(path):
            raise ASRModelError(
                f"Qwen ASR model is not ready: {path}. "
                "Run scripts/download_review_models.py --model qwen3-asr-1.7b."
            )
        try:
            module = importlib.import_module("qwen_asr")
            model_type = getattr(module, "Qwen3ASRModel")
        except (ImportError, AttributeError) as exc:
            raise ASRDependencyError(
                "Qwen ASR runtime is unavailable. Install qwen-asr with --no-deps "
                "and its compatible runtime dependencies."
            ) from exc
        try:
            kwargs = (
                {}
                if self._device in {"", "auto", None}
                else {"device_map": self._device}
            )
            self._model = model_type.from_pretrained(str(path), **kwargs)
        except Exception as exc:
            raise ASRModelError(f"Failed to load Qwen ASR model {path}: {exc}") from exc

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
        **kwargs: Any,
    ) -> list[TranscriptionSegment]:
        self.load_model()
        window = SimpleNamespace(
            id="qwen-primary", start=0.0, end=len(audio) / sample_rate
        )
        try:
            qwen_language = qwen_language_name(language or self._language)
            raw = self._model.transcribe(
                audio=(np.asarray(audio, dtype=np.float32), sample_rate),
                language=qwen_language,
                **kwargs,
            )
        except TypeError:
            raw = self._model.transcribe(
                (np.asarray(audio, dtype=np.float32), sample_rate),
                language=qwen_language,
            )
        except Exception as exc:
            raise ASRModelError(f"Qwen ASR transcription failed: {exc}") from exc

        try:
            candidates = _adapt_qwen_segments(
                raw, window=window, model_name=self.model_name
            )
        except (TypeError, ValueError, ReviewEngineUnavailableError) as exc:
            raise ASRModelError(f"Qwen ASR returned an invalid result: {exc}") from exc

        results: list[TranscriptionSegment] = []
        for candidate in candidates:
            words = [
                WordTimestamp(
                    word=word.text,
                    start=word.start if word.start is not None else candidate.start,
                    end=word.end if word.end is not None else candidate.end,
                    confidence=word.confidence if word.confidence is not None else 1.0,
                )
                for word in candidate.words
            ]
            results.append(
                TranscriptionSegment(
                    text=candidate.text,
                    start=candidate.start,
                    end=candidate.end,
                    words=words,
                    language=internal_language_name(candidate.language)
                    or language
                    or self._language,
                    avg_logprob=0.0,
                )
            )
        return results


__all__ = [
    "DEFAULT_QWEN_MODEL",
    "QwenASREngine",
    "default_qwen_model_path",
    "qwen_model_cache_dir",
    "qwen_model_path_ready",
]
