"""Lazy adapters for optional multi-engine evidence providers.

The adapters keep third-party SDKs outside the evidence contracts. They load
only when a configured review window reaches the corresponding stage and turn
missing runtimes/models into a typed, diagnosable degradation.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

from .evidence import CandidateEvidence, EvidenceWord, candidates_from_segments
from .model_paths import (
    default_qwen_model_path,
    internal_language_name,
    qwen_language_name,
)
from .review_engines import ReviewEngineUnavailable


ModelLoader = Callable[[str, str], Any]


def _require_local_model(model_path: str, *, allow_remote: bool) -> None:
    if allow_remote:
        return
    if not model_path or not Path(model_path).expanduser().exists():
        raise ReviewEngineUnavailable(
            "model_unavailable",
            f"local review model does not exist: {model_path or '<empty>'}",
        )


def _relative_window_audio(audio: Any, sample_rate: int, window: Any) -> Any:
    start = max(0, int(float(window.start) * sample_rate))
    end = max(start + 1, int(float(window.end) * sample_rate))
    return audio[start:end]


def _value(item: Any, *names: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        for name in names:
            if name in item:
                return item[name]
        return default
    for name in names:
        value = getattr(item, name, None)
        if value is not None:
            return value
    return default


def _word(
    value: Any,
    *,
    offset: float,
    index: int,
    time_source: str = "qwen_forced_alignment",
) -> Optional[EvidenceWord]:
    if isinstance(value, EvidenceWord):
        return value
    text = str(_value(value, "word", "text", "token", default="")).strip()
    if not text:
        return None
    start = _value(value, "start", "start_time", "begin", default=None)
    end = _value(value, "end", "end_time", "finish", default=None)
    if isinstance(value, dict) and "timestamp" in value:
        timestamp = value["timestamp"]
        if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
            start, end = timestamp[0], timestamp[1]
    try:
        start = None if start is None else float(start) + offset
        end = None if end is None else float(end) + offset
        if start is not None and end is not None and end <= start:
            return None
    except (TypeError, ValueError):
        start = end = None
    return EvidenceWord(
        id=f"review-word:{index:06d}",
        text=text,
        start=start,
        end=end,
        confidence=_value(value, "confidence", "score", default=None),
        time_source=time_source,
        diagnostics={"adapter": "qwen", "raw_type": type(value).__name__},
    )


def _segments(raw: Any, *, window: Any) -> list[Any]:
    """Normalize common Qwen result containers without importing its types."""
    if raw is None:
        return []
    if isinstance(raw, dict) and "segments" in raw:
        raw = raw["segments"]
    elif hasattr(raw, "segments"):
        raw = raw.segments
    if isinstance(raw, (list, tuple)):
        return list(raw)
    text = str(_value(raw, "text", "transcript", default="")).strip()
    if not text:
        return []
    return [
        SimpleNamespace(
            text=text,
            start=_value(raw, "start", "start_time", default=0.0),
            end=_value(raw, "end", "end_time", default=float(window.end - window.start)),
            words=_value(raw, "words", "timestamps", default=[]),
            language=_value(raw, "language", default=None),
        )
    ]


def _adapt_qwen_segments(raw: Any, *, window: Any, model_name: str) -> list[CandidateEvidence]:
    adapted = []
    for index, segment in enumerate(_segments(raw, window=window)):
        words = []
        for word_index, item in enumerate(
            _value(segment, "words", "timestamps", "time_stamps", default=[]) or ()
        ):
            item = _word(
                item,
                offset=float(window.start),
                index=index * 10000 + word_index,
                time_source="native_word_timestamp",
            )
            if item is not None:
                words.append(item)
        start = _value(segment, "start", "start_time", default=0.0)
        end = _value(segment, "end", "end_time", default=float(window.end - window.start))
        try:
            start = float(start) + float(window.start)
            end = float(end) + float(window.start)
        except (TypeError, ValueError):
            start, end = float(window.start), float(window.end)
        if end <= start:
            continue
        normalized = SimpleNamespace(
            text=str(_value(segment, "text", "transcript", default="")).strip(),
            start=start,
            end=end,
            words=words,
            language=internal_language_name(_value(segment, "language", default=None)),
        )
        if normalized.text:
            adapted.extend(
                candidates_from_segments(
                    [normalized],
                    source="qwen",
                    engine="qwen",
                    model=model_name,
                    window_id=window.id,
                )
            )
    return adapted


class LazyQwenASR:
    """Qwen3-ASR bounded-window adapter with injectable model loading."""

    name = "qwen"

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        allow_remote: bool = False,
        model_loader: Optional[ModelLoader] = None,
    ) -> None:
        if model_path:
            self.model_path = str(Path(model_path).expanduser())
        else:
            self.model_path = str(default_qwen_model_path())
        self.device = device
        self.allow_remote = allow_remote
        self._model_loader = model_loader
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self.model_path or "qwen3-asr"

    def availability(self) -> dict[str, Any]:
        if self._model_loader is not None:
            return {"status": "ready", "engine": self.name, "model": self.model_name}
        if not self.model_path:
            return {"status": "unavailable", "reason": "model_path_missing", "engine": self.name}
        if not self.allow_remote and not Path(self.model_path).expanduser().exists():
            return {"status": "unavailable", "reason": "model_path_missing", "engine": self.name}
        try:
            importlib.import_module("qwen_asr")
        except ImportError:
            return {"status": "unavailable", "reason": "qwen_asr_not_installed", "engine": self.name}
        return {"status": "ready", "engine": self.name, "model": self.model_name}

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._model_loader is not None:
            self._model = self._model_loader(self.model_path, self.device)
            return self._model
        _require_local_model(self.model_path, allow_remote=self.allow_remote)
        try:
            module = importlib.import_module("qwen_asr")
            model_type = getattr(module, "Qwen3ASRModel")
            kwargs = {} if self.device == "auto" else {"device_map": self.device}
            self._model = model_type.from_pretrained(self.model_path, **kwargs)
        except (ImportError, AttributeError) as exc:
            raise ReviewEngineUnavailable(
                "dependency_unavailable", "install qwen-asr to enable Qwen3-ASR"
            ) from exc
        except Exception as exc:
            raise ReviewEngineUnavailable("model_unavailable", str(exc)) from exc
        return self._model

    def review(
        self,
        audio: Any,
        sample_rate: int,
        window: Any,
        *,
        language: Optional[str] = None,
        cancellation_token: Any = None,
    ) -> Sequence[CandidateEvidence]:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        model = self._load()
        clipped = _relative_window_audio(audio, sample_rate, window)
        qwen_language = qwen_language_name(language)
        try:
            raw = model.transcribe(audio=(clipped, sample_rate), language=qwen_language)
        except TypeError:
            raw = model.transcribe((clipped, sample_rate), language=qwen_language)
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        return _adapt_qwen_segments(raw, window=window, model_name=self.model_name)


class LazyQwenForcedAligner:
    """Qwen3-ForcedAligner adapter that emits only secondary word timing."""

    name = "forced-aligner"

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        allow_remote: bool = False,
        model_loader: Optional[ModelLoader] = None,
    ) -> None:
        self.model_path = str(model_path or "")
        self.device = device
        self.allow_remote = allow_remote
        self._model_loader = model_loader
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self.model_path or "qwen3-forced-aligner"

    def availability(self) -> dict[str, Any]:
        if self._model_loader is not None:
            return {"status": "ready", "engine": self.name, "model": self.model_name}
        if not self.model_path:
            return {"status": "unavailable", "reason": "model_path_missing", "engine": self.name}
        if not self.allow_remote and not Path(self.model_path).expanduser().exists():
            return {"status": "unavailable", "reason": "model_path_missing", "engine": self.name}
        try:
            importlib.import_module("qwen_asr")
        except ImportError:
            return {"status": "unavailable", "reason": "qwen_asr_not_installed", "engine": self.name}
        return {"status": "ready", "engine": self.name, "model": self.model_name}

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._model_loader is not None:
            self._model = self._model_loader(self.model_path, self.device)
            return self._model
        _require_local_model(self.model_path, allow_remote=self.allow_remote)
        try:
            module = importlib.import_module("qwen_asr")
            model_type = getattr(module, "Qwen3ForcedAligner")
            kwargs = {} if self.device == "auto" else {"device_map": self.device}
            self._model = model_type.from_pretrained(self.model_path, **kwargs)
        except (ImportError, AttributeError) as exc:
            raise ReviewEngineUnavailable(
                "dependency_unavailable", "install qwen-asr to enable ForcedAligner"
            ) from exc
        except Exception as exc:
            raise ReviewEngineUnavailable("model_unavailable", str(exc)) from exc
        return self._model

    def align(self, audio: Any, sample_rate: int, text: str, window: Any, *, language: Optional[str] = None) -> Sequence[EvidenceWord]:
        model = self._load()
        clipped = _relative_window_audio(audio, sample_rate, window)
        try:
            raw = model.align(audio=(clipped, sample_rate), text=text, language=language)
        except TypeError:
            raw = model.align((clipped, sample_rate), text, language=language)
        values = raw.get("words", raw) if isinstance(raw, dict) else getattr(raw, "words", raw)
        if values and isinstance(values[0], (list, tuple)):
            values = values[0]
        result = []
        for index, item in enumerate(values or ()):
            word = _word(item, offset=float(window.start), index=index)
            if word is not None:
                result.append(word)
        return result


class LazyAudioClassifierSED:
    """Transformers audio-classification adapter for physical sound evidence."""

    name = "sed"

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        allow_remote: bool = False,
        model_loader: Optional[ModelLoader] = None,
    ) -> None:
        self.model_path = str(model_path or "")
        self.device = device
        self.allow_remote = allow_remote
        self._model_loader = model_loader
        self._pipeline: Any = None

    @property
    def model_name(self) -> str:
        return self.model_path or "audio-classification"

    def availability(self) -> dict[str, Any]:
        if self._model_loader is not None:
            return {"status": "ready", "engine": self.name, "model": self.model_name}
        if not self.model_path:
            return {"status": "unavailable", "reason": "model_path_missing", "engine": self.name}
        if not self.allow_remote and not Path(self.model_path).expanduser().exists():
            return {"status": "unavailable", "reason": "model_path_missing", "engine": self.name}
        try:
            importlib.import_module("transformers")
        except ImportError:
            return {"status": "unavailable", "reason": "transformers_not_installed", "engine": self.name}
        return {"status": "ready", "engine": self.name, "model": self.model_name}

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if self._model_loader is not None:
            self._pipeline = self._model_loader(self.model_path, self.device)
            return self._pipeline
        _require_local_model(self.model_path, allow_remote=self.allow_remote)
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ReviewEngineUnavailable(
                "dependency_unavailable", "install transformers to enable SED"
            ) from exc
        try:
            if self.device in {"auto", "cpu"}:
                device = -1
            elif str(self.device).startswith("cuda:"):
                device = int(str(self.device).split(":", 1)[1])
            else:
                device = int(self.device)
            self._pipeline = pipeline(
                "audio-classification",
                model=self.model_path,
                device=device,
                model_kwargs={"local_files_only": not self.allow_remote},
            )
        except Exception as exc:
            raise ReviewEngineUnavailable("model_unavailable", str(exc)) from exc
        return self._pipeline

    def detect(self, audio: Any, sample_rate: int, window: Any) -> dict[str, Any]:
        classifier = self._load()
        clipped = _relative_window_audio(audio, sample_rate, window)
        try:
            values = classifier({"raw": clipped, "sampling_rate": sample_rate})
        except TypeError:
            values = classifier(clipped)
        labels = [
            {
                "label": str(_value(item, "label", default="unknown")),
                "score": _value(item, "score", default=None),
            }
            for item in (values or ())
        ]
        return {
            "status": "ok",
            "engine": self.name,
            "model": self.model_name,
            "window_id": getattr(window, "id", None),
            "start": float(window.start),
            "end": float(window.end),
            "labels": labels,
        }


__all__ = ["LazyAudioClassifierSED", "LazyQwenASR", "LazyQwenForcedAligner"]
