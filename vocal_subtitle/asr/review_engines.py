"""Optional review-engine ports and conservative unavailable defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

from .evidence import CandidateEvidence, candidates_from_segments


class ReviewEnginePort(Protocol):
    name: str

    def review(
        self,
        audio: Any,
        sample_rate: int,
        window: Any,
        *,
        language: Optional[str] = None,
        cancellation_token: Any = None,
    ) -> Sequence[CandidateEvidence]:
        """Return evidence for a bounded review window."""


class WindowTranscriptionPort(ReviewEnginePort, Protocol):
    """Common port for primary, context and heterogeneous secondary ASR."""

    family: str
    model_name: str


class ContextReASRPort(ReviewEnginePort, Protocol):
    pass


class QwenASRPort(ReviewEnginePort, Protocol):
    pass


class ForcedAlignerPort(Protocol):
    name: str

    def align(self, audio: Any, sample_rate: int, text: str, window: Any, *, language: Optional[str] = None) -> Sequence[Any]:
        ...


class SEDPort(Protocol):
    name: str

    def detect(self, audio: Any, sample_rate: int, window: Any) -> dict[str, Any]:
        ...


class SemanticReviewPort(Protocol):
    name: str

    def review(self, text: str, context: str = "") -> dict[str, Any]:
        ...


class ReviewEngineUnavailable(RuntimeError):
    """A review backend cannot run without breaking the base subtitle path."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class ReviewUnavailable:
    """Default optional component that records why a stage was skipped."""

    name: str
    reason: str = "optional_component_disabled"

    def review(self, *args: Any, **kwargs: Any) -> Sequence[CandidateEvidence]:
        return ()

    def align(self, *args: Any, **kwargs: Any) -> Sequence[Any]:
        return ()

    def detect(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "unavailable", "reason": self.reason, "engine": self.name}


class CallbackContextReASR:
    """Adapter for the existing main-engine transcribe callback."""

    name = "context-reasr"

    def __init__(self, callback: Callable[..., Sequence[CandidateEvidence]]):
        self._callback = callback

    def review(self, audio: Any, sample_rate: int, window: Any, *, language: Optional[str] = None) -> Sequence[CandidateEvidence]:
        return self._callback(audio, sample_rate, window, language=language)


class CallbackForcedAligner:
    """Inject an external forced-aligner without coupling ASR to its SDK."""

    def __init__(self, callback: Callable[..., Sequence[Any]], name: str = "forced-aligner"):
        self.name = name
        self._callback = callback

    def align(
        self,
        audio: Any,
        sample_rate: int,
        text: str,
        window: Any,
        *,
        language: Optional[str] = None,
    ) -> Sequence[Any]:
        return self._callback(
            audio,
            sample_rate,
            text,
            window,
            language=language,
        )


class CallbackSED:
    """Inject a sound-event detector through the narrow SED port."""

    def __init__(self, callback: Callable[..., dict[str, Any]], name: str = "sed"):
        self.name = name
        self._callback = callback

    def detect(self, audio: Any, sample_rate: int, window: Any) -> dict[str, Any]:
        return self._callback(audio, sample_rate, window)


class CallbackSemanticReview:
    """Inject structured semantic review while keeping LLM details outside ASR."""

    def __init__(self, callback: Callable[..., dict[str, Any]], name: str = "semantic-review"):
        self.name = name
        self._callback = callback

    def review(self, text: str, context: str = "") -> dict[str, Any]:
        return self._callback(text, context)


class WindowedASRContextReASR:
    """Use the selected ASR engine to review one bounded physical window."""

    name = "context-reasr"
    source = "context_reasr"

    def __init__(self, engine_factory: Callable[[], Any]):
        self._engine_factory = engine_factory

    @property
    def model_name(self) -> str:
        return str(getattr(self, "_model_name", "context-reasr"))

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
        engine = self._engine_factory()
        engine.load_model()
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        start = int(window.start * sample_rate)
        end = int(window.end * sample_rate)
        segments = engine.transcribe(audio[start:end], sample_rate, language=language)
        if not isinstance(segments, (list, tuple)):
            segments = [segments]
        return candidates_from_segments(
            segments,
            source=self.source,
            engine=getattr(engine, "name", None),
            model=getattr(engine, "model_name", None),
            window_id=window.id,
            offset=window.start,
        )


class WindowedASREngine(WindowedASRContextReASR):
    """Adapt an existing synchronous ASR engine as a bounded evidence port."""

    def __init__(
        self,
        engine_factory: Callable[[], Any],
        *,
        name: str,
        source: str,
        family: str,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__(engine_factory)
        self.name = name
        self.source = source
        self.family = family
        self._model_name = model_name or name


class WindowedASRQwen(WindowedASRContextReASR):
    """Qwen-compatible bounded ASR adapter with an explicit evidence source."""

    name = "qwen"
    source = "qwen"


__all__ = [
    "CallbackForcedAligner",
    "CallbackContextReASR",
    "CallbackSED",
    "CallbackSemanticReview",
    "ContextReASRPort",
    "ForcedAlignerPort",
    "QwenASRPort",
    "ReviewEnginePort",
    "ReviewEngineUnavailable",
    "ReviewUnavailable",
    "SEDPort",
    "SemanticReviewPort",
    "WindowedASRContextReASR",
    "WindowTranscriptionPort",
    "WindowedASREngine",
    "WindowedASRQwen",
]
