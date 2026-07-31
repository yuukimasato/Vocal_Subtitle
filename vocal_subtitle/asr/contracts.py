"""Explicit contracts for ASR application services.

The contracts deliberately contain only data and injected capabilities.  They
must not depend on ``Pipeline`` so global, segmented and review paths can be
tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .base import ASREngine


class ProgressPort(Protocol):
    def update_stage(self, index: int, **kwargs: Any) -> Any:
        """Report progress without coupling the service to ProgressManager."""


@dataclass(frozen=True)
class ASRRuntimePorts:
    """Capabilities supplied by the application layer to ASR services."""

    config: Any
    get_engine: Callable[[], ASREngine]
    get_engine_for: Callable[..., ASREngine]
    get_language: Callable[[], Optional[str]]
    set_language: Callable[[Optional[str]], None]
    quality_gate_kwargs: Callable[[], Dict[str, Any]]
    global_runner: Optional[Callable[["GlobalASRRequest"], Any]] = None
    segmented_runner: Optional[Callable[["SegmentedASRRequest"], Any]] = None
    progress: Optional[ProgressPort] = None


@dataclass(frozen=True)
class GlobalASRRequest:
    """Input evidence for one global transcription attempt."""

    audio: np.ndarray
    sample_rate: int
    shadow: Any
    stats: Any
    vad_segments: Optional[Sequence[Any]] = None
    ffmpeg_result: Optional[Dict[str, Any]] = None
    noise_profile: Any = None


@dataclass
class GlobalASRResult:
    """Result of a global transcription attempt."""

    events: List[Any] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    transcript: Any = None


@dataclass(frozen=True)
class SegmentedASRRequest:
    """Input data for segmented/skeleton ASR."""

    audio: np.ndarray
    sample_rate: int
    vocals_path: Any
    segments: Sequence[Any]
    context: Any = None
    chunk_label: str = ""
    run_asr: bool = True


@dataclass
class SegmentedASRResult:
    """Result of segmented ASR and its diagnostics."""

    events: List[Any] = field(default_factory=list)
    segment_count: int = 0
    context: Any = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ASRReviewRequest:
    """Input for global result validation and failure classification."""

    events: Sequence[Any]
    transcript: Any
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class ASRFailureRequest:
    """Normalized exception input for failure classification."""

    error: Exception
