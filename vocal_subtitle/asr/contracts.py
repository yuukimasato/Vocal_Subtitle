"""Explicit contracts for ASR application services.

The contracts deliberately contain only data and injected capabilities.  They
must not depend on ``Pipeline`` so global, segmented and review paths can be
tested independently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .base import ASREngine
from .evidence import CandidateEvidence, EvidenceDecision


class ProgressPort(Protocol):
    def update_stage(self, index: int, **kwargs: Any) -> Any:
        """Report progress without coupling the service to ProgressManager."""


@dataclass(frozen=True)
class ASRRuntimePorts:
    """Capabilities supplied by the application layer to ASR services."""

    config: Any
    get_engine: Callable[[], ASREngine]
    get_engine_for: Callable[..., ASREngine]
    get_language: Callable[[], str | None]
    set_language: Callable[[str | None], None]
    quality_gate_kwargs: Callable[[], dict[str, Any]]
    global_runner: Callable[[GlobalASRRequest], Any] | None = None
    segmented_runner: Callable[[SegmentedASRRequest], Any] | None = None
    progress: ProgressPort | None = None


@dataclass(frozen=True)
class GlobalASRRequest:
    """Input evidence for one global transcription attempt."""

    audio: np.ndarray
    sample_rate: int
    shadow: Any
    stats: Any
    vad_segments: Sequence[Any] | None = None
    ffmpeg_result: dict[str, Any] | None = None
    noise_profile: Any = None


@dataclass
class GlobalASRResult:
    """Result of a global transcription attempt."""

    events: list[Any] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    transcript: Any = None
    # New evidence is appended after the legacy positional fields so callers
    # constructing GlobalASRResult(events, diagnostics, transcript) remain
    # compatible during the migration.
    evidence: list[Any] = field(default_factory=list)


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

    events: list[Any] = field(default_factory=list)
    segment_count: int = 0
    context: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceReviewRequest:
    """Input for the independent evidence review service."""

    events: Sequence[Any]
    audio: Any = None
    sample_rate: int = 16000
    physical_timeline: Any = None
    global_evidence: Sequence[CandidateEvidence] = ()
    recovery_evidence: Sequence[CandidateEvidence] = ()
    input_hash: str = ""
    audio_hash: str = ""
    physical_timeline_version: str = "physical-timeline-v1"
    route_version: str = ""
    engine: str = ""
    model: str = ""
    review_policy: str = "risk_only"
    secondary_engine: str = ""
    pair_route_version: str = ""
    review_policy_version: str = "review-policy-v1"
    risk_policy_version: str = "risk-policy-v1"
    decision_policy_version: str = "decision-policy-v1"
    evidence_schema_version: str = "evidence-v1"


@dataclass
class EvidenceReviewResult:
    """Output of evidence scoring, review and final decision stages."""

    events: list[Any] = field(default_factory=list)
    decisions: list[EvidenceDecision] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ASRReviewRequest:
    """Input for global result validation and failure classification."""

    events: Sequence[Any]
    transcript: Any
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ASRFailureRequest:
    """Normalized exception input for failure classification."""

    error: Exception
