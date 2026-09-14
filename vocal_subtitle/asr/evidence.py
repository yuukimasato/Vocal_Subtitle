"""Evidence contracts shared by ASR, review and subtitle decision stages."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence


EVIDENCE_SCHEMA_VERSION = "evidence-v1"
RISK_POLICY_VERSION = "risk-policy-v1"
DECISION_POLICY_VERSION = "decision-policy-v1"
TIME_SOURCES = {
    "native_word_timestamp",
    "qwen_forced_alignment",
    "segment_boundary",
    "physical_acoustic_boundary",
    # 高精度链路的词级时间来源（优化方案 3.1）。
    "whisperx_alignment",
    "faster_whisper_word",
}
EVIDENCE_SOURCES = {
    "segmented",
    "global",
    "context_reasr",
    "local_recovery",
    "qwen",
    "forced_aligner",
    "sed",
    "llm",
}
DECISIONS = {"keep", "replace", "split", "drop", "unresolved"}
RISK_LEVELS = {"low", "medium", "high", "critical"}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _number(value: Any, name: str, *, allow_none: bool = False) -> Optional[float]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _confidence(value: Any) -> Optional[float]:
    result = _number(value, "confidence", allow_none=True)
    if result is not None and not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return result


def _range(start: Any, end: Any, name: str) -> tuple[float, float]:
    first = _number(start, f"{name}_start")
    last = _number(end, f"{name}_end")
    assert first is not None and last is not None
    if first < 0 or last <= first:
        raise ValueError(f"{name} must satisfy 0 <= start < end")
    return first, last


def _optional_range(
    start: Any,
    end: Any,
    name: str,
) -> tuple[Optional[float], Optional[float]]:
    if start is None and end is None:
        return None, None
    if start is None or end is None:
        raise ValueError(f"{name} requires both start and end")
    return _range(start, end, name)


@dataclass(frozen=True)
class EvidenceWord:
    """A word observation from one ASR or alignment source."""

    id: str
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: Optional[float] = None
    time_source: str = "native_word_timestamp"
    speaker_id: Optional[int] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    source_id: Optional[str] = None
    offset_id: Optional[str] = None
    window_id: Optional[str] = None
    dedup_key: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "id"))
        object.__setattr__(self, "text", _text(self.text, "text"))
        start, end = _optional_range(self.start, self.end, "word time")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if self.time_source not in TIME_SOURCES:
            raise ValueError(f"unsupported time_source: {self.time_source}")
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if self.speaker_id is not None:
            if isinstance(self.speaker_id, bool) or not isinstance(self.speaker_id, int):
                raise ValueError("speaker_id must be an integer or None")
        object.__setattr__(self, "diagnostics", copy.deepcopy(dict(self.diagnostics)))
        source_id = self.source_id or self.diagnostics.get("source")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "offset_id", self.offset_id or self.diagnostics.get("offset_id"))
        object.__setattr__(self, "window_id", self.window_id or self.diagnostics.get("window_id"))
        dedup_key = self.dedup_key or hashlib.sha1(
            f"{self.text}|{self.start}|{self.end}|{source_id or ''}".encode("utf-8")
        ).hexdigest()[:16]
        object.__setattr__(self, "dedup_key", dedup_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "time_source": self.time_source,
            "speaker_id": self.speaker_id,
            "diagnostics": copy.deepcopy(self.diagnostics),
            "source_id": self.source_id,
            "offset_id": self.offset_id,
            "window_id": self.window_id,
            "dedup_key": self.dedup_key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceWord":
        return cls(**dict(payload))


@dataclass(frozen=True)
class CandidateEvidence:
    """A subtitle candidate or supporting observation."""

    id: str
    source: str
    text: str
    start: float
    end: float
    engine: Optional[str] = None
    model: Optional[str] = None
    window_id: Optional[str] = None
    words: tuple[EvidenceWord, ...] = ()
    confidence: Optional[float] = None
    no_speech_prob: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    physical_clip_id: Optional[str] = None
    language: Optional[str] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    source_id: Optional[str] = None
    offset_id: Optional[str] = None
    candidate_role: str = "primary"
    alternative_for: Optional[str] = None
    trace_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "id"))
        if self.source not in EVIDENCE_SOURCES:
            raise ValueError(f"unsupported evidence source: {self.source}")
        object.__setattr__(self, "text", _text(self.text, "text"))
        start, end = _range(self.start, self.end, "candidate time")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "words", tuple(self.words))
        for name in ("no_speech_prob", "avg_logprob", "compression_ratio"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, allow_none=True)
            )
        object.__setattr__(self, "diagnostics", copy.deepcopy(dict(self.diagnostics)))
        source_id = self.source_id or self.id
        object.__setattr__(self, "source_id", str(source_id))
        role = str(self.candidate_role or "primary")
        if self.source == "global" and role == "primary":
            role = "global_signal"
        object.__setattr__(self, "candidate_role", role)
        object.__setattr__(self, "trace_context", copy.deepcopy({
            "schema_version": "offline-trace-v1",
            "source_id": source_id,
            "offset_id": self.offset_id,
            "window_id": self.window_id,
            "candidate_id": self.id,
            "candidate_role": role,
            "physical_span_ids": [self.physical_clip_id] if self.physical_clip_id else [],
            **dict(self.trace_context or {}),
        }))

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def has_word_times(self) -> bool:
        return bool(self.words) and all(
            word.start is not None and word.end is not None for word in self.words
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "engine": self.engine,
            "model": self.model,
            "window_id": self.window_id,
            "words": [word.to_dict() for word in self.words],
            "confidence": self.confidence,
            "no_speech_prob": self.no_speech_prob,
            "avg_logprob": self.avg_logprob,
            "compression_ratio": self.compression_ratio,
            "physical_clip_id": self.physical_clip_id,
            "language": self.language,
            "diagnostics": copy.deepcopy(self.diagnostics),
            "source_id": self.source_id,
            "offset_id": self.offset_id,
            "candidate_role": self.candidate_role,
            "alternative_for": self.alternative_for,
            "trace_context": copy.deepcopy(self.trace_context),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateEvidence":
        data = dict(payload)
        data["words"] = tuple(EvidenceWord.from_dict(item) for item in data.get("words", ()))
        return cls(**data)


@dataclass(frozen=True)
class EvidenceDecision:
    """The only contract allowed to become a final subtitle event."""

    candidate_ids: tuple[str, ...]
    decision: str
    final_text: str
    final_words: tuple[EvidenceWord, ...]
    start: Optional[float]
    end: Optional[float]
    time_source: str
    confidence: Optional[float]
    risk_score: float
    risk_level: str
    evidence_codes: tuple[str, ...] = ()
    physical_validation: dict[str, Any] = field(default_factory=dict)
    revision_trace: tuple[dict[str, Any], ...] = ()
    decision_id: Optional[str] = None
    selected_candidate_id: Optional[str] = None
    trace_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_ids:
            raise ValueError("candidate_ids must not be empty")
        if self.decision not in DECISIONS:
            raise ValueError(f"unsupported decision: {self.decision}")
        if self.risk_level not in RISK_LEVELS:
            raise ValueError(f"unsupported risk_level: {self.risk_level}")
        _optional_range(self.start, self.end, "decision time")
        if self.time_source not in TIME_SOURCES:
            raise ValueError(f"unsupported time_source: {self.time_source}")
        score = _number(self.risk_score, "risk_score")
        assert score is not None
        if not 0.0 <= score <= 1.0:
            raise ValueError("risk_score must be between 0 and 1")
        object.__setattr__(self, "risk_score", score)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "final_words", tuple(self.final_words))
        object.__setattr__(self, "evidence_codes", tuple(self.evidence_codes))
        object.__setattr__(self, "physical_validation", copy.deepcopy(dict(self.physical_validation)))
        revision_trace = [copy.deepcopy(item) for item in self.revision_trace]
        decision_id = self.decision_id or (
            next((item.get("decision_id") for item in revision_trace if item.get("decision_id")), None)
            or "decision:" + hashlib.sha1(
                "|".join(map(str, (
                    *self.candidate_ids,
                    self.decision,
                    self.start,
                    self.end,
                    self.final_text,
                ))).encode("utf-8")
            ).hexdigest()[:12]
        )
        object.__setattr__(self, "decision_id", str(decision_id))
        selected = self.selected_candidate_id or (
            next((item.get("selected_candidate_id") for item in revision_trace if item.get("selected_candidate_id")), None)
            or self.candidate_ids[0]
        )
        object.__setattr__(self, "selected_candidate_id", str(selected))
        if not any(item.get("stage") == "evidence_decision" for item in revision_trace):
            revision_trace.append({
                "stage": "evidence_decision",
                "decision_id": str(decision_id),
                "candidate_ids": list(self.candidate_ids),
                "selected_candidate_id": str(selected),
                "decision": self.decision,
            })
        object.__setattr__(self, "revision_trace", tuple(revision_trace))
        physical_span_ids = []
        if isinstance(self.physical_validation, dict):
            physical_span_ids.extend(self.physical_validation.get("physical_span_ids", []) or [])
            if self.physical_validation.get("physical_clip_id"):
                physical_span_ids.append(self.physical_validation["physical_clip_id"])
        object.__setattr__(self, "trace_context", copy.deepcopy({
            "schema_version": "offline-trace-v1",
            "source_id": "evidence_decision",
            "candidate_id": self.candidate_ids[0],
            "candidate_ids": list(self.candidate_ids),
            "decision_id": str(decision_id),
            "physical_span_ids": list(dict.fromkeys(item for item in physical_span_ids if item is not None)),
            **dict(self.trace_context or {}),
        }))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "decision": self.decision,
            "final_text": self.final_text,
            "final_words": [word.to_dict() for word in self.final_words],
            "start": self.start,
            "end": self.end,
            "time_source": self.time_source,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "evidence_codes": list(self.evidence_codes),
            "physical_validation": copy.deepcopy(self.physical_validation),
            "revision_trace": list(self.revision_trace),
            "decision_id": self.decision_id,
            "selected_candidate_id": self.selected_candidate_id,
            "trace_context": copy.deepcopy(self.trace_context),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceDecision":
        data = dict(payload)
        data["candidate_ids"] = tuple(data.get("candidate_ids", ()))
        data["final_words"] = tuple(
            EvidenceWord.from_dict(item) for item in data.get("final_words", ())
        )
        data["evidence_codes"] = tuple(data.get("evidence_codes", ()))
        data["revision_trace"] = tuple(data.get("revision_trace", ()))
        return cls(**data)


@dataclass(frozen=True)
class DecisionEvidenceBundle:
    """Normalized secondary evidence available to the decision engine."""

    source: str
    status: str
    candidate_ids: tuple[str, ...] = ()
    window_id: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _text(self.source, "source"))
        object.__setattr__(self, "status", _text(self.status, "status"))
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
        object.__setattr__(self, "evidence", copy.deepcopy(dict(self.evidence)))
        object.__setattr__(self, "diagnostics", copy.deepcopy(dict(self.diagnostics)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "candidate_ids": list(self.candidate_ids),
            "window_id": self.window_id,
            "evidence": copy.deepcopy(self.evidence),
            "diagnostics": copy.deepcopy(self.diagnostics),
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """Inputs to scoring and decision without exposing Pipeline internals."""

    segmented: tuple[CandidateEvidence, ...] = ()
    global_evidence: tuple[CandidateEvidence, ...] = ()
    review: tuple[CandidateEvidence, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def all_candidates(self) -> tuple[CandidateEvidence, ...]:
        return self.segmented + self.global_evidence + self.review

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "segmented": [item.to_dict() for item in self.segmented],
            "global_evidence": [item.to_dict() for item in self.global_evidence],
            "review": [item.to_dict() for item in self.review],
            "diagnostics": copy.deepcopy(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceBundle":
        data = dict(payload)
        version = data.pop("schema_version", EVIDENCE_SCHEMA_VERSION)
        if version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema: {version}")
        return cls(
            segmented=tuple(
                CandidateEvidence.from_dict(item) for item in data.get("segmented", ())
            ),
            global_evidence=tuple(
                CandidateEvidence.from_dict(item)
                for item in data.get("global_evidence", ())
            ),
            review=tuple(
                CandidateEvidence.from_dict(item) for item in data.get("review", ())
            ),
            diagnostics=copy.deepcopy(dict(data.get("diagnostics", {}))),
        )


def evidence_word_from_asr(word: Any, word_id: str, *, time_source: str = "native_word_timestamp") -> EvidenceWord:
    """Adapt existing ASR word objects while preserving missing values."""
    start = getattr(word, "raw_start", getattr(word, "start", None))
    end = getattr(word, "raw_end", getattr(word, "end", None))
    metadata = dict(getattr(word, "metadata", {}) or {})
    if start is None or end is None:
        start = end = None
    else:
        try:
            start_value = float(start)
            end_value = float(end)
        except (TypeError, ValueError):
            start_value = end_value = None
        if (
            start_value is None
            or end_value is None
            or not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or start_value < 0.0
            or end_value <= start_value
        ):
            metadata = {
                **metadata,
                "timing_degraded": True,
                "invalid_timing": {
                    "start": start,
                    "end": end,
                    "reason": "missing_or_non_monotonic_word_time",
                },
            }
            start = end = None
        else:
            start, end = start_value, end_value
    time_source = metadata.get("time_source", time_source)
    return EvidenceWord(
        id=word_id,
        text=str(getattr(word, "word", getattr(word, "text", ""))).strip(),
        start=start,
        end=end,
        confidence=getattr(word, "confidence", None),
        time_source=time_source,
        speaker_id=getattr(word, "speaker_id", None),
        diagnostics=metadata,
    )


def candidate_from_subtitle_event(event: Any, *, source: str = "segmented") -> CandidateEvidence:
    """Adapt the existing SubtitleEvent into the evidence contract."""
    words = []
    event_start = float(getattr(event, "start", 0.0))
    event_time_source = getattr(event, "time_source", "")
    default_time_source = (
        event_time_source
        if event_time_source in TIME_SOURCES
        else "segment_boundary"
        if any(
            marker in str(event_time_source).casefold()
            for marker in ("degrad", "boundary")
        )
        else "native_word_timestamp"
    )
    for index, word in enumerate(getattr(event, "words", ()) or ()):
        word_id = (
            (getattr(event, "source_word_ids", ()) or ())[index]
            if index < len(getattr(event, "source_word_ids", ()) or ())
            else f"{source}:{getattr(event, 'index', 0)}:word:{index:04d}"
        )
        adapted = evidence_word_from_asr(
            word,
            word_id,
            time_source=default_time_source,
        )
        # Global IR words already use absolute raw_* coordinates. Legacy
        # SubtitleEvent words are relative to the event and need projection.
        if (
            adapted.start is not None
            and adapted.end is not None
            and not hasattr(word, "raw_start")
        ):
            adapted = EvidenceWord(
                id=adapted.id,
                text=adapted.text,
                start=adapted.start + event_start,
                end=adapted.end + event_start,
                confidence=adapted.confidence,
                time_source=adapted.time_source,
                speaker_id=adapted.speaker_id,
                diagnostics=adapted.diagnostics,
            )
        words.append(adapted)
    confidence_values = [word.confidence for word in words if word.confidence is not None]
    invalid_word_timing_count = sum(
        1 for word in words if (word.diagnostics or {}).get("invalid_timing")
    )
    return CandidateEvidence(
        id=f"{source}:event:{getattr(event, 'index', 0):06d}",
        source=source,
        text=str(getattr(event, "text", "")).strip(),
        start=float(event.start),
        end=float(event.end),
        words=tuple(words),
        confidence=(sum(confidence_values) / len(confidence_values)) if confidence_values else None,
        physical_clip_id=getattr(event, "physical_region_id", None) or getattr(event, "physical_bin_id", None),
        diagnostics={
            "legacy_event": True,
            "invalid_word_timing_count": invalid_word_timing_count,
        },
    )


def candidates_from_segments(
    segments: Sequence[Any],
    *,
    source: str,
    engine: Optional[str] = None,
    model: Optional[str] = None,
    window_id: Optional[str] = None,
    offset: float = 0.0,
) -> list[CandidateEvidence]:
    """Adapt transcription segments into one candidate per segment."""
    result = []
    for index, segment in enumerate(segments):
        text = str(getattr(segment, "text", "")).strip()
        # ASR backends may emit timing-only or empty segments.  They cannot
        # satisfy the evidence contract and must not abort the whole review.
        if not text:
            continue
        words = []
        for word_index, word in enumerate(getattr(segment, "words", ()) or ()):
            adapted = evidence_word_from_asr(
                word,
                f"{source}:{window_id or 'audio'}:seg:{index:04d}:word:{word_index:04d}",
            )
            if adapted.start is not None:
                adapted = EvidenceWord(
                    id=adapted.id,
                    text=adapted.text,
                    start=adapted.start + offset,
                    end=adapted.end + offset if adapted.end is not None else None,
                    confidence=adapted.confidence,
                    time_source=adapted.time_source,
                    speaker_id=adapted.speaker_id,
                    diagnostics=adapted.diagnostics,
                )
            words.append(adapted)
        start = float(getattr(segment, "start", 0.0)) + offset
        end = float(getattr(segment, "end", start + 0.01)) + offset
        confidence_values = [word.confidence for word in words if word.confidence is not None]
        invalid_word_timing_count = sum(
            1 for word in words if (word.diagnostics or {}).get("invalid_timing")
        )
        result.append(CandidateEvidence(
            id=f"{source}:segment:{index:06d}:{window_id or 'audio'}",
            source=source,
            text=text,
            start=start,
            end=max(start + 0.01, end),
            engine=engine,
            model=model,
            window_id=window_id,
            words=tuple(words),
            confidence=(sum(confidence_values) / len(confidence_values)) if confidence_values else None,
            no_speech_prob=getattr(segment, "no_speech_prob", None),
            avg_logprob=getattr(segment, "avg_logprob", None),
            compression_ratio=getattr(segment, "compression_ratio", None),
            language=getattr(segment, "language", None),
            diagnostics={
                "invalid_word_timing_count": invalid_word_timing_count,
            },
        ))
    return result


def candidates_from_global_transcript(
    transcript: Any,
    *,
    source: str = "global",
) -> list[CandidateEvidence]:
    """Adapt the global IR without passing through legacy subtitle events."""
    if transcript is None:
        return []
    words_by_id = {
        getattr(word, "id", ""): word
        for word in (getattr(transcript, "words", ()) or ())
        if getattr(word, "id", None)
    }
    result: list[CandidateEvidence] = []
    for index, segment in enumerate(getattr(transcript, "segments", ()) or ()):
        text = str(getattr(segment, "text", "") or "").strip()
        segment_word_ids = list(getattr(segment, "word_ids", ()) or ())
        segment_words = []
        for word_id in segment_word_ids:
            word = words_by_id.get(word_id)
            if word is None:
                continue
            metadata = getattr(word, "metadata", {}) or {}
            time_source = metadata.get("time_source", "native_word_timestamp")
            if time_source not in TIME_SOURCES:
                time_source = "segment_boundary"
            segment_words.append(evidence_word_from_asr(
                word,
                str(word.id),
                time_source=time_source,
            ))
        if not text and segment_words:
            text = " ".join(item.text for item in segment_words).strip()
        if not text:
            continue
        start = getattr(segment, "raw_start", None)
        end = getattr(segment, "raw_end", None)
        if start is None or end is None:
            continue
        confidence_values = [
            item.confidence for item in segment_words if item.confidence is not None
        ]
        metadata = dict(getattr(segment, "metadata", {}) or {})
        result.append(CandidateEvidence(
            id=f"{source}:segment:{getattr(segment, 'id', index)}",
            source=source,
            text=text,
            start=float(start),
            end=float(end),
            engine=getattr(transcript, "backend", None),
            window_id=(
                getattr(words_by_id.get(segment_word_ids[0]), "source_window_id", None)
                if segment_word_ids else None
            ),
            words=tuple(segment_words),
            confidence=(
                sum(confidence_values) / len(confidence_values)
                if confidence_values else None
            ),
            no_speech_prob=metadata.get("no_speech_prob"),
            avg_logprob=getattr(segment, "avg_logprob", None),
            compression_ratio=metadata.get("compression_ratio"),
            physical_clip_id=metadata.get("physical_clip_id"),
            language=getattr(segment, "language", None),
            diagnostics={
                "global_transcript_status": getattr(transcript, "status", None),
                "segment_metadata": metadata,
                "invalid_word_timing_count": sum(
                    1 for item in segment_words
                    if (item.diagnostics or {}).get("invalid_timing")
                ),
            },
        ))
    return result


def bundle_from_candidates(
    segmented: Iterable[CandidateEvidence] = (),
    global_evidence: Iterable[CandidateEvidence] = (),
    review: Iterable[CandidateEvidence] = (),
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> EvidenceBundle:
    return EvidenceBundle(
        segmented=tuple(segmented),
        global_evidence=tuple(global_evidence),
        review=tuple(review),
        diagnostics=copy.deepcopy(dict(diagnostics or {})),
    )


__all__ = [
    "CandidateEvidence",
    "DECISIONS",
    "EVIDENCE_SCHEMA_VERSION",
    "RISK_POLICY_VERSION",
    "DECISION_POLICY_VERSION",
    "EvidenceBundle",
    "EvidenceDecision",
    "EvidenceWord",
    "RISK_LEVELS",
    "TIME_SOURCES",
    "bundle_from_candidates",
    "candidate_from_subtitle_event",
    "candidates_from_global_transcript",
    "candidates_from_segments",
    "evidence_word_from_asr",
]
