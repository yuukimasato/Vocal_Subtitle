"""Global intermediate representations for offline subtitle processing."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..asr.base import TranscriptionSegment
from ..diarization.base import DiarizationResult, SpeakerTurn


SCHEMA_VERSION = "global-ir-v1"


def _number(value: Any, name: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (non_negative and result < 0):
        raise ValueError(f"{name} must be a finite number")
    return result


def _range(start: Any, end: Any, name: str, *, duration: Optional[float] = None) -> tuple[float, float]:
    first = _number(start, f"{name}_start", non_negative=True)
    last = _number(end, f"{name}_end", non_negative=True)
    if last <= first:
        raise ValueError(f"{name} must satisfy 0 <= start < end")
    if duration is not None and last > duration:
        raise ValueError(f"{name} exceeds audio duration")
    return first, last


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_number(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    return _number(value, name)


def _metadata(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    return copy.deepcopy(dict(value))


def _speaker_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("speaker_id must be a non-negative integer or None")
    return value


@dataclass(frozen=True)
class GlobalWord:
    id: str
    text: str
    raw_start: float
    raw_end: float
    confidence: Optional[float] = None
    source_window_id: str = ""
    segment_id: str = ""
    language: Optional[str] = None
    speaker_id: Optional[int] = None
    no_speech_prob: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "id"))
        object.__setattr__(self, "text", _text(self.text, "text"))
        start, end = _range(self.raw_start, self.raw_end, "raw time")
        object.__setattr__(self, "raw_start", start)
        object.__setattr__(self, "raw_end", end)
        object.__setattr__(self, "source_window_id", _text(self.source_window_id, "source_window_id"))
        object.__setattr__(self, "segment_id", _text(self.segment_id, "segment_id"))
        for name in ("confidence", "no_speech_prob", "avg_logprob", "compression_ratio"):
            object.__setattr__(self, name, _optional_number(getattr(self, name), name))
        object.__setattr__(self, "speaker_id", _speaker_id(self.speaker_id))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "raw_start": self.raw_start,
            "raw_end": self.raw_end,
            "confidence": self.confidence,
            "source_window_id": self.source_window_id,
            "segment_id": self.segment_id,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "no_speech_prob": self.no_speech_prob,
            "avg_logprob": self.avg_logprob,
            "compression_ratio": self.compression_ratio,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalWord":
        if not isinstance(payload, Mapping):
            raise ValueError("global word must be a mapping")
        return cls(**dict(payload))


@dataclass(frozen=True)
class GlobalTranscriptSegment:
    id: str
    text: str
    raw_start: float
    raw_end: float
    word_ids: List[str] = field(default_factory=list)
    language: Optional[str] = None
    avg_logprob: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "id"))
        object.__setattr__(self, "text", self.text if isinstance(self.text, str) else str(self.text))
        start, end = _range(self.raw_start, self.raw_end, "segment time")
        object.__setattr__(self, "raw_start", start)
        object.__setattr__(self, "raw_end", end)
        if not isinstance(self.word_ids, list) or any(not isinstance(item, str) or not item.strip() for item in self.word_ids):
            raise ValueError("word_ids must be a list of non-empty strings")
        object.__setattr__(self, "word_ids", list(self.word_ids))
        object.__setattr__(self, "avg_logprob", _optional_number(self.avg_logprob, "avg_logprob"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "raw_start": self.raw_start,
            "raw_end": self.raw_end,
            "word_ids": list(self.word_ids),
            "language": self.language,
            "avg_logprob": self.avg_logprob,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalTranscriptSegment":
        if not isinstance(payload, Mapping):
            raise ValueError("global transcript segment must be a mapping")
        return cls(**dict(payload))


@dataclass
class GlobalTranscript:
    schema_version: str = SCHEMA_VERSION
    audio_duration: Optional[float] = None
    words: List[GlobalWord] = field(default_factory=list)
    segments: List[GlobalTranscriptSegment] = field(default_factory=list)
    backend: str = "unknown"
    status: str = "unknown"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported global IR schema")
        if self.audio_duration is not None:
            object.__setattr__(self, "audio_duration", _number(self.audio_duration, "audio_duration", non_negative=True))
        if not isinstance(self.words, list) or not all(isinstance(item, GlobalWord) for item in self.words):
            raise ValueError("words must be a list of GlobalWord")
        if not isinstance(self.segments, list) or not all(isinstance(item, GlobalTranscriptSegment) for item in self.segments):
            raise ValueError("segments must be a list of GlobalTranscriptSegment")
        self.words.sort(key=lambda item: (item.raw_start, item.raw_end, item.id))
        self.segments.sort(key=lambda item: (item.raw_start, item.raw_end, item.id))
        self.backend = _text(self.backend, "backend")
        self.status = _text(self.status, "status")
        self.diagnostics = _metadata(self.diagnostics)
        self.validate()

    def validate(self) -> List[str]:
        errors: List[str] = []
        word_by_id: Dict[str, GlobalWord] = {}
        for word in self.words:
            if word.id in word_by_id:
                errors.append(f"duplicate global word id: {word.id}")
            word_by_id[word.id] = word
            if self.audio_duration is not None and word.raw_end > self.audio_duration:
                errors.append(f"word {word.id} exceeds audio duration")
        segment_ids = set()
        for segment in self.segments:
            if segment.id in segment_ids:
                errors.append(f"duplicate transcript segment id: {segment.id}")
            segment_ids.add(segment.id)
            if len(segment.word_ids) != len(set(segment.word_ids)):
                errors.append(f"segment {segment.id} repeats a word id")
            referenced = [word_by_id.get(item) for item in segment.word_ids]
            if any(word is None for word in referenced):
                errors.append(f"segment {segment.id} has a dangling word id")
                continue
            words = [word for word in referenced if word is not None]
            ordered = sorted(words, key=lambda item: (item.raw_start, item.raw_end, item.id))
            if [item.id for item in words] != [item.id for item in ordered]:
                errors.append(f"segment {segment.id} word_ids are not time ordered")
            if words and (segment.raw_start > min(item.raw_start for item in words) or segment.raw_end < max(item.raw_end for item in words)):
                errors.append(f"segment {segment.id} does not cover its words")
        if errors:
            raise ValueError("invalid global transcript: " + "; ".join(errors))
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audio_duration": self.audio_duration,
            "words": [item.to_dict() for item in self.words],
            "segments": [item.to_dict() for item in self.segments],
            "backend": self.backend,
            "status": self.status,
            "diagnostics": copy.deepcopy(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalTranscript":
        if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported global IR schema")
        return cls(
            schema_version=payload["schema_version"],
            audio_duration=payload.get("audio_duration"),
            words=[GlobalWord.from_dict(item) for item in payload.get("words", [])],
            segments=[GlobalTranscriptSegment.from_dict(item) for item in payload.get("segments", [])],
            backend=payload.get("backend", "unknown"),
            status=payload.get("status", "unknown"),
            diagnostics=payload.get("diagnostics", {}),
        )


@dataclass
class GlobalSpeakerTimeline:
    schema_version: str = SCHEMA_VERSION
    duration: float = 0.0
    turns: List[SpeakerTurn] = field(default_factory=list)
    exclusive_turns: List[SpeakerTurn] = field(default_factory=list)
    speaker_ids: List[int] = field(default_factory=list)
    backend: str = "unknown"
    status: str = "unknown"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported global IR schema")
        self.duration = _number(self.duration, "duration", non_negative=True)
        if self.duration <= 0:
            raise ValueError("duration must be greater than zero")
        self.turns = self._normalize_turns(self.turns, "turns")
        self.exclusive_turns = self._normalize_turns(self.exclusive_turns, "exclusive_turns")
        self.speaker_ids = sorted({turn.speaker_id for turn in self.turns + self.exclusive_turns})
        self.backend = _text(self.backend, "backend")
        self.status = _text(self.status, "status")
        self.diagnostics = _metadata(self.diagnostics)

    def _normalize_turns(self, turns: Iterable[SpeakerTurn], name: str) -> List[SpeakerTurn]:
        if not isinstance(turns, list):
            turns = list(turns)
        normalized = []
        for turn in turns:
            if not isinstance(turn, SpeakerTurn):
                raise ValueError(f"{name} must contain SpeakerTurn")
            _range(turn.start, turn.end, name, duration=self.duration)
            _speaker_id(turn.speaker_id)
            if turn.confidence is not None:
                _number(turn.confidence, "turn confidence")
            normalized.append(turn)
        return sorted(normalized, key=lambda item: (item.start, item.end, item.speaker_id))

    def to_dict(self) -> Dict[str, Any]:
        def turn_dict(turn: SpeakerTurn) -> Dict[str, Any]:
            return {
                "start": turn.start,
                "end": turn.end,
                "speaker_id": turn.speaker_id,
                "confidence": turn.confidence,
                "overlapped": turn.overlapped,
            }
        return {
            "schema_version": self.schema_version,
            "duration": self.duration,
            "turns": [turn_dict(item) for item in self.turns],
            "exclusive_turns": [turn_dict(item) for item in self.exclusive_turns],
            "speaker_ids": list(self.speaker_ids),
            "backend": self.backend,
            "status": self.status,
            "diagnostics": copy.deepcopy(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalSpeakerTimeline":
        if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported global IR schema")
        def parse_turn(item: Mapping[str, Any]) -> SpeakerTurn:
            if not isinstance(item, Mapping):
                raise ValueError("speaker turn must be a mapping")
            return SpeakerTurn(
                start=item.get("start"), end=item.get("end"),
                speaker_id=item.get("speaker_id"),
                confidence=item.get("confidence"),
                overlapped=bool(item.get("overlapped", False)),
            )
        return cls(
            schema_version=payload["schema_version"],
            duration=payload.get("duration"),
            turns=[parse_turn(item) for item in payload.get("turns", [])],
            exclusive_turns=[parse_turn(item) for item in payload.get("exclusive_turns", [])],
            speaker_ids=[],
            backend=payload.get("backend", "unknown"),
            status=payload.get("status", "unknown"),
            diagnostics=payload.get("diagnostics", {}),
        )


def adapt_transcription_segments(
    segments: Sequence[TranscriptionSegment],
    *,
    source_window_id: str,
    segment_id_prefix: str,
    time_offset: float = 0.0,
    language: Optional[str] = None,
    audio_duration: Optional[float] = None,
) -> GlobalTranscript:
    """Adapt relative-window ASR segments into absolute global IR."""
    offset = _number(time_offset, "time_offset")
    source_window_id = _text(source_window_id, "source_window_id")
    prefix = _text(segment_id_prefix, "segment_id_prefix")
    words: List[GlobalWord] = []
    transcript_segments: List[GlobalTranscriptSegment] = []
    diagnostics: Dict[str, Any] = {"skipped_segments": 0, "skipped_words": 0}
    for index, segment in enumerate(segments):
        if not isinstance(segment, TranscriptionSegment):
            diagnostics["skipped_segments"] += 1
            continue
        segment_id = f"{prefix}:{index:04d}"
        try:
            seg_start, seg_end = _range(segment.start + offset, segment.end + offset, "segment", duration=audio_duration)
        except ValueError:
            diagnostics["skipped_segments"] += 1
            continue
        word_ids: List[str] = []
        for word_index, word in enumerate(segment.words or []):
            try:
                word_start, word_end = _range(word.start + offset, word.end + offset, "word", duration=audio_duration)
                word_id = f"gw:{source_window_id}:{segment_id}:w{word_index:04d}"
                global_word = GlobalWord(
                    id=word_id,
                    text=word.word,
                    raw_start=word_start,
                    raw_end=word_end,
                    confidence=word.confidence,
                    source_window_id=source_window_id,
                    segment_id=segment_id,
                    language=segment.language or language,
                    speaker_id=word.speaker_id,
                    avg_logprob=segment.avg_logprob,
                    no_speech_prob=segment.no_speech_prob,
                    compression_ratio=segment.compression_ratio,
                    metadata={"source": "transcription_segment"},
                )
            except (AttributeError, TypeError, ValueError):
                diagnostics["skipped_words"] += 1
                continue
            words.append(global_word)
            word_ids.append(word_id)
        transcript_segments.append(GlobalTranscriptSegment(
            id=segment_id,
            text=segment.text,
            raw_start=seg_start,
            raw_end=seg_end,
            word_ids=word_ids,
            language=segment.language or language,
            avg_logprob=segment.avg_logprob,
            metadata={
                "source": "transcription_segment",
                **({"speaker_id": segment.speaker_id} if segment.speaker_id is not None else {}),
            },
        ))
    status = "ok" if not diagnostics["skipped_segments"] else "degraded"
    return GlobalTranscript(
        audio_duration=audio_duration,
        words=words,
        segments=transcript_segments,
        backend="legacy-segment-adapter",
        status=status,
        diagnostics=diagnostics,
    )


def adapt_diarization_result(result: DiarizationResult, *, duration: float) -> GlobalSpeakerTimeline:
    if not isinstance(result, DiarizationResult):
        raise ValueError("result must be a DiarizationResult")
    return GlobalSpeakerTimeline(
        schema_version=SCHEMA_VERSION,
        duration=duration,
        turns=list(result.turns),
        exclusive_turns=list(result.exclusive_turns),
        backend=result.backend,
        status=result.status,
        diagnostics=copy.deepcopy(result.diagnostics),
    )
