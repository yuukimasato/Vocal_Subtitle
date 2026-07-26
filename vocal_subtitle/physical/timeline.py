"""Physical timeline domain objects.

The objects in this module use absolute seconds and have no dependency on
audio or model runtimes.  They are deliberately small so they can be used by
adapters and cache loaders without importing the main pipeline.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


SCHEMA_VERSION = "physical-timeline-v1"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _range(start: Any, end: Any, *, duration: Optional[float] = None) -> Tuple[float, float]:
    start_value = _finite(start, "start")
    end_value = _finite(end, "end")
    if start_value < 0 or end_value <= start_value:
        raise ValueError("time range must satisfy 0 <= start < end")
    if duration is not None and end_value > duration:
        raise ValueError("time range exceeds timeline duration")
    return start_value, end_value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _metadata(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    return copy.deepcopy(dict(value))


def _confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    result = _finite(value, "confidence")
    if not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return result


@dataclass(frozen=True)
class PhysicalClip:
    id: str
    start: float
    end: float
    source: str = "input"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "id"))
        start, end = _range(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "source", _text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class SpeechEvidenceSpan:
    id: str
    start: float
    end: float
    source: str
    confidence: Optional[float] = None
    physical_clip_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "id"))
        start, end = _range(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "source", _text(self.source, "source"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if self.physical_clip_id is not None:
            object.__setattr__(
                self,
                "physical_clip_id",
                _text(self.physical_clip_id, "physical_clip_id"),
            )
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class ContextWindow:
    id: str
    start: float
    end: float
    physical_clip_id: str
    left_context: float
    right_context: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "id"))
        start, end = _range(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(
            self,
            "physical_clip_id",
            _text(self.physical_clip_id, "physical_clip_id"),
        )
        left = _finite(self.left_context, "left_context")
        right = _finite(self.right_context, "right_context")
        if left < 0 or right < 0:
            raise ValueError("context values must be non-negative")
        object.__setattr__(self, "left_context", left)
        object.__setattr__(self, "right_context", right)
        object.__setattr__(self, "metadata", _metadata(self.metadata))


class PhysicalTimeline:
    """Validated collection of clips, evidence and context windows."""

    def __init__(
        self,
        duration: float,
        *,
        physical_clips: Optional[List[PhysicalClip]] = None,
        speech_evidence_spans: Optional[List[SpeechEvidenceSpan]] = None,
        context_windows: Optional[List[ContextWindow]] = None,
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.duration = _finite(duration, "duration")
        if self.duration <= 0:
            raise ValueError("duration must be greater than zero")
        self.physical_clips = list(physical_clips or [])
        self.speech_evidence_spans = list(speech_evidence_spans or [])
        self.context_windows = list(context_windows or [])
        self.diagnostics = copy.deepcopy(dict(diagnostics or {}))
        self._sort_and_validate()

    @classmethod
    def from_duration(cls, duration: float) -> "PhysicalTimeline":
        timeline = cls(duration)
        timeline.add_clip(0.0, timeline.duration, source="input", clip_id="clip-000001")
        return timeline

    def add_clip(
        self,
        start: float,
        end: float,
        source: str = "input",
        clip_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PhysicalClip:
        start_value, end_value = _range(start, end, duration=self.duration)
        selected_id = clip_id or self._next_clip_id()
        clip = PhysicalClip(
            id=selected_id,
            start=start_value,
            end=end_value,
            source=source,
            metadata=dict(metadata or {}),
        )
        if any(existing.id == clip.id for existing in self.physical_clips):
            raise ValueError(f"duplicate physical clip id: {clip.id}")
        for existing in self.physical_clips:
            if start_value < existing.end and end_value > existing.start:
                raise ValueError("physical clips must not overlap")
        self.physical_clips.append(clip)
        self.physical_clips.sort(key=lambda item: (item.start, item.end, item.id))
        return clip

    def add_evidence(
        self,
        start: float,
        end: float,
        source: str,
        confidence: Optional[float] = None,
        physical_clip_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        evidence_id: Optional[str] = None,
    ) -> SpeechEvidenceSpan:
        raw_start, raw_end = _range(start, end)
        clip = self._resolve_clip(raw_start, raw_end, physical_clip_id)
        clipped_start = max(raw_start, clip.start)
        clipped_end = min(raw_end, clip.end)
        if clipped_end <= clipped_start:
            raise ValueError("evidence is outside its physical clip")
        evidence_metadata = dict(metadata or {})
        if clipped_start != raw_start or clipped_end != raw_end:
            evidence_metadata.update({
                "clipped": True,
                "original_start": raw_start,
                "original_end": raw_end,
            })
        selected_id = evidence_id or f"evidence-{len(self.speech_evidence_spans) + 1:06d}"
        evidence = SpeechEvidenceSpan(
            id=selected_id,
            start=clipped_start,
            end=clipped_end,
            source=source,
            confidence=confidence,
            physical_clip_id=clip.id,
            metadata=evidence_metadata,
        )
        if any(item.id == evidence.id for item in self.speech_evidence_spans):
            raise ValueError(f"duplicate evidence id: {evidence.id}")
        self.speech_evidence_spans.append(evidence)
        self.speech_evidence_spans.sort(key=lambda item: (item.start, item.end, item.id))
        return evidence

    def add_context_window(
        self,
        physical_clip_id: str,
        left_context: float,
        right_context: float,
        *,
        window_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ContextWindow:
        clip = self._clip_by_id(physical_clip_id)
        left = _finite(left_context, "left_context")
        right = _finite(right_context, "right_context")
        if left < 0 or right < 0:
            raise ValueError("context values must be non-negative")
        start = max(0.0, clip.start - left)
        end = min(self.duration, clip.end + right)
        if end <= start:
            raise ValueError("context window must have positive duration")
        selected_id = window_id or f"ctx:{clip.id}:l{round(left * 1000):04d}:r{round(right * 1000):04d}"
        window = ContextWindow(
            id=selected_id,
            start=start,
            end=end,
            physical_clip_id=clip.id,
            left_context=left,
            right_context=right,
            metadata=dict(metadata or {}),
        )
        if any(item.id == window.id for item in self.context_windows):
            raise ValueError(f"duplicate context window id: {window.id}")
        self.context_windows.append(window)
        self.context_windows.sort(key=lambda item: (item.start, item.end, item.id))
        return window

    def clip_range(
        self,
        start: float,
        end: float,
        clip_id: Optional[str] = None,
    ) -> Optional[Tuple[float, float]]:
        start_value, end_value = _range(start, end)
        matches = [
            clip for clip in self.physical_clips
            if clip.start < end_value and clip.end > start_value
            and (clip_id is None or clip.id == clip_id)
        ]
        if len(matches) != 1:
            return None
        return matches[0].start, matches[0].end

    def intersections(self, start: float, end: float) -> List[SpeechEvidenceSpan]:
        start_value, end_value = _range(start, end)
        return [
            item for item in self.speech_evidence_spans
            if item.start < end_value and item.end > start_value
        ]

    def validate(self) -> List[str]:
        errors: List[str] = []
        try:
            duration = _finite(self.duration, "duration")
            if duration <= 0:
                errors.append("duration must be greater than zero")
        except ValueError as exc:
            errors.append(str(exc))
            duration = None
        clip_ids = set()
        previous_end = -1.0
        for clip in self.physical_clips:
            if clip.id in clip_ids:
                errors.append(f"duplicate physical clip id: {clip.id}")
            clip_ids.add(clip.id)
            try:
                _range(clip.start, clip.end, duration=duration)
            except ValueError as exc:
                errors.append(f"clip {clip.id}: {exc}")
            if clip.start < previous_end:
                errors.append(f"overlapping physical clip: {clip.id}")
            previous_end = max(previous_end, clip.end)
        evidence_ids = set()
        clips_by_id = {clip.id: clip for clip in self.physical_clips}
        for evidence in self.speech_evidence_spans:
            if evidence.id in evidence_ids:
                errors.append(f"duplicate evidence id: {evidence.id}")
            evidence_ids.add(evidence.id)
            try:
                _range(evidence.start, evidence.end, duration=duration)
            except ValueError as exc:
                errors.append(f"evidence {evidence.id}: {exc}")
            if evidence.physical_clip_id not in clip_ids:
                errors.append(f"evidence {evidence.id}: unknown physical clip")
            else:
                owner = clips_by_id[evidence.physical_clip_id]
                if evidence.start < owner.start or evidence.end > owner.end:
                    errors.append(f"evidence {evidence.id}: exceeds physical clip")
        window_ids = set()
        for window in self.context_windows:
            if window.id in window_ids:
                errors.append(f"duplicate context window id: {window.id}")
            window_ids.add(window.id)
            try:
                _range(window.start, window.end, duration=duration)
            except ValueError as exc:
                errors.append(f"context {window.id}: {exc}")
            if window.physical_clip_id not in clip_ids:
                errors.append(f"context {window.id}: unknown physical clip")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        errors = self.validate()
        if errors:
            raise ValueError("invalid physical timeline: " + "; ".join(errors))
        return {
            "schema_version": SCHEMA_VERSION,
            "duration": self.duration,
            "physical_clips": [self._clip_dict(item) for item in self.physical_clips],
            "speech_evidence_spans": [self._evidence_dict(item) for item in self.speech_evidence_spans],
            "context_windows": [self._context_dict(item) for item in self.context_windows],
            "diagnostics": copy.deepcopy(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicalTimeline":
        if not isinstance(payload, Mapping):
            raise ValueError("physical timeline payload must be a mapping")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported physical timeline schema")
        clips_payload = payload.get("physical_clips", [])
        evidence_payload = payload.get("speech_evidence_spans", [])
        context_payload = payload.get("context_windows", [])
        if not all(isinstance(value, list) for value in (clips_payload, evidence_payload, context_payload)):
            raise ValueError("timeline collections must be lists")
        timeline = cls(payload.get("duration"))
        for item in clips_payload:
            if not isinstance(item, Mapping):
                raise ValueError("physical clip must be a mapping")
            timeline.add_clip(
                item.get("start"), item.get("end"), item.get("source", "input"),
                item.get("id"), item.get("metadata"),
            )
        for item in evidence_payload:
            if not isinstance(item, Mapping):
                raise ValueError("evidence span must be a mapping")
            timeline.add_evidence(
                item.get("start"), item.get("end"), item.get("source"),
                item.get("confidence"), item.get("physical_clip_id"),
                item.get("metadata"), item.get("id"),
            )
        for item in context_payload:
            if not isinstance(item, Mapping):
                raise ValueError("context window must be a mapping")
            window = timeline.add_context_window(
                item.get("physical_clip_id"), item.get("left_context"),
                item.get("right_context"), window_id=item.get("id"),
                metadata=item.get("metadata"),
            )
            if (window.start, window.end) != (item.get("start"), item.get("end")):
                raise ValueError("context window range does not match its context values")
        timeline.diagnostics = _metadata(payload.get("diagnostics", {}))
        return timeline

    def _resolve_clip(self, start: float, end: float, clip_id: Optional[str]) -> PhysicalClip:
        if clip_id is not None:
            return self._clip_by_id(clip_id)
        matches = [clip for clip in self.physical_clips if clip.start < end and clip.end > start]
        if len(matches) != 1:
            raise ValueError("evidence must match exactly one physical clip")
        return matches[0]

    def _clip_by_id(self, clip_id: str) -> PhysicalClip:
        selected = [clip for clip in self.physical_clips if clip.id == clip_id]
        if len(selected) != 1:
            raise ValueError(f"unknown physical clip: {clip_id}")
        return selected[0]

    def _next_clip_id(self) -> str:
        index = len(self.physical_clips) + 1
        existing = {clip.id for clip in self.physical_clips}
        while f"clip-{index:06d}" in existing:
            index += 1
        return f"clip-{index:06d}"

    def _sort_and_validate(self) -> None:
        self.physical_clips.sort(key=lambda item: (item.start, item.end, item.id))
        self.speech_evidence_spans.sort(key=lambda item: (item.start, item.end, item.id))
        self.context_windows.sort(key=lambda item: (item.start, item.end, item.id))
        errors = self.validate()
        if errors:
            raise ValueError("invalid physical timeline: " + "; ".join(errors))

    @staticmethod
    def _clip_dict(item: PhysicalClip) -> Dict[str, Any]:
        return {"id": item.id, "start": item.start, "end": item.end, "source": item.source, "metadata": copy.deepcopy(item.metadata)}

    @staticmethod
    def _evidence_dict(item: SpeechEvidenceSpan) -> Dict[str, Any]:
        return {"id": item.id, "start": item.start, "end": item.end, "source": item.source, "confidence": item.confidence, "physical_clip_id": item.physical_clip_id, "metadata": copy.deepcopy(item.metadata)}

    @staticmethod
    def _context_dict(item: ContextWindow) -> Dict[str, Any]:
        return {"id": item.id, "start": item.start, "end": item.end, "physical_clip_id": item.physical_clip_id, "left_context": item.left_context, "right_context": item.right_context, "metadata": copy.deepcopy(item.metadata)}
