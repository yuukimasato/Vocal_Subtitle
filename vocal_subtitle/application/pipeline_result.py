"""Serializable results produced by the subtitle pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PipelineStats:
    """Pipeline execution statistics and quality diagnostics."""

    input_path: Path
    duration_seconds: float
    stage_timings: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    segment_count: int = 0
    subtitle_count: int = 0
    speaker_count: int = 0
    diarization_silhouette: Optional[float] = None
    diagnostic_report: Optional[Dict] = None
    quality_diagnostics: Dict[str, Any] = field(default_factory=dict)
    quality_status: str = "pass"
    requested_engine: str = ""
    selected_engine: str = ""
    final_engine: str = ""
    detected_language: str = "unknown"
    language_probability: float = 0.0
    asr_route_version: str = ""
    quality_gate_version: str = ""

    asr_path: str = ""
    global_attempted: bool = False
    fallback_category: str = ""
    fallback_reason: str = ""
    global_diagnostics: Dict = field(default_factory=dict)

    raw_diarization_speaker_count: int = 0
    canonical_speaker_count: int = 0
    speaker_merge_map: Dict = field(default_factory=dict)
    canonicalization_status: str = ""

    diarization_backend: str = ""
    diarization_status: str = ""
    mixed_event_count: int = 0
    atomic_span_count: int = 0
    local_speaker_split_count: int = 0
    speaker_conflict_count: int = 0
    unknown_speaker_count: int = 0

    def to_dict(self) -> dict:
        result = {
            "input_path": str(self.input_path),
            "duration_seconds": self.duration_seconds,
            "stage_timings": self.stage_timings,
            "total_time": self.total_time,
            "segment_count": self.segment_count,
            "subtitle_count": self.subtitle_count,
            "asr_path": self.asr_path,
            "global_attempted": self.global_attempted,
            "fallback_category": self.fallback_category,
            "fallback_reason": self.fallback_reason,
            "global_diagnostics": self.global_diagnostics,
            "raw_diarization_speaker_count": self.raw_diarization_speaker_count,
            "canonical_speaker_count": self.canonical_speaker_count,
            "speaker_merge_map": self.speaker_merge_map,
            "canonicalization_status": self.canonicalization_status,
            "diarization_backend": self.diarization_backend,
            "diarization_status": self.diarization_status,
            "mixed_event_count": self.mixed_event_count,
            "atomic_span_count": self.atomic_span_count,
            "local_speaker_split_count": self.local_speaker_split_count,
            "speaker_conflict_count": self.speaker_conflict_count,
            "unknown_speaker_count": self.unknown_speaker_count,
            "quality_diagnostics": self.quality_diagnostics,
            "quality_status": self.quality_status,
            "requested_engine": self.requested_engine,
            "selected_engine": self.selected_engine,
            "final_engine": self.final_engine,
            "detected_language": self.detected_language,
            "language_probability": self.language_probability,
            "asr_route_version": self.asr_route_version,
            "quality_gate_version": self.quality_gate_version,
            "hallucination_filter_version": getattr(self, "hallucination_filter_version", ""),
            "hallucination_dropped_count": getattr(self, "hallucination_dropped_count", 0),
        }
        if self.speaker_count:
            result["speaker_count"] = self.speaker_count
        if self.diarization_silhouette is not None:
            result["diarization_silhouette"] = self.diarization_silhouette
        if self.diagnostic_report:
            result["diagnostic_report"] = self.diagnostic_report
        return result

    @classmethod
    def from_dict(
        cls, input_path: Path, payload: dict, duration_seconds: float = 0.0
    ) -> "PipelineStats":
        stats = cls(input_path=input_path, duration_seconds=duration_seconds)
        stats.total_time = payload.get("total_time", 0.0)
        stats.segment_count = payload.get("segment_count", 0)
        stats.subtitle_count = payload.get("subtitle_count", 0)
        stats.asr_path = payload.get("asr_path", "")
        stats.global_attempted = payload.get("global_attempted", False)
        stats.fallback_category = payload.get("fallback_category", "")
        stats.fallback_reason = payload.get("fallback_reason", "")
        stats.global_diagnostics = payload.get("global_diagnostics", {})
        stats.raw_diarization_speaker_count = payload.get("raw_diarization_speaker_count", 0)
        stats.canonical_speaker_count = payload.get("canonical_speaker_count", 0)
        stats.speaker_merge_map = payload.get("speaker_merge_map", {})
        stats.canonicalization_status = payload.get("canonicalization_status", "")
        stats.diarization_backend = payload.get("diarization_backend", "")
        stats.diarization_status = payload.get("diarization_status", "")
        stats.mixed_event_count = payload.get("mixed_event_count", 0)
        stats.atomic_span_count = payload.get("atomic_span_count", 0)
        stats.local_speaker_split_count = payload.get("local_speaker_split_count", 0)
        stats.speaker_conflict_count = payload.get("speaker_conflict_count", 0)
        stats.unknown_speaker_count = payload.get("unknown_speaker_count", 0)
        stats.speaker_count = payload.get("speaker_count", 0)
        stats.diarization_silhouette = payload.get("diarization_silhouette")
        stats.diagnostic_report = payload.get("diagnostic_report")
        stats.quality_diagnostics = payload.get("quality_diagnostics", {})
        stats.quality_status = payload.get("quality_status", "pass")
        stats.requested_engine = payload.get("requested_engine", "")
        stats.selected_engine = payload.get("selected_engine", "")
        stats.final_engine = payload.get("final_engine", "")
        stats.detected_language = payload.get("detected_language", "unknown")
        stats.language_probability = payload.get("language_probability", 0.0)
        stats.asr_route_version = payload.get("asr_route_version", "")
        stats.quality_gate_version = payload.get("quality_gate_version", "")
        if "hallucination_filter_version" in payload:
            stats.hallucination_filter_version = payload["hallucination_filter_version"]
        if "hallucination_dropped_count" in payload:
            stats.hallucination_dropped_count = payload["hallucination_dropped_count"]
        return stats
