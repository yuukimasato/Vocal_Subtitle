"""Serializable results produced by the subtitle pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineStats:
    """Pipeline execution statistics and quality diagnostics."""

    input_path: Path
    duration_seconds: float
    run_id: str = ""
    task_id: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    segment_count: int = 0
    subtitle_count: int = 0
    speaker_count: int = 0
    diarization_silhouette: float | None = None
    diagnostic_report: dict | None = None
    quality_diagnostics: dict[str, Any] = field(default_factory=dict)
    quality_status: str = "pass"
    status: str = "completed"
    error_category: str = ""
    diagnostics_complete: bool = False
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
    global_diagnostics: dict = field(default_factory=dict)
    production_path: str = ""
    review_status: str = ""
    decision_count: int = 0

    raw_diarization_speaker_count: int = 0
    canonical_speaker_count: int = 0
    speaker_merge_map: dict = field(default_factory=dict)
    canonicalization_status: str = ""

    diarization_backend: str = ""
    diarization_status: str = ""
    mixed_event_count: int = 0
    atomic_span_count: int = 0
    local_speaker_split_count: int = 0
    speaker_conflict_count: int = 0
    unknown_speaker_count: int = 0

    hallucination_filter_version: str = ""
    hallucination_dropped_count: int = 0
    hallucination_drop_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "run_id": self.run_id,
            "task_id": self.task_id,
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
            "production_path": self.production_path,
            "review_status": self.review_status,
            "decision_count": self.decision_count,
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
            "status": self.status,
            "error_category": self.error_category,
            "diagnostics_complete": self.diagnostics_complete,
            "requested_engine": self.requested_engine,
            "selected_engine": self.selected_engine,
            "final_engine": self.final_engine,
            "detected_language": self.detected_language,
            "language_probability": self.language_probability,
            "asr_route_version": self.asr_route_version,
            "quality_gate_version": self.quality_gate_version,
            "hallucination_filter_version": self.hallucination_filter_version,
            "hallucination_dropped_count": self.hallucination_dropped_count,
            "hallucination_drop_reasons": self.hallucination_drop_reasons,
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
    ) -> PipelineStats:
        stats = cls(input_path=input_path, duration_seconds=duration_seconds)
        stats.run_id = payload.get("run_id", "")
        stats.task_id = payload.get("task_id", "")
        stats.total_time = payload.get("total_time", 0.0)
        stats.segment_count = payload.get("segment_count", 0)
        stats.subtitle_count = payload.get("subtitle_count", 0)
        stats.asr_path = payload.get("asr_path", "")
        stats.global_attempted = payload.get("global_attempted", False)
        stats.fallback_category = payload.get("fallback_category", "")
        stats.fallback_reason = payload.get("fallback_reason", "")
        stats.global_diagnostics = payload.get("global_diagnostics", {})
        stats.production_path = payload.get("production_path", "")
        stats.review_status = payload.get("review_status", "")
        stats.decision_count = payload.get("decision_count", 0)
        stats.raw_diarization_speaker_count = payload.get(
            "raw_diarization_speaker_count", 0
        )
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
        stats.status = payload.get("status", "completed")
        stats.error_category = payload.get("error_category", "")
        stats.diagnostics_complete = bool(payload.get("diagnostics_complete", False))
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
        if "hallucination_drop_reasons" in payload:
            stats.hallucination_drop_reasons = payload["hallucination_drop_reasons"]
        return stats
