"""统一运行报告 Schema 数据类 (run-report-v1)

对应 RUN_REPORT_SCHEMA.md 定义的完整 JSON Schema。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InputInfo:
    path: str = ""
    file_hash: str = ""
    file_size_bytes: int = 0
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    format: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "file_hash": self.file_hash,
            "file_size_bytes": self.file_size_bytes,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format,
        }


@dataclass
class EngineStatusEntry:
    engine: str = ""
    model: str = ""
    status: str = "unavailable"  # unavailable|model_missing|ready_shadow|ready_review|ready_default
    device: str = ""
    model_path: str = ""
    model_hash: str = ""
    reason: str = ""
    lifecycle: str = "unknown"
    enabled: bool | None = None
    selected: bool = False
    available: bool | None = None
    windows_processed: int = 0
    windows_failed: int = 0

    def to_dict(self) -> dict:
        result: dict = {"engine": self.engine, "status": self.status}
        result.update({
            "lifecycle": self.lifecycle,
            "enabled": self.enabled,
            "selected": self.selected,
            "available": self.available,
            "windows_processed": self.windows_processed,
            "windows_failed": self.windows_failed,
        })
        if self.model:
            result["model"] = self.model
        if self.device:
            result["device"] = self.device
        if self.model_path:
            result["model_path"] = self.model_path
        if self.model_hash:
            result["model_hash"] = self.model_hash
        if self.reason:
            result["reason"] = self.reason
        return result


@dataclass
class PipelinePathInfo:
    mode: str = "offline"
    production_path: str = "quality_first"
    route_version: str = ""
    quality_gate_version: str = ""
    review_policy_version: str = ""
    cover_policy: str = ""
    engine_policy: str = ""
    risk_policy_version: str = ""
    decision_policy_version: str = ""
    evidence_schema_version: str = ""
    golden_quality_gate_version: str = ""

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "production_path": self.production_path,
            "route_version": self.route_version,
            "quality_gate_version": self.quality_gate_version,
            "review_policy_version": self.review_policy_version,
            "cover_policy": self.cover_policy,
            "engine_policy": self.engine_policy,
            "risk_policy_version": self.risk_policy_version,
            "decision_policy_version": self.decision_policy_version,
            "evidence_schema_version": self.evidence_schema_version,
            "golden_quality_gate_version": self.golden_quality_gate_version,
        }


@dataclass
class StageInfo:
    status: str = "completed"  # completed|degraded|skipped|failed|unavailable
    duration_seconds: float = 0.0
    engine: str = ""
    model: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result: dict = {"status": self.status}
        if self.duration_seconds:
            result["duration_seconds"] = self.duration_seconds
        if self.engine:
            result["engine"] = self.engine
        if self.model:
            result["model"] = self.model
        result.update(self.extra)
        return result


@dataclass
class QualityInfo:
    status: str = "pass"
    coverage_audit: dict = field(default_factory=dict)
    acoustic_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "diagnostics": {},
            "coverage_audit": self.coverage_audit,
            "acoustic_report": self.acoustic_report,
        }


@dataclass
class DegradationInfo:
    overall_mode: str = "full"
    events: list[dict] = field(default_factory=list)
    fallback_category: str = ""
    fallback_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "overall_mode": self.overall_mode,
            "events": self.events,
            "fallback_category": self.fallback_category,
            "fallback_reason": self.fallback_reason,
        }


@dataclass
class OutputInfo:
    subtitle_count: int = 0
    speaker_count: int = 0
    detected_language: str = "unknown"
    language_probability: float = 0.0
    total_duration_seconds: float = 0.0
    files: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subtitle_count": self.subtitle_count,
            "speaker_count": self.speaker_count,
            "detected_language": self.detected_language,
            "language_probability": self.language_probability,
            "total_duration_seconds": self.total_duration_seconds,
            "files": self.files,
        }


@dataclass
class RunReport:
    """run-report-v1 顶层结构"""

    run_id: str = ""
    task_id: str = ""
    created_at: str = ""

    input: InputInfo = field(default_factory=InputInfo)
    config_snapshot: dict = field(default_factory=dict)
    engine_availability: dict = field(default_factory=dict)
    engine_status: dict = field(default_factory=dict)
    pipeline_path: PipelinePathInfo = field(default_factory=PipelinePathInfo)
    stages: dict[str, StageInfo] = field(default_factory=dict)
    output: OutputInfo = field(default_factory=OutputInfo)
    quality: QualityInfo = field(default_factory=QualityInfo)
    degradation: DegradationInfo = field(default_factory=DegradationInfo)
    capability_maturity: dict = field(default_factory=dict)
    noise_shadow: dict = field(default_factory=dict)
    feedback_profile: dict = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    timing: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "$schema": "run-report-v1",
            "run_id": self.run_id,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "input": self.input.to_dict(),
            "config_snapshot": self.config_snapshot,
            "engine_availability": self.engine_availability,
            "engine_status": self.engine_status,
            "pipeline_path": self.pipeline_path.to_dict(),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "output": self.output.to_dict(),
            "quality": self.quality.to_dict(),
            "degradation": self.degradation.to_dict(),
            "capability_maturity": self.capability_maturity,
            "noise_shadow": self.noise_shadow,
            "feedback_profile": self.feedback_profile,
            "errors": self.errors,
            "warnings": self.warnings,
            "timing": self.timing,
        }
