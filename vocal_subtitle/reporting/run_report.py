"""统一运行报告生成器

每次离线生产任务完成后生成符合 run-report-v1 schema 的统一运行报告。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..utils.file_hasher import compute_file_hash
from ..utils.session_manager import create_config_snapshot
from .degradation_log import DegradationLogger
from .engine_availability import EngineAvailabilityChecker, EngineAvailabilitySnapshot
from .run_report_schema import (
    DegradationInfo,
    InputInfo,
    OutputInfo,
    PipelinePathInfo,
    QualityInfo,
    RunReport,
    StageInfo,
)

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = "run-report-v1"


def _sanitize_for_yaml(obj):
    """递归转换 tuple → list，避免 YAML 输出 !!python/tuple 标签。"""
    if isinstance(obj, tuple):
        return [_sanitize_for_yaml(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_yaml(item) for item in obj]
    return obj


class RunReportBuilder:
    """构建统一运行报告并持久化到 cache/reports/{run_id}/。"""

    def __init__(
        self,
        run_id: str,
        task_id: str,
        *,
        reports_root: Optional[Path] = None,
    ):
        self.run_id = run_id
        self.task_id = task_id
        self._reports_root = reports_root or (
            Path(__file__).parent.parent.parent / "cache" / "reports"
        )
        self._report_dir = self._reports_root / run_id
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._degradation_logger = DegradationLogger(self._report_dir)

        # 可在 build 之前逐步填充
        self.input_info = InputInfo()
        self.engine_availability: dict = {}
        self.engine_status: dict = {}
        self.pipeline_path = PipelinePathInfo()
        self.stages: dict[str, StageInfo] = {}
        self.output_info = OutputInfo()
        self.quality_info = QualityInfo()
        self.degradation = DegradationInfo()
        self.capability_maturity: dict = {}
        self.noise_shadow: dict = {}
        self.feedback_profile: dict = {}
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.stage_timings: dict[str, float] = {}

    @property
    def degradation_logger(self):
        """Expose the internal DegradationLogger for real-time stage logging."""
        return self._degradation_logger

    # ---- 便捷设置 ----

    def set_input(self, path: Path, duration: float, sample_rate: int = 0, channels: int = 0) -> None:
        suffix = path.suffix.lower().lstrip(".")
        try:
            file_size = path.stat().st_size
        except OSError:
            file_size = 0
        self.input_info = InputInfo(
            path=str(path),
            file_hash=compute_file_hash(path),
            file_size_bytes=file_size,
            duration_seconds=round(duration, 2),
            sample_rate=sample_rate,
            channels=channels,
            format=suffix,
        )

    def set_engine_snapshot(self, snapshot: EngineAvailabilitySnapshot) -> None:
        self.engine_availability = {
            engine: entry.to_dict()
            for engine, entry in snapshot.entries.items()
        }
        self.engine_status = {
            engine: {
                **entry,
                "lifecycle": entry.get("lifecycle", "unknown"),
                "enabled": entry.get("enabled", False),
                "selected": entry.get("selected", False),
                "available": (
                    entry.get("available")
                    if entry.get("available") is not None
                    else str(entry.get("status", "")).startswith("ready")
                ),
                "windows_processed": entry.get("windows_processed", 0),
                "windows_failed": entry.get("windows_failed", 0),
            }
            for engine, entry in self.engine_availability.items()
        }
        # 保存 host 信息以便 persist 时写入 engine_availability.json
        if snapshot.host_info:
            self._engine_snapshot_host = dict(snapshot.host_info)

    def set_engine_status(self, entries: dict[str, Any]) -> None:
        """Merge runtime execution facts into the availability snapshot."""
        aliases = {
            "faster-whisper": "asr_faster_whisper",
            "funasr": "asr_funasr",
            "qwen": "asr_qwen",
            "qwen-asr": "asr_qwen",
            "whisper-cpp": "asr_whisper_cpp",
            "whisper.cpp": "asr_whisper_cpp",
            "context_reasr": "review_context_reasr",
            "global": "review_global_evidence",
        }
        for engine, value in (entries or {}).items():
            if not isinstance(value, dict):
                continue
            canonical_engine = aliases.get(str(engine), str(engine))
            self.engine_status[canonical_engine] = {
                **self.engine_status.get(canonical_engine, {}),
                **value,
            }

    def set_pipeline_path(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.pipeline_path, k):
                setattr(self.pipeline_path, k, v)

    def set_stage(
        self, name: str, status: str = "completed", duration: float = 0.0,
        engine: str = "", model: str = "", **extra,
    ) -> None:
        self.stages[name] = StageInfo(
            status=status, duration_seconds=duration,
            engine=engine, model=model, extra=extra,
        )

    def add_warning(self, stage: str, message: str) -> None:
        self.warnings.append({"stage": stage, "message": message})

    def add_error(self, stage: str, message: str, category: str = "") -> None:
        self.errors.append({"stage": stage, "message": message, "category": category})

    def record_degradation(
        self, stage: str, from_path: str, to_path: str, reason: str, category: str = ""
    ) -> None:
        self._degradation_logger.record(
            stage=stage, from_path=from_path, to_path=to_path,
            reason=reason, category=category,
        )

    # ---- 配置快照 ----

    def build_config_snapshot(self, config, profile: str = "default", overrides: dict | None = None) -> dict:
        return create_config_snapshot(config, profile=profile, overrides=overrides).to_dict()

    # ---- 构建 ----

    def build(self, config_snapshot: dict | None = None) -> RunReport:
        """聚合所有信息生成 RunReport。"""
        report = RunReport(
            run_id=self.run_id,
            task_id=self.task_id,
            created_at=self._created_at,
            input=self.input_info,
            config_snapshot=config_snapshot or {},
            engine_availability=self.engine_availability,
            engine_status=self.engine_status,
            pipeline_path=self.pipeline_path,
            stages=self.stages,
            output=self.output_info,
            quality=self.quality_info,
            degradation=self.degradation,
            capability_maturity=self.capability_maturity,
            noise_shadow=self.noise_shadow,
            feedback_profile=self.feedback_profile,
            errors=self.errors,
            warnings=self.warnings,
            timing={
                "total_wall_seconds": round(sum(self.stage_timings.values()), 2),
                "stage_timings": self.stage_timings,
            },
        )
        return report

    def persist(self, report: RunReport, *, config: Any = None) -> Path:
        """将报告和附件持久化到 cache/reports/{run_id}/。"""
        self._report_dir.mkdir(parents=True, exist_ok=True)

        # 主报告
        report_path = self._report_dir / "run_report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Run report written: %s", report_path)

        # 配置快照
        if config is not None:
            import yaml

            try:
                from dataclasses import asdict
                config_dict = asdict(config)
            except Exception:
                config_dict = {}
            snapshot_path = self._report_dir / "config_snapshot.yaml"
            snapshot_path.write_text(
                yaml.dump(
                    _sanitize_for_yaml(config_dict),
                    allow_unicode=True,
                    default_flow_style=False,
                ),
                encoding="utf-8",
            )

        # 引擎可用性快照
        if self.engine_availability:
            ea_path = self._report_dir / "engine_availability.json"
            ea_data = {
                "run_id": self.run_id,
                "timestamp": self._created_at,
                "engines": self.engine_availability,
            }
            if hasattr(self, "_engine_snapshot_host") and self._engine_snapshot_host:
                ea_data["host"] = self._engine_snapshot_host
            ea_path.write_text(
                json.dumps(ea_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # 物理覆盖审计
        if self.quality_info.coverage_audit:
            cov_path = self._report_dir / "coverage_audit.json"
            cov_path.write_text(
                json.dumps(self.quality_info.coverage_audit, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # 阶段耗时
        if self.stage_timings:
            timing_path = self._report_dir / "stage_timings.json"
            timing_path.write_text(
                json.dumps(self.stage_timings, indent=2),
                encoding="utf-8",
            )

        # 降级日志已由 DegradationLogger 写入
        self._degradation_logger.flush()

        return report_path

    @property
    def report_dir(self) -> Path:
        return self._report_dir
