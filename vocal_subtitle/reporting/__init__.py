"""运行报告系统 — 统一运行报告生成、引擎可用性检查、降级日志。

对应文档:
  - RUN_REPORT_SCHEMA.md (run-report-v1)
  - TASK_STATE_MACHINE.md (task-state-v1)
  - ENGINE_LIFECYCLE.md (engine-lifecycle-v1)
"""

from .degradation_log import DegradationLogger
from .capability_maturity import build_capability_maturity
from .engine_availability import EngineAvailabilityChecker, EngineAvailabilitySnapshot
from .feedback_profile import inspect_feedback_profile
from .noise_shadow import build_noise_shadow
from .run_report import RunReportBuilder
from .run_report_schema import (
    EngineStatusEntry,
    PipelinePathInfo,
    QualityInfo,
    RunReport,
    StageInfo,
)

__all__ = [
    "RunReportBuilder",
    "RunReport",
    "StageInfo",
    "PipelinePathInfo",
    "QualityInfo",
    "EngineStatusEntry",
    "EngineAvailabilityChecker",
    "EngineAvailabilitySnapshot",
    "DegradationLogger",
    "build_capability_maturity",
    "build_noise_shadow",
    "inspect_feedback_profile",
]
