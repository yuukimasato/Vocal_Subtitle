"""阶段质量报告聚合(2026-09-15 重构计划 Task 6)。

把 ``RunContext.diagnostics`` 中的阶段条目折叠为可序列化的
``StageReportAggregator``,聚合结果写入
``PipelineStats.quality_diagnostics["stage_reports"]``(不破坏既有 JSON 结构)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..application.run_context import RunContext

STATUS_OK = "ok"


@dataclass
class StageReport:
    """单阶段摘要。"""

    stage: str
    status: str
    elapsed_seconds: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            **self.detail,
        }


@dataclass
class StageReportAggregator:
    """按阶段收集 StageReport 并输出汇总。"""

    reports: list[StageReport] = field(default_factory=list)

    def add(self, report: StageReport) -> None:
        self.reports.append(report)

    def to_dict(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        total_elapsed = 0.0
        serialized: list[dict[str, Any]] = []
        for report in self.reports:
            status_counts[report.status] = status_counts.get(report.status, 0) + 1
            total_elapsed += report.elapsed_seconds
            serialized.append(report.to_dict())
        return {
            "stage_count": len(self.reports),
            "status_counts": status_counts,
            "total_elapsed_seconds": round(total_elapsed, 6),
            "stages": serialized,
        }


def aggregate_run_diagnostics(context: RunContext) -> dict[str, Any]:
    """把 RunContext 的阶段诊断折叠为 stage_reports 聚合。"""
    aggregator = StageReportAggregator()
    for stage, entries in context.diagnostics.items():
        for entry in entries:
            detail = {
                key: value
                for key, value in dict(entry or {}).items()
                if key not in {"status", "elapsed_seconds"}
            }
            aggregator.add(
                StageReport(
                    stage=stage,
                    status=str((entry or {}).get("status", STATUS_OK)),
                    elapsed_seconds=float(
                        (entry or {}).get("elapsed_seconds", 0.0) or 0.0
                    ),
                    detail=detail,
                )
            )
    return aggregator.to_dict()


__all__ = [
    "StageReport",
    "StageReportAggregator",
    "aggregate_run_diagnostics",
]
