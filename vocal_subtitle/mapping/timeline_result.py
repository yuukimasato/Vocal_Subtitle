"""时间轴只读校验模型(2026-09-15 重构计划 Task 5)。

时间轴写入的单一 Owner 是 ``mapping.final_validator.enforce_non_overlap``;
本模块是唯一验证出口:只读、不修改事件,产出可序列化的
``TimelineValidationReport`` 供诊断与导出阶段消费。

规则:
- 相邻事件按 start 排序后,前一事件 end 超过后续事件 start 即为 overlap;
- 相等边界(end == next.start)合法;
- 标记 ``genuine_overlap`` 的事件对(真实重叠对白)不计违规。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

OVERLAP_TOLERANCE = 1e-9


@dataclass(frozen=True)
class TimelineIssue:
    """一处相邻字幕重叠违规。"""

    kind: str
    first_index: int
    second_index: int
    overlap_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "first_index": self.first_index,
            "second_index": self.second_index,
            "overlap_seconds": round(self.overlap_seconds, 6),
        }


@dataclass(frozen=True)
class TimelineValidationReport:
    """只读校验结果(不携带事件引用,防误改)。"""

    ok: bool
    issues: tuple[TimelineIssue, ...] = ()
    checked_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_count": self.checked_count,
            "issue_count": len(self.issues),
            "issues": [item.to_dict() for item in self.issues],
            **self.diagnostics,
        }


def _genuine(event: Any) -> bool:
    return bool(getattr(event, "genuine_overlap", False))


def validate_timeline(events: Sequence[Any]) -> TimelineValidationReport:
    """按相邻不变量只读校验事件序列(不修改任何事件)。"""
    ordered = sorted(
        enumerate(events),
        key=lambda pair: (
            float(getattr(pair[1], "start", 0.0)),
            float(getattr(pair[1], "end", 0.0)),
        ),
    )
    issues: list[TimelineIssue] = []
    for (first_index, first), (second_index, second) in zip(ordered, ordered[1:]):
        first_end = float(getattr(first, "end", 0.0))
        second_start = float(getattr(second, "start", 0.0))
        overlap = first_end - second_start
        if overlap <= OVERLAP_TOLERANCE:
            continue
        if _genuine(first) or _genuine(second):
            continue
        issues.append(
            TimelineIssue(
                kind="overlap",
                first_index=first_index,
                second_index=second_index,
                overlap_seconds=overlap,
            )
        )
    return TimelineValidationReport(
        ok=not issues,
        issues=tuple(issues),
        checked_count=len(events),
    )


__all__ = [
    "TimelineIssue",
    "TimelineValidationReport",
    "validate_timeline",
]
