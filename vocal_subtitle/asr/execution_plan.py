"""ASR 执行计划(2026-09-15 重构计划 Task 4)。

把 primary/fallback/global/segmented 路径选择表达为显式状态机,
供生命周期与诊断消费;不改变任何现有路由决策逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ROUTE_VERSION = "asr-plan-v1"

FALLBACK_TRIGGER_NONE = "none"
FALLBACK_TRIGGER_FUNASR_QUALITY_GATE = "funasr_quality_gate"
FALLBACK_TRIGGER_GATE_OR_FAILURE = "gate_or_failure"


@dataclass(frozen=True)
class ASRExecutionPlan:
    """一次运行的 ASR 路径与回退策略。"""

    requested_path: str = "segmented"
    requested_engine: str = "auto"
    primary_engine: str = "faster-whisper"
    fallback_engine: str | None = None
    fallback_trigger: str = FALLBACK_TRIGGER_NONE
    # 用户显式要求 global 路径时,主路径失败不允许静默降级。
    hard_fail_on_primary_error: bool = False
    route_version: str = ROUTE_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_path": self.requested_path,
            "requested_engine": self.requested_engine,
            "primary_engine": self.primary_engine,
            "fallback_engine": self.fallback_engine,
            "fallback_trigger": self.fallback_trigger,
            "hard_fail_on_primary_error": self.hard_fail_on_primary_error,
            "route_version": self.route_version,
            **self.extra,
        }


def build_execution_plan(
    config: Any,
    decision: Any = None,
    *,
    requested_path: str = "segmented",
) -> ASRExecutionPlan:
    """根据路由决策与请求路径生成显式执行计划。

    ``decision`` 是 ``_prepare_asr_route`` 产出的路由决策(带
    requested_engine / selected_engine / fallback_engine);缺失时按
    faster-whisper 主引擎的保守计划处理。
    """
    requested_engine = str(getattr(decision, "requested_engine", "auto") or "auto")
    selected_engine = str(
        getattr(decision, "selected_engine", "faster-whisper") or "faster-whisper"
    )
    route_fallback = getattr(decision, "fallback_engine", None)

    fallback_engine: str | None = None
    fallback_trigger = FALLBACK_TRIGGER_NONE
    hard_fail = False

    if requested_path == "global":
        # 用户显式要求 global 路径,失败不应静默降级(生命周期既有语义)。
        hard_fail = True
    elif requested_path == "global_primary":
        # 实验路由:门禁失败或执行失败回退 segmented。
        fallback_engine = "segmented"
        fallback_trigger = FALLBACK_TRIGGER_GATE_OR_FAILURE

    if (
        requested_engine == "auto"
        and selected_engine == "funasr"
        and route_fallback == "faster-whisper"
    ):
        fallback_engine = "faster-whisper"
        fallback_trigger = FALLBACK_TRIGGER_FUNASR_QUALITY_GATE

    return ASRExecutionPlan(
        requested_path=requested_path,
        requested_engine=requested_engine,
        primary_engine=selected_engine,
        fallback_engine=fallback_engine,
        fallback_trigger=fallback_trigger,
        hard_fail_on_primary_error=hard_fail,
        route_version=ROUTE_VERSION,
    )


__all__ = [
    "ASRExecutionPlan",
    "build_execution_plan",
    "ROUTE_VERSION",
]
