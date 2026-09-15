"""Preflight 阶段:状态重置、preflight 校验与报告构建器初始化。"""

from __future__ import annotations

from typing import Any

from ..run_context import RunContext


class PreflightStage:
    """执行 preflight;失败时返回需提前返回的结果 dict,否则返回 None。"""

    name = "preflight"

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def execute(self, context: RunContext) -> dict[str, Any] | None:
        return self.pipeline._run_preflight_stage(context)


__all__ = ["PreflightStage"]
