"""ASR 阶段:路由决策与 global/骨架/多块/legacy 各识别路径。"""

from __future__ import annotations

from typing import Any

from ..run_context import RunContext


class ASRStage:
    """产出事件与识别诊断(写入 context.state["events"] 等)。"""

    name = "asr"

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def execute(self, context: RunContext) -> RunContext:
        self.pipeline._run_asr_stage(context)
        return context


__all__ = ["ASRStage"]
