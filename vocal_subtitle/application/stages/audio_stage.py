"""音频阶段:模块预算、人声分离、宏切块、音频加载与 early turns。"""

from __future__ import annotations

from typing import Any

from ..run_context import RunContext


class AudioStage:
    """准备后续 ASR 所需的音频与物理前置状态(写入 context.state)。"""

    name = "audio"

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def execute(self, context: RunContext, *, progress_callback=None) -> RunContext:
        self.pipeline._run_audio_stage(context, progress_callback)
        return context


__all__ = ["AudioStage"]
