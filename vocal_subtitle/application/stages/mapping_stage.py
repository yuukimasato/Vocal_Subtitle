"""Mapping 阶段:review 之后的事件→后处理交接。

当前生命周期里 review→postprocess 的调用发生在各 ASR 路径内部与
导出前尾部;该尾部区域包含尚未落库的并行修改,整体搬移被推迟到
这些修改合入之后。本阶段先以组合形式封装 ``_post_process_events``
交接点,供独立测试与后续接线使用(不改变现有调用时序)。
"""

from __future__ import annotations

from typing import Any

from ..run_context import RunContext
from ..stage_protocol import PipelineStage


class MappingStage:
    """组合现有 postprocess 交接方法,保持事件顺序与时间不变。"""

    name = "mapping"

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def execute(
        self,
        context: RunContext,
        events: Any,
        *,
        audio: Any,
        sample_rate: int,
        vocals_path: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self.pipeline._post_process_events(
            events,
            vocals_path,
            audio,
            sample_rate,
            context.stats,
            **kwargs,
        )


__all__ = ["MappingStage"]
