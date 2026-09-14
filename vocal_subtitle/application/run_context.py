"""显式管线运行上下文(2026-09-15 重构计划 Task 2)。

``RunContext`` 承载一次 ``Pipeline.run`` 的输入、中间产物、诊断与取消令牌,
替代散落在 Mixin 实例属性上的隐式状态:

- 输入/输出参数在入口一次性固化;
- 中间产物(audio / events / stats)为可空字段,由各阶段按序填充;
- ``add_diagnostic(stage, payload)`` 是唯一的诊断入口,按阶段分组合并;
- 取消令牌沿用 ``asr.window_execution.CancellationToken``,
  ``cancelled`` / ``raise_if_cancelled`` 向下透传。

上下文实例之间不共享任何可变默认值(隔离性由测试保证)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..asr.window_execution import CancellationToken
from .pipeline_result import PipelineStats

SCHEMA_VERSION = "run-context-v1"


@dataclass
class RunContext:
    """One offline pipeline run's explicit state container."""

    input_path: Path
    output_path: Optional[Path] = None
    output_format: str = "srt"
    skip_separation: bool = False
    task_id: Optional[str] = None
    session_dir: Optional[Path] = None
    feedback_reference: Optional[Path] = None
    overrides: Dict[str, Any] = field(default_factory=dict)

    # 中间产物(按阶段填充;缺失即 None/空)。
    audio: Optional[np.ndarray] = None
    sample_rate: int = 16000
    vocals_path: Optional[Path] = None
    duration_seconds: Optional[float] = None
    events: List[Any] = field(default_factory=list)
    stats: Optional[PipelineStats] = None

    # 协作取消与诊断。
    cancellation_token: Optional[CancellationToken] = None
    diagnostics: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def add_diagnostic(self, stage: str, payload: Dict[str, Any]) -> None:
        """记录一条阶段诊断(按阶段分组合并,保持插入顺序)。"""
        if not stage:
            raise ValueError("diagnostic stage must be a non-empty string")
        self.diagnostics.setdefault(stage, []).append(dict(payload))

    def diagnostics_for(self, stage: str) -> tuple:
        return tuple(self.diagnostics.get(stage, ()))

    def cancelled(self) -> bool:
        return bool(
            self.cancellation_token is not None and self.cancellation_token.cancelled
        )

    def cancel(self, reason: str = "cancelled") -> None:
        if self.cancellation_token is None:
            self.cancellation_token = CancellationToken()
        self.cancellation_token.cancel(reason)

    def raise_if_cancelled(self) -> None:
        if self.cancellation_token is not None:
            self.cancellation_token.raise_if_cancelled()

    def ensure_stats(self) -> PipelineStats:
        """懒创建 stats,保证多次获取拿到同一实例。"""
        if self.stats is None:
            self.stats = PipelineStats(
                input_path=str(self.input_path),
                duration_seconds=float(self.duration_seconds or 0.0),
            )
        return self.stats
