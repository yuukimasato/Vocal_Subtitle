"""阶段协议(2026-09-15 重构计划 Task 2)。

``PipelineStage`` 是生命周期阶段的统一形状:接收并返回 ``RunContext``,
执行结果(状态/耗时/负载)统一写回 ``context.add_diagnostic``。
``Stage`` 基类提供计时与失败包装,子类只实现 :meth:`Stage.run`。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Protocol, runtime_checkable

from .run_context import RunContext

logger = logging.getLogger(__name__)

STAGE_STATUS_OK = "ok"
STAGE_STATUS_DEGRADED = "degraded"
STAGE_STATUS_FAILED = "failed"
STAGE_STATUS_SKIPPED = "skipped"


@runtime_checkable
class PipelineStage(Protocol):
    """一个生命周期阶段:context in → context out。"""

    name: str

    def execute(self, context: RunContext) -> RunContext:
        ...


@dataclass
class StageResult:
    """单次阶段执行的摘要(同时写入 context 诊断)。"""

    stage: str
    status: str
    elapsed_seconds: float
    payload: Dict[str, Any]

    def as_payload(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            **self.payload,
        }


class Stage:
    """带计时与诊断记录的阶段基类。

    子类实现 :meth:`run`;异常被捕获并记为 failed 后重新抛出,
    保证阶段边界之外的调用方仍能看到失败(诊断先行)。
    """

    name: str = "stage"

    def execute(self, context: RunContext) -> RunContext:
        started = time.perf_counter()
        try:
            result = self.run(context)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            context.add_diagnostic(self.name, {
                "status": STAGE_STATUS_FAILED,
                "elapsed_seconds": round(elapsed, 6),
                "error": str(exc),
            })
            logger.warning("Stage %s failed after %.3fs: %s", self.name, elapsed, exc)
            raise
        elapsed = time.perf_counter() - started
        payload = dict(result or {})
        status = str(payload.pop("status", STAGE_STATUS_OK))
        context.add_diagnostic(self.name, {
            "status": status,
            "elapsed_seconds": round(elapsed, 6),
            **payload,
        })
        return context

    def run(self, context: RunContext) -> Dict[str, Any]:
        """执行阶段本体,返回摘要 payload(可含 'status')。"""
        raise NotImplementedError


__all__ = [
    "PipelineStage",
    "Stage",
    "StageResult",
    "STAGE_STATUS_OK",
    "STAGE_STATUS_DEGRADED",
    "STAGE_STATUS_FAILED",
    "STAGE_STATUS_SKIPPED",
]
