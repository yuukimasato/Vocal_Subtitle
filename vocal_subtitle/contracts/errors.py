"""结构化管线错误(2026-09-15 重构计划 Task 6)。

领域异常按类别建模并携带可序列化状态;阶段边界把非领域异常翻译为
对应类别(保留原异常为 ``__cause__``,不丢失 traceback)。
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PipelineError(Exception):
    """所有管线领域异常的基类。"""

    category = "pipeline_error"
    recoverable = False

    def __init__(
        self,
        message: str,
        *,
        category: Optional[str] = None,
        recoverable: Optional[bool] = None,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        if category is not None:
            self.category = category
        if recoverable is not None:
            self.recoverable = recoverable
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "message": str(self),
            "detail": self.detail,
            "recoverable": self.recoverable,
        }


class DependencyUnavailableError(PipelineError):
    """可选依赖(模型/运行时)不可用;可降级。"""

    category = "dependency_unavailable"
    recoverable = True


class AudioDecodeError(PipelineError):
    """音频解码/加载失败。"""

    category = "audio_decode"


class ASRExecutionError(PipelineError):
    """ASR 执行失败;可按执行计划回退。"""

    category = "asr_execution"
    recoverable = True


class TimelineViolationError(PipelineError):
    """时间轴不变量被破坏(跨静音/重叠/倒序)。"""

    category = "timeline_violation"


class RecoverableStageError(PipelineError):
    """阶段可恢复失败:记录降级原因后继续。"""

    category = "stage_recoverable"
    recoverable = True


_CATEGORY_HINTS = (
    ("no module named", DependencyUnavailableError),
    ("not installed", DependencyUnavailableError),
    ("is not installed", DependencyUnavailableError),
    ("model missing", DependencyUnavailableError),
    ("invalid data", AudioDecodeError),
    ("decode", AudioDecodeError),
)


def classify_exception(exc: BaseException) -> str:
    """把任意异常映射到稳定类别名(不抛出)。"""
    if isinstance(exc, PipelineError):
        return exc.category
    if isinstance(exc, FileNotFoundError):
        return "input_not_found"
    if isinstance(exc, TimeoutError):
        return "stage_timeout"
    if isinstance(exc, (EOFError, OSError)):
        return "audio_decode"
    if isinstance(exc, ImportError):
        return "dependency_unavailable"
    message = str(exc).casefold()
    for marker, error_type in _CATEGORY_HINTS:
        if marker in message:
            return error_type.category
    return "execution_failed"


def translate_exception(exc: BaseException) -> PipelineError:
    """把非领域异常翻译为 PipelineError(保留原因为 __cause__ 的信息)。"""
    if isinstance(exc, PipelineError):
        return exc
    category = classify_exception(exc)
    recoverable = category in {
        "dependency_unavailable",
        "asr_execution",
        "stage_recoverable",
    }
    return PipelineError(
        str(exc),
        category=category,
        recoverable=recoverable,
    )


__all__ = [
    "PipelineError",
    "DependencyUnavailableError",
    "AudioDecodeError",
    "ASRExecutionError",
    "TimelineViolationError",
    "RecoverableStageError",
    "classify_exception",
    "translate_exception",
]
