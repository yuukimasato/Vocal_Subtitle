"""降级日志记录器

以 JSONL 格式记录管道执行过程中的所有降级事件。
对应 ENGINE_LIFECYCLE.md §降级行为矩阵 和 RUN_REPORT_SCHEMA.md。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class DegradationLogger:
    """记录管道降级事件到 JSONL 文件。

    使用示例:
        dlog = DegradationLogger(report_dir)
        dlog.record("asr", "faster-whisper", "whisper_cpp",
                    reason="CUDA OOM", category="resource_exhausted")
        dlog.flush()
    """

    def __init__(self, report_dir: Path):
        self._report_dir = Path(report_dir)
        self._path = self._report_dir / "degradation_log.jsonl"
        self._buffer: list[dict] = []
        self._count = 0

    def record(
        self,
        stage: str,
        from_path: str,
        to_path: str,
        *,
        reason: str = "",
        category: str = "",
    ) -> None:
        """记录一条降级事件。

        Args:
            stage: 发生降级的阶段名称 (separation, vad, asr, review, postprocess, export)
            from_path: 降级前的生产路径
            to_path: 降级后的生产路径
            reason: 降级原因描述
            category: 降级分类 (model_missing, resource_exhausted, engine_timeout, ...)
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "from": from_path,
            "to": to_path,
            "reason": reason,
            "fallback_category": category,
        }
        self._buffer.append(event)
        self._count += 1
        logger.info("Degradation [%s]: %s → %s (%s)", stage, from_path, to_path, reason)

    def flush(self) -> None:
        """将缓冲的降级事件写入磁盘。"""
        if not self._buffer:
            return
        self._report_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            for event in self._buffer:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        logger.info("Degradation log flushed: %d events to %s", self._count, self._path)
        self._buffer.clear()

    @property
    def count(self) -> int:
        return self._count

    @property
    def path(self) -> Path:
        return self._path
