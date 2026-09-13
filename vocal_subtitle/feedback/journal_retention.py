"""编辑日志 sink 文件保留策略（D30）。

cache/journal_sink/<session_id>.jsonl 的生命周期管理：
  - CLI ingest 成功消费后按策略处理源文件：archive=移入 consumed/ 子目录（可审计）
    或 delete=直接删除；
  - 从未被消费的文件按 TTL（默认 30 天，feedback.journal_sink_ttl_days）兜底清理，
    防止无人 ingest 时无限堆积。

目录约定：sink 顶层是「待消费」文件；已消费归档固定放 sink/consumed/。
TTL 清理只扫 sink 顶层的 *.jsonl，consumed/ 子目录天然不受二次清理。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

CONSUMED_SUBDIR = "consumed"

RETENTION_ARCHIVE = "archive"  # 消费后移入 consumed/（可审计，默认）
RETENTION_DELETE = "delete"    # 消费后直接删除

_VALID_POLICIES = {RETENTION_ARCHIVE, RETENTION_DELETE}


def consumed_dir(sink_dir: Path | str) -> Path:
    """已消费日志的归档目录（sink/consumed/）"""
    return Path(sink_dir) / CONSUMED_SUBDIR


@dataclass
class RetentionResult:
    """单个文件保留策略的执行结果"""

    path: Path
    action: str        # "archived" | "deleted" | "skipped"
    reason: str = ""   # skipped 时的原因：missing=源已不存在（幂等）| outside-sink=不在 sink 顶层
    detail: str = ""   # archived 时的归档目标路径


def apply_retention(path: Path | str, *, policy: str, sink_dir: Path | str) -> RetentionResult:
    """对成功消费的 sink 文件执行保留策略（归档或删除）。

    只处理 sink 顶层文件（consumed/ 内或 sink 外的路径一律跳过，
    避免误动用户的本地 .journal.jsonl 导出文件）；源文件已不存在时幂等跳过。
    """
    path = Path(path)
    sink_dir = Path(sink_dir)
    # 仅认 sink 顶层文件：consumed/ 内的归档不重复处理，sink 外的用户文件不动
    if path.parent.resolve() != sink_dir.resolve():
        logger.info("Journal retention skipped (outside sink): %s", path)
        return RetentionResult(path=path, action="skipped", reason="outside-sink")
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"未知保留策略: {policy!r}（feedback.journal_sink_retention 仅支持 {_VALID_POLICIES}）"
        )
    if not path.exists():
        logger.info("Journal retention skipped (already consumed): %s", path)
        return RetentionResult(path=path, action="skipped", reason="missing")

    if policy == RETENTION_ARCHIVE:
        target = consumed_dir(sink_dir)
        target.mkdir(parents=True, exist_ok=True)
        destination = target / path.name
        # os.replace 原子覆盖：同 session 重新上送后再次 ingest 时覆盖旧归档（归档保留最新全量）
        os.replace(path, destination)
        logger.info("Journal sink file archived: %s -> %s", path, destination)
        return RetentionResult(path=path, action="archived", detail=str(destination))

    path.unlink()
    logger.info("Journal sink file deleted after ingest: %s", path)
    return RetentionResult(path=path, action="deleted")


def cleanup_expired(
    sink_dir: Path | str,
    *,
    ttl_days: int,
    now: Optional[datetime] = None,
) -> List[Path]:
    """TTL 兜底清理：删除 sink 顶层从未被消费且超过 TTL 的 *.jsonl 文件。

    以文件 mtime（sink 追加式落盘，mtime 即最后上送时间）判定超期；
    只扫顶层，consumed/ 归档不受影响；ttl_days <= 0 视为禁用清理。
    返回被删除的文件列表（幂等：重复运行时残留文件已删，返回空）。
    """
    if ttl_days <= 0:
        logger.info("Journal sink TTL cleanup disabled (journal_sink_ttl_days=%d)", ttl_days)
        return []
    sink_dir = Path(sink_dir)
    if not sink_dir.is_dir():
        return []
    current = now or datetime.now(timezone.utc)
    deadline = current - timedelta(days=ttl_days)

    removed: List[Path] = []
    for path in sorted(sink_dir.glob("*.jsonl")):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if mtime > deadline:
                continue
            path.unlink()
        except OSError as exc:
            logger.warning("Journal sink TTL cleanup failed for %s: %s", path, exc)
            continue
        removed.append(path)
        logger.info("Journal sink file expired (TTL %d days): %s", ttl_days, path)
    return removed


__all__ = [
    "CONSUMED_SUBDIR",
    "RETENTION_ARCHIVE",
    "RETENTION_DELETE",
    "RetentionResult",
    "apply_retention",
    "cleanup_expired",
    "consumed_dir",
]
