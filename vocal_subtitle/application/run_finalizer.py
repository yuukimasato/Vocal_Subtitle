"""Task-history finalization for offline pipeline runs."""

from __future__ import annotations

import logging
from typing import Any

from ..application.pipeline_result import PipelineStats

logger = logging.getLogger(__name__)


def finalize_task_state(pipeline: Any, stats: PipelineStats) -> None:
    """Record the final task state without changing pipeline semantics."""
    task_id = getattr(pipeline, "_effective_task_id", None) or stats.task_id
    if not task_id:
        return
    try:
        history = pipeline._get_history()
        if stats.status == "failed":
            history.set_failed(
                task_id,
                reason=stats.fallback_reason or "pipeline execution failed",
                category=stats.error_category or "unrecoverable_failure",
            )
        elif stats.status == "degraded_completed":
            history.set_degraded_completed(
                task_id,
                reason=stats.fallback_reason or "",
                category=stats.fallback_category or "",
            )
        elif stats.status in ("completed",):
            history.set_completed(task_id)
    except Exception as exc:
        logger.warning("Failed to finalize task state: %s", exc)
