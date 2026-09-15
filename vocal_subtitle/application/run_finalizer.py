"""Task-history finalization for offline pipeline runs."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..application.pipeline_result import PipelineStats

logger = logging.getLogger(__name__)


def build_result_payload(
    *,
    task_id: str,
    stats: PipelineStats,
    events: list[Any],
    input_path: Any = None,
    subtitle_path: Any = None,
    clean_subtitle_path: Any = None,
    llm_subtitle_path: Any = None,
    vocals_path: Any = None,
    accompaniment_path: Any = None,
    from_cache: bool = False,
) -> dict[str, Any]:
    """组装与 WebUI task_result 同构的结果载荷。

    CLI 任务写入历史库后，WebUI 的字幕导出、音频下载、任务详情面板
    均按该载荷回退渲染（见 routes_subtitles / api_serializers）。
    """
    event_dicts: list[dict[str, Any]] = []
    for event in events:
        try:
            event_dicts.append(
                event.to_dict() if hasattr(event, "to_dict") else dict(event)
            )
        except Exception:
            continue
    artifacts = {
        name: str(path)
        for name, path in {
            "input": input_path,
            "subtitle": subtitle_path,
            "subtitle_clean": clean_subtitle_path,
            "subtitle_llm": llm_subtitle_path,
            "vocals": vocals_path,
            "accompaniment": accompaniment_path,
        }.items()
        if path
    }
    return {
        "task_id": task_id,
        "run_id": stats.run_id,
        "status": stats.status,
        "input_path": str(input_path) if input_path else str(stats.input_path),
        "subtitle_path": str(subtitle_path) if subtitle_path else None,
        "clean_subtitle_path": str(clean_subtitle_path)
        if clean_subtitle_path
        else None,
        "llm_subtitle_path": str(llm_subtitle_path) if llm_subtitle_path else None,
        "stats": stats.to_dict(),
        "events": event_dicts,
        "from_cache": from_cache,
        "segment_count": stats.segment_count,
        "subtitle_count": stats.subtitle_count,
        "quality_status": stats.quality_status,
        "error_category": stats.error_category or None,
        "final_engine": stats.final_engine,
        "detected_language": stats.detected_language,
        "vocals_path": str(vocals_path) if vocals_path else None,
        "accompaniment_path": str(accompaniment_path) if accompaniment_path else None,
        "artifacts": artifacts,
        "diagnostics": stats.to_dict(),
    }


def finalize_task_state(
    pipeline: Any,
    stats: PipelineStats,
    *,
    result_payload: dict[str, Any] | None = None,
) -> None:
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
        if result_payload is not None and stats.status != "failed":
            history.update(
                task_id,
                result_json=json.dumps(result_payload, default=str),
                total_duration_seconds=stats.total_time,
            )
    except Exception as exc:
        logger.warning("Failed to finalize task state: %s", exc)
