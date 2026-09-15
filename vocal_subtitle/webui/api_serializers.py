"""Stable WebUI response serializers shared by route groups."""

from __future__ import annotations

import json
from typing import Any


def result_summary(
    result_json: str | None, *, detail: bool = False
) -> dict[str, Any] | None:
    if not result_json:
        return None
    try:
        result = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return None
    stats = result.get("stats") or {}
    if not isinstance(stats, dict):
        stats = {}
    if detail:
        return {
            "contract_version": result.get("contract_version"),
            "task_id": result.get("task_id"),
            "run_id": result.get("run_id") or stats.get("run_id"),
            "subtitle_path": result.get("subtitle_path"),
            "subtitle_count": result.get("subtitle_count", 0),
            "segment_count": result.get("segment_count", 0),
            "from_cache": result.get("from_cache", False),
            "stats": result.get("stats"),
            "artifacts": result.get("artifacts") or {},
            "diagnostics": result.get("diagnostics") or {},
            "input_path": result.get("input_path"),
            "vocals_path": result.get("vocals_path"),
            "accompaniment_path": result.get("accompaniment_path"),
            # 内部学习任务的学习报告（冷重跑完成后挂载，D28）；普通任务为 None
            "learn_report": result.get("learn_report"),
        }
    return {
        "contract_version": result.get("contract_version"),
        "task_id": result.get("task_id"),
        "run_id": result.get("run_id") or stats.get("run_id"),
        "subtitle_count": result.get("subtitle_count", 0),
        "segment_count": result.get("segment_count", 0),
        "from_cache": result.get("from_cache", False),
        "quality_status": stats.get("quality_status"),
        "final_engine": stats.get("final_engine"),
    }


def history_item(task: dict[str, Any]) -> dict[str, Any]:
    summary = result_summary(task.get("result_json")) or {}
    return {
        "id": task["id"],
        "task_id": task["id"],
        "run_id": task.get("run_id") or summary.get("run_id") or "",
        "contract_version": summary.get("contract_version"),
        "input_file_name": task["input_file_name"],
        "input_file_hash": task.get("input_file_hash", ""),
        "input_file_size": task.get("input_file_size", 0),
        "profile": task.get("profile", "default"),
        "status": task["status"],
        # "学习"标记与场景标签（内部学习任务，D28/D27）；普通任务为空串
        "task_type": task.get("task_type") or "",
        "scenario": task.get("scenario") or "",
        "error": task.get("error"),
        "total_duration_seconds": task.get("total_duration_seconds", 0),
        "created_at": task.get("created_at", ""),
        "completed_at": task.get("completed_at"),
        "result_summary": summary or None,
    }


def history_detail(task: dict[str, Any]) -> dict[str, Any]:
    result = result_summary(task.get("result_json"), detail=True)
    events = []
    if task.get("result_json"):
        try:
            events = json.loads(task["result_json"]).get("events", [])
        except (json.JSONDecodeError, TypeError):
            pass
    item = history_item(task)
    item["result_summary"] = result
    item["events"] = events
    return item


__all__ = ["history_detail", "history_item", "result_summary"]
