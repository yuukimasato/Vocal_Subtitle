"""Stable WebUI response serializers shared by route groups."""

from __future__ import annotations

import json
from typing import Any, Dict


def result_summary(result_json: str | None, *, detail: bool = False) -> Dict[str, Any] | None:
    if not result_json:
        return None
    try:
        result = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if detail:
        return {
            "subtitle_path": result.get("subtitle_path"),
            "subtitle_count": result.get("subtitle_count", 0),
            "segment_count": result.get("segment_count", 0),
            "from_cache": result.get("from_cache", False),
            "stats": result.get("stats"),
            "vocals_path": result.get("vocals_path"),
            "accompaniment_path": result.get("accompaniment_path"),
        }
    stats = result.get("stats") or {}
    return {
        "subtitle_count": result.get("subtitle_count", 0),
        "segment_count": result.get("segment_count", 0),
        "from_cache": result.get("from_cache", False),
        "quality_status": stats.get("quality_status"),
        "final_engine": stats.get("final_engine"),
    }


def history_item(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": task["id"],
        "input_file_name": task["input_file_name"],
        "input_file_hash": task.get("input_file_hash", ""),
        "input_file_size": task.get("input_file_size", 0),
        "profile": task.get("profile", "default"),
        "status": task["status"],
        "error": task.get("error"),
        "total_duration_seconds": task.get("total_duration_seconds", 0),
        "created_at": task.get("created_at", ""),
        "completed_at": task.get("completed_at"),
        "result_summary": result_summary(task.get("result_json")),
    }


def history_detail(task: Dict[str, Any]) -> Dict[str, Any]:
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
