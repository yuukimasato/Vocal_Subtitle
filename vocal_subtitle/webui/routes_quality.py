"""Read-only quality report projection for the WebUI workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from .runtime_state import state

router = APIRouter()


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _result_payload(task: Dict[str, Any]) -> Dict[str, Any]:
    raw = task.get("result")
    if isinstance(raw, dict):
        return raw
    raw = task.get("result_json")
    if not raw:
        return {}
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return result if isinstance(result, dict) else {}


def _stats_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    stats = result.get("stats") or {}
    return stats if isinstance(stats, dict) else {}


def _safe_run_id(task: Dict[str, Any], result: Dict[str, Any], stats: Dict[str, Any]) -> str:
    run_id = str(
        stats.get("run_id")
        or result.get("run_id")
        or task.get("run_id")
        or ""
    )
    return run_id if run_id and Path(run_id).name == run_id else ""


def _report_path(run_id: str) -> Path:
    return Path(__file__).parent.parent.parent / "cache" / "reports" / run_id / "run_report.json"


def _history_summary(task: Dict[str, Any], result: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    run_id = _safe_run_id(task, result, stats)
    stage_timings = stats.get("stage_timings") or {}
    diagnostics = stats.get("quality_diagnostics") or stats.get("diagnostics") or {}
    report = {
        "$schema": "run-report-v1",
        "run_id": run_id,
        "task_id": task.get("id", ""),
        "status": task.get("status", stats.get("status", "")),
        "pipeline_path": {
            "mode": "offline",
            "production_path": stats.get("production_path", ""),
            "route_version": stats.get("asr_route_version", ""),
            "quality_gate_version": stats.get("quality_gate_version", ""),
        },
        "stages": {
            name: {"duration_seconds": value, "status": "completed"}
            for name, value in stage_timings.items()
        },
        "quality": {
            "status": stats.get("quality_status", ""),
            "diagnostics": diagnostics,
            "acoustic_report": stats.get("diagnostic_report") or {},
        },
        "engine_availability": {},
        "degradation": {
            "category": stats.get("fallback_category", ""),
            "reason": stats.get("fallback_reason", ""),
        },
        "errors": ([
            {
                "stage": "pipeline",
                "message": task.get("error", ""),
                "category": task.get("error_category", ""),
            }
        ] if task.get("error") else []),
        "warnings": [],
        "output": {
            "subtitle_path": result.get("subtitle_path"),
            "subtitle_count": result.get("subtitle_count", stats.get("subtitle_count", 0)),
        },
        "diagnostics": {
            "asr": stats.get("global_diagnostics", {}),
            "review_status": stats.get("review_status", ""),
            "decision_count": stats.get("decision_count", 0),
        },
    }
    return report


@router.get("/quality/report/{task_id}")
async def get_quality_report(task_id: str):
    task = state.task_store.get(task_id)
    if task is None:
        task = state.task_history.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    result = _result_payload(task)
    stats = _stats_payload(result)
    run_id = _safe_run_id(task, result, stats)
    report = _load_json(_report_path(run_id)) if run_id else None
    source = "run_report" if report else "history_summary"
    report = report or _history_summary(task, result, stats)
    return {
        "task_id": task_id,
        "run_id": run_id or report.get("run_id", ""),
        "report_source": source,
        "report_schema_version": report.get("$schema", "run-report-v1"),
        "report": report,
    }


__all__ = ["get_quality_report", "router"]
