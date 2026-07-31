"""HTTP adapters for pipeline submission and task status."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .models import TaskStatus
from .pipeline_tasks import (
    PipelineSubmissionError,
    PipelineTaskService,
    run_pipeline_in_thread,
)
from .runtime_state import state

router = APIRouter()
_service = PipelineTaskService()


def _run_pipeline_in_thread(
    task_id: str,
    input_path: Path,
    output_path: Path,
    profile: str,
    output_format: str,
    skip_separation: bool,
    overrides: Dict[str, Any],
    session_dir: Optional[Path] = None,
) -> None:
    """Compatibility wrapper for the historical background-task entrypoint."""
    return run_pipeline_in_thread(
        task_id,
        input_path,
        output_path,
        profile,
        output_format,
        skip_separation,
        overrides,
        session_dir,
    )


def _legacy_thread_target():
    """Resolve the historical API hook before using the split implementation."""
    api = sys.modules.get("vocal_subtitle.webui.api")
    return getattr(api, "_run_pipeline_in_thread", _run_pipeline_in_thread)


@router.post("/run")
async def run_pipeline(
    file: UploadFile = File(...),
    profile: str = Form(default="default"),
    output_format: str = Form(default="srt"),
    skip_separation: bool = Form(default=False),
    overrides: str = Form(default="{}"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}") from exc
    try:
        return await _service.submit(
            contents,
            file.filename,
            profile=profile,
            output_format=output_format,
            skip_separation=skip_separation,
            overrides_text=overrides,
            thread_target=_legacy_thread_target(),
        )
    except PipelineSubmissionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in state.task_store:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    task = state.task_store[task_id]
    result = task.get("result") or {}
    return TaskStatus(
        task_id=task["task_id"],
        status=task["status"],
        progress=task.get("progress"),
        result=task.get("result"),
        error=task.get("error"),
        quality_status=result.get("quality_status"),
    )


@router.get("/tasks")
async def list_tasks():
    return [
        {
            "task_id": task["task_id"],
            "status": task["status"],
            "error": task.get("error"),
        }
        for task in state.task_store.values()
    ]


__all__ = ["_run_pipeline_in_thread", "list_tasks", "router", "run_pipeline"]
