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
    if api is not None:
        return getattr(api, "_run_pipeline_in_thread", _run_pipeline_in_thread)
    return _run_pipeline_in_thread


@router.post("/run")
async def run_pipeline(
    file: UploadFile = File(...),
    profile: str = Form(default="default"),
    output_format: str = Form(default="srt"),
    skip_separation: bool = Form(default=False),
    overrides: str = Form(default="{}"),
):
    """启动单文件 Pipeline 处理

    接收音频文件上传，在后台线程中运行全链路处理，
    通过 WebSocket 实时推送进度。

    基于输入文件 SHA256 哈希的会话目录：
    - 快速查重：os.path.exists(session_dir) → 已处理过
    - 标准化输出命名：ASR-generated.{srt,vtt,ass} / LLM-optimized.{srt,vtt,ass}
    """
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
    stats = result.get("stats") or {}
    return TaskStatus(
        task_id=task["task_id"],
        status=task["status"],
        run_id=task.get("run_id") or result.get("run_id") or stats.get("run_id"),
        contract_version=result.get("contract_version"),
        progress=task.get("progress"),
        result=task.get("result"),
        error=task.get("error"),
        quality_status=result.get("quality_status"),
        error_category=result.get("error_category"),
        diagnostics_complete=result.get("diagnostics_complete"),
        artifacts=result.get("artifacts"),
        diagnostics=result.get("diagnostics"),
        # "学习"标记与场景标签（内部学习任务，D28）；普通任务为 None
        task_type=task.get("task_type"),
        scenario=task.get("scenario"),
        learn_report=task.get("learn_report"),
    )


@router.get("/tasks/{task_id}/manifest")
async def get_task_manifest(task_id: str):
    """获取任务的 review-manifest-v1 审核清单（按任务结果即时构建）"""
    from ..contracts.review_manifest import REVIEW_MANIFEST_SCHEMA, build_review_manifest
    from .routes_subtitles import _load_completed_subtitle_task

    task, result = _load_completed_subtitle_task(task_id)
    stats = result.get("stats") or {}
    events = result.get("events", [])
    engines = {
        "asr": stats.get("final_engine") or stats.get("selected_engine") or "",
    }
    input_path = result.get("input_path") or ""
    manifest = build_review_manifest(
        task_id=task_id,
        run_id=result.get("run_id", ""),
        subtitle_path=result.get("subtitle_path") or "subtitle.srt",
        events=events,
        input_name=Path(input_path).name if input_path else None,
        duration=stats.get("duration_seconds"),
        engines={k: v for k, v in engines.items() if v} or None,
    )
    manifest["schema"] = REVIEW_MANIFEST_SCHEMA
    return manifest


@router.get("/tasks")
async def list_tasks():
    return [
        {
            "task_id": task["task_id"],
            "status": task["status"],
            "run_id": task.get("run_id") or (task.get("result") or {}).get("run_id"),
            "error": task.get("error"),
            # "学习"标记与场景标签（内部学习任务，D28/D27）；普通任务为空串
            "task_type": task.get("task_type") or "",
            "scenario": task.get("scenario") or "",
        }
        for task in state.task_store.values()
    ]


__all__ = ["_run_pipeline_in_thread", "list_tasks", "router", "run_pipeline"]
