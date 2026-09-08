"""Task history, cache and persisted-file routes.

This router owns storage lifecycle endpoints.  Pipeline execution remains in
``routes_pipeline``; both modules read the compatibility-aware runtime state.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .api_serializers import history_detail, history_item
from .models import CacheConfigUpdate, CacheInfoResponse, PersistenceSettingsModel
from .runtime_state import state
from .storage_services import WebUIStorageService

logger = logging.getLogger(__name__)
router = APIRouter()
_persistence_mgr = None
_storage = WebUIStorageService(state)


@router.get("/history")
async def list_history(
    limit: int = Query(default=20, le=200),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
):
    tasks = state.task_history.list(limit=limit, offset=offset, status=status)
    items = [history_item(task) for task in tasks]
    return {
        "items": items,
        "total": state.task_history.count(),
        "limit": limit,
        "offset": offset,
    }


@router.get("/history/{task_id}")
async def get_history_detail(task_id: str):
    task = state.task_history.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return history_detail(task)


@router.delete("/history/{task_id}")
async def delete_history(task_id: str):
    if not state.task_history.delete(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return {"status": "ok", "deleted": task_id}


@router.delete("/history")
async def clear_history(older_than_days: Optional[int] = Query(default=None)):
    count = state.task_history.clear(older_than_days=older_than_days)
    if older_than_days is not None:
        return {"status": "ok", "deleted_count": count}

    active_dirs = {
        task.get("session_dir", "")
        for task in state.task_store.values()
        if task.get("status") == "running" and task.get("session_dir")
    }
    state.task_store.clear()
    persistent_cleaned = 0
    try:
        persistent_cleaned = _storage.clear_persistent_files()
    except Exception as exc:
        logger.warning("Failed to clear persistent files: %s", exc)

    uploads_cleaned = _storage.clear_uploads(
        skip_names={path.rsplit("/", 1)[-1] for path in active_dirs}
    )
    return {
        "status": "ok",
        "deleted_count": count,
        "persistent_files_cleaned": persistent_cleaned,
        "uploads_cleaned": uploads_cleaned,
    }


def _dir_size_mb(directory: Path) -> float:
    return _storage.directory_size_mb(directory)


@router.get("/cache/info", response_model=CacheInfoResponse)
async def get_cache_info():
    return CacheInfoResponse(
        stages={},
        total_mb=round(_dir_size_mb(state.upload_dir), 2),
        total_items=state.task_history.count(),
        files_dir_mb=0,
        cache_dir=str(state.upload_dir.resolve()),
        ttl_map={},
        task_history_count=state.task_history.count(),
        task_history_db_mb=round(state.task_history.get_db_size_mb(), 2),
    )


@router.delete("/cache")
async def clear_cache(stage: Optional[str] = Query(default=None)):
    """清除缓存

    Args:
        stage: 指定阶段名称清除部分缓存，None 则清除 uploads 目录

    清除全部时清理 uploads 目录下的历史上传文件。
    计算阶段缓存（分离/转录等）保持不变，加速后续处理。
    持久化文件不受影响，随历史记录生命周期管理。
    """
    if stage:
        _storage.clear_stage(stage)
        return {"status": "ok", "cleared_stage": stage}
    cleaned = _storage.clear_uploads()
    return {"status": "ok", "cleared": "all", "uploads_cleaned": cleaned}


@router.put("/cache/config")
async def update_cache_config(body: CacheConfigUpdate):
    changes = {}
    if body.ttl_separation is not None:
        changes["ttl_separation"] = body.ttl_separation
    if body.ttl_transcription is not None:
        changes["ttl_transcription"] = body.ttl_transcription
    if body.history_retention_days is not None:
        changes["history_retention_days"] = body.history_retention_days
        state.task_history.clear(older_than_days=body.history_retention_days)
    return {
        "status": "ok",
        "message": "运行时缓存配置已更新（重启后从 YAML 读取）",
        "changes": changes,
    }


def _get_persistence_mgr():
    global _persistence_mgr
    # The pre-componentization API exposed this cache on ``webui.api`` and
    # callers may still replace it in tests or embedding applications.
    legacy_api = sys.modules.get("vocal_subtitle.webui.api")
    legacy_mgr = getattr(legacy_api, "_persistence_mgr", None)
    if legacy_mgr is not None:
        return legacy_mgr
    if _persistence_mgr is None:
        from ..utils.persistence_manager import PersistenceManager

        _persistence_mgr = PersistenceManager()
        if legacy_api is not None:
            setattr(legacy_api, "_persistence_mgr", _persistence_mgr)
    return _persistence_mgr


@router.get("/persistence/settings")
async def get_persistence_settings():
    return _get_persistence_mgr().get_settings().to_dict()


@router.put("/persistence/settings")
async def update_persistence_settings(body: PersistenceSettingsModel):
    from ..utils.persistence_manager import PersistenceSettings

    settings = PersistenceSettings(
        persist_asr_subtitle=body.persist_asr_subtitle,
        persist_llm_subtitle=body.persist_llm_subtitle,
        persist_final_ass=body.persist_final_ass,
        persist_final_srt=body.persist_final_srt,
        persist_vocals=body.persist_vocals,
        persist_accompaniment=body.persist_accompaniment,
        ttl_subtitle_days=body.ttl_subtitle_days,
        ttl_audio_days=body.ttl_audio_days,
    )
    _get_persistence_mgr().save_settings(settings)
    return {"status": "ok", "settings": settings.to_dict()}


@router.post("/persistence/apply/{task_id}")
async def apply_persistence(task_id: str):
    task = state.task_store.get(task_id)
    task_result = task.get("result") if task else None
    if not task_result:
        historical = state.task_history.get(task_id)
        if historical and historical.get("result_json"):
            try:
                task_result = json.loads(historical["result_json"])
            except (json.JSONDecodeError, TypeError):
                task_result = None
    if not task_result:
        raise HTTPException(status_code=404, detail="Task not found or has no result")
    manifest = _get_persistence_mgr().persist_task(task_id, task_result)
    return {"status": "ok", "task_id": task_id, "files": manifest.get("files", [])}


@router.get("/persistence/files/{task_id}")
async def get_persisted_files(task_id: str):
    files = _get_persistence_mgr().get_persisted_files(task_id)
    if not files:
        raise HTTPException(status_code=404, detail="No persisted files found for this task")
    return files


@router.delete("/persistence/files/{task_id}")
async def delete_persisted_files(task_id: str):
    ok = _get_persistence_mgr().delete_persisted_task(task_id)
    return {"status": "ok" if ok else "not_found", "task_id": task_id}


@router.post("/persistence/cleanup")
async def cleanup_expired_persistence():
    """清理所有已过期的持久化文件"""
    return {"status": "ok", "cleaned_dirs": _get_persistence_mgr().cleanup_expired()}


@router.get("/persistence/stats")
async def get_persistence_stats():
    return _get_persistence_mgr().get_persistence_stats()


__all__ = ["router", "_get_persistence_mgr"]
