"""Shared business services for API routes.

Extracted from api.py: pipeline runner thread, persistence manager,
VAD engine factory, subtitle batch editing, history clearing.
"""

import asyncio
import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ConfigLoader, PipelineConfig
from ..mapping.time_mapper import SubtitleEvent
from ..pipeline import Pipeline
from ..utils.file_hasher import compute_config_hash, compute_file_hash
from ..utils.session_manager import OUTPUT_NAMES, SessionManager
from ..utils.task_history import TaskHistoryManager
from .websocket import ws_manager

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Module-level globals (shared across routes)
# ------------------------------------------------------------

UPLOAD_DIR = Path(__file__).resolve().parents[3] / "uploads"

_task_store: Dict[str, Dict[str, Any]] = {}
_task_history = TaskHistoryManager()
_persistence_mgr = None


def _get_persistence_mgr():
    """延迟初始化 PersistenceManager"""
    global _persistence_mgr
    if _persistence_mgr is None:
        from ..utils.persistence_manager import PersistenceManager
        _persistence_mgr = PersistenceManager()
    return _persistence_mgr


def _dir_size_mb(directory: Path) -> float:
    """计算目录总大小 (MB)，不存在则返回 0"""
    import os
    if not directory.exists():
        return 0.0
    total = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


# ------------------------------------------------------------
# Pipeline background thread
# ------------------------------------------------------------

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
    """在后台线程中运行 Pipeline"""
    try:
        # 加载配置
        loader = ConfigLoader()
        config = loader.load_profile(profile)
        config = loader.merge_with_overrides(config, **overrides)

        # 创建 Pipeline
        pipeline = Pipeline(config)

        # 创建进度回调（桥接到 WebSocket）
        progress_callback = ws_manager.create_progress_callback(task_id)

        # 更新任务状态
        _task_store[task_id]["status"] = "running"
        _task_history.update(task_id, status="running")

        # 发送开始事件（通过主事件循环广播到 WebSocket）
        ws_manager.broadcast_from_thread(
            task_id,
            {
                "type": "stage_start",
                "stage": "pipeline",
                "total": 1,
                "description": "Pipeline 启动",
            },
        )

        # 运行 Pipeline（传入 session_dir 用于多格式输出）
        result = pipeline.run(
            input_path=input_path,
            output_path=output_path,
            output_format=output_format,
            progress_callback=progress_callback,
            skip_separation=skip_separation,
            task_id=task_id,
            session_dir=session_dir,
        )

        # 序列化结果并存储
        from .api_serializers import _build_task_result
        stats = result["stats"]
        from_cache = result.get("from_cache", False)

        task_result = _build_task_result(result, from_cache, output_path)

        _task_store[task_id].update(
            {
                "status": "completed",
                "result": task_result,
            }
        )

        # 更新持久化历史
        result_json = json.dumps(task_result, default=str)
        _task_history.update(
            task_id,
            status="completed",
            result_json=result_json,
            total_duration_seconds=stats.duration_seconds,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        # 发送完成事件
        ws_manager.broadcast_from_thread(
            task_id,
            {
                "type": "complete",
                "result": task_result,
            },
        )

        # 存储到 WebSocket manager
        ws_manager.store_task_result(task_id, task_result)

        # 自动应用持久化设置
        try:
            mgr = _get_persistence_mgr()
            mgr.persist_task(task_id, task_result)
        except Exception as e:
            logger.warning("Auto-persist failed for task %s: %s", task_id, e)

    except Exception as e:
        logger.exception("Pipeline task %s failed", task_id)
        error_msg = str(e)
        if task_id in _task_store:
            _task_store[task_id].update(
                {"status": "failed", "error": error_msg}
            )
        else:
            logger.warning(
                "Task %s was already removed from store before error handler",
                task_id,
            )
        _task_history.update(
            task_id,
            status="failed",
            error=error_msg,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        ws_manager.broadcast_from_thread(
            task_id,
            {"type": "error", "message": error_msg},
        )


# ------------------------------------------------------------
# History clearing
# ------------------------------------------------------------

def _clear_history_full():
    """Clear all history: memory tasks, persistent files, uploads."""
    _task_store.clear()

    # 同步清理持久化文件
    try:
        loader = ConfigLoader()
        config = loader.load_profile("default")
        pipeline = Pipeline(config)
        cache = pipeline._services.get_cache()
        persistent_cleaned = cache.clear_persistent_files()
        logger.info("Cleared %d persistent file entries", persistent_cleaned)
    except Exception as e:
        logger.warning("Failed to clear persistent files: %s", e)
        persistent_cleaned = 0

    # 清理 uploads 目录
    uploads_cleaned = 0
    if UPLOAD_DIR.exists():
        active_dirs = set()
        for task_info in _task_store.values():
            if task_info.get("status") == "running":
                sd = task_info.get("session_dir", "")
                if sd:
                    active_dirs.add(Path(sd).name)

        for item in UPLOAD_DIR.iterdir():
            if item.is_dir() and item.name in active_dirs:
                logger.info("Skipping upload dir for running task: %s", item.name)
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                elif item.is_file():
                    item.unlink()
                uploads_cleaned += 1
            except Exception as e:
                logger.warning("Failed to clean upload item %s: %s", item, e)

    return persistent_cleaned, uploads_cleaned
