"""REST API 端点 — 配置管理、Pipeline 执行、字幕操作、导出、任务历史、缓存管理"""

import asyncio
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from ..config import ConfigLoader, PipelineConfig
from ..mapping.time_mapper import SubtitleEvent
from ..pipeline import Pipeline
from ..utils.file_hasher import compute_config_hash, compute_file_hash
from ..utils.gpu_detector import GPUDetector
from ..utils.session_manager import OUTPUT_NAMES, SessionManager
from ..utils.task_history import TaskHistoryManager
from .models import (
    BatchRunRequest,
    CacheConfigUpdate,
    CacheInfoResponse,
    ConflictInfo,
    ConflictResolutionRequest,
    DeviceInfoResponse,
    FingerprintInfo,
    FingerprintListResponse,
    FingerprintMatchResponse,
    HealthScoreDetail,
    HealthTrendEntry,
    ImpactPredictionInfo,
    PersistenceSettingsModel,
    ProfileInfo,
    ShadowModeStatus,
    ShadowModeToggleRequest,
    SubtitleEditRequest,
    SubtitleEventResponse,
    TaskHistoryItem,
    TaskStatus,
)
from .websocket import ws_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# 任务存储（内存）
# ---------------------------------------------------------------------------

# task_id -> TaskStatus
_task_store: Dict[str, Dict[str, Any]] = {}

# 持久化任务历史管理器
_task_history = TaskHistoryManager()

# 上传文件临时目录
UPLOAD_DIR = Path(__file__).parent.parent.parent / "cache" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 场景模板描述
# ---------------------------------------------------------------------------

_PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "default": "通用场景，分离引擎 Spleeter，适合日常音频处理",
    "podcast": "播客/访谈场景，UVR 高品质分离，中文优化，低 VAD 阈值捕捉更多语音",
    "education": "教学/演讲场景，Spleeter 分离，较大的合并间隙适应讲课节奏",
    "variety_show": "综艺/直播场景，UVR 分离，背景音乐较多时的最佳选择",
    "music_live": "音乐现场场景，UVR 高品质分离，专为含背景音乐的语音优化",
}


def _get_profile_description(name: str) -> str:
    return _PROFILE_DESCRIPTIONS.get(name, "自定义配置")


def _config_summary(config: PipelineConfig) -> Dict[str, Any]:
    """提取配置关键字段摘要"""
    return {
        "separation_engine": config.separation.engine,
        "vad_engine": config.vad.engine,
        "asr_engine": config.asr.engine,
        "asr_model": config.asr.model,
        "language": config.asr.language,
        "device": config.asr.device,
        "llm_enabled": config.llm_optimize.enabled,
    }


def _config_to_overrides_dict(config: PipelineConfig) -> Dict[str, Any]:
    """将完整配置转为前端可用的参数字典"""
    return {
        "separator": config.separation.engine,
        "uvr_model": config.separation.uvr_model,
        "vad_engine": config.vad.engine,
        "vad_threshold": config.vad.threshold,
        "vad_min_speech_ms": config.vad.min_speech_duration_ms,
        "vad_min_silence_ms": config.vad.min_silence_duration_ms,
        "merge_min_silence_gap": config.merging.min_silence_gap,
        "merge_max_segment": config.merging.max_segment_length,
        "merge_padding": config.merging.padding,
        "asr_engine": config.asr.engine,
        "asr_model": config.asr.model,
        "asr_device": config.asr.device,
        "asr_compute_type": config.asr.compute_type,
        "language": config.asr.language or "",
        "asr_beam_size": config.asr.beam_size,
        "subtitle_min_duration": config.subtitle.min_duration,
        "subtitle_max_duration": config.subtitle.max_duration,
        "subtitle_max_chars_cjk": config.subtitle.max_chars_cjk,
        "subtitle_max_chars_latin": config.subtitle.max_chars_latin,
        "llm_enabled": config.llm_optimize.enabled,
        "llm_model": config.llm_optimize.model,
        "llm_batch_num": config.llm_optimize.batch_num,
        "llm_thread_num": config.llm_optimize.thread_num,
        "llm_base_url": config.llm_optimize.base_url or "",
        "llm_api_key": config.llm_optimize.api_key or "",
        "diarization_enabled": config.diarization.enabled,
        "diarization_distance_threshold": config.diarization.distance_threshold,
        "diarization_min_speakers": config.diarization.min_speakers,
        "diarization_max_speakers": config.diarization.max_speakers,
        "diarization_use_pca": config.diarization.use_pca,
        "diarization_pca_variance": config.diarization.pca_variance,
        "speaker_role_enabled": config.speaker_role.enabled,
        "speaker_role_model": config.speaker_role.model,
        "speaker_role_temperature": config.speaker_role.temperature,
        "speaker_role_context_hint": config.speaker_role.context_hint or "",
        # 说话人嵌入模型
        "speaker_embedding_enabled": config.speaker_embedding.enabled,
        "speaker_embedding_engine": config.speaker_embedding.engine,
        "speaker_embedding_model_ref": config.speaker_embedding.model_ref,
        "speaker_embedding_hf_token": "***" if config.speaker_embedding.hf_token else "",
        # 骨架分段模式
        "acoustic_skeleton_mode": config.acoustic_validation.skeleton_mode,
        "acoustic_export_skeleton": config.acoustic_validation.export_skeleton_segments,
        # 语义合并决策 (merge_decision) — ★ 反馈面板需要
        "fast_merge_max_gap": config.merge_decision.fast_merge_max_gap,
        "llm_decision_min_gap": config.merge_decision.llm_decision_min_gap,
        "llm_decision_max_gap": config.merge_decision.llm_decision_max_gap,
        "hard_split_min_gap": config.merge_decision.hard_split_min_gap,
        "llm_tier": config.merge_decision.llm_tier,
        "llm_merge_model": config.merge_decision.llm_model,
        # 反馈学习
        "feedback_enabled": config.feedback.enabled,
        "feedback_active_profile": config.feedback.active_profile,
    }


# ---------------------------------------------------------------------------
# 配置相关端点
# ---------------------------------------------------------------------------


@router.get("/profiles", response_model=List[ProfileInfo])
async def list_profiles():
    """获取所有可用场景模板"""
    loader = ConfigLoader()
    profiles = []
    for name in loader.list_profiles():
        try:
            config = loader.load_profile(name)
            profiles.append(
                ProfileInfo(
                    name=name,
                    description=_get_profile_description(name),
                    config_summary=_config_summary(config),
                )
            )
        except Exception as e:
            logger.warning("Failed to load profile '%s': %s", name, e)
    return profiles


@router.get("/profiles/{name}")
async def get_profile_config(name: str):
    """获取指定场景模板的完整配置"""
    loader = ConfigLoader()
    try:
        config = loader.load_profile(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

    return {
        "name": name,
        "description": _get_profile_description(name),
        "config": _config_to_overrides_dict(config),
    }


# ---------------------------------------------------------------------------
# 设备信息
# ---------------------------------------------------------------------------


@router.get("/device", response_model=DeviceInfoResponse)
async def get_device_info():
    """获取系统/GPU 设备信息"""
    info = GPUDetector.get_device_info()
    device_type = GPUDetector.get_best_device()
    return DeviceInfoResponse(
        device_type=info["device_type"],
        device_count=info["device_count"],
        device_names=info["device_names"],
        memory_mb=info["memory_mb"],
        recommended_compute_type=info["recommended_compute_type"],
        gpu_memory_used_mb=GPUDetector.get_gpu_memory_used_mb(),
        recommended_model=GPUDetector.select_whisper_model(device_type),
    )


@router.get("/speaker-embedding/license")
async def get_speaker_embedding_license():
    """获取说话人嵌入模型的协议信息

    前端展示 pyannote 模型的协议要求，
    用户需在 huggingface.co 上接受协议后才能使用。
    """
    try:
        from vocal_subtitle.diarization.speaker_embedding import (
            PyannoteEmbeddingEngine,
        )

        info = PyannoteEmbeddingEngine.license_info()
        return {
            "engine": info["engine"],
            "code_license": info["code_license"],
            "model_license": info["model_license"],
            "license_url": info["license_url"],
            "preset_models": [
                {
                    "ref": k,
                    "name": v["name"],
                    "description": v["description"],
                    "embedding_dim": v["embedding_dim"],
                    "license_url": v["license_url"],
                    "model_license_type": v["model_license_type"],
                    "size_mb": v["size_mb"],
                    "requires_token": v["requires_token"],
                }
                for k, v in info["preset_models"].items()
            ],
        }
    except ImportError:
        return {
            "engine": "unavailable",
            "note": "pyannote.audio 未安装。安装: pip install pyannote.audio",
        }


# ---------------------------------------------------------------------------
# Pipeline 执行
# ---------------------------------------------------------------------------


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

        # 提取字幕事件
        events: List[SubtitleEvent] = result.get("events", [])
        subtitle_events = [
            {
                "index": e.index,
                "start": e.start,
                "end": e.end,
                "text": e.text,
                "original_text": e.original_text,
                "speaker_id": e.speaker_id,
                "speaker_label": e.speaker_label,
                "physical_start": e.physical_start,
                "physical_end": e.physical_end,
                "source_word_ids": e.source_word_ids,
                "speaker_status": e.speaker_status,
                "speaker_source": e.speaker_source,
                "alignment_warning": e.alignment_warning,
            }
            for e in events
        ]

        stats = result["stats"]
        from_cache = result.get("from_cache", False)

        # 存储结果
        task_result = {
            "subtitle_path": str(result["subtitle_path"]),
            "llm_subtitle_path": str(result["llm_subtitle_path"]) if result.get("llm_subtitle_path") else None,
            "stats": stats.to_dict(),
            "events": subtitle_events,
            "from_cache": from_cache,
            "segment_count": stats.segment_count,
            "subtitle_count": stats.subtitle_count,
            "vocals_path": result.get("vocals_path"),
            "accompaniment_path": result.get("accompaniment_path"),
        }

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

        # 发送完成事件（通过主事件循环广播到 WebSocket）
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
        # 任务可能在运行期间被清除（如用户调用了 DELETE /api/history）
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
    import json

    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # 读取文件内容并计算哈希（用于快速查重）
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    original_filename = file.filename
    task_id = str(uuid.uuid4())[:8]

    # ---- 基于哈希的会话目录 ----
    session_mgr = SessionManager(UPLOAD_DIR)
    session_key = SessionManager.compute_session_key_from_bytes(contents)
    session_dir = session_mgr.session_dir(session_key)

    # 快速查重：检查是否已有处理过的会话目录
    existing_metadata = session_mgr.read_metadata(session_dir)
    if existing_metadata:
        logger.info(
            "📂 Session dir already exists for %s (hash=%s) — previously processed at %s. "
            "Pipeline will reuse cached vocals if available.",
            original_filename, session_key, existing_metadata.get("processed_at", "unknown"),
        )

    # 创建会话目录并保存输入文件
    session_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_filename).suffix or ".wav"
    input_path = session_dir / f"input{suffix}"

    try:
        input_path.write_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # 如果是视频文件，提取音轨为 WAV
    from ..utils.audio_utils import AudioUtils

    pipeline_input = input_path
    is_video = AudioUtils.is_video_file(input_path)
    if is_video:
        import logging
        _log = logging.getLogger(__name__)
        _log.info("Detected video file, extracting audio: %s", original_filename)
        try:
            pipeline_input = AudioUtils.extract_audio_from_video(
                input_path, session_dir
            )
            _log.info("Audio extracted: %s -> %s", input_path.name, pipeline_input.name)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract audio from video: {e}",
            )

    # 解析 overrides
    try:
        overrides_dict = json.loads(overrides)
    except json.JSONDecodeError:
        overrides_dict = {}

    # 加载配置
    loader = ConfigLoader()
    config = loader.load_profile(profile)
    config = loader.merge_with_overrides(config, **overrides_dict)

    # 计算文件哈希和配置哈希
    file_hash = compute_file_hash(pipeline_input)
    config_hash = compute_config_hash(config)
    file_size = pipeline_input.stat().st_size

    # ---- 写入初始 metadata.json ----
    try:
        session_mgr.write_metadata(
            session_dir,
            original_filename=original_filename,
            input_sha256=file_hash,
            profile=profile,
            config_hash=config_hash,
            task_id=task_id,
        )
    except Exception as e:
        logger.warning("Failed to write session metadata: %s", e)

    # 默认输出路径（保持兼容）
    output_path = session_dir / f"output.{output_format}"

    # ---- 检查全管道缓存（DB 级：相同文件 + 相同配置） ----
    cache_cfg = config.cache
    if cache_cfg.enabled and cache_cfg.full_pipeline_cache:
        cached_task = _task_history.find_by_hash(file_hash, config_hash)
        if cached_task and cached_task.get("result_json"):
            try:
                cached_result = json.loads(cached_task["result_json"])
                cached_subtitle_path = Path(cached_result.get("subtitle_path", ""))
                if cached_subtitle_path.exists():
                    logger.info(
                        "DB Cache HIT for %s (previous task: %s)",
                        original_filename, cached_task["id"],
                    )

                    # 复制字幕到会话目录
                    subtitle_text = cached_subtitle_path.read_text(encoding="utf-8")
                    output_path.write_text(subtitle_text, encoding="utf-8")

                    _task_store[task_id] = {
                        "task_id": task_id,
                        "status": "completed",
                        "result": cached_result,
                        "from_cache": True,
                        "input_file_name": original_filename,
                        "session_dir": str(session_dir),
                    }

                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "from_cache": True,
                    }
            except Exception as e:
                logger.warning("Failed to restore cached result: %s", e)

    # 创建内存任务
    _task_store[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": None,
        "result": None,
        "error": None,
        "from_cache": False,
        "input_file_name": original_filename,
        "session_dir": str(session_dir),
    }

    # 创建持久化历史记录
    try:
        _task_history.create(
            task_id=task_id,
            file_name=original_filename,
            file_hash=file_hash,
            file_size=file_size,
            profile=profile,
            config=config,
        )
    except Exception as e:
        logger.warning("Failed to create history record: %s", e)

    # 确保 WebSocket Manager 持有主事件循环引用（在后台线程启动前设置）
    try:
        ws_manager.set_main_loop(asyncio.get_running_loop())
    except RuntimeError:
        pass  # 不在异步上下文中，降级处理

    # 在后台线程中运行 Pipeline
    thread = threading.Thread(
        target=_run_pipeline_in_thread,
        args=(
            task_id,
            pipeline_input,  # 视频已提前提取为 WAV
            output_path,
            profile,
            output_format,
            skip_separation,
            overrides_dict,
            session_dir,  # 传入会话目录
        ),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id, "status": "pending"}


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """查询任务状态和结果"""
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = _task_store[task_id]
    return TaskStatus(
        task_id=task["task_id"],
        status=task["status"],
        progress=task.get("progress"),
        result=task.get("result"),
        error=task.get("error"),
    )


@router.get("/tasks")
async def list_tasks():
    """列出所有任务"""
    return [
        {
            "task_id": t["task_id"],
            "status": t["status"],
            "error": t.get("error"),
        }
        for t in _task_store.values()
    ]


# ---------------------------------------------------------------------------
# 任务历史（持久化）
# ---------------------------------------------------------------------------


@router.get("/history")
async def list_history(
    limit: int = Query(default=20, le=200),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
):
    """获取持久化任务历史列表"""
    tasks = _task_history.list(limit=limit, offset=offset, status=status)
    total = _task_history.count()
    items = []
    for t in tasks:
        result_summary = None
        if t.get("result_json"):
            try:
                r = json.loads(t["result_json"])
                result_summary = {
                    "subtitle_count": r.get("subtitle_count", 0),
                    "segment_count": r.get("segment_count", 0),
                    "from_cache": r.get("from_cache", False),
                }
            except (json.JSONDecodeError, TypeError):
                pass

        items.append(
            {
                "id": t["id"],
                "input_file_name": t["input_file_name"],
                "input_file_hash": t.get("input_file_hash", ""),
                "input_file_size": t.get("input_file_size", 0),
                "profile": t.get("profile", "default"),
                "status": t["status"],
                "error": t.get("error"),
                "total_duration_seconds": t.get("total_duration_seconds", 0),
                "created_at": t.get("created_at", ""),
                "completed_at": t.get("completed_at"),
                "result_summary": result_summary,
            }
        )

    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/history/{task_id}")
async def get_history_detail(task_id: str):
    """获取任务历史详情（含字幕事件）"""
    task = _task_history.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    events = []
    result_summary = None
    if task.get("result_json"):
        try:
            r = json.loads(task["result_json"])
            events = r.get("events", [])
            result_summary = {
                "subtitle_path": r.get("subtitle_path"),
                "subtitle_count": r.get("subtitle_count", 0),
                "segment_count": r.get("segment_count", 0),
                "from_cache": r.get("from_cache", False),
                "stats": r.get("stats"),
                "vocals_path": r.get("vocals_path"),
                "accompaniment_path": r.get("accompaniment_path"),
            }
        except (json.JSONDecodeError, TypeError):
            pass

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
        "result_summary": result_summary,
        "events": events,
    }


@router.delete("/history/{task_id}")
async def delete_history(task_id: str):
    """删除单条历史记录"""
    ok = _task_history.delete(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return {"status": "ok", "deleted": task_id}


@router.delete("/history")
async def clear_history(
    older_than_days: Optional[int] = Query(default=None),
):
    """清除历史记录

    Args:
        older_than_days: 只清除 N 天前的记录，None 则清除全部

    清除全部时，同步清理：
    - 内存中的任务状态
    - 持久化文件（分离音频、字幕等）
    - 上传目录中的临时文件
    """
    count = _task_history.clear(older_than_days=older_than_days)
    # 如果清除全部，也重置内存中的任务
    if older_than_days is None:
        _task_store.clear()

        # 同步清理持久化文件
        from ..pipeline import Pipeline
        try:
            loader = ConfigLoader()
            config = loader.load_profile("default")
            pipeline = Pipeline(config)
            cache = pipeline._get_cache()
            persistent_cleaned = cache.clear_persistent_files()
            logger.info("Cleared %d persistent file entries", persistent_cleaned)
        except Exception as e:
            logger.warning("Failed to clear persistent files: %s", e)
            persistent_cleaned = 0

        # 清理 uploads 目录（跳过正在运行的任务目录）
        uploads_cleaned = 0
        if UPLOAD_DIR.exists():
            # 收集所有运行中任务使用的 session 目录
            active_dirs = set()
            for task_info in _task_store.values():
                if task_info.get("status") == "running":
                    sd = task_info.get("session_dir", "")
                    if sd:
                        active_dirs.add(Path(sd).name)

            for item in UPLOAD_DIR.iterdir():
                # 跳过当前正在运行的任务目录
                if item.is_dir() and item.name in active_dirs:
                    logger.info(
                        "Skipping upload dir for running task: %s", item.name
                    )
                    continue
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    elif item.is_file():
                        item.unlink()
                    uploads_cleaned += 1
                except Exception as e:
                    logger.warning("Failed to clean upload item %s: %s", item, e)

        return {
            "status": "ok",
            "deleted_count": count,
            "persistent_files_cleaned": persistent_cleaned,
            "uploads_cleaned": uploads_cleaned,
        }

    return {"status": "ok", "deleted_count": count}


# ---------------------------------------------------------------------------
# 缓存管理
# ---------------------------------------------------------------------------


def _dir_size_mb(directory: Path) -> float:
    """计算目录总大小 (MB)，不存在则返回 0"""
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


@router.get("/cache/info", response_model=CacheInfoResponse)
async def get_cache_info():
    """获取缓存统计信息

    仅统计 uploads 目录大小（用户上传的临时文件），
    不包含模型下载缓存和处理阶段计算缓存。
    """
    # 只统计 uploads 目录
    uploads_mb = round(_dir_size_mb(UPLOAD_DIR), 2)
    history_count = _task_history.count()
    history_db_mb = _task_history.get_db_size_mb()

    return CacheInfoResponse(
        stages={},
        total_mb=uploads_mb,
        total_items=history_count,
        files_dir_mb=0,
        cache_dir=str(UPLOAD_DIR.resolve()),
        ttl_map={},
        task_history_count=history_count,
        task_history_db_mb=round(history_db_mb, 2),
    )


@router.delete("/cache")
async def clear_cache(
    stage: Optional[str] = Query(default=None),
):
    """清除缓存

    Args:
        stage: 指定阶段名称清除部分缓存，None 则清除 uploads 目录

    清除全部时清理 uploads 目录下的历史上传文件。
    计算阶段缓存（分离/转录等）保持不变，加速后续处理。
    持久化文件不受影响，随历史记录生命周期管理。
    """
    if stage:
        from ..pipeline import Pipeline

        loader = ConfigLoader()
        config = loader.load_profile("default")
        pipeline = Pipeline(config)
        cache = pipeline._get_cache()
        cache.clear_stage(stage)
        return {"status": "ok", "cleared_stage": stage}
    else:
        # 清理 uploads 目录下的历史上传文件
        uploads_cleaned = 0
        if UPLOAD_DIR.exists():
            for item in UPLOAD_DIR.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    elif item.is_file():
                        item.unlink()
                    uploads_cleaned += 1
                except Exception as e:
                    logger.warning("Failed to clean upload item %s: %s", item, e)
        logger.info("Cleared %d upload directories", uploads_cleaned)

        return {
            "status": "ok",
            "cleared": "all",
            "uploads_cleaned": uploads_cleaned,
        }


@router.put("/cache/config")
async def update_cache_config(body: CacheConfigUpdate):
    """更新缓存配置（运行时生效，不持久化到 YAML）"""
    # 此端点仅返回确认信息，实际配置更新需重启后从 YAML 读取
    changes = {}
    if body.ttl_separation is not None:
        changes["ttl_separation"] = body.ttl_separation
    if body.ttl_transcription is not None:
        changes["ttl_transcription"] = body.ttl_transcription
    if body.history_retention_days is not None:
        changes["history_retention_days"] = body.history_retention_days
        # 立即清理过期历史
        _task_history.clear(older_than_days=body.history_retention_days)

    return {
        "status": "ok",
        "message": "运行时缓存配置已更新（重启后从 YAML 读取）",
        "changes": changes,
    }


# ---------------------------------------------------------------------------
# 持久化文件管理
# ---------------------------------------------------------------------------

_persistence_mgr = None


def _get_persistence_mgr():
    """延迟初始化 PersistenceManager"""
    global _persistence_mgr
    if _persistence_mgr is None:
        from ..utils.persistence_manager import PersistenceManager
        _persistence_mgr = PersistenceManager()
    return _persistence_mgr


@router.get("/persistence/settings")
async def get_persistence_settings():
    """获取文件持久化设置"""
    mgr = _get_persistence_mgr()
    settings = mgr.get_settings()
    return settings.to_dict()


@router.put("/persistence/settings")
async def update_persistence_settings(body: PersistenceSettingsModel):
    """更新文件持久化设置"""
    from ..utils.persistence_manager import PersistenceSettings
    mgr = _get_persistence_mgr()
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
    mgr.save_settings(settings)
    return {"status": "ok", "settings": settings.to_dict()}


@router.post("/persistence/apply/{task_id}")
async def apply_persistence(task_id: str):
    """对已完成的任务应用持久化（复制文件到持久化目录）

    优先从内存任务存储查找，其次从历史记录查找。
    """
    task = _task_store.get(task_id)
    task_result = None

    if task and task.get("result"):
        task_result = task["result"]
    else:
        hist_task = _task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                task_result = json.loads(hist_task["result_json"])
            except (json.JSONDecodeError, TypeError):
                pass

    if not task_result:
        raise HTTPException(status_code=404, detail="Task not found or has no result")

    mgr = _get_persistence_mgr()
    manifest = mgr.persist_task(task_id, task_result)
    return {"status": "ok", "task_id": task_id, "files": manifest.get("files", [])}


@router.get("/persistence/files/{task_id}")
async def get_persisted_files(task_id: str):
    """获取任务的持久化文件列表"""
    mgr = _get_persistence_mgr()
    files = mgr.get_persisted_files(task_id)
    if not files:
        raise HTTPException(status_code=404, detail="No persisted files found for this task")
    return files


@router.delete("/persistence/files/{task_id}")
async def delete_persisted_files(task_id: str):
    """删除任务的持久化文件"""
    mgr = _get_persistence_mgr()
    ok = mgr.delete_persisted_task(task_id)
    return {"status": "ok" if ok else "not_found", "task_id": task_id}


@router.post("/persistence/cleanup")
async def cleanup_expired_persistence():
    """清理所有已过期的持久化文件"""
    mgr = _get_persistence_mgr()
    cleaned = mgr.cleanup_expired()
    return {"status": "ok", "cleaned_dirs": cleaned}


@router.get("/persistence/stats")
async def get_persistence_stats():
    """获取持久化存储统计"""
    mgr = _get_persistence_mgr()
    return mgr.get_persistence_stats()


# ---------------------------------------------------------------------------
# 字幕操作
# ---------------------------------------------------------------------------


@router.get("/subtitle/{task_id}", response_model=List[SubtitleEventResponse])
async def get_subtitles(task_id: str):
    """获取任务的字幕事件列表"""
    task = _task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")

    result = task.get("result", {})
    events = result.get("events", [])

    return [SubtitleEventResponse(**e) for e in events]


def _subtitle_event_from_payload(payload: dict) -> SubtitleEvent:
    """Reconstruct a SubtitleEvent from a dict payload, preserving all provenance fields."""
    return SubtitleEvent.from_dict(payload)


def _subtitle_event_to_payload(event: SubtitleEvent) -> dict:
    """Serialize a SubtitleEvent to a dict, preserving all provenance fields."""
    return event.to_dict()


def _rewrite_subtitle_files(task_result: Dict[str, Any]) -> None:
    """将内存中的字幕事件写回磁盘文件

    同时更新主字幕文件和 LLM 字幕文件（如果存在）。
    """
    events = task_result.get("events", [])
    if not events:
        return

    # 重建 SubtitleEvent 对象
    from ..mapping.subtitle_builder import SubtitleBuilder, SubtitleRule
    from ..config import SubtitleBuildConfig

    rebuilt_events = [
        SubtitleEvent(
            index=e["index"],
            start=e["start"],
            end=e["end"],
            text=e["text"],
            original_text=e.get("original_text"),
            speaker_id=e.get("speaker_id"),
            speaker_label=e.get("speaker_label"),
            physical_start=e.get("physical_start"),
            physical_end=e.get("physical_end"),
            source_word_ids=e.get("source_word_ids", []),
            speaker_status=e.get("speaker_status", ""),
            speaker_source=e.get("speaker_source", ""),
            alignment_warning=e.get("alignment_warning"),
        )
        for e in events
    ]

    # 加载字幕构建规则
    loader = ConfigLoader()
    try:
        config = loader.load_profile("default")
        sub_cfg = config.subtitle
    except Exception:
        sub_cfg = SubtitleBuildConfig()

    builder = SubtitleBuilder(
        rule=SubtitleRule(
            min_duration=sub_cfg.min_duration,
            max_duration=sub_cfg.max_duration,
            max_chars_cjk=sub_cfg.max_chars_cjk,
            max_chars_latin=sub_cfg.max_chars_latin,
            max_lines=sub_cfg.max_lines,
        )
    )

    # 写回主字幕文件
    subtitle_path = task_result.get("subtitle_path")
    if subtitle_path:
        try:
            srt_text = builder.build_to_string(rebuilt_events, fmt="srt")
            Path(subtitle_path).write_text(srt_text, encoding="utf-8")
            logger.info("Rewrote subtitle file: %s", subtitle_path)
        except Exception as e:
            logger.warning("Failed to rewrite subtitle file: %s", e)

    # 写回 LLM 字幕文件（如果存在）
    llm_path = task_result.get("llm_subtitle_path")
    if llm_path:
        try:
            llm_text = builder.build_to_string(rebuilt_events, fmt="srt")
            Path(llm_path).write_text(llm_text, encoding="utf-8")
            logger.info("Rewrote LLM subtitle file: %s", llm_path)
        except Exception as e:
            logger.warning("Failed to rewrite LLM subtitle file: %s", e)


@router.put("/subtitle/{task_id}/{index}")
async def update_subtitle(task_id: str, index: int, body: SubtitleEditRequest):
    """编辑单条字幕文本，并自动保存到磁盘文件"""
    task = _task_store.get(task_id)
    if not task:
        # 尝试从历史记录中查找
        hist_task = _task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                r = json.loads(hist_task["result_json"])
                # 构建临时任务结构
                task = {"task_id": task_id, "status": "completed", "result": r}
                _task_store[task_id] = task
            except (json.JSONDecodeError, TypeError):
                raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        else:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    result = task.get("result", {})
    events = result.get("events", [])

    for e in events:
        if e["index"] == index:
            e["text"] = body.text
            e["original_text"] = None  # 手动编辑后清除原始文本标记
            # 自动保存到磁盘
            _rewrite_subtitle_files(result)
            return {"status": "ok", "index": index, "text": body.text}

    raise HTTPException(status_code=404, detail=f"Subtitle {index} not found")


@router.get("/subtitle/{task_id}/export")
async def export_subtitle(
    task_id: str,
    fmt: str = Query(default="srt", alias="format"),
):
    """导出自幕文件为指定格式（支持内存任务和历史任务）"""
    raw_events = []

    # 先尝试内存中的任务
    task = _task_store.get(task_id)
    if task and task.get("status") == "completed":
        result = task.get("result", {})
        raw_events = result.get("events", [])

    # 如果内存中没有，尝试历史记录
    if not raw_events:
        hist_task = _task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                r = json.loads(hist_task["result_json"])
                raw_events = r.get("events", [])
            except (json.JSONDecodeError, TypeError):
                pass

    if not raw_events:
        raise HTTPException(status_code=404, detail="Task not found or has no events")

    # 重建 SubtitleEvent 对象
    events = [
        SubtitleEvent(
            index=e["index"],
            start=e["start"],
            end=e["end"],
            text=e["text"],
            original_text=e.get("original_text"),
            speaker_id=e.get("speaker_id"),
            speaker_label=e.get("speaker_label"),
        )
        for e in raw_events
    ]

    # 使用 SubtitleBuilder 生成字符串
    from ..config import SubtitleBuildConfig
    from ..mapping.subtitle_builder import SubtitleBuilder, SubtitleRule

    # 从默认配置加载规则
    loader = ConfigLoader()
    try:
        config = loader.load_profile("default")
        sub_cfg = config.subtitle
    except Exception:
        sub_cfg = SubtitleBuildConfig()

    builder = SubtitleBuilder(
        rule=SubtitleRule(
            min_duration=sub_cfg.min_duration,
            max_duration=sub_cfg.max_duration,
            max_chars_cjk=sub_cfg.max_chars_cjk,
            max_chars_latin=sub_cfg.max_chars_latin,
            max_lines=sub_cfg.max_lines,
        )
    )

    subtitle_text = builder.build_to_string(events, fmt=fmt)

    media_types = {
        "srt": "text/plain; charset=utf-8",
        "vtt": "text/vtt; charset=utf-8",
        "ass": "text/plain; charset=utf-8",
    }

    return PlainTextResponse(
        content=subtitle_text,
        media_type=media_types.get(fmt, "text/plain"),
        headers={
            "Content-Disposition": f'attachment; filename="subtitle.{fmt}"'
        },
    )


# ---------------------------------------------------------------------------
# 分离音频导出
# ---------------------------------------------------------------------------


@router.get("/tasks/{task_id}/audio")
async def download_separated_audio(
    task_id: str,
    type: str = Query(default="vocals", description="vocals 或 accompaniment"),
):
    """下载人声分离产出的音频文件

    Args:
        task_id: 任务 ID
        type: 音频类型 — 'vocals'（人声）或 'accompaniment'（背景声/伴奏）
    """
    if type not in ("vocals", "accompaniment"):
        raise HTTPException(status_code=400, detail="type must be 'vocals' or 'accompaniment'")

    # 先查内存中的任务
    task = _task_store.get(task_id)
    file_path = None

    if task and task.get("result"):
        file_path = task["result"].get(f"{type}_path")

    # 内存中找不到，查持久化历史
    if not file_path:
        hist_task = _task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                r = json.loads(hist_task["result_json"])
                file_path = r.get(f"{type}_path")
            except (json.JSONDecodeError, TypeError):
                pass

    if not file_path:
        raise HTTPException(
            status_code=404,
            detail=f"No {type} audio found for this task. "
                    "The task may not have run separation, or the files have been cleaned up.",
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{type.capitalize()} audio file no longer exists on disk.",
        )

    # 确定下载文件名
    type_label = "人声" if type == "vocals" else "背景声"
    original_name = task.get("input_file_name", "audio") if task else "audio"
    download_name = f"{Path(original_name).stem}_{type_label}.wav"

    from fastapi.responses import FileResponse

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=download_name,
    )


@router.get("/tasks/{task_id}/audio/stream")
async def stream_audio(
    task_id: str,
    type: str = Query(default="vocals", description="vocals, accompaniment, 或 input"),
):
    """流式传输音频文件（支持 HTTP Range 请求，用于 HTML5 Audio 播放）

    与 download 端点不同，此端点不设置 Content-Disposition，
    浏览器可直接用于 <audio> 元素的 src 属性，支持 seek 操作。

    音频来源优先级（type=vocals 时）：
    1. 分离后的人声 (vocals_path)
    2. 原始上传文件 (input_path)

    Args:
        task_id: 任务 ID
        type: 音频类型 — 'vocals'（人声）, 'accompaniment'（背景声）, 或 'input'（原始文件）
    """
    file_path = None

    # 辅助函数：在会话目录中查找 input 文件
    def _find_input_file(session_dir_str: str) -> Optional[str]:
        task_dir = Path(session_dir_str)
        if task_dir.exists():
            for f in task_dir.iterdir():
                if f.name.startswith("input") and f.suffix in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
                    return str(f)
        return None

    # 先查内存中的任务
    task = _task_store.get(task_id)
    if task and task.get("result"):
        if type == "input":
            # 优先从 session_dir 查找
            session_dir = task.get("session_dir", "")
            if session_dir:
                file_path = _find_input_file(session_dir)
            # 回退到旧路径格式
            if not file_path:
                file_path = _find_input_file(str(UPLOAD_DIR / task_id))
        else:
            file_path = task["result"].get(f"{type}_path")

    # 内存中找不到，查持久化历史
    if not file_path:
        hist_task = _task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                r = json.loads(hist_task["result_json"])
                if type == "input":
                    # 尝试从 session_dir 查找
                    if task and task.get("session_dir"):
                        file_path = _find_input_file(task["session_dir"])
                    if not file_path:
                        file_path = _find_input_file(str(UPLOAD_DIR / task_id))
                else:
                    file_path = r.get(f"{type}_path")
            except (json.JSONDecodeError, TypeError):
                pass

    # Fallback: 如果请求 vocals 但找不到，尝试 input
    if not file_path and type == "vocals":
        if task and task.get("session_dir"):
            file_path = _find_input_file(task["session_dir"])
        if not file_path:
            file_path = _find_input_file(str(UPLOAD_DIR / task_id))

    if not file_path:
        raise HTTPException(
            status_code=404,
            detail=f"No audio found for streaming. The task may not have audio files available.",
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file no longer exists on disk.",
        )

    # FileResponse 原生支持 Range 请求（Accept-Ranges: bytes）
    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
    )


# ---------------------------------------------------------------------------
# 字幕文件直接下载（干净版 / LLM 优化版）
# ---------------------------------------------------------------------------


@router.get("/tasks/{task_id}/subtitle-file")
async def download_subtitle_file(
    task_id: str,
    version: str = Query(default="clean", description="clean 或 llm"),
):
    """下载 Pipeline 产出的字幕文件（直接返回磁盘文件）

    Args:
        task_id: 任务 ID
        version: 'clean' — LLM 优化前的干净版, 'llm' — LLM 优化版
    """
    if version not in ("clean", "llm"):
        raise HTTPException(status_code=400, detail="version must be 'clean' or 'llm'")

    # 先查内存中的任务
    task = _task_store.get(task_id)
    file_path = None
    input_name = "subtitle"

    if task and task.get("result"):
        if version == "llm":
            file_path = task["result"].get("llm_subtitle_path")
        if not file_path:
            file_path = task["result"].get("subtitle_path")
        input_name = task.get("input_file_name", "subtitle") if hasattr(task, "get") else "subtitle"

    # 内存中找不到，查持久化历史
    if not file_path:
        hist_task = _task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                r = json.loads(hist_task["result_json"])
                if version == "llm":
                    file_path = r.get("llm_subtitle_path")
                if not file_path:
                    file_path = r.get("subtitle_path")
            except (json.JSONDecodeError, TypeError):
                pass

    if not file_path:
        raise HTTPException(
            status_code=404,
            detail=f"No subtitle file found for this task.",
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Subtitle file no longer exists on disk.",
        )

    # 确定下载文件名
    version_label = "LLM优化版" if version == "llm" else "干净版"
    download_name = f"{file_path.stem}_{version_label}{file_path.suffix}"

    from fastapi.responses import FileResponse

    return FileResponse(
        path=str(file_path),
        media_type="text/plain; charset=utf-8",
        filename=download_name,
    )


# ---------------------------------------------------------------------------
# LLM 模型管理
# ---------------------------------------------------------------------------

# 预设供应商（OpenAI 兼容协议，截至 2026-06）
# 参考 qwen-tts-webui 的 _LLM_PROVIDER_PRESETS 设计
LLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    # ── 国际主流 ──────────────────────────────────────────────────
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek（深度求索）",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-pro",
        "default_models": [
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "base_url": "https://api.openai.com",
        "default_model": "gpt-5.5",
        "default_models": [
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5",
            "o4-mini",
        ],
    },
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic (Claude)",
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-fable-5",
        "default_models": [
            "claude-fable-5",
            "claude-mythos-5",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
    },
    "google": {
        "id": "google",
        "name": "Google (Gemini)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-3.5-flash",
        "default_models": [
            "gemini-3.5-flash",
            "gemini-3.5-pro",
            "gemini-3.0-pro",
        ],
    },
    # ── 国内主流 ──────────────────────────────────────────────────
    "zhipu": {
        "id": "zhipu",
        "name": "智谱 AI (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "GLM-5.2",
        "default_models": [
            "GLM-5.2",
            "GLM-5.1",
            "glm-4-plus",
        ],
    },
    "dashscope": {
        "id": "dashscope",
        "name": "阿里百炼 (Qwen)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode",
        "default_model": "qwen3.7-max",
        "default_models": [
            "qwen3.7-max",
            "qwen-plus",
            "qwen-max",
            "qwen-turbo",
        ],
    },
    "hunyuan": {
        "id": "hunyuan",
        "name": "腾讯混元 (Hunyuan)",
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "default_model": "hunyuan-hy3-preview",
        "default_models": [
            "hunyuan-hy3-preview",
            "hunyuan-turbo",
            "hunyuan-pro",
        ],
    },
    "moonshot": {
        "id": "moonshot",
        "name": "月之暗面 (Kimi)",
        "base_url": "https://api.moonshot.cn",
        "default_model": "kimi-k2.6",
        "default_models": [
            "kimi-k2.6",
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
        ],
    },
    "minimax": {
        "id": "minimax",
        "name": "MiniMax",
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "minimax-m3",
        "default_models": [
            "minimax-m3",
            "minimax-m2.7",
            "abab6.5s-chat",
        ],
    },
    "siliconflow": {
        "id": "siliconflow",
        "name": "硅基流动 (SiliconFlow)",
        "base_url": "https://api.siliconflow.cn",
        "default_model": "deepseek-ai/DeepSeek-V3",
        "default_models": [
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "Pro/deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen3-235B-A22B",
        ],
    },
    "ollama": {
        "id": "ollama",
        "name": "Ollama（本地）",
        "base_url": "http://localhost:11434",
        "default_model": "llama3",
        "default_models": [],
    },
    "custom": {
        "id": "custom",
        "name": "自定义",
        "base_url": "",
        "default_model": "",
        "default_models": [],
    },
}


@router.get("/llm/providers")
async def list_llm_providers():
    """获取预设 LLM 供应商列表（不含敏感信息）"""
    return [
        {
            "id": pid,
            "name": info["name"],
            "base_url": info["base_url"],
            "default_model": info["default_model"],
            "default_models": info.get("default_models", []),
        }
        for pid, info in LLM_PROVIDERS.items()
    ]


@router.post("/llm/models")
async def fetch_llm_models(body: dict):
    """通过 OpenAI 兼容 API 获取可用模型列表。

    优先从 API 实时获取；获取失败或未配置 API 密钥时，
    根据 base_url 匹配预设供应商的 default_models 降级返回。

    Request body:
        {"base_url": "https://api.deepseek.com", "api_key": "sk-..."}

    支持 OpenAI 兼容格式、Ollama 本地格式等。
    Ollama 本地调用时 api_key 可留空。
    """
    base_url = body.get("base_url", "").strip()
    api_key = body.get("api_key", "").strip()

    if not base_url:
        raise HTTPException(status_code=400, detail="base_url is required")

    # 规范化 URL
    try:
        from llm_subtitle_optimizer.llm_client import normalize_base_url
        base_url = normalize_base_url(base_url)
    except ImportError:
        # 简单追加 /v1
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

    # ── 1) 尝试从 API 实时获取 ──
    if api_key:
        try:
            import urllib.request
            import json

            headers = {"Content-Type": "application/json"}
            headers["Authorization"] = f"Bearer {api_key}"

            req = urllib.request.Request(
                f"{base_url}/models",
                headers=headers,
            )

            models = []
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            # 处理不同供应商的响应格式
            if "data" in data:
                # OpenAI 兼容格式: {"data": [{"id": "model-name", ...}]}
                for m in data.get("data", []):
                    model_id = m.get("id", "")
                    if model_id:
                        models.append({
                            "id": model_id,
                            "owned_by": m.get("owned_by", ""),
                        })
            elif "models" in data:
                # Ollama 格式: {"models": [{"name": "llama3:latest", ...}]}
                for m in data.get("models", []):
                    model_id = m.get("name") or m.get("id") or m.get("model", "")
                    if model_id:
                        models.append({
                            "id": model_id,
                            "owned_by": m.get("provider", m.get("owned_by", "")),
                        })
            elif isinstance(data, list):
                # 纯列表格式
                for m in data:
                    if isinstance(m, dict):
                        model_id = m.get("id") or m.get("name") or str(m)
                    else:
                        model_id = str(m)
                    if model_id:
                        models.append({"id": model_id, "owned_by": ""})
            else:
                # 未知格式，尝试提取 id/name 字段
                for key in ["data", "models", "result", "items"]:
                    items = data.get(key, [])
                    if isinstance(items, list):
                        for m in items:
                            if isinstance(m, dict):
                                model_id = m.get("id") or m.get("name") or m.get("model", "")
                                if model_id:
                                    models.append({"id": model_id, "owned_by": m.get("owned_by", m.get("provider", ""))})

            if models:
                # 排序：chat/completion 类模型排前面
                models.sort(key=lambda m: (
                    not any(kw in m["id"].lower() for kw in
                            ("chat", "gpt", "claude", "deepseek", "qwen", "glm", "llama", "command")),
                    m["id"],
                ))
                return {"models": models, "total": len(models), "source": "api"}

        except Exception:
            pass  # 降级到预设

    # ── 2) 降级：根据 api_base 匹配预设供应商 ──
    # 去掉 /v1 后缀进行匹配
    base_url_clean = base_url.rstrip("/").removesuffix("/v1").rstrip("/")
    for p in LLM_PROVIDERS.values():
        if p["base_url"].rstrip("/") == base_url_clean and p.get("default_models"):
            models = [{"id": m, "owned_by": p["name"]} for m in p["default_models"]]
            return {"models": models, "total": len(models), "source": "preset"}

    # ── 3) 无密钥也尝试匿名请求（Ollama 等本地服务） ──
    try:
        import urllib.request
        import json

        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(f"{base_url}/models", headers=headers)

        models = []
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        if "data" in data:
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if model_id:
                    models.append({"id": model_id, "owned_by": m.get("owned_by", "")})
        elif "models" in data:
            for m in data.get("models", []):
                model_id = m.get("name") or m.get("id") or m.get("model", "")
                if model_id:
                    models.append({"id": model_id, "owned_by": m.get("provider", m.get("owned_by", ""))})

        if models:
            models.sort(key=lambda x: x["id"])
            return {"models": models, "total": len(models), "source": "api"}

    except Exception:
        pass

    # ── 4) 完全无结果 ──
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="未提供 API 密钥且 API 地址需要认证，请先输入密钥后重试",
        )
    raise HTTPException(
        status_code=502,
        detail="获取模型列表失败：API 不可达或密钥无效。将使用预设默认模型。",
    )


# ---------------------------------------------------------------------------
# 反馈学习端点 (Phase 5: 自适应参数学习)
# ---------------------------------------------------------------------------


@router.post("/feedback/learn")
async def feedback_learn(
    audio: UploadFile = File(...),
    reference: UploadFile = File(...),
    profile: str = Form(default="default"),
    feedback_profile: str = Form(default="user_default"),
    run_pipeline_first: bool = Form(default=True),
    dry_run: bool = Form(default=False),
):
    """上传音频 + 修订字幕，触发反馈学习

    流程:
    1. 若 run_pipeline_first=True，先用当前参数生成自动字幕
    2. 对齐自动版与用户修订版
    3. 分析差异、更新用户配置

    Returns:
        差异分析报告（含参数调整建议）
    """
    import tempfile

    from ..config import ConfigLoader
    from ..feedback import (
        DiffAnalyzer,
        FewShotBuilder,
        ParamLearner,
        SubtitleAligner,
        UserProfileManager,
    )
    from ..feedback.aligner import AlignmentError, parse_subtitle_file
    from ..pipeline import Pipeline

    # 验证参考字幕格式
    ref_filename = reference.filename or ""
    ref_suffix = Path(ref_filename).suffix.lower()
    if ref_suffix not in (".srt", ".ass"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的字幕格式: {ref_suffix}，仅支持 .srt / .ass",
        )

    # 保存上传文件
    audio_dir = UPLOAD_DIR / f"feedback_{uuid.uuid4().hex[:8]}"
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / (audio.filename or "audio.wav")
    ref_path = audio_dir / (reference.filename or "reference.srt")

    try:
        audio_content = await audio.read()
        ref_content = await reference.read()

        audio_path.write_bytes(audio_content)
        ref_path.write_bytes(ref_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    try:
        # 加载配置
        loader = ConfigLoader()
        config = loader.load_profile(profile)
        feedback_cfg = config.feedback

        # 解析用户修订字幕
        manual_events = parse_subtitle_file(ref_path)
        if not manual_events:
            return {
                "status": "error",
                "message": "未从修订文件中解析到字幕事件",
            }

        # 生成自动版字幕
        auto_events = []
        if run_pipeline_first:
            pipeline = Pipeline(config)
            result = pipeline.run(
                input_path=audio_path,
                skip_separation=True,
            )
            auto_events = result.get("events", [])

        if not auto_events and run_pipeline_first:
            return {
                "status": "error",
                "message": "管道未生成字幕事件",
            }

        # 对齐
        aligner = SubtitleAligner(
            min_iou=feedback_cfg.alignment_min_iou,
            min_coverage=feedback_cfg.alignment_min_coverage,
            text_weight=feedback_cfg.alignment_text_weight,
            semantic_weight=feedback_cfg.alignment_semantic_weight,
            semantic_enabled=feedback_cfg.alignment_semantic_enabled,
        )
        try:
            pairs = aligner.align(auto_events, manual_events)
        except AlignmentError as e:
            return {
                "status": "error",
                "message": f"对齐失败: {e}",
                "alignment_coverage": round(e.coverage, 3) if e.coverage else 0,
                "total_pairs": e.n_matched,
                "auto_event_count": e.n_auto,
                "manual_event_count": e.n_manual,
                "time_shifts_count": 0,
                "merge_actions_count": 0,
                "text_edits_count": 0,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"对齐失败: {e}",
            }

        # 差异分析
        analyzer = DiffAnalyzer(
            param_isolation_enabled=feedback_cfg.param_isolation_enabled,
        )
        diff_report = analyzer.analyze(pairs)

        # 构建响应
        response = {
            "status": "ok",
            "alignment_coverage": round(diff_report.alignment_coverage, 3),
            "total_pairs": diff_report.total_pairs,
            "time_shifts_count": len(diff_report.time_shifts),
            "merge_actions_count": len(diff_report.merge_actions),
            "text_edits_count": len(diff_report.text_edits),
            "param_adjustments": {
                path: {
                    "direction": adj.direction,
                    "confidence": round(adj.confidence, 3),
                    "learn_weight": round(adj.learn_weight, 3),
                    "reason": adj.reason,
                    "param_tier": adj.param_tier,
                }
                for path, adj in diff_report.attribution.items()
            },
            "structural_revision": diff_report.structural_revision,
            "message": "",
        }

        # 应用学习（非 dry-run）
        if not dry_run and diff_report.attribution:
            profile_mgr = UserProfileManager(feedback_cfg)
            profile = profile_mgr.load(feedback_profile)
            current_overrides = profile.get("overrides", {})

            learner = ParamLearner(profile_mgr)
            learner.learn_from_diff(
                diff_report=diff_report,
                current_config_overrides=current_overrides,
                profile_name=feedback_profile,
            )

            # Few-shot
            if feedback_cfg.few_shot_enabled:
                few_shot = FewShotBuilder(max_examples=feedback_cfg.few_shot_max_examples)
                few_shot.load_cache(feedback_profile)
                few_shot.build_merge_examples(diff_report.merge_actions)
                if diff_report.text_edits:
                    few_shot.build_format_examples(diff_report.text_edits)
                few_shot.save_cache(feedback_profile)

            response["message"] = f"已学习 {len(diff_report.attribution)} 个参数调整"
        elif dry_run:
            response["message"] = "[dry-run] 未实际更新配置"
        elif not diff_report.attribution:
            response["message"] = "无需调整参数"

        return response

    except Exception as e:
        logger.exception("Feedback learn failed")
        return {
            "status": "error",
            "message": str(e),
        }
    finally:
        # 清理临时文件
        try:
            import shutil
            shutil.rmtree(audio_dir, ignore_errors=True)
        except Exception:
            pass


@router.post("/feedback/preview")
async def feedback_preview(
    audio: UploadFile = File(...),
    reference: UploadFile = File(...),
    profile: str = Form(default="default"),
):
    """预览：只分析差异，不更新配置（dry-run 模式）"""
    return await feedback_learn(
        audio=audio,
        reference=reference,
        profile=profile,
        feedback_profile="user_default",
        run_pipeline_first=True,
        dry_run=True,
    )


@router.get("/feedback/profiles")
async def list_feedback_profiles():
    """列出所有用户配置及其学习统计"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    profile_names = mgr.list_profiles()

    profiles = []
    for name in profile_names:
        p = mgr.load(name)
        profiles.append({
            "profile_id": name,
            "base_profile": p.get("base_profile", "default"),
            "feedback_count": p.get("feedback_count", 0),
            "created_at": p.get("created_at", ""),
            "updated_at": p.get("updated_at", ""),
            "is_active": p.get("is_active", True),
            "overrides": p.get("overrides", {}),
            "history_count": len(p.get("history", [])),
        })

    return {"profiles": profiles, "total": len(profiles)}


@router.get("/feedback/profile/{name}")
async def get_feedback_profile(name: str):
    """获取指定用户配置的详细信息"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    profile = mgr.load(name)

    if not profile or profile.get("profile_id") != name:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

    return {
        "profile_id": profile.get("profile_id"),
        "base_profile": profile.get("base_profile", "default"),
        "description": profile.get("description", ""),
        "feedback_count": profile.get("feedback_count", 0),
        "created_at": profile.get("created_at", ""),
        "updated_at": profile.get("updated_at", ""),
        "is_active": profile.get("is_active", True),
        "overrides": profile.get("overrides", {}),
        "fingerprint": profile.get("fingerprint", {}),
        "history": profile.get("history", [])[-10:],  # 最近 10 条
        "few_shot_examples_count": len(profile.get("few_shot_examples", [])),
    }


@router.post("/feedback/profile/{name}/rollback")
async def rollback_feedback_profile(name: str):
    """回滚指定用户配置"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    try:
        profile = mgr.rollback(name)
        return {
            "status": "ok",
            "profile_id": name,
            "updated_at": profile.get("updated_at", ""),
            "feedback_count": profile.get("feedback_count", 0),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No backup found for profile: {name}")


@router.delete("/feedback/profile/{name}")
async def delete_feedback_profile(name: str):
    """删除指定用户配置"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    if name == "user_default":
        raise HTTPException(status_code=400, detail="Cannot delete the default profile")

    mgr = UserProfileManager(FeedbackConfig())
    ok = mgr.delete(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")
    return {"status": "ok", "deleted": name}


# ---------------------------------------------------------------------------
# 指纹管理端点 (Phase 5.3)
# ---------------------------------------------------------------------------


@router.get("/feedback/fingerprints", response_model=FingerprintListResponse)
async def list_fingerprints():
    """列出所有音频指纹"""
    from ..config import FeedbackConfig
    from ..feedback import AudioFingerprinter

    cfg = FeedbackConfig()
    fingerprinter = AudioFingerprinter(
        distance_method=cfg.fingerprint_distance_method,
    )

    fps = fingerprinter.list_all()
    return FingerprintListResponse(
        fingerprints=[
            FingerprintInfo(
                id=f["id"],
                profile_id=f["profile_id"],
                audio_hash=f["audio_hash"],
                audio_signature=f.get("audio_signature", ""),
                feedback_count=f.get("feedback_count", 1),
                created_at=f.get("created_at", ""),
            )
            for f in fps
        ],
        total=len(fps),
        db_path=str(fingerprinter._db_path),
    )


@router.post("/feedback/fingerprints/match")
async def match_fingerprint(
    audio: UploadFile = File(...),
    profile: str = Form(default="user_default"),
):
    """上传音频，查找匹配的指纹和配置

    Returns:
        匹配结果（含 profile_id 和 confidence）
    """
    import tempfile

    from ..config import FeedbackConfig
    from ..feedback import AudioFingerprinter

    cfg = FeedbackConfig()

    # 保存音频到临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        fingerprinter = AudioFingerprinter(
            distance_method=cfg.fingerprint_distance_method,
            knn_k=cfg.fingerprint_knn_k,
            min_absolute_similarity=cfg.fingerprint_min_absolute_similarity,
            relative_margin=cfg.fingerprint_relative_margin,
        )

        # 拟合匹配器（从已有指纹库）
        fingerprinter.refit_matcher()

        fp = fingerprinter.extract(tmp_path)
        if fp is None:
            return FingerprintMatchResponse(matched=False)

        result = fingerprinter.find_similar(fp)
        if result:
            matched_profile, confidence = result
            return FingerprintMatchResponse(
                matched=True,
                profile_id=matched_profile,
                confidence=round(confidence, 4),
                audio_signature=fp.audio_signature,
            )

        return FingerprintMatchResponse(
            matched=False,
            audio_signature=fp.audio_signature,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.delete("/feedback/fingerprints/{fp_id}")
async def delete_fingerprint(fp_id: int):
    """删除指定指纹"""
    from ..config import FeedbackConfig
    from ..feedback import AudioFingerprinter

    cfg = FeedbackConfig()
    fingerprinter = AudioFingerprinter(
        distance_method=cfg.fingerprint_distance_method,
    )
    ok = fingerprinter.delete_by_id(fp_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Fingerprint not found: {fp_id}")
    return {"status": "ok", "deleted": fp_id}


# ---------------------------------------------------------------------------
# 健康度评分端点 (Phase 5.4)
# ---------------------------------------------------------------------------


@router.get("/feedback/health/{profile_name}")
async def get_health_trend(
    profile_name: str,
    limit: int = Query(default=20, le=100),
):
    """获取健康度趋势数据（用于前端趋势图）"""
    from ..config import FeedbackConfig
    from ..feedback import AudioFingerprinter

    cfg = FeedbackConfig()
    fingerprinter = AudioFingerprinter(
        distance_method=cfg.fingerprint_distance_method,
    )

    trend = fingerprinter.get_health_trend(profile_name, limit=limit)
    return {
        "profile_id": profile_name,
        "data_points": len(trend),
        "trend": [
            HealthTrendEntry(
                timestamp=e["timestamp"],
                health_before=e["health_before"],
                health_after=e["health_after"],
                shadow_mode=e["shadow_mode"],
                summary=e["summary"],
            )
            for e in trend
        ],
    }


@router.post("/feedback/health/compute")
async def compute_health(
    auto_subtitle: UploadFile = File(...),
    reference_subtitle: UploadFile = File(...),
):
    """上传自动版与修订版字幕，计算健康度评分

    Returns:
        HealthScoreDetail 包含综合评分和各子项得分
    """
    import tempfile

    from ..feedback.aligner import SubtitleAligner, parse_subtitle_file
    from ..feedback.health_scorer import health_score_result

    # 保存文件
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        auto_path = tmp_dir / (auto_subtitle.filename or "auto.srt")
        ref_path = tmp_dir / (reference_subtitle.filename or "reference.srt")

        auto_path.write_bytes(await auto_subtitle.read())
        ref_path.write_bytes(await reference_subtitle.read())

        # 解析
        auto_events = parse_subtitle_file(auto_path)
        manual_events = parse_subtitle_file(ref_path)

        if not auto_events or not manual_events:
            raise HTTPException(status_code=400, detail="Empty subtitle files")

        # 计算健康度
        result = health_score_result(auto_events, manual_events)

        return HealthScoreDetail(
            overall=result.overall,
            alignment_coverage=result.alignment_coverage,
            semantic_similarity=result.semantic_similarity,
            time_iou=result.time_iou,
            structure_consistency=result.structure_consistency,
            grade=result.grade,
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 影子模式端点 (Phase 5.4)
# ---------------------------------------------------------------------------


# 全局影子模式状态（会话级）
_shadow_evaluators: Dict[str, Any] = {}


@router.get("/feedback/shadow/{profile_name}")
async def get_shadow_status(profile_name: str):
    """获取影子模式状态"""
    from ..feedback import ShadowModeEvaluator

    evaluator = _shadow_evaluators.get(profile_name)
    if evaluator is None:
        return ShadowModeStatus(
            enabled=False,
            total_runs=0,
            reason="Shadow mode not active for this profile",
        )

    eval_result = evaluator.should_upgrade()
    return ShadowModeStatus(
        enabled=True,
        total_runs=evaluator.run_count,
        current_mean_health=eval_result.current_mean_health,
        shadow_mean_health=eval_result.shadow_mean_health,
        health_delta=eval_result.health_delta,
        recommendation=eval_result.recommendation,
        reason=eval_result.reason,
        runs=evaluator.to_dict().get("runs", []),
    )


@router.post("/feedback/shadow/{profile_name}/toggle")
async def toggle_shadow_mode(profile_name: str, body: ShadowModeToggleRequest):
    """启用/停用影子模式"""
    from ..config import FeedbackConfig
    from ..feedback import ShadowModeEvaluator

    if body.enabled:
        cfg = FeedbackConfig()
        evaluator = ShadowModeEvaluator(
            min_shadow_runs=cfg.shadow_min_runs,
            upgrade_threshold=cfg.shadow_upgrade_threshold,
            max_shadow_duration_days=cfg.shadow_max_duration_days,
        )
        _shadow_evaluators[profile_name] = evaluator
        return {
            "status": "ok",
            "enabled": True,
            "message": f"Shadow mode enabled for '{profile_name}'",
        }
    else:
        _shadow_evaluators.pop(profile_name, None)
        return {
            "status": "ok",
            "enabled": False,
            "message": f"Shadow mode disabled for '{profile_name}'",
        }


@router.post("/feedback/shadow/{profile_name}/record")
async def record_shadow_run(
    profile_name: str,
    health_current: float = Form(...),
    health_shadow: float = Form(...),
):
    """记录一次影子运行结果"""
    from ..feedback import ShadowModeEvaluator, ShadowRunResult

    evaluator = _shadow_evaluators.get(profile_name)
    if evaluator is None:
        cfg = FeedbackConfig()
        evaluator = ShadowModeEvaluator(
            min_shadow_runs=cfg.shadow_min_runs,
            upgrade_threshold=cfg.shadow_upgrade_threshold,
            max_shadow_duration_days=cfg.shadow_max_duration_days,
        )
        _shadow_evaluators[profile_name] = evaluator

    evaluator.add_run(ShadowRunResult(
        health_current=health_current,
        health_shadow=health_shadow,
    ))

    eval_result = evaluator.should_upgrade()
    return {
        "status": "ok",
        "total_runs": evaluator.run_count,
        "should_upgrade": eval_result.should_upgrade,
        "recommendation": eval_result.recommendation,
        "reason": eval_result.reason,
    }


# ---------------------------------------------------------------------------
# 冲突检测端点 (Phase 5.4)
# ---------------------------------------------------------------------------


@router.get("/feedback/conflicts/{profile_name}")
async def detect_conflicts(profile_name: str):
    """检测参数冲突（震荡）"""
    from ..config import FeedbackConfig
    from ..feedback import ConflictDetector, UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    profile = mgr.load(profile_name)
    history = profile.get("history", [])

    detector = ConflictDetector(window=5)
    reports = detector.detect_all_oscillations(history)

    return {
        "profile_id": profile_name,
        "conflicts": [
            ConflictInfo(
                param_path=r.param_path,
                is_oscillating=r.is_oscillating,
                oscillation_count=r.oscillation_count,
                severity=r.severity,
                recommended_action=r.recommended_action,
                possible_causes=r.possible_causes,
                suggested_actions=r.suggested_actions,
                entries=[
                    {
                        "timestamp": e.timestamp,
                        "direction": e.direction,
                        "delta": e.delta,
                        "summary": e.summary,
                    }
                    for e in r.entries
                ],
            )
            for r in reports
        ],
        "total": len(reports),
    }


@router.post("/feedback/conflicts/{profile_name}/resolve")
async def resolve_conflict(profile_name: str, body: ConflictResolutionRequest):
    """解决参数冲突"""
    from ..config import FeedbackConfig
    from ..feedback import ConflictDetector, UserProfileManager

    detector = ConflictDetector(window=5)
    mgr = UserProfileManager(FeedbackConfig())

    # 查找冲突
    profile = mgr.load(profile_name)
    history = profile.get("history", [])
    reports = detector.detect_all_oscillations(history)

    target = None
    for r in reports:
        if r.param_path == body.param_path:
            target = r
            break

    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"No oscillation detected for param: {body.param_path}",
        )

    result = detector.resolve(target, body.action)

    # 如果锁定，写入 locked_params 到 profile
    if body.action == "lock":
        locked = profile.get("locked_params", [])
        if body.param_path not in locked:
            locked.append(body.param_path)
        profile["locked_params"] = locked
        mgr.save(profile)

    return result


# ---------------------------------------------------------------------------
# 影响预估端点 (Phase 5.4)
# ---------------------------------------------------------------------------


@router.post("/feedback/impact/preview")
async def preview_impact(profile_name: str = "user_default"):
    """预览当前待调整参数的影响

    基于用户配置中最近的差异分析结果。
    """
    from ..config import FeedbackConfig
    from ..feedback import ImpactEstimator, UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    profile = mgr.load(profile_name)
    overrides = profile.get("overrides", {})
    history = profile.get("history", [])

    if not history:
        return {"impacts": [], "message": "尚无学习历史"}

    # 从最近的历史记录中获取归因
    last_entry = history[-1]
    latest_adjustments = last_entry.get("adjustments", {})
    if not latest_adjustments:
        return {"impacts": [], "message": "最近一次反馈无参数调整"}

    # 构建简易 ParamAdjustment 列表
    from ..feedback.diff_analyzer import ParamAdjustment

    adj_map = {}
    for param_path, (old_val, new_val) in latest_adjustments.items():
        delta = new_val - old_val
        adj_map[param_path] = ParamAdjustment(
            param_path=param_path,
            param_tier="medium_term",
            observed_value=abs(delta),
            confidence=0.8,
            learn_weight=1.0,
            direction="increase" if delta > 0 else "decrease",
            reason="Historical adjustment",
        )

    estimator = ImpactEstimator()
    impacts = estimator.estimate(adj_map, overrides)

    return {
        "profile_id": profile_name,
        "impacts": [
            ImpactPredictionInfo(
                param_path=ip.param_path,
                current_value=ip.current_value,
                new_value=ip.new_value,
                delta=ip.delta,
                delta_pct=ip.delta_pct,
                summary=ip.summary,
                avg_duration_change_pct=ip.avg_duration_change_pct,
                merge_frequency_change_pct=ip.merge_frequency_change_pct,
                split_frequency_change_pct=ip.split_frequency_change_pct,
                end_truncation_change_pct=ip.end_truncation_change_pct,
                total_line_count_change_pct=ip.total_line_count_change_pct,
                confidence_low=ip.confidence_low,
                confidence_high=ip.confidence_high,
            )
            for ip in impacts
        ],
        "total": len(impacts),
    }
