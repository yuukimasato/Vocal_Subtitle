"""Subtitle editing, export and task audio routes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from ..config import ConfigLoader
from ..mapping.time_mapper import SubtitleEvent
from .models import SubtitleEventResponse
from .runtime_state import state

logger = logging.getLogger(__name__)
router = APIRouter()

_TERMINAL_TASK_STATUSES = {"completed", "degraded_completed"}
_INPUT_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".webm"}


def _find_input_file(session_dir_str: str) -> Optional[str]:
    """Find the uploaded input audio in a session directory.

    The helper is shared by download and streaming routes.  Session metadata
    is user-controlled only through the upload workflow, but keeping the
    lookup bounded to one directory avoids accidentally scanning the cache.
    """
    if not session_dir_str:
        return None
    task_dir = Path(session_dir_str)
    if not task_dir.is_dir():
        return None
    try:
        candidates = sorted(task_dir.iterdir(), key=lambda item: item.name)
    except OSError:
        return None
    for candidate in candidates:
        if (
            candidate.is_file()
            and candidate.name.startswith("input")
            and candidate.suffix.lower() in _INPUT_AUDIO_SUFFIXES
        ):
            return str(candidate)
    return None


def _input_path_from_result(result: dict) -> Optional[str]:
    """Return the original input audio path recorded by an offline (CLI) run.

    CLI 任务不经过 WebUI 上传，没有 session 目录可查，只能依赖
    result_json 中记录的 input_path 定位原始音频。
    """
    raw = result.get("input_path")
    if raw and Path(raw).is_file():
        return str(raw)
    return None


@router.get("/subtitle/{task_id}", response_model=List[SubtitleEventResponse])
async def get_subtitles(task_id: str):
    """获取任务的字幕事件列表"""
    task = state.task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if task["status"] not in _TERMINAL_TASK_STATUSES:
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


def _load_completed_subtitle_task(task_id: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a completed task from memory, falling back to persisted history."""
    task = state.task_store.get(task_id)
    if not task:
        hist_task = state.task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                result = json.loads(hist_task["result_json"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise HTTPException(status_code=404, detail=f"Task not found: {task_id}") from exc
            task = {
                "task_id": task_id,
                "status": result.get("status", "completed"),
                "result": result,
            }
            state.task_store[task_id] = task
        else:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if task.get("status") not in _TERMINAL_TASK_STATUSES:
        raise HTTPException(status_code=400, detail="Task not completed yet")
    result = task.get("result", {})
    if not isinstance(result, dict) or not isinstance(result.get("events"), list):
        raise HTTPException(status_code=404, detail=f"Task has no subtitle events")
    return task, result


@router.get("/subtitle/{task_id}/export")
async def export_subtitle(
    task_id: str,
    fmt: str = Query(default="srt", alias="format"),
):
    """导出自幕文件为指定格式（支持内存任务和历史任务）"""
    raw_events = []

    # 先尝试内存中的任务
    task = state.task_store.get(task_id)
    if task and task.get("status") in _TERMINAL_TASK_STATUSES:
        result = task.get("result", {})
        raw_events = result.get("events", [])

    # 如果内存中没有，尝试历史记录
    if not raw_events:
        hist_task = state.task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                r = json.loads(hist_task["result_json"])
                raw_events = r.get("events", [])
            except (json.JSONDecodeError, TypeError):
                pass

    if not raw_events:
        raise HTTPException(status_code=404, detail="Task not found or has no events")

    # 重建 SubtitleEvent 对象
    events = [_subtitle_event_from_payload(e) for e in raw_events]

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

    # 验证格式（提前捕获不支持的格式，避免 500）
    SUPPORTED_FORMATS = {"srt", "vtt", "ass"}
    normalized_fmt = fmt.lower()
    if normalized_fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported subtitle format: '{fmt}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    subtitle_text = builder.build_to_string(events, fmt=normalized_fmt)

    media_types = {
        "srt": "text/plain; charset=utf-8",
        "vtt": "text/vtt; charset=utf-8",
        "ass": "text/plain; charset=utf-8",
    }

    return PlainTextResponse(
        content=subtitle_text,
        media_type=media_types.get(normalized_fmt, "text/plain"),
        headers={
            "Content-Disposition": f'attachment; filename="subtitle.{normalized_fmt}"'
        },
    )


# ---------------------------------------------------------------------------
# 分离音频导出
# ---------------------------------------------------------------------------


@router.get("/tasks/{task_id}/audio")
async def download_separated_audio(
    task_id: str,
    type: str = Query(default="vocals", description="vocals、accompaniment 或 input"),
):
    """下载人声分离产出的音频文件

    Args:
        task_id: 任务 ID
        type: 音频类型 — 'vocals'（人声）或 'accompaniment'（背景声/伴奏）
    """
    if type not in ("vocals", "accompaniment", "input"):
        raise HTTPException(status_code=400, detail="type must be 'vocals', 'accompaniment' or 'input'")

    # 先查内存中的任务
    task = state.task_store.get(task_id)
    file_path = None

    if task and task.get("result"):
        if type == "input":
            session_dir = task.get("session_dir", "")
            if session_dir:
                file_path = _find_input_file(session_dir)
            if not file_path:
                file_path = _find_input_file(str(state.upload_dir / task_id))
        else:
            file_path = task["result"].get(f"{type}_path")

    # 内存中找不到，查持久化历史
    if not file_path:
        hist_task = state.task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                r = json.loads(hist_task["result_json"])
                if type == "input":
                    session_dir = str(Path(r.get("subtitle_path", "")).parent) if r.get("subtitle_path") else ""
                    file_path = _find_input_file(session_dir)
                    if not file_path:
                        file_path = _find_input_file(str(state.upload_dir / task_id))
                    if not file_path:
                        file_path = _input_path_from_result(r)
                else:
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
    type_label = {"vocals": "人声", "accompaniment": "背景声", "input": "原始音频"}[type]
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

    # 先查内存中的任务
    task = state.task_store.get(task_id)
    if task and task.get("result"):
        if type == "input":
            # 优先从 session_dir 查找
            session_dir = task.get("session_dir", "")
            if session_dir:
                file_path = _find_input_file(session_dir)
            # 回退到旧路径格式
            if not file_path:
                file_path = _find_input_file(str(state.upload_dir / task_id))
        else:
            file_path = task["result"].get(f"{type}_path")

    # 内存中找不到，查持久化历史
    if not file_path:
        hist_task = state.task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                r = json.loads(hist_task["result_json"])
                if type == "input":
                    # 尝试从 session_dir 查找
                    if task and task.get("session_dir"):
                        file_path = _find_input_file(task["session_dir"])
                    # 回退：从 subtitle_path 推断 session_dir
                    if not file_path:
                        sp = r.get("subtitle_path", "")
                        inferred_dir = str(Path(sp).parent) if sp else ""
                        if inferred_dir:
                            file_path = _find_input_file(inferred_dir)
                    if not file_path:
                        file_path = _find_input_file(str(state.upload_dir / task_id))
                    # 回退：CLI 任务直接记录了原始音频路径
                    if not file_path:
                        file_path = _input_path_from_result(r)
                else:
                    file_path = r.get(f"{type}_path")
            except (json.JSONDecodeError, TypeError):
                pass

    # Fallback: 如果请求 vocals 但找不到，尝试 input
    if not file_path and type == "vocals":
        if task and task.get("session_dir"):
            file_path = _find_input_file(task["session_dir"])
        # 回退：从历史记录的 subtitle_path 推断 session_dir
        if not file_path:
            hist_task = state.task_history.get(task_id)
            if hist_task and hist_task.get("result_json"):
                try:
                    r = json.loads(hist_task["result_json"])
                    sp = r.get("subtitle_path", "")
                    inferred_dir = str(Path(sp).parent) if sp else ""
                    if inferred_dir:
                        file_path = _find_input_file(inferred_dir)
                    if not file_path:
                        file_path = _input_path_from_result(r)
                except (json.JSONDecodeError, TypeError):
                    pass
        if not file_path:
            file_path = _find_input_file(str(state.upload_dir / task_id))

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

    from fastapi.responses import FileResponse

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
    task = state.task_store.get(task_id)
    file_path = None
    input_name = "subtitle"

    if task and task.get("result"):
        if version == "llm":
            file_path = task["result"].get("llm_subtitle_path")
        else:
            file_path = task["result"].get("clean_subtitle_path")
        if not file_path:
            file_path = task["result"].get("subtitle_path")
        input_name = task.get("input_file_name", "subtitle") if hasattr(task, "get") else "subtitle"

    # 内存中找不到，查持久化历史
    if not file_path:
        hist_task = state.task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                r = json.loads(hist_task["result_json"])
                if version == "llm":
                    file_path = r.get("llm_subtitle_path")
                else:
                    file_path = r.get("clean_subtitle_path")
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
