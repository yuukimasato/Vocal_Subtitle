"""Application service for asynchronous WebUI pipeline tasks.

The HTTP routes do not construct or own ``Pipeline`` instances.  This module
owns the background-task lifecycle and keeps the historical task store,
history database, WebSocket events, and persistence hooks unchanged.
"""

from __future__ import annotations

import json
import hashlib
import logging
import asyncio
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from ..asr.funasr_manager import FunASRPrepareError, ensure_funasr_ready
from ..config import ConfigLoader
from ..utils.audio_utils import AudioUtils
from ..utils.file_hasher import compute_config_hash, compute_file_hash
from ..utils.session_manager import SessionManager
from ..pipeline import Pipeline
from .runtime_state import state
from .websocket import ws_manager
from ..contracts.common import CONTRACT_VERSION

logger = logging.getLogger(__name__)


def _pipeline_class() -> Type[Pipeline]:
    """Resolve the legacy API override before falling back to the real class."""
    api = sys.modules.get("vocal_subtitle.webui.api")
    candidate = getattr(api, "Pipeline", None)
    if candidate is not None:
        return candidate
    return Pipeline


def _serialize_events(events: List[Any]) -> List[Dict[str, Any]]:
    serialized = []
    for event in events:
        payload = {
            "index": event.index,
            "start": event.start,
            "end": event.end,
            "text": event.text,
            "original_text": event.original_text,
            "speaker_id": event.speaker_id,
            "speaker_label": event.speaker_label,
            "physical_start": event.physical_start,
            "physical_end": event.physical_end,
            "physical_spans": event.physical_spans,
            "physical_bin_id": event.physical_bin_id,
            "physical_bin_start": event.physical_bin_start,
            "physical_bin_end": event.physical_bin_end,
            "time_source": event.time_source,
            "source_word_ids": event.source_word_ids,
            "speaker_status": event.speaker_status,
            "speaker_source": event.speaker_source,
            "speaker_confidence": event.speaker_confidence,
            "speaker_model": event.speaker_model,
            "speaker_repair_reason": event.speaker_repair_reason,
            "alignment_warning": event.alignment_warning,
            "revision_trace": event.revision_trace,
        }
        # 词级时间戳（可选字段，供 review-manifest-v1 出处）：相对时间原样保留
        if getattr(event, "words", None):
            payload["words"] = [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                    "confidence": word.confidence,
                }
                for word in event.words
            ]
        serialized.append(payload)
    return serialized


def _persistence_manager():
    """Resolve the compatibility accessor at call time for old monkeypatches."""
    from . import routes_history

    return routes_history._get_persistence_mgr()


def _global_toggle_config():
    """全局"应用学习参数"开关的配置源（default 模板；载入失败回退内置默认）。"""
    try:
        return ConfigLoader().load_profile("default").feedback
    except Exception:
        return None


def _apply_active_profile_overrides(config):
    """按 feedback.apply_overrides_on_run 合并 active_profile 的学习参数（D38）。

    开关判定：场景模板 YAML 的 feedback.apply_overrides_on_run 显式开启，
    或用户级开关文件（8613 反馈档案工作区写入）开启。默认两者皆关，
    原样返回输入配置，保持既有行为零变化。
    """
    from ..feedback.user_profile import (
        UserProfileManager,
        load_apply_overrides_on_run,
    )

    feedback_cfg = config.feedback
    if not getattr(feedback_cfg, "apply_overrides_on_run", False):
        # 开关文件位置全局唯一：按 default 模板解析（与 8613 开关端点同源），
        # 不随场景模板的 user_profile_dir 漂移，避免"写 A 读 B"静默失效
        if not load_apply_overrides_on_run(_global_toggle_config()):
            return config
    profile_name = feedback_cfg.active_profile or "user_default"
    try:
        profile = UserProfileManager(feedback_cfg).load(profile_name)
    except Exception as exc:
        logger.warning("Failed to load user profile '%s' for overrides: %s", profile_name, exc)
        return config
    learned = profile.get("overrides") or {}
    if not learned:
        return config
    logger.info(
        "Applying learned overrides from profile '%s' (%d top-level group(s), apply_overrides_on_run=on)",
        profile_name,
        len(learned),
    )
    return UserProfileManager.merge_with_base(config, learned)


def _build_run_config(loader: ConfigLoader, profile: str, overrides: Dict[str, Any]):
    """构建一次任务运行的最终配置。

    顺序：场景模板 → （可选）active_profile 学习参数 → 单次任务显式 overrides，
    显式 overrides 后应用以保证用户当次意图优先于学习参数（D38）。
    """
    config = _apply_active_profile_overrides(loader.load_profile(profile))
    return loader.merge_with_overrides(config, **overrides)


def _ensure_task_entry(task_id: str, session_dir: Optional[Path] = None) -> Dict[str, Any]:
    """取回任务条目；条目已从 store 消失时重建最小结构。

    运行线程与 HTTP 侧（清空历史等清理操作）并发访问 task_store，条目可能在
    任务执行中途被移除。这里重建而不是让 KeyError 把整个成功运行记成 failed。
    """
    entry = state.task_store.get(task_id)
    if entry is None:
        entry = {
            "task_id": task_id,
            "status": "running",
            "progress": None,
            "result": None,
            "error": None,
            "from_cache": False,
        }
        if session_dir is not None:
            entry["session_dir"] = str(session_dir)
        state.task_store[task_id] = entry
        logger.warning(
            "Task %s entry missing from store (cleared while running?); recreated",
            task_id,
        )
    return entry


def run_pipeline_in_thread(
    task_id: str,
    input_path: Path,
    output_path: Path,
    profile: str,
    output_format: str,
    skip_separation: bool,
    overrides: Dict[str, Any],
    session_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Run one pipeline task and publish the legacy WebUI task contract.

    ``progress_callback`` 允许调用方包装默认的 WebSocket 进度回调
    （内部学习任务借此在每条进度事件上注入"学习"标记与场景标签，D28）。
    """
    try:
        from ..config import ConfigLoader

        loader = ConfigLoader()
        config = _build_run_config(loader, profile, overrides)
        pipeline = _pipeline_class()(config)
        if progress_callback is None:
            progress_callback = ws_manager.create_progress_callback(task_id)

        _ensure_task_entry(task_id, session_dir)["status"] = "running"
        ws_manager.broadcast_from_thread(
            task_id,
            {
                "type": "stage_start",
                "stage": "pipeline",
                "total": 1,
                "description": "Pipeline 启动",
            },
        )

        result = pipeline.run(
            input_path=input_path,
            output_path=output_path,
            output_format=output_format,
            progress_callback=progress_callback,
            skip_separation=skip_separation,
            task_id=task_id,
            session_dir=session_dir,
        )
        events = _serialize_events(result.get("events", []))
        stats = result["stats"]
        from ..utils.session_manager import create_run_id

        run_id = stats.run_id or create_run_id(task_id)
        stats.run_id = run_id
        stats.task_id = task_id
        task_contract_version = CONTRACT_VERSION
        artifacts = {
            name: str(path)
            for name, path in {
                "input": input_path,
                "subtitle": result.get("subtitle_path"),
                "subtitle_clean": result.get("clean_subtitle_path"),
                "subtitle_llm": result.get("llm_subtitle_path"),
                "vocals": result.get("vocals_path"),
                "accompaniment": result.get("accompaniment_path"),
            }.items()
            if path
        }
        diagnostics = result.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = stats.to_dict()
        task_result = {
            "contract_version": task_contract_version,
            "task_id": task_id,
            "run_id": run_id,
            "status": stats.status,
            "input_path": str(input_path),
            "subtitle_path": str(result["subtitle_path"]),
            "clean_subtitle_path": str(result["clean_subtitle_path"]) if result.get("clean_subtitle_path") else None,
            "llm_subtitle_path": str(result["llm_subtitle_path"]) if result.get("llm_subtitle_path") else None,
            "stats": stats.to_dict(),
            "events": events,
            "from_cache": result.get("from_cache", False),
            "segment_count": stats.segment_count,
            "subtitle_count": stats.subtitle_count,
            "quality_status": stats.quality_status,
            "error_category": stats.error_category or None,
            "diagnostics_complete": stats.diagnostics_complete,
            "requested_engine": stats.requested_engine,
            "selected_engine": stats.selected_engine,
            "final_engine": stats.final_engine,
            "detected_language": stats.detected_language,
            "fallback_reason": stats.fallback_reason or None,
            "vocals_path": result.get("vocals_path"),
            "accompaniment_path": result.get("accompaniment_path"),
            "artifacts": artifacts,
            "diagnostics": diagnostics,
        }

        _ensure_task_entry(task_id, session_dir).update({
            "status": stats.status,
            "run_id": run_id,
            "result": task_result,
        })
        state.task_history.update(
            task_id,
            run_id=run_id,
            status=stats.status,
            result_json=json.dumps(task_result, default=str),
            total_duration_seconds=stats.duration_seconds,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        ws_manager.broadcast_from_thread(task_id, {"type": "complete", "result": task_result})
        ws_manager.store_task_result(task_id, task_result)
        try:
            _persistence_manager().persist_task(task_id, task_result)
        except Exception as exc:
            logger.warning("Auto-persist failed for task %s: %s", task_id, exc)
    except Exception as exc:
        logger.exception("Pipeline task %s failed", task_id)
        error_msg = str(exc)
        if task_id in state.task_store:
            state.task_store[task_id].update({"status": "failed", "error": error_msg})
        else:
            logger.warning("Task %s was already removed from store before error handler", task_id)
        state.task_history.update(
            task_id,
            status="failed",
            error=error_msg,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        ws_manager.broadcast_from_thread(task_id, {"type": "error", "message": error_msg})


class PipelineTaskService:
    """Explicit service boundary for task execution."""

    run_in_thread = staticmethod(run_pipeline_in_thread)

    async def submit(
        self,
        contents: bytes,
        original_filename: str,
        *,
        profile: str = "default",
        output_format: str = "srt",
        skip_separation: bool = False,
        overrides_text: str = "{}",
        thread_target: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        """Validate, cache-check, persist, and enqueue one uploaded input."""
        task_id = str(uuid.uuid4())[:8]
        session_mgr = SessionManager(state.upload_dir)
        session_key = SessionManager.compute_session_key_from_bytes(contents)
        session_dir = session_mgr.session_dir(session_key)
        existing_metadata = session_mgr.read_metadata(session_dir)
        if existing_metadata:
            logger.info(
                "Session dir already exists for %s (hash=%s), previously processed at %s",
                original_filename,
                session_key,
                existing_metadata.get("processed_at", "unknown"),
            )
        session_dir.mkdir(parents=True, exist_ok=True)
        input_path = session_dir / f"input{Path(original_filename).suffix or '.wav'}"
        try:
            input_path.write_bytes(contents)
        except Exception as exc:
            raise ValueError(f"Failed to save file: {exc}") from exc

        pipeline_input = input_path
        if AudioUtils.is_video_file(input_path):
            try:
                pipeline_input = AudioUtils.extract_audio_from_video(input_path, session_dir)
            except Exception as exc:
                raise ValueError(f"Failed to extract audio from video: {exc}") from exc

        try:
            overrides = json.loads(overrides_text)
        except json.JSONDecodeError:
            overrides = {}
        loader = ConfigLoader()
        config = _build_run_config(loader, profile, overrides)
        if config.asr.engine == "funasr":
            try:
                prepare = _legacy_callable("ensure_funasr_ready", ensure_funasr_ready)
                await asyncio.to_thread(prepare, config.asr.model)
            except FunASRPrepareError as exc:
                raise PipelineSubmissionError(503, str(exc)) from exc

        file_hash = compute_file_hash(pipeline_input)
        config_hash = compute_config_hash(config)
        file_size = pipeline_input.stat().st_size
        try:
            session_mgr.write_metadata(
                session_dir,
                original_filename=original_filename,
                input_sha256=file_hash,
                profile=profile,
                config_hash=config_hash,
                task_id=task_id,
            )
        except Exception as exc:
            logger.warning("Failed to write session metadata: %s", exc)

        output_path = session_dir / f"output.{output_format}"
        if config.cache.enabled and config.cache.full_pipeline_cache:
            cached_task = state.task_history.find_by_hash(file_hash, config_hash)
            if cached_task and cached_task.get("result_json"):
                try:
                    cached_result = json.loads(cached_task["result_json"])
                    cached_stats = cached_result.get("stats") or {}
                    cached_run_id = cached_result.get("run_id") or cached_stats.get("run_id", "")
                    cached_result.setdefault("contract_version", CONTRACT_VERSION)
                    cached_result.setdefault("task_id", task_id)
                    cached_result.setdefault("run_id", cached_run_id)
                    cached_result.setdefault("input_path", str(pipeline_input))
                    if not isinstance(cached_result.get("artifacts"), dict):
                        cached_result["artifacts"] = {}
                    cached_result["artifacts"].setdefault("input", str(pipeline_input))
                    if cached_result.get("subtitle_path"):
                        cached_result["artifacts"].setdefault("subtitle", str(cached_result["subtitle_path"]))
                    cached_result.setdefault("diagnostics", cached_stats if isinstance(cached_stats, dict) else {})
                    cached_subtitle_path = Path(cached_result.get("subtitle_path", ""))
                    if cached_subtitle_path.exists():
                        output_path.write_text(cached_subtitle_path.read_text(encoding="utf-8"), encoding="utf-8")
                        state.task_store[task_id] = {
                            "task_id": task_id,
                            "status": cached_result.get("status", "completed"),
                            "run_id": cached_result.get("run_id") or (cached_result.get("stats") or {}).get("run_id", ""),
                            "result": cached_result,
                            "from_cache": True,
                            "input_file_name": original_filename,
                            "session_dir": str(session_dir),
                        }
                        return {
                            "task_id": task_id,
                            "status": cached_result.get("status", "completed"),
                            "from_cache": True,
                        }
                except Exception as exc:
                    logger.warning("Failed to restore cached result: %s", exc)

        state.task_store[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": None,
            "result": None,
            "error": None,
            "from_cache": False,
            "input_file_name": original_filename,
            "session_dir": str(session_dir),
        }
        try:
            state.task_history.create(
                task_id=task_id,
                file_name=original_filename,
                file_hash=file_hash,
                file_size=file_size,
                profile=profile,
                config=config,
            )
        except Exception as exc:
            logger.warning("Failed to create history record: %s", exc)

        try:
            ws_manager.set_main_loop(asyncio.get_running_loop())
        except RuntimeError:
            pass
        target = thread_target or run_pipeline_in_thread
        threading.Thread(
            target=target,
            args=(
                task_id,
                pipeline_input,
                output_path,
                profile,
                output_format,
                skip_separation,
                overrides,
                session_dir,
            ),
            daemon=True,
        ).start()
        return {"task_id": task_id, "status": "pending"}


class PipelineSubmissionError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _legacy_callable(name: str, default: Callable[..., Any]) -> Callable[..., Any]:
    api = sys.modules.get("vocal_subtitle.webui.api")
    return getattr(api, name, default)


# ---------------------------------------------------------------------------
# 内部学习任务（冷重跑异步化，D28）
#
# 缺 task_id 且需要跑管线的 V3/V4 学习请求（存量字幕学习 / 从零打轴）不再
# 同步阻塞 HTTP，而是作为内部任务提交进现有任务系统：任务列表 / 详情 /
# WebSocket 进度推送带"学习"标记（task_type="learn"）与场景标签，
# 复用普通任务队列与进度机制不插队；管线完成后自动执行与同步路径同一套
# "对齐→diff→入库"逻辑，学习报告挂在任务详情（learn_report 字段）。
# ---------------------------------------------------------------------------

LEARN_TASK_TYPE = "learn"

# D28：异步化仅作用于冷重跑场景（V3 存量字幕学习 / V4 从零打轴）；
# V1/V2 走 task_id 引用同步路径，不带场景的旧客户端音频路径保持原行为。
ASYNC_LEARN_SCENARIOS = ("existing-subtitle", "from-scratch-timing")

# 进程内学习任务登记表：幂等键 → task_id，配合 task_history 哈希查询实现
# "同一音频+同一参考字幕+同一场景"重复提交去重
_active_learn_tasks: Dict[str, str] = {}
_learn_registry_lock = threading.Lock()


def _learn_dedupe_hash(base_config_hash: str, reference_hash: str, scenario: str) -> str:
    """学习请求幂等键：管线配置哈希 + 参考字幕哈希 + 场景标签。

    存入 task_history 的 config_hash 列，使学习任务的幂等去重直接复用
    现成的 (input_file_hash, config_hash) 索引，不另建存储；
    与普通任务的全管线缓存键（纯配置哈希）天然不冲突。
    """
    raw = f"learn-task-v1|{base_config_hash}|{reference_hash}|{scenario}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _learn_envelope(task_id: str, status: str, scenario: str, deduplicated: bool) -> Dict[str, Any]:
    """学习任务提交响应（不阻塞请求，任务标识供进度订阅与详情查询）"""
    return {
        "task_id": task_id,
        "status": status,
        "task_type": LEARN_TASK_TYPE,
        "scenario": scenario,
        "deduplicated": deduplicated,
    }


def _finalize_learn_phase(task_id: str, report: Dict[str, Any], scenario: str) -> None:
    """学习阶段收尾：报告挂任务详情并广播；失败则任务状态与错误可见"""
    if report.get("status") != "ok":
        _mark_learn_failed(task_id, str(report.get("message") or "学习阶段失败"), scenario)
        return
    task = state.task_store.get(task_id)
    if task is not None:
        task["learn_report"] = report
    # 报告挂任务详情：合并进任务历史 result_json 的 learn_report 字段
    record = state.task_history.get(task_id)
    if record is not None:
        try:
            result = json.loads(record.get("result_json") or "{}")
        except (TypeError, ValueError):
            result = {}
        if isinstance(result, dict):
            result["learn_report"] = report
            state.task_history.update(task_id, result_json=json.dumps(result, default=str))
    ws_manager.broadcast_from_thread(
        task_id,
        {
            "type": "learn_complete",
            "task_type": LEARN_TASK_TYPE,
            "scenario": scenario,
            "report": report,
        },
    )


def _mark_learn_failed(task_id: str, message: str, scenario: str) -> None:
    """学习阶段失败：任务状态与错误在任务存储/历史/推送三处可见"""
    from ..utils.task_history import ErrorCategory

    logger.warning("Learn phase of task %s failed: %s", task_id, message)
    task = state.task_store.get(task_id)
    if task is not None:
        task.update({"status": "failed", "error": message})
    try:
        state.task_history.update(
            task_id,
            status="failed",
            error=message,
            error_category=ErrorCategory.UNRECOVERABLE_FAILURE,
        )
    except Exception as exc:
        logger.warning("Failed to record learn failure for task %s: %s", task_id, exc)
    ws_manager.broadcast_from_thread(
        task_id,
        {
            "type": "error",
            "stage": "learn",
            "task_type": LEARN_TASK_TYPE,
            "scenario": scenario,
            "message": message,
        },
    )


def run_learn_task_in_thread(
    task_id: str,
    input_path: Path,
    reference_path: Path,
    output_path: Path,
    profile: str,
    scenario: str = "",
    feedback_profile: str = "user_default",
    consent: str = "anonymous",
    session_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """内部学习任务线程：管线冷重跑 → 对齐 → diff → 入库（D28）。

    阶段一复用 run_pipeline_in_thread（同一套进度推送/任务历史/落库行为，
    普通任务队列不插队）；阶段二通过 task_id 引用本任务刚落库的 events，
    走与 V1/V2 同步路径完全相同的 FeedbackLearningService.learn()。
    """
    from .feedback_services import FeedbackLearningService

    overrides = overrides or {}
    try:
        ws_manager.broadcast_from_thread(
            task_id,
            {
                "type": "learn_started",
                "stage": "pipeline",
                "task_type": LEARN_TASK_TYPE,
                "scenario": scenario,
                "message": "学习任务开始：先重跑管线生成对照基线",
            },
        )
        base_callback = ws_manager.create_progress_callback(task_id)

        def marked_callback(event: Dict[str, Any]) -> None:
            # 进度推送注入"学习"标记与场景标签（契约只追加可选字段）
            payload = dict(event)
            payload.setdefault("task_type", LEARN_TASK_TYPE)
            payload.setdefault("scenario", scenario)
            base_callback(payload)

        run_pipeline_in_thread(
            task_id,
            input_path,
            output_path,
            profile,
            "srt",
            False,
            overrides,
            session_dir,
            progress_callback=marked_callback,
        )
        status = (state.task_store.get(task_id) or {}).get("status")
        if status in ("failed", "cancelled"):
            # 管线失败/取消：状态与错误已由 run_pipeline_in_thread 落库并广播
            return

        # 阶段二：与同步路径同一套"对齐→diff→入库"，基线=本任务已存 events
        report = FeedbackLearningService().learn(
            None,
            reference_path,
            task_id=task_id,
            scenario=scenario,
            profile=profile,
            feedback_profile=feedback_profile,
            run_pipeline_first=True,
            dry_run=False,
            consent=consent,
        )
        _finalize_learn_phase(task_id, report, scenario)
    except Exception as exc:
        logger.exception("Learn task %s failed", task_id)
        _mark_learn_failed(task_id, f"学习任务异常: {exc}", scenario)


async def submit_learn_task(
    audio_contents: bytes,
    original_filename: str,
    reference_contents: bytes,
    reference_filename: str,
    *,
    scenario: str,
    profile: str = "default",
    feedback_profile: str = "user_default",
    consent: str = "anonymous",
    thread_target: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """提交内部学习任务：不阻塞请求，返回任务标识（D28）。

    幂等/去重（同一音频+同一参考字幕+同一场景）：
    - 进程内登记表拦截进行中的重复提交（快速双击）；
    - task_history 按 (input_file_hash, 幂等键) 查已完成学习任务，命中即
      幂等返回原任务（不重复跑管线、不重复入库 D2 样本）；失败任务不拦截
      （允许重试）。
    - 管线已跑过（全管线缓存命中且会话产物可用）时不进队列，直接按
      task_id 引用路径同步秒级完成学习（D28"未命中缓存"才冷重跑）。
    """
    from .feedback_services import FeedbackLearningService, TaskBaselineError

    session_mgr = SessionManager(state.upload_dir)
    session_key = SessionManager.compute_session_key_from_bytes(audio_contents)
    session_dir = session_mgr.session_dir(session_key)
    session_dir.mkdir(parents=True, exist_ok=True)
    input_path = session_dir / f"input{Path(original_filename).suffix or '.wav'}"
    try:
        input_path.write_bytes(audio_contents)
    except Exception as exc:
        raise ValueError(f"Failed to save file: {exc}") from exc

    pipeline_input = input_path
    if AudioUtils.is_video_file(input_path):
        try:
            pipeline_input = AudioUtils.extract_audio_from_video(input_path, session_dir)
        except Exception as exc:
            raise ValueError(f"Failed to extract audio from video: {exc}") from exc

    ref_hash = hashlib.sha256(reference_contents).hexdigest()
    ref_suffix = Path(reference_filename).suffix.lower() or ".srt"
    reference_path = session_dir / f"learn_reference_{ref_hash[:12]}{ref_suffix}"
    try:
        reference_path.write_bytes(reference_contents)
    except Exception as exc:
        raise ValueError(f"Failed to save reference subtitle: {exc}") from exc

    loader = ConfigLoader()
    config = _build_run_config(loader, profile, {})
    audio_hash = compute_file_hash(pipeline_input)
    base_config_hash = compute_config_hash(config)
    dedupe_hash = _learn_dedupe_hash(base_config_hash, ref_hash, scenario)

    # 全管线缓存命中 → 不进队列，同步秒级完成学习（与 task_id 引用同路径）。
    # 幂等与冷重跑路径同口径：先查学习登记（in-flight/已完成），避免重复提交重复入库 D2
    cached = state.task_history.find_by_hash(audio_hash, base_config_hash)
    if cached and cached.get("result_json"):
        with _learn_registry_lock:
            inflight = _active_learn_tasks.get(dedupe_hash)
            inflight_status = (state.task_store.get(inflight) or {}).get("status") if inflight else None
            if inflight is not None and inflight_status in ("pending", "running"):
                return _learn_envelope(inflight, inflight_status, scenario, True)
            done_sync = state.task_history.find_by_hash(audio_hash, dedupe_hash, task_type=LEARN_TASK_TYPE)
        if done_sync:
            # 同一学习请求已完成：幂等返回原任务，不重复学/入库
            return _learn_envelope(done_sync["id"], str(done_sync.get("status") or "completed"), scenario, True)
        try:
            report = FeedbackLearningService().learn(
                pipeline_input,
                reference_path,
                task_id=cached["id"],
                scenario=scenario,
                profile=profile,
                feedback_profile=feedback_profile,
                run_pipeline_first=True,
                dry_run=False,
                consent=consent,
            )
            logger.info(
                "Learn request hit pipeline cache (task %s); answered synchronously",
                cached["id"],
            )
            # 补登一条学习任务记录：后续同请求经上方幂等查询直接命中
            # （并发同请求在秒级窗口内可能双双放行，与冷重跑路径的进程内登记口径一致）
            try:
                state.task_history.create(
                    task_id=str(uuid.uuid4())[:8],
                    file_name=original_filename,
                    file_hash=audio_hash,
                    file_size=pipeline_input.stat().st_size,
                    profile=profile,
                    config=config,
                    task_type=LEARN_TASK_TYPE,
                    scenario=scenario,
                    config_hash=dedupe_hash,
                )
            except Exception as exc:
                logger.warning("Failed to record cache-hit learn for idempotency: %s", exc)
            return report
        except TaskBaselineError as exc:
            # 缓存记录的会话产物已清理：回退为冷重跑内部任务
            logger.info("Cached task %s unusable for learn (%s); falling back to rerun", cached["id"], exc)

    if config.asr.engine == "funasr":
        try:
            prepare = _legacy_callable("ensure_funasr_ready", ensure_funasr_ready)
            await asyncio.to_thread(prepare, config.asr.model)
        except FunASRPrepareError as exc:
            raise PipelineSubmissionError(503, str(exc)) from exc

    file_size = pipeline_input.stat().st_size
    with _learn_registry_lock:
        existing = _active_learn_tasks.get(dedupe_hash)
        if existing is not None:
            existing_status = (state.task_store.get(existing) or {}).get("status")
            if existing_status in ("pending", "running"):
                return _learn_envelope(existing, existing_status, scenario, True)
            _active_learn_tasks.pop(dedupe_hash, None)
        done = state.task_history.find_by_hash(audio_hash, dedupe_hash, task_type=LEARN_TASK_TYPE)
        if done:
            # 同一学习请求已完成：幂等返回原任务，不重复跑管线/入库
            return _learn_envelope(done["id"], str(done.get("status") or "completed"), scenario, True)

        task_id = str(uuid.uuid4())[:8]
        state.task_store[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": None,
            "result": None,
            "error": None,
            "from_cache": False,
            "input_file_name": original_filename,
            "session_dir": str(session_dir),
            "task_type": LEARN_TASK_TYPE,
            "scenario": scenario,
        }
        try:
            state.task_history.create(
                task_id=task_id,
                file_name=original_filename,
                file_hash=audio_hash,
                file_size=file_size,
                profile=profile,
                config=config,
                task_type=LEARN_TASK_TYPE,
                scenario=scenario,
                config_hash=dedupe_hash,
            )
        except Exception as exc:
            logger.warning("Failed to create learn task history record: %s", exc)
        if session_mgr.read_metadata(session_dir) is None:
            try:
                session_mgr.write_metadata(
                    session_dir,
                    original_filename=original_filename,
                    input_sha256=audio_hash,
                    profile=profile,
                    config_hash=dedupe_hash,
                    task_id=task_id,
                )
            except Exception as exc:
                logger.warning("Failed to write session metadata: %s", exc)
        _active_learn_tasks[dedupe_hash] = task_id

    try:
        ws_manager.set_main_loop(asyncio.get_running_loop())
    except RuntimeError:
        pass
    target = thread_target or run_learn_task_in_thread
    try:
        threading.Thread(
            target=target,
            args=(
                task_id,
                pipeline_input,
                reference_path,
                # 文件名带 task_id：同音频不同参考字幕的学习任务并发时产物不互相覆盖
                session_dir / f"learn_{task_id}.srt",
                profile,
                scenario,
                feedback_profile,
                consent,
                session_dir,
            ),
            daemon=True,
        ).start()
    except Exception:
        # 线程启动失败：清除幂等登记，允许客户端重试
        with _learn_registry_lock:
            if _active_learn_tasks.get(dedupe_hash) == task_id:
                _active_learn_tasks.pop(dedupe_hash, None)
        raise
    return _learn_envelope(task_id, "pending", scenario, False)


__all__ = [
    "ASYNC_LEARN_SCENARIOS",
    "LEARN_TASK_TYPE",
    "Pipeline",
    "PipelineSubmissionError",
    "PipelineTaskService",
    "run_learn_task_in_thread",
    "run_pipeline_in_thread",
    "submit_learn_task",
]
