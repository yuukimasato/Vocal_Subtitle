"""Application service for asynchronous WebUI pipeline tasks.

The HTTP routes do not construct or own ``Pipeline`` instances.  This module
owns the background-task lifecycle and keeps the historical task store,
history database, WebSocket events, and persistence hooks unchanged.
"""

from __future__ import annotations

import json
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
    return [
        {
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
        for event in events
    ]


def _persistence_manager():
    """Resolve the compatibility accessor at call time for old monkeypatches."""
    from . import routes_history

    return routes_history._get_persistence_mgr()


def run_pipeline_in_thread(
    task_id: str,
    input_path: Path,
    output_path: Path,
    profile: str,
    output_format: str,
    skip_separation: bool,
    overrides: Dict[str, Any],
    session_dir: Optional[Path] = None,
) -> None:
    """Run one pipeline task and publish the legacy WebUI task contract."""
    try:
        from ..config import ConfigLoader

        loader = ConfigLoader()
        config = loader.load_profile(profile)
        config = loader.merge_with_overrides(config, **overrides)
        pipeline = _pipeline_class()(config)
        progress_callback = ws_manager.create_progress_callback(task_id)

        state.task_store[task_id]["status"] = "running"
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

        state.task_store[task_id].update({
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
        config = loader.merge_with_overrides(
            loader.load_profile(profile),
            **overrides,
        )
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


__all__ = [
    "Pipeline",
    "PipelineSubmissionError",
    "PipelineTaskService",
    "run_pipeline_in_thread",
]
