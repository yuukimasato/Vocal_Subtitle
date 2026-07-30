"""Request/response serialization, SubtitleEvent conversion, and file rewriting helpers.

Extracted from api.py to keep route functions focused on HTTP concerns.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from ..config import ConfigLoader, SubtitleBuildConfig
from ..mapping.time_mapper import SubtitleEvent
from .api_services import _task_history, _task_store

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Config summary helpers
# ------------------------------------------------------------

_PROFILE_DESCRIPTIONS = {
    "default": "默认（通用访谈/播客）",
    "podcast": "播客（长音频、宽泛声学吸附）",
    "education": "教育（安全敏感、严格边界）",
    "variety_show": "综艺（多人对话、激进合并）",
    "music_live": "音乐现场（跳过静音检测、宽松重连）",
}


def _get_profile_description(name: str) -> str:
    return _PROFILE_DESCRIPTIONS.get(name, "自定义配置")


def _config_summary(config) -> Dict[str, Any]:
    """提取配置关键字段摘要"""
    return {
        "separation_engine": config.separation.engine,
        "vad_engine": config.vad.engine,
        "asr_engine": config.asr.engine,
        "asr_model": config.asr.model,
        "language": config.asr.language,
        "device": config.asr.device,
        "asr_route_version": config.asr.auto_routing.route_version,
        "asr_quality_gate_version": config.asr.auto_routing.quality_gate_version,
        "llm_enabled": config.llm_optimize.enabled,
    }


def _config_to_overrides_dict(config) -> Dict[str, Any]:
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
        "asr_route_version": config.asr.auto_routing.route_version,
        "asr_quality_gate_version": config.asr.auto_routing.quality_gate_version,
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
        "speaker_fusion": config.diarization.fusion_mode,
        "global_diarization_model": config.diarization.global_model,
        "speaker_diarization_scope": config.diarization.diarization_scope,
        "local_speaker_refinement": config.diarization.local_refinement,
        "expected_speakers": config.diarization.expected_speakers,
        "diarization_local_context": config.diarization.local_context_seconds,
        "diarization_min_local_segment": config.diarization.min_local_segment_seconds,
        "diarization_min_change_confidence": config.diarization.min_change_confidence,
        "diarization_distance_threshold": config.diarization.distance_threshold,
        "diarization_min_speakers": config.diarization.min_speakers,
        "diarization_max_speakers": config.diarization.max_speakers,
        "speaker_role": config.speaker_role.enabled,
        "speaker_role_model": config.speaker_role.model,
        "speaker_role_context_hint": config.speaker_role.context_hint or "",
        "speaker_embedding": config.speaker_embedding.enabled,
        "speaker_embedding_model": config.speaker_embedding.model_ref,
        "speaker_embedding_token": config.speaker_embedding.hf_token,
        "ffmpeg_enabled": config.vad.ffmpeg_enabled,
        "ffmpeg_noise_db": config.vad.ffmpeg_noise_db,
        "skeleton_mode": config.acoustic_validation.skeleton_mode,
        "export_skeleton_segments": config.acoustic_validation.export_skeleton_segments,
        "export_skeleton_dir": config.acoustic_validation.export_skeleton_dir,
        "merge_llm_tier": config.merge_decision.llm_tier,
        "merge_fast_max_gap": config.merge_decision.fast_merge_max_gap,
        "merge_llm_decision_max_gap": config.merge_decision.llm_decision_max_gap,
        "merge_hard_split_min_gap": config.merge_decision.hard_split_min_gap,
        "merge_max_combined_duration": config.merge_decision.max_combined_duration,
        "noise_reduction": config.noise_reduction.enabled,
        "noise_reduction_engine": config.noise_reduction.engine,
        "asr_path": config.asr_path,
    }


# ------------------------------------------------------------
# SubtitleEvent serialization
# ------------------------------------------------------------

class SerializationError(RuntimeError):
    """Raised when SubtitleEvent payloads cannot be round-tripped."""
    pass


def _build_subtitle_event_payloads(events: list) -> list:
    """Serialize a list of SubtitleEvents into API dict payloads."""
    return [
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
            "speaker_confidence": e.speaker_confidence,
            "speaker_model": e.speaker_model,
            "speaker_repair_reason": e.speaker_repair_reason,
            "alignment_warning": e.alignment_warning,
        }
        for e in events
    ]


def _build_task_result(result, from_cache, output_path) -> dict:
    """Build the task_result dict from pipeline output."""
    events = result.get("events", [])
    stats = result["stats"]

    subtitle_events = _build_subtitle_event_payloads(events)

    return {
        "subtitle_path": str(result["subtitle_path"]),
        "clean_subtitle_path": str(result["clean_subtitle_path"]) if result.get("clean_subtitle_path") else None,
        "llm_subtitle_path": str(result["llm_subtitle_path"]) if result.get("llm_subtitle_path") else None,
        "stats": stats.to_dict(),
        "events": subtitle_events,
        "from_cache": from_cache,
        "segment_count": stats.segment_count,
        "subtitle_count": stats.subtitle_count,
        "quality_status": stats.quality_status,
        "requested_engine": stats.requested_engine,
        "selected_engine": stats.selected_engine,
        "final_engine": stats.final_engine,
        "detected_language": stats.detected_language,
        "fallback_reason": stats.fallback_reason or None,
        "vocals_path": result.get("vocals_path"),
        "accompaniment_path": result.get("accompaniment_path"),
    }


def _subtitle_event_from_payload(payload: dict):
    """Reconstruct a SubtitleEvent from a dict payload, preserving all provenance fields."""
    return SubtitleEvent.from_dict(payload)


def _subtitle_event_to_payload(event):
    """Serialize a SubtitleEvent to a dict, preserving all provenance fields."""
    return event.to_dict()


def _load_completed_subtitle_task(task_id: str) -> tuple:
    """Load a completed task from memory, falling back to persisted history."""
    task = _task_store.get(task_id)
    if not task:
        hist_task = _task_history.get(task_id)
        if hist_task and hist_task.get("result_json"):
            try:
                result = json.loads(hist_task["result_json"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise HTTPException(status_code=404, detail=f"Task not found: {task_id}") from exc
            task = {
                "task_id": task_id,
                "status": "completed",
                "result": result,
            }
            _task_store[task_id] = task
        else:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    result = task.get("result", {})
    if not isinstance(result, dict) or not isinstance(result.get("events"), list):
        raise HTTPException(status_code=404, detail="Task has no subtitle events")
    return task, result


def _persist_subtitle_result(task_id: str, task: dict, result: dict) -> None:
    """Persist the final event list used by the UI and every subtitle export."""
    task["status"] = "completed"
    task["result"] = result
    _task_store[task_id] = task
    _task_history.update(
        task_id,
        status="completed",
        result_json=json.dumps(result, default=str),
    )


def _rewrite_subtitle_files(task_result: dict) -> list:
    """将内存中的字幕事件写回磁盘文件"""
    events = task_result.get("events", [])
    if not events:
        return []

    errors: List[str] = []

    from ..mapping.subtitle_builder import SubtitleBuilder, SubtitleRule

    rebuilt_events = [_subtitle_event_from_payload(e) for e in events]

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

    # Rewrite subtitle_path
    subtitle_path_str = task_result.get("subtitle_path")
    if subtitle_path_str:
        path = Path(subtitle_path_str)
        try:
            text = builder.build_to_string(rebuilt_events, fmt=path.suffix.lstrip(".") or "srt")
            path.write_text(text, encoding="utf-8")
        except Exception as e:
            errors.append(f"subtitle_path {path}: {e}")

    # Rewrite LLM path
    llm_path_str = task_result.get("llm_subtitle_path")
    if llm_path_str:
        llm_path = Path(llm_path_str)
        try:
            text = builder.build_to_string(rebuilt_events, fmt=llm_path.suffix.lstrip(".") or "srt")
            llm_path.write_text(text, encoding="utf-8")
        except Exception as e:
            errors.append(f"llm_subtitle_path {llm_path}: {e}")

    return errors


def _parse_cache_status_response(cache) -> dict:
    """Build cache info response from cache manager state."""
    stages = {}
    for stage_name in ["separation", "transcription"]:
        try:
            stage_cache = getattr(cache, f"_{stage_name}_cache", None)
            if stage_cache is not None:
                stages[stage_name] = len(stage_cache)
        except Exception:
            stages[stage_name] = 0
    return stages
