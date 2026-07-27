"""字幕事件最终化模块

集中编排字幕事件最终化流程，确保 WebUI、API、缓存、SRT/VTT/ASS 和
质量报告消费同一份最终事件列表。

流程：
    PhysicalFragment -> strict segmentation -> LLM/local semantic decision
    -> display mapping -> final validation -> numbering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .display_timeline import DisplayCue, DisplayTimelineConfig, map_to_display_timeline
from .time_mapper import SubtitleEvent


@dataclass
class FinalizeConfig:
    """最终化配置。"""

    strict_segmentation_enabled: bool = True
    llm_post_enabled: bool = False
    display: DisplayTimelineConfig = field(default_factory=DisplayTimelineConfig)
    validate: bool = True


@dataclass
class FinalizeResult:
    """最终化结果 — 单一出口。"""

    events: List[SubtitleEvent]
    display_cues: List[DisplayCue]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": [
                {
                    "index": e.index,
                    "start": e.start,
                    "end": e.end,
                    "text": e.text,
                    "physical_start": getattr(e, "physical_start", None),
                    "physical_end": getattr(e, "physical_end", None),
                    "speaker_id": getattr(e, "speaker_id", None),
                    "speaker_label": getattr(e, "speaker_label", None),
                }
                for e in self.events
            ],
            "display_cues": [c.to_dict() for c in self.display_cues],
            "diagnostics": dict(self.diagnostics),
        }

    @property
    def subtitle_count(self) -> int:
        """事件的逻辑数量，与预览和每个导出文件的 cue 数一致。"""
        return len(self.events)


def finalize_subtitle_events(
    events: Sequence[SubtitleEvent],
    *,
    config: Optional[FinalizeConfig] = None,
    audio_duration: Optional[float] = None,
) -> FinalizeResult:
    """唯一字幕事件最终化入口。

    所有离线路径在导出前只调用一次 finalizer。主 subtitle_path、
    result.events、WebSocket 完成事件和质量报告使用同一 list。

    Args:
        events: 字幕事件列表（已含物理、来源、speaker 和修订溯源）。
        config: 最终化配置。
        audio_duration: 音频总时长，用于夹持。

    Returns:
        FinalizeResult 包含事件和显示 cue。
    """
    cfg = config or FinalizeConfig()
    diagnostics: Dict[str, Any] = {
        "input_event_count": len(events),
        "output_event_count": 0,
        "step": [],
    }

    # Step 1: 验证输入事件
    valid_events = _validate_input_events(events, diagnostics)

    # Step 2: 转换为语义组格式
    semantic_groups = _events_to_semantic_groups(valid_events)

    # Step 3: 映射到显示时间轴
    display_cues = map_to_display_timeline(
        semantic_groups,
        config=cfg.display,
        audio_duration=audio_duration,
    )
    diagnostics["display_cue_count"] = len(display_cues)
    diagnostics["step"].append("display_mapping")

    # Step 4: 同步事件时间到显示时间
    for i, event in enumerate(valid_events):
        if i < len(display_cues):
            cue = display_cues[i]
            event.start = cue.display_start
            event.end = cue.display_end

    # Step 5: 最终校验
    if cfg.validate:
        valid_events, validation_diag = _run_final_validation(
            valid_events, audio_duration
        )
        diagnostics.update(validation_diag)
        diagnostics["step"].append("final_validation")

    # Step 6: 编号
    for new_index, event in enumerate(valid_events, start=1):
        event.index = new_index
    diagnostics["output_event_count"] = len(valid_events)

    return FinalizeResult(
        events=list(valid_events),
        display_cues=display_cues,
        diagnostics=diagnostics,
    )


def _validate_input_events(
    events: Sequence[SubtitleEvent],
    diagnostics: Dict[str, Any],
) -> List[SubtitleEvent]:
    """验证并过滤输入事件。"""
    valid: List[SubtitleEvent] = []
    skipped = 0

    for event in events:
        if not event.text or not event.text.strip():
            skipped += 1
            continue
        if event.end <= event.start:
            skipped += 1
            continue
        valid.append(event)

    diagnostics["skipped_invalid"] = skipped
    diagnostics["valid_input_count"] = len(valid)
    return valid


def _events_to_semantic_groups(
    events: Sequence[SubtitleEvent],
) -> List[Dict[str, Any]]:
    """将 SubtitleEvent 列表转为语义组字典列表。"""
    groups = []
    for i, event in enumerate(events, start=1):
        physical_start = getattr(event, "physical_start", None)
        physical_end = getattr(event, "physical_end", None)
        if physical_start is None or physical_start < 0:
            physical_start = event.start
        if physical_end is None or physical_end <= 0:
            physical_end = event.end

        groups.append({
            "index": i,
            "text": event.text,
            "start": event.start,
            "end": event.end,
            "physical_start": physical_start,
            "physical_end": physical_end,
            "speaker_id": getattr(event, "speaker_id", None),
            "speaker_label": getattr(event, "speaker_label", None),
            "source_word_ids": list(getattr(event, "source_word_ids", []) or []),
            "physical_spans": list(getattr(event, "physical_spans", []) or []),
            "timing_degraded": bool(getattr(event, "alignment_warning", None)),
            "overlap_group_id": getattr(event, "overlap_group_id", None),
            "hard_split_before": bool(getattr(event, "hard_split_before", False)),
            "hard_split_after": False,
        })
    return groups


def _run_final_validation(
    events: Sequence[SubtitleEvent],
    audio_duration: Optional[float],
) -> tuple[List[SubtitleEvent], Dict[str, Any]]:
    """运行最终校验，返回通过校验的事件和诊断信息。"""
    from .final_validator import validate_events

    result = validate_events(events, audio_duration=audio_duration, strict=True)
    return list(result.events), result.diagnostics
