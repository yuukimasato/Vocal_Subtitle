"""显示时间轴映射

将不可变的物理时间和语义边界投影结果映射到适合阅读的显示时间轴。

规则：
- physical_start/end 取已接受的 BoundaryDecision 首尾对齐词边界，进入本阶段后不可修改。
- display_start/end 可向最近候选静音边界吸附，但必须被 PhysicalClip、hard split、
  下一人声起点和最大前导/尾随窗口夹持。
- 显示时间必须覆盖物理人声，除非事件已标记 timing_degraded。
- 相邻不同 speaker 时优先人声边界；阅读时长不足写入质量警告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class DisplayTimelineConfig:
    """显示时间轴映射配置。"""

    max_lead_ms: float = 150.0          # 最大前导静音窗口（ms）
    max_trail_ms: float = 150.0         # 最大尾随静音窗口（ms）
    min_reading_duration_ms: float = 600.0  # 最短阅读时间（ms）
    max_reading_duration_ms: float = 6000.0  # 最长阅读时间（ms）
    silence_snap_threshold_ms: float = 50.0   # 静音吸附阈值（ms）
    cps_cjk: float = 5.0                # 中文字速（字符/秒）
    cps_latin: float = 12.0             # 拉丁字速（字符/秒）

    def __post_init__(self) -> None:
        if self.max_lead_ms < 0:
            raise ValueError("max_lead_ms must be non-negative")
        if self.max_trail_ms < 0:
            raise ValueError("max_trail_ms must be non-negative")
        if self.min_reading_duration_ms <= 0:
            raise ValueError("min_reading_duration_ms must be > 0")
        if self.max_reading_duration_ms < self.min_reading_duration_ms:
            raise ValueError("max_reading_duration_ms must be >= min_reading_duration_ms")


@dataclass
class DisplayCue:
    """最终显示 cue — 唯一供 WebUI、API 和导出器消费的事件。"""

    index: int
    physical_start: float
    physical_end: float
    display_start: float
    display_end: float
    text: str
    speaker_id: Optional[int] = None
    speaker_label: Optional[str] = None
    timing_degraded: bool = False
    source_word_ids: List[str] = field(default_factory=list)
    physical_spans: List[Dict[str, Any]] = field(default_factory=list)
    overlap_group_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.physical_start < 0 or self.physical_end <= self.physical_start:
            raise ValueError("physical times must satisfy 0 <= start < end")
        if self.display_start < 0 or self.display_end <= self.display_start:
            raise ValueError("display times must satisfy 0 <= start < end")
        if not self.timing_degraded:
            if self.display_start > self.physical_start + 0.001:
                raise ValueError("display must cover physical start")
            if self.display_end < self.physical_end - 0.001:
                raise ValueError("display must cover physical end")

    @property
    def duration_ms(self) -> float:
        return (self.display_end - self.display_start) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "physical_start": self.physical_start,
            "physical_end": self.physical_end,
            "display_start": self.display_start,
            "display_end": self.display_end,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "speaker_label": self.speaker_label,
            "timing_degraded": self.timing_degraded,
            "source_word_ids": list(self.source_word_ids),
            "physical_spans": list(self.physical_spans),
            "overlap_group_id": self.overlap_group_id,
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
        }


def map_to_display_timeline(
    semantic_groups: Sequence[Dict[str, Any]],
    *,
    config: Optional[DisplayTimelineConfig] = None,
    audio_duration: Optional[float] = None,
) -> List[DisplayCue]:
    """将语义分组映射为显示时间轴的 DisplayCue 列表。

    Args:
        semantic_groups: 语义分组列表，每项需包含：
            - physical_start / physical_end（不可变物理时间）
            - text
            - index
            - 可选的 projected_start/end、speaker_id 等。
        config: 显示映射配置。
        audio_duration: 音频总时长，用于夹持。

    Returns:
        DisplayCue 列表，已排序编号。
    """
    cfg = config or DisplayTimelineConfig()
    max_lead = cfg.max_lead_ms / 1000.0
    max_trail = cfg.max_trail_ms / 1000.0
    min_reading = cfg.min_reading_duration_ms / 1000.0

    cues: List[DisplayCue] = []

    for group in semantic_groups:
        physical_start = float(group["physical_start"])
        physical_end = float(group["physical_end"])
        index = int(group.get("index", len(cues) + 1))

        # 从投影结果获取显示候选
        projected_start = float(group.get("projected_start", physical_start))
        projected_end = float(group.get("projected_end", physical_end))

        # 显示时间 = 物理时间向静音边界吸附，但受夹持
        display_start = max(
            physical_start,
            projected_start - max_lead,
        )
        display_end = min(
            physical_end,
            projected_end + max_trail,
        )
        # 但也不能短于物理人声，除非 timing_degraded
        timing_degraded = bool(group.get("timing_degraded", False))
        if not timing_degraded:
            display_start = min(display_start, physical_start)
            display_end = max(display_end, physical_end)

        # 满足阅读时间要求
        if display_end - display_start < min_reading:
            extension_needed = min_reading - (display_end - display_start)
            display_end = min(
                display_end + extension_needed,
                audio_duration if audio_duration is not None else display_end + extension_needed,
            )

        # 夹持到音频范围
        if audio_duration is not None:
            display_start = max(0.0, display_start)
            display_end = min(audio_duration, display_end)

        if display_end <= display_start:
            display_end = display_start + min_reading

        # 检查跨 speaker 约束
        warnings = list(group.get("warnings", []))
        if cues:
            prev = cues[-1]
            prev_speaker = prev.speaker_id
            this_speaker = group.get("speaker_id")
            if (prev_speaker is not None and this_speaker is not None
                    and prev_speaker != this_speaker
                    and display_start < prev.display_end):
                # 不同 speaker 时优先人声边界
                display_start = min(display_start, physical_start)
                warnings.append("speaker_boundary_display_conflict")

        cue = DisplayCue(
            index=index,
            physical_start=physical_start,
            physical_end=physical_end,
            display_start=max(0.0, display_start),
            display_end=display_end,
            text=str(group.get("text", "")),
            speaker_id=group.get("speaker_id"),
            speaker_label=group.get("speaker_label"),
            timing_degraded=timing_degraded,
            source_word_ids=list(group.get("source_word_ids", [])),
            physical_spans=list(group.get("physical_spans", [])),
            overlap_group_id=group.get("overlap_group_id"),
            warnings=warnings,
            diagnostics=group.get("projection_diagnostics", {}),
        )
        cues.append(cue)

    # 重新编号
    for new_index, cue in enumerate(cues, start=1):
        cue.index = new_index

    return cues
