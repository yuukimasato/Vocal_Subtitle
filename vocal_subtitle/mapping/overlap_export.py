"""重叠对白导出模块

对少量真实重叠说话，保留独立 speaker 轨道和物理跨度。

规则：
1. 只有事件带可验证 genuine_overlap=True 且 speaker 不同时进入重叠分组。
2. SRT 将该逻辑 cue 渲染为双行，每行使用可配置 speaker prefix，不超过两行。
3. ASS 可将同一 overlap_group_id 渲染为多个展示事件和轨道/位置。
4. 重叠分离不可信时保留 UNKNOWN 和 review warning，不丢弃次要可听对白。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class OverlapExportConfig:
    """重叠对白导出配置。"""

    max_tracks: int = 2
    speaker_prefix_format: str = "name_only"  # "name_only" | "colon" | "bracket"
    srt_max_lines: int = 2
    ass_track_positions: Tuple[int, ...] = (2, 8)  # ASS {\an} 位置代码


@dataclass
class OverlapTrack:
    """重叠组中的一个 speaker 轨道。"""

    text: str
    speaker_id: int
    speaker_label: Optional[str] = None
    start: float = 0.0
    end: float = 0.0
    source_word_ids: List[str] = field(default_factory=list)

    def format_srt_line(self, prefix_format: str = "name_only") -> str:
        """格式化为 SRT 行。"""
        label = self.speaker_label or f"Speaker {self.speaker_id}"
        if prefix_format == "colon":
            return f"{label}: {self.text}"
        elif prefix_format == "bracket":
            return f"[{label}] {self.text}"
        else:
            # name_only — just include the name for ASS-style rendering
            return f"{label}: {self.text}"


@dataclass
class OverlapGroup:
    """一组真实重叠的 DisplayCue。"""

    group_id: str
    tracks: List[OverlapTrack]
    start: float = 0.0
    end: float = 0.0
    verified: bool = True
    warnings: List[str] = field(default_factory=list)

    @property
    def track_count(self) -> int:
        return len(self.tracks)

    def srt_text(self, config: Optional[OverlapExportConfig] = None) -> str:
        """生成 SRT 双行文本。

        两行格式，每行带 speaker prefix，行间用 \\n 分隔。
        """
        cfg = config or OverlapExportConfig()
        lines = []
        for i, track in enumerate(self.tracks[:cfg.srt_max_lines]):
            lines.append(track.format_srt_line(cfg.speaker_prefix_format))
        return "\n".join(lines)

    def ass_events(self, config: Optional[OverlapExportConfig] = None) -> List[Dict[str, Any]]:
        """生成 ASS 展示事件。

        每个 speaker 轨道生成一个 ASS 事件，使用不同的 \\an 位置代码。
        """
        cfg = config or OverlapExportConfig()
        events = []
        for i, track in enumerate(self.tracks):
            position = cfg.ass_track_positions[min(i, len(cfg.ass_track_positions) - 1)]
            events.append({
                "start": track.start,
                "end": track.end,
                "text": track.text,
                "speaker_id": track.speaker_id,
                "speaker_label": track.speaker_label,
                "an_position": position,
                "overlap_group_id": self.group_id,
                "track_index": i,
            })
        return events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "tracks": [
                {
                    "text": t.text,
                    "speaker_id": t.speaker_id,
                    "speaker_label": t.speaker_label,
                    "start": t.start,
                    "end": t.end,
                    "source_word_ids": list(t.source_word_ids),
                }
                for t in self.tracks
            ],
            "start": self.start,
            "end": self.end,
            "verified": self.verified,
            "warnings": list(self.warnings),
        }


def group_overlapping_events(
    events: Sequence[Any],
    *,
    config: Optional[OverlapExportConfig] = None,
    max_gap: float = 0.05,
) -> List[OverlapGroup]:
    """将带 genuine_overlap 标记的事件分组为 OverlapGroup。

    Args:
        events: SubtitleEvent 或 DisplayCue 序列。
        config: 导出配置。
        max_gap: 归为同一重叠组的最小时间间隙。

    Returns:
        OverlapGroup 列表（非重叠事件保持单独但不在返回列表中）。
    """
    cfg = config or OverlapExportConfig()

    # 筛选真正重叠的事件
    overlap_events = [
        e for e in events
        if getattr(e, "genuine_overlap", False)
        and getattr(e, "overlap_group_id", None) is not None
    ]

    if not overlap_events:
        return []

    # 按 overlap_group_id 分组
    groups_by_id: Dict[str, List[Any]] = {}
    for event in overlap_events:
        gid = str(getattr(event, "overlap_group_id", ""))
        groups_by_id.setdefault(gid, []).append(event)

    result: List[OverlapGroup] = []
    for gid, members in sorted(groups_by_id.items()):
        # 排序
        members.sort(key=lambda e: (getattr(e, "start", 0), getattr(e, "end", 0)))

        # 按 speaker 区分轨道
        tracks: List[OverlapTrack] = []
        speakers_seen: set = set()
        for member in members[:cfg.max_tracks]:
            sid = getattr(member, "speaker_id", -1)
            # 去重同 speaker
            if sid in speakers_seen:
                continue
            speakers_seen.add(sid)

            tracks.append(OverlapTrack(
                text=str(getattr(member, "text", "")),
                speaker_id=sid if sid is not None else -1,
                speaker_label=getattr(member, "speaker_label", None),
                start=float(getattr(member, "physical_start", getattr(member, "start", 0))),
                end=float(getattr(member, "physical_end", getattr(member, "end", 0))),
                source_word_ids=list(getattr(member, "source_word_ids", []) or []),
            ))

        if not tracks:
            continue

        group_start = min(t.start for t in tracks)
        group_end = max(t.end for t in tracks)
        verified = all(
            getattr(m, "speaker_source", "") not in ("llm_guess", "unknown")
            for m in members
        )
        warnings: List[str] = []
        if len(members) > cfg.max_tracks:
            warnings.append(f"truncated_{len(members) - cfg.max_tracks}_tracks")
        if not verified:
            warnings.append("unverified_speaker_attribution")

        result.append(OverlapGroup(
            group_id=gid,
            tracks=tracks,
            start=group_start,
            end=group_end,
            verified=verified,
            warnings=warnings,
        ))

    # 按时间排序
    result.sort(key=lambda g: (g.start, g.end, g.group_id))
    return result


def render_overlap_srt(
    groups: Sequence[OverlapGroup],
    *,
    config: Optional[OverlapExportConfig] = None,
    audio_duration: Optional[float] = None,
) -> str:
    """将重叠组渲染为 SRT 文本。

    每个 OverlapGroup 渲染为一个 SRT cue（双行），
    格式化为标准 SRT 编号 + 时间 + 双行文本。
    """
    cfg = config or OverlapExportConfig()

    def _format_ms(total_seconds: float) -> str:
        total_seconds = max(0.0, min(total_seconds, audio_duration or total_seconds))
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        millis = int(round((total_seconds % 1) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    lines: List[str] = []
    for index, group in enumerate(groups, start=1):
        start_str = _format_ms(group.start)
        end_str = _format_ms(group.end)
        lines.append(str(index))
        lines.append(f"{start_str} --> {end_str}")
        lines.append(group.srt_text(cfg))
        lines.append("")  # blank line between cues

    return "\n".join(lines)
