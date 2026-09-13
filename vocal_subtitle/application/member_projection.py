"""把聚合 ASR 窗口的输出重投影回原始物理骨架成员段。

聚合骨架窗口（group_speech_intervals）给 ASR 提供了跨短停顿的上下文，
提升识别质量；代价是字幕边界改由窗口内部的 VAD/合并 padding 决定，
不再锚定物理静音边界。本模块恢复 v0.2.0 逐段切片的端点契约：

- 跨越成员间隙的事件按词级时间戳拆分（间隙 >= split_min_gap 的才拆，
  句内微停顿保持整行，避免"四个半"式碎片）；
- 所有事件端点钳制到所属物理成员段的边界；
- 完全落在上下文 padding 静音区、不属于任何成员段的事件被丢弃。

输入事件与成员段必须处于同一坐标系（骨架路径中为窗口局部坐标）。
"""

from __future__ import annotations

import copy
import logging
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Sequence, Tuple

from ..mapping.time_mapper import SubtitleEvent

logger = logging.getLogger(__name__)

# 钳制后事件仍退化为非正时长时的最小展示时长（秒）。
_MIN_EVENT_DURATION = 0.05

# 成员间隙小于该阈值时视为句内微停顿（换气/强调），不据此拆分文本；
# 骨架成员间隙范围是 [skeleton_min_silence, 聚合硬静音 0.4s)，
# 默认 0.3s 只让真正的短语间停顿成为字幕行边界。
DEFAULT_MEMBER_SPLIT_MIN_GAP = 0.3

# 聚合组超出该时长/显示宽度时，即使间隙 < split_min_gap 也要在成员间隙处
# 拆分（展示驱动的兜底，避免整窗合并成一条长字幕）。宽度按 CJK=2 计，
# 84 ≈ 2 行 × 42 拉丁字符，与 subtitle.max_chars_latin 的双行约束对齐；
# 时长上限与合并级联的 max_combined_duration(5.0s) 对齐。
DEFAULT_MEMBER_SPLIT_MAX_DURATION = 5.0
_MAX_DISPLAY_WIDTH = 84


def _display_width(text: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        for char in text
    )


@dataclass
class MemberProjectionStats:
    """重投影统计，用于日志与诊断。"""

    events_in: int = 0
    events_out: int = 0
    split_events: int = 0
    clamped_events: int = 0
    dropped_events: int = 0
    unworded_events: int = 0
    over_limit_splits: int = 0

    def as_dict(self) -> dict:
        return {
            "events_in": self.events_in,
            "events_out": self.events_out,
            "split_events": self.split_events,
            "clamped_events": self.clamped_events,
            "dropped_events": self.dropped_events,
            "unworded_events": self.unworded_events,
            "over_limit_splits": self.over_limit_splits,
        }


def _word_field(word: Any, name: str) -> Optional[float]:
    value = getattr(word, name, None)
    if value is None and isinstance(word, dict):
        value = word.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _word_text(word: Any) -> str:
    text = getattr(word, "word", None)
    if text is None and isinstance(word, dict):
        text = word.get("word")
    return str(text) if text is not None else ""


def _absolute_word_spans(event: SubtitleEvent) -> List[Tuple[Any, float, float]]:
    """词时间相对 event.start；换算回事件所在坐标系的绝对时间。"""
    spans: List[Tuple[Any, float, float]] = []
    for word in event.words or []:
        start = _word_field(word, "start")
        end = _word_field(word, "end")
        if start is None or end is None:
            continue
        spans.append((word, event.start + start, event.start + end))
    return spans


def _assign_member(
    word_start: float,
    word_end: float,
    members: Sequence[Tuple[float, float]],
) -> int:
    """按最大重叠归属词；无重叠时归属最近成员（silencedetect 边缘抖动）。"""
    best_index = 0
    best_key: Optional[Tuple[bool, float]] = None
    for index, (start, end) in enumerate(members):
        overlap = min(word_end, end) - max(word_start, start)
        distance = max(start - word_end, word_start - end, 0.0)
        key = (overlap > 0.0, overlap if overlap > 0.0 else -distance)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
    return best_index


def _join_word_text(words: Sequence[Any]) -> str:
    # ASR segment 文本即词字符串的直接拼接（空格由词自带），保持一致。
    return "".join(_word_text(word) for word in words).strip()


def _relative_words(
    spans: Sequence[Tuple[Any, float, float]],
    event_start: float,
) -> List[Any]:
    relative: List[Any] = []
    for word, abs_start, abs_end in spans:
        copied = copy.copy(word)
        copied.start = abs_start - event_start
        copied.end = abs_end - event_start
        relative.append(copied)
    return relative


def _trace(event: SubtitleEvent, reason: str, detail: dict) -> List[dict]:
    entry = {"stage": "member_reprojection", "source": reason, **detail}
    return list(event.revision_trace) + [entry]


def _copy_event(event: SubtitleEvent, **fields: Any) -> SubtitleEvent:
    """replace() 的浅拷贝会让多个新事件共享可变字段（如 time_offset_trace），
    导致后续一次性偏移契约误判；这里显式拷贝所有可变容器。"""
    fields.setdefault("words", list(event.words))
    fields.setdefault("physical_spans", list(event.physical_spans))
    fields.setdefault("source_word_ids", list(event.source_word_ids))
    fields.setdefault("overlap_tracks", list(event.overlap_tracks))
    fields.setdefault("revision_trace", list(event.revision_trace))
    fields.setdefault("time_offset_trace", list(event.time_offset_trace))
    fields.setdefault("trace_context", dict(event.trace_context))
    return replace(event, **fields)


def _clamped_copy(
    event: SubtitleEvent,
    new_start: float,
    new_end: float,
    member: Tuple[float, float],
) -> SubtitleEvent:
    fields: dict = {
        "start": new_start,
        "end": new_end,
        "revision_trace": _trace(event, "clamp_to_member", {
            "original_start": round(event.start, 6),
            "original_end": round(event.end, 6),
            "member_start": round(member[0], 6),
            "member_end": round(member[1], 6),
        }),
    }
    if event.physical_start is not None:
        fields["physical_start"] = max(event.physical_start, member[0])
    if event.physical_end is not None:
        fields["physical_end"] = min(event.physical_end, member[1])
    return _copy_event(event, **fields)


def _entries_over_limit(
    entries: List[Tuple[Tuple[Any, float, float], int]],
    max_duration: float,
) -> bool:
    """词条组是否超出时长/显示宽度限制（``max_duration <= 0`` 视为不限制）。"""
    if not entries or max_duration is None or max_duration <= 0:
        return False
    span_start = entries[0][0][1]
    span_end = entries[-1][0][2]
    if span_end - span_start > max_duration:
        return True
    width = sum(
        _display_width(_word_text(span[0]))
        for span, _ in entries
    )
    return width > _MAX_DISPLAY_WIDTH


def _split_entries_over_limit(
    entries: List[Tuple[Tuple[Any, float, float], int]],
    max_duration: float,
) -> List[List[Tuple[Tuple[Any, float, float], int]]]:
    """把同一拆分组的词条按成员间隙贪心切成不超过限制的片段。

    切分点只允许落在成员索引变化处（真实的静音间隙）；单个成员内部的
    连续语流没有声学证据，保持不切，交由下游 finalize 兜底。
    ``max_duration <= 0`` 视为不限制，返回原组。
    """
    if not entries or max_duration is None or max_duration <= 0:
        return [entries]

    pieces: List[List[Tuple[Tuple[Any, float, float], int]]] = []
    current: List[Tuple[Tuple[Any, float, float], int]] = [entries[0]]
    current_width = _display_width(_word_text(entries[0][0][0]))

    for entry in entries[1:]:
        span, member_index = entry
        at_member_gap = member_index != current[-1][1]
        piece_start = current[0][0][1]
        width_next = current_width + _display_width(_word_text(span[0]))
        over_limit = (
            span[2] - piece_start > max_duration
            or width_next > _MAX_DISPLAY_WIDTH
        )
        if at_member_gap and over_limit and current:
            pieces.append(current)
            current = [entry]
            current_width = _display_width(_word_text(span[0]))
        else:
            current.append(entry)
            current_width = width_next

    pieces.append(current)
    return pieces


def reproject_events_to_members(
    events: List[SubtitleEvent],
    members: Sequence[Tuple[float, float]],
    *,
    split_min_gap: float = DEFAULT_MEMBER_SPLIT_MIN_GAP,
    max_duration: float = DEFAULT_MEMBER_SPLIT_MAX_DURATION,
) -> Tuple[List[SubtitleEvent], MemberProjectionStats]:
    """把事件拆分/钳制到不相交的物理成员段上。

    Args:
        events: 待重投影的字幕事件（不修改原对象）。
        members: 与事件同坐标系的物理成员段 [(start, end)]，升序不重叠。
        split_min_gap: 相邻成员间隙小于该值时合并为同一拆分组（句内
            微停顿不产生字幕行边界）；``0`` 恢复逐成员拆分的旧行为。
        max_duration: 单个拆分组允许的最大时长（秒）；超过时即使组内
            间隙 < split_min_gap 也要在成员间隙处继续拆分（<=0 关闭）。

    Returns:
        (重投影后的事件列表, 统计)
    """
    stats = MemberProjectionStats(events_in=len(events))
    if not members:
        stats.events_out = len(events)
        return list(events), stats

    ordered = sorted(members)
    # 成员索引 → 拆分组编号：间隙 >= split_min_gap 的相邻成员才分属不同组。
    group_of = [0]
    for index in range(1, len(ordered)):
        gap = ordered[index][0] - ordered[index - 1][1]
        group_of.append(group_of[-1] if gap < split_min_gap else group_of[-1] + 1)
    group_member_ranges: dict = {}
    for index, span in enumerate(ordered):
        group_member_ranges.setdefault(group_of[index], []).append(span)

    projected: List[SubtitleEvent] = []

    for event in events:
        spans = _absolute_word_spans(event)

        if not spans:
            # 无词级时间戳：无法可靠拆分文本，只钳制到重叠成员段的包络。
            stats.unworded_events += 1
            overlapped = [
                member for member in ordered
                if member[0] < event.end and member[1] > event.start
            ]
            if not overlapped:
                stats.dropped_events += 1
                continue
            new_start = max(event.start, overlapped[0][0])
            new_end = min(event.end, overlapped[-1][1])
            if new_end - new_start < _MIN_EVENT_DURATION:
                stats.dropped_events += 1
                continue
            if new_start == event.start and new_end == event.end:
                projected.append(event)
                continue
            stats.clamped_events += 1
            projected.append(_clamped_copy(
                event, new_start, new_end, overlapped[-1],
            ))
            continue

        assignments = [
            _assign_member(word_start, word_end, ordered)
            for _, word_start, word_end in spans
        ]
        # 分组词条携带成员索引：拆分点只允许落在成员间隙上（有真实静音证据）。
        groups: dict = {}
        for span, member_index in zip(spans, assignments):
            groups.setdefault(group_of[member_index], []).append((span, member_index))

        # 单组且不超限：微停顿合并成立，端点钳制到组成员包络后透传；
        # 超限（单一大组）时继续走下方的成员间隙拆分，避免整段长字幕。
        if len(groups) <= 1 and not _entries_over_limit(
            groups[group_of[assignments[0]]], max_duration,
        ):
            # 同一组可能覆盖多个相邻成员（句内微停顿未拆分），
            # 端点钳制到该组成员范围的包络，而不是首个词的成员端点。
            group_spans = group_member_ranges[group_of[assignments[0]]]
            envelope_start = group_spans[0][0]
            envelope_end = group_spans[-1][1]
            new_start = max(event.start, envelope_start)
            new_end = min(event.end, envelope_end)
            if new_start == event.start and new_end == event.end:
                projected.append(event)
                continue
            if new_end - new_start < _MIN_EVENT_DURATION:
                stats.dropped_events += 1
                continue
            stats.clamped_events += 1
            projected.append(_clamped_copy(
                event, new_start, new_end,
                (envelope_start, envelope_end),
            ))
            continue

        def _emit_piece(
            entries: List[Tuple[Tuple[Any, float, float], int]],
            *,
            hard_before: bool,
            over_limit: bool,
        ) -> None:
            member_indexes = [member_index for _, member_index in entries]
            span_start = ordered[min(member_indexes)][0]
            span_end = ordered[max(member_indexes)][1]
            piece_spans = [span for span, _ in entries]
            start = max(span_start, piece_spans[0][1])
            end = min(span_end, piece_spans[-1][2])
            if end - start < _MIN_EVENT_DURATION:
                end = min(span_end, start + _MIN_EVENT_DURATION)
            if end <= start:
                stats.dropped_events += 1
                return
            detail = {
                "member_start": round(span_start, 6),
                "member_end": round(span_end, 6),
                "source_start": round(event.start, 6),
                "source_end": round(event.end, 6),
            }
            if over_limit:
                detail["over_limit"] = True
            projected.append(_copy_event(
                event,
                start=start,
                end=end,
                text=_join_word_text([span[0] for span in piece_spans])
                or event.text,
                words=_relative_words(piece_spans, start),
                hard_split_before=hard_before,
                revision_trace=_trace(event, "split_across_members", detail),
            ))

        stats.split_events += 1
        for group_id in sorted(groups):
            entries = groups[group_id]
            pieces = _split_entries_over_limit(entries, max_duration)
            if len(pieces) > 1:
                stats.over_limit_splits += len(pieces) - 1
            for piece_index, piece in enumerate(pieces):
                _emit_piece(
                    piece,
                    hard_before=(
                        piece_index > 0
                        or group_id != group_of[assignments[0]]
                    ),
                    over_limit=len(pieces) > 1,
                )

    stats.events_out = len(projected)
    return projected, stats
