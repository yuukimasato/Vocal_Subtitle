"""物理语音边界与全局 speaker turns 的融合。

该模块不通过停顿猜测说话人。停顿只影响同一 speaker 的物理区间是否
可以合并；speaker identity 和换人边界必须来自全局 diarization。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple

from .base import AtomicSpeechSpan, SpeakerTurn

logger = logging.getLogger(__name__)


def normalize_turns(
    turns: Iterable[SpeakerTurn],
    duration: Optional[float] = None,
    min_duration: float = 0.0,
) -> List[SpeakerTurn]:
    """清理全局 turns，保持 speaker id，只裁剪时间范围。"""
    result: List[SpeakerTurn] = []
    limit = max(0.0, duration) if duration is not None else None
    for turn in turns:
        start = max(0.0, float(turn.start))
        end = float(turn.end)
        if limit is not None:
            start = min(start, limit)
            end = min(end, limit)
        if end <= start or end - start < min_duration:
            continue
        result.append(
            replace(
                turn,
                start=start,
                end=end,
                speaker_id=int(turn.speaker_id),
            )
        )
    result.sort(key=lambda item: (item.start, item.end, item.speaker_id))
    return result


def shift_turns(
    turns: Sequence[SpeakerTurn],
    offset: float,
    duration: Optional[float] = None,
) -> List[SpeakerTurn]:
    """将全局 turns 投影到以 ``offset`` 为起点的局部音频。"""
    shifted = [
        replace(turn, start=turn.start - offset, end=turn.end - offset)
        for turn in turns
    ]
    return normalize_turns(shifted, duration=duration)


def _snap_to_physical_boundary(
    value: float,
    region_start: float,
    region_end: float,
    collar: float,
) -> float:
    if abs(value - region_start) <= collar:
        return region_start
    if abs(value - region_end) <= collar:
        return region_end
    return value


def reconcile_regions(
    regions: Iterable,
    turns: Sequence[SpeakerTurn],
    *,
    physical_source: str = "fused_vad",
    speaker_source: str = "diarization",
    boundary_collar_ms: float = 80.0,
    duration: Optional[float] = None,
    keep_unknown: bool = True,
) -> List[AtomicSpeechSpan]:
    """将物理语音区间与 speaker turns 求交集。

    ``regions`` 只需要提供 ``start`` 和 ``end`` 属性，也可传入
    ``(start, end)`` 元组。每个返回 span 最多归属于一个 speaker。
    """
    collar = max(0.0, float(boundary_collar_ms)) / 1000.0
    normalized = normalize_turns(turns, duration=duration)
    output: List[AtomicSpeechSpan] = []

    for region in regions:
        if isinstance(region, (tuple, list)):
            raw_start, raw_end = region[0], region[1]
        else:
            raw_start, raw_end = region.start, region.end
        start = max(0.0, float(raw_start))
        end = float(raw_end)
        if duration is not None:
            end = min(end, duration)
        if end <= start:
            continue

        overlaps = [
            turn for turn in normalized
            if turn.end > start and turn.start < end
        ]
        if not overlaps:
            if keep_unknown:
                output.append(
                    AtomicSpeechSpan(
                        start=start,
                        end=end,
                        speaker_id=None,
                        physical_source=physical_source,
                        speaker_source="unknown",
                    )
                )
            continue

        covered_until = start
        for turn in overlaps:
            span_start = max(start, turn.start)
            span_end = min(end, turn.end)
            span_start = _snap_to_physical_boundary(
                span_start, start, end, collar,
            )
            span_end = _snap_to_physical_boundary(
                span_end, start, end, collar,
            )
            span_start = max(start, min(span_start, end))
            span_end = max(start, min(span_end, end))

            if keep_unknown and span_start > covered_until + 1e-6:
                output.append(
                    AtomicSpeechSpan(
                        start=covered_until,
                        end=span_start,
                        speaker_id=None,
                        physical_source=physical_source,
                        speaker_source="unknown",
                    )
                )
            if span_end > span_start:
                output.append(
                    AtomicSpeechSpan(
                        start=span_start,
                        end=span_end,
                        speaker_id=turn.speaker_id,
                        physical_source=physical_source,
                        speaker_source=speaker_source,
                        overlapped=turn.overlapped,
                    )
                )
                covered_until = max(covered_until, span_end)

        if keep_unknown and covered_until < end - 1e-6:
            output.append(
                AtomicSpeechSpan(
                    start=covered_until,
                    end=end,
                    speaker_id=None,
                    physical_source=physical_source,
                    speaker_source="unknown",
                )
            )

    output.sort(key=lambda item: (item.start, item.end))
    return output


def merge_same_speaker_spans(
    spans: Sequence[AtomicSpeechSpan],
    *,
    max_gap: float = 0.12,
) -> List[AtomicSpeechSpan]:
    """只合并相邻且 speaker id 相同的 span。

    ``None`` 和不同 speaker 永远不会合并。该函数不通过 gap 推断身份。
    """
    if not spans:
        return []
    result: List[AtomicSpeechSpan] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if span.end <= span.start:
            continue
        if result:
            previous = result[-1]
            gap = span.start - previous.end
            if (
                previous.speaker_id is not None
                and previous.speaker_id == span.speaker_id
                and 0.0 <= gap <= max(0.0, max_gap)
                and not previous.overlapped
                and not span.overlapped
            ):
                result[-1] = replace(
                    previous,
                    end=max(previous.end, span.end),
                )
                continue
        result.append(span)
    return result


def split_event_intervals(
    start: float,
    end: float,
    turns: Sequence[SpeakerTurn],
) -> List[Tuple[float, float, Optional[int]]]:
    """按全局 turns 拆分一个字幕事件，供无词级时间戳 fallback 使用。"""
    if end <= start:
        return []
    pieces: List[Tuple[float, float, Optional[int]]] = []
    cursor = start
    for turn in normalize_turns(turns):
        if turn.end <= start or turn.start >= end:
            continue
        piece_start = max(start, turn.start)
        piece_end = min(end, turn.end)
        if piece_start > cursor:
            pieces.append((cursor, piece_start, None))
        if piece_end > piece_start:
            pieces.append((piece_start, piece_end, turn.speaker_id))
            cursor = max(cursor, piece_end)
    if cursor < end:
        pieces.append((cursor, end, None))
    return pieces
