"""静音幻听碎片吸收（时间轴前置修复，2026-09-13 诊断定案）

ASR 词级时间戳在换人/换句边界普遍偏早时，真实下一句的第一个音节会被
配上"骑在静音区"的时间（实例：「得了吧」被拆成骑在静音上的「得」
12.52-12.88 + 提前截止的「了吧」12.88-13.10，而真实语音在 12.92-13.36），
产生一整行几乎不含语音能量的幻听碎片。

该吸收器在说话人归属之前运行：用能量骨架计算每个事件的真实语音覆盖率，
把覆盖率过低的碎片文本并入"碎片结束之后第一个语音起点"所在的后继事件
（后继事件 start 锚定到该真实语音起点），前方无语音可并时整行丢弃。
有真实语音的事件一律不动。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..utils.text_utils import smart_join_texts

logger = logging.getLogger(__name__)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speech_coverage(
    start: float, end: float, skeleton: List[Tuple[float, float]],
) -> float:
    span = max(1e-6, end - start)
    return sum(_overlap(start, end, s, e) for s, e in skeleton) / span


def _next_onset_after(
    t: float, skeleton: List[Tuple[float, float]],
) -> Optional[float]:
    best: Optional[float] = None
    for s_start, _s_end in skeleton:
        if s_start > t - 0.02 and (best is None or s_start < best):
            best = s_start
    return best


def _merge_words(fragment: Any, target: Any) -> None:
    fw = list(getattr(fragment, "words", []) or [])
    tw = list(getattr(target, "words", []) or [])
    if fw or tw:
        target.words = fw + tw


def reanchor_word_timestamps(
    events: List[Any],
    audio: Any,
    sample_rate: int,
    *,
    max_reanchor_distance: float = 0.45,
) -> int:
    """词级时间戳能量重锚定：把"骑在静音上"的词起点吸到真实语音起点。

    finalize 的显示拆行以词时间戳为准（new_start = min(word.start)），
    ASR 词起点在句内停顿/换句边界偏早 200~300ms 时，拆出的字幕行会把
    停顿吞进行首。这里对整段静音（词首前方窗口无语音能量）的词做
    后向重锚定；词内已有语音能量的词一律不动。
    """
    if not events or audio is None:
        return 0
    from ..acoustic.validator import (
        _leading_silence_ahead,
        _next_energy_onset_after,
    )

    shifted = 0
    for event in events:
        words = list(getattr(event, "words", None) or [])
        for index, word in enumerate(words):
            try:
                w_start = float(word.start)
                w_end = float(word.end)
            except (TypeError, ValueError):
                continue
            if w_end - w_start <= 0:
                continue
            # 词首已有语音能量 → 时间戳可信，不动
            if not _leading_silence_ahead(
                audio, sample_rate, w_start, window_sec=0.10,
            ):
                continue
            onset = _next_energy_onset_after(
                audio, sample_rate, w_start, max_reanchor_distance,
            )
            if onset is None:
                continue
            # 保持单调：不能越过下一个词的起点
            if index + 1 < len(words):
                try:
                    next_start = float(words[index + 1].start)
                except (TypeError, ValueError):
                    next_start = None
                if next_start is not None and onset > next_start + 1e-6:
                    continue
            word.start = onset
            word.end = onset + (w_end - w_start)
            shifted += 1
    if shifted:
        logger.info("Word timestamps re-anchored: %d words", shifted)
    return shifted



def resolve_speech_skeleton(
    vocals_path,
    ffmpeg_unified_result,
    acoustic_config,
    audio,
    sample_rate: int,
) -> List[Tuple[float, float]]:
    """获取能量骨架：优先复用统一 ffmpeg 结果，否则独立检测。

    惰性导入 AcousticValidator 以避免 merging → acoustic 的模块级循环
    依赖（acoustic.validator 在模块级导入 merging.merge_engine）。
    """
    from ..acoustic.validator import AcousticValidator

    validator = AcousticValidator(acoustic_config)
    return validator._get_skeleton(
        vocals_path, ffmpeg_unified_result, audio, sample_rate,
    )


def absorb_silent_fragments(
    events: List[Any],
    speech_skeleton: List[Tuple[float, float]],
    *,
    min_coverage: float = 0.4,
    max_fragment_duration: float = 0.8,
    max_reanchor_distance: float = 0.5,
) -> List[Any]:
    """吸收骑在静音上的幻听碎片事件，返回精简后的事件列表。

    Args:
        events: 字幕事件列表（按时间顺序，含 start/end/text）
        speech_skeleton: 能量骨架语音区间 [(start, end), ...]
        min_coverage: 事件与骨架的最低重叠覆盖率，低于该值视为碎片
        max_fragment_duration: 只考虑短于该时长的碎片（长事件不吸收）
        max_reanchor_distance: 碎片结束到后继语音起点的最大重锚距离

    Returns:
        精简后的事件列表（原对象就地修改，碎片对象被移除）
    """
    if not events or not speech_skeleton:
        return events

    from .merge_engine import _physical_owner_compatible_for_events

    ordered = sorted(events, key=lambda e: float(getattr(e, "start", 0.0)))
    absorbed: set[int] = set()
    forward_count = 0
    backward_count = 0
    dropped_count = 0

    for index, frag in enumerate(ordered):
        if id(frag) in absorbed:
            continue
        duration = float(getattr(frag, "end", 0.0)) - float(
            getattr(frag, "start", 0.0)
        )
        if duration <= 0 or duration > max_fragment_duration:
            continue
        coverage = _speech_coverage(
            float(frag.start), float(frag.end), speech_skeleton,
        )
        if coverage >= min_coverage:
            continue

        text = (getattr(frag, "text", "") or "").strip()
        if not text:
            absorbed.add(id(frag))
            dropped_count += 1
            continue

        # 前向吸收：碎片结束后第一个语音起点所在的后继事件
        onset = _next_onset_after(float(frag.end), speech_skeleton)
        target = None
        if (
            onset is not None
            and onset - float(frag.end) <= max_reanchor_distance
        ):
            for candidate in ordered[index + 1:]:
                if id(candidate) in absorbed:
                    continue
                if (
                    float(candidate.start) <= onset + 0.1
                    and _physical_owner_compatible_for_events(frag, candidate)
                ):
                    target = candidate
                break
        if target is not None:
            target.text = smart_join_texts(
                [text, getattr(target, "text", "") or ""]
            )
            _merge_words(frag, target)
            # 后继事件 start 锚定到真实语音起点（只后移，不前拖）
            if float(target.start) < onset:
                target.start = onset
            absorbed.add(id(frag))
            forward_count += 1
            logger.info(
                "Silent fragment absorbed forward: %.2f-%.2f '%s' → onset %.2f",
                float(frag.start), float(frag.end), text[:12], onset,
            )
            continue

        # 后向吸收：碎片前方 0.5s 内有语音终点时并入前一事件
        prev_end = None
        for prev in reversed(ordered[:index]):
            if id(prev) in absorbed:
                continue
            prev_end = float(prev.end)
            if (
                0 <= float(frag.start) - prev_end <= max_reanchor_distance
                and _physical_owner_compatible_for_events(prev, frag)
            ):
                prev.text = smart_join_texts(
                    [getattr(prev, "text", "") or "", text]
                )
                _merge_words(frag, prev)
                absorbed.add(id(frag))
                backward_count += 1
                logger.info(
                    "Silent fragment absorbed backward: %.2f-%.2f '%s'",
                    float(frag.start), float(frag.end), text[:12],
                )
                target = prev
                break
        if target is not None:
            continue

        # 前后都没有可归属的语音：纯幻听，整行丢弃
        absorbed.add(id(frag))
        dropped_count += 1
        logger.info(
            "Silent fragment dropped: %.2f-%.2f '%s'",
            float(frag.start), float(frag.end), text[:12],
        )

    if absorbed:
        result = [e for e in events if id(e) not in absorbed]
        logger.info(
            "Fragment absorber: %d events in, %d removed "
            "(forward=%d, backward=%d, dropped=%d)",
            len(events), len(events) - len(result),
            forward_count, backward_count, dropped_count,
        )
        return result
    return events
