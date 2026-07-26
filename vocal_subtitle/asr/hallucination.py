"""ASR 结果的保守幻觉过滤。

该模块只消费 ASR 结果和质量元数据，不访问音频、模型或 Pipeline 状态，
因此可以同时用于新识别结果和磁盘缓存命中结果。
"""

from dataclasses import dataclass, field
import math
import re
import unicodedata
from typing import Any, Dict, List, Sequence

from .base import TranscriptionSegment


_MIN_WORD_CONFIDENCE = 0.35
_ADJACENT_DUPLICATE_GAP = 0.20
_TRAINING_PHRASES = (
    "字幕志愿者",
    "中文字幕志愿者",
    "感谢观看",
    "感谢收看",
    "thanks for watching",
)
_NORMALIZED_TRAINING_PHRASES = frozenset(
    unicodedata.normalize("NFKC", phrase).casefold().replace(" ", "")
    for phrase in _TRAINING_PHRASES
)


@dataclass(frozen=True)
class HallucinationFilterPolicy:
    """幻觉过滤策略。

    质量字段缺失时，过滤器跳过对应规则而不是猜测异常。
    """

    enabled: bool = True
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    filter_training_phrases: bool = True
    filter_adjacent_duplicates: bool = True
    version: str = "v1"


@dataclass(frozen=True)
class HallucinationFilterResult:
    """过滤结果和可序列化诊断。"""

    segments: List[TranscriptionSegment]
    dropped: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    filter_version: str = "v1"


def filter_transcription_segments(
    segments: Sequence[TranscriptionSegment],
    policy: HallucinationFilterPolicy,
) -> HallucinationFilterResult:
    """过滤明显的 ASR 幻觉，同时保留有词级证据的短词。

    输入对象不会被修改，返回列表也只包含原对象引用，便于保留现有数据契约。
    """
    if not policy.enabled:
        return HallucinationFilterResult(
            segments=list(segments),
            counts={"disabled": 1},
            filter_version=policy.version,
        )

    dropped: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    candidates = []

    for index, segment in enumerate(segments):
        text = str(getattr(segment, "text", "") or "").strip()
        normalized = _normalize_text(text)
        base = _diagnostic(index, segment, text)

        if not normalized:
            _record(dropped, base, "empty_text")
            continue

        if (
            policy.filter_training_phrases
            and normalized in _NORMALIZED_TRAINING_PHRASES
        ):
            _record(dropped, base, "training_phrase")
            continue

        has_word_evidence = _has_valid_word_evidence(segment)
        no_speech_prob = _finite_number(getattr(segment, "no_speech_prob", None))
        if (
            no_speech_prob is not None
            and no_speech_prob >= policy.no_speech_threshold
        ):
            if not has_word_evidence:
                _record(dropped, base, "high_no_speech_without_word_evidence")
                continue
            _record(warnings, base, "high_no_speech_with_word_evidence")

        avg_logprob = _finite_number(getattr(segment, "avg_logprob", None))
        if avg_logprob is not None and avg_logprob < policy.log_prob_threshold:
            if not has_word_evidence:
                _record(dropped, base, "low_logprob_without_word_evidence")
                continue
            _record(warnings, base, "low_logprob_with_word_evidence")

        compression_ratio = _finite_number(
            getattr(segment, "compression_ratio", None)
        )
        if (
            compression_ratio is not None
            and compression_ratio > policy.compression_ratio_threshold
        ):
            if _has_repetition(text):
                _record(dropped, base, "repetitive_compression")
                continue
            _record(warnings, base, "high_compression_without_repetition")

        candidates.append((index, segment, normalized, text))

    if policy.filter_adjacent_duplicates:
        candidates, duplicate_drops = _deduplicate_adjacent(candidates)
        dropped.extend(duplicate_drops)

    counts: Dict[str, int] = {}
    for item in dropped:
        counts[item["reason"]] = counts.get(item["reason"], 0) + 1
    for item in warnings:
        counts[item["reason"]] = counts.get(item["reason"], 0) + 1

    return HallucinationFilterResult(
        segments=[item[1] for item in candidates],
        dropped=dropped,
        warnings=warnings,
        counts=counts,
        filter_version=policy.version,
    )


def _deduplicate_adjacent(candidates):
    kept = []
    dropped = []
    for candidate in candidates:
        if not kept:
            kept.append(candidate)
            continue

        previous = kept[-1]
        if not _is_adjacent(previous[1], candidate[1]):
            kept.append(candidate)
            continue
        if not _is_duplicate_or_contained(previous[2], candidate[2]):
            kept.append(candidate)
            continue

        previous_score = _completeness_score(previous[1], previous[3])
        current_score = _completeness_score(candidate[1], candidate[3])
        if current_score > previous_score:
            kept.pop()
            dropped.append(
                _diagnostic(previous[0], previous[1], previous[3], "adjacent_duplicate")
            )
            kept.append(candidate)
        else:
            dropped.append(
                _diagnostic(candidate[0], candidate[1], candidate[3], "adjacent_duplicate")
            )
    return kept, dropped


def _is_adjacent(left: TranscriptionSegment, right: TranscriptionSegment) -> bool:
    left_end = _finite_number(getattr(left, "end", None))
    right_start = _finite_number(getattr(right, "start", None))
    if left_end is None or right_start is None:
        return False
    return right_start - left_end <= _ADJACENT_DUPLICATE_GAP


def _is_duplicate_or_contained(left: str, right: str) -> bool:
    return left == right or left in right or right in left


def _completeness_score(segment: TranscriptionSegment, text: str):
    duration = max(
        0.0,
        float(getattr(segment, "end", 0.0) or 0.0)
        - float(getattr(segment, "start", 0.0) or 0.0),
    )
    return (len(text), len(getattr(segment, "words", []) or []), duration)


def _has_valid_word_evidence(segment: TranscriptionSegment) -> bool:
    for word in getattr(segment, "words", []) or []:
        text = str(getattr(word, "word", "") or "").strip()
        start = _finite_number(getattr(word, "start", None))
        end = _finite_number(getattr(word, "end", None))
        confidence = _finite_number(getattr(word, "confidence", None))
        if (
            text
            and start is not None
            and end is not None
            and start < end
            and confidence is not None
            and confidence >= _MIN_WORD_CONFIDENCE
        ):
            return True
    return False


def _has_repetition(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.casefold())
    if len(tokens) >= 2 and any(a == b for a, b in zip(tokens, tokens[1:])):
        return True
    compact = _normalize_text(text)
    if len(compact) < 4:
        return False
    return bool(re.search(r"(.{2,})\1", compact))


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized, flags=re.UNICODE)


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _diagnostic(index, segment, text, reason=None):
    result = {
        "index": index,
        "text": text,
        "no_speech_prob": getattr(segment, "no_speech_prob", None),
        "avg_logprob": getattr(segment, "avg_logprob", None),
        "compression_ratio": getattr(segment, "compression_ratio", None),
    }
    if reason is not None:
        result["reason"] = reason
    return result


def _record(target, diagnostic, reason):
    item = dict(diagnostic)
    item["reason"] = reason
    target.append(item)
