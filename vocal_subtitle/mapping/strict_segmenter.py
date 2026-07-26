"""Word-safe subtitle segmentation for phase four."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .time_mapper import SubtitleEvent


_PUNCTUATION = frozenset(
    " \t\r\n,，。.!！？?；;：:、………()（）[]【】{}｛｝<>《》\"'“”‘’"
)
_HARD_BOUNDARY_WARNINGS = frozenset(
    {"speaker_conflict", "discontinuous_physical_boundary", "hard_split"}
)


@dataclass(frozen=True)
class StrictSegmentationConfig:
    enabled: bool = True
    silence_gap: float = 0.35
    max_duration: float = 5.0
    max_chars_cjk: int = 20
    max_chars_latin: int = 42
    max_lines: int = 2
    split_on_sentence_end: bool = True
    split_on_soft_punctuation: bool = True
    soft_punctuation_min_duration: float = 1.15
    latin_pause_gap: float = 0.30
    latin_pause_min_duration: float = 2.0
    latin_max_duration: float = 4.5
    allow_short_same_owner_merge: bool = True
    unknown_run_max_duration: float = 1.2
    unknown_fragment_max_chars: int = 2
    unknown_run_max_gap: float = 0.35


@dataclass
class SegmentationResult:
    events: list[SubtitleEvent]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def normalize_event_text(
    events: Sequence[SubtitleEvent],
    *,
    max_fragment_chars: int = 2,
    max_gap: float = 0.15,
) -> list[SubtitleEvent]:
    """Normalize punctuation ownership and join safe short word fragments.

    ASR/physical-bin boundaries may leave a sentence-ending punctuation mark
    at the start of the next event.  Move that mark to the preceding event's
    text, then join only very short adjacent fragments with the same speaker
    and continuous physical ownership.  The operation is intentionally
    idempotent because it runs at more than one pipeline boundary.
    """
    result: list[SubtitleEvent] = []
    for event in sorted(events, key=lambda item: (item.start, item.end, item.index)):
        text = str(getattr(event, "text", "") or "").strip()
        leading, text = _split_leading_punctuation(text)

        if leading and result and _has_content(result[-1].text):
            result[-1].text = _append_text(result[-1].text, leading)

        event.text = text
        if not _has_content(text):
            # A punctuation-only event has no independent spoken content.
            # Its punctuation was already attached to the previous event.
            continue

        if result and _can_merge_short_fragment(
            result[-1], event, max_fragment_chars=max_fragment_chars, max_gap=max_gap
        ):
            _merge_event_in_place(result[-1], event)
        else:
            result.append(event)

    for index, event in enumerate(result, start=1):
        event.index = index
    return result


def repair_short_unknown_fragments(
    events: Sequence[SubtitleEvent],
    *,
    max_fragment_chars: int = 1,
    max_gap: float = 0.35,
    decision_fn: Callable[[SubtitleEvent, SubtitleEvent, SubtitleEvent], bool | None]
    | None = None,
) -> list[SubtitleEvent]:
    """Repair conservative speaker-unknown tail fragments.

    Speaker-turn projection can leave the final word of a physical region
    uncovered even when both neighboring events belong to the same speaker.
    Only the three-way consensus case is repaired here; an unknown event is
    never assigned from one neighbor alone, and interjections remain separate.
    """
    result = [event for event in sorted(
        events, key=lambda item: (item.start, item.end, item.index)
    )]
    index = 1
    while index < len(result) - 1:
        previous = result[index - 1]
        current = result[index]
        following = result[index + 1]
        if not _is_safe_unknown_tail_candidate(
            previous,
            current,
            following,
            max_fragment_chars=max_fragment_chars,
            max_gap=max_gap,
        ):
            index += 1
            continue

        if decision_fn is not None:
            try:
                # The callback may veto a merge, but it cannot authorize a
                # candidate that failed the deterministic safety gate above.
                if decision_fn(previous, current, following) is False:
                    index += 1
                    continue
            except Exception:
                # LLM failures must preserve the deterministic fallback.
                pass

        current.speaker_id = previous.speaker_id
        current.speaker_label = previous.speaker_label
        _merge_event_in_place(previous, current)
        result.pop(index)

    for item_index, event in enumerate(result, start=1):
        event.index = item_index
    return result


def repair_unknown_runs(
    events: Sequence[SubtitleEvent],
    *,
    max_fragment_chars: int = 2,
    max_run_duration: float = 1.2,
    max_gap: float = 0.35,
) -> tuple[list[SubtitleEvent], dict[str, Any]]:
    """Inherit a bounded unknown run only when both sides agree.

    Global diarization is authoritative. This helper repairs only the small
    holes that can be caused by projecting a speaker turn onto word events;
    it never invents a speaker from one neighbor or from pause alternation.
    The events remain separate so recognition provenance and physical bounds
    are preserved for the display layer to decide whether to merge them.
    """
    ordered = sorted(events, key=lambda item: (item.start, item.end, item.index))
    diagnostics: dict[str, Any] = {
        "input_unknown_event_count": sum(
            1 for event in ordered if getattr(event, "speaker_id", None) is None
        ),
        "repaired_run_count": 0,
        "repaired_event_count": 0,
        "repaired_event_indices": [],
        "blocked_reasons": {},
    }
    result = list(ordered)
    index = 0
    while index < len(result):
        if getattr(result[index], "speaker_id", None) is not None:
            index += 1
            continue

        run_start = index
        while (
            index + 1 < len(result)
            and getattr(result[index + 1], "speaker_id", None) is None
        ):
            index += 1
        run_end = index
        previous = result[run_start - 1] if run_start > 0 else None
        following = result[run_end + 1] if run_end + 1 < len(result) else None

        reason = _unknown_run_rejection_reason(
            result[run_start : run_end + 1],
            previous,
            following,
            max_fragment_chars=max_fragment_chars,
            max_run_duration=max_run_duration,
            max_gap=max_gap,
        )
        if reason is not None:
            diagnostics["blocked_reasons"][reason] = (
                diagnostics["blocked_reasons"].get(reason, 0) + 1
            )
            index += 1
            continue

        speaker_id = previous.speaker_id
        for event in result[run_start : run_end + 1]:
            event.speaker_id = speaker_id
            event.speaker_label = previous.speaker_label
            event.speaker_source = "unknown_run_inheritance"
            event.speaker_repair_reason = "bounded_unknown_run_same_neighbor"
            diagnostics["repaired_event_count"] += 1
            diagnostics["repaired_event_indices"].append(event.index)
        diagnostics["repaired_run_count"] += 1
        index += 1

    for item_index, event in enumerate(result, start=1):
        event.index = item_index
    diagnostics["output_unknown_event_count"] = sum(
        1 for event in result if getattr(event, "speaker_id", None) is None
    )
    return result, diagnostics


def segment_events(
    events: Sequence[SubtitleEvent],
    *,
    config: StrictSegmentationConfig | None = None,
    audio_duration: float | None = None,
) -> SegmentationResult:
    """Split events only at word boundaries and preserve provenance."""
    cfg = config or StrictSegmentationConfig()
    diagnostics = {
        "input_event_count": len(events),
        "output_event_count": 0,
        "split_count": 0,
        "sentence_split_count": 0,
        "duration_split_count": 0,
        "character_split_count": 0,
        "hard_boundary_count": 0,
    }
    normalized_input = normalize_event_text(events)
    if not cfg.enabled:
        result = _normalize(normalized_input, audio_duration)
        diagnostics["output_event_count"] = len(result)
        return SegmentationResult(result, diagnostics)

    output: list[SubtitleEvent] = []
    for event in normalized_input:
        pieces = _split_word_safe(event, cfg, diagnostics)
        if len(pieces) > 1:
            diagnostics["split_count"] += len(pieces) - 1
        output.extend(pieces)

    normalized = _normalize(normalize_event_text(output), audio_duration)
    diagnostics["output_event_count"] = len(normalized)
    return SegmentationResult(normalized, diagnostics)


def _split_word_safe(
    event: SubtitleEvent,
    cfg: StrictSegmentationConfig,
    diagnostics: dict[str, Any],
) -> list[SubtitleEvent]:
    words = list(getattr(event, "words", []) or [])
    if not words:
        return [event]

    is_latin = _is_latin_text(event.text)
    max_chars = cfg.max_chars_cjk * cfg.max_lines if _is_cjk_text(event.text) else cfg.max_chars_latin * cfg.max_lines
    max_duration = (
        min(cfg.max_duration, cfg.latin_max_duration)
        if is_latin
        else cfg.max_duration
    )
    groups: list[list[Any]] = []
    current: list[Any] = []
    current_chars = 0
    current_start = event.start

    for word in words:
        word_start = event.start + float(getattr(word, "start", 0.0))
        word_end = event.start + float(getattr(word, "end", 0.0))
        token = str(getattr(word, "word", "")).strip()
        token_chars = len(token.replace(" ", ""))
        gap = word_start - (event.start + float(getattr(current[-1], "end", 0.0))) if current else 0.0
        next_token = token
        sentence_boundary = bool(
            cfg.split_on_sentence_end
            and current
            and _ends_sentence(str(getattr(current[-1], "word", "")))
        )
        soft_boundary = bool(
            current
            and cfg.split_on_soft_punctuation
            and _should_split_soft_punctuation(
                event,
                current[-1],
                next_token,
                current_chars,
                current_start,
                cfg,
            )
        )
        over_limit = bool(
            current
            and (
                word_end - current_start > max_duration
                or current_chars + token_chars > max_chars
            )
        )
        duration_limit_hit = bool(
            current and word_end - current_start > max_duration
        )
        character_limit_hit = bool(
            current and current_chars + token_chars > max_chars
        )
        hard_boundary = bool(
            current and not _compatible_word_gap(event, word_start, gap, cfg)
        )
        # A comma followed by a normal list item is not a hard speech break;
        # keep the phrase together even when ASR leaves a slightly long gap.
        if (
            hard_boundary
            and is_latin
            and gap <= cfg.silence_gap + 0.15
            and _ends_soft_punctuation(
                str(getattr(current[-1], "word", ""))
            )
        ):
            hard_boundary = False
        # English ASR often omits commas at clause boundaries. A moderate
        # pause without a comma is still a safe whole-word split candidate.
        natural_pause_boundary = bool(
            current
            and is_latin
            and gap >= cfg.latin_pause_gap
            and (
                event.start
                + float(getattr(current[-1], "end", 0.0))
                - current_start
                >= cfg.latin_pause_min_duration
            )
            and not _ends_soft_punctuation(
                str(getattr(current[-1], "word", ""))
            )
        )

        if (
            sentence_boundary
            or soft_boundary
            or over_limit
            or hard_boundary
            or natural_pause_boundary
        ):
            groups.append(current)
            current = []
            current_chars = 0
            current_start = word_start
            if sentence_boundary:
                diagnostics["sentence_split_count"] += 1
            if soft_boundary:
                diagnostics["soft_punctuation_split_count"] = (
                    diagnostics.get("soft_punctuation_split_count", 0) + 1
                )
            if over_limit:
                diagnostics["duration_split_count"] += int(duration_limit_hit)
                diagnostics["character_split_count"] += int(character_limit_hit)
            if hard_boundary:
                diagnostics["hard_boundary_count"] += 1
            if natural_pause_boundary:
                diagnostics["natural_pause_split_count"] = (
                    diagnostics.get("natural_pause_split_count", 0) + 1
                )
        current.append(word)
        current_chars += token_chars
    if current:
        groups.append(current)

    if len(groups) == 1:
        return [event]
    return [_make_piece(event, group, index) for index, group in enumerate(groups)]


def _compatible_word_gap(
    event: SubtitleEvent,
    word_start: float,
    gap: float,
    cfg: StrictSegmentationConfig,
) -> bool:
    if gap < 0 or gap > cfg.silence_gap:
        return False
    if not cfg.allow_short_same_owner_merge:
        return False
    warning = str(getattr(event, "alignment_warning", "") or "")
    if any(item in warning.split(";") for item in ("speaker_conflict", "discontinuous_physical_boundary")):
        return False
    return True


_LATIN_CLAUSE_STARTERS = frozenset(
    {
        "but",
        "because",
        "either",
        "for",
        "if",
        "i'm",
        "it's",
        "once",
        "or",
        "so",
        "that",
        "they're",
        "you",
        "you'll",
    }
)


def _should_split_soft_punctuation(
    event: SubtitleEvent,
    previous_word: Any,
    next_token: str,
    current_chars: int,
    current_start: float,
    cfg: StrictSegmentationConfig,
) -> bool:
    """Split an English clause at a whole-word comma boundary."""
    if not _is_latin_text(event.text):
        return False
    previous = str(getattr(previous_word, "word", "")).strip()
    if not re.search(r"[,;；]$", previous) or not next_token:
        return False
    duration = (
        event.start
        + float(getattr(previous_word, "end", 0.0))
        - current_start
    )
    if duration < cfg.soft_punctuation_min_duration:
        return False
    normalized_next = next_token.strip(" \t\n\r\"'([{“‘").casefold()
    if normalized_next in _LATIN_CLAUSE_STARTERS:
        return True
    return duration >= cfg.max_duration - 0.5 and current_chars >= 24


def _make_piece(
    source: SubtitleEvent,
    words: list[Any],
    piece_index: int,
) -> SubtitleEvent:
    start = source.start + float(getattr(words[0], "start", 0.0))
    end = source.start + float(getattr(words[-1], "end", 0.0))
    text = _join_words([str(getattr(word, "word", "")) for word in words])
    source_word_ids = [
        str(getattr(word, "id"))
        for word in words
        if getattr(word, "id", None) is not None
    ] or _filter_source_ids(source, len(words), words)
    piece = copy.copy(source)
    piece.index = piece_index
    piece.start = max(source.start, start)
    piece.end = min(source.end, end)
    piece.text = text or source.text
    piece.asr_text = piece.text
    piece.original_text = None
    piece.llm_text = None
    # SubtitleEvent word timestamps are relative to the event start. A piece
    # can be segmented again by the exporter, so normalize the copied words
    # to the new piece origin instead of retaining the source-relative values.
    piece.words = []
    for word in words:
        copied = copy.copy(word)
        copied.start = float(getattr(word, "start", 0.0)) - (start - source.start)
        copied.end = float(getattr(word, "end", 0.0)) - (start - source.start)
        piece.words.append(copied)
    piece.hard_split_before = bool(
        getattr(source, "hard_split_before", False) or piece_index > 0
    )
    piece.physical_region_id = getattr(source, "physical_region_id", None)
    piece.source_word_ids = source_word_ids
    return piece


def _filter_source_ids(source: SubtitleEvent, count: int, words: list[Any]) -> list[str]:
    ids = list(getattr(source, "source_word_ids", []) or [])
    if not ids:
        return []
    all_words = list(getattr(source, "words", []) or [])
    try:
        start = all_words.index(words[0])
        return ids[start : start + count]
    except (ValueError, IndexError):
        return ids[:count]


def _normalize(events: list[SubtitleEvent], audio_duration: float | None) -> list[SubtitleEvent]:
    result = []
    for event in events:
        if audio_duration is not None:
            event.start = max(0.0, min(float(audio_duration), event.start))
            event.end = max(0.0, min(float(audio_duration), event.end))
        if event.end > event.start and _has_content(str(event.text)):
            result.append(event)
    for index, event in enumerate(result, start=1):
        event.index = index
    return result


def _is_cjk_text(text: str) -> bool:
    chars = re.sub(r"\s", "", text)
    return bool(chars) and sum("\u4e00" <= char <= "\u9fff" for char in chars) / len(chars) > 0.5


def _is_latin_text(text: str) -> bool:
    chars = re.sub(r"\s", "", text)
    if not chars:
        return False
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in chars)
    latin_count = sum(char.isascii() and char.isalpha() for char in chars)
    return latin_count > cjk_count and latin_count > 0


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?！？。]$", text.strip()))


def _ends_soft_punctuation(text: str) -> bool:
    return bool(re.search(r"[,;；]$", text.strip()))


def _split_leading_punctuation(text: str) -> tuple[str, str]:
    """Return leading punctuation separately from the spoken content."""
    index = 0
    while index < len(text) and text[index] in _PUNCTUATION:
        index += 1
    return text[:index], text[index:].strip()


def _has_content(text: str) -> bool:
    return any(char.isalnum() or "\u3400" <= char <= "\u9fff" for char in text)


def _append_text(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if left[-1] in ".。!?！？" and right[0] in ".。!?！？":
        return left
    if right[0] in _PUNCTUATION or left[-1] in "([{（【《“‘":
        return left + right
    if _is_cjk(right[0]) or _is_cjk(left[-1]):
        return left + right
    return f"{left} {right}"


def _is_cjk(value: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in value)


def _can_merge_short_fragment(
    previous: SubtitleEvent,
    current: SubtitleEvent,
    *,
    max_fragment_chars: int,
    max_gap: float,
) -> bool:
    previous_text = str(getattr(previous, "text", ""))
    current_text = str(getattr(current, "text", ""))
    previous_chars = len(previous_text.replace(" ", ""))
    current_chars = len(current_text.replace(" ", ""))
    if min(previous_chars, current_chars) > max_fragment_chars:
        return False
    if getattr(previous, "speaker_id", None) != getattr(current, "speaker_id", None):
        return False
    if _ends_sentence(previous_text):
        return False
    previous_end = float(getattr(previous, "end", 0.0))
    current_start = float(getattr(current, "start", 0.0))
    if current_start - previous_end < -0.03 or current_start - previous_end > max_gap:
        return False
    warnings = _warning_set(previous) | _warning_set(current)
    if warnings.intersection(_HARD_BOUNDARY_WARNINGS):
        return False
    if not _physical_owner_compatible(previous, current, max_gap=max_gap):
        return False
    return True


_STANDALONE_INTERJECTIONS = frozenset(
    {"啊", "哎", "唉", "哦", "喔", "嗯", "呃", "咦", "哈", "嘿", "呀"}
)


def _is_safe_unknown_tail_candidate(
    previous: SubtitleEvent,
    current: SubtitleEvent,
    following: SubtitleEvent,
    *,
    max_fragment_chars: int,
    max_gap: float,
) -> bool:
    if getattr(current, "speaker_id", None) is not None:
        return False
    speaker_id = getattr(previous, "speaker_id", None)
    if speaker_id is None or speaker_id != getattr(following, "speaker_id", None):
        return False

    text = str(getattr(current, "text", "") or "").strip()
    content = re.sub(r"[\s,，。.!！？?；;：:、\"'“”‘’()（）\[\]【】{}<>《》]", "", text)
    if not content or len(content) > max_fragment_chars:
        return False
    if content in _STANDALONE_INTERJECTIONS:
        return False
    if _ends_sentence(str(getattr(previous, "text", ""))) or _ends_soft_punctuation(
        str(getattr(previous, "text", ""))
    ):
        return False
    if bool(getattr(previous, "hard_split_after", False)) or bool(
        getattr(current, "hard_split_before", False)
    ):
        return False

    previous_gap = float(current.start) - float(previous.end)
    following_gap = float(following.start) - float(current.end)
    if not (0.0 <= previous_gap <= max_gap and 0.0 <= following_gap <= max_gap):
        return False
    if not _physical_owner_compatible(previous, current, max_gap=max_gap):
        return False
    if not _physical_owner_compatible(current, following, max_gap=max_gap):
        return False
    warnings = _warning_set(previous) | _warning_set(current) | _warning_set(following)
    return not warnings.intersection(_HARD_BOUNDARY_WARNINGS)


def _unknown_run_rejection_reason(
    run: Sequence[SubtitleEvent],
    previous: SubtitleEvent | None,
    following: SubtitleEvent | None,
    *,
    max_fragment_chars: int,
    max_run_duration: float,
    max_gap: float,
) -> str | None:
    if previous is None or following is None:
        return "missing_known_neighbor"
    if previous.speaker_id is None or previous.speaker_id != following.speaker_id:
        return "neighbor_speaker_conflict"
    if not run or run[-1].end - run[0].start > max_run_duration:
        return "run_duration_limit"

    for event in run:
        text = str(getattr(event, "text", "") or "").strip()
        content = re.sub(
            r"[\s,，。.!！？?；;：:、\"'“”‘’()（）\[\]【】{}<>《》]",
            "",
            text,
        )
        if not content:
            return "empty_fragment"
        if len(content) > max_fragment_chars:
            return "fragment_length_limit"
        if content in _STANDALONE_INTERJECTIONS:
            return "standalone_interjection"

    surrounding = [previous, *run, following]
    for left, right in zip(surrounding, surrounding[1:]):
        gap = float(right.start) - float(left.end)
        if gap < 0.0 or gap > max_gap:
            return "gap_limit"
        if not _physical_owner_compatible(left, right, max_gap=max_gap):
            return "physical_owner_conflict"
        if bool(getattr(left, "hard_split_after", False)) or bool(
            getattr(right, "hard_split_before", False)
        ):
            return "hard_boundary"
        if _warning_set(left).intersection(_HARD_BOUNDARY_WARNINGS) or _warning_set(
            right
        ).intersection(_HARD_BOUNDARY_WARNINGS):
            return "hard_boundary"

    if _ends_sentence(str(previous.text)) or _ends_soft_punctuation(str(previous.text)):
        return "preceding_punctuation_boundary"
    return None


def _warning_set(event: SubtitleEvent) -> set[str]:
    value = str(getattr(event, "alignment_warning", "") or "")
    return {item for item in value.split(";") if item}


def _physical_owner_compatible(
    left: SubtitleEvent,
    right: SubtitleEvent,
    *,
    max_gap: float = 0.02,
) -> bool:
    left_region = getattr(left, "physical_region_id", None)
    right_region = getattr(right, "physical_region_id", None)
    if left_region is not None or right_region is not None:
        if left_region != right_region:
            return False
    left_spans = list(getattr(left, "physical_spans", []) or [])
    right_spans = list(getattr(right, "physical_spans", []) or [])
    if not left_spans and not right_spans:
        return True
    if not left_spans or not right_spans:
        return False

    def clip_id(span: Any) -> str | None:
        if isinstance(span, dict):
            value = span.get("physical_clip_id", span.get("clip_id"))
        else:
            value = getattr(span, "clip_id", None)
        return str(value) if value is not None else None

    def span_start(span: Any) -> float:
        return float(span.get("start", 0.0) if isinstance(span, dict) else getattr(span, "start", 0.0))

    def span_end(span: Any) -> float:
        return float(span.get("end", 0.0) if isinstance(span, dict) else getattr(span, "end", 0.0))

    left_clip_ids = {clip_id(span) for span in left_spans if clip_id(span)}
    right_clip_ids = {clip_id(span) for span in right_spans if clip_id(span)}
    if left_clip_ids and right_clip_ids and not left_clip_ids.intersection(right_clip_ids):
        return False

    ordered = sorted([*left_spans, *right_spans], key=span_start)
    return all(
        span_start(next_span) - span_end(span) <= max_gap
        for span, next_span in zip(ordered, ordered[1:])
    )


def _merge_event_in_place(previous: SubtitleEvent, current: SubtitleEvent) -> None:
    """Merge a short event while preserving relative word timestamps."""
    base_start = float(previous.start)
    offset = float(current.start) - base_start
    shifted_words = []
    for word in list(getattr(current, "words", []) or []):
        copied = copy.copy(word)
        copied.start = float(getattr(word, "start", 0.0)) + offset
        copied.end = float(getattr(word, "end", 0.0)) + offset
        shifted_words.append(copied)

    previous.text = _append_text(previous.text, current.text)
    previous.end = max(float(previous.end), float(current.end))
    previous.words = [*(getattr(previous, "words", []) or []), *shifted_words]
    previous.source_word_ids = list(dict.fromkeys([
        *(getattr(previous, "source_word_ids", []) or []),
        *(getattr(current, "source_word_ids", []) or []),
    ]))
    previous.physical_spans = [
        *(getattr(previous, "physical_spans", []) or []),
        *(getattr(current, "physical_spans", []) or []),
    ]
    for field_name in ("physical_start", "physical_end"):
        left = getattr(previous, field_name, None)
        right = getattr(current, field_name, None)
        if left is None:
            setattr(previous, field_name, right)
        elif right is not None:
            setattr(previous, field_name, min(left, right) if field_name.endswith("start") else max(left, right))
    warnings = _warning_set(previous) | _warning_set(current)
    previous.alignment_warning = ";".join(sorted(warnings)) or None
    if getattr(previous, "speaker_source", None) != getattr(
        current, "speaker_source", None
    ):
        previous.speaker_source = "merged"
    if getattr(current, "speaker_repair_reason", None):
        previous.speaker_repair_reason = current.speaker_repair_reason
    if getattr(current, "physical_bin_end", None) is not None:
        previous.physical_bin_end = current.physical_bin_end


def _join_words(words: Sequence[str]) -> str:
    result = ""
    for raw in words:
        token = raw.strip()
        if not token:
            continue
        if not result:
            result = token
        elif (
            token[0] in '，。！？；：、,.!?;:)]}）】》”’"'
            or result[-1] in "([{（【《“‘"
            or _is_cjk_text(token[0])
            or _is_cjk_text(result[-1])
        ):
            result += token
        else:
            result += " " + token
    return result
