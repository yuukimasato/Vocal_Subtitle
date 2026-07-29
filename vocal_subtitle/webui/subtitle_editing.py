"""Pure subtitle event transformations used by the WebUI batch editor."""

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple


class SubtitleBatchEditError(ValueError):
    """Raised when a batch subtitle operation cannot be applied safely."""


def _indexed_positions(
    events: List[Dict[str, Any]],
    indexes: Iterable[int],
    *,
    require_multiple: bool = False,
    require_contiguous: bool = False,
) -> List[int]:
    requested = list(indexes)
    if not requested:
        raise SubtitleBatchEditError("至少选择一条字幕")
    if len(set(requested)) != len(requested):
        raise SubtitleBatchEditError("字幕序号不能重复")

    positions_by_index = {event.get("index"): pos for pos, event in enumerate(events)}
    missing = [index for index in requested if index not in positions_by_index]
    if missing:
        raise SubtitleBatchEditError(f"字幕序号不存在: {missing}")
    if require_multiple and len(requested) < 2:
        raise SubtitleBatchEditError("合并至少需要选择两条字幕")

    positions = sorted(positions_by_index[index] for index in requested)
    if require_contiguous and positions != list(range(positions[0], positions[-1] + 1)):
        raise SubtitleBatchEditError("合并只能选择连续字幕")
    return positions


def _next_speaker_id(events: List[Dict[str, Any]]) -> int:
    used = {
        event.get("speaker_id")
        for event in events
        if isinstance(event.get("speaker_id"), int) and event.get("speaker_id") >= 0
    }
    candidate = 0
    while candidate in used:
        candidate += 1
    return candidate


def _generated_speaker_label(speaker_id: int) -> str:
    if speaker_id < 26:
        return f"说话人 {chr(ord('A') + speaker_id)}"
    return f"说话人 {speaker_id}"


def _resolve_speaker(
    events: List[Dict[str, Any]],
    speaker_id: Optional[int],
    speaker_label: Optional[str],
) -> Tuple[Optional[int], str]:
    label = (speaker_label or "").strip()
    if speaker_id is None and not label:
        raise SubtitleBatchEditError("说话人不能为空")
    if speaker_id is not None and speaker_id < 0:
        raise SubtitleBatchEditError("说话人 ID 无效")

    if speaker_id is None and label:
        matching = [
            event
            for event in events
            if (event.get("speaker_label") or "").strip() == label
        ]
        if matching:
            existing_id = matching[0].get("speaker_id")
            if existing_id is not None:
                speaker_id = existing_id
            else:
                # A label-only speaker is already a valid identity; do not
                # manufacture a second identity for the same label.
                return None, label

    if speaker_id is None:
        speaker_id = _next_speaker_id(events)
    if not label:
        label = next(
            (
                (event.get("speaker_label") or "").strip()
                for event in events
                if event.get("speaker_id") == speaker_id
                and (event.get("speaker_label") or "").strip()
            ),
            _generated_speaker_label(speaker_id),
        )
    return speaker_id, label


def apply_speaker_batch(
    events: List[Dict[str, Any]],
    indexes: Iterable[int],
    *,
    speaker_id: Optional[int] = None,
    speaker_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Apply one speaker identity to selected final subtitle events."""
    result = deepcopy(events)
    positions = _indexed_positions(result, indexes)
    resolved_id, resolved_label = _resolve_speaker(result, speaker_id, speaker_label)
    for position in positions:
        result[position]["speaker_id"] = resolved_id
        result[position]["speaker_label"] = resolved_label
    return result


def _merge_list_fields(selected: List[Dict[str, Any]], key: str) -> List[Any]:
    merged: List[Any] = []
    for event in selected:
        value = event.get(key) or []
        if isinstance(value, list):
            merged.extend(deepcopy(value))
    return merged


def _same_speaker(selected: List[Dict[str, Any]]) -> bool:
    first = selected[0]
    first_id = first.get("speaker_id")
    first_label = (first.get("speaker_label") or "").strip()
    return all(
        event.get("speaker_id") == first_id
        and (event.get("speaker_label") or "").strip() == first_label
        for event in selected[1:]
    )


def apply_merge(
    events: List[Dict[str, Any]], indexes: Iterable[int], *, separator: str = "newline"
) -> List[Dict[str, Any]]:
    """Merge a contiguous selection into one final subtitle event."""
    if separator not in {"newline", "space"}:
        raise SubtitleBatchEditError("合并分隔符只能是 newline 或 space")

    result = deepcopy(events)
    positions = _indexed_positions(
        result, indexes, require_multiple=True, require_contiguous=True
    )
    selected = [result[position] for position in positions]
    merged = deepcopy(selected[0])
    merged["start"] = min(event["start"] for event in selected)
    merged["end"] = max(event["end"] for event in selected)
    delimiter = "\n" if separator == "newline" else " "
    merged["text"] = delimiter.join(
        str(event.get("text") or "") for event in selected
    )
    merged["original_text"] = None

    for key in ("words", "physical_spans", "source_word_ids", "overlap_tracks", "revision_trace"):
        if key in merged:
            merged[key] = _merge_list_fields(selected, key)
    for start_key, end_key in (
        ("physical_start", "physical_end"),
        ("physical_bin_start", "physical_bin_end"),
    ):
        starts = [event.get(start_key) for event in selected if event.get(start_key) is not None]
        ends = [event.get(end_key) for event in selected if event.get(end_key) is not None]
        if starts:
            merged[start_key] = min(starts)
        if ends:
            merged[end_key] = max(ends)

    if not _same_speaker(selected):
        merged["speaker_id"] = None
        merged["speaker_label"] = None
        merged["speaker_status"] = ""
    merged["genuine_overlap"] = any(event.get("genuine_overlap") for event in selected)
    overlap_ids = {event.get("overlap_group_id") for event in selected}
    merged["overlap_group_id"] = overlap_ids.pop() if len(overlap_ids) == 1 else None

    first_position = positions[0]
    selected_positions = set(positions)
    result = [
        event for position, event in enumerate(result) if position not in selected_positions
    ]
    result.insert(first_position, merged)
    for index, event in enumerate(result, start=1):
        event["index"] = index
    return result


def apply_batch_edit(
    events: List[Dict[str, Any]],
    *,
    action: str,
    indexes: Iterable[int],
    speaker_id: Optional[int] = None,
    speaker_label: Optional[str] = None,
    separator: str = "newline",
) -> List[Dict[str, Any]]:
    if action == "speaker":
        return apply_speaker_batch(
            events,
            indexes,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
        )
    if action == "merge":
        return apply_merge(events, indexes, separator=separator)
    raise SubtitleBatchEditError("不支持的批量字幕操作")
