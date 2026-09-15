"""Adapt evidence decisions into the validated global physical IR."""

from __future__ import annotations

from collections.abc import Sequence

from ..asr.evidence import EvidenceDecision, EvidenceWord
from .ir import GlobalTranscript, GlobalTranscriptSegment, GlobalWord


def decisions_to_global_transcript(
    decisions: Sequence[EvidenceDecision],
    *,
    audio_duration: float | None = None,
    backend: str = "evidence-decision",
) -> GlobalTranscript:
    """Materialize accepted decisions as one global segment per decision.

    Decision coordinates are absolute. Missing word timestamps are filled with
    explicit segment-boundary observations so downstream allocation can work
    without pretending they were native word timings.
    """
    words: list[GlobalWord] = []
    segments: list[GlobalTranscriptSegment] = []
    diagnostics: dict[str, object] = {
        "decision_count": len(decisions),
        "skipped_decisions": [],
        "synthetic_word_count": 0,
        "clamped_word_count": 0,
        "word_order_repaired_count": 0,
    }

    for index, decision in enumerate(decisions, start=1):
        if decision.decision == "drop":
            continue
        if not decision.final_text or decision.start is None or decision.end is None:
            diagnostics["skipped_decisions"].append(
                {
                    "index": index,
                    "candidate_ids": list(decision.candidate_ids),
                    "reason": "missing_final_text_or_time",
                }
            )
            continue

        start = max(0.0, float(decision.start))
        end = float(decision.end)
        if audio_duration is not None:
            end = min(end, float(audio_duration))
        if end <= start:
            diagnostics["skipped_decisions"].append(
                {
                    "index": index,
                    "candidate_ids": list(decision.candidate_ids),
                    "reason": "outside_audio_duration",
                }
            )
            continue

        segment_id = f"decision-segment:{index:06d}"
        decision_id = f"decision:{index:06d}"
        materialized = _materialize_words(
            decision.final_words,
            start=start,
            end=end,
            decision_id=decision_id,
            segment_id=segment_id,
            decision=decision,
            diagnostics=diagnostics,
        )
        words.extend(materialized)
        word_ids = [word.id for word in materialized]
        segments.append(
            GlobalTranscriptSegment(
                id=segment_id,
                text=decision.final_text,
                raw_start=start,
                raw_end=end,
                word_ids=word_ids,
                metadata={
                    "decision_id": decision_id,
                    "candidate_ids": list(decision.candidate_ids),
                    "decision": decision.decision,
                    "risk_level": decision.risk_level,
                    "risk_score": decision.risk_score,
                    "evidence_codes": list(decision.evidence_codes),
                    "revision_trace": list(decision.revision_trace),
                },
            )
        )

    return GlobalTranscript(
        audio_duration=audio_duration,
        words=words,
        segments=segments,
        backend=backend,
        status="degraded" if diagnostics["skipped_decisions"] else "ok",
        diagnostics=diagnostics,
    )


def _materialize_words(
    source_words: Sequence[EvidenceWord],
    *,
    start: float,
    end: float,
    decision_id: str,
    segment_id: str,
    decision: EvidenceDecision,
    diagnostics: dict[str, object],
) -> list[GlobalWord]:
    items = list(source_words)
    ordered_items = sorted(
        enumerate(items),
        key=lambda item: (
            item[1].start is None,
            item[1].start if item[1].start is not None else float("inf"),
            item[1].end if item[1].end is not None else float("inf"),
            item[0],
        ),
    )
    if [index for index, _ in ordered_items] != list(range(len(items))):
        diagnostics["word_order_repaired_count"] += 1
    items = [item for _, item in ordered_items]
    if not items:
        diagnostics["synthetic_word_count"] += 1
        items = [
            EvidenceWord(
                id=f"{decision_id}:text",
                text=decision.final_text,
                time_source="segment_boundary",
            )
        ]

    ranges: list[tuple[float, float, bool]] = []
    for word in items:
        if word.start is None or word.end is None:
            ranges.append((0.0, 0.0, False))
            continue
        word_start = max(start, float(word.start))
        word_end = min(end, float(word.end))
        if word_end <= word_start:
            ranges.append((0.0, 0.0, False))
            continue
        clamped = word_start != float(word.start) or word_end != float(word.end)
        if clamped:
            diagnostics["clamped_word_count"] += 1
        ranges.append((word_start, word_end, clamped))

    cursor = 0
    while cursor < len(items):
        if ranges[cursor][2] or ranges[cursor][1] > ranges[cursor][0]:
            cursor += 1
            continue
        run_end = cursor
        while run_end < len(items) and not (
            ranges[run_end][2] or ranges[run_end][1] > ranges[run_end][0]
        ):
            run_end += 1
        left = ranges[cursor - 1][1] if cursor else start
        right = next(
            (
                ranges[pos][0]
                for pos in range(run_end, len(items))
                if ranges[pos][2] or ranges[pos][1] > ranges[pos][0]
            ),
            end,
        )
        if right <= left:
            left, right = start, end
        width = (right - left) / (run_end - cursor)
        for pos in range(cursor, run_end):
            item_start = left + width * (pos - cursor)
            item_end = left + width * (pos - cursor + 1)
            ranges[pos] = (item_start, item_end, False)
            diagnostics["synthetic_word_count"] += 1
        cursor = run_end

    result: list[GlobalWord] = []
    for index, (source, timing) in enumerate(zip(items, ranges)):
        word_start, word_end, clamped = timing
        if word_end <= word_start:
            continue
        metadata = {
            **dict(source.diagnostics),
            "decision_id": decision_id,
            "source_word_id": source.id,
            "decision": decision.decision,
            "candidate_ids": list(decision.candidate_ids),
            "risk_level": decision.risk_level,
            "risk_score": decision.risk_score,
            "evidence_codes": list(decision.evidence_codes),
            "revision_trace": list(decision.revision_trace),
            "time_source": source.time_source if not clamped else "segment_boundary",
        }
        if clamped:
            metadata["timestamp_clamped"] = True
        result.append(
            GlobalWord(
                id=f"{decision_id}:word:{index:04d}:{source.id}",
                text=source.text,
                raw_start=word_start,
                raw_end=word_end,
                confidence=source.confidence,
                source_window_id=decision_id,
                segment_id=segment_id,
                speaker_id=source.speaker_id,
                metadata=metadata,
            )
        )
    final_order = sorted(
        enumerate(result),
        key=lambda item: (item[1].raw_start, item[1].raw_end, item[1].id),
    )
    if [index for index, _ in final_order] != list(range(len(result))):
        diagnostics["word_order_repaired_count"] += 1
    return [item for _, item in final_order]


__all__ = ["decisions_to_global_transcript"]
