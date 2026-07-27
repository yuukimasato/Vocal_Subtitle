"""Deterministic assignment of global words to physical audio ranges."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..diarization.base import SpeakerTurn
from .ir import (
    GlobalSpeakerTimeline,
    GlobalTranscript,
    GlobalTranscriptSegment,
    GlobalWord,
)
from .subtitle_bins import PhysicalSubtitleBin
from .boundary_arbiter import BoundaryDecision
from .timeline import PhysicalTimeline, SpeechEvidenceSpan


@dataclass(frozen=True)
class PhysicalSpan:
    """The portion of a word that belongs to one physical clip."""

    clip_id: str
    start: float
    end: float
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_clip_id": self.clip_id,
            "start": self.start,
            "end": self.end,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class WordAllocation:
    word: GlobalWord
    physical_spans: tuple[PhysicalSpan, ...]
    evidence_ids: tuple[str, ...] = ()
    evidence_spans: tuple[SpeechEvidenceSpan, ...] = ()
    speaker_id: int | None = None
    speaker_source: str = "unknown"
    warnings: tuple[str, ...] = ()
    accepted: bool = True
    aligned_start: float | None = None
    aligned_end: float | None = None
    physical_bin_id: str | None = None
    boundary_confidence: float = 0.0
    alignment_status: str = "unaligned"
    start_boundary_decision: BoundaryDecision | None = None
    end_boundary_decision: BoundaryDecision | None = None
    boundary_evidence_ids: tuple[str, ...] = ()

    @property
    def clip_ids(self) -> tuple[str, ...]:
        return tuple(span.clip_id for span in self.physical_spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word.to_dict(),
            "physical_spans": [item.to_dict() for item in self.physical_spans],
            "evidence_ids": list(self.evidence_ids),
            "speaker_id": self.speaker_id,
            "speaker_source": self.speaker_source,
            "warnings": list(self.warnings),
            "accepted": self.accepted,
            "aligned_start": self.aligned_start,
            "aligned_end": self.aligned_end,
            "physical_bin_id": self.physical_bin_id,
            "boundary_confidence": self.boundary_confidence,
            "alignment_status": self.alignment_status,
            "start_boundary_decision": (
                self.start_boundary_decision.to_dict()
                if self.start_boundary_decision else None
            ),
            "end_boundary_decision": (
                self.end_boundary_decision.to_dict()
                if self.end_boundary_decision else None
            ),
            "boundary_evidence_ids": list(self.boundary_evidence_ids),
        }


@dataclass
class AllocationResult:
    allocations: list[WordAllocation] = field(default_factory=list)
    rejected: list[WordAllocation] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> list[WordAllocation]:
        return [item for item in self.allocations if item.accepted]

    @property
    def accepted_words(self) -> list[GlobalWord]:
        return [item.word for item in self.accepted]


def allocate_words(
    transcript: GlobalTranscript,
    physical_timeline: PhysicalTimeline,
    speaker_timeline: GlobalSpeakerTimeline | None = None,
    *,
    boundary_epsilon: float = 0.005,
    strict_discontinuous_boundary: bool = False,
    subtitle_bins: Sequence[PhysicalSubtitleBin] | None = None,
) -> AllocationResult:
    """Assign words without changing their text or raw timestamps.

    Physical clips constrain ownership; they are not fixed-size subtitle bins.
    A word crossing adjacent clips remains whole and receives multiple spans.
    """
    if not isinstance(transcript, GlobalTranscript):
        raise ValueError("transcript must be a GlobalTranscript")
    if not isinstance(physical_timeline, PhysicalTimeline):
        raise ValueError("physical_timeline must be a PhysicalTimeline")
    if speaker_timeline is not None and not isinstance(
        speaker_timeline, GlobalSpeakerTimeline
    ):
        raise ValueError("speaker_timeline must be a GlobalSpeakerTimeline or None")
    if boundary_epsilon < 0:
        raise ValueError("boundary_epsilon must be non-negative")
    if subtitle_bins is not None and any(
        item.end <= item.start for item in subtitle_bins
    ):
        raise ValueError("subtitle_bins must contain positive ranges")

    segment_by_word = _segment_index(transcript.segments)
    result = AllocationResult(
        diagnostics={
            "word_count": len(transcript.words),
            "accepted_count": 0,
            "rejected_count": 0,
            "cross_boundary_count": 0,
            "evidence_missing_count": 0,
            "speech_bin_rejected_count": 0,
            "speaker_fallback_count": 0,
        }
    )

    for word in transcript.words:
        clips = [
            clip
            for clip in physical_timeline.physical_clips
            if clip.start < word.raw_end and clip.end > word.raw_start
        ]
        spans: list[PhysicalSpan] = []
        evidence_ids: list[str] = []
        evidence_spans: list[SpeechEvidenceSpan] = []
        for clip in clips:
            start = max(word.raw_start, clip.start)
            end = min(word.raw_end, clip.end)
            if end <= start:
                continue
            evidence = [
                item
                for item in physical_timeline.speech_evidence_spans
                if item.physical_clip_id == clip.id
                and item.start < end
                and item.end > start
            ]
            ids = tuple(item.id for item in evidence)
            evidence_ids.extend(ids)
            evidence_spans.extend(evidence)
            spans.append(PhysicalSpan(clip.id, start, end, ids))

        warnings: list[str] = []
        if not spans:
            allocation = WordAllocation(
                word=word,
                physical_spans=(),
                evidence_ids=(),
                warnings=("outside_physical_clip",),
                accepted=False,
            )
            result.rejected.append(allocation)
            result.diagnostics["rejected_count"] += 1
            continue

        # Macro physical clips may include long silent intervals. The global
        # path supplies precise speech bins so ASR hallucinations in those
        # intervals cannot become subtitle events merely because they remain
        # inside a broad clip.
        if subtitle_bins is not None and not _overlaps_speech_bin(
            word,
            subtitle_bins,
            epsilon=boundary_epsilon,
        ):
            allocation = WordAllocation(
                word=word,
                physical_spans=tuple(spans),
                evidence_ids=tuple(dict.fromkeys(evidence_ids)),
                evidence_spans=tuple(evidence_spans),
                warnings=("outside_speech_bin",),
                accepted=False,
            )
            result.rejected.append(allocation)
            result.diagnostics["rejected_count"] += 1
            result.diagnostics["speech_bin_rejected_count"] += 1
            continue

        if len(spans) > 1:
            result.diagnostics["cross_boundary_count"] += 1
            warnings.append("cross_physical_boundary")
            if strict_discontinuous_boundary and not _spans_are_contiguous(
                spans, boundary_epsilon
            ):
                warnings.append("discontinuous_physical_boundary")
                allocation = WordAllocation(
                    word=word,
                    physical_spans=tuple(spans),
                    evidence_ids=tuple(dict.fromkeys(evidence_ids)),
                    evidence_spans=tuple(evidence_spans),
                    warnings=tuple(warnings),
                    accepted=False,
                )
                result.rejected.append(allocation)
                result.diagnostics["rejected_count"] += 1
                continue

        unique_evidence_ids = tuple(dict.fromkeys(evidence_ids))
        if not unique_evidence_ids:
            warnings.append("missing_speech_evidence")
            result.diagnostics["evidence_missing_count"] += 1

        speaker_id, speaker_source, speaker_warning = _resolve_speaker(
            word,
            segment_by_word.get(word.id),
            speaker_timeline,
        )
        if speaker_source not in ("word", "unknown"):
            result.diagnostics["speaker_fallback_count"] += 1
        if speaker_warning:
            warnings.append(speaker_warning)

        allocation = WordAllocation(
            word=word,
            physical_spans=tuple(spans),
            evidence_ids=unique_evidence_ids,
            evidence_spans=tuple(evidence_spans),
            speaker_id=speaker_id,
            speaker_source=speaker_source,
            warnings=tuple(dict.fromkeys(warnings)),
        )
        result.allocations.append(allocation)
        result.diagnostics["accepted_count"] += 1

    result.allocations.sort(
        key=lambda item: (item.word.raw_start, item.word.raw_end, item.word.id)
    )
    result.rejected.sort(
        key=lambda item: (item.word.raw_start, item.word.raw_end, item.word.id)
    )
    return result


def repair_late_words(
    transcript: GlobalTranscript,
    bins: Sequence[PhysicalSubtitleBin],
    *,
    max_late_start: float = 0.08,
    max_previous_gap: float = 0.5,
) -> GlobalTranscript:
    """Move ASR words that start just after a physical speech bin.

    Some ASR windows report a trailing word after the acoustic speech has
    already ended. When the word is immediately adjacent to the previous
    word from the same window, the physical bin provides a safer end time:
    ``previous_word.end -> speech_bin.end``. This preserves valid trailing
    words such as a final Chinese character while rejecting distant silence
    hallucinations during allocation.
    """
    if not isinstance(transcript, GlobalTranscript):
        raise ValueError("transcript must be a GlobalTranscript")
    if max_late_start < 0 or max_previous_gap < 0:
        raise ValueError("repair tolerances must be non-negative")
    ordered_bins = sorted(bins, key=lambda item: (item.start, item.end, item.id))
    ordered_words = sorted(
        transcript.words, key=lambda item: (item.raw_start, item.raw_end, item.id)
    )
    repaired: list[GlobalWord] = []
    repairs: list[dict[str, Any]] = []
    for word in ordered_words:
        if _overlaps_speech_bin(word, ordered_bins, epsilon=0.0):
            repaired.append(word)
            continue

        previous = repaired[-1] if repaired else None
        late_bin = next(
            (
                item
                for item in ordered_bins
                if 0.0 <= word.raw_start - item.end <= max_late_start
            ),
            None,
        )
        if (
            previous is None
            or late_bin is None
            or previous.source_window_id != word.source_window_id
            or previous.segment_id != word.segment_id
            or previous.raw_end > word.raw_start
            or word.raw_start - previous.raw_end > max_previous_gap
            or previous.raw_end >= late_bin.end
        ):
            repaired.append(word)
            continue

        repaired_start = max(previous.raw_end, late_bin.start)
        repaired_end = float(late_bin.end)
        if repaired_end <= repaired_start:
            repaired.append(word)
            continue
        repaired_word = GlobalWord(
            id=word.id,
            text=word.text,
            raw_start=repaired_start,
            raw_end=repaired_end,
            confidence=word.confidence,
            source_window_id=word.source_window_id,
            segment_id=word.segment_id,
            language=word.language,
            speaker_id=word.speaker_id,
            no_speech_prob=word.no_speech_prob,
            avg_logprob=word.avg_logprob,
            compression_ratio=word.compression_ratio,
            metadata={
                **word.metadata,
                "timestamp_repaired": True,
                "original_raw_start": word.raw_start,
                "original_raw_end": word.raw_end,
                "repair_bin_id": late_bin.id,
            },
        )
        repaired.append(repaired_word)
        repairs.append({
            "word_id": word.id,
            "original": [word.raw_start, word.raw_end],
            "repaired": [repaired_start, repaired_end],
            "bin_id": late_bin.id,
        })

    if not repairs:
        return transcript

    words_by_id = {word.id: word for word in repaired}
    segments = []
    for segment in transcript.segments:
        segment_words = [words_by_id[word_id] for word_id in segment.word_ids]
        segments.append(
            GlobalTranscriptSegment(
                id=segment.id,
                text=segment.text,
                raw_start=min(
                    [segment.raw_start, *(word.raw_start for word in segment_words)]
                ),
                raw_end=max(
                    [segment.raw_end, *(word.raw_end for word in segment_words)]
                ),
                word_ids=list(segment.word_ids),
                language=segment.language,
                avg_logprob=segment.avg_logprob,
                metadata=dict(segment.metadata),
            )
        )
    return GlobalTranscript(
        schema_version=transcript.schema_version,
        audio_duration=transcript.audio_duration,
        words=repaired,
        segments=segments,
        backend=transcript.backend,
        status=transcript.status,
        diagnostics={
            **transcript.diagnostics,
            "late_word_repairs": repairs,
        },
    )


def _segment_index(
    segments: Sequence[GlobalTranscriptSegment],
) -> dict[str, GlobalTranscriptSegment]:
    index: dict[str, GlobalTranscriptSegment] = {}
    for segment in segments:
        for word_id in segment.word_ids:
            index[word_id] = segment
    return index


def _spans_are_contiguous(spans: Sequence[PhysicalSpan], epsilon: float) -> bool:
    ordered = sorted(spans, key=lambda item: (item.start, item.end, item.clip_id))
    return all(
        right.start - left.end <= epsilon for left, right in zip(ordered, ordered[1:])
    )


def _overlaps_speech_bin(
    word: GlobalWord,
    bins: Sequence[PhysicalSubtitleBin],
    *,
    epsilon: float,
) -> bool:
    return any(
        word.raw_start < float(item.end) + epsilon
        and word.raw_end > float(item.start) - epsilon
        for item in bins
    )


def _resolve_speaker(
    word: GlobalWord,
    segment: GlobalTranscriptSegment | None,
    timeline: GlobalSpeakerTimeline | None,
) -> tuple[int | None, str, str | None]:
    if word.speaker_id is not None:
        return word.speaker_id, "word", None

    segment_speaker = None
    if segment is not None:
        candidate = segment.metadata.get("speaker_id")
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and candidate >= 0
        ):
            segment_speaker = candidate
    if segment_speaker is not None:
        return segment_speaker, "segment", None

    if timeline is None:
        return None, "unknown", None

    speaker, source, warning = _speaker_from_turns(
        word, timeline.exclusive_turns, "exclusive_turn"
    )
    if source != "unknown":
        return speaker, source, warning
    return _speaker_from_turns(word, timeline.turns, "turn")


def _speaker_from_turns(
    word: GlobalWord,
    turns: Sequence[SpeakerTurn],
    source: str,
) -> tuple[int | None, str, str | None]:
    midpoint = (word.raw_start + word.raw_end) / 2.0
    matches = [turn for turn in turns if turn.start <= midpoint < turn.end]
    speaker_ids = sorted({turn.speaker_id for turn in matches})
    if len(speaker_ids) == 1:
        return speaker_ids[0], source, None
    if len(speaker_ids) > 1:
        return None, "mixed", "speaker_conflict"
    return None, "unknown", None
