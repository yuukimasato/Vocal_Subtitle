"""Shadow construction for the phase-two physical/global IRs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..asr.base import TranscriptionSegment
from .context import build_context_windows
from .evidence_adapter import EvidenceAdaptResult, build_timeline_from_context
from .ir import (
    GlobalSpeakerTimeline,
    GlobalTranscript,
    GlobalTranscriptSegment,
    GlobalWord,
    adapt_diarization_result,
    adapt_transcription_segments,
)
from .timeline import PhysicalTimeline


@dataclass
class ShadowBuildResult:
    physical_timeline: PhysicalTimeline
    global_transcript: GlobalTranscript
    global_speaker_timeline: GlobalSpeakerTimeline
    status: str = "ok"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "diagnostics": copy.deepcopy(self.diagnostics),
            "statistics": copy.deepcopy(self.statistics),
            "physical_timeline": self.physical_timeline.to_dict(),
            "global_transcript": self.global_transcript.to_dict(),
            "global_speaker_timeline": self.global_speaker_timeline.to_dict(),
        }


def _build_ownership_timeline(duration: float, macro_chunks: Optional[Sequence[Any]]) -> PhysicalTimeline:
    timeline = PhysicalTimeline(duration)
    chunks = sorted(list(macro_chunks or []), key=lambda item: (float(item.start), float(item.end), int(getattr(item, "index", 0))))
    if not chunks:
        timeline.add_clip(0.0, duration, clip_id="clip-000001")
        return timeline

    previous_end = 0.0
    for index, chunk in enumerate(chunks, start=1):
        original_start = float(chunk.start)
        original_end = float(chunk.end)
        start = max(0.0, original_start, previous_end)
        end = min(duration, original_end)
        if end <= start:
            continue
        timeline.add_clip(
            start,
            end,
            clip_id=f"clip-{index:06d}",
            source="macro_chunk",
            metadata={
                "original_start": original_start,
                "original_end": original_end,
                "overlap_assigned_to_previous": original_start < previous_end,
            },
        )
        previous_end = end
    if not timeline.physical_clips:
        timeline.add_clip(0.0, duration, clip_id="clip-000001")
    return timeline


def _empty_transcript(duration: float, status: str = "unknown") -> GlobalTranscript:
    return GlobalTranscript(
        audio_duration=duration,
        words=[],
        segments=[],
        backend="shadow",
        status=status,
    )


def build_shadow_artifacts(
    contexts: Sequence[Any],
    duration: float,
    *,
    macro_chunks: Optional[Sequence[Any]] = None,
    context_offsets: Optional[Sequence[float]] = None,
    diarization_result: Any = None,
    left_context: float = 0.5,
    right_context: float = 0.5,
) -> ShadowBuildResult:
    """Build all phase-two artifacts from already-produced pipeline outputs."""
    timeline = _build_ownership_timeline(duration, macro_chunks)
    diagnostics: Dict[str, Any] = {}
    evidence = EvidenceAdaptResult()
    all_words: List[GlobalWord] = []
    all_segments: List[GlobalTranscriptSegment] = []
    offsets = list(context_offsets or [])
    if len(offsets) < len(contexts):
        offsets.extend([0.0] * (len(contexts) - len(offsets)))

    for index, context in enumerate(contexts or []):
        offset = offsets[index]
        owner_clip_id = None
        if macro_chunks and index < len(timeline.physical_clips):
            owner_clip_id = timeline.physical_clips[index].id
        try:
            context._physical_timeline = timeline
            adapted = build_timeline_from_context(
                context,
                duration,
                time_offset=offset,
                physical_clip_id=owner_clip_id,
            )
            evidence.evidence_spans.extend(adapted.evidence_spans)
            evidence.skipped_count += adapted.skipped_count
            for key, value in adapted.source_counts.items():
                evidence.source_counts[key] = evidence.source_counts.get(key, 0) + value
            for key, value in adapted.diagnostics.items():
                if isinstance(value, list):
                    diagnostics.setdefault(key, []).extend(value)
                else:
                    diagnostics[key] = value
        except Exception as exc:
            diagnostics.setdefault("context_errors", []).append(f"context[{index}]: {exc}")

        raw_asr_segments = getattr(context, "asr_segments", []) or []
        # The legacy mapper stores one list of TranscriptionSegment per VAD
        # segment; accept a flat sequence as well for standalone adapters.
        asr_segments: List[TranscriptionSegment] = []
        for item in raw_asr_segments:
            if isinstance(item, TranscriptionSegment):
                asr_segments.append(item)
            elif isinstance(item, (list, tuple)):
                asr_segments.extend(
                    candidate for candidate in item
                    if isinstance(candidate, TranscriptionSegment)
                )
        if asr_segments:
            try:
                transcript = adapt_transcription_segments(
                    asr_segments,
                    source_window_id=f"shadow-window-{index:04d}",
                    segment_id_prefix=f"shadow-segment-{index:04d}",
                    time_offset=offset,
                    audio_duration=duration,
                )
                all_words.extend(transcript.words)
                all_segments.extend(transcript.segments)
                if transcript.diagnostics.get("skipped_segments"):
                    diagnostics.setdefault("transcript_diagnostics", []).append(transcript.diagnostics)
            except Exception as exc:
                diagnostics.setdefault("transcript_errors", []).append(f"context[{index}]: {exc}")

    timeline.speech_evidence_spans.sort(key=lambda item: (item.start, item.end, item.id))
    evidence.evidence_spans.sort(key=lambda item: (item.start, item.end, item.id))
    timeline.diagnostics.update({
        "evidence_source_counts": dict(evidence.source_counts),
        "evidence_skipped_count": evidence.skipped_count,
    })
    for window in build_context_windows(timeline, left_context, right_context):
        timeline.add_context_window(
            window.physical_clip_id,
            window.left_context,
            window.right_context,
            window_id=window.id,
            metadata=window.metadata,
        )

    if all_words or all_segments:
        transcript = GlobalTranscript(
            audio_duration=duration,
            words=all_words,
            segments=all_segments,
            backend="legacy-segment-adapter",
            status="degraded" if diagnostics.get("transcript_errors") else "ok",
            diagnostics=diagnostics.get("transcript_diagnostics", {}),
        )
    else:
        transcript = _empty_transcript(duration)

    if diarization_result is not None:
        try:
            speaker_timeline = adapt_diarization_result(diarization_result, duration=duration)
        except Exception as exc:
            diagnostics.setdefault("speaker_errors", []).append(str(exc))
            speaker_timeline = GlobalSpeakerTimeline(duration=duration, turns=[], exclusive_turns=[], status="degraded")
    else:
        speaker_timeline = GlobalSpeakerTimeline(duration=duration, turns=[], exclusive_turns=[], status="unknown")

    degraded_keys = {"context_errors", "transcript_errors", "speaker_errors", "skipped", "skipped_by_source"}
    if evidence.skipped_count or any(key in diagnostics for key in degraded_keys):
        status = "degraded"
    else:
        status = "ok"
    statistics = {
        "physical_clip_count": len(timeline.physical_clips),
        "evidence_count": len(timeline.speech_evidence_spans),
        "context_window_count": len(timeline.context_windows),
        "global_word_count": len(transcript.words),
        "global_segment_count": len(transcript.segments),
        "speaker_turn_count": len(speaker_timeline.turns),
        "speaker_count": len(speaker_timeline.speaker_ids),
        "evidence_source_counts": dict(evidence.source_counts),
        "skipped_count": evidence.skipped_count,
    }
    return ShadowBuildResult(
        physical_timeline=timeline,
        global_transcript=transcript,
        global_speaker_timeline=speaker_timeline,
        status=status,
        diagnostics=diagnostics,
        statistics=statistics,
    )
