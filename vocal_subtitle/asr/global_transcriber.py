"""Global ASR orchestration and deterministic overlap deduplication."""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..physical.context import build_context_windows
from ..physical.ir import GlobalTranscript, GlobalTranscriptSegment, GlobalWord
from ..physical.timeline import ContextWindow, PhysicalTimeline
from .base import ASREngine
from .whisperx_engine import normalize_whisperx_transcript


@dataclass(frozen=True)
class GlobalTranscriberConfig:
    overlap_dedup: bool = True
    dedup_overlap_ratio: float = 0.5
    dedup_time_tolerance: float = 0.08
    alignment: bool = True
    left_context: float = 0.5
    right_context: float = 0.5


@dataclass
class GlobalTranscriptionResult:
    transcript: GlobalTranscript
    diagnostics: dict[str, Any] = field(default_factory=dict)


class GlobalTranscriber:
    """Run any ASR engine over owned context windows and return global IR."""

    def __init__(
        self, engine: ASREngine, config: GlobalTranscriberConfig | None = None
    ) -> None:
        if not isinstance(engine, ASREngine):
            raise ValueError("engine must implement ASREngine")
        self.engine = engine
        self.config = config or GlobalTranscriberConfig()

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        windows: Sequence[ContextWindow] | None = None,
        physical_timeline: PhysicalTimeline | None = None,
        language: str | None = None,
    ) -> GlobalTranscriptionResult:
        duration = len(audio) / max(sample_rate, 1)
        selected_windows = list(windows or [])
        if not selected_windows and physical_timeline is not None:
            selected_windows = build_context_windows(
                physical_timeline,
                self.config.left_context,
                self.config.right_context,
            )
        if not selected_windows:
            selected_windows = [
                ContextWindow(
                    id="window:full",
                    start=0.0,
                    end=duration,
                    physical_clip_id="clip:full",
                    left_context=0.0,
                    right_context=0.0,
                )
            ]

        candidates: list[GlobalTranscript] = []
        diagnostics: dict[str, Any] = {
            "window_count": len(selected_windows),
            "failed_windows": [],
            "alignment_status": "disabled"
            if not self.config.alignment
            else "unavailable",
            "alignment_failures": [],
            "raw_word_count": 0,
            "deduplicated_word_count": 0,
        }
        for window in selected_windows:
            start_sample = max(
                0, min(len(audio), int(round(window.start * sample_rate)))
            )
            end_sample = max(
                start_sample, min(len(audio), int(round(window.end * sample_rate)))
            )
            try:
                raw_segments = self.engine.transcribe(
                    audio[start_sample:end_sample],
                    sample_rate=sample_rate,
                    language=language,
                )
                if isinstance(raw_segments, GlobalTranscript):
                    transcript = raw_segments
                else:
                    if self.config.alignment and hasattr(self.engine, "align"):
                        try:
                            raw_segments = self.engine.align(
                                audio[start_sample:end_sample],
                                sample_rate=sample_rate,
                                segments=raw_segments,
                                language=language,
                            )
                            diagnostics["alignment_status"] = "applied"
                        except Exception as exc:
                            diagnostics["alignment_failures"].append(
                                {"window_id": window.id, "error": str(exc)}
                            )
                            # Alignment is an accuracy enhancement. Keep the
                            # valid ASR timestamps when the optional model fails.
                    transcript = normalize_whisperx_transcript(
                        raw_segments,
                        source_window_id=window.id,
                        segment_id_prefix=f"seg:{window.id}",
                        time_offset=window.start,
                        language=language,
                        audio_duration=duration,
                    )
                candidates.append(transcript)
                diagnostics["raw_word_count"] += len(transcript.words)
            except Exception as exc:
                diagnostics["failed_windows"].append(
                    {"window_id": window.id, "error": str(exc)}
                )

        transcript = self._combine(candidates, duration, diagnostics)
        return GlobalTranscriptionResult(transcript=transcript, diagnostics=diagnostics)

    def combine_transcripts(
        self,
        transcripts: Sequence[GlobalTranscript],
        duration: float,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> GlobalTranscript:
        """Merge existing global transcripts using the normal dedup policy."""
        merge_diagnostics = dict(diagnostics or {})
        merge_diagnostics.setdefault("window_count", len(transcripts))
        merge_diagnostics.setdefault("failed_windows", [])
        merge_diagnostics.setdefault("alignment_status", "not_applicable")
        merge_diagnostics.setdefault("alignment_failures", [])
        merge_diagnostics.setdefault(
            "raw_word_count", sum(len(item.words) for item in transcripts)
        )
        return self._combine(transcripts, duration, merge_diagnostics)

    def _combine(
        self,
        transcripts: Sequence[GlobalTranscript],
        duration: float,
        diagnostics: dict[str, Any],
    ) -> GlobalTranscript:
        all_words = [word for transcript in transcripts for word in transcript.words]
        selected: list[GlobalWord] = []
        duplicate_ids: set[str] = set()
        aliases: dict[str, str] = {}
        for word in sorted(
            all_words, key=lambda item: (item.raw_start, item.raw_end, item.id)
        ):
            duplicate = next(
                (
                    existing
                    for existing in selected
                    if self._is_duplicate(existing, word)
                ),
                None,
            )
            if duplicate is None or not self.config.overlap_dedup:
                selected.append(word)
                continue
            duplicate_ids.add(word.id)
            if self._priority(word) > self._priority(duplicate):
                selected.remove(duplicate)
                selected.append(word)
                aliases[duplicate.id] = word.id
            else:
                aliases[word.id] = duplicate.id
        selected.sort(key=lambda item: (item.raw_start, item.raw_end, item.id))
        diagnostics["deduplicated_word_count"] = len(all_words) - len(selected)
        diagnostics["duplicate_word_ids"] = sorted(duplicate_ids)
        selected_ids = {word.id for word in selected}

        segments: list[GlobalTranscriptSegment] = []
        seen_segment_words: set[tuple[str, ...]] = set()
        used_segment_ids: set[str] = set()
        selected_by_id = {word.id: word for word in selected}
        for transcript in transcripts:
            for segment in transcript.segments:
                ids = []
                for word_id in segment.word_ids:
                    canonical_id = aliases.get(word_id, word_id)
                    if canonical_id in selected_ids and canonical_id not in ids:
                        ids.append(canonical_id)
                # Overlapping windows can produce the same segment twice.
                # Keep one segment while retaining all canonical word links.
                ids = [word.id for word in selected if word.id in ids]
                if not ids or tuple(ids) in seen_segment_words:
                    continue
                seen_segment_words.add(tuple(ids))
                words = [selected_by_id[word_id] for word_id in ids]
                if words:
                    start = min(word.raw_start for word in words)
                    end = max(word.raw_end for word in words)
                else:
                    start, end = segment.raw_start, segment.raw_end
                segment_id = segment.id
                if segment_id in used_segment_ids:
                    suffix = 2
                    while f"{segment_id}:dup{suffix}" in used_segment_ids:
                        suffix += 1
                    segment_id = f"{segment_id}:dup{suffix}"
                used_segment_ids.add(segment_id)
                segments.append(
                    GlobalTranscriptSegment(
                        id=segment_id,
                        text=segment.text,
                        raw_start=start,
                        raw_end=end,
                        word_ids=ids,
                        language=segment.language,
                        avg_logprob=segment.avg_logprob,
                        metadata=copy.deepcopy(segment.metadata),
                    )
                )
        return GlobalTranscript(
            audio_duration=duration,
            words=selected,
            segments=segments,
            backend=self.engine.name,
            status="degraded"
            if diagnostics["failed_windows"]
            or any(transcript.status != "ok" for transcript in transcripts)
            else ("empty" if not selected else "ok"),
            diagnostics=copy.deepcopy(diagnostics),
        )

    def _is_duplicate(self, left: GlobalWord, right: GlobalWord) -> bool:
        if _normalize_text(left.text) != _normalize_text(right.text):
            return False
        if (
            left.language
            and right.language
            and left.language.casefold() != right.language.casefold()
        ):
            return False
        overlap = min(left.raw_end, right.raw_end) - max(
            left.raw_start, right.raw_start
        )
        if overlap < -self.config.dedup_time_tolerance:
            return False
        shortest = min(left.raw_end - left.raw_start, right.raw_end - right.raw_start)
        return (
            overlap
            >= shortest * self.config.dedup_overlap_ratio
            - self.config.dedup_time_tolerance
        )

    @staticmethod
    def _priority(word: GlobalWord) -> tuple[float, float, int]:
        confidence = word.confidence if word.confidence is not None else 0.0
        return confidence, word.raw_end - word.raw_start, -len(word.id)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).casefold()
