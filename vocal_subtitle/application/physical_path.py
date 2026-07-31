"""Physical shadow and speaker-boundary adapters for the application pipeline."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..pipeline_context import NoiseProfile, PipelineContext
from ..utils.audio_utils import AudioUtils
from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


class PipelinePhysicalPathMixin:
    def _run_early_detection(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vocals_path: Path,
    ) -> Tuple[PipelineContext, List[SpeechSegment], Optional[Dict], NoiseProfile]:
        """Run one full-audio detector pass for the physical shadow.

        This deliberately reuses the existing detector implementations and
        does not change their thresholds or default fusion/denoise policy.
        """
        chunk_duration = len(audio) / max(sample_rate, 1)
        ctx = PipelineContext(
            audio_path=vocals_path,
            audio=audio,
            sample_rate=sample_rate,
        )
        noise = AudioUtils.estimate_noise_floor_per_chunk(
            audio, sample_rate, chunk_duration=chunk_duration,
        )
        noise_profile = NoiseProfile(
            noise_rms=noise["noise_rms"],
            speech_threshold=noise["speech_threshold"],
            is_noisy_environment=noise["is_noisy_environment"],
        )
        ctx.noise_profile = noise_profile

        silero_segments = self._run_vad(audio, sample_rate)
        ctx.silero_segments = list(silero_segments)
        ffmpeg_result = None
        if self.config.vad.ffmpeg_enabled:
            ffmpeg_result = self._run_ffmpeg_vad(vocals_path, ctx, "[global] ")
        ctx.ffmpeg_unified_result = ffmpeg_result
        if ffmpeg_result:
            ctx.ffmpeg_segments = list(ffmpeg_result.get("coarse_speech", []) or [])

        selected_segments = list(silero_segments)
        if ffmpeg_result and self.config.fusion.enabled:
            from ..vad.boundary_fusion import BoundaryFusion

            fused_segments = BoundaryFusion(self.config.fusion).fuse(
                silero_segments,
                ffmpeg_result.get("coarse_speech", []) or [],
                audio,
                sample_rate,
            )
            ctx.fused_segments = list(fused_segments)
            selected_segments = list(fused_segments)

        ctx.add_diagnostic(
            "Global shadow detection: silero=%d, ffmpeg=%d, fused=%d"
            % (
                len(ctx.silero_segments),
                len(ctx.ffmpeg_segments),
                len(ctx.fused_segments),
            )
        )
        return ctx, selected_segments, ffmpeg_result, noise_profile

    def _build_physical_shadow(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vocals_path: Path,
        context: PipelineContext,
        vad_segments: List[SpeechSegment],
        ffmpeg_result: Optional[Dict],
        noise_profile: NoiseProfile,
    ):
        """Adapt detector output into one validated physical shadow."""
        from ..physical.shadow import build_shadow_artifacts

        duration = len(audio) / max(sample_rate, 1)
        shadow = build_shadow_artifacts([context], duration)
        timeline_errors = shadow.physical_timeline.validate()
        if timeline_errors:
            raise ValueError("invalid global physical timeline: " + "; ".join(timeline_errors))

        # Keep detector artifacts available to the alignment and acoustic
        # validation stages without changing the ShadowBuildResult schema.
        shadow.ffmpeg_unified_result = ffmpeg_result
        shadow.vad_segments = vad_segments
        shadow.noise_profile = noise_profile
        shadow.diagnostics.setdefault("early_detection", list(context.diagnostics))
        shadow.diagnostics["statistics"] = dict(shadow.statistics)
        return shadow

    @staticmethod
    def _clamp_to_physical_envelopes(events: list) -> list:
        """Clamp display time of each event to its physical envelope."""
        for event in events:
            physical_start = getattr(event, "physical_start", None)
            physical_end = getattr(event, "physical_end", None)
            if physical_start is not None and physical_end is not None:
                event.start = max(event.start, physical_start)
                event.end = min(event.end, physical_end)
        return events

    @staticmethod
    def _is_usable_diarization_cache(diarization, requested_backend: str) -> bool:
        """Check whether a cached diarization result matches the requested backend.

        A legacy/fallback result must not be reused when pyannote is available,
        and vice-versa.
        """
        backend = getattr(diarization, "backend", "")
        if requested_backend == "auto":
            return backend != "legacy-global-fallback"
        if requested_backend == "pyannote":
            return backend in ("pyannote", "pyannote-community-1", "fused")
        if requested_backend == "legacy":
            return True
        return backend == requested_backend

    def _project_global_speakers(
        self,
        segments: list,
        time_offset: float = 0.0,
        duration: float = 0.0,
    ):
        """Project global diarization turns onto VAD speech segments.

        Returns (projected_segments, speaker_ids) where each input segment is
        split at speaker turn boundaries.
        """
        turns = getattr(self, "_global_turns", []) or []
        projected_segments = []
        speaker_ids = []
        for seg in segments:
            seg_start = seg.start + time_offset
            seg_end = seg.end + time_offset
            # Clip to audio duration
            seg_start = max(0.0, seg_start)
            seg_end = min(duration if duration > 0 else float("inf"), seg_end)
            relevant = [
                t for t in turns
                if t.end > seg_start and t.start < seg_end
            ]
            if not relevant:
                # No diarization data — return segment as-is with unknown speaker
                projected_segments.append(type(seg)(seg_start, seg_end))
                speaker_ids.append(-1)
                continue
            # When only one speaker covers the entire segment, don't split
            speakers_in_seg = {t.speaker_id for t in relevant}
            if len(speakers_in_seg) == 1:
                projected_segments.append(type(seg)(seg_start, seg_end))
                speaker_ids.append(speakers_in_seg.pop())
                continue
            # Split at turn boundaries
            boundaries = sorted(set(
                [max(seg_start, t.start) for t in relevant]
                + [min(seg_end, t.end) for t in relevant]
            ))
            for b_start, b_end in zip(boundaries, boundaries[1:]):
                if b_end <= b_start:
                    continue
                # Find the speaker at the midpoint of this sub-segment
                midpoint = (b_start + b_end) / 2.0
                speaker = next(
                    (t.speaker_id for t in turns if t.start <= midpoint < t.end),
                    -1,
                )
                projected_segments.append(type(seg)(b_start, b_end))
                speaker_ids.append(speaker)
        return projected_segments, speaker_ids

    def _enforce_speaker_boundaries(
        self,
        events: list,
        stats,
    ):
        """Split subtitle events at speaker-turn boundaries.

        When an event spans a speaker change, it is split so each piece
        carries the correct speaker label and only the words that belong
        to that speaker.
        """
        from ..asr.base import WordTimestamp
        turns = getattr(self, "_global_turns", []) or []
        if not turns or not events:
            return events

        result = []
        for event in events:
            words = list(getattr(event, "words", []) or [])
            if not words:
                # No word timestamps — can't split by speaker. When the event
                # crosses a speaker boundary, mark it as UNKNOWN.
                crosses_boundary = any(
                    t.start > event.start and t.start < event.end
                    for t in turns
                )
                if crosses_boundary:
                    first_turn = next(
                        (t for t in turns if t.start <= event.start < t.end),
                        None,
                    )
                    event.end = min(event.end, first_turn.end if first_turn else turns[0].start)
                    event.speaker_id = None
                    event.speaker_label = None
                else:
                    for turn in turns:
                        if turn.start <= event.start < turn.end:
                            event.speaker_id = turn.speaker_id
                            break
                    if event.speaker_id is not None:
                        event.speaker_label = self._make_speaker_label(
                            self._resolved_language_or_config(), event.speaker_id
                        )
                stats.mixed_event_count += 1
                result.append(event)
                continue

            # Split at speaker boundary
            split_points = sorted(set(
                [event.start]
                + [t.start for t in turns if event.start < t.start < event.end]
                + [t.end for t in turns if event.start < t.end < event.end]
                + [event.end]
            ))
            piece_index = 0
            for b_start, b_end in zip(split_points, split_points[1:]):
                if b_end <= b_start:
                    continue
                midpoint = (b_start + b_end) / 2.0
                turn_speaker = next(
                    (t.speaker_id for t in turns if t.start <= midpoint < t.end),
                    None,
                )
                # Find words whose midpoint falls in this sub-segment
                piece_words = [
                    w for w in words
                    if (event.start + float(getattr(w, "start", 0.0)) + event.start + float(getattr(w, "end", 0.0))) / 2.0 < b_end
                    and (event.start + float(getattr(w, "start", 0.0)) + event.start + float(getattr(w, "end", 0.0))) / 2.0 > b_start
                ]
                if not piece_words:
                    continue
                piece_text = " ".join(str(getattr(w, "word", "")) for w in piece_words)
                # Build a piece event
                import copy
                piece = copy.copy(event)
                piece.index = piece_index
                piece.start = b_start
                piece.end = b_end
                piece.text = piece_text or event.text
                piece.speaker_id = turn_speaker
                piece.speaker_label = self._make_speaker_label(
                    self._resolved_language_or_config(), turn_speaker
                )
                # Adjust word timestamps relative to the new piece start.
                # For the first piece, b_start == event.start so offsets are zero.
                offset = b_start - event.start
                piece.words = []
                for w in piece_words:
                    copied = copy.copy(w)
                    copied.start = max(0.0, float(getattr(w, "start", 0.0)) - offset)
                    copied.end = float(getattr(w, "end", 0.0)) - offset
                    piece.words.append(copied)
                # Filter source_word_ids to words in this piece
                all_words = list(getattr(event, "words", []) or [])
                all_source_ids = list(getattr(event, "source_word_ids", []) or [])
                piece.source_word_ids = []
                for w in piece_words:
                    try:
                        idx = all_words.index(w)
                        if idx < len(all_source_ids):
                            piece.source_word_ids.append(all_source_ids[idx])
                    except ValueError:
                        pass
                if not piece.source_word_ids:
                    piece.source_word_ids = all_source_ids
                piece_index += 1
                result.append(piece)
            if piece_index > 1:
                stats.mixed_event_count += 1

        return result

