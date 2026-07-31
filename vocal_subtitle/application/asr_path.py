"""ASR route and global evidence stages for the application pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..asr.base import ASREngine
from ..asr.contracts import ASRRuntimePorts, GlobalASRRequest
from ..asr.global_path import GlobalASRService
from ..asr.review_path import ASRFailureRequest, ASRReviewRequest, ASRReviewService
from ..asr.router import ASRRouter
from ..pipeline_context import NoiseProfile, PipelineContext
from ..utils.audio_utils import AudioUtils
from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


class PipelineASRPathMixin:
    @staticmethod
    def _classify_global_failure(exc: Exception) -> str:
        """Compatibility hook for global ASR failure classification."""
        return ASRReviewService.classify_failure(ASRFailureRequest(exc))

    def _get_global_asr_engine(self):
        """Return the global ASR engine.

        Tests monkeypatch this method; the default returns the standard engine.
        """
        return self._get_asr_engine()

    def _prepare_asr_route(self, audio, sample_rate: int, speech_intervals=None):
        """Resolve and cache the task-level ASR route exactly once."""
        if self._asr_route_decision is not None:
            return self._asr_route_decision

        def factory(engine_name, model=None, probe=False):
            return self._get_asr_engine_for(
                engine_name, model=model, cache=not probe
            )

        decision = ASRRouter(self.config, factory).decide(
            audio, sample_rate, speech_intervals
        )
        self._asr_route_decision = decision
        self._resolved_language = decision.language
        self._asr_engine = self._get_asr_engine_for(decision.selected_engine)
        logger.info(
            "ASR route: requested=%s selected=%s language=%s probability=%.3f reason=%s",
            decision.requested_engine,
            decision.selected_engine,
            decision.detected_language,
            decision.language_probability,
            decision.decision_reason,
        )
        if self._progress is not None:
            try:
                self._progress.update_stage(
                    0,
                    extra={
                        "detail": (
                            f"语言检测完成：{decision.detected_language}，"
                            f"路由到 {decision.selected_engine}"
                        )
                    },
                )
            except Exception:
                # Global ASR may resolve before the segmented ASR stage exists.
                pass
        return decision

    def _run_global_transcription_path(
        self,
        audio,
        sample_rate,
        shadow,
        stats,
        *,
        vad_segments=None,
        ffmpeg_result=None,
        noise_profile=None,
    ):
        """Compatibility adapter for the explicit global ASR service."""
        request = GlobalASRRequest(
            audio=audio,
            sample_rate=sample_rate,
            shadow=shadow,
            stats=stats,
            vad_segments=vad_segments,
            ffmpeg_result=ffmpeg_result,
            noise_profile=noise_profile,
        )
        ports = ASRRuntimePorts(
            config=self.config,
            get_engine=self._get_global_asr_engine,
            get_engine_for=self._get_asr_engine_for,
            get_language=self._resolved_language_or_config,
            set_language=lambda value: setattr(self, "_resolved_language", value),
            quality_gate_kwargs=self._quality_gate_kwargs,
            progress=self._progress,
            global_runner=lambda item: self._run_global_transcription_path_legacy(
                audio=item.audio,
                sample_rate=item.sample_rate,
                shadow=item.shadow,
                stats=item.stats,
                vad_segments=item.vad_segments,
                ffmpeg_result=item.ffmpeg_result,
                noise_profile=item.noise_profile,
            ),
        )
        result = GlobalASRService().run(request, ports)
        return result.events, result.diagnostics, result.transcript

    def _run_global_transcription_path_legacy(
        self,
        audio,
        sample_rate,
        shadow,
        stats,
        *,
        vad_segments=None,
        ffmpeg_result=None,
        noise_profile=None,
    ):
        """Execute the global transcription path.

        Transcribes the full audio, allocates words to physical bins,
        builds SubtitleEvents, audits coverage, and attempts tail recovery.
        The tests in test_coverage_recovery and test_phase_three verify
        this integration path.
        """
        from ..physical.ir import GlobalTranscript, GlobalTranscriptSegment, GlobalWord
        from ..physical.subtitle_bins import build_physical_subtitle_bins
        from ..physical.allocator import (
            AllocationResult,
            PhysicalSpan,
            WordAllocation,
            allocate_words,
        )
        from ..physical.coverage import audit_physical_coverage
        from ..physical.events import build_events
        from ..physical.word_alignment import align_words_to_physical
        from ..mapping.time_mapper import SubtitleEvent as SE

        engine = self._get_global_asr_engine()
        engine.load_model()

        language = self._resolved_language_or_config()
        if language is None:
            detector = getattr(engine, "detect_language", None)
            if callable(detector):
                try:
                    language = detector(audio, sample_rate)
                except Exception as exc:
                    logger.warning("Global language detection failed: %s", exc)
            if language:
                self._resolved_language = language

        # Transcribe the full audio
        segments = engine.transcribe(audio, sample_rate, language=language)
        if not isinstance(segments, list):
            segments = [segments]

        # Build GlobalTranscript
        words = []
        transcript_segments = []
        word_idx = 0
        for seg in segments:
            seg_words = getattr(seg, "words", []) or []
            seg_word_ids = []
            for w in seg_words:
                word_id = f"word-{word_idx:06d}"
                word = GlobalWord(
                    id=word_id, text=str(getattr(w, "word", "")),
                    raw_start=float(getattr(w, "start", seg.start)),
                    raw_end=float(getattr(w, "end", seg.end)),
                    confidence=float(getattr(w, "confidence", 0.9)),
                    source_window_id="global",
                    segment_id=f"seg-{len(transcript_segments):03d}",
                )
                words.append(word)
                seg_word_ids.append(word_id)
                word_idx += 1
            if not seg_words:
                word_id = f"word-{word_idx:06d}"
                word = GlobalWord(
                    id=word_id, text=str(getattr(seg, "text", "")).strip(),
                    raw_start=float(getattr(seg, "start", 0.0)),
                    raw_end=float(getattr(seg, "end", 1.0)),
                    confidence=0.9, source_window_id="global",
                    segment_id=f"seg-{len(transcript_segments):03d}",
                )
                words.append(word)
                seg_word_ids.append(word_id)
                word_idx += 1
            transcript_segments.append(GlobalTranscriptSegment(
                id=f"seg-{len(transcript_segments):03d}",
                text=str(getattr(seg, "text", "")).strip(),
                raw_start=float(getattr(seg, "start", 0.0)),
                raw_end=float(getattr(seg, "end", 1.0)),
                word_ids=seg_word_ids,
            ))

        global_transcript = GlobalTranscript(
            audio_duration=float(stats.duration_seconds),
            words=words, segments=transcript_segments,
            backend=engine.name,
            status="ok" if words else "degraded",
        )

        timeline = getattr(shadow, "physical_timeline", None)
        if timeline is None:
            return [], {"recovery": {"status": "no_timeline"}, "physical_coverage": {"complete": False}}, global_transcript

        tail_repair = self._repair_tail_evidence(timeline, stats.duration_seconds)

        bins = build_physical_subtitle_bins(
            timeline, audio=audio, sample_rate=sample_rate,
        )
        speaker_timeline = getattr(shadow, "global_speaker_timeline", None)
        allocation_result = allocate_words(global_transcript, timeline, speaker_timeline=speaker_timeline, subtitle_bins=bins)

        bin_owner_map = {
            item.id: item.physical_clip_id
            for item in bins
            if item.physical_clip_id
        }
        clip_bounds = {
            item.id: (item.start, item.end)
            for item in timeline.physical_clips
        }
        # Global callers may provide detector outputs directly. When they do
        # not, reconstruct the same absolute-time evidence from the shadow
        # timeline so BoundaryArbiter still receives all available sources.
        if vad_segments is None:
            from ..vad.base import SpeechSegment

            vad_sources = {"silero", "ten", "webrtc", "boundary_fusion"}
            vad_segments = [
                SpeechSegment(item.start, item.end, item.confidence or 0.5)
                for item in timeline.speech_evidence_spans
                if item.source in vad_sources
            ]
        if ffmpeg_result is None:
            ffmpeg_result = {
                "skeleton": [
                    (item.start, item.end)
                    for item in timeline.speech_evidence_spans
                    if item.source == "ffmpeg_skeleton"
                ],
                "coarse_speech": [
                    (item.start, item.end)
                    for item in timeline.speech_evidence_spans
                    if item.source == "ffmpeg_coarse"
                ],
                "raw_silence_intervals": [],
            }
        if noise_profile is None:
            from ..physical.noise_profile import estimate_noise_profile

            noise_profile = estimate_noise_profile(audio, sample_rate)

        aligned_allocations = align_words_to_physical(
            allocation_result.allocations,
            bins,
            bin_owner_map=bin_owner_map,
            timeline=timeline,
            clip_bounds=clip_bounds,
            audio=audio,
            sample_rate=sample_rate,
            vad_segments=vad_segments,
            ffmpeg_result=(
                getattr(shadow, "ffmpeg_unified_result", None)
                or ffmpeg_result
            ),
            noise_profile=noise_profile,
        )
        aligned_result = AllocationResult(
            allocations=aligned_allocations,
            rejected=list(allocation_result.rejected),
            diagnostics=dict(allocation_result.diagnostics),
        )
        events = [
            item.to_subtitle_event()
            for item in build_events(aligned_result, subtitle_bins=bins)
        ]

        coverage = audit_physical_coverage(bins, allocation_result.allocations)
        from ..asr.quality_gate import evaluate_asr_quality

        quality = evaluate_asr_quality(
            events,
            [(item.start, item.end) for item in bins],
            stats.duration_seconds,
            **self._quality_gate_kwargs(),
        )
        diag = {
            "physical_coverage": coverage.to_dict(),
            "quality_gate": quality.to_dict(),
            "tail_evidence_repair": tail_repair,
            "recovery": {"status": "recovered" if coverage.complete else "incomplete"},
        }

        # Tail recovery
        if not coverage.complete and coverage.recovery_ranges:
            try:
                recovered_allocations = []
                for rec_range in coverage.recovery_ranges:
                    ss = int(rec_range.start * sample_rate)
                    es = int(min(rec_range.end, stats.duration_seconds) * sample_rate)
                    if es <= ss:
                        continue
                    seg_audio = audio[ss:es]
                    recovery_segs = engine.transcribe(
                        seg_audio, sample_rate, language=language
                    )
                    for rseg in recovery_segs:
                        rwlist = getattr(rseg, "words", []) or []
                        for rw in rwlist:
                            wid = f"word-rec-{word_idx:06d}"
                            # Clamp recovery word times to the audio duration
                            r_start = min(rec_range.start + float(getattr(rw, "start", 0.0)), stats.duration_seconds)
                            r_end = min(rec_range.start + float(getattr(rw, "end", 0.1)), stats.duration_seconds)
                            rword = GlobalWord(id=wid, text=str(getattr(rw, "word", "")),
                                               raw_start=r_start,
                                               raw_end=max(r_start + 0.01, r_end),
                                               confidence=float(getattr(rw, "confidence", 0.9)),
                                               source_window_id="recovery", segment_id="recovery")
                            words.append(rword)
                            word_idx += 1
                            clip_id = rec_range.physical_clip_id or "clip-000001"
                            rec_span = PhysicalSpan(
                                clip_id=clip_id,
                                start=r_start,
                                end=max(r_start + 0.01, r_end),
                            )
                            events.append(SE(
                                len(events) + 1,
                                rword.raw_start,
                                rword.raw_end,
                                rword.text,
                                physical_start=rword.raw_start,
                                physical_end=rword.raw_end,
                                physical_spans=[rec_span.to_dict()],
                                source_word_ids=[rword.id],
                                speaker_source="recovery",
                                alignment_warning="timing_degraded:local_recovery",
                                time_source="timing_degraded",
                                revision_trace=[{
                                    "stage": "local_recovery",
                                    "status": "timing_degraded",
                                    "reason": "recovered_word_not_boundary_aligned",
                                }],
                            ))
                            recovered_allocations.append(WordAllocation(
                                word=rword,
                                physical_spans=(rec_span,),
                                warnings=("timing_degraded",),
                                alignment_status="degraded",
                                accepted=True,
                            ))

                coverage2 = audit_physical_coverage(bins, list(allocation_result.allocations) + recovered_allocations)
                diag["physical_coverage"] = coverage2.to_dict()
                if coverage2.complete:
                    diag["recovery"]["status"] = "recovered"
                    global_transcript = GlobalTranscript(audio_duration=stats.duration_seconds, words=words, segments=transcript_segments, backend=engine.name, status="ok")
                else:
                    diag["recovery"]["status"] = "incomplete"
                    global_transcript = GlobalTranscript(audio_duration=stats.duration_seconds, words=words, segments=transcript_segments, backend=engine.name, status="degraded")
            except Exception as exc:
                diag["recovery"] = {"status": "failed", "error": str(exc)}
                global_transcript = GlobalTranscript(audio_duration=stats.duration_seconds, words=words, segments=transcript_segments, backend=engine.name, status="degraded")
        elif not coverage.complete:
            global_transcript = GlobalTranscript(audio_duration=stats.duration_seconds, words=words, segments=transcript_segments, backend=engine.name, status="degraded")

        quality = evaluate_asr_quality(
            events,
            [(item.start, item.end) for item in bins],
            stats.duration_seconds,
            **self._quality_gate_kwargs(),
        )
        diag["quality_gate"] = quality.to_dict()
        return events, diag, global_transcript

    def _quality_gate_kwargs(self) -> dict:
        policy = self.config.asr.auto_routing
        return {
            "min_coverage_ratio": policy.min_coverage_ratio,
            "min_text_density": policy.min_text_density,
            "max_event_duration": policy.max_event_duration,
            "long_audio_seconds": policy.long_audio_seconds,
            "long_audio_min_text_chars": policy.long_audio_min_text_chars,
            "max_overlap_ratio": policy.max_overlap_ratio,
        }

    @staticmethod
    def _repair_tail_evidence(timeline, duration_seconds: float) -> dict:
        """Preserve a longer corroborating tail when ffmpeg ends early.

        The preferred ffmpeg skeleton remains the source for normal bins. A
        longer Silero/coarse span is copied only across the disputed tail,
        which keeps precise boundaries while preventing silent truncation.
        """
        evidence = list(getattr(timeline, "speech_evidence_spans", []) or [])
        preferred = [item for item in evidence if item.source == "ffmpeg_skeleton"]
        alternatives = [
            item for item in evidence
            if item.source != "ffmpeg_skeleton"
            and item.source in {"silero", "ten", "webrtc", "boundary_fusion", "ffmpeg_coarse"}
        ]
        preferred_end = max((float(item.end) for item in preferred), default=0.0)
        alternative = max(alternatives, key=lambda item: float(item.end), default=None)
        alternative_end = float(getattr(alternative, "end", 0.0) or 0.0)
        gap = alternative_end - preferred_end
        if alternative is None or gap <= 0.15:
            return {
                "status": "not_needed",
                "preferred_end": preferred_end,
                "alternative_end": alternative_end,
                "gap_seconds": max(0.0, gap),
            }
        end = min(float(duration_seconds), alternative_end)
        start = max(0.0, min(float(alternative.start), preferred_end - 0.25))
        try:
            timeline.add_evidence(
                start=start,
                end=end,
                source="ffmpeg_skeleton",
                confidence=getattr(alternative, "confidence", None),
                physical_clip_id=getattr(alternative, "physical_clip_id", None),
                metadata={
                    "tail_recheck": True,
                    "source_evidence": getattr(alternative, "source", "unknown"),
                    "original_preferred_end": preferred_end,
                },
            )
        except Exception as exc:
            return {
                "status": "failed",
                "preferred_end": preferred_end,
                "alternative_end": alternative_end,
                "gap_seconds": gap,
                "error": str(exc),
            }
        return {
            "status": "extended",
            "preferred_end": preferred_end,
            "alternative_end": alternative_end,
            "gap_seconds": gap,
            "source": getattr(alternative, "source", "unknown"),
            "recheck_start": start,
            "recheck_end": end,
        }

    @staticmethod
    def _validate_global_result(events, transcript, diagnostics):
        return ASRReviewService.validate(
            ASRReviewRequest(
                events=events,
                transcript=transcript,
                diagnostics=diagnostics,
            )
        )

    @staticmethod
    def _classify_global_diagnostics(diagnostics: dict) -> str:
        """Extract the worst failure category from global diagnostics."""
        failed_windows = diagnostics.get("failed_windows", [])
        for window in failed_windows:
            error = window.get("error", "")
            if "is not installed" in error:
                return "dependency_unavailable"
        return "execution_failed"

    @staticmethod
    def _global_diagnostic_errors(diagnostics: dict) -> list:
        """Extract error messages from global diagnostics."""
        return [w.get("error", "") for w in diagnostics.get("failed_windows", [])]

    @staticmethod
    def _safe_failure_reason(exc: Exception) -> str:
        """Redact credentials from failure messages."""
        return ASRReviewService.safe_failure_reason(exc)

    def _resolve_asr_path(self) -> str:
        """Determine the requested offline ASR route.

        ``auto`` is kept as a distinct route so the caller can record whether
        global ASR actually succeeded or whether it fell back to segmented.
        """
        if self.config.mode == "streaming":
            return "segmented"
        # Respect explicit override from merge_with_overrides or config
        explicit = getattr(self.config, "asr_path", None)
        if explicit:
            return str(explicit)
        if self._requested_asr_path:
            return self._requested_asr_path
        if self.config.asr.global_asr.enabled:
            routing = self.config.asr.global_asr.routing
            if routing in ("global", "auto"):
                return routing
        return "segmented"

    @staticmethod
    def _is_usable_global_transcript(transcript) -> bool:
        """Check whether a global transcript is usable for subtitle production."""
        return ASRReviewService.is_usable_transcript(transcript)

    def _is_usable_full_pipeline_cache(self, cache_entry: dict) -> bool:
        """Check whether a cached full-pipeline result matches the current ASR path."""
        cached_stats = cache_entry.get("stats", {})
        cached_path = cached_stats.get("asr_path", "")
        requested_path = self._resolve_asr_path()
        if self.config.asr.engine == "auto" and requested_path in ("global", "auto"):
            policy = self.config.asr.auto_routing
            if (
                cached_stats.get("asr_route_version") != policy.route_version
                or cached_stats.get("quality_gate_version") != policy.quality_gate_version
                or not cached_stats.get("requested_engine")
                or not cached_stats.get("selected_engine")
            ):
                return False
        if requested_path in ("global", "auto"):
            return cached_path == "global"
        # Old cache entries did not carry asr_path; retain compatibility for
        # explicitly requested segmented/legacy runs.
        return cached_path in ("", "legacy", "legacy_degraded")
