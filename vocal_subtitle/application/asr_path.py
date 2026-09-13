"""ASR route and global evidence stages for the application pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..asr.base import ASREngine
from ..asr.contracts import ASRRuntimePorts, GlobalASRRequest
from ..asr.evidence import candidate_from_subtitle_event
from ..asr.evidence_review import EvidenceReviewRuntimePorts
from ..asr.optional_adapters import (
    LazyAudioClassifierSED,
    LazyQwenASR,
    LazyQwenForcedAligner,
)
from ..asr.review_engines import WindowedASRContextReASR, WindowedASREngine
from .offline_production import (
    OfflineProductionCoordinator,
    OfflineProductionRequest,
)
from ..asr.global_transcriber import GlobalTranscriber, GlobalTranscriberConfig
from ..asr.global_path import GlobalASRService
from ..asr.engine_pairing import EnginePairRouter
from ..asr.review_path import ASRFailureRequest, ASRReviewRequest, ASRReviewService
from ..asr.router import ASRRouter
from ..pipeline_context import NoiseProfile, PipelineContext
from ..utils.audio_utils import AudioUtils
from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


class PipelineASRPathMixin:
    @staticmethod
    def _optional_asr_confidence(value):
        """Keep absent backend confidence absent in the evidence contract."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _apply_evidence_review(
        self,
        events,
        *,
        audio=None,
        sample_rate: int = 16000,
        physical_timeline=None,
        stats=None,
    ):
        """Run the componentized evidence path over segmented candidates."""
        config = getattr(self.config, "evidence_review", None)
        if physical_timeline is not None:
            self._ensure_global_evidence(
                audio=audio,
                sample_rate=sample_rate,
                physical_timeline=physical_timeline,
                stats=stats,
            )
        review_engine = None
        if config is not None and config.context_reasr_enabled and audio is not None:
            review_engine = WindowedASRContextReASR(self._get_asr_engine)
        qwen_engine = None
        secondary_engine = None
        secondary_name = None
        forced_aligner = None
        sed_engine = None
        qwen_model_path = (
            getattr(config, "qwen_model_path", None)
            if config is not None
            else None
        ) or getattr(getattr(self.config, "asr", None), "qwen_model_path", None)
        if config is not None and getattr(config, "qwen_enabled", False):
            qwen_engine = LazyQwenASR(
                qwen_model_path or "",
                device=getattr(config, "review_device", "auto"),
                allow_remote=getattr(config, "allow_remote_model_download", False),
            )
        if config is not None and getattr(config, "forced_aligner_enabled", False):
            forced_aligner = LazyQwenForcedAligner(
                getattr(config, "forced_aligner_model_path", None) or "",
                device=getattr(config, "review_device", "auto"),
                allow_remote=getattr(config, "allow_remote_model_download", False),
            )
        if config is not None and getattr(config, "sed_enabled", False):
            sed_engine = LazyAudioClassifierSED(
                getattr(config, "sed_model_path", None) or "",
                device=getattr(config, "review_device", "auto"),
                allow_remote=getattr(config, "allow_remote_model_download", False),
            )
        cache_config = getattr(self.config, "cache", None)
        cache = None
        cache_ttl = None
        if cache_config is not None and getattr(cache_config, "enabled", False):
            cache = self._get_cache()
            cache_ttl = getattr(cache_config, "ttl_transcription", None)
        route = getattr(self, "_asr_route_decision", None)
        engine_name = getattr(route, "selected_engine", None) or getattr(
            getattr(self.config, "asr", None), "engine", ""
        )
        model_name = getattr(getattr(self.config, "asr", None), "model", "")
        route_version = getattr(
            getattr(getattr(self.config, "asr", None), "auto_routing", None),
            "route_version",
            "",
        )
        pair_decision = None
        tail_repair = None
        if physical_timeline is not None and stats is not None:
            duration = float(
                getattr(stats, "duration_seconds", 0.0)
                or getattr(physical_timeline, "duration", 0.0)
                or 0.0
            )
            tail_repair = self._repair_tail_evidence(
                physical_timeline,
                duration,
            )
        pair_config = getattr(getattr(self.config, "asr", None), "engine_pair", None)
        if pair_config is not None and getattr(pair_config, "enabled", True):
            pair_decision = EnginePairRouter(
                getattr(pair_config, "route_version", "asr-pair-v1")
            ).route(
                language=self._resolved_language_or_config(),
                primary=getattr(pair_config, "primary", "auto"),
                secondary=getattr(pair_config, "secondary", "auto"),
                policy=getattr(pair_config, "policy", "risk_only"),
                selected_primary=engine_name,
                same_family_policy=getattr(pair_config, "same_family_policy", "reject"),
            )
            secondary_name = pair_decision.secondary
            # ``secondary=auto`` records Qwen as the preferred heterogeneous
            # pair, but must not load it while the optional review capability
            # is disabled. An explicit ``secondary=qwen`` remains an opt-in
            # path for CLI/WebUI callers.
            qwen_pair_enabled = bool(
                config is not None
                and (
                    getattr(config, "qwen_enabled", False)
                    or getattr(pair_config, "secondary", "auto") == "qwen"
                )
            )
            if secondary_name == "qwen" and qwen_pair_enabled:
                secondary_engine = LazyQwenASR(
                    qwen_model_path or "",
                    device=getattr(config, "review_device", "auto"),
                    allow_remote=getattr(config, "allow_remote_model_download", False),
                ) if config is not None else None
            elif secondary_name in {"faster-whisper", "whisper-cpp", "funasr"}:
                secondary_engine = WindowedASREngine(
                    lambda name=secondary_name: self._get_asr_engine_for(name),
                    name=secondary_name,
                    source="context_reasr",
                    family=getattr(pair_decision, "secondary_family", "whisper") or "whisper",
                    model_name=model_name,
                )
        result = OfflineProductionCoordinator().run(
            OfflineProductionRequest(
                events=events,
                audio=audio,
                sample_rate=sample_rate,
                physical_timeline=physical_timeline,
                global_evidence=tuple(getattr(self, "_global_evidence", ()) or ()),
                recovery_engine=self._get_asr_engine(),
                recovery_language=self._resolved_language_or_config(),
                input_hash=getattr(self, "_file_hash", ""),
                physical_timeline_version="physical-timeline-v1",
                route_version=route_version,
                engine=engine_name,
                model=model_name,
                pair_decision=pair_decision,
                secondary_engine=secondary_name or "",
                pair_route_version=getattr(pair_decision, "route_version", "") if pair_decision else "",
            ),
            EvidenceReviewRuntimePorts(
                config=config,
                context_reasr=review_engine,
                qwen=qwen_engine,
                secondary=secondary_engine,
                secondary_name=secondary_name,
                forced_aligner=forced_aligner,
                sed=sed_engine,
                language=self._resolved_language_or_config(),
                cache=cache,
                cache_ttl=cache_ttl,
            ),
        )
        if stats is not None:
            stats.production_path = result.diagnostics.get("production_path", "")
            stats.review_status = result.diagnostics.get("review_status", "")
            stats.decision_count = int(result.diagnostics.get("decision_count", 0) or 0)
            runtime_status = result.diagnostics.get("status", "completed")
            if runtime_status not in {"completed", "degraded", "failed"}:
                runtime_status = "degraded" if result.diagnostics.get("degraded") else "completed"
            stats.status = runtime_status
            stats.error_category = result.diagnostics.get("error_category", "")
            stats.diagnostics_complete = all(
                key in result.diagnostics
                for key in ("production_path", "review_status", "decision_count", "physical_projection")
            )
            if result.diagnostics.get("fallback_reason"):
                stats.fallback_reason = result.diagnostics["fallback_reason"]
        if tail_repair is not None:
            result.diagnostics["tail_evidence_repair"] = tail_repair
        if route is not None:
            result.diagnostics["route"] = route.to_dict()
        return result.events, result.diagnostics

    def _publish_arbitration_evidence(self) -> None:
        """把全程识别 evidence 发布为时间轴仲裁层 R1 的参照文本区域。

        简化为 (start, end, text) 三元组写入共享的
        ``acoustic_validation.arbitration_evidence_regions``：声学校验在
        postprocess_runner 内部构造 AcousticValidator（该文件属于并行任务
        领地,不改动）,共享配置对象是管线层到 validator 的唯一通道。
        每次发布整体覆盖,不跨任务累积;evidence 不可用时清空,R1 不触发。
        """
        acoustic_config = getattr(self.config, "acoustic_validation", None)
        if acoustic_config is None:
            return
        # 开关关闭时不做任何运行期注入,保持现状行为。
        if not getattr(acoustic_config, "timeline_arbitration", False):
            return
        regions = []
        for item in getattr(self, "_global_evidence", ()) or ():
            text = str(getattr(item, "text", "") or "")
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            if not text.strip() or start is None or end is None:
                continue
            try:
                regions.append((float(start), float(end), text))
            except (TypeError, ValueError):
                continue
        acoustic_config.arbitration_evidence_regions = tuple(regions)

    def _ensure_global_evidence(
        self,
        *,
        audio,
        sample_rate: int,
        physical_timeline=None,
        stats=None,
    ) -> dict:
        """Run global ASR once as evidence without changing primary events."""
        if getattr(self, "_global_evidence_attempted", False):
            return dict(getattr(self, "_global_evidence_diagnostics", {}) or {})

        self._global_evidence_attempted = True
        global_config = getattr(getattr(self.config, "asr", None), "global_asr", None)
        if global_config is None or not getattr(global_config, "enabled", False):
            diagnostics = {"status": "disabled", "role": "evidence"}
            self._global_evidence = ()
            self._global_evidence_diagnostics = diagnostics
            self._publish_arbitration_evidence()
            return diagnostics
        if not getattr(global_config, "evidence_enabled", True):
            diagnostics = {
                "status": "disabled",
                "reason": "evidence_disabled",
                "role": "evidence",
            }
            self._global_evidence = ()
            self._global_evidence_diagnostics = diagnostics
            self._publish_arbitration_evidence()
            return diagnostics
        if audio is None or stats is None:
            diagnostics = {
                "status": "unavailable",
                "reason": "audio_or_stats_missing",
                "role": "evidence",
            }
            self._global_evidence = ()
            self._global_evidence_diagnostics = diagnostics
            self._publish_arbitration_evidence()
            return diagnostics

        stats.global_attempted = True
        shadow = SimpleNamespace(
            physical_timeline=physical_timeline,
            global_speaker_timeline=None,
        )
        try:
            _events, diagnostics, _transcript = self._run_global_transcription_path(
                audio=audio,
                sample_rate=sample_rate,
                shadow=shadow,
                stats=stats,
            )
            diagnostics = {
                **dict(diagnostics or {}),
                "status": "ok" if self._global_evidence else "empty",
                "role": "evidence",
                "candidate_count": len(self._global_evidence),
            }
        except Exception as exc:
            diagnostics = {
                "status": "unavailable",
                "role": "evidence",
                "reason": self._safe_failure_reason(exc),
                "failure_category": self._classify_global_failure(exc),
                "candidate_count": 0,
            }
            self._global_evidence = ()
            logger.warning("Global evidence unavailable: %s", diagnostics["reason"])
        self._global_evidence_diagnostics = diagnostics
        stats.global_diagnostics = {
            **dict(getattr(stats, "global_diagnostics", {}) or {}),
            "evidence": diagnostics,
        }
        self._publish_arbitration_evidence()
        return diagnostics

    def _run_offline_production_review(
        self,
        events,
        *,
        audio=None,
        sample_rate: int = 16000,
        physical_timeline=None,
        stats=None,
    ):
        """Apply the shared offline review component and persist diagnostics."""
        reviewed_events, diagnostics = self._apply_evidence_review(
            events,
            audio=audio,
            sample_rate=sample_rate,
            physical_timeline=physical_timeline,
            stats=stats,
        )
        if stats is not None:
            stats.quality_diagnostics["evidence_review"] = diagnostics
        return reviewed_events

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
        pair_config = getattr(getattr(self.config, "asr", None), "engine_pair", None)
        if pair_config is not None and getattr(pair_config, "enabled", True):
            pair = EnginePairRouter(
                getattr(pair_config, "route_version", "asr-pair-v1")
            ).route(
                language=decision.language,
                primary=getattr(pair_config, "primary", "auto"),
                secondary=getattr(pair_config, "secondary", "auto"),
                policy=getattr(pair_config, "policy", "risk_only"),
                selected_primary=decision.selected_engine,
                same_family_policy=getattr(pair_config, "same_family_policy", "reject"),
            )
            selected_model = decision.selected_model
            if pair.primary == "qwen":
                selected_model = (
                    getattr(self.config.asr, "qwen_model_path", None)
                    or "qwen3-asr"
                )
            decision = replace(
                decision,
                selected_engine=pair.primary,
                selected_model=selected_model,
                decision_reason=f"{decision.decision_reason};pair={pair.decision_reason}",
                fallback_engine=(
                    getattr(self.config.asr.auto_routing, "fallback_engine", None)
                    if pair.primary == "funasr"
                    else None
                ),
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
        self._global_review_timeline = getattr(shadow, "physical_timeline", None)
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
        # Global output is retained as evidence. The default segmented route
        # can use it for risk scoring or bounded replacement, but never as a
        # raw final-event bypass.
        self._global_evidence = tuple(
            result.evidence
            or [candidate_from_subtitle_event(event, source="global") for event in result.events]
        )
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
        from ..physical.allocator import AllocationResult, allocate_words
        from ..physical.coverage import audit_physical_coverage
        from ..physical.events import build_events
        from ..physical.word_alignment import align_words_to_physical

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

        timeline = getattr(shadow, "physical_timeline", None)
        windowed_diagnostics = {}
        global_config = getattr(self.config.asr, "global_asr", None)
        max_window_duration = float(
            getattr(
                global_config,
                "max_window_duration",
                self.GLOBAL_ASR_MAX_DURATION_SECONDS,
            )
        )
        if stats.duration_seconds > max_window_duration:
            transcriber = GlobalTranscriber(
                engine,
                GlobalTranscriberConfig(
                    left_context=float(getattr(global_config, "left_context", 0.5)),
                    right_context=float(getattr(global_config, "right_context", 0.5)),
                    max_window_duration=max_window_duration,
                    window_overlap=float(getattr(global_config, "window_overlap", 0.5)),
                ),
            )
            windowed = transcriber.transcribe(
                audio,
                sample_rate,
                physical_timeline=timeline,
                language=language,
            )
            global_transcript = windowed.transcript
            windowed_diagnostics = {
                "mode": "bounded_windows",
                "windowed_transcription": windowed.diagnostics,
            }
            words = list(global_transcript.words)
            transcript_segments = list(global_transcript.segments)
            word_idx = len(words)
        else:
            # Keep the existing short-audio path stable while long audio uses
            # the independent windowing component above.
            segments = engine.transcribe(audio, sample_rate, language=language)
            if not isinstance(segments, list):
                segments = [segments]

            words = []
            transcript_segments = []
            word_idx = 0
            for seg in segments:
                seg_words = getattr(seg, "words", []) or []
                seg_word_ids = []
                segment_start = float(getattr(seg, "start", 0.0))
                segment_end = float(getattr(seg, "end", 1.0))
                word_ranges = []
                for w in seg_words:
                    word_id = f"word-{word_idx:06d}"
                    word_start = getattr(w, "start", None)
                    word_end = getattr(w, "end", None)
                    native_word_time = word_start is not None and word_end is not None
                    if native_word_time:
                        word_start = float(word_start)
                        word_end = float(word_end)
                        if word_end <= word_start:
                            continue
                        word_ranges.append((word_start, word_end))
                    word = GlobalWord(
                        id=word_id, text=str(getattr(w, "word", "")),
                        raw_start=float(word_start if word_start is not None else seg.start),
                        raw_end=float(word_end if word_end is not None else seg.end),
                        confidence=self._optional_asr_confidence(getattr(w, "confidence", None)),
                        source_window_id="global",
                        segment_id=f"seg-{len(transcript_segments):03d}",
                        metadata={
                            "time_source": (
                                "native_word_timestamp"
                                if native_word_time
                                else "segment_boundary"
                            )
                        },
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
                        confidence=None, source_window_id="global",
                        segment_id=f"seg-{len(transcript_segments):03d}",
                        metadata={"time_source": "segment_boundary"},
                    )
                    words.append(word)
                    seg_word_ids.append(word_id)
                    word_idx += 1
                elif word_ranges:
                    # Some native backends emit a word end just beyond the
                    # segment end. Keep that valid acoustic evidence and
                    # repair the IR container instead of dropping the whole
                    # global transcript during validation.
                    segment_start = min(segment_start, min(item[0] for item in word_ranges))
                    segment_end = max(segment_end, max(item[1] for item in word_ranges))
                transcript_segments.append(GlobalTranscriptSegment(
                    id=f"seg-{len(transcript_segments):03d}",
                    text=str(getattr(seg, "text", "")).strip(),
                    raw_start=segment_start,
                    raw_end=segment_end,
                    word_ids=seg_word_ids,
                ))

            global_transcript = GlobalTranscript(
                audio_duration=float(stats.duration_seconds),
                words=words, segments=transcript_segments,
                backend=engine.name,
                status="ok" if words else "degraded",
            )

        if timeline is None:
            return [], {
                **windowed_diagnostics,
                "recovery": {"status": "no_timeline"},
                "physical_coverage": {"complete": False},
            }, global_transcript

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
            **windowed_diagnostics,
            "physical_coverage": coverage.to_dict(),
            "quality_gate": quality.to_dict(),
            "tail_evidence_repair": tail_repair,
            "recovery": {
                "status": "not_needed" if coverage.complete else "deferred_to_evidence_review"
            },
        }

        if not coverage.complete:
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

        ``global`` is retained as an explicit compatibility route. The
        default and ``auto`` routes use segmented ASR as the primary source.
        """
        if self.config.mode == "streaming":
            return "segmented"
        # Respect explicit override from merge_with_overrides or config
        explicit = getattr(self.config, "asr_path", None)
        if explicit:
            return str(explicit)
        if self._requested_asr_path:
            return "global" if self._requested_asr_path == "global" else "segmented"
        if self.config.asr.global_asr.enabled:
            routing = self.config.asr.global_asr.routing
            if routing == "global":
                return "global"
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
            return cached_path in ("global", "global_evidence")
        # Old cache entries did not carry asr_path; retain compatibility for
        # explicitly requested segmented/legacy runs.
        return cached_path in ("", "segmented", "legacy", "legacy_degraded")
