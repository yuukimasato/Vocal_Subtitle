"""Main offline pipeline lifecycle component."""
from __future__ import annotations
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from ..application.pipeline_result import PipelineStats
from ..mapping.time_mapper import SubtitleEvent
from ..pipeline_context import ASRFragment, NoiseProfile, PipelineContext
from ..utils.audio_utils import AudioUtils
from ..utils.file_hasher import compute_config_hash, compute_file_hash
from ..utils.progress import ProgressManager
logger = logging.getLogger(__name__)
class PipelineLifecycleMixin:
    def run(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        output_format: str = "srt",
        progress_callback: Optional[callable] = None,
        skip_separation: bool = False,
        task_id: Optional[str] = None,
        session_dir: Optional[Path] = None,
        feedback_reference: Optional[Path] = None,  # ★ 反馈学习：用户修订字幕路径
        **overrides,
    ) -> dict:
        """执行全链路处理
        Args:
            input_path: 输入音频文件路径
            output_path: 字幕输出路径，默认与输入同名 .srt
            output_format: 输出格式 (srt / vtt / ass)
            progress_callback: 进度回调函数
            skip_separation: 跳过分离阶段（输入已是人声）
            task_id: 任务 ID（用于历史记录关联）
            session_dir: 会话目录，提供时输出到标准化命名文件，
                         并生成 SRT/VTT/ASS 三种格式
            feedback_reference: 用户修订字幕文件路径（.srt / .ass），
                               提供时在管道完成后自动执行反馈学习
        Returns:
            dict: {
                "subtitle_path": Path,
                "stats": PipelineStats,
                "events": List[SubtitleEvent],
                "from_cache": bool,
            }
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if output_path is None:
            output_path = input_path.with_suffix(f".{output_format}")
        else:
            output_path = Path(output_path)
        if self.config.mode == "streaming":
            logger.info("Pipeline running in streaming mode")
            return self.run_streaming(
                input_path=input_path,
                output_path=output_path,
                output_format=output_format,
                progress_callback=progress_callback,
                skip_separation=skip_separation,
                task_id=task_id,
                **overrides,
            )
        self._file_hash = compute_file_hash(input_path)
        self._config_hash = compute_config_hash(self.config)
        cache_cfg = self.config.cache
        if cache_cfg.enabled and cache_cfg.full_pipeline_cache and not skip_separation:
            history = self._get_history()
            cached_task = history.find_by_hash(
                self._file_hash, self._config_hash
            )
            if cached_task and cached_task.get("result_json"):
                try:
                    cached_result = json.loads(cached_task["result_json"])
                    cached_subtitle_path = Path(cached_result.get("subtitle_path", ""))
                    if (
                        cached_subtitle_path.exists()
                        and self._is_usable_full_pipeline_cache(cached_result)
                    ):
                        logger.info(
                            "Full pipeline cache HIT for %s (task: %s)",
                            input_path.name, cached_task["id"],
                        )
                        subtitle_text = cached_subtitle_path.read_text(encoding="utf-8")
                        output_path.write_text(subtitle_text, encoding="utf-8")
                        cached_stats = cached_result.get("stats", {})
                        stats = PipelineStats.from_dict(
                            input_path,
                            cached_stats if isinstance(cached_stats, dict) else {},
                            duration_seconds=cached_task.get(
                                "total_duration_seconds", 0
                            ),
                        )
                        stats.total_time = 0
                        stats.segment_count = cached_result.get(
                            "segment_count", stats.segment_count
                        )
                        stats.subtitle_count = cached_result.get(
                            "subtitle_count", stats.subtitle_count
                        )
                        cached_events = [
                            item
                            if isinstance(item, SubtitleEvent)
                            else SubtitleEvent.from_dict(item)
                            for item in cached_result.get("events", [])
                        ]
                        stats.subtitle_count = len(cached_events)
                        self._generate_run_report(
                            input_path, stats, task_id,
                            sample_rate=cached_result.get("sample_rate", 0),
                            skip_separation=skip_separation,
                        )
                        return {
                            "subtitle_path": output_path,
                            "stats": stats,
                            "events": cached_events,
                            "from_cache": True,
                            "status": stats.status,
                            "diagnostics": stats.to_dict(),
                        }
                except Exception as e:
                    logger.warning("Failed to restore cached result: %s", e)
        start_time = time.time()
        stats = PipelineStats(input_path=input_path, duration_seconds=0)
        self._resolved_language = None
        self._asr_route_decision = None
        self._asr_engine = None
        if self.config.mode != "streaming":
            try:
                self._run_preflight(
                    input_path,
                    output_path,
                    task_id,
                    stats,
                    skip_separation=skip_separation,
                )
            except Exception as e:
                logger.error("Preflight failed: %s", e)
                return {
                    "subtitle_path": output_path,
                    "stats": stats,
                    "events": [],
                    "from_cache": False,
                    "status": "failed",
                    "diagnostics": stats.to_dict(),
                    "error": str(e),
                }
        self._report_builder = None
        if self.config.mode != "streaming":
            try:
                from ..reporting import RunReportBuilder
                from ..utils.session_manager import create_run_id, create_task_id
                effective_task_id = task_id or create_task_id(input_path)
                effective_run_id = create_run_id(effective_task_id)
                stats.run_id = effective_run_id
                stats.task_id = effective_task_id
                self._report_builder = RunReportBuilder(
                    run_id=effective_run_id, task_id=effective_task_id,
                )
                try:
                    self._services.attach_degradation_logger(self._report_builder.degradation_logger)
                except Exception:
                    pass
            except Exception as e:
                logger.warning("Failed to init report builder early: %s", e)
        active = self._resolve_active_modules()
        if self.config.degradation.mode != "full":
            logger.info(
                "Degradation mode: %s — active modules: %s",
                self.config.degradation.mode,
                {k: v for k, v in active.items() if v},
            )
        total_stages = 5  # vad + merging + asr + mapping + (llm_optimize)
        if not skip_separation:
            total_stages += 1  # separation
        if active["macro_chunk"]:
            total_stages += 1  # macro_chunk (may run)
        if active["ffmpeg_vad"]:
            total_stages += 1  # ffmpeg_vad
        if active["boundary_refinement"]:
            total_stages += 1  # boundary_refinement
        if active["acoustic_validation"]:
            total_stages += 1  # acoustic_validation
        if active["diarization"]:
            total_stages += 1  # diarization
        if active["speaker_role"]:
            total_stages += 1  # role_labeling
        if active["llm_merge"]:
            total_stages += 1  # llm_merge
        self._progress = ProgressManager(
            total_stages=total_stages,
            callback=progress_callback,
        )
        logger.info("Pipeline started: %s → %s", input_path, output_path)
        separation_result = None
        _cached_vocals_path = None
        _cached_accomp_path = None
        if session_dir and not skip_separation:
            from ..utils.session_manager import OUTPUT_NAMES
            _cv = session_dir / OUTPUT_NAMES["vocals"]
            _ca = session_dir / OUTPUT_NAMES["accompaniment"]
            if _cv.exists():
                logger.info(
                    "♻️ Reusing cached vocals from session dir: %s (skipping separation)",
                    _cv,
                )
                _cached_vocals_path = _cv
                _cached_accomp_path = _ca if _ca.exists() else None
                skip_separation = True
        if not skip_separation:
            self._progress.start_stage(
                "separation", description="人声分离"
            )
            def _sep_progress(current: int, total: int) -> None:
                if self._progress:
                    self._progress.report_progress(
                        current, total,
                        extra={"detail": f"处理音频块: {current}/{total}"},
                    )
            separation_result = self._run_separation(
                input_path, progress_callback=_sep_progress
            )
            stats.stage_timings["separation"] = separation_result.processing_time
            vocals_path = separation_result.vocals_path
            self._progress.finish_stage()
        elif _cached_vocals_path is not None:
            vocals_path = _cached_vocals_path
            logger.info("Using cached vocals: %s", vocals_path)
        else:
            vocals_path = input_path
        macro_chunks = None
        if self.config.macro_chunking.enabled:
            from ..macro_chunker import MacroChunker, MacroChunkConfig
            chunker = MacroChunker(self.config.macro_chunking)
            audio, sample_rate = AudioUtils.load_audio(vocals_path)
            stats.duration_seconds = len(audio) / sample_rate
            if chunker.should_split(stats.duration_seconds):
                self._progress.start_stage(
                    "macro_chunk", description="宏观切块", total_items=1,
                )
                try:
                    macro_chunks = chunker.split(vocals_path, audio, sample_rate)
                    logger.info(
                        "Macro chunking: %d chunks from %.1fs audio",
                        len(macro_chunks), stats.duration_seconds,
                    )
                    self._progress.update_stage(
                        1, extra={"detail": f"切分为 {len(macro_chunks)} 块"}
                    )
                    stats.stage_timings["macro_chunk"] = self._progress.finish_stage()
                except Exception as e:
                    logger.warning("Macro chunking failed, treating as single chunk: %s", e)
                    macro_chunks = None
                    self._progress.finish_stage()
        if "audio" not in locals() or "sample_rate" not in locals():
            audio, sample_rate = AudioUtils.load_audio(vocals_path)
            stats.duration_seconds = len(audio) / sample_rate
        # ---- [层1] 身份主干 P1:全局 diarization 前置（early_turns） ----
        # 分离之后、chunk 处理之前对完整人声音频跑一次全局 pass，
        # turns 贯通 chunk_runner（合并硬约束）与 postprocess_runner
        # （事件标签来源）。early_turns=false 时此处仅记 disabled，
        # 后续行为与现状完全一致（后处理事件级聚类照常运行）。
        self._early_turns_state = None
        self._early_turn_spans: List[Any] = []
        self._run_early_global_turns(audio, sample_rate, stats)
        requested_asr_path = self._resolve_asr_path()
        stats.asr_path = requested_asr_path
        global_completed = False
        self._global_evidence = ()
        self._global_evidence_attempted = False
        self._global_evidence_diagnostics = {}
        self._global_review_timeline = None
        quality_speech_intervals = None
        if requested_asr_path in ("global", "global_primary"):
            stats.global_attempted = True
            global_diag = {
                "route": requested_asr_path,
                "duration_seconds": stats.duration_seconds,
            }
            stats.global_diagnostics = dict(global_diag)
            try:
                shadow_context, shadow_vad, shadow_ffmpeg, shadow_noise = (
                    self._run_early_detection(audio, sample_rate, vocals_path)
                )
                shadow = self._build_physical_shadow(
                    audio,
                    sample_rate,
                    vocals_path,
                    shadow_context,
                    shadow_vad,
                    shadow_ffmpeg,
                    shadow_noise,
                )
                self._global_review_timeline = getattr(
                    shadow, "physical_timeline", None
                )
                decision = self._prepare_asr_route(
                    audio,
                    sample_rate,
                    speech_intervals=[
                        (item.start, item.end) for item in shadow_vad
                    ] + list((shadow_ffmpeg or {}).get("skeleton", [])),
                )
                stats.requested_engine = decision.requested_engine
                stats.selected_engine = decision.selected_engine
                stats.final_engine = decision.selected_engine
                stats.detected_language = decision.detected_language
                stats.language_probability = decision.language_probability
                stats.asr_route_version = decision.route_version
                stats.quality_gate_version = decision.quality_gate_version
                global_diag["route"] = decision.to_dict()
                events, global_diag, global_transcript = (
                    self._run_global_transcription_path(
                        audio=audio,
                        sample_rate=sample_rate,
                        shadow=shadow,
                        stats=stats,
                        vad_segments=shadow_vad,
                        ffmpeg_result=shadow_ffmpeg,
                        noise_profile=shadow_noise,
                    )
                )
                self._global_evidence_attempted = True
                global_diag = dict(global_diag or {})
                global_diag["route"] = decision.to_dict()
                global_diag["shadow"] = {
                    "status": getattr(shadow, "status", "unknown"),
                    "diagnostics": getattr(shadow, "diagnostics", {}),
                    "statistics": getattr(shadow, "statistics", {}),
                }
                stats.global_diagnostics = global_diag
                quality_payload = global_diag.get("quality_gate", {})
                stats.quality_status = quality_payload.get("status", "pass")
                stats.quality_diagnostics["asr_quality_gate"] = quality_payload
                if (
                    decision.selected_engine == "funasr"
                    and quality_payload.get("status") == "failed"
                ):
                    raise RuntimeError(
                        "FunASR quality gate failed: "
                        + ", ".join(quality_payload.get("reasons", []))
                    )
                self._validate_global_result(events, global_transcript, global_diag)
                events = self._run_offline_production_review(
                    events,
                    audio=audio,
                    sample_rate=sample_rate,
                    physical_timeline=self._global_review_timeline,
                    stats=stats,
                )
                events = self._post_process_events(
                    events,
                    vocals_path,
                    audio,
                    sample_rate,
                    stats,
                    ffmpeg_unified_result=shadow_ffmpeg,
                )
                stats.asr_path = "global"
                stats.segment_count = len(global_transcript.segments)
                stats.subtitle_count = len(events)
                global_diag["role"] = "primary"
                global_completed = bool(events)
            except Exception as exc:
                category = self._classify_global_failure(exc)
                reason = self._safe_failure_reason(exc)
                stats.fallback_category = category
                stats.fallback_reason = reason
                stats.global_diagnostics = {
                    **stats.global_diagnostics,
                    "status": "failed",
                    "failure_category": category,
                    "failure_reason": reason,
                }
                logger.warning(
                    "Global ASR failed (%s): %s", category, reason,
                )
                decision = self._asr_route_decision
                can_fallback = bool(
                    requested_asr_path == "auto"
                    and decision is not None
                    and decision.requested_engine == "auto"
                    and decision.selected_engine == "funasr"
                    and decision.fallback_engine == "faster-whisper"
                )
                if can_fallback:
                    try:
                        logger.warning("Falling back once from FunASR to faster-whisper")
                        self._asr_engine = self._get_asr_engine_for(
                            "faster-whisper"
                        )
                        fallback_events, fallback_diag, fallback_transcript = (
                            self._run_global_transcription_path(
                                audio=audio,
                                sample_rate=sample_rate,
                                shadow=shadow,
                                stats=stats,
                                vad_segments=shadow_vad,
                                ffmpeg_result=shadow_ffmpeg,
                                noise_profile=shadow_noise,
                            )
                        )
                        fallback_diag = dict(fallback_diag or {})
                        fallback_diag["route"] = decision.to_dict()
                        fallback_diag["fallback"] = {
                            "from": "funasr",
                            "to": "faster-whisper",
                            "reason": reason,
                        }
                        self._validate_global_result(
                            fallback_events, fallback_transcript, fallback_diag
                        )
                        events = fallback_events
                        global_transcript = fallback_transcript
                        global_diag = {
                            "primary": stats.global_diagnostics,
                            "fallback_result": fallback_diag,
                        }
                        stats.global_diagnostics = global_diag
                        stats.fallback_category = category
                        stats.fallback_reason = reason
                        stats.final_engine = "faster-whisper"
                        fallback_quality = fallback_diag.get("quality_gate", {})
                        stats.quality_status = fallback_quality.get(
                            "status", "degraded"
                        )
                        stats.quality_diagnostics["asr_quality_gate"] = {
                            "primary": stats.global_diagnostics.get("primary", {}).get(
                                "quality_gate", {}
                            ),
                            "fallback": fallback_quality,
                        }
                        events = self._post_process_events(
                            events,
                            vocals_path,
                            audio,
                            sample_rate,
                            stats,
                            ffmpeg_unified_result=shadow_ffmpeg,
                        )
                        stats.asr_path = "global"
                        stats.segment_count = len(fallback_transcript.segments)
                        stats.subtitle_count = len(events)
                        stats.global_diagnostics["role"] = "primary"
                        global_completed = bool(events)
                    except Exception as fallback_exc:
                        fallback_reason = self._safe_failure_reason(fallback_exc)
                        stats.global_diagnostics["fallback_error"] = fallback_reason
                        stats.quality_status = "failed"
                        logger.warning(
                            "faster-whisper fallback failed: %s", fallback_reason
                        )
                if requested_asr_path == "global":
                    stats.asr_path = "global"
                    raise  # 用户显式要求 global 路径，不应静默降级
                if not global_completed:
                    stats.asr_path = "legacy_degraded"
        if self._asr_route_decision is None:
            decision = self._prepare_asr_route(audio, sample_rate)
            stats.requested_engine = decision.requested_engine
            stats.selected_engine = decision.selected_engine
            stats.final_engine = decision.selected_engine
            stats.detected_language = decision.detected_language
            stats.language_probability = decision.language_probability
            stats.asr_route_version = decision.route_version
            stats.quality_gate_version = decision.quality_gate_version
        self._prepare_task_language(audio, sample_rate)
        if global_completed:
            pass
        elif self.config.acoustic_validation.skeleton_mode:
            logger.info("Skeleton segmentation mode enabled")
            if 'audio' not in dir() or 'sample_rate' not in dir():
                audio, sample_rate = AudioUtils.load_audio(vocals_path)
                stats.duration_seconds = len(audio) / sample_rate
            events, seg_count, skeleton_ffmpeg_result = self._process_skeleton_segmented(
                audio=audio,
                sample_rate=sample_rate,
                vocals_path=vocals_path,
            )
            stats.segment_count = seg_count
            stats.subtitle_count = len(events)
            quality_speech_intervals = (skeleton_ffmpeg_result or {}).get("skeleton", [])
            if getattr(self.config, "evidence_review", None) is not None:
                events = self._run_offline_production_review(
                    events,
                    audio=audio,
                    sample_rate=sample_rate,
                    physical_timeline=getattr(self, "_global_review_timeline", None),
                    stats=stats,
                )
                stats.subtitle_count = len(events)
            events = self._post_process_events(
                events, vocals_path, audio, sample_rate, stats,
                ffmpeg_unified_result=skeleton_ffmpeg_result,
            )
            from ..asr.quality_gate import evaluate_asr_quality
            quality = evaluate_asr_quality(
                events,
                quality_speech_intervals or [
                    (getattr(item, "physical_start", item.start),
                     getattr(item, "physical_end", item.end))
                    for item in events
                ],
                stats.duration_seconds,
                **self._quality_gate_kwargs(),
            )
            stats.quality_status = quality.status
            stats.quality_diagnostics["asr_quality_gate"] = quality.to_dict()
            decision = self._asr_route_decision
            if (
                quality.status == "failed"
                and decision is not None
                and decision.requested_engine == "auto"
                and decision.selected_engine == "funasr"
                and decision.fallback_engine == "faster-whisper"
            ):
                logger.warning("Segmented FunASR quality gate failed; retrying with faster-whisper")
                self._asr_engine = self._get_asr_engine_for("faster-whisper")
                events, seg_count, skeleton_ffmpeg_result = self._process_skeleton_segmented(
                    audio=audio,
                    sample_rate=sample_rate,
                    vocals_path=vocals_path,
                )
                stats.segment_count = seg_count
                stats.subtitle_count = len(events)
                if getattr(self.config, "evidence_review", None) is not None:
                    events = self._run_offline_production_review(
                        events,
                        audio=audio,
                        sample_rate=sample_rate,
                        physical_timeline=getattr(self, "_global_review_timeline", None),
                        stats=stats,
                    )
                    stats.subtitle_count = len(events)
                events = self._post_process_events(
                    events,
                    vocals_path,
                    audio,
                    sample_rate,
                    stats,
                    ffmpeg_unified_result=skeleton_ffmpeg_result,
                )
                stats.final_engine = "faster-whisper"
                fallback_quality = evaluate_asr_quality(
                    events,
                    (skeleton_ffmpeg_result or {}).get("skeleton", []),
                    stats.duration_seconds,
                    **self._quality_gate_kwargs(),
                )
                stats.quality_status = fallback_quality.status
                stats.quality_diagnostics["asr_quality_gate"] = {
                    "primary": quality.to_dict(),
                    "fallback": fallback_quality.to_dict(),
                }
        elif macro_chunks is not None and len(macro_chunks) > 1:
            logger.info(
                "Multi-chunk path: processing %d chunks", len(macro_chunks),
            )
            all_chunk_events: List[SubtitleEvent] = []
            total_segments = 0
            speaker_offset = 0
            max_speaker_per_chunk = 0
            # [层1] early_turns 生效时标签来自全局 turns（全局唯一），
            # 无需跨块偏移；关闭时保持跨块 speaker_offset 累加补丁。
            early_turns_active = self._early_turns_active()
            for idx, chunk in enumerate(macro_chunks):
                chunk_audio = chunk.audio
                chunk_sr = sample_rate
                chunk_duration = len(chunk_audio) / chunk_sr
                logger.info(
                    "Chunk %d/%d: %.1fs → %.1fs (duration=%.1fs)",
                    idx + 1, len(macro_chunks),
                    chunk.start, chunk.end, chunk_duration,
                )
                import tempfile
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False,
                ) as tmp_f:
                    tmp_path = Path(tmp_f.name)
                try:
                    AudioUtils.save_audio(chunk_audio, tmp_path, chunk_sr)
                    chunk_events, chunk_seg_count, _chunk_ctx = self._process_chunk_pipeline(
                        audio=chunk_audio,
                        sample_rate=chunk_sr,
                        vocals_path=tmp_path,
                        chunk_label=f"Chunk {idx+1}/{len(macro_chunks)}",
                        parallel_vad=False,  # 多块嵌套线程，避免 PyTorch 死锁
                        time_offset=chunk.start,
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)
                # ★ 跨块 speaker_id 偏移（段级聚类遗留；early_turns 生效时跳过）
                chunk_speakers = set()
                for evt in chunk_events:
                    if evt.speaker_id is not None:
                        chunk_speakers.add(evt.speaker_id)
                if chunk_speakers and not early_turns_active:
                    max_speaker_per_chunk = max(chunk_speakers)
                    if speaker_offset > 0:
                        for evt in chunk_events:
                            if evt.speaker_id is not None:
                                evt.speaker_id += speaker_offset
                    speaker_offset += max_speaker_per_chunk + 1
                for evt in chunk_events:
                    from ..mapping.time_mapper import offset_subtitle_event
                    offset_subtitle_event(
                        evt,
                        chunk.start,
                        source=f"macro_chunk:{chunk.index}",
                    )
                total_segments += chunk_seg_count
                all_chunk_events.extend(chunk_events)
            all_chunk_events.sort(key=lambda e: e.start)
            if len(macro_chunks) > 1:
                from ..mapping.time_mapper import _merge_distinct_texts
                stitched_events: List[SubtitleEvent] = []
                for i in range(len(all_chunk_events)):
                    evt = all_chunk_events[i]
                    if stitched_events:
                        last = stitched_events[-1]
                        if abs(evt.start - last.start) < 0.05:
                            if len(evt.text) > len(last.text):
                                stitched_events[-1] = evt
                            continue
                        overlap_start = max(last.start, evt.start)
                        overlap_end = min(last.end, evt.end)
                        overlap_dur = overlap_end - overlap_start
                        if overlap_dur > 0:
                            last_dur = last.end - last.start
                            evt_dur = evt.end - evt.start
                            min_dur = min(last_dur, evt_dur)
                            if min_dur > 0:
                                overlap_ratio = overlap_dur / min_dur
                                if overlap_ratio >= 0.5:
                                    last_spk = getattr(last, "speaker_id", None)
                                    evt_spk = getattr(evt, "speaker_id", None)
                                    same_or_unknown = not (
                                        last_spk is not None
                                        and evt_spk is not None
                                        and last_spk != evt_spk
                                    )
                                    if same_or_unknown:
                                        if last.start <= evt.start and last.end >= evt.end:
                                            last.text = _merge_distinct_texts(
                                                last.text, evt.text
                                            )
                                            logger.debug(
                                                "Cross-chunk merge: #%d absorbs #%d",
                                                last.index, evt.index,
                                            )
                                            continue
                                        elif evt.start <= last.start and evt.end >= last.end:
                                            evt.text = _merge_distinct_texts(
                                                evt.text, last.text
                                            )
                                            stitched_events[-1] = evt
                                            logger.debug(
                                                "Cross-chunk merge: #%d absorbs #%d",
                                                evt.index, last.index,
                                            )
                                            continue
                    stitched_events.append(evt)
                all_chunk_events = stitched_events
            for i, evt in enumerate(all_chunk_events):
                evt.index = i + 1
            events = all_chunk_events
            stats.segment_count = total_segments
            stats.subtitle_count = len(events)
            if getattr(self.config, "evidence_review", None) is not None:
                events = self._run_offline_production_review(
                    events,
                    audio=audio,
                    sample_rate=sample_rate,
                    physical_timeline=getattr(self, "_global_review_timeline", None),
                    stats=stats,
                )
                stats.subtitle_count = len(events)
            multi_ffmpeg_result = None
            try:
                from ..vad.ffmpeg_vad import unified_ffmpeg_pass
                multi_ffmpeg_result = unified_ffmpeg_pass(vocals_path)
            except Exception:
                pass
            events = self._post_process_events(
                events, vocals_path, audio, sample_rate, stats,
                ffmpeg_unified_result=multi_ffmpeg_result,
            )
        else:
            if macro_chunks is not None and len(macro_chunks) == 1:
                chunk = macro_chunks[0]
                audio = chunk.audio
            if 'audio' not in dir() or 'sample_rate' not in dir():
                audio, sample_rate = AudioUtils.load_audio(vocals_path)
                stats.duration_seconds = len(audio) / sample_rate
            events, seg_count, ctx = self._process_chunk_pipeline(
                audio=audio,
                sample_rate=sample_rate,
                vocals_path=vocals_path,
                chunk_label="",
            )
            stats.segment_count = seg_count
            stats.subtitle_count = len(events)
            if self._global_review_timeline is None:
                self._global_review_timeline = self._build_review_timeline_from_context(
                    ctx, stats.duration_seconds
                )
            if getattr(self.config, "evidence_review", None) is not None:
                events = self._run_offline_production_review(
                    events,
                    audio=audio,
                    sample_rate=sample_rate,
                    physical_timeline=getattr(self, "_global_review_timeline", None),
                    stats=stats,
                )
                stats.subtitle_count = len(events)
            events = self._post_process_events(
                events, vocals_path, audio, sample_rate, stats,
                ffmpeg_unified_result=ctx.ffmpeg_unified_result,
            )
        if stats.detected_language == "unknown" and self._resolved_language:
            stats.detected_language = str(self._resolved_language)
        try:
            from ..mapping.end_time_validator import EndTimePostValidator
            validator = EndTimePostValidator()
            events = validator.validate(events)
        except Exception as e:
            logger.warning("EndTimePostValidator failed: %s", e)
        builder = self._get_subtitle_builder()
        clean_subtitle_paths: Dict[str, str] = {}
        clean_subtitle_path = str(output_path)
        if self.config.llm_optimize.enabled:
            clean_subtitle_paths = self._export_subtitles_multi_format(
                builder, events, output_path, output_format, session_dir, label="asr"
            )
            clean_subtitle_path = clean_subtitle_paths.get(
                "srt" if session_dir else output_format,
                str(output_path),
            )
        if self.config.acoustic_validation.export_skeleton_segments:
            try:
                from ..acoustic import export_skeleton_segments
                export_dir = self.config.acoustic_validation.export_skeleton_dir
                if not export_dir:
                    export_dir = str(output_path.parent / "skeleton_export")
                export_result = export_skeleton_segments(
                    audio_path=vocals_path,
                    output_dir=Path(export_dir),
                    noise_db=self.config.acoustic_validation.skeleton_noise_db,
                    min_silence_duration=self.config.acoustic_validation.skeleton_min_silence,
                    min_speech_duration=self.config.acoustic_validation.skeleton_min_speech,
                )
                logger.info(
                    "Exported %d skeleton segments (speech=%d, silence=%d) → %s",
                    export_result["total_segments"],
                    export_result["speech_segments"],
                    export_result["silence_segments"],
                    export_result["output_dir"],
                )
            except Exception as e:
                logger.warning("Skeleton segment export failed: %s", e)
        llm_subtitle_path = None
        llm_subtitle_paths: Dict[str, str] = {}
        if self.config.llm_optimize.enabled:
            self._progress.start_stage(
                "llm", description="LLM 优化", total_items=1
            )
            events = self._run_llm_optimize(events)
            self._progress.update_stage(
                1, extra={"detail": f"LLM 优化完成，共 {len(events)} 条字幕"}
            )
            stats.stage_timings["llm"] = self._progress.finish_stage()
        events = self._finalize_events(events, stats, stats.duration_seconds)
        # ---- 显示 cue 能量对齐（2026-09-13 诊断定案） ----
        # finalize 的显示拆行以词起点/文本宽度为准，句内停顿处偏早的
        # 行首会把静音吞进行首；骨架路径的合并事件无词表，校验阶段无法
        # 修正。这里在拆行之后、导出之前对最终 cue 做一次能量对齐：
        # start 后向吸附到真实语音起点，end 在连续语音骨架段内延长。
        if self.config.acoustic_validation.enabled:
            try:
                from ..acoustic import AcousticValidator
                from ..merging.fragment_absorber import resolve_speech_skeleton

                _validator = AcousticValidator(
                    self.config.acoustic_validation
                )
                _skeleton = resolve_speech_skeleton(
                    vocals_path, None,
                    self.config.acoustic_validation, audio, sample_rate,
                )
                if _skeleton:
                    events, _align_report = _validator._physical_snap_validation(
                        events, _skeleton, audio=audio, sample_rate=sample_rate,
                    )
                    stats.quality_diagnostics["display_energy_alignment"] = {
                        "snapped_starts": _align_report["snapped_starts"],
                        "ends_extended": _align_report.get("ends_extended", 0),
                        "snapped_ends": _align_report["snapped_ends"],
                    }
            except Exception as e:
                logger.warning("Display energy alignment failed: %s", e)
        export_label = "llm" if self.config.llm_optimize.enabled else "asr"
        final_paths = self._export_subtitles_multi_format(
            builder, events, output_path, output_format, session_dir, label=export_label
        )
        final_subtitle_path = final_paths.get(output_format, str(output_path))
        if session_dir:
            final_subtitle_path = final_paths.get("srt", final_subtitle_path)
        if self.config.llm_optimize.enabled:
            llm_subtitle_paths = final_paths
            llm_subtitle_path = final_subtitle_path
        else:
            clean_subtitle_paths = final_paths
            clean_subtitle_path = final_subtitle_path
        logger.info(
            "Final subtitle exported: %s (%d events)",
            final_subtitle_path, len(events),
        )
        if session_dir and separation_result:
            import shutil
            from ..utils.session_manager import OUTPUT_NAMES
            vocals_dest = session_dir / OUTPUT_NAMES["vocals"]
            accomp_dest = session_dir / OUTPUT_NAMES["accompaniment"]
            if separation_result.vocals_path and Path(separation_result.vocals_path).exists():
                shutil.copy2(separation_result.vocals_path, vocals_dest)
            if separation_result.accompaniment_path and Path(separation_result.accompaniment_path).exists():
                shutil.copy2(separation_result.accompaniment_path, accomp_dest)
        if session_dir:
            try:
                from ..utils.session_manager import SessionManager
                mgr = SessionManager(session_dir.parent)
                outputs_info = {}
                for fmt_key, path in {**clean_subtitle_paths, **llm_subtitle_paths}.items():
                    p = Path(path)
                    if p.exists():
                        outputs_info[p.name] = {
                            "sha256": compute_file_hash(p),
                            "size": p.stat().st_size,
                        }
                mgr.write_metadata(
                    session_dir,
                    original_filename="",
                    input_sha256=self._file_hash,
                    profile=getattr(self.config, '_profile_name', ''),
                    config_hash=self._config_hash,
                    task_id=task_id or "",
                    outputs=outputs_info,
                )
            except Exception as e:
                logger.warning("Failed to write session metadata: %s", e)
        stats.total_time = time.time() - start_time
        logger.info(
            "Pipeline complete: %.1fs total, %d subtitle events",
            stats.total_time,
            stats.subtitle_count,
        )
        if self._progress:
            pm_stats = self._progress.get_stats()
            for stage_name, elapsed in pm_stats.get("stage_timings", {}).items():
                if stage_name not in stats.stage_timings:
                    stats.stage_timings[stage_name] = elapsed
        from ..utils.session_manager import OUTPUT_NAMES
        vocals_result = (
            str(session_dir / OUTPUT_NAMES["vocals"])
            if session_dir and (session_dir / OUTPUT_NAMES["vocals"]).exists()
            else str(separation_result.vocals_path) if separation_result else None
        )
        accomp_result = (
            str(session_dir / OUTPUT_NAMES["accompaniment"])
            if session_dir and (session_dir / OUTPUT_NAMES["accompaniment"]).exists()
            else str(separation_result.accompaniment_path) if separation_result else None
        )
        feedback_report = None
        if feedback_reference and self.config.feedback.enabled:
            feedback_report = self._run_feedback_learning(
                auto_events=events,
                reference_path=feedback_reference,
                audio_path=str(vocals_path),
            )
        if stats.fallback_reason and stats.status == "completed":
            stats.status = "degraded_completed"
        from .run_finalizer import build_result_payload

        result_payload = build_result_payload(
            task_id=getattr(self, "_effective_task_id", None) or stats.task_id or task_id or "",
            stats=stats,
            events=events,
            input_path=input_path,
            subtitle_path=final_subtitle_path,
            clean_subtitle_path=clean_subtitle_path,
            llm_subtitle_path=llm_subtitle_path,
            vocals_path=vocals_result,
            accompaniment_path=accomp_result,
        )
        self._finalize_task_state(stats, result_payload=result_payload)
        self._generate_run_report(
            input_path, stats, task_id,
            sample_rate=sample_rate if "sample_rate" in dir() else 0,
            final_subtitle_path=final_subtitle_path,
        )
        return {
            "subtitle_path": final_subtitle_path,
            "clean_subtitle_path": clean_subtitle_path,
            "llm_subtitle_path": llm_subtitle_path,
            "stats": stats,
            "events": events,
            "from_cache": False,
            "status": stats.status,
            "diagnostics": stats.to_dict(),
            "vocals_path": vocals_result,
            "accompaniment_path": accomp_result,
            "clean_subtitle_paths": clean_subtitle_paths,
            "llm_subtitle_paths": llm_subtitle_paths,
            "feedback_report": feedback_report,  # ★ 反馈学习报告
        }
