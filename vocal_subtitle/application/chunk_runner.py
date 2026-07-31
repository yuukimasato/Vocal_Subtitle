"""Chunk and skeleton-mode execution stages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..asr.base import ASRInvalidResultError
from ..asr.contracts import ASRRuntimePorts, SegmentedASRRequest
from ..asr.segmented_path import SegmentedASRService
from ..mapping.time_mapper import SubtitleEvent
from ..pipeline_context import NoiseProfile, PipelineContext
from ..utils.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class PipelineChunkMixin:
    def _resolve_active_modules(self) -> Dict[str, bool]:
        """根据降级模式和运行时模式决定启用哪些模块 (文档 5.5.2 + 5.12.5)

        降级模式 (degradation.mode):
          - "full":      所有模块按配置启用
          - "degraded":  禁用所有 LLM 调用，使用规则替代
          - "minimal":   仅 VAD + ASR + 规则合并

        运行时模式 (config.mode):
          - "offline":   按降级模式正常启用
          - "streaming": 自动降级依赖全局视角的模块（方案〇/二/七），
                         LLM 合并降级为本地 NLP

        Returns:
            dict: 模块名 → 是否启用
        """
        # 流式模式下，从流式降级映射开始
        if self.config.mode == "streaming":
            from .streaming import resolve_streaming_modules
            streaming_modules = resolve_streaming_modules()

            # 如果同时有降级模式 (degraded/minimal)，叠加降级
            if self.config.degradation.mode == "minimal":
                # minimal 叠加：禁用更多
                streaming_modules.update({
                    "ffmpeg_vad": False,
                    "pre_split": False,
                    "asr_refine": False,
                    "llm_merge": False,
                    "frame_seamless": False,
                    "diarization": False,
                    "speaker_role": False,
                })
            elif self.config.degradation.mode == "degraded":
                # degraded 叠加：关闭 LLM 相关
                streaming_modules.update({
                    "llm_merge": False,
                    "speaker_role": False,
                    "llm_optimize": False,
                })

            return streaming_modules

        # ---- 离线模式：原逻辑 ----
        mode = self.config.degradation.mode

        if mode == "minimal":
            # 仅 VAD + ASR + 规则合并
            return {
                "macro_chunk": False,
                "ffmpeg_vad": False,
                "fusion": False,
                "pre_split": False,
                "adaptive_padding": False,
                "boundary_refinement": False,
                "llm_merge": False,
                "acoustic_validation": False,
                "diarization": False,
                "speaker_role": False,
                "llm_optimize": False,
            }

        if mode == "degraded":
            # 禁用所有 LLM 调用
            return {
                "macro_chunk": True,
                "ffmpeg_vad": True,
                "fusion": self.config.fusion.enabled,
                "pre_split": True,
                "adaptive_padding": True,
                "boundary_refinement": True,
                "llm_merge": False,
                "acoustic_validation": True,
                "diarization": self.config.diarization.enabled,
                "speaker_role": False,
                "llm_optimize": False,
            }

        # mode == "full": 全部按配置启用
        return {
            "macro_chunk": self.config.macro_chunking.enabled,
            "ffmpeg_vad": self.config.vad.ffmpeg_enabled,
            "fusion": self.config.fusion.enabled,
            "pre_split": self.config.merging.pre_split_silence,
            "adaptive_padding": self.config.merging.adaptive_padding,
            "boundary_refinement": self.config.boundary_refinement.enabled,
            "llm_merge": self.config.merge_decision.llm_tier != "rule_only",
            "acoustic_validation": self.config.acoustic_validation.enabled,
            "diarization": self.config.diarization.enabled,
            "speaker_role": self.config.speaker_role.enabled,
            "llm_optimize": self.config.llm_optimize.enabled,
        }

    def _process_chunk_pipeline(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vocals_path: Path,
        chunk_label: str = "",
        parallel_vad: bool = True,
        run_asr: bool = True,
    ) -> tuple:
        """处理单个音频块的完整管线 (VAD → Merge → ASR → Refine → Mapping)

        供单块路径和多块路径共用。使用 PipelineContext 作为数据载体
        在模块间传递共享数据（文档 5.1.2）。

        Args:
            audio: 音频 numpy 数组
            sample_rate: 采样率
            vocals_path: 音频文件路径（供 ffmpeg 调用）
            chunk_label: 块标签（多块模式下用于日志）
            parallel_vad: 是否用 ThreadPoolExecutor 并行执行 Silero + ffmpeg VAD。
                          骨架分段/多块/流式等嵌套线程场景应设为 False，
                          避免 PyTorch 推理与 ThreadPoolExecutor 的三层嵌套死锁。

        Returns:
            (events: List[SubtitleEvent], segment_count: int, ctx: PipelineContext)
            ctx 含声学骨架等信息，供后处理阶段复用。
        """
        prefix = f"[{chunk_label}] " if chunk_label else ""
        chunk_duration = len(audio) / sample_rate

        # ---- 初始化 PipelineContext（统一数据载体） ----
        ctx = PipelineContext(
            audio_path=vocals_path,
            audio=audio,
            sample_rate=sample_rate,
        )

        # ---- 前置降噪（可选，5.12.1） ----
        if self.config.noise_reduction.enabled:
            try:
                from ..audio_preprocessor import AudioPreprocessor, DenoiseConfig
                denoise_cfg = DenoiseConfig(
                    enabled=True,
                    engine=self.config.noise_reduction.engine,
                    spectral_noise_reduction_db=(
                        self.config.noise_reduction.spectral_noise_reduction_db
                    ),
                    spectral_noise_estimation_frames=(
                        self.config.noise_reduction.spectral_noise_estimation_frames
                    ),
                    burst_noise_protection=(
                        self.config.noise_reduction.burst_noise_protection
                    ),
                    burst_noise_threshold_db=(
                        self.config.noise_reduction.burst_noise_threshold_db
                    ),
                    burst_noise_max_duration_ms=(
                        self.config.noise_reduction.burst_noise_max_duration_ms
                    ),
                )
                preprocessor = AudioPreprocessor(denoise_cfg)
                audio, denoise_report = preprocessor.process(audio, sample_rate)
                ctx.add_diagnostic(
                    f"Denoise: engine={denoise_report.get('engine', '?')}, "
                    f"rms_reduction={denoise_report.get('rms_reduction_db', 0):.1f}dB, "
                    f"burst={denoise_report.get('burst_events_detected', 0)}"
                )
                logger.info(
                    "%sDenoise: engine=%s, rms_reduction=%.1fdB, burst=%d",
                    prefix,
                    denoise_report.get("engine", "?"),
                    denoise_report.get("rms_reduction_db", 0),
                    denoise_report.get("burst_events_detected", 0),
                )
            except Exception as e:
                logger.warning("%sDenoise failed, continuing with original: %s", prefix, e)
                ctx.add_diagnostic(f"Denoise FAILED: {e}")

        # ---- 环境底噪自适应采样 ----
        noise_profile = AudioUtils.estimate_noise_floor_per_chunk(
            audio, sample_rate, chunk_duration=chunk_duration,
        )
        ctx.noise_profile = NoiseProfile(
            noise_rms=noise_profile["noise_rms"],
            speech_threshold=noise_profile["speech_threshold"],
            is_noisy_environment=noise_profile["is_noisy_environment"],
        )
        ctx.add_diagnostic(
            f"Noise: rms={noise_profile['noise_rms']:.6f}, "
            f"threshold={noise_profile['speech_threshold']:.6f}, "
            f"noisy={noise_profile['is_noisy_environment']}"
        )
        logger.info(
            "%sNoise profile: rms=%.6f, threshold=%.6f, noisy=%s",
            prefix,
            noise_profile["noise_rms"],
            noise_profile["speech_threshold"],
            noise_profile["is_noisy_environment"],
        )

        # ---- Stage 2+2.5: Silero VAD 和 ffmpeg VAD 执行 ----
        # 两种执行模式：
        # 1) parallel_vad=True（默认）：ThreadPoolExecutor 并行执行，
        #    适用单块路径（无嵌套线程风险）
        # 2) parallel_vad=False：串行执行，先 Silero 后 ffmpeg，
        #    适用骨架分段/多块/流式路径，避免 PyTorch 推理在
        #    ThreadPoolExecutor worker 中与 daemon 线程形成三层嵌套死锁

        self._progress.start_stage("vad", description=f"{chunk_label}语音检测")

        vad_segments = []
        ffmpeg_result = None
        if self.config.vad.ffmpeg_enabled:
            if parallel_vad:
                # 并行执行 Silero VAD 和 ffmpeg VAD（单块路径）
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_silero = executor.submit(
                        self._run_vad, audio, sample_rate,
                    )
                    future_ffmpeg = executor.submit(
                        self._run_ffmpeg_vad, vocals_path, ctx, prefix,
                    )
                    vad_segments = future_silero.result()
                    ffmpeg_result = future_ffmpeg.result()
            else:
                # 串行执行（骨架分段 / 多块 / 流式路径）
                # 避免 ThreadPoolExecutor 嵌套带来的 PyTorch 线程死锁
                vad_segments = self._run_vad(audio, sample_rate)
                ffmpeg_result = self._run_ffmpeg_vad(
                    vocals_path, ctx, prefix,
                )

            # 三方法融合（如果启用，逻辑不变）
            if ffmpeg_result is not None and self.config.fusion.enabled:
                from ..vad.boundary_fusion import BoundaryFusion

                fusion_engine = BoundaryFusion(self.config.fusion)
                ffmpeg_segments = ffmpeg_result.get("coarse_speech", [])
                vad_segments = fusion_engine.fuse(
                    vad_segments, ffmpeg_segments, audio, sample_rate,
                )
                ctx.add_diagnostic(
                    f"Fusion: {len(vad_segments)} segments after 3-method fusion"
                )
        else:
            vad_segments = self._run_vad(audio, sample_rate)

        # 报告 VAD 检测结果，让前端显示有意义的进度信息
        self._progress.update_stage(
            1, extra={"detail": f"检测到 {len(vad_segments)} 个语音段"}
        )
        self._progress.finish_stage()

        # ---- Stage 3: 片段合并 ----
        self._progress.start_stage(
            "merging", description=f"{chunk_label}片段合并", total_items=1,
        )
        merged_segments = self._run_merging(
            vad_segments, audio, sample_rate, chunk_duration,
        )
        self._progress.update_stage(
            1, extra={"detail": f"片段合并: {len(vad_segments)} → {len(merged_segments)} 段"}
        )
        self._progress.finish_stage()

        # ---- Stage 3.5: 说话人分离 ----
        # 段级 diarization 已废弃，改用事件级聚类（见 Stage 5.1）。
        # 保留空 speaker_ids 使下游文本降级/碎片过滤正确跳过。
        speaker_ids: List[int] = []

        # ---- Stage 4: ASR 识别 ----
        self._progress.start_stage(
            "asr", description=f"{chunk_label}语音识别",
            total_items=len(merged_segments),
        )
        if run_asr:
            asr_results = self._run_asr(audio, sample_rate, merged_segments)
        else:
            asr_results = [[] for _ in merged_segments]
        self._progress.finish_stage()

        # ---- 文本降级说话人分离 ----
        # 段级 diarization 已废弃，文本降级不再需要。
        # 事件级聚类（Stage 5.1）在字幕粒度上做声学聚类，效果更好。

        # ---- 过滤超短内容片段（编号碎片如 "1." "2."） ----
        # 激进的预切分可能把编号/列表标记切成独立段（<3 个有效字符）。
        # 将它们合并到下一段，避免字幕中出现孤立的 "1." "2."
        #
        # ★ 说话人安全检查：仅当碎片与下一段属于同一说话人（或说话人
        #    信息不可用）时才合并。不同说话人的碎片保留为独立段，
        #    避免将说话人 A 的内容错标给说话人 B。
        import re
        _meaningful_pattern = re.compile(r'[A-Za-z一-鿿㐀-䶿]')
        if len(merged_segments) > 1 and len(asr_results) == len(merged_segments):
            filtered_segments = []
            filtered_asr = []
            filtered_speaker_ids = []
            for i in range(len(merged_segments)):
                seg = merged_segments[i]
                asr = asr_results[i]
                text = " ".join(ts.text for ts in asr).strip()
                meaningful = len(_meaningful_pattern.findall(text))

                # 检查是否可以安全合并：说话人相同或信息不可用
                can_merge = False
                if meaningful < 3 and i + 1 < len(merged_segments):
                    if speaker_ids and i < len(speaker_ids) and i + 1 < len(speaker_ids):
                        # 有说话人信息 → 仅当同一说话人时合并
                        if speaker_ids[i] == speaker_ids[i + 1]:
                            can_merge = True
                        else:
                            logger.debug(
                                "%sTiny fragment speaker mismatch: "
                                "'%.40s' (spk=%d) vs next (spk=%d) — keeping separate",
                                prefix, text, speaker_ids[i], speaker_ids[i + 1],
                            )
                    else:
                        # 无说话人信息 → 安全合并
                        can_merge = True

                if can_merge:
                    # 超短内容碎片：合并到下一段（原地修改，下一轮迭代正常处理）
                    next_seg = merged_segments[i + 1]
                    next_asr = asr_results[i + 1]
                    merged_segments[i + 1] = type(next_seg)(
                        start=seg.start,
                        end=next_seg.end,
                        confidence=next_seg.confidence,
                    )
                    asr_results[i + 1] = asr + next_asr
                    # speaker 继承下一段的值（同一说话人，无需修改）
                    logger.debug(
                        "%sFiltered tiny fragment: '%.40s' (%.2fs-%.2fs) → "
                        "merged into next segment (same speaker)",
                        prefix, text, seg.start, seg.end,
                    )
                    # 跳过当前段（不追加到 filtered），下一轮迭代处理合并后的段
                    continue

                filtered_segments.append(seg)
                filtered_asr.append(asr)
                if speaker_ids and i < len(speaker_ids):
                    filtered_speaker_ids.append(speaker_ids[i])

            if len(filtered_segments) < len(merged_segments):
                logger.info(
                    "%sFiltered %d tiny fragments (numbered-list artifacts)",
                    prefix, len(merged_segments) - len(filtered_segments),
                )
                merged_segments = filtered_segments
                asr_results = filtered_asr
                if filtered_speaker_ids:
                    speaker_ids = filtered_speaker_ids

        # ---- Stage 4.5: ASR 边界双向精修（方案四） ----
        if self.config.boundary_refinement.enabled:
            try:
                from ..asr.boundary_refiner import BoundaryRefiner

                self._progress.start_stage(
                    "boundary_refine", description=f"{chunk_label}边界精修",
                    total_items=len(merged_segments),
                )
                refiner = BoundaryRefiner(self.config.boundary_refinement)
                merged_segments, asr_results = refiner.refine_all(
                    merged_segments, asr_results, audio, sample_rate,
                )
                self._progress.update_stage(
                    len(merged_segments),
                    extra={"detail": f"已精修 {len(merged_segments)} 个片段边界"},
                )
                self._progress.finish_stage()
                ctx.add_diagnostic(
                    f"Boundary refine: {len(merged_segments)} segments refined"
                )
            except Exception as e:
                logger.warning("%sBoundary refinement failed: %s", prefix, e)
                ctx.add_diagnostic(f"Boundary refinement FAILED: {e}")

        # ---- Stage 4.6: 边界滑动窗口冗余识别（方案八） ----
        if self.config.boundary_redundancy.enabled and len(merged_segments) > 1:
            try:
                merged_segments, asr_results = self._run_boundary_redundancy(
                    merged_segments, asr_results, audio, sample_rate,
                    chunk_label=chunk_label,
                )
            except Exception as e:
                logger.warning(
                    "%sBoundary redundancy failed, continuing: %s", prefix, e,
                )
                ctx.add_diagnostic(f"Boundary redundancy FAILED: {e}")

        # ---- Stage 5: 时间轴映射 ----
        # 说话人信息在 _post_process_events 中通过事件级聚类统一注入，
        # 确保单块/多块/骨架三种路径都使用全局事件集合进行聚类。
        self._progress.start_stage(
            "mapping", description=f"{chunk_label}字幕生成", total_items=1,
        )
        events = self._run_mapping(
            asr_results, merged_segments,
            audio=audio, sample_rate=sample_rate,
            speaker_ids=None, role_names=None,
        )
        self._progress.update_stage(
            1, extra={"detail": f"生成 {len(events)} 条字幕"}
        )
        self._progress.finish_stage()

        return events, len(merged_segments), ctx

    # ------------------------------------------------------------------
    # 骨架分段独立处理模式
    # ------------------------------------------------------------------

    def _process_skeleton_segmented(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vocals_path: Path,
    ) -> Tuple[List[Any], int, Optional[Dict]]:
        """Compatibility adapter for the explicit segmented ASR service."""
        request = SegmentedASRRequest(
            audio=audio,
            sample_rate=sample_rate,
            vocals_path=vocals_path,
            segments=(),
        )
        ports = ASRRuntimePorts(
            config=self.config,
            get_engine=self._get_asr_engine,
            get_engine_for=self._get_asr_engine_for,
            get_language=self._resolved_language_or_config,
            set_language=lambda value: setattr(self, "_resolved_language", value),
            quality_gate_kwargs=self._quality_gate_kwargs,
            progress=self._progress,
            segmented_runner=lambda item: self._process_skeleton_segmented_legacy(
                audio=item.audio,
                sample_rate=item.sample_rate,
                vocals_path=item.vocals_path,
            ),
        )
        result = SegmentedASRService().run(request, ports)
        return result.events, result.segment_count, result.context

    def _process_skeleton_segmented_legacy(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vocals_path: Path,
    ) -> Tuple[List[Any], int, Optional[Dict]]:
        """按声学骨架分段，每段独立处理（骨架分段模式）。

        与 VAD 分段不同，此方法使用 ffmpeg silencedetect 的物理
        声学骨架作为分段依据。每个骨架语音段内部是物理连续的语音，
        按段独立处理后拼接，从根本上避免了跨段时间戳漂移。

        处理流程:
        1. 构建声学骨架（ffmpeg silencedetect）
        2. 每个骨架语音段 → 独立 _process_chunk_pipeline
        3. 调整时间戳到全局坐标
        4. 拼接所有事件

        Returns:
            (events: List[SubtitleEvent], total_segment_count: int, ffmpeg_result: Optional[Dict])
        """
        import tempfile

        from ..vad.ffmpeg_vad import unified_ffmpeg_pass

        cfg = self.config.acoustic_validation
        skeleton_noise_db = cfg.skeleton_noise_db
        skeleton_min_silence = cfg.skeleton_min_silence
        min_speech_duration = cfg.skeleton_min_speech

        # Step 1: 构建声学骨架
        self._progress.start_stage("skeleton", description="声学骨架提取")
        ffmpeg_result = unified_ffmpeg_pass(
            vocals_path,
            noise_db=skeleton_noise_db,
            min_silence_duration=skeleton_min_silence,
        )
        speech_skeleton = ffmpeg_result["skeleton"]  # [(start, end), ...]
        self._progress.finish_stage()

        total_duration = len(audio) / sample_rate
        logger.info(
            "Skeleton segmentation: %d speech segments from %.1fs audio "
            "(noise=%.0fdB, min_silence=%.2fs, min_speech=%.2fs)",
            len(speech_skeleton), total_duration,
            skeleton_noise_db, skeleton_min_silence, min_speech_duration,
        )

        if not speech_skeleton:
            logger.warning("No speech detected in skeleton, falling back to single chunk")
            events, seg_count, _fallback_ctx = self._process_chunk_pipeline(
                audio=audio, sample_rate=sample_rate,
                vocals_path=vocals_path, chunk_label="",
            )
            return events, seg_count, ffmpeg_result

        # 过滤过短的段（< min_speech_duration 的孤立爆发可能是噪音）
        filtered_skeleton = [
            (s, e) for s, e in speech_skeleton
            if (e - s) >= min_speech_duration
        ]

        if len(filtered_skeleton) < len(speech_skeleton):
            logger.info(
                "Filtered %d short segments (< %.2fs)",
                len(speech_skeleton) - len(filtered_skeleton),
                min_speech_duration,
            )
        speech_skeleton = filtered_skeleton

        # Step 2: 逐段独立处理
        all_events: List[Any] = []
        total_seg_count = 0

        # ★ 跨段说话人偏移量（同多块路径）：每个骨架段独立运行 diarization，
        # 从 0 开始编号。为防止不同段的 "说话人0" 混淆，累加偏移量。
        speaker_offset = 0
        empty_asr_segments = 0
        first_empty_asr_error: Optional[Exception] = None

        total_segments = len(speech_skeleton)
        for idx, (seg_start, seg_end) in enumerate(speech_skeleton):
            seg_duration = seg_end - seg_start
            start_sample = int(seg_start * sample_rate)
            end_sample = int(seg_end * sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            if end_sample <= start_sample:
                continue

            seg_audio = audio[start_sample:end_sample].copy()

            # 创建临时 WAV 文件供 ffmpeg 子进程调用
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False,
            ) as tmp_f:
                tmp_path = Path(tmp_f.name)

            try:
                AudioUtils.save_audio(seg_audio, tmp_path, sample_rate)

                chunk_label = f"Seg {idx+1}/{total_segments}"
                logger.info(
                    "Processing skeleton segment %d/%d: %.2fs → %.2fs (%.2fs)",
                    idx + 1, total_segments, seg_start, seg_end, seg_duration,
                )

                try:
                    seg_events, seg_count, _seg_ctx = self._process_chunk_pipeline(
                        audio=seg_audio,
                        sample_rate=sample_rate,
                        vocals_path=tmp_path,
                        chunk_label=chunk_label,
                        parallel_vad=False,  # 骨架分段嵌套线程，避免 PyTorch 死锁
                    )
                except ASRInvalidResultError as exc:
                    # A physical skeleton can contain a very short/noisy burst
                    # that VAD keeps but ASR cannot transcribe. Skip only this
                    # independent segment and continue with the rest.
                    empty_asr_segments += 1
                    if first_empty_asr_error is None:
                        first_empty_asr_error = exc
                    logger.warning(
                        "%sASR produced no usable subtitles; skipping skeleton segment: %s",
                        chunk_label,
                        exc,
                    )
                    continue
            finally:
                tmp_path.unlink(missing_ok=True)

            # ★ 跨段 speaker_id 偏移（同多块路径）
            seg_speakers = set()
            for evt in seg_events:
                if evt.speaker_id is not None:
                    seg_speakers.add(evt.speaker_id)
            if seg_speakers:
                max_spk = max(seg_speakers)
                if speaker_offset > 0:
                    for evt in seg_events:
                        if evt.speaker_id is not None:
                            evt.speaker_id += speaker_offset
                speaker_offset += max_spk + 1

            # 偏移到全局时间轴
            for evt in seg_events:
                evt.start += seg_start
                evt.end += seg_start

            total_seg_count += seg_count
            all_events.extend(seg_events)

        if speech_skeleton and not all_events and first_empty_asr_error is not None:
            raise ASRInvalidResultError(
                "ASR returned no usable subtitles for any of "
                f"{len(speech_skeleton)} skeleton speech segments "
                f"({empty_asr_segments} segment failures)"
            ) from first_empty_asr_error

        # Step 3: 按 start 排序
        all_events.sort(key=lambda e: e.start)

        # 重新编号
        for i, evt in enumerate(all_events):
            evt.index = i + 1

        logger.info(
            "Skeleton segmented: %d segments → %d events (%d VAD sub-segments)",
            len(speech_skeleton), len(all_events), total_seg_count,
        )

        return all_events, total_seg_count, ffmpeg_result

    # ------------------------------------------------------------------
    # 后处理公共方法（三种路径共用）
    # ------------------------------------------------------------------
