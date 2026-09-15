"""Chunk and skeleton-mode execution stages."""

from __future__ import annotations  # noqa: I001

import logging
from pathlib import Path
from typing import Any

import numpy as np

# 导入顺序敏感（勿重排）：asr 包必须先于 acoustic 初始化——asr/__init__ 经
# global_transcriber → physical → decision_projection 反向依赖 mapping.time_mapper，
# 若 acoustic/merging 链先触发 mapping 包初始化，time_mapper 尚未定义 SubtitleEvent
# 即被 physical 引用，产生部分初始化 ImportError。
# noqa: I001
from ..asr.base import ASRInvalidResultError
from ..asr.contracts import ASRRuntimePorts, SegmentedASRRequest
from ..asr.segmented_path import SegmentedASRService
from ..acoustic.skeleton import adaptive_silence_threshold_db, group_speech_intervals
from ..diarization.early_turns import (
    dominant_span_speaker,
    dominant_speaker_at,
    spans_from_skeleton,
)
from ..pipeline_context import NoiseProfile, PipelineContext
from ..utils.audio_utils import AudioUtils
from .member_projection import reproject_events_to_members

logger = logging.getLogger(__name__)
ASR_CONTEXT_PADDING_SECONDS = 0.25


class PipelineChunkMixin:
    def _resolve_active_modules(self) -> dict[str, bool]:
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
                streaming_modules.update(
                    {
                        "ffmpeg_vad": False,
                        "pre_split": False,
                        "asr_refine": False,
                        "llm_merge": False,
                        "frame_seamless": False,
                        "diarization": False,
                        "speaker_role": False,
                    }
                )
            elif self.config.degradation.mode == "degraded":
                # degraded 叠加：关闭 LLM 相关
                streaming_modules.update(
                    {
                        "llm_merge": False,
                        "speaker_role": False,
                        "llm_optimize": False,
                    }
                )

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

        # mode == "full": 全部按配置启用，并叠加实验注册表中已启用的实验
        active = {
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

        # ---- 查询实验注册表：已启用的实验自动激活对应配置 ----
        self._apply_enabled_experiments(active)

        return active

    @staticmethod
    def _apply_enabled_experiments(active: dict[str, bool]) -> None:
        """查询 ExperimentRegistry，将已启用的实验映射到配置开关。

        每个 status=enabled 的实验对应一组配置覆盖。
        此方法在 full 降级模式之外独立存在，允许 degraded/minimal
        模式也受益于实验注册。
        """
        try:
            from ..governance.experiment_registry import (
                ExperimentRegistry,
                ExperimentStatus,
            )

            registry = ExperimentRegistry()
            enabled_exps = registry.list_by_status(ExperimentStatus.ENABLED.value)

            for exp in enabled_exps:
                exp_id = exp.experiment_id
                logger.debug("Applying enabled experiment: %s (%s)", exp_id, exp.name)

                if exp_id == "exp-20260802-qwen-review":
                    # Qwen3-ASR 复核引擎
                    if hasattr(active, "__setitem__"):
                        pass  # active is dict, no extra module flags needed
                    logger.info("Experiment %s active: Qwen review enabled", exp_id)
                elif exp_id == "exp-20260802-forced-aligner":
                    logger.info("Experiment %s active: ForcedAligner enabled", exp_id)
                elif exp_id == "exp-20260802-sed-non-speech":
                    logger.info(
                        "Experiment %s active: SED non-speech detection enabled", exp_id
                    )
                elif exp_id == "exp-20260802-vad-fusion":
                    if "fusion" in active:
                        active["fusion"] = True
                    logger.info("Experiment %s active: VAD fusion enabled", exp_id)
                elif exp_id == "exp-20260802-llm-optimize":
                    if "llm_optimize" in active:
                        active["llm_optimize"] = True
                    logger.info("Experiment %s active: LLM optimize enabled", exp_id)
                elif exp_id == "exp-20260802-noise-reduction":
                    logger.info("Experiment %s active: Noise reduction enabled", exp_id)
        except Exception:
            pass  # 非致命操作

    # ------------------------------------------------------------------
    # [层1] 说话人身份主干（early_turns，2026-09-11 定案）
    # ------------------------------------------------------------------

    def _attach_early_turns_context(
        self, ctx: PipelineContext, time_offset: float
    ) -> None:
        """把前置全局 turns / 骨架×turns 跨度注入窗口 ctx。

        early_turns 未生效（关闭或全局 pass 失败）时不做任何事，
        ctx 字段保持为空，全链路与现状一致。
        """
        state = getattr(self, "_early_turns_state", None)
        if state is None or not state.active:
            return
        ctx.early_turns = list(state.turns)
        ctx.early_turn_spans = list(getattr(self, "_early_turn_spans", []) or [])
        ctx.early_turns_window_offset = float(time_offset or 0.0)
        ctx.add_diagnostic(
            f"Early turns: status={state.status}, "
            f"global_turn_count={len(ctx.early_turns)}, "
            f"span_count={len(ctx.early_turn_spans)}, "
            f"window_offset={ctx.early_turns_window_offset:.2f}s"
        )

    def _early_speaker_ids_for_segments(
        self,
        segments: list[Any],
        ctx: PipelineContext,
        duration: float,
    ) -> list[int | None]:
        """从 ctx 的全局 turns/spans 推导每个 ASR 段的说话人。

        - 有骨架×turns 跨度（spans）时优先按跨度取主覆盖身份；
        - 否则按全局 turns 直接求主覆盖说话人（turns/spans 均为全局
          时间轴坐标，段坐标加 ``early_turns_window_offset`` 还原）；
        - 无覆盖返回 None（无信息，合并检查按"安全合并"处理）；
        - early_turns 未生效时返回空列表 → 行为与现状一致。
        """
        if not getattr(ctx, "early_turns", None):
            return []
        offset = float(getattr(ctx, "early_turns_window_offset", 0.0) or 0.0)
        spans = list(getattr(ctx, "early_turn_spans", []) or [])
        speaker_ids: list[int | None] = []
        for seg in segments:
            start = float(seg.start) + offset
            end = float(seg.end) + offset
            if spans:
                speaker_ids.append(dominant_span_speaker(spans, start, end))
            else:
                speaker_ids.append(dominant_speaker_at(ctx.early_turns, start, end))
        return speaker_ids

    def _filter_tiny_fragments(
        self,
        merged_segments: list[Any],
        asr_results: list[Any],
        speaker_ids: list[int | None],
        prefix: str = "",
    ) -> tuple[list[Any], list[Any], list[int | None]]:
        """过滤超短内容片段（编号碎片如 "1." "2."），合并到下一段。

        ★ 说话人安全检查：仅当碎片与下一段属于同一说话人（或说话人
        信息不可用）时才合并。early_turns 生效时 speaker_ids 来自全局
        turns/spans，"不同说话人的碎片保留为独立段"分支真正生效，
        避免将说话人 A 的内容错标给说话人 B；speaker_ids 为空时保持
        "无信息 → 安全合并"的现状行为。
        """
        import re

        _meaningful_pattern = re.compile(r"[A-Za-z一-鿿㐀-䶿]")
        if len(merged_segments) <= 1 or len(asr_results) != len(merged_segments):
            return merged_segments, asr_results, speaker_ids
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
                    # （None 表示 turn 未覆盖：None == 已知为 False → 保守
                    #   不合并；None == None → 无冲突可合并）
                    if speaker_ids[i] == speaker_ids[i + 1]:
                        can_merge = True
                    else:
                        logger.debug(
                            "%sTiny fragment speaker mismatch: "
                            "'%.40s' (spk=%s) vs next (spk=%s) — keeping separate",
                            prefix,
                            text,
                            speaker_ids[i],
                            speaker_ids[i + 1],
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
                    prefix,
                    text,
                    seg.start,
                    seg.end,
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
                prefix,
                len(merged_segments) - len(filtered_segments),
            )
            return (
                filtered_segments,
                filtered_asr,
                filtered_speaker_ids if filtered_speaker_ids else speaker_ids,
            )
        return merged_segments, asr_results, speaker_ids

    def _process_chunk_pipeline(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vocals_path: Path,
        chunk_label: str = "",
        parallel_vad: bool = True,
        run_asr: bool = True,
        time_offset: float = 0.0,
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
            time_offset: 本块在全局时间轴中的起点（秒）。[层1] early_turns
                         生效时用于把全局 turns/spans 对齐到本块局部时间轴。

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
        # ---- [层1] early_turns:身份主干上下文注入 ----
        self._attach_early_turns_context(ctx, time_offset)

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
                logger.warning(
                    "%sDenoise failed, continuing with original: %s", prefix, e
                )
                ctx.add_diagnostic(f"Denoise FAILED: {e}")

        # ---- 环境底噪自适应采样 ----
        noise_profile = AudioUtils.estimate_noise_floor_per_chunk(
            audio,
            sample_rate,
            chunk_duration=chunk_duration,
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
                        self._run_vad,
                        audio,
                        sample_rate,
                    )
                    future_ffmpeg = executor.submit(
                        self._run_ffmpeg_vad,
                        vocals_path,
                        ctx,
                        prefix,
                    )
                    vad_segments = future_silero.result()
                    ffmpeg_result = future_ffmpeg.result()
            else:
                # 串行执行（骨架分段 / 多块 / 流式路径）
                # 避免 ThreadPoolExecutor 嵌套带来的 PyTorch 线程死锁
                vad_segments = self._run_vad(audio, sample_rate)
                ffmpeg_result = self._run_ffmpeg_vad(
                    vocals_path,
                    ctx,
                    prefix,
                )

            # 三方法融合（如果启用，逻辑不变）
            if ffmpeg_result is not None and self.config.fusion.enabled:
                from ..vad.boundary_fusion import BoundaryFusion

                fusion_engine = BoundaryFusion(self.config.fusion)
                ffmpeg_segments = ffmpeg_result.get("coarse_speech", [])
                vad_segments = fusion_engine.fuse(
                    vad_segments,
                    ffmpeg_segments,
                    audio,
                    sample_rate,
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
            "merging",
            description=f"{chunk_label}片段合并",
            total_items=1,
        )
        merged_segments = self._run_merging(
            vad_segments,
            audio,
            sample_rate,
            chunk_duration,
        )
        self._progress.update_stage(
            1,
            extra={
                "detail": f"片段合并: {len(vad_segments)} → {len(merged_segments)} 段"
            },
        )
        self._progress.finish_stage()

        # ---- Stage 3.5: 说话人分离 ----
        # 段级 diarization 已废弃。speaker_ids 现有两个来源：
        # 1) [层1] early_turns 生效时，从 ctx 的全局 turns/spans 推导
        #    （tiny-fragment 合并的说话人安全检查据此生效）；
        # 2) 否则保持空列表，下游碎片过滤按"无信息 → 安全合并"跳过，
        #    说话人标签由后处理统一注入（现状行为）。
        speaker_ids: list[int | None] = self._early_speaker_ids_for_segments(
            merged_segments,
            ctx,
            chunk_duration,
        )

        # ---- Stage 4: ASR 识别 ----
        self._progress.start_stage(
            "asr",
            description=f"{chunk_label}语音识别",
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
        # （说话人安全检查见 _filter_tiny_fragments）。
        merged_segments, asr_results, speaker_ids = self._filter_tiny_fragments(
            merged_segments,
            asr_results,
            speaker_ids,
            prefix=prefix,
        )

        # ---- Stage 4.5: ASR 边界双向精修（方案四） ----
        if self.config.boundary_refinement.enabled:
            try:
                from ..asr.boundary_refiner import BoundaryRefiner

                self._progress.start_stage(
                    "boundary_refine",
                    description=f"{chunk_label}边界精修",
                    total_items=len(merged_segments),
                )
                refiner = BoundaryRefiner(self.config.boundary_refinement)
                merged_segments, asr_results = refiner.refine_all(
                    merged_segments,
                    asr_results,
                    audio,
                    sample_rate,
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
                    merged_segments,
                    asr_results,
                    audio,
                    sample_rate,
                    chunk_label=chunk_label,
                )
            except Exception as e:
                logger.warning(
                    "%sBoundary redundancy failed, continuing: %s",
                    prefix,
                    e,
                )
                ctx.add_diagnostic(f"Boundary redundancy FAILED: {e}")

        # ---- Stage 5: 时间轴映射 ----
        # 说话人信息在 _post_process_events 中通过事件级聚类统一注入，
        # 确保单块/多块/骨架三种路径都使用全局事件集合进行聚类。
        self._progress.start_stage(
            "mapping",
            description=f"{chunk_label}字幕生成",
            total_items=1,
        )
        events = self._run_mapping(
            asr_results,
            merged_segments,
            audio=audio,
            sample_rate=sample_rate,
            speaker_ids=None,
            role_names=None,
        )
        self._progress.update_stage(1, extra={"detail": f"生成 {len(events)} 条字幕"})
        self._progress.finish_stage()

        from ..asr.trace_contract import attach_event_trace

        source_id = "chunk" if not chunk_label else f"chunk:{chunk_label}"
        for event_index, event in enumerate(events, start=1):
            attach_event_trace(
                event,
                source_id=source_id,
                offset_id=f"chunk-offset:{event_index:06d}",
                window_id=f"chunk-window:{chunk_label or 'single'}",
            )

        return events, len(merged_segments), ctx

    # ------------------------------------------------------------------
    # 骨架分段独立处理模式
    # ------------------------------------------------------------------

    def _process_skeleton_segmented(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vocals_path: Path,
    ) -> tuple[list[Any], int, dict | None]:
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
    ) -> tuple[list[Any], int, dict | None]:
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
        skeleton_noise_db = adaptive_silence_threshold_db(
            audio,
            sample_rate,
            enabled=cfg.skeleton_adaptive_noise_db,
            fallback_db=cfg.skeleton_noise_db,
            margin_db=cfg.skeleton_noise_margin_db,
        )
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
        if getattr(self, "_global_review_timeline", None) is None:
            review_context = PipelineContext(
                audio_path=vocals_path,
                audio=audio,
                sample_rate=sample_rate,
            )
            review_context.ffmpeg_unified_result = ffmpeg_result
            self._global_review_timeline = self._build_review_timeline_from_context(
                review_context, total_duration
            )
        logger.info(
            "Skeleton segmentation: %d speech segments from %.1fs audio "
            "(noise=%.0fdB, min_silence=%.2fs, min_speech=%.2fs)",
            len(speech_skeleton),
            total_duration,
            skeleton_noise_db,
            skeleton_min_silence,
            min_speech_duration,
        )

        if not speech_skeleton:
            logger.warning(
                "No speech detected in skeleton, falling back to single chunk"
            )
            events, seg_count, _fallback_ctx = self._process_chunk_pipeline(
                audio=audio,
                sample_rate=sample_rate,
                vocals_path=vocals_path,
                chunk_label="",
            )
            return events, seg_count, ffmpeg_result

        # 过滤过短的段（< min_speech_duration 的孤立爆发可能是噪音）
        filtered_skeleton = [
            (s, e) for s, e in speech_skeleton if (e - s) >= min_speech_duration
        ]

        if len(filtered_skeleton) < len(speech_skeleton):
            logger.info(
                "Filtered %d short segments (< %.2fs)",
                len(speech_skeleton) - len(filtered_skeleton),
                min_speech_duration,
            )
        speech_skeleton = filtered_skeleton

        # Preserve the original physical skeleton, but provide enough context
        # for ASR when ordinary pauses split one short utterance into tiny
        # windows. Hard silence remains a strict ASR-window boundary.
        asr_skeleton = group_speech_intervals(speech_skeleton)
        if len(asr_skeleton) != len(speech_skeleton):
            logger.info(
                "Grouped %d physical skeleton segments into %d ASR windows",
                len(speech_skeleton),
                len(asr_skeleton),
            )

        # ---- [层1] 骨架区间 × 全局 turns 求交（reconcile_regions 接线） ----
        # 物理骨架只产时间区间；speaker identity 一律来自前置全局 turns。
        # 相邻且同 speaker 的跨度合并，不同 speaker 永不合并；结果存入
        # 窗口 ctx（_attach_early_turns_context），供 tiny-fragment 硬约束
        # 与后续时间轴仲裁层（P3）复用。
        if self._early_turns_active():
            self._early_turn_spans = spans_from_skeleton(
                speech_skeleton,
                self._early_turns_state.turns,
                duration=total_duration,
            )
            logger.info(
                "Early turns reconcile: %d physical segments × %d global turns "
                "→ %d speaker spans",
                len(speech_skeleton),
                len(self._early_turns_state.turns),
                len(self._early_turn_spans),
            )
        else:
            self._early_turn_spans = []

        # Step 2: 先用带上下文的 ASR 窗口处理；若一个聚合窗口失败，
        # 只回退该窗口包含的原始物理段，避免扩大失败范围或跨硬静音重试。
        all_events: list[Any] = []
        total_seg_count = 0

        # ★ 跨段说话人偏移量（同多块路径）：每个骨架段独立运行 diarization，
        # 从 0 开始编号。为防止不同段的 "说话人0" 混淆，累加偏移量。
        # [层1] early_turns 生效时标签来自全局 turns（全局唯一），无需偏移。
        speaker_offset = 0
        empty_asr_segments = 0
        first_empty_asr_error: Exception | None = None
        grouped_window_fallbacks = 0

        total_segments = len(asr_skeleton)
        for idx, (window_start, window_end) in enumerate(asr_skeleton):
            window_members = [
                item
                for item in speech_skeleton
                if item[0] >= window_start - 1e-9 and item[1] <= window_end + 1e-9
            ]
            # Give grouped windows limited context to compensate for
            # silencedetect clipping low-energy word edges. Single physical
            # windows retain their exact bounds, preserving the legacy timing
            # contract. Use the midpoint of a hard-silence gap as the limit
            # between neighboring grouped windows.
            input_start = window_start
            input_end = window_end
            if len(window_members) > 1:
                input_start = max(
                    0.0,
                    window_start - ASR_CONTEXT_PADDING_SECONDS,
                )
                input_end = min(
                    total_duration,
                    window_end + ASR_CONTEXT_PADDING_SECONDS,
                )
                if idx > 0:
                    previous_end = asr_skeleton[idx - 1][1]
                    input_start = max(
                        input_start,
                        (previous_end + window_start) / 2.0,
                    )
                if idx + 1 < total_segments:
                    next_start = asr_skeleton[idx + 1][0]
                    input_end = min(
                        input_end,
                        (window_end + next_start) / 2.0,
                    )

            attempts = [(input_start, input_end, f"skeleton:{idx}")]
            if len(window_members) > 1:
                attempts.extend(
                    (
                        member_start,
                        member_end,
                        f"skeleton:{idx}:fallback:{member_idx}",
                    )
                    for member_idx, (member_start, member_end) in enumerate(
                        window_members
                    )
                )

            window_succeeded = False
            for attempt_index, (seg_start, seg_end, event_source) in enumerate(
                attempts
            ):
                seg_duration = seg_end - seg_start
                start_sample = max(0, int(seg_start * sample_rate))
                end_sample = min(len(audio), int(seg_end * sample_rate))

                if end_sample <= start_sample:
                    continue

                seg_audio = audio[start_sample:end_sample].copy()

                # 创建临时 WAV 文件供 ffmpeg 子进程调用
                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=False,
                ) as tmp_f:
                    tmp_path = Path(tmp_f.name)

                try:
                    AudioUtils.save_audio(seg_audio, tmp_path, sample_rate)

                    if attempt_index == 0:
                        chunk_label = f"Seg {idx + 1}/{total_segments}"
                    else:
                        chunk_label = (
                            f"Seg {idx + 1}/{total_segments} fallback "
                            f"{attempt_index}/{len(attempts) - 1}"
                        )
                    logger.info(
                        "Processing skeleton segment %d/%d: %.2fs → %.2fs (%.2fs)%s",
                        idx + 1,
                        total_segments,
                        seg_start,
                        seg_end,
                        seg_duration,
                        " [physical fallback]" if attempt_index else "",
                    )

                    try:
                        seg_events, seg_count, _seg_ctx = self._process_chunk_pipeline(
                            audio=seg_audio,
                            sample_rate=sample_rate,
                            vocals_path=tmp_path,
                            chunk_label=chunk_label,
                            parallel_vad=False,  # 骨架分段嵌套线程，避免 PyTorch 死锁
                            time_offset=seg_start,
                        )
                    except ASRInvalidResultError as exc:
                        empty_asr_segments += 1
                        if first_empty_asr_error is None:
                            first_empty_asr_error = exc
                        logger.warning(
                            "%sASR produced no usable subtitles; trying next bounded window: %s",
                            chunk_label,
                            exc,
                        )
                        continue
                finally:
                    tmp_path.unlink(missing_ok=True)

                # 聚合窗口的 ASR 上下文不能改变物理时间契约：把结果投影
                # 回原始物理成员段（在成员静音间隙处按词拆分、端点钳制到
                # 骨架边界），恢复 v0.2.0 逐段切片的端点静音对齐精度。
                if (
                    attempt_index == 0
                    and len(window_members) > 1
                    and self.config.acoustic_validation.reproject_grouped_windows
                ):
                    local_members = [
                        (member_start - seg_start, member_end - seg_start)
                        for member_start, member_end in window_members
                    ]
                    seg_events, projection_stats = reproject_events_to_members(
                        seg_events,
                        local_members,
                        split_min_gap=self.config.acoustic_validation.member_split_min_gap,
                        max_duration=self.config.acoustic_validation.member_split_max_duration,
                    )
                    if any(
                        (
                            projection_stats.split_events,
                            projection_stats.clamped_events,
                            projection_stats.dropped_events,
                        )
                    ):
                        logger.info(
                            "%s member reprojection: %s",
                            chunk_label,
                            projection_stats.as_dict(),
                        )
                    if not seg_events:
                        logger.warning(
                            "%s all events dropped by member reprojection; "
                            "retrying with physical member slices",
                            chunk_label,
                        )
                        continue

                window_succeeded = True
                if attempt_index > 0:
                    grouped_window_fallbacks += 1

                # ★ 跨段 speaker_id 偏移（同多块路径；early_turns 生效时跳过，
                # 标签来自全局 turns 无需偏移）
                if not self._early_turns_active():
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

                # 偏移到全局时间轴；物理范围和来源追踪必须同步偏移。
                from ..mapping.time_mapper import offset_subtitle_event

                for evt in seg_events:
                    offset_subtitle_event(
                        evt,
                        seg_start,
                        source=event_source,
                    )
                    from ..asr.trace_contract import attach_event_trace

                    attach_event_trace(
                        evt,
                        source_id="skeleton",
                        offset_id=event_source,
                        window_id=f"skeleton-window:{idx:06d}",
                    )

                total_seg_count += seg_count
                all_events.extend(seg_events)

                # 聚合成功后不再执行其成员段；聚合失败时必须继续遍历
                # 所有物理成员，避免第一个成员成功掩盖后续成员漏检。
                if attempt_index == 0:
                    break

            if not window_succeeded:
                logger.warning(
                    "Skeleton ASR window %d/%d failed for all %d bounded attempts",
                    idx + 1,
                    total_segments,
                    len(attempts),
                )

        if asr_skeleton and not all_events and first_empty_asr_error is not None:
            raise ASRInvalidResultError(
                "ASR returned no usable subtitles for any of "
                f"{len(asr_skeleton)} skeleton speech segments "
                "(ASR windows) "
                f"({empty_asr_segments} segment failures)"
            ) from first_empty_asr_error

        # Step 3: 按 start 排序
        all_events.sort(key=lambda e: e.start)

        # 重新编号
        for i, evt in enumerate(all_events):
            evt.index = i + 1

        logger.info(
            "Skeleton segmented: %d ASR windows (%d physical fallbacks) → "
            "%d events (%d VAD sub-segments)",
            len(asr_skeleton),
            grouped_window_fallbacks,
            len(all_events),
            total_seg_count,
        )

        return all_events, total_seg_count, ffmpeg_result

    # ------------------------------------------------------------------
    # 后处理公共方法（三种路径共用）
    # ------------------------------------------------------------------
