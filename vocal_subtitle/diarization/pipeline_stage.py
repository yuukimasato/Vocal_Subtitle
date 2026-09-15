"""Diarization and speaker enrichment stages for Pipeline."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..application.pipeline_result import PipelineStats
    from ..asr.base import TranscriptionSegment
    from ..mapping.time_mapper import SubtitleEvent
    from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


class PipelineDiarizationMixin:
    def _run_diarization(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: list[SpeechSegment],
        stats,
    ) -> list[int]:
        """Stage 3.5: 声学特征聚类 → 说话人分离"""
        diar_cfg = self.config.diarization

        try:
            from ..diarization.speaker_clusterer import SpeakerDiarizer
        except ImportError as e:
            logger.error("Diarization dependencies missing: %s", e)
            return []

        diarizer = SpeakerDiarizer(
            distance_threshold=diar_cfg.distance_threshold,
            min_speakers=diar_cfg.min_speakers,
            max_speakers=diar_cfg.max_speakers,
            use_pca=diar_cfg.use_pca,
            pca_variance=diar_cfg.pca_variance,
        )
        diarizer.load_model()

        speaker_ids = diarizer.diarize(segments, audio, sample_rate)
        stats.speaker_count = len(set(speaker_ids)) if speaker_ids else 0
        stats.diarization_silhouette = (
            diarizer.last_silhouette_ if hasattr(diarizer, "last_silhouette_") else None
        )
        logger.info("Diarization: %d speakers detected", len(set(speaker_ids)))
        return speaker_ids

    def _run_role_labeling(
        self,
        asr_results: list[list[TranscriptionSegment]],
        speaker_ids: list[int],
    ) -> dict[int, str]:
        """Stage 4.5: LLM 上下文分析 → 说话人角色命名"""
        role_cfg = self.config.speaker_role

        try:
            from ..diarization.role_labeler import RoleLabeler
        except ImportError as e:
            logger.error("Role labeler import failed: %s", e)
            return {}

        # 按说话人聚合对话文本
        speaker_texts: dict[int, list[str]] = defaultdict(list)
        for seg_result, spk_id in zip(asr_results, speaker_ids):
            text = " ".join(ts.text for ts in seg_result).strip()
            if text:
                speaker_texts[spk_id].append(text)

        if not speaker_texts:
            logger.warning("No transcribed text for role labeling")
            return {}

        labeler = RoleLabeler()
        role_names = labeler.label_roles(
            transcript_by_speaker=speaker_texts,
            model=role_cfg.model,
            base_url=role_cfg.base_url,
            api_key=role_cfg.api_key,
            temperature=role_cfg.temperature,
            context_hint=role_cfg.context_hint,
        )
        logger.info("Role labeling: %d speakers named", len(role_names))
        return role_names

    def _run_early_global_turns(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stats: PipelineStats,
    ) -> None:
        """[层1] 身份主干 P1:全局 diarization 前置（early_turns）。

        分离之后、chunk 处理之前对完整人声音频只跑一次全局 pass
        （复用 speaker_fusion 的引擎加载与模型选择逻辑，模型缓存沿用
        现有机制），turns 归一到全局时间轴后保存在
        ``self._early_turns_state``。pyannote 不可用时状态非 ok，
        后处理自动回退到事件级聚类（降级链 global→embedding→MFCC→
        unknown 不变），日志记录回退原因。
        """
        from ..diarization.early_turns import run_early_global_pass

        self._early_turns_state = run_early_global_pass(
            audio,
            sample_rate,
            self.config,
            duration=len(audio) / max(sample_rate, 1),
        )
        state = self._early_turns_state
        if state.attempted:
            stats.quality_diagnostics["early_turns"] = {
                "status": state.status,
                "turn_count": len(state.turns),
                "speaker_count": state.speaker_count,
                "model": state.model_ref,
                "single_speaker": state.single_speaker,
            }

    def _early_turns_active(self) -> bool:
        """[层1] 前置全局 turns 是否可用（early_turns 开启且全局 pass 成功）。"""
        state = getattr(self, "_early_turns_state", None)
        return bool(state is not None and state.active)

    def _get_embedding_engine(self):
        """获取说话人嵌入引擎（惰性初始化 + 缓存）

        Returns:
            SpeakerEmbeddingEngine 或 None（无法加载时保留 unknown）
        """
        if self._embedding_engine is not None:
            return self._embedding_engine

        emb_cfg = self.config.speaker_embedding
        if not emb_cfg.enabled:
            return None

        try:
            from ..diarization.speaker_embedding import (
                DummyEmbeddingEngine,
                create_embedding_engine,
            )

            engine = create_embedding_engine(emb_cfg)
            # Dummy 引擎表示加载失败，返回 None 以保留 unknown。
            if engine is None or isinstance(engine, DummyEmbeddingEngine):
                return None
            if engine.model_loaded:
                logger.info(
                    "Speaker embedding engine loaded: %s (dim=%d)",
                    engine.name,
                    engine.embedding_dim,
                )
                self._embedding_engine = engine
                return engine
        except Exception as e:
            logger.warning(
                "Failed to load speaker embedding engine: %s. "
                "Speaker identity will remain unknown unless global diarization is available.",
                e,
            )

        return None

    def _run_event_speaker_clustering(
        self,
        events: list[SubtitleEvent],
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[SubtitleEvent]:
        """事件级说话人聚类（替代段级 diarization）

        采用滑动窗口策略：将音频切成重叠的固定长度窗口（3s），
        在每个窗口上提取说话人特征（足够长的音频确保特征可靠），
        聚类窗口后将每个字幕事件分配给最佳匹配窗口的说话人。

        这解决了直接对短事件（0.5-2s）提取特征不可靠的问题。
        """
        if len(events) <= 1:
            for e in events:
                e.speaker_id = None
                e.speaker_label = None
                e.speaker_status = "unknown"
                e.speaker_source = "unknown"
            return events

        try:
            from ..diarization.speaker_clusterer import SpeakerDiarizer
        except ImportError as e:
            logger.error("Diarization dependencies missing: %s", e)
            return events

        # ---- Step 1: 滑动窗口特征提取 ----
        window_sec = 3.0
        hop_sec = 1.0
        audio_duration = len(audio) / sample_rate

        # 尝试加载说话人嵌入引擎（pyannote 等）。
        # 加载失败时不使用停顿或序号伪造身份。
        embedding_engine = self._get_embedding_engine()

        window_features = []
        window_times = []  # (start, end) per window

        t = 0.0
        while t + window_sec <= audio_duration:
            s = int(t * sample_rate)
            e = int((t + window_sec) * sample_rate)
            snippet = audio[s:e].astype(np.float32)

            if embedding_engine is not None:
                feats = embedding_engine.extract_embedding(snippet, sample_rate)
            else:
                feats = self._extract_pitch_energy_features_single(snippet, sample_rate)

            if feats is not None and np.any(feats):
                window_features.append(feats)
                window_times.append((t, t + window_sec))

            t += hop_sec

        if len(window_features) < 2:
            logger.warning("Too few windows for clustering")
            for ev in events:
                ev.speaker_id = None
                ev.speaker_label = None
                ev.speaker_status = "unknown"
                ev.speaker_source = "unknown"
            return events

        feature_matrix = np.vstack(window_features)
        logger.info(
            "Sliding window: %d windows (%.1fs each, hop=%.1fs)",
            len(window_features),
            window_sec,
            hop_sec,
        )

        # ---- Step 2: 窗口聚类 ----
        diar_cfg = self.config.diarization
        diarizer = SpeakerDiarizer(
            distance_threshold=diar_cfg.distance_threshold,
            min_speakers=max(2, diar_cfg.min_speakers),
            max_speakers=min(4, diar_cfg.max_speakers),
            use_pca=False,
        )
        diarizer.load_model()

        window_speaker_ids = diarizer._cluster(feature_matrix)
        silhouette = diarizer._evaluate_clustering(feature_matrix, window_speaker_ids)
        n_window_speakers = len(set(window_speaker_ids))

        # 单簇重试
        if n_window_speakers == 1 and len(window_features) >= 5:
            best, best_score = None, -1.0
            orig = diarizer.distance_threshold
            try:
                for thresh in [0.15, 0.20, 0.25, 0.30]:
                    diarizer.distance_threshold = thresh
                    alt = diarizer._cluster(feature_matrix)
                    alt_score = diarizer._evaluate_clustering(feature_matrix, alt)
                    alt_n = len(set(alt))
                    if 2 <= alt_n <= 4 and alt_score > best_score:
                        best, best_score = alt, alt_score
            finally:
                diarizer.distance_threshold = orig

            if best is not None and best_score > 0.03:
                window_speaker_ids = best
                silhouette = best_score
                n_window_speakers = len(set(best))
                logger.info(
                    "Window retry: %d speakers (silhouette=%.3f)",
                    n_window_speakers,
                    silhouette,
                )

        # 质量门控：聚类质量太差时保留 unknown，禁止伪造 speaker。
        if n_window_speakers < 2 or silhouette < 0.1:
            logger.warning(
                "Window clustering quality insufficient "
                "(silhouette=%.3f, %d speakers). Keeping speakers unknown.",
                silhouette,
                n_window_speakers,
            )
            for ev in events:
                ev.speaker_id = None
                ev.speaker_label = None
                ev.speaker_status = "unknown"
                ev.speaker_source = "unknown"
            return events

        logger.info(
            "Window clustering: %d windows → %d speakers (silhouette=%.3f)",
            len(window_features),
            n_window_speakers,
            silhouette,
        )

        # ---- Step 3: 事件→窗口→说话人映射 ----
        for evt in events:
            evt_mid = (evt.start + evt.end) / 2
            # 找到覆盖事件中点的窗口
            best_window = 0
            best_dist = float("inf")
            for wi, (ws, we) in enumerate(window_times):
                dist = abs(evt_mid - (ws + we) / 2)
                if dist < best_dist:
                    best_dist = dist
                    best_window = wi
            evt.speaker_id = int(window_speaker_ids[best_window])

        # 规范化 speaker_id（从 0 连续编号）
        unique_sorted = sorted(set(e.speaker_id for e in events))
        remap = {old: new for new, old in enumerate(unique_sorted)}
        for evt in events:
            evt.speaker_id = remap[evt.speaker_id]

        # 注入 speaker_label（根据语言国际化）
        lang = self._resolved_language_or_config()
        for evt in events:
            sid = evt.speaker_id
            evt.speaker_label = self._make_speaker_label(lang, sid)

        n_event_speakers = len(set(e.speaker_id for e in events))
        logger.info(
            "Event mapping: %d events → %d speakers",
            len(events),
            n_event_speakers,
        )

        return events

    @staticmethod
    def _extract_pitch_energy_features_single(
        snippet: np.ndarray, sample_rate: int
    ) -> np.ndarray | None:
        """从单个音频片段提取音高+能量特征（8 维）"""
        try:
            import librosa
        except ImportError:
            return None

        feats = []
        try:
            f0, voiced_flag, _ = librosa.pyin(
                snippet,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
            )
            f0_voiced = (
                f0[voiced_flag]
                if voiced_flag is not None and np.any(voiced_flag)
                else f0
            )
            f0_clean = f0_voiced[~np.isnan(f0_voiced)]
            if len(f0_clean) > 0:
                feats.extend(
                    [
                        float(np.mean(f0_clean)),
                        float(np.std(f0_clean)),
                        float(np.median(f0_clean)),
                    ]
                )
                feats.append(
                    float(np.sum(voiced_flag) / len(voiced_flag))
                    if voiced_flag is not None
                    else 0.0
                )
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])
        except Exception:
            feats.extend([0.0, 0.0, 0.0, 0.0])

        try:
            rms = librosa.feature.rms(y=snippet)
            feats.extend([float(np.mean(rms)), float(np.std(rms))])
        except Exception:
            feats.extend([0.0, 0.0])

        try:
            S = np.abs(librosa.stft(snippet, n_fft=2048, hop_length=512))  # noqa: N806 (librosa 惯例：S=幅度谱)
            centroid = librosa.feature.spectral_centroid(S=S, sr=sample_rate)
            feats.extend([float(np.mean(centroid)), float(np.std(centroid))])
        except Exception:
            feats.extend([0.0, 0.0])

        result = np.array(feats, dtype=np.float64)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _gap_based_speaker_assignment(
        self,
        events: list[SubtitleEvent],
    ) -> list[SubtitleEvent]:
        """纯间隙驱动的说话人交替（声学特征完全失效时的最后兜底）

        计算段间间隙的分布，使用中位数作为基准：
        - 间隙 >= 中位数 × 1.0 → 可能是说话人切换
        - 固定下界 0.08s，上界 0.5s
        - 支持多人交替：每次切换递增 speaker_id（而非仅二元交替），
          短间隙恢复上一说话人（回切检测）
        """
        if len(events) <= 1:
            lang = self._resolved_language_or_config()
            for e in events:
                e.speaker_id = 0
                e.speaker_label = self._make_speaker_label(lang, 0)
            return events

        import numpy as np

        gaps = []
        for i in range(len(events) - 1):
            gap = events[i + 1].start - events[i].end
            if gap > 0:
                gaps.append(gap)

        if not gaps:
            median_gap = 0.3
        else:
            median_gap = float(np.median(gaps))

        # 自适应阈值：中位数 × 1.0，限制在 [0.08, 0.50] 范围
        # 更低的乘数和下界，更好捕获快速多人对话中的切换
        switch_threshold = min(0.50, max(0.08, median_gap * 1.0))
        # 回切阈值：间隙 < 下界 → 恢复到上一说话人（快速交替）
        back_switch_threshold = max(0.03, median_gap * 0.3)
        logger.info(
            "Gap-based: median=%.3fs, switch_threshold=%.3fs, "
            "back_threshold=%.3fs, %d gaps",
            median_gap,
            switch_threshold,
            back_switch_threshold,
            len(gaps),
        )

        # 多人交替：每次切换递增 speaker_id
        # 短间隙 → 回切到上一说话人（A→B→A 模式）
        current_speaker = 0
        next_speaker = 1
        prev_speaker = None  # 用于回切检测
        speaker_stack = []  # 说话人栈，用于回切

        for i, evt in enumerate(events):
            evt.speaker_id = current_speaker
            if i < len(events) - 1:
                gap = events[i + 1].start - evt.end
                if gap >= switch_threshold:
                    # 明确切换 → 新说话人
                    if speaker_stack:
                        prev_speaker = speaker_stack.pop()
                    else:
                        speaker_stack.append(current_speaker)
                        prev_speaker = current_speaker
                    current_speaker = next_speaker
                    next_speaker += 1
                elif 0 < gap <= back_switch_threshold and prev_speaker is not None:
                    # 极短间隙 → 恢复到上一说话人（A-B-A 回切）
                    speaker_stack.append(current_speaker)
                    current_speaker = prev_speaker
                    prev_speaker = speaker_stack.pop() if speaker_stack else None

        lang = self._resolved_language_or_config()
        for evt in events:
            evt.speaker_label = self._make_speaker_label(lang, evt.speaker_id)

        n_speakers = len(set(e.speaker_id for e in events))
        logger.info(
            "Gap-based: %d events → %d speakers",
            len(events),
            n_speakers,
        )
        return events

    def _run_event_role_labeling(
        self,
        events: list[SubtitleEvent],
    ) -> list[SubtitleEvent]:
        """事件级 LLM 说话人角色标注

        从已聚类的 SubtitleEvent 按 speaker_id 聚合文本，
        调用 LLM 推断角色名称并更新 speaker_label。
        """
        try:
            from ..diarization.role_labeler import RoleLabeler
        except ImportError as e:
            logger.error("Role labeler import failed: %s", e)
            return events

        # 按说话人聚合文本
        speaker_texts: dict[int, list[str]] = defaultdict(list)
        for evt in events:
            if evt.speaker_id is not None and evt.text.strip():
                speaker_texts[evt.speaker_id].append(evt.text)

        if len(speaker_texts) < 2:
            logger.info("Event-level role labeling: only 1 speaker, skipping LLM call")
            return events

        role_cfg = self.config.speaker_role
        labeler = RoleLabeler()
        role_names = labeler.label_roles(
            transcript_by_speaker=speaker_texts,
            model=role_cfg.model,
            base_url=role_cfg.base_url,
            api_key=role_cfg.api_key,
            temperature=role_cfg.temperature,
            context_hint=role_cfg.context_hint,
        )

        if not role_names:
            return events

        # 应用角色名称
        for evt in events:
            if evt.speaker_id is not None and evt.speaker_id in role_names:
                evt.speaker_label = role_names[evt.speaker_id]

        logger.info(
            "Event-level role labeling: %d speakers named",
            len(role_names),
        )
        return events

    def _run_boundary_redundancy(
        self,
        segments: list[SpeechSegment],
        asr_results: list[list[TranscriptionSegment]],
        audio: np.ndarray,
        sample_rate: int,
        chunk_label: str = "",
    ) -> tuple[list[SpeechSegment], list[list[TranscriptionSegment]]]:
        """Stage 4.6: 边界滑动窗口冗余识别

        对低置信度边界执行偏移窗口重 ASR + LLM 语义仲裁，
        修正因语速快、词间粘连导致的边界分词错误。

        流程:
        1. BoundaryConfidenceEstimator 评估所有边界
        2. 对低分边界 → SlidingWindowReASR 创建重叠窗口重新识别
        3. BoundaryArbitrator 用 LLM/规则决定词归属
        4. 应用仲裁结果到 segments 和 asr_results
        """
        from ..asr.boundary_arbitration import (
            ArbitrationConfig,
            BoundaryArbitrator,
            apply_arbitration_results,
        )
        from ..asr.boundary_confidence import (
            BoundaryConfidenceEstimator,
            BoundaryRedundancyConfig,
        )
        from ..asr.boundary_reasr import (
            SlidingWindowConfig,
            SlidingWindowReASR,
        )

        cfg = self.config.boundary_redundancy
        progress = self._progress  # may be None in tests

        # Step 1: 评估边界置信度
        prefix = f"[{chunk_label}] " if chunk_label else ""
        if progress:
            progress.start_stage(
                "boundary_confidence",
                description=f"{chunk_label}边界置信度评估",
                total_items=len(segments) - 1,
            )

        estimator_cfg = BoundaryRedundancyConfig(
            enabled=True,
            min_gap_trigger=cfg.min_gap_trigger,
            max_energy_slope_trigger=cfg.max_energy_slope_trigger,
            confidence_threshold=cfg.confidence_threshold,
        )
        estimator = BoundaryConfidenceEstimator(estimator_cfg)

        boundaries = estimator.evaluate_all(
            segments,
            asr_results,
            audio,
            sample_rate,
        )
        low_conf_indices = estimator.get_low_confidence_boundaries(boundaries)

        if progress:
            progress.update_stage(
                len(boundaries),
                extra={
                    "detail": f"{len(low_conf_indices)}/{len(boundaries)} 边界需冗余"
                },
            )
            progress.finish_stage()

        if not low_conf_indices:
            logger.info("%sAll boundaries clear — skipping redundancy", prefix)
            return segments, asr_results

        # Step 2: 滑动窗口重 ASR
        if progress:
            progress.start_stage(
                "boundary_reasr",
                description=f"{chunk_label}滑动窗口重识别",
                total_items=len(low_conf_indices),
            )

        window_cfg = SlidingWindowConfig(
            base_overlap_ms=cfg.base_overlap_ms,
            fast_speech_wps=cfg.fast_speech_wps,
            fast_overlap_ms=cfg.fast_overlap_ms,
            very_fast_overlap_ms=cfg.very_fast_overlap_ms,
            fusion_window_sec=cfg.fusion_window_sec,
            max_workers=cfg.max_workers,
        )

        # ★ 边界窗口 ASR 也需要准确的 language 参数。
        # 窗口音频极短（500-1000ms），自动检测几乎必定失败。
        # 如果用户未锁定语言，从完整音频中做一次全局检测。
        boundary_language: str | None = self.config.asr.language
        if boundary_language is None:
            asr_engine = self._get_asr_engine()
            asr_engine.load_model()
            detector = getattr(asr_engine, "detect_language", None)
            if callable(detector):
                boundary_language = detector(audio, sample_rate)
            else:
                detect = getattr(asr_engine, "detect_language_info", None)
                if callable(detect):
                    lang_info = detect(audio, sample_rate)
                    boundary_language = (
                        getattr(lang_info, "language", None) or lang_info
                    )
            if boundary_language:
                logger.info(
                    "Boundary re-ASR: using detected language=%s",
                    boundary_language,
                )
            else:
                logger.warning(
                    "Boundary re-ASR: language detection failed, "
                    "short-window auto-detection may be unreliable"
                )

        reasr = SlidingWindowReASR(
            config=window_cfg,
            asr_engine=self._get_asr_engine(),
            cache=self._get_cache() if self.config.cache.enabled else None,
            language=boundary_language,
        )
        reasr_results = reasr.process_boundaries(
            low_conf_indices,
            segments,
            asr_results,
            audio,
            sample_rate,
            total_duration=len(audio) / sample_rate,
        )
        if progress:
            progress.update_stage(
                len(low_conf_indices),
                extra={"detail": f"完成 {len(reasr_results)} 个边界冗余"},
            )
            progress.finish_stage()

        if not reasr_results:
            logger.info("%sRe-ASR produced no usable results", prefix)
            return segments, asr_results

        # Step 3: LLM 语义仲裁
        if progress:
            progress.start_stage(
                "boundary_arbitration",
                description=f"{chunk_label}语义仲裁",
                total_items=len(reasr_results),
            )

        arb_cfg = ArbitrationConfig(
            llm_model=cfg.llm_model,
            llm_base_url=cfg.llm_base_url,
            llm_api_key=cfg.llm_api_key,
            llm_temperature=cfg.llm_temperature,
            llm_timeout=cfg.llm_timeout,
            auto_apply_confidence=cfg.auto_apply_confidence,
            review_threshold=cfg.review_threshold,
            fallback_to_rules=cfg.fallback_to_rules,
        )
        arb = BoundaryArbitrator(arb_cfg)

        arbitration_results = {}
        for idx, reasr_result in reasr_results.items():
            # 构建上下文（前后各取 2 个段的文本）
            left_ctx = ""
            right_ctx = ""
            for j in range(max(0, idx - 2), idx):
                text = " ".join(ts.text for ts in asr_results[j]).strip()
                if text:
                    left_ctx += text + " "
            for j in range(idx + 1, min(len(asr_results), idx + 3)):
                text = " ".join(ts.text for ts in asr_results[j]).strip()
                if text:
                    right_ctx += text + " "

            left_end = segments[idx].end
            right_start = segments[idx + 1].start

            result = arb.arbitrate(
                reasr_result,
                left_context=left_ctx.strip(),
                right_context=right_ctx.strip(),
                left_seg_end=left_end,
                right_seg_start=right_start,
            )
            arbitration_results[idx] = result

        if progress:
            progress.update_stage(
                len(arbitration_results),
                extra={
                    "detail": (
                        f"自动应用 {sum(1 for a in arbitration_results.values() if a.auto_applied)}, "
                        f"待复核 {sum(1 for a in arbitration_results.values() if a.needs_review)}"
                    ),
                },
            )
            progress.finish_stage()

        # Step 4: 应用仲裁结果
        asr_results, segments = apply_arbitration_results(
            arbitration_results,
            asr_results,
            segments,
        )

        return segments, asr_results
