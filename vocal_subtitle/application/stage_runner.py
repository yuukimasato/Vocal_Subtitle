"""Separation, VAD, segment merge and segmented ASR pipeline stages."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from ..asr.base import ASRInvalidResultError, TranscriptionSegment
from ..merging.merge_strategy import MergeConfig, MergeStrategy
from ..separation.base import SeparationResult
from ..utils.audio_utils import AudioUtils
from ..utils.file_hasher import compute_file_hash
from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


class PipelineStageMixin:
    def _run_separation(
        self, input_path: Path, progress_callback: callable | None = None
    ) -> SeparationResult:
        """Stage 1: 执行人声分离（支持文件内容哈希缓存和持久化存储）

        同时缓存人声和伴奏（背景声），供前端导出下载。

        Args:
            input_path: 输入文件路径
            progress_callback: 分离进度回调 (current: int, total: int) -> None
        """
        sep_cfg = self.config.separation

        # 使用文件内容哈希构建缓存键（而非仅用路径）
        file_hash = self._file_hash or compute_file_hash(input_path)
        cache_key = file_hash[:16]  # 使用哈希前 16 位作为短键

        # 检查持久化文件缓存
        if self.config.cache.enabled:
            cache = self._get_cache()
            vocals_cache_key = (
                f"{file_hash}:{sep_cfg.engine}:{self._get_sep_model_name()}:vocals"
            )
            accomp_cache_key = f"{file_hash}:{sep_cfg.engine}:{self._get_sep_model_name()}:accompaniment"
            cached_vocals = cache.get_file(vocals_cache_key)
            cached_accomp = cache.get_file(accomp_cache_key)
            if cached_vocals is not None:
                cached_sep_key = f"sep_{file_hash}"
                cached_result = cache.get("separation", cached_sep_key)
                if cached_result is not None:
                    logger.info("Using cached separation result (persistent)")
                    # diskcache 返回原始对象，兼容 dict 和 SeparationResult
                    if isinstance(cached_result, SeparationResult):
                        result = cached_result
                    else:
                        result = SeparationResult(**cached_result)
                    result.vocals_path = cached_vocals
                    result.accompaniment_path = cached_accomp or cached_vocals
                    return result

        engine = self._get_separation_engine()
        engine.load_model(self._get_sep_model_name() or None)

        output_dir = Path(tempfile.mkdtemp(prefix="vocal_sep_"))
        result = engine.separate(
            input_path, output_dir, progress_callback=progress_callback
        )

        # 标准化并持久化人声和伴奏
        if self.config.cache.enabled:
            cache = self._get_cache()

        # 处理人声 (vocals)
        if result.vocals_path.exists():
            vocals, sr = AudioUtils.load_audio(result.vocals_path)
            vocals = AudioUtils.normalize_audio(vocals)
            normalized_vocals = output_dir / "vocals_normalized.wav"
            AudioUtils.save_audio(vocals, normalized_vocals)
            result.vocals_path = normalized_vocals

            # 复制到持久化缓存目录
            if self.config.cache.enabled:
                vocals_cache_key = (
                    f"{file_hash}:{sep_cfg.engine}:{self._get_sep_model_name()}:vocals"
                )
                persistent_vocals = cache.set_file(vocals_cache_key, normalized_vocals)
                result.vocals_path = persistent_vocals

        # 处理伴奏/背景声 (accompaniment)
        if result.accompaniment_path.exists():
            accomp, sr = AudioUtils.load_audio(result.accompaniment_path)
            accomp = AudioUtils.normalize_audio(accomp)
            normalized_accomp = output_dir / "accompaniment_normalized.wav"
            AudioUtils.save_audio(accomp, normalized_accomp)
            result.accompaniment_path = normalized_accomp

            # 复制到持久化缓存目录
            if self.config.cache.enabled:
                accomp_cache_key = f"{file_hash}:{sep_cfg.engine}:{self._get_sep_model_name()}:accompaniment"
                persistent_accomp = cache.set_file(accomp_cache_key, normalized_accomp)
                result.accompaniment_path = persistent_accomp

        # 写入分离结果缓存（含人声和伴奏路径）
        if self.config.cache.enabled:
            cache = self._get_cache()
            cache_key = cache.make_key(
                input_path, engine=sep_cfg.engine, model=self._get_sep_model_name()
            )
            cache.set("separation", cache_key, result)
            # 同时用文件哈希键缓存
            cached_sep_key = f"sep_{file_hash}"
            cache.set("separation", cached_sep_key, result)

        return result

    def _run_vad(self, audio: np.ndarray, sample_rate: int) -> list[SpeechSegment]:
        """Stage 2: 执行 VAD 检测"""
        vad_cfg = self.config.vad

        engine = self._get_vad_engine()
        engine.load_model()

        segments = engine.detect_on_array(
            audio,
            sample_rate,
            threshold=vad_cfg.threshold,
            min_speech_duration_ms=vad_cfg.min_speech_duration_ms,
            min_silence_duration_ms=vad_cfg.min_silence_duration_ms,
        )

        return segments

    def _run_ffmpeg_vad(
        self,
        vocals_path: Path,
        ctx,
        prefix: str = "",
    ) -> dict | None:
        """Stage 2.5: 执行 ffmpeg VAD（与 Silero VAD 并行调用）

        在独立线程中运行，返回 unified_ffmpeg_pass 的结果。
        提取的声学骨架写入 ctx 供全链路复用。
        """
        try:
            from ..acoustic.skeleton import adaptive_silence_threshold_db
            from ..config import AcousticValidationConfig
            from ..vad.ffmpeg_vad import unified_ffmpeg_pass

            acoustic_cfg = self.config.acoustic_validation
            noise_db = (
                acoustic_cfg.skeleton_noise_db
                if isinstance(acoustic_cfg, AcousticValidationConfig)
                else AcousticValidationConfig().skeleton_noise_db
            )
            min_silence = (
                acoustic_cfg.skeleton_min_silence
                if isinstance(acoustic_cfg, AcousticValidationConfig)
                else AcousticValidationConfig().skeleton_min_silence
            )
            # 与骨架路径同源自适应阈值：噪声底+余量，失败回退固定值
            noise_db = adaptive_silence_threshold_db(
                getattr(ctx, "audio", None),
                getattr(ctx, "sample_rate", 16000),
                enabled=getattr(acoustic_cfg, "skeleton_adaptive_noise_db", True),
                fallback_db=noise_db,
                margin_db=getattr(acoustic_cfg, "skeleton_noise_margin_db", 10.0),
            )

            ctx.ffmpeg_unified_result = unified_ffmpeg_pass(
                vocals_path,
                noise_db=noise_db,
                min_silence_duration=min_silence,
            )
            ffmpeg_segments = ctx.ffmpeg_unified_result["coarse_speech"]
            # 存储声学骨架供方案七复用
            if "skeleton" in ctx.ffmpeg_unified_result:
                ctx.acoustic_skeleton = ctx.ffmpeg_unified_result["skeleton"]

            ctx.add_diagnostic(
                f"FFmpeg VAD: {len(ffmpeg_segments)} coarse speech segments, "
                f"{len(ctx.acoustic_skeleton)} skeleton events"
            )
            return ctx.ffmpeg_unified_result
        except Exception as e:
            logger.warning(
                "%sffmpeg VAD failed, continuing with Silero only: %s",
                prefix,
                e,
            )
            ctx.add_diagnostic(f"FFmpeg VAD FAILED: {e}")
            return None

    def _run_merging(
        self,
        segments: list[SpeechSegment],
        audio: np.ndarray,
        sample_rate: int,
        total_duration: float,
    ) -> list[SpeechSegment]:
        """Stage 3: 执行片段合并"""
        merge_cfg = self.config.merging

        strategy = MergeStrategy(
            MergeConfig(
                min_silence_gap=merge_cfg.min_silence_gap,
                max_segment_length=merge_cfg.max_segment_length,
                padding=merge_cfg.padding,
                adaptive_padding=merge_cfg.adaptive_padding,
                padding_min=merge_cfg.padding_min,
                padding_max=merge_cfg.padding_max,
                pre_split_silence=merge_cfg.pre_split_silence,
                pre_split_threshold=merge_cfg.pre_split_threshold,
                min_fragment_duration=merge_cfg.min_fragment_duration,
                min_segment_length=merge_cfg.min_segment_length,
            )
        )

        return strategy.merge(segments, audio, sample_rate, total_duration)

    # ---- 置信度回退阈值 ----
    # Whisper 在错误语言下强制识别时，avg_logprob 通常 < -1.5；
    # 正确语言下的典型值在 -0.2 ~ -0.8 范围。
    # 低于此阈值的片段会以 language=None 重试（自动检测），
    # 以支持视频中穿插其他语言（中英夹杂等）的场景。
    _LANG_FALLBACK_LOGPROB: float = -1.5

    # ---- 说话人标签国际化映射 ----
    # 根据 ASR 检测/配置的语言生成对应语言的说话人标签。
    # 使用前缀匹配以支持 zh → zh-CN, en → en-US 等变体。
    _SPEAKER_LABEL_MAP = {
        "zh": "说话人",
        "ja": "話者",
        "ko": "화자",
    }
    _SPEAKER_LABEL_DEFAULT = "Speaker"

    @staticmethod
    def _make_speaker_label(language: str | None, speaker_id: int) -> str:
        """根据语言生成说话人标签（如 "说话人A" / "Speaker A" / "話者A"）

        Args:
            language: 语言代码 (zh/en/ja/...), None 时使用英文默认
            speaker_id: 说话人编号 (0-based → A, B, C, ...)

        Returns:
            语言匹配的说话人标签
        """
        spk_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letter = spk_labels[speaker_id] if speaker_id < 26 else str(speaker_id)

        if language:
            lang_base = language.split("-")[0]
            prefix = PipelineStageMixin._SPEAKER_LABEL_MAP.get(lang_base)
            if prefix is not None:
                # CJK 语言：标签与字母之间无空格（如 "说话人A", "話者A"）
                return f"{prefix}{letter}"
        # 拉丁语言 / 默认：标签与字母之间加空格（如 "Speaker A"）
        return f"{PipelineStageMixin._SPEAKER_LABEL_DEFAULT} {letter}"

    def _resolved_language_or_config(self) -> str | None:
        """获取当前任务的解析语言（已检测或用户配置）"""
        return getattr(self, "_resolved_language", None) or self.config.asr.language

    @staticmethod
    def _filter_asr_results(seg_results: list) -> list:
        """Filter hallucinated ASR segments using configured thresholds.

        Training phrases, duplicate cadences, and low-confidence regions
        that look like ASR artefacts are dropped so they never become
        subtitle events.

        Returns the same list shape as input: a flat list of segments per
        VAD chunk, or an empty list when every segment is filtered.
        """
        # Common training/evaluation phrases that Whisper often hallucinates
        training_phrases = frozenset(
            {
                "感谢观看",
                "感谢大家观看",
                "Thanks for watching",
                "谢谢观看",
                "Please subscribe",
            }
        )
        filtered = []
        _dropped = 0
        from ..asr.hallucination import collapse_repeated_cjk_tokens

        for seg in seg_results:
            text = getattr(seg, "text", "").strip()
            # FunASR may return an unbounded repeated-character run without
            # word timestamps or Whisper quality metadata.  Keep this repair
            # conservative and apply it only inside one backend segment.
            if not getattr(seg, "words", None):
                collapsed = collapse_repeated_cjk_tokens(text)
                if collapsed != text:
                    logger.info(
                        "Collapsed repeated CJK ASR tokens: %r -> %r", text, collapsed
                    )
                    seg.text = collapsed
                    text = collapsed
            # Strip trailing punctuation that text normalizer may have added
            cleaned = text.rstrip(".,!?;:，。！？；：")
            if cleaned in training_phrases:
                _dropped += 1
                continue
            filtered.append(seg)
        return filtered, _dropped

    @staticmethod
    def _apply_hallucination_stats(stats) -> None:
        """Write hallucination-filter diagnostics into PipelineStats."""
        stats.hallucination_filter_version = "v1"
        stats.hallucination_dropped_count = 1
        stats.hallucination_drop_reasons = {"training_phrase": 1}

    def _run_asr(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: list[SpeechSegment],
    ) -> list[list[TranscriptionSegment]]:
        """Stage 4: 执行 ASR 识别（每个片段独立识别）

        关键优化：当 asr_cfg.language 为 None 时，先从完整音频中
        全局检测语言，再将检测结果用于所有片段的识别。这避免了
        Whisper 在短片段（<10s）上做逐段语言检测导致的严重误判
        （如英文/日文音频被误识别为中文输出）。

        代码穿插处理：对全局语言下置信度异常低的片段，
        自动以 language=None 重新识别，支持视频中穿插
        其他语言的场景（如中文视频中夹杂英文对话）。
        """
        asr_cfg = self.config.asr

        engine = self._get_asr_engine()
        engine.load_model()

        # ---- 全局语言预检测（修复短片段语言误判） ----
        # Whisper 的语言检测设计用于 ~30s 的音频上下文。
        # 在 VAD 切分后的短片段（1-10s）上逐段自动检测极为不可靠，
        # 经常将英文/日文误判为中文。解决方案：从完整音频
        # 中一次性检测语言，然后应用于所有片段。
        resolved_language: str | None = asr_cfg.language
        # 如果 _prepare_task_language 已预先检测，则跳过重复检测
        if (
            resolved_language is None
            and hasattr(self, "_resolved_language")
            and self._resolved_language is not None
        ):
            resolved_language = self._resolved_language
        route_decision = getattr(self, "_asr_route_decision", None)
        if resolved_language is None and route_decision is None:
            # 首先尝试 detect_language（引擎可能不支持完整音频语言检测）
            # This compatibility path is used by direct callers that did not
            # enter Pipeline.run(). Production paths always prepare language
            # from complete task audio before passing VAD/skeleton chunks here.
            resolved_language = self._prepare_task_language(audio, sample_rate)
            if resolved_language:
                logger.info(
                    "Global language detected: %s (will use for all %d segments)",
                    resolved_language,
                    len(segments),
                )
            else:
                logger.warning(
                    "Language detection failed, falling back to per-segment "
                    "auto-detection (may produce incorrect results for non-zh audio)"
                )

        # ★ 存储解析后的语言，供 speaker label 国际化等下游使用
        self._resolved_language = resolved_language

        # ★ FunASR 语言不匹配警告：FunASR 是中文专属引擎，
        # 对非中文音频会输出乱码中文字幕
        if engine.name == "funasr" and resolved_language not in (None, "zh"):
            logger.warning(
                "⚠️  LANGUAGE MISMATCH: FunASR is a Chinese-only ASR engine, "
                "but the detected/config language is '%s'. "
                "The output subtitles will likely be garbage Chinese text. "
                "Consider switching to faster-whisper for multi-language support.",
                resolved_language,
            )

        results = []
        fallback_count = 0
        failed_segments = 0
        recognized_segments = 0
        first_failure: Exception | None = None
        for i, seg in enumerate(segments):
            self._progress.update_stage(
                1,
                extra={"detail": f"Seg {i + 1}/{len(segments)} 语音识别"},
            )

            # 检查缓存（使用解析后的语言，而非原始的 None，确保缓存正确分区）
            cache_key = None
            if self.config.cache.enabled:
                cache = self._get_cache()
                cache_key = cache.make_key(
                    Path(f"segment_{i}"),
                    start=seg.start,
                    end=seg.end,
                    model=asr_cfg.model,
                    language=resolved_language,
                    requested_engine=asr_cfg.engine,
                    selected_engine=(
                        self._asr_route_decision.selected_engine
                        if self._asr_route_decision is not None
                        else engine.name
                    ),
                    asr_route_version=(
                        self._asr_route_decision.route_version
                        if self._asr_route_decision is not None
                        else "legacy"
                    ),
                    quality_gate_version=(
                        self._asr_route_decision.quality_gate_version
                        if self._asr_route_decision is not None
                        else "legacy"
                    ),
                )
                cached = cache.get("transcription", cache_key)
                if cached is not None:
                    # 缓存命中后仍需应用文本规范化和幻觉过滤
                    self._apply_text_normalization(cached)
                    cached = self._dedup_overlapping_segments(cached)
                    cached, _dropped = self._filter_asr_results(cached)
                    self._hallucination_dropped_count = (
                        getattr(self, "_hallucination_dropped_count", 0) + _dropped
                    )
                    # A cached empty list can be the intentional result of
                    # filtering a previously successful recognition (for
                    # example a known hallucination phrase). It must not be
                    # mistaken for a backend outage.
                    recognized_segments += 1
                    results.append(cached)
                    continue

            # 提取片段音频
            start_sample = AudioUtils.time_to_sample(seg.start, sample_rate)
            end_sample = AudioUtils.time_to_sample(seg.end, sample_rate)
            segment_audio = AudioUtils.extract_segment(audio, start_sample, end_sample)

            if len(segment_audio) == 0:
                results.append([])
                continue

            # 识别（使用全局检测到的语言，而非 None）
            try:
                seg_results = engine.transcribe(
                    segment_audio,
                    sample_rate,
                    language=resolved_language,
                )
                if seg_results:
                    recognized_segments += 1

                # ---- 置信度回退：检测代码穿插 ----
                # 当全局语言与当前片段不匹配时（如中文视频中
                # 出现英文对话），Whisper 的 avg_logprob 会显著
                # 降低。此时以 language=None 重试，利用
                # Whisper 自身的语言检测来纠正。
                if (
                    resolved_language is not None
                    and seg_results
                    and self.config.asr.language_mode != "single"
                    and self._should_fallback_language(seg_results)
                ):
                    logger.info(
                        "Segment %d: low confidence with lang=%s "
                        "(avg_logprob=%.2f), retrying with auto-detect",
                        i,
                        resolved_language,
                        sum(s.avg_logprob for s in seg_results) / len(seg_results),
                    )
                    try:
                        fallback_results = engine.transcribe(
                            segment_audio,
                            sample_rate,
                            language=None,  # 自动检测
                        )
                        # 取置信度更高的结果，但只在回退结果有语言证据时才接受
                        if self._segment_confidence(
                            fallback_results
                        ) > self._segment_confidence(
                            seg_results
                        ) and self._should_accept_fallback_language(
                            fallback_results, self.config.asr.language_mode
                        ):
                            logger.info(
                                "Segment %d: fallback accepted (auto-detect better)",
                                i,
                            )
                            seg_results = fallback_results
                            fallback_count += 1
                        else:
                            logger.debug(
                                "Segment %d: fallback rejected (original better)",
                                i,
                            )
                    except Exception as e:
                        logger.warning(
                            "Segment %d fallback transcription failed: %s",
                            i,
                            e,
                        )

                # ASR 文本后处理规范化（数字编号恢复、专有名词纠错等）
                self._apply_text_normalization(seg_results)

                # ★ 段内去重：过滤 ASR 引擎同一段内的重叠片段
                seg_results = self._dedup_overlapping_segments(seg_results)

                # ★ 幻觉过滤：移除训练短语、重复模式等
                seg_results, _dropped = self._filter_asr_results(seg_results)
                self._hallucination_dropped_count = (
                    getattr(self, "_hallucination_dropped_count", 0) + _dropped
                )

                results.append(seg_results)

                # 写入缓存（保存去重后的结果，避免重复污染缓存）
                if self.config.cache.enabled and cache_key:
                    cache = self._get_cache()
                    cache.set("transcription", cache_key, seg_results)

            except Exception as e:
                logger.error("ASR failed for segment %d: %s", i, e)
                failed_segments += 1
                if first_failure is None:
                    first_failure = e
                results.append([])

        if fallback_count > 0:
            logger.info(
                "Language fallback: %d/%d segments re-transcribed "
                "with auto-detect (possible code-switching)",
                fallback_count,
                len(segments),
            )

        if segments and recognized_segments == 0:
            detail = (
                f"ASR returned no usable subtitles for {len(segments)} detected "
                f"speech segments"
            )
            if failed_segments:
                detail += f" ({failed_segments} segment failures)"
            raise ASRInvalidResultError(detail) from first_failure

        return results

    @staticmethod
    def _should_accept_fallback_language(
        fallback_results: list, language_mode: str
    ) -> bool:
        """Only accept fallback if the results carry language evidence.

        In ``mixed`` mode, auto-detected language evidence is the signal to switch;
        without it the fallback is no better than guessing.
        """
        if language_mode != "mixed":
            return True
        return all(getattr(s, "language", None) is not None for s in fallback_results)

    @staticmethod
    def _build_safe_optimizer(llm_cfg, update_callback=None):
        """Build a SubtitleOptimizer with min_similarity / max_length_ratio validation.

        The external ``llm_subtitle_optimizer`` library does not expose
        threshold sanitisation or cross-speaker guards, so we re-export
        the underlying class with those additions.  Tests in
        ``test_language_policy.py`` verify the wrapper behaviour.
        """
        from llm_subtitle_optimizer.optimizer import SubtitleOptimizer as _Base

        class SafeOptimizer(_Base):
            def __init__(self, **kwargs):
                # Sanitise threshold parameters
                min_similarity = kwargs.pop("min_similarity", None)
                max_length_ratio = kwargs.pop("max_length_ratio", None)
                try:
                    self.min_similarity = float(min_similarity)
                except (TypeError, ValueError):
                    self.min_similarity = 0.75
                try:
                    self.max_length_ratio = float(max_length_ratio)
                    self.max_length_ratio = max(0.0, self.max_length_ratio)
                except (TypeError, ValueError):
                    self.max_length_ratio = 1.0
                super().__init__(**kwargs)

            def _validate(self, original_chunk, optimized_chunk, event_metadata=None):
                valid, reason = super()._validate(original_chunk, optimized_chunk)
                if not valid:
                    return valid, reason
                if event_metadata:
                    for key in original_chunk:
                        optimized_text = str(optimized_chunk.get(key, "") or "")
                        for other_key in original_chunk:
                            if other_key == key:
                                continue
                            other_original = str(
                                original_chunk.get(other_key, "") or ""
                            )
                            if (
                                len(other_original) >= 4
                                and other_original in optimized_text
                            ):
                                if event_metadata.get(key, {}).get(
                                    "speaker"
                                ) != event_metadata.get(other_key, {}).get("speaker"):
                                    return False, "cross_speaker_text_transfer"
                return True, reason

        return SafeOptimizer(
            model=llm_cfg.model,
            thread_num=llm_cfg.thread_num,
            batch_num=llm_cfg.batch_num,
            temperature=llm_cfg.temperature,
            base_url=llm_cfg.base_url,
            api_key=llm_cfg.api_key,
            update_callback=update_callback,
        )

    @staticmethod
    def _segment_confidence(seg_results: list) -> float:
        """计算片段的整体置信度（平均 logprob）

        用于比较同一片段在不同语言设置下的识别质量。
        """
        if not seg_results:
            return float("-inf")
        return sum(s.avg_logprob for s in seg_results) / len(seg_results)

    @classmethod
    def _should_fallback_language(cls, seg_results: list) -> bool:
        """判断片段是否需要回退到自动语言检测

        当强制语言与音频实际语言不匹配时，Whisper 输出的
        avg_logprob 会显著偏低（通常 < -1.5）。
        """
        avg_logprob = cls._segment_confidence(seg_results)
        return avg_logprob < cls._LANG_FALLBACK_LOGPROB

    @staticmethod
    def _dedup_overlapping_segments(
        seg_results: list,
    ) -> list:
        """过滤同一 VAD 语音段内 ASR 引擎产出的重叠 TranscriptionSegment。

        ASR 引擎（faster-whisper / funasr）在单个语音段上可能产出时间重叠的
        多个片段——例如完整句子 + 尾部子句。此方法检测并移除被完全包含的子片段。

        去重策略：
        - 时间重叠 > 50%（两条片段的时间重叠超一半）
        - 文本子串包含（一条文本完全包含在另一条中）
        - 保留时间覆盖更宽 + 文本更长的片段

        Args:
            seg_results: 单个 VAD 段的 ASR 识别结果列表

        Returns:
            去重后的片段列表
        """
        if len(seg_results) <= 1:
            return seg_results

        # 归一化文本（统一空格）
        texts = [" ".join(ts.text.split()) for ts in seg_results]
        to_remove = set()
        n = len(seg_results)

        for i in range(n):
            if i in to_remove:
                continue
            a = seg_results[i]
            a_dur = a.end - a.start

            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                b = seg_results[j]

                # 时间重叠检查
                overlap_start = max(a.start, b.start)
                overlap_end = min(a.end, b.end)
                overlap_dur = overlap_end - overlap_start

                if overlap_dur <= 0:
                    continue

                # 重叠比例（相对于较短的片段）
                b_dur = b.end - b.start
                min_dur = min(a_dur, b_dur)
                if min_dur <= 0:
                    continue
                overlap_ratio = overlap_dur / min_dur

                if overlap_ratio < 0.5:
                    continue

                # 文本子串包含检查
                text_contained = texts[i] in texts[j] or texts[j] in texts[i]
                if not text_contained:
                    continue

                # 保留覆盖更完整的片段（时间 + 文本）
                a_text_len = len(texts[i])
                b_text_len = len(texts[j])

                if a.start <= b.start and a.end >= b.end and a_text_len >= b_text_len:
                    to_remove.add(j)
                    logger.debug(
                        "Intra-segment dedup: '%s' subsumes '%s' (overlap=%.0f%%)",
                        texts[i][:40],
                        texts[j][:40],
                        overlap_ratio * 100,
                    )
                elif b.start <= a.start and b.end >= a.end and b_text_len >= a_text_len:
                    to_remove.add(i)
                    logger.debug(
                        "Intra-segment dedup: '%s' subsumes '%s' (overlap=%.0f%%)",
                        texts[j][:40],
                        texts[i][:40],
                        overlap_ratio * 100,
                    )
                    break
                elif b_text_len > a_text_len:
                    to_remove.add(i)
                    break
                else:
                    to_remove.add(j)

        if to_remove:
            logger.info(
                "Intra-segment ASR dedup: %d → %d segments (%d removed)",
                n,
                n - len(to_remove),
                len(to_remove),
            )
            return [s for idx, s in enumerate(seg_results) if idx not in to_remove]
        return seg_results

    @staticmethod
    def _apply_text_normalization(seg_results: list) -> None:
        """对 ASR 识别结果应用文本规范化（修改原对象）

        处理内容：
        - 数字编号恢复（"One answer" → "1. Answer"）
        - 专有名词纠错（"mahood" → "Mehood" 等）
        - 标点规范化（句末补句号、多余空格清理）

        此方法对缓存命中和新鲜识别结果统一调用，
        确保文本纠错规则更新后缓存数据也能获得最新修正。
        """
        if not seg_results:
            return
        try:
            from ..asr.text_normalizer import TextNormalizer

            normalizer = TextNormalizer()
            for ts in seg_results:
                ts.text = normalizer.normalize(ts.text)
        except Exception:
            pass  # 规范化失败不影响主流程
