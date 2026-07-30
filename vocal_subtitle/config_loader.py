"""配置加载与校验模块

YAML 配置文件加载、参数覆盖和配置一致性校验。
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .config.models import (
    AcousticValidationConfig,
    ASRAutoRoutingConfig,
    ASRConfig,
    BoundaryRedundancyConfig,
    BoundaryRefinementConfig,
    CacheConfig,
    ContextReASRConfig,
    DegradationConfig,
    DiarizationConfig,
    FeedbackConfig,
    FFmpegVADConfig,
    FusionConfig,
    GapHandlingConfig,
    GlobalASRConfig,
    LLMOptimizeConfig,
    LoggingConfig,
    MacroChunkConfig,
    MergeDecisionConfig,
    MergingConfig,
    NoiseReductionConfig,
    PipelineConfig,
    SeparationConfig,
    SpeakerEmbeddingConfig,
    SpeakerRoleConfig,
    StreamingConfig,
    SubtitleBuildConfig,
    VADConfig,
)


# ---------------------------------------------------------------------------
# 配置加载器
# ---------------------------------------------------------------------------


class ConfigLoader:
    """YAML 配置文件加载与校验

    支持场景模板加载和参数覆盖。
    """

    # 内置场景模板路径
    BUILTIN_PROFILES: Dict[str, str] = {
        "default": "default.yaml",
        "podcast": "podcast.yaml",
        "education": "education.yaml",
        "variety_show": "variety_show.yaml",
        "music_live": "music_live.yaml",
    }

    def __init__(self, configs_dir: Optional[Path] = None):
        if configs_dir is None:
            configs_dir = Path(__file__).parent.parent / "configs"
        self._configs_dir = Path(configs_dir)

    def list_profiles(self) -> List[str]:
        """列出可用的场景模板名称"""
        profiles = list(self.BUILTIN_PROFILES.keys())
        return profiles

    def load_profile(self, profile: str = "default") -> PipelineConfig:
        """加载场景模板配置

        Args:
            profile: 场景模板名称 (default / podcast / education / ...)

        Returns:
            PipelineConfig 实例

        Raises:
            FileNotFoundError: 配置文件不存在
        """
        filename = self.BUILTIN_PROFILES.get(profile, f"{profile}.yaml")
        config_path = self._configs_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config profile '{profile}' not found at {config_path}"
            )

        return self.load_file(config_path)

    @staticmethod
    def load_file(config_path: Path) -> PipelineConfig:
        """从 YAML 文件加载管道配置

        Args:
            config_path: YAML 配置文件路径

        Returns:
            PipelineConfig 实例
        """
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return ConfigLoader._parse_config(raw)

    @staticmethod
    def _parse_config(raw: dict) -> PipelineConfig:
        """将 YAML 字典解析为 PipelineConfig 对象"""
        pipeline_raw = raw.get("pipeline", raw)

        # 解析各阶段配置
        sep_raw = pipeline_raw.get("separation", {})
        separation = SeparationConfig(
            engine=sep_raw.get("engine", "spleeter"),
            uvr_model=sep_raw.get(
                "uvr_model",
                "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            ),
            output_sample_rate=sep_raw.get("output", {}).get("sample_rate", 16000),
            output_channels=sep_raw.get("output", {}).get("channels", 1),
            output_format=sep_raw.get("output", {}).get("format", "wav"),
            output_bit_depth=sep_raw.get("output", {}).get("bit_depth", 16),
        )

        vad_raw = pipeline_raw.get("vad", {})
        vad = VADConfig(
            engine=vad_raw.get("engine", "silero"),
            threshold=vad_raw.get("threshold", 0.5),
            min_speech_duration_ms=vad_raw.get("min_speech_duration_ms", 150),
            min_silence_duration_ms=vad_raw.get("min_silence_duration_ms", 400),
            ffmpeg_enabled=vad_raw.get("ffmpeg_enabled", True),
            ffmpeg_noise_db=vad_raw.get("ffmpeg_noise_db", -35.0),
            ffmpeg_weight=vad_raw.get("ffmpeg_weight", 0.4),
        )

        merge_raw = pipeline_raw.get("merging", {})
        merging = MergingConfig(
            min_silence_gap=merge_raw.get("min_silence_gap", 0.4),
            max_segment_length=merge_raw.get("max_segment_length", 20.0),
            padding=merge_raw.get("padding", 0.10),
            adaptive_padding=merge_raw.get("adaptive_padding", True),
            padding_min=merge_raw.get("padding_min", 0.05),
            padding_max=merge_raw.get("padding_max", 0.20),
            pre_split_silence=merge_raw.get("pre_split_silence", True),
            pre_split_threshold=merge_raw.get("pre_split_threshold", 0.8),
            min_fragment_duration=merge_raw.get("min_fragment_duration", 0.15),
            min_segment_length=merge_raw.get("min_segment_length", 0.5),
            protect_single_word=merge_raw.get("protect_single_word", True),
            min_word_gap_ms=merge_raw.get("min_word_gap_ms", 80),
        )

        diar_raw = pipeline_raw.get("diarization", {})
        diarization = DiarizationConfig(
            enabled=diar_raw.get("enabled", True),
            engine=diar_raw.get("engine", "agglomerative"),
            backend=diar_raw.get("backend", "auto"),
            fusion_mode=diar_raw.get("fusion_mode", "auto"),
            global_model=diar_raw.get("global_model", "auto"),
            diarization_scope=diar_raw.get("diarization_scope", "hierarchical"),
            distance_threshold=diar_raw.get("distance_threshold", 0.5),
            min_speakers=diar_raw.get("min_speakers", 1),
            max_speakers=diar_raw.get("max_speakers", 10),
            expected_speakers=diar_raw.get("expected_speakers"),
            use_pca=diar_raw.get("use_pca", True),
            pca_variance=diar_raw.get("pca_variance", 0.95),
            text_fallback=diar_raw.get("text_fallback", True),
            local_refinement=diar_raw.get("local_refinement", "embedding"),
            local_context_seconds=diar_raw.get("local_context_seconds", 0.6),
            min_local_segment_seconds=diar_raw.get("min_local_segment_seconds", 0.25),
            min_change_confidence=diar_raw.get("min_change_confidence", 0.70),
        )

        asr_raw = pipeline_raw.get("asr", {})
        global_asr_raw = asr_raw.get("global_asr", {})
        auto_routing_raw = asr_raw.get("auto_routing", {})
        global_asr = GlobalASRConfig(
            enabled=global_asr_raw.get("enabled", True),
            routing=global_asr_raw.get("routing", "auto"),
            backend=global_asr_raw.get("backend", "faster-whisper"),
            left_context=global_asr_raw.get("left_context", 0.5),
            right_context=global_asr_raw.get("right_context", 0.5),
            min_word_confidence=global_asr_raw.get("min_word_confidence", 0.0),
            hallucination_filter=global_asr_raw.get("hallucination_filter", True),
            language_switch_threshold=global_asr_raw.get("language_switch_threshold", 0.7),
            mode=global_asr_raw.get("mode", "evidence"),
            direct_max_duration_seconds=global_asr_raw.get("direct_max_duration_seconds", 180.0),
            macro_window_seconds=global_asr_raw.get("macro_window_seconds", 45.0),
            macro_overlap_seconds=global_asr_raw.get("macro_overlap_seconds", 2.0),
        )
        context_raw = asr_raw.get("context_reasr", {})
        context_reasr = ContextReASRConfig(
            enabled=context_raw.get("enabled", True),
            left_context_seconds=context_raw.get("left_context_seconds", 0.8),
            right_context_seconds=context_raw.get("right_context_seconds", 0.8),
            max_window_seconds=context_raw.get("max_window_seconds", 15.0),
            max_group_seconds=context_raw.get("max_group_seconds", 10.0),
            min_duration_trigger=context_raw.get("min_duration_trigger", 0.8),
            cps_warning_threshold=context_raw.get("cps_warning_threshold", 15.0),
            cps_high_risk_threshold=context_raw.get("cps_high_risk_threshold", 20.0),
            primary_engine=context_raw.get("primary_engine", "faster-whisper"),
            primary_model=context_raw.get("primary_model", "large-v3"),
            secondary_engine=context_raw.get("secondary_engine", "whisper-cpp"),
            secondary_model=context_raw.get("secondary_model", "medium"),
            secondary_on_conflict=context_raw.get("secondary_on_conflict", True),
            secondary_on_training_phrase=context_raw.get("secondary_on_training_phrase", True),
            version=context_raw.get("version", "context-reasr-v1"),
            cross_engine_policy_version=context_raw.get("cross_engine_policy_version", "cross-engine-v1"),
        )
        auto_routing = ASRAutoRoutingConfig(
            enabled=auto_routing_raw.get("enabled", True),
            language_probe_model=auto_routing_raw.get("language_probe_model", "tiny"),
            language_probe_window_seconds=auto_routing_raw.get(
                "language_probe_window_seconds", 8.0
            ),
            language_probe_max_windows=auto_routing_raw.get(
                "language_probe_max_windows", 8
            ),
            zh_min_probability=auto_routing_raw.get("zh_min_probability", 0.85),
            zh_required_window_ratio=auto_routing_raw.get(
                "zh_required_window_ratio", 1.0
            ),
            uncertain_window_policy=auto_routing_raw.get(
                "uncertain_window_policy", "faster-whisper"
            ),
            fallback_on_quality_failure=auto_routing_raw.get(
                "fallback_on_quality_failure", True
            ),
            fallback_engine=auto_routing_raw.get("fallback_engine", "faster-whisper"),
            route_version=auto_routing_raw.get("route_version", "asr-route-v1"),
            quality_gate_version=auto_routing_raw.get(
                "quality_gate_version", "asr-quality-v1"
            ),
            min_coverage_ratio=auto_routing_raw.get("min_coverage_ratio", 0.80),
            min_text_density=auto_routing_raw.get("min_text_density", 0.20),
            max_event_duration=auto_routing_raw.get("max_event_duration", 12.0),
            long_audio_seconds=auto_routing_raw.get("long_audio_seconds", 60.0),
            long_audio_min_text_chars=auto_routing_raw.get(
                "long_audio_min_text_chars", 12
            ),
            max_overlap_ratio=auto_routing_raw.get("max_overlap_ratio", 0.35),
        )
        asr = ASRConfig(
            engine=asr_raw.get("engine", "auto"),
            model=asr_raw.get("model", "large-v3"),
            device=asr_raw.get("device", "cuda"),
            compute_type=asr_raw.get("compute_type", "float16"),
            language=asr_raw.get("language"),
            whisper_cpp_bin=asr_raw.get("whisper_cpp_bin"),
            whisper_cpp_model_path=asr_raw.get("whisper_cpp_model_path"),
            beam_size=asr_raw.get("beam_size", 5),
            word_timestamps=asr_raw.get("word_timestamps", True),
            condition_on_previous_text=asr_raw.get(
                "condition_on_previous_text", False
            ),
            vad_filter=asr_raw.get("vad_filter", False),
            language_mode=asr_raw.get("language_mode", "single"),
            global_asr=global_asr,
            context_reasr=context_reasr,
            auto_routing=auto_routing,
        )
        if asr.engine not in {"auto", "faster-whisper", "funasr", "whisper-cpp"}:
            raise ValueError(
                "Unknown asr.engine '%s'; expected auto, faster-whisper, funasr or whisper-cpp"
                % asr.engine
            )
        if asr.global_asr.mode not in {"evidence", "production"}:
            raise ValueError("asr.global_asr.mode must be evidence or production")
        if asr.global_asr.direct_max_duration_seconds <= 0:
            raise ValueError("asr.global_asr.direct_max_duration_seconds must be positive")
        if asr.global_asr.macro_window_seconds <= 0:
            raise ValueError("asr.global_asr.macro_window_seconds must be positive")
        if not 0 <= asr.global_asr.macro_overlap_seconds < asr.global_asr.macro_window_seconds:
            raise ValueError("asr.global_asr.macro_overlap_seconds must be within macro window")
        context = asr.context_reasr
        if context.max_window_seconds <= 0 or context.max_group_seconds <= 0:
            raise ValueError("context_reasr window limits must be positive")
        if context.max_group_seconds > context.max_window_seconds:
            raise ValueError("context_reasr.max_group_seconds cannot exceed max_window_seconds")
        if context.left_context_seconds < 0 or context.right_context_seconds < 0:
            raise ValueError("context_reasr context values cannot be negative")
        if context.cps_warning_threshold <= 0 or context.cps_high_risk_threshold < context.cps_warning_threshold:
            raise ValueError("context_reasr CPS thresholds are invalid")
        if not 0.0 <= float(asr.auto_routing.zh_min_probability) <= 1.0:
            raise ValueError("asr.auto_routing.zh_min_probability must be between 0 and 1")
        if not 0.0 <= float(asr.auto_routing.zh_required_window_ratio) <= 1.0:
            raise ValueError(
                "asr.auto_routing.zh_required_window_ratio must be between 0 and 1"
            )
        if float(asr.auto_routing.language_probe_window_seconds) <= 0:
            raise ValueError("asr.auto_routing.language_probe_window_seconds must be positive")
        if int(asr.auto_routing.language_probe_max_windows) < 2:
            raise ValueError("asr.auto_routing.language_probe_max_windows must be at least 2")
        if asr.auto_routing.uncertain_window_policy != "faster-whisper":
            raise ValueError(
                "asr.auto_routing.uncertain_window_policy must be faster-whisper"
            )

        spk_role_raw = pipeline_raw.get("speaker_role", {})
        speaker_role = SpeakerRoleConfig(
            enabled=spk_role_raw.get("enabled", False),
            model=spk_role_raw.get("model", "deepseek-v4-pro"),
            temperature=spk_role_raw.get("temperature", 0.2),
            base_url=spk_role_raw.get("base_url"),
            api_key=spk_role_raw.get("api_key"),
            context_hint=spk_role_raw.get("context_hint"),
        )

        emb_raw = pipeline_raw.get("speaker_embedding", {})
        speaker_embedding = SpeakerEmbeddingConfig(
            enabled=emb_raw.get("enabled", True),
            engine=emb_raw.get("engine", "pyannote"),
            model_ref=emb_raw.get("model_ref", "speechbrain/spkrec-ecapa-voxceleb"),
            hf_token=emb_raw.get("hf_token", ""),
            cache_dir=emb_raw.get("cache_dir", ""),
        )

        sub_raw = pipeline_raw.get("subtitle", {})
        gap_raw = sub_raw.get("gap_handling", {})
        subtitle = SubtitleBuildConfig(
            min_duration=sub_raw.get("min_duration", 0.8),
            max_duration=sub_raw.get("max_duration", 5.0),
            max_chars_cjk=sub_raw.get("max_chars_cjk", 20),
            max_chars_latin=sub_raw.get("max_chars_latin", 42),
            max_lines=sub_raw.get("max_lines", 2),
            gap_handling=GapHandlingConfig(
                seamless_threshold=gap_raw.get("seamless_threshold", 0.2),
                natural_pause_max=gap_raw.get("natural_pause_max", 1.0),
            ),
            frame_seamless=sub_raw.get("frame_seamless", True),
            max_stitch_gap=sub_raw.get("max_stitch_gap", 0.12),
        )

        llm_raw = pipeline_raw.get("llm_optimize", {})
        llm_optimize = LLMOptimizeConfig(
            enabled=llm_raw.get("enabled", False),
            model=llm_raw.get("model", "deepseek-v4-pro"),
            batch_num=llm_raw.get("batch_num", 5),
            thread_num=llm_raw.get("thread_num", 4),
            temperature=llm_raw.get("temperature", 0.2),
            base_url=llm_raw.get("base_url"),
            api_key=llm_raw.get("api_key"),
        )

        # 噪声抑制 — 支持 noise_reduction 和 noise_adaptation 两种命名
        noise_raw = raw.get("noise_reduction") or raw.get("noise_adaptation") or {}
        noise_reduction = NoiseReductionConfig(
            enabled=noise_raw.get("enabled", False),
            stationary=noise_raw.get("stationary", False),
            engine=noise_raw.get("engine", "spectral_gate"),
            spectral_noise_reduction_db=noise_raw.get(
                "spectral_noise_reduction_db", 12.0,
            ),
            spectral_noise_estimation_frames=noise_raw.get(
                "spectral_noise_estimation_frames", 10,
            ),
            burst_noise_protection=noise_raw.get(
                "burst_noise_protection", True,
            ),
            burst_noise_threshold_db=noise_raw.get(
                "burst_noise_threshold_db", 15.0,
            ),
            burst_noise_max_duration_ms=noise_raw.get(
                "burst_noise_max_duration_ms", 200,
            ),
        )

        cache_raw = raw.get("cache", {})
        cache = CacheConfig(
            enabled=cache_raw.get("enabled", True),
            directory=cache_raw.get("directory", "./cache"),
            ttl_separation=cache_raw.get("ttl_separation", 86400 * 7),
            ttl_transcription=cache_raw.get("ttl_transcription", 604800),
            max_size_mb=cache_raw.get("max_size_mb", 5000),
            history_retention_days=cache_raw.get("history_retention_days", 30),
            full_pipeline_cache=cache_raw.get("full_pipeline_cache", True),
        )

        log_raw = raw.get("logging", {})
        logging = LoggingConfig(
            level=log_raw.get("level", "INFO"),
            format=log_raw.get("format", "json"),
            file=log_raw.get("file", "logs/pipeline.log"),
        )

        # 降级模式 (pipeline.degradation first, then top-level degradation overrides)
        deg_raw = {}
        if pipeline_raw.get("degradation"):
            deg_raw.update(pipeline_raw["degradation"])
        if raw.get("degradation"):
            deg_raw.update(raw["degradation"])
        degradation = DegradationConfig(
            mode=deg_raw.get("mode", "full"),
            per_module_timeout=deg_raw.get("per_module_timeout", 60.0),
            ffmpeg_timeout=deg_raw.get("ffmpeg_timeout", 30.0),
            llm_api_timeout=deg_raw.get("llm_api_timeout", 15.0),
        )

        # 流式模式 (5.12.5)
        mode = pipeline_raw.get("mode", "offline")
        streaming_raw = pipeline_raw.get("streaming", {})
        streaming = StreamingConfig(
            chunk_duration=streaming_raw.get("chunk_duration", 2.0),
            overlap_duration=streaming_raw.get("overlap_duration", 0.5),
            max_latency=streaming_raw.get("max_latency", 3.0),
            llm_fallback=streaming_raw.get("llm_fallback", "local_nlp"),
            vad_engine=streaming_raw.get("vad_engine", "silero"),
        )

        # 方案〇：宏观切块
        macro_raw = pipeline_raw.get("macro_chunking", {})
        macro_chunking = MacroChunkConfig(
            enabled=macro_raw.get("enabled", True),
            auto_enable_threshold=macro_raw.get("auto_enable_threshold", 180.0),
            silence_threshold_db=macro_raw.get("silence_threshold_db", -30),
            min_silence_duration=macro_raw.get("min_silence_duration", 2.0),
            target_chunk_duration=macro_raw.get("target_chunk_duration", 60.0),
            max_chunk_duration=macro_raw.get("max_chunk_duration", 180.0),
            overlap_ms=macro_raw.get("overlap_ms", 200),
            recursive=macro_raw.get("recursive", True),
        )

        # 方案一：ffmpeg VAD
        ffmpeg_raw = pipeline_raw.get("ffmpeg_vad", {})
        ffmpeg_vad = FFmpegVADConfig(
            enabled=ffmpeg_raw.get("enabled", True),
            noise_db=ffmpeg_raw.get("noise_db", -35.0),
            min_speech_duration_ms=ffmpeg_raw.get("min_speech_duration_ms", 250),
            min_silence_duration_ms=ffmpeg_raw.get("min_silence_duration_ms", 400),
        )

        # 方案二：三方法边界融合
        fusion_raw = pipeline_raw.get("fusion", {})
        fusion = FusionConfig(
            enabled=fusion_raw.get("enabled", False),
            grid_resolution=fusion_raw.get("grid_resolution", 0.01),
            min_consensus=fusion_raw.get("min_consensus", 2),
            high_conf_padding=fusion_raw.get("high_conf_padding", 0.03),
            low_conf_padding=fusion_raw.get("low_conf_padding", 0.12),
            min_speech_duration=fusion_raw.get("min_speech_duration", 0.25),
        )

        # 方案四：ASR 边界精修
        boundary_raw = pipeline_raw.get("boundary_refinement", {})
        boundary_refinement = BoundaryRefinementConfig(
            enabled=boundary_raw.get("enabled", True),
            max_shrink_ms=boundary_raw.get("max_shrink_ms", 0),
            max_extend_ms=boundary_raw.get("max_extend_ms", 100),
            check_frames=boundary_raw.get("check_frames", 3),
            frame_ms=boundary_raw.get("frame_ms", 10),
            min_boundary_confidence=boundary_raw.get("min_boundary_confidence", 0.3),
            shrink_end_enabled=boundary_raw.get("shrink_end_enabled", False),
        )

        # 方案五：LLM 语义合并决策
        merge_dec_raw = pipeline_raw.get("merge_decision", {})
        merge_decision = MergeDecisionConfig(
            fast_merge_max_gap=merge_dec_raw.get("fast_merge_max_gap", 0.20),
            llm_decision_min_gap=merge_dec_raw.get("llm_decision_min_gap", 0.20),
            llm_decision_max_gap=merge_dec_raw.get("llm_decision_max_gap", 1.20),
            hard_split_min_gap=merge_dec_raw.get("hard_split_min_gap", 1.20),
            max_combined_duration=merge_dec_raw.get("max_combined_duration", 5.0),
            min_fragment_duration=merge_dec_raw.get("min_fragment_duration", 0.15),
            local_nlp_gap_range=tuple(merge_dec_raw.get(
                "local_nlp_gap_range", [0.15, 0.60]
            )),
            cloud_llm_gap_range=tuple(merge_dec_raw.get(
                "cloud_llm_gap_range", [0.60, 1.20]
            )),
            llm_tier=merge_dec_raw.get("llm_tier", "cascading"),
            llm_model=merge_dec_raw.get("llm_model", "deepseek-v4-pro"),
            llm_base_url=merge_dec_raw.get("llm_base_url"),
            llm_api_key=merge_dec_raw.get("llm_api_key"),
            llm_temperature=merge_dec_raw.get("llm_temperature", 0.1),
            llm_timeout=merge_dec_raw.get("llm_timeout", 15.0),
            llm_fallback_to_rules=merge_dec_raw.get("llm_fallback_to_rules", True),
        )

        # 方案七：全局声学标尺校验
        acoustic_raw = pipeline_raw.get("acoustic_validation", {})
        acoustic_validation = AcousticValidationConfig(
            enabled=acoustic_raw.get("enabled", True),
            skeleton_noise_db=acoustic_raw.get("skeleton_noise_db", -40.0),
            skeleton_min_silence=acoustic_raw.get("skeleton_min_silence", 0.1),
            skeleton_min_speech=acoustic_raw.get("skeleton_min_speech", 0.05),
            max_snap_distance=acoustic_raw.get("max_snap_distance", 0.5),
            snap_start_margin=acoustic_raw.get("snap_start_margin", 0.03),
            snap_end_margin=acoustic_raw.get("snap_end_margin", 0.01),
            confidence_threshold=acoustic_raw.get("confidence_threshold", 0.6),
            rms_override_threshold=acoustic_raw.get("rms_override_threshold", 0.15),
            generate_report=acoustic_raw.get("generate_report", True),
            flag_threshold_ms=acoustic_raw.get("flag_threshold_ms", 200),
            unified_ffmpeg_pass=acoustic_raw.get("unified_ffmpeg_pass", True),
            skeleton_mode=acoustic_raw.get("skeleton_mode", True),
            export_skeleton_segments=acoustic_raw.get("export_skeleton_segments", False),
            export_skeleton_dir=acoustic_raw.get("export_skeleton_dir", ""),
            allow_end_shorten=acoustic_raw.get("allow_end_shorten", True),
            allow_start_pull_earlier=acoustic_raw.get("allow_start_pull_earlier", True),
        )

        # Stage 4.6：边界滑动窗口冗余识别
        boundary_red_raw = pipeline_raw.get("boundary_redundancy", {})
        confidence_raw = boundary_red_raw.get("confidence", {})
        window_raw = boundary_red_raw.get("sliding_window", {})
        arb_raw = boundary_red_raw.get("arbitration", {})
        boundary_redundancy = BoundaryRedundancyConfig(
            enabled=boundary_red_raw.get("enabled", True),
            min_gap_trigger=confidence_raw.get("min_gap_trigger", 0.05),
            max_energy_slope_trigger=confidence_raw.get("max_energy_slope_trigger", 3.0),
            confidence_threshold=confidence_raw.get("confidence_threshold", 0.5),
            base_overlap_ms=window_raw.get("base_overlap_ms", 500),
            fast_speech_wps=window_raw.get("fast_speech_wps", 4.0),
            fast_overlap_ms=window_raw.get("fast_overlap_ms", 750),
            very_fast_overlap_ms=window_raw.get("very_fast_overlap_ms", 1000),
            fusion_window_sec=window_raw.get("fusion_window_sec", 1.0),
            max_workers=window_raw.get("max_workers", 3),
            llm_model=arb_raw.get("llm_model", "deepseek-v4-pro"),
            llm_base_url=arb_raw.get("llm_base_url"),
            llm_api_key=arb_raw.get("llm_api_key"),
            llm_temperature=arb_raw.get("llm_temperature", 0.1),
            llm_timeout=arb_raw.get("llm_timeout", 15.0),
            auto_apply_confidence=arb_raw.get("auto_apply_confidence", 0.8),
            review_threshold=arb_raw.get("review_threshold", 0.5),
            fallback_to_rules=arb_raw.get("fallback_to_rules", True),
        )

        # 反馈学习 (Phase 5)
        feedback_raw = raw.get("feedback", {})
        feedback = FeedbackConfig(
            enabled=feedback_raw.get("enabled", True),
            user_profile_dir=feedback_raw.get("user_profile_dir", "~/.vocal_subtitle/profiles"),
            active_profile=feedback_raw.get("active_profile", "user_default"),
            alignment_min_iou=feedback_raw.get("alignment_min_iou", 0.3),
            alignment_min_coverage=feedback_raw.get("alignment_min_coverage", 0.60),
            alignment_text_weight=feedback_raw.get("alignment_text_weight", 0.30),
            alignment_semantic_weight=feedback_raw.get("alignment_semantic_weight", 0.35),
            alignment_semantic_enabled=feedback_raw.get("alignment_semantic_enabled", True),
            min_samples_to_learn=feedback_raw.get("min_samples_to_learn", 3),
            base_learn_rate=feedback_raw.get("base_learn_rate", 0.10),
            max_learn_rate=feedback_raw.get("max_learn_rate", 0.35),
            param_isolation_enabled=feedback_raw.get("param_isolation_enabled", True),
            decay_long_term_days=feedback_raw.get("decay_long_term_days", 180),
            decay_medium_term_days=feedback_raw.get("decay_medium_term_days", 90),
            decay_short_term_days=feedback_raw.get("decay_short_term_days", 60),
            fingerprint_enabled=feedback_raw.get("fingerprint_enabled", True),
            fingerprint_distance_method=feedback_raw.get("fingerprint_distance_method", "mahalanobis"),
            fingerprint_knn_k=feedback_raw.get("fingerprint_knn_k", 3),
            fingerprint_min_absolute_similarity=feedback_raw.get("fingerprint_min_absolute_similarity", 0.70),
            fingerprint_relative_margin=feedback_raw.get("fingerprint_relative_margin", 0.08),
            shadow_mode_enabled=feedback_raw.get("shadow_mode_enabled", False),
            shadow_min_runs=feedback_raw.get("shadow_min_runs", 10),
            shadow_upgrade_threshold=feedback_raw.get("shadow_upgrade_threshold", 0.05),
            shadow_max_duration_days=feedback_raw.get("shadow_max_duration_days", 14),
            auto_rollback_on_quality_drop=feedback_raw.get("auto_rollback_on_quality_drop", True),
            quality_drop_threshold=feedback_raw.get("quality_drop_threshold", 0.3),
            oscillation_detection_window=feedback_raw.get("oscillation_detection_window", 5),
            few_shot_max_examples=feedback_raw.get("few_shot_max_examples", 3),
            few_shot_max_cache=feedback_raw.get("few_shot_max_cache", 20),
            few_shot_min_weight_to_inject=feedback_raw.get("few_shot_min_weight_to_inject", 0.3),
            few_shot_enabled=feedback_raw.get("few_shot_enabled", True),
        )

        return PipelineConfig(
            mode=mode,
            streaming=streaming,
            separation=separation,
            vad=vad,
            merging=merging,
            diarization=diarization,
            asr=asr,
            speaker_role=speaker_role,
            speaker_embedding=speaker_embedding,
            subtitle=subtitle,
            llm_optimize=llm_optimize,
            macro_chunking=macro_chunking,
            ffmpeg_vad=ffmpeg_vad,
            fusion=fusion,
            boundary_refinement=boundary_refinement,
            merge_decision=merge_decision,
            acoustic_validation=acoustic_validation,
            boundary_redundancy=boundary_redundancy,
            noise_reduction=noise_reduction,
            degradation=degradation,
            cache=cache,
            logging=logging,
            feedback=feedback,
        )

    def merge_with_overrides(self, config: PipelineConfig, **overrides) -> PipelineConfig:
        """通过关键字参数覆盖配置字段

        Args:
            config: 基础配置
            **overrides: 覆盖参数，使用点分隔路径:
                e.g. asr_model="medium", vad_threshold=0.3

        Returns:
            合并后的新 PipelineConfig
        """
        # 简单的命名映射: asr_model → asr.model
        field_map = {
            "separator": "separation.engine",
            "uvr_model": "separation.uvr_model",
            "vad_engine": "vad.engine",
            "vad_threshold": "vad.threshold",
            "ffmpeg_enabled": "vad.ffmpeg_enabled",
            "ffmpeg_noise_db": "vad.ffmpeg_noise_db",
            "asr_model": "asr.model",
            "asr_engine": "asr.engine",
            "language": "asr.language",
            "whisper_cpp_bin": "asr.whisper_cpp_bin",
            "whisper_cpp_model_path": "asr.whisper_cpp_model_path",
            "language_mode": "asr.language_mode",
            "mixed_language": "asr.language_mode",
            "device": "asr.device",
            "llm_optimize": "llm_optimize.enabled",
            "llm_model": "llm_optimize.model",
            "llm_base_url": "llm_optimize.base_url",
            "llm_api_key": "llm_optimize.api_key",
            "diarization": "diarization.enabled",
            "speaker_fusion": "diarization.fusion_mode",
            "global_diarization_model": "diarization.global_model",
            "speaker_diarization_scope": "diarization.diarization_scope",
            "local_speaker_refinement": "diarization.local_refinement",
            "diarization_local_context": "diarization.local_context_seconds",
            "diarization_min_local_segment": "diarization.min_local_segment_seconds",
            "diarization_min_change_confidence": "diarization.min_change_confidence",
            "expected_speakers": "diarization.expected_speakers",
            "diarization_distance_threshold": "diarization.distance_threshold",
            "diarization_min_speakers": "diarization.min_speakers",
            "diarization_max_speakers": "diarization.max_speakers",
            "speaker_role": "speaker_role.enabled",
            "speaker_role_context_hint": "speaker_role.context_hint",
            # 说话人嵌入模型
            "speaker_embedding": "speaker_embedding.enabled",
            "speaker_embedding_model": "speaker_embedding.model_ref",
            "speaker_embedding_token": "speaker_embedding.hf_token",
            # 骨架分段模式
            "skeleton_mode": "acoustic_validation.skeleton_mode",
            "export_skeleton_segments": "acoustic_validation.export_skeleton_segments",
            "export_skeleton_dir": "acoustic_validation.export_skeleton_dir",
        }

        import copy

        new_config = copy.deepcopy(config)

        for key, value in overrides.items():
            if value is None:
                continue

            # 如果 value 是 dict 且目标路径对应的属性也是 dataclass，
            # 则递归展开，逐字段设置，避免 dict 覆盖整个 dataclass 对象
            if isinstance(value, dict) and not field_map.get(key):
                # 直接 key（非映射路径）且为字典，尝试递归展开
                if hasattr(new_config, key):
                    target = getattr(new_config, key)
                    if hasattr(target, "__dataclass_fields__"):
                        for sub_key, sub_value in value.items():
                            sub_path = f"{key}.{sub_key}"
                            resolved_path = field_map.get(sub_key, sub_path)
                            if resolved_path == sub_path:
                                resolved_path = f"{key}.{sub_key}"
                            _set_nested_attr(new_config, resolved_path, sub_value)
                        continue

            # Normalize boolean overrides: True → "mixed", False → "single"
            if key == "mixed_language":
                if value is True or (isinstance(value, str) and value.lower() == "true"):
                    value = "mixed"
                elif value is False or (isinstance(value, str) and value.lower() == "false"):
                    value = "single"

            # Normalize other boolean string overrides
            normalized = value
            if isinstance(value, str):
                low = value.lower()
                if low in ("true", "false"):
                    normalized = low == "true"

            path = field_map.get(key, key)
            _set_nested_attr(new_config, path, normalized)

        return new_config

    @staticmethod
    def apply_user_profile_overrides(
        config: PipelineConfig,
        user_overrides: Dict[str, Any],
    ) -> PipelineConfig:
        """将用户配置文件的覆盖参数合并到管道配置

        递归遍历用户配置字典，使用 _set_nested_attr() 设置嵌套数据类字段。
        用户配置作为"补丁"覆盖默认配置，而非替换。

        Args:
            config: 基础 PipelineConfig（如从场景模板加载）
            user_overrides: 用户配置中的 overrides 字典，
                e.g. {"merging": {"padding": 0.14}, "merge_decision": {"fast_merge_max_gap": 0.25}}

        Returns:
            合并后的新 PipelineConfig（深拷贝，不影响原对象）
        """
        import copy

        new_config = copy.deepcopy(config)

        def _apply_nested(base_obj: Any, overrides: dict, prefix: str = ""):
            for key, value in overrides.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    # 递归应用嵌套字典
                    _apply_nested(base_obj, value, path)
                else:
                    _set_nested_attr(new_config, path, value)

        _apply_nested(new_config, user_overrides)
        return new_config


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """按点分隔路径设置嵌套属性

    自动将字符串值转换为目标字段声明的类型（int/float/bool），
    防止前端 HTML 表单的字符串值导致类型错误。
    """
    import typing

    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)

    # 类型强制转换：从 dataclass 字段声明中推断目标类型
    target_type = None
    if hasattr(obj, "__dataclass_fields__"):
        field_info = obj.__dataclass_fields__.get(parts[-1])
        if field_info is not None:
            target_type = field_info.type

    if target_type is not None and isinstance(value, str):
        # 处理 Optional[int] / int | None 等联合类型
        origin = typing.get_origin(target_type)
        args = typing.get_args(target_type)
        if origin is not None and args:
            # 联合类型：取第一个非 None 的具体类型
            for arg in args:
                if arg is not type(None):
                    target_type = arg
                    break

        if target_type is int:
            value = int(value)
        elif target_type is float:
            value = float(value)
        elif target_type is bool:
            value = value.lower() not in ("false", "0", "")

    setattr(obj, parts[-1], value)


def validate_config_consistency(config: PipelineConfig) -> List[str]:
    """校验配置参数一致性，返回警告列表

    Args:
        config: PipelineConfig 实例

    Returns:
        警告字符串列表（空列表 = 无问题）
    """
    warnings = []

    # 检查1: ffmpeg 统一调用时，方案一和方案七的参数兼容性
    acoustic = config.acoustic_validation
    if acoustic.unified_ffmpeg_pass and config.vad.ffmpeg_enabled:
        skeleton_db = acoustic.skeleton_noise_db
        vad_db = config.vad.ffmpeg_noise_db
        if vad_db > skeleton_db:
            warnings.append(
                f"统一ffmpeg模式下，VAD阈值({vad_db}dB) > 声学校验阈值({skeleton_db}dB)。"
                f"将使用声学校验阈值({skeleton_db}dB)运行，VAD从中过滤。"
            )

    # 检查2: 预切分阈值不应小于最小片段时长
    merging = config.merging
    pre_split = merging.pre_split_threshold
    min_frag = merging.min_fragment_duration
    if pre_split < min_frag:
        warnings.append(
            f"pre_split_threshold ({pre_split}s) < min_fragment_duration "
            f"({min_frag}s)，可能出现无效切分。"
        )

    # 检查3: max_snap_distance 不应超过段间最小合并间隙
    # 注意：声学吸附与快速合并在不同阶段独立运行，
    # 吸附仅做边界微调（非拆分事件），RMS 能量确认防止误吸附。
    # 这里使用合并间隙作为更合理的参考基线。
    max_snap = acoustic.max_snap_distance
    min_gap = merging.min_silence_gap
    if max_snap > min_gap:
        warnings.append(
            f"max_snap_distance ({max_snap}s) > min_silence_gap ({min_gap}s)。"
            f"吸附后的边界微调可能导致相邻事件重叠，"
            f"建议确保吸附距离不超过段间合并间隙。"
        )

    # 检查4: 宏观切块重叠量 > 2× 预切分最大段长（警告）
    macro = config.macro_chunking
    overlap = macro.overlap_ms / 1000.0
    max_seg = merging.max_segment_length
    if overlap > max_seg:
        warnings.append(
            f"宏观切块重叠 ({overlap}s) > 单段最大长度 ({max_seg}s)，"
            f"重叠区内可能无法找到合适的缝合点。"
        )

    # 检查5: 幻觉过滤阈值应在合理范围
    if config.asr.no_speech_threshold < 0:
        warnings.append(
            f"no_speech_threshold ({config.asr.no_speech_threshold}) "
            f"must be >= 0 (range 0.0–1.0)"
        )
    if config.asr.no_speech_threshold > 1.0:
        warnings.append(
            f"no_speech_threshold ({config.asr.no_speech_threshold}) "
            f"must be <= 1.0"
        )

    # 检查5: LLM tier 但未配置 API
    # 注意：未配置 llm_base_url 时，云端 LLM 路径自动跳过，仅使用
    # 本地 NLP（CPU 推理）+ 规则降级。如需云端裁决，
    # 在 llm_base_url 和 llm_api_key 中填入 API 凭证。
    if config.merge_decision.llm_tier not in ("rule_only",):
        if not config.merge_decision.llm_base_url:
            warnings.append(
                "LLM merge 未配置 llm_base_url，云端 LLM 裁决路径自动跳过，"
                "仅使用本地 NLP + 规则降级（完全离线，无需 GPU）。"
                "如需云端 LLM 裁决，请配置 llm_base_url 和 llm_api_key。"
            )

    return warnings
