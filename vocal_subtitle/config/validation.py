"""Cross-field configuration consistency checks."""

from __future__ import annotations

from .models import PipelineConfig


def validate_config_consistency(config: PipelineConfig) -> list[str]:
    """Return warnings for incompatible or suspicious configuration values."""
    warnings = []
    acoustic = config.acoustic_validation
    if acoustic.unified_ffmpeg_pass and config.vad.ffmpeg_enabled:
        skeleton_db = acoustic.skeleton_noise_db
        vad_db = config.vad.ffmpeg_noise_db
        if vad_db > skeleton_db:
            warnings.append(
                f"统一ffmpeg模式下，VAD阈值({vad_db}dB) > 声学校验阈值({skeleton_db}dB)。"
                f"将使用声学校验阈值({skeleton_db}dB)运行，VAD从中过滤。"
            )

    merging = config.merging
    if merging.pre_split_threshold < merging.min_fragment_duration:
        warnings.append(
            f"pre_split_threshold ({merging.pre_split_threshold}s) < "
            f"min_fragment_duration ({merging.min_fragment_duration}s)，可能出现无效切分。"
        )

    if acoustic.max_snap_distance > merging.min_silence_gap:
        warnings.append(
            f"max_snap_distance ({acoustic.max_snap_distance}s) > "
            f"min_silence_gap ({merging.min_silence_gap}s)。"
            "吸附后的边界微调可能导致相邻事件重叠，"
            "建议确保吸附距离不超过段间合并间隙。"
        )

    overlap = config.macro_chunking.overlap_ms / 1000.0
    if overlap > merging.max_segment_length:
        warnings.append(
            f"宏观切块重叠 ({overlap}s) > 单段最大长度 ({merging.max_segment_length}s)，"
            "重叠区内可能无法找到合适的缝合点。"
        )

    if config.asr.no_speech_threshold < 0:
        warnings.append(
            f"no_speech_threshold ({config.asr.no_speech_threshold}) must be >= 0 (range 0.0–1.0)"
        )
    if config.asr.no_speech_threshold > 1.0:
        warnings.append(
            f"no_speech_threshold ({config.asr.no_speech_threshold}) must be <= 1.0"
        )

    if config.merge_decision.llm_tier not in ("rule_only",):
        if not config.merge_decision.llm_base_url:
            warnings.append(
                "LLM merge 未配置 llm_base_url，云端 LLM 裁决路径自动跳过，"
                "仅使用本地 NLP + 规则降级（完全离线，无需 GPU）。"
                "如需云端 LLM 裁决，请配置 llm_base_url 和 llm_api_key。"
            )
    return warnings
