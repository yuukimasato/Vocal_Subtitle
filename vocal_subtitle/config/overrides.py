"""Configuration override application independent of YAML loading."""

from __future__ import annotations

import copy
import typing
from typing import Any, Dict

from .models import PipelineConfig


FIELD_MAP = {
    "separator": "separation.engine",
    "uvr_model": "separation.uvr_model",
    "vad_engine": "vad.engine",
    "vad_threshold": "vad.threshold",
    "ffmpeg_enabled": "vad.ffmpeg_enabled",
    "ffmpeg_noise_db": "vad.ffmpeg_noise_db",
    "asr_model": "asr.model",
    "asr_engine": "asr.engine",
    "qwen_model_path": "asr.qwen_model_path",
    "primary_engine": "asr.engine_pair.primary",
    "secondary_engine": "asr.engine_pair.secondary",
    "engine_pair_policy": "asr.engine_pair.policy",
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
    "speaker_embedding": "speaker_embedding.enabled",
    "speaker_embedding_model": "speaker_embedding.model_ref",
    "speaker_embedding_token": "speaker_embedding.hf_token",
    "skeleton_mode": "acoustic_validation.skeleton_mode",
    "export_skeleton_segments": "acoustic_validation.export_skeleton_segments",
    "export_skeleton_dir": "acoustic_validation.export_skeleton_dir",
}


def set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a dotted dataclass attribute and coerce form values to its type."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)

    target_type = None
    if hasattr(obj, "__dataclass_fields__"):
        field_info = obj.__dataclass_fields__.get(parts[-1])
        if field_info is not None:
            # Config models use postponed annotations, so ``field_info.type``
            # may be a string. Resolve it against the concrete dataclass.
            target_type = typing.get_type_hints(type(obj)).get(
                parts[-1], field_info.type
            )

    if target_type is not None and isinstance(value, str):
        origin = typing.get_origin(target_type)
        args = typing.get_args(target_type)
        if origin is not None and args:
            target_type = next(
                (arg for arg in args if arg is not type(None)), target_type
            )
        if target_type is int:
            value = int(value)
        elif target_type is float:
            value = float(value)
        elif target_type is bool:
            value = value.lower() not in ("false", "0", "")

    setattr(obj, parts[-1], value)


def _normalise_override(key: str, value: Any) -> Any:
    if key == "mixed_language":
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            return "mixed"
        if value is False or (isinstance(value, str) and value.lower() == "false"):
            return "single"
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def merge_with_overrides(config: PipelineConfig, **overrides: Any) -> PipelineConfig:
    """Return a deep-copied config with CLI/WebUI-style overrides applied."""
    new_config = copy.deepcopy(config)
    for key, value in overrides.items():
        if value is None:
            continue

        if isinstance(value, dict) and not FIELD_MAP.get(key):
            if hasattr(new_config, key):
                target = getattr(new_config, key)
                if hasattr(target, "__dataclass_fields__"):
                    for sub_key, sub_value in value.items():
                        path = FIELD_MAP.get(sub_key, f"{key}.{sub_key}")
                        set_nested_attr(new_config, path, sub_value)
                    continue

        path = FIELD_MAP.get(key, key)
        set_nested_attr(new_config, path, _normalise_override(key, value))
    return new_config


def apply_user_profile_overrides(
    config: PipelineConfig, user_overrides: Dict[str, Any]
) -> PipelineConfig:
    """Apply a nested user profile patch without mutating the base config."""
    new_config = copy.deepcopy(config)

    def apply_nested(overrides: dict, prefix: str = "") -> None:
        for key, value in overrides.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                apply_nested(value, path)
            else:
                set_nested_attr(new_config, path, value)

    apply_nested(user_overrides)
    return new_config


# Historical private helper name retained for callers importing loader internals.
_set_nested_attr = set_nested_attr
