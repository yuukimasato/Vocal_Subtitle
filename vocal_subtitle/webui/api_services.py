"""Pure/shared WebUI services used by route groups.

This module contains response shaping and compatibility adapters only. It has
no router registration and does not import the API module, which keeps route
imports acyclic.
"""

from __future__ import annotations

import sys
from typing import Any

from ..config import PipelineConfig
from ..utils.hf_token_store import has_hf_token

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "default": "通用场景，分离引擎 UVR (BS-RoFormer)，适合日常音频处理",
    "podcast": "播客/访谈场景，UVR 高品质分离，中文优化，低 VAD 阈值捕捉更多语音",
    "education": "教学/演讲场景，UVR 分离，较大的合并间隙适应讲课节奏",
    "variety_show": "综艺/直播场景，UVR 分离，背景音乐较多时的最佳选择",
    "music_live": "音乐现场场景，UVR 高品质分离，专为含背景音乐的语音优化",
}

# Stable, non-secret provider metadata used by the legacy LLM endpoints and
# the split model-management route group.
LLM_PROVIDERS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek（深度求索）",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-pro",
        "default_models": [
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "base_url": "https://api.openai.com",
        "default_model": "gpt-5.5",
        "default_models": ["gpt-5.5", "gpt-5.4", "gpt-5", "o4-mini"],
    },
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic (Claude)",
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-fable-5",
        "default_models": [
            "claude-fable-5",
            "claude-mythos-5",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
    },
    "google": {
        "id": "google",
        "name": "Google (Gemini)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-3.5-flash",
        "default_models": ["gemini-3.5-flash", "gemini-3.5-pro", "gemini-3.0-pro"],
    },
    "zhipu": {
        "id": "zhipu",
        "name": "智谱 AI (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "GLM-5.2",
        "default_models": ["GLM-5.2", "GLM-5.1", "glm-4-plus"],
    },
    "dashscope": {
        "id": "dashscope",
        "name": "阿里百炼 (Qwen)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode",
        "default_model": "qwen3.7-max",
        "default_models": ["qwen3.7-max", "qwen-plus", "qwen-max", "qwen-turbo"],
    },
    "hunyuan": {
        "id": "hunyuan",
        "name": "腾讯混元 (Hunyuan)",
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "default_model": "hunyuan-hy3-preview",
        "default_models": ["hunyuan-hy3-preview", "hunyuan-turbo", "hunyuan-pro"],
    },
    "moonshot": {
        "id": "moonshot",
        "name": "月之暗面 (Kimi)",
        "base_url": "https://api.moonshot.cn",
        "default_model": "kimi-k2.6",
        "default_models": [
            "kimi-k2.6",
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
        ],
    },
    "minimax": {
        "id": "minimax",
        "name": "MiniMax",
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "minimax-m3",
        "default_models": ["minimax-m3", "minimax-m2.7", "abab6.5s-chat"],
    },
    "siliconflow": {
        "id": "siliconflow",
        "name": "硅基流动 (SiliconFlow)",
        "base_url": "https://api.siliconflow.cn",
        "default_model": "deepseek-ai/DeepSeek-V3",
        "default_models": [
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "Pro/deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen3-235B-A22B",
        ],
    },
    "ollama": {
        "id": "ollama",
        "name": "Ollama（本地）",
        "base_url": "http://localhost:11434",
        "default_model": "llama3",
        "default_models": [],
    },
    "custom": {
        "id": "custom",
        "name": "自定义",
        "base_url": "",
        "default_model": "",
        "default_models": [],
    },
}


def has_saved_hf_token() -> bool:
    try:
        return has_hf_token()
    except (ImportError, OSError, ValueError):
        return False


def speaker_model_download_detail(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    error_name = type(exc).__name__.lower()
    error_text = str(exc).lower()
    if status_code in (401, 403) or "gatedrepo" in error_name or "gated" in error_text:
        return "Hugging Face Token 无效或尚未获得该模型权限；请检查 Token 类型、模型协议和账号授权"
    if isinstance(exc, ImportError):
        return "缺少 huggingface_hub 依赖，请先安装 WebUI/模型下载依赖"
    if any(
        marker in error_text
        for marker in ("ssl", "httpsconnectionpool", "timeout", "connection")
    ):
        return "无法连接 Hugging Face，请检查网络、代理或 HF_ENDPOINT"
    if "incomplete" in error_text or "integrity" in error_text:
        return "模型下载未形成完整本地缓存，请重试下载"
    return "speaker model download failed"


def profile_description(name: str) -> str:
    return PROFILE_DESCRIPTIONS.get(name, "自定义配置")


def config_summary(config: PipelineConfig) -> dict[str, Any]:
    return {
        "separation_engine": config.separation.engine,
        "vad_engine": config.vad.engine,
        "asr_engine": config.asr.engine,
        "asr_model": config.asr.model,
        "asr_primary_engine": config.asr.engine_pair.primary,
        "asr_secondary_engine": config.asr.engine_pair.secondary,
        "asr_engine_pair_policy": config.asr.engine_pair.policy,
        "language": config.asr.language,
        "device": config.asr.device,
        "asr_route_version": config.asr.auto_routing.route_version,
        "asr_quality_gate_version": config.asr.auto_routing.quality_gate_version,
        "llm_enabled": config.llm_optimize.enabled,
    }


def config_to_overrides(config: PipelineConfig) -> dict[str, Any]:
    """Return the stable frontend configuration payload."""
    return {
        "separator": config.separation.engine,
        "uvr_model": config.separation.uvr_model,
        "vad_engine": config.vad.engine,
        "vad_threshold": config.vad.threshold,
        "vad_min_speech_ms": config.vad.min_speech_duration_ms,
        "vad_min_silence_ms": config.vad.min_silence_duration_ms,
        "merge_min_silence_gap": config.merging.min_silence_gap,
        "merge_max_segment": config.merging.max_segment_length,
        "merge_padding": config.merging.padding,
        "asr_engine": config.asr.engine,
        "asr_model": config.asr.model,
        "primary_engine": config.asr.engine_pair.primary,
        "secondary_engine": config.asr.engine_pair.secondary,
        "engine_pair_policy": config.asr.engine_pair.policy,
        "asr_device": config.asr.device,
        "asr_compute_type": config.asr.compute_type,
        "language": config.asr.language or "",
        "asr_beam_size": config.asr.beam_size,
        "asr_route_version": config.asr.auto_routing.route_version,
        "asr_quality_gate_version": config.asr.auto_routing.quality_gate_version,
        "subtitle_min_duration": config.subtitle.min_duration,
        "subtitle_max_duration": config.subtitle.max_duration,
        "subtitle_max_chars_cjk": config.subtitle.max_chars_cjk,
        "subtitle_max_chars_latin": config.subtitle.max_chars_latin,
        "llm_enabled": config.llm_optimize.enabled,
        "llm_model": config.llm_optimize.model,
        "llm_batch_num": config.llm_optimize.batch_num,
        "llm_thread_num": config.llm_optimize.thread_num,
        "llm_base_url": config.llm_optimize.base_url or "",
        "llm_api_key": config.llm_optimize.api_key or "",
        "diarization_enabled": config.diarization.enabled,
        "speaker_fusion": config.diarization.fusion_mode,
        "global_diarization_model": config.diarization.global_model,
        "speaker_diarization_scope": config.diarization.diarization_scope,
        "local_speaker_refinement": config.diarization.local_refinement,
        "expected_speakers": config.diarization.expected_speakers,
        "diarization_local_context": config.diarization.local_context_seconds,
        "diarization_min_local_segment": config.diarization.min_local_segment_seconds,
        "diarization_min_change_confidence": config.diarization.min_change_confidence,
        "diarization_distance_threshold": config.diarization.distance_threshold,
        "diarization_min_speakers": config.diarization.min_speakers,
        "diarization_max_speakers": config.diarization.max_speakers,
        "diarization_use_pca": config.diarization.use_pca,
        "diarization_pca_variance": config.diarization.pca_variance,
        "speaker_role_enabled": config.speaker_role.enabled,
        "speaker_role_model": config.speaker_role.model,
        "speaker_role_temperature": config.speaker_role.temperature,
        "speaker_role_context_hint": config.speaker_role.context_hint or "",
        "speaker_embedding_enabled": config.speaker_embedding.enabled,
        "speaker_embedding_engine": config.speaker_embedding.engine,
        "speaker_embedding_model_ref": config.speaker_embedding.model_ref,
        "speaker_embedding_hf_token": "***"
        if config.speaker_embedding.hf_token or has_saved_hf_token()
        else "",
        "acoustic_skeleton_mode": config.acoustic_validation.skeleton_mode,
        "acoustic_export_skeleton": config.acoustic_validation.export_skeleton_segments,
        "fast_merge_max_gap": config.merge_decision.fast_merge_max_gap,
        "llm_decision_min_gap": config.merge_decision.llm_decision_min_gap,
        "llm_decision_max_gap": config.merge_decision.llm_decision_max_gap,
        "hard_split_min_gap": config.merge_decision.hard_split_min_gap,
        "llm_tier": config.merge_decision.llm_tier,
        "llm_merge_model": config.merge_decision.llm_model,
        "feedback_enabled": config.feedback.enabled,
        "feedback_active_profile": config.feedback.active_profile,
    }


def _legacy_api_attr(name: str, fallback):
    api = sys.modules.get("vocal_subtitle.webui.api")
    candidate = getattr(api, name, None)
    # ``webui.api`` re-exports these adapters for legacy callers. Do not
    # resolve the re-export back to the adapter itself, or FunASR requests
    # recurse instead of reaching the manager implementation.
    if candidate is None or candidate is globals().get(name):
        return fallback
    return candidate


def funasr_status(model: str = ""):
    from ..asr.funasr_manager import funasr_status as default

    return _legacy_api_attr("funasr_status", default)(model)


def ensure_funasr_ready(model: str = ""):
    from ..asr.funasr_manager import ensure_funasr_ready as default

    return _legacy_api_attr("ensure_funasr_ready", default)(model)
