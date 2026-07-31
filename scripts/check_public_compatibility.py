#!/usr/bin/env python3
"""Check public compatibility names frozen at the componentization baseline."""

from __future__ import annotations

import argparse
import importlib


CONTRACT = {
    "vocal_subtitle.pipeline": {
        "Pipeline", "PipelineStats",
    },
    "vocal_subtitle.config": {
        "ASRAutoRoutingConfig", "ASRConfig", "AcousticValidationConfig",
        "BoundaryRedundancyConfig", "BoundaryRefinementConfig", "CacheConfig",
        "ConfigLoader", "DegradationConfig", "DiarizationConfig",
        "FFmpegVADConfig", "FeedbackConfig", "FusionConfig", "GapHandlingConfig",
        "GlobalASRConfig", "LLMOptimizeConfig", "LoggingConfig", "MacroChunkConfig",
        "MergeDecisionConfig", "MergingConfig", "NoiseReductionConfig",
        "PipelineConfig", "SeparationConfig", "SpeakerEmbeddingConfig",
        "SpeakerRoleConfig", "StreamingConfig", "SubtitleBuildConfig", "VADConfig",
        "validate_config_consistency",
    },
    "vocal_subtitle.acoustic_validator": {
        "AcousticValidationConfig", "AcousticValidator", "classify_acoustic_events",
        "export_skeleton_segments",
    },
    "vocal_subtitle.merging.llm_merge_engine": {
        "LLMMergeEngine", "MergeDecisionConfig", "MERGE_DECISION_PROMPT",
        "SUBTITLE_LAYOUT_RULES", "apply_frame_seamless_stitching",
        "apply_layout_suggestions", "auto_layout_events", "auto_line_break_fallback",
    },
    "vocal_subtitle.webui.api": {
        "BatchRunRequest", "CacheConfigUpdate", "CacheInfoResponse", "ConfigLoader",
        "ConflictInfo", "ConflictResolutionRequest", "DeviceInfoResponse",
        "FeedbackLearnRequest", "FeedbackLearnResponse", "FeedbackPreviewResponse",
        "FingerprintInfo", "FingerprintListResponse", "FingerprintMatchResponse",
        "FunASRPrepareError", "FunASRPrepareRequest", "GPUDetector",
        "HealthScoreDetail", "HealthTrendEntry", "ImpactPredictionInfo",
        "LLM_PROVIDERS", "OUTPUT_NAMES", "PersistenceSettingsModel", "Pipeline",
        "PipelineConfig", "PipelineStats", "ProfileInfo", "RunRequest",
        "SessionManager", "ShadowModeStatus", "ShadowModeToggleRequest",
        "SubtitleBatchEditRequest", "SubtitleEditRequest", "SubtitleEvent",
        "SubtitleEventResponse", "TaskHistoryItem", "TaskHistoryManager", "TaskStatus",
        "UPLOAD_DIR", "UserProfileInfo", "apply_persistence", "clear_cache",
        "clear_history", "compute_health", "delete_feedback_profile", "delete_fingerprint",
        "delete_history", "delete_persisted_files", "detect_conflicts",
        "download_separated_audio", "download_speaker_model", "download_subtitle_file",
        "export_subtitle", "feedback_learn", "feedback_preview", "fetch_llm_models",
        "funasr_status", "get_cache_info", "get_device_info", "get_feedback_profile",
        "get_funasr_status", "get_health_trend", "get_history_detail",
        "get_persisted_files", "get_persistence_settings", "get_persistence_stats",
        "get_profile_config", "get_shadow_status", "get_speaker_embedding_license",
        "get_speaker_model_status", "get_subtitles", "get_task_status", "list_feedback_profiles",
        "list_fingerprints", "list_history", "list_llm_providers", "list_profiles",
        "list_speaker_models", "list_tasks", "prepare_funasr", "preview_impact",
        "record_shadow_run", "resolve_conflict", "rollback_feedback_profile", "run_pipeline",
        "router", "stream_audio", "toggle_shadow_mode", "update_cache_config",
        "update_persistence_settings", "update_subtitle", "update_subtitles_batch",
        "ensure_funasr_ready",
    },
}

PRIVATE_HOOKS = {
    "vocal_subtitle.webui.api": {
        "_load_completed_subtitle_task", "_persist_subtitle_result", "_rewrite_subtitle_files",
        "_run_pipeline_in_thread", "_shadow_evaluators", "_subtitle_event_from_payload",
        "_subtitle_event_to_payload", "_task_history", "_task_store", "_persistence_mgr",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    missing: list[str] = []
    for module_name, names in CONTRACT.items():
        module = importlib.import_module(module_name)
        for name in sorted(names):
            if not hasattr(module, name):
                missing.append(f"missing:{module_name}:{name}")
    for module_name, names in PRIVATE_HOOKS.items():
        module = importlib.import_module(module_name)
        for name in sorted(names):
            if not hasattr(module, name):
                missing.append(f"missing-hook:{module_name}:{name}")
    for item in missing:
        print(item)
    if args.report:
        return 0
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
