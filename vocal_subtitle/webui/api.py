"""WebUI API router assembly and legacy compatibility exports."""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..application.pipeline_result import PipelineStats
from ..asr.funasr_manager import FunASRPrepareError
from ..config import ConfigLoader, PipelineConfig
from ..mapping.time_mapper import SubtitleEvent
from ..utils.file_hasher import compute_config_hash, compute_file_hash
from ..utils.gpu_detector import GPUDetector
from ..utils.session_manager import OUTPUT_NAMES, SessionManager
from ..utils.task_history import TaskHistoryManager
from .api_services import (
    LLM_PROVIDERS,
    ensure_funasr_ready,
    funasr_status,
)
from .api_services import (
    config_summary as _config_summary,  # noqa: F401 (legacy façade)
)
from .api_services import (
    config_to_overrides as _config_to_overrides_dict,  # noqa: F401 (legacy façade)
)
from .api_services import (
    has_saved_hf_token as _has_saved_hf_token,  # noqa: F401 (legacy façade)
)
from .api_services import (
    profile_description as _get_profile_description,  # noqa: F401 (legacy façade)
)
from .api_services import (
    speaker_model_download_detail as _speaker_model_download_detail,  # noqa: F401 (legacy façade)
)
from .pipeline_tasks import Pipeline
from .runtime_state import state

# Legacy names remain importable for integrations and monkeypatch-based tests.
_task_store = state.task_store
_shadow_evaluators = state.shadow_evaluators
_task_history = state.task_history
UPLOAD_DIR = state.upload_dir
_persistence_mgr = None

# Keep the historical module-level API as a façade over the split route
# modules.  Integrations have imported these names directly, and a number of
# callers monkeypatch them before creating the FastAPI application.  The
# implementation remains owned by the route/service modules.
from . import routes_subtitles as subtitles_routes
from .models import (  # noqa: E402
    BatchRunRequest,
    CacheConfigUpdate,
    CacheInfoResponse,
    ConflictInfo,
    ConflictResolutionRequest,
    DeviceInfoResponse,
    FeedbackLearnRequest,
    FeedbackLearnResponse,
    FeedbackPreviewResponse,
    FingerprintInfo,
    FingerprintListResponse,
    FingerprintMatchResponse,
    FunASRPrepareRequest,
    HealthScoreDetail,
    HealthTrendEntry,
    ImpactPredictionInfo,
    PersistenceSettingsModel,
    ProfileInfo,
    RunRequest,
    ShadowModeStatus,
    ShadowModeToggleRequest,
    SubtitleEventResponse,
    TaskHistoryItem,
    TaskStatus,
    UserProfileInfo,
)
from .routes_dataset import router as dataset_router
from .routes_editor_sync import router as editor_sync_router
from .routes_feedback import (
    compute_health,
    delete_feedback_profile,
    delete_fingerprint,
    detect_conflicts,
    get_feedback_profile,
    get_health_trend,
    get_shadow_status,
    list_feedback_profiles,
    list_fingerprints,
    match_fingerprint,  # noqa: F401 (legacy façade)
    preview_impact,
    record_shadow_run,
    resolve_conflict,
    rollback_feedback_profile,
    toggle_shadow_mode,
)
from .routes_feedback import router as feedback_router
from .routes_feedback_learning import (
    feedback_learn,
    feedback_preview,
)
from .routes_feedback_learning import (
    router as feedback_learning_router,
)
from .routes_history import (
    _dir_size_mb,  # noqa: F401 (legacy façade)
    _get_persistence_mgr,  # noqa: F401 (legacy façade)
    apply_persistence,
    cleanup_expired_persistence,
    clear_cache,
    clear_history,
    delete_history,
    delete_persisted_files,
    find_history_by_hash,
    get_cache_info,
    get_history_detail,
    get_persisted_files,
    get_persistence_settings,
    get_persistence_stats,
    list_history,
    update_cache_config,
    update_persistence_settings,
)
from .routes_history import router as history_router
from .routes_journal import router as journal_router
from .routes_llm import (
    fetch_llm_models,
    list_llm_providers,
)
from .routes_llm import (
    router as llm_router,
)
from .routes_models import (
    download_speaker_model,
    get_device_info,
    get_funasr_status,
    get_profile_config,
    get_speaker_embedding_license,
    get_speaker_model_status,
    list_profiles,
    list_speaker_models,
    prepare_funasr,
)
from .routes_models import router as models_router
from .routes_pipeline import (
    _run_pipeline_in_thread,  # noqa: F401 (legacy façade)
    get_task_status,
    list_tasks,
    run_pipeline,
)
from .routes_pipeline import router as pipeline_router
from .routes_quality import router as quality_router
from .routes_subtitles import (
    download_separated_audio,
    download_subtitle_file,
    export_subtitle,
    get_subtitles,
    stream_audio,
)
from .routes_subtitles import router as subtitles_router

router = APIRouter()
router.include_router(llm_router)
router.include_router(feedback_learning_router)
router.include_router(feedback_router)
router.include_router(dataset_router)
router.include_router(models_router)
router.include_router(history_router)
router.include_router(quality_router)
router.include_router(pipeline_router)
router.include_router(journal_router)
router.include_router(editor_sync_router)
router.include_router(subtitles_router)


@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "version": __version__}


# Compatibility exports used by integrations and historical tests.
_subtitle_event_from_payload = subtitles_routes._subtitle_event_from_payload
_subtitle_event_to_payload = subtitles_routes._subtitle_event_to_payload

__all__ = [
    "BatchRunRequest",
    "CacheConfigUpdate",
    "CacheInfoResponse",
    "ConfigLoader",
    "ConflictInfo",
    "ConflictResolutionRequest",
    "DeviceInfoResponse",
    "FeedbackLearnRequest",
    "FeedbackLearnResponse",
    "FeedbackPreviewResponse",
    "FingerprintInfo",
    "FingerprintListResponse",
    "FingerprintMatchResponse",
    "FunASRPrepareError",
    "FunASRPrepareRequest",
    "GPUDetector",
    "HealthScoreDetail",
    "HealthTrendEntry",
    "ImpactPredictionInfo",
    "LLM_PROVIDERS",
    "OUTPUT_NAMES",
    "PersistenceSettingsModel",
    "Pipeline",
    "PipelineConfig",
    "PipelineStats",
    "ProfileInfo",
    "RunRequest",
    "SessionManager",
    "ShadowModeStatus",
    "ShadowModeToggleRequest",
    "SubtitleEvent",
    "SubtitleEventResponse",
    "TaskHistoryItem",
    "TaskStatus",
    "TaskHistoryManager",
    "UserProfileInfo",
    "UPLOAD_DIR",
    "compute_config_hash",
    "compute_file_hash",
    "compute_health",
    "apply_persistence",
    "cleanup_expired_persistence",
    "clear_cache",
    "clear_history",
    "delete_feedback_profile",
    "delete_fingerprint",
    "delete_history",
    "delete_persisted_files",
    "detect_conflicts",
    "download_separated_audio",
    "download_speaker_model",
    "download_subtitle_file",
    "export_subtitle",
    "feedback_learn",
    "feedback_preview",
    "fetch_llm_models",
    "find_history_by_hash",
    "funasr_status",
    "get_cache_info",
    "get_device_info",
    "get_feedback_profile",
    "get_funasr_status",
    "get_health_trend",
    "get_history_detail",
    "get_persisted_files",
    "get_persistence_settings",
    "get_persistence_stats",
    "get_profile_config",
    "get_shadow_status",
    "get_speaker_embedding_license",
    "get_speaker_model_status",
    "get_subtitles",
    "get_task_status",
    "list_profiles",
    "list_feedback_profiles",
    "list_fingerprints",
    "list_history",
    "list_speaker_models",
    "list_tasks",
    "list_llm_providers",
    "prepare_funasr",
    "preview_impact",
    "record_shadow_run",
    "resolve_conflict",
    "rollback_feedback_profile",
    "run_pipeline",
    "router",
    "stream_audio",
    "toggle_shadow_mode",
    "update_cache_config",
    "update_persistence_settings",
    "ensure_funasr_ready",
    "health_check",
]
