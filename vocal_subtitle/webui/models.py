"""Pydantic 请求/响应数据模型"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """单文件处理请求"""

    profile: str = Field(default="default", description="场景模板名称")
    output_format: str = Field(default="srt", description="输出格式 (srt/vtt/ass)")
    skip_separation: bool = Field(default=False, description="跳过人声分离")
    overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="参数覆盖，如 {'asr_model': 'medium', 'language': 'zh'}",
    )


class BatchRunRequest(BaseModel):
    """批量处理请求"""

    profile: str = Field(default="default")
    output_format: str = Field(default="srt")
    glob_pattern: str = Field(default="*.mp3", description="文件匹配模式")
    skip_separation: bool = Field(default=False)
    overrides: Dict[str, Any] = Field(default_factory=dict)


class FunASRPrepareRequest(BaseModel):
    """FunASR dependency/model preparation request."""

    model: str = Field(default="", description="FunASR 模型 ID 或通用模型名")


class SubtitleEditRequest(BaseModel):
    """字幕编辑请求"""

    index: int = Field(..., description="字幕序号")
    text: str = Field(..., description="修改后的文本")


class SubtitleBatchEditRequest(BaseModel):
    """批量字幕编辑请求"""

    action: str = Field(..., description="批量操作: speaker 或 merge")
    indexes: List[int] = Field(..., min_length=1, description="字幕序号列表")
    speaker_id: Optional[int] = Field(default=None, description="说话人编号")
    speaker_label: Optional[str] = Field(default=None, description="说话人标签")
    separator: str = Field(default="newline", description="合并分隔符: newline 或 space")


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------


class ProfileInfo(BaseModel):
    """场景模板摘要信息"""

    name: str
    description: str
    config_summary: Dict[str, Any]


class TaskStatus(BaseModel):
    """任务状态"""

    task_id: str
    status: str  # pending | running | completed | failed
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SubtitleEventResponse(BaseModel):
    """单条字幕事件"""

    index: int
    start: float
    end: float
    text: str
    original_text: Optional[str] = None  # LLM 优化前的原始文本
    speaker_id: Optional[int] = None  # 说话人编号
    speaker_label: Optional[str] = None  # 说话人标签: "张三(嘉宾)"
    # 物理时间轴与溯源字段（前端可选渲染，保持旧响应兼容）
    physical_start: Optional[float] = None
    physical_end: Optional[float] = None
    source_word_ids: Optional[List[str]] = None
    speaker_status: Optional[str] = None
    speaker_source: Optional[str] = None
    speaker_confidence: Optional[float] = None
    speaker_model: Optional[str] = None
    speaker_repair_reason: Optional[str] = None
    alignment_warning: Optional[str] = None
    genuine_overlap: Optional[bool] = None
    overlap_group_id: Optional[str] = None


class DeviceInfoResponse(BaseModel):
    """设备信息"""

    device_type: str
    device_count: int
    device_names: List[str]
    memory_mb: List[int]
    recommended_compute_type: str
    gpu_memory_used_mb: Optional[float] = None
    recommended_model: str


# ---------------------------------------------------------------------------
# 任务历史 / 缓存管理
# ---------------------------------------------------------------------------


class TaskHistoryItem(BaseModel):
    """任务历史记录条目"""

    id: str
    input_file_name: str
    input_file_hash: str = ""
    input_file_size: int = 0
    profile: str = "default"
    status: str  # pending | running | completed | failed
    error: Optional[str] = None
    total_duration_seconds: float = 0
    created_at: str = ""
    completed_at: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None


class CacheInfoResponse(BaseModel):
    """缓存统计信息"""

    stages: Dict[str, Any] = Field(default_factory=dict)
    total_mb: float = 0
    total_items: int = 0
    files_dir_mb: float = 0
    cache_dir: str = ""
    ttl_map: Dict[str, int] = Field(default_factory=dict)
    task_history_count: int = 0
    task_history_db_mb: float = 0


class CacheConfigUpdate(BaseModel):
    """缓存配置更新请求"""

    ttl_separation: Optional[int] = None
    ttl_transcription: Optional[int] = None
    max_size_mb: Optional[int] = None
    history_retention_days: Optional[int] = None
    full_pipeline_cache: Optional[bool] = None


# ---------------------------------------------------------------------------
# 持久化配置
# ---------------------------------------------------------------------------


class PersistenceSettingsModel(BaseModel):
    """文件持久化设置"""

    persist_asr_subtitle: bool = True
    persist_llm_subtitle: bool = True
    persist_final_ass: bool = False
    persist_final_srt: bool = True
    persist_vocals: bool = True
    persist_accompaniment: bool = False
    ttl_subtitle_days: int = 90
    ttl_audio_days: int = 30


# ---------------------------------------------------------------------------
# 反馈学习模型 (Phase 5)
# ---------------------------------------------------------------------------


class FeedbackLearnRequest(BaseModel):
    """反馈学习请求"""

    profile: str = Field(default="default", description="场景模板名称")
    feedback_profile: str = Field(default="user_default", description="用户配置名称")
    run_pipeline_first: bool = Field(default=True, description="是否先用当前参数生成自动版")
    dry_run: bool = Field(default=False, description="仅预览差异，不更新配置")


class FeedbackLearnResponse(BaseModel):
    """反馈学习响应"""

    status: str  # "ok" | "skipped" | "error"
    alignment_coverage: float = 0.0
    total_pairs: int = 0
    time_shifts_count: int = 0
    merge_actions_count: int = 0
    text_edits_count: int = 0
    param_adjustments: Dict[str, Any] = Field(default_factory=dict)
    structural_revision: bool = False
    message: str = ""


class FeedbackPreviewResponse(BaseModel):
    """反馈预览响应（与 learn 相同，但不更新配置）"""

    status: str
    alignment_coverage: float = 0.0
    total_pairs: int = 0
    time_shifts_count: int = 0
    merge_actions_count: int = 0
    text_edits_count: int = 0
    param_adjustments: Dict[str, Any] = Field(default_factory=dict)
    structural_revision: bool = False
    message: str = ""


class UserProfileInfo(BaseModel):
    """用户配置信息"""

    profile_id: str
    base_profile: str = "default"
    feedback_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    is_active: bool = True
    overrides: Dict[str, Any] = Field(default_factory=dict)
    history_count: int = 0


# ---------------------------------------------------------------------------
# 指纹管理模型 (Phase 5.3)
# ---------------------------------------------------------------------------


class FingerprintInfo(BaseModel):
    """音频指纹信息"""

    id: int
    profile_id: str = ""
    audio_hash: str = ""
    audio_signature: str = ""
    feedback_count: int = 1
    created_at: str = ""


class FingerprintListResponse(BaseModel):
    """指纹列表响应"""

    fingerprints: List[FingerprintInfo] = Field(default_factory=list)
    total: int = 0
    db_path: str = ""


class FingerprintMatchResponse(BaseModel):
    """指纹匹配响应"""

    matched: bool = False
    profile_id: Optional[str] = None
    confidence: Optional[float] = None
    audio_signature: Optional[str] = None


# ---------------------------------------------------------------------------
# 健康度评分模型 (Phase 5.4)
# ---------------------------------------------------------------------------


class HealthScoreDetail(BaseModel):
    """健康度评分子项"""

    overall: float = 0.0
    alignment_coverage: float = 0.0
    semantic_similarity: float = 0.0
    time_iou: float = 0.0
    structure_consistency: float = 0.0
    grade: str = "fair"  # excellent | good | fair | poor


class HealthTrendEntry(BaseModel):
    """健康度趋势数据点"""

    timestamp: str = ""
    health_before: Optional[float] = None
    health_after: Optional[float] = None
    shadow_mode: bool = False
    summary: str = ""


# ---------------------------------------------------------------------------
# 影子模式模型 (Phase 5.4)
# ---------------------------------------------------------------------------


class ShadowModeStatus(BaseModel):
    """影子模式状态"""

    enabled: bool = False
    total_runs: int = 0
    current_mean_health: float = 0.0
    shadow_mean_health: float = 0.0
    health_delta: float = 0.0
    recommendation: str = "continue"
    reason: str = ""
    runs: List[Dict[str, Any]] = Field(default_factory=list)


class ShadowModeToggleRequest(BaseModel):
    """影子模式开关请求"""

    enabled: bool = True


# ---------------------------------------------------------------------------
# 冲突检测模型 (Phase 5.4)
# ---------------------------------------------------------------------------


class ConflictInfo(BaseModel):
    """参数冲突信息"""

    param_path: str
    is_oscillating: bool = False
    oscillation_count: int = 0
    severity: str = "low"  # low | medium | high
    recommended_action: str = ""
    possible_causes: List[str] = Field(default_factory=list)
    suggested_actions: List[Dict[str, str]] = Field(default_factory=list)
    entries: List[Dict[str, Any]] = Field(default_factory=list)


class ConflictResolutionRequest(BaseModel):
    """冲突解决请求"""

    param_path: str
    action: str  # "lock" | "branch" | "continue"


# ---------------------------------------------------------------------------
# 影响预估模型 (Phase 5.4)
# ---------------------------------------------------------------------------


class ImpactPredictionInfo(BaseModel):
    """影响预估信息"""

    param_path: str
    current_value: float = 0.0
    new_value: float = 0.0
    delta: float = 0.0
    delta_pct: float = 0.0
    summary: str = ""
    avg_duration_change_pct: Optional[float] = None
    merge_frequency_change_pct: Optional[float] = None
    split_frequency_change_pct: Optional[float] = None
    end_truncation_change_pct: Optional[float] = None
    total_line_count_change_pct: Optional[float] = None
    confidence_low: float = 0.0
    confidence_high: float = 0.0
