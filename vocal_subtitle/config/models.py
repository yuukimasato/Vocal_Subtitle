"""Configuration dataclasses and stable defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------


@dataclass
class SeparationConfig:
    """Stage 1: 人声分离配置"""

    engine: str = "uvr"
    uvr_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    output_sample_rate: int = 16000
    output_channels: int = 1
    output_format: str = "wav"
    output_bit_depth: int = 16


@dataclass
class VADConfig:
    """Stage 2: VAD 语音检测配置"""

    engine: str = "silero"
    threshold: float = 0.5
    min_speech_duration_ms: int = 150
    min_silence_duration_ms: int = 400

    # 方案一：ffmpeg silencedetect 并行 VAD
    ffmpeg_enabled: bool = True
    ffmpeg_noise_db: float = -35.0
    ffmpeg_weight: float = 0.4


@dataclass
class FFmpegVADConfig:
    """方案一：ffmpeg silencedetect 独立配置"""

    enabled: bool = True
    noise_db: float = -35.0
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 400


@dataclass
class FusionConfig:
    """方案二：三方法边界融合配置"""

    enabled: bool = False
    grid_resolution: float = 0.01       # 10ms
    min_consensus: int = 2               # 最少共识方法数
    high_conf_padding: float = 0.03
    low_conf_padding: float = 0.12
    min_speech_duration: float = 0.25


@dataclass
class BoundaryRefinementConfig:
    """方案四：ASR 边界双向精修配置"""

    enabled: bool = True
    max_shrink_ms: float = 0          # 默认禁用段尾收缩（由反向能量扫描负责 end 精度）
    max_extend_ms: float = 100
    check_frames: int = 3
    frame_ms: int = 10
    min_boundary_confidence: float = 0.3
    shrink_end_enabled: bool = False  # 段尾收缩开关（默认关闭）


@dataclass
class MergeDecisionConfig:
    """方案五：LLM 语义合并决策配置"""

    # Fast-Slow Path 分流阈值
    fast_merge_max_gap: float = 0.20
    llm_decision_min_gap: float = 0.20
    llm_decision_max_gap: float = 1.20
    hard_split_min_gap: float = 1.20

    # 合并约束
    max_combined_duration: float = 5.0
    min_fragment_duration: float = 0.15

    # LLM 降本策略：渐进降级间隙范围
    local_nlp_gap_range: Tuple[float, float] = (0.15, 0.60)  # 此范围内优先本地NLP
    cloud_llm_gap_range: Tuple[float, float] = (0.60, 1.20)  # 仅此范围调用云端LLM

    # LLM 策略
    llm_tier: str = "cascading"          # "cascading" | "all_llm" | "rule_only"
    llm_model: str = "deepseek-v4-pro"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.1
    llm_timeout: float = 15.0
    llm_fallback_to_rules: bool = True


@dataclass
class MacroChunkConfig:
    """方案〇：宏观静音切块配置"""

    enabled: bool = True
    auto_enable_threshold: float = 180.0  # >3分钟自动启用
    silence_threshold_db: float = -30
    min_silence_duration: float = 2.0
    target_chunk_duration: float = 60.0
    max_chunk_duration: float = 180.0
    overlap_ms: int = 200
    recursive: bool = True


@dataclass
class AcousticValidationConfig:
    """方案七：全局声学标尺校验配置"""

    enabled: bool = True
    skeleton_noise_db: float = -40.0
    skeleton_min_silence: float = 0.1
    skeleton_min_speech: float = 0.05
    max_snap_distance: float = 0.5   # 扩大声学吸附范围（原 0.25s → 0.5s）
    snap_start_margin: float = 0.03
    snap_end_margin: float = 0.01
    confidence_threshold: float = 0.6
    rms_override_threshold: float = 0.15
    generate_report: bool = True
    flag_threshold_ms: float = 200
    unified_ffmpeg_pass: bool = True
    # 双向修正（默认开启）
    allow_end_shorten: bool = True         # ★ 允许声学标尺缩短结束时间
    allow_start_pull_earlier: bool = True  # ★ 允许声学标尺将 start 向前吸附
    # 骨架分段独立处理模式：跳过 VAD 分段，直接按声学骨架逐段
    # 独立处理，然后拼接时间轴。每个骨架段是物理隔离的连续语音。
    skeleton_mode: bool = True
    # 导出骨架段音频供人工验证
    export_skeleton_segments: bool = False
    export_skeleton_dir: str = ""


@dataclass
class MergingConfig:
    """Stage 3: 片段合并配置

    Attributes:
        min_silence_gap: 相邻段间隔 < 此值则合并
        max_segment_length: 最大段长，超出则切分
        padding: 基础两端填充（秒）
        adaptive_padding: 是否根据边界能量梯度自适应调整 padding
        padding_min: 自适应最小 padding（能量清晰边界）
        padding_max: 自适应最大 padding（能量模糊边界）
        pre_split_silence: 是否在段内静音处预切分
        pre_split_threshold: 内部静音 > 此值则切分（秒）
        min_fragment_duration: 最小语音片段（秒），短于此值强制合并
        min_segment_length: 最小段长（秒），过短则丢弃
    """

    min_silence_gap: float = 0.4
    max_segment_length: float = 20.0
    padding: float = 0.10
    adaptive_padding: bool = True
    padding_min: float = 0.05
    padding_max: float = 0.20
    pre_split_silence: bool = True
    pre_split_threshold: float = 0.8   # 减少过度切分（原 0.5s → 0.8s）
    min_fragment_duration: float = 0.15
    min_segment_length: float = 0.5
    protect_single_word: bool = True     # 禁止在单词中间切分
    min_word_gap_ms: int = 80            # 单词内部允许的最大"静音"（清辅音间隔）


@dataclass
class GlobalASRConfig:
    """全局转录配置 — 以完整音频为窗口进行 ASR"""

    enabled: bool = True
    routing: str = "auto"  # auto | global | segmented
    backend: str = "faster-whisper"
    left_context: float = 0.5
    right_context: float = 0.5
    min_word_confidence: float = 0.0
    hallucination_filter: bool = True
    language_switch_threshold: float = 0.7


@dataclass
class ASRAutoRoutingConfig:
    """Automatic engine routing and result quality policy."""

    enabled: bool = True
    language_probe_model: str = "tiny"
    language_probe_window_seconds: float = 8.0
    language_probe_max_windows: int = 8
    zh_min_probability: float = 0.85
    zh_required_window_ratio: float = 1.0
    uncertain_window_policy: str = "faster-whisper"
    fallback_on_quality_failure: bool = True
    fallback_engine: str = "faster-whisper"
    route_version: str = "asr-route-v1"
    quality_gate_version: str = "asr-quality-v1"
    min_coverage_ratio: float = 0.80
    min_text_density: float = 0.20
    max_event_duration: float = 12.0
    long_audio_seconds: float = 60.0
    long_audio_min_text_chars: int = 12
    max_overlap_ratio: float = 0.35


@dataclass
class ASRConfig:
    """Stage 4: ASR 识别配置"""

    engine: str = "auto"
    model: str = "large-v3"
    device: str = "auto"  # auto = 自动检测 GPU/CPU
    compute_type: str = "float16"
    language: Optional[str] = None
    whisper_cpp_bin: Optional[str] = None
    whisper_cpp_model_path: Optional[str] = None
    beam_size: int = 5
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    vad_filter: bool = False
    language_mode: str = "single"  # single | mixed | auto
    global_asr: "GlobalASRConfig" = field(default_factory=GlobalASRConfig)
    auto_routing: ASRAutoRoutingConfig = field(default_factory=ASRAutoRoutingConfig)
    # Hallucination filter thresholds
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    hallucination_filter_version: str = "v1"


@dataclass
class GapHandlingConfig:
    """段间间隙处理配置"""

    seamless_threshold: float = 0.2  # matches default.yaml
    natural_pause_max: float = 1.0


@dataclass
class SubtitleBuildConfig:
    """Stage 5: 字幕构建配置"""

    min_duration: float = 0.8
    max_duration: float = 5.0
    max_chars_cjk: int = 20
    max_chars_latin: int = 42
    max_lines: int = 2
    gap_handling: GapHandlingConfig = field(default_factory=GapHandlingConfig)

    # 帧级无缝衔接（消除字幕闪烁）
    frame_seamless: bool = True           # 非句尾字幕自动衔接到下一句
    max_stitch_gap: float = 0.12          # 最多衔接 120ms 的间隙


@dataclass
class LLMOptimizeConfig:
    """LLM 后处理配置"""

    enabled: bool = False
    model: str = "deepseek-v4-pro"
    batch_num: int = 5   # 较小的批次减少 LLM 跨条目混淆
    thread_num: int = 4
    temperature: float = 0.2
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class NoiseReductionConfig:
    """前置降噪配置 (文档 5.12.1)

    在 VAD 处理前对音频进行降噪，减少突发噪音和稳态底噪干扰。

    Attributes:
        enabled: 是否启用降噪（默认关闭，纯净录音不需要）
        engine: 降噪引擎 ("spectral_gate" | "rnnoise" | "deepfilternet")
        spectral_noise_reduction_db: 谱减法降噪量 (dB)
        spectral_noise_estimation_frames: 噪声估计帧数
        burst_noise_protection: 是否启用突发噪音保护
        burst_noise_threshold_db: 突发噪音判定阈值 (dB)
        burst_noise_max_duration_ms: 突发噪音最大持续时长
    """

    enabled: bool = False
    stationary: bool = False  # 兼容旧字段
    engine: str = "spectral_gate"

    # 谱减法参数
    spectral_noise_reduction_db: float = 12.0
    spectral_noise_estimation_frames: int = 10

    # 突发噪音保护
    burst_noise_protection: bool = True
    burst_noise_threshold_db: float = 15.0
    burst_noise_max_duration_ms: int = 200


@dataclass
class DiarizationConfig:
    """Stage 3.5: 说话人分离配置（基于音色聚类）"""

    enabled: bool = True   # 默认启用说话人分离
    engine: str = "agglomerative"  # 聚类引擎: agglomerative
    backend: str = "auto"  # 后端: auto | pyannote | legacy
    fusion_mode: str = "auto"  # auto | embedding | dual
    global_model: str = "auto"  # auto | none | community-1 | diarization-3.1
    diarization_scope: str = "hierarchical"  # global | hierarchical
    distance_threshold: float = 0.5  # 凝聚聚类合并阈值（余弦距离）
    min_speakers: int = 1  # 最少说话人数
    max_speakers: int = 10  # 最多说话人数
    expected_speakers: Optional[int] = None  # 已知说话人数（None = 自动推断）
    use_pca: bool = True  # 聚类前是否 PCA 降维
    pca_variance: float = 0.95  # PCA 保留的方差比例
    text_fallback: bool = True  # 声学聚类失败时启用文本模式降级
    local_refinement: str = "embedding"  # off | embedding | full
    local_context_seconds: float = 0.6
    min_local_segment_seconds: float = 0.25
    min_change_confidence: float = 0.70


@dataclass
class SpeakerRoleConfig:
    """Stage 4.5: LLM 说话人角色标注配置"""

    enabled: bool = False
    model: str = "deepseek-v4-pro"  # LLM 模型
    temperature: float = 0.2
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    context_hint: Optional[str] = None  # 场景提示: podcast, lecture, interview


@dataclass
class SpeakerEmbeddingConfig:
    """说话人嵌入模型配置（pyannote.audio 等）"""

    enabled: bool = True  # 默认启用（speechbrain 无需协议）
    engine: str = "pyannote"  # 嵌入引擎: pyannote | dummy
    model_ref: str = "speechbrain/spkrec-ecapa-voxceleb"  # 默认使用 Apache 2.0 模型
    hf_token: str = ""  # HuggingFace API token（speechbrain 模型无需）
    cache_dir: str = ""  # 模型缓存目录，空 = 默认 {project}/cache/speaker_models/


@dataclass
class CacheConfig:
    """缓存配置"""

    enabled: bool = True
    directory: str = "./cache"
    ttl_separation: int = 86400 * 7  # 7 天
    ttl_transcription: int = 604800  # 7 天
    max_size_mb: int = 5000  # 最大缓存大小 (MB)
    history_retention_days: int = 30  # 历史记录保留天数
    full_pipeline_cache: bool = True  # 是否启用全管道结果缓存


@dataclass
class BoundaryRedundancyConfig:
    """Stage 4.6: 边界滑动窗口冗余识别配置

    针对语速快、词间静音不清晰的场景，
    对低置信度边界做偏移窗口多次 ASR + LLM 语义仲裁。
    """

    enabled: bool = True

    # 置信度阈值
    min_gap_trigger: float = 0.05         # gap < 50ms 触发
    max_energy_slope_trigger: float = 3.0  # 能量斜率 < 3.0 触发
    confidence_threshold: float = 0.5      # score < 此值触发冗余

    # 滑动窗口
    base_overlap_ms: int = 500             # 基础重叠量
    fast_speech_wps: float = 4.0           # 快速语速阈值（词/秒）
    fast_overlap_ms: int = 750             # 快速语速重叠量
    very_fast_overlap_ms: int = 1000       # 极快语速重叠量
    fusion_window_sec: float = 1.0         # 融合窗半宽
    max_workers: int = 3                   # 并行 ASR 线程

    # LLM 仲裁
    llm_model: str = "deepseek-v4-pro"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.1
    llm_timeout: float = 15.0
    auto_apply_confidence: float = 0.8     # > 此值自动应用
    review_threshold: float = 0.5          # 50-80% 标记复核
    fallback_to_rules: bool = True         # 无 LLM 时降级到规则


@dataclass
class DegradationConfig:
    """全局降级模式配置 (文档 5.5.2)

    控制系统级降级策略，用于生产环境排查或降本。

    mode="full":      所有模块启用（默认）
    mode="degraded":  禁用所有 LLM 调用，使用规则替代
    mode="minimal":   仅 VAD + ASR + 规则合并（回退到基线）
    """

    mode: str = "full"                   # "full" | "degraded" | "minimal"
    per_module_timeout: float = 60.0     # 每个模块最大执行秒数
    ffmpeg_timeout: float = 30.0
    llm_api_timeout: float = 15.0


@dataclass
class LoggingConfig:
    """日志配置"""

    level: str = "INFO"
    format: str = "json"  # json | console
    file: str = "logs/pipeline.log"


@dataclass
class StreamingConfig:
    """流式处理配置 (文档 5.12.5)

    控制 Pipeline 离线/流式双模式运行参数。
    """

    chunk_duration: float = 2.0        # 每次处理的音频窗口（秒）
    overlap_duration: float = 0.5      # 窗口重叠（秒）
    max_latency: float = 3.0           # 最大允许延迟（秒）
    # 流式降级
    llm_fallback: str = "local_nlp"   # "local_nlp" | "rule_only"
    vad_engine: str = "silero"         # 流式模式下只用 Silero（最快）


@dataclass
class FeedbackConfig:
    """用户反馈学习配置 (Phase 5)

    基于用户修订字幕的自适应参数学习引擎配置。
    """

    enabled: bool = True                    # 是否启用反馈学习
    user_profile_dir: str = "~/.vocal_subtitle/profiles"
    active_profile: str = "user_default"    # 当前活跃的用户配置

    # 对齐参数
    alignment_min_iou: float = 0.3          # 最小时间交并比
    alignment_min_coverage: float = 0.60    # 最低对齐覆盖率（低于此值拒绝学习）
    alignment_text_weight: float = 0.30     # 字面文本相似度权重
    alignment_semantic_weight: float = 0.35 # 语义相似度权重
    alignment_semantic_enabled: bool = True # 是否启用语义相似度

    # 学习参数
    min_samples_to_learn: int = 3           # 最少样本数才触发参数更新
    base_learn_rate: float = 0.10           # 基础学习率
    max_learn_rate: float = 0.35            # 最大学习率
    param_isolation_enabled: bool = True    # 是否启用参数隔离调整

    # 分级衰减 (天)
    decay_long_term_days: int = 180         # 长期偏好半衰期
    decay_medium_term_days: int = 90        # 中期偏好半衰期（默认）
    decay_short_term_days: int = 60         # 短期环境半衰期

    # 指纹匹配
    fingerprint_enabled: bool = True
    fingerprint_distance_method: str = "mahalanobis"  # "mahalanobis" | "cosine"
    fingerprint_knn_k: int = 3              # 动态阈值 KNN 的 K 值
    fingerprint_min_absolute_similarity: float = 0.70
    fingerprint_relative_margin: float = 0.08

    # 影子模式
    shadow_mode_enabled: bool = False
    shadow_min_runs: int = 10
    shadow_upgrade_threshold: float = 0.05
    shadow_max_duration_days: int = 14

    # 安全机制
    auto_rollback_on_quality_drop: bool = True
    quality_drop_threshold: float = 0.3     # 健康度下降 30% 触发回滚
    oscillation_detection_window: int = 5   # 震荡检测窗口（次）

    # Few-shot
    few_shot_max_examples: int = 3
    few_shot_max_cache: int = 20            # 最大缓存示例数
    few_shot_min_weight_to_inject: float = 0.3  # 注入 Prompt 的最低权重
    few_shot_enabled: bool = True


@dataclass
class PipelineConfig:
    """完整管道配置"""

    # 运行模式
    mode: str = "offline"              # "offline" | "streaming"
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    separation: SeparationConfig = field(default_factory=SeparationConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    merging: MergingConfig = field(default_factory=MergingConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    speaker_role: SpeakerRoleConfig = field(default_factory=SpeakerRoleConfig)
    speaker_embedding: SpeakerEmbeddingConfig = field(
        default_factory=SpeakerEmbeddingConfig
    )
    subtitle: SubtitleBuildConfig = field(default_factory=SubtitleBuildConfig)
    llm_optimize: LLMOptimizeConfig = field(default_factory=LLMOptimizeConfig)

    # 方案〇~七 新模块配置
    macro_chunking: MacroChunkConfig = field(default_factory=MacroChunkConfig)
    ffmpeg_vad: FFmpegVADConfig = field(default_factory=FFmpegVADConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    boundary_refinement: BoundaryRefinementConfig = field(
        default_factory=BoundaryRefinementConfig
    )
    merge_decision: MergeDecisionConfig = field(default_factory=MergeDecisionConfig)
    acoustic_validation: AcousticValidationConfig = field(
        default_factory=AcousticValidationConfig
    )

    boundary_redundancy: BoundaryRedundancyConfig = field(
        default_factory=BoundaryRedundancyConfig
    )
    noise_reduction: NoiseReductionConfig = field(
        default_factory=NoiseReductionConfig
    )
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # 反馈学习
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)

    # Transient / override-only fields — not persisted to YAML profiles
    asr_path: str = ""


# ---------------------------------------------------------------------------
# 配置加载器
# ---------------------------------------------------------------------------

