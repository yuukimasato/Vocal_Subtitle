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
    fast_merge_max_gap: float = 0.30
    llm_decision_min_gap: float = 0.30
    llm_decision_max_gap: float = 1.20
    hard_split_min_gap: float = 1.20

    # 合并约束
    max_combined_duration: float = 5.0
    min_fragment_duration: float = 0.15

    # LLM 降本策略：渐进降级间隙范围
    local_nlp_gap_range: Tuple[float, float] = (0.30, 0.60)  # 此范围内优先本地NLP
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
    # ★ 截尾修复：事件 end 落在连续语音骨架段内部时，允许延长到该骨架段
    # 语音终点（同时钳制到下一事件 start 之前，绝不跨静音/吞下一句）。
    # 过去只标记 possible_truncation 不修复，是 ASR 词尾普遍偏早 60~300ms
    # 时字幕"硬切"的直接原因。
    allow_end_extend: bool = True
    # ★ 吞静音修复：start 后向吸附（吸附到下一个真实语音起点）的限幅。
    # faster-whisper 词起点在换人/换句边界普遍偏早 200~300ms，超过
    # max_snap_distance 的偏差只标记不修，导致下一句开头吞掉静音区。
    # 与 max_snap_distance 分开限幅，需能量确认兜底。
    max_start_snap_distance: float = 0.45
    # 骨架分段独立处理模式：跳过 VAD 分段，直接按声学骨架逐段
    # 独立处理，然后拼接时间轴。每个骨架段是物理隔离的连续语音。
    skeleton_mode: bool = True
    # 聚合 ASR 窗口结果重投影回原始物理成员段：在成员静音间隙处
    # 按词拆分事件并把端点钳制到骨架边界（恢复 v0.2.0 逐段切片
    # 的端点静音对齐契约），同时保留聚合窗口的识别上下文收益。
    reproject_grouped_windows: bool = True
    # 重投影拆分阈值：成员间隙 < 该值视为句内微停顿，不拆分文本，
    # 端点跟随词时间延伸到最后一个成员段（避免"四/个半"式碎片行）。
    member_split_min_gap: float = 0.3
    # 聚合组超过该时长（秒）时，即使间隙 < member_split_min_gap 也要在
    # 成员间隙处继续拆分（展示驱动兜底，<=0 关闭）；与合并级联的
    # max_combined_duration(5.0) 对齐。
    member_split_max_duration: float = 5.0
    # 自适应骨架阈值：按音频噪声底（底部 20% 帧 RMS 中位数）+ 余量动态
    # 推导，钳制 [-45, -30]（与 noise-shadow 建议策略一致）；估计失败或
    # 关闭时回退固定 skeleton_noise_db。
    skeleton_adaptive_noise_db: bool = True
    skeleton_noise_margin_db: float = 10.0
    # 导出骨架段音频供人工验证
    export_skeleton_segments: bool = False
    export_skeleton_dir: str = ""
    # ---- 时间轴仲裁层(2026-09-11 定案,层2) ----
    # 三规则信任策略表:骨架管段级真值,ASR 管词级真值,能量检测当裁判,
    # 文本一致性决定信任级别。命名用 timeline_arbitration,
    # 避免与 LLM 语义仲裁(asr.arbitration)混淆。
    timeline_arbitration: bool = False
    # R1 共识门槛:分段基线与全程 evidence 字符对齐一致、且一致字符数
    # ≥ 该值时才整段信骨架(解除 max_snap_distance 限幅),防短重复短语
    # 假阳性把真实尾音钳掉。
    arbitration_r1_min_overlap_chars: int = 6
    # R1 共识判定的最小字符重合率(对齐重合字符 / 较短方字符数)。
    arbitration_r1_min_similarity: float = 0.85
    # R2 盲区能量确认:True 时用事件周边局部噪声画像,而非全局噪声底,
    # 防止音乐残留等非均匀噪声被误确认为"真语音"。
    arbitration_r2_local_noise: bool = True
    # R1 共识参照文本区域(运行期注入,非用户配置):ASR 管线把全程识别
    # evidence 简化为 (start, end, text) 三元组发布到该字段,validator
    # 读取它做区域字符对齐。postprocess_runner 的 validate 调用点不传参,
    # 共享配置对象是管线层到 validator 的唯一通道;None/空时 R1 不触发。
    arbitration_evidence_regions: Optional[Tuple[Tuple[float, float, str], ...]] = None


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
    routing: str = "segmented"  # segmented | global (compatibility) | global_primary (experimental)
    evidence_enabled: bool = True
    # global_primary 实验路由门禁阈值:全局转录通过物理语音覆盖与文本密度
    # 检查后才允许作为主候选,否则记录 global_primary_fallback_reason 并回退
    # segmented 路径。
    global_primary_min_speech_coverage: float = 0.6
    global_primary_min_chars_per_second: float = 0.5
    global_primary_max_chars_per_second: float = 100.0
    # WhisperX 词级强制对齐开关。默认 True 与既有 GlobalTranscriber 行为一致
    # （后端不支持 align() 时自动跳过）；仅当后端实现 align() 时才会实际调用。
    alignment_enabled: bool = True
    backend: str = "faster-whisper"
    left_context: float = 0.5
    right_context: float = 0.5
    max_window_duration: float = 180.0
    window_overlap: float = 0.5
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
class ASREnginePairConfig:
    """Primary/secondary engine pairing policy for offline review."""

    enabled: bool = True
    primary: str = "auto"
    secondary: str = "auto"
    policy: str = "risk_only"
    same_family_policy: str = "reject"
    fallback_secondary: bool = True
    max_workers: int = 2
    route_version: str = "asr-pair-v1"


@dataclass
class ASRConfig:
    """Stage 4: ASR 识别配置"""

    engine: str = "auto"  # auto | faster-whisper | whisper-cpp | funasr | qwen
    model: str = "large-v3"
    qwen_model_path: Optional[str] = None
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
    engine_pair: ASREnginePairConfig = field(default_factory=ASREnginePairConfig)
    # Hallucination filter thresholds
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    hallucination_filter_version: str = "v1"


@dataclass
class EvidenceReviewConfig:
    """Phase 0-3 evidence review and conservative fallback policy."""

    enabled: bool = True
    shadow_mode: bool = True
    authoritative_mode: bool = False
    context_reasr_enabled: bool = False
    # global 候选角色准入（优化方案 8.1）：默认 global 只作为 signal 参与风险
    # 评分；显式开启后才允许通过校验的 global 候选进入替代候选集合。
    global_alternative_enabled: bool = False
    qwen_enabled: bool = False
    forced_aligner_enabled: bool = False
    sed_enabled: bool = False
    semantic_review_enabled: bool = False
    unresolved_keeps_candidate: bool = True
    require_multi_source_drop: bool = True
    fallback_to_segmented: bool = True
    local_recovery_enabled: bool = True
    local_recovery_max_attempts: int = 3
    local_recovery_min_confidence: float = 0.5
    local_recovery_context_seconds: float = 0.5
    local_recovery_request_tolerance: float = 0.15
    qwen_model_path: Optional[str] = None
    forced_aligner_model_path: Optional[str] = None
    sed_model_path: Optional[str] = None
    review_device: str = "auto"
    allow_remote_model_download: bool = False
    left_context: float = 0.8
    right_context: float = 0.8
    max_group_duration: float = 12.0
    max_window_duration: float = 15.0
    medium_threshold: float = 0.25
    high_threshold: float = 0.50
    critical_threshold: float = 0.75
    max_workers: int = 2
    window_timeout_seconds: Optional[float] = 60.0
    review_policy_version: str = "review-policy-v1"
    cover_policy: str = ""
    engine_policy: str = ""
    risk_policy_version: str = "risk-policy-v1"
    decision_policy_version: str = "decision-policy-v1"
    evidence_schema_version: str = "evidence-v1"
    golden_quality_gate_version: str = "golden-quality-v1"


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
    # ---- 说话人身份主干(2026-09-11 定案,层1) ----
    # turns 前置:全局 diarization 在分离之后立即运行一次(结果进缓存),
    # turns 贯通 ctx 供骨架×turns 求交、合并硬约束与词级切分使用;
    # 置 false 回到后处理事件级聚类现状。
    early_turns: bool = False
    # 词级后切分:字幕事件在 turn 翻转点按最近词间隙切开,
    # 两段各自继承 turn 标签(标签先天正确);重叠区标 overlapped。
    word_split_on_turn: bool = False
    # 单说话人短路:全局 turns 归一后 ≤1 个说话人时跳过切分与
    # 多说话人路径(TTS/口播素材零额外开销)。
    single_speaker_shortcut: bool = True


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
    preflight_mode: str = "report"       # "report" | "enforce" — 预检失败时仅报告还是阻断
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

    # 运行时应用（D38：overrides 接线，默认关——维持纯收集语义）
    apply_overrides_on_run: bool = False    # 任务提交构建配置时合并 active_profile 的 overrides

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

    # 编辑日志摄取（edit-journal-v1 第三触发通道）
    journal_enabled: bool = True            # 是否消费编辑日志数据

    # sink 保留策略（D30：防止 cache/journal_sink/ 在无人 ingest 时无限堆积）
    journal_sink_retention: str = "archive"  # 消费成功后源文件处理：archive=归档到 consumed/ | delete=直接删除
    journal_sink_ttl_days: int = 30          # 未消费文件 TTL 兜底清理（天；<=0 禁用清理）

    # V3 触发机制（D16：只定机制与可配置阈值，不定数值——等 V1 数据分布校准）
    v3_trigger_min_samples: Optional[int] = None       # D2+journal 样本数下限
    v3_trigger_min_coverage: Optional[float] = None    # 对齐/出处覆盖率下限 [0,1]
    v3_trigger_max_conflict_rate: Optional[float] = None  # 参数冲突率上限 [0,1]


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
    evidence_review: EvidenceReviewConfig = field(default_factory=EvidenceReviewConfig)
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
