# Vocal Subtitle 技术架构文档

> 版本: 0.1.0 | 更新日期: 2026-07-07 | 协议: MIT

---

## 目录

1. [系统概览](#1-系统概览)
2. [核心 Pipeline 架构](#2-核心-pipeline-架构)
3. [模块详细设计](#3-模块详细设计)
4. [数据流与数据模型](#4-数据流与数据模型)
5. [七层优化方案 (Plan 0-7)](#5-七层优化方案-plan-0-7)
6. [引擎抽象与多态设计](#6-引擎抽象与多态设计)
7. [运行模式架构](#7-运行模式架构)
8. [缓存架构](#8-缓存架构)
9. [Web GUI 与实时通信](#9-web-gui-与实时通信)
10. [LLM 集成架构](#10-llm-集成架构)
11. [自适应反馈学习 (Phase 5)](#11-自适应反馈学习-phase-5)
12. [配置系统](#12-配置系统)
13. [测试架构](#13-测试架构)
14. [部署与运维](#14-部署与运维)

---

## 1. 系统概览

### 1.1 项目定位

Vocal Subtitle 是一个**模块化多阶段音频处理管道**，将原始音视频文件自动转换为带精确时间戳的字幕文件 (SRT/VTT/ASS)。所有依赖均使用 MIT/Apache 2.0/BSD 宽松协议，确保可自由商用。

### 1.2 技术栈

```
┌──────────────────────────────────────────────────┐
│  应用层    │ Click CLI  │ FastAPI WebGUI  │ Python API  │
├──────────────────────────────────────────────────┤
│  编排层    │ Pipeline (pipeline.py)  │ StreamingPipeline  │
├──────────────────────────────────────────────────┤
│  处理层    │ Separation │ VAD │ Merging │ ASR │ Mapping  │
├──────────────────────────────────────────────────┤
│  优化层    │ Plans 0-7 │ LLM Optimizer │ Diarization  │
├──────────────────────────────────────────────────┤
│  基础设施  │ diskcache │ SQLite │ structlog │ ffmpeg  │
├──────────────────────────────────────────────────┤
│  引擎后端  │ CTranslate2 │ PyTorch │ ONNX │ TensorFlow  │
└──────────────────────────────────────────────────┘
```

### 1.3 系统边界

- **输入**: 音频文件 (WAV/MP3/FLAC/AAC/OGG/M4A) 或视频文件 (MP4/MKV/AVI/MOV/WebM)
- **输出**: 字幕文件 (SRT/VTT/ASS) + 可选音频导出 (人声/伴奏 WAV)
- **外部依赖**: ffmpeg (系统级，通过 subprocess 调用，非链接)
- **可选外部服务**: LLM API (DeepSeek/OpenAI 等兼容协议)

---

## 2. 核心 Pipeline 架构

### 2.1 Pipeline 生命周期

```
┌─────────────────────────────────────────────────────────┐
│                   Pipeline.run()                        │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ 输入校验  │→│ 缓存查询  │→│ 阶段执行              │→ 输出
│  │+ 文件哈希│  │L1→L2→L3  │  │1→0→1.5→2→2.5→3→     │  │
│  └──────────┘  └──────────┘  │3.5→4→4.5→5→Post      │  │
│                      ↓ 命中    └──────────────────────┘  │
│                  直接返回缓存结果                          │
│                                                         │
│  注意：Stage 0 (宏观切块) 在 Stage 1 (分离) 之后执行，   │
│        因为需要先获取完整 vocals 才能进行切分。            │
└─────────────────────────────────────────────────────────┘
```

### 2.2 阶段编排

`Pipeline.run()` 方法按序编排以下阶段：

```python
# pipeline.py 核心流程 (简化伪代码)
def run(input_path, output_path, **overrides):
    # 0. 输入加载 + 缓存查询
    audio, sr = AudioUtils.load(input_path)
    file_hash = compute_file_hash(input_path)

    # Stage 1: 人声分离 (在切块之前, 获取完整 vocals)
    vocals_path = separation_engine.separate(input_path)

    # Stage 1.5: 音频预处理 (可选)
    vocals = audio_preprocessor.process(vocals)

    # Stage 0: 宏观静音切块 (Plan 0, >3min 长音频)
    if audio_duration > 180s:
        chunks = macro_chunker.split(vocals_path, audio, sr)
    else:
        chunks = [single_full_chunk]

    for chunk in chunks:
        # Stage 2: VAD 检测 + Plan 1 ffmpeg 并行
        segments_silero = silero_vad.detect(chunk.audio)
        segments_ffmpeg = ffmpeg_vad.detect(chunk.audio)  # ThreadPoolExecutor 并行

        # Plan 2: 三方法边界融合 (可选)
        segments = boundary_fusion.fuse(segments_silero, segments_ffmpeg, rms)

        # Stage 3: 片段合并 + Plan 3 段内预切分
        segments = merge_strategy.merge(segments)

        # Stage 3.5: 说话人分离 (可选)
        segments = diarization.diarize(segments, chunk.audio)

        # Stage 4: ASR 识别
        fragments = asr_engine.transcribe(segments)

        # Stage 4.5: 边界精修 (Plan 4) + 冗余识别 + LLM 仲裁
        fragments = boundary_refiner.refine(fragments, chunk.audio)
        low_conf_boundaries = boundary_confidence.assess(fragments)
        if low_conf_boundaries:
            fragments = boundary_reasr.re_recognize(fragments, chunk.audio)
            fragments = boundary_arbitration.arbitrate(fragments)
        fragments = role_labeler.label(fragments)  # 可选

    # Stage 5: 时间轴映射 + 事件去重
    events = time_mapper.map(all_fragments)

    # 后处理层 (顺序经精心设计)
    # 0. 事件级说话人聚类 (全局集合, 替代段级 diarization)
    events = event_speaker_clustering.cluster(events, audio, sr)
    events = role_labeler.label(events)
    # 1. Plan 6: 帧级无缝衔接 (不改变边界)
    events = apply_frame_seamless_stitching(events)
    # 2. Plan 5: LLM 语义合并 (改变边界, 必须在声学校验前)
    events = llm_merge_engine.merge(events, audio, sr)
    # 3. Plan 7: 声学标尺校验 (最终关卡)
    events, report = acoustic_validator.validate(events, vocals_path)
    events = end_time_validator.validate(events)
    # 4. 可选 LLM 后处理
    events = llm_optimizer.optimize(events)

    # 输出
    subtitle_builder.build(events, output_format).save(output_path)
```

### 2.3 模块依赖图

```
pipeline.py ─────────────────────────────────────────────
    │
    ├── config.py (PipelineConfig, ConfigLoader)
    ├── pipeline_context.py (PipelineContext)
    │
    ├── separation/
    │   ├── base.py (SeparationEngine ABC)
    │   ├── uvr_engine.py ─── audio-separator (ONNX)
    │   ├── spleeter_engine.py ─── spleeter (TF)
    │   └── openunmix_engine.py ─── openunmix (PyTorch)
    │
    ├── vad/
    │   ├── base.py (VADEngine ABC, SpeechSegment)
    │   ├── silero_vad.py ─── torch + torchaudio
    │   ├── webrtc_vad.py ─── webrtcvad
    │   ├── ten_vad.py (energy-based fallback)
    │   ├── ffmpeg_vad.py ─── ffmpeg subprocess
    │   └── boundary_fusion.py (Plan 2)
    │
    ├── merging/
    │   ├── merge_strategy.py (MergeStrategy + MergingConfig)
    │   └── llm_merge_engine.py (Plan 5)
    │
    ├── diarization/
    │   ├── base.py (DiarizationEngine ABC)
    │   ├── feature_extractor.py ─── librosa
    │   ├── speaker_clusterer.py ─── scikit-learn
    │   ├── speaker_embedding.py ─── SpeechBrain / pyannote
    │   └── role_labeler.py ─── LLM API
    │
    ├── feedback/
    │   ├── aligner.py (SubtitleAligner)
    │   ├── diff_analyzer.py (DiffAnalyzer)
    │   ├── param_learner.py (ParamLearner)
    │   ├── audio_fingerprint.py (AudioFingerprinter)
    │   ├── health_scorer.py (HealthScorer)
    │   ├── conflict_detector.py (ConflictDetector)
    │   ├── few_shot_builder.py (FewShotBuilder)
    │   ├── impact_estimator.py (ImpactEstimator)
    │   ├── shadow_mode.py (ShadowMode)
    │   └── user_profile.py (UserProfileManager)
    │
    ├── asr/
    │   ├── base.py (ASREngine ABC, TranscriptionSegment)
    │   ├── faster_whisper_engine.py ─── CTranslate2
    │   ├── whisper_cpp_engine.py ─── whisper.cpp subprocess
    │   ├── funasr_engine.py ─── FunASR
    │   ├── boundary_refiner.py (Plan 4: 双向精修)
    │   ├── boundary_confidence.py (边界置信度评估)
    │   ├── boundary_reasr.py (滑动窗口冗余 ASR)
    │   ├── boundary_arbitration.py (LLM 语义仲裁)
    │   └── text_normalizer.py
    │
    ├── mapping/
    │   ├── time_mapper.py (TimeMapper, SubtitleEvent)
    │   ├── subtitle_builder.py ─── pysubs2
    │   └── end_time_validator.py
    │
    ├── utils/
    │   ├── audio_utils.py (AudioUtils)
    │   ├── cache_manager.py ─── diskcache
    │   ├── persistence_manager.py (持久化文件管理)
    │   ├── session_manager.py (会话管理: hash 目录 + 去重)
    │   ├── file_hasher.py (SHA256)
    │   ├── gpu_detector.py (CUDA/MPS/CPU)
    │   ├── progress.py (ProgressManager)
    │   ├── task_history.py ─── SQLite
    │   └── logger.py ─── structlog
    │
    └── webui/
        ├── app.py (FastAPI factory)
        ├── api.py (REST endpoints, ~1705 行)
        ├── websocket.py (WebSocketManager)
        ├── cli_runner.py (GUI 启动入口)
        ├── models.py (Pydantic)
        └── static/
            ├── index.html (SPA frontend)
            └── speaker-embedding-guide.html
```

---

## 3. 模块详细设计

### 3.1 Pipeline 编排器 (`pipeline.py`)

**核心类**: `Pipeline`, `PipelineStats`

**职责**:
- 阶段生命周期管理（Init → Run → Post-process）
- 引擎延迟初始化（工厂方法模式，仅在使用时加载 ML 模型）
- 跨阶段数据流通过 `PipelineContext` 传递
- 全局进度事件通过 `ProgressManager` 分发
- 缓存键计算（文件哈希 + 配置哈希）
- 异常处理 + 降级策略 (degradation.mode)
- 反馈学习集成 (接受外部 auto_events 用于对齐分析)

**关键设计决策**:
- 引擎实例化延迟到首次调用 (`_get_*_engine()`)，避免启动时加载所有 ML 模型
- 进度回调支持 CLI (tqdm) 和 WebSocket 双路分发
- 离线模式三种路径：单块路径、多块路径 (宏观切块后)、骨架分段路径
- 宏观切块 (Plan 0) 在 Stage 1 (人声分离) 之后处理——先获取完整人声，再 >2s 静音处切分，每个 chunk 独立走 Stage 2-4.5，最后全局拼接
- 后处理统一入口 `_post_process_events()`：事件级聚类 → 帧级无缝 → LLM 合并 → 声学校验 → LLM 优化，三种路径共用
- `events` 属性公开暴露，供反馈学习模块 (feedback/) 获取自动版字幕用于对齐分析

### 3.2 配置管理 (`config.py`)

**核心类**: `ConfigLoader`, 24 个 `@dataclass` 配置类

**配置类层级**:
```
PipelineConfig (顶级容器)
├── SeparationConfig          (Stage 1: 人声分离)
├── VADConfig                 (Stage 2: VAD 检测)
├── FFmpegVADConfig           (Plan 1: ffmpeg VAD)
├── FusionConfig              (Plan 2: 三方法融合)
├── MergingConfig             (Stage 3: 片段合并)
├── BoundaryRefinementConfig  (Plan 4: 边界精修)
├── BoundaryReASRConfig?      → BoundaryRedundancyConfig (Plan 4: 冗余 ASR)
├── BoundaryScoreConfig       (Plan 4: 边界置信度评分)
├── MergeDecisionConfig       (Plan 5: LLM 合并决策)
├── MacroChunkConfig          (Plan 0: 宏观切块)
├── AcousticValidationConfig  (Plan 7: 声学标尺)
├── DiarizationConfig         (Stage 3.5: 说话人分离)
├── SpeakerEmbeddingConfig    (Stage 3.5: 声学嵌入模型)
├── ASRConfig                 (Stage 4: ASR 识别)
├── SpeakerRoleConfig         (Stage 4.5: 角色标注)
├── SubtitleBuildConfig       (Stage 5: 字幕构建)
├── GapHandlingConfig         (Stage 5: 间隙处理)
├── NoiseReductionConfig      (预处理: 降噪)
├── LLMOptimizeConfig         (LLM 后处理)
├── CacheConfig               (缓存配置)
├── PersistenceConfig?        → (合并入 CacheConfig / 独立持久化)
├── DegradationConfig         (降级模式)
├── StreamingConfig           (流式处理参数)
├── LoggingConfig             (日志配置)
└── FeedbackConfig            (Phase 5: 自适应反馈学习)
```

**模板系统**:
- 5 个 YAML 模板文件位于 `configs/`
- 模板通过 `--profile <name>` 选择，可叠加 `--config custom.yaml` 自定义
- 支持运行时 `overrides` 参数覆盖任意配置键

### 3.3 数据上下文 (`pipeline_context.py`)

**核心类**: `PipelineContext`, `ASRFragment`, `NoiseProfile`

作为 Pipeline 各阶段间的统一数据载体，避免参数爆炸：

```python
@dataclass
class PipelineContext:
    audio: np.ndarray              # 原始音频 (降噪后)
    original_audio: np.ndarray     # 未降噪的原始音频
    sample_rate: int               # 统一采样率 (16000)
    segments: List[SpeechSegment]  # VAD 分段结果
    asr_fragments: List[ASRFragment]  # ASR 识别结果 (含 word timestamps)
    acoustic_skeleton: List[SpeechSegment]  # ffmpeg 声学骨架 (Plan 7)
    noise_profile: NoiseProfile    # 环境噪声特征
    diagnostics: dict              # 诊断信息
```

### 3.4 进度管理 (`utils/progress.py`)

**核心类**: `ProgressManager`

支持双路分发模式：
- **CLI 模式**: 通过 tqdm monkey-patch 捕获进度 → 标准进度条
- **WebSocket 模式**: 回调 → `ws_manager.broadcast()` → 前端实时渲染

消息协议：
```json
{
  "type": "stage_start|progress|stage_finish|complete|error",
  "stage": "separation|vad|merging|asr|mapping|optimize",
  "progress": 0.0-1.0,
  "message": "描述文本",
  "timestamp": 1234567890.123
}
```

---

## 4. 数据流与数据模型

### 4.1 核心数据结构

#### SpeechSegment (`vad/base.py`)
```python
@dataclass
class SpeechSegment:
    start: float           # 开始时间 (秒)
    end: float             # 结束时间 (秒)
    confidence: float      # VAD 置信度 (0-1)
    audio: np.ndarray      # 音频片段
    speaker_id: int = -1   # 说话人 ID (diarization 后)
    source: str = "silero" # 来源: silero/ffmpeg/fusion
```

#### TranscriptionSegment (`asr/base.py`)
```python
@dataclass
class TranscriptionSegment:
    text: str              # 识别文本
    start: float           # 开始时间
    end: float             # 结束时间
    words: List[WordTimestamp]  # 词级时间戳
    avg_logprob: float     # 平均对数概率
    language: str          # 检测语言
```

#### WordTimestamp (`asr/base.py`)
```python
@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float      # 词级置信度
```

#### SubtitleEvent (`mapping/time_mapper.py`)
```python
@dataclass
class SubtitleEvent:
    index: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp]
    original_text: str           # LLM 优化前的原文 (用于前端对比)
    speaker_id: Optional[int]
    speaker_label: Optional[str] # 角色标注: 主持人/嘉宾/旁白
```

### 4.2 各阶段数据流转

```
Stage 1: [Input Path] → Separation → [vocals.wav + accompaniment.wav paths]
Stage 0: [vocals.wav] → Macro Chunker → [List[AudioChunk]] (长音频自动切分)
Stage 1.5: [Audio ndarray] → Audio Preprocessor → [denoised audio] (可选)
Stage 2: [Audio] → VAD (Silero + ffmpeg 并行) → [List[SpeechSegment]]
Stage 2.5: [Segments] → Boundary Fusion → [Fused Segments] (Plan 2, 可选)
Stage 3: [List[SpeechSegment]] → Merge Strategy → [List[SpeechSegment]] (合并+预切分)
Stage 3.5: [Segments + Audio] → Diarization → [Segments with speaker_id] (可选)
Stage 4: [Segments + Audio] → ASR → [List[TranscriptionSegment]]
Stage 4.5: [TranscriptionSegments] → Boundary Refiner + Confidence + ReASR + Arbitration
          → [Refined TranscriptionSegments]
Stage 5: [All Segments] → TimeMapper → [List[SubtitleEvent]] (去重 + 跨块拼接)
Post 0: [SubtitleEvents] → Event Speaker Clustering → [Events with speaker_id]
Post 1: [SubtitleEvents] → Frame Seamless Stitching (Plan 6) → [Seamless Events]
Post 2: [SubtitleEvents] → LLM Merge Engine (Plan 5) → [Merged Events]
Post 3: [SubtitleEvents] → Acoustic Validator (Plan 7) → [Validated Events + Diagnostics]
Post 4: [SubtitleEvents] → LLM Optimizer → [Optimized Events] (可选)
Output: [SubtitleEvents] → SubtitleBuilder → [SRT/VTT/ASS file]
```

---

## 5. 七层优化方案 (Plan 0-7)

### 5.1 Plan 0: 宏观静音切块 (`macro_chunker.py`)

**问题**: 长音频 (>3min) 在全局范围内处理效率低、内存占用高。

**方案**: 在 >2s 静音处切分音频，每个 chunk 独立处理，最后全局拼接。

```
原始音频 (10min)
    │
    ├── 检测 >2s 静音断点
    │
    ├── Chunk 1 (0:00-3:15) → 独立 Pipeline → Subtitles 1
    ├── Chunk 2 (3:15-6:30) → 独立 Pipeline → Subtitles 2
    ├── Chunk 3 (6:30-10:00) → 独立 Pipeline → Subtitles 3
    │
    └── 全局拼接 (时间偏移 + speaker ID 偏移 + overlap 去重)
```

**配置**:
```yaml
macro_chunking:
  enabled: true
  auto_enable_threshold: 180.0   # >3min 自动启用
  silence_threshold_db: -30
  min_silence_duration: 2.0
  target_chunk_duration: 60.0
  max_chunk_duration: 180.0
  overlap_ms: 200                # chunk 间重叠
  recursive: true                # 递归切分
```

### 5.2 Plan 1: ffmpeg silencedetect 并行 VAD (`vad/ffmpeg_vad.py`)

**问题**: 单一 VAD 引擎有盲区 (Silero 对低音量语尾不敏感，ffmpeg 对短停顿过度敏感)。

**方案**: Silero VAD 与 ffmpeg silencedetect 并行运行，互补盲区。

```
┌─────────────────┐     ┌──────────────────────┐
│  Silero VAD      │     │  ffmpeg silencedetect │
│  (神经网络)      │     │  (信号处理)           │
└────────┬─────────┘     └──────────┬───────────┘
         │                          │
         └──────────┬───────────────┘
                    │
         ┌──────────▼───────────┐
         │  unified_ffmpeg_pass  │ ← 统一管理 ffmpeg 调用 (与 Plan 7 共用)
         │  (ThreadPoolExecutor) │
         └──────────────────────┘
```

**配置**:
```yaml
vad:
  ffmpeg_enabled: true
  ffmpeg_noise_db: -35.0
  ffmpeg_weight: 0.4
```

### 5.3 Plan 2: 三方法边界融合 (`vad/boundary_fusion.py`)

**问题**: Silero + ffmpeg 两种方法边界有分歧，人工查看成本高。

**方案**: 10ms 网格投票制，3 种方法 (Silero + ffmpeg + RMS Energy) 2/3 共识决定。

```
音频 → 离散化 (10ms 网格)
    │
    ├── Silero VAD 投票: 每个网格 0/1
    ├── ffmpeg 投票:     每个网格 0/1
    └── RMS Energy 投票: 每个网格 0/1
    │
    ▼
每个网格: sum(votes) >= 2 → 语音
          sum(votes) == 1 → 低置信度语音
          sum(votes) == 0 → 静音
    │
    ▼
连续语音网格 → 语音段 → 按置信度加不同 padding
  - 高置信 (3/3): +30ms padding
  - 中置信 (2/3): +30ms padding
  - 低置信 (1/3): +120ms padding (更保守)
```

**配置**:
```yaml
fusion:
  enabled: false          # 默认关闭
  grid_resolution: 0.01   # 10ms
  min_consensus: 2        # 2/3 多数决
  high_conf_padding: 0.03
  low_conf_padding: 0.12
  min_speech_duration: 0.25
```

### 5.4 Plan 3: 段内静音预切分 (merged into `merging/merge_strategy.py`)

**问题**: VAD 合并后仍有段内长静音 (0.5s+)，ASR 在此处可能产生错误。

**方案**: 段内检测 >0.55s 静音间隙 → 预切分，保护单词不被切断。

```
原始段: [0.0s ─── 静音 ─── 静音 ─── 5.0s]
                    ↓ 检测 >0.55s 静音间隙
切分段: [0.0s ─ 2.0s]  [2.7s ─ 5.0s]
```

**配置**:
```yaml
merging:
  pre_split_silence: true
  pre_split_threshold: 0.55
  min_fragment_duration: 0.25
  protect_single_word: true
  min_word_gap_ms: 80        # 单词内部允许的最大静音
```

### 5.5 Plan 4: ASR 边界双向精修 (`asr/boundary_refiner.py`)

**问题**: VAD 边界可能截断 ASR 识别出的完整词汇。

**方案**: 利用 ASR 词级时间戳 (word_timestamps) + 三帧能量斜率校验，双向调整边界。

```
修正前:  [VAD: 1.0s ───── 3.5s]  ASR words: ["今天" 1.1s, "天气" 1.5s, "很好" 1.9s]
                                          ↑ 词尾 1.9s，远早于 VAD 3.5s

操作:  收缩 end → 词尾 + 3帧能量确认 + 安全 margin
修正后:  [1.0s ───────── 2.2s]  (缩短 1.3s 无效静音)
```

**双向策略**:
- **Start (向左)**: 检查词首时间戳 → 如早于 VAD start → 向外扩展
- **Start (向右)**: 检查词首时间戳 → 如晚于 VAD start → 向内收缩 (保守)
- **End (向左)**: 检查词尾时间戳 → 如早于 VAD end → 向内收缩 (慎用)
- **End (向右)**: 检查词尾时间戳 → 如晚于 VAD end → 向外扩展

**配置**:
```yaml
boundary_refinement:
  enabled: true
  max_shrink_ms: 80
  max_extend_ms: 100
  check_frames: 3             # 三帧能量斜率校验
  shrink_end_enabled: false   # end 向内收缩默认关闭 (语尾低能量不可靠)
  min_boundary_confidence: 0.5
```

### 5.6 Plan 5: LLM 语义合并 (`merging/llm_merge_engine.py`)

**问题**: 短间隙 (<1.2s) 的两句话可能是同一句话的断续，也可能是不同说话人的交替。纯声学规则无法判断。

**方案**: 三层级联决策流水线 (Fast-Slow Path)。

```
相邻字幕间隙
    │
    ├── gap < 200ms ──────────→ 快路径：规则强制合并 (<1ms, 纯 CPU)
    │
    ├── gap 200-600ms ────────→ 本地 NLP：sentence-transformers (~30ms, CPU)
    │                            │
    │                      语义相似度 > 阈值? ──→ 合并
    │                      语义相似度 ≤ 阈值? ──→ 不合并
    │                      无法裁决? ──→ 升级到云端 LLM
    │
    ├── gap 600-1200ms ───────→ 云端 LLM：API 调用 (~1s)
    │                            │
    │                      LLM 裁决合并 / 不合并
    │                      API 不可用 → 规则降级
    │
    └── gap > 1200ms ─────────→ 硬规则：强制不合并
```

**语义边界检测** (降级规则):
- 编号列表检测 (`^\d+[.\)]\s`, `^[一二三四五六七八九十][、，.]`)
- 段落标题检测 (`^(Summary|Example|Effective Communication)`)
- 段落分隔符检测 (句末 `with courtesy.` 等)

**配置**:
```yaml
merge_decision:
  fast_merge_max_gap: 0.20
  llm_decision_min_gap: 0.20
  llm_decision_max_gap: 1.20
  hard_split_min_gap: 1.20
  local_nlp_gap_range: [0.15, 0.60]
  cloud_llm_gap_range: [0.60, 1.20]
  llm_tier: "cascading"      # cascading | all_llm | rule_only
  llm_fallback_to_rules: true
```

### 5.7 Plan 6: 帧级无缝衔接 (embedded in `llm_merge_engine.py`)

**问题**: 相邻字幕间有微小间隙 (几帧)，播放器在间隙期无字幕显示，造成闪烁感。

**方案**: 非句尾字幕的 end time 扩展至下一句 start time。

```
修正前:  [sub 1: 0.0-2.5s]  [gap 0.15s]  [sub 2: 2.65-5.0s]
                                              ↑ 0.15s 无字幕闪烁

修正后:  [sub 1: 0.0-2.65s]  [sub 2: 2.65-5.0s]  ← 无缝衔接
                                                        ↑ 仅限非句尾
句尾字幕 (.!?。！？结尾) 不扩展，保持自然停顿
```

**配置**:
```yaml
subtitle:
  frame_seamless: true
  max_stitch_gap: 0.12       # 最多衔接 120ms (超过为自然停顿)
```

### 5.8 Plan 7: 声学标尺校验 (`acoustic_validator.py`)

**问题**: 多轮后处理后的字幕边界可能与原始音频的物理静音不一致。

**方案**: 以 ffmpeg silencedetect 声学骨架为物理基准 (ground truth)，双向 snap 修正。

```
                    ffmpeg 声学骨架 (物理基准)
                    ├── speech: 0.5-2.3s ──┤
SubtitleEvent:  [0.7s ────── 2.2s]
                      ↑                  ↑
                  偏差 200ms          偏差 100ms
                      │                  │
                snap 到 0.53s      snap 到 2.30s

修正逻辑:
- start 向前吸附: 如果 ffmpeg 骨架 start 早于字幕 start ≤ max_snap_distance
  → snap start = skeleton.start + snap_start_margin
- end 向后吸附: 如果 ffmpeg 骨架 end 晚于字幕 end ≤ max_snap_distance
  → snap end = skeleton.end - snap_end_margin
- RMS override: 如果能量 RMS > threshold，跳过吸附 (避免切到语音中间)
```

**配置**:
```yaml
acoustic_validation:
  enabled: true
  skeleton_mode: true            # 骨架分段独立处理
  max_snap_distance: 0.15
  snap_start_margin: 0.03
  snap_end_margin: 0.003
  allow_end_shorten: true
  allow_start_pull_earlier: true
```

---

## 6. 引擎抽象与多态设计

### 6.1 设计模式

所有处理引擎遵循 **策略模式 (Strategy Pattern)**，通过 ABC 抽象基类定义统一接口：

```
SeparationEngine (ABC)          VADEngine (ABC)           ASREngine (ABC)
├── separate(audio) → Result    ├── detect(audio) → [Seg]  ├── transcribe(audio) → [Seg]
├── UVREngine                   ├── SileroVAD              ├── FasterWhisperEngine
├── SpleeterEngine              ├── WebRTCVAD              ├── WhisperCppEngine
└── OpenUnmixEngine             └── TENVAD                 └── FunASREngine

DiarizationEngine (ABC)
├── diarize(segments, audio) → [ClusteredSegment]
└── SpeakerDiarizer (agglomerative clustering)
```

### 6.2 引擎延迟初始化

Pipeline 使用工厂方法模式 + 懒加载，避免启动时加载所有 ML 模型：

```python
def _get_separation_engine(self) -> SeparationEngine:
    if self._separation_engine is not None:
        return self._separation_engine  # 已加载，直接返回
    # 首次调用时才初始化，加载 ML 模型
    if engine_name == "uvr":
        self._separation_engine = UVREngine()
    # ...
    return self._separation_engine
```

### 6.3 引擎选型矩阵

| 标准 | UVR (BS-RoFormer) | Spleeter | Open-Unmix | Silero VAD | WebRTC VAD | faster-whisper | whisper.cpp | FunASR |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 质量 | ★★★★★ | ★★★ | ★★★★ | ★★★★ | ★★★ | ★★★★★ | ★★★★ | ★★★★ |
| 速度 | ★★★ | ★★★ | ★★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★ |
| 内存 | 2-4GB | 2-3GB | 3-4GB | 500MB | 10MB | 2-4GB | 500MB-2GB | 2-3GB |
| GPU | ONNX | TF | PyTorch | PyTorch | 不需要 | CTranslate2 | 不需要 | PyTorch |
| Python版本 | 3.10+ | <3.12 | 3.10+ | 3.10+ | 3.10+ | 3.10+ | 3.10+ | 3.10+ |
| 语言 | 通用 | 通用 | 通用 | 通用 | 通用 | 99+语言 | 99+语言 | 中文优化 |
| 默认 | ✅ | | | ✅ | | ✅ | | |

---

## 7. 运行模式架构

### 7.1 离线模式 (Offline, 默认)

```
输入 → [完整 Pipeline] → 输出
     → Plan 0-7 全部可用
     → 全局声学标尺可用
     → 批量 LLM 合并可用
     → 适合：音视频文件批处理
```

### 7.2 流式模式 (Streaming)

```
实时音频流 → [滑动窗口 Pipeline] → 实时字幕流
     → 窗口大小: 2s (可配)
     → 窗口重叠: 0.5s (可配)
     → 最大延迟: 3s (可配)
     → 自动降级全局依赖模块
```

**模块可用性**:

| 模块 | 离线 | 流式 | 说明 |
|------|:----:|:----:|------|
| Plan 0 (宏观切块) | ✅ | ❌ | 无全局视角 |
| Plan 1 (ffmpeg VAD) | ✅ | ⚠️ | 窗口内运行 |
| Plan 2 (三方法融合) | ✅ | ❌ | 仅 Silero VAD |
| Plan 3 (段内预切分) | ✅ | ✅ | 窗口内 |
| Plan 4 (边界精修) | ✅ | ✅ | 窗口内 |
| Plan 5 (LLM 合并) | ✅ | ⚠️ | 仅本地 NLP |
| Plan 6 (帧无缝) | ✅ | ✅ | 窗口内 |
| Plan 7 (声学标尺) | ✅ | ❌ | 无全局标尺 |

### 7.3 骨架分段模式 (Skeleton Mode)

跳过 VAD，直接以 ffmpeg 声学骨架为分段依据：

```
输入 → ffmpeg silencedetect (敏感模式: -40dB) → 骨架分段
     → 每段独立 ASR → 直接构建字幕
     → 适用场景: 纯净人声、已有高质量人声分离结果
```

### 7.4 降级模式

通过 `degradation.mode` 控制应对异常：

```yaml
degradation:
  mode: "full"          # 所有模块
  # mode: "degraded"    # 关闭 LLM，仅本地规则
  # mode: "minimal"     # 仅 VAD + ASR + 规则合并
  per_module_timeout: 60
  ffmpeg_timeout: 30
  llm_api_timeout: 15
```

---

## 8. 缓存架构

### 8.1 三层缓存

```
┌─────────────────────────────────────────────────────┐
│ L1: 分离缓存 (diskcache)                            │
│ Key: SHA256(file) + engine + model                  │
│ Value: vocals.wav + accompaniment.wav paths         │
│ TTL: 7 天 (可配)                                     │
├─────────────────────────────────────────────────────┤
│ L2: 转录缓存 (diskcache)                            │
│ Key: SHA256(segment_audio + model + language)        │
│ Value: List[TranscriptionSegment]                   │
│ TTL: 7 天 (可配)                                     │
├─────────────────────────────────────────────────────┤
│ L3: 任务历史 (SQLite)                                │
│ Key: task_id                                        │
│ Value: 完整 PipelineResult + events + timings        │
│ TTL: 永久 (手动清理)                                  │
└─────────────────────────────────────────────────────┘
```

### 8.2 缓存键生成

```python
# L1 分离缓存键
cache_key = f"{file_hash}:{engine}:{model}:{sample_rate}"

# L2 转录缓存键
cache_key = f"{segment_hash}:{model}:{language}:{beam_size}"

# L3 全量 Pipeline 缓存键
cache_key = f"{file_hash}:{config_hash}"
```

### 8.3 缓存目录结构

```
cache/
├── models/                   # 下载的 ML 模型权重
├── separation/               # L1: diskcache 分离缓存
├── transcription/            # L2: diskcache 转录缓存
├── persistent_files/         # 持久化音频/字幕文件 (按 hash 分目录)
├── uploads/                  # Web GUI 上传临时文件
├── speaker_models/           # 说话人识别模型缓存
├── task_history.db           # L3: SQLite 任务历史数据库
└── persistence_settings.json # 服务端配置持久化
```

---

## 9. Web GUI 与实时通信

### 9.1 系统架构

```
┌──────────────────────────────────────────────────┐
│                    Browser (SPA)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ 文件上传  │ │ 进度展示  │ │ 字幕编辑 + 对比   │  │
│  └────┬─────┘ └────▲─────┘ └────────┬─────────┘  │
│       │            │               │             │
├───────┼────────────┼───────────────┼─────────────┤
│       │    HTTP    │   WebSocket   │    HTTP     │
│       ▼            │               ▼             │
│  ┌──────────────────────────────────────────┐    │
│  │           FastAPI (uvicorn)               │    │
│  │                                           │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │    │
│  │  │ REST API  │ │WebSocket │ │ Static   │  │    │
│  │  │ /api/*    │ │ /ws/*    │ │ /        │  │    │
│  │  └────┬─────┘ └────┬─────┘ └──────────┘  │    │
│  │       │            │                       │    │
│  │       ▼            ▼                       │    │
│  │  ┌──────────────────────────────┐         │    │
│  │  │      Pipeline (后台线程)       │         │    │
│  │  │  进度回调 → WebSocket 广播     │         │    │
│  │  └──────────────────────────────┘         │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

### 9.2 REST API 端点 (46 个)

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/profiles` | 模板列表 + 描述 |
| GET | `/api/profiles/{name}` | 模板完整配置 |
| GET | `/api/device` | GPU/系统设备信息 |
| POST | `/api/run` | 上传 + 启动 Pipeline (multipart) |
| GET | `/api/tasks/{task_id}` | 任务状态 + 结果 |
| GET | `/api/tasks` | 内存中任务列表 |
| GET | `/api/history` | 持久化历史 (分页) |
| GET | `/api/history/{task_id}` | 历史详情 + 事件 |
| DELETE | `/api/history/{task_id}` | 删除单条历史 |
| DELETE | `/api/history` | 清空历史 (可选按时间) |
| GET | `/api/cache/info` | 缓存统计 |
| DELETE | `/api/cache` | 清空缓存 (可选按 stage) |
| PUT | `/api/cache/config` | 运行时修改缓存配置 |
| GET | `/api/subtitle/{task_id}` | 获取字幕 |
| PUT | `/api/subtitle/{task_id}/{index}` | 编辑单条字幕 |
| GET | `/api/subtitle/{task_id}/export` | 导出字幕 (format 参数) |
| GET | `/api/tasks/{task_id}/audio` | 下载分离音频 |
| GET | `/api/llm/providers` | LLM Provider 预设列表 |
| POST | `/api/llm/models` | 获取可用 LLM 模型 |
| GET/PUT | `/api/persistence/settings` | 持久化文件管理设置 |
| POST | `/api/persistence/apply/{task_id}` | 对已完成任务应用持久化 |
| GET | `/api/persistence/files/{task_id}` | 列出已持久化文件 |
| DELETE | `/api/persistence/files/{task_id}` | 删除已持久化文件 |
| POST | `/api/persistence/cleanup` | 强制执行 TTL 清理 |
| GET | `/api/persistence/stats` | 持久化文件统计 |
| GET | `/api/tasks/{task_id}/audio/stream` | 流式下载分离音频 |
| GET | `/api/tasks/{task_id}/subtitle-file` | 下载任务生成的字幕文件 |
| GET | `/api/speaker-embedding/license` | 说话人嵌入模型许可证信息 |
| POST | `/api/feedback/learn` | 上传修订字幕 + 音频，触发学习 |
| POST | `/api/feedback/preview` | 预览学习效果 (dry-run) |
| GET | `/api/feedback/profiles` | 列出所有用户配置 |
| GET | `/api/feedback/profile/{name}` | 查看用户配置详情 |
| POST | `/api/feedback/profile/{name}/rollback` | 回滚配置 |
| DELETE | `/api/feedback/profile/{name}` | 删除配置 |
| GET | `/api/feedback/fingerprints` | 音频指纹库列表 |
| POST | `/api/feedback/fingerprints/match` | 匹配最相似音频指纹 |
| DELETE | `/api/feedback/fingerprints/{fp_id}` | 删除指纹 |
| GET | `/api/feedback/health/{profile_name}` | 健康度趋势 |
| POST | `/api/feedback/health/compute` | 手动计算健康度 |
| GET | `/api/feedback/shadow/{profile_name}` | Shadow Mode 状态 |
| POST | `/api/feedback/shadow/{profile_name}/toggle` | 切换 Shadow Mode |
| POST | `/api/feedback/shadow/{profile_name}/record` | 记录 Shadow 对比结果 |
| GET | `/api/feedback/conflicts/{profile_name}` | 参数震荡状态 |
| POST | `/api/feedback/conflicts/{profile_name}/resolve` | 手动解决震荡 |
| POST | `/api/feedback/impact/preview` | 预览参数变更影响 |

### 9.3 WebSocket 协议

**连接**: `ws://host:port/ws/tasks/{task_id}`

**消息类型**:

| Type | 触发时机 | 字段 |
|------|---------|------|
| `stage_start` | 阶段开始 | stage, message, timestamp |
| `progress` | 阶段进行中 | stage, progress (0-1), message, timestamp |
| `stage_finish` | 阶段完成 | stage, result_summary, timestamp |
| `complete` | Pipeline 完成 | stats, subtitle_count, output_path, timestamp |
| `error` | Pipeline 错误 | stage, error_message, timestamp |

**跨线程广播**:
Pipeline 在后台线程运行，WebSocket 广播在主事件循环执行。`WebSocketManager.broadcast()` 使用 `asyncio.run_coroutine_threadsafe()` 实现跨线程安全投递。

### 9.4 前端 (SPA)

单文件 SPA (`static/index.html`)，零构建步骤：
- 暗色主题 UI
- 拖拽上传 (File API)
- 实时进度条 (WebSocket)
- 字幕表格编辑 (双击编辑，contenteditable)
- LLM 对比视图 (diff 高亮)
- 参数 localStorage 持久化
- 反馈学习面板 (上传修订字幕 / 预览差异 / 查看健康度 / Shadow Mode 开关)

---

## 10. LLM 集成架构

### 10.1 LLM 功能矩阵

```
┌─────────────────────────────────────────────────────┐
│               LLM 功能模块                           │
├─────────────────┬───────────────────────────────────┤
│ llm_subtitle_   │ 字幕后处理优化                     │
│ optimizer       │ ├── Agent Loop (最多 3 轮)         │
│ (独立 Package)  │ ├── 并发批处理 (ThreadPoolExecutor) │
│                 │ ├── Diff 对齐修复                   │
│                 │ └── 关键帧校验 + 去重                │
├─────────────────┼───────────────────────────────────┤
│ llm_merge_      │ 语义合并裁决 (Plan 5)              │
│ engine          │ ├── 600-1200ms 间隙区间            │
│ (Pipeline 内)   │ ├── 语义边界检测                    │
│                 │ └── 降级: 规则模式                  │
├─────────────────┼───────────────────────────────────┤
│ role_labeler    │ 说话人角色标注 (Stage 4.5)          │
│ (Pipeline 内)   │ ├── 主持人/嘉宾/旁白识别            │
│                 │ └── 上下文提示 (context_hint)       │
└─────────────────┴───────────────────────────────────┘
```

### 10.2 LLM Client (`llm_subtitle_optimizer/llm_client.py`)

**协议**: OpenAI 兼容 API (Chat Completions)

**默认 Provider**: DeepSeek (降低成本)

**12 个内置 Provider 预设** (10 云端 + Ollama 本地 + 自定义):
- DeepSeek (v3, r1, v4-pro)
- OpenAI (GPT-4o, GPT-4.1)
- Anthropic (Claude 5 Sonnet, Claude Opus 4.8)
- Google Gemini (2.5 Pro, 2.5 Flash)
- 智谱 GLM (GLM-4.7, GLM-5)
- 阿里百炼 Qwen (Qwen3, Qwen-Max)
- 腾讯混元 (Hunyuan-T1, Hunyuan-Large)
- Moonshot Kimi (Kimi-K2)
- MiniMax (MiniMax-M2)
- 硅基流动 SiliconFlow (各类开源模型)
- Ollama (本地部署)

**容错机制**:
- tenacity 重试 (指数退避，max 3 次)
- API 不可用 → 自动降级到规则/本地模式
- 超时控制 (llm_timeout: 15s)

### 10.3 Subtitle Optimizer Agent Loop

```
输入: 原始字幕
    │
    ▼
┌─────────────────────────────┐
│  Round 1: LLM 优化           │
│  ├── 提示词 + 字幕 + 上下文   │
│  └── → 优化后字幕             │
├─────────────────────────────┤
│  Validate: 关键帧校验         │
│  ├── 相似度 > 60% (短)       │
│  ├── 相似度 > 70% (长)       │
│  ├── 无跨帧重复               │
│  └── → 通过/失败 + 反馈      │
├─────────────────────────────┤
│  Round 2: LLM 修正           │
│  ├── 原始 + 反馈              │
│  └── → 修正后字幕             │
├─────────────────────────────┤
│  Round 3: 最终尝试            │
│  └── → 通过或使用原文         │
└─────────────────────────────┘
    │
    ▼
输出: 优化后字幕 (保留 original_text 用于前端对比)
```

---

## 11. 自适应反馈学习 (Phase 5)

### 11.1 概述

自适应反馈学习引擎允许系统从用户修订的字幕中学习偏好，自动调整管道参数。用户上传手动修订的字幕文件 (.srt/.ass) + 原始音频，系统自动对齐、分析差异，并将学习到的参数偏离量持久化到用户配置中。

**核心能力**:
- 自动版 vs 修订版字幕对齐 (基于 IoU + 语义相似度双锚定)
- 差异分析 + 参数归因 (定位具体哪些配置参数导致了偏差)
- 参数学习 + 梯度累积 (逐步收敛，避免单次过拟合)
- 音频指纹匹配 (复用相似音频场景的学习经验)
- 参数震荡检测 (防止反复调参)
- 健康度评分 (5 维度评估对齐质量)
- Shadow Mode 安全试错 (在不影响用户的前提下测试新参数)
- Few-shot 示例缓存 (为 LLM 合并/优化提供适配用户的上下文)
- CLI + Web GUI 双通道

### 11.2 模块架构

```
vocal_subtitle/feedback/
├── aligner.py              # SubtitleAligner — 自动版/修订版对齐
│                            #   时间轴 IoU 粗匹配 + sentence-transformers 语义精匹配
├── diff_analyzer.py        # DiffAnalyzer — 差异检测 + 参数归因
│                            #   时间偏移检测 / 合并拆分检测 / 文本修改检测
├── param_learner.py        # ParamLearner — 参数学习 + 梯度累积
│                            #   学习率衰减 / 置信度加权 / 分级参数保护
├── audio_fingerprint.py    # AudioFingerprinter — 音频指纹提取与匹配
│                            #   跨越时间轴的声学特征向量 (MFCC + 频谱统计)
├── health_scorer.py        # HealthScorer — 5 维度健康度评分
│                            #   时间对齐 / 内容完整性 / 断句质量 / 边界精度 / 稳定性
├── conflict_detector.py    # ConflictDetector — 参数震荡检测
│                            #   滑动窗口检测参数反复翻转 (A→B→A→B)
├── few_shot_builder.py     # FewShotBuilder — 示例提取与缓存
│                            #   合并决策示例 / 文本格式化示例
├── impact_estimator.py     # ImpactEstimator — 变更影响预估
│                            #   预估参数调整对下游的影响范围
├── shadow_mode.py          # ShadowMode — 影子模式安全试错
│                            #   双轨运行 (active vs shadow config) + 效果对比
└── user_profile.py         # UserProfileManager — 用户配置持久化
                             #   JSON 配置存储 / 备份 / 回滚
```

### 11.3 学习流程

```
┌─────────────────────────────────────────────────────┐
│                   Feedback Learn                      │
│                                                       │
│  1. 解析修订字幕       2. 运行 Pipeline (自动版)       │
│  ┌──────────────┐     ┌──────────────────────┐       │
│  │ parse .srt/.ass│    │ Pipeline.run(         │       │
│  │ → manual_events│    │   skip_separation=True│       │
│  └──────┬───────┘     │ ) → auto_events       │       │
│         │             └──────────┬───────────┘       │
│         └────────────┬───────────┘                   │
│                      ▼                               │
│  3. 对齐 (SubtitleAligner)                            │
│     IoU 粗匹配 → 语义相似度精匹配 → AlignmentPair[]    │
│                      │                               │
│                      ▼                               │
│  4. 差异分析 (DiffAnalyzer)                           │
│     时间偏移 / 合并拆分 / 文本修改 → DiffReport        │
│                      │                               │
│                      ▼                               │
│  5. 参数归因 (ParamLearner)                           │
│     差异 → 具体配置参数 → 建议调整方向 + 置信度 + 学习权重│
│                      │                               │
│                      ▼                               │
│  6. 安全检查                                           │
│     ├── 震荡检测 (ConflictDetector): 是否反复翻转?      │
│     ├── 健康度评分 (HealthScorer): 是否恶化?           │
│     └── 影响预估 (ImpactEstimator): 变更范围多大?      │
│                      │                               │
│                      ▼                               │
│  7. 持久化                                             │
│     ├── UserProfileManager.save(overrides)            │
│     ├── AudioFingerprinter.store(fingerprint)         │
│     └── FewShotBuilder.save_cache(examples)           │
│                                                       │
│  可选: Shadow Mode — 双轨运行，对比效果后再决定是否应用  │
└─────────────────────────────────────────────────────┘
```

### 11.4 CLI 接口

```bash
# 学习用户偏好
vocal-subtitle feedback learn -a input.wav -r revised.srt
vocal-subtitle feedback learn -a input.mp3 -r fixed.ass --dry-run   # 仅预览

# 查看学习到的参数
vocal-subtitle feedback show
vocal-subtitle feedback show --profile user_default

# 回滚 / 重置
vocal-subtitle feedback rollback      # 回滚到上一备份版本
vocal-subtitle feedback reset         # 清除所有学习记录

# 音频指纹管理
vocal-subtitle feedback fingerprints  # 查看指纹库

# 导入/导出配置分享
vocal-subtitle feedback export -o my_profile.yaml
vocal-subtitle feedback import -i friend_profile.yaml
```

### 11.5 Web GUI 集成

反馈学习通过 REST API + WebSocket 集成到 Web GUI：

- `POST /api/feedback/learn` — 上传修订字幕 + 音频，触发学习
- `POST /api/feedback/preview` — 预览学习效果（dry-run）
- `GET /api/feedback/profiles` — 列出所有用户配置
- `GET /api/feedback/profile/{name}` — 查看用户配置详情
- `POST /api/feedback/profile/{name}/rollback` — 回滚配置
- `DELETE /api/feedback/profile/{name}` — 删除配置
- `GET /api/feedback/fingerprints` — 指纹库列表
- `POST /api/feedback/fingerprints/match` — 匹配最相似音频指纹
- `DELETE /api/feedback/fingerprints/{fp_id}` — 删除指纹
- `GET /api/feedback/health/{profile}` — 健康度趋势
- `POST /api/feedback/health/compute` — 手动计算健康度
- `GET /api/feedback/shadow/{profile}` — Shadow Mode 状态
- `POST /api/feedback/shadow/{profile}/toggle` — 切换 Shadow Mode
- `POST /api/feedback/shadow/{profile}/record` — 记录 Shadow 对比结果
- `GET /api/feedback/conflicts/{profile}` — 参数震荡状态
- `POST /api/feedback/conflicts/{profile}/resolve` — 手动解决震荡
- `POST /api/feedback/impact/preview` — 预览参数变更影响

### 11.6 配置项

```yaml
feedback:
  enabled: true
  alignment_min_iou: 0.3
  alignment_min_coverage: 0.4
  alignment_text_weight: 0.6
  alignment_semantic_weight: 0.4
  alignment_semantic_enabled: true
  param_isolation_enabled: true       # 参数隔离（避免交叉影响）
  learning_rate: 0.3
  learning_rate_decay: 0.9
  min_confidence_threshold: 0.5
  oscillation_detection_window: 5     # 震荡检测窗口 (次)
  fingerprint_enabled: true
  fingerprint_distance_method: "cosine"
  fingerprint_knn_k: 3
  fingerprint_min_absolute_similarity: 0.7
  fingerprint_relative_margin: 0.1
  few_shot_enabled: true
  few_shot_max_examples: 20
  health_scoring_enabled: true
  shadow_mode_enabled: false
  shadow_comparison_metric: "alignment_coverage"
  auto_rollback_enabled: false
  auto_rollback_threshold: -10.0      # 健康度下降阈值 (pts)
```

---

## 12. 配置系统

### 12.1 配置解析流程

```
configs/*.yaml ──→ ConfigLoader.load_profile(name)
    │                    │
    │              ┌─────▼──────┐
    │              │  YAML 解析  │
    │              │  + 默认值   │
    │              │  + 类型校验 │
    │              └─────┬──────┘
    │                    │
    │              ┌─────▼──────┐
    │              │ PipelineConfig │
    │              │ (dataclass)    │
    │              └─────┬──────┘
    │                    │
    │         ┌──────────▼──────────┐
    │         │ Pipeline.__init__()  │
    │         │ + run(overrides=...) │
    │         └─────────────────────┘
```

### 12.2 配置覆盖机制

```python
# 优先级: CLI 参数 > overrides > 模板 YAML > 默认值
pipeline.run(
    input_path=...,
    overrides={
        "vad.threshold": 0.45,         # 点号分隔的键路径
        "asr.language": "zh",
        "merging.min_silence_gap": 0.6,
    }
)
```

### 12.3 YAML 模板变体

5 个场景模板的关键差异：

| 参数 | default | podcast | education | variety_show | music_live |
|------|---------|---------|-----------|-------------|------------|
| `vad.threshold` | 0.40 | 0.40 | 0.40 | 0.40 | **0.30** |
| `merging.min_silence_gap` | 0.4s | 0.4s | **0.6s** | **0.3s** | 0.4s |
| `subtitle.max_chars_cjk` | 20 | 20 | **24** | 20 | 20 |
| `noise_reduction.enabled` | false | false | false | **true** | **true** |
| `acoustic_validation.max_snap_distance` | 0.15 | 0.15 | 0.15 | 0.15 | **0.20** |
| `speaker_role.enabled` | false | **true** | false | false | false |

---

## 13. 测试架构

### 13.1 测试分层

```
tests/
├── test_pipeline.py            # 全链路集成测试
├── test_cli.py                 # CLI 端到端测试
├── test_webui.py               # Web GUI API 测试
├── test_audio_preprocessor.py  # 预处理模块测试
├── test_acoustic_validator.py  # 声学标尺校验测试
├── test_macro_chunker.py       # 宏观切块测试
├── test_end_time_fixes.py      # 结束时间修正测试
├── test_streaming.py           # 流式架构测试
├── test_session_manager.py     # 会话管理器测试
├── test_feedback.py            # 反馈学习模块测试
├── test_separation/            # 分离引擎测试 (3 文件)
│   ├── test_uvr_engine.py
│   ├── test_spleeter_engine.py
│   └── test_openunmix_engine.py
├── test_vad/                   # VAD 引擎测试 (5 文件)
│   ├── test_silero_vad.py
│   ├── test_webrtc_vad.py
│   ├── test_ten_vad.py
│   ├── test_ffmpeg_vad.py
│   └── test_boundary_fusion.py
├── test_merging/               # 合并模块测试 (2 文件)
│   ├── test_merge_strategy.py
│   └── test_llm_merge_engine.py
├── test_asr/                   # ASR 引擎测试 (4 文件)
│   ├── test_faster_whisper_engine.py
│   ├── test_whisper_cpp_engine.py
│   ├── test_funasr_engine.py
│   └── test_boundary_refiner.py
├── test_mapping/               # 映射模块测试 (2 文件)
│   ├── test_time_mapper.py
│   └── test_subtitle_builder.py
├── test_diarization/           # 说话人分离测试 (3 文件)
│   ├── test_feature_extractor.py
│   ├── test_speaker_clusterer.py
│   └── test_role_labeler.py
├── test_utils/                 # 工具模块测试 (3 文件)
│   ├── test_audio_utils.py
│   ├── test_cache_manager.py
│   └── test_gpu_detector.py
├── benchmarks/                 # Benchmark 场景 (6 目录)
│   ├── podcast_conversation/
│   ├── studio_monologue/
│   ├── hotel_front_desk/
│   ├── music_voiceover/
│   ├── outdoor_interview/
│   └── meeting_3_speakers/
└── fixtures/                   # 测试配置 + 数据 fixtures
    ├── audio/
    ├── expected/
    └── configs/
```

### 13.2 测试策略

- **单元测试**: 每个引擎/模块独立测试，使用 mock 外部依赖
- **集成测试**: `test_pipeline.py` 覆盖完整 Pipeline
- **端到端测试**: `test_cli.py` 通过 CLI 入口测试完整流程
- **API 测试**: `test_webui.py` 测试 FastAPI 端点和 WebSocket
- **Benchmark**: `scripts/benchmark.py` 性能回归测试

---

## 14. 部署与运维

### 14.1 安装方式

| 方式 | 命令 | 适用场景 |
|------|------|---------|
| 一键脚本 | `bash install.sh --gui` | 新手/快速部署 |
| 最小安装 | `pip install -e ".[faster-whisper,silero-vad,uvr]"` | 仅 CLI |
| Web GUI | `pip install -e ".[faster-whisper,silero-vad,uvr,webui]"` | 个人工作站 |
| 全量安装 | `pip install -e ".[all]"` | 开发/全功能 |
| Docker | (计划中) | 服务器部署 |

### 14.2 系统依赖

- **ffmpeg >= 4.4**: 音频解码、视频解复用、silencedetect
  - 依赖方式: subprocess 调用 (非动态链接，规避 GPL 传染)
- **libsndfile**: Open-Unmix 音频 I/O (可选)

### 14.3 GPU 支持

```python
# gpu_detector.py 自动检测
def get_best_device() -> Device:
    if torch.cuda.is_available():
        return Device.CUDA
    if torch.backends.mps.is_available():
        return Device.MPS  # CTranslate2 不支持，降级 CPU
    return Device.CPU
```

**已知限制**:
- CTranslate2 (faster-whisper) 不支持 Apple MPS，MPS 环境自动降级 CPU
- float16 在 CPU 上不可用，自动降级 int8
- Spleeter 仅支持 Python < 3.12 (已于 2022 年停维)

### 14.4 日志系统

基于 `structlog` 的结构化日志：

```yaml
logging:
  level: "INFO"           # DEBUG/INFO/WARNING/ERROR
  format: "json"          # json (生产) | console (开发)
  file: "logs/pipeline.log"
```

JSON 格式输出示例：
```json
{
  "event": "stage_start",
  "stage": "separation",
  "timestamp": "2026-07-05T10:30:00.123Z",
  "level": "info"
}
```

### 14.5 运行时目录

```
Vocal_Subtitle/
├── cache/                      # 运行时缓存 (gitignored)
│   ├── models/                 # ML 模型权重 (首次运行自动下载)
│   ├── separation/             # L1: diskcache 分离缓存
│   ├── transcription/          # L2: diskcache 转录缓存
│   ├── persistent_files/       # 持久化音频/字幕文件 (hash 前缀分目录)
│   ├── uploads/                # Web GUI 上传临时文件
│   ├── speaker_models/         # 说话人嵌入模型缓存
│   ├── task_history.db         # L3: SQLite 任务历史
│   └── persistence_settings.json  # 服务端配置持久化
├── logs/                       # 运行时日志
│   ├── pipeline.log
│   └── webui.log
└── venv/                       # Python 虚拟环境
```

---

## 附录 A: 关键文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `vocal_subtitle/pipeline.py` | 3537 | 管道编排器 |
| `vocal_subtitle/config.py` | 1110 | 配置管理 (24 个 dataclass) |
| `vocal_subtitle/webui/api.py` | 2452 | REST API (46 端点) |
| `vocal_subtitle/cli.py` | 817 | CLI 命令行入口 (含 feedback 子命令组) |
| `vocal_subtitle/acoustic_validator.py` | 823 | 声学标尺校验 + 诊断报告 (Plan 7) |
| `vocal_subtitle/merging/merge_strategy.py` | 728 | 合并策略 + Plan 3 段内预切分 |
| `vocal_subtitle/merging/llm_merge_engine.py` | 1167 | LLM 语义合并 (Plan 5) + 帧级无缝 (Plan 6) |
| `vocal_subtitle/utils/audio_utils.py` | 753 | 音频加载/转换/重采样 |
| `vocal_subtitle/mapping/subtitle_builder.py` | 729 | 字幕构建输出 (SRT/VTT/ASS) |
| `vocal_subtitle/asr/boundary_arbitration.py` | 681 | LLM 语义仲裁器 |
| `llm_subtitle_optimizer/optimizer.py` | 640 | LLM Agent Loop 优化器 |
| `vocal_subtitle/diarization/speaker_clusterer.py` | 633 | 说话人聚类 |
| `vocal_subtitle/mapping/time_mapper.py` | 562 | 时间轴映射 + 事件去重 |
| `vocal_subtitle/asr/boundary_refiner.py` | 528 | ASR 边界双向精修 (Plan 4) |
| `vocal_subtitle/diarization/speaker_embedding.py` | 522 | 声学嵌入提取 (SpeechBrain) |
| `vocal_subtitle/audio_preprocessor.py` | 477 | 预 VAD 降噪 (Stage 1.5) |
| `vocal_subtitle/asr/boundary_reasr.py` | 459 | 滑动窗口冗余 ASR |
| `vocal_subtitle/utils/persistence_manager.py` | 405 | 持久化文件管理 |
| `vocal_subtitle/streaming.py` | 402 | 流式处理架构 |
| `vocal_subtitle/asr/boundary_confidence.py` | 400 | 边界置信度评估 (5维度) |
| `vocal_subtitle/diarization/feature_extractor.py` | 369 | 87维声学特征提取 |
| `vocal_subtitle/utils/task_history.py` | 354 | SQLite 任务历史 |
| `vocal_subtitle/macro_chunker.py` | 345 | 宏观静音切块 (Plan 0) |
| `vocal_subtitle/utils/cache_manager.py` | 345 | diskcache 缓存管理 |
| `vocal_subtitle/diarization/role_labeler.py` | 334 | LLM 角色标注 |
| `vocal_subtitle/vad/boundary_fusion.py` | 268 | 三方法边界融合 (Plan 2) |
| `vocal_subtitle/utils/session_manager.py` | 245 | 会话管理 (hash 目录 + 去重) |
| `vocal_subtitle/feedback/aligner.py` | 420 | 自动版/修订版字幕对齐 |
| `vocal_subtitle/feedback/diff_analyzer.py` | 350 | 差异分析 + 参数归因 |
| `vocal_subtitle/feedback/param_learner.py` | 220 | EMA 参数学习 |
| `vocal_subtitle/feedback/audio_fingerprint.py` | 380 | 音频指纹提取与匹配 |
| `vocal_subtitle/feedback/health_scorer.py` | 250 | 5维度健康度评分 |
| `vocal_subtitle/feedback/conflict_detector.py` | 200 | 参数震荡检测 |
| `vocal_subtitle/feedback/impact_estimator.py` | 180 | 变更影响预估 |
| `vocal_subtitle/feedback/shadow_mode.py` | 230 | Shadow Mode 安全试错 |
| `vocal_subtitle/feedback/user_profile.py` | 200 | 用户配置管理 |
| `vocal_subtitle/webui/websocket.py` | 156 | WebSocket 实时通信 |
| `vocal_subtitle/asr/text_normalizer.py` | 152 | 文本后处理标准化 |
| `vocal_subtitle/webui/models.py` | 220 | Pydantic 数据模型 |
| `vocal_subtitle/utils/progress.py` | 186 | 进度管理器 (CLI + WebSocket) |
| `vocal_subtitle/utils/gpu_detector.py` | 185 | CUDA/MPS/CPU 自动检测 |
| `vocal_subtitle/utils/file_hasher.py` | 133 | SHA256 文件哈希 |
| `vocal_subtitle/mapping/end_time_validator.py` | 104 | 结束时间校验 |
| `vocal_subtitle/utils/logger.py` | 103 | structlog 结构化日志 |
| `llm_subtitle_optimizer/llm_client.py` | 202 | OpenAI 兼容 API 客户端 |
| `llm_subtitle_optimizer/aligner.py` | 179 | Diff 对齐修复 |
| `install.sh` | 1111 | 一键安装脚本 |

## 附录 B: 外部依赖协议合规

所有外部依赖协议均为 MIT / Apache 2.0 / BSD / ISC，确保商用零合规风险：

| 包 | 协议 | 用途 |
|---|------|------|
| numpy | BSD-3 | 数值计算 |
| pydub | MIT | 音频转换 |
| pysubs2 | MIT | 字幕 I/O |
| pyyaml | MIT | 配置解析 |
| click | BSD-3 | CLI |
| tqdm | MPL-2.0 / MIT | 进度条 |
| diskcache | Apache-2.0 | 磁盘缓存 |
| structlog | MIT / Apache-2.0 | 日志 |
| faster-whisper | MIT | ASR |
| torch / torchaudio | BSD-3 | ML 推理 |
| audio-separator | MIT | UVR 分离 |
| fastapi | MIT | Web 框架 |
| uvicorn | BSD-3 | ASGI 服务器 |
| librosa | ISC | 音频分析 |
| scikit-learn | BSD-3 | 聚类 |
| sentence-transformers | Apache-2.0 | 语义相似度 |
| openai | Apache-2.0 | LLM API |
