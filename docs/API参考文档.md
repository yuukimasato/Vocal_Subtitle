# API 参考文档

## 核心模块

### vocal_subtitle.Pipeline

管道编排器，负责调度 5+2 个处理阶段的有序执行（含 2 个可选的说话人标注阶段）。

```python
from vocal_subtitle import Pipeline
from pathlib import Path

pipeline = Pipeline()
result = pipeline.run(
    input_path=Path("input.mp3"),
    output_path=Path("output.srt"),
    output_format="srt",
    skip_separation=False,
)
```

#### Pipeline(config=None)

创建管道实例。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | `PipelineConfig` | `None` | 管道配置，默认加载 `default` 模板 |

#### Pipeline.run(input_path, output_path=None, output_format="srt", progress_callback=None, skip_separation=False, **overrides) → dict

执行全链路处理。

| 参数 | 类型 | 说明 |
|------|------|------|
| `input_path` | `Path` | 输入音频文件路径 |
| `output_path` | `Path \| None` | 字幕输出路径，默认与输入同名 `.srt` |
| `output_format` | `str` | 输出格式: `"srt"` / `"vtt"` / `"ass"` |
| `progress_callback` | `callable \| None` | 进度回调函数 |
| `skip_separation` | `bool` | 跳过分离阶段 |

**返回值**: `dict`

```python
{
    "subtitle_path": Path,              # 输出字幕文件路径
    "stats": PipelineStats,             # 管道执行统计
    "events": List[SubtitleEvent],      # 字幕事件列表
    "from_cache": bool,                 # 是否来自全管道缓存
    "vocals_path": str | None,          # 分离人声 WAV 路径
    "accompaniment_path": str | None,   # 分离伴奏/背景声 WAV 路径
}
```

#### Pipeline.run_batch(input_dir, output_dir, output_format="srt", glob_pattern="*.mp3", **overrides) → List[dict]

批量处理音频文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `input_dir` | `Path` | 输入目录 |
| `output_dir` | `Path` | 输出目录 |
| `output_format` | `str` | 输出字幕格式 |
| `glob_pattern` | `str` | 文件匹配模式 |

---

### vocal_subtitle.config.ConfigLoader

YAML 配置文件加载器。

```python
from vocal_subtitle.config import ConfigLoader

loader = ConfigLoader()

# 列出可用模板
profiles = loader.list_profiles()
# → ["default", "podcast", "education", "variety_show", "music_live"]

# 加载模板
config = loader.load_profile("podcast")

# 加载自定义文件
config = loader.load_file(Path("my_config.yaml"))

# 覆盖参数
config = loader.merge_with_overrides(config,
    separator="uvr",
    language="ja",
    vad_threshold=0.3,
)
```

#### 配置数据类

```python
from vocal_subtitle.config import (
    PipelineConfig,
    SeparationConfig,
    VADConfig,
    MergingConfig,
    DiarizationConfig,
    ASRConfig,
    SpeakerRoleConfig,
    SubtitleBuildConfig,
    GapHandlingConfig,
    LLMOptimizeConfig,
    NoiseReductionConfig,
    CacheConfig,
    LoggingConfig,
)
```

---

### vocal_subtitle.separation — 人声分离引擎

```python
from vocal_subtitle.separation import (
    SeparationEngine,   # 抽象基类
    SeparationResult,   # 分离结果
    LicenseInfo,        # 协议信息
    SpleeterEngine,     # Spleeter 引擎 (MIT)
    OpenUnmixEngine,    # Open-Unmix 引擎 (MIT)
    UVREngine,          # UVR 引擎 (MIT)
)
```

#### SeparationEngine.separate(input_path, output_dir, **kwargs) → SeparationResult

执行人声分离。

```python
@dataclass
class SeparationResult:
    vocals_path: Path           # 人声输出路径（已标准化、已缓存）
    accompaniment_path: Path    # 伴奏输出路径（已标准化、已缓存）
    engine_name: str            # 引擎名称
    processing_time: float      # 处理耗时（秒）
```

> **缓存说明**：人声和伴奏均会标准化（峰值归一化、单声道 16kHz）并复制到
> `cache/persistent_files/` 持久化目录。下次处理相同文件时直接命中缓存，
> 跳过分离阶段。分离音频可通过 Web GUI 或
> `GET /api/tasks/{task_id}/audio?type=vocals|accompaniment` 下载。

---

### vocal_subtitle.vad — 语音活动检测

```python
from vocal_subtitle.vad import (
    VADEngine,      # 抽象基类
    SpeechSegment,  # 语音片段
    SileroVAD,      # Silero VAD (MIT)
    TENVAD,         # TEN VAD (Apache 2.0)
    WebRTCVAD,      # WebRTC VAD (BSD-3)
)
```

#### VADEngine.detect(audio_path, threshold=0.5, ...) → List[SpeechSegment]

检测文件中的语音区间。

#### VADEngine.detect_on_array(audio, sample_rate, ...) → List[SpeechSegment]

在 numpy 数组上检测。

```python
@dataclass
class SpeechSegment:
    start: float        # 起始时间（秒）
    end: float          # 结束时间（秒）
    confidence: float   # 置信度 (0.0–1.0)
    duration: float     # 片段时长（属性）
```

---

### vocal_subtitle.asr — 语音识别

```python
from vocal_subtitle.asr import (
    ASREngine,              # 抽象基类
    TranscriptionSegment,   # 转录结果
    WordTimestamp,          # 词级时间戳
    FasterWhisperEngine,    # faster-whisper (MIT)
    WhisperCppEngine,       # whisper.cpp (MIT)
    FunASREngine,           # Fun-ASR (Apache 2.0)
)
```

#### ASREngine.transcribe(audio, sample_rate=16000, language=None, **kwargs) → List[TranscriptionSegment]

识别音频片段。

```python
@dataclass
class TranscriptionSegment:
    text: str                       # 转录文本
    start: float                    # 段内起始时间
    end: float                      # 段内结束时间
    words: List[WordTimestamp]      # 词级时间戳
    avg_logprob: float              # 平均对数概率
```

```python
@dataclass
class WordTimestamp:
    word: str           # 词
    start: float        # 起始时间
    end: float          # 结束时间
    confidence: float   # 置信度
```

---

### vocal_subtitle.diarization — 说话人分离与角色标注

```python
from vocal_subtitle.diarization import (
    DiarizationEngine,     # 抽象基类
    ClusteredSegment,      # 带说话人编号的片段
    SpeakerRole,           # 角色标注结果
    SpeakerDiarizer,       # 音色聚类引擎
    RoleLabeler,           # LLM 角色标注器
)
```

#### SpeakerDiarizer — 音色特征聚类

```python
diarizer = SpeakerDiarizer(
    distance_threshold=0.5,   # 凝聚聚类合并阈值（余弦距离）
    min_speakers=1,           # 最少说话人数
    max_speakers=10,          # 最多说话人数
    use_pca=True,             # 聚类前 PCA 降维
    pca_variance=0.95,        # PCA 保留方差比例
)
diarizer.load_model()
speaker_ids = diarizer.diarize(segments, audio, sample_rate=16000)
# → [0, 1, 0, 0, 1, 2, 1, ...]
```

聚类后自动计算 Silhouette Score 评估质量：
- \> 0.5：清晰分离
- 0.25–0.5：可接受
- \< 0.25：低置信度

#### RoleLabeler — LLM 角色标注

三级命名策略：身份识别 → 角色推断 → 兜底。

```python
labeler = RoleLabeler()
role_names = labeler.label_roles(
    transcript_by_speaker={0: ["大家好..."], 1: ["谢谢..."]},
    model="deepseek-v4-pro",
    temperature=0.2,
    context_hint="a podcast interview",
)
# → {0: "主持人", 1: "张三(嘉宾)"}
```

LLM 不可用时自动兜底为 `说话人A`、`说话人B` 等序列标签。

#### 数据结构

```python
@dataclass
class SpeakerRole:
    speaker_id: int
    name: Optional[str] = None       # 从上下文挖掘的名字
    role: Optional[str] = None       # 推断的角色类型
    label: str = ""                  # 最终显示标签
    confidence: str = "fallback"     # "identity" | "role" | "fallback"
```

---

### vocal_subtitle.merging — 片段合并

```python
from vocal_subtitle.merging import MergeStrategy, MergeConfig

strategy = MergeStrategy(MergeConfig(
    min_silence_gap=0.4,
    max_segment_length=30.0,
    padding=0.1,
    min_segment_length=0.5,
))
merged = strategy.merge(segments, audio, sample_rate, total_duration)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_silence_gap` | 0.4s | 合并阈值 |
| `max_segment_length` | 30.0s | 最大段长 |
| `padding` | 0.1s | 两端填充 |
| `min_segment_length` | 0.5s | 最小段长，过短丢弃 |

---

### vocal_subtitle.mapping — 时间轴映射与字幕构建

```python
from vocal_subtitle.mapping import TimeMapper, SubtitleBuilder, SubtitleRule
```

#### TimeMapper

```python
mapper = TimeMapper(
    seamless_threshold=0.2,
    natural_pause_max=1.0,
)
events = mapper.map(asr_results, speech_segments)
```

#### SubtitleBuilder

```python
builder = SubtitleBuilder(rule=SubtitleRule(
    min_duration=0.8,
    max_duration=5.0,
    max_chars_cjk=20,
    max_chars_latin=42,
    max_lines=2,
    speaker_label_format="bracket",  # "bracket" | "prefix" | "none"
))
builder.build(events, output_path, fmt="srt")

# bracket:  [主持人] 大家好
# prefix:   主持人: 大家好
# none:     大家好（无标签）
```

---

### vocal_subtitle.utils — 工具层

```python
from vocal_subtitle.utils import (
    AudioUtils,         # 音频处理
    CacheManager,       # 磁盘缓存
    GPUDetector,        # GPU 检测
    ProgressManager,    # 进度管理
    setup_logging,      # 日志配置
)
```

#### AudioUtils

| 方法 | 说明 |
|------|------|
| `load_audio(path, target_sr)` | 加载音频 → (np.ndarray, sr) |
| `save_audio(audio, path, sr)` | 保存音频为 WAV |
| `normalize_audio(audio)` | 标准化音频 |
| `extract_segment(audio, start, end)` | 提取片段 |
| `time_to_sample(time, sr)` | 时间 → 采样点 |
| `sample_to_time(n, sr)` | 采样点 → 时间 |
| `get_duration_seconds(path)` | 获取音频时长 |
| `get_audio_info(path)` | 获取音频元信息 |
| `convert_format(in_path, out_path, fmt)` | 转换音频格式 |

#### GPUDetector

| 方法 | 说明 |
|------|------|
| `get_best_device()` | 获取最佳设备 → `DeviceType` |
| `get_device_info()` | 获取设备详细信息 |
| `get_gpu_memory_used_mb()` | 获取当前显存使用量 |
| `detect_cuda()` | 检测 CUDA 可用性 |
| `detect_mps()` | 检测 Apple Silicon 可用性 |
| `select_whisper_model(device_type)` | 根据设备推荐模型 |

#### CacheManager

```python
cache = CacheManager(
    cache_dir="./cache",
    ttl_separation=86400,
    ttl_transcription=604800,
)

key = cache.make_key(input_path, engine="uvr")
cached = cache.get("separation", key)
cache.set("separation", key, result)
cache.clear_all()
```

---

## 数据模型（更新）

### SubtitleEvent（v0.3+）

```python
@dataclass
class SubtitleEvent:
    index: int                          # 字幕序号 (1-based)
    start: float                        # 全局开始时间（秒）
    end: float                          # 全局结束时间（秒）
    text: str                           # 字幕文本（LLM 优化后）
    words: List = field(default_factory=list)
    original_text: Optional[str] = None # LLM 优化前的原始 ASR 文本
    speaker_id: Optional[int] = None    # 说话人编号 (0, 1, 2, ...)
    speaker_label: Optional[str] = None # 说话人标签: "张三(嘉宾)", "主持人"
```

### Pipeline.run() 返回值（v0.2+）

```python
{
    "subtitle_path": Path,              # 输出字幕文件路径
    "stats": PipelineStats,             # 管道执行统计
    "events": List[SubtitleEvent],      # 字幕事件列表
    "from_cache": bool,                 # 是否来自全管道缓存
    "vocals_path": str | None,          # 分离人声 WAV 路径
    "accompaniment_path": str | None,   # 分离伴奏/背景声 WAV 路径
}
```

---

## WebSocket 进度推送协议

前端通过 WebSocket 连接 `/ws/tasks/{task_id}` 接收实时进度更新。

### 消息类型

| type | 方向 | 说明 | 字段 |
|------|------|------|------|
| `stage_start` | 服务端→客户端 | 阶段开始 | `stage`, `total`, `description` |
| `progress` | 服务端→客户端 | 阶段进度 | `stage`, `current`, `total`, `extra` |
| `stage_finish` | 服务端→客户端 | 阶段完成 | `stage`, `elapsed_seconds` |
| `complete` | 服务端→客户端 | 全流程完成 | `result` (含 stats + events) |
| `error` | 服务端→客户端 | 处理失败 | `message` |
| `pong` | 服务端→客户端 | 心跳响应 | — |
| `ping` | 客户端→服务端 | 心跳请求 | `type: "ping"` |

### 阶段名称映射

| stage 值 | 对应处理阶段 | 前端 DOM ID |
|----------|-------------|-------------|
| `separation` | 人声分离 | `#stage-separation` |
| `vad` | VAD 语音检测 | `#stage-vad` |
| `merging` | 片段合并 | `#stage-merging` |
| `diarization` | 说话人分离（可选） | `#stage-diarization` |
| `asr` | ASR 语音识别 | `#stage-asr` |
| `role_labeling` | 角色标注（可选） | `#stage-role-labeling` |
| `mapping` | 字幕生成 | `#stage-mapping` |
| `llm` | LLM 优化 | `#stage-llm` |

### 架构说明

Pipeline 在后台线程执行，通过 `ProgressManager` 回调将事件发送到 `WebSocketManager`，
由 `asyncio.run_coroutine_threadsafe` 将消息路由到 FastAPI 主事件循环，
最终推送到前端 WebSocket 客户端。

事件循环跨线程传递的关键点：
- `WebSocketManager._main_loop` 存储 FastAPI 主事件循环引用
- `set_main_loop()` 在 `/api/run` 端点中调用（异步上下文）
- `connect()` 中作为后备自动捕获
- 所有跨线程广播通过 `broadcast_from_thread()` 安全调度

---

## Web GUI REST API

### 任务管理

#### `POST /api/run`
启动单文件 Pipeline 处理（multipart/form-data）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `file` | File | 上传的音频/视频文件 |
| `profile` | str | 场景模板名称，默认 `"default"` |
| `output_format` | str | 输出格式，默认 `"srt"` |
| `skip_separation` | bool | 跳过分离阶段 |
| `overrides` | str | JSON 格式的参数覆盖 |

返回 `{"task_id": str, "status": "pending"}`。

#### `GET /api/tasks/{task_id}`
查询任务状态和结果。

返回 `TaskStatus`:
```json
{
  "task_id": "...",
  "status": "pending|running|completed|failed",
  "progress": null,
  "result": {
    "subtitle_path": "...",
    "stats": {...},
    "events": [...],
    "vocals_path": "...",
    "accompaniment_path": "..."
  },
  "error": null
}
```

#### `GET /api/tasks/{task_id}/audio?type=vocals|accompaniment`
下载分离后的音频文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `type` | str | `"vocals"`（人声）或 `"accompaniment"`（背景声/伴奏） |

返回 `audio/wav` 文件下载。

### 字幕操作

#### `GET /api/subtitle/{task_id}`
获取字幕事件列表（`SubtitleEventResponse[]`），事件包含 `speaker_id` 和 `speaker_label` 字段（说话人分离开启时）。

#### `PUT /api/subtitle/{task_id}/{index}`
编辑单条字幕文本。

```json
{"index": 1, "text": "修改后的文本"}
```

#### `GET /api/subtitle/{task_id}/export?format=srt|vtt|ass`
导出字幕文件。

### 场景模板

#### `GET /api/profiles`
获取可用场景模板列表。

#### `GET /api/profiles/{name}`
获取指定模板的完整配置参数。

### 设备信息

#### `GET /api/device`
获取系统 GPU/CPU 设备信息和推荐配置。

### 任务历史

#### `GET /api/history?limit=20&offset=0`
获取持久化任务历史列表。

#### `GET /api/history/{task_id}`
获取任务详情（含完整字幕事件）。

#### `DELETE /api/history/{task_id}`
删除单条历史记录。

#### `DELETE /api/history`
清除全部历史记录。

### 缓存管理

#### `GET /api/cache/info`
获取缓存统计信息。

#### `DELETE /api/cache?stage=separation`
清除指定阶段或全部缓存。

### LLM 模型管理

#### `GET /api/llm/providers`
获取预设 LLM 供应商列表（DeepSeek / OpenAI / Anthropic / 智谱 / 阿里百炼 等）。

#### `POST /api/llm/models`
传入 `base_url` + `api_key`，实时获取可用模型列表。

---

## 前端设置持久化

Web GUI 的所有用户参数通过 `localStorage` 自动保存和恢复。

### 存储键

统一存储键：`vocal_settings`（JSON 格式）。

旧版独立键（首次加载后自动迁移）：`vocal_llm_url`、`vocal_llm_key`、`vocal_llm_model`、`vocal_llm_enabled`。

### 持久化的参数

| 参数 | 对应配置 |
|------|---------|
| `profile` | 选中的场景模板 |
| `separator` | 分离引擎 |
| `uvr_model` | UVR 模型 |
| `vad_engine` | VAD 引擎 |
| `vad_threshold` | VAD 阈值 |
| `asr_engine` | ASR 引擎 |
| `asr_model` | ASR 模型 |
| `asr_device` | 计算设备 |
| `language` | 识别语言 |
| `subtitle_min_duration` | 字幕最小时长 |
| `subtitle_max_duration` | 字幕最大时长 |
| `llm_enabled` | LLM 优化开关 |
| `llm_model` | LLM 模型 |
| `llm_base_url` | LLM API 地址 |
| `llm_api_key` | LLM API 密钥 |
| `diarization_enabled` | 说话人分离开关 |
| `speaker_role_enabled` | LLM 角色标注开关 |

### 优先级

用户保存值 > 模板默认值。切换模板时先加载新模板默认值，再覆盖用户保存值。```
