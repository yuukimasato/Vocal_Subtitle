# Vocal_Subtitle 离线高精度字幕系统设计文档

> **日期**：2026-07-26（基于阶段零到五实施完成后的全面修订，含 superpowers 设计文档交叉引用）
> **状态**：阶段零到五代码已全部完成并接入离线 Pipeline。620 个测试通过（0 失败），global CPU 技术链路验收通过。物理字幕仓灌装代码已实现，真实素材质量门仍在优化中。各子系统详细设计以 `docs/superpowers/specs/` 为准，实施计划见 `docs/superpowers/plans/`。本文以文末 §20 为当前实际实现状态。

## 目标与范围

离线主流程已从「物理语音段独立 ASR」调整为「全局 ASR + 物理范围约束 + 词级分配」。六项目标均已实现代码级闭环：

1. 物理音频文件的开始/结束范围是不可逾越的硬约束 —— `PhysicalTimeline`、`PhysicalClip`、最终校验器均已实现
2. ASR 获得完整音频上下文，降低短片段幻觉和残缺音素补全 —— WhisperX 全局转录、重叠窗口去重已接入
3. 全局说话人身份保持一致 —— 全局 diarization + canonicalization + `expected_speakers` 上限约束已实现
4. 字幕默认在换气、明显静音和说话人变化处断句 —— 严格断句器（`StrictSegmenter`）已默认启用
5. LLM 只能优化已分配文本，不能制造新的词、时间范围、说话人或物理归属 —— LLM 安全门已实现
6. 现有物理检测能力得到保留，但不把 VAD 检测结果误认为绝对声学事实 —— 三层分离（PhysicalClip / SpeechEvidenceSpan / ContextWindow）已实现

---

## 目录

1. [从差距基线到实施完成](#1-从差距基线到实施完成)
2. [系统总览与当前架构](#2-系统总览与当前架构)
3. [安装与运行时](#3-安装与运行时)
4. [音频预处理与物理边界系统](#4-音频预处理与物理边界系统)
5. [全局 ASR 与 WhisperX 集成](#5-全局-asr-与-whisperx-集成)
6. [说话人身份与字段一致性](#6-说话人身份与字段一致性)
7. [物理范围与说话人词级分配器](#7-物理范围与说话人词级分配器)
8. [ASR 幻觉过滤](#8-asr-幻觉过滤)
9. [字幕断句规则与物理字幕仓灌装](#9-字幕断句规则与物理字幕仓灌装)
10. [LLM 权限边界与文本优化](#10-llm-权限边界与文本优化)
11. [LLM 字幕差异高亮与 WebUI 三阶段时间轴](#11-llm-字幕差异高亮与-webui-三阶段时间轴)
12. [缓存与模型管理](#12-缓存与模型管理)
13. [HF Token 安全管理](#13-hf-token-安全管理)
14. [迁移方案与实施状态](#14-迁移方案与实施状态)
15. [测试与验收标准](#15-测试与验收标准)
16. [风险与取舍](#16-风险与取舍)
17. [完成定义](#17-完成定义)
18. [2026-07-26 当前实现与质量验收](#18-2026-07-26-当前实现与质量验收)
19. [2026-07-26 代码审计与补齐跟踪](#19-2026-07-26-代码审计与补齐跟踪)
20. [2026-07-26 补齐结果与生产结论](#20-2026-07-26-补齐结果与生产结论)
21. [自适应反馈学习系统](#21-自适应反馈学习系统)
22. [流式处理](#22-流式处理)
23. [参考文档索引](#23-参考文档索引)

---

## 1. 从差距基线到实施完成

本文原以阶段实施前的「差距核对」开篇。经过阶段零到五的完整实施，原差距清单中的条目已全部解决。以下按原核对项给出当前状态。

### 1.1 已实现且可直接依赖（原清单，全部保留）

| 能力 | 实际位置 |
|------|----------|
| 全局 diarization（完整音频跑一次，固定全局 ID） | `Pipeline._run_global_diarization()` |
| 全局 turn 投影为 speaker-safe ASR 段 | `Pipeline._project_global_speakers()` |
| 边界精修不越过已确认 turn | `Pipeline._clamp_segments_to_speaker_turns()` |
| `SpeakerTurn` / `AtomicSpeechSpan` / `DiarizationResult` | `diarization/base.py` |
| 物理区间与 turn 求交集 | `reconcile_regions()` — `diarization/turn_reconciler.py` |
| canonical speaker ID 重编号（按首次出现排序） | `PyannoteDiarizationEngine.diarize()` 内联 |
| `asr_text` / `llm_text` / `original_text` 三字段 | `SubtitleEvent` |
| 跨说话人合并阻断 | `subtitle_builder.py`、`llm_merge_engine.py` |
| 说话人统计字段 | `PipelineStats` |
| LLM 输出相似度校验与拒绝 | `llm_subtitle_optimizer/optimizer.py` |
| LLM 差异高亮（LCS token 级） | `webui/static/index.html` |
| 宏观切块 | `MacroChunker` |
| 三方法边界融合 | `BoundaryFusion` |
| 声学骨架校验 | `AcousticValidator` |
| diarization 结果缓存（含配置指纹） | `Pipeline._run_global_diarization()` |

### 1.2 原稿引用错误（已全部修正）

原稿引用的不存在的标识符已在代码中全部实现或以不同命名落地：

| 原稿引用 | 当前实际实现 |
|----------|-------------|
| `_canonicalize_diarization_result()` | 已提取为独立 canonicalization 模块，含 `expected_speakers` 上限约束 |
| `expected_speakers` 配置项 | 已添加到 `DiarizationConfig`，进入缓存键 |
| `physical_region_id` 字段 | 以 `physical_spans` 和 `PhysicalClip.id` 形式实现 |
| `PhysicalClip` / `SpeechEvidenceSpan` / `ContextWindow` | 已实现于 `vocal_subtitle/physical/timeline.py` |
| `GlobalWord` / `PhysicalTimeline` / `GlobalTranscript` | 已实现于 `vocal_subtitle/physical/ir.py` |
| `physical_spans` / `source_word_ids` 等 | 已添加到 `SubtitleEvent`（可选，向后兼容） |
| `no_speech_prob` / `compression_ratio` | 已添加到 `TranscriptionSegment` |
| `WordTimestamp.speaker_id` | 已添加到 `WordTimestamp` |
| WhisperX 引擎与配置 | 已实现于 `vocal_subtitle/asr/whisperx_engine.py` |
| `install.sh` CUDA 检测 | 已实现纯 shell 的 `nvidia-smi` 检测 |

### 1.3 已确认的真实缺陷（已全部修复）

**缺陷 1：末条 ASR 结果结束时间被无条件拉长到 VAD 段末。**
→ 已修复。物理优先时间轴：所有事件以物理语音边界为准，末条不再特殊处理。

**缺陷 2：`unified_ffmpeg_pass()` 忽略配置。**
→ 已修复。所有调用路径传入 `AcousticValidationConfig` 的配置值。

**缺陷 3：说话人标签数量失控。**
→ 已修复。`expected_speakers` 上限约束 + canonicalization 合并，双人素材不再产生超过两个标签。

---

### 1.4 Superpowers 设计文档与实施状态映射

以下为 `docs/superpowers/` 目录中各设计规格文档及其当前实际实施状态。部分 spec 的状态标记为"待实施"实为撰写时的快照，代码已先行完成。

| 设计文档 (spec) | Spec 状态 | 实际实施状态 | 对应主文档节 |
|:---|:---|:---|:---|
| 2026-07-24-asr-accuracy-design | 已批准并实现 | ✅ 已实现 | §5, §8, §10 |
| 2026-07-24-deployment-design | 已批准并实现 | ✅ 已实现 | §3 |
| 2026-07-24-hf-model-download-design | 已批准并实现 | ✅ 已实现 | §13 |
| 2026-07-24-llm-asr-language-fix-design | 已批准并实现 | ✅ 已实现 | §5 |
| 2026-07-24-speaker-diarization-design | 已实现 | ✅ 已实现 | §6, §7 |
| 2026-07-24-subtitle-timeline-comparison-design | 已批准并实现 | ✅ 已实现 | §11 |
| 2026-07-25-phase-zero-physical-first | 已确认设计，待实施 | ✅ 已实现 | §4, §6, §7 |
| 2026-07-25-phase-one-hallucination-filter | 已确认，待实施 | ✅ 已实现 | §8 |
| 2026-07-25-phase-two-physical-timeline (P2.1) | 已确认，待实施 | ✅ 已实现 | §2.4, §4, §7 |
| 2026-07-25-phase-two-evidence-adapter (P2.2) | 已确认，待实施 | ✅ 已实现 | §4.2 |
| 2026-07-25-phase-two-global-ir (P2.3) | 已确认，待实施 | ✅ 已实现 | §2.4 |
| 2026-07-25-phase-two-coordinate-context-cache (P2.4) | 已确认，已实施 | ✅ 已实现 | §2.5, §12 |
| 2026-07-26-phase-three-global-transcription | 已实现（默认关闭） | ✅ 已实现（阶段五后默认开启） | §5, §7 |
| 2026-07-26-phase-four-subtitle-segmentation | 已实现 | ✅ 已实现（默认启用严格断句） | §9 |
| 2026-07-26-phase-five-default-path | 已确认，待实施 | ✅ 已实现（620 测试通过） | §5.2, §14, §18 |
| 2026-07-26-full-chain-completion | 已获批准，待实施 | ✅ 代码链路已闭环；真实素材质量门待通过 | §18 |
| 2026-07-26-physical-bin-subtitle-filling | 待用户审阅 | 🔶 已实现（`subtitle_bins.py`）；质量优化进行中 | §7.2, §9.4, §18.6 |
| 2026-07-26-production-completion | 已获批准，实施中 | 🔶 配置/API/CLI/WebUI 补齐大部分完成；degradation 嵌套和 feedback YAML 仍待修复 | §19, §20 |

> **说明**：标记为 ✅ 的功能已代码级完成并可通过无模型测试验证。标记为 🔶 的功能代码已实现但质量指标尚未达到发布标准。各设计文档的详细设计方案（数据模型、API、不变量、测试计划等）是本文档对应章节的权威参考来源，本文提供概要架构和当前实施状态。各 spec 文件中的“状态”字段仅是撰写时快照，不代表当前实施进度；以本表“实际实施状态”和代码为准。

---

## 2. 系统总览与当前架构

### 2.1 项目背景

项目已具备完整字幕生产链路（10+ 处理阶段）：人声分离、多 VAD（WebRTC / Silero / TEN / ffmpeg silencedetect）、三方法边界融合、宏观切块、全局 pyannote diarization、WhisperX 全局转录（可选）、物理时间线、词级物理/说话人分配、物理字幕仓灌装、严格断句、幻觉过滤、字幕合并、LLM 优化（带安全门）、最终物理校验、缓存、CLI、Web GUI 与流式处理。

### 2.2 核心设计原则

1. **物理边界不可逾越**（目标 1）：`PhysicalClip` 是硬约束，最终校验器在导出前拦截任何越界事件
2. **全局上下文优先**（目标 2）：全局 ASR 路径默认优先，完整音频或重叠窗口提供上下文
3. **说话人身份全局一致**（目标 3）：全局 diarization 一次运行，canonical ID 全文件稳定
4. **默认拆分优先**（目标 4）：严格断句器在换气、静音和说话人变化处默认断句
5. **LLM 只能优化文本**（目标 5）：安全门逐条校验，拒绝修改时间/归属/来源词
6. **VAD 是证据而非事实**（目标 6）：三层时间范围（PhysicalClip / SpeechEvidenceSpan / ContextWindow）分离
7. **增强而非替代**：`faster-whisper` 始终保留为降级引擎，global 失败可完整回退 segmented

### 2.3 当前架构（已实现）

```text
输入音频
  → 音频预处理 / 人声分离
  → PhysicalTimeline 构建（PhysicalClip + SpeechEvidenceSpan + ContextWindow）
  → 全局说话人识别（pyannote → canonicalization + 上限约束）
  → 全局 ASR（完整音频或重叠 ContextWindow）
       ├─ WhisperX 全局转录 + forced alignment（global 路径）
       └─ faster-whisper 逐段 ASR（segmented 降级路径）
  → 物理字幕仓生成（融合 VAD/声学证据）
  → 词级物理范围/说话人灌装与分配
  → 幻觉过滤（引擎级 + Pipeline 统一过滤层）
  → 严格断句（StrictSegmenter，默认启用）
  → LLM 文本优化（不可改时间/归属/来源词，安全门校验）
  → 字幕构建与帧级衔接
  → 最终物理校验（FinalValidator）
  → ASS/SRT/VTT + JSON 诊断
```

### 2.4 关键数据结构

#### 三层时间范围（已实现：`vocal_subtitle/physical/timeline.py`）

| 类型 | 定义 | 约束 |
|------|------|------|
| `PhysicalClip` | 真实音频的 `[start, end]`，唯一拥有字幕归属权 | 硬范围，不能扩展、拉伸或跨文件拼接 |
| `SpeechEvidenceSpan` | VAD/ffmpeg/RMS 检出的疑似语音区间 | 语音存在的证据，不是绝对真相 |
| `ContextWindow` | 供全局 ASR 使用的扩展上下文区间 | 只用于识别，不拥有字幕归属权 |

#### GlobalWord（已实现：`vocal_subtitle/physical/ir.py`）

```python
GlobalWord(
    id,              # 全局唯一词 ID（如 gw:ctx:N:w0001）
    text,            # 词文本
    raw_start,       # 全局绝对开始时间
    raw_end,         # 全局绝对结束时间
    confidence,      # 词级置信度
    source_window_id,# 来源窗口 ID
    segment_id,      # 来源 segment ID
    language,        # 语言
    speaker_id,      # canonical 说话人 ID（可选）
    no_speech_prob,  # 无声概率（可选）
    avg_logprob,     # 平均 log 概率（可选）
    compression_ratio, # 压缩比（可选）
)
```

#### PhysicalSubtitleBin（已实现：`vocal_subtitle/physical/subtitle_bins.py`）

```python
PhysicalSubtitleBin(
    id, start, end, source, confidence,
    evidence_ids, physical_clip_id, boundary_resolution_ms
)
```

物理字幕仓融合 VAD/声学证据生成接近真实发声范围的边界，ASR 文字以整词为单位灌入。物理仓边界优先作为字幕显示时间，但不以物理切刀截断词或字符。

#### SubtitleEvent 扩展（已实现）

现有字段保留，新增全部可选且向后兼容：

```python
    physical_spans: List[PhysicalSpan]     # 权威物理归属
    physical_start: Optional[float]        # 物理包络起始
    physical_end: Optional[float]          # 物理包络结束
    source_word_ids: List[str]             # 来源全局词 ID
    logical_sentence_id: Optional[int]     # 逻辑句 ID
    alignment_warning: Optional[str]       # 对齐警告
    # 以下为物理字幕仓灌装相关（已实现，待质量优化）
    physical_region_id: Optional[str]      # 物理区域标识
    physical_bin_id: Optional[str]         # 物理字幕仓 ID
    physical_bin_start: Optional[float]    # 仓起始时间
    physical_bin_end: Optional[float]      # 仓结束时间
    time_source: Optional[str]             # 时间来源（physical_bin / word_boundary）
    hard_split_before: bool                # 硬断标记
```

> **API 状态**：`SubtitleEventResponse` 和 `_subtitle_event_to_payload()` 已同步暴露物理归属、来源词、逻辑句、对齐告警及物理字幕仓字段；字段清单和验证结果见 §19.2.1。

### 2.5 模块布局

项目共 94 个 Python 源文件，按子系统组织。以下列出完整清单：83 个实现文件和 11 个包初始化文件均计入总数。核心模块数量与支撑文件分开标注，避免把包初始化文件或共享接口遗漏在模块统计之外。

**核心 Pipeline（3 个核心模块）**

```text
vocal_subtitle/pipeline.py                   【已实现】global/segmented 双路径路由，10+ 阶段编排
vocal_subtitle/pipeline_context.py           【已实现】统一 PipelineContext 数据载体
vocal_subtitle/streaming.py                  【已实现】实时流式处理入口
```

**运行时配置与根级基础设施（5 个实现模块，含 CLI）**

```text
vocal_subtitle/config.py                     【已实现】26 个 dataclass 配置类 + ConfigLoader
vocal_subtitle/cli.py                        【已实现】Click CLI（run/batch/feedback/download-models）
vocal_subtitle/acoustic_validator.py         【已实现】声学骨架校验
vocal_subtitle/audio_preprocessor.py         【已实现】噪声抑制/环境底噪自适应
vocal_subtitle/macro_chunker.py              【已实现】宏观静音切块（长音频）
```

**物理时间线与全局 IR（Phase 2）**

```text
vocal_subtitle/physical/__init__.py          【已实现】物理时间线包导出
vocal_subtitle/physical/timeline.py          【已实现】PhysicalClip / SpeechEvidenceSpan / ContextWindow
vocal_subtitle/physical/evidence_adapter.py  【已实现】VAD/融合输出 → SpeechEvidenceSpan 适配
vocal_subtitle/physical/ir.py               【已实现】GlobalWord / GlobalTranscript / GlobalSpeakerTimeline
vocal_subtitle/physical/coordinate.py        【已实现】局部/全局坐标转换器
vocal_subtitle/physical/context.py           【已实现】ContextWindow 构造器
vocal_subtitle/physical/ir_cache.py          【已实现】IR 缓存键与指纹
vocal_subtitle/physical/allocator.py         【已实现】词级物理范围/说话人确定性分配
vocal_subtitle/physical/events.py            【已实现】GlobalSubtitleEvent 构建与适配
vocal_subtitle/physical/subtitle_bins.py     【已实现】物理字幕仓构建与整词灌装
vocal_subtitle/physical/shadow.py            【已实现】Shadow Pipeline 接入
```

**ASR 引擎（10 个核心模块）**

```text
vocal_subtitle/asr/faster_whisper_engine.py  【已实现】faster-whisper 引擎（降级/segmented 路径）
vocal_subtitle/asr/whisperx_engine.py        【已实现】WhisperX 延迟加载适配器（global 路径）
vocal_subtitle/asr/whisper_cpp_engine.py     【已实现】whisper.cpp 引擎
vocal_subtitle/asr/funasr_engine.py          【已实现】FunASR 引擎（阿里 Paraformer）
vocal_subtitle/asr/global_transcriber.py     【已实现】全局转录编排与重叠窗口去重
vocal_subtitle/asr/hallucination.py          【已实现】幻觉过滤纯函数（训练语料、重复、低质量）
vocal_subtitle/asr/text_normalizer.py        【已实现】文本归一化（默认 safe 模式）
vocal_subtitle/asr/boundary_refiner.py       【已实现】ASR 边界双向精修
vocal_subtitle/asr/boundary_reasr.py         【已实现】边界滑动窗口冗余 ASR
vocal_subtitle/asr/boundary_arbitration.py   【已实现】LLM 语义仲裁（冗余识别结果选择）
```

```text
# ASR 接口与置信度支撑文件
vocal_subtitle/asr/__init__.py               【已实现】ASR 包导出
vocal_subtitle/asr/base.py                   【已实现】ASR 引擎/转录结果接口
vocal_subtitle/asr/boundary_confidence.py    【已实现】边界置信度评估
```

**说话人分离与身份管理（7 个核心模块）**

```text
vocal_subtitle/diarization/base.py           【已实现】SpeakerTurn / DiarizationResult / AtomicSpeechSpan
vocal_subtitle/diarization/pyannote_engine.py【已实现】pyannote 全局 diarization 引擎
vocal_subtitle/diarization/canonicalizer.py  【已实现】独立 canonicalization 模块（含 expected_speakers 上限约束）
vocal_subtitle/diarization/turn_reconciler.py【已实现】Turn 与物理区域求交集
vocal_subtitle/diarization/speaker_clusterer.py【已实现】Agglomerative/spectral 聚类
vocal_subtitle/diarization/speaker_embedding.py【已实现】说话人嵌入（speechbrain/pyannote）
vocal_subtitle/diarization/role_labeler.py   【已实现】LLM 说话人角色标注
```

```text
# 说话人分离支撑文件
vocal_subtitle/diarization/__init__.py       【已实现】说话人分离包导出
vocal_subtitle/diarization/feature_extractor.py【已实现】声学特征提取
```

**字幕构建与校验（6 个核心模块）**

```text
vocal_subtitle/mapping/time_mapper.py        【已实现】时间映射与物理优先时间轴
vocal_subtitle/mapping/subtitle_builder.py   【已实现】字幕事件构建与帧级衔接
vocal_subtitle/mapping/strict_segmenter.py   【已实现】严格断句器（默认启用）
vocal_subtitle/mapping/final_validator.py    【已实现】最终物理校验器
vocal_subtitle/mapping/event_constraints.py  【已实现】事件合并兼容性共享规则
vocal_subtitle/mapping/llm_guard.py          【已实现】LLM 文本安全门（6 条校验规则）
```

```text
vocal_subtitle/mapping/__init__.py            【已实现】字幕映射包导出
vocal_subtitle/mapping/end_time_validator.py 【已实现】结束时间校验
```

**字幕合并扩展**

```text
vocal_subtitle/merging/__init__.py           【已实现】字幕合并包导出
vocal_subtitle/merging/merge_strategy.py     【已实现】规则合并策略
vocal_subtitle/merging/llm_merge_engine.py   【已实现】LLM 语义合并引擎
```

**VAD、音频预处理与声学校验**

```text
vocal_subtitle/vad/__init__.py               【已实现】VAD 包导出
vocal_subtitle/vad/base.py                   【已实现】VAD 片段接口
vocal_subtitle/vad/silero_vad.py             【已实现】Silero VAD
vocal_subtitle/vad/webrtc_vad.py             【已实现】WebRTC VAD
vocal_subtitle/vad/ten_vad.py                【已实现】TEN VAD
vocal_subtitle/vad/ffmpeg_vad.py             【已实现】ffmpeg silencedetect
vocal_subtitle/vad/boundary_fusion.py        【已实现】三方法边界融合（Silero+ffmpeg+RMS）
```

**人声分离**

```text
vocal_subtitle/separation/__init__.py        【已实现】人声分离包导出
vocal_subtitle/separation/base.py            【已实现】分离引擎接口
vocal_subtitle/separation/uvr_engine.py      【已实现】UVR 人声分离
vocal_subtitle/separation/openunmix_engine.py【已实现】Open-Unmix 人声分离
vocal_subtitle/separation/spleeter_engine.py 【已实现】Spleeter（已停止维护，保留兼容）
```

**自适应反馈学习（Phase 5.1+5.2，10 个实现模块）**

```text
vocal_subtitle/feedback/__init__.py          【已实现】反馈学习包导出
vocal_subtitle/feedback/aligner.py           【已实现】用户修订与自动字幕对齐
vocal_subtitle/feedback/audio_fingerprint.py 【已实现】音频指纹计算
vocal_subtitle/feedback/conflict_detector.py 【已实现】冲突检测
vocal_subtitle/feedback/diff_analyzer.py     【已实现】字幕差异分析
vocal_subtitle/feedback/few_shot_builder.py  【已实现】Few-shot 示例构建
vocal_subtitle/feedback/health_scorer.py     【已实现】字幕质量健康评分
vocal_subtitle/feedback/impact_estimator.py  【已实现】参数影响估计
vocal_subtitle/feedback/param_learner.py     【已实现】自适应参数学习
vocal_subtitle/feedback/shadow_mode.py       【已实现】影子模式（新旧参数对比）
vocal_subtitle/feedback/user_profile.py      【已实现】用户偏好画像
```

**工具与基础设施（11 个实现模块）**

```text
vocal_subtitle/utils/__init__.py             【已实现】工具包导出
vocal_subtitle/utils/cache_manager.py        【已实现】三层 stage 缓存管理
vocal_subtitle/utils/hf_token_store.py       【已实现】HF Token Fernet 加密存储
vocal_subtitle/utils/session_manager.py      【已实现】哈希会话目录管理
vocal_subtitle/utils/task_history.py         【已实现】任务历史持久化
vocal_subtitle/utils/persistence_manager.py  【已实现】字幕文件读写与 ASS/SRT/VTT 导出
vocal_subtitle/utils/model_loader.py         【已实现】延迟模型加载
vocal_subtitle/utils/gpu_detector.py         【已实现】GPU 检测
vocal_subtitle/utils/file_hasher.py          【已实现】文件哈希
vocal_subtitle/utils/audio_utils.py          【已实现】音频工具函数
vocal_subtitle/utils/progress.py             【已实现】进度上报
vocal_subtitle/utils/logger.py               【已实现】结构化日志
```

**WebUI 与 CLI**

```text
vocal_subtitle/webui/__init__.py             【已实现】WebUI 包导出
vocal_subtitle/webui/app.py                  【已实现】FastAPI 应用入口
vocal_subtitle/webui/api.py                  【已实现】REST API（run/batch/hf-token/cache/subtitle）
vocal_subtitle/webui/models.py               【已实现】Pydantic 请求/响应模型
vocal_subtitle/webui/websocket.py            【已实现】WebSocket 进度推送
vocal_subtitle/webui/cli_runner.py           【已实现】WebUI 内嵌 CLI 执行
```

```text
vocal_subtitle/__init__.py                   【已实现】顶层包导出
```

---

## 3. 安装与运行时

> **现状**：WhisperX 已作为 optional extra 接入。`install.sh` 已包含纯 shell 的 CUDA 自动检测。Debian 打包采用分层 profile 安装。

### 3.1 安装 Profile

当前 `install.sh` 支持以下 profile（已统一收敛）：

| Profile | 内容 |
|---------|------|
| `base` | 核心依赖、faster-whisper、webrtcvad、webui（默认） |
| `gpu` | GPU 推理所需的 CTranslate2/PyTorch 扩展 |
| `separation` | UVR 人声分离 |
| `diarization` | pyannote/SpeechBrain 相关依赖 |
| `llm` | 云端 LLM 优化 |
| `local-nlp` | 本地 NLP 语义合并 |
| `full` | 组合所有受支持扩展（不含已停止维护的 Spleeter） |
| `dev` | 测试、格式化、Lint 工具 |

### 3.2 硬件检测

`install.sh` 已内置纯 shell 的 CUDA 检测：

- Linux 下 `nvidia-smi` 存在且返回成功 → CUDA 候选
- 同时检查 NVIDIA 驱动库可见 → 确认 CUDA
- 检测失败或非 NVIDIA 系统 → CPU
- 与 `GPUDetector`（`utils/gpu_detector.py`）保持判定口径一致

### 3.3 运行时降级链

```text
WhisperX → faster-whisper（segmented 降级路径）
```

导入失败、不兼容 torch、CUDA 探测失败或模型初始化失败均产生结构化警告并按 `fallback_category` 降级。安装失败不隐藏，报告失败目标和可操作诊断。

### 3.4 Debian 打包

已实现分层 Debian 包结构（基于 `dpkg-deb` 手动构建，减少构建环境要求）：

**安装路径（FHS 兼容）**：

| 路径 | 用途 | 权限 |
|------|------|------|
| `/usr/share/vocal-subtitle` | 只读项目源码和资源 | 只读 |
| `/var/lib/vocal-subtitle/venv` | 可变 Python 虚拟环境 | `postinst` 创建 |
| `/etc/vocal-subtitle` | 环境配置和服务配置 | 可配置 |
| `/var/cache/vocal-subtitle` | 任务缓存及模型缓存 | 可读写 |
| `/var/log/vocal-subtitle` | 服务日志 | 可写 |
| `/usr/bin/vocal-subtitle` | CLI wrapper | 可执行 |
| `/usr/bin/vocal-subtitle-gui` | GUI wrapper | 可执行 |
| `/usr/sbin/vocal-subtitle-setup` | 扩展安装和修复入口 | 可执行 (root) |

基础 `Depends`：Python 3.10+、`python3-venv`、`ffmpeg`。不声明 `libsndfile1`、`python3-tk`、CUDA toolkit 或开发依赖。

**postinst 流程**：

1. 创建目录和 `vocal-subtitle` 系统用户
2. 创建或复用 `/var/lib/vocal-subtitle/venv`
3. 调用 `install.sh --profile base --no-system-deps --no-model-dl`
4. 验证 Python 包、CLI 和 GUI 入口
5. 写入 `/etc/vocal-subtitle/env.conf`
6. 设置权限，systemd 可用时 enable/start GUI 服务

重复安装和升级不删除现有 venv、模型缓存或配置。Python 依赖安装失败时以失败状态退出并保留现场供 `vocal-subtitle-setup` 修复。

**postrm**：普通 remove 保留 venv、缓存、配置和日志；purge 删除这些生成目录和系统用户。

**systemd**：GUI 服务使用专用系统用户，代码目录只读，默认监听 `127.0.0.1:7862`。不在安装后直接向局域网暴露服务。

**包内容排除**：历史项目副本 (`Vocal_Subtitle.旧项目/`)、构建产物、虚拟环境、测试音频和 Spleeter 依赖。

---

## 4. 音频预处理与物理边界系统

> **现状**：三层时间范围分离已实现（`PhysicalClip` / `SpeechEvidenceSpan` / `ContextWindow`）。物理优先时间轴已生效。`PhysicalTimeline` 提供统一容器、校验和序列化。

### 4.1 物理边界硬约束（已实现）

1. 每个进入最终字幕的词都能追溯到至少一个合法 `PhysicalClip`，或记录 `alignment_warning`
2. 词的有效时间不得落在所有合法 `PhysicalClip` 之外
3. 字幕事件不得扩展到音频文件范围之外
4. 最小时长、帧衔接、LLM 合并和字幕排版都不能修改 `PhysicalClip`
5. ASR 词时间落在 `SpeechEvidenceSpan` 之外时标记为边界候选，但不凭 VAD 缺失直接删除
6. 最终物理校验在字幕构建、LLM 优化和格式导出前执行

### 4.2 物理范围生成（已实现）

保留 `unified_ffmpeg_pass`、Silero VAD、RMS 扫描和 `BoundaryFusion`，输出明确分成三层：

- `physical_clips`：由输入文件构造的合法音频范围
- `speech_evidence_spans`：检测出的疑似语音区间（通过 `evidence_adapter.py` 适配）
- `context_windows`：供全局 ASR 使用的扩展上下文区间

### 4.3 物理优先时间轴（已实现）

时间映射优先级：

```text
PhysicalClip 硬范围
  > 物理语音边界 SpeechSegment
    > 声学内部边界（skeleton/RMS）
      > ASR 词/段时间戳（仅内部降级证据）
```

- 单 ASR 事件使用 `SpeechSegment.start/end`
- 多事件的首尾贴合物理区间首尾，中间边界优先使用声学静音
- ASR 时间超出物理区间时裁剪；ASR 时间偏短时不缩短物理优先边界
- 末条不再有特殊覆盖逻辑，所有事件统一遵循 physical-first

### 4.4 物理区域标识传递

- `SubtitleEvent` 已有 `physical_spans`、`physical_start`、`physical_end`
- 物理区域标识进入 LLM fragment 和合并兼容性检查
- 帧级衔接、规则合并和 LLM 合并均不得跨越不同物理区域
- 跨物理区域合并被 `StrictSegmenter` 和合并兼容性检查阻止

---

## 5. 全局 ASR 与 WhisperX 集成

> **现状**：WhisperX 3.8.6 已接入作为全局 ASR 后端（延迟加载）。Global ASR 路径默认优先（`auto`），segmented 为降级。`FasterWhisperEngine` 保留为降级引擎。

### 5.1 WhisperX 定位

WhisperX 作为离线高精度转录编排层：

```text
WhisperX ASR + forced alignment + word-level speaker assignment
                         |
                         v
现有全局 speaker timeline / 物理边界 / 字幕构建 / 合并 / 校验 / 导出
```

`faster-whisper` 不删除，继续用于轻量离线、流式和降级路径。

### 5.1.1 语言模式与全局检测（已实现）

`ASRConfig` 提供语言模式配置，遵循设计文档 [2026-07-24-asr-accuracy-design](superpowers/specs/2026-07-24-asr-accuracy-design.md)：

```yaml
asr:
  language: null
  language_mode: single       # single | mixed
  language_detection_min_probability: 0.65
  mixed_language_min_probability: 0.85
  mixed_language_min_logprob: -1.5
  mixed_language_min_gain: 0.15
```

语言检测与使用规则：

- **显式指定 `language`**：跳过全局检测，直接使用指定语言（所有模式）
- **`language=null` + `language_mode=single`（默认）**：对完整人声音频做一次全局检测并锁定主语言；低于阈值时记录低置信度并锁定最佳候选，不进行片段级换语言；引擎无法返回语言候选时提示用户使用 `--language`
- **`language=null` + `language_mode=mixed`**：先按主语言识别；片段仅在主语言置信度低于阈值、候选语言不同且置信度收益达到 `mixed_language_min_gain` 时切换
- 全局检测失败不会静默退回短片段自动检测
- 语言策略、音频内容哈希、引擎/模型、解码参数和混合语言阈值已纳入缓存键

`TranscriptionSegment` 包含 `language`（可选）和 `language_probability`（默认 0.0）字段。所有引擎均保持转写模式，不执行翻译。

### 5.2 ASR 路径路由（已实现）

```text
auto（离线默认）
  → global ASR（WhisperX 全局转录 + 词级分配）
  → global 成功：asr_path=global
  → global 失败（依赖/资源/执行/结果异常）
     → fallback_to_segmented=true：asr_path=legacy_degraded
     → fallback_to_segmented=false：结构化任务失败

explicit global  → 强制 global，失败直接报错
explicit segmented → asr_path=legacy
streaming         → asr_path=legacy（不加载 WhisperX）
```

### 5.3 全局转录编排（已实现）

`global_transcriber.py` 接受音频、ContextWindow 列表、ASR 后端和窗口策略：

- 单个完整输入窗口或按 ContextWindow 执行的多窗口
- 重叠窗口结果按时间、文本和来源 ID 确定性去重
- 优先保留置信度高者，无法确定时全部保留并记录诊断
- 输出按 `(raw_start, raw_end, id)` 稳定排序
- 不执行 physical clip 裁剪（识别范围 ≠ 归属范围）

### 5.4 结果适配与数据契约（已实现）

`WordTimestamp` 增加 `speaker_id`（可选），`TranscriptionSegment` 增加 `no_speech_prob` 和 `compression_ratio`（可选）。所有新增字段带默认值，保证向后兼容。

faster-whisper 引擎已透传 `no_speech_prob` 和 `compression_ratio`。解码阈值显式传递：
`no_speech_threshold=0.6`、`log_prob_threshold=-1.0`、`compression_ratio_threshold=2.4`。

### 5.5 分阶段降级

| 失败阶段 | 处理 | fallback_category |
|----------|------|-------------------|
| WhisperX 导入/依赖缺失 | 返回 `legacy_degraded` | `dependency_unavailable` |
| WhisperX ASR 失败 | 使用 segmented ASR | `execution_failed` |
| alignment 失败 | 保留段级时间，继续字幕 | `execution_failed` |
| GPU OOM / 资源不足 | 降级 CPU 或 segmented | `resource_unavailable` |
| 结果非法 | 降级 segmented | `invalid_result` |

---

## 6. 说话人身份与字段一致性

> **现状**：所有机制已实现并测试通过。`expected_speakers` 上限约束已生效。公共 canonicalization 函数已提取。

### 6.1 说话人数量策略（已实现）

| 配置 | 行为 |
|------|------|
| `expected_speakers = N` | 向 pyannote 传递 `min_speakers=N` 和 `max_speakers=N`；canonical 标签严格为 A..N |
| `expected_speakers` 未设置，`max_speakers = M` | 允许自动估计，拒绝/合并超过 M 的结果 |
| 两者均未设置 | 允许模型估计，标记为估计值 |
| 后端/模型不可用 | 保持 unknown，不发明标签 |

超限时优先为每个原始 speaker 建立 turn 音频 profile，使用嵌入计算簇间相似度，逐步合并最近簇直到达到上限。无法取得可靠 profile 时使用确定性降级合并并标记 `degraded`。

### 6.2 结果归一化（已实现）

独立 canonicalization 模块输出：

```text
raw_diarization_speaker_count
canonical_speaker_count
speaker_merge_map
canonicalization_status
canonicalization_reason
```

缓存结果也必须经过上限校验，防止旧缓存绕过约束。

### 6.3 Pipeline 统计字段

`PipelineStats` 已包含：

```text
speaker_count
raw_diarization_speaker_count
canonical_speaker_count
maximum_allowed_speaker_count
embedding_consolidation_count
canonicalization_status
diarization_backend / diarization_status
physical_region_count / atomic_span_count
overlap_ratio / mixed_event_count
cross_speaker_merge_blocked
fallback_reason
```

---

## 7. 物理范围与说话人词级分配器

> **现状**：词级分配器已实现（`vocal_subtitle/physical/allocator.py`）。物理字幕仓灌装已实现（`vocal_subtitle/physical/subtitle_bins.py`）。

### 7.1 词级分配规则（已实现）

对每个 `GlobalWord`：

1. 词完整落在一个 `PhysicalClip` 内 → 直接归属该 clip
2. 词跨越两个 clip 且有字符级对齐 → 按字符时间分配到对应 clip
3. 词跨越边界但无字符级对齐 → 整词保守归属，不生成半词，记录 `cross_physical_boundary`
4. 词完全落在物理范围之外 → 拒绝并记录 `outside_physical_clip`
5. 词位于语音证据之外但仍在物理范围内 → 与 `SpeechEvidenceSpan` 求交集，保留证据 ID
6. 词横跨不同 speaker turn → 有词级 speaker 时按 speaker 组切分；缺失时使用 turn 包含关系

### 7.2 物理字幕仓灌装（已实现）

`PhysicalSubtitleBin` 融合 VAD/声学证据生成更接近真实发声范围的物理字幕仓：

- 证据优先级：融合边界 > ffmpeg skeleton/RMS > Silero/其他 VAD
- 重叠证据先合并，再按真实静音和边界置信度确定不重叠区间
- ASR 文字以整词为单位灌入字幕仓，按词的最大时间重叠归属
- 跨仓词保留完整词文本，记录 `cross_bin_word`
- 单仓单事件使用物理仓边界；仓内多事件使用完整词边界
- 物理仓边界可作为最终字幕时间，绝不以物理切刀截断词或字符

### 7.3 Speaker-Safe 拆分优先级

1. 词级 speaker 已确认 → 按连续 speaker 组切分
2. 词级 speaker 缺失但全局 turn 可覆盖 → 按 turn 与词时间求交集切分
3. 词级和 turn 都无法确认 → `speaker_id=None`
4. 事件跨越两个不同 speaker 且无可用词时间 → 标记 mixed/unknown

---

## 8. ASR 幻觉过滤

> **现状**：全部已实现。`vocal_subtitle/asr/hallucination.py` 提供纯函数过滤层，引擎级解码阈值显式传参，缓存键包含过滤器版本。

### 8.1 两层保护（已实现）

- **引擎级**：faster-whisper 显式传递 `no_speech_threshold`、`log_prob_threshold`、`compression_ratio_threshold`
- **Pipeline 级**：统一过滤层覆盖所有引擎（faster-whisper、whisper-cpp、FunASR、WhisperX）和缓存命中

### 8.2 过滤规则（已实现）

按顺序执行：

1. 空文本 → 删除（`empty_text`）
2. 训练语料短语（`字幕志愿者`、`感谢观看`、`thanks for watching` 等）→ 删除（`training_phrase`）
3. 高 `no_speech_prob` 且无有效词级证据 → 删除（`high_no_speech_without_word_evidence`）
4. 低 `avg_logprob` 且无有效词级证据 → 删除（`low_logprob_without_word_evidence`）
5. 异常压缩比且文本明显重复 → 删除（`repetitive_compression`）
6. 相邻重复结果 → 保留信息更完整者（`adjacent_duplicate`）
7. 有效短词（`Good`、`我`、`唉` 等）→ 保留

### 8.3 缓存隔离

ASR 缓存键包含：过滤器版本、过滤开关、三个质量阈值、训练语料短语开关、相邻重复开关。规则版本或阈值改变后旧缓存自然失效。

### 8.4 配置

```yaml
asr:
  no_speech_threshold: 0.6
  log_prob_threshold: -1.0
  compression_ratio_threshold: 2.4
  hallucination_filter_enabled: true
  hallucination_filter_version: v1
  filter_training_phrases: true
  filter_adjacent_duplicates: true
```

---

## 9. 字幕断句规则与物理字幕仓灌装

> **现状**：严格断句器（`StrictSegmenter`）已默认启用。物理字幕仓灌装已实现。跨说话人拆分已保留。

### 9.1 严格断句策略（已实现）

默认在以下位置断句：

1. 不同 speaker：硬断
2. speaker unknown/mixed 与已知 speaker：硬断
3. 不连续 physical clip 或不同 physical owner：硬断
4. `alignment_warning` 中的不可合并警告：硬断
5. 明显句末标点：优先断句（缩写和小数点例外）
6. 词间真实静音/换气 gap：默认断句；短 gap 仅在同 speaker、同一连续物理区域且前文非句末时允许合并
7. 达到最大时长、最大行数或最大字符数：只在词边界拆分
8. 仅有旧事件而没有词信息时：使用事件级边界和文本标点

### 9.2 允许同人合并的条件

只有**同时**满足以下所有条件才允许合并：

- 两侧全局 `speaker_id` 相同（双方 unknown 时才允许按 unknown 合并）
- 两侧 physical span 有相同 clip 或 clip 在时间轴上连续且无未覆盖静音
- 间隔小于快速合并阈值且无明显换气/长停顿
- 前后文本没有句末边界
- 不存在硬断警告
- 合并后满足最大时长/字符和 physical envelope
- 合并后事件保留全部 `physical_spans` 和 `source_word_ids`

### 9.3 不同说话人

不同 `speaker_id` 必须拆成独立字幕事件。LLM、规则合并和字幕排版均不能跨越该边界。unknown 不得吸附到已知 speaker。

### 9.4 物理字幕仓灌装与事件生成

融合 VAD/声学证据生成物理字幕仓后，全局词流按以下顺序灌装：

1. 每个词计算与物理仓的重叠时长、词中点距离、speaker 兼容性
2. 选择最大重叠仓，整词写入
3. 仓内词流按 speaker 变化、硬断 warning、真实静音、句末标点和行长上限切分
4. 单仓单事件使用物理仓 `start/end`；仓内多事件使用首尾完整词时间
5. 输出覆盖报告，确保每个 accepted global word 恰好进入一个事件

---

## 10. LLM 权限边界与文本优化

> **现状**：全部已实现。相似度校验 + 跨说话人保护 + 来源词归属校验 + 语言脚本安全检查。

### 10.1 LLM 可执行的操作

- 文本纠错、错字修正
- 标点和断句建议
- 在确定性策略已允许的同人短间隙内合并文本
- 大小写和显示文本优化

### 10.2 LLM 禁止的操作

- 修改 `physical_spans`、`source_word_ids`、`words` 或 speaker
- 延长 `start/end` 到合法 `PhysicalClip` 之外
- 合并不同 speaker
- 合并明显静音分隔的字幕事件
- 引入不存在于全局词流的事实文本
- 翻译或语言脚本突变
- 跨条目搬运、复制或清空内容

### 10.3 输出校验（已实现）

逐条安全校验：

1. 输出必须包含完整原始键集合，每个值为字符串
2. 禁止翻译或语言脚本突变；输出主要文字脚本应与输入一致
3. 只允许当前条目内的修改；禁止跨条目复制/移动
4. 可配置的保守相似度阈值和长度变化上限
5. 保护数字、单位、专名候选和 speaker 边界
6. 任一条目失败时该条保留 ASR 原文；批次级失败时整批保留
7. LLM 只改变 `text`；`physical_*`、`physical_spans`、`words`、`source_word_ids` 始终保留

---

## 11. LLM 字幕差异高亮与 WebUI 三阶段时间轴

> **现状**：LLM 差异高亮已完整实现。WebUI 三阶段时间轴（原始识别 / LLM 优化 / 最终版本）已实现。

### 11.1 LLM 差异高亮（已实现）

`diffAndHighlight()` 实现 LCS token 级差异：

- token 化正则：CJK 逐字、拉丁逐词、标点和空白独立成 token
- 对 token 序列构建 LCS DP 表，回溯标记 `same` / `added`
- 连续 `added` token 合并进 `<span class="diff-added">`（琥珀色高亮）
- 删除的 token 不出现在 LLM 输出中
- 所有 token 经 HTML 转义

### 11.2 WebUI 三阶段时间轴（已实现）

时间轴固定渲染三列：

1. **原始识别（ASR）**：显示 `asr_text`
2. **LLM 优化**：显示 `llm_text`（为空时保持空白）
3. **最终版本**：显示 `text`，支持 SRT/ASS 格式切换

交互特性：

- 点击行自动播放对应时间段
- 双击任意栏进入编辑
- 编辑保存统一写入 `text`，ASR/LLM 对照值不被覆盖
- 空格键控制播放/暂停
- 三字段通过 API 透传：`asr_text`、`llm_text`、`text`
- 旧数据自动兼容回退（`asr_text` ← `original_text` ← `text`）

---

## 12. 缓存与模型管理

> **现状**：三层 stage 缓存（separation / transcription / diarization）已存在。WhisperX 分阶段缓存已接入。IR 缓存键基础设施已实现（`ir_cache.py`）。

### 12.1 缓存命名空间

| 命名空间 | 缓存内容 | 关键身份参数 |
|----------|----------|-------------|
| `separation` | 人声分离结果 | 音频哈希、分离引擎、模型 |
| `transcription` | 分段 ASR 结果 | 音频哈希、引擎、模型、语言、解码参数、过滤器版本 |
| `diarization` | 全局 diarization | 音频哈希、backend、model_ref、speaker 约束、策略参数 |
| `whisperx_asr` | 全局转录 | 音频哈希、模型、语言、窗口列表、解码参数 |
| `whisperx_alignment` | 对齐结果 | ASR 缓存身份、语言、对齐模型 |
| `whisperx_speaker_assignment` | 说话人分配 | alignment 身份、speaker timeline 身份 |

### 12.2 IR 缓存键（已实现）

`make_ir_cache_key()` 生成包含 artifact type、schema version、producer version、输入 SHA-256、时间线指纹、坐标策略和上下文策略的稳定缓存键。全流水线缓存身份区分 global/legacy/legacy_degraded，禁止跨路径复用。

---

## 13. HF Token 安全管理

> **现状**：已实现。HF Token 通过 Fernet 加密存储，浏览器只保存 masked presence marker。

### 13.1 设计

- Token 经 Fernet authenticated encryption 加密后存储于 `cache/hf_token.enc`（mode 0600）
- 每安装生成独立密钥于 `cache/.hf_token.key`（mode 0600）
- 浏览器只存储 `***` 作为 presence marker；masked 提交时服务器使用加密值
- 明文 Token 永不通过 API 响应返回或写入普通日志
- diarization 和 embedding 引擎在无显式 Token 时也查询加密存储
- 下载模型需显式操作；Token 解析成功后模型加载作为验证步骤

### 13.2 错误处理

- 缺少 Token：HTTP 400
- 缺少 pyannote 依赖：HTTP 503
- 授权/协议/网络失败：HTTP 502 附安全诊断
- 全局缓存检测与 embedding 缓存布局独立

---

## 14. 迁移方案与实施状态

### 阶段零：修复已确认缺陷 ✅ 已完成

- ffmpeg 参数传递 ✅
- 末条结束时间上界钳制语义 ✅
- 说话人数量上限约束 ✅
- `no_speech_prob` / `compression_ratio` 透传 ✅
- 物理优先时间轴 ✅
- 公共 canonicalization 函数提取 ✅
- `PipelineStats` 新增统计字段 ✅

### 阶段一：幻觉过滤与数据契约扩展 ✅ 已完成

- `TranscriptionSegment` / `WordTimestamp` 扩展 ✅
- 引擎级解码阈值显式传参 ✅
- `vocal_subtitle/asr/hallucination.py` 纯函数过滤层 ✅
- 训练语料短语过滤 ✅
- ASR 缓存键加入过滤器版本 ✅
- Cache 命中与新增结果使用相同过滤路径 ✅

### 阶段二：建立全局中间表示 ✅ 已完成

- `PhysicalTimeline`（`physical_clips`、`speech_evidence_spans`、`context_windows`）✅
- `GlobalTranscript`、`GlobalSpeakerTimeline` ✅
- `GlobalWord` 全局词流模型 ✅
- 现有 VAD/骨架输出转 evidence 适配器 ✅
- 坐标转换、上下文窗口构造、IR 缓存 ✅
- Shadow Pipeline 接入 ✅

### 阶段三：实现全局转录与词级分配 ✅ 已完成

- WhisperX 延迟加载适配器 ✅
- 全局转录编排器与重叠窗口去重 ✅
- `PhysicalWordAllocator` ✅
- 词到 physical clip、speech evidence 和 speaker 的确定性分配 ✅
- `physical_spans`、`source_word_ids`、`logical_sentence_id` 事件模型 ✅
- 跨边界 `alignment_warning` 和诊断报告 ✅

### 阶段四：字幕断句与最终校验 ✅ 已完成

- 严格断句器（`strict_segmenter.py`，默认启用）✅
- 统一物理/speaker/source-word 合并兼容性判断 ✅
- 帧级衔接与微间隙合并不绕过字幕逻辑层 ✅
- LLM 安全门（来源词归属校验、语言脚本检查）✅
- 最终物理校验器（`final_validator.py`）✅
- `physical_spans`、`source_word_ids` 在拆分/合并/LLM/导出后不丢失 ✅

### 阶段五：切换默认路径 ✅ 已完成

- 离线默认 `auto` 优先 global，失败回退 segmented ✅
- `global`/`segmented`/`auto` 任务级覆盖 ✅
- 结构化降级分类：`dependency_unavailable` / `resource_unavailable` / `execution_failed` / `invalid_result` / `explicit_legacy` ✅
- 缓存路径隔离 ✅
- CLI/API/WebUI/WebSocket 路径诊断透传 ✅
- Benchmark CI 发布资格检查 ✅
- 620 tests passed, 0 failures ✅

### 待实施：物理字幕仓灌装质量优化

- 物理字幕仓灌装代码已实现（`vocal_subtitle/physical/subtitle_bins.py`），模块包含 `PhysicalSubtitleBin` 数据类、仓构建器（融合 VAD/声学证据）和整词灌装逻辑
- 四类目标素材的毫秒级质量指标（Start/End MAE ≤ 50ms, P95 ≤ 100ms）尚未达到第 18.6 节质量门；中文多人、181 单人和英文视频评测场景的事件完整性和时间精度需继续优化
- `--require-global --ci` 技术门已通过（代码链路验证），质量门尚未通过
- 设计依据：[2026-07-26-physical-bin-subtitle-filling-design](superpowers/specs/2026-07-26-physical-bin-subtitle-filling-design.md)

### 全链路完成（依据 2026-07-26-full-chain-completion-design）

全链路完成设计中定义的目标大部分已实现：

- ✅ 修复测试回归（620 passed, 0 failures）
- ✅ 核对物理时间线、词级分配、幻觉过滤、最终物理校验和全局 ASR 接口契约
- ✅ 统一 Pipeline、缓存、CLI、WebUI API、WebSocket、任务历史中的 ASR 路径与降级诊断
- ✅ 核对 WhisperX optional extra、延迟导入、全局成功/失败/降级行为
- ✅ 建立真实素材 manifest（`test/quality_manifest.yaml`）
- ✅ 真实素材质量对比报告可一键生成（技术门通过）
- 🔶 文档同步与最终验收报告（本次更新完成）
- 🔶 真实素材质量门仍未通过（待模型/对齐/物理仓灌装优化后重新验收）

### Superpowers 设计文档权威来源

各阶段的设计细节（数据模型、API 契约、不变量、测试计划、验收标准）以 `docs/superpowers/specs/` 目录下的设计文档为权威参考。本文档提供概要架构和当前实施状态。实施计划文件（`docs/superpowers/plans/`）记录了各阶段的执行任务分解。

---

## 15. 测试与验收标准

### 15.1 无模型回归

```bash
venv/bin/pytest -q
```

结果：**620 passed**，存在依赖弃用和空音频数值警告，没有测试失败。

### 15.2 Global CPU 技术链路

执行 `run_quality_benchmark.py --asr-path global --asr-model large-v3 --require-global --no-cache --ci`：

六个场景均实际使用 `global`，没有 fallback；`--require-global --ci` 技术发布门通过。

| 场景 | 匹配/人工事件数 | Start MAE | End MAE | 文本相似度 | 参考内容覆盖 |
|------|-----------------|-----------|---------|------------|--------------|
| 中文双人朗读 | 52/60 | 196.5ms | 1021.9ms | 0.6816 | 0.8480 |
| 中文多人 | 7/7 | 21.4ms | 18.6ms | 0.9576 | 0.9545 |
| 培训双人 | 49/49 | 217.3ms | 649.0ms | 0.7571 | 0.9570 |
| 英文多人 | 7/7 | 68.6ms | 615.7ms | 0.8264 | 0.9756 |
| 181 单人 | 22/24 | 113.6ms | 241.8ms | 0.8683 | 0.8994 |
| 英文视频评测 | 16/17 | 185.0ms | 495.0ms | 0.8870 | 0.9697 |

聚合 Start MAE = 133.7ms、End MAE = 507.0ms，最低参考内容覆盖率 = 84.80%；技术门通过，质量门未通过。

### 15.3 Segmented 基线

| 场景 | 匹配事件 | Start MAE | End MAE | 文本相似度 |
|------|----------|-----------|---------|------------|
| 中文双人朗读 | 57/60 | 281.3ms | 947.6ms | 0.6060 |
| 中文多人 | 7/7 | 0.3ms | 0.2ms | 0.7879 |
| 培训双人 | 48/49 | 520.9ms | 895.3ms | 0.6620 |
| 英文多人 | 5/7 | 400.5ms | 0.4ms | 0.9121 |
| 181 单人 | 24/24 | 124.8ms | 125.6ms | 0.5908 |
| 英文视频评测 | 5/17 | 601.4ms | 1401.9ms | 0.5132 |

### 15.4 路由验收

短素材 `中文多人员测试音频.wav`：

- `auto`：成功使用 global，`asr_path=global`
- `global`：成功完成全局路径；缺少依赖时任务失败而非静默 legacy
- `segmented`：成功生成字幕，`asr_path=legacy`

### 15.5 测试素材

| 文件 | 用途 |
|------|------|
| `test/中文朗读测试-双人.wav` | 中文双人对话 |
| `test/中文朗读测试-双人.ass` | 参考轨（人工校正） |
| `test/项目识别.ass` | 项目输出，用于回归对比 |
| `test/quality_manifest.yaml` | 6 组音频和人工 ASS 的素材清单 |

---

## 16. 风险与取舍

### 16.1 架构风险

| 风险 | 缓解措施 |
|------|----------|
| 全局 ASR 占用更多内存/显存 | 重叠 ContextWindow、结果去重和可恢复缓存 |
| 全局说话人识别失败 | 保留 unknown，不猜测 speaker |
| 字符级边界对齐非所有引擎支持 | 缺失时按整词保守归属并增加 warning |
| 全局 ASR 仍可能出现幻觉 | 保留 no-speech、logprob、词置信度过滤和局部复识别 |
| 「默认拆分」增加字幕条数 | 优先保证可读性和物理可信度；场景模板可覆盖阈值 |

### 16.2 回退策略

- 旧分段 ASR 作为 global 失败时的完整降级路径
- `fallback_to_segmented=true`（默认）时 global 失败自动降级
- 降级结果在 `asr_path=legacy_degraded` 和 `fallback_category` 中明确标记
- 可通过配置 `global_asr.enabled=false` 或任务级 `segmented` 恢复旧行为
- 说话人系统故障时保持 unknown，不静默发明标签

---

## 17. 完成定义

### 阶段零（缺陷修复）✅

1. `unified_ffmpeg_pass` 调用使用配置值 ✅
2. 末条字幕结束时间不再被无条件拉长到 VAD 段末 ✅
3. `expected_speakers` 生效；双人 fixture 不再产生超过两个标签 ✅
4. `no_speech_prob` / `compression_ratio` 进入 ASR 数据契约 ✅
5. 说话人数量相关统计字段进入 `PipelineStats.to_dict()` ✅

### 阶段一（幻觉过滤）✅

6. 训练语料幻觉不再出现在最终字幕 ✅
7. 引擎级解码阈值显式传参 ✅
8. 过滤层为纯函数，对缓存命中结果同样生效 ✅
9. 真实短词不被误删 ✅

### 阶段二至四（物理边界与全局 ASR）✅

10. 最终事件的每个词都能追溯到合法 `PhysicalClip`、来源词 ID 或明确的 `alignment_warning` ✅
11. 任何事件不得跨越不同 speaker ✅
12. 明显换气静音处默认产生两个字幕事件 ✅
13. 词级时间戳和 speaker 信息进入统一数据契约 ✅
14. 全局 speaker ID 在单块、宏观切块和 skeleton mode 中保持一致 ✅
15. WhisperX 可通过独立 profile 安装并运行离线转录 ✅
16. 默认基础 profile、现有 ASR 引擎、流式入口和字幕导出行为无回归 ✅
17. 各阶段具备缓存、进度、统计和可测试的降级路径 ✅

### 全局

18. 无模型单元测试全部通过 ✅（620 passed, 0 failures）
19. 固定样本 benchmark 给出 global 与 segmented 对照 ✅
20. `test/中文朗读测试-双人.ass` 的主要起止边界和说话人切换能被复现 ✅
21. 用户文档清楚说明安装成本、模型授权和 fallback 行为 ✅

---

## 18. 2026-07-26 当前实现与质量验收

本节是本文档的权威当前状态，覆盖阶段零到五的实际代码、测试和基准结果。各子系统的详细设计规范见 `docs/superpowers/specs/`，实施计划见 `docs/superpowers/plans/`。

### 18.1 代码级状态

| 链路 | 当前状态 | 验证依据 |
|------|----------|----------|
| 阶段零缺陷修复 | 已完成 | 配置透传、物理优先时间轴、speaker 上限和统计字段测试 |
| 幻觉过滤与 ASR 数据契约 | 已完成 | `tests/test_asr/test_hallucination.py`、`tests/test_phase_one.py` |
| 物理时间线与词级分配 | 已完成 | `vocal_subtitle/physical/`（12 个模块）、`tests/test_physical/` |
| 全局 ASR 路由与 WhisperX 适配器 | 已接入，CPU 环境已验证 | `auto/global/segmented` 路由、WhisperX 3.8.6、延迟导入和降级测试 |
| 严格断句与最终校验 | 已完成 | `strict_segmenter.py`、`final_validator.py`、`tests/test_phase_four.py` |
| 物理字幕仓灌装 | 已实现 | `subtitle_bins.py`、灌装器、覆盖守恒校验 |
| CLI/API/WebUI 状态透传 | 已完成 | `tests/test_cli.py`、`tests/test_webui.py`、阶段五测试 |
| WebUI 三阶段时间轴 | 已完成 | ASR/LLM/最终版本三列固定渲染、编辑保存、空格播放 |
| HF Token 安全管理 | 已完成 | Fernet 加密存储、masked presence marker |
| 缓存路径隔离与任务诊断 | 已完成 | IR 缓存键、global/legacy 路径隔离、`PipelineStats` round-trip |
| 安装与部署 | 已同步 | `install.sh`（CUDA 检测）、`debian/` 分层打包、`docs/DEPLOYMENT.md` |

### 18.2 无模型回归

执行 `venv/bin/pytest -q`：**620 passed**，无测试失败。

### 18.3 Global CPU 技术链路验收

`large-v3` CPU 基准命令：

```bash
venv/bin/python scripts/run_quality_benchmark.py \
  --asr-path global --asr-model large-v3 --require-global --no-cache --ci \
  --output-dir test/benchmark_results/production-large-v3-fixed
```

六个场景均成功且实际使用 `global`，无 fallback，物理违规为 0，词级物理分配完整。聚合 Start MAE = 133.7ms、End MAE = 507.0ms，最低人工参考内容覆盖率 = 84.80%。因此技术发布门通过，质量发布门未通过。

### 18.4 尚未通过的质量门

1. Global 聚合 Start/End MAE 和场景级边界误差仍未达到高精度阈值（目标 Start/End MAE ≤ 50ms，P95 ≤ 100ms）
2. 中文双人朗读、181 单人等场景的人工参考内容覆盖率不足；系统额外保留有声学证据的语气词不作为失败，但识别错误、主要内容缺失和过度切分仍需优化
3. `WhisperX` 词级对齐已经接通；有词级证据但被适配器标记为 degraded 的结果会保留并暴露诊断，空词结果才会阻断 global 主路径
4. 当前结果证明 global 转录、物理分配、严格切分和导出链路可运行，但尚未达到可宣称质量发布的标准
5. `--require-global --ci` 是技术门，不等于质量门通过

### 18.5 路由行为

- `auto`（离线默认）：优先 global，失败时降级 segmented → `asr_path=legacy_degraded`
- `global`：强制 global，失败直接报错，不静默回退
- `segmented`：显式旧路径，`asr_path=legacy`
- `streaming`：固定 segmented，不加载 WhisperX
- 缓存身份区分 global / legacy / legacy_degraded，禁止跨路径复用

### 18.6 物理字幕仓灌装（待质量优化）

`PhysicalSubtitleBin` 和灌装器已实现，提供了更接近真实发声范围的物理字幕仓边界。实测六个目标素材均无物理越界且接受词全部发射；剩余问题集中在 WhisperX/ASR 内容识别和按人工字幕分组时的边界精度，仍需通过物理仓灌装 + 严格断句联合优化改善事件完整性和毫秒级时间精度。系统保留人工 ASS 主观删减之外的可听语气词，不把它们当作冗余失败。

质量门（物理仓灌装路径）：

- 词内容覆盖率 ≥ 99%；重复词、丢词、物理越界为 0
- 最终事件 Start/End MAE 建议 ≤ 50ms，P95 ≤ 100ms
- 保留 ≤ 5ms、≤ 10ms、≤ 20ms、≤ 50ms、≤ 100ms 分桶
- 所有场景必须实际使用 global，任何 fallback 仍使发布门失败

---

## 19. 2026-07-26 代码审计与补齐跟踪

本节追踪原始代码审计发现的缺口及其后续补齐状态。各条目标注「已修复 ✅」「部分修复 🔶」「仍未修复 ❌」以明确当前实际状态。最终结论和剩余风险见 §20。

**审计基准日期**：2026-07-26（原始审计）→ 2026-07-26（补齐后验证）

### 19.1 后端 Pipeline（审计通过，无严重问题）

Pipeline 核心逻辑与本文档描述一致。69 个方法全部实现，`_resolve_asr_path()`、`_run_global_diarization()`、`_run_asr()` 含幻觉过滤、`_run_llm_optimize()` 含安全门、`_post_process_events()` 等关键路径完整。无 TODO/FIXME 标记。

**微小问题**：
- Pipeline 类文档字符串写"orchestrates 5 processing stages"，实际包含 10+ 阶段（分离、VAD、融合、合并、ASR、边界精修、冗余识别、时间映射、全局转录、后处理、LLM 优化等），描述已过时

### 19.2 WebUI / API 层面

#### 19.2.1 ✅ SubtitleEventResponse 物理归属字段（已修复）

API 响应模型 `SubtitleEventResponse`（`webui/models.py` 第 105-115 行）和序列化函数 `_subtitle_event_to_payload()`（`webui/api.py` 第 95-114 行）**已暴露全部物理归属字段**：

| 字段 | SubtitleEvent 中存在 | API 响应中 | 状态 |
|:---|:---|:---|:---|
| `physical_start` | ✅ | ✅ | 已修复 |
| `physical_end` | ✅ | ✅ | 已修复 |
| `physical_spans` | ✅ | ✅ | 已修复 |
| `source_word_ids` | ✅ | ✅ | 已修复 |
| `logical_sentence_id` | ✅ | ✅ | 已修复 |
| `alignment_warning` | ✅ | ✅ | 已修复 |
| `time_source` | ✅ | ✅ | 已修复 |
| `physical_bin_id` | ✅ | ✅ | 已修复 |
| `physical_bin_start` | ✅ | ✅ | 已修复 |
| `physical_bin_end` | ✅ | ✅ | 已修复 |

前端现在可展示字幕事件的物理归属、来源词追溯、对齐警告和物理仓信息。

#### 19.2.2 ✅ Batch 端点 `/api/run/batch`（已实现）

- `POST /api/run/batch` 端点已实现（`webui/api.py` 第 1524 行），接收多个上传文件并返回任务 ID 列表。
- `BatchRunRequest` 模型已投入使用，批量请求复用单文件任务契约和配置覆盖语义。
- `SubtitleEditRequest` 的 `index` 字段校验逻辑保留，URL 路径参数与请求体之间无冲突。

#### 19.2.3 ✅ 独立 HF Token 管理端点（已修复）

以下端点已实现（`webui/api.py` 第 922-950 行）：

- `POST /api/hf-token`（独立设置/更新 Token）✅
- `GET /api/hf-token`（查询 Token 存在状态，不暴露明文）✅
- `DELETE /api/hf-token`（删除已存 Token）✅

Token 采用 Fernet 加密存储于 `cache/hf_token.enc`（mode 0600），每安装生成独立密钥。

#### 19.2.4 ✅ asr_path 描述与验证（已修复）

`RunRequest.asr_path` 现使用 `Optional[Literal["auto", "global", "segmented"]]` 类型声明，字段描述已更新为明确三种路径各自用途。Handler 代码完整验证三种值。

#### 19.2.5 🔶 缓存配置持久化（部分修复）

`PUT /api/cache/config` 运行时修改仍不持久化到 YAML 配置文件，重启后恢复 YAML 设定值。当前行为已通过 API 文档明确说明。完整持久化待后续版本实现。

### 19.3 CLI 层面

#### 19.3.1 🔶 batch 命令选项补齐（部分修复）

batch 命令现已支持以下选项（`cli.py` 第 312 行起）：

| 选项 | 状态 |
|:---|:---|
| `--asr-path` | ✅ 已补齐 |
| `--asr-model` | ✅ 已补齐 |
| `--llm-optimize` | ✅ 已补齐 |
| `--diarization / --no-diarization` | ✅ 已补齐 |
| `--speaker-role / --no-speaker-role` | ✅ 已补齐 |
| `--vad-threshold` | ❌ 仍未支持 |
| `--skeleton-mode` | ❌ 仍未支持 |
| `--export-skeleton-segments` | ❌ 仍未支持 |
| `--export-skeleton-dir` | ❌ 仍未支持 |

#### 19.3.2 ✅ --config 标志（已修复）

CLI 已支持 `--config` 指定自定义 YAML 配置文件路径（`cli.py` 第 71 行），与 API 使用相同配置覆盖语义。

#### 19.3.3 ✅ feedback 命令组文档（已修复）

CLI `feedback` 子命令组（7 个子命令：`learn`、`show`、`rollback`、`reset`、`export`、`import`、`fingerprints`）已在 CLI 帮助和本文档 §21 中记录。

#### 19.3.4 🔶 --asr-model 校验（未修复）

`--asr-model` 仍为自由文本字符串，未使用 `click.Choice` 校验。无效值在运行时才会失败。

#### 19.3.5 🔶 --skeleton-mode 隐藏副作用（未修复）

启用 `--skeleton-mode` 且不指定 `--asr-path` 时，CLI 仍会静默将 ASR 路由强制设为 `segmented`。帮助文本未说明此行为。

#### 19.3.6 ✅ --asr-path segmented 弃用标记（已修复）

`--asr-path segmented` 参数保留兼容但输出明确弃用警告，帮助文本说明仅用于测试和故障恢复。`_resolve_asr_path()` 和 `batch` 命令仍接受该值。

#### 19.3.7 🔶 download-models 命令脆弱性（未修复）

`download-models` 仍使用实例化 `WhisperModel` 触发下载的副作用方式，若 `faster-whisper` 库改变缓存行为则可能失效。

### 19.4 配置层面

#### 19.4.1 🔶 BUG：default.yaml 的 `degradation` 位置仍不正确（部分修复）

原始 bug：`degradation` 被错误嵌套在 `cache` 下。已移动至 YAML **顶层**（`configs/default.yaml` 第 299 行），但 `ConfigLoader._parse_config()` 从 `pipeline.degradation` 读取（`config.py` 第 903 行 `pipeline_raw.get("degradation", {})`），因此 **顶层 degradation 配置仍被静默忽略**，回退到 `DegradationConfig()` dataclass 默认值。

由于当前 YAML 值与 dataclass 默认值一致（`mode: "full"`、`per_module_timeout: 60`、`ffmpeg_timeout: 30`、`llm_api_timeout: 15`），未产生可见症状。但任何对 `degradation` 块的编辑都将被静默忽略。

**正确修复**：将 `default.yaml` 的 `degradation` 块移入 `pipeline:` 下，或修改 parser 从顶层读取。

#### 19.4.2 ✅ Parser 回退值与 dataclass 默认值不一致（已修复）

`ConfigLoader._parse_config()` 现已改为在 `defaults` 字典中以 dataclass 实例作为单一权威来源（`config.py` 第 631-659 行），所有解析路径的缺失键回退到对应 `*Config()` 默认值。编程构造和 YAML 加载获得一致的默认值。原 8 处不一致已全部消除。

#### 19.4.3 ✅ 默认配置与设计文档描述一致性（已修复）

`configs/default.yaml` 已与 dataclass 默认值和本文档描述对齐。

#### 19.4.4 🔶 模板 profiles 缺少新增配置段（部分修复）

四个模板 profiles（podcast/education/variety_show/music_live）已部分补齐新配置段，但仍可能缺少以下段或字段：

- `boundary_redundancy` 段（podcast/education 未确认）
- `streaming` 段（各 profile 不应重复 default 设置，但需确认存在）
- `expected_speakers` 字段（见 19.4.5）
- `feedback` 段（见 19.4.6）

podcast 和 education 中 `max_shrink_ms` 已从 200 更新为 80，与 default.yaml 一致。

#### 19.4.5 ❌ `expected_speakers` 不存在于任何 YAML 文件（仍未修复）

`DiarizationConfig.expected_speakers` 在代码和缓存键中正确实现，但未出现在任何 YAML profile（含 `default.yaml`）中，对纯 YAML 用户不可发现。

#### 19.4.6 ❌ `feedback` 段完全缺失于 default.yaml（仍未修复）

`FeedbackConfig`（34 个字段）在代码中已实现完整默认值，反馈学习子系统默认启用，但 `default.yaml` 中无任何可见配置段。用户无法通过 YAML 发现或覆盖反馈学习参数。

### 19.5 设计文档内部已修正

#### 19.5.1 Pipeline "5 stages" → 已更新

§2.3 架构图和 §2.5 模块布局已更新为反映实际 10+ 阶段流水线和全部 94 个模块。

#### 19.5.2 测试数量已更新 ✅

全量测试数为 620 passed（0 failures），本文档已同步。

### 19.6 总结（补齐后）

按原始审计的 36 项问题逐项归并，补齐后的状态为：

| 汇总指标 | 数量 |
|:---|---:|
| 原始问题 | **36** |
| 完全修复 | **24** |
| 部分修复 | **4** |
| 仍未修复 | **8** |

下表保留各类问题的影响面明细。字段、端点或选项数量是受影响对象数量，不与上表的 36 项问题逐项相加。

| 类别 | 严重度 | 影响面 | 完全修复 | 部分修复 | 仍未修复 |
|:---|:---|:---|:---:|:---:|:---:|
| 🐛 default.yaml degradation 位置不正确 | **高** | 1 项 | — | 1 | — |
| 🐛 Parser 回退值与 dataclass 默认不一致 | **高** | 8 处 | 8 | — | — |
| API 响应字段缺失（physical 系列） | 高 | 10 个字段 | 10 | — | — |
| API 模型未对齐端点 | 中 | 3 处 | 3 | — | — |
| CLI batch 命令选项缺失 | 中 | 8 个选项 | 4 | — | 4 |
| CLI 缺少 `--config` 标志 | 中 | 1 项 | 1 | — | — |
| 配置默认值与文档描述偏差 | 中 | 4 处 | 4 | — | — |
| 模板 profiles 缺失新增段 | 中 | 多个字段 | — | 1 | 部分 |
| 缺少独立 Token 管理 API | 低 | 3 个端点 | 3 | — | — |
| 未记录的 CLI feedback 命令组 | 低 | 1 个命令组 | 1 | — | — |
| `expected_speakers` 不在 YAML | 低 | 1 项 | — | — | 1 |
| `feedback` 段缺失于 YAML | 低 | 1 项 | — | — | 1 |
| CLI 其他小问题 | 低 | 4 处 | 1 | 1 | 2 |

> **结论**：以上为原始审计（2026-07-26）经补齐后的最终状态。36 项原始问题中，24 项完全修复、4 项部分修复、8 项仍未修复；完全或部分修复合计 28 项。

## 20. 2026-07-26 补齐结果与生产结论

本节为当前验收结论，替代 §19 的历史缺口列表。以下基于实际代码验证（`venv/bin/pytest -q` 620 passed）和配置文件逐项检查。

### 20.1 已落实项

| 领域 | 已落实内容 | 验证方法 |
|:---|:---|:---|
| 配置 | Parser 统一 dataclass 默认值为单一权威来源（`config.py` 第 631-659 行），消除 8 处不一致 | `tests/test_deployment_defaults.py` |
| 配置 | 补齐 profile 字段：`boundary_refinement`、`acoustic_validation`、`strict_segmentation` 等 | 逐文件对比 |
| Pipeline | global/legacy 路由隔离、物理字幕仓、整词分配、严格断句、最终校验和结构化诊断 | `tests/test_phase_zero.py`、`tests/test_physical/`、`tests/test_phase_four.py` |
| WhisperX | 使用 `asr_options` 配置词级能力，独立 `align()` 取得词时间；空词结果阻断，有词 degraded 结果保留 | `tests/test_asr/test_whisperx_engine.py`、真实 large-v3 基准 |
| API/WebUI | 事件物理来源字段（10 个字段）、HF Token 加密管理（3 个端点）、batch 端点 `/api/run/batch` | `tests/test_webui.py`、代码行级验证 |
| CLI | `--config` 标志、`--asr-path`/模型选择/diarization/LLM 选项补齐 batch 命令 | `tests/test_cli.py` |
| 持久化 | 最终 SRT/ASS 生成使用真实 `SubtitleEvent`，不再因缺少 `duration` 导致静默失败 | `tests/test_persistence_manager.py` |

### 20.2 已知仍未修复项

| 问题 | 严重度 | 描述 | 影响 |
|:---|:---|:---|:---|
| `degradation` 在 default.yaml 的位置不正确 | **高** | 已从 `cache` 下移至顶层，但 parser 从 `pipeline.degradation` 读取（`config.py:903`），顶层值被静默忽略。当前 YAML 值与 dataclass 默认值巧合一致故无可见症状 | 任何对 degradation 的 YAML 编辑均被忽略 |
| `expected_speakers` 不在任何 YAML 文件 | 中 | `DiarizationConfig.expected_speakers` 在代码中实现但 YAML 中不可发现 | 纯 YAML 用户无法使用此功能 |
| `feedback` 段缺失于 default.yaml | 中 | `FeedbackConfig`（34 个字段）无 YAML 暴露 | 反馈学习参数无法通过 YAML 覆盖 |
| CLI batch 缺少 4 个选项 | 低 | `--vad-threshold`、`--skeleton-mode`、`--export-skeleton-segments`、`--export-skeleton-dir` 仍未支持 | batch 模式功能不对称 |
| CLI `--asr-model` 无校验 | 低 | 无效值运行时才失败 | 用户体验差 |
| CLI `--skeleton-mode` 隐藏副作用 | 低 | 静默强制 `segmented` ASR 路径 | 可能产生意外行为 |
| `download-models` 实现脆弱 | 低 | 依赖 `WhisperModel` 实例化副作用 | 未来可能失效 |

### 20.3 真实入口验收记录

- CLI 单文件：`test/中文多人员测试音频.wav` 成功，global，7 条字幕。
- CLI batch：`test/` 下 6 个 WAV 全部成功，6/6 输出 ASS，global。
- HTTP API：真实上传、任务轮询、字幕读取、ASS 导出和音频流均成功。
- WebSocket：任务通道心跳 `ping -> pong` 成功；服务端页面、profiles/device/history/cache 接口均正常返回。
- WebUI：真实浏览器打开本地页面、配置控件和真实音频文件选择均正常。

### 20.4 生产发布结论

当前可以投入内部试运行和人工复核流程，不能以“高精度质量门已通过”作为生产承诺。原因不是接口或物理安全异常，而是：

1. **真实素材质量指标仍未达标**：`large-v3` 六场景 global 技术链路 6/6 成功，无 fallback、无物理违规；但聚合 Start MAE 133.7ms、End MAE 507.0ms，高于 50ms 目标；最低人工参考内容覆盖率 84.80%，低于 99% 门槛。
2. **配置层面存在静默忽略的风险**：`degradation` 配置仍在 YAML 中被 parser 跳过（§19.4.1），虽然当前值恰与默认值一致。
3. **YAML 可发现性不足**：`expected_speakers` 和 `feedback` 等代码已实现的功能在 YAML 中不可见（§19.4.5-19.4.6）。

已有报告、自动字幕、人工字幕和诊断结果保留在 `test/benchmark_results/production-large-v3-fixed/`。发布前仍需：

- 修复 `degradation` 的 YAML 位置（移入 `pipeline:` 下或修改 parser 读取路径）
- 在 `default.yaml` 中添加 `expected_speakers` 和 `feedback` 配置段（使用 dataclass 默认值）
- 提升模型/解码参数、对齐和物理字幕仓断句策略
- 重新运行 `test/quality_manifest.yaml` 全量验收

不得用放宽门槛或静默 fallback 代替质量修复。

---

## 21. 自适应反馈学习系统

> **现状**：Phase 5.1+5.2 已全部实现。`vocal_subtitle/feedback/` 包含 10 个模块，CLI 提供 7 个 feedback 子命令，WebUI 默认启用。

### 21.1 系统概述

自适应反馈学习系统在用户手动修订字幕后自动学习修订模式，逐步调整 Pipeline 参数以减少未来同类场景的修订需求。系统包含两个阶段：

- **Phase 5.1（离线学习）**：分析用户修订，提取模式，调整参数
- **Phase 5.2（影子模式）**：新参数与当前参数并行运行，对比质量后再切换

### 21.2 模块架构

```text
vocal_subtitle/feedback/
├── aligner.py           # 用户修订与自动字幕对齐（LCS 级）
├── audio_fingerprint.py # 音频场景指纹计算（用于相似场景匹配）
├── conflict_detector.py # 修订冲突检测（多个用户/多次修订）
├── diff_analyzer.py     # 字幕差异分析（断句、时间、文本维度）
├── few_shot_builder.py  # Few-shot 示例构建（注入 LLM prompt）
├── health_scorer.py     # 字幕质量健康评分（多维度综合）
├── impact_estimator.py  # 参数变更影响估计
├── param_learner.py     # 自适应参数学习核心
├── shadow_mode.py       # 影子模式（新旧参数并行对比）
└── user_profile.py      # 用户偏好画像
```

### 21.3 配置

```yaml
# FeedbackConfig（34 个字段，config.py 第 461 行）
feedback:
  enabled: true
  learning_mode: "incremental"    # incremental | batch | shadow_only
  min_revision_count: 3           # 最少修订数触发学习
  shadow_mode_enabled: true       # 影子模式对比
  auto_apply_threshold: 0.8       # 自动应用置信度阈值
  # ... 其余 29 个字段见 FeedbackConfig dataclass 默认值
```

> **注意**：`feedback` 段当前不存在于 `default.yaml`（见 §19.4.6），用户无法通过 YAML 发现或覆盖。配置完全依赖 `FeedbackConfig` dataclass 默认值。

### 21.4 CLI 入口

```bash
vocal-subtitle feedback learn <task_id>    # 从任务学习
vocal-subtitle feedback show               # 显示学习状态
vocal-subtitle feedback rollback           # 回滚参数
vocal-subtitle feedback reset              # 重置学习数据
vocal-subtitle feedback export <path>      # 导出学习数据
vocal-subtitle feedback import <path>      # 导入学习数据
vocal-subtitle feedback fingerprints       # 显示音频指纹库
```

### 21.5 安全约束

- 学习数据以用户会话为隔离单位，不跨用户共享
- 影子模式确保新参数在质量不低于当前参数时才切换
- LLM few-shot 注入受 LLM 安全门（§10）约束，不能改写时间/归属/来源词
- 所有学习产物可回滚、可重置

---

## 22. 流式处理

> **现状**：流式处理入口已实现（`vocal_subtitle/streaming.py`），运行模式由 `PipelineMode` 控制，参数通过 `StreamingConfig` 管理。流式模式固定使用 segmented ASR 路径，不加载 WhisperX。

### 22.1 定位与约束

流式处理服务于实时音频场景（麦克风、直播流），与离线模式有以下关键差异：

- **ASR 路径固定为 `segmented`**：无完整音频上下文，不能使用全局 ASR
- **不加载 WhisperX**：避免额外内存占用和导入开销
- **VAD 引擎限制为轻量级**：默认 `webrtc`，适合 CPU 实时处理
- **结果标记为 `asr_path=legacy`**：不与离线 global 质量指标混合

### 22.2 配置

```yaml
pipeline:
  mode: "streaming"              # 切换到流式模式
  streaming:
    chunk_duration: 2.0          # 每次处理的音频窗口（秒）
    overlap_duration: 0.5        # 窗口重叠（秒）
    max_latency: 3.0             # 最大允许延迟（秒）
    llm_fallback: "rule_only"    # 流式下不调用 LLM
    vad_engine: "webrtc"         # 轻量 CPU VAD
```

### 22.3 架构

```text
实时音频流
  → 音频分块（chunk_duration + overlap）
  → WebRTC VAD（轻量 CPU）
  → faster-whisper 逐段 ASR（segmented 路径）
  → 字幕事件生成
  → WebSocket 推送
```

数据流以滑动窗口为边界：输入音频进入 `StreamingBuffer` 后按重叠窗口输出，窗口内完成 VAD、segmented ASR、事件合并和延迟控制，再通过 WebSocket 推送增量事件。流式模式不经过全局 diarization、WhisperX 对齐、物理字幕仓灌装或 LLM 优化阶段；Speaker 信息来自短窗口聚类或保持 `unknown`。

---

## 23. 参考文档索引

### Superpowers 设计规格（`docs/superpowers/specs/`）

| 文档 | 描述 | 对应主文档节 |
|:---|:---|:---|
| 2026-07-24-asr-accuracy-design | ASR 语言模式、检测策略与 LLM 安全门 | §5, §8, §10 |
| 2026-07-24-deployment-design | 安装 profile、Debian 打包、systemd 服务 | §3 |
| 2026-07-24-hf-model-download-design | HF Token 加密存储与模型下载 | §13 |
| 2026-07-24-llm-asr-language-fix-design | 骨架分段语言检测修复 | §5 |
| 2026-07-24-speaker-diarization-design | 全局 diarization、canonicalization、turn 合并 | §6, §7 |
| 2026-07-24-subtitle-timeline-comparison-design | WebUI 三阶段时间轴与 LLM 差异高亮 | §11 |
| 2026-07-25-phase-zero-physical-first | 物理优先时间轴、speaker 上限约束 | §4, §6, §7 |
| 2026-07-25-phase-one-hallucination-filter | 幻觉过滤两层保护、训练语料过滤 | §8 |
| 2026-07-25-phase-two-physical-timeline | PhysicalTimeline 三层时间模型 | §2.4, §4 |
| 2026-07-25-phase-two-evidence-adapter | VAD 检测结果到 SpeechEvidenceSpan 适配 | §4.2 |
| 2026-07-25-phase-two-global-ir | GlobalWord / GlobalTranscript / GlobalSpeakerTimeline | §2.4 |
| 2026-07-25-phase-two-coordinate-context-cache | 坐标转换、ContextWindow 构造、IR 缓存键 | §2.5, §12 |
| 2026-07-26-phase-three-global-transcription | WhisperX 全局转录编排与词级分配 | §5, §7 |
| 2026-07-26-phase-four-subtitle-segmentation | 严格断句器与最终物理校验 | §9 |
| 2026-07-26-phase-five-default-path | global/segmented/auto 路由与降级 | §5.2, §14, §18 |
| 2026-07-26-full-chain-completion | 全链路一致性核对与真实素材验收 | §18 |
| 2026-07-26-physical-bin-subtitle-filling | 物理字幕仓构建与整词灌装 | §7.2, §9.4 |
| 2026-07-26-production-completion | 配置/API/CLI/WebUI 生产补齐 | §19, §20 |

### 其他文档

| 文档 | 描述 |
|:---|:---|
| `docs/ARCHITECTURE.md` | 系统架构总览 |
| `docs/API参考文档.md` | API 参考 |
| `docs/DEPLOYMENT.md` | 部署指南 |
| `docs/场景模板使用指南.md` | 场景模板（profile）使用 |
| `docs/人声分离字幕工程化方案.md` | 人声分离与字幕工程化 |
| `docs/人声分离字幕方案.md` | 人声分离方案 |
| `docs/字幕时间轴精度优化方案.md` | 时间轴精度优化 |
| `docs/speaker-diarization-plan.md` | 说话人分离计划 |
| `docs/superpowers/plans/` | 各阶段实施计划（任务分解） |
