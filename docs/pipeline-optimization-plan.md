# 管线数据流优化方案

> 基于 2026-07-05 对 `vocal_subtitle/pipeline.py` 全链路代码审查

---

## 一、当前状态

### 1.1 现有数据流

```
[原始音频]
    │
    ├─ Stage 0: 宏观切块 (>3min 长音频 → N 个大块)
    │
    ├─ Stage 1: 人声分离 (UVR/Spleeter/OpenUnmix → vocals + accompaniment)
    │
    ├─ Stage 2: Silero VAD (→ 碎片级语音区间)
    │
    ├─ Stage 2.5: ffmpeg VAD + 三方法融合 (→ 精修语音区间 + 声学骨架)
    │                  ↑ 串行，等 Stage 2 完成后才启动
    │
    ├─ Stage 3: 片段合并 (合并/切分/自适应padding)
    │
    ├─ Stage 3.5: 说话人分离 (声学聚类 + 文本降级)
    │
    ├─ Stage 4: ASR 识别 (faster-whisper/whisper.cpp/FunASR)
    │
    ├─ Stage 4.5: 边界双向精修 (ASR词级时间戳 + 三帧能量斜率)
    │
    ├─ Stage 5: 时间轴映射 + 字幕构建
    │
    ├─ [后处理] 帧级无缝衔接
    ├─ [后处理] 声学校验          ← 在 LLM 合并之前
    ├─ [后处理] LLM 语义合并       ← 改变边界，使上一步校验作废
    └─ [后处理] LLM 文本优化
            │
            ▼
       SRT / VTT / ASS
```

### 1.2 当前架构特征

| 特征 | 现状 |
|------|------|
| PipelineContext | 已定义完整的数据载体结构，但仅用于诊断日志收集 |
| 阶段间数据传递 | 函数参数 + 局部变量，ctx 被旁路 |
| VAD 双引擎 | Silero 和 ffmpeg **串行**执行 |
| ffmpeg 调用次数 | 同一音频最多调用 **3 次** silencedetect |
| 声学骨架 | Stage 2.5 提取后闲置，到后处理才使用 |
| 骨架分段模式 | 独立代码路径，与常规路径大量重复 |

---

## 二、发现的问题

### 问题 1（关键 Bug）：声学校验在 LLM 语义合并之前执行

**位置**：三个后处理路径均存在
- 骨架分段模式：[pipeline.py:458-564](vocal_subtitle/pipeline.py#L458-L564)
- 多块路径：[pipeline.py:644-754](vocal_subtitle/pipeline.py#L644-L754)
- 单块路径：[pipeline.py:779-891](vocal_subtitle/pipeline.py#L779-L891)

**现象**：
```
帧级无缝衔接 → 声学校验(吸附修正边界) → LLM语义合并(改变边界) → LLM文本优化
                   ↑                                              ↑
              校验结果                                        合并后边界变化
              在合并时作废
```

声学校验用 ffmpeg 物理骨架对每个字幕事件的 start/end 做了吸附修正、生成了健康评分；紧接着 LLM 合并将多条相邻字幕合并为一条——合并后的字幕边界 `start = frag[0].start, end = frag[-1].end`，之前的吸附修正和评分全部失效。

**影响**：声学校验的吸附修正对最终输出无实际效果；健康评分反映的是合并前的状态而非用户看到的字幕质量。

---

### 问题 2（性能）：Silero VAD 和 ffmpeg VAD 串行执行

**位置**：[pipeline.py:1824-1865](vocal_subtitle/pipeline.py#L1824-L1865)

```python
# Stage 2: Silero VAD（先跑，约 0.3-2s）
vad_segments = self._run_vad(audio, sample_rate)

# Stage 2.5: ffmpeg VAD（后跑，约 0.5-3s）
ffmpeg_result = unified_ffmpeg_pass(vocals_path)

# 然后融合
vad_segments = fusion_engine.fuse(vad_segments, ffmpeg_segments, ...)
```

两个操作完全独立：
- Silero VAD 读取 `np.ndarray`，GPU/CPU 推理
- ffmpeg VAD 读取文件路径，子进程 I/O

**无任何数据依赖**，当前串行浪费 30-50% 的阶段耗时。

---

### 问题 3（重复计算）：ffmpeg silencedetect 被多次调用

| 调用位置 | 目的 | 可复用？ |
|----------|------|----------|
| `_process_chunk_pipeline` → `unified_ffmpeg_pass()` | VAD 融合骨架 | ✅ |
| `_process_skeleton_segmented` → `unified_ffmpeg_pass()` | 骨架分段 | 与上重复 |
| `AcousticValidator._get_skeleton()` | 声学校验骨架 | 可复用但未传参 |

代码中 `AcousticValidator.validate()` 接受 `ffmpeg_unified_result` 参数用于复用，但三种后处理路径调用时都传了 `None`：

```python
# 骨架分段路径 (line 476)
validator.validate(events, ..., ffmpeg_unified_result=None)  # ← 未复用

# 多块路径 (line 664)
validator.validate(events, ..., ffmpeg_unified_result=None)  # ← 未复用

# 单块路径 (line 799)
validator.validate(events, ..., ffmpeg_unified_result=None)  # ← 未复用
```

---

### 问题 4（架构）：PipelineContext 设计但未投入使用

`PipelineContext`（[pipeline_context.py](vocal_subtitle/pipeline_context.py)）定义了完整的数据流字段：

```python
# 各阶段应有产出
silero_segments: List       # 方案一+二
ffmpeg_segments: List       # 方案一+二
fused_segments: List        # 融合结果
pre_split_segments: List    # 方案三
asr_fragments: List         # ASR 产出
refined_fragments: List     # 方案四
merged_events: List         # 方案五
seamless_events: List       # 方案六
acoustic_skeleton: List     # 方案七
```

但 `_process_chunk_pipeline` 中 ctx 的实际使用情况：

```python
ctx = PipelineContext(...)           # 创建
ctx.noise_profile = ...              # ✅ 写入
ctx.ffmpeg_unified_result = ...      # ✅ 写入
ctx.add_diagnostic(...)              # ✅ 写入诊断
# ❌ 下游阶段全部通过局部变量传递，不读 ctx
# ❌ silero_segments, fused_segments, asr_fragments 等字段从未被填充
```

结果：ctx 退化为诊断日志收集器，没有起到"统一数据总线"的设计初衷。

---

### 问题 5（信息闲置）：声学骨架早提取、晚使用

ffmpeg 骨架在 Stage 2.5 提取后，包含采样级精度的物理语音/静音边界信息。但它在以下阶段完全未被利用：

| 阶段 | 骨架可提供的帮助 | 当前状态 |
|------|-----------------|---------|
| 片段合并 (Stage 3) | 确认间隙是否为真静音，辅助合并决策 | ❌ 未使用 |
| 边界精修 (Stage 4.5) | 约束搜索范围，防止过度外扩 | ❌ 未使用 |
| 时间轴映射 (Stage 5) | 判断段间 gap 类型（无缝/停顿/段落） | ❌ 未使用 |

---

### 问题 6（代码冗余）：三种后处理路径大量重复

骨架分段模式、多块路径、单块路径的后处理代码（帧级衔接 + 声学校验 + LLM 合并）几乎完全相同（~150 行 × 3），仅 `ffmpeg_unified_result` 参数传入不同。这些应抽取为一个公共方法。

---

## 三、优化方案

### 概览

```
                        [原始音频]
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        人声分离(UVR)   ffmpeg骨架提取   噪声档案采样
        (全文件一次,    (silencedetect,   (RMS profile,
         最昂贵阶段)     全程只跑一次)     底噪阈值)
              │             │             │
              ▼             └──────┬──────┘
          纯净人声                  │
              │              PipelineContext
              │              (统一数据总线,全程可读)
              │                    │
              ▼                    ▼
    ┌─ 宏观切块(>3min时) ◄─── 声学骨架(复用)
    │         │
    │    ┌────┴────┐
    │    ▼         ▼
    │  Silero VAD  ffmpeg VAD    ← 🆕 并行执行 (ThreadPool)
    │  (GPU推理)   (子进程I/O)
    │    │         │
    │    └────┬────┘
    │         ▼
    │    三方法边界融合 ←─── 声学骨架(复用,非重新提取)
    │         │
    │         ▼
    │    片段合并 ←───────── 声学骨架(辅助静音间隙判断) 🆕
    │         │
    │         ▼
    │    说话人分离(声学聚类 + 文本降级)
    │         │
    │         ▼
    │    ASR 语音识别
    │         │
    │         ▼
    │    边界双向精修 ←───── 声学骨架(约束外扩范围) 🆕
    │         │
    │         ▼
    │    时间轴映射 + 字幕构建
    │         │
    └────┬────┘
         ▼
    ┌─ _post_process_events() ─┐  🆕 抽取公共方法
    │                           │
    │  1. 帧级无缝衔接          │
    │  2. LLM 语义合并  ←───────┤  🐛 修复：移到声学校验之前
    │  3. 声学标尺校验  ←───────┤  🐛 修复：放到合并之后
    │                           │
    └──────────┬────────────────┘
               ▼
        LLM 文本优化 (只改文字,不改时间轴)
               │
               ▼
        [SRT / VTT / ASS]
```

---

### 改动 1 🐛：修正后处理阶段顺序

**优先级**：P0（逻辑 Bug）  
**影响范围**：`pipeline.py` 三个后处理路径  
**风险**：低（纯顺序调整，各阶段接口不变）

**当前顺序**：
```
帧级无缝衔接 → 声学校验 → LLM 语义合并 → LLM 文本优化
```

**修正后**：
```
帧级无缝衔接 → LLM 语义合并 → 声学校验 → LLM 文本优化
              ↑ 改变边界      ↑ 校验最终边界
```

**实现**：将三个路径中的声学校验代码块从 LLM 合并之前移到之后。骨架分段模式（line 466-486 ↔ 488-564），多块路径（line 653-674 ↔ 676-754），单块路径（line 789-809 ↔ 811-891）。

同时更新 `AcousticValidator` 的 docstring（line 12）从：
> 执行时机: LLM 语义合并完成后、最终字幕输出前

这是它本来就写的正确时机，只是调用方没遵守。

---

### 改动 2 ⚡：Silero VAD 与 ffmpeg VAD 并行化

**优先级**：P1（性能优化）  
**影响范围**：`_process_chunk_pipeline` 方法  
**风险**：低（两个操作完全独立，合并点在 barrier 处语义不变）

**现状**（串行）：
```python
# Stage 2: Silero VAD
vad_segments = self._run_vad(audio, sample_rate)
self._progress.finish_stage()

# Stage 2.5: ffmpeg VAD
ffmpeg_result = unified_ffmpeg_pass(vocals_path)
self._progress.finish_stage()

# 融合
vad_segments = fusion_engine.fuse(vad_segments, ffmpeg_segments, ...)
```

**改造后**（并行）：
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=2) as executor:
    future_silero = executor.submit(self._run_vad, audio, sample_rate)
    future_ffmpeg = executor.submit(unified_ffmpeg_pass, vocals_path)

    # 等待双方完成
    vad_segments = future_silero.result()
    ffmpeg_result = future_ffmpeg.result()

# 融合（逻辑不变）
if self.config.fusion.enabled:
    vad_segments = fusion_engine.fuse(vad_segments, ffmpeg_segments, ...)
```

预期耗时：`max(Silero时间, ffmpeg时间)` 替代 `Silero时间 + ffmpeg时间`，减少 **30-50%**。

**注意**：Silero VAD 首次调用会触发模型加载（~100ms），此部分无法并行；但模型加载后 `detect_on_array` 是纯推理，与 ffmpeg 子进程完全可并行。

---

### 改动 3 🔄：ffmpeg 骨架全程只跑一次

**优先级**：P1（消除重复计算）  
**影响范围**：`_process_chunk_pipeline`、`_process_skeleton_segmented`、后处理路径  
**风险**：低（骨架结果缓存后语义不变）

**改造点**：

1. **`_process_chunk_pipeline`** 返回 ctx（或通过 ctx 传递骨架）
   ```python
   # 当前只返回 (events, seg_count)
   return events, len(merged_segments)
   
   # 改造：同时返回 ctx（含声学骨架）
   return events, len(merged_segments), ctx
   ```

2. **后处理阶段复用骨架**：将 `ctx.ffmpeg_unified_result` 传给 `AcousticValidator.validate()`
   ```python
   # 当前（三个路径都传 None）
   validator.validate(events, ..., ffmpeg_unified_result=None)
   
   # 改造：传入上游已提取的骨架
   validator.validate(events, ..., ffmpeg_unified_result=ctx.ffmpeg_unified_result)
   ```
   这样 `AcousticValidator._get_skeleton()` 走复用分支（line 136-141），跳过重复的 ffmpeg 子进程调用。

3. **骨架分段模式**：`_process_skeleton_segmented` 不再自己调用 `unified_ffmpeg_pass()`，改为复用 Pipeline 级别预先提取的骨架。

---

### 改动 4 🏗️：抽取公共后处理方法

**优先级**：P1（消除 ~150 行 × 3 重复代码）  
**影响范围**：`Pipeline` 类  
**风险**：中等（涉及三个路径的统一，需逐个路径验证）

将三个路径中完全相同的后处理逻辑抽取为一个方法：

```python
def _post_process_events(
    self,
    events: List[SubtitleEvent],
    vocals_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    stats: PipelineStats,
    ffmpeg_unified_result: Optional[Dict] = None,
) -> List[SubtitleEvent]:
    """后处理管线（三种路径共用）
    
    执行顺序：
    1. 帧级无缝衔接
    2. LLM 语义合并（改变边界）
    3. 声学标尺校验（校验最终边界，使用物理骨架）
    
    顺序是精心设计的：LLM 合并改变边界，声学校验必须在其之后执行。
    """
    # 1. 帧级无缝衔接
    try:
        from .merging.llm_merge_engine import apply_frame_seamless_stitching
        stitch_gap = self.config.subtitle.max_stitch_gap
        events = apply_frame_seamless_stitching(events, max_stitch_gap=stitch_gap)
    except Exception as e:
        logger.warning("Frame seamless stitching failed: %s", e)
    
    # 2. LLM 语义合并（先于声学校验，因为合并改变边界）
    if (
        self.config.merge_decision.llm_tier != "rule_only"
        and len(events) > 1
    ):
        events = self._run_llm_merge(events, audio, sample_rate, stats)
    
    # 3. 声学标尺校验（在合并之后，校验最终边界）
    if self.config.acoustic_validation.enabled:
        try:
            from .acoustic_validator import AcousticValidator
            self._progress.start_stage("acoustic", description="声学校验")
            validator = AcousticValidator(self.config.acoustic_validation)
            events, validation_report = validator.validate(
                events,
                audio_path=vocals_path,
                audio=audio,
                sample_rate=sample_rate,
                ffmpeg_unified_result=ffmpeg_unified_result,  # 🆕 传入复用
            )
            if validation_report.get("health_score") is not None:
                logger.info(
                    "Acoustic validation health: %.1f%%",
                    validation_report["health_score"],
                )
            stats.diagnostic_report = validation_report
            stats.stage_timings["acoustic"] = self._progress.finish_stage()
        except Exception as e:
            logger.warning("Acoustic validation failed: %s", e)
    
    return events
```

三个调用方简化为一行：
```python
events = self._post_process_events(
    events, vocals_path, audio, sample_rate, stats,
    ffmpeg_unified_result=ctx.ffmpeg_unified_result,
)
```

---

### 改动 5 🏗️：PipelineContext 逐步激活

**优先级**：P2（架构改善，可逐步推进）  
**影响范围**：各模块  
**风险**：中高（涉及模块间接口变更，建议分步实施）

**Phase 1**：ctx 承载声学骨架（改动 3 的延伸）
- `_process_chunk_pipeline` 将骨架写入 `ctx.acoustic_skeleton`
- 下游通过 `ctx.has_ffmpeg_skeleton()` 判断是否可用

**Phase 2**：ctx 承载融合后的 VAD 结果
- 合并阶段从 `ctx.fused_segments` 读取，而非参数传入
- 好处：骨架分段模式和多块路径的缝合逻辑可以直接读取 ctx 中的分段信息

**Phase 3**：ctx 承载 ASR 片段
- `ctx.asr_fragments` 承载 `ASRFragment` 列表（含词级时间戳、合并标记等）
- 边界精修读取 `ctx.asr_fragments`，输出到 `ctx.refined_fragments`
- LLM 合并读取 `ctx.refined_fragments`

**最终目标**：`_process_chunk_pipeline` 的签名从 5 个参数 + 返回 tuple 简化为：
```python
def _process_chunk_pipeline(self, ctx: PipelineContext) -> PipelineContext:
    """处理单个音频块，所有中间结果写入 ctx"""
    ...
    return ctx
```

---

### 改动 6 🔧：声学骨架指导下游阶段（可选增强）

**优先级**：P2（精度提升）  
**影响范围**：`merge_strategy.py`、`boundary_refiner.py`、`time_mapper.py`  
**风险**：中（需逐个模块验证精度变化）

| 模块 | 改造内容 |
|------|---------|
| **MergeStrategy** | 合并两段前，用骨架确认间隙是否为**物理静音**（而非 VAD 漏检），降低误合并风险 |
| **BoundaryRefiner** | 三帧能量外扩时，以骨架边界为硬上限（不超过骨架端点外 50ms），防止边界过度膨胀 |
| **TimeMapper** | gap ≤ 骨架静音阈值 → 无缝衔接；gap 在语音段内 → 自然停顿；gap 跨越多骨架段 → 段落间隔 |

所有模块通过 `PipelineContext.acoustic_skeleton` 访问骨架，**不改函数签名**。

---

## 四、实施计划

### 第一批（无风险，立即可做）

| 步骤 | 改动 | 预计改动行数 |
|------|------|-------------|
| 1.1 | 修正三个路径的后处理顺序：声学校验移到 LLM 合并之后 | ~30 行调整 |
| 1.2 | 抽取 `_post_process_events()` 公共方法 | +60 行, -140 行 |
| 1.3 | 后处理调用 `AcousticValidator` 时传入 `ffmpeg_unified_result` | ~3 行修改 |

### 第二批（性能优化，低风险）

| 步骤 | 改动 | 预计改动行数 |
|------|------|-------------|
| 2.1 | `_process_chunk_pipeline` 中 VAD 并行化 | ~25 行 |
| 2.2 | `_process_chunk_pipeline` 返回 ctx | ~10 行 |

### 第三批（架构改善，需逐模块验证）

| 步骤 | 改动 | 预计改动行数 |
|------|------|-------------|
| 3.1 | ctx 承载骨架 → MergeStrategy 读取 | ~20 行 |
| 3.2 | ctx 承载骨架 → BoundaryRefiner 读取 | ~15 行 |
| 3.3 | ctx 承载骨架 → TimeMapper 读取 | ~15 行 |

### 第四批（长期架构演进）

| 步骤 | 改动 |
|------|------|
| 4.1 | ctx 承载全部中间结果，模块间通过 ctx 交换数据 |
| 4.2 | 骨架分段模式与常规模式统一为同一代码路径 |

---

## 五、风险与回滚

- **所有改动均为内部重构**，不影响对外 API（`Pipeline.run()` 的输入输出不变）
- **第一批和第二批改动**不改变任何模块的内部逻辑，仅调整调用顺序和并行化
- **第三批改动**涉及模块内部行为，建议每次改一个模块后跑对应的测试用例确认精度
- 回滚策略：每批改动独立提交，出问题单独 revert 即可
