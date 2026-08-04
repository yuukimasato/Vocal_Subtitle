# Global ASR 时间轴方案设计

日期：2026-07-28
状态：草案，待评审

---

## 1. 现状分析

### 1.1 当前架构：逐片段 ASR

```
音频 → VAD → Merge → 逐片段 ASR (每个合并段一次调用)
                         ↓
                   片段内相对时间戳
                         ↓
                    TimeMapper 偏移到全局轴
                         ↓
                    AcousticValidator (ffmpeg 骨架吸附)
```

问题：
- VAD 边界不精确时，上下文被切断（如「你好」被切成`"你" | "好"`），两个 ASR 调用各自独立
- BoundaryRefiner/ReASR 只能做局部修正，无法从根本解决上下文断裂
- ffmpeg 声学骨架（能量检测，精度 ~50-100ms）作为唯一物理基准，但 Whisper cross-attention 词级时间戳精度可达 ~20ms

### 1.2 已存在但未连接的全局 ASR 基础设施

| 组件 | 文件 | 行数 | 状态 |
|---|---|---|---|
| `_run_global_transcription_path()` | pipeline.py:219-454 | 235 行 | ✅ 完整实现，**从未被调用** |
| `_resolve_asr_path()` | pipeline.py:479-493 | 15 行 | ✅ 定义但**从未被调用** |
| `_get_global_asr_engine()` | pipeline.py:212-217 | 6 行 | ✅ 定义 |
| `GlobalTranscriber` | asr/global_transcriber.py | 272 行 | ✅ 完整实现 |
| `GlobalTranscript` / `GlobalWord` / `GlobalTranscriptSegment` | physical/ir.py | ~500 行 | ✅ 完整 IR |
| `PhysicalTimeline` + `SpeechEvidenceSpan` + `PhysicalClip` | physical/timeline.py | ~400 行 | ✅ 完整 |
| `GlobalASRConfig` | config.py:181-191 | 11 行 | ✅ `routing: auto/global/segmented` |
| CLI `--asr-path` | cli.py | — | ✅ 暴露但无效果 |
| `PipelineStats.asr_path` / `global_attempted` | pipeline.py:59-64 | — | ✅ 定义但从未被设置 |

### 1.3 `_run_global_transcription_path()` 缺失的依赖

该方法签名：

```python
def _run_global_transcription_path(
    self, audio, sample_rate, shadow, stats, *,
    vad_segments=None, ffmpeg_result=None, noise_profile=None,
):
```

关键行（pipeline.py:305-307）：
```python
timeline = getattr(shadow, "physical_timeline", None)
if timeline is None:
    return [], {"recovery": {"status": "no_timeline"}}, global_transcript
```

**缺失的连线**：`shadow` 对象需要带有 `physical_timeline` 属性。目前 `shadow` 只在 `PipelineContext` 中（构建于 `_process_chunk_pipeline()`），这是一个鸡生蛋问题——全局 ASR 路径需要物理骨架来分配词到字幕 bins，但物理骨架目前是在逐片段路径中才构建的。

### 1.4 现有测试

- `tests/test_physical/test_phase_three.py:381` — 调用 `_run_global_transcription_path()` 的集成测试
- `tests/test_physical/test_coverage_recovery.py:219` — 覆盖率恢复测试

两个测试通过构造 mock shadow 来绕过依赖，验证了方法的内部逻辑是正确的。

---

## 2. 架构设计

### 2.1 核心思想：时间轴权威转移

```
                     现状                              目标
             ┌──────────────────┐            ┌──────────────────┐
             │  ffmpeg 骨架      │            │  全局 ASR 词级    │
             │  (物理基准)       │            │  时间戳 (主基准)  │
             │                  │            │                  │
             │  逐片段 ASR      │     →      │  ffmpeg 骨架      │
             │  (被骨架吸附)    │            │  (安全网/兜底)    │
             └──────────────────┘            └──────────────────┘
```

- **主时间轴**：全局 ASR 的词级时间戳（cross-attention 精度 ~20ms）
- **安全网**：ffmpeg 声学骨架（能量检测精度 ~50-100ms）
- **门控逻辑**：置信度门控决定何时用 ASR 时间戳、何时回退到物理骨架

### 2.2 新架构全景

```
                              ┌─────────────────────────────┐
                              │     音频输入 (whole audio)    │
                              └─────────────┬───────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
              ┌─────▼─────┐          ┌──────▼──────┐          ┌─────▼─────┐
              │ 语言检测    │          │ ffmpeg 骨架  │          │ VAD 检测   │
              │ (全音频)    │          │ (物理基准)   │          │ (Silero)  │
              └─────┬─────┘          └──────┬──────┘          └─────┬─────┘
                    │                       │                       │
                    └───────────────────────┼───────────────────────┘
                                            │
                                  ┌─────────▼─────────┐
                                  │  全局 ASR (一次)    │
                                  │  engine.transcribe │
                                  │  (完整音频)         │
                                  └─────────┬─────────┘
                                            │
                                  ┌─────────▼─────────┐
                                  │  GlobalTranscript  │
                                  │  · words (词级)    │
                                  │  · segments (段级) │
                                  │  · 绝对时间坐标     │
                                  └─────────┬─────────┘
                                            │
                                  ┌─────────▼──────────────┐
                                  │  词 → 物理 Bin 分配     │
                                  │  allocate_words()       │
                                  │  + align_words_to_physical()
                                  └─────────┬──────────────┘
                                            │
                                  ┌─────────▼─────────┐
                                  │  build_events()    │
                                  │  → SubtitleEvent[] │
                                  └─────────┬─────────┘
                                            │
                                  ┌─────────▼──────────────┐
                                  │  AcousticValidator      │
                                  │  ┌────────────────────┐ │
                                  │  │ 置信度门控:          │ │
                                  │  │ · 高置信度 → 保留    │ │
                                  │  │ · 低置信度 → 骨架吸附│ │
                                  │  │ · 明显越界 → 修正    │ │
                                  │  │ · 疑似截断 → 报告    │ │
                                  │  └────────────────────┘ │
                                  └─────────┬──────────────┘
                                            │
                                  ┌─────────▼─────────┐
                                  │  后处理 (不变)      │
                                  │  · 说话人融合       │
                                  │  · LLM 语义合并     │
                                  │  · 帧级无缝衔接     │
                                  └─────────┬─────────┘
                                            │
                                  ┌─────────▼─────────┐
                                  │  导出 SRT/VTT/ASS  │
                                  └───────────────────┘
```

### 2.3 两种路径对比

| 维度 | 逐片段路径 (现状) | 全局 ASR 路径 (目标) |
|---|---|---|
| ASR 调用方式 | N 次（每个 VAD 合并段一次） | 1 次（完整音频） |
| 时间戳坐标系 | 片段内相对 → TimeMapper 偏移 | 全局绝对坐标 |
| 语言检测 | 全局一次，传给各段 | 全局一次 |
| 代码穿插回退 | 逐段 avg_logprob 检测 | 全局后按 bin 检测低置信度区域 |
| 物理骨架角色 | 主导框架（吸附基准） | 安全网（边界兜底） |
| 上下文完整性 | VAD 边界处可能断裂 | 完整上下文 |
| 内存占用 | 低（逐段处理小音频） | 高（完整音频送入 Whisper） |
| 长音频 (>1h) | 天然支持 | 需分块 + GlobalTranscriber 合并 |

### 2.4 路由策略

```
config.asr.global_asr.routing:
  "global"    → 始终走全局路径，失败则报错
  "segmented" → 始终走逐片段路径（现状）
  "auto"      → 优先全局，失败回退到逐片段
```

回退条件（`auto` 模式下）：
1. `global_transcript.status == "empty"` — 全局 ASR 无输出
2. `global_transcript.status == "degraded"` 且词覆盖率 < 50%
3. 音频时长 > 3600s（1 小时）— Whisper 单次调用可能 OOM
4. 全局 ASR 执行异常（`_classify_global_failure()` 分类）

---

## 3. 时序关系设计

### 3.1 全局 ASR 时间戳即主时间轴

全局 ASR 产出 `GlobalTranscript`，其中每个 `GlobalWord` 携带**绝对时间坐标**：

```python
@dataclass(frozen=True)
class GlobalWord:
    id: str
    text: str
    raw_start: float       # ← 全局绝对起始时间
    raw_end: float         # ← 全局绝对结束时间
    confidence: float      # ← 时间戳置信度
```

步骤：

1. **词分配到物理 Bins**：`allocate_words(global_transcript, physical_timeline)` 将每个词分配到它所属的物理语音段
2. **词对齐精修**：`align_words_to_physical()` 在 bin 边界附近微调归属（防止边界词被错分）
3. **构建字幕事件**：`build_events()` 从分配结果生成 `SubtitleEvent`，事件的 `start/end` = 首尾词的 `raw_start/raw_end`
4. **置信度门控**：`AcousticValidator` 检查每个事件的词级置信度
   - 高置信度 → 保留 ASR 时间轴
   - 低置信度 → 用 ffmpeg 骨架修正边界

### 3.2 全局 ASR 作为声学验证的参考轴

当全局 ASR 提供词级时间戳后，`AcousticValidator` 的角色变为：

```
                    ┌──────────────────────────────┐
                    │  AcousticValidator.validate() │
                    │                              │
   SubtitleEvent[] ─┤  对每个事件:                  │
   (from global     │                              │
    ASR + bins)     │  1. 检查词级置信度            │
                    │     ├─ ≥ 阈值 → 保留边界       │
   ffmpeg skeleton  ─┤     └─ < 阈值 → 进入物理校验  │
   (安全网)          │                              │
                    │  2. 边界是否在骨架语音段内？    │
                    │     ├─ 是 → 保留              │
                    │     └─ 否 → 方向感知吸附       │
                    │                              │
                    │  3. 物理越界检测              │
                    │     └─ 超过 max_snap_distance │
                    │        → 标记需复核            │
                    └──────────────────────────────┘
```

关键变化：现在的「高置信度」判断不再只依赖 `event.confidence` 字段，还包含**词级时间戳的存在性和置信度**。全局 ASR 路径天然产生词级时间戳，因此大多数事件的保留率更高。

### 3.3 时间轴参考优先级（新增）

```
1. GlobalWord.raw_start/raw_end (来自全局 ASR cross-attention)
   └─ 条件：word.confidence ≥ 0.6 且位于骨架语音段内或间隙 ≤ 50ms
   
2. ffmpeg skeleton 方向感知吸附
   └─ 条件：全局 ASR 边界明显在静音区（> 50ms 间隙）
   
3. 事件自身 physical_start/physical_end
   └─ 条件：全局 ASR 无词级时间戳（降级场景）
```

---

## 4. 实施步骤

### Phase A：构建物理 Shadow（前置条件）

**文件**：`pipeline.py`（新增 `_build_physical_shadow()`）

这一步解决 `_run_global_transcription_path()` 缺少 `shadow` 参数的问题。

```python
def _build_physical_shadow(self, audio, sample_rate, vocals_path, ctx):
    """构建全局 ASR 路径所需的物理 shadow 对象。
    
    从 ffmpeg + VAD 检测结果构建 PhysicalTimeline，
    供 _run_global_transcription_path() 使用。
    """
    from .physical.shadow import PhysicalShadow
    from .physical.timeline import PhysicalTimeline, PhysicalClip, SpeechEvidenceSpan
    from .physical.evidence_adapter import build_timeline_from_detectors
    
    # 复用 ffmpeg skeleton + VAD segments
    timeline = build_timeline_from_detectors(
        audio_duration=len(audio) / sample_rate,
        skeleton=ctx.ffmpeg_unified_result.get("skeleton", []),
        coarse_speech=ctx.ffmpeg_unified_result.get("coarse_speech", []),
        vad_segments=ctx.vad_segments,
    )
    
    return PhysicalShadow(
        physical_timeline=timeline,
        ffmpeg_unified_result=ctx.ffmpeg_unified_result,
        noise_profile=ctx.noise_profile,
        vad_segments=ctx.vad_segments,
    )
```

**预估改动量**：~50 行（pipeline.py），复用现有 `evidence_adapter.py`

### Phase B：补充全局 ASR 的语言检测

**文件**：`pipeline.py`（在 `run()` 中）

全局 ASR 路径应与逐片段路径共享语言检测逻辑。当前 `_run_asr()` 中的全局语言检测（pipeline.py:2083-2107）需要提取为独立方法：

```python
def _detect_global_language(self, audio, sample_rate, asr_cfg):
    """在完整音频上运行一次语言检测，所有路径共享。"""
    if asr_cfg.language:
        return asr_cfg.language
    engine = self._get_asr_engine()
    engine.load_model()
    detector = getattr(engine, "detect_language", None)
    if callable(detector):
        return detector(audio, sample_rate)
    return None
```

**预估改动量**：~30 行提取 + ~10 行在 `_run_asr()` 中调用

### Phase C：连接路径分发

**文件**：`pipeline.py:run()`（约在第 1100-1310 行之间）

这是核心变更。在当前 `_process_chunk_pipeline()` 或 `_process_skeleton_segmented()` 被调用之前，插入路径分发：

```python
# ---- ASR 路径分发 ----
asr_path = self._resolve_asr_path()
stats.asr_path = asr_path

if asr_path == "global":
    stats.global_attempted = True
    try:
        # 1. 构建物理 shadow（复用骨架路径的 VAD + ffmpeg）
        ffmpeg_result, vad_segments, noise_profile = self._run_early_detection(
            audio, sample_rate, vocals_path
        )
        shadow = self._build_physical_shadow(
            audio, sample_rate, vocals_path, ffmpeg_result, vad_segments, noise_profile
        )
        
        # 2. 运行全局 ASR
        events, global_diag, global_transcript = self._run_global_transcription_path(
            audio=audio,
            sample_rate=sample_rate,
            shadow=shadow,
            stats=stats,
            vad_segments=vad_segments,
            ffmpeg_result=ffmpeg_result,
            noise_profile=noise_profile,
        )
        stats.global_diagnostics = global_diag
        
        # 3. 检查可用性
        if not events and self._is_usable_global_transcript(global_transcript):
            stats.fallback_category = "empty_events_but_usable_transcript"
        elif not events:
            stats.fallback_category = "global_returned_no_events"
            if asr_path == "auto":
                logger.warning("Global ASR returned no events, falling back to segmented")
                asr_path = "segmented"
                stats.fallback_reason = global_diag.get("recovery", {}).get("status", "unknown")
        
    except Exception as exc:
        stats.fallback_category = Pipeline._classify_global_failure(exc)
        stats.fallback_reason = str(exc)[:200]
        logger.warning("Global ASR failed (%s): %s", stats.fallback_category, exc)
        if asr_path == "auto":
            asr_path = "segmented"
        else:
            raise

# 回退到逐片段路径
if asr_path == "segmented":
    events, seg_count, ctx = self._process_chunk_pipeline(
        audio=audio, sample_rate=sample_rate, vocals_path=vocals_path,
    )
    stats.segment_count = seg_count
    stats.subtitle_count = len(events)
```

**预估改动量**：~80 行（pipeline.py）

### Phase D：全局 ASR 时间轴参考集成

**文件**：`acoustic_validator.py`

`AcousticValidator._physical_snap_validation()` 已支持置信度门控（`_preserve_reliable_asr_boundary()`）。全局 ASR 路径下，事件天然携带词级时间戳，因此高置信度保留率自动提高。

需要新增的逻辑：

```python
def _has_global_word_timestamps(event) -> bool:
    """检查事件是否来自全局 ASR 且携带可靠的词级时间戳。"""
    source_word_ids = getattr(event, "source_word_ids", []) or []
    words = getattr(event, "words", []) or []
    if not source_word_ids and not words:
        return False
    
    # 全局 ASR 词级时间戳的置信度检查
    if words:
        high_conf_words = [w for w in words if getattr(w, "confidence", 0) >= 0.6]
        return len(high_conf_words) >= len(words) * 0.5
    
    # 如果只有 source_word_ids（来自 GlobalTranscript 分配），
    # 视为有全局时间戳参考
    return True


def _preserve_reliable_asr_boundary(event, boundary_type, config):
    """增强版：全局 ASR 词级时间戳优先。"""
    # 现有检查：event.confidence + word timestamps
    if _has_global_word_timestamps(event):
        # 全局 ASR 路径：词级时间戳可靠 → 优先保留
        event_conf = getattr(event, "confidence", None)
        if event_conf is None or event_conf >= config.confidence_threshold:
            return True
    
    # ... 现有逻辑保持不变 ...
```

**预估改动量**：~40 行（acoustic_validator.py）

### Phase E：GlobalTranscriber 用于长音频

**文件**：`pipeline.py`

对于 >1 小时的音频，单次 `engine.transcribe()` 可能 OOM。使用已有的 `GlobalTranscriber`：

```python
def _run_global_asr_with_chunking(self, audio, sample_rate, engine, shadow):
    """对长音频使用 GlobalTranscriber 分窗口转录。"""
    from .asr.global_transcriber import GlobalTranscriber, GlobalTranscriberConfig
    
    transcriber = GlobalTranscriber(
        engine=engine,
        config=GlobalTranscriberConfig(
            left_context=0.5,
            right_context=0.5,
            overlap_dedup=True,
        ),
    )
    
    result = transcriber.transcribe(
        audio=audio,
        sample_rate=sample_rate,
        physical_timeline=shadow.physical_timeline,
        language=self._resolved_language,
    )
    
    return result.transcript, result.diagnostics
```

**预估改动量**：~30 行（pipeline.py，复用 GlobalTranscriber）

### Phase F：测试

**文件**：`tests/test_global_asr_path.py`（新文件）

测试场景：
1. 短音频（< 30s）：全局 ASR 产生正确词级时间戳
2. 全局 ASR 词级时间戳被 AcousticValidator 识别为可靠边界
3. 全局 ASR 失败时正确回退到逐片段路径
4. 全局 ASR 低置信度词被 ffmpeg 骨架吸附
5. 物理 shadow 构建正确性
6. 路由决策（global / segmented / auto）正确性

---

## 5. 改动范围汇总

| Phase | 文件 | 改动性质 | 行数估计 |
|---|---|---|---|
| A | `pipeline.py` | 新增 `_build_physical_shadow()` | +50 |
| A | `physical/evidence_adapter.py` | 可能需要新增 `build_timeline_from_detectors()` | +30 |
| B | `pipeline.py` | 提取 `_detect_global_language()` | +30 / -20 |
| C | `pipeline.py` | 在 `run()` 中插入路径分发 | +80 |
| D | `acoustic_validator.py` | 增强全局 ASR 时间戳识别 | +40 |
| E | `pipeline.py` | 长音频分块（复用 GlobalTranscriber） | +30 |
| F | `tests/test_global_asr_path.py` | 新测试文件 | +200 |
| — | `config.py` | 无需改动（GlobalASRConfig 已完整） | 0 |
| — | `asr/global_transcriber.py` | 无需改动（已完整） | 0 |
| — | `physical/` | 无需改动（已完整） | 0 |
| **总计** | | | **~440 行新增** |

### 5.1 不改动的范围

- `asr/base.py` — ASR 引擎接口不变
- `asr/faster_whisper_engine.py` — 引擎实现不变
- `asr/global_transcriber.py` — 已完整，无需改动
- `physical/ir.py` — GlobalTranscript/GlobalWord IR 已完整
- `physical/allocator.py` — 词分配逻辑不变
- `physical/events.py` — 事件构建不变
- `physical/word_alignment.py` — 词对齐不变
- `mapping/time_mapper.py` — 全局路径下不再需要偏移映射
- `asr/boundary_*.py` — 边界精修/ReASR/仲裁在全局路径下不再需要（词级时间戳已足够精确）
- 现有的 `_run_asr()` 逐片段路径保持不变

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 长音频 Whisper OOM | GlobalTranscriber 分窗口转录 + overlap dedup |
| 全局 ASR 时间戳在静音区偏移 | ffmpeg 骨架作为安全网兜底 |
| VAD / ffmpeg 骨架必须提前构建 | Phase A 中将骨架构建提前到路径分发之前 |
| 全局路径与现有后处理兼容性 | 后处理（说话人融合、语义合并）接收相同的 SubtitleEvent[]，接口不变 |
| FunASR 不支持词级时间戳 | 全局路径默认仅对 faster-whisper 启用；FunASR 走分段路径 |

---

## 7. 验收标准

1. `--asr-path global` 对 <1h 音频产出正确字幕，时间戳精度优于逐片段路径
2. `--asr-path auto` 在全局 ASR 成功时不走逐片段路径
3. `--asr-path auto` 在全局 ASR 失败时正确回退
4. 全局 ASR 路径下 `AcousticValidator` 的 `skipped_high_confidence` 计数显著高于分段路径
5. 现有 tests/test_physical/test_phase_three.py 和 test_coverage_recovery.py 继续通过
6. 现有逐片段路径（默认行为）不受影响，所有现有测试通过
