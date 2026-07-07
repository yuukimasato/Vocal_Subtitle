# 边界感知滑动窗口冗余识别

## 概述

针对语速较快、词间静音间隙不明显、单词粘连的场景，在现有 ASR 管线基础上增加**边界冗余识别**层，通过偏移窗口多次 ASR + LLM 语义仲裁来解决边界分词错误和时间轴偏移问题。

## 问题场景

当语速较快（>4 词/秒）且词间无静音间隙时：

1. VAD 边界可能落在语义粘连的词之间（如副词和形容词之间）
2. 孤立片段 ASR 可能给非句末文本错误添加句号
3. 句末标点阻断了现有 LLM Merge 引擎的快速合并路径
4. 语义单元被撕裂，产生错误的字幕分段

**典型案例：**

```
自动生成:
  Seg N:   "other room in the house that gets unbearably."  [22.81→25.30]
  Seg N+1: "hot you'll know exactly what i mean..."          [25.30→28.95]

人工修订:
  Seg N:   "other room in the house that gets unbearably hot" [23.09→25.71]
  Seg N+1: "you'll know exactly what i mean..."               [26.00→27.45]
```

## 架构设计

### 插入位置

在 Stage 4.5 (ASR Boundary Refinement) 之后、Stage 5 (Time Mapping) 之前：

```
现有管线:
  Stage 4: ASR → Text Normalize → Stage 4.5: Boundary Refinement
       ↓
  [NEW] Boundary Confidence Estimation  ← 评估各边界是否需要冗余
       ↓
  [NEW] Sliding Window Re-ASR          ← 仅对低置信度边界做偏移窗口重识别
       ↓
  [NEW] LLM Boundary Arbitration       ← 语义仲裁决定词归属 + 时间戳
       ↓
  Stage 5: Time Mapping (含词重分配数据)
```

### 三个核心组件

#### 1. BoundaryConfidenceEstimator（边界置信度评估器）

对每对相邻片段边界计算置信度分数 (0.0–1.0)，低分触发冗余：

| 低置信度信号 | 说明 |
|---|---|
| 段间 gap < 50ms | 几乎紧贴，可能是词间粘连 |
| 前段末词以人工句号结尾 | ASR 给不完整句子加了句号 |
| 后段首词是"孤儿词" | 孤立副词、形容词、连词等 |
| 边界能量斜率 < 3.0 | 渐变边界，非阶跃跳变 |
| 前段末词时间戳紧贴段尾 | 词被边沿截断 |
| 后段首词紧贴段首 | 词可能是前段溢出 |

#### 2. SlidingWindowReASR（滑动窗口重识别）

对低置信度边界创建 3 个重叠音频窗口，重新执行 ASR：

```
原始分段:
  [------ Seg N ------][------ Seg N+1 ------]
                        ^ 边界 index

窗口 A (左扩展):
  Seg N 音频 + 向前延伸 overlap_ms → 捕获可能被截断的尾词

窗口 B (右扩展):
  Seg N+1 音频 - 向后延伸 overlap_ms → 捕获可能被截断的首词

窗口 C (融合窗):
  边界前后各 1s 的音频 → 完整上下文，获取跨边界词的精确时间戳
```

重叠量自适应：
- 正常语速 (<4 词/秒): 500ms
- 快速 (4-5 词/秒): 750ms
- 极快 (>5 词/秒): 1000ms

3 个窗口并行执行 ASR（ThreadPoolExecutor），结果缓存。

#### 3. BoundaryArbitrator（LLM 语义仲裁器）

构建词级共识格，LLM 决定每个词归属哪个段：

```
LLM 输入:
  上文: "... room in the house that gets"
  边界候选（来自多次 ASR 的不同假设）:
    假设A (原始):    "unbearably." | "hot you'll know..."
    假设B (窗口A):   "unbearably hot" | [out of window]
    假设C (窗口B):   [out of window] | "you'll know exactly..."
    假设D (窗口C):   "unbearably hot you'll know..."
  下文: "... exactly what i mean so when people talk"

LLM 输出 (JSON):
  {
    "word_assignments": [
      {"word":"unbearably","segment":"left","confidence":1.0},
      {"word":"hot","segment":"left","confidence":0.95},
      {"word":"you'll","segment":"right","confidence":0.98}
    ],
    "rationale": "hot is modified by unbearably, belonging to left segment"
  }
```

自动应用策略：
- LLM 置信度 > 80%：自动应用词重分配和时间轴修正
- 50%–80%：应用修正在 ASS 注释字段标记
- < 50%：保持原始分段

### 时间轴修正

词重分配后，从对应窗口的 ASR 词级时间戳获取精确时间：

```
"hot" 从 Seg N+1 重分配到 Seg N:
  → 使用窗口 A/C 中 "hot" 的词级时间戳
  → global_time = window_start + word_local_time
  → Seg N end_time 更新
  → Seg N+1 start_time 更新
```

## 配置

```yaml
boundary_redundancy:
  enabled: true

  confidence:
    min_gap_trigger: 0.05           # gap < 50ms 触发
    max_energy_slope_trigger: 3.0   # 渐变边界触发
    orphan_patterns:
      - '^(very|quite|rather|extremely|unbearably|really|so|too|just|only)$'
      - '^(hot|cold|big|small|good|bad|nice|great|loud|quiet|dark|bright)$'
      - '^(and|but|or|so|because|if|when|that|which|who)$'

  sliding_window:
    base_overlap_ms: 500
    fast_speech_wps: 4.0
    fast_overlap_ms: 750
    very_fast_overlap_ms: 1000

  arbitration:
    llm_model: "deepseek-v4-pro"
    auto_apply_confidence: 0.8
    review_threshold: 0.5
```

## 性能影响

| 指标 | 估算 |
|---|---|
| 触发率 | 10–20% 的边界（仅快语速、粘连场景） |
| 额外 ASR 调用 | 每触发边界 3 次（10–30s 语音窗口，可并行） |
| LLM 调用 | 每触发边界 1 次（约 200–500 tokens） |
| 延迟增加 | 每触发边界 +2–4 秒（主要为 ASR） |
| ASR 缓存 | 按 `(audio_hash, time_range)` 缓存窗口结果 |

## 模块文件

| 文件 | 说明 |
|---|---|
| `vocal_subtitle/asr/boundary_confidence.py` | 边界置信度评估器 |
| `vocal_subtitle/asr/boundary_reasr.py` | 滑动窗口重识别 |
| `vocal_subtitle/asr/boundary_arbitration.py` | LLM 语义仲裁器 |
| `vocal_subtitle/pipeline.py` | 集成为 Stage 4.6 |
| `vocal_subtitle/mapping/time_mapper.py` | 接受词重分配数据 |
| `vocal_subtitle/config.py` | 新增 BoundaryRedundancyConfig |
| `configs/default.yaml` | 新增默认配置 |
