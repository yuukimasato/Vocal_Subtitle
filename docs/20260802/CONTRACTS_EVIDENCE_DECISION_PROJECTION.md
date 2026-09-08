# 证据、决策与投影契约

**版本**: contracts-v1
**更新日期**: 2026-08-02
**依据**: [ADR-001](../adr/001-unified-decision-export.md), [ADR-002](../adr/002-physical-boundary-priority.md)

## 1. 三层模型概览

```
┌─────────────────────────────────────────────────────┐
│                    证据层 (Evidence)                  │
│  CandidateEvidence + EvidenceWord                   │
│  来源: segmented, global, qwen, context_reasr, ...  │
│  职责: 统一候选格式，携带来源、时间、置信度           │
└──────────────────────┬──────────────────────────────┘
                       │ EvidenceBundle
                       ▼
┌─────────────────────────────────────────────────────┐
│                    决策层 (Decision)                  │
│  EvidenceDecisionEngine + RiskScorer                │
│  动作: keep | replace | split | drop | unresolved   │
│  职责: 文本选择、风险评估、物理校验                   │
└──────────────────────┬──────────────────────────────┘
                       │ EvidenceDecision[]
                       ▼
┌─────────────────────────────────────────────────────┐
│                    投影层 (Projection)                │
│  DecisionEventProjector                             │
│  路径: decisions → IR → bins → allocate → events    │
│  职责: 词级分配、物理分箱、覆盖审计、字幕事件投影      │
└──────────────────────┬──────────────────────────────┘
                       │ SubtitleEvent[]
                       ▼
┌─────────────────────────────────────────────────────┐
│                    展示层 (Display)                   │
│  后处理: 说话人、语义合并、格式化、导出               │
│  约束: 只能在物理和决策边界内组织展示                  │
└─────────────────────────────────────────────────────┘
```

## 2. 候选项统一模型 (CandidateEvidence)

### 核心字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | `str` | ✅ | 全局唯一标识，格式 `{source}:segment:{index}:{window}` |
| `source` | `str` | ✅ | 来源：`segmented`/`global`/`context_reasr`/`qwen`/`forced_aligner`/`local_recovery`/`sed`/`llm` |
| `text` | `str` | ✅ | 文本内容 |
| `start` | `float` | ✅ | 开始时间（绝对坐标） |
| `end` | `float` | ✅ | 结束时间（绝对坐标） |
| `engine` | `str?` | | 引擎名：`faster-whisper`/`funasr`/`qwen-asr` |
| `model` | `str?` | | 模型名：`large-v3`/`paraformer-zh`/`Qwen3-ASR-1.7B` |
| `window_id` | `str?` | | 所属窗口 ID（global ASR 场景） |
| `words` | `tuple[EvidenceWord]` | | 词级时间戳 |
| `confidence` | `float?` | | 综合置信度 [0,1] |
| `language` | `str?` | | 检测语言 |
| `physical_clip_id` | `str?` | | 关联的物理 clip |

### 词级模型 (EvidenceWord)

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | `str` | ✅ | 词级唯一标识 |
| `text` | `str` | ✅ | 词文本 |
| `start` | `float?` | | 词开始时间（绝对坐标） |
| `end` | `float?` | | 词结束时间（绝对坐标） |
| `confidence` | `float?` | | 词置信度 [0,1] |
| `time_source` | `str` | ✅ | 时间来源：`native_word_timestamp`/`qwen_forced_alignment`/`segment_boundary`/`physical_acoustic_boundary` |
| `speaker_id` | `int?` | | 说话人 ID |
| `diagnostics` | `dict` | | 诊断信息（含 `invalid_timing` 标记） |

### 候选构建入口

| 函数 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `candidates_from_segments()` | ASR segments | `list[CandidateEvidence]` | 分段主候选 |
| `candidates_from_global_transcript()` | GlobalTranscript IR | `list[CandidateEvidence]` | 全局证据 |
| `candidate_from_subtitle_event()` | SubtitleEvent | `CandidateEvidence` | 旧事件适配 |
| `evidence_word_from_asr()` | ASR word | `EvidenceWord` | 词级适配 |

### 不变量

1. 所有时间坐标使用**原始音频的绝对时间**（秒），禁止混合相对/绝对坐标
2. `start` < `end`，且两者的值必须在 `[0, audio_duration]` 范围内
3. `text` 不能为空字符串
4. `words` 中的词按时间排序，如有词级时间则必须在候选时间范围内
5. `source` 必须来自预定义的 8 种来源之一，禁止自定义来源绕过审计

---

## 3. 决策层 (EvidenceDecisionEngine)

### 决策动作

| 动作 | 含义 | 触发条件 | 对输出的影响 |
|------|------|----------|-------------|
| `keep` | 保留候选 | 风险评估 low/medium，无更好的替代 | 候选文本和时间直接进入 final |
| `replace` | 用 evidence 替换 | 替代候选有词级时间戳且语义相似 | 替换文本和时间，保留 candidate_id 追踪 |
| `split` | 拆分为多个事件 | context_reasr 提供了更细粒度的拆分 | 拆分为 N 个独立决策 |
| `drop` | 删除候选 | 多源非语音证据（SED + semantic）+ 物理无覆盖 | 不生成字幕事件 |
| `unresolved` | 证据不足，保守保留 | 高风险但无多源证据支持 drop | 保留候选但标记为 unresolved |

### 决策记录

每个 `EvidenceDecision` 包含完整追踪信息：

```json
{
  "candidate_ids": ["segmented:segment:000042:audio", "qwen:segment:15:window_03"],
  "decision": "replace",
  "final_text": "修正后的文本",
  "final_words": [...],
  "start": 12.34, "end": 14.56,
  "time_source": "native_word_timestamp",
  "confidence": 0.87,
  "risk_score": 0.22,
  "risk_level": "low",
  "evidence_codes": ["qwen_agreement"],
  "physical_validation": {"valid": true, "status": "supported", "overlap_seconds": 2.20},
  "revision_trace": [
    {"stage": "risk_scoring", "score": 0.22, "level": "low"},
    {"stage": "alternative_selection", "source": "qwen", "selected_candidate_id": "..."},
    {"stage": "physical_validation", "valid": true}
  ]
}
```

### 决策配置 (DecisionConfig)

```python
@dataclass(frozen=True)
class DecisionConfig:
    unresolved_keeps_candidate: bool = True   # unresolved 时保守保留文本
    physical_tolerance: float = 0.30          # 物理覆盖容忍度（秒）
    replace_min_similarity: float = 0.35      # 替换的最低文本相似度
    require_multi_source_drop: bool = True    # drop 需要至少 2 个独立源
```

### 决策不变量

1. **`EvidenceDecision` 是文本选择的唯一出口**：任何模块不得绕过此层直接修改最终字幕文本
2. **drop 必须有至少 2 个独立证据源**（当 `require_multi_source_drop=True`）
3. **物理校验失败的 alternative 不被接受**：回退到原始候选
4. **unresolved 默认保守保留**（`unresolved_keeps_candidate=True`）
5. **所有决策必须有 `revision_trace`**：从 risk_scoring → alternative → physical_validation 完整记录

---

## 4. 投影层 (DecisionEventProjector)

### 投影路径

```
EvidenceDecision[]
    │
    ├─ 无 PhysicalTimeline → _project_one() 直接投影
    │   └─ diagnostics.mode = "direct"
    │
    └─ 有 PhysicalTimeline → _project_physical()
        │
        ├─ 1. decisions_to_global_transcript()
        │      EvidenceDecision → GlobalTranscript IR
        │
        ├─ 2. build_physical_subtitle_bins()
        │      物理时间线 → 字幕分箱
        │
        ├─ 3. repair_late_words()
        │      修复迟到词（词时间在候选之后）
        │
        ├─ 4. allocate_words()
        │      词级物理分配
        │
        ├─ 5. audit_physical_coverage()
        │      覆盖审计
        │
        └─ 6. build_events()
               GlobalSubtitleEvent → SubtitleEvent
```

### 诊断输出 (DecisionProjectionResult.diagnostics)

```json
{
  "mode": "physical",
  "decision_count": 42,
  "event_count": 41,
  "bin_count": 8,
  "bins": [...],
  "allocation": {"word_count": 156, "late_repair_count": 3},
  "rejected_word_ids": ["..."],
  "physical_violation_count": 0,
  "cross_silence_count": 0,
  "decision_trace_missing_count": 0,
  "raw_event_bypass_count": 0,
  "coverage": {
    "physical_bin_count": 8,
    "covered_physical_bin_count": 8,
    "uncovered_physical_bin_count": 0,
    "complete": true
  }
}
```

### 投影不变量

1. **投影是唯一的事件构建路径**：禁止 raw bypass（raw_event_bypass_count 必须为 0）
2. **物理 bin 约束不可绕过**：字幕事件的 `physical_start`/`physical_end` 受限于所在的物理 bin
3. **跨静音检测**：相邻物理 span 间隔 > 0.35s 时计入 `cross_silence_count`
4. **decision_trace_missing_count** 必须为 0：每个事件必须有完整的决策追踪
5. **直接投影模式仅在无 PhysicalTimeline 时使用**：有物理时间线时必须走完整物理投影

---

## 5. 物理覆盖审计 (PhysicalCoverageReport)

### 审计维度

| 维度 | 指标 | 告警条件 |
|------|------|----------|
| 覆盖率 | `covered / total` | < 0.80 |
| 未覆盖 bin | `uncovered_physical_bin_count` | > 0 |
| 尾部间隙 | `tail_gap_seconds` | > 2.0s |
| 过分配 | `over_allocated_segments` | > 0 |
| 欠分配 | `under_allocated_segments` | > 0 |

### 覆盖区间 (PhysicalCoverageRange)

连续未覆盖的物理 bin 组，用于局部恢复决策：

```json
{
  "start": 45.2, "end": 48.7, "duration": 3.5,
  "bin_ids": ["bin_12", "bin_13"],
  "physical_clip_id": "clip_03"
}
```

---

## 6. 坐标转换约定

### 绝对坐标体系

- **所有内部时间坐标使用原始音频的绝对时间**（秒）
- `CoordinateMapper` 负责将碎片化时间轴统一映射到原始音频时间轴
- 同一个偏移只应用一次，禁止重复应用

### 坐标转换检查

| 检查项 | 测试位置 | 预期 |
|--------|----------|------|
| offset 只应用一次 | `tests/test_physical/test_coordinate.py` | 绝对坐标无重复偏移 |
| 宏块坐标一致 | 同文件 | chunk 内坐标 = 原始坐标 - chunk_offset |
| 骨架时间与原始对齐 | 同文件 | ffmpeg 骨架时间 = PhysicalTimeline 时间 |
| 全局时间与分段一致 | `tests/test_global_asr_path.py` | global 和 segmented 在同一坐标系 |

### 时间来源优先级

1. `native_word_timestamp` — ASR 原生词级时间戳（最高精度）
2. `qwen_forced_alignment` — 强制对齐结果
3. `physical_acoustic_boundary` — 物理声学边界
4. `segment_boundary` — 段边界（最粗粒度，仅在无词级时间时使用）

---

## 7. 可追溯性：从词到字幕

### 追溯链

```
EvidenceWord.id
  └─ CandidateEvidence.id
      └─ EvidenceDecision.candidate_ids
          └─ SubtitleEvent.source_word_ids
              └─ SubtitleEvent.revision_trace[].decision_id
```

### 追溯性保证

对任意字幕 cue（最终输出的一条字幕），必须能够回答：

1. **文本来源**：来自哪个候选？哪个引擎？哪个模型？
2. **决策理由**：keep/replace/split/drop/unresolved？
3. **物理范围**：被分配到哪个物理 clip？覆盖了哪些 speech span？
4. **时间来源**：词级时间戳来自原生 ASR 还是强制对齐？
5. **修改历史**：经过了哪些后处理步骤？

### 检查方法

```python
# 对任意 SubtitleEvent
event.text           # 最终文本
event.source_word_ids  # 追溯到 EvidenceWord
event.revision_trace    # 追溯到 EvidenceDecision
event.physical_region_id  # 追溯到 PhysicalTimeline
event.physical_start / event.physical_end  # 物理约束范围
```

---

## 8. 后处理显示层约束

### 允许的操作（显示层）

| 操作 | 允许 | 约束 |
|------|------|------|
| 说话人标签分配 | ✅ | 不改变文本和时间 |
| 标点规范化 | ✅ | 不改变语义 |
| 语义合并 | ✅ (在物理 bin 内) | 不跨硬静音/物理 bin 边界 |
| 帧级无缝衔接 | ✅ | 受 `max_stitch_gap` 和声学校验门控 |
| 格式化（大小写、数字） | ✅ | 不改变文本内容 |
| 字幕分箱重组 | ✅ | 保持在物理 bin 约束内 |

### 禁止的操作（显示层）

| 操作 | 禁止原因 |
|------|----------|
| 修改最终文本内容 | 绕过 EvidenceDecision 出口 |
| 跨越物理 bin 合并字幕 | 违反物理边界优先（ADR-002） |
| 修改时间戳越过物理边界 | 违反声学校验 |
| 在硬静音区伪造语音 | 伪造物理证据 |
| 删除 trace/溯源信息 | 破坏可追溯性 |

### 声学校验最终关卡

`acoustic_validator.py` (Plan 7) 作为后处理的最终关卡：
- 方向感知查询：禁止跨静音延长、允许向静音缩短
- 置信度门控：保留可靠 ASR 边界，仅修正低置信度边界
- 诊断报告：健康评分 + 逐决策审计日志

---

## 9. 降级投影

### 降级条件

当以下条件之一满足时，启用降级投影：

1. `PhysicalTimeline` 不可用（分离/VAD 失败）
2. `build_physical_subtitle_bins()` 返回空列表
3. `EvidenceDecision` 无有效的物理验证

### 降级行为

```
降级投影:
  - 跳过物理分箱和词级分配
  - 直接 _project_one() 逐个投影
  - diagnostics.mode = "direct"
  - 标记 raw_event_bypass_count（降级投影中允许）
  - 字幕事件的 physical_* 字段为 None
```

### 降级追踪

降级投影的输出必须标记：
- `production_path = "projector_degraded"`
- `diagnostics.mode = "direct"`
- 原因写入 `fallback_reason`

---

## 10. 契约测试

### 必测场景

| 测试 | 类型 | 验证点 |
|------|------|--------|
| CandidateEvidence 构建与序列化 | 单元 | id/text 非空，start<end，source 合法 |
| EvidenceDecision keep 动作 | 单元 | trace 完整，risk_score 在 [0,1] |
| EvidenceDecision replace 动作 | 单元 | similarity >= replace_min_similarity |
| EvidenceDecision drop 动作 | 单元 | 需要 2+ 独立源，risk_level high/critical |
| DecisionEventProjector 物理投影 | 集成 | mode=physical，raw_bypass=0 |
| DecisionEventProjector 直接投影 | 集成 | mode=direct，无物理字段 |
| 覆盖审计检测 gap | 单元 | uncovered_bin_count > 0 触发告警 |
| 跨静音检测 | 集成 | span gap > 0.35s → cross_silence_count += 1 |
| 后处理不跨物理 bin | 集成 | 合并后的 event physical_bin 一致 |
| 降级投影追踪 | 集成 | production_path 标记正确 |

### 回归测试

```bash
# 运行契约相关测试
pytest tests/test_physical/ -v
pytest tests/test_asr/test_router.py -v
pytest tests/test_global_asr_path.py -v
pytest tests/test_offline_production.py -v
```

---

## 11. 核心不变量总结

1. **分段主候选是默认生产基线**：global ASR 和副引擎首先是证据来源
2. **`EvidenceDecision` 是文本选择出口**：任何路径不得绕过
3. **`DecisionEventProjector` 是事件投影出口**：禁止 raw bypass
4. **`PhysicalTimeline` 决定字幕物理边界**：后处理不能伪造语音范围
5. **所有时间坐标使用绝对时间**：同一偏移只应用一次
6. **所有决策有 revision_trace**：从候选到最终字幕完整可追溯
7. **后处理只能在显示层操作**：不能破坏物理边界、决策文本和时间追踪
