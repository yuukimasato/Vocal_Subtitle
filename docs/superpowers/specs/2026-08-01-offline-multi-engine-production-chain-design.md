# 离线多引擎复核生产链调整方案

日期：2026-08-01  
状态：待实施设计基线  
范围：离线字幕生产链；流式模式不纳入本次生产化迁移  
目标：让“主引擎 + 异质副引擎 + 统一证据裁决 + 物理时间轴约束”成为最终字幕生产链

## 1. 结论摘要

当前项目已经具备新复核链的大部分基础组件：分段 ASR、全局 ASR 证据、物理时间线、风险评分、上下文 Re-ASR、Qwen/ForcedAligner/SED 适配边界、证据缓存和统一裁决契约。

但当前运行链仍是“旧字幕主链 + 新复核诊断层”：

- 默认 `shadow_mode=true`，复核结果不会替换最终字幕；
- 复核在后处理之后执行，顺序与目标流程相反；
- ForcedAligner、SED、语义审查在裁决之后执行，只进入诊断；
- Qwen 尚未按“异质副引擎”对称接入；
- `split/drop` 只存在于安全 API，未由实际批量裁决编排；
- 物理分箱和全局事件主要用于全局观察，不约束最终分段事件；
- 默认配置开启骨架分段，和总览图标注的“标准离线默认”不一致。

本方案不重写 VAD、说话人融合、字幕格式和 WebUI，而是重排离线编排边界，使最终 `SubtitleEvent` 只能由 `EvidenceDecision` 和物理事件构建阶段产生。

## 2. 范围与目标

### 2.1 本次范围

本次只改造 `Pipeline.run()` 的离线生产路径，包含：

1. 主引擎和异质副引擎的统一路由；
2. 全量主识别和风险窗口副识别；
3. Qwen、Whisper、FunASR 的双引擎组合；
4. 风险、文本、词级时间和声学证据的统一收集；
5. `EvidenceDecision` 作为最终事件的唯一入口；
6. 物理时间轴、词分配、字幕分箱和后处理的正确顺序；
7. 缓存、诊断、降级、质量模式和回归验收。

流式模式继续使用当前的滑动窗口离线模拟实现，不在本方案中承诺实时输入或实时字幕推送。

### 2.2 目标

- 默认生产模式下，主引擎全量识别，异质副引擎只复核高风险窗口；
- 质量模式下，主副引擎对所有物理语音窗口并行识别，不以耗时和能耗为约束；
- Whisper、Qwen、FunASR 可以按语言和配置组成异质引擎对；
- 全局 ASR 只能提供上下文、覆盖、重复和冲突证据，不能绕过裁决器生成最终字幕；
- ForcedAligner 只提供次级词级时间，SED 只提供声学标签，LLM 只提供语义证据；
- 所有最终字幕事件均可追溯到主候选、副候选、物理范围和裁决原因；
- 可选模型缺失、加载失败或超时时，主链仍能输出可解释的降级结果；
- 在真实黄金集验证通过后，关闭 shadow 输出，正式切换为新复核链生产。

### 2.3 非目标

- 不在本次重写 VAD、FFmpeg silencedetect、说话人聚类或字幕导出格式；
- 不把多个模型做成无条件的简单多数投票；
- 不用 CPS、黑名单、SED、LLM 或单个模型结果独立触发删除；
- 不把复核窗口起止时间直接作为字幕边界；
- 不把强制对齐后的错误文本伪装为可靠时间戳；
- 不把质量模式的高耗时作为默认生产性能目标。

## 3. 当前差距与必须调整项

### 3.1 复核没有成为最终输出链

当前 `EvidenceReviewService` 在默认 shadow 模式下直接保留输入事件；即使关闭 shadow，证据复核仍发生在后处理之后，决策事件也没有重新经过物理分配和字幕分箱。

必须调整为：

```text
主/全局/副引擎证据
    -> 风险评分与复核
    -> EvidenceDecision
    -> 物理边界校验与词分配
    -> SubtitleEvent 构建
    -> 说话人/语义合并/声学校验
    -> 最终化与导出
```

### 3.2 次级证据必须在裁决前收集

当前 ForcedAligner、SED、语义审查在 `EvidenceDecision` 之后运行，只产生诊断。必须先构造完整的 `EvidenceBundle`，再执行裁决：

```text
CandidateEvidence
  + ContextReASREvidence
  + SecondaryEngineEvidence
  + ForcedAlignmentEvidence
  + SEDEvidence
  + SemanticEvidence
  + PhysicalEvidence
    -> EvidenceDecisionEngine
```

其中：

- ForcedAligner 可以影响 `split`、时间一致性和 `unresolved`；
- SED 可以提高风险等级，但不能单独 `drop`；
- LLM 可以提供结构化语义风险，但不能生成时间戳或直接删除；
- 副 ASR 可以参与 `replace`、`split`、`unresolved`，但必须满足文本和物理保护条件。

### 3.3 副引擎必须按风险策略调度

当前 Qwen 开关打开后会对全部复核窗口运行，且没有明确“主引擎 Re-ASR 后仍冲突”的门控。改造后分为两种策略：

- `risk_only`：仅对 high/critical，或 medium 经主引擎 Re-ASR 后仍冲突的窗口调用副引擎；
- `full_quality`：副引擎对所有物理语音窗口运行，适合黄金集、模型对比和上线前质量验收。

### 3.4 `split/drop` 必须进入真实编排

`EvidenceDecisionEngine.drop()` 和 `split()` 继续保留安全前提，但需要由编排层根据证据条件调用。默认保护规则：

- `drop` 至少需要 high/critical 风险；
- SED、LLM 或单个副引擎不能单独 `drop`；
- 主副文本冲突且无法确认时使用 `unresolved`；
- `unresolved_keeps_candidate=true` 时保留主候选并标记诊断；
- `split` 必须有至少两个有效、物理合法且有时间证据的部分。

## 4. 双引擎生产模型

### 4.1 引擎角色

每个离线任务都明确两个角色：

```text
primary_engine    主引擎：负责默认字幕候选和全量主识别
secondary_engine  副引擎：提供异质文本和时间证据
```

主引擎和副引擎必须使用不同的模型家族。`faster-whisper` 和 `whisper.cpp` 都归入 Whisper 家族，默认不能互相作为异质副引擎；它们只能作为 Whisper 家族内的实现回退。

### 4.2 支持的组合

| 主引擎 | 副引擎 | 语言约束 | 适用场景 |
|---|---|---|---|
| FunASR | Qwen | 中文优先 | 中文音频，FunASR 主识别、Qwen 异质复核 |
| Qwen | FunASR | 仅中文 | 中文音频，Qwen 主识别、FunASR 异质复核 |
| Qwen | Whisper | 多语言 | Qwen 主识别，Whisper 复核 |
| Whisper | Qwen | 多语言 | Whisper 主识别，Qwen 复核 |

自动路由规则：

1. 主引擎为 FunASR 时，副引擎默认 Qwen；非中文任务自动降级主引擎到 Whisper，或按显式策略报告路由错误；
2. 主引擎为 Qwen 时，中文选择 FunASR，非中文选择 Whisper；
3. 主引擎为 Whisper 时，副引擎选择 Qwen；
4. 用户显式指定副引擎时，仍校验语言能力和模型家族，不允许同源副引擎静默替代异质复核；
5. 副引擎不可用时不阻断主链，诊断中记录 `secondary_unavailable`、原因、模型和降级动作。

### 4.3 两种运行模式

#### 生产模式：`risk_only`

默认策略：

```text
主引擎分段识别全部物理语音段
    -> 风险评分
    -> low: 保留主候选
    -> medium: 主引擎上下文 Re-ASR
    -> high/critical: 主引擎 Re-ASR + 异质副引擎复核
    -> 仍无法确认: unresolved
    -> 物理裁决与最终事件构建
```

该模式的“副引擎只复核高风险窗口”不等于副引擎不可用于普通质量评测；质量评测使用 `full_quality`。

#### 质量模式：`full_quality`

```text
主引擎全量识别所有物理语音窗口
副引擎全量识别相同窗口
全局 ASR/上下文/声学证据并行收集
    -> 所有窗口统一风险评分和裁决
```

质量模式不以耗时、GPU 显存和能耗为约束，但必须有并发上限、取消机制和缓存，避免同一窗口重复推理导致资源失控。

### 4.4 配置草案

```yaml
pipeline:
  mode: "offline"

  asr:
    engine: "auto"
    model: "large-v3"
    language: null
    engine_pair:
      enabled: true
      primary: "auto"          # auto | funasr | qwen | whisper
      secondary: "auto"        # auto | funasr | qwen | whisper
      policy: "risk_only"      # risk_only | full_quality
      same_family_policy: "reject"
      fallback_secondary: true
      max_workers: 2
      pair_route_version: "asr-pair-v1"

  evidence_review:
    enabled: true
    shadow_mode: false
    context_reasr_enabled: true
    qwen_enabled: true
    forced_aligner_enabled: false
    sed_enabled: false
    semantic_review_enabled: false
    unresolved_keeps_candidate: true

  acoustic_validation:
    skeleton_mode: false
```

说明：`skeleton_mode` 是否继续作为独立实验模式由场景配置决定；新生产默认链不应因为全局证据尝试而自动进入骨架分支。

## 5. 目标离线调用链

### 5.1 主链

```text
输入音频
  -> 人声分离（可选）
  -> 音频预处理和噪声画像
  -> 宏观切块（长音频可选）
  -> VAD/FFmpeg/RMS
  -> PhysicalTimeline/Shadow
  -> 主引擎分段识别
  -> 全局 ASR 证据（可选但不得生成最终事件）
  -> CandidateEvidence 归一化
  -> 主副引擎路由与风险调度
  -> Context Re-ASR / 异质副引擎 / ForcedAligner / SED / LLM
  -> EvidenceDecision
  -> 物理边界校验
  -> WordAllocation / PhysicalSubtitleBins
  -> SubtitleEvent 构建
  -> 说话人融合 / 角色标注 / 语义合并 / 声学校验
  -> EndTimeValidator
  -> LLM 文本优化（可选）
  -> Finalize
  -> SRT/VTT/ASS
```

### 5.2 关键顺序约束

1. `GlobalTranscript` 只能进入 `EvidenceBundle.global_evidence`；
2. 主副引擎结果都必须先归一化为 `CandidateEvidence`；
3. 所有次级证据必须在 `EvidenceDecisionEngine.decide_bundle()` 前完成收集；
4. `EvidenceDecision` 是唯一允许转换为 `SubtitleEvent` 的数据结构；
5. 物理校验不是只返回诊断，而是最终事件构建的必要门槛；
6. 后处理只能处理已经通过物理约束的事件；
7. `SubtitleBuilder` 和 `finalize_subtitle_events()` 的实际顺序为先最终化、后导出，流程图必须同步修正；
8. 全局 ASR 失败或副引擎失败只影响证据完整度，不应让主引擎字幕链失效。

## 6. 组件调整方案

### 6.1 新增引擎配对服务

新增 `vocal_subtitle/asr/engine_pairing.py`，职责：

- 将 `funasr`、`qwen`、`faster-whisper`、`whisper.cpp` 归一为模型家族；
- 根据显式配置、检测语言和自动路由选择主副引擎；
- 校验 FunASR 中文能力和 Qwen/Whisper 可用性；
- 生成不可变的 `EnginePairDecision`；
- 记录 `pair_route_version`、路由原因、语言和降级动作；
- 禁止同源主副引擎组合。

建议契约：

```python
@dataclass(frozen=True)
class EnginePairDecision:
    primary: str
    secondary: str | None
    primary_family: str
    secondary_family: str | None
    language: str | None
    policy: str
    decision_reason: str
    degraded: bool
    fallback_reason: str | None
    route_version: str
```

### 6.2 统一主副引擎适配端口

当前 `LazyQwenASR` 只适合作为复核端口，FunASR 和 Whisper 主要通过现有 `ASREngine` 使用。需要新增统一窗口识别端口：

```python
class WindowTranscriptionPort(Protocol):
    name: str
    family: str
    model_name: str

    def transcribe_window(
        self,
        audio: Any,
        sample_rate: int,
        window: ReviewWindow,
        *,
        language: str | None,
    ) -> Sequence[CandidateEvidence]: ...
```

适配器：

- `WhisperWindowAdapter`：包装 faster-whisper 或 whisper.cpp；
- `FunASRWindowAdapter`：包装现有 FunASR 引擎，保留中文语言限制；
- `QwenWindowAdapter`：复用惰性 Qwen 适配器，输出绝对时间；
- 所有适配器必须保留 engine、family、model、window、language、time source 和失败诊断。

主引擎和副引擎共用这个端口，避免出现“主引擎一套事件契约、副引擎另一套证据契约”。

### 6.3 新增离线生产协调器

新增 `vocal_subtitle/asr/offline_production.py`，不要继续把所有新职责堆入 `pipeline_runner.py`。协调器负责：

1. 接收 `PhysicalTimeline`、主候选、全局证据和配置；
2. 选择主副引擎对；
3. 执行风险评分和窗口调度；
4. 按 `risk_only/full_quality` 调用副引擎；
5. 收集 ForcedAligner、SED、LLM 和上下文证据；
6. 调用统一裁决器；
7. 调用物理事件构建器；
8. 返回最终事件、决策列表和完整诊断。

`pipeline_runner.py` 只负责生命周期、进度、缓存和端口注入。

### 6.4 改造 `EvidenceReviewService`

将当前顺序：

```text
主候选 -> 风险 -> Context/Qwen -> 决策 -> ForcedAligner/SED/LLM
```

改为：

```text
主候选 -> 风险 -> 窗口调度
       -> Context Re-ASR
       -> 异质副引擎
       -> ForcedAligner/SED/LLM
       -> 完整 EvidenceBundle
       -> EvidenceDecision
```

需要补充：

- `secondary_engine`、`engine_family`、`role`、`review_policy` 字段；
- 副引擎结果与对应主候选的关联；
- 复核阶段顺序和并行执行状态；
- 每个证据来源的成功、失败、超时和缓存命中信息；
- 语义审查端口的应用层接线；
- Qwen 仅在 risk-only 的高风险门控后运行，full-quality 则全量运行。

### 6.5 改造最终事件构建

新增或扩展 `vocal_subtitle/physical/decision_events.py`：

```text
EvidenceDecision[]
  -> accepted words
  -> physical clip validation
  -> allocate_words()
  -> repair_late_words()
  -> audit_physical_coverage()
  -> build_physical_subtitle_bins()
  -> build_events()
```

禁止在非 shadow 生产路径中直接调用 `decisions_to_subtitle_events()` 作为最终实现，因为它只能做基础对象转换，不能替代 PhysicalAllocator、Coverage Audit 和 SubtitleBins。

最终事件必须包含：

- `decision_id` 或等价裁决关联；
- 主候选和副候选 ID；
- `source_word_ids`；
- `physical_region_id`；
- `time_source`；
- `alignment_warning`；
- `revision_trace`；
- `risk_level` 和 `evidence_codes`。

## 7. 证据与裁决规则

### 7.1 证据优先级

这不是直接覆盖优先级，而是冲突分析优先级：

```text
物理语音范围合法性
  -> 主引擎原生词级时间
  -> 副引擎原生词级时间
  -> ForcedAligner 次级时间
  -> 分段边界时间
  -> 文本、置信度、上下文和语义证据
```

任何文本证据都不能绕过物理边界；任何时间证据都不能单独证明文本真实。

### 7.2 主副文本一致

- 文本相似、时间一致、物理范围一致：`keep` 主候选，附加 corroboration；
- 副引擎文本更完整且与主候选有足够相似度，词级时间和物理范围合法：`replace` 或 `split`；
- 主副文本明显不同且缺少可靠时间或语义证据：`unresolved`；
- 主候选高风险，副候选确认非语义声音，且物理/语义证据共同支持：允许 `drop`；
- 主候选高风险但副候选提供合理对白：优先保留或替换，不因高 CPS 自动删除。

### 7.3 允许删除的最低条件

`drop` 必须同时满足：

1. 风险等级为 high 或 critical；
2. 主候选缺少可靠词级时间，或存在明显无语音冲突；
3. 至少两个独立证据支持非语义声音或训练幻觉；
4. 没有可靠副引擎对白证据；
5. 物理时间线允许将该候选标记为空白；
6. 决策记录完整 `revision_trace`。

## 8. 缓存与诊断

每个主副引擎窗口结果的缓存键必须至少包含：

```text
input_hash
audio_hash
sample_rate
physical_timeline_version
window_start/window_end
role: primary/secondary
engine_family
engine
model
language
pair_route_version
review_policy
evidence_schema_version
risk_policy_version
decision_policy_version
```

诊断至少记录：

- 主副引擎选择、语言和路由原因；
- 每个窗口的起止时间、物理片段、风险等级和触发码；
- 主引擎、副引擎、Re-ASR、ForcedAligner、SED、LLM 的耗时和状态；
- 缓存命中、失败、超时、模型缺失和降级原因；
- 文本相似度、时间差、物理覆盖率和最终动作；
- `risk_only` 是否跳过副引擎；
- `full_quality` 是否完成主副全量覆盖。

不保存完整 LLM chain-of-thought，只保存结构化证据码和简短原因。

## 9. 错误处理与降级

| 场景 | 处理 |
|---|---|
| 主引擎不可用 | 按现有路由回退到可用主引擎；无法回退则任务失败并返回明确原因 |
| 副引擎不可用 | 保留主候选，标记 `secondary_unavailable`，不阻断任务 |
| FunASR 用于非中文 | 自动路由到 Whisper；显式强制时产生语言不匹配诊断 |
| Qwen 模型缺失 | 保留主候选，标记 `qwen_model_missing` |
| 副引擎超时 | 窗口降级为主候选，记录超时和资源信息 |
| 主副文本冲突 | 进入 `unresolved` 或满足条件后 `replace/split`，不做多数投票 |
| ForcedAligner 失败 | 保留文本证据和未对齐状态，不伪造时间 |
| SED 失败 | 风险评分缺少 SED 因子，不独立改变动作 |
| LLM 失败 | 使用规则证据，不阻断字幕输出 |
| 物理范围非法 | 不直接输出该决策事件；标记 unresolved 或 drop，保留诊断 |
| 全局 ASR 失败 | 继续主引擎链，不影响基础字幕生产 |

## 10. 分阶段实施顺序

### Phase 0：基线与契约冻结

- 固定当前 942 个测试和现有质量黄金集结果；
- 增加 `role`、`engine_family`、`pair_route_version`、`review_policy` 字段；
- 明确 `EvidenceDecision` 是唯一最终事件来源；
- 增加 final event 必须关联 decision 的契约测试；
- 保留 `legacy`、`shadow`、`production`、`full_quality` 四种 rollout 状态。

### Phase 1：引擎配对和统一适配

- 实现 `engine_pairing.py`；
- 增加 Whisper、FunASR、Qwen 的统一窗口适配器；
- 实现四种合法组合和语言门控；
- 增加模型不可用、同源拒绝、中文/非中文路由测试；
- 在不改变最终字幕的情况下记录主副引擎诊断。

### Phase 2：复核编排重排

- 将次级证据收集移动到裁决前；
- 接通语义审查端口；
- 统一 `risk_only/full_quality` 调度；
- 对 high/critical 加入副引擎门控；
- 对 full-quality 保证每个物理语音窗口都有主副结果或结构化失败诊断；
- 增加主副一致、冲突、超时、缓存命中和降级测试。

### Phase 3：物理事件生产化

- 将 `EvidenceDecision` 接到 WordAllocation、Coverage、SubtitleBins 和 `build_events()`；
- 删除全局事件直接参与最终导出的旧路径；
- 确保分段、骨架和长音频路径都使用同一物理事件构建入口；
- 后处理只接收物理校验通过的事件；
- 增加物理越界、跨静音、迟到词、分裂词和 unresolved 测试。

### Phase 4：生产模式切换

- 引入 `production` rollout 状态和配置级 `legacy` 回退开关；
- 在真实模型和黄金集验收完成前，默认仍保持 `shadow_mode=true`；
- 允许通过显式配置启用 `production + risk_only`，用于受控验收；
- 默认复核策略设为 `risk_only`；
- 质量评测和诊断命令支持 `full_quality`；
- 默认主引擎仍由语言路由决定，副引擎按配对矩阵自动选择；
- `skeleton_mode` 改为显式场景选项，不再成为新生产链的隐式默认路径；
- 同步更新流程图，使其明确显示主副引擎策略和最终事件顺序。

### Phase 5：真实模型和黄金集验收

- 下载并固定 Qwen、FunASR、Whisper 真实模型版本；
- 对中文、英文、混合语言、呻吟/呼吸、多人和长音频样本执行 `risk_only` 与 `full_quality`；
- 记录 CPU/GPU、显存、耗时、缓存命中和失败降级；
- 校准风险阈值和副引擎触发规则；
- 只有通过质量门禁后，才将 `shadow_mode=false` 和 `production + risk_only` 作为默认最终输出路径。

## 11. 测试与验收标准

### 11.1 单元与契约

- 四种引擎组合均能生成合法 `EnginePairDecision`；
- FunASR 非中文路由不会静默产生中文副字幕；
- 主副引擎不能同源；
- 所有候选保留 engine、family、role、window 和 time source；
- 所有最终事件都有 decision trace；
- 缺少置信度和时间时保持 `None`；
- `drop`、`split`、`unresolved` 的保护条件可测试。

### 11.2 集成流程

- 复核服务在裁决前收集所有次级证据；
- `risk_only` 的低风险窗口不调用副引擎；
- `risk_only` 的高风险窗口调用副引擎；
- `full_quality` 对全部物理语音窗口调用主副引擎；
- 副引擎失败仍能输出主候选；
- 全局 ASR 事件不能直接进入最终导出；
- 非 shadow 路径最终事件必须经过物理分箱和覆盖审计；
- 后处理、最终化和多格式导出使用同一批决策生成事件。

### 11.3 质量门禁

最低硬门禁：

- 全部现有测试保持通过；
- 最终字幕事件物理越界数为 0；
- 最终字幕事件未关联裁决数为 0；
- full-quality 模式主副覆盖率达到 100%，失败窗口必须有诊断；
- same-family 主副组合数为 0；
- 可选副引擎缺失不能导致任务异常退出。

黄金集门禁：

- 以当前生产链建立文本、幻觉保留、误删除、时间边界和未解决率基线；
- 新链不得增加真实字幕误删除率；
- 新链应降低重复训练短语和非语义声音字幕的保留率；
- 所有下降或提升都必须按语言、主副引擎组合和场景拆分报告；
- 未有足够黄金集样本的指标标记为 `uncalibrated`，不能当作通过。

## 12. 发布与回滚

发布状态：

```text
legacy      旧链最终输出，新链不运行
shadow      旧链最终输出，新链完整运行并记录差异
production  新链最终输出，失败时按配置回退旧链
full_quality新链主副全量输出，主要用于验收和对比
```

推荐切换顺序：

1. `shadow + risk_only` 验证决策、物理边界和模型组合；
2. `shadow + full_quality` 建立主副引擎黄金集对比；
3. 修复误删除、未解决率和时间边界问题；
4. 切换 `production + risk_only`；
5. 保留一次配置级旧链回退能力；
6. 质量模式作为独立验收和回归工具，不作为默认生产策略。

任何模型依赖、GPU、网络或第三方 API 失败都应通过结构化诊断和回退处理，而不是依赖异常文本判断是否成功。

## 13. 流程图同步调整

现有流程图需要在代码改造完成后同步修订以下内容：

- 将“标准离线默认”改为实际配置默认，明确 `risk_only`；
- 增加 `EnginePairRouter` 和主副引擎矩阵；
- 将 Qwen/Whisper/FunASR 表示为互为异质复核，而不是 Qwen 固定第二引擎；
- 增加 `full_quality` 全量双引擎分支；
- 将 ForcedAligner、SED、LLM 放到裁决前；
- 将 `EvidenceDecision` 放在物理事件构建之前；
- 将 `WordAllocation → Coverage → SubtitleBins → build_events` 画成最终必经路径；
- 修正最终化和导出顺序为 `Finalize → SubtitleBuilder.build → SRT/VTT/ASS`；
- 将流式分支标为“当前离线文件模拟，实时输入另行建设”。

## 14. 完成定义

当以下条件全部满足时，才认为新复核链已经成为最终生产链：

1. `production` 模式默认关闭 shadow，最终字幕只能由 `EvidenceDecision` 生成；
2. Whisper、Qwen、FunASR 至少完成四种合法组合的真实模型 smoke test；
3. risk-only 和 full-quality 两种模式均有集成测试；
4. 副引擎、ForcedAligner、SED、LLM 的证据都在裁决前完成汇总；
5. 所有最终事件都通过物理范围约束、覆盖审计和最终化；
6. 全局 ASR 不再直接生成最终字幕事件；
7. 黄金集质量门禁通过，误删除和幻觉保留率达到发布阈值；
8. 模型缺失、超时、语言不匹配和主副冲突都有可解释降级；
9. 流程图、默认配置、API 诊断和实际调用链一致；
10. 旧链回退开关经过一次真实回滚演练。
