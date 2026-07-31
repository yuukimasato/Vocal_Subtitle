# 全局 ASR 多引擎复核组件化落地设计

日期：2026-07-31  
状态：已确认，作为 Phase 0-3 实施基线  
关联基线：`2026-07-31-global-asr-multi-engine-review-final.md`

## 1. 目标与范围

本次实现基于仓库现有组件化架构，完成全局 ASR 语义迁移：分段 ASR 提供默认主字幕候选，全局 ASR 只提供上下文、漏识别、重复和冲突证据，所有最终字幕事件统一经过 `EvidenceDecision`。

第一阶段包含：

- Phase 0：证据契约、缺失置信度语义、序列化和回归基线；
- Phase 1：全局 ASR 证据化，保留 `global/auto/segmented` 配置兼容；
- Phase 2：可解释风险评分、物理窗口调度和主引擎 Context Re-ASR；
- Phase 3：统一 `keep/replace/split/drop/unresolved` 裁决、物理边界校验和诊断链路；
- 为 Qwen3-ASR、ForcedAligner、SED、LLM 提供可选适配器协议和降级接口，但本阶段不要求安装或运行具体模型。

本阶段不改变 SRT/VTT/ASS 导出格式，不重写 VAD、说话人聚类、分离器、流式路径或现有后处理组件，不把全部逻辑重新集中到 `pipeline.py`。

## 2. 当前组件边界

现有可复用组件：

- `vocal_subtitle/asr/contracts.py`：ASR 应用服务端口和请求/结果数据结构；
- `vocal_subtitle/asr/global_path.py`：全局 ASR 服务边界；
- `vocal_subtitle/asr/segmented_path.py`：分段 ASR 路径；
- `vocal_subtitle/asr/router.py`：任务级引擎和语言路由；
- `vocal_subtitle/application/asr_path.py`：当前应用层兼容适配；
- `vocal_subtitle/physical/timeline.py`、`context.py`、`shadow.py`：物理范围、上下文和声学证据；
- `vocal_subtitle/physical/ir.py`、`allocator.py`、`events.py`：当前全局词流和物理分配实现；
- `vocal_subtitle/mapping/*`：最终字幕事件约束、分段和导出。

必须迁移的旧语义：

1. `GlobalASRResult.events` 不能继续作为最终字幕契约；
2. `PipelineASRPathMixin` 不能在全局成功后直接把全局事件送入后处理；
3. 全局路径中的缺失词置信度不能继续默认写成 `0.9`；
4. `GlobalSubtitleEvent.to_subtitle_event()` 只能保留为兼容/测试工具，不能成为新路径的最终事件入口。

## 3. 新增组件

### 3.1 `vocal_subtitle/asr/evidence.py`

定义稳定的数据契约：

- `EvidenceWord`：词文本、可选时间、可选置信度、`time_source`；
- `CandidateEvidence`：来源、引擎、窗口、文本、词、ASR 指标、物理 clip、诊断；
- `EvidenceDecision`：候选集合、动作、最终文本/词/时间、风险、证据代码、物理校验和 `revision_trace`；
- `EvidenceBundle`：分段主候选、全局证据和窗口证据的集合，供评分器和裁决器消费；
- `EVIDENCE_SCHEMA_VERSION`：用于缓存和可复现诊断。

所有模型使用 `None` 表示缺失置信度或缺失时间，不使用伪造的高置信度默认值。契约提供 `to_dict/from_dict`，拒绝非法范围、空 ID 和倒序时间。

### 3.2 `vocal_subtitle/asr/risk_scoring.py`

实现规则型、可解释的风险评分器。输入只读候选和物理/全局证据，输出 `RiskAssessment`，包括分数、等级、证据代码和建议动作。初始规则覆盖：短时长、高 CPS、重复、全局冲突、缺失时间、缺失置信度、模型质量指标、物理覆盖不足和上下文不完整。

评分器只负责调度，不负责删除。`drop` 必须由裁决器根据多证据策略决定。

### 3.3 `vocal_subtitle/asr/review_scheduler.py`

根据风险等级合并相邻异常事件，调用 `PhysicalTimeline`/`context.py` 生成有界窗口。默认上下文左右各 0.8 秒，窗口组上限 8-12 秒，单窗上限 15 秒；参数通过配置注入。窗口不直接成为字幕边界，必须保留原始候选 ID、物理坐标和任务语言。

该组件通过 `ReviewEnginePort` 协议调用主引擎 Context Re-ASR，所有异常转换为诊断并走降级路径。

### 3.4 `vocal_subtitle/asr/evidence_decision.py`

实现唯一最终裁决入口：

- `keep`：保留分段主候选；
- `replace`：采用复核文本，但重新验证词级时间和物理范围；
- `split`：依据可靠词级时间、停顿和物理边界重新分段；
- `drop`：仅在多证据满足保护规则时执行；
- `unresolved`：冲突未解决，不强行选择证据。

裁决器不直接处理音频模型加载，不依赖 `Pipeline`，通过端口取得物理校验、候选匹配和可选复核结果。输出之后才允许进入现有 mapping/postprocess/export 组件。

### 3.5 `vocal_subtitle/asr/review_engines.py`

定义可选复核能力协议和默认空实现：

- `ContextReASRPort`；
- `QwenASRPort`；
- `ForcedAlignerPort`；
- `SEDPort`；
- `SemanticReviewPort`。

第一阶段只有 Context Re-ASR 接入主引擎；其他端口返回不可用诊断，不阻塞基础字幕生成。该边界允许后续新增具体适配器而不修改裁决器和应用层流程。

## 4. 应用层数据流

```text
segmented ASR
    -> CandidateEvidence(source=segmented)
global ASR
    -> GlobalTranscript / CandidateEvidence(source=global)
PhysicalTimeline + ASR diagnostics
    -> RiskAssessment
RiskAssessment
    -> low: direct candidate
    -> medium/high: ReviewScheduler -> Context Re-ASR evidence
CandidateEvidence + all review evidence
    -> EvidenceDecision
EvidenceDecision
    -> physical boundary validation
    -> existing mapping/postprocess/export
```

`GlobalASRService` 的结果类型改为同时携带 `transcript` 和 `evidence`，保留只读的 legacy events 字段以支持迁移期调用方，但新应用路径不消费该字段。`PipelineASRPathMixin` 在 global/auto 路径中先取得分段主候选，再生成全局证据，统一调用编排器；全局失败只记录降级诊断并继续分段主路径。

为减少一次性风险，`segmented` 配置直接走同一裁决器但不调用全局服务；`global/auto` 配置默认启用全局证据，不能再把全局事件标记为最终成功结果。

## 5. 物理边界与时间证据

时间证据明确区分：

- `native_word_timestamp`：ASR 原生词级时间，优先观察值；
- `qwen_forced_alignment`：已接受文本的次级对齐时间；
- `segment_boundary`：原始分段边界；
- `physical_acoustic_boundary`：VAD/FFmpeg/RMS/SED 物理范围。

最终校验复用现有 `PhysicalTimeline`，依次检查词级时间、50-100ms 初始容差、声学证据范围、长静音和说话人边界。超出范围且无法解释时输出 `unresolved`。不得把窗口起止直接当字幕边界，也不得用 ForcedAligner 证明错误文本为真。

## 6. 配置、缓存和诊断

新增配置保持默认保守：

- `evidence_review.enabled` 默认开启契约和裁决；
- `evidence_review.shadow_mode` 初始开启，记录建议动作但保持基线输出；
- `context_reasr.enabled` 默认开启但受 medium/high 风险触发；
- `qwen/align/sed/semantic_review.enabled` 默认关闭；
- `unresolved` 默认保留原分段候选并标记诊断，不自动删除。

证据缓存键至少包括输入哈希、音轨哈希、物理时间线版本、窗口坐标、阶段、引擎/模型、语言、路由版本、证据 schema 版本、评分策略版本和裁决策略版本。现有缓存调用方通过兼容字段读取，不能把不同引擎或不同窗口结果混用。

每个阶段记录 source、engine、model、language、window、cache hit、wall time、风险分数、evidence codes、decision、物理校验和降级原因。

## 7. 测试策略

新增单元测试覆盖：

- 契约序列化、`None` 置信度、非法时间和证据来源；
- 风险规则的 low/medium/high/critical 分级；
- 相邻风险事件窗口合并、长静音不跨越和语言透传；
- 全局证据不能直接生成最终事件；
- `EvidenceDecision` 五种动作、revision trace 和 unresolved；
- 物理范围越界、缺失词级时间和 ForcedAligner 次级时间；
- Qwen/SED/LLM 不可用时的降级；
- global/auto/segmented 现有配置兼容和基础导出回归。

保留现有全局路径测试，在迁移完成后将“全局成功后直接使用 global events”的断言改为“全局证据进入统一裁决器”。Phase 3 结束前默认 shadow mode 输出必须与当前基线一致，除非测试明确验证了新的裁决动作。

## 8. 交付顺序

1. 契约和策略对象；
2. 全局结果转证据并移除新路径对 global events 的直接消费；
3. 风险评分、窗口调度和 Context Re-ASR 端口；
4. EvidenceDecision 和物理边界桥接；
5. 应用层接线、配置、缓存诊断和回归测试；
6. 运行完整测试集并输出基线对比报告。

实现完成的必要条件：所有最终字幕事件可追溯到 `EvidenceDecision`，全局失败不丢失分段主路径，缺失置信度不会被视为高置信度，且 Qwen/ForcedAligner/SED/LLM 不可用时基础流程仍可完成。
