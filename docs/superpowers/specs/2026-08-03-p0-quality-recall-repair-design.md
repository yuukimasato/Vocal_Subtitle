# P0 离线字幕质量召回收敛设计

日期：2026-08-03  
状态：已确认，待实施

## 1. 目标与范围

本阶段承接 `docs/20260802` 中已经完成的离线生产链架构收敛，目标是修复黄金集中的真实语音漏检和文本不匹配，同时保持物理安全契约不回退。

生产主链保持为：

```text
segmented primary
  -> global evidence
  -> risk/review
  -> EvidenceDecision
  -> DecisionEventProjector
  -> postprocess/export
```

本阶段包含：

- 基于同一黄金集报告定位 `speech_candidate_missing`、`asr_text_mismatch` 和 `timeline_shift_or_boundary_mismatch`；
- 修复候选召回、词级时间、局部 recovery 或字幕投影中能够由代码确定的缺陷；
- 保持 global evidence、local recovery 和 authoritative projection 的统一追踪链；
- 为每个修复补充最小单元测试和路径回归测试；
- 重新生成 baseline、shadow、authoritative 三种报告，并执行严格 gate。

本阶段不包含：

- 放宽 `real_speech_drop_rate` 或其他质量门禁；
- 将 Qwen、FunASR、ForcedAligner 或 SED 设为 release-default；
- 让反馈 profile 自动修改生产参数；
- 流式模式改造；
- 与当前漏检无关的 WebUI 重构或大规模组件化。

## 2. 当前基线与问题分类

当前 P0 相关测试已经覆盖默认 segmented 路由、global evidence、统一决策/投影、local recovery 和黄金集归因。现有黄金集报告表明：物理违规、跨静音、幻觉保留、decision trace 和 raw bypass 已满足安全要求，但真实语音漏检仍超过 `0.05` 发布线。

每个未匹配参考事件使用以下优先级归因：

1. 有时间重叠但文本相似度不足：`asr_text_mismatch`；
2. 文本相似但时间偏移在可诊断范围内：`timeline_shift_or_boundary_mismatch`；
3. 没有有效候选覆盖或相近文本：`speech_candidate_missing`。

修复必须先根据归因决定所属阶段，不能通过改变匹配阈值掩盖识别质量，也不能把跨 physical bin 的事件合并来提高匹配率。

## 3. 组件与职责

### 3.1 质量报告与归因

`vocal_subtitle/quality/golden_gate.py` 和 `scripts/run_offline_golden_production.py` 负责统一输出事件级归因、候选摘要和阶段计数。它们只负责测量和解释，不参与生产决策，不修改 gate 阈值。

### 3.2 分段候选与 local recovery

`vocal_subtitle/application/asr_path.py`、`vocal_subtitle/application/offline_production.py` 和 `vocal_subtitle/asr/local_recovery.py` 负责产生有绝对坐标的候选。recovery window 必须来自 physical coverage 的 bounded `recovery_ranges`，词时间必须正确加回窗口 offset，结果以 `CandidateEvidence(source="local_recovery")` 重新进入 review。

### 3.3 证据、决策与物理投影

`EvidenceReviewService` 负责风险窗口和候选竞争，`EvidenceDecisionEngine` 负责 `keep/replace/split/drop/unresolved`，`DecisionEventProjector` 负责最终事件构造、物理边界、覆盖审计和 trace。任何修复都不能绕过这三个边界。

### 3.4 后处理与导出

后处理只能消费 coordinator/projector 返回的事件。它可以进行显示层的标点、合并和格式化，但不能凭空扩大 speech evidence、跨 hard silence 拼接或创建无 decision trace 的事件。

## 4. 数据流与实现顺序

1. 先执行现有 P0 相关测试和黄金集 runner，保存基线报告。
2. 按报告归因逐类检查：候选是否存在、词时间是否有效、物理覆盖是否完整、投影后是否丢词、后处理是否改变了匹配所需的文本/时间。
3. 每次只修改一个责任边界，并先写回归测试。
4. 对候选缺失问题，优先修复 bounded recovery、chunk offset、候选准入和词去重；不得扩大静音区。
5. 对文本不匹配问题，优先确认引擎输入、语言状态、分段上下文和候选选择；多引擎替换仅在满足现有 evidence contract 时启用。
6. 对时间轴问题，优先修复绝对/相对坐标转换、physical bin 归属和最终化边界；不得放宽跨静音规则。
7. 修复完成后重新运行单元、组件、全量 pytest 和严格黄金集 gate，并保留报告中的归因变化。

## 5. 错误处理与降级

- global evidence 不可用：保留 segmented 主链，记录 `global_evidence_unavailable`；
- review 失败：使用带 trace 的 segmented projected fallback，任务可标记 degraded；
- projector 失败：遵循现有安全 fallback，禁止返回 raw events；
- recovery 失败、无词级时间或无 physical overlap：记录 recovery 失败，不改变物理 speech span；
- 任何候选转换异常：丢弃该候选并记录安全的 `error_category`、route 和版本信息；
- 报告中不得包含 token、密码、完整凭证或未脱敏路径中的敏感信息。

## 6. 测试与验收

### 单元测试

- 候选缺失、文本不匹配和时间偏移归因稳定；
- recovery window 的绝对 offset、边界和无词结果处理正确；
- global/local candidate 的物理准入和 source trace 正确；
- projector 不跨 hard silence、不产生 raw bypass；
- 后处理不破坏 decision trace 和物理覆盖。

### 路径回归

- 默认配置仍为 segmented primary + global evidence；
- global 失败、review 失败、projector fallback 和 recovery 失败均可降级；
- shadow 返回 baseline 但保留完整诊断，authoritative 返回 projector 事件；
- 显式 global 兼容路径仍受 coordinator 管控。

### 黄金集验收

使用同一 `test/quality_manifest.yaml`、相同模型和参数生成三种模式报告。必须同时观察：

- `real_speech_drop_rate`；
- `hallucination_retention_rate`；
- `physical_violation_rate`；
- `cross_silence_rate`；
- `unresolved_rate`；
- `trace_missing_rate`；
- `raw_bypass_count`；
- `miss_attribution_counts` 和每个事件的 trace。

只有 authoritative 结果满足既有 gate 才能进入发布候选；否则发布状态必须继续标记为不可发布，并根据新的归因进入下一轮修复。

## 7. 完成定义

- P0 相关测试和全量 pytest 通过；
- 默认链路、降级链路和 trace 契约没有回退；
- 安全类指标仍满足既有阈值；
- 黄金集报告可以解释剩余漏检，而不是只给总数；
- 质量改进可复现、可回滚，且没有通过放宽 gate 获得的虚假通过。
