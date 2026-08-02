# P0 离线字幕生产链收敛设计

日期：2026-08-02

状态：已确认，待实施

## 1. 目标与范围

本设计落实《离线字幕全链路方案-2026-08-02》的 P0，目标是在不放宽黄金集门禁的前提下，把离线默认链路收敛为：

```text
分段主 ASR
  -> global evidence（可选）
  -> 风险评分与局部复核
  -> EvidenceDecision
  -> DecisionEventProjector
  -> 后处理与导出
```

P0 包含：

- 默认入口从 global primary 切换为 segmented primary。
- global ASR 从直接产出事件改为独立证据来源。
- 分段候选、global 合法候选、局部复核候选统一进入 `EvidenceReviewService`。
- 所有决策都产生 `EvidenceDecision`，authoritative 模式通过 `DecisionEventProjector` 投影。
- 统一记录候选来源、决策、物理覆盖、尾段修复和漏识别归因。
- 修复物理未覆盖区间的局部召回链，不通过扩张静音边界伪造字幕覆盖。
- 以同一黄金集比较 baseline、evidence review 和 authoritative projection 的质量变化。

本 P0 不包含：

- FunASR、Qwen、ForcedAligner、SED 进入 release-default。
- 噪声画像自动调节 VAD 阈值。
- 反馈 profile 自动改变生产参数。
- 流式模式迁移。
- 放宽 `real_speech_drop_rate` 或其他黄金集阈值。

## 2. 当前问题

当前 `PipelineRunMixin.run()` 在 `global_asr.routing` 为 `auto` 或 `global` 时，global 成功后会直接执行后处理和导出，跳过 `OfflineProductionCoordinator`。global evidence 虽然被保存并参与风险评分，但没有进入 `EvidenceDecisionEngine.decide_bundle()` 的替代候选集合。

当前协调器在 shadow 模式下会运行 review，但返回 baseline 事件；authoritative 模式才使用投影事件。该行为保留，但必须明确记录为 shadow 观察，不得报告为已完成物理投影。

最新黄金集结果为 `real_speech_drop_rate=0.116667`，门禁为 `0.05`。因此架构收敛与漏识别归因必须同时进行，不能用切换 global 主链或修改阈值掩盖内容召回问题。

## 3. 设计方案

### 3.1 路由与配置

默认离线配置改为：

```yaml
pipeline:
  asr:
    global_asr:
      enabled: true
      routing: "segmented"
      evidence_enabled: true
  evidence_review:
    enabled: true
    authoritative_mode: false
    shadow_mode: true
```

`routing=segmented` 表示分段 ASR 是主候选来源；`evidence_enabled=true` 表示在分段候选准备完成后，global ASR 可以独立运行并提供证据。新增 `evidence_enabled` 是为了避免把“是否启用 global”与“global 是否直接成为主链”混为一项配置。

显式 `routing=global` 保留为兼容和诊断入口，但它也必须进入统一的 coordinator。该入口不再是默认发布路径，并在诊断中标记 `role=compatibility_global`。

### 3.2 调用顺序

离线运行顺序调整为：

1. 执行早期检测、物理时间线和分段主 ASR。
2. 若 `evidence_enabled`，运行 global ASR；global 失败只记录 `global_evidence` 降级原因，不影响分段主链。
3. 将 global transcript 转换为候选证据，保留全局词时间、来源窗口、引擎、模型和物理重叠信息。
4. 调用 `OfflineProductionCoordinator`，请求同时携带分段事件与 global evidence。
5. `EvidenceReviewService` 执行风险评分、Context Re-ASR、异质副引擎和辅助证据收集。
6. `EvidenceDecisionEngine` 选择 `keep/replace/split/drop/unresolved`。
7. shadow 模式返回分段 baseline，同时保留完整 decision/projection 诊断；authoritative 模式返回 projector 结果。
8. 后处理只消费 coordinator 返回的事件；最终化与导出不得再次绕过决策结果。

### 3.3 Global 候选的安全准入

global 结果分为两种角色：

- `global_signal`：只有风险、覆盖、重复或文本冲突信息，不得替换分段文本。
- `global_alternative`：同时满足以下条件时，才作为决策候选参与 `replace/split`：
  - 有有效的词级时间戳；
  - 候选时间落在音频时长和对应 review window 内；
  - 与分段候选或 physical speech span 有正重叠；
  - 词序、文本和时间字段可序列化为 `CandidateEvidence`；
  - 不跨越已知 physical clip/bin 或 hard silence；
  - 诊断明确记录来源 candidate ID 和准入结果。

不满足条件的 global 结果仍可用于风险评分，但不能替换或删除分段候选。`drop` 继续要求多源非语音证据；`unresolved` 默认保留分段候选。

P0 默认只对高风险或存在文本冲突的窗口开放 `global_alternative` 竞争，低风险候选保持分段结果，避免 global 长上下文错误覆盖稳定基线。

### 3.4 统一决策与投影

`OfflineProductionRequest` 承载：

- segmented baseline events；
- global evidence；
- audio、sample rate、physical timeline；
- input/audio hash；
- ASR route、engine pair 和版本信息。

`EvidenceReviewResult` 承载 decisions 与结构化 diagnostics。`DecisionEventProjector` 是 authoritative 事件的唯一构造器，负责：

- 词时间转为绝对时间；
- 词分配到 physical span/bin；
- 物理覆盖审计；
- 事件来源和 decision trace；
- rejected word 与跨静音诊断。

shadow 仍可调用 projector 生成诊断结果，但必须返回 baseline，并写入：

```json
{
  "production_path": "shadow",
  "physical_projection": {"mode": "shadow_observed"},
  "raw_event_bypass_count": 0,
  "decision_count": 0
}
```

这里的 `raw_event_bypass_count=0` 只表示事件已被 coordinator 接管；`shadow_observed` 不得被解释为 authoritative 投影成功。

### 3.5 物理尾段与召回

物理时间线审计产生的 `recovery_ranges` 是唯一的局部召回输入。召回流程：

1. 读取未覆盖 bin 的开始、结束、physical clip 和原因。
2. 在原音频坐标中创建有限长度的 recovery window。
3. 使用已选主引擎执行局部 ASR，词时间加回 window offset。
4. 将 recovery 结果作为带 `source=local_recovery` 的候选重新进入决策，而不是直接追加字幕事件。
5. 再次执行物理投影和覆盖审计。

恢复失败、没有词级时间或仍无覆盖时，保留结构化 `recovery.status`，不扩大 speech span，不把静音改成语音。

## 4. 错误处理与降级

- global 模型缺失、依赖缺失、超时或识别失败：继续 segmented 主链，诊断为 `global_evidence_unavailable`。
- review 服务失败：按现有 `fallback_to_segmented` 生成 baseline keep decisions，并通过 projector 投影；状态为 `degraded`。
- projector 失败：若配置允许 fallback，返回带诊断的 segmented 投影结果；禁止返回未经决策追踪的原始事件。
- global 候选物理校验失败：降级为 `global_signal`，不得触发替换。
- recovery 失败：保留已有事件和 recovery 诊断，不修改物理证据。
- 所有异常都必须带 `error_category`、安全错误原因和 route/version 信息；不得写入密钥或完整凭证。

## 5. 测试设计

### 单元测试

- 默认配置解析为 segmented primary、global evidence enabled。
- `global_signal` 不进入替代候选；合法 `global_alternative` 可以参与决策。
- 缺少词时间、越过 window、无 physical overlap 的 global 候选被拒绝为 alternative。
- global 失败不阻断 segmented 输出。
- shadow 返回 baseline，但产生 decision/projection 诊断。
- authoritative 使用 projector 结果，事件带 decision trace。
- recovery window 正确应用绝对时间 offset，失败时不伪造物理覆盖。
- 显式 global 兼容路径仍可运行，但不产生 raw bypass。

### 组件与回归测试

- 更新 `tests/test_global_asr_path.py`，验证 auto 默认不再 global short-circuit。
- 扩展 `tests/test_offline_production.py`，验证 global evidence 进入 coordinator。
- 扩展 `tests/test_asr/test_evidence_review.py` 与 `tests/test_secondary_decisions.py`，覆盖来源角色、替代候选和保守 drop/unresolved。
- 扩展物理投影测试，覆盖尾段、跨 bin、词去重、覆盖缺口和诊断完整性。
- 执行全量 pytest，确保已有组件服务、WebUI 和缓存回归不变。

### 黄金集验收

使用同一 `test/quality_manifest.yaml` 和相同模型/依赖，生成三组可比较结果：

1. segmented baseline；
2. segmented + global evidence + shadow diagnostics；
3. segmented + global evidence + authoritative projection。

每组报告必须包含：

- `real_speech_drop_rate`；
- `physical_violation_rate`；
- `cross_silence_rate`；
- `hallucination_retention_rate`；
- `unresolved_rate`；
- `raw_bypass_count`；
- 每个 expected speech event 的归因阶段；
- recovery window、global alternative 准入和最终决策 trace。

只有 authoritative 结果同时满足既有门禁，才允许下一阶段考虑将 authoritative 设为默认。P0 不改变任何门禁阈值。

## 6. 实施顺序

1. 新增配置字段和 route diagnostics，先锁定 segmented primary 行为。
2. 把 global 执行从 primary 分支拆为 evidence stage，并保留兼容入口。
3. 扩展证据合约和 review service，接入经过校验的 global alternatives。
4. 统一 shadow/authoritative 的 coordinator 与 projector 诊断。
5. 收敛 recovery window 到 evidence/decision/projection 链。
6. 更新单元、组件和路径回归测试。
7. 执行全量 pytest 和三组黄金集；根据归因结果再做下一轮 P0 修复。

## 7. 完成标准

- 默认离线运行不再因 global 成功跳过 segmented primary。
- global 失败不会阻断 segmented 生产。
- 最终事件不再由 global raw events 直接导出。
- 每个最终事件都能追溯到 candidate/decision/projection 或明确 degraded fallback。
- 物理违规、跨静音、幻觉保留和 raw bypass 门禁不回退。
- 黄金集报告能定位真实语音漏识别发生在物理候选、ASR、决策、投影、后处理或匹配阶段。
- 未达到 `real_speech_drop_rate <= 0.05` 前，不将结果标为 `gated` 或 `release-default`。
