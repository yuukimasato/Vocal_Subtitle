# P0 离线字幕生产链收敛实施计划

依据：[P0 设计文档](../specs/2026-08-02-p0-offline-production-convergence-design.md)

## 执行状态（2026-08-02）

代码实施已完成，发布质量门禁未通过，因此状态为“P0 架构收敛完成，禁止发布”。

| 阶段 | 状态 | 落实结果 |
|---|---|---|
| 1. 配置与路由 | 已完成 | 默认使用 segmented primary；global 在主链完成后作为 evidence 执行。 |
| 2. Global evidence 合约 | 已完成 | 有效词时间、窗口和物理范围校验后的 global 候选才可作为高风险窗口的替代项。 |
| 3. Coordinator/projector 诊断 | 已完成 | shadow 记录 `shadow_observed`，authoritative 和降级路径均产出决策与投影追踪。 |
| 4. 物理尾段与局部召回 | 已完成 | 由 projector coverage 的 `recovery_ranges` 驱动；LocalRecovery 候选以 `local_recovery` 证据重进 review/decision/projector，不再由 global 路径直接追加事件。 |
| 5. 黄金集归因与门禁 | 已完成 | runner 支持 baseline、shadow、authoritative 三种可比较模式，输出漏识别归因和生产模式元数据。 |

本轮已完成真实的 10 场景、`faster-whisper large-v3` 三模式验收，并执行 `.venv-production/bin/python -m pytest -q`（`1014 passed, 1 skipped`）。baseline 与 shadow 的 `real_speech_drop_rate` 均为 `0.125`；baseline 按设计不具备 review diagnostics，shadow 的诊断、物理违规、跨静音、trace 和 raw bypass 均通过。authoritative 在修复语言状态污染及物理 bin 跨界合并后，从 `0.175` 降至 `0.100`，但仍高于 `0.05` 发布线；物理违规、跨静音、幻觉保留、unresolved、trace、raw bypass 和诊断完整性均通过。

authoritative 未匹配归因：`asr_text_mismatch=8`、`speech_candidate_missing=3`、`timeline_shift_or_boundary_mismatch=1`。剩余问题是候选文本/召回质量，而非门禁阈值或 projector 契约问题；不得通过放宽门禁或把跨物理 bin 的片段拼接为一次匹配来标记发布。

本地 Qwen3-ASR-1.7B smoke 可加载，但对重复短语样本只返回无词级时间的文本结果；它不满足 P0 的替代候选时间契约，因此黄金集 runner 继续将 Qwen 保持为禁用的替换能力，不能以其文本结果绕过 physical projection。

## 实施原则

- 保留工作区已有用户改动，不做回滚或无关重构。
- 先锁定主链和证据契约，再启用替代候选，避免一次性改变所有决策。
- 每个阶段先补回归测试，再改实现；global 失败必须可降级到 segmented。
- 不修改黄金集门禁阈值，不把未通过结果标记为 `gated` 或 `release-default`。

## 阶段 1：配置与路由收敛

目标：默认离线任务执行 segmented primary，global 只作为可选 evidence。

文件：

- `vocal_subtitle/config/models.py`
- `vocal_subtitle/config/loader.py`
- `configs/default.yaml`
- `vocal_subtitle/application/pipeline_runner.py`
- `vocal_subtitle/application/asr_path.py`
- `tests/test_production_defaults.py`
- `tests/test_global_asr_path.py`

动作：

1. 为 `GlobalASRConfig` 增加 `evidence_enabled`，兼容旧配置默认值。
2. 默认 YAML 使用 `routing: segmented` 和 `evidence_enabled: true`。
3. 将 global primary 分支改为显式兼容路径；默认 `auto` 不再在 global 成功后短路 segmented。
4. 在 segmented 候选完成、物理 timeline 可用后调用 global evidence stage。
5. global evidence 异常只写入结构化诊断，不能阻断 segmented 生产。
6. 更新缓存可用性判断与 stats 的 `asr_path`/role 字段，避免旧 global 缓存误当 segmented 结果。

验证：默认配置断言、auto 不再 short-circuit、explicit global 兼容路径、global 失败降级测试。

## 阶段 2：Global evidence 合约与候选准入

目标：global 结果可作为风险信号，并在严格条件下成为替代候选。

文件：

- `vocal_subtitle/asr/evidence.py`
- `vocal_subtitle/asr/evidence_review.py`
- `vocal_subtitle/asr/evidence_decision.py`
- `vocal_subtitle/application/offline_production.py`
- `tests/test_asr/test_evidence_review.py`
- `tests/test_secondary_decisions.py`
- `tests/test_offline_production.py`

动作：

1. 为 CandidateEvidence 增加或复用 diagnostics 中的 evidence role，不扩张 source 枚举。
2. 新增 global candidate 校验：词时间、音频范围、window 归属、physical overlap、hard boundary。
3. 将合法 global alternatives 按 review window 传入 `decide_bundle()`；不合法项只进入风险诊断。
4. 限制 global alternative 只参与高风险/文本冲突窗口，保持低风险 segmented keep。
5. 保留 multi-source drop 和 unresolved 保留主候选的现有规则。
6. 在 diagnostics 中记录 global candidate 总数、signal 数、alternative 数、拒绝原因和来源 ID。

验证：合法/非法候选、跨物理边界、无词时间、无重叠、global-only failure、保守 drop/unresolved。

## 阶段 3：统一 coordinator/projector 诊断

目标：shadow 和 authoritative 都有一致的决策与投影可观测性，最终事件不再 raw bypass。

文件：

- `vocal_subtitle/application/offline_production.py`
- `vocal_subtitle/physical/decision_projection.py`
- `vocal_subtitle/application/asr_path.py`
- `vocal_subtitle/application/pipeline_result.py`
- `tests/test_offline_production.py`
- `tests/test_physical/test_phase_three.py`
- `tests/test_physical/test_evidence_adapter.py`

动作：

1. shadow 模式也执行 projector 诊断，但返回 baseline，并将 projection mode 明确为 `shadow_observed`。
2. authoritative 模式只返回 projector events；review/projector 异常统一进入 segmented projected fallback。
3. 为每个结果补全 `production_path`、`review_status`、`decision_count`、projection diagnostics、route/version。
4. 将 global compatibility path 也包装到 coordinator，禁止无 trace 事件直接进入后处理。
5. 保证 source word ID、decision ID、physical region 和 revision trace 在 projection 后保留。

验证：shadow baseline、authoritative projection、review failure、projector failure、diagnostic completeness、raw bypass。

## 阶段 4：物理尾段与局部召回

目标：物理未覆盖区间进入统一 evidence/decision/projector 链，不直接追加字幕。

文件：

- `vocal_subtitle/application/asr_path.py`
- `vocal_subtitle/asr/local_recovery.py`
- `vocal_subtitle/physical/coverage.py`
- `vocal_subtitle/physical/decision_projection.py`
- `tests/test_physical/test_coverage_recovery.py`
- `tests/test_global_asr_path.py`

动作：

1. 将 coverage `recovery_ranges` 映射为带绝对坐标的 bounded recovery windows。
2. 局部 ASR 结果转换为 `local_recovery` CandidateEvidence，应用 offset 后进入 review。
3. 删除/旁路现有直接 append SubtitleEvent 的 recovery 行为，或将其限制为兼容诊断而非输出。
4. 重新执行 decisions → projector → coverage audit，记录恢复前后覆盖状态。
5. recovery 失败或无词时间时保留降级诊断，不扩大 physical speech evidence。

验证：尾段 offset、局部恢复、恢复失败、静音区拒绝、二次覆盖审计。

## 阶段 5：黄金集归因与门禁

文件：

- `vocal_subtitle/quality/golden_gate.py`
- `scripts/run_offline_golden_production.py`
- `scripts/run_golden_quality_gate.py`
- `tests/test_golden_quality_gate.py`
- `tests/test_offline_golden_production.py`
- `tests/test_calibrate_quality_gate.py`

动作：

1. 为 normalized case 增加可选的 miss attribution 字段和阶段计数，不改变既有指标含义。
2. 让 production runner 生成 baseline、shadow evidence、authoritative projection 三种可比较输入。
3. 使用同一 manifest、模型和参数运行全量黄金集。
4. 先执行全量 pytest，再执行黄金集脚本和 `--ci` gate。
5. 输出每个漏检事件的阶段归因，确认下一轮修复点。

## 完成条件

- 所有阶段测试通过，且全量 pytest 无新增失败。
- 默认路径为 segmented primary + global evidence，global 成功不能短路 segmented。
- global 失败、review 失败、projector 失败都有可追踪降级。
- authoritative 事件无 raw bypass，shadow 明确标记为观察模式。
- 黄金集报告包含三组对比和漏识别归因。
- 若 `real_speech_drop_rate` 仍大于 `0.05`，保持发布状态为不可发布，并据归因结果继续后续 P0 修复。
