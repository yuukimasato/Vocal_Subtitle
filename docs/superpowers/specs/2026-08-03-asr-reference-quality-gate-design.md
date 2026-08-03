# ASR 参考质量与离线安全门禁设计

日期：2026-08-03  
状态：已确认，待实施  
范围：`test/` 音频回放、ASR 召回诊断、黄金质量报告与 CI 门禁

## 1. 背景与目标

当前离线生产 runner 使用 `test/quality_manifest.yaml` 的人工字幕生成 `expected_events`，并由 `real_speech_drop_rate` 参与默认 `--ci` 发布判定。这会把人工字幕隐含为绝对真值，无法表达人工字幕可能包含的删减、风格偏好、时间取舍或结构改写。

本阶段采用双轨评估：

- `test/` 下的音频是离线生产回放集，必须继续真实运行并记录稳定性、资源、链路和诊断结果；
- 人工字幕是 `reference/advisory`，用于衡量“接近人工识别质量”的文本、时间和事件覆盖，并定位 ASR 召回问题；
- 默认 CI 门禁只阻断安全与链路契约问题，不因人工字幕相似度不足单独阻断；
- 研发可通过显式 `--strict-reference` 运行参考质量门禁，但该模式不是默认生产发布门禁。

目标不是降低 ASR 质量要求，而是把“结果是否安全可发布”和“结果是否接近当前人工参考”拆成两个可解释状态，继续用参考差异驱动召回改进。

## 2. 范围与非目标

包含：

- manifest 中声明参考文件的角色和评估策略；
- golden gate 输出 `safety_gate` 与 `reference_quality` 两组指标；
- 默认 `--ci`、显式 `--strict-reference` 和报告状态的语义固定；
- 保留逐事件参考匹配、漏检阶段归因和 ASR/物理覆盖诊断；
- 对 ASR 候选生成、局部恢复、时间覆盖和语言策略增加可观测性与回归测试。

不包含：

- 修改人工字幕文件或把人工字幕重新命名为绝对真值；
- 放宽物理边界、跨静音、幻觉、raw bypass、决策追踪和诊断完整性要求；
- 修改 ASR 模型默认版本、黄金集阈值以制造通过结果；
- 用参考质量通过替代真实人工抽检；
- 以无参考音频伪造文本准确率。

## 3. 数据契约

### 3.1 Manifest

现有 `ground_truth` 字段继续兼容，但新增可选字段：

```yaml
reference_role: advisory
gate_profile: safety
```

缺省规则：有 `ground_truth` 时 `reference_role=advisory`，没有参考时为 `none`；`gate_profile` 缺省为 `safety`。未来若有经过独立审核、明确授权的严格数据集，可显式声明 `reference_role=strict`，但不能由旧 manifest 自动升级。

### 3.2 Runner 输入

`run_offline_golden_production.py` 保留参考事件和 `miss_attribution`，同时在每个 case 写入：

- `reference_role`；
- `gate_profile`；
- `reference_status`；
- `audio`、引擎/模型、生产模式和路由版本；
- 候选、决策、投影、覆盖和降级诊断。

参考文件只影响 `reference_quality` 计算，不影响 production events 的生成。

## 4. Gate 语义

### 4.1 安全门禁

默认 `--ci` 检查以下硬约束：

- 幻觉/非语音参考场景的保留率；
- 物理越界率；
- 跨静音率；
- unresolved、决策追踪缺失和 raw bypass；
- 诊断完整性；
- 必要时的运行失败、空输出和报告 schema 完整性。

这些约束不是人工字幕相似度，仍然必须通过。安全门禁失败时 CI 返回非零，并保持不可发布状态。

### 4.2 参考质量

有人工参考的 case 继续计算：

- event recall / real speech drop rate；
- 文本相似度与文本不匹配；
- 时间重叠、边界偏移；
- `speech_candidate_missing`、`asr_text_mismatch`、`timeline_shift_or_boundary_mismatch`、`physical_coverage_or_recovery` 等阶段归因。

这些指标写入 `reference_quality`，单独展示 `reference_case_count`、`advisory_case_count` 和按场景/类别的分布。默认安全门禁的 `passed` 不使用 `max_real_speech_drop_rate`；阈值保留用于 `--strict-reference` 或趋势报警。

### 4.3 CLI 与报告状态

- 默认运行：`run_golden_quality_gate.py --ci`，结果由 `safety_gate.passed` 决定；
- 严格研发运行：`run_golden_quality_gate.py --ci --strict-reference`，在安全通过后再要求参考质量阈值；
- 报告顶层提供 `gate_mode`、`safety_gate`、`reference_quality`、`release_status`；
- `release_status=blocked` 只由安全门禁或运行契约失败触发；安全通过但参考质量较弱时为 `reference_improvement_required`，不得伪造为高质量通过；
- 参考质量不通过时必须在报告和控制台明确显示，不能静默忽略。

## 5. ASR 召回改进路径

召回修复使用参考质量作为诊断信号，按以下顺序收敛：

1. 以候选事件、global evidence、local recovery、decision 和 projection 的 trace 对齐每个参考事件；
2. 优先修复 `speech_candidate_missing`：检查分段边界、局部 recovery window、语言提示和候选过滤；
3. 再修复 `asr_text_mismatch`：比较 primary/secondary/global 候选，禁止跨 physical bin 或跨静音拼接掩盖问题；
4. 最后修复时间边界和投影覆盖：保留绝对坐标、recovery offset、physical bin 和 coverage audit；
5. 每次修复同时运行安全门禁和参考质量报告，确认没有以召回提升换来幻觉、物理违规或 trace 缺失。

任何候选只有通过现有 evidence → decision → projection 链才能进入 authoritative 输出，不能为提高参考 recall 直接追加 raw subtitle event。

## 6. 测试与验收

- manifest 角色缺省与显式字段测试；
- 默认 gate 忽略参考 recall 阈值、严格模式启用参考阈值的单元测试；
- 安全约束失败仍阻断默认 CI 的测试；
- 报告同时包含安全指标、参考质量指标和漏检归因；
- runner 使用 `test/` 音频生成输入，参考字幕只标记为 advisory；
- ASR 召回和局部恢复回归测试保持 physical/trace/raw bypass 硬约束；
- 全量 pytest、compileall、JavaScript 语法和 `git diff --check`；
- 在可用模型环境执行同一 manifest 的离线回放，并记录真实模型、设备和耗时。

验收条件：人工字幕不再是默认 CI 的硬基线，但报告仍能清晰回答“距离人工参考还有哪些漏检”；任何安全或链路契约问题仍会阻断默认门禁。
