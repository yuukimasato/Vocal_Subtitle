# 离线字幕全链路收敛设计

日期：2026-08-01
状态：已确认设计，待实施
范围：离线高质量字幕生产链；流式链路不在本次范围内。

## 1. 目标

以当前工作区代码为基线，把已有的物理证据、分段 ASR、global evidence、风险复核、EvidenceDecision、物理投影和字幕后处理收敛为一条可追踪、可降级、可验收的生产链。

本次工作的成功标准是：

- 只有分段主候选经过统一决策和物理投影后才能进入最终字幕。
- global、Qwen、SED、ForcedAligner 和 LLM 只能提供证据或建议，不能绕过决策层直接导出。
- 所有时间坐标、候选来源、模型状态、失败原因和降级结果可诊断。
- 模型不可用时保留主候选并结构化报告，不把组件接线误报成质量通过。
- faster-whisper 仍是未完成真实黄金集准入前的发布基线。

## 2. 现状与约束

工作区已有一批未提交的离线多引擎链路代码、测试和报告。本次实现必须在其上增量完成，不回滚、不覆盖与本任务无关的用户改动。

现有方案定义了 R0-R8 阶段：测量修正、生产基线、时间契约、ASR 粒度、物理尾段、Qwen 复核、FunASR 准入、自适应证据和发布治理。本设计沿用该阶段划分，但优先修复可由本地测试验证的契约和降级行为；生产模型缺失时只生成明确的 unavailable 诊断。

不在本次范围内：

- 重新设计流式识别链路。
- 让 LLM 生成时间戳、充当声学判定器或单独删除真实语音。
- 通过放宽门禁、静默 fallback 或复用旧缓存制造通过结论。
- 在真实黄金集通过前把 FunASR 或 Qwen 设为默认发布引擎。

## 3. 架构与数据流

```text
输入/音频准备
  -> 物理证据与 PhysicalTimeline
  -> 宏观切块与分段主 ASR
  -> global evidence
  -> 风险评分与 Context Re-ASR
  -> risk_only 或 full_quality 异质复核
  -> EvidenceDecision
  -> DecisionEventProjector
  -> 说话人/确定性字幕归并/可选语义建议
  -> 声学门禁与 EndTime 校验
  -> final validator 与 SRT/VTT/ASS 导出
```

### 3.1 职责边界

- 物理层描述语音概率、能量和合法时间范围，不产生字幕文字。
- 识别层产生主候选、词时间、置信度和复核证据，不直接导出。
- 决策层根据多源证据选择 `keep/replace/split/drop/unresolved`。
- 展示层只组织已投影的合法事件，不扩大未经校验的时间范围。

### 3.2 时间契约

系统明确区分窗口相对时间、全局词时间、物理范围和展示事件时间。每个 offset 只应用一次；跨窗口去重使用 source word ID、时间和文本联合判断。缺少词级时间的候选必须标记 `timing_degraded`，不得伪造强对齐。

### 3.3 唯一决策出口

主候选和所有证据统一进入 `EvidenceDecision`。`unresolved` 默认保留主候选并写入诊断；`drop` 必须有多源支持，单一 LLM、SED、CPS 或低置信度不能单独删除真实语音。任何无法物理投影的替代结果都回退到主候选并记录失败原因，禁止 raw event bypass。

## 4. 引擎组合与配置

- 生产基线：faster-whisper large-v3。
- FunASR：仅作为中文候选主引擎；质量门禁失败时最多一次性回退 faster-whisper，并记录最终引擎和原因。
- Qwen：先 shadow，后仅针对 Context Re-ASR 后仍为 high/critical 的窗口启用 risk-only；默认不开启全量复核。
- global ASR：只写 evidence，不直接导出字幕。
- `full_quality`：只用于生产模型验收、高价值任务和对比，不作为普通任务默认档位。

所有引擎端口返回结构化 `ok/unavailable/failed/timeout` 状态，配对服务只负责选择和校验，不负责加载模型。默认路径、显式路径和无效路径必须具有可解释诊断。

## 5. 实施顺序

1. R0：修正黄金门禁统计，覆盖五类决策，补齐模型/路由元数据和 unavailable 诊断。
2. R1：按生产配对生成可比较的模型、物理投影和字幕质量报告；报告不自动等于发布通过。
3. R2：统一 offset、window、global word、physical span 和 display event 契约，修复 skeleton/VAD 重复职责。
4. R3：保留词级证据，新增确定性词/片段归并；尊重句末、speaker、硬静音、CPS、时长和字符限制。
5. R4：合并尾段物理证据，局部宽松复检最多一次，传播投影拒绝和 fallback 原因。
6. R5：完成 Qwen shadow/risk-only 调度、缓存、超时、并发上限和可回放 evidence bundle。
7. R6：完成 FunASR 中文质量门禁、一次性回退和语言不匹配保护；未达准入不改变发布默认。
8. R7：仅在场景化数据证明收益后启用自适应声学、ForcedAligner、SED 和语义证据。
9. R8：收敛为 quality-first、full-quality、budget 三个用户可理解档位，更新文档中的实现/准入状态。

如果前一阶段的门禁或回归失败，保留上一阶段行为，不继续打开下一阶段默认开关。

## 6. 错误处理与可观测性

每次运行至少记录：输入哈希、route version、policy version、主/副引擎和模型、模型路径摘要、语言、窗口、决策动作、物理范围、source trace、复核状态、缓存命中、耗时和 fallback 原因。

模型缺失、超时、语言不匹配、重复识别、物理越界和投影失败均是结构化诊断，不通过异常文本推断模型成功。发生复核失败时保留主候选；发生物理投影失败时回退主候选；发生最终校验失败时阻止不合法事件导出。

## 7. 测试与验收

按以下顺序执行：

```text
黄金门禁单测
  -> EvidenceDecision/投影/时间契约组件测试
  -> fake engine 路由、回退、超时和 unavailable 测试
  -> 现有离线/映射/物理/字幕回归
  -> 生产模型 smoke（环境允许时）
  -> 生产配对黄金集与三类报告
```

必须覆盖：五类决策分母、空集合、无效 Qwen 路径、一次性 FunASR 回退、重复 offset、跨窗口去重、无词级时间降级、尾段覆盖、短回应保护、speaker/硬静音边界、raw bypass 为零和 `unresolved` 保留主候选。

发布门禁沿用方案文档：`real_speech_drop_rate <= 0.05`、`physical_violation_rate == 0`、`unresolved_rate <= 0.30`、`hallucination_retention_rate == 0`、`raw_bypass_count == 0`，且报告元数据完整。门禁未通过时发布状态保持不可发布。

## 8. 交付物

- 仅针对上述链路的代码和测试增量。
- 生产模型可用时的模型/路由、物理/投影、字幕质量三类报告；不可用时的结构化诊断。
- 配置档位和文档状态更新。
- 测试命令、结果、已知限制和回滚说明。
