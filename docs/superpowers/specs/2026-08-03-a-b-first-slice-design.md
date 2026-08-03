# Vocal Subtitle A+B 第一阶段产品闭环设计

日期：2026-08-03  
状态：已确认，待实施  
范围：A 可运行的完整产品闭环 + B 离线生产质量共同基础与第一阶段可验收切片

## 1. 目标

本阶段把现有离线字幕能力收敛成一条可操作、可追溯、可降级的产品闭环：

```text
上传音频
  → 创建任务 / 预检
  → segmented 主 ASR + global evidence
  → EvidenceDecision + DecisionEventProjector
  → 字幕后处理与导出
  → run_report / 任务历史
  → WebUI 审核、保存修订
  → 反馈差异预览（显式确认后才写入反馈配置）
```

本阶段不扩张识别模型能力，不把 Qwen、FunASR、ForcedAligner、SED、LLM 或反馈 profile 设为默认主链，不改变黄金质量门禁阈值，也不迁移流式链路。

## 2. 第一阶段验收切片

必须能够在本地无外部服务的条件下完成以下操作：

1. WebUI 选择音频和配置，提交离线任务。
2. 任务经过 `pending → preflight → running`，页面展示阶段进度与可恢复降级。
3. 主链完成后产出字幕文件、任务历史记录和 `run_report-v1` 报告。
4. 历史任务可进入审核工作区，加载音频与字幕，编辑并保存单条或批量修改。
5. 当前编辑结果可进入反馈预览，展示差异、覆盖率和风险；只有用户显式确认才写入反馈候选集，不自动消费到生产配置。
6. 无报告文件时，质量工作区可从历史结果生成只读 `history_summary`，并明确标识报告来源。
7. 运行主引擎缺失或可选复核失败时，任务仍能按契约降级完成或以结构化错误失败，不产生无追踪的原始事件。
8. 使用现有黄金样本脚本比较 segmented baseline、shadow evidence 和 authoritative projection，报告漏识别归因；门禁失败时保持不可发布。

## 3. 后端组件边界

### 3.1 Task/Run Coordinator

`TaskPort` 是任务状态唯一写入口。协调器负责创建任务、预检、启动运行、转换终态、登记产物和汇总诊断。`Pipeline.run()`、旧 WebUI route 和 CLI 入口保留为兼容 façade，由 adapter 转换到 `TaskRequest`、`RunRequest` 和 `RunResult`。

任务状态沿用 `task-state-v1`：`pending`、`preflight`、`running`、`completed`、`degraded_completed`、`failed`、`cancelled`。状态转换不由 ASR、报告或前端组件直接执行。

### 3.2 Offline Production

离线主链默认使用 segmented ASR 作为主候选。global ASR 只作为独立 evidence stage；global 失败、模型缺失或超时不能阻断 segmented 主链。

所有可影响最终事件的候选必须经过：

```text
CandidateEvidence → EvidenceReviewService → EvidenceDecision → DecisionEventProjector
```

global candidate 只有同时具备有效词级时间、音频范围、窗口归属、物理重叠且不跨硬边界时，才允许作为 `global_alternative` 参与 `replace/split`；否则只能作为 `global_signal` 进入诊断和风险评分。低风险窗口保持 segmented baseline，`drop` 继续要求多源非语音证据，证据不足默认 `unresolved` 并保留主候选。

物理覆盖产生的 `recovery_ranges` 是局部召回的唯一输入。局部结果必须作为 `local_recovery` evidence 重新进入决策和投影，不直接 append 最终字幕，也不通过扩张 speech span 伪造覆盖。

### 3.3 Report/Artifact

每次运行都应关联 `task_id`、`run_id`、配置快照、引擎可用性、pipeline path、阶段耗时、质量/声学、证据/决策/投影诊断、降级事件、错误和输出产物。报告或附属产物写入失败不能覆盖已经生成的字幕；错误写入结构化诊断。

报告目录沿用 `cache/reports/{run_id}/`。WebUI 质量接口优先读取 `run_report.json`，缺失时返回历史摘要，不修改任务历史或报告文件。

## 4. WebUI 工作区设计

继续使用现有静态 HTML/CSS/JavaScript 与 FastAPI，不引入 React、Vite 或新的构建链。保留现有 DOM id、`App` 方法、请求响应字段和路由。

### 4.1 两栏生产布局

保留现有两栏模式：

- 左栏：上传区域、场景模板、高级配置、引擎与降级选项、运行入口、历史快捷入口。
- 右栏从上到下：
  1. 处理信息：任务状态、阶段进度、运行标识和降级提示；
  2. 操作面：字幕审核、质量报告、反馈预览/确认入口；
  3. 音频波形：浏览器端 Web Audio API 解码，HTML5 Canvas 绘制可缩放波形；
  4. 字幕时间轴：与波形共享绝对秒坐标，显示可编辑字幕 cue、当前播放位置和质量标记。

桌面端右侧各区域独立滚动，页面不出现横向溢出。移动端按“上传与配置 → 处理信息 → 操作面 → 波形 → 字幕时间轴”纵向排列，字幕列表和波形内部滚动。

### 4.2 波形与时间轴契约

- Web Audio API 只负责浏览器预览解码，不改变后端音频、VAD 或字幕结果。
- Canvas 使用音频 buffer 的峰值采样绘制波形，绘制宽度由容器尺寸决定，缩放只改变显示采样密度。
- 波形与字幕时间轴统一使用音频绝对秒坐标；任何窗口相对时间必须在服务端转换后再传给前端。
- 播放头移动时高亮对应 cue；点击 cue 跳转播放器；拖动或编辑边界必须调用现有字幕编辑 API，并遵守服务端物理边界校验。
- 音频解码失败、音频未提供或浏览器不支持 Web Audio 时，保留字幕审核能力并显示明确的波形降级状态。

### 4.3 其他工作区

- 审核：任务选择、播放定位、字幕编辑、批量说话人/合并、保存和导出。
- 报告：任务状态和质量状态分开显示，呈现阶段、引擎、质量/声学、证据/决策/投影、降级、错误和产物。
- 反馈：修订来源、差异预览、学习确认、健康趋势与审核队列分开呈现；预览失败保留已有修订。
- 历史：状态筛选、分页、详情、审核入口、质量入口、产物下载和显式删除。

## 5. 错误与降级

统一使用结构化错误：类别、代码、用户消息、阶段、引擎、是否可重试、是否可恢复和安全诊断。不得向外部 payload 写入 API key、HF token 或完整 traceback。

- 输入缺失、格式不支持、无可用 VAD、无可用 ASR、磁盘不可写：任务失败。
- 可选分离、复核、global evidence、后处理或报告组件不可用：记录降级事件，能保留有效主链时进入 `degraded_completed`。
- WebUI 404 显示任务已清理；报告缺失回退 `history_summary`；编辑或反馈请求失败时保留用户已有内容。
- 取消只允许通过任务协调器执行，运行中不得留下未归类的 `running` 任务。

## 6. 测试与验收

### 自动化

- 契约 round-trip：旧历史、旧报告、`Pipeline.run()` 返回值与新契约双向转换。
- 协调器 fake port：成功、降级、预检失败、取消、产物缺失、报告失败。
- P0 主链：global evidence 失败降级、alternative 准入、shadow/authoritative projector、recovery offset 和 raw bypass。
- WebUI：上传/任务状态/历史/字幕编辑/质量报告回退/反馈预览与确认边界。
- `compileall`、JavaScript syntax check、结构门禁、`git diff --check`。

### 端到端与质量

- 本地 WebUI smoke 覆盖桌面与 390px 移动视口，验证两栏布局、工作区操作、波形空/成功/失败状态、字幕时间轴不重叠。
- 黄金样本运行使用同一 manifest 比较 baseline、shadow 和 authoritative，必须输出漏识别归因、物理违规、跨静音、幻觉保留、unresolved、raw bypass 和决策追踪。
- `real_speech_drop_rate > 0.05` 时保持“不可发布”，不得通过修改阈值或绕过 projector 改写结论。

## 7. 后续衔接

本阶段完成后，再进入总计划的 C 阶段，逐步收敛质量运营、反馈审核、D1/D3 数据资产、实验启用和发布治理。反馈 profile 在后续明确评估和授权前不自动改变默认生产行为。
