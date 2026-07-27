# 声学发布门禁与完整边界仲裁设计

**日期：** 2026-07-27  
**状态：** 已获用户批准，待实施

## 1. 目标与发布判定

本任务补齐影视级发布声明所需的四项证据：

1. 可版本化的人工词级声学金标准和评估器；
2. `BoundaryArbiter` 对 RMS、VAD、FFmpeg、局部噪声和 ASR 证据的逐候选仲裁；
3. 修复后 8 个真实媒体的完整回归；
4. 产物、代码改动和测试结果的可追溯整理。

代码测试通过不等于发布通过。只有人工 gold 指标和 8 个真实媒体回归都满足门禁，最终报告才能写“可发布”；否则必须列出剩余阻塞项。

## 2. 数据契约

`acoustic-gold-v1` 使用同一词级 JSON 结构承载人工真值和预测结果。顶层包含：

- `schema_version`：固定为 `acoustic-gold-v1`；
- `media_id`、`audio_path`、`sample_rate` 和标注来源等媒体元数据；
- 非空 `words` 数组。

每个词必须包含稳定且唯一的 `id`、非空 `text`、非空 `speaker`、有限非负的 `onset` 和 `offset`，并满足 `onset < offset`。允许额外保留 `reviewer`、`confidence`、`notes`、`source`、原始时间和 `revision_trace`，但评估器只使用稳定词 ID、首尾时间和 speaker。

评估器必须拒绝缺失版本、重复 ID、非法时间、空词数组和无法匹配任何词 ID 的输入。它输出 median/P95 首点、尾点和整体边界误差、切头/切尾率、speaker error、缺词/额外词及结构化门禁原因。

CLI 支持单媒体评估、报告写入和 `--ci` 模式。默认门禁为整体边界 median `<=80ms`、P95 `<=160ms`、切头和切尾率各不超过 `0.5%`；缺词直接不通过。默认阈值可显式覆盖，报告必须记录实际阈值。

## 3. 边界仲裁架构

`BoundaryArbiter` 是唯一有权接受词级物理端点的组件。流程为：

`PhysicalTimeline` 证据准备 -> 构造局部候选 -> 硬约束过滤 -> 特征归一化与评分 -> 确定性选择 -> `BoundaryDecision`。

每个 `BoundaryCandidate` 在搜索窗及所属 `PhysicalSubtitleBin` 内计算并记录：

- ASR 原始时间和校准置信度；
- RMS 能量、梯度、局部谷底距离；
- VAD 人声概率或状态；
- FFmpeg 静音/语音边界距离及来源状态；
- `LocalNoiseProfile` 噪声等级、局部阈值和稳定性；
- evidence IDs、特征缺失和硬约束拒绝原因。

候选必须先通过 PhysicalClip、PhysicalSubtitleBin、speaker/hard split、相邻词单调性、正时长和已确认人声覆盖等硬约束；高分非法候选永远不能获选。通过候选才进入归一化评分，不能建立永久的单信号优先级。起点和终点可以使用不同损失，但必须保持确定性同分排序。

`BoundaryDecision` 保存接受状态、端点、边界类型、置信度、证据 IDs、reason codes、每个候选的分项/总分、拒绝候选和降级信息。没有合法候选时沿用受约束的 ASR/片段降级结果并写入结构化原因，不伪造词级精度。

## 4. Pipeline 接线

Global 词级路径固定为：

`global ASR -> physical bin 分配 -> 候选生成 -> BoundaryArbiter -> WordAllocation -> SubtitleEvent`。

ASR 时间是观测值；VAD、FFmpeg、RMS 和局部噪声是候选锚点及合法范围。特征计算限制在词端点搜索窗和所属 bin，复用 Pipeline 已生成的绝对时间证据与缓存。信号缺失时降低置信度并记录 warning，不把缺失信号解释为通过。

末词不得无条件延长到 VAD 或 bin 末端。无词级时间或无可接受候选时必须标记 `timing_degraded`。BoundaryDecision 和候选诊断写入 allocation/event 的 revision provenance，供评估器和真实回归报告读取。

## 5. 验证与真实媒体回归

验证分三层：

1. 单元测试覆盖 schema、指标、硬约束、四源特征、确定性排序和降级原因；
2. 集成测试使用合成音频确认 RMS/VAD/FFmpeg/局部噪声逐候选注入，并验证非法高分候选不会获选；
3. 修复后完整运行 8 个真实媒体，记录代码版本、配置、模型、实际 ASR path、运行时间、输出位置和失败原因。

旧 dogfood 输出只作历史对照，不能混入新报告。成功重跑后更新对应结果；未能重跑的样本标记为 `stale` 或 `not_run`，不能被计作通过。人工 gold 未提供或指标未达标时，发布状态必须保持阻塞。

## 6. 产物与 Git 边界

保留本轮相关的 `audio_utils.py` 和测试改动。只清理已确认属于旧回归的重复生成物，不删除未知用途或用户代码。完成验证后只暂存本轮相关文件，创建一个说明声学门禁、完整仲裁和真实媒体回归状态的提交；无关工作区改动不得被覆盖或混入。

## 7. 验收标准

- `acoustic-gold-v1` schema、评估器和测试可独立运行；
- 每个接受或降级的词级端点都有可审计 `BoundaryDecision`；
- 每个真实候选都能追溯四源声学特征或明确记录缺失；
- 完整测试、静态检查和 8 媒体回归结果可复现；
- 最终报告不会把未完成的人工 gold 或 stale dogfood 误报为发布通过。
