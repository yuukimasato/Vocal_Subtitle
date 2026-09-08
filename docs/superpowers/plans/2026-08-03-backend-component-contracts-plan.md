# Vocal Subtitle 后端组件契约实施计划

依据：[后端组件契约设计](../specs/2026-08-03-backend-component-contracts-design.md)

## 实施约束

- 采用兼容优先、非破坏式演进；不替换现有 CLI、WebUI、缓存、任务历史和 `Pipeline.run()`。
- 契约层只使用标准库类型和项目内已有结果对象，不引入 FastAPI、Click 或模型 SDK 依赖。
- 旧对象通过 adapter 接入；adapter 不复制字幕、ASR、黄金集或决策业务规则。
- 每一组迁移先通过定向测试，再运行全量测试和编译检查。
- 不改变 ASR 召回、质量阈值、黄金集门禁和发布结论。

## 步骤

1. 建立 `vocal_subtitle/contracts/`，定义版本、错误、任务、运行、引擎、报告和外部边界数据对象，并保证 JSON round-trip。
2. 定义 `TaskPort`、`PipelineRunPort`、`EngineRegistryPort`、`ReportPort`、`ArtifactPort`、证据复核和决策投影 Protocol，明确依赖方向。
3. 实现 `TaskHistoryManager`、`Pipeline.run()`、`EngineRegistry` 和 `RunReportBuilder` 的兼容 adapter；保留旧字段、状态和序列化行为。
4. 实现只依赖 Protocol 的运行协调器，覆盖正常、降级、失败、取消和报告失败路径；不接入旧 CLI/WebUI 调用路径。
5. 增加契约 round-trip、端口 fake、adapter 映射、错误归一化和版本兼容测试。
6. 执行定向测试、全量 pytest、compileall 和 `git diff --check`，更新后端契约实施报告。

## 验收

- 新契约可表达任务、运行、引擎结果、错误、报告和产物，不携带凭证或完整异常堆栈。
- 旧任务历史 payload、`PipelineStats`、报告 JSON 和 `Pipeline.run()` 返回值均可继续读取。
- 协调器只依赖 Protocol，所有状态变化经 `TaskPort`，失败不会覆盖已有主字幕产物。
- 现有全量测试无新增失败，新增测试覆盖协议成功/降级/失败/取消分支。
