# 后端组件契约实施报告

日期：2026-08-03  
依据：[后端组件契约设计](../specs/2026-08-03-backend-component-contracts-design.md)  
计划：[后端组件契约实施计划](../plans/2026-08-03-backend-component-contracts-plan.md)

## 完成内容

新增 `vocal_subtitle/contracts/` 契约包：

- `common.py`：契约、路由、决策和报告版本常量，以及 `ErrorInfo` 错误信封；异常转换不序列化 traceback。
- `task.py`：`TaskRequest`、`TaskSnapshot`、`TaskState`，兼容旧任务历史列和 `progress_json`。
- `run.py`：`RunRequest`、`RunResult`，支持旧 `Pipeline.run()` 字典返回值和 `PipelineStats` 原对象保留。
- `engine.py`：`EngineIdentity`、`EngineAvailability`、`EngineRequest`、`PrepareResult`、`EngineResult`。
- `report.py`：版本化 `RunReport`，保留现有 `run-report-v1` JSON payload。
- `external.py`：框架无关的 CLI/WebUI 请求转换。
- `ports.py`：任务、运行、引擎注册表、引擎、报告、产物、证据复核和决策投影 Protocol。

新增兼容适配器：

- `TaskHistoryAdapter`：接入现有 SQLite `TaskHistoryManager` 和状态机。
- `PipelineRunAdapter`：接入现有 `Pipeline.run()`，不改变参数和返回值。
- `EngineRegistryAdapter`：接入治理层生命周期状态并映射为统一可用性。
- `ASREngineAdapter`：接入现有 ASR `load_model/transcribe`，统一执行结果和错误类别。
- `RunReportAdapter`：接入 `RunReportBuilder`，保持报告 schema 与附件持久化。
- `ArtifactRegistryAdapter`：提供协调器和测试使用的产物端口实现。

新增 `BackendRunCoordinator`/`RunCoordinator`：只依赖 Protocol，统一处理任务创建、预检/运行状态、终态、产物登记和报告生成。报告失败只写入结构化诊断，不覆盖已经生成的字幕结果；该协调器暂未替换旧 CLI/WebUI 调用路径。

## 兼容边界

- 未修改现有 `Pipeline.run()`、CLI 命令、WebUI route、任务历史 schema、缓存 payload、`PipelineStats` 字段或 ASR 召回逻辑。
- 旧报告对象和旧任务记录都通过 adapter 读取；新契约对象提供 `to_dict/from_dict` round-trip。
- 契约层不依赖 Click、FastAPI、SQLite 或具体模型 SDK。
- 本阶段不改变黄金集门禁、质量阈值、发布结论和真实模型选择。

## 验证结果

定向验证：

```text
60 passed, 1 warning
```

全量验证：

```text
1159 passed, 1 skipped, 2 warnings
```

其他检查：

- `python -m compileall -q vocal_subtitle tests scripts`：通过。
- `git diff --check`：通过。
- 新增模块均小于 1000 行。

警告均为现有依赖/兼容性警告，不是本阶段新增失败。

## 后续阶段

下一阶段进入前端组件契约：先通过 `WebUIAdapter` 固定现有请求/响应与 route 兼容边界，再逐步将页面状态消费迁移到 `TaskSnapshot`、`RunResult` 和报告契约；继续保留旧 WebUI 模型和 route 作为兼容入口。
