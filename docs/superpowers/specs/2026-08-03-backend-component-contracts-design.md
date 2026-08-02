# Vocal Subtitle 后端组件契约设计

日期：2026-08-03  
状态：已确认，待实施  
前置阶段：组件化收口

## 1. 目标与范围

本阶段在组件化收口后的 application façade、运行报告、任务状态和 CLI/WebUI 兼容入口之上，建立统一的后端组件契约。目标是让任务、运行、引擎、证据/决策/投影、报告和产物管理都通过明确的 Protocol、版本化 dataclass 或 typed envelope 交互。

本阶段采用兼容优先、非破坏式演进：保留现有 CLI、WebUI route、请求/响应字段、缓存结构、任务历史 payload、`Pipeline.run()` 返回值和 `PipelineStats` 字段；通过 adapter 逐步迁移内部实现。

本阶段不包含：

- ASR 召回、模型策略或黄金集阈值修复；
- WebUI 新交互和前端组件；
- 破坏性 API v2；
- 流式链路重构；
- 反馈参数自动消费到生产链。

## 2. 契约分层

```text
External Adapter
  CLI / WebUI / Python API
          ↓
Application Contract
  RunRequest / RunResult / TaskSnapshot
          ↓
Domain Contract
  Evidence / Decision / Projection
          ↓
Infrastructure Port
  ASR / VAD / Separation / Diarization / Storage / Report
```

依赖规则：

- 外部适配器不得直接操作具体引擎或 Pipeline 私有方法；
- application 只编排，不定义 ASR 或字幕业务规则；
- domain 只消费明确输入并返回可序列化结果；
- infrastructure 通过 Protocol 接入；
- 组件间使用版本化 dataclass、Protocol 或 typed envelope；
- 旧入口保留为兼容代理，不保留第二套业务逻辑。

目标新增目录：

```text
vocal_subtitle/contracts/
  common.py      # 版本、标识、错误、诊断
  task.py        # 任务请求、状态、快照
  run.py         # 运行请求、结果、降级信息
  engine.py      # 引擎身份、能力、生命周期、执行结果
  report.py      # 运行报告和报告存储 Port
  external.py    # CLI/WebUI 边界对象
```

现有 `asr/contracts.py`、`physical/decision_ir.py`、`reporting/` 和 `governance/` 逐步适配这些契约，不在一次迁移中删除旧类型。

## 3. 核心数据契约

### 3.1 任务

```python
TaskRequest:
    task_id: str | None
    input_path: str
    output_path: str | None
    profile: str
    mode: Literal["offline", "streaming"]
    overrides: Mapping[str, Any]
    requested_by: str
    contract_version: str

TaskSnapshot:
    task_id: str
    run_id: str | None
    state: TaskState
    progress: float
    stage: str
    status: str
    error: ErrorInfo | None
    created_at: str
    updated_at: str
```

`TaskState` 继续沿用现有状态机；任何组件不得绕过 `TaskPort` 跳过状态。

### 3.2 运行

```python
RunRequest:
    task: TaskRequest
    config: PipelineConfig
    input_fingerprint: str
    services: RuntimeServices

RunResult:
    task: TaskSnapshot
    subtitle_path: str | None
    events: tuple[SubtitleEvent, ...]
    stats: PipelineStats
    report: RunReport | None
    artifacts: Mapping[str, str]
    diagnostics: Mapping[str, Any]
```

现有 `Pipeline.run()` 的 dict 返回值通过兼容 adapter 转换为 `RunResult`，不删除旧调用路径。

### 3.3 引擎

```python
EnginePort:
    identity() -> EngineIdentity
    availability() -> EngineAvailability
    prepare(request) -> PrepareResult
    execute(request) -> EngineResult
    release() -> None
```

`EngineResult` 统一承载候选文本或声学结果、时间坐标类型、词级时间、引擎/模型/版本、降级原因和可序列化诊断。现有 ASR `transcribe()` 通过 adapter 接入，不要求所有引擎立即重写。

### 3.4 报告

`RunReport` 继续遵守 `RUN_REPORT_SCHEMA.md`。契约层只提供类型化构建和持久化 Port，不改变现有 JSON 字段。报告必须包含：运行标识、pipeline path 和版本、阶段状态/耗时、引擎可用性、质量/降级信息、证据/决策/投影诊断、产物、错误类别和可恢复性。

## 4. Ports 与 Adapters

### Application Ports

- `TaskPort`：创建任务、状态转换、快照、失败/降级/取消；
- `PipelineRunPort`：接收 `RunRequest`，返回 `RunResult`；
- `EngineRegistryPort`：查询能力、可用性、路由选择和实例复用；
- `ReportPort`：构建、持久化、查询报告和记录降级事件；
- `ArtifactPort`：注册、检查和清理字幕、音频、报告产物；
- `EvidenceReviewPort`、`DecisionProjectionPort`：封装现有证据复核与物理投影能力。

### Adapters

- `PipelineRunAdapter`：现有 `Pipeline.run()` → `RunResult`；
- `TaskHistoryAdapter`：`TaskHistoryManager` → `TaskPort`；
- `EngineRegistryAdapter`：引擎工厂/治理注册表 → `EngineRegistryPort`；
- `RunReportAdapter`：`RunReportBuilder` → `ReportPort`；
- `WebUIAdapter`：FastAPI request/response model ↔ 外部契约；
- `CLIAdapter`：Click 参数 → `TaskRequest`。

调用关系：

```text
WebUI / CLI
    ↓
External Adapter
    ↓
TaskPort + PipelineRunPort
    ↓
Application Coordinator
    ├─ EngineRegistryPort
    ├─ EvidenceReviewPort
    ├─ DecisionProjectionPort
    ├─ ReportPort
    └─ ArtifactPort
    ↓
Existing domain implementations
```

依赖注入优先通过构造函数完成；application coordinator 不依赖 FastAPI/Click，domain 不依赖 WebUI、CLI、SQLite 或模型 SDK。

## 5. 错误与版本

统一错误对象：

```python
ErrorInfo:
    category: str
    code: str
    message: str
    retryable: bool
    recoverable: bool
    stage: str | None
    engine: str | None
    details: Mapping[str, Any]
```

错误分类沿用：`input_invalid`、`dependency_missing`、`model_unavailable`、`engine_failed`、`timeout`、`resource_exhausted`、`preflight_failed`、`contract_invalid`、`output_failed`、`unrecoverable_failure`。

组件只能返回结构化错误，不能吞错或直接修改任务状态；状态转换由 application coordinator 统一处理。报告或产物失败不得覆盖已生成的主字幕结果。

版本字段独立管理：

- `contract_version`：数据对象和 API envelope；
- `route_version`：ASR 路由；
- `decision_policy_version`：决策策略；
- `report_schema_version`：运行报告。

同一主版本只允许新增可选字段；删除字段或改变类型必须升级主版本。读取端支持当前版本和上一兼容版本，旧缓存和任务历史通过 adapter 转换。外部 payload 不写入密钥、完整凭证或完整异常堆栈。

## 6. 迁移顺序

1. 新增 contracts 类型和 Protocol；
2. 为 TaskHistory、Pipeline、EngineRegistry、RunReportBuilder 建立 adapters；
3. application coordinator 改为依赖 Protocol；
4. CLI/WebUI 改为调用 coordinator；
5. 保留旧方法作为兼容代理；
6. 增加旧 payload 与新契约的双向 round-trip 测试；
7. 执行全量测试和现有 API/CLI/WebUI 契约检查。

## 7. 测试与门禁

测试分为：

1. schema round-trip：旧 payload ↔ 新契约；
2. Port contract：fake adapter 验证成功、降级、重试、失败和取消；
3. application coordinator：正常、预检失败、引擎缺失、review/projector 降级、报告失败、产物缺失、缓存命中；
4. external adapter：CLI、WebUI、HTTP 错误映射和旧 hook；
5. 结构与版本门禁：Protocol 实现完整性、版本字段、依赖方向、schema 兼容性。

完成标准：

- 所有任务状态只能通过 `TaskPort` 变化；
- 所有运行结果都能转换为 `RunResult`；
- 所有引擎都有统一 identity、availability 和 error 结果；
- 所有报告包含运行、版本、降级和产物信息；
- 旧 CLI、WebUI、缓存和历史数据可继续读取；
- 新契约失败不会覆盖原始字幕结果；
- 不改变 ASR 召回、黄金集门禁和发布结论。
