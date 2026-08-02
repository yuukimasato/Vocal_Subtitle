# Vocal Subtitle 组件化收口设计

日期：2026-08-03  
状态：已确认，待实施  
实施顺序：组件化收口 → 后端组件契约 → 前端组件 → ASR 召回与黄金集门禁

## 1. 背景与范围

项目已经完成第一轮大文件组件化，但当前工作区在继续加入离线生产、治理、报告、质量和 ASR 能力后，仍有部分应用层和入口层文件超过模块规模边界。现有组件化报告证明了兼容入口、WebUI 路由和领域边界的基本做法，本阶段的目标是把这些边界收口为可持续维护的结构。

本阶段只做结构性改造：拆分职责、明确依赖方向、统一装配入口、补充结构门禁和兼容测试。现有字幕结果、ASR 路由、质量阈值、配置语义、缓存键、任务历史 payload、CLI 行为和 WebUI 契约必须保持不变。

本阶段不包含：

- 后端 API 字段或协议重新设计；
- WebUI 交互和前端功能新增；
- ASR 文本召回、模型选择或黄金集阈值修复；
- 反馈参数自动改变生产行为；
- 流式链路迁移；
- 与组件化无关的业务重构。

## 2. 当前问题

当前代码中主要的结构收口点如下：

| 文件 | 当前规模/职责 | 收口方向 |
|---|---:|---|
| `vocal_subtitle/cli.py` | 约 1697 行，包含所有 CLI 命令、参数、配置覆盖和输出 | 保留入口 façade，命令实现迁移到 `cli_commands/` |
| `vocal_subtitle/application/pipeline_runner.py` | 约 1265 行，同时承担运行、预检、任务收尾和报告 | 拆为生命周期、预检、收尾和报告组件 |
| `vocal_subtitle/application/asr_path.py` | ASR 路径兼容编排与运行上下文耦合 | 保持当前行为，补齐与 application/service 的边界 |
| `tests/test_asr/test_evidence_review.py` | 约 1173 行，集中承载多类证据复核测试 | 按候选校验、复核决策、诊断与降级拆分 |
| 新增治理/质量/报告模块 | 已有独立模块，但尚未全部接入统一结构门禁 | 纳入检查范围和公共导出约束 |

工作区存在用户未提交的实现变更。本阶段不得通过回滚、重置或覆盖方式处理这些变更；若拆分涉及同一文件，必须基于当前内容迁移。

## 3. 设计方案

### 3.1 组件边界与依赖方向

系统采用四层依赖方向：

```text
CLI / WebUI façade
        ↓
Application orchestration
        ↓
Domain components
        ↓
Infrastructure adapters
```

目标目录和职责如下：

#### CLI 组件

由于 `vocal_subtitle.cli` 是现有公开模块，不能新增同名包造成导入歧义。CLI 内部实现放在 `vocal_subtitle/cli_commands/`：

- `common.py`：公共 Click 参数、配置覆盖和输出格式化；
- `run_commands.py`：单文件和批处理命令；
- `feedback_commands.py`：反馈学习、样本和 profile 命令；
- `quality_commands.py`：质量、数据资产和趋势命令；
- `governance_commands.py`：实验、发布和预检命令；
- `__init__.py`：命令组装；
- `vocal_subtitle/cli.py`：继续作为 `vocal-subtitle` 的兼容 façade，只负责组装并导出 `main`。

#### Application 组件

- `run_lifecycle.py`：任务启动、阶段调用、异常转换和主运行结果汇总；
- `run_preflight.py`：输入文件、配置、引擎和运行环境预检；
- `run_finalizer.py`：任务状态收尾、统计补全和失败状态归类；
- `run_reporter.py`：标准运行报告构建、诊断合并和报告落盘；
- `pipeline_runner.py`：保留兼容方法与依赖编排，不再承载上述组件的完整实现。

#### 公共 Pipeline 入口

`vocal_subtitle/pipeline.py` 继续作为稳定公共入口，负责配置读取、`PipelineServices` 装配、惰性组件访问和现有 Mixin 组合。领域模块不得反向导入 `Pipeline`，也不得通过 Pipeline 私有方法获取跨层依赖。

#### WebUI 边界

继续使用现有 routes/services/models/serializers 划分。本阶段只修正结构性越界：路由不得直接构造 Pipeline 或具体引擎，旧路由、响应模型、私有 hook 和持久化入口保持兼容。

### 3.2 运行数据流

单文件运行统一遵循：

```text
CLI 参数
  → ConfigLoader + OverrideResolver
  → RunPreflight
  → Pipeline facade
  → Application RunLifecycle
  → 现有阶段 Mixin / Domain Service
  → PipelineResult + PipelineStats
  → RunFinalizer
  → RunReporter
  → CLI 输出 / WebUI 任务状态 / 字幕文件
```

数据对象职责保持现状：

- `PipelineConfig` 只描述配置，不保存单次运行状态；
- `PipelineContext` 只描述单次任务上下文；
- `PipelineStats` 保存统计、状态、降级和诊断；
- `PipelineResult` 保存字幕、音频、统计和可选报告；
- `RunReport` 保存标准化运行报告，不替代 `PipelineStats`；
- `TaskState` 负责任务状态机，不参与 ASR 或字幕决策。

组件间通过现有 dataclass、Protocol、request/result 对象传递数据，禁止依赖隐式全局状态。必要的运行状态从 façade 显式传入组件。

### 3.3 兼容策略

以下接口冻结：

- `Pipeline.run()`、`Pipeline.run_batch()` 和流式入口签名；
- `vocal_subtitle.cli:main` 和已有 CLI 命令、参数、默认值、退出码；
- `from vocal_subtitle.pipeline import Pipeline, PipelineStats`；
- 现有配置字段、缓存键、任务历史 payload 和字幕事件结构；
- WebUI method/path、请求字段、响应字段和旧兼容 hook。

迁移顺序为：先建立新组件并迁移实现，再让 façade 转发，最后删除重复实现。禁止同时保留两套会改变业务结果的实现路径。

## 4. 错误处理与降级

组件化不得改变现有错误语义：

- 预检失败仍返回当前错误类别和可读原因；
- 阶段异常仍由 Pipeline 转换为现有失败或降级状态；
- 任务收尾保证状态、统计和诊断在成功与失败路径都被补全；
- 报告生成失败不得覆盖主字幕结果，并记录结构化诊断；
- CLI 继续使用现有退出码和错误输出；
- WebUI 继续使用现有任务状态和错误响应。

若组件化暴露已有业务问题，只记录为回归风险或后续阶段任务，不在本阶段放宽门禁或修改 ASR/字幕行为。

## 5. 测试与结构门禁

### 5.1 结构检查

完善并统一使用以下检查：

- `check_module_size.py`：新增模块默认不超过 1000 行，1200 行为硬失败；
- `check_import_boundaries.py`：检查循环依赖和领域层反向依赖；
- `check_component_contracts.py`：禁止路由直接构造 Pipeline/引擎，禁止领域组件调用 Pipeline 私有方法；
- `check_public_compatibility.py`：检查公共导入和旧兼容 hook；
- `check_api_contract.py`：冻结 WebUI method/path；
- 新增 CLI 契约检查：检查命令名、参数名、默认值、帮助输出和退出码。

现有超大测试文件也按职责拆分，测试迁移必须保留原测试路径可运行，或提供明确的兼容导入/收集方式。

### 5.2 验证层次

1. 新组件单元测试，不加载真实模型；
2. `Pipeline`、CLI、WebUI façade 回归测试；
3. 结构门禁和公共契约测试；
4. 全量 pytest；
5. `compileall`、`git diff --check` 和必要的 JavaScript 语法检查。

### 5.3 完成标准

- `cli.py` 和 `pipeline_runner.py` 不再超过 1200 行硬阈值；
- 新增组件均有单一职责和明确导入边界；
- 结构检查全部通过；
- 公共导入、CLI、WebUI 路由和任务状态契约无回归；
- 全量测试无新增失败；
- 组件化文档和检查命令同步更新；
- 不改变 ASR 质量指标、黄金集门禁和最终字幕业务结果。

## 6. 实施顺序

1. 增加 CLI 和 Pipeline runner 的目标组件骨架及最小接口；
2. 将现有实现按职责迁移到目标组件，保留原 façade；
3. 拆分超大证据复核测试，并纳入新增结构检查；
4. 扩展结构门禁覆盖治理、报告、质量和新增 ASR 模块；
5. 执行定向测试和兼容检查；
6. 执行全量测试、编译检查和差异检查；
7. 更新组件化实施报告，记录未覆盖的真实模型/部署验证项。

## 7. 后续阶段衔接

组件化收口完成后，下一阶段再设计后端组件契约，重点处理运行报告、任务状态、引擎生命周期、证据/决策/投影和 API 请求响应对象。前端组件化和 ASR 召回修复不提前混入本阶段，以保持每一阶段的验收边界清晰。
