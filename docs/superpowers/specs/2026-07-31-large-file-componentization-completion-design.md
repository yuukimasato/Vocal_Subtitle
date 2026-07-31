# 大文件组件化完整收敛设计

日期：2026-07-31
状态：已确认方案，待实现
关联设计：[大文件组件化设计方案](2026-07-31-large-file-componentization-design.md)
关联核对：[大文件组件化落实核对报告](../reports/2026-07-31-large-file-componentization-implementation.md)

## 1. 目标

本设计用于完成大文件组件化的剩余收敛工作。目标不是继续降低文件行数，而是让组件之间通过显式数据契约和依赖端口交互，使 `Pipeline`、ASR、LLM 合并和 WebUI 路由之间不再通过私有方法、隐式 `self` 状态或全局变量形成耦合。

以 `5079e43^` 及更早历史为行为基线，完成后必须满足：

1. 现有 CLI、Pipeline 公共入口、WebUI endpoint、配置字段、缓存/历史字段和字幕导出格式保持兼容。
2. global ASR、segmented ASR、review/fallback、LLM 本地决策、云端决策和规则降级均可以脱离 `Pipeline` 单独实例化和测试。
3. `Pipeline` 只负责依赖装配、阶段顺序、状态汇总、进度和兼容委托，不再实现领域规则。
4. WebUI route 只负责 HTTP 参数校验、服务调用和响应映射；任务执行、模型准备、feedback、历史和字幕业务由服务对象负责。
5. 旧导入路径和历史 monkeypatch 入口继续可用，但兼容层不得成为新的业务实现位置。
6. 静态门禁能够阻止组件重新调用 `Pipeline` 私有方法、路由直接创建模型和跨层反向依赖。

## 2. 不变性约束

以下行为不能因组件化改变：

- `Pipeline.run()`、`run_batch()`、`run_streaming()` 的参数、返回字段和失败降级语义。
- `auto`、显式 `global`、显式 `segmented`、global evidence、context re-ASR、语言回退和多引擎 fallback 的选择逻辑。
- 物理时间坐标、物理 clip 所有权、声学边界、说话人边界和字幕事件字段。
- `PipelineStats`、配置 dataclass、缓存 key、历史 payload 和 WebUI response 字段。
- 52 个 WebUI method/path、HTTP 状态码、WebSocket 消息类型和静态资源 URL。
- rule-only、local NLP、cloud LLM、LLM 超时/失败降级和布局/断行行为。

组件化允许改变模块位置、构造方式和内部调用方式，不允许顺便修改上述业务规则。

## 3. 统一边界模型

### 3.1 依赖方向

```text
CLI / WebUI routes
        |
Application services / PipelineRunner
        |
Domain services / policies / adapters
        |
Contracts / dataclasses / pure utilities
```

约束：

- `asr`、`acoustic`、`merging`、`physical`、`mapping`、`diarization`、`feedback` 不得导入 `pipeline`、`webui.api`、`webui.routes_*` 或 CLI。
- `application` 不得导入 WebUI 层。
- `webui/routes_*` 不得直接实例化 ASR、VAD、分离、说话人或 LLM 模型。
- 领域服务不得接收 `Pipeline` 类型或通过 `context._...` 调用编排器私有方法。
- 需要环境能力时使用 Protocol、Callable 或显式 `*Ports` dataclass 注入。

### 3.2 请求、结果和端口

组件接口采用以下形式：

```python
@dataclass(frozen=True)
class ServiceRequest:
    ...

@dataclass
class ServiceResult:
    ...

@dataclass(frozen=True)
class ServicePorts:
    engine_factory: Callable[..., Engine]
    cache: CachePort | None
    progress: ProgressPort | None
```

请求对象只包含输入数据和配置快照；结果对象只包含领域结果、诊断和可恢复状态；端口对象只包含外部能力。服务不能从调用者对象反向读取属性。

旧入口通过构造 `ServiceRequest` 和 `ServicePorts` 调用新服务。旧私有方法可以保留为兼容 hook，但新服务只调用端口，不知道旧入口的存在。

## 4. ASR 组件收敛

### 4.1 目标结构

```text
vocal_subtitle/asr/
  contracts.py       # request/result/runtime ports/protocols
  global_path.py     # GlobalASRService
  segmented_path.py  # SegmentedASRService
  review_path.py     # ASRReviewService
  runtime.py         # engine/cache/language/quality port adapter
```

`application/asr_path.py` 不再保存 global/segmented/review 的领域实现，只保留 Pipeline 兼容 hook 或调用新服务的应用层装配代码。

### 4.2 Global ASR

`GlobalASRRequest` 至少包含音频、采样率、物理 shadow/timeline、VAD/ffmpeg 证据、配置快照和统计初始值。`GlobalASRService.run()` 返回 `GlobalASRResult`，包含事件、`GlobalTranscript`、coverage/recovery diagnostics、检测语言和失败分类所需信息。

global 逻辑中的 engine 加载、语言选择、词分配、物理 bin、coverage audit、tail recovery 和 quality gate 只能访问 `ASRRuntimePorts` 与 request 数据，不得访问 `Pipeline`。

### 4.3 Segmented ASR

`SegmentedASRRequest` 至少包含音频、采样率、VAD/合并片段、chunk 偏移、语言策略、缓存端口和引擎端口。`SegmentedASRService.run()` 返回原始片段、过滤统计、语言回退诊断和缓存信息。

`_process_skeleton_segmented()` 的旧入口转为兼容适配器；实际的骨架分段、ASR、过滤、去重、文本规范化和语言回退逻辑迁移到服务内部。

### 4.4 Review 和兼容 hook

`ASRReviewService.validate()` 和 `classify_failure()` 只接受 transcript、events、diagnostics、异常和配置，不接受 `Pipeline`。旧 `_validate_global_result()`、`_classify_global_failure()` 包装新服务，保留历史测试和 monkeypatch 入口。

## 5. LLM 合并组件收敛

### 5.1 目标结构

```text
vocal_subtitle/merging/
  contracts.py          # fragments, decision groups, ports
  merge_engine.py      # LLMMergeService/兼容外壳
  merge_policy.py      # fast/local/cloud/rule routing
  merge_constraints.py # speaker/gap/physical hard constraints
  local_decider.py     # local rule + sentence-transformer decision
  llm_decider.py       # cloud request, parse, timeout, fallback
  layout.py             # seamless, line break, layout
  applicator.py         # apply decision groups through event operations
```

### 5.2 决策端口

`LocalMergeDecider` 和 `LLMMergeDecider` 必须独立持有各自实现。它们通过 `MergeDecisionPorts` 接收模型加载、HTTP 请求、日志和 fallback 能力，不得接收 `LLMMergeEngine` 实例。

`MergeService.merge()` 的顺序固定为：补齐 gap -> fast policy -> hard constraints -> local decision -> cloud decision -> rule fallback -> decision applicator -> layout。所有 `start/end/text` 修改集中在 applicator 或已有 `mapping/event_ops.py`，不在 decider 中直接修改事件。

### 5.3 兼容策略

旧 `LLMMergeEngine` 继续导出 `merge()`、配置类、布局函数和历史私有辅助函数。兼容外壳在每次调用时构造端口，使旧的 `_local_merge_decision()`、`_call_llm_merge_decision()` 等 monkeypatch 仍能作为端口 hook 生效；新服务本身不依赖外壳。

## 6. Pipeline 收敛

### 6.1 PipelineRuntime

新增显式 `PipelineRuntime`，持有配置、引擎端口、缓存/历史端口、进度端口、阶段服务和任务状态。`PipelineRunner` 接收 runtime 和 request，负责阶段顺序、失败降级、进度和结果汇总。

阶段服务通过 request/result 交互，不再通过 mixin 的 `self._run_*()` 链接。最少拆为：

- `SeparationService`
- `PhysicalEvidenceService`
- `SegmentedASRService` / `GlobalASRService`
- `DiarizationService`
- `MappingService`
- `PostprocessService`
- `FeedbackService`
- `ExportService`

### 6.2 兼容外壳

`vocal_subtitle.pipeline.Pipeline` 保留原类名和公共方法。它负责：

- 将旧构造参数转换为 runtime 和 request。
- 将旧 monkeypatch 入口转换为端口实现。
- 调用 `PipelineRunner`。
- 维护旧属性和结果字段的兼容视图。

它不得再包含 ASR 评分、声学边界、说话人聚类、字幕合并或 WebUI 业务规则。现有 mixin 中的实现按领域迁移后删除；仅保留确有兼容需要的薄方法。

## 7. WebUI 服务层收敛

### 7.1 目标结构

```text
vocal_subtitle/webui/
  api.py                 # router 汇总 + 兼容导出
  api_services.py        # service container and shared ports
  runtime_state.py       # explicit runtime state
  services/
    pipeline_tasks.py    # upload, background task, websocket progress
    models.py            # model/device/FunASR operations
    history.py           # history/cache/persistence
    subtitles.py         # subtitle read/edit/export
    feedback.py          # learning/profile/health/shadow
  routes_*.py            # HTTP parameter/response mapping only
  api_serializers.py
```

路由函数不直接读取模块级任务字典，不直接创建 Pipeline，不直接执行外部 LLM 或模型准备。它们只从 `WebUIServiceContainer` 获取服务并映射响应。

### 7.2 兼容接口

`webui.api` 继续导出历史公共函数、router、`_task_store`、`_task_history` 等兼容符号。兼容别名指向 runtime state 或 service 对象，不复制业务实现。

### 7.3 WebSocket 和资源

WebSocket 的路径、连接生命周期、进度消息和结束状态不变。静态入口继续由 `webui/app.py` 挂载，CSS/JS URL、脚本顺序和 localStorage key 不变。

## 8. 分阶段迁移

### 阶段 A：契约和门禁

新增 contracts/ports、公共兼容测试和 AST 门禁；先不改变默认生产路径。完成后必须通过旧导入、路由快照和全量测试。

### 阶段 B：ASR

将 `application/asr_path.py` 的实现迁入 ASR 服务，Pipeline 通过 request/result 和 ports 调用。保留旧 hook，运行 ASR、physical、phase 和 pipeline 定向测试。

### 阶段 C：LLM 合并

迁移本地/云端决策和 applicator，缩小 `merge_engine.py` 为服务外壳和兼容 API。运行 merging、mapping、phase-four 和 pipeline 测试。

### 阶段 D：Pipeline

将 mixin 中的阶段逻辑转成服务，`PipelineRunner` 只保留生命周期和结果汇总。先支持旧 hook，再删除无业务职责的 mixin 方法。运行全链路 fake dependency、streaming、cache 和 export 测试。

### 阶段 E：WebUI

迁移 API 中的任务、LLM、feedback 和持久化业务到 services；route 只保留 HTTP 映射。运行 WebUI、batch、runtime、subtitle editing、session 和浏览器 smoke。

### 阶段 F：收口和审计

执行公共符号/路由基线比对、模块规模、导入边界、禁止私有调用、静态资源、全量测试和浏览器验证。更新实现报告，列出真实模型验证限制。

每个阶段只提交本阶段文件，不回退或清理工作区已有的无关改动。

## 9. 门禁设计

新增或增强以下检查：

- `check_import_boundaries.py`：层级依赖、循环依赖、domain/application 禁止 WebUI 依赖。
- `check_component_contracts.py`：禁止 ASR/merging/acoustic/domain 文件出现 `context._...`、禁止 route 文件出现 Pipeline/Engine 构造，检查服务接口是否使用 request/result/ports。
- `check_api_contract.py`：从 `5079e43^` 生成或读取固定 method/path 快照，禁止只检查数量。
- `check_public_compatibility.py`：检查旧模块公共导入、类方法和兼容别名。
- `check_module_size.py`：继续执行 1000 行提示、1200 行硬门禁，并检查新增文件。

门禁脚本本身使用 AST/静态解析，不导入真实模型或启动应用。所有检查都纳入 pytest 或 CI 命令。

## 10. 测试矩阵

每阶段至少执行对应定向测试；最终执行：

```text
./.venv/bin/pytest -q tests/test_asr tests/test_physical tests/test_mapping
./.venv/bin/pytest -q tests/test_diarization tests/test_merging
./.venv/bin/pytest -q tests/test_pipeline.py tests/test_streaming.py
./.venv/bin/pytest -q tests/test_webui.py tests/test_webui_batch.py tests/test_webui_runtime.py
./.venv/bin/pytest -q tests/test_feedback.py tests/feedback_suite
./.venv/bin/pytest -q
```

必须增加的边界测试：

- ASR 服务使用 fake engine/cache/physical timeline，不依赖 Pipeline 实例。
- LLM decider 使用 fake model、fake HTTP transport 和失败 fallback，不依赖 LLMMergeEngine。
- PipelineRunner 使用 fake stage services 验证顺序、失败降级、进度和结果汇总。
- WebUI route 使用 fake service container 验证请求/响应，不加载模型。
- 旧 monkeypatch 入口仍能改变对应端口行为。
- 运行时导入不会重复加载模型，旧缓存和历史 payload 能往返读取。

## 11. 完成标准

只有同时满足以下条件，才能将组件化标记为完成：

1. ASR/LLM/WebUI 新服务不接收或调用 `Pipeline` 私有方法。
2. `Pipeline` 和旧大文件只剩兼容入口、装配、生命周期和结果汇总。
3. route 文件不直接创建模型、Pipeline 或执行长耗时业务。
4. 结构门禁、公共兼容、完整路由快照和模块规模检查全部通过。
5. 全量 pytest 通过，且新增边界测试覆盖服务独立实例化和失败降级。
6. WebUI 首页、静态资源、初始化 API 和长流程 smoke 均通过；无法执行真实模型时明确记录环境限制。
7. 实现报告明确列出每个目标文件的最终职责、测试结果、保留兼容层和残余风险。

不以“文件低于 1000 行”单独作为完成依据；若职责、依赖和测试边界仍不清晰，即使规模门禁通过也只能判定为未完成。
