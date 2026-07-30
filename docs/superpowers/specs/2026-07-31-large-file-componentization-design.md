# 大文件组件化设计方案

日期：2026-07-31  
状态：设计已确认，按兼容优先方案执行
关联设计：[全局 ASR、上下文重识别与多引擎复核设计](2026-07-30-global-asr-context-reasr-design.md)

## 1. 背景

当前仓库已经形成较完整的离线字幕生产链路，但多个文件同时承担数据模型、生命周期编排、领域规则、兼容适配和输出逻辑。文件规模本身不是运行速度的主要瓶颈，但超大文件会增加以下维护成本：

- 修改一个领域规则时需要理解不相关的生命周期和兼容代码。
- 测试难以按职责定位，局部修改容易引起隐含状态回归。
- 模块之间通过私有方法和动态属性传递数据，边界不清晰。
- Python 模块导入时会加载整个文件；WebUI 单页文件会同时承载模板、样式、状态、API 和渲染逻辑，首次加载和修改成本都较高。
- 第三方 AI 执行重构时容易一次性重写大文件，导致回归范围不可控。

本方案只针对当前超过 1000 行的文件，不扩展到所有中小文件的普遍重构。拆分依据优先级是职责、状态所有权、依赖方向和测试边界，行数只是触发检查的门槛。

## 2. 范围与基线

基线使用 `wc -l` 统计 Python、HTML 和 CSS/JavaScript 源文件，排除 `__pycache__`、构建产物和缓存。当前主要目标文件如下：

| 文件 | 约行数 | 主要问题 | 优先级 |
|---|---:|---|---:|
| `vocal_subtitle/pipeline.py` | 5422 | 生命周期、ASR、VAD、说话人、映射、导出、反馈全部集中 | P0 |
| `vocal_subtitle/webui/static/index.html` | 4279 | HTML、CSS、API client、状态和渲染逻辑集中 | P1 |
| `vocal_subtitle/webui/api.py` | 2705 | 路由、任务执行、历史、缓存、字幕编辑、模型管理集中 | P1 |
| `vocal_subtitle/config.py` | 1375 | dataclass、YAML 解析、覆盖、校验和 profile 逻辑集中 | P0 |
| `tests/test_feedback.py` | 1348 | 多个 feedback 子系统和集成测试集中 | P2 |
| `vocal_subtitle/merging/llm_merge_engine.py` | 1168 | 规则合并、本地模型、云端调用、布局处理集中 | P1 |
| `vocal_subtitle/acoustic_validator.py` | 1037 | 骨架、边界吸附、校验、诊断和音频导出集中 | P1 |

若执行期间某文件已经低于 1000 行，应继续检查其职责边界；不能为了达到行数目标重新合并模块。若新文件超过 1000 行，必须在同一任务中说明原因并重新评估边界。

## 3. 目标与非目标

### 3.1 目标

1. 将每个目标文件拆成职责单一、可独立测试的模块。
2. 保留现有 CLI、WebUI API、配置字段、缓存兼容和字幕导出行为。
3. 让 `Pipeline` 只负责阶段顺序、依赖注入、进度、失败降级和结果汇总。
4. 让领域组件通过小型数据对象交互，不读取全局变量，不调用 `Pipeline` 私有方法形成隐式依赖。
5. 让第三方 AI 可以逐任务执行、逐任务测试和逐任务回滚。
6. 建立可重复的文件规模检查、导入检查、循环依赖检查和回归测试门禁。

### 3.2 非目标

- 不重写 ASR、VAD、说话人聚类、字幕时间轴或 WebUI 的业务行为。
- 不改变 ASS/SRT 导出格式和现有 API 响应字段的语义。
- 不为了降低行数创建只包含一个函数、没有独立职责的碎片模块。
- 不同时迁移数据库/缓存格式和模块结构，除非目标模块明确需要兼容适配。
- 不在没有基线测试或 fake engine 覆盖的情况下切换默认生产路径。

## 4. 设计原则

### 4.1 先契约，后搬运

每个拆分任务必须先确定输入、输出、异常和副作用，再移动实现。优先使用现有的 `PipelineContext`、`PipelineStats`、`GlobalTranscript`、`SubtitleEvent`、配置 dataclass 和现有缓存对象；不复制一套相似的数据结构。

### 4.2 编排器不持有领域规则

`pipeline.py` 只保留以下责任：

- 创建任务上下文和依赖。
- 按顺序调用阶段服务。
- 更新进度和 `PipelineStats`。
- 选择失败降级路径。
- 汇总最终事件、诊断和导出结果。

ASR 评分、物理边界、事件合并、说话人处理、反馈学习等规则必须放到对应领域模块。

### 4.3 依赖方向单向

推荐依赖方向：

```text
入口/CLI/WebUI
      |
Pipeline/Application Services
      |
Domain Services + Policies
      |
IR / dataclasses / pure utilities
```

领域模块不得反向导入 `Pipeline`、WebUI 路由或 CLI。需要回调时使用协议、函数参数或小型接口对象。避免通过 `getattr(self, ...)` 访问编排器的私有状态。

### 4.4 兼容外壳优先

原文件在过渡期可以保留兼容导出或薄包装方法，但包装层不应继续增加业务逻辑。迁移完成后，原模块只保留稳定公共入口、类型别名和向后兼容导入。

### 4.5 文件规模门槛

目标是职责清晰，而不是绝对行数。建议：

- 生产组件通常控制在 600～800 行以内。
- 测试文件按单一领域控制在 500～800 行以内。
- 超过 1000 行必须由计划任务明确说明其职责是否仍然单一。
- 任何新文件超过 1200 行，必须暂停拆分并重新提交边界说明。

## 5. 目标组件边界

### 5.1 `pipeline.py`

建议的目标结构：

```text
vocal_subtitle/application/
  pipeline_runner.py          # offline/streaming 生命周期与阶段顺序
  pipeline_services.py        # 依赖构造、引擎/缓存/历史服务
  pipeline_result.py          # PipelineStats、结果汇总和兼容序列化
  stage_runner.py             # 单块、多块、骨架块阶段执行

vocal_subtitle/asr/
  global_path.py              # global production/evidence 编排适配
  segmented_path.py           # segmented ASR、过滤、语言回退
  review_path.py              # anomaly/context review 调度

vocal_subtitle/diarization/
  pipeline_stage.py           # diarization 阶段适配与 speaker 边界

vocal_subtitle/mapping/
  pipeline_stage.py           # mapping、post-process、finalize 适配
```

`Pipeline` 可以暂时保留在 `vocal_subtitle/pipeline.py`，但应逐步成为兼容入口，内部委托给 `PipelineRunner` 和阶段服务。不得把原有 5422 行整体搬到一个新的 5422 行文件。

拆分顺序：统计/结果对象 -> 引擎和依赖工厂 -> global/segmented ASR -> chunk stage -> speaker -> mapping/post-process -> run/streaming 外壳。

### 5.2 `config.py`

建议拆成：

```text
vocal_subtitle/config/
  models.py       # 所有配置 dataclass 和版本常量
  loader.py       # YAML/profile 加载与默认值
  overrides.py    # merge_with_overrides、嵌套字段设置
  validation.py   # validate_config_consistency 与范围校验
  __init__.py     # 兼容导出 ConfigLoader、PipelineConfig 等
```

配置字段名称、默认值、profile 名称、敏感字段脱敏和旧 YAML 缺省行为必须保持兼容。第三方 AI 不得顺便重命名配置字段。

### 5.3 `llm_merge_engine.py`

建议拆成：

```text
vocal_subtitle/merging/
  merge_engine.py       # LLMMergeEngine 外壳和公共 merge 接口
  merge_policy.py       # fast/local/cloud/rule 决策
  local_decider.py      # sentence-transformers 生命周期和相似度
  llm_decider.py        # 云端请求、超时和降级
  layout.py             # frame seamless、断行、布局建议
  merge_constraints.py  # physical owner、speaker、gap 硬约束
```

合并策略必须继续复用 `mapping/event_ops.py` 和物理边界约束，禁止在新模块中重新实现一套 `start/end/text` 直接修改逻辑。

### 5.4 `acoustic_validator.py`

建议拆成：

```text
vocal_subtitle/acoustic/
  validator.py          # AcousticValidator 公共入口
  skeleton.py           # 声学骨架和区间查询
  boundary.py           # 起止边界搜索、吸附和置信度
  event_checks.py       # 物理事件/静音/重叠/覆盖校验
  diagnostics.py        # 报告和 reason codes
  export.py             # 骨架段音频导出
```

所有模块使用绝对秒坐标，不能在拆分过程中引入另一套时间坐标或把显示时间当成物理时间。

### 5.5 `webui/api.py`

建议拆成：

```text
vocal_subtitle/webui/
  routes_pipeline.py       # /api/run、任务状态、WebSocket 协调
  routes_subtitles.py      # 字幕查询、编辑、导出和序列化
  routes_history.py        # 历史、缓存和持久化
  routes_models.py         # ASR、speaker、FunASR 模型管理
  routes_feedback.py       # feedback/profile/health/shadow API
  api_serializers.py       # request/response payload 适配
  api_services.py          # 路由共享的业务服务和依赖
```

路由函数只做参数校验、服务调用和 HTTP 响应映射；不能在路由函数中运行整条 pipeline 或直接修改字幕领域对象。

### 5.6 `webui/static/index.html`

建议保持入口文件可直接被现有 WebUI 服务返回，同时将内联内容拆到：

```text
vocal_subtitle/webui/static/
  index.html               # 页面结构和资源引用
  css/base.css
  css/components.css
  js/api-client.js
  js/state.js
  js/progress.js
  js/subtitles.js
  js/settings.js
  js/feedback.js
  js/app.js
```

如果当前服务只支持单文件返回，先扩展静态资源路由并保留资源加载失败时的明确错误。不能直接把脚本拆出而不更新 `webui/app.py` 的静态目录配置。

### 5.7 `tests/test_feedback.py`

建议拆成：

```text
tests/test_feedback/
  test_aligner.py
  test_diff_analyzer.py
  test_param_learner.py
  test_profile_manager.py
  test_few_shot_cache.py
  test_health_scorer.py
  test_conflict_detector.py
  test_audio_fingerprint.py
  test_shadow_mode.py
  test_integration.py
```

测试拆分只改变收集路径，不改变 fixture、测试名称语义和断言。跨模块流程测试放在 `test_integration.py`，纯算法测试放到对应领域文件。

## 6. 迁移与兼容策略

1. 在每个阶段开始前记录基线测试结果和目标文件行数。
2. 先新增目标模块，再从原文件移动实现；移动期间原文件通过导入或薄包装继续提供旧入口。
3. 每完成一个职责组，运行该职责的定向测试和 import smoke test。
4. 只有定向测试通过后，才删除原实现；不要在同一个提交中同时大范围改名、改行为和改数据结构。
5. 保留至少一个过渡周期的兼容导出，避免外部调用方和测试直接导入旧路径时失败。
6. 每个阶段形成独立提交，第三方 AI 遇到回归时可以回退单个阶段。

## 7. 测试和质量门禁

### 7.1 行为门禁

- 运行与目标模块相关的既有 pytest 测试。
- 运行完整测试集。
- 对 `PipelineConfig`、`PipelineStats`、WebUI response、缓存 key 做 round-trip 检查。
- 对 global/segmented、streaming、fallback、LLM 不可用和模型不可用做 fake dependency 测试。

### 7.2 结构门禁

新增 `scripts/check_module_size.py`，默认检查：

- `vocal_subtitle/**/*.py`、`vocal_subtitle/webui/static/**/*.{html,js,css}` 和 `tests/**/*.py`。
- 排除 `__pycache__`、缓存、生成输出和第三方 vendor 文件。
- 输出超过 1000 行的文件、超过 1200 行的新文件和目标文件变更前后行数。

新增 `scripts/check_import_boundaries.py` 或等价静态检查，至少阻止：

- domain 模块导入 `pipeline`。
- domain 模块导入 `webui`。
- route 模块直接创建 ASR/VAD 模型。
- 新模块之间形成循环导入。

### 7.3 性能门禁

组件化不应以“文件变小”宣称运行时加速。第三方 AI 只需记录：

- `python -X importtime` 的核心入口导入耗时变化。
- WebUI 首屏资源加载是否成功及资源总大小。
- pipeline fake engine 的启动和单任务耗时不能出现明显回归。
- 模型只能按原有惰性生命周期加载，不得因为拆分而重复加载。

## 8. 风险和回滚

| 风险 | 处理方式 |
|---|---|
| 隐式 `self` 状态丢失 | 先定义 context/service 对象，迁移后用状态快照测试 |
| 循环导入 | 纯数据对象下沉到 IR/config 模块，运行时依赖使用注入 |
| WebUI 静态资源路径错误 | 先加静态资源 smoke test，再拆 HTML 内联脚本 |
| 缓存或 API 字段变化 | 保留旧字段和旧入口，增加 round-trip 测试 |
| 第三方 AI 一次性重写过多 | 按本方案任务顺序执行，每任务独立提交和测试 |
| 文件变小但职责更混乱 | 以边界、依赖和测试归属作为验收标准，不以行数单独验收 |

任何阶段失败时，只回退该阶段提交，不回退用户已有的无关工作区修改。

## 9. 完成标准

- 所有基线超过 1000 行的目标文件均已完成职责拆分，或在文档中明确记录保留原因。
- `pipeline.py` 不再承载所有领域实现，且应用入口行为保持兼容。
- WebUI API 和静态入口可正常加载，原有 endpoint、payload 和下载行为不变。
- 配置、缓存、字幕事件、物理时间轴和 ASR 证据契约均通过往返测试。
- 新增规模和依赖边界检查进入持续验证流程。
- 相关测试和完整测试通过；真实模型或浏览器测试受环境限制时，计划中必须记录 fake/静态验证结果和剩余风险。

## 10. 本次执行口径

本次实施以 `5079e43` 及其父提交之前的行为为基线，保留工作区中已有的组件化改动，不回退无关用户修改。执行采用兼容优先的方案：先恢复旧导入路径和公共调用入口，再收敛阶段边界，最后验证 WebUI 和静态资源。

测试统一优先使用仓库虚拟环境中的 `.venv/bin/python` 与 `.venv/bin/pytest`。可选模型依赖缺失时不得删除、跳过或放宽对应测试；应区分环境限制与代码回归，并在最终报告中分别列出。浏览器验证使用可用的浏览器 skill 或 CLI，对首页、静态资源、配置加载、任务状态、字幕查看/编辑/导出和错误响应执行 smoke flow；浏览器不可用时使用 FastAPI TestClient、静态资源请求和 JavaScript 语法检查替代，并记录限制。

阻断性问题包括：旧模块无法导入、公共符号或 monkeypatch 入口丢失、路由未注册、静态资源 404、配置/统计/缓存/字幕字段不兼容、ASR global/segmented/context re-ASR/multi-engine fallback 行为改变，以及任何未解释的既有测试回归。
