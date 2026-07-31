# 大文件组件化实施报告

日期：2026-07-31  
冻结功能基线：`5079e43` 及其父提交 `1b4a705` 之前的实现历史  
关联设计：[大文件组件化设计方案](../specs/2026-07-31-large-file-componentization-design.md)  
关联收口设计：[大文件组件化完整收敛设计](../specs/2026-07-31-large-file-componentization-completion-design.md)

## 1. 基线口径

`5079e43` 只包含组件化设计和执行计划，不包含新的功能实现。因此本次验收使用以下口径：

- `1b4a705` 及更早历史用于确认原有功能、公共入口、API 和数据契约。
- `5079e43` 作为功能冻结验收点，组件化不得引入其后的新业务行为。
- 组件化只允许改变代码位置、依赖装配和内部调用方式，不允许改变字幕时间坐标、事件字段、配置字段、缓存/历史 payload、API response、降级语义和导出格式。

## 2. 实施结果

组件化已完成，当前代码保留兼容入口并将职责拆到以下边界：

| 原大文件 | 组件化结果 |
|---|---|
| `vocal_subtitle/pipeline.py` | 收敛为约 235 行兼容入口和依赖装配；离线生命周期、阶段执行、结果统计、流式生命周期和后处理分别位于 `application/` 及领域阶段模块。 |
| ASR 路径 | `asr/contracts.py` 提供 request/result/ports；`global_path.py`、`segmented_path.py`、`review_path.py` 可独立实例化，支持 global、segmented、review、quality gate 和 fallback 契约。 |
| `vocal_subtitle/config.py` | 拆为 `config/models.py`、`loader.py`、`overrides.py`、`validation.py`；`config/__init__.py` 保留旧导入路径。 |
| `vocal_subtitle/acoustic_validator.py` | 拆为 `acoustic/validator.py`、`skeleton.py`、`boundary.py`、`event_checks.py`、`diagnostics.py`、`export.py`；根模块保留兼容导出。 |
| `vocal_subtitle/merging/llm_merge_engine.py` | 拆为合并外壳、contracts、policy、constraints、local/cloud decider、layout；旧模块继续导出 `LLMMergeEngine` 和历史辅助符号。 |
| `vocal_subtitle/webui/api.py` | 收敛为路由汇总和兼容 façade；任务、历史/缓存、模型、feedback、LLM、字幕编辑和序列化由独立 route/service 模块负责。 |
| `vocal_subtitle/webui/static/index.html` | 页面结构、样式、API client、状态、设置、进度、字幕、feedback 和 WebSocket 逻辑拆为独立静态资源，原资源 URL 继续可用。 |
| `tests/test_feedback.py` | 按音频、核心算法、质量和集成职责拆到 `tests/feedback_suite/`，历史测试路径仍可显式运行。 |

## 3. 兼容收口

本次补齐了组件化过程中最容易遗漏的旧入口：

- `webui.api` 继续暴露冻结点的路由函数、请求/响应模型、字幕序列化 helper、FunASR 入口和任务线程入口。
- `runtime_state` 在调用时解析 `api._task_store`、`api._task_history`、`api._shadow_evaluators`、`api.UPLOAD_DIR`，旧 monkeypatch 仍会作用于实际路由。
- 任务路由在提交线程时解析旧的 `api._run_pipeline_in_thread` hook。
- 持久化服务在调用时解析旧的 `api._persistence_mgr` hook。
- `ConfigLoader`、`PipelineConfig`、`PipelineStats`、`AcousticValidator`、`LLMMergeEngine` 等旧导入路径保持可用。
- 52 个冻结点 WebUI method/path 逐项保持一致。

新增验证脚本：

- `scripts/check_public_compatibility.py`：检查冻结点公共导入和兼容 hook。
- `scripts/check_api_contract.py`：默认从 `5079e43` 原始 `webui/api.py` 解析路由基线，不再把当前结果当作自身基线。
- `scripts/check_component_contracts.py`：检查领域反向依赖、私有 context 调用和路由直接构造模型。
- `scripts/check_import_boundaries.py`：检查组件内部导入边界和循环依赖。
- `scripts/check_module_size.py`：检查源文件规模和新增文件硬阈值。

## 4. 验证结果

在仓库 `.venv` 环境中执行：

```text
./.venv/bin/pytest -q
899 passed in 5.97s

./.venv/bin/pytest -q /tmp/vocal-subtitle-baseline.pJ3erE/tests \
  --confcutdir=/home/hope/Tools/Vocal_Subtitle \
  --rootdir=/home/hope/Tools/Vocal_Subtitle
887 passed, 2 failed
```

冻结点原始测试的 2 个失败均为归档临时目录缺少仓库外质量音频 fixture：

- `test_quality_manifest_contains_existing_fixture_pairs`
- `test_quality_manifest_rejects_duplicate_names`

失败发生在 fixture 文件存在性检查，未进入组件化代码路径；当前仓库全量测试中的对应测试均通过。

结构与兼容门禁：

```text
./.venv/bin/python scripts/check_public_compatibility.py
通过

./.venv/bin/python scripts/check_api_contract.py
通过；52 个 method/path 与 5079e43 基线一致

./.venv/bin/python scripts/check_component_contracts.py
通过

./.venv/bin/python scripts/check_import_boundaries.py
通过

./.venv/bin/python scripts/check_module_size.py --root vocal_subtitle --root tests --changed-only
通过；变更文件无超过 1200 行硬阈值的文件

./.venv/bin/python -m compileall -q vocal_subtitle tests scripts
通过

git diff --check
通过
```

定向组件和 WebUI 测试：

```text
./.venv/bin/pytest -q tests/test_webui_component_services.py \
  tests/test_webui.py tests/test_webui_batch.py tests/test_webui_runtime.py
31 passed
```

静态资源验证：

- `node --check` 对 `app.js` 和全部拆分 JS 通过。
- `/styles/app.css`、`/app.js` 和全部 `/js/*.js` 请求均返回 200。
- 首页、配置、设备、历史、缓存、默认 profile、speaker model、LLM provider、持久化 settings 初始化请求均返回 200。
- 使用 `agent-browser` 访问 `http://127.0.0.1:8765/`，首页正常渲染上传区、场景模板、处理进度、配置区和反馈区。

## 5. 未执行项目与剩余风险

以下项目需要真实部署环境或外部依赖，不能用当前 fake/静态验证替代：

- faster-whisper、whisper.cpp、FunASR 的真实 global/segmented/context re-ASR 逐字、逐时间结果。
- 真实 VAD、人声分离、声纹/说话人模型、GPU 显存和多任务并发行为。
- 外部 LLM 的实际请求、超时、错误和云端降级。
- 浏览器真实音频上传、长流程 Pipeline 完成、字幕编辑后 SRT/VTT/ASS 导出和历史恢复。
- 生产环境导入耗时、模型惰性加载和缓存复用性能。

这些是运行环境覆盖范围，不是当前自动化测试失败。组件化验收已确认结构、兼容契约、冻结点测试行为和 WebUI 静态资源没有发现功能缺漏；发布前仍应在目标环境执行一次真实音频全链路验收。

## 6. 结论

按 `5079e43` 冻结功能基线，组件化实现已完成并通过当前可执行的自动化与浏览器 smoke 验收。旧入口、路由、配置、字幕事件、缓存/历史和静态资源契约已保留；真实模型和外部服务相关的生产等价性属于部署环境验收项，已在本报告中明确标注。
