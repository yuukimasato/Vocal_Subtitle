# 大文件组件化执行计划

日期：2026-07-31  
关联设计：[大文件组件化设计方案](../specs/2026-07-31-large-file-componentization-design.md)  
执行者：第三方 AI  
执行方式：按任务顺序逐项完成，每项独立验证并提交

## 0. 给执行 AI 的总指令

本计划是可直接执行的重构任务，不要求一次性重写整个仓库。开始前必须：

1. 阅读关联设计文档和当前目标文件。
2. 检查 `git status --short`，不得覆盖、回退或清理用户已有修改。
3. 创建独立分支或独立工作区。
4. 运行基线测试并记录结果；如果环境缺少 pytest，使用项目已有环境或等价隔离环境，并记录命令。
5. 每个任务只处理该任务列出的职责；发现行为问题时先记录，不借组件化之名改业务规则。
6. 每个任务结束后运行定向测试、`git diff --check` 和导入检查，再提交。
7. 不把多个职责重新集中到一个新的大文件；新文件超过 1000 行必须暂停并重新划分。

推荐提交前缀：`refactor:`。不要提交缓存、模型文件、截图、`__pycache__` 或生成的字幕文件。

## 1. 总体任务顺序

| 阶段 | 目标 | 主要文件 | 依赖 |
|---|---|---|---|
| 0 | 基线、规模检查和边界检查 | `scripts/`、`tests/` | 无 |
| 1 | `pipeline.py` 组件化 | `vocal_subtitle/application/`、`vocal_subtitle/asr/`、`pipeline.py` | 0 |
| 2 | `config.py` 组件化 | `vocal_subtitle/config/`、`config.py` | 0 |
| 3 | 声学校验组件化 | `vocal_subtitle/acoustic/`、`acoustic_validator.py` | 1 |
| 4 | LLM 合并组件化 | `vocal_subtitle/merging/` | 1、3 |
| 5 | WebUI API 组件化 | `vocal_subtitle/webui/` | 1、2 |
| 6 | WebUI 静态页面组件化 | `vocal_subtitle/webui/static/`、`webui/app.py` | 5 |
| 7 | feedback 测试组件化 | `tests/test_feedback/` | 0～6 |
| 8 | 全量验收和残余大文件审计 | `scripts/`、`docs/` | 1～7 |

## 2. 阶段 0：建立基线和自动门禁

### 任务 0.1：记录规模基线

**目标：** 固定本次重构的目标文件清单和行数，避免执行过程中凭印象判断。

**操作：**

- 使用 `find`/`wc -l` 统计源文件。
- 排除 `__pycache__`、缓存、截图、模型和构建目录。
- 保存目标文件、当前行数、计划拆分出的目标模块和当前测试入口。

**完成条件：** 目标清单与设计文档的 7 个大文件一致；若发现新文件超过 1000 行，追加到清单并说明原因。

### 任务 0.2：增加 `scripts/check_module_size.py`

**目标：** 让后续任务可以自动发现大文件和新产生的大文件。

**要求：**

- 支持路径参数，默认检查 `vocal_subtitle`、`tests`。
- 支持 `.py`、`.html`、`.js`、`.css`。
- 排除 `__pycache__` 和明确的生成目录。
- `--max-lines` 默认 1000，`--hard-limit` 默认 1200。
- 输出机器可读 JSON 或稳定表格。
- 不因已有目标文件超过阈值而让当前基线任务失败；提供 `--baseline`/`--changed-only` 模式。

**测试：**

```bash
python scripts/check_module_size.py --help
python scripts/check_module_size.py --root vocal_subtitle --root tests
```

**提交：** `refactor: add module size audit`

### 任务 0.3：建立导入边界检查

**目标：** 阻止 domain、application 和 web 层互相反向导入。

**要求：**

- 至少检查 `vocal_subtitle/physical`、`vocal_subtitle/asr`、`vocal_subtitle/acoustic`、`vocal_subtitle/merging` 不导入 `pipeline` 或 `webui`。
- 检查新增模块是否形成循环导入。
- 无需引入重量级架构检查库；使用 AST 或现有依赖即可。

**完成条件：** 对当前仓库运行无未解释的新违规；已有例外写入 allowlist 并附迁移任务。

## 3. 阶段 1：拆分 `pipeline.py`

### 任务 1.1：提取统计和结果契约

**目标：** 让 `PipelineStats`、序列化和结果汇总不再与 5000 行编排器绑定。

**新增建议文件：**

- `vocal_subtitle/application/pipeline_result.py`

**迁移内容：**

- `PipelineStats`、`to_dict()`、`from_dict()`。
- global/fallback/quality/review/hallucination 字段。
- 只保留稳定类型和序列化逻辑，不把模型调用放入结果对象。
- `pipeline.py` 保留兼容导出：`from .application.pipeline_result import PipelineStats`。

**测试：**

```bash
pytest -q tests/test_phase_five.py tests/test_phase_one.py tests/test_pipeline.py
```

**完成条件：** 统计 round-trip 保持现有字段；旧缓存/历史 payload 缺字段仍可读取。

### 任务 1.2：提取引擎、缓存和依赖服务

**目标：** 消除 `Pipeline` 中引擎工厂和缓存初始化对业务阶段的耦合。

**新增建议文件：**

- `vocal_subtitle/application/pipeline_services.py`
- `vocal_subtitle/application/dependencies.py`（仅在确有独立依赖对象时创建）

**迁移内容：**

- ASR、VAD、separation、embedding、cache、history 的构造和惰性加载。
- 保持 engine/model/device/cache 身份规则。
- 不改变 `Pipeline._get_*` 的兼容调用；旧方法委托给 service。

**测试：**

```bash
pytest -q tests/test_asr tests/test_utils/test_cache_manager.py tests/test_pipeline.py
```

**完成条件：** fake engine 可注入；同一任务不重复加载模型；旧测试中的 monkeypatch 仍可工作。

### 任务 1.3：提取 global、segmented 和 review ASR 路径

**目标：** 让 ASR 证据、生产路径和上下文复核有独立边界。

**新增建议文件：**

- `vocal_subtitle/asr/global_path.py`
- `vocal_subtitle/asr/segmented_path.py`
- `vocal_subtitle/asr/review_path.py`

**迁移内容：**

- global production/evidence 入口。
- segmented ASR、过滤、缓存和语言回退。
- global/segment 对齐、异常窗口调度和 review diagnostics。
- 每个服务接收 engine/cache/timeline/config/context，不读取 `Pipeline` 私有属性。

**测试：**

```bash
pytest -q tests/test_global_asr_path.py tests/test_phase_five.py
pytest -q tests/test_asr/test_context_reasr.py tests/test_physical/test_phase_three.py
```

**完成条件：** `auto`、显式 `global`、显式 `segmented`、evidence shadow、失败回退行为不变；长音频 macro window fake engine 测试通过。

### 任务 1.4：提取块处理和阶段执行器

**目标：** 缩小 `_process_chunk_pipeline()`，让单块、多块、骨架块共享明确阶段接口。

**新增建议文件：**

- `vocal_subtitle/application/stage_runner.py`
- `vocal_subtitle/application/chunk_runner.py`

**迁移内容：**

- separation/VAD/merge/ASR/mapping 的阶段调用顺序。
- `PipelineContext` 作为输入输出总线。
- 每个阶段失败的降级和进度更新由 stage runner 统一处理。
- 不改变物理时间和 chunk 偏移规则。

**测试：**

```bash
pytest -q tests/test_pipeline.py tests/test_phase_zero.py tests/test_phase_four.py
pytest -q tests/test_physical tests/test_vad
```

**完成条件：** 单块、多块、skeleton mode、streaming 入口均通过 fake dependency 测试。

### 任务 1.5：提取 speaker、post-process 和 feedback 适配

**目标：** 清理 `pipeline.py` 中剩余的说话人、LLM/post-process、导出和 feedback 方法。

**新增建议文件：**

- `vocal_subtitle/diarization/pipeline_stage.py`
- `vocal_subtitle/application/postprocess_runner.py`
- `vocal_subtitle/application/export_runner.py`
- `vocal_subtitle/feedback/pipeline_stage.py`

**迁移内容：**

- 说话人投影、边界拆分、角色标注和标签生成。
- event post-process、finalize、subtitle export。
- feedback learning 入口只保留服务调用。

**测试：**

```bash
pytest -q tests/test_diarization tests/test_mapping tests/test_phase_four.py
pytest -q tests/test_feedback.py
```

**完成条件：** `pipeline.py` 只保留兼容入口和生命周期编排；对每个剩余私有方法给出“保留原因”或迁移目标。

## 4. 阶段 2：拆分 `config.py`

### 任务 2.1：移动 dataclass 和版本常量

**新增：** `vocal_subtitle/config/models.py`

移动所有配置 dataclass、默认值和与数据结构直接相关的版本常量。禁止在 models 中读取文件、环境变量或调用模型。

### 任务 2.2：移动 loader/profile 逻辑

**新增：** `vocal_subtitle/config/loader.py`

移动 `ConfigLoader` 的 profile 查找、YAML 读取和 `_parse_config()`。保持五个 profile 和旧配置缺省值兼容。原 `config.py` 通过 `config/__init__.py` 暴露旧符号。

### 任务 2.3：移动 overrides 和 validation

**新增：** `vocal_subtitle/config/overrides.py`、`vocal_subtitle/config/validation.py`

分别移动 nested override、类型转换、范围校验和一致性 warning。避免 loader 直接依赖 pipeline。

**测试：**

```bash
pytest -q tests/test_deployment_defaults.py tests/test_phase_one.py tests/test_phase_five.py
pytest -q tests/test_cli.py tests/test_language_policy.py
```

**完成条件：** `from vocal_subtitle.config import PipelineConfig, ConfigLoader` 继续有效；所有 profile 能加载；非法字段错误信息稳定。

## 5. 阶段 3：拆分 `acoustic_validator.py`

### 任务 3.1：提取 skeleton 和查询逻辑

**新增：** `vocal_subtitle/acoustic/skeleton.py`

移动 `_get_skeleton()`、语音区间查询、RMS/VAD overlap 查询等纯声学操作。输入输出使用绝对坐标，禁止依赖 Pipeline。

### 任务 3.2：提取边界和事件校验

**新增：** `vocal_subtitle/acoustic/boundary.py`、`vocal_subtitle/acoustic/event_checks.py`

移动边界吸附、置信度、micro-gap、物理覆盖、静音跨越和重叠校验。保留 `AcousticValidator.validate()` 作为兼容入口。

### 任务 3.3：提取诊断和导出

**新增：** `vocal_subtitle/acoustic/diagnostics.py`、`vocal_subtitle/acoustic/export.py`

移动报告、reason codes 和骨架音频导出。诊断对象必须可 JSON 序列化。

**测试：**

```bash
pytest -q tests/test_acoustic_validator.py tests/test_acoustic_gold.py
pytest -q tests/test_physical/test_timeline.py tests/test_physical/test_coverage_recovery.py
```

**完成条件：** 物理时间和显示时间约束不变；长静音不会被错误覆盖；导出骨架段行为不变。

## 6. 阶段 4：拆分 `llm_merge_engine.py`

### 任务 4.1：提取合并约束和规则策略

**新增：** `vocal_subtitle/merging/merge_constraints.py`、`merge_policy.py`

移动 gap、speaker、physical owner、hard split、fast merge 等无网络规则。所有 event 修改必须通过 `mapping/event_ops.py`。

### 任务 4.2：提取本地模型和云端 LLM

**新增：** `vocal_subtitle/merging/local_decider.py`、`llm_decider.py`

分别处理模型加载、相似度、API、超时、解析和降级。模型生命周期由依赖注入管理，不能在每条字幕中重新加载。

### 任务 4.3：提取布局和兼容入口

**新增：** `vocal_subtitle/merging/layout.py`、`merge_engine.py`

移动 frame seamless、断行、layout suggestion，并让原 `LLMMergeEngine` 成为兼容外壳。

**测试：**

```bash
pytest -q tests/test_merging tests/test_mapping/test_event_ops.py
pytest -q tests/test_phase_four.py tests/test_pipeline.py
```

**完成条件：** rule-only、local NLP、cloud LLM、LLM 失败降级和物理硬边界行为不变。

## 7. 阶段 5：拆分 `webui/api.py`

### 任务 5.1：提取序列化和共享服务

**新增：** `vocal_subtitle/webui/api_serializers.py`、`api_services.py`

移动 SubtitleEvent payload 转换、任务结果读取、文件重写和共享依赖。序列化函数必须保留旧字段，并对新增字段使用追加兼容策略。

### 任务 5.2：拆分路由组

按 endpoint 组移动到 `routes_pipeline.py`、`routes_subtitles.py`、`routes_history.py`、`routes_models.py`、`routes_feedback.py`。每个路由只做 HTTP 层工作，业务逻辑调用 service。

**注意：**

- 保持 FastAPI route path、method、参数名、状态码和 response shape。
- 不在路由 import 时加载 ASR、FunASR、speaker 或 LLM 重型模型。
- WebSocket 任务进度必须保持连接、消息类型和结束状态兼容。

**测试：**

```bash
pytest -q tests/test_webui.py tests/test_webui_batch.py tests/test_webui_runtime.py
pytest -q tests/test_subtitle_editing.py tests/test_session_manager.py
```

**完成条件：** API 测试通过；旧客户端请求和历史任务 payload 可正常读取；导出、编辑、模型准备和 feedback endpoint 均可用。

## 8. 阶段 6：拆分 `webui/static/index.html`

### 任务 6.1：建立静态资源目录和加载契约

将内联 `<style>`、`<script>` 和页面结构拆成设计文档指定的 CSS/JS 文件。先更新 `webui/app.py` 或静态服务配置，使资源能被正确访问，再删除内联实现。

**完成条件：** 直接打开 WebUI 时没有 404；页面入口、CSS、JavaScript 资源和 favicon 等请求可追踪。

### 任务 6.2：按职责拆 JavaScript

按以下顺序拆分：

1. `api-client.js`：fetch、WebSocket、错误处理。
2. `state.js`：任务、字幕、设置和 profile 状态。
3. `progress.js`：阶段进度和任务状态。
4. `subtitles.js`：字幕表格、编辑、预览和导出。
5. `settings.js`：profile、ASR、模型和参数设置。
6. `feedback.js`：feedback/profile/health/shadow UI。
7. `app.js`：初始化和事件绑定。

不要在拆分时改变文案、endpoint、localStorage key、设置字段名或 API payload。

**测试：**

```bash
pytest -q tests/test_webui.py tests/test_webui_runtime.py
python scripts/check_module_size.py --root vocal_subtitle/webui/static
```

如果环境支持浏览器测试，执行现有 WebUI smoke flow；否则至少使用静态资源请求检查和 JavaScript 语法检查，并记录缺失的真实浏览器验证。

## 9. 阶段 7：拆分 `tests/test_feedback.py`

按测试类和依赖域移动到 `tests/test_feedback/`。要求：

- 保留原测试函数/方法名称，避免历史测试报告失去可比性。
- 每个测试文件只导入对应领域组件。
- 跨 align/diff/learn/profile 的流程测试放 `test_integration.py`。
- 不因为拆分测试顺便修改生产参数或降低断言。

**测试：**

```bash
pytest -q tests/test_feedback
pytest -q tests/test_feedback.py  # 若保留兼容代理文件
```

若删除原文件会破坏外部测试入口，保留一个只导入新测试模块的兼容代理；代理不得复制测试代码。

## 10. 阶段 8：全量验收

### 任务 8.1：规模和边界审计

执行：

```bash
python scripts/check_module_size.py --root vocal_subtitle --root tests
python scripts/check_import_boundaries.py
git diff --check
```

列出仍超过 1000 行的文件及保留理由。`pipeline.py`、`config.py`、`webui/api.py` 和 `index.html` 不得仍然承载已计划迁出的领域实现。

### 任务 8.2：测试矩阵

至少执行：

```bash
pytest -q tests/test_asr tests/test_physical tests/test_mapping
pytest -q tests/test_diarization tests/test_merging
pytest -q tests/test_webui.py tests/test_webui_batch.py tests/test_webui_runtime.py
pytest -q tests/test_feedback
pytest -q
```

真实模型、GPU、浏览器或外部 LLM 不可用时，必须在交付报告中明确：已执行的 fake/static 验证、未执行的真实验证、原因和剩余风险。

### 任务 8.3：导入和启动性能记录

记录核心入口的：

```bash
python -X importtime -c "import vocal_subtitle.pipeline"
python -X importtime -c "import vocal_subtitle.webui.api"
```

若使用浏览器，记录静态资源加载总大小和首屏是否成功。不要声称“文件变小”必然带来运行时加速；若出现性能回归，优先检查重复 import、模型提前加载和资源请求数量。

## 11. 第三方 AI 交付格式

第三方 AI 每个阶段结束时必须输出一份简短报告，包含：

- 已完成的任务编号。
- 新增/修改/删除文件。
- 兼容入口和数据契约是否保持。
- 定向测试、全量测试和静态检查命令及结果。
- 仍超过 1000 行的文件和原因。
- 未解决的问题、风险和下一阶段建议。
- 该阶段的提交哈希。

最终交付必须包含一份汇总报告，明确说明哪些文件已组件化、哪些文件仍保留、所有测试结果以及真实模型/WebUI 验收限制。
