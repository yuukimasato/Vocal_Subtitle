# 组件化落地与基准行为校正设计

日期：2026-07-31  
基准提交：`5079e43`  
关联设计：[大文件组件化设计方案](2026-07-31-large-file-componentization-design.md)、[全局 ASR、上下文重识别与多引擎复核设计](2026-07-30-global-asr-context-reasr-design.md)

## 1. 目标

本次工作继续完成 `5079e43` 之后的大文件组件化，并校正当前分支中因迁移产生的行为和资源问题。最终结果必须同时满足：

1. 保留基准代码已有的 CLI、WebUI、配置、缓存、字幕导出、物理时间轴和 streaming 行为。
2. 保留并正确接入全局 ASR evidence、上下文重识别、多引擎局部复核和结构化诊断。
3. 默认模式仍为兼容的 `evidence/shadow`，不改变现有最终字幕。
4. 提供显式 `production review` 模式，只有经过词级、声学和物理时间轴校验后才应用 `replace`、`split`、`drop`。
5. 让 Pipeline 逐步只负责生命周期、阶段顺序、进度、降级和结果汇总。

本次不进行全量业务重写，不改变 ASS/SRT 字段语义，不删除旧入口，不清理用户当前未提交的改动。

## 2. 当前问题与优先级

### P0：必须先修复

- `webui/static/js/state.js`、`api-client.js`、`progress.js` 当前是对象片段，作为独立 script 加载会产生 JavaScript 语法错误。
- 静态资源拆分必须保证每个浏览器请求都得到合法、可执行的资源，`app.js` 不得依赖失败的片段脚本才能运行。

### P1：功能和契约校正

- `PipelineServiceFactory` 的 ASR 实例缓存必须按 `engine + model + device + compute_type` 隔离。
- 片段、全局、上下文主引擎和第二引擎必须使用包含音频/人声 hash、绝对时间、模型和策略版本的独立缓存身份。
- 主 ASR 路径必须统一复用 `asr/hallucination.py`，不能与 Pipeline 内部简化过滤器并存两套规则。
- 稳定事件 ID 不能包含 `SubtitleEvent.index`。
- global evidence 必须在语言和引擎路由契约明确后执行，不能绕过任务级路由造成证据来源不一致。
- production review 必须真正将裁决动作转换为最终事件，并执行词时间、声学边界、说话人边界、物理时间和显示时间校验。

### P2：结构和门禁补齐

- 把 global、segmented、review ASR 从 `pipeline.py` 移到独立服务。
- 把 chunk/stage/post-process/export 生命周期从 `pipeline.py` 移到 application 服务。
- 将配置 loader、overrides、validation 形成明确模块。
- 将 WebUI API 路由按领域拆分，保留原 endpoint 和 response shape。
- 将静态页面脚本拆成合法模块，`app.js` 只保留初始化和事件绑定。

## 3. 兼容模式和数据流

```text
音频/人声
   |
   +--> 任务语言与 ASR 路由
   |
   +--> 物理时间线、VAD、声学证据
   |
   +--> segmented path ------------------> 初始字幕事件
   |                                             |
   +--> global evidence path --------------> 词流与诊断
                                                 |
                                      anomaly/review path
                                                 |
                                primary + secondary 局部复核
                                                 |
                              ReviewReport + ReviewDecision
                                                 |
                    shadow: 只记录        production: 应用裁决
                                                 |
                              词级映射与物理时间轴校验
                                                 |
                                      finalizer / ASS / SRT
```

### 3.1 默认行为

`asr.global_asr.mode = evidence` 时：

- segmented 结果仍是默认生产结果；
- global/context/secondary 结果只写入 `PipelineStats.asr_review_diagnostics`；
- 不删除、替换或拆分最终事件；
- 诊断必须保留 event ID、source、engine、model、window ID、word ID、触发原因和失败降级信息。

### 3.2 正式复核行为

新增明确的 review application 配置开关，默认关闭。开启后：

- `keep` 保留原事件，但可按证据重新校正时间；
- `replace` 使用候选词和绝对时间替换事件；
- `split` 按候选词时间和长静音拆成多个事件；
- `drop` 仅允许在没有有效词级和声学证据时执行；
- `unresolved` 保留原候选并输出人工复核诊断。

所有动作必须经过最终物理事件检查，显示 cue 的最短阅读时长不得修改 `physical_start`/`physical_end`。

## 4. 目标模块边界

### 4.1 Application

```text
vocal_subtitle/application/
  pipeline_runner.py       # offline/streaming 生命周期与阶段顺序
  stage_runner.py          # 单块、骨架块和多块阶段执行
  postprocess_runner.py    # speaker、merge、acoustic、finalize 适配
  export_runner.py         # 字幕文件和多格式输出
  pipeline_result.py       # stats、结果汇总和兼容序列化
  pipeline_services.py     # 惰性依赖和 engine/cache factory
```

`pipeline.py` 保留 `Pipeline` 兼容入口和薄代理；不再新增领域规则。

### 4.2 ASR

```text
vocal_subtitle/asr/
  global_path.py           # global production/evidence 编排
  segmented_path.py        # segmented ASR、缓存、过滤和语言回退
  review_path.py           # anomaly、context window、primary/secondary 复核
  review_apply.py          # production review 的事件替换、拆分和删除
  evidence.py              # evidence/review 数据对象和绝对坐标转换
  global_transcriber.py    # 宏观分窗、重叠去重和全局词流
  context_scheduler.py     # 异常窗口调度
  hallucination.py         # 统一幻觉过滤策略
```

领域服务只依赖显式参数和数据对象，不通过 `getattr(Pipeline, ...)` 读取编排器私有状态。

### 4.3 Configuration

```text
vocal_subtitle/config/
  models.py
  loader.py
  overrides.py
  validation.py
  __init__.py              # 兼容导出
```

旧的 `vocal_subtitle.config`、`ConfigLoader` 和字段名保持可用。

### 4.4 WebUI

```text
vocal_subtitle/webui/
  routes_pipeline.py
  routes_subtitles.py
  routes_history.py
  routes_models.py
  routes_feedback.py
  api_serializers.py
  api_services.py

vocal_subtitle/webui/static/js/
  api-client.js
  state.js
  progress.js
  subtitles.js
  settings.js
  feedback.js
  app.js
```

每个 JS 文件必须是独立合法脚本，或者明确使用 ES module 方式加载；不允许把对象字面量中间片段当作独立资源。

## 5. 缓存和来源契约

所有 ASR 缓存键必须至少包含：

```text
artifact_type
schema_version
audio_hash
vocals_hash
engine
model
language
start
end
word_timestamps
condition_on_previous_text
device
compute_type
asr_route_version
quality_gate_version
hallucination_filter_version
context_reasr_version
cross_engine_policy_version
window_policy_version
```

缓存命中后仍需执行当前版本的文本规范化和幻觉过滤。缺少词级证据的缓存文本不能被视为已验证证据。

每条最终事件必须保留：

- 独立稳定 event ID；
- segmented/global/context/secondary 来源；
- 原始 segment ID 和 window ID；
- candidate word ID 和最终 word ID；
- engine/model；
- physical_start/physical_end；
- display_start/display_end；
- review decision 和 reason。

## 6. 实施顺序

1. 建立基准测试、契约快照和当前工作树隔离说明。
2. 修复 WebUI 静态脚本和资源 smoke test。
3. 修复 ASR engine/model 生命周期、缓存身份和稳定 ID。
4. 抽取 `global_path`、`segmented_path`、`review_path`，保持 shadow 行为。
5. 实现 `review_apply`，增加默认关闭的 production review 集成测试。
6. 抽取 stage/chunk/post-process/export runner，缩小 Pipeline。
7. 拆分 config loader、overrides、validation。
8. 拆分 WebUI 路由和剩余静态 JS。
9. 运行全量结构、行为和资源门禁，记录真实模型/GPU/浏览器限制。

每一步先新增模块和测试，再切换调用方；原入口保留薄兼容代理。不得在一次提交中同时重命名公共字段、改变默认生产行为和迁移大量实现。

## 7. 验收标准

- `5079e43` 之前已有测试和公共入口保持兼容。
- 默认 evidence 模式最终字幕与原 segmented 生产路径一致。
- 249 秒以上音频使用 global evidence 宏观分窗，不因 180 秒限制静默跳过。
- context review 只处理异常窗口，不重复识别完整音频。
- secondary engine 不可用时能降级并留下结构化诊断。
- production review 的 replace/split/drop 有 fake engine 覆盖，并确保不跨长静音、不跨说话人、不使用窗口边界作为字幕边界。
- ASR cache round-trip 和 engine/model 切换测试通过。
- `pipeline.py`、`webui/api.py`、静态 JS 不再承载已迁出的完整领域实现；超过 1000 行的文件必须有明确保留原因。
- Python 编译、导入边界、模块规模、JS 语法和 WebUI 静态资源请求检查通过。
- 全量测试通过；可选依赖缺失时明确列出 fake/static 验证和未完成的真实验证。
