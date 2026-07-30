# ASR 语言路由与引擎质量优化落实计划

日期：2026-07-29
关联设计：[ASR 语言路由与引擎质量优化方案](../specs/2026-07-29-asr-language-routing-and-engine-optimization-design.md)

## 实施目标

在保留现有全局 ASR、物理时间轴和字幕导出链路的前提下，增加任务级 ASR 路由决策、FunASR 质量门禁、一次性 faster-whisper 回退、whisper.cpp 前置校验、可追踪质量状态和缓存隔离。自动模式只在全局有效探测窗口全部满足中文门槛时选择 FunASR；其他情况统一选择 faster-whisper。

## 实施步骤

### 1. 配置与稳定数据契约

修改 `vocal_subtitle/config.py`：

- 增加 `ASRAutoRoutingConfig`，集中承载探测模型、窗口数、窗口长度、中文概率、质量门禁、回退引擎和版本号。
- 将 `ASRConfig.engine` 支持 `auto`，保留旧的三个具体引擎值。
- 在 `_parse_config()` 中校验引擎、语言、概率/比例/窗口参数，旧配置缺省值保持可加载。

修改 `configs/default.yaml` 和五个场景配置：

- 新安装默认使用 `auto`。
- 写入自动路由和质量门禁默认值。
- 保留具体引擎配置的兼容注释和 whisper.cpp 路径配置。

新增 `vocal_subtitle/asr/router.py`：

- 定义不可变 `ASRRouteDecision` 和窗口证据结构。
- 实现显式引擎优先、显式语言优先、自动多窗口探测、中文硬门和 faster-whisper 默认分支。
- 探测失败、有效窗口不足和证据冲突都生成可序列化的诊断，不抛出为普通路由异常。
- 不允许 `auto` 自动选择 whisper.cpp。

### 2. 全局语言探测与引擎生命周期

修改 `vocal_subtitle/pipeline.py`：

- 为每个任务缓存一次 `ASRRouteDecision`，后续阶段不重复检测语言。
- 引入“探测语言”和“正式识别”两个明确阶段，并将阶段名传给现有进度管理器。
- 让 `_get_asr_engine()` 接受已解析的具体引擎或由路由结果构造，避免 `auto` 进入引擎工厂。
- 统一记录 `requested_engine`、`selected_engine`、`detected_language`、概率、决策原因和探测窗口。
- 正式 ASR 使用同一决策中的语言提示，不根据单个分段重新猜测引擎。

### 3. FunASR 质量门禁与回退

新增 `vocal_subtitle/asr/quality_gate.py`：

- 从最终候选字幕事件和物理语音区间计算 `coverage_ratio`、`text_density`、`max_event_duration`、`long_event_count`、`overlap_ratio` 和 `invalid_event_count`。
- 纯中文短音频可通过基础门禁；长音频低文本量/低密度/长事件组合触发失败，避免单条字幕仅凭条数判定。
- 输出 `pass/warning/failed/degraded` 和结构化原因；阈值全部来自配置。

修改 `vocal_subtitle/pipeline.py`：

- FunASR 正式事件在质量检查前保留原始诊断。
- 模型准备、推理、空结果、物理映射、最终化或导出失败时，自动模式最多回退一次 faster-whisper。
- 回退时丢弃 FunASR 正式事件，复用同一音频、物理证据和语言决策；记录初始/最终引擎与原因，禁止循环。
- 显式 FunASR + 非中文语言在启动阶段硬失败。
- 更新 `PipelineStats.to_dict()/from_dict()`，增加质量状态和路由字段，同时保持旧字段可读。

### 4. faster-whisper 尾段与物理覆盖

修改现有全局 ASR/物理证据接口：

- 检查并合并 Silero、ffmpeg coarse、ffmpeg skeleton 的尾段证据，计算完整 `last_physical_speech_end`。
- 尾段证据冲突时执行一次宽松复检，保留不确定区域，不用较短证据静默截断输入。
- 将全局 transcript、物理 bins 和最终事件的覆盖统计统一写入质量诊断。
- 局部恢复最多执行一次，并记录 `recovery_category`、缺口范围和恢复后覆盖结果。

优先复用 `vocal_subtitle/physical/coverage.py`、`vocal_subtitle/asr/local_recovery.py` 和现有全局路径，避免重写时间轴。

### 5. whisper.cpp 前置检查

修改 `vocal_subtitle/asr/whisper_cpp_engine.py`：

- 在模型加载/任务启动前检查可执行文件存在、可执行权限、模型存在、模型扩展名/格式和可运行性。
- 错误信息包含具体缺失路径及修复建议。
- 对骨架降级和 `display must cover physical end` 等契约错误保留 `degraded/failed` 质量状态，不伪装成普通成功。

### 6. 缓存、API 和 WebUI

修改 `vocal_subtitle/utils/cache_manager.py`、`vocal_subtitle/pipeline.py`：

- 将请求引擎、选中/最终引擎、路由版本、质量门禁版本、探测模型/窗口/阈值、语言摘要加入转录与完整管道缓存元数据。
- 缺失新版本字段的旧自动缓存不可复用；旧的显式引擎缓存继续按兼容规则读取。

修改 `vocal_subtitle/webui/api.py`、`vocal_subtitle/webui/models.py`、`vocal_subtitle/webui/static/index.html`：

- 增加“自动（中文优先）”选项，并在自动模式显示策略摘要。
- 进度显示语言检测、路由、质量校验和回退阶段。
- 结果区显示检测语言、选中/最终引擎、质量状态、回退原因和诊断摘要。
- FunASR 准备期间沿用现有禁用开始按钮逻辑；自动模式准备失败时显示具体状态。
- 具体引擎设置保持用户原值，不被默认值迁移覆盖。

### 7. 测试与验收

新增/扩展测试：

- `tests/test_asr/test_router.py`：语言窗口、显式优先级、探测失败、混合语言、whisper.cpp 不自动路由。
- `tests/test_asr/test_quality_gate.py`：覆盖、密度、长事件、重叠、空结果和质量状态。
- `tests/test_global_asr_path.py`、`tests/test_pipeline.py`：决策复用、FunASR 一次回退、统计传播、尾段恢复。
- `tests/test_asr/test_whisper_cpp_engine.py`：二进制/权限/模型格式前置检查和降级标记。
- `tests/test_utils/test_cache_manager.py`、`tests/test_phase_five.py`：路由/质量版本导致缓存隔离。
- `tests/test_webui.py`：自动选项、阶段文案和结果诊断字段。
- 配置测试：旧显式配置、新自动配置、非法路由参数和未知引擎。

验证顺序：

1. 运行新增 ASR、配置、Pipeline 和缓存单测。
2. 运行完整 `pytest`，修复由默认引擎变化引起的兼容性回归。
3. 在无法加载真实模型的环境中使用 fake engine 完成确定性路由/回退测试，并明确记录真实 3×3 浏览器验收是否受模型或音频资产限制。

## 完成标准

- 自动模式对所有有效探测窗口为高概率中文时选择 FunASR；任一非中文、不确定、冲突或探测失败时选择 faster-whisper。
- FunASR 质量失败最多回退一次，最终状态和原因可从 PipelineStats、任务结果、日志和 WebUI 读取。
- whisper.cpp 不会被自动策略选中，资源缺失在启动前明确失败。
- 英文长音频尾段物理覆盖诊断不再静默使用较短证据截断。
- 路由和质量策略变更会隔离缓存，旧显式配置和 API 字段保持兼容。
- 新增测试通过，完整回归结果和未覆盖的真实模型验收限制在交付说明中明确列出。
