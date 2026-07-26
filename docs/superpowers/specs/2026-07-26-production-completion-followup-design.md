# 生产补齐后续设计：配置、CLI 与 WebUI 全链路契约

## 状态

已获用户批准，范围限定为可实现的工程缺口补齐。真实素材的 Start/End MAE、内容覆盖率和质量门优化不包含在本设计中。

## 背景与目标

最新系统设计文档确认 Pipeline 的 global/segmented、物理字幕仓、严格断句和最终校验链路已经具备。本次后续补齐聚焦入口层和配置层，目标是让同一套运行语义在 YAML、CLI、HTTP API 和 WebUI 之间保持一致，并能在进程重启后复现。

目标：

1. 配置文件中的 `degradation`、`expected_speakers` 和 `feedback` 可被正确解析、发现和覆盖。
2. CLI 的 `run` 与 `batch` 对同一组运行参数使用相同语义，明确 skeleton legacy 路径行为。
3. `download-models` 使用明确的 faster-whisper 下载 API，不依赖实例化模型的副作用。
4. WebUI runtime cache 配置更新后可持久化，并被后续任务重新加载。
5. 用无模型/无网络测试固定上述跨端契约，不改变已完成的 Pipeline 物理边界和 global 主路径行为。

非目标：

- 不重新设计 Pipeline、ASR、diarization 或字幕分段算法。
- 不在本轮调整真实素材质量阈值或宣称达到高精度质量门。
- 不覆盖工作区中已有的无关未提交改动。

## 方案与取舍

采用兼容性补丁加契约测试的方案。保留现有 `ConfigLoader`、API runtime cache JSON 和 CLI 配置覆盖机制，只修复其边界行为并补齐测试。

相比重构成统一配置服务，该方案改动面小，能保护当前已完成的 Pipeline；相比只修改 YAML，它还能够验证 API 重启后的配置持久化和 CLI/API 的参数语义一致性。

## 架构与数据流

### 配置加载

`ConfigLoader` 继续是所有离线任务的配置入口。解析 `degradation` 时同时接受顶层 `degradation` 和 `pipeline.degradation`，优先使用明确的顶层配置并在缺失时读取 pipeline 嵌套配置，以兼容当前两种 profile 写法。缺省值继续来自对应 dataclass 默认实例。

内置 YAML profile 明确暴露 `diarization.expected_speakers` 和完整 `feedback` 配置。字段值与 `DiarizationConfig`、`FeedbackConfig` 默认值保持一致，避免文档可见值和代码默认值分叉。

### CLI

`run` 和 `batch` 共享一个 CLI override 构建函数，统一处理设备、语言、分离、VAD、ASR 模型/路径、LLM、diarization、speaker role 和骨架导出参数。参数仍通过 `ConfigLoader.merge_with_overrides` 进入 Pipeline。

`--skeleton-mode` 是显式的 legacy 调试入口：启用后会提示并将未显式指定的 ASR 路径置为 segmented；生产 global 验收必须显式传入 `--asr-path global`。该行为写入帮助文本和回归测试，不再作为隐藏副作用存在。

`download-models` 使用 faster-whisper 提供的显式下载函数，传入确定的模型名和下载目录，不加载推理模型。分离模型下载行为保持现有引擎接口，错误按模型逐项收集并在命令结束时报告。

### WebUI/API

API 继续将 runtime cache 覆盖保存到 `cache/runtime_cache_config.json`。运行任务和 batch 任务在加载 profile 后统一调用 runtime override 应用函数，因此 API 进程重启后仍使用更新后的缓存配置。

更新接口对数值字段执行非负校验；配置写入采用临时文件替换，避免进程中断产生半截 JSON。更新成功后返回实际变更值，历史保留天数变更继续立即清理过期任务历史。

任务请求仍复用 `RunRequest`、`BatchRunRequest` 和 `SubtitleEventResponse`，不新增一套与 CLI 不同的字段命名。已有 physical/source 字段、ASR path 校验和 batch 任务行为保持兼容。

## 错误处理与兼容性

- 同时存在顶层和嵌套 `degradation` 时，顶层值优先，避免旧配置被静默覆盖。
- 无效 `expected_speakers` 继续通过配置一致性校验报告，不在解析阶段改变既有容错策略。
- 缺少 faster-whisper 依赖时，`download-models` 输出可操作错误并以失败状态结束；单个模型失败不隐藏其他模型结果。
- 下载网络或缓存错误不泄露 Token，也不写入明文凭据。
- runtime cache 文件不存在、格式损坏或字段缺失时回退到 profile 配置；写入失败返回 HTTP 500 并保留旧文件。
- skeleton 模式的 legacy 行为只影响显式启用该模式的 CLI 任务，不改变 API 默认 global/auto 路径。

## 测试策略

增加以下聚焦测试：

1. 配置解析：顶层/嵌套 `degradation` 优先级、默认 profile 的 `expected_speakers`、完整 feedback 字段、所有内置 profile 加载。
2. CLI：`run`/`batch` help 选项对称、模型非法值被 Click 拒绝、skeleton 模式的显式提示与 ASR 路径覆盖、下载函数被调用而非模型实例化。
3. WebUI/API：PUT cache config 写入文件、重新加载后应用到新任务、非法负数拒绝、batch 请求与单任务使用相同配置覆盖。
4. 回归：现有 CLI、部署默认值、WebUI 测试全部通过。

不依赖真实 ASR、模型下载、GPU 或 LLM 网络；外部依赖均通过 monkeypatch/mock 隔离。

## 验收标准

- `ConfigLoader` 不再静默忽略任一支持的 `degradation` 配置位置。
- 默认 YAML 可发现并覆盖 `expected_speakers` 与 feedback 参数。
- `run` 与 `batch` 对已支持参数的最终 `PipelineConfig` 结果一致。
- skeleton 行为在 help、日志和测试中明确；默认 API/global 路径不受影响。
- 模型预下载不实例化 `WhisperModel`，且失败结果可诊断。
- API runtime cache 更新跨重启生效，异常文件安全回退。
- 现有回归测试和新增契约测试全部通过。

