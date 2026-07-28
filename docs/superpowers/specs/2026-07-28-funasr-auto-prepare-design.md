# FunASR WebUI 自动准备设计

## 背景

WebUI 选择 FunASR 后，当前流程会把通用 ASR 模型名 `large-v3` 传给 FunASR，并在运行阶段直接导入 `funasr`。在未安装 Python 包时任务失败；即使安装了包，也没有明确的本地模型检查和准备流程。

## 目标

- 常规全量安装包含 FunASR 依赖。
- WebUI 选择 FunASR 时，按本地优先顺序准备运行环境：检查 Python 包、检查模型缓存、仅在缺失时安装/下载。
- FunASR 使用专用模型 ID，不再使用 `large-v3` 作为其模型名。
- 已有本地模型直接复用，不因选择引擎重复联网或下载。
- 缺包、下载失败和模型缓存不完整时，WebUI 显示明确错误，任务不进入不可解释的失败状态。

## 方案

采用“安装依赖 + WebUI 按需准备”的混合方案。`pyproject.toml` 增加 `funasr` 可选依赖并纳入 `all`；WebUI 只在用户选择 FunASR 时调用准备接口。服务启动不会因为未使用 FunASR 而安装包或下载模型。

## 运行流程

1. WebUI 的 ASR 引擎切换为 `funasr`，立即调用准备接口。
2. 后端使用当前进程的 `sys.executable` 检查 `funasr`，避免误用系统其他 Python。
3. 若包缺失，执行 `[sys.executable, "-m", "pip", "install", "funasr"]`；安装失败则返回非敏感的错误信息，不继续模型下载。
4. 将通用配置中的空值、`large-v3`、`medium`、`small`、`tiny` 映射为现有 FunASR 默认中文模型 `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`。
5. 先检查 FunASR/ModelScope 的本地缓存；命中完整模型时直接返回 ready。缓存不存在时才调用模型下载机制，完成后再次检查缓存。
6. 准备成功后，后续 Pipeline 使用同一 FunASR 模型 ID；切回其他引擎时保留其原有模型选择。
7. `/api/run` 在启动 Pipeline 前再次执行同一准备检查，防止用户在 WebUI 准备完成前点击运行造成竞态失败。

## 组件边界

- `vocal_subtitle/asr/funasr_engine.py`：定义 FunASR 默认模型、模型 ID 规范化和加载错误信息；不在模块导入阶段下载。
- `vocal_subtitle/webui/api.py`：提供 FunASR 状态/准备接口，负责异步线程执行安装与下载，避免阻塞事件循环。
- `vocal_subtitle/webui/static/index.html`：引擎切换时更新模型选项并调用准备接口，显示准备状态。
- `pyproject.toml`、`requirements-all.txt`、安装文档：声明常规安装依赖。

## 并发和错误处理

- 同一进程内使用锁避免多个切换事件并行安装或下载同一模型。
- 本地检查和已缓存路径不访问网络。
- 自动安装只使用当前 Python 解释器；子进程输出不回传完整环境信息或凭据。
- `pip` 不可用、网络失败、模型缓存不完整均返回可读错误；不伪造 ready 状态。

## 验证

- FunASR 引擎单测覆盖默认模型、通用模型名规范化和已有自定义模型保留。
- WebUI API 测试覆盖已安装/未安装、缓存命中/缺失、安装失败和下载失败。
- 配置测试确认 FunASR 不再将 `large-v3` 传给引擎。
- 静态 WebUI 测试确认 FunASR 选择包含准备触发和状态反馈逻辑。
- 在可用环境中验证选择 FunASR 后：已有缓存不联网，缺失缓存只下载一次，准备完成后任务可正常进入 ASR。

## 非目标

- 不在服务启动时安装 FunASR 或下载模型。
- 不替换 faster-whisper、whisper.cpp 的模型缓存策略。
- 不自动切换到其他 ASR 引擎掩盖 FunASR 安装/下载失败。
