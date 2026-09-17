# 更新日志 (Changelog)

本项目的所有显著变更都记录在此文件中。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

版本号唯一来源为 `vocal_subtitle/__version__`，CLI（`--version`）、WebUI（`/api/health`、FastAPI 元数据）与打包脚本均引用该值。

## [Unreleased]

## [0.3.0] — 2026-09-18

自 v0.2.0 以来共 90+ 个提交，聚焦高精度生产链、说话人身份主干与时间轴仲裁、四场景反馈学习闭环与流水线结构化治理。

### 新增

- **高精度生产链**：门控高精度生产 profile（`high_precision`），TTS 干净人声骨架优先 profile，按残余风险门控的上下文重识别（context re-ASR），高精度批量处理实现与冻结基线。
- **骨架分段主干模式**：跳过 VAD，以 ffmpeg 声学骨架直接分段（`--skeleton-mode`），配套全程识别辅助校验。
- **说话人身份主干与时间轴仲裁层**：物理边界优先级统一、决策门控的全局主路由、全局证据经候选角色路由、WhisperX 强制对齐溯源、无词说话人事件安全保留、字幕时间轴仲裁集中化。
- **四场景反馈学习闭环**：会话学习流程（inline-review / external-correction / existing-subtitle / from-scratch-timing 场景标签，`vocal-subtitle feedback learn --scenario`）、编辑日志摄取（journal sink/ingest）、参数学习任务异步化。
- **自动 ASR 路由与质量门禁**：全局语言证据补全、多引擎复核（Qwen3-ASR / ForcedAligner / SED）与显式引擎配对策略（`--primary-engine` / `--secondary-engine` / `--engine-pair-policy`）。
- **结构化流水线**：显式 Pipeline 运行上下文、生命周期阶段拆分、结构化流水线错误与阶段质量报告、ASR 执行计划与并发统一、基于内容+版本的缓存身份。
- **质量治理**：决策追踪（decision trace）、运行报告、黄金集质量门禁（幻觉保留 / 真实语音漏删 / 物理越界 / 跨静音）、生产链预检（preflight）、遗留路径使用证据与计数器。
- **WebUI**：任务执行与事件分离、数据集工作区、学习参数面板、历史清理；8613 处理台与 8631 编辑台双界面收敛及统一启动脚本（`scripts/launch_dual_ui.sh`）；review-manifest-v1 出处协议。
- **字幕打轴工作台**：`tools/subtitle-editor` 抽取为嵌套独立仓库（[yuukimasato/subtitle-editor](https://github.com/yuukimasato/subtitle-editor)），增加 Agent 访问层（CLI/harness/MCP）与双击即用单文件版。

### 修复

- 说话人分离 auto 模式：首个模型塌缩为单人时，自动用第二候选模型重跑复核。
- 字幕合并塌缩事件的说话人补偿；CJK 字间空格压缩。
- LLM 优化器：单请求 300s 超时，批次失败告警去重。
- UVR 分离短音频 pad 修复（BS/Mel-Roformer 推理 chunk 下限）。
- 字幕时间轴硬切与说话人空缺诊断修复（物理覆盖仲裁定案）。
- WebUI WebSocket：未匹配路径由 HTTP 500 改为干净拒绝。
- ASR 幻觉过滤诊断：记录真实每轮计数（此前为静态占位）。
- 字幕打轴工作台：`loadSubtitle` 未写入 `subtitleSource='file'`，导致"打开存量字幕"学习场景被误标为 from-scratch-timing。
- 打包：`build-deb.sh` 内嵌版本号停留在 0.1.0，与项目版本脱节；现在与发布版本一致。

### 变更

- 版本号集中化：CLI、WebUI 健康检查与 FastAPI 元数据统一引用 `vocal_subtitle.__version__`，不再各自硬编码。
- 大文件组件化重构完成（webui/asr/pipeline 拆分为子模块），仓库范围风格收敛。
- 流水线兼容基线锁定测试（1636 项测试套件，全量通过）。

## [0.2.0] — 2026-08-07

### 新增

- 物理优先架构 Phase 0-3：全局坐标系 (CoordinateMapper)、物理时间线 IR (PhysicalTimeline)、词级物理分配 (WordAllocation)、物理覆盖审计 (PhysicalCoverageReport)、物理字幕分箱 (PhysicalSubtitleBin)。
- 自适应反馈学习引擎 (Phase 5)：音频指纹匹配、参数震荡检测、健康度评分、Few-shot 示例缓存、Shadow Mode 安全试错。
- 字幕打轴工作台（tools/subtitle-editor）初版与单文件独立版。
- 多引擎复核模型注册与下载脚本（Qwen3-ASR / ForcedAligner / SED）。

## [0.1.0] — 2026-07-24

### 新增

- 首个可用版本：人声分离 → VAD → ASR → 字幕生成全链路 CLI，faster-whisper / Silero VAD / UVR 引擎接入，场景模板与 YAML 配置驱动，本地 Web GUI（FastAPI + WebSocket）。

[Unreleased]: https://github.com/yuukimasato/Vocal_Subtitle/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/yuukimasato/Vocal_Subtitle/releases/tag/v0.3.0
[0.2.0]: https://github.com/yuukimasato/Vocal_Subtitle/releases/tag/v0.2.0
