# Vocal_Subtitle 全链路完成实施计划

设计依据：[2026-07-26-full-chain-completion-design.md](../specs/2026-07-26-full-chain-completion-design.md)

## 实施原则

- 保留工作区已有用户改动，只修改本任务涉及的契约、链路、测试和文档。
- 先修复确定性回归，再扩展验收能力；每个阶段完成后运行对应的最小测试集。
- 全局 ASR 依赖保持可选。无 WhisperX 或模型时验证降级契约，不把 legacy 结果标为 global 成功。
- 真实素材报告与代码测试分离保存，报告记录实际路径、降级原因和不可发布原因。

## 步骤 1：基线与缺口清单

目标：把当前状态变成可重复的基线。

任务：

- 记录 `venv/bin/pytest -q` 当前 580 passed / 5 failed 的结果。
- 核对 PipelineStats、SubtitleEvent、API/WebSocket、CLI、任务历史和 WebUI 的字段映射。
- 核对 shell 安装脚本、optional extras、Debian postinst 和 fallback 文档。
- 建立 `test/quality_manifest.yaml`，明确音频与人工 ASS 配对、语言、场景类别和 speaker 数量。

验收：缺口清单不依赖未提交的临时输出，manifest 中的每个输入文件存在且人工字幕可解析。

## 步骤 2：修复测试契约与后端确定性回归

目标：恢复阶段五默认路径的测试一致性，并保留 LLM 安全边界。

任务：

- 将阶段前旧默认断言同步为当前部署契约：`none` 分离、WebRTC VAD、CPU small、diarization 关闭。
- 保留 profile-specific 的高质量分离和 diarization 断言，避免把默认轻量 profile 与 podcast 等模板混淆。
- 将 LLM 测试夹具改为满足安全门的同脚本、相近文本，增加明确的低相似度拒绝断言。
- 验证 `PipelineStats.to_dict/from_dict`、LLM 字段回写和 empty/fallback 行为。

验收：`tests/test_pipeline.py tests/test_deployment_defaults.py tests/test_phase_five.py` 全部通过。

## 步骤 3：后端路径、物理契约与缓存闭环

目标：确保 global、legacy 和降级状态在 Pipeline 内部一致，且最终事件不可越过物理边界。

任务：

- 检查 `_resolve_asr_path`、全局尝试、异常分类、结果校验和 segmented fallback 的所有分支。
- 检查 `PhysicalClip`、`SpeechEvidenceSpan`、`ContextWindow`、allocator、strict segmenter 和 final validator 的输入输出。
- 检查 LLM merge、字幕 builder、time mapper、结束时间修正是否保留 physical spans/source word IDs。
- 检查 full-pipeline、ASR、diarization 缓存键和兼容性判断是否隔离路径及配置版本。
- 补齐缺失的纯函数/模拟引擎测试，不依赖模型下载。

验收：全局成功、依赖缺失、执行失败、非法结果、显式 legacy 五类链路测试均能得到预期状态；所有非法事件在导出前被拒绝或带 warning 留在诊断中。

## 步骤 4：CLI、API、WebSocket 和 WebUI

目标：用户从任意入口都能选择路径并看到实际结果状态。

任务：

- 检查 CLI `run`、`batch`、配置覆盖和完成输出，补齐 `asr_path` 与降级摘要。
- 检查 API 请求模型、配置接口、任务结果、任务历史和 WebSocket complete payload 的字段兼容。
- 检查 WebUI 路径选择、提交参数、统计卡片、降级提示和事件归一化逻辑。
- 运行 API/CLI 测试，必要时用 TestClient/模拟 Pipeline 覆盖 global success/fallback/failure。
- 启动本地 WebUI 做最小浏览器冒烟：页面加载、路径选择、任务提交参数和完成状态渲染。

验收：旧请求不传新字段仍可运行，新请求可显式选择三种路径，前端不依赖缺失字段猜测状态。

## 步骤 5：安装、部署和模型能力探测

目标：安装文档和运行时行为与可选 WhisperX 一致。

任务：

- 核对 `pyproject.toml` WhisperX extra、Python 版本范围、requirements 文件和安装脚本。
- 补齐或修复安装期 CPU/CUDA 探测、元数据脱敏和失败诊断。
- 核对 Debian postinst 与源码安装共享入口，保证基础安装不强制加载 WhisperX。
- 运行 `bash -n`、安装脚本探测单元测试和基础导入检查。

验收：无 NVIDIA 选择 CPU；可模拟 NVIDIA 选择 CUDA；探测失败可回落 CPU；WhisperX 未安装时基础 profile 导入和 legacy 路径不失败。

## 步骤 6：真实素材 benchmark

目标：用 `test/` 音频和人工字幕形成可审计的质量对比。

任务：

- 为 manifest 配套读取音频、人工 ASS 和场景元数据。
- 扩展 benchmark 入口支持 manifest、`auto/global/segmented` 路径和输出目录。
- 复用时间轴对比逻辑，增加文本相似度、speaker 统计、物理越界、来源词缺失、warning 和降级诊断。
- 为每个素材分别保存字幕、对比报告和诊断报告；汇总报告区分 global 成功、legacy 降级、依赖不可用和运行失败。
- CI 模式在素材不存在、人工字幕缺失、global 未成功或指标失败时返回非零。

验收：可以用一个命令扫描现有 `test/`；模型不可用时仍产出明确的不可发布报告；有可用模型时能完成实际音频推理并与人工字幕比较。

## 步骤 7：文档同步与最终验证

目标：让用户文档、开发计划和验收状态与代码完全一致。

任务：

- 更新主设计文档 §1、§13、§14、§16 的实现状态，区分代码完成与素材质量通过。
- 更新 README、`docs/DEPLOYMENT.md`、`docs/API参考文档.md` 和阶段五开发说明。
- 增加真实素材运行命令、降级排障和模型授权说明。
- 运行全量 pytest、benchmark smoke/CI、CLI help、API smoke、Shell 语法、编译导入和 `git diff --check`。
- 写入最终验收报告，列出通过项、未通过项和模型/素材环境限制。

验收：所有可执行测试结果可复现，文档中的命令与参数在当前仓库有效，工作区未被无关格式化或清理。

## 预计产物

- `docs/superpowers/plans/2026-07-26-full-chain-completion-plan.md`
- `test/quality_manifest.yaml`
- benchmark/对比工具及其测试
- 后端、CLI、API、WebUI、部署和文档同步修改
- `test/benchmark_results/` 下的真实素材报告（不伪造 global 通过）
- 最终验收报告

