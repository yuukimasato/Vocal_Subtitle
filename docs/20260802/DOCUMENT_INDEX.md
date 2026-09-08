# 文档状态索引

**更新日期**: 2026-08-02
**维护规则**: 每次新增、修改或废弃文档时更新本索引。

## 状态标记说明

| 标记 | 含义 |
|------|------|
| ✅ 已实施 | 方案已在代码中实现并验证 |
| 🔧 部分实施 | 核心逻辑已实现，边缘场景或配套流程待完善 |
| 📋 待实施 | 方案已设计，尚未开始实现 |
| 🗑️ 已废弃 | 被更新的方案替代或不再适用 |
| 📖 仅供参考 | 历史记录、背景知识或外部参考 |
| 🏗️ 进行中 | 正在实施中 |

---

## 一、核心方案文档

| 文档 | 状态 | 说明 |
|------|------|------|
| [离线字幕全链路方案-2026-08-02.md](离线字幕全链路方案-2026-08-02.md) | 📖 参考 | 当前整体方案，代码审计与融合修订版 |
| [离线字幕全链路-整体落实计划-2026-08-02.md](离线字幕全链路-整体落实计划-2026-08-02.md) | 🏗️ 进行中 | 本计划的总体实施方案 |
| [统一优化方案.md](统一优化方案.md) | 📖 参考 | Plan 0-7 统一优化方案概览 |
| [人声分离字幕方案.md](人声分离字幕方案.md) | 📖 参考 | 早期完整方案（全链路可商用） |
| [人声分离字幕工程化方案.md](人声分离字幕工程化方案.md) | 📖 参考 | 早期工程化方案 |
| [Vocal_Subtitle-离线高精度字幕系统设计文档.md](Vocal_Subtitle-离线高精度字幕系统设计文档.md) | 🔧 部分实施 | 高精度字幕系统设计 |

## 二、架构与运维文档

| 文档 | 状态 | 说明 |
|------|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | ✅ 已实施 | 技术架构文档（含组件化后模块索引） |
| [ARCHITECTURE_STATE.md](ARCHITECTURE_STATE.md) | ✅ 已实施 | 架构状态表（版本、配置、依赖锁定） |
| [DEPLOYMENT.md](DEPLOYMENT.md) | ✅ 已实施 | 部署指南 |
| [API参考文档.md](API参考文档.md) | 🔧 部分实施 | API 参考（需随组件化重构更新） |
| [场景模板使用指南.md](场景模板使用指南.md) | ✅ 已实施 | 5 种场景模板说明 |

## 三、治理规范文档（2026-08-02 新建）

| 文档 | 状态 | 说明 |
|------|------|------|
| [ARCHITECTURE_STATE.md](ARCHITECTURE_STATE.md) | ✅ 已实施 | 架构状态表（版本、配置、依赖锁定、已知限制） |
| [NAMING_CONVENTIONS.md](NAMING_CONVENTIONS.md) | ✅ 已实施 | 命名规范与运行标识约定 |
| [DATA_ASSETS.md](DATA_ASSETS.md) | ✅ 已实施 | 数据资产登记表（D0 已登记，D1-D4 待建立） |
| [RUN_REPORT_SCHEMA.md](RUN_REPORT_SCHEMA.md) | ✅ 已实施 | 统一运行报告 JSON Schema |
| [TASK_STATE_MACHINE.md](TASK_STATE_MACHINE.md) | ✅ 已实施 | 任务状态机（pending→running→completed/failed） |
| [ENGINE_LIFECYCLE.md](ENGINE_LIFECYCLE.md) | ✅ 已实施 | 引擎生命周期、注册表、降级行为矩阵 |
| [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md) | ✅ 已实施 | 实验注册表（6 个已注册实验） |
| [CONTRACTS_EVIDENCE_DECISION_PROJECTION.md](CONTRACTS_EVIDENCE_DECISION_PROJECTION.md) | ✅ 已实施 | 证据/决策/投影三层契约 |
| [FEEDBACK_LOOP.md](FEEDBACK_LOOP.md) | 🔧 部分实施 | 反馈学习闭环（pipeline/CLI/WebUI 已接入 D2 入库 + D3 抽样；前端审核界面待开发） |
| [QUALITY_OPERATIONS.md](QUALITY_OPERATIONS.md) | 🔧 部分实施 | 质量运营规范（代码已实施，运营数据待积累；136 测试覆盖） |
| [RELEASE_GOVERNANCE.md](RELEASE_GOVERNANCE.md) | 🔧 部分实施 | 发布治理（检查单 + 观测 + 回滚已实施，流程待执行） |

### 测试覆盖 (2026-08-03 新建)

| 文件 | 说明 |
|------|------|
| `tests/test_governance.py` | 治理模块测试 (36 tests): EngineLifecycle, EngineRegistry, ExperimentRegistry, ReleaseManager |
| `tests/test_reporting.py` | 报告模块测试 (28 tests): RunReportBuilder, EngineAvailabilityChecker, DegradationLogger |
| `tests/test_quality_module.py` | 质量模块测试 (50 tests): IssueClassifier, SceneSlicer, PriorityCalculator, TrendReporter, ControlledRunner, DataVersionManager |
| `tests/test_task_history.py` | 任务历史测试 (22 tests): 状态机转换, 错误分类, 预检清单, CRUD |

## 四、ADR 架构决策记录

| 文档 | 状态 | 说明 |
|------|------|------|
| [adr/001-unified-decision-export.md](../adr/001-unified-decision-export.md) | ✅ 已实施 | 统一决策出口 |
| [adr/002-physical-boundary-priority.md](../adr/002-physical-boundary-priority.md) | ✅ 已实施 | 物理边界优先 |
| [adr/003-feedback-shadow-consumption.md](../adr/003-feedback-shadow-consumption.md) | 🔧 部分实施 | 反馈影子消费 |
| [adr/004-dataset-tiering.md](../adr/004-dataset-tiering.md) | 🔧 部分实施 | 数据集分层 |

## 五、专项优化方案

| 文档 | 状态 | 说明 |
|------|------|------|
| [字幕时间轴精度优化方案.md](字幕时间轴精度优化方案.md) | ✅ 已实施 | Plan 4-7 时间轴精度优化 |
| [时间轴精度修复方案.md](时间轴精度修复方案.md) | ✅ 已实施 | 边界精修修复方案 |
| [时间轴精度修复-实施计划.md](时间轴精度修复-实施计划.md) | ✅ 已实施 | 修复方案实施计划 |
| [boundary_redundancy.md](boundary_redundancy.md) | ✅ 已实施 | 边界滑动窗口冗余识别 (Plan 4 扩展) |
| [pipeline-optimization-plan.md](pipeline-optimization-plan.md) | 🔧 部分实施 | 管线数据流优化方案 |
| [speaker-diarization-plan.md](speaker-diarization-plan.md) | ✅ 已实施 | 说话人分离方案对比 |
| [speaker-embedding-guide.md](speaker-embedding-guide.md) | ✅ 已实施 | 说话人嵌入模型配置指南 |

## 六、分析与报告

| 文档 | 状态 | 说明 |
|------|------|------|
| [全链路验收报告-2026-07-26.md](全链路验收报告-2026-07-26.md) | 📖 参考 | 2026-07-26 全链路验收 |
| [测试与修复报告-2026-07-06.md](测试与修复报告-2026-07-06.md) | 📖 参考 | 2026-07-06 测试与修复 |
| [复杂音频ASR代码审计与优化方案-2026-07-29.md](复杂音频ASR代码审计与优化方案-2026-07-29.md) | 📖 参考 | 复杂音频 ASR 审计 |
| [analysis_complex_audio_asr_degradation.md](analysis_complex_audio_asr_degradation.md) | 📖 参考 | 复杂场景 ASR 质量下降分析 |
| [方案B-自适应声学与VAD优化方案-2026-07-29.md](方案B-自适应声学与VAD优化方案-2026-07-29.md) | 📋 待实施 | 方案 B：自适应声学与 VAD |
| [字幕识别浏览器实测优化方案-2026-07-27.md](字幕识别浏览器实测优化方案-2026-07-27.md) | 📖 参考 | 浏览器实测优化 |
| [字幕识别浏览器三音频三引擎分析报告-2026-07-29.md](字幕识别浏览器三音频三引擎分析报告-2026-07-29.md) | 📖 参考 | 三音频三引擎对比分析 |
| [[ASR 优化设计方案].md]([ASR%20优化设计方案].md) | 📖 参考 | ASR 准确率与字幕安全优化 |

## 七、反馈学习

| 文档 | 状态 | 说明 |
|------|------|------|
| [基于用户反馈的自适应学习机制.md](基于用户反馈的自适应学习机制.md) | 🔧 部分实施 | 反馈学习工程方案 |
| [phase-five-default-path-development.md](phase-five-default-path-development.md) | ✅ 已实施 | Phase 5 全局 ASR 默认路径开发说明 |

## 八、Superpowers 设计规范（specs/）

### Phase 0-2: 物理优先基础设施

| 文档 | 状态 | 说明 |
|------|------|------|
| [specs/2026-07-25-phase-zero-physical-first-design.md](superpowers/specs/2026-07-25-phase-zero-physical-first-design.md) | ✅ 已实施 | 阶段零：物理边界优先设计 |
| [specs/2026-07-25-phase-one-hallucination-filter-design.md](superpowers/specs/2026-07-25-phase-one-hallucination-filter-design.md) | ✅ 已实施 | 阶段一：幻觉过滤设计 |
| [specs/2026-07-25-phase-two-physical-timeline-design.md](superpowers/specs/2026-07-25-phase-two-physical-timeline-design.md) | ✅ 已实施 | 阶段二 P2.1：物理时间线核心模型 |
| [specs/2026-07-25-phase-two-evidence-adapter-design.md](superpowers/specs/2026-07-25-phase-two-evidence-adapter-design.md) | ✅ 已实施 | 阶段二 P2.2：证据适配器 |
| [specs/2026-07-25-phase-two-global-ir-design.md](superpowers/specs/2026-07-25-phase-two-global-ir-design.md) | ✅ 已实施 | 阶段二 P2.3：全局 IR 契约 |
| [specs/2026-07-25-phase-two-coordinate-context-cache-design.md](superpowers/specs/2026-07-25-phase-two-coordinate-context-cache-design.md) | ✅ 已实施 | 阶段二 P2.4：坐标与缓存 |

### Phase 3-4: 字幕构建与边界优化

| 文档 | 状态 | 说明 |
|------|------|------|
| [specs/2026-07-26-physical-bin-subtitle-filling-design.md](superpowers/specs/2026-07-26-physical-bin-subtitle-filling-design.md) | ✅ 已实施 | 物理字幕仓与文字灌装 |
| [specs/2026-07-26-subtitle-punctuation-and-fragment-integrity-design.md](superpowers/specs/2026-07-26-subtitle-punctuation-and-fragment-integrity-design.md) | ✅ 已实施 | 标点归属与碎片完整性 |
| [specs/2026-07-27-acoustic-release-gates-design.md](superpowers/specs/2026-07-27-acoustic-release-gates-design.md) | ✅ 已实施 | 声学发布门禁 |
| [specs/2026-07-28-physical-boundary-gating-design.md](superpowers/specs/2026-07-28-physical-boundary-gating-design.md) | ✅ 已实施 | 物理边界方向与置信度门控 |

### Phase 5: 全局 ASR 与生产链

| 文档 | 状态 | 说明 |
|------|------|------|
| [specs/2026-07-26-phase-five-default-path-design.md](superpowers/specs/2026-07-26-phase-five-default-path-design.md) | ✅ 已实施 | 阶段五：全局 ASR 默认路径 |
| [specs/2026-07-28-global-asr-timeline-design.md](superpowers/specs/2026-07-28-global-asr-timeline-design.md) | ✅ 已实施 | Global ASR 时间轴方案 |
| [specs/2026-07-29-phase-a-global-asr-implementation-design.md](superpowers/specs/2026-07-29-phase-a-global-asr-implementation-design.md) | ✅ 已实施 | 方案 A：Global ASR 主路径接入 |

### 生产补齐与组件化

| 文档 | 状态 | 说明 |
|------|------|------|
| [specs/2026-07-26-full-chain-completion-design.md](superpowers/specs/2026-07-26-full-chain-completion-design.md) | ✅ 已实施 | 全链路完成设计 |
| [specs/2026-07-26-production-completion-design.md](superpowers/specs/2026-07-26-production-completion-design.md) | ✅ 已实施 | 生产补齐设计 |
| [specs/2026-07-26-production-completion-followup-design.md](superpowers/specs/2026-07-26-production-completion-followup-design.md) | ✅ 已实施 | 生产补齐后续：CLI/WebUI 契约 |
| [specs/2026-08-02-p0-offline-production-convergence-design.md](superpowers/specs/2026-08-02-p0-offline-production-convergence-design.md) | 🏗️ 进行中 | P0 离线生产链收敛 |

### 说话人与质量

| 文档 | 状态 | 说明 |
|------|------|------|
| [specs/2026-07-27-global-subtitle-quality-design.md](superpowers/specs/2026-07-27-global-subtitle-quality-design.md) | ✅ 已实施 | 影视级字幕质量优化 |
| [specs/2026-07-27-high-precision-speaker-fusion-design.md](superpowers/specs/2026-07-27-high-precision-speaker-fusion-design.md) | ✅ 已实施 | 高精度说话人融合 |
| [specs/2026-07-27-speaker-chain-repair-design.md](superpowers/specs/2026-07-27-speaker-chain-repair-design.md) | ✅ 已实施 | 双人说话人链路修复 |

### WebUI 与交互

| 文档 | 状态 | 说明 |
|------|------|------|
| [specs/2026-07-24-deployment-design.md](superpowers/specs/2026-07-24-deployment-design.md) | ✅ 已实施 | 一键部署设计 |
| [specs/2026-07-24-hf-model-download-design.md](superpowers/specs/2026-07-24-hf-model-download-design.md) | ✅ 已实施 | HF 模型下载设计 |
| [specs/2026-07-24-llm-asr-language-fix-design.md](superpowers/specs/2026-07-24-llm-asr-language-fix-design.md) | ✅ 已实施 | ASR 语言检测修复 |
| [specs/2026-07-24-subtitle-timeline-comparison-design.md](superpowers/specs/2026-07-24-subtitle-timeline-comparison-design.md) | ✅ 已实施 | 字幕时间轴对比 |
| [specs/2026-07-27-hf-token-model-cache-validation-design.md](superpowers/specs/2026-07-27-hf-token-model-cache-validation-design.md) | ✅ 已实施 | HF Token 与模型缓存 |
| [specs/2026-07-28-funasr-auto-prepare-design.md](superpowers/specs/2026-07-28-funasr-auto-prepare-design.md) | ✅ 已实施 | FunASR 自动准备 |
| [specs/2026-07-28-password-manager-prompt-design.md](superpowers/specs/2026-07-28-password-manager-prompt-design.md) | ✅ 已实施 | 密码管理器提示修复 |
| [specs/2026-07-28-subtitle-batch-edit-design.md](superpowers/specs/2026-07-28-subtitle-batch-edit-design.md) | ✅ 已实施 | 字幕批量编辑 |
| [specs/2026-07-28-subtitle-row-interactions-design.md](superpowers/specs/2026-07-28-subtitle-row-interactions-design.md) | ✅ 已实施 | 字幕行级交互 |
| [specs/2026-07-29-asr-language-routing-and-engine-optimization-design.md](superpowers/specs/2026-07-29-asr-language-routing-and-engine-optimization-design.md) | ✅ 已实施 | ASR 语言路由与引擎优化 |
| [specs/2026-07-29-asr-runtime-repair-design.md](superpowers/specs/2026-07-29-asr-runtime-repair-design.md) | ✅ 已实施 | ASR 运行时修复 |

## 九、Superpowers 实施计划（plans/）

| 文档 | 状态 | 说明 |
|------|------|------|
| [plans/2026-07-25-phase-one-hallucination-filter-plan.md](superpowers/plans/2026-07-25-phase-one-hallucination-filter-plan.md) | ✅ 已完成 | 幻觉过滤实施 |
| [plans/2026-07-26-full-chain-completion-plan.md](superpowers/plans/2026-07-26-full-chain-completion-plan.md) | ✅ 已完成 | 全链路完成实施 |
| [plans/2026-07-26-phase-five-default-path-plan.md](superpowers/plans/2026-07-26-phase-five-default-path-plan.md) | ✅ 已完成 | 全局 ASR 默认路径实施 |
| [plans/2026-07-27-global-subtitle-quality-plan.md](superpowers/plans/2026-07-27-global-subtitle-quality-plan.md) | ✅ 已完成 | 影视级字幕质量实施 |
| [plans/2026-07-27-high-precision-speaker-fusion-plan.md](superpowers/plans/2026-07-27-high-precision-speaker-fusion-plan.md) | ✅ 已完成 | 高精度说话人融合实施 |
| [plans/2026-07-29-asr-language-routing-and-engine-optimization-plan.md](superpowers/plans/2026-07-29-asr-language-routing-and-engine-optimization-plan.md) | ✅ 已完成 | ASR 语言路由实施 |
| [plans/2026-08-02-p0-offline-production-convergence-plan.md](superpowers/plans/2026-08-02-p0-offline-production-convergence-plan.md) | 🏗️ 进行中 | P0 离线生产链收敛 |

## 十、Superpowers 验收报告（reports/）

| 文档 | 状态 | 说明 |
|------|------|------|
| [reports/2026-07-27-acoustic-release-gates-acceptance.md](superpowers/reports/2026-07-27-acoustic-release-gates-acceptance.md) | 📖 参考 | 声学发布门禁验收 |
| [reports/2026-07-27-global-subtitle-quality-code-audit.md](superpowers/reports/2026-07-27-global-subtitle-quality-code-audit.md) | 📖 参考 | 影视级字幕质量代码审计 |
| [reports/2026-07-27-task-15-acceptance.md](superpowers/reports/2026-07-27-task-15-acceptance.md) | 📖 参考 | 任务 15 验收报告 |
| [reports/2026-07-31-large-file-componentization-implementation.md](superpowers/reports/2026-07-31-large-file-componentization-implementation.md) | 📖 参考 | 大文件组件化实施报告 |
| [reports/2026-08-02-offline-full-chain-runtime-validation.md](superpowers/reports/2026-08-02-offline-full-chain-runtime-validation.md) | 📖 参考 | 离线全链路运行时验收报告 |

## 十一、存档中的过期文档（不在当前目录，仅 tar.gz 中）

以下文档已被更新的方案替代，仅保留在 `docs.tar.gz` 存档中：

| 文档 (tar.gz 内路径) | 被替代原因 |
|----------------------|-----------|
| `superpowers/plans/2026-08-01-offline-subtitle-full-chain-implementation-plan.md` | 被 2026-08-02 整体落实计划替代 |
| `superpowers/specs/2026-07-30-global-asr-context-reasr-design.md` | 被 08-02 离线字幕全链路方案融合 |
| `superpowers/specs/2026-07-31-global-asr-multi-engine-componentized-design.md` | 被 08-02 方案替代 |
| `superpowers/specs/2026-07-31-global-asr-multi-engine-review-design.md` | 被 08-02 方案替代 |
| `superpowers/specs/2026-07-31-large-file-componentization-design.md` | 已实施完成（见实施报告） |
| `superpowers/specs/2026-07-31-large-file-componentization-completion-design.md` | 已实施完成 |
| `superpowers/specs/2026-08-01-offline-multi-engine-production-chain-design.md` | 被 08-02 方案融合 |
| `superpowers/specs/2026-08-01-offline-subtitle-full-chain-convergence-design.md` | 被 08-02 方案替代 |
