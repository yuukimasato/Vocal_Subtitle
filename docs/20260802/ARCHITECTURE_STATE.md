# 架构状态表 (Architecture State Table)

**更新日期**: 2026-08-02
**基于分支**: `refactor/componentize-large-files`

## 1. 版本信息

| 项目 | 值 |
|------|-----|
| 项目版本 | 0.1.0 (Alpha) |
| Python | >= 3.10 (支持 3.10-3.12) |
| 配置版本 | default.yaml (v1, 无显式 schema 版本) |
| 核心包行数 | ~33,000 |
| 测试覆盖 | ~6,178 行测试代码，~100+ 测试文件 |

## 2. 默认配置概要

| 配置域 | 默认值 | 备注 |
|--------|--------|------|
| 运行模式 | offline | streaming 只保持兼容 |
| 分离引擎 | UVR (BS-RoFormer) | ONNX 推理 |
| VAD 引擎 | Silero VAD | 阈值 0.40 |
| ffmpeg VAD | 并行启用 | Plan 1 |
| 三方法融合 | 关闭 | Plan 2, 实验阶段 |
| 宏观切块 | 自动启用 (>180s) | Plan 0 |
| 声学校验 | 启用 | Plan 7, 骨架模式 |
| ASR 引擎 | auto (默认 faster-whisper large-v3) | 支持 FunASR/Qwen/whisper.cpp |
| 引擎配对 | 启用, policy=risk_only | primary/secondary 自动选择 |
| 全局 ASR 证据 | 启用 (shadow_mode) | 不直接影响默认输出 |
| Context Re-ASR | 关闭 | 实验阶段 |
| Qwen 复核 | 关闭 | 实验阶段 |
| ForcedAligner | 关闭 | 实验阶段 |
| SED | 关闭 | 实验阶段 |
| 语义复核 | 关闭 | 实验阶段 |
| 说话人分离 | 启用 (agglomerative + ECAPA) | |
| 语义合并 | cascading (三层级联) | 规则 → 本地 NLP → 云端 LLM |
| 帧级无缝 | 启用 | Plan 6 |
| LLM 后处理 | 关闭 | 需 API Key |
| 噪声抑制 | 关闭 | 纯净录音不需要 |

## 3. 依赖与模型锁定

| 类别 | 依赖/模型 | 版本 |
|------|-----------|------|
| ASR 核心 | faster-whisper | >= 1.0 (CTranslate2 >= 4.0) |
| ASR 中文 | FunASR | >= 1.0 |
| ASR 复核 | qwen-asr | == 0.0.6 (固定, 不兼容 Transformers 5.x) |
| Transformers | | == 4.57.6 (与 qwen-asr 配套锁定) |
| VAD | Silero VAD | ONNX 模型, ~1.5MB |
| 分离 | audio-separator | >= 0.18 |
| 说话人 | speechbrain | >= 1.0 (ECAPA-TDNN, Apache 2.0) |
| 语义合并 | sentence-transformers | >= 3.0 |
| LLM 客户端 | openai | >= 1.0 |
| 缓存 | diskcache | >= 5.6 |
| 日志 | structlog | >= 23.0 |
| CLI | click | >= 8.0 |
| Web | fastapi + uvicorn + websockets | >= 0.110 |

### 复核模型注册表

| 用途 | HuggingFace 仓库 | 默认路径 |
|------|-----------------|---------|
| Qwen3-ASR 1.7B | Qwen/Qwen3-ASR-1.7B | ~/.cache/vocal-subtitle/review-models/qwen3-asr-1.7b |
| Qwen3-ASR 0.6B | Qwen/Qwen3-ASR-0.6B | ~/.cache/vocal-subtitle/review-models/qwen3-asr-0.6b |
| ForcedAligner 0.6B | Qwen/Qwen3-ForcedAligner-0.6B | ~/.cache/vocal-subtitle/review-models/ |
| SED AST | MIT/ast-finetuned-audioset-10-10-0.4593 | ~/.cache/vocal-subtitle/review-models/ |

## 4. 支持平台

| 平台 | 状态 | 备注 |
|------|------|------|
| Ubuntu 20.04+ | 主要支持 | CI 运行环境 |
| Ubuntu 22.04 LTS | 推荐 | 生产环境 |
| macOS 12+ | 支持 | 开发环境 |
| Windows | 未正式测试 | 理论兼容，ffmpeg 路径需处理 |
| Docker | 未提供 | 可自行构建 |

## 5. 已知限制

| 限制 | 影响范围 | 计划 |
|------|----------|------|
| 多引擎复核（Qwen/FunASR/SED）仅 shadow 模式 | 高质量场景需手动启用 | Phase D 建立引擎生命周期后逐步开放 |
| 反馈学习默认不消费 | 自动优化闭环未形成 | Phase E 建立反馈审核和影子消费 |
| 无正式 D3 反馈回归集 | 版本质量对比缺少真实数据 | Phase E 从真实授权反馈中积累 |
| 长音频 (>1h) 资源消耗大 | 大文件处理时间长、内存/显存峰值高 | Phase B 建立资源边界 |
| 多人重叠/高噪声场景质量待提升 | 复杂场景准确率不足 | Phase F 按真实需求优化 |
| 测试素材未授权为黄金集 | 不满足正式发布认证 | ADR-004 建立 D0-D4 分层 |
| 缺少模型可用性预检 | 首次运行可能因模型缺失失败 | Phase B 统一预检 |
| 流式模式仅保持兼容 | 流式路径未经充分测试 | 不纳入本计划范围 |

## 6. 生产档位

| 档位 | 配置 | 适用场景 |
|------|------|----------|
| `quality-first` (默认) | segmented primary + global evidence + 保守复核 + 物理/声学校验 | 离线批量生产 |
| `degraded` | 关闭 LLM 调用，仅本地规则 | 无 API Key / 网络受限 |
| `minimal` | 仅 VAD + ASR + 规则合并 | 快速预览 / 低资源设备 |

## 7. 关键版本标识符

| 标识符 | 用途 | 当前值 |
|--------|------|--------|
| `route_version` | ASR 路由策略版本 | asr-route-v1 |
| `quality_gate_version` | ASR 质量门禁版本 | asr-quality-v1 |
| `review_policy_version` | 复核策略版本 | review-policy-v1 |
| `risk_policy_version` | 风险策略版本 | risk-policy-v1 |
| `decision_policy_version` | 决策策略版本 | decision-policy-v1 |
| `evidence_schema_version` | 证据模型版本 | evidence-v1 |
| `golden_quality_gate_version` | 黄金质量门禁版本 | golden-quality-v1 |
