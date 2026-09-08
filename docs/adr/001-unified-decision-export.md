# ADR-001: 统一决策出口

**状态**: 已实施
**日期**: 2026-08-02
**决策者**: yuukimasato

## 背景

离线字幕生产链中，多个阶段可能生成或修改字幕文本候选：分段主 ASR、global ASR 证据、异质引擎复核（Qwen、FunASR）、后处理合并/拆分等。若无统一出口，将导致：
- 无法追溯最终字幕的文本来源
- 不同路径可能绕过物理边界约束
- 后处理可能篡改 ASR 输出而不留痕迹

## 决策

1. **`EvidenceDecision` 是唯一的文本选择出口**：所有候选（segmented primary、global evidence、review engines）必须经过 `EvidenceDecision` 做出 keep/replace/split/drop/unresolved 动作，任何模块不得绕过此出口直接写入最终字幕文本。

2. **`DecisionEventProjector` 是唯一的事件投影出口**：所有最终字幕事件（SubtitleEvent）必须通过 `DecisionEventProjector` 从决策结果投影生成，包含 projection trace。

3. **原始候选保留在证据层**：`EvidenceDecision` 的输入（候选证据）和输出（决策动作）均持久化或可重建，允许事后审计。

4. **降级路径也必须留痕**：引擎不可用、超时或决策失败时的降级输出（如直接使用 segmented primary）须标记 `production_path=segmented_fallback` 和降级原因。

## 后果

- 新增或替换 ASR 引擎时只需接入证据层，不改变投影逻辑
- 所有最终字幕 cue 可追溯到其候选来源和决策理由
- 禁止任何模块通过 raw bypass 直接写入字幕事件
- 维护者必须理解证据/决策/投影三层模型后才能添加新路径

## 相关

- [ADR-002: 物理边界优先](./002-physical-boundary-priority.md)
- [ADR-004: 数据集分层](./004-dataset-tiering.md)
