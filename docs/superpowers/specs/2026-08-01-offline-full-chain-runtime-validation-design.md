# 离线字幕全链路运行验证设计

日期：2026-08-01  
范围：离线字幕生产链的可运行闭环、CLI、Web 后端/前端验证；黄金集质量门禁最后执行。  
依据：`docs/离线字幕全链路优化与链路组合方案-2026-08-01.md`。

## 1. 目标与边界

首要目标是让现有离线生产链在当前环境完成一次可重复的真实音频闭环，并能被 CLI、Web 后端和前端共同调用。闭环成功不等于质量通过：当前环境可用的 whisper.cpp tiny 仅用于 `runnable` smoke，不能替代 faster-whisper large-v3、FunASR 或 Qwen 的生产配对验收。

本次不重设计流式链路，不把 global ASR、Qwen、LLM 或单一声学证据变成最终字幕来源，不改变默认发布主模型，不在黄金集之前把 FunASR/Qwen 标记为发布准入。

## 2. 生产职责与数据流

统一离线链路为：

```text
CLI / WebUI
  -> Pipeline.run
  -> 分离（可跳过）/物理骨架/VAD/噪声画像
  -> 分段 ASR 主候选
  -> global ASR evidence
  -> OfflineProductionCoordinator
       -> risk scoring
       -> Context Re-ASR / 可选异质副引擎
       -> EvidenceDecision
       -> DecisionEventProjector
  -> 说话人/语义后处理
  -> 合并后声学校验
  -> final validator
  -> SRT/VTT/ASS + 诊断
```

固定不变量：

- `Pipeline.run` 是 CLI、WebUI 和脚本共用的任务入口。
- 分段 ASR 是主候选来源；global、Context Re-ASR、Qwen 和其他审查器只能提供证据。
- `EvidenceDecision` 是唯一允许改变候选的决策出口。
- `DecisionEventProjector` 是唯一把决策事件投影到物理时间线的出口。
- `unresolved` 保留主候选并记录原因；模型缺失、超时和投影失败不能静默处理。
- 没有决策追踪或物理范围的事件不得绕过投影直接导出。

## 3. 运行状态和错误契约

跨入口统一使用以下运行状态：

| 状态 | 含义 |
|---|---|
| `completed` | 主链完成并导出结果 |
| `degraded` | 主链完成，但发生明确 fallback、模型不可用或复核未完成 |
| `failed` | 无法生成合法结果 |

能力发布状态独立记录为：`implemented`、`runnable`、`benchmarked`、`gated`、`release-default`。不得用前一状态替代后一状态。

错误必须保留结构化类别：`input_invalid`、`dependency_unavailable`、`model_unavailable`、`engine_failed`、`empty_candidate`、`projection_failed`、`export_failed`。

降级规则：

- global evidence、Context Re-ASR、Qwen 等可选能力失败时保留主候选，状态为 `degraded`，记录 `fallback_reason`。
- 主 ASR 质量门禁失败时最多回退一次，并同时保留 primary/fallback 诊断。
- 物理投影失败时不得 raw bypass；可回退到主候选则标记 `projection_failed`，否则任务失败。
- 缓存键必须包含输入、配置、主/副引擎、模型和 route/policy/projection 版本，避免历史 smoke 结果污染生产结果。

跨入口结果至少包含：

```json
{
  "status": "completed|degraded|failed",
  "subtitle_path": "...",
  "stats": {},
  "events": [],
  "diagnostics": {},
  "quality_status": "pass|fail|unverified"
}
```

## 4. 实施与验证顺序

### 4.1 全链路 smoke

以现有 `Pipeline.run` 和 `OfflineProductionCoordinator` 为主，修复实际运行中发现的状态透传、诊断完整性、缓存隔离和失败处理问题。使用本地 whisper.cpp tiny 与 `test/golden/repeated_phrase_me.wav`，输出到独立临时目录。

通过条件是：输入、物理准备、分段 ASR、EvidenceReview、投影、后处理和 SRT/VTT/ASS 导出均能完成；输出可解析，诊断完整，无 raw bypass。允许 `degraded`，不要求质量通过。

### 4.2 CLI

通过 subprocess 执行真实 `vocal_subtitle run` 和 batch 入口，验证参数覆盖、输出文件、退出码、状态和输出路径。CLI 不自行重建生产链，只展示 `Pipeline` 的统一结果。

### 4.3 Web 后端与前端

后端验证上传、任务创建、进度/完成事件、任务结果和字幕下载。前端通过真实本地服务验证上传到结果展示的主流程，并明确区分 `completed`、`degraded` 和 `failed`。测试使用临时目录，不覆盖已有缓存和用户文件。

### 4.4 黄金集

最后使用生产配对模式生成带主/副引擎、模型、route、review policy、projection 版本等元数据的输入，再运行 golden gate。报告必须区分 `runnable`、`benchmarked` 和 `gated`；tiny smoke 只能支持前两者，不能伪造质量通过。

## 5. 测试层次

```text
现有契约测试
  -> 本地真实离线 smoke
  -> CLI subprocess
  -> Web backend integration
  -> 前端浏览器 smoke
  -> 生产配对黄金集 + quality gate
```

主要修改范围限定在 `application/pipeline_runner.py`、`application/offline_production.py`、`application/pipeline_result.py`、`cli.py`、`webui/pipeline_tasks.py`、`webui/models.py` 及其测试；不做无关重构，不切换默认主模型。

## 6. 验收输出

每一阶段保留可重复命令、测试结果和诊断报告。最终报告必须同时说明：代码是否已实现、当前环境是否可运行、是否有生产模型基准、是否通过黄金集门禁，以及哪些能力仍不得进入默认发布配置。
