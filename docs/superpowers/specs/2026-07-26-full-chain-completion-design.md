# Vocal_Subtitle 全链路完成设计

日期：2026-07-26

状态：已获用户批准，待实施计划与代码实施

## 1. 背景与目标

阶段零到阶段五已经在仓库中形成了大部分后端、物理时间线、全局 ASR、默认路径、CLI 和 WebUI 实现，但当前状态仍存在三类问题：

1. 既有回归测试有 5 个失败，主要涉及默认配置契约和 LLM 字段回写。
2. 真实音频与人工字幕位于 `test/`，尚未接入现有 benchmark 发现场景，无法形成可重复的质量验收。
3. 后端路径状态、缓存、CLI、API、WebSocket、前端展示、部署说明和主设计文档还需要进行一次全链路一致性核对。

本设计的目标是完成设计文档 §16 的代码级闭环，并用 `test/` 中的音频和人工 ASS 产生真实素材质量报告。模型不可用或全局路径不可用时，必须明确记录降级，而不是将旧路径结果伪装成全局路径质量结果。

## 2. 范围与非目标

### 2.1 范围

- 修复当前测试暴露的配置和 LLM 字段回归。
- 核对并补齐物理时间线、词级分配、幻觉过滤、最终物理校验和全局 ASR 的接口契约。
- 统一 Pipeline、缓存、CLI、WebUI API、WebSocket、任务历史和静态页面中的 ASR 路径与降级诊断。
- 核对 WhisperX optional extra、延迟导入、全局成功/失败/降级行为，以及 faster-whisper legacy 路径。
- 建立真实素材 manifest，配对 `test/` 音频与人工 ASS，输出可审计的时间、文本、说话人和物理边界指标。
- 更新主设计文档、开发计划、README、部署/API 说明和验收报告。

### 2.2 非目标

- 不删除或重置工作区中已有的用户改动。
- 不把真实素材 benchmark 的结果硬编码为发布通过；素材指标是否达标由报告决定。
- 不强制下载大型 ASR、对齐、diarization 或分离模型；测试应优先复用本地缓存，缺少模型时记录可诊断的降级。
- 不在本次工作中重写现有 Pipeline 或重新设计 WebUI 视觉风格。

## 3. 统一架构

离线流程保持以下边界：

```text
输入
  -> 配置与路径策略
  -> 分离/VAD/物理时间线与语音证据
  -> 全局 diarization
  -> global ASR 或 segmented legacy ASR
  -> 词级物理范围与 speaker 分配
  -> 严格断句与字幕事件
  -> LLM 文本优化（只改文本）
  -> 最终物理校验
  -> SRT/VTT/ASS、JSON 诊断、API/CLI/WebUI 状态
```

`PhysicalClip` 是唯一的物理归属硬边界；`SpeechEvidenceSpan` 只表示检测证据；`ContextWindow` 只为识别提供上下文，不能取得字幕归属权。后处理、LLM 合并、最小时长和导出均不得扩大 `PhysicalClip`。

## 4. 稳定契约

### 4.1 ASR 路径

- `auto`：离线默认优先 global；全局依赖、资源、执行或结果失败时完整回退 segmented。
- `global`：要求 global 成功；失败时抛出带分类和原因的错误，不静默回退。
- `segmented`：显式使用现有分段路径。
- streaming：固定使用 segmented，不加载 WhisperX。

结果至少暴露：

- `asr_path`: `global`、`legacy` 或 `legacy_degraded`
- `global_attempted`
- `fallback_category`
- `fallback_reason`
- `global_diagnostics`

### 4.2 字幕事件

保留既有 `text`、`original_text`、`asr_text`、`llm_text` 和 speaker 字段，并保证新增字段在所有映射、合并、序列化和导出路径中不丢失：

- `physical_spans`
- `source_word_ids`
- `logical_sentence_id`
- `alignment_warning`

LLM 只能返回经过校验的文本。时间、speaker、physical spans、source word IDs 和物理归属不能由 LLM 改写。

### 4.3 配置与缓存

配置默认值必须同时满足现有兼容测试、设计文档和部署行为。全流水线与 ASR 缓存身份至少区分 ASR 路径、全局配置、diarization speaker 上限、过滤器版本和关键模型参数，避免 global/segmented 结果互相命中。

## 5. 前后端与 CLI 闭环

后端 Pipeline 负责产生唯一的路径和诊断状态。API、WebSocket complete 事件、任务历史和 CLI 直接透传该状态。WebUI 的路径选择、任务提交、进度完成、结果统计和降级提示均消费显式字段，不通过缺失字段或事件数量猜测状态。

兼容要求：旧请求不传 `asr_path` 时继续按 profile/default 行为运行；旧字幕事件缺少新增字段时按空值处理；现有 SRT/VTT/ASS 输出格式不改变。

## 6. 真实素材质量验收

建立明确的素材清单，至少覆盖当前 `test/` 中可配对的中文双人、中文多人、英文多人、培训双人、单人和视频评测素材。每条 manifest 记录音频、人工 ASS、语言、speaker 数量和场景类别。

每次对比输出：

- 自动字幕和人工字幕的事件匹配数、未匹配数。
- 起始/结束时间 MAE、P95、超过 300ms/500ms 的比例。
- 文本 CER/WER 或 CJK 字符级相似度。
- speaker 标签数量、切换位置、unknown/mixed 比例。
- 物理边界越界数量、来源词缺失数量、alignment warning 数量。
- 实际 ASR 路径、fallback 分类、耗时和缓存命中信息。

global 与 segmented 的结果必须分别保存。WhisperX 或模型不可用时，报告标记为“代码链路验证/legacy 降级”，不计入 global 质量通过。

## 7. 错误处理与可观测性

所有降级原因采用稳定分类：`dependency_unavailable`、`resource_unavailable`、`execution_failed`、`invalid_result`、`explicit_legacy`。错误信息包含下一步操作建议，但不包含 token、凭据或完整敏感路径。

最终物理校验失败时保留诊断数据并阻止非法事件进入导出；可恢复的全局失败才允许 `auto` 回退。缓存恢复结果必须带原始路径和诊断，不能只恢复字幕文本。

## 8. 测试策略

1. 无模型单元测试：数据契约、路由、缓存隔离、物理分配、幻觉过滤、LLM guard、API/CLI 序列化。
2. 全量回归：修复当前 5 个失败后运行完整 `venv/bin/pytest`。
3. 链路测试：模拟 global 成功、依赖缺失、执行失败、显式 legacy，覆盖 Pipeline、CLI、API 和 WebSocket。
4. 真实素材测试：在可用本地模型/缓存条件下运行 manifest；没有模型时仍生成明确的不可发布报告。
5. 静态校验：`git diff --check`、Python 编译/导入、Shell `bash -n`、CLI help 和 benchmark CI 空素材行为。

## 9. 完成标准

- 全量无模型测试通过。
- 当前默认配置和 LLM 阶段字段回归修复。
- global/legacy/降级三种路径在后端、缓存、CLI、API、WebSocket、WebUI 和任务历史中状态一致。
- 每个进入最终字幕的物理事件可追溯，越界事件被最终校验阻止。
- `test/` 真实素材可一键生成对比报告；报告清楚区分 global 成功、legacy 降级和模型不可用。
- 主设计文档不再把已实现、代码级完成和真实素材验收完成混为一谈。
- README、部署/API 文档、开发计划和验收报告与实际命令一致。

