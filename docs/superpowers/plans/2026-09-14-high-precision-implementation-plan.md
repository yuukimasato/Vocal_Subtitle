# 高精度时间轴与识别文本实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏现有基线行为的前提下，把 global context transcription、WhisperX forced alignment、统一 EvidenceDecision 和物理边界投影接入高精度离线字幕主链。

**Architecture:** 保留 faster-whisper large-v3 作为可回退基线；新增“全局上下文识别 → 词级强制对齐 → 证据决策 → 物理投影”的可开关路径。所有阶段先以 shadow 或实验配置运行，只有通过基线回归和黄金集门禁后才允许改变默认生产行为。

**Tech Stack:** Python 3.10+, pytest, NumPy, faster-whisper, WhisperX（可选依赖）, PyYAML, 现有 PhysicalTimeline/EvidenceDecision 数据结构。

**Spec:** [优化方案_高精度时间轴与识别文本.md](../../高精度时间轴与识别文本优化方案-2026-09-14.md)

## Global Constraints

- 基线提交为 `d3d49d1`；开始前必须保存工作树状态，不得覆盖用户未提交修改。
- 每个任务开始前运行对应基线测试；实现后必须运行同一测试集。
- 默认配置保持兼容：新路径必须有显式开关，默认关闭或保持当前行为，直到黄金集门禁通过。
- 不得伪造词级时间戳；没有原生词时间时必须使用 `time_source=segment_boundary` 并记录降级。
- 所有时间 offset 只能应用一次；窗口相对时间必须转换为全局时间后再进入物理投影。
- global、Qwen、ForcedAligner、SED 和 semantic review 结果不得绕过 `EvidenceDecision` 直接导出。
- 不得删除或改写现有基线测试来适应新实现。
- 每个任务完成后创建独立 commit，commit 前确认只包含该任务文件。

---

## Task 0: 冻结基线与建立回归快照

**Files:**
- Create: `test/baselines/high_precision_baseline.json`
- Create: `scripts/capture_high_precision_baseline.py`
- Test: `tests/test_release_check.py`, `tests/test_golden_quality_gate.py`, `tests/test_global_pipeline_contract.py`

**Interfaces:**
- Produces: 可重复的基线命令、测试结果摘要和配置快照，供后续任务比较。

- [ ] **Step 1: 保存工作树和提交信息**

```bash
git status --short
git rev-parse HEAD
git diff --stat
```

确认 HEAD 为 `d3d49d1`，将已有未提交修改记录在基线快照中，不回退、不覆盖。

- [ ] **Step 2: 运行基础回归测试**

```bash
pytest tests/test_release_check.py tests/test_golden_quality_gate.py tests/test_global_pipeline_contract.py -q
```

记录通过数量、失败数量和完整输出路径。

- [ ] **Step 3: 运行 ASR、物理时间轴和映射测试**

```bash
pytest tests/test_asr tests/test_physical tests/test_mapping -q
```

- [ ] **Step 4: 写入基线快照脚本**

脚本调用 `subprocess.run` 执行上述两组测试，并把 HEAD、工作树 diff stat、配置中的 ASR/声学校验关键值和退出码写入 JSON；测试失败时脚本返回非零退出码。

- [ ] **Step 5: 重新运行快照并提交**

```bash
python scripts/capture_high_precision_baseline.py
git add test/baselines/high_precision_baseline.json scripts/capture_high_precision_baseline.py
git commit -m "test: freeze high precision baseline"
```

**Gate:** 基线测试必须全部通过；若已有失败，先记录为已知失败，后续任务不得扩大失败集合。

---

## Task 1: 接通 WhisperX 词级强制对齐

**Files:**
- Modify: `vocal_subtitle/asr/global_transcriber.py`
- Modify: `vocal_subtitle/asr/whisperx_engine.py`
- Modify: `vocal_subtitle/asr/contracts.py`
- Modify: `vocal_subtitle/config/models.py`
- Modify: `configs/default.yaml`
- Create: `tests/test_asr/test_global_alignment_preference.py`

**Interfaces:**
- Consumes: `GlobalTranscriber.transcribe()` 当前返回的 `GlobalTranscript`。
- Produces: 每个 `GlobalWord` 带可靠 `time_source`；对齐失败时保留原始词时间并记录 diagnostics。

- [ ] **Step 1: 写失败测试**

测试 fake engine 同时提供 `transcribe()` 和 `align()`，断言对齐后的词使用 alignment 时间，且 `diagnostics["alignment_status"] == "applied"`；另一个 fake engine 让 `align()` 抛异常，断言保留原始词时间并记录 `alignment_failures`。

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_asr/test_global_alignment_preference.py -q
```

- [ ] **Step 3: 实现最小改动**

在 `GlobalTranscriber` 中为每个词写入 `time_source="whisperx_alignment"`；原始 ASR 词写入 `faster_whisper_word`。对齐异常时不得丢弃 transcript，只更新失败诊断。

- [ ] **Step 4: 增加配置开关**

增加 `asr.global_asr.alignment_enabled`，默认值保持当前兼容行为；只有显式开启时才调用 alignment。

- [ ] **Step 5: 运行测试并提交**

```bash
pytest tests/test_asr/test_global_alignment_preference.py tests/test_asr/test_whisperx_engine.py tests/test_global_pipeline_contract.py -q
git add vocal_subtitle/asr vocal_subtitle/config/models.py configs/default.yaml tests/test_asr/test_global_alignment_preference.py
git commit -m "feat: add explicit whisperx alignment provenance"
```

**Gate:** 无 WhisperX 依赖的环境中，基线路径行为不变。

---

## Task 2: 统一 global 与 segmented 的候选角色

**Files:**
- Modify: `vocal_subtitle/asr/evidence.py`
- Modify: `vocal_subtitle/asr/evidence_review.py`
- Modify: `vocal_subtitle/asr/evidence_decision.py`
- Modify: `vocal_subtitle/application/asr_path.py`
- Create: `tests/test_asr/test_global_candidate_roles.py`

**Interfaces:**
- Consumes: segmented candidates、global evidence、physical timeline。
- Produces: `global_signal` 默认只参与风险评分；`global_alternative` 显式准入后才参与替换/拆分决策。

- [ ] **Step 1: 写失败测试**

覆盖三种情况：默认 signal 不进入替换集合；开启 alternative 且物理范围合法时进入 alternatives；global candidate 缺词时间或窗口归属非法时被拒绝并记录原因。

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_asr/test_global_candidate_roles.py -q
```

- [ ] **Step 3: 增加 candidate role 字段**

在 CandidateEvidence diagnostics 或显式字段中保存 `candidate_role`，取值为 `global_signal` 或 `global_alternative`，不扩展现有 source 枚举。

- [ ] **Step 4: 修改 EvidenceReviewService**

默认仅把 global evidence 传给风险评分；当 `global_alternative_enabled=true` 且候选通过词时间、语言、窗口归属和物理校验后，才加入 EvidenceDecision 的替代候选集合。

- [ ] **Step 5: 运行相关回归并提交**

```bash
pytest tests/test_asr/test_global_candidate_roles.py tests/test_asr/test_evidence_review.py tests/test_asr/test_evidence_review_decision.py tests/test_global_pipeline_contract.py -q
git add vocal_subtitle/asr vocal_subtitle/application/asr_path.py tests/test_asr/test_global_candidate_roles.py
git commit -m "feat: route global evidence through candidate roles"
```

**Gate:** global 不能再绕过统一决策出口直接替换最终事件。

---

## Task 3: 增加 global-primary 实验路由

**Files:**
- Modify: `vocal_subtitle/application/asr_path.py`
- Modify: `vocal_subtitle/application/pipeline_runner.py`
- Modify: `vocal_subtitle/config/models.py`
- Modify: `configs/default.yaml`
- Create: `tests/test_global_primary_routing.py`

**Interfaces:**
- Consumes: Task 2 的候选角色和 Task 1 的词级对齐结果。
- Produces: `routing="global_primary"` 时，全局窗口作为主候选但仍经过 EvidenceDecision 和 DecisionEventProjector；失败时回退 segmented。

- [ ] **Step 1: 写失败测试**

测试 global 成功、global 失败、global 覆盖不足三种情况，断言分别选择 global candidate、回退 segmented、输出明确 diagnostics。

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_global_primary_routing.py -q
```

- [ ] **Step 3: 实现路由选择**

增加 `global_primary` 分支；不得复用当前直接进入 postprocess 的 bypass。global 结果必须构造 CandidateEvidence，进入统一 review/decision/projector。

- [ ] **Step 4: 增加覆盖和失败门禁**

至少检查 global transcript 非空、物理 speech coverage 达标、时间范围合法、文本密度不异常；失败时记录 `global_primary_fallback_reason` 并执行 segmented 路径。

- [ ] **Step 5: 运行回归并提交**

```bash
pytest tests/test_global_primary_routing.py tests/test_long_audio_global_path.py tests/test_global_pipeline_contract.py tests/test_pipeline.py -q
git add vocal_subtitle/application vocal_subtitle/config/models.py configs/default.yaml tests/test_global_primary_routing.py
git commit -m "feat: add decision-gated global primary routing"
```

**Gate:** 默认 `routing: auto` 行为保持不变；实验路由只能通过显式配置启用。

---

## Task 4: 统一词级物理对齐和边界裁决

**Files:**
- Modify: `vocal_subtitle/physical/word_alignment.py`
- Modify: `vocal_subtitle/asr/boundary_refiner.py`
- Modify: `vocal_subtitle/acoustic/validator.py`
- Modify: `vocal_subtitle/application/postprocess_runner.py`
- Create: `tests/test_physical/test_boundary_decision_precedence.py`

**Interfaces:**
- Consumes: WhisperX/faster-whisper word timestamps、physical timeline、声学骨架。
- Produces: 单一边界决策结果，包含 boundary time、confidence、time source、evidence IDs 和 revision trace。

- [ ] **Step 1: 写失败测试**

覆盖高置信词边界、中置信声学吸附、低置信不自动移动、相邻字幕不重叠和跨硬静音拒绝五种情况。

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_physical/test_boundary_decision_precedence.py -q
```

- [ ] **Step 3: 定义裁决优先级**

实现固定优先级：WhisperX word alignment → faster-whisper word → segment boundary；字幕整体边界再受 physical skeleton、相邻事件和硬静音约束。

- [ ] **Step 4: 限制自动吸附**

高置信最大 120ms，中置信最大 200ms，低置信仅诊断；超过 200ms 必须通过局部 RMS、global 文本一致性和相邻事件约束。

- [ ] **Step 5: 运行回归并提交**

```bash
pytest tests/test_physical tests/test_asr/test_boundary_refiner.py tests/test_end_time_fixes.py tests/test_mapping/test_time_mapper.py -q
git add vocal_subtitle/physical vocal_subtitle/asr/boundary_refiner.py vocal_subtitle/acoustic/validator.py vocal_subtitle/application/postprocess_runner.py tests/test_physical/test_boundary_decision_precedence.py
git commit -m "feat: unify physical boundary precedence"
```

**Gate:** 不得增加跨静音、时间倒序或字幕重叠测试失败。

---

## Task 5: 修复无词级时间戳的 speaker 切分

**Files:**
- Modify: `vocal_subtitle/diarization/turn_reconciler.py`
- Modify: `vocal_subtitle/diarization/early_turns.py`
- Modify: `vocal_subtitle/physical/events.py`
- Modify: `vocal_subtitle/config/models.py`
- Create: `tests/test_diarization/test_unworded_event_split.py`

**Interfaces:**
- Consumes: 带词和不带词的 SubtitleEvent、speaker turns。
- Produces: 有词时按词归属切分；无词时保留整段并标记 `speaker_split_degraded`，不把整句文本压到首个 speaker。

- [ ] **Step 1: 写失败测试**

测试带词事件切分为两个 speaker；无词事件保持单事件且包含降级诊断；非法词时间不会被伪造。

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_diarization/test_unworded_event_split.py -q
```

- [ ] **Step 3: 实现条件切分**

仅当每个词的 start/end 合法且能映射到 turn 时切分；否则保留原事件，设置 `time_source=segment_boundary` 和 `speaker_split_degraded=true`。

- [ ] **Step 4: 开启配置前保持默认兼容**

保持 `word_split_on_turn=false` 为默认值，新增测试专门验证显式开启后的行为。

- [ ] **Step 5: 运行回归并提交**

```bash
pytest tests/test_diarization tests/test_mapping/test_merged_event_speaker_compensation.py -q
git add vocal_subtitle/diarization vocal_subtitle/physical/events.py vocal_subtitle/config/models.py tests/test_diarization/test_unworded_event_split.py
git commit -m "fix: preserve unworded speaker events safely"
```

**Gate:** 不得出现整句文本被错误压到首个 speaker 的回归。

---

## Task 6: 高风险 Context Re-ASR 与可观测性

**Files:**
- Modify: `vocal_subtitle/asr/review_scheduler.py`
- Modify: `vocal_subtitle/asr/evidence_review.py`
- Modify: `vocal_subtitle/asr/review_telemetry.py`
- Modify: `vocal_subtitle/config/models.py`
- Create: `tests/test_asr/test_risk_context_reasr.py`

**Interfaces:**
- Consumes: EvidenceRiskScorer 结果和候选事件。
- Produces: 仅对 high/critical 或明确冲突窗口执行 Context Re-ASR；输出替换收益、失败原因、耗时和调用比例。

- [ ] **Step 1: 写失败测试**

测试低风险窗口不调用 re-ASR，高风险窗口调用一次，re-ASR 失败时保留主候选并记录失败原因。

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_asr/test_risk_context_reasr.py -q
```

- [ ] **Step 3: 实现风险门控**

使用现有 `ReviewScheduler` 和 `EvidenceRiskScorer`，禁止全量重复识别；同一 candidate/window 在一次运行中最多调用一次。

- [ ] **Step 4: 补充诊断字段**

输出 `reviewed_window_count`、`reasr_replaced_count`、`reasr_failed_count`、`reasr_time_ms` 和每个候选的替换依据。

- [ ] **Step 5: 运行回归并提交**

```bash
pytest tests/test_asr/test_risk_context_reasr.py tests/test_asr/test_evidence_review.py tests/test_asr/test_local_recovery.py -q
git add vocal_subtitle/asr vocal_subtitle/config/models.py tests/test_asr/test_risk_context_reasr.py
git commit -m "feat: gate context reasr by residual risk"
```

**Gate:** 低风险样本的处理耗时和调用次数不得无条件增加。

---

## Task 7: 黄金集门禁、配置准入和最终切换

**Files:**
- Modify: `configs/default.yaml`
- Modify: `scripts/run_offline_golden_production.py`
- Modify: `vocal_subtitle/quality/golden_gate.py`
- Create: `configs/high_precision.yaml`
- Create: `tests/test_high_precision_config.py`

**Interfaces:**
- Consumes: Task 1–6 的诊断字段和决策追踪。
- Produces: 可独立运行的高精度配置、黄金集对比报告和默认配置切换准入结论。

- [ ] **Step 1: 写配置与门禁失败测试**

断言 `configs/high_precision.yaml` 显式开启 global alignment、global-primary、timeline arbitration 和 risk-only Context Re-ASR；断言默认配置仍保持兼容值。

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_high_precision_config.py -q
```

- [ ] **Step 3: 创建高精度配置**

高精度配置使用 faster-whisper large-v3、词级时间戳、global context、WhisperX alignment、Context Re-ASR 和受限声学吸附；Qwen/FunASR 先保持 shadow 或 risk_only。

- [ ] **Step 4: 加强黄金集门禁**

至少检查：

- `real_speech_drop_rate <= 0.05`
- `physical_violation_rate == 0`
- `cross_silence_rate == 0`
- `hallucination_retention_rate <= 0.05`
- `raw_bypass_count == 0`
- 词级时间覆盖率达到 95%
- 字幕重叠率为 0

- [ ] **Step 5: 运行完整验证**

```bash
pytest tests/test_high_precision_config.py tests/test_release_check.py tests/test_golden_quality_gate.py tests/test_pipeline.py -q
python scripts/run_offline_golden_production.py --config configs/high_precision.yaml
```

- [ ] **Step 6: 只在门禁通过后切换默认配置**

如果高精度配置连续两轮黄金集通过，才修改 `configs/default.yaml`；否则保留独立配置并记录失败指标，不降低门禁阈值。

- [ ] **Step 7: 提交并生成交接报告**

```bash
git add configs scripts vocal_subtitle/quality tests/test_high_precision_config.py
git commit -m "feat: add gated high precision production profile"
```

交接报告必须包含：变更 commit 列表、测试命令和结果、黄金集指标、已知失败、回滚 commit 和是否建议切换默认配置。

**Gate:** 未通过黄金集门禁时，默认生产路径不得改变。

---

## 给其他 AI 的执行规则

1. 一次只执行一个 Task；先阅读该 Task 的文件范围，再开始编辑。
2. 不要重置、清理或覆盖其他未提交修改；发现同一文件已有并行修改时，先保留并在其基础上调整。
3. 先写失败测试，再改实现；测试名称必须对应具体行为。
4. 每个 Task 完成后运行该 Task 的 Gate 测试，并提交独立 commit。
5. 不得修改或删除既有基线测试来消除失败。
6. 如果基线测试失败，停止当前 Task，报告失败测试、首次出现的 commit 和最小复现命令。
7. 如果需要修改接口，必须同时更新调用方、类型/数据结构、测试和诊断字段。
8. 默认配置只有在 Task 7 的黄金集门禁连续通过后才能切换。
9. 最终交付必须包含测试结果、黄金集结果、性能变化和回滚方式。


## Task 8: 增加 TTS 清晰人声场景 profile

**Files:**
- Create: `configs/tts_clean.yaml`
- Modify: `vocal_subtitle/config/models.py`
- Modify: `vocal_subtitle/application/pipeline_runner.py`
- Modify: `vocal_subtitle/acoustic/validator.py`
- Create: `tests/test_tts_clean_profile.py`
- Create: `tests/test_physical/test_tts_skeleton_priority.py`

**Interfaces:**
- Consumes: 现有声学骨架、PhysicalTimeline、词级 ASR 结果。
- Produces: TTS profile 下由物理骨架决定 cue 的合法 start/end，词级 ASR 只负责段内文本和细化；通用 profile 行为不变。

- [ ] **Step 1: 写失败测试**

构造单人低噪声音频和两条骨架段，断言 cue 起点不早于骨架起点允许余量、终点不晚于骨架终点，骨架间静音不被跨越；再断言通用配置的边界结果不发生变化。

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_tts_clean_profile.py tests/test_physical/test_tts_skeleton_priority.py -q
```

- [ ] **Step 3: 增加显式场景 profile**

新增 `configs/tts_clean.yaml`，启用 `skeleton_mode`、`timeline_arbitration`、`reproject_grouped_windows` 和双向边界修正；不修改 `configs/default.yaml` 的通用默认值。

- [ ] **Step 4: 实现骨架优先边界策略**

在 `AcousticValidator` 或其调用层增加 profile 分支：TTS 模式使用骨架段作为 cue 的硬物理范围，首词/末词只用于范围内细化；禁止跨骨架段合并。

- [ ] **Step 5: 增加诊断字段**

输出 `skeleton_priority=true`、`skeleton_start_delta_ms`、`skeleton_end_delta_ms`、`cross_skeleton_merge_count` 和 `micro_pause_split_count`。

- [ ] **Step 6: 运行回归并提交**

```bash
pytest tests/test_tts_clean_profile.py tests/test_physical/test_tts_skeleton_priority.py tests/test_physical tests/test_end_time_fixes.py -q
git add configs/tts_clean.yaml vocal_subtitle/config/models.py vocal_subtitle/application/pipeline_runner.py vocal_subtitle/acoustic/validator.py tests/test_tts_clean_profile.py tests/test_physical/test_tts_skeleton_priority.py
git commit -m "feat: add tts clean voice skeleton priority profile"
```

**Gate:** TTS profile 的骨架边界误差不高于当前 ASR 段边界误差；通用配置和既有测试不受影响。

### Task 9: 增加 TTS 专项黄金集门禁

**Files:**
- Modify: `scripts/run_offline_golden_production.py`
- Modify: `vocal_subtitle/quality/golden_gate.py`
- Create: `tests/test_tts_golden_gate.py`

**Interfaces:**
- Consumes: TTS profile 诊断字段和人工标注的 TTS 黄金集。
- Produces: 独立的 TTS 边界质量结论，不与复杂真人录音门禁混为一组。

- [ ] **Step 1: 写失败测试**

断言 TTS 黄金集报告包含骨架 start/end 偏差、语音覆盖率、跨骨架合并数和微停顿拆分率。

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_tts_golden_gate.py -q
```

- [ ] **Step 3: 实现独立指标**

增加 TTS 专项门禁：骨架覆盖率、cue 跨骨架数、cue 尾随静音、cue 起止偏差 P50/P90。未达到门禁时只阻止 TTS profile 默认化，不影响通用模式。

- [ ] **Step 4: 运行验证并提交**

```bash
pytest tests/test_tts_golden_gate.py tests/test_golden_quality_gate.py tests/test_release_check.py -q
git add scripts/run_offline_golden_production.py vocal_subtitle/quality/golden_gate.py tests/test_tts_golden_gate.py
git commit -m "test: gate tts skeleton timing separately"
```

**Gate:** TTS 场景单独报告可重复，且不会通过放宽通用门禁来隐藏其他场景回归。
