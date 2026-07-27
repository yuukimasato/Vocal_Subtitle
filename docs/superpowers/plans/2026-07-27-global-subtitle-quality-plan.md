# 影视级字幕全链路质量优化实施计划

设计依据：[2026-07-27-global-subtitle-quality-design.md](../specs/2026-07-27-global-subtitle-quality-design.md)

## 1. 目标与执行契约

本计划把现有 VAD/FFmpeg/RMS 物理时间轴接入唯一生产链路，建立局部噪声模型、结构化边界仲裁、可追溯词级对齐、自适应语义分片、受约束 LLM 滑窗、有限回投状态机、双时间轴和唯一最终事件出口。

执行模型必须遵守：

1. 在仓库根目录 `/home/hope/Tools/Vocal_Subtitle` 执行所有命令。
2. 保留用户已有改动，特别是 `vocal_subtitle/utils/audio_utils.py`、未跟踪的 `test/`、`dogfood-output/` 和浏览器实测报告。禁止回退、清理或批量暂存这些内容。
3. 每个任务使用测试驱动：先增加或修正目标测试，确认它因预期原因失败，再做最小实现。
4. 每个任务单独提交，不夹带其他任务或用户改动。
5. 不得删除或 `xfail` 质量测试来获得绿色结果。旧测试与已批准设计冲突时，在同一提交中按新不变量改写断言，并在提交说明中写明冲突。
6. 默认无模型测试不得下载模型、访问网络或需要 API key。真实模型和浏览器验收放在独立门禁。
7. `vocal_subtitle/pipeline.py` 由单一集成执行者负责。其他执行者先在独立模块和测试中完成纯函数，再由集成执行者接入 Pipeline。

## 2. 当前基线

基线提交：`bf62809 docs: design global subtitle quality pipeline`。

2026-07-27 执行：

```bash
venv/bin/pytest -q
```

结果为 `531` 个测试被发现，在收集阶段出现 `13` 组错误。主要缺口：

- `asr/base.py` 缺少 `LanguageDetection` 及新质量/语言/speaker 字段。
- `diarization/base.py` 缺少 `SpeakerTurn` 和 `DiarizationResult`。
- `speaker_embedding.py` 缺少模型缓存探测契约。
- `scripts/run_benchmarks.py` 缺少 rollout 评估函数与序列化字段。
- `SubtitleEvent` 不接受已有物理/词来源模块需要的字段。
- `config.py` 和 `pipeline.py` 未完整实现已提交测试中的 global ASR 与 speaker 契约。

另一组可收集的聚焦测试当前为 `31 passed, 18 failed`，其中 17 个失败由 `SubtitleEvent` 契约断层引起，1 个失败由 LLM fast merge 跨物理边界引起。

实施时先恢复测试可收集性，然后用更小的失败簇推进，不得在收集错误未解决时开始修改 LLM prompt。

## 3. 核心实现选择

- 复用 `PhysicalTimeline` / `PhysicalClip` / `SpeechEvidenceSpan` / `PhysicalSubtitleBin`，不新建第二套物理时间轴。
- `LocalNoiseProfile` 只校准物理候选，不改写原始声学证据；固定阈值回退必须显式降置信度。
- `BoundaryArbiter` 先执行硬约束，再对多源特征评分；所有冻结物理边界保存结构化 `BoundaryDecision`。
- 将 `GlobalWord + WordAllocation` 作为设计中 `WordToken` 的实现，在 allocation 上增加对齐结果，避免复制词对象。
- 保留 `SubtitleEvent` 作为外部兼容类型；`start/end` 在最终阶段代表显示时间，`physical_start/end` 代表冻结的人声证据时间。
- `SubtitleBuilder` 最终只负责渲染。合并、拆分、去重、显示映射和最终校验必须在导出前完成。
- LLM 输出只引用 word/fragment ID，禁止直接采信它返回的数值时间。
- 语义边界没有合法投影时先确定性合并，最多一次受限 LLM 修复，随后回退 PhysicalFragment。

## 4. 任务 1：恢复基础数据契约和测试收集

**目标：** 让所有测试模块可被 pytest 导入，不在这一步接入新 Pipeline。

**修改文件：**

- `vocal_subtitle/asr/base.py`
- `vocal_subtitle/diarization/base.py`
- `vocal_subtitle/diarization/speaker_embedding.py`
- `scripts/run_benchmarks.py`
- 相应 `tests/test_asr/`、`tests/test_diarization/`、`tests/test_benchmark_rollout.py`

**先写/确认的测试：**

- `LanguageDetection(language, probability, source)` 可序列化。
- `WordTimestamp` 支持可选 `speaker_id`；`TranscriptionSegment` 支持语言、语言概率、`no_speech_prob`、`compression_ratio` 且兼容旧位置参数。
- `SpeakerTurn` 支持 start/end/speaker/confidence/overlapped；`DiarizationResult` 支持 turns/exclusive turns/count/backend/status/diagnostics。
- Hugging Face 缓存探测不发起网络请求。
- rollout 无真实场景时明确不可发布，并序列化 actual ASR path 与 fallback category。

**实现要点：**

1. 对 dataclass 只追加带默认值的字段，不破坏旧 positional construction。
2. 验证时间有限且 `end > start`，但不在基础类中引入模型依赖。
3. rollout 逻辑复用现有 `BenchmarkSummary` / `SceneResult`，不创建重复脚本。

**验证：**

```bash
venv/bin/pytest --collect-only -q
venv/bin/pytest -q tests/test_asr/test_whisperx_engine.py tests/test_diarization tests/test_benchmark_rollout.py tests/test_language_policy.py
```

**完成条件：** 全量测试收集错误为 0；未要求全量断言在此任务全部通过。

**建议提交：** `fix: restore foundational ASR and diarization contracts`

## 5. 任务 2：恢复配置、统计和路由契约

**目标：** 让 global/segmented 路径、语言策略和 speaker 能力具有稳定的配置与诊断面。

**修改文件：**

- `vocal_subtitle/config.py`
- `configs/default.yaml`
- `configs/podcast.yaml`
- `configs/education.yaml`
- `configs/variety_show.yaml`
- `configs/music_live.yaml`
- `vocal_subtitle/pipeline.py` 中仅限 `PipelineStats` 和纯路由辅助函数
- `vocal_subtitle/cli.py`
- `tests/test_deployment_defaults.py`
- `tests/test_phase_five.py`
- `tests/test_language_policy.py`

**实现要点：**

1. 向 `ASRConfig` 追加 `GlobalASRConfig`、`language_mode`、幻觉过滤和语言切换字段，完整接入 `_parse_config()` 和 override。
2. 实现 `auto/global/segmented` 请求路由；streaming 始终返回 segmented。
3. 恢复 global 失败分类、凭证脱敏和 full-pipeline cache 路径兼容性纯函数。
4. `PipelineStats.to_dict/from_dict` 包含 ASR path、global attempted、fallback、speaker canonicalization、物理覆盖和质量分项诊断。
5. 导入 `vocal_subtitle` 不得导入 WhisperX、Torch 或 pyannote 重依赖。

**验证：**

```bash
venv/bin/pytest -q tests/test_deployment_defaults.py tests/test_phase_five.py tests/test_language_policy.py tests/test_cli.py
```

**完成条件：** 旧 YAML 仍可加载；离线默认 global 优先；streaming 不进入 global；统计往返不丢字段。

**建议提交：** `feat: restore quality pipeline configuration contracts`

## 6. 任务 3：建立兼容的统一 SubtitleEvent

**目标：** 修复物理模块与旧字幕对象的契约断层，定义双时间轴语义。

**修改文件：**

- `vocal_subtitle/mapping/time_mapper.py`
- `vocal_subtitle/physical/events.py`
- `vocal_subtitle/webui/models.py`
- `tests/test_phase_zero.py`
- `tests/test_phase_four.py`
- `tests/test_physical/test_phase_three.py`
- `tests/test_physical/test_coverage_recovery.py`

**字段契约：**

`SubtitleEvent` 追加带默认值的 `asr_text`、`speaker_status`、`speaker_source`、`speaker_repair_reason`、`physical_start/end`、`physical_spans`、`source_word_ids`、`logical_sentence_id`、`alignment_warning`、`hard_split_before`、`physical_region_id`、`physical_bin_id/start/end`、`time_source`、`genuine_overlap`、`overlap_group_id`、`overlap_tracks` 和 `revision_trace`。

`start/end` 保持旧 API 名称，但定义为当前显示时间。最终化之前它们初始等于物理时间；最终化之后可在受限范围内扩展。`physical_start/end` 是语义阶段不可改的人声证据时间。

**步骤：**

1. 增加 `to_dict()` / `from_dict()`，复制 list/dict 字段，避免缓存或 WebUI 修改共享可变对象。
2. 修正 `GlobalSubtitleEvent.to_subtitle_event()`，保留全部 provenance；词 ID 继续由 `source_word_ids` 管理。
3. 更新 Pydantic response 为追加字段且保持旧响应兼容。
4. 更新与新设计冲突的旧测试：`TimeMapper` 不再把最后一词硬填到整个 VAD 尾部；VAD 外包络只写入物理证据，不伪装成词时间。

**验证：**

```bash
venv/bin/pytest -q tests/test_mapping/test_time_mapper.py tests/test_phase_zero.py tests/test_phase_four.py tests/test_physical/test_phase_three.py tests/test_physical/test_coverage_recovery.py
```

**完成条件：** 物理事件可转换、序列化、缓存往返；原始 VAD/FFmpeg 证据不被替换。

**建议提交：** `feat: add provenance-safe subtitle event contract`

## 7. 任务 4：集中事件合并、拆分和偏移操作

**目标：** 消除各模块直接修改 `start/end/text` 造成的 provenance 丢失。

**新增/修改文件：**

- 新增 `vocal_subtitle/mapping/event_ops.py`
- `vocal_subtitle/mapping/strict_segmenter.py`
- `vocal_subtitle/mapping/subtitle_builder.py`
- `vocal_subtitle/mapping/time_mapper.py`
- `vocal_subtitle/mapping/event_constraints.py`
- `vocal_subtitle/acoustic_validator.py`
- `vocal_subtitle/merging/llm_merge_engine.py`
- 新增 `tests/test_mapping/test_event_ops.py`
- 扩展 `tests/test_phase_four.py` 和 `tests/test_mapping/test_subtitle_builder.py`

**纯操作接口：**

- `clone_event(event, **changes)`
- `shift_event(event, offset)`
- `merge_event_group(events, text: Optional[str], reason: str)`
- `split_event_by_word_ranges(event, ranges, reason: str)`

**必须保证：**

1. 合并时按顺序去重合并 `source_word_ids` 和 `physical_spans`，合并 revision trace，不改已确认 speaker。
2. 拆分时按词范围切分 source IDs 和 spans，子事件词时间重建为子事件相对坐标。
3. 偏移时同时移动 display、physical、bin 和 span 时间；禁止只移 `start/end`。
4. 任何合并先调用 `can_merge_events()`，硬边界、物理 owner 冲突、已确认 speaker 冲突和真实重叠均拒绝。
5. `LLMMergeEngine._apply_fast_merges()` 必须复用相同约束，修复当前跨 PhysicalClip 快速合并失败。

**验证：**

```bash
venv/bin/pytest -q tests/test_mapping/test_event_ops.py tests/test_mapping/test_subtitle_builder.py tests/test_phase_four.py tests/test_diarization/test_global_pipeline_contract.py
```

**建议提交：** `refactor: centralize provenance-preserving event operations`

## 8. 任务 5：接通现有物理时间轴和 global ASR 主路径

**目标：** 让已存在的 phase 2/3 模块进入真实 Pipeline，建立可缓存的局部噪声模型，并保证 global 失败时整体回退，不混用部分结果。

**修改文件：**

- `vocal_subtitle/pipeline.py`
- `vocal_subtitle/pipeline_context.py`
- `vocal_subtitle/physical/shadow.py`
- `vocal_subtitle/physical/evidence_adapter.py`
- `vocal_subtitle/physical/subtitle_bins.py`
- 新增 `vocal_subtitle/physical/noise_profile.py`
- `vocal_subtitle/physical/allocator.py`
- `vocal_subtitle/physical/events.py`
- `vocal_subtitle/vad/ffmpeg_vad.py`
- `vocal_subtitle/config.py`
- `configs/default.yaml`
- `vocal_subtitle/asr/global_transcriber.py`
- `tests/test_physical/test_shadow.py`
- `tests/test_physical/test_evidence_adapter.py`
- 新增 `tests/test_physical/test_noise_profile.py`
- `tests/test_physical/test_phase_three.py`
- `tests/test_phase_five.py`

**步骤：**

1. `_process_chunk_pipeline()` 完成 VAD/FFmpeg 后构建并在 context 中保留唯一 `PhysicalTimeline`。
2. 在物理阶段使用滚动分位数与 MAD 建立 `LocalNoiseProfile`，通过变点检测、滞回、最小稳定区间和上下限夹持生成稳定噪声区间。
3. FFmpeg `silencedetect` 在稳定噪声区间使用分段阈值；若分段执行的额外成本超出性能预算，则保留一次敏感 FFmpeg pass，并由局部 RMS/VAD 补充候选。不得把一次固定阈值调用标记为动态阈值。
4. 噪声估计失败时回退配置阈值、降低对应 evidence confidence 并记录 warning，不扩大 PhysicalClip，也不把无静音结果直接解释为全程人声。
5. 为单块、宏块和骨架路径使用绝对时间 offset；宏块拼接必须调用 `shift_event()`。
6. 实现 `_run_global_transcription_path()`：`GlobalTranscriber -> physical bins -> allocate_words -> build_events -> SubtitleEvent`。
7. 实现 `_resolve_asr_path()` 的真实编排。global 结果为 empty/degraded/非法时，丢弃全部 global 事件，再运行 segmented fallback。
8. 缓存记录 schema、noise-profile 配置指纹与 actual path，global 请求不能命中无 path 的旧 full-pipeline 缓存。
9. 保留所有原始 evidence IDs、局部噪声区间和 allocation warnings，不为了产出事件扩大 PhysicalClip。

**验证：**

```bash
venv/bin/pytest -q tests/test_physical/test_noise_profile.py tests/test_physical tests/test_vad/test_ffmpeg_vad.py tests/test_phase_five.py tests/test_utils/test_cache_manager.py
```

**完成条件：** fake global engine 成功时不调用 legacy ASR；任一 global 失败类别均只产生完整 legacy 结果或明确失败。合成底噪阶跃能生成不同的稳定噪声区间；噪声估计失败可审计地回退配置值。

**建议提交：** `feat: connect physical timeline to global subtitle path`

## 9. 任务 6：实现受约束词级边界对齐

**目标：** 将 ASR 词时间视为观测，将 VAD/FFmpeg/RMS 视为候选锚点和合法范围，不做 VAD 硬填。

**新增/修改文件：**

- 新增 `vocal_subtitle/physical/word_alignment.py`
- 新增 `vocal_subtitle/physical/boundary_arbiter.py`
- `vocal_subtitle/physical/allocator.py`
- `vocal_subtitle/physical/events.py`
- `vocal_subtitle/physical/subtitle_bins.py`
- 新增 `tests/test_physical/test_word_alignment.py`
- 新增 `tests/test_physical/test_boundary_arbiter.py`
- 扩展 `tests/test_physical/test_phase_three.py`
- 更新 `tests/test_mapping/test_time_mapper.py`

**数据契约：**

`WordAllocation` 追加 `aligned_start/end`、`physical_bin_id`、`boundary_confidence`、`alignment_status`、起止 `BoundaryDecision` 和使用的 boundary evidence IDs。`GlobalWord.raw_start/end` 保持不变。

`BoundaryDecision` 至少包含 `accepted`、`boundary_time`、`boundary_type`、`confidence`、`evidence_ids` 和 `reason_codes`。拒绝候选同样保留原因，便于质量报告区分“没有候选”和“候选违反硬约束”。

**对齐规则：**

1. 先按词与 PhysicalSubtitleBin 的正重叠、中点距离和 evidence 质量分配候选仓。
2. 在同一词序列上保证单调、有限、正时长和不互相逆序。
3. `BoundaryArbiter` 先过滤越出 PhysicalSubtitleBin/PhysicalClip、跨 speaker/hard split、截断确认人声、覆盖下一确认人声或破坏单调性的候选，再对归一化后的 ASR 置信度、RMS 梯度/谷底、VAD 概率、FFmpeg 静音、来源一致性和局部噪声状态评分。
4. 起点与终点使用不同损失函数，但不实现永久的 `RMS > VAD > ASR` 或 `FFmpeg > RMS > VAD > ASR` 固定排序。连续 200-400ms 低能静音是强终点候选，不是所有字幕换条的必要条件。
5. ASR 置信度按后端校准为等级。高置信度初始使用约 `±200ms` 搜索窗，低置信度初始使用约 `±500ms`；窗口被相邻词、物理仓、PhysicalClip 和 speaker 硬边界夹持。`0.9`、`0.6` 和物理权重 `80%` 只允许作为实验初值。
6. 只在词端点落入合法候选窗口时移动；记录原始值、新值、候选评分、拒绝原因和 evidence IDs。
7. 末词结束时间不得无条件改为 `speech_seg.end` 或 `physical_bin.end`。
8. 跨仓词保留整词并标记冲突；不从中间截断文字。
9. 无词级时间或无可接受边界时降级为片段证据，写入 `timing_degraded` 和结构化降级决策，不伪造词精度。
10. `build_events()` 使用首尾已接受的边界决策生成 `physical_start/end`；PhysicalSubtitleBin 外边界只作为候选锚点和合法上限，不直接取代词边界。

**验证：**

```bash
venv/bin/pytest -q tests/test_physical/test_boundary_arbiter.py tests/test_physical/test_word_alignment.py tests/test_physical/test_phase_three.py tests/test_mapping/test_time_mapper.py
```

**完成条件：** 违反硬约束的高分候选永不被选中；连续语音可在合法词间边界换条；相同输入与配置产生完全一致的 BoundaryDecision；所有降级边界都有 reason code。

**建议提交：** `feat: align ASR words to physical boundary evidence`

## 10. 任务 7：自适应停顿分级与 PhysicalFragment

**目标：** 把连续词流分成适合语义判断的短片段，先建立物理硬边界，再允许语义组合。

**新增/修改文件：**

- 新增 `vocal_subtitle/mapping/semantic_fragments.py`
- `vocal_subtitle/mapping/strict_segmenter.py`
- `vocal_subtitle/config.py`
- `configs/default.yaml`
- 新增 `tests/test_mapping/test_semantic_fragments.py`
- 扩展 `tests/test_phase_four.py`

**实现要点：**

1. 定义 `PauseClass`、`AdaptivePauseThresholds` 和 `PhysicalFragment`，片段只保存 ID 和证据引用，不拷贝自由时间。
2. 按 speaker 使用稳健分位数估计停顿分布，再夹持在设计范围：微停顿约 120-250ms，句间停顿约 250-500ms，长停顿初始约 500ms 及以上。
3. PhysicalClip 变化、已确认 speaker 变化、长停顿和真实重叠轨道边界标记为 hard split。
4. 细粒度句号/问号/叹号断句对正常时长事件也生效，不再只对超长事件调用。
5. 语义断句必须在词边界；没有词的事件使用保守的文本/停顿降级。
6. 连续语音允许在受 BoundaryDecision 支持的词间边界换条；不得因为缺少 200-400ms 静音而强制合并全部语义句。

**验证：**

```bash
venv/bin/pytest -q tests/test_mapping/test_semantic_fragments.py tests/test_phase_four.py tests/test_mapping/test_subtitle_builder.py
```

**建议提交：** `feat: add adaptive physical subtitle fragmentation`

## 11. 任务 8：建立受约束的 LLM 语义滑窗

**目标：** 让 LLM 只决定 ID 组合、标点、文本覆盖和 UNKNOWN speaker 推断，完全剥离时间所有权。

**新增/修改文件：**

- 新增 `vocal_subtitle/merging/semantic_window.py`
- `vocal_subtitle/merging/llm_merge_engine.py`
- `vocal_subtitle/mapping/llm_guard.py`
- `vocal_subtitle/config.py`
- 新增 `tests/test_merging/test_semantic_window.py`
- 扩展 `tests/test_merging/test_llm_merge_engine.py`
- 扩展 `tests/test_phase_four.py`

**结构化协议：**

- 输入：window ID、fragment IDs、word IDs、只读 physical range、文本、语言、speaker/status/confidence、pause class、hard split 和 overlap 标记。
- 输出：有序 fragment/word ID groups、绑定原始 ID 的 normalized text overlay、仅针对 UNKNOWN 的 speaker decision、review requests 和 reason。
- 修复输入额外包含可重组的 ID 与只读合法 boundary candidate IDs；修复输出仍只能重组 ID，不能生成候选。
- 输出 schema 明确禁止 `start`、`end`、`duration` 等数值时间字段。

**实现要点：**

1. 窗口在 hard split 处截断；可以使用少量只读上下文，但不赋予操作权。
2. 重叠窗口使用稳定 ID 合并结果，相同 ID 冲突时按置信度与原始窗口主权解决。
3. 验证未知 ID、重复/丢失/逆序 ID、跨 hard split、跨已确认 speaker、无来源增词和越权 speaker 修改。
4. 联网且已配置 LLM 时优先 quality provider；单窗口 schema 失败有限重试一次，仍失败则仅该窗口本地降级。
5. 修正现有 `_run_llm_merge()` 直接新建丢字段 `SubtitleEvent` 的行为；它只应用经验证的 ID 操作。
6. 保留每个窗口的 provider/model/retry/fallback/rejection 诊断，不记录 API key。
7. 物理投影失败时只允许一次受限语义修复；第二次失败必须确定性回退，不得循环请求 LLM。

**验证：**

```bash
venv/bin/pytest -q tests/test_merging/test_semantic_window.py tests/test_merging/test_llm_merge_engine.py tests/test_phase_four.py
```

**建议提交：** `feat: constrain LLM subtitle edits to evidence ids`

## 12. 任务 9：局部再识别与 UNKNOWN speaker 补全

**目标：** 让“漏词补全”通过新声学证据完成，并以受约束的声学异常门控 UNKNOWN speaker，不把 LLM 猜测当成转录或说话人真值。

**新增/修改文件：**

- 新增 `vocal_subtitle/asr/local_recovery.py`
- `vocal_subtitle/asr/boundary_reasr.py`
- `vocal_subtitle/physical/coverage.py`
- `vocal_subtitle/physical/allocator.py`
- `vocal_subtitle/mapping/strict_segmenter.py`
- 新增 `vocal_subtitle/diarization/speaker_change.py`
- `vocal_subtitle/diarization/feature_extractor.py`
- 新增 `tests/test_asr/test_local_recovery.py`
- 新增 `tests/test_diarization/test_speaker_change.py`
- 扩展 `tests/test_physical/test_coverage_recovery.py`
- 扩展 `tests/test_phase_four.py`

**实现要点：**

1. review request 、未覆盖 PhysicalSubtitleBin 或低置信度词组发起有界局部再识别。
2. 再识别可使用上下文 prompt、备用 ASR 或现有 sliding-window re-ASR，但每个区间有明确尝试上限。
3. 候选补词先转为新 `GlobalWord` ID，再重跑 allocation/alignment/coverage；无正物理重叠或置信度不足时拒绝。
4. 文本覆盖若改变词数，必须引用新词 ID；只改标点/大小写时可继续引用原始 ID。
5. UNKNOWN speaker 先使用两侧一致性和物理 owner 规则，再允许 LLM 推断。任何已确认 speaker 标签必须保持 100%。
6. 全局 diarization change point 或 speaker embedding 距离是 speaker 异常判断的主要证据；阈值按模型和场景校准。
7. RMS 均值差只产生 `SPK_CHANGE_CANDIDATE`，不能单独形成 hard split。MFCC/F0 可在 embedding 不可用时补强；仅在样本足以稳定估计协方差时允许使用马氏距离。
8. 候选只有在 diarization 确认或多源声学一致时才升级为 speaker hard split。LLM 只能补全硬边界范围内的 UNKNOWN，不能跨边界合并或覆盖已确认 speaker。

**验证：**

```bash
venv/bin/pytest -q tests/test_asr/test_local_recovery.py tests/test_diarization/test_speaker_change.py tests/test_physical/test_coverage_recovery.py tests/test_phase_four.py
```

**完成条件：** 同一 speaker 的音量突变不会单独产生 hard split；经确认的声学 speaker 变化不可被 LLM 跨越；embedding 不可用时的特征降级有明确诊断。

**建议提交：** `feat: recover low-confidence speech with local evidence`

## 13. 任务 10：双时间轴显示映射与最终校验

**目标：** 冻结物理边界，在明确的显示范围内吸附静音点和提供阅读停留时间。

**新增/修改文件：**

- 新增 `vocal_subtitle/mapping/display_timeline.py`
- 新增 `vocal_subtitle/mapping/boundary_projection.py`
- `vocal_subtitle/mapping/final_validator.py`
- `vocal_subtitle/mapping/end_time_validator.py`
- `vocal_subtitle/config.py`
- `configs/default.yaml`
- 新增 `tests/test_mapping/test_display_timeline.py`
- 新增 `tests/test_mapping/test_boundary_projection.py`
- 扩展 `tests/test_phase_four.py`
- 扩展 `tests/test_end_time_fixes.py`

**映射规则：**

1. `boundary_projection.py` 实现 `PROPOSED -> PROJECTED -> ACCEPTED/MERGED/ONE_REPAIR/FALLBACK` 有限状态机，并记录每次转移原因。
2. 语义组首尾只能投影到任务 6 产生的合法候选。无合法候选时先确定性合并相邻兼容组；仍冲突时最多发起一次受限修复，失败后回退 PhysicalFragment。
3. `physical_start/end` 取已接受的 SemanticCue 首尾对齐词边界，进入本阶段后不可修改。
4. display start/end 可向最近候选静音边界吸附，但必须被 PhysicalClip、hard split、下一人声起点和最大前导/尾随窗口夹持。
5. 显示时间必须覆盖物理人声，除非事件已标记 `timing_degraded`；不得用阅读时长要求截断人声。
6. 相邻不同 speaker 时优先人声边界；阅读时长不足写入质量警告，不篡改下一 speaker 的时间。
7. 修正当前 `final_validator` 把 display 强制夹进 physical envelope 的旧逻辑。新校验验证“display 合法覆盖 physical”，而不是要求两者必须相等。
8. 非真实重叠在最终阶段解决；不允许盲目裁前一条而导致切尾。无合法解时回退 PhysicalFragment 并标记需复核。

**验证：**

```bash
venv/bin/pytest -q tests/test_mapping/test_boundary_projection.py tests/test_mapping/test_display_timeline.py tests/test_phase_four.py tests/test_end_time_fixes.py
```

**完成条件：** 无合法投影时最多调用一次 LLM 修复；失败后稳定回退 PhysicalFragment；状态机不存在无界循环，物理时间在进入显示映射后保持不可变。

**建议提交：** `feat: map immutable physical cues to display timing`

## 14. 任务 11：统一 Pipeline 最终化、WebUI 和导出

**目标：** 从根本上消除“预览 6 条、导出 5 条”和缓存恢复字段丢失。

**新增/修改文件：**

- 新增 `vocal_subtitle/mapping/finalize.py`
- `vocal_subtitle/pipeline.py`
- `vocal_subtitle/mapping/subtitle_builder.py`
- `vocal_subtitle/webui/api.py`
- `vocal_subtitle/webui/models.py`
- `vocal_subtitle/utils/task_history.py`
- `vocal_subtitle/utils/cache_manager.py`
- `tests/test_pipeline.py`
- `tests/test_webui.py`
- 新增 `tests/test_mapping/test_finalize.py`

**步骤：**

1. 实现唯一 `finalize_subtitle_events()`：物理 fragment -> 严格断句 -> LLM/local semantic decision -> display mapping -> final validation -> 编号。
2. 移除 `SubtitleBuilder.build()` / `build_to_string()` 内部的隐式合并、拆分和去重。Builder 对输入事件不可变，只负责换行和格式渲染。
3. 所有离线路径在导出前只调用一次 finalizer。主 `subtitle_path`、`result.events`、WebSocket 完成事件和质量报告使用同一 list。
4. 已配置 LLM 时，LLM 语义结果是主字幕。如保留 pre-LLM 输出，它只能作为明确标记的诊断产物，不得与主结果混用。
5. WebUI 、编辑写回和 export API 全部使用 `SubtitleEvent.from_dict/to_dict`，保留 physical/source/speaker/revision 字段。
6. 缓存恢复时重建 `SubtitleEvent`，不允许 Pipeline cache hit 返回 dict 而 cache miss 返回 object 的两种契约。
7. `PipelineStats.subtitle_count` 在 finalizer 后设置，必须与预览和每个导出文件的 cue 数一致。

**验证：**

```bash
venv/bin/pytest -q tests/test_mapping/test_finalize.py tests/test_mapping/test_subtitle_builder.py tests/test_pipeline.py tests/test_webui.py
```

**完成条件：** 相同任务的 API events、WebUI 预览、SRT、VTT 和 ASS 逻辑 cue 的文本/起止时间可逐条对应。任务 12 完成后，ASS 允许将一个重叠逻辑 cue 渲染为多条展示记录，但必须共享 `overlap_group_id`。

**建议提交：** `refactor: use one finalized subtitle sequence everywhere`

## 15. 任务 12：真实重叠对白的 SRT/ASS 导出

**目标：** 在不改动物理时间的前提下表达少量同期对白，且不在导出器内隐式改变逻辑 cue 数。

**新增/修改文件：**

- 新增 `vocal_subtitle/mapping/overlap_export.py`
- `vocal_subtitle/mapping/subtitle_builder.py`
- `vocal_subtitle/config.py`
- 新增 `tests/test_mapping/test_overlap_export.py`

**规则：**

1. 只有事件带可验证 `genuine_overlap=True` 且 speaker 不同时进入重叠分组。
2. 重叠分组在 finalizer 中生成一个带 `overlap_tracks` 的逻辑 DisplayCue；WebUI、API 和 SRT 都展示这一逻辑 cue，不由 Builder 临时合并。
3. SRT 将该逻辑 cue 渲染为双行，每行使用可配置 speaker prefix，不超过两行。
4. ASS 可将同一 `overlap_group_id` 渲染为多个展示事件和轨道/位置，但它们必须可反查到同一逻辑 cue，不串行化物理时间。
5. 重叠分离不可信时保留 `UNKNOWN` 和 review warning，不丢弃次要可听对白。

**验证：**

```bash
venv/bin/pytest -q tests/test_mapping/test_overlap_export.py tests/test_mapping/test_subtitle_builder.py
```

**建议提交：** `feat: export verified overlapping dialogue`

## 16. 任务 13：修复对比工具并建立双基准质量门禁

**目标：** 分开测量声学边界与成片字幕效果，对当前 8 个实测音频产生可审计回归报告。

**新增/修改文件：**

- `scripts/compare_timeline.py`
- `scripts/run_quality_benchmark.py`
- `scripts/run_benchmarks.py`
- 新增 `scripts/evaluate_acoustic_gold.py`
- 新增 `tests/test_compare_timeline.py`
- `tests/test_quality_benchmark.py`
- `tests/test_benchmark_rollout.py`
- 新增 `test/quality_manifest.yaml`（只暂存该 manifest，不暂存用户媒体）

**步骤：**

1. 修正 ASS 时间解析中小数部分已归一化后又除以 100 的错误；为 1/2/3 位小数时间增加单元测试。
2. 取消“事件数相同则按索引强配”，改为时间重叠 + 文本相似度的单调全局匹配，保存 unmatched 事件。
3. manifest 记录音频、成片字幕、语言、场景、speaker 数、是否有词级声学金标准，不根据文件名猜测。
4. 成片基准报告 CER/WER、句段匹配、CPS、行数和显示时间；不用它评分词级发声误差。
5. 声学金标准使用独立 JSON schema 记录 word onset/offset/speaker，报告 start/end 中位数、P95、超过 80ms 切头/切尾率。
6. 词级金标准必须由人工听辨标注或复核。执行模型不得用 ASR 输出自动生成“金标准”并宣称通过。
7. CI 模式在缺真实场景、缺 global path、指标失败或金标准未就绪时返回非零，报告写明阻断理由。
8. 合成回归覆盖连续语音语义换条、气音/弱起声、渐弱尾音、底噪阶跃与缓变、音乐/撞击高能、同 speaker 音量突变、无静音 speaker 切换，以及语义边界无合法投影。
9. 分别报告不同 ASR 后端的置信度校准曲线和边界误差；未经校准的 `0.9/0.6` 阈值不得进入发布默认值。

**验证：**

```bash
venv/bin/pytest -q tests/test_compare_timeline.py tests/test_quality_benchmark.py tests/test_benchmark_rollout.py
venv/bin/python scripts/run_quality_benchmark.py --help
```

**建议提交：** `test: add dual subtitle quality benchmarks`

## 17. 任务 14：质量报告、缓存指纹与性能上限

**目标：** 使质量状态可观测、可局部重跑，且质量模式不超过约 2x 音频时长。

**修改文件：**

- `vocal_subtitle/pipeline.py`
- `vocal_subtitle/pipeline_context.py`
- `vocal_subtitle/utils/cache_manager.py`
- `vocal_subtitle/utils/persistence_manager.py`
- `vocal_subtitle/webui/models.py`
- `vocal_subtitle/webui/api.py`
- `tests/test_pipeline.py`
- `tests/test_persistence_manager.py`
- `tests/test_webui.py`

**实现要点：**

1. 分阶段缓存 physical evidence、local noise profile、global ASR、speaker、boundary decisions、alignment/fragments、LLM decisions 和 finalized cues。
2. 每层缓存指纹包含 schema 版本与影响该层的配置。LLM prompt 变化不重跑声学；VAD/FFmpeg/RMS/noise-profile 或置信度校准变化使边界仲裁及全部下游失效。
3. 质量报告分开声学边界、语义断句、speaker、最终结构四类健康度，不使用单一 100% 覆盖失败。
4. 记录局部 ASR/LLM 请求数、重试数、降级窗口数和各阶段耗时。
5. 设置每窗口和每任务复核上限，测试证明失败 provider 不会无限重试。
6. 边界报告记录搜索窗口、候选来源与归一化分数、硬约束拒绝原因、噪声区间、最终 evidence IDs、timing-degraded 决策和投影状态机终态。

**验证：**

```bash
venv/bin/pytest -q tests/test_pipeline.py tests/test_persistence_manager.py tests/test_utils/test_cache_manager.py tests/test_webui.py
```

**建议提交：** `feat: expose staged subtitle quality diagnostics`

## 18. 任务 15：全量回归和真实浏览器验收

**目标：** 证明主链路、格式导出、真实网页操作和已批准质量指标一致。

**执行顺序：**

1. 格式和编译检查。

   ```bash
   git diff --check
   venv/bin/python -m compileall -q vocal_subtitle scripts
   ```

2. 全量无模型测试。

   ```bash
   venv/bin/pytest -q
   ```

3. 用 fake ASR/fake LLM 跑端到端 Pipeline，验证 global 成功、global 整体降级、LLM 单窗口失败、无合法边界的一次修复/回退和缓存恢复五种链路。
4. 启动本地 WebUI，使用浏览器把 `test/` 下 8 个样本重跑到新的独立输出目录。
5. 逐任务验证页面无 console 错误、任务状态 completed、预览和下载 SRT/ASS 一致，且质量分项诊断可见。
6. 对 6 个有人工成片字幕的样本运行成片基准；对已完成人工词级标注的样本运行声学基准。
7. 生成 before/after 汇总，列出每个样本的 CER/WER、事件数、start/end median/P95、speaker 错误、结构错误和 real-time factor。

**发布门禁：**

- 全量无模型测试通过。
- 字幕 100% 可追溯至词 ID 和物理证据。
- 每个冻结物理起止点均有 accepted BoundaryDecision 或明确的 timing-degraded 降级决策。
- 跨硬边界、跨已确认 speaker、重复灌词和非真实重叠均为 0。
- 同 speaker 音量突变不得仅因 RMS 差异创建硬边界；无静音 speaker 切换不得被语义阶段跨越。
- 已确认 speaker 保持率 100%；UNKNOWN 推断准确率目标不低于 95%。
- 清晰语音中文 CER/英文 WER 不高于 5%，复杂多人场景不高于 10%。
- 在词级声学金标准就绪后，边界中位绝对误差不高于 80ms，P95 不高于 160ms，超过 80ms 的可听切头/切尾率不高于 0.5%。
- 质量模式 real-time factor 不高于 2.0。

如词级声学金标准尚未完成，可以报告代码与成片基准完成，但状态必须是“声学发布门禁待人工金标准”，不得宣称已达到影视级边界指标。

**建议提交：** `test: verify production subtitle quality pipeline`

## 19. 多模型协作分工

在同一共享工作树中不要同时修改同一文件。建议分工：

| 工作流 | 任务 | 主要文件所有权 |
|---|---|---|
| 契约组 | 1、3、4 | base dataclasses、`SubtitleEvent`、`event_ops.py` |
| 物理与仲裁组 | 5 的纯模块、6-7 | `noise_profile.py`、`boundary_arbiter.py`、`word_alignment.py`、`semantic_fragments.py` |
| 语义与恢复组 | 8-9 | `semantic_window.py`、LLM guard、local recovery、`speaker_change.py` |
| 时间与导出组 | 10、12 | boundary projection、display timeline、final validator、overlap export |
| 评测组 | 13 | `scripts/`、benchmark tests、manifest |
| 集成组 | 2、5 的 Pipeline 接入、11、14、15 | config/routing、`pipeline.py`、WebUI、cache、最终验收 |

依赖关系：

```text
1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12
                                                   +-> 14

4 -> 13
12 + 13 + 14 -> 15
```

各组可在前置数据契约稳定后并行编写纯模块和测试，但任务 10 的状态机集成必须等待任务 6、8、9。只有集成组修改 `pipeline.py`、`webui/api.py` 和全流水线缓存。

## 20. 总完成定义

1. 测试可收集且全量无模型测试通过。
2. 现有 VAD/FFmpeg/RMS 物理证据从 Pipeline 头到最终导出不丢失。
3. 无任何路径把 ASR 末词无条件延长到 VAD 段末尾。
4. LLM 只能对已存在的词/片段 ID 做受约束操作，已确认 speaker 和物理时间不可变。
5. 主结果同时保存物理时间与显示时间，两者的差异可诊断。
6. WebUI、API、缓存、SRT/VTT/ASS 和质量报告消费同一份最终事件。
7. 质量报告可复现，未完成的人工标注或模型环境限制被明确标记，不伪造通过。
8. 每个冻结物理起止点都有 accepted BoundaryDecision 或明确的 timing-degraded 决策，且投影修复不会无限循环。
9. 局部底噪突变不会使合法边界整体消失；RMS 音量差不会单独创建 speaker 硬边界。
