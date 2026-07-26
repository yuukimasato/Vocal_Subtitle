# 阶段一：ASR 幻觉过滤与数据契约扩展设计

日期：2026-07-25

状态：设计已确认，待实施

关联设计：[Vocal_Subtitle 离线高精度字幕系统设计文档](../../Vocal_Subtitle-离线高精度字幕系统设计文档.md)

## 1. 目标与范围

阶段零已经完成 ASR 质量元数据的数据契约扩展：

- `WordTimestamp.speaker_id`
- `TranscriptionSegment.no_speech_prob`
- `TranscriptionSegment.compression_ratio`

阶段一消费这些字段，建立统一的 ASR 幻觉过滤层，降低短片段解码产生的训练语料幻觉，同时保持既有引擎、缓存、流式入口和字幕导出兼容。

本阶段包含：

1. faster-whisper 显式传递解码保护阈值。
2. 新建纯函数过滤模块，统一处理所有引擎输出。
3. 训练语料短语和相邻重复结果过滤。
4. 缓存命中与新识别结果使用完全相同的过滤路径。
5. 过滤器版本和阈值进入 ASR 缓存键。
6. 过滤诊断统计进入 Pipeline 任务结果。
7. 为过滤规则和兼容行为增加单元、适配层和 Pipeline 回归测试。

本阶段不包含：

- WhisperX、强制对齐或全局 ASR。
- `PhysicalClip`、`PhysicalTimeline`、全局词流和词级物理分配。
- 新的 VAD 或声学模型。
- 基于神经网络的二次幻觉判定。
- 默认 profile 切换。
- streaming 识别行为改造。

## 2. 设计原则

### 2.1 两层保护

采用“引擎级解码保护 + Pipeline 统一过滤”的组合：

- 引擎级保护控制 faster-whisper 的候选生成，避免依赖第三方库默认值。
- Pipeline 过滤器是所有 ASR 结果的最终统一入口，覆盖 faster-whisper、whisper-cpp、FunASR、缓存命中和未来的全局 ASR 适配器。

只在引擎侧过滤会使缓存和其他引擎绕过规则；只在 Pipeline 过滤又无法稳定控制底层解码行为。两层职责独立，互不替代。

### 2.2 保守删除

过滤器只删除有明确异常证据的结果。缺失 `no_speech_prob`、`compression_ratio` 或词级时间戳时，不因字段缺失直接删除；短词也不因时长或字符数过短直接删除。

有效词级证据定义为：词文本非空、`start` 和 `end` 为有限数值且 `start < end`、置信度为有限数值且不低于 `0.35`。只要结果包含至少一个有效词级证据，单项低质量指标默认只记录警告，不自动删除；“已知训练语料短语”是明确例外。

## 3. 模块边界与接口

新增模块：

```text
vocal_subtitle/asr/hallucination.py
```

模块提供以下稳定接口：

```python
filter_transcription_segments(
    segments: Sequence[TranscriptionSegment],
    policy: HallucinationFilterPolicy,
) -> HallucinationFilterResult
```

`HallucinationFilterPolicy` 包含：

- `enabled: bool`
- `no_speech_threshold: float = 0.6`
- `log_prob_threshold: float = -1.0`
- `compression_ratio_threshold: float = 2.4`
- `filter_training_phrases: bool = True`
- `filter_adjacent_duplicates: bool = True`
- `version: str = "v1"`

`HallucinationFilterResult` 包含：

- `segments`：保留的 `TranscriptionSegment` 列表。
- `dropped`：按输入索引记录的删除原因、原始文本和质量指标。
- `warnings`：保留但存在异常指标的输入索引及原因。
- `counts`：各删除原因和保留告警的计数。
- `filter_version`：实际使用的策略版本。

过滤函数必须满足：

- 不修改输入列表、segment 对象或 `words` 对象。
- 不重排保留结果。
- 删除原因稳定、可序列化，供诊断报告使用。
- 输入为空、字段为 `None` 或字段类型异常时不抛出未说明的错误；策略配置本身在配置校验阶段验证。

## 4. 过滤规则

规则按以下顺序执行：

1. **空文本**：去除首尾空白后为空，删除，原因 `empty_text`。
2. **训练语料短语**：文本规范化后匹配以下短语时删除，原因 `training_phrase`：
   - `字幕志愿者`
   - `中文字幕志愿者`
   - `感谢观看`
   - `感谢收看`
   - `thanks for watching`
3. **无声概率**：`no_speech_prob >= no_speech_threshold` 且没有有效词级证据时删除，原因 `high_no_speech_without_word_evidence`。
4. **平均 logprob**：`avg_logprob < log_prob_threshold` 且没有有效词级证据时删除，原因 `low_logprob_without_word_evidence`。
5. **压缩比与重复**：`compression_ratio > compression_ratio_threshold` 且文本存在明显重复特征时删除，原因 `repetitive_compression`。明显重复包括规范化文本重复 token/字符循环，或与相邻候选结果规范化后完全相同。
6. **相邻重复**：同一序列内，两个结果时间相邻且文本规范化后重复时只保留信息更完整者；“短前缀 + 包含完整短语”的关系保留完整结果，原因 `adjacent_duplicate`。
7. **有效短词**：如 `Good`、`我`、`唉`，只要有有效词级证据，不因时长或字符长度删除。

规则优先级确保明确的训练语料幻觉先被删除；重复去重不会删除更完整的相邻结果。若质量字段缺失，相关规则跳过而不是假设异常。

## 5. 配置与引擎接入

`ASRConfig` 增加以下配置，默认值固定为：

```yaml
asr:
  no_speech_threshold: 0.6
  log_prob_threshold: -1.0
  compression_ratio_threshold: 2.4
  hallucination_filter_enabled: true
  hallucination_filter_version: v1
  filter_training_phrases: true
  filter_adjacent_duplicates: true
```

配置加载器负责读取和校验：

- `no_speech_threshold`、`compression_ratio_threshold` 必须为有限非负数。
- `log_prob_threshold` 必须为有限数。
- 过滤器版本必须为非空字符串。
- 布尔配置按现有配置解析规则处理。

`FasterWhisperEngine.transcribe()` 在调用底层模型时显式传递：

```text
no_speech_threshold=0.6
log_prob_threshold=-1.0
compression_ratio_threshold=2.4
```

实际值由 Pipeline 配置注入。现有 `beam_size`、`word_timestamps`、`condition_on_previous_text` 和 `vad_filter` 保持不变。faster-whisper 返回的 `no_speech_prob`、`compression_ratio` 和词级概率继续映射到项目数据类。

其他引擎不必实现 faster-whisper 专属的底层参数，但其输出统一进入 Pipeline 过滤器；缺少质量字段时按保守策略处理。

## 6. Pipeline 流程

`Pipeline._run_asr()` 对每个物理片段使用以下顺序：

```text
缓存读取或引擎识别
  → 文本规范化
  → 段内重叠去重
  → hallucination.py 统一过滤
  → 累计过滤诊断
  → 边界精修与时间映射
```

缓存命中和新识别必须调用同一辅助函数，不能让缓存分支提前 `continue` 绕过过滤。过滤关闭时保留原始结果，诊断中记录 `disabled` 状态。

过滤器默认不改变结果的时间戳和文本，只把删除结果排除在后续边界精修、时间映射和字幕导出之外。过滤异常时记录错误并保留原始结果继续任务，避免单条诊断逻辑阻断整条离线任务；配置错误则在任务开始前直接失败并给出配置字段。

`PipelineStats` 增加可选的过滤统计或诊断字段，至少包含：

- `hallucination_filter_version`
- `hallucination_filter_enabled`
- `hallucination_dropped_count`
- `hallucination_drop_reasons`
- `hallucination_warning_count`

既有统计字段和导出格式保持兼容。

## 7. 缓存隔离

现有 ASR `transcription` 缓存键增加：

- 过滤器版本。
- 过滤开关。
- 三个质量阈值。
- 训练语料短语开关。
- 相邻重复开关。

这些参数必须同时出现在缓存命中和写入使用的键构造中。规则版本或阈值改变后，旧缓存自然失效；不迁移旧缓存内容，也不在运行时对旧结果做不透明兼容。

## 8. 错误处理与降级

- 空列表输入：返回空结果和零计数。
- 缺失质量字段：跳过对应规则。
- 词级数据损坏：该词不计为有效词级证据，但不影响其他词或其他规则。
- 过滤器内部异常：Pipeline 记录过滤失败诊断并保留原始结果。
- faster-whisper 不接受某个显式参数：视为引擎兼容错误，按现有 ASR 异常路径处理，不静默改回库默认值。
- 缓存中存在旧数据：由于键包含版本和阈值，不应命中新策略；若外部直接注入旧对象，按现有对象兼容逻辑读取并经过统一过滤。

## 9. 测试与验收

### 9.1 纯函数

- 空文本被删除。
- 五个训练语料短语被删除，大小写和空白归一化有效。
- 高 `no_speech_prob` 且无词级证据被删除。
- 低 `avg_logprob` 且无词级证据被删除。
- 异常压缩比和重复文本被删除。
- 缺失质量字段不误删。
- 有效词级证据的 `Good`、`我`、`唉` 保留。
- 相邻短前缀和完整短语只保留完整结果。
- 过滤器不修改输入对象和词时间戳。
- 删除原因、警告和汇总计数稳定可序列化。

### 9.2 引擎与 Pipeline

- mock faster-whisper，断言三个解码阈值显式传入。
- 断言质量字段透传到 `TranscriptionSegment`。
- 新识别结果经过过滤后再进入后续阶段。
- 缓存命中结果也经过相同过滤。
- 过滤器异常时任务保留原始结果并记录诊断。
- 过滤关闭时结果集合保持旧行为。

### 9.3 配置与缓存

- 默认配置值正确加载。
- 非法阈值和空版本被拒绝。
- 过滤器版本、阈值或开关变化会生成不同缓存键。
- 旧 positional dataclass 构造、whisper-cpp、FunASR 和 streaming 入口保持兼容。

阶段一验收要求：纯函数、配置、缓存和 mock 引擎测试在无模型环境可运行；真实模型测试缺少依赖时明确跳过；阶段零相关回归不得因本阶段改动失败。

## 10. 完成定义

阶段一完成需同时满足：

1. faster-whisper 不再依赖库默认解码保护阈值。
2. 统一过滤器对新识别和缓存命中均生效。
3. 已知训练语料幻觉不会进入最终字幕。
4. 有词级证据的真实短词不会因短时长被误删。
5. 过滤规则、版本和阈值进入缓存隔离与任务诊断。
6. 其他 ASR 引擎、streaming、既有字幕导出和阶段零行为保持兼容。
7. 阶段一测试在当前可用依赖范围内通过，并记录无法运行的外部依赖项。
