# 阶段零：物理边界优先的字幕时间轴与全局说话人约束设计

日期：2026-07-25

状态：已确认设计，待实施

关联设计：[Vocal_Subtitle 离线高精度字幕系统设计文档](../../Vocal_Subtitle-离线高精度字幕系统设计文档.md)

## 1. 目标与边界

阶段零先修复当前代码中已确认、且不依赖 WhisperX 的问题，建立后续阶段可依赖的契约：

1. 字幕事件的默认开始和结束时间以现有物理语音边界为准，不以 ASR 的偏移时间戳覆盖它。
2. diarization 的最终 canonical speaker 数量不得超过用户配置的上限。
3. ffmpeg 声学骨架路径必须使用调用方配置，而不是静默使用函数默认值。
4. ASR 适配层保留后续幻觉过滤所需的元数据，同时不改变本阶段的结果过滤行为。
5. 既有 faster-whisper、whisper-cpp、FunASR、流式入口和旧 positional dataclass 构造保持兼容。

本阶段明确不实现：WhisperX 安装与运行时、全局 ASR、强制对齐、幻觉过滤决策、PhysicalTimeline/GlobalTranscript、LLM 来源词约束和默认 profile 切换。这些能力按总设计文档的后续阶段单独验收。

## 2. 术语与权威边界

本阶段区分两种边界：

- **PhysicalClip**：输入文件或任务明确允许处理的绝对范围，是不可扩展的硬约束。阶段零暂不新建完整 `PhysicalClip` 类型，但不允许任何现有映射结果越过 `SpeechSegment` 的合法范围。
- **物理语音边界**：现有 VAD、ffmpeg、RMS、边界融合和合并流程产出的 `SpeechSegment(start, end)`。当前项目对这些边界已有较稳定的时间轴质量，本阶段把它作为字幕事件默认显示范围。
- **ASR 时间戳**：用于词排序、文本归属、重复判断以及没有可靠物理内部边界时的降级切分证据；不再作为字幕事件外层边界的默认权威。

优先级为：

```text
PhysicalClip 硬范围
  > 物理语音边界 SpeechSegment
    > 声学内部边界（skeleton/RMS）
      > ASR 词/段时间戳（仅内部降级证据）
```

任何降级使用 ASR 时间作为内部边界的情况都必须保留诊断信息，便于阶段二建立完整物理时间线后替换。

## 3. 方案选择

### 3.1 采用方案

采用“现有流水线增量修复 + 小型可复用归一化策略”的方案：

- 时间轴逻辑留在 `mapping/time_mapper.py`，直接利用现有 `SpeechSegment`、声学扫描和 gap 处理，避免引入尚未成熟的第二套时间模型。
- 说话人数量处理抽成独立的 canonicalization 模块，供阶段二的全局 speaker timeline 复用。
- ASR 数据结构只追加可选字段，避免修改既有构造调用。

### 3.2 不采用的方案

- 不在本阶段一次性接入 WhisperX；它会引入独立的依赖、缓存和降级面。
- 不把 ASR 的末词时间作为“更精确”的字幕边界；当前项目已观察到 ASR 时间存在前后偏移，物理边界才是本项目的优先证据。
- 不用停顿轮换或事件位置猜测来补造缺失说话人身份。

## 4. 物理优先时间轴

### 4.1 单一 ASR 事件

当一个 `SpeechSegment` 只生成一条有效 ASR 事件时：

```text
event.start = speech_segment.start
event.end   = speech_segment.end
```

ASR 的首词和末词时间仍保留在 `event.words`，但不改写事件的外层时间。开始和结束必须在物理语音边界内。

### 4.2 一个物理区间内有多个 ASR 事件

一个物理区间不能把完整的 `[start, end]` 重复赋给每一条字幕，否则会产生重叠。因此采用以下边界构造：

1. 第一条事件从 `speech_segment.start` 开始。
2. 最后一条事件到 `speech_segment.end` 结束。
3. 中间边界优先从现有 acoustic skeleton/RMS 静音扫描获得。ASR 时间只用于确定搜索窗口和事件顺序，不直接作为边界。
4. 声学内部边界不可用时，才回退到相邻 ASR 词/段时间，并写入现有日志/管道诊断；结构化的 `alignment_warning` 字段留到阶段二扩展 `SubtitleEvent` 时加入。
5. 生成的边界必须单调、不反向、不越过 `speech_segment`，过短或反向事件按现有规则丢弃。

现有 `_merge_gaps()` 继续负责段间 gap 的自然停顿处理，但不得把事件扩展到物理语音边界之外。后续阶段二将把这些内部边界正式建模为 `SpeechEvidenceSpan` 和 `PhysicalSpan`。

### 4.3 结束时间缺失与越界

- 有词级时间但词尾早于物理结束：保留物理结束作为事件结束，不用 ASR 时间缩短事件。
- ASR 词尾或段尾晚于物理结束：裁剪到物理结束。
- 没有词级时间：仍使用物理边界；ASR 段时间仅参与内部事件的降级切分。
- 任意后处理（最小时长、帧衔接、LLM、声学校验）都不能突破该物理区间。

为让约束跨越后处理阶段，阶段零在 `SubtitleEvent` 末尾追加可选的
`physical_start` / `physical_end` 内部包络字段。它们是当前阶段的过渡契约；阶段二建立 `physical_spans` 后再统一迁移，不把完整 `PhysicalClip` 数据模型提前引入。

这取代当前“最后一条 ASR 事件无条件把 `end` 设为声学边界”的特殊逻辑：新的行为是所有事件都遵循 physical-first，而不是只对末条做特殊覆盖。

## 5. 说话人数量与 canonicalization

### 5.1 配置契约

`DiarizationConfig` 新增：

```python
expected_speakers: Optional[int] = None
```

- `None`：保持现有后端自动估计行为。
- `N > 0`：向 pyannote 传递 `min_speakers=N` 和 `max_speakers=N`；fallback 聚类也使用同一上限。
- `N <= 0`：配置校验失败，给出明确错误，不静默回退。

diarization 缓存键必须包含 `expected_speakers`。旧缓存没有该约束时不得直接绕过新的结果归一化。

### 5.2 结果归一化

新增独立策略函数/模块，输入 `DiarizationResult`、可选音频和上限，输出归一化结果及诊断：

```text
raw_diarization_speaker_count
canonical_speaker_count
speaker_merge_map
canonicalization_status
canonicalization_reason
```

处理规则：

1. 以 regular/exclusive turn 的首次出现顺序建立稳定 canonical ID，保证跨 chunk、VAD 段和字幕后处理不重新编号。
2. 没有上限，或原始数量不超过上限时，只做稳定重编号，不改变 turn 时间。
3. 超过上限时，优先为每个原始 speaker 建立 turn 音频 profile，使用现有声学特征/嵌入计算簇间相似度，逐步合并最近的簇直到达到上限。
4. 无法取得可靠 profile 时，使用稳定的最小信息降级合并：保留首次出现的前 `N` 个 canonical 簇，将溢出簇按有效时长和相邻 turn 的确定性规则映射到已有簇，并把结果标记为 `degraded`。这保证数量上限，但明确暴露身份置信度下降。
5. 数量不足时不拆分、不补造缺失 ID；最终数量可以小于 `N`。
6. 所有 regular turns、exclusive turns 和后续选中的 turns 必须使用同一 `speaker_merge_map`，不能只修一套结果。

归一化只合并已有身份，不按停顿、字幕顺序或轮换模式发明 speaker。超限合并不会改变 turn 的时间范围，只改变 speaker ID 和诊断状态。

### 5.3 Pipeline 与统计

`Pipeline._run_global_diarization()` 在缓存读取和后端返回后都执行结果可用性检查与 canonicalization。缓存结果也必须经过新的上限校验，避免旧结果绕过约束。

`PipelineStats` 新增可选字段：

- `raw_diarization_speaker_count`
- `canonical_speaker_count`
- `speaker_merge_map`
- `canonicalization_status`

既有 `speaker_count` 继续表示最终投影到字幕的 canonical 数量，保持现有调用方语义。

## 6. ffmpeg 配置透传

`unified_ffmpeg_pass()` 当前接受 `noise_db` 和 `min_silence_duration`，但部分调用点使用函数默认值。阶段零审计并修复所有调用路径：

- 多宏观块后处理路径。
- `_run_ffmpeg_vad()` 的主 VAD 路径。
- 已经正确传参的声学校验/骨架导出路径保持不变。

对于 skeleton/声学校验调用，唯一配置来源是 `AcousticValidationConfig.skeleton_noise_db` 和 `skeleton_min_silence`。`FFmpegVADConfig` 仍可控制其自身的粗粒度 VAD 语义，但不得覆盖 skeleton 调用的两个参数。

## 7. ASR 数据契约扩展

追加字段，不改变既有字段顺序：

```python
@dataclass
class WordTimestamp:
    speaker_id: Optional[int] = None

@dataclass
class TranscriptionSegment:
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None
```

字段语义：

- `speaker_id`：词级 speaker 信息；旧引擎没有该信息时为 `None`。
- `no_speech_prob`：底层 ASR 的无声概率，供阶段一过滤层使用。
- `compression_ratio`：底层 ASR 的压缩比，供阶段一异常输出判断使用。

`FasterWhisperEngine` 从底层 segment 透传这两个字段。阶段零不添加新的阈值参数、不删除结果、不改变缓存结果集合。whisper-cpp、FunASR 和所有测试构造的旧对象继续使用默认值。

## 8. 错误处理与降级

- 物理边界缺失：沿用现有 `SpeechSegment` 输入校验；无法构造合法物理范围时不生成越界事件，并保留现有异常/诊断路径。
- 声学内部边界扫描失败：保留事件文本，使用 ASR 时间作为内部降级边界，写入现有日志/管道诊断，不扩大物理范围。
- diarization 后端不可用：沿用现有 pyannote → embedding/legacy → unknown 降级链；canonicalization 只约束已有结果，不改变后端选择。
- speaker profile 提取失败：执行确定性超限合并并标记 `degraded`，不因为诊断失败而生成新 speaker。
- ffmpeg 配置无效或执行失败：沿用现有异常捕获与 Silero-only 继续处理行为。
- 新增字段缺失：按 `None`/默认值处理，不让旧缓存和旧引擎反序列化失败。

## 9. 测试与验收

### 9.1 时间轴

- 单事件使用 `SpeechSegment.start/end`，即使 ASR 首词/末词偏移也不改外层边界。
- 多事件的首尾分别贴合物理区间首尾。
- 中间边界优先使用声学静音，声学边界缺失时才走 ASR 降级并写入现有诊断。
- ASR 时间超出物理区间时被裁剪；ASR 时间偏短时不缩短物理优先边界。
- 非末条和无词级时间戳路径保持有效，不能产生反向或越界事件。

### 9.2 说话人

- `expected_speakers=N` 正确解析、传入 pyannote、进入缓存键。
- 后端返回超过 `N` 个 speaker 时，最终 canonical 数量不超过 `N`，并有 merge map/降级状态。
- 后端返回少于 `N` 个 speaker 时不补造 ID。
- regular/exclusive turns 使用一致映射，canonical ID 按首次出现稳定。
- 未启用 expected speaker 约束时，现有默认行为不变。

### 9.3 ffmpeg 与 ASR 元数据

- mock `unified_ffmpeg_pass()`，验证所有目标调用路径收到配置值。
- mock faster-whisper segment，验证 `no_speech_prob` 和 `compression_ratio` 透传。
- 旧引擎和旧 positional 构造的默认字段为 `None`，流式入口不受影响。

### 9.4 回归验证

先运行阶段零相关测试，再运行完整 `pytest`。真实模型、GPU、HF token 缺失时只跳过对应集成测试；纯函数、配置、缓存、映射和数据契约测试必须在 CPU/无模型环境通过。

## 10. 完成定义

阶段零完成需同时满足：

1. 本规范中列出的代码路径已实现并有对应测试。
2. 物理优先时间轴不再由 ASR 时间戳覆盖，且后处理不越过物理范围。
3. 说话人数量上限和缓存隔离生效，超限结果不会原样进入字幕。
4. ffmpeg 配置默认值漂移问题已消除。
5. ASR 元数据可被阶段一直接消费，未提前改变过滤策略。
6. 阶段零测试和可运行范围内的完整测试通过，并记录外部依赖导致的跳过项。
