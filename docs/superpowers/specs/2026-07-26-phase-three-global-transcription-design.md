# 阶段三：全局转录与词级分配设计

日期：2026-07-26

状态：设计已确认，待实现

关联设计：

- [离线高精度字幕系统总设计](../../Vocal_Subtitle-离线高精度字幕系统设计文档.md)
- [阶段二物理时间线](2026-07-25-phase-two-physical-timeline-design.md)
- [阶段二全局中间表示](2026-07-25-phase-two-global-ir-design.md)

## 1. 目标与范围

阶段三建立一条可显式启用的“全局 ASR -> 词级物理/说话人分配 -> 事件”路径，利用完整音频或带重叠的上下文窗口降低短片段识别错误，同时保证字幕归属不越过 `PhysicalClip`。

本阶段包含：

- WhisperX 的 optional、延迟加载适配器和结果归一化。
- 全局转录编排、上下文窗口执行和重叠窗口结果去重。
- `GlobalTranscript` 词流到物理 clip、语音证据和 speaker 的确定性分配。
- 无字符对齐时的整词保守归属和对齐诊断。
- 保留物理范围、来源词和逻辑句 ID 的事件模型。
- Pipeline 的显式 global 模式、缓存、进度和失败降级。

本阶段不包含：

- 默认路径切换；阶段五之前旧的分段 ASR 仍是默认路径。
- 默认断句策略和全局最终物理校验；阶段四负责这些行为。
- LLM 来源词约束；阶段四接入事件字段后再实施。
- 流式入口改造；流式路径不能加载 WhisperX。
- 安装脚本、CUDA 自动探测和模型权重打包。

## 2. 现状与接口边界

阶段二已经提供以下输入：

- `PhysicalTimeline`：包含不可扩展的 `PhysicalClip`、检测证据和上下文窗口。
- `GlobalTranscript`：全局绝对时间的 `GlobalWord` 与段级摘要。
- `GlobalSpeakerTimeline`：canonical speaker turn 和 exclusive turn。
- `build_shadow_artifacts()`：现有 Pipeline 可构建上述对象，但当前只使用旧分段 ASR 结果。

阶段三新增模块只依赖这些公开对象，不修改旧 ASR 数据类的 positional 构造契约，也不把窗口局部时间再次当成全局时间。

## 3. 模块设计

### 3.1 WhisperX 适配器

文件：`vocal_subtitle/asr/whisperx_engine.py`

适配器实现现有 ASR 引擎的最小接口，并将 `whisperx`、`torch` 和 `torchaudio` 的导入放在 `load_model()` 内。适配器负责：

- 显式检测依赖和模型能力；未安装时抛出可识别的能力异常。
- 把 WhisperX segment、word、字符对齐和 speaker 字段转换为项目数据结构。
- 词级 speaker 优先于 segment 级 speaker；缺失时保留 `None`，不在适配器内猜测。
- 拒绝空 ID、负时间、反向时间、重复词和越过输入窗口的结果，并记录诊断。
- 直接生成已带全局 offset 的 `GlobalTranscript`，避免重复叠加 offset。

适配器不改变 WhisperX 默认模型下载行为，不在模块导入阶段联网或加载模型。

### 3.2 全局转录编排器

文件：`vocal_subtitle/asr/global_transcriber.py`

编排器接受音频、`ContextWindow` 列表、ASR 后端和窗口策略。窗口策略支持：

- 单个完整输入窗口；
- 按 `ContextWindow` 执行的多个窗口；
- 按窗口重叠范围裁剪输入并为每个窗口生成稳定来源 ID。

每个窗口结果先归一化为全局绝对时间，再按以下顺序去重：

1. 只比较同一语言或兼容语言的候选词；
2. 以时间重叠和规范化文本判断重复候选；
3. 优先保留置信度更高者；置信度相同时优先完整时间范围、主窗口归属和稳定 ID；
4. 无法确定为重复时全部保留，并在诊断中记录重叠候选，不静默丢词。

输出必须按 `(raw_start, raw_end, id)` 稳定排序，并更新 transcript 的 segment 词引用。编排器不执行 physical clip 裁剪，避免把识别范围误当作归属范围。

### 3.3 物理词分配器

文件：`vocal_subtitle/physical/allocator.py`

核心 API：

```python
allocate_words(
    transcript: GlobalTranscript,
    physical_timeline: PhysicalTimeline,
    speaker_timeline: Optional[GlobalSpeakerTimeline] = None,
) -> AllocationResult
```

每个 `GlobalWord` 产生一个分配记录或明确的拒绝诊断：

- 完全位于一个 `PhysicalClip` 内：归属于该 clip；
- 跨 clip 边界：不扩展 clip，不截断词文本；整词保守归属，记录 `cross_physical_boundary`；
- 完全位于 clip 外：拒绝该词并记录 `outside_physical_clip`；不能仅通过钳制事件时间保留文本；
- 与 `SpeechEvidenceSpan` 相交：保存证据 ID；没有证据不代表词必然无效，但要记录证据缺失；
- speaker 优先级为词级 speaker、segment 级 speaker、exclusive turn 包含关系、普通 turn 包含关系；冲突或重叠无法确定时使用 `mixed/unknown` 状态，不猜测已知 speaker。

无字符级对齐时始终以整词为最小单位，不切半词、不生成新的文本片段。

### 3.4 事件模型

文件：`vocal_subtitle/physical/events.py`

事件由按全局时间排序的已分配词构成，包含：

- `start`、`end`：来源词的音频相对时间包络；正常情况下不超出对应 physical clip；
- `text` 和词列表；
- `speaker_id` 或 unknown/mixed 状态；
- `physical_spans`：一个或多个物理归属范围；
- `source_word_ids`：事件使用的全局词 ID；
- `logical_sentence_id`：稳定的逻辑句编号；
- `alignment_warning`：跨物理边界、证据缺失、speaker 冲突等诊断。

`PhysicalClip` 是合法范围包络，不是按固定时长灌装字幕的容器。阶段三禁止按 clip 边界切割字符或词，也不能因为 clip 边界而丢弃 ASR 原始词时间。

跨边界词采用以下保守策略：

- 词完全位于一个 clip 内时，事件时间使用词时间并受 clip 包络校验。
- 词跨越相邻 clip 时，保留完整词，不生成半词文本；事件可保留多个连续 `physical_spans`，并记录 `cross_physical_boundary`。
- 词跨越真实静音或不连续物理范围时，不强行拼接为两个事件；保留完整词作为待复识别候选，或在严格模式中拒绝并记录诊断。
- 事件只在词边界上拆分，不在字符中间拆分。不同 speaker、不同不连续 physical clip、被拒绝的词不能进入同一事件。

静音/换气优先断句的完整策略留给阶段四；同 speaker 的短间隙是否合并也不能覆盖 physical clip 边界。

### 3.5 时间语义与校正层

阶段三区分四种时间语义，禁止用后者无条件覆盖前者：

1. `raw_*`：ASR 输出的音频相对时间，来自全局窗口归一化后的绝对坐标。
2. `aligned_*`：WhisperX/forced alignment 或可靠声学锚点校正后的时间，可选。
3. `physical_*`：PhysicalClip 和 PhysicalSpan 提供的合法范围，不是字幕显示时间。
4. `display_*`：阶段四及之后的字幕排版结果。

离线 ASR 的模型计算延迟不会自动形成时间戳漂移，因此阶段三不执行按处理时钟累积的动态补偿。只有检测到可靠的窗口 offset 或对齐锚点时，才允许执行有界、单调的校正；校正必须保留原始时间、记录修正原因，并再次通过物理范围校验。不得承诺固定的 5ms 残差，验收应以固定素材上的越界率、词级覆盖率和边界误差分布为准。

事件模型提供到现有 `SubtitleEvent` 的显式适配，新增字段使用默认值保持旧调用方兼容。适配过程不修改 `PhysicalTimeline`。

## 4. Pipeline 接入

新增显式配置，默认关闭：

```yaml
asr:
  global:
    enabled: false
    backend: whisperx
    window_mode: context_windows
    overlap_dedup: true
    alignment: true
    fallback_to_segmented: true
```

运行顺序为：

```text
音频预处理/VAD/diarization
  -> PhysicalTimeline 与 ContextWindow
  -> GlobalTranscriber
  -> PhysicalWordAllocator
  -> GlobalEventBuilder
  -> 现有字幕输出/导出
```

全局模式的结果与旧路径隔离，旧路径仍按当前流程执行。出现依赖缺失、模型失败、非法结果、缓存损坏或分配校验失败时：

- 记录 stage、异常类型、窗口和词级诊断；
- 将 global 结果标记为 `degraded`；
- 在 `fallback_to_segmented=true` 时回到当前分段 ASR；
- 不修改旧路径的输入对象和缓存。

显式关闭 fallback 时，全局失败应返回可诊断错误，不产生部分字幕。

## 5. 缓存与可观测性

新增缓存命名空间：

- `whisperx_asr`：音频哈希、模型、语言、窗口列表、解码参数和实现版本；
- `whisperx_alignment`：ASR 缓存身份、语言、对齐模型和对齐策略；
- `whisperx_speaker_assignment`：alignment 身份、speaker timeline 身份和分配策略。

每个缓存 payload 带 schema/version；命中后仍执行归一化、分配和校验。缓存不可用只触发重新计算或降级，不绕过物理约束。

统计至少包含窗口数、原始词数、去重词数、拒绝词数、跨边界词数、无证据词数、speaker fallback 数、事件数、降级原因和耗时。

声学吸附、微间隙合并、字幕构建和 LLM 后处理都必须在阶段三事件字段存在时同步维护 `words`、`physical_spans`、`source_word_ids` 和 warning，并拒绝跨不连续 physical clip 的隐式合并。

## 6. 错误处理与不变量

- 所有时间使用有限的绝对秒数；全局结果不允许再次叠加 offset。
- `PhysicalClip` 只能由时间线创建和验证，阶段三任何模块不得扩展或重写它。
- 不可确定的重复、speaker 或边界使用诊断和保守结果，不使用隐式猜测。
- 外部引擎对象在转换失败时保持不变。
- 空 transcript 是合法的 `empty` 结果；空 speaker timeline 保持 `unknown`。
- 旧分段路径和流式入口不导入 WhisperX，也不受全局模式实现影响。

## 7. 测试计划

不依赖模型、GPU、HF token 或网络的测试：

1. WhisperX 未安装时基础模块可导入，显式调用得到能力错误。
2. mock WhisperX segment/word/character 输出能转换为全局绝对 IR。
3. 非法时间、重复词、缺失字段、窗口越界和 speaker 标签映射被拒绝或诊断。
4. 重叠窗口重复词按确定性规则去重，边界相似词不会误删。
5. 词完全落在 clip 内、跨 clip、clip 外和缺少 evidence 的分配结果正确。
6. speaker 词级、segment 级、exclusive turn、普通 turn 的 fallback 顺序正确。
7. 同 speaker 的连续词形成事件，不同 speaker、不同 clip 或拒绝词强制分开。
8. 事件保留全部 `physical_spans`、`source_word_ids`、逻辑句 ID 和 warning。
9. 全局 Pipeline 成功路径、依赖缺失降级路径和 fallback 关闭错误路径可测试。
10. 默认配置下现有 Pipeline 测试和流式行为保持不变。
11. 单个 ASR 段包含词时不回退为整个 PhysicalClip；无词时才使用物理包络并记录 warning。
12. 相邻 clip 的跨边界词保持完整文本和多个连续 physical spans，不生成半词。
13. 后处理尝试扩展、合并或吸附到物理范围外时被拒绝或钳制并产生诊断。

## 8. 完成定义

阶段三完成需要满足：

- 显式 global 模式能够在 mock 后端下跑通从窗口到事件的完整链路；
- 真实 WhisperX 依赖通过延迟适配器接入，未安装不影响现有模式；
- 所有词级分配和物理边界不变量有独立测试；
- Pipeline 失败时有明确、可观测且不破坏旧路径的降级；
- 默认配置、现有导出格式、流式入口和旧 ASR 引擎行为不变；
- 阶段四可直接消费事件中的物理范围、来源词和对齐诊断字段。
