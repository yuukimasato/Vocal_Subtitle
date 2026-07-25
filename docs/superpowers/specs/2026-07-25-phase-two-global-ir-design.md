# 阶段二 P2.3：全局中间表示契约设计

日期：2026-07-25

状态：设计已确认，待实施

关联规格：

- [阶段二 P2.1：物理时间线核心模型设计](2026-07-25-phase-two-physical-timeline-design.md)
- [阶段二 P2.2：现有检测结果适配器设计](2026-07-25-phase-two-evidence-adapter-design.md)

## 1. 目标与边界

P2.3 建立全局词流和全局说话人轨道的数据契约，隔离现有分段 ASR 的局部时间语义，为 P2.4 缓存/坐标基础设施和阶段三全局 ASR、词级分配器提供稳定输入。

本子项目包含：

- `GlobalWord`：带稳定 ID 的全局绝对时间词。
- `GlobalTranscriptSegment`：全局 ASR 段级摘要。
- `GlobalTranscript`：有序全局词流和段引用关系。
- `GlobalSpeakerTimeline`：canonical speaker turn 和状态。
- 旧 `TranscriptionSegment`、`WordTimestamp`、`DiarizationResult` 到 IR 的显式适配器。
- 严格的时间、ID、引用和 schema 校验，以及 JSON-safe 序列化。

本子项目不包含：

- 全局 ASR、WhisperX、强制对齐或重叠窗口去重。
- 词到 PhysicalClip、SpeechEvidenceSpan 或 speaker 的最终分配。
- `SubtitleEvent`、`physical_spans`、`source_word_ids` 等阶段四事件字段。
- Pipeline 默认路径切换。

## 2. 模块布局

新增：

```text
vocal_subtitle/physical/ir.py
```

模块只依赖标准库和项目已有的 ASR/diarization 数据类，不依赖 numpy、模型或网络。`vocal_subtitle/physical/__init__.py` 统一导出 P2.1 与 P2.3 的公共类型。

## 3. GlobalWord

字段：

```text
id: str
text: str
raw_start: float
raw_end: float
confidence: Optional[float]
source_window_id: str
segment_id: str
language: Optional[str]
speaker_id: Optional[int]
no_speech_prob: Optional[float]
avg_logprob: Optional[float]
compression_ratio: Optional[float]
metadata: Dict[str, Any]
```

语义：

- `raw_start/raw_end` 是全局绝对时间，任何下游不得再次叠加 offset。
- `id` 在同一任务、同一输入和同一来源窗口下稳定，不使用列表位置作为唯一身份。
- 默认 ID 形式为 `gw:{source_window_id}:{segment_id}:w{index:04d}`；来源 ID 作为不透明字符串保存，序列化/反序列化不得重写。
- `source_window_id` 标识 ASR 输入窗口；没有窗口对象时由调用方传入明确的默认 ID。
- `segment_id` 标识原始 ASR segment，不等同于字幕事件 ID。
- `speaker_id` 只接受已经 canonical 化的全局 speaker ID；缺失时保持 `None`，IR 层不按间隙猜 speaker。
- 质量字段只保存证据；幻觉过滤由阶段一或未来全局 ASR 编排层负责。
- `metadata` 保存引擎、字符对齐和 alignment 等扩展信息，但不能覆盖核心字段。

不变量：

- `id`、`text`、`source_window_id`、`segment_id` 非空字符串。
- 时间为有限数且 `0 <= raw_start < raw_end`；有 duration 校验时不得越界。
- `confidence` 和质量字段若非 `None`，必须是有限数。
- `speaker_id` 只能为 `None` 或非负整数。
- 构造 GlobalWord 不修改输入 `WordTimestamp`。

## 4. GlobalTranscript

### 4.1 GlobalTranscriptSegment

字段：

```text
id: str
text: str
raw_start: float
raw_end: float
word_ids: List[str]
language: Optional[str]
avg_logprob: Optional[float]
metadata: Dict[str, Any]
```

它是 ASR 段级摘要，不代替 `TranscriptionSegment`，也不承载局部坐标。允许 `word_ids` 为空，用于只有段级时间的降级结果；不得伪造不存在的词 ID。

### 4.2 容器字段和关系

```text
schema_version: str = "global-ir-v1"
audio_duration: Optional[float]
words: List[GlobalWord]
segments: List[GlobalTranscriptSegment]
backend: str
status: str
diagnostics: Dict[str, Any]
```

关系约束：

1. `words` 按 `(raw_start, raw_end, id)` 稳定排序。
2. 每个 `word.id` 全局唯一。
3. 每个 segment 的 `word_ids` 必须引用本 transcript 已存在的词。
4. `word_ids` 的顺序必须与对应词的时间顺序一致；同一词不能被同一 segment 重复引用。
5. segment 时间必须覆盖其引用词的时间范围；只有无词段才允许独立使用段级范围。
6. `audio_duration` 为 `None` 时仍校验非负时间；不执行输入范围校验。
7. P2.3 不去重重叠窗口词；重复候选由 P2.4/P3 编排层处理。

## 5. GlobalSpeakerTimeline

字段：

```text
schema_version: str = "global-ir-v1"
duration: float
turns: List[SpeakerTurn]
exclusive_turns: List[SpeakerTurn]
speaker_ids: List[int]
backend: str
status: str
diagnostics: Dict[str, Any]
```

约束：

- `turns` 和 `exclusive_turns` 使用阶段零 canonicalization 产生的 ID，不重新编号。
- 两组 turn 均使用全局绝对时间并满足 `0 <= start < end <= duration`。
- turn 按 `(start, end, speaker_id)` 稳定排序；overlap 信息保留。
- `speaker_ids` 由两组 turns 的 ID 并集推导并升序排列，不接受独立手工列表。
- `speaker_ids` 可以为空；没有可靠 diarization 时使用空 turns 和 `status="unknown"` 或 `"degraded"`。
- P2.3 不改变 turn 时间、不做超限合并、不从 gap 发明 speaker。

## 6. 适配器与绝对时间

新增适配函数：

```python
adapt_transcription_segments(
    segments: Sequence[TranscriptionSegment],
    *,
    source_window_id: str,
    segment_id_prefix: str,
    time_offset: float = 0.0,
    language: Optional[str] = None,
    audio_duration: Optional[float] = None,
) -> GlobalTranscript

adapt_diarization_result(
    result: DiarizationResult,
    *,
    duration: float,
) -> GlobalSpeakerTimeline
```

`adapt_transcription_segments` 的输入时间明确规定为“窗口/片段内相对时间”，因此统一执行：

```text
raw_start = time_offset + word.start
raw_end   = time_offset + word.end
```

调用方若已有全局绝对时间，必须传入已归零的 segment 结构或使用单独的绝对时间构造入口；不允许通过隐式猜测避免 offset 重复。阶段三的全局 ASR 适配器直接生成绝对时间 GlobalWord，不经过相对时间路径。

适配规则：

- 每个 `TranscriptionSegment` 生成一个稳定 segment ID；有词时按词序生成 GlobalWord。
- 词级 speaker 优先使用 `WordTimestamp.speaker_id`；缺失时保持 `None`，不在 P2.3 回退到 turn。
- segment 的语言、质量字段和引擎信息进入对应字段/metadata。
- 非法时间、空 ID、负 speaker ID、重复生成 ID 或越过 duration 的结果拒绝并记录诊断；不裁剪词时间。物理裁剪和 alignment warning 属于阶段三分配器。
- `adapt_diarization_result` 只包装已有 canonical turns，保留 regular/exclusive 两套轨道和 diagnostics。

## 7. 序列化与版本

所有四类 IR 对象提供 `to_dict()` / `from_dict()` 或等价类方法，结果满足：

- 仅包含 JSON-safe 标量、列表和字典；numpy 数值在边界处转成 Python 标量。
- `None` 字段显式保留，避免读取时改变“缺失证据”语义。
- `schema_version="global-ir-v1"` 必须存在且严格匹配。
- `from_dict` 拒绝未知 schema、重复 ID、悬空 word reference、错误字段类型和非法时间。
- 序列化前按稳定排序输出，等价输入得到等价 JSON 结构。
- 诊断字段可以扩展，但不得覆盖核心字段；未知 metadata 字段原样保留。

P2.4 可以直接把 `to_dict()` 结果放入 CacheManager；缓存 key 负责区分输入、模型、窗口和策略，P2.3 不负责生成缓存 key。

## 8. 错误和降级

- 空 transcript 是合法结果：`words=[]`、`segments=[]`、`status="empty"`。
- 空 speaker timeline 是合法结果：`turns=[]`、`speaker_ids=[]`、`status="unknown"`。
- 单个词或单个 turn 非法时，适配器返回可诊断的拒绝记录；容器构造阶段拒绝包含非法对象的结果。
- 适配失败不会回写或修改旧 ASR/diarization 对象；P2.5 shadow 接入时标记 IR 为 `degraded`，旧路径继续运行。
- 不把未知 speaker、缺失词时间或缺失质量字段转换成确定值。

## 9. 测试与验收

新增 `tests/test_physical/test_global_ir.py`，覆盖：

1. GlobalWord 的稳定 ID、绝对时间和质量字段映射。
2. 相对 segment 时间加 offset 后不重复加 offset。
3. 非法时间、speaker、ID 和 duration 越界被拒绝。
4. GlobalTranscript 的词/段引用一致性和排序。
5. 无词段、空 transcript 和 unknown speaker timeline 的合法状态。
6. regular/exclusive turns 和 canonical speaker ID 原样保留。
7. dangling word ID、重复 word ID、未知 schema 和错误字段类型被拒绝。
8. `to_dict/from_dict` 往返保持 ID、时间、状态和 diagnostics。
9. 旧 `TranscriptionSegment`、`WordTimestamp` 和 `DiarizationResult` 不被修改。
10. 测试不依赖模型、GPU、HF token 或网络。

P2.3 完成定义：GlobalWord、GlobalTranscript、GlobalSpeakerTimeline 及旧对象适配器具备稳定的全局坐标、ID、speaker 引用、序列化和严格校验；阶段三可以直接消费，阶段一/零和现有分段路径不受影响。
