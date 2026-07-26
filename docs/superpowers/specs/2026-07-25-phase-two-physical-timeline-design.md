# 阶段二 P2.1：物理时间线核心模型设计

日期：2026-07-25

状态：设计已确认，待实施

关联设计：[Vocal_Subtitle 离线高精度字幕系统设计文档](../../Vocal_Subtitle-离线高精度字幕系统设计文档.md)

## 1. 目标与边界

P2.1 为阶段二建立物理时间线领域模型，明确“合法音频归属范围”“语音检测证据”和“识别上下文”三类时间概念的边界，供 P2.2 检测适配器、P2.3 全局中间表示、P2.4 坐标/缓存基础设施和 P2.5 Pipeline shadow 接入共同消费。

本子项目包含：

- `PhysicalClip`：不可扩展的物理归属硬范围。
- `SpeechEvidenceSpan`：VAD、ffmpeg、RMS 或 fusion 提供的语音证据。
- `ContextWindow`：为识别提供上下文的扩展范围。
- `PhysicalTimeline`：上述对象的有序容器、校验、交集查询和序列化。
- 确定性的 ID、排序、错误处理和版本化序列化契约。

本子项目不包含：

- VAD、ffmpeg、RMS 或 diarization 算法实现。
- ASR、WhisperX、强制对齐或词级分配。
- `SubtitleEvent.physical_spans` 等阶段四事件字段。
- Pipeline 默认路径切换或字幕导出行为改变。

## 2. 模块布局

新增：

```text
vocal_subtitle/physical/
  __init__.py
  timeline.py
```

`__init__.py` 导出 `PhysicalClip`、`SpeechEvidenceSpan`、`ContextWindow` 和 `PhysicalTimeline`。核心实现不依赖 numpy、VAD、ASR 或任何模型包，以便在无模型环境中单独测试和序列化。

## 3. 数据模型

### 3.1 PhysicalClip

字段：

```text
id: str
start: float
end: float
source: str = "input"
metadata: Dict[str, Any] = {}
```

语义：输入文件或任务明确允许处理的绝对时间范围，是唯一拥有字幕归属权的范围。clip 按起点、终点和稳定 ID 排序，clip 之间不得重叠。

默认输入时间线由完整音频构造一个 `[0, duration]` clip，默认 ID 为 `clip-000001`。显式未提供 ID 时，按创建序号分配确定性 ID；后续列表排序不改变已有 ID，序列化/反序列化必须保留已有 ID。

### 3.2 SpeechEvidenceSpan

字段：

```text
id: str
start: float
end: float
source: str
confidence: Optional[float] = None
physical_clip_id: Optional[str] = None
metadata: Dict[str, Any] = {}
```

语义：检测器对“此处可能存在语音”的证据，不是物理事实，也不拥有字幕归属权。不同来源的 evidence 可以相互重叠；同一来源的重叠结果由适配器负责处理，P2.1 不擅自合并检测结果。

### 3.3 ContextWindow

字段：

```text
id: str
start: float
end: float
physical_clip_id: str
left_context: float
right_context: float
metadata: Dict[str, Any] = {}
```

语义：围绕某个 physical clip 扩展出的识别上下文。context window 可以与其他 window 重叠，也可以跨出所属 clip，但最终必须处于整段输入的 `[0, duration]` 内；它不能改变或扩大所属 clip 的合法归属范围。

### 3.4 PhysicalTimeline

字段：

```text
schema_version: str = "physical-timeline-v1"
duration: float
physical_clips: List[PhysicalClip]
speech_evidence_spans: List[SpeechEvidenceSpan]
context_windows: List[ContextWindow]
diagnostics: Dict[str, Any]
```

时间线只保存绝对秒坐标，不保存音频数组。所有列表在构造完成后按时间、结束时间和稳定 ID 排序，保证相同输入得到相同 `to_dict()` 结果。

## 4. 时间不变量

所有时间字段必须是有限数，且满足：

```text
0 <= start < end <= duration
```

额外规则：

1. `duration` 必须为有限正数。
2. clip 必须位于输入范围内，任意两个 clip 不能重叠。
3. evidence 必须关联一个存在的 clip，或由构造器根据唯一匹配 clip 自动关联。
4. evidence 越过所属 clip 时裁剪到 clip 内，并在 metadata 中记录 `clipped=true` 和原始范围；裁剪后为空则抛出 `ValueError`。
5. context 的左右扩展量必须为有限非负数；生成窗口后钳制到 `[0, duration]`，原始扩展量保留在字段中。
6. metadata 只承载来源和诊断信息，不得覆盖或放宽时间约束。

## 5. 构造与查询 API

`PhysicalTimeline` 提供确定性 API：

```python
PhysicalTimeline.from_duration(duration: float) -> PhysicalTimeline

timeline.add_clip(
    start: float,
    end: float,
    source: str = "input",
    clip_id: Optional[str] = None,
) -> PhysicalClip

timeline.add_evidence(
    start: float,
    end: float,
    source: str,
    confidence: Optional[float] = None,
    physical_clip_id: Optional[str] = None,
) -> SpeechEvidenceSpan

timeline.add_context_window(
    physical_clip_id: str,
    left_context: float,
    right_context: float,
) -> ContextWindow
```

查询和持久化 API：

```python
timeline.validate() -> List[str]
timeline.clip_range(start, end, clip_id=None) -> Optional[TimeRange]
timeline.intersections(start, end) -> List[SpeechEvidenceSpan]
timeline.to_dict() -> dict
PhysicalTimeline.from_dict(payload) -> PhysicalTimeline
```

具体实现可以使用内部 `TimeRange` 值对象承载通用范围校验，但不向外暴露第二套物理语义。核心四类对象使用不可变值语义或等价的受控修改方式，避免调用方绕过时间线校验直接改变边界。

## 6. 错误处理

- `add_clip` 对负时间、越过 duration、反向范围和与已有 clip 重叠直接抛出 `ValueError`。
- evidence 无法关联 clip、类型错误或裁剪后为空时抛出 `ValueError`。
- context 的非法扩展量直接抛出 `ValueError`；合法扩展超出输入范围时执行钳制。
- `from_dict` 严格检查 `schema_version`、核心字段类型和时间不变量；未知 schema 版本拒绝加载，不静默降级。
- `validate()` 返回结构化可读错误列表，不修改时间线。
- 构造和查询不依赖音频或模型，因此不会因外部依赖缺失改变结果。

P2.5 接入时，shadow 时间线校验失败将标记新产物为 `degraded` 并保留旧字幕路径；只有输入 duration 或默认硬 clip 非法才阻止任务。

## 7. 测试与验收

新增 `tests/test_physical/test_timeline.py`，覆盖：

1. `from_duration()` 生成完整默认 clip。
2. 合法 clip、evidence 和 context 的创建及稳定排序。
3. clip 重叠、负时间、反向时间和越界输入被拒绝。
4. evidence 越界被裁剪并记录诊断，裁剪为空时被拒绝。
5. evidence 之间允许重叠，不能因此改变 clip。
6. context 可跨出所属 clip，但不能越过输入 duration。
7. 交集查询只返回合法 evidence。
8. `to_dict()`/`from_dict()` 往返保持对象、ID、顺序和诊断一致。
9. 未知 schema 版本和损坏核心字段被拒绝。
10. 无 numpy、VAD、ASR 或模型依赖也可导入并运行测试。

P2.1 完成定义：四类核心对象、时间不变量、确定性 ID/排序、序列化、错误处理和单元测试全部完成；阶段一、阶段零和现有默认 Pipeline 行为不因 P2.1 改动而变化。
