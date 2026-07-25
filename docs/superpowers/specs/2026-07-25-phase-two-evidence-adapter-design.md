# 阶段二 P2.2：现有检测结果适配器设计

日期：2026-07-25

状态：设计已确认，待实施

关联规格：[阶段二 P2.1：物理时间线核心模型设计](2026-07-25-phase-two-physical-timeline-design.md)

## 1. 目标与边界

P2.2 将现有 Silero/WebRTC/TEN VAD、ffmpeg coarse、ffmpeg skeleton、RMS/fusion 等检测输出转换为 P2.1 定义的 `SpeechEvidenceSpan`，让后续阶段获得统一的绝对坐标、来源和边界诊断。

本子项目只做数据适配，不重写检测算法，不改变现有 `SpeechSegment`、裸 skeleton tuple、`PipelineContext` 或字幕主路径的行为。

包含：

- 通用 `SpeechSegment` 到 evidence 的适配器。
- ffmpeg unified result 的 coarse/skeleton 适配器。
- `PipelineContext` 到 `PhysicalTimeline` evidence 的组装器。
- 局部时间偏移、clip 归属、越界裁剪、诊断和来源统计。
- 无模型适配器单元测试。

不包含：

- VAD、ffmpeg、RMS、BoundaryFusion 算法修改。
- 不同来源 evidence 的合并或新的投票算法。
- PhysicalClip 创建逻辑。
- PipelineContext 新字段和 Pipeline shadow 接入；这些属于 P2.5。
- ASR、WhisperX、词级分配和字幕事件构建。

## 2. 模块与返回契约

新增：

```text
vocal_subtitle/physical/evidence_adapter.py
```

核心接口：

```python
adapt_speech_segments(
    segments: Sequence[SpeechSegment],
    timeline: PhysicalTimeline,
    source: str,
    *,
    time_offset: float = 0.0,
    physical_clip_id: Optional[str] = None,
) -> EvidenceAdaptResult

adapt_ffmpeg_result(
    result: Mapping[str, Any],
    timeline: PhysicalTimeline,
    *,
    time_offset: float = 0.0,
    physical_clip_id: Optional[str] = None,
) -> EvidenceAdaptResult

build_timeline_from_context(
    context: PipelineContext,
    duration: float,
    *,
    time_offset: float = 0.0,
) -> EvidenceAdaptResult
```

`EvidenceAdaptResult` 包含：

```text
evidence_spans
diagnostics
skipped_count
source_counts
```

适配器不修改输入检测对象，也不返回新的 PhysicalClip。它只通过 `timeline.add_evidence()` 写入 evidence，并使用 P2.1 的统一 ID、裁剪和排序规则。

## 3. 来源契约

| 输入 | source | confidence | 备注 |
|------|--------|------------|------|
| Silero/WebRTC/TEN `SpeechSegment` | 调用方传入 | 保留 `SpeechSegment.confidence` | source 必须显式声明 |
| ffmpeg coarse | `ffmpeg_coarse` | 保留输入值 | 来自 unified result 或 context |
| ffmpeg skeleton tuple | `ffmpeg_skeleton` | `None` | 裸 tuple 没有置信度，不伪造数值 |
| BoundaryFusion 输出 | `boundary_fusion` | 保留 fusion confidence | 不推断内部投票来源 |

每条 evidence 的 metadata 至少包含：

```text
boundary_type = "detected_evidence"
source
time_offset
```

若输入有额外来源统计、原始阈值或检测器诊断，则原样放入 metadata；没有的信息不补造。不同 source 的重叠 evidence 是合法结果，因为它们代表独立证据，不代表重复字幕归属。

## 4. 坐标与合法性处理

适配器将局部时间转换为全局时间：

```text
global_start = local_start + time_offset
global_end   = local_end + time_offset
```

处理顺序：

1. 检查时间为有限数且 `start < end`。
2. 检查 `time_offset` 为有限数。
3. 使用显式 `physical_clip_id`，或要求时间范围只能匹配一个 clip。
4. 将 evidence 裁剪到所属 clip 内，由 P2.1 记录 `clipped=true` 和原始范围。
5. 若越过输入 duration、反向、为空或字段类型非法，则跳过该条并增加诊断。
6. 所有成功结果按 P2.1 规则稳定排序。

`time_offset` 不负责宏观块重叠去重；宏观块全局坐标、重叠窗口和跨块缝合属于 P2.4。

## 5. 去重与失败降级

适配器遵循以下规则：

- 不合并不同 source 的重叠 evidence。
- 同一 source 的输入顺序和边界保持不变；输入自身重复时只记录 `duplicate_candidate`，不擅自删除。
- 不重新计算 BoundaryFusion 投票，不重建 boundary confidence。
- 某一检测器为空、字段缺失或单条适配失败时，其他 source 继续生成 evidence。
- 只有 duration 或默认 PhysicalClip 非法时才向上抛出异常。

这样保持现有“ffmpeg 失败后继续使用 Silero”的降级语义。P2.5 接入时，适配失败只让 shadow timeline 标记为 `degraded`，旧 `SpeechSegment` 字幕路径继续运行。

## 6. Context 组装规则

`build_timeline_from_context()` 只消费 `PipelineContext` 已有字段，按以下来源读取：

```text
silero_segments       → silero
ffmpeg_segments       → ffmpeg_coarse
fused_segments        → boundary_fusion
ffmpeg_unified_result["coarse_speech"]
                       → ffmpeg_coarse
ffmpeg_unified_result["skeleton"]
                       → ffmpeg_skeleton
```

为避免 unified result 与 `ctx.ffmpeg_segments` 重复写入 coarse evidence：

- unified result 存在时，使用其中的 `coarse_speech` 作为唯一 ffmpeg coarse 来源；
- unified result 不存在时，使用 `ctx.ffmpeg_segments`；
- Silero 与 fusion 始终作为独立来源保留，即使时间重叠也不合并。

P2.2 不修改 `PipelineContext` 结构。P2.5 负责将组装结果挂入 context 并接入 shadow 流程。

## 7. 测试与验收

新增 `tests/test_physical/test_evidence_adapter.py`，覆盖：

1. 四类来源的 source、confidence 和 metadata 映射。
2. 局部 offset 到全局坐标的转换。
3. clip 裁剪、越界跳过和诊断统计。
4. 空结果、损坏结果和非法 offset 的处理。
5. unified ffmpeg coarse 优先于 context coarse，确保不重复。
6. 不同 source 的重叠 evidence 保留。
7. 输入 `SpeechSegment`、skeleton tuple 和 `PipelineContext` 不被修改。
8. 单一 source 失败时其余 source 仍能生成结果。
9. 结果排序、source_counts 和 skipped_count 稳定。
10. 测试不需要模型、GPU 或外部 token。

P2.2 完成定义：所有现有检测输出都能转换为可追溯的 `SpeechEvidenceSpan`；适配过程不改变检测算法、不扩大 PhysicalClip、不阻断既有降级路径，并具备独立单元测试。
