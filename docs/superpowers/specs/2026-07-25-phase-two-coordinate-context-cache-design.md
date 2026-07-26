# 阶段二 P2.4：坐标、上下文与缓存基础设施设计

日期：2026-07-25

状态：设计已确认，已实施

关联规格：

- [阶段二 P2.1：物理时间线核心模型设计](2026-07-25-phase-two-physical-timeline-design.md)
- [阶段二 P2.2：现有检测结果适配器设计](2026-07-25-phase-two-evidence-adapter-design.md)
- [阶段二 P2.3：全局中间表示契约设计](2026-07-25-phase-two-global-ir-design.md)

## 1. 目标与边界

P2.4 为后续 shadow Pipeline 和全局 ASR 提供三个无模型基础设施：显式的局部/全局坐标转换、由 PhysicalClip 构造识别上下文窗口、以及与 IR schema 和处理策略绑定的缓存键。

本子项目包含：

- `CoordinateMapper` 和只读 `MacroChunkCoordinate` 视图。
- `build_context_windows()` 上下文窗口构造器。
- canonical JSON 指纹和版本化 IR cache key。
- 上述模块的严格输入校验、JSON-safe 边界和独立单元测试。

本子项目不包含：

- PipelineContext 字段扩展或默认 Pipeline 切换。
- overlap 区域去重、文本缝合、词级分配或 ASR/WhisperX。
- 对既有 `CacheManager.make_key()` 语义的修改。
- 把 context window 当作字幕物理归属范围。

## 2. 模块布局

新增：

```text
vocal_subtitle/physical/
  __init__.py
  timeline.py
  coordinate.py
  context.py
  ir_cache.py
```

`timeline.py` 是 P2.1 领域模型的无模型实现，供 P2.4 的 context builder 使用。其余模块只依赖标准库和时间线模型，不依赖 numpy、模型、网络或磁盘缓存实现。

## 3. 坐标转换契约

### 3.1 `CoordinateMapper`

```python
CoordinateMapper(
    origin_offset: float,
    duration: Optional[float],
    source_id: str,
)
```

规则：

- `origin_offset` 是当前局部窗口起点对应的全局秒数；所有局部到全局转换显式执行 `global = local + origin_offset`。
- 所有全局到局部转换显式执行 `local = global - origin_offset`。
- 不根据数值大小、对象类型或字段名称自动猜测坐标空间。
- `to_global()` 只接收局部范围，`to_local()` 只接收全局范围；使用带 `coordinate_space` 的 `CoordinateRange` 重复转换时抛出 `ValueError`。
- 输入范围必须有限且 `start < end`；有 `duration` 时全局结果必须位于 `[0, duration]`，除非先显式调用 `clamp_global()`。
- 转换不修改输入对象，`map_segment()` 返回新的 `MacroChunkCoordinate` 只读视图。

`MacroChunkCoordinate` 至少保留：`source_id`、`index`、local/global start/end、`overlap_with_prev`、`overlap_with_next`。它只表达坐标映射，不执行 overlap 去重或事件缝合。

### 3.2 `clamp_global`

`clamp_global(start, end)` 只在 mapper 配置了 duration 时把范围裁剪到 `[0, duration]`，裁剪后必须仍满足 `start < end`。它返回新 tuple，不改变输入。

## 4. ContextWindow 构造

核心接口：

```python
build_context_windows(
    timeline: PhysicalTimeline,
    left_context: float,
    right_context: float,
    id_prefix: str = "ctx",
) -> List[ContextWindow]
```

规则：

- 每个 `PhysicalClip` 生成一个 owner window；clip 的顺序只影响返回排序，不改变 clip ID。
- 窗口范围为 `[clip.start-left_context, clip.end+right_context]` 钳制到 `[0, timeline.duration]`。
- `physical_clip_id` 始终保留原 owner；context 扩展不获得字幕归属权。
- ContextWindow 允许相互重叠，不合并相邻或重叠窗口。
- 默认稳定 ID 为 `ctx:{clip_id}:l{left_ms}:r{right_ms}`；同一 clip 和策略只能产生一个窗口。
- 左右扩展量必须是有限非负秒数；参数由上层 profile/config 传入，P2.4 不硬编码 ASR 默认值。
- 构造器是纯函数，返回的新窗口不写回 `PipelineContext`，P2.5 决定如何挂载 shadow 结果。

## 5. IR 缓存契约

### 5.1 指纹

`fingerprint_ir(payload)` 使用递归 JSON-safe 规范化、UTF-8 canonical JSON（排序 key、紧凑分隔符）和 SHA-256，返回 64 位十六进制摘要。字典 key 必须可稳定转换为字符串；非 JSON-safe 对象拒绝，而不是使用不稳定的 `repr()`。

### 5.2 Cache key

```python
make_ir_cache_key(
    *,
    artifact_type: str,
    schema_version: str,
    producer_version: str,
    input_sha256: str,
    audio_duration: Optional[float],
    timeline_fingerprint: str,
    coordinate_policy: str,
    context_policy: str,
    additional_params: Optional[Mapping[str, Any]] = None,
) -> str
```

key payload 至少包含以上字段，以及固定的 `ir_cache_version="ir-cache-v1"`。返回值是 canonical payload 的 SHA-256，不包含路径、token、凭据或不可复现的对象地址。`audio_duration` 明确区分 `None` 和具体数值。

示例 artifact type：`physical_timeline`、`context_windows`、`global_transcript`、`global_speaker_timeline`。

### 5.3 缓存值边界

- IR 对象写入缓存前必须通过 `to_dict()` 或等价 mapping 变为 JSON-safe 字典。
- `encode_ir_value()` 不直接 pickle Python 对象；非 mapping 或不可序列化值拒绝。
- `decode_ir_value(payload, loader)` 校验 mapping 后调用显式 loader；schema、引用、时间或 loader 失败统一返回 `None`，由调用方按 cache miss 处理。
- P2.4 不修改既有 `CacheManager` 的文件路径 key 语义，也不负责改变 TTL。

## 6. 错误和降级

- 坐标参数、时间线 duration、上下文扩展量和缓存 key 的核心字符串字段非法时抛出 `ValueError`。
- 单个 IR 缓存值损坏只返回 miss，不阻断主流程；P2.5 负责记录 shadow diagnostics。
- 任何转换失败都不回写或修改旧 `MacroChunk`、检测结果或 PipelineContext。

## 7. 测试与验收

新增：

- `tests/test_physical/test_timeline.py`
- `tests/test_physical/test_coordinate.py`
- `tests/test_physical/test_context.py`
- `tests/test_physical/test_ir_cache.py`

覆盖：

1. local/global 加减 offset、duration 边界、显式 clamp 和重复转换拒绝。
2. `MacroChunkCoordinate` 保留 overlap 标记且不修改 `MacroChunk`。
3. 每个 clip 生成一个稳定 context window，窗口钳制、重叠和 owner 关系正确。
4. 非法上下文参数、空 ID 和损坏时间线被拒绝。
5. canonical JSON 指纹顺序稳定且内容变化会改变摘要。
6. IR key 隔离 artifact、schema、producer、输入、时间线、坐标和 context 策略。
7. `None` duration 与具体 duration 区分；路径、token 不进入 key。
8. JSON-safe 编码和损坏 loader 的 cache miss 行为。
9. 测试不依赖模型、GPU、HF token、ffmpeg 或网络。

P2.4 完成定义：所有局部/全局时间转换必须显式可追踪，context 不扩大物理归属范围，IR 缓存能被 schema、输入和策略稳定隔离，且不改变既有 Pipeline 与 CacheManager 行为。
