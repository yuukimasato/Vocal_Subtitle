# 全局说话人识别与物理边界融合设计

> 日期：2026-07-24
> 状态：方案已确认，待实现
> 范围：说话人识别、物理语音边界、ASR 分段、字幕合并

## 1. 背景

当前项目已经具备较完整的物理时间边界链路：Silero VAD、ffmpeg silencedetect、RMS 边界精修、声学骨架和 `AcousticValidator`。这些模块能够给出较精确的语音开始和结束时间，应继续作为物理时间尺度的主要来源。

当前说话人识别则是事件级固定窗口聚类：在 [pipeline.py](../../../vocal_subtitle/pipeline.py) 的 `_run_event_speaker_clustering()` 中以 3 秒窗口提取 embedding，再用事件中点映射到最近窗口。这种做法无法表达一个窗口内的 A→B，也无法阻止一个已经包含两个人的 ASR 事件继续作为单一 speaker 输出。

## 2. 实际问题定位

### 2.1 固定窗口不能表达窗口内换人

当前实现使用固定 3 秒窗口和 1 秒 hop，并根据字幕事件中点选择窗口 speaker。一个窗口内出现 A→B 时，整条字幕只能获得一个 speaker id，无法拆分内容。

### 2.2 宏观块和骨架段会破坏全局身份

多块路径和 skeleton 路径都会让每个块/段从 speaker 0 重新开始，再通过偏移量避免冲突。这只能避免编号碰撞，却不能识别“后一个块的 speaker 0 就是前一个块的 speaker 0”，因此同一个人会被拆成多个角色。

### 2.3 间隙降级逻辑会制造假 speaker

`_gap_based_speaker_assignment()` 在检测到切换时递增 speaker id，不能稳定表达 A→B→A；间隙只能描述停顿，不能单独证明说话人发生变化。

### 2.4 现有聚类质量门控存在缺陷

`speaker_clusterer.py` 的质量门控使用未定义的 `n_segments`。此外，单簇 silhouette 被定义为 1.0，不能证明单簇结果真的正确。

### 2.5 现有合并器无法修复已经混合的事件

当前 LLM/规则合并器可以阻止两个已有事件跨 speaker 合并，但无法识别单个事件内部已经混入两个 speaker。必须在 ASR 前或词级映射阶段建立 speaker 硬边界。

## 3. 外部技术依据

以 2026-07-24 可访问资料为准。用户要求的 2026-07-25 资料日期晚于当前环境日期，因此不将其作为已发布资料引用。

- [pyannote-audio](https://github.com/pyannote/pyannote-audio)：最新 release 为 4.0.7（2026-06-30）。官方推荐 `pyannote/speaker-diarization-community-1` 作为开源 speaker diarization pipeline，并提供 regular 与 `exclusive_speaker_diarization` 输出。
- [pyannote-audio CHANGELOG](https://github.com/pyannote/pyannote-audio/blob/main/CHANGELOG.md)：4.0 的 community-1 使用 VBx clustering，并改进 speaker counting 和 speaker assignment；4.0 同时明确了完整 pipeline 与 embedding 模型的职责差异。
- [WhisperX](https://github.com/m-bain/whisperX)：先进行独立 diarization，再将 speaker labels 分配到词级时间戳；其文档也明确重叠语音仍不是完全解决的问题。
- [NVIDIA NeMo Speaker Diarization](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html)：区分端到端 Sortformer 和 VAD + speaker embedding + clustering 级联方案，说明 diarization 应是独立的“谁在何时说话”阶段。
- [SpeechBrain SpeakerRecognition](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.speaker.html)：ECAPA 适合 speaker recognition/verification embedding，不能替代完整的 speech activity、speaker change 和 diarization pipeline。

## 4. 目标与非目标

### 4.1 目标

1. 保留当前物理边界精度，不用 diarization 时间戳粗暴覆盖 RMS/ffmpeg 物理边界。
2. 对任意交替模式保持全局稳定身份，例如：

   ```text
   A → B → A → A → B → C → B
   0 → 1 → 0 → 0 → 1 → 2 → 1
   ```

3. 不依赖说话间隙创建 speaker，也不依赖说话间隙判断 speaker 切换。
4. 一个物理语音区间包含换人时，能按 speaker turn 切分成多个原子语音区间。
5. 任意字幕合并、帧级衔接和去重都不能跨越已确认的 speaker 边界。
6. 在单块、宏观切块和 ffmpeg skeleton 三条路径中使用同一份全局 speaker timeline。
7. pyannote、embedding 和依赖不可用时明确降级，不伪造多个 speaker id。

### 4.2 非目标

1. 不承诺单通道、完全重叠语音中每个词都能可靠归属；这是音频可观测性限制。
2. 不让 LLM 角色命名参与声学 speaker identity 判断。
3. 不在本阶段重写 Silero、ffmpeg、RMS 或声学校验算法本身，除非融合接口需要最小改动。

## 5. 系统不变量

以下规则必须通过单元测试和集成测试验证：

1. speaker id 是全文件作用域的 canonical identity，不能按块、骨架段或字幕事件重新编号。
2. 间隙长度只能影响同一 speaker 的字幕组合和排版，不能创建新 speaker 或改变 speaker identity。
3. 一个 `SubtitleEvent` 不得跨越两个不同 `speaker_id` 的确认 turn。
4. 同一 speaker 再次出现必须复用原 speaker id，不得因间隙或宏观块产生新 id。
5. 角色显示名称变化不能改变 `speaker_id`。
6. 不同 speaker 的事件即使文本相似、时间重叠或间隙很短，也不能被去重为一个事件。
7. 无法可靠判定的重叠/混合内容必须标记 `unknown` 或 `mixed`，不能伪装成确定 speaker。

## 6. 总体架构

```text
输入音视频
  → 人声分离
  → 加载完整 vocals
  ├─ 物理边界支线
  │    Silero VAD + ffmpeg VAD + RMS
  │    → fused speech regions
  │    → ffmpeg acoustic skeleton
  │
  └─ 全局说话人支线
       pyannote community-1
       → regular speaker turns
       → exclusive speaker turns
       → overlap diagnostics

两条支线
  → physical speech region 与 exclusive speaker turn 求交集
  → AtomicSpeechSpan
  → 宏观切块 / skeleton 处理 / ASR
  → 词级时间戳映射
  → speaker-aware SubtitleEvent
  → 同 speaker 帧衔接与语义合并
  → AcousticValidator 最终物理校验
  → 字幕输出
```

pyannote 全文件只运行一次，位置在完整 vocals 加载后、宏观切块和 skeleton 分段前。宏观切块只用于性能和上下文管理，不再负责 speaker identity。

## 7. 数据结构与模块边界

### 7.1 `diarization/base.py`

增加全局 diarization 数据结构，同时保留旧 `DiarizationEngine` 以兼容现有测试和 fallback：

```python
@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker_id: int
    confidence: float | None = None
    overlapped: bool = False

@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    exclusive_turns: list[SpeakerTurn]
    speaker_count: int
    backend: str
    status: str
    overlap_duration: float = 0.0
    diagnostics: dict = field(default_factory=dict)

@dataclass
class AtomicSpeechSpan:
    start: float
    end: float
    speaker_id: int | None
    physical_source: str
    speaker_source: str
    overlapped: bool = False
```

`SpeakerTurn` 使用全局秒数。所有后续路径只能消费这份全局结果，不得重新局部聚类或偏移 speaker id。

### 7.2 新增 `diarization/pyannote_engine.py`

职责：

- 延迟加载 `pyannote/speaker-diarization-community-1`。
- 支持 Hugging Face token、环境变量和本地离线模型目录。
- 对完整 vocals 运行一次 pipeline。
- 读取 regular 与 exclusive diarization tracks。
- 将 `SPEAKER_00` 等原始标签按首次出现顺序映射到连续稳定整数。
- 关闭或显式配置 pyannote telemetry，避免在无提示情况下发送可选使用统计。
- 不输出 token、完整音频路径或敏感信息到普通日志。

### 7.3 新增 `diarization/turn_reconciler.py`

职责：

- 清理和排序 speaker turns。
- 裁剪越界区间，移除无效区间。
- 以 physical region 与 exclusive turn 的交集生成 atomic spans。
- 在 speaker turn 靠近物理边界时使用 collar 规则吸附到物理边界。
- 对相邻同 speaker 区间做受控合并。
- 对 overlap 和无 speaker 覆盖区间保留诊断状态。
- 不使用 gap 单独推断 speaker identity。

## 8. 物理边界与 speaker turn 融合

### 8.1 物理边界链路保留

以下模块继续保留并作为物理时间轴来源：

- Silero VAD：高召回 speech candidate。
- ffmpeg VAD：粗语音区间和 acoustic skeleton。
- RMS：局部 onset/offset 精修。
- `BoundaryFusion`：三方法边界融合。
- `MergeStrategy`：在 speaker 保护边界下进行物理区间处理。
- `AcousticValidator`：最终物理边界校验和报告。

pyannote 的 turn 边界不能直接替换物理起止时间。它只提供 speaker change 候选和 speaker identity。

### 8.2 融合顺序

1. 加载完整 vocals。
2. 全局运行 pyannote，得到 exclusive turns。
3. 运行现有 Silero + ffmpeg + RMS 物理边界链路。
4. 在 physical regions 进入普通相邻合并之前，按 speaker turns 切分内部换人。
5. `MergeStrategy` 接收 protected speaker boundaries，仅允许在同 speaker 范围内合并。
6. 产生 `AtomicSpeechSpan` 后进入 ASR。

如果 physical region 覆盖 A 和 B：

```text
physical:       [----------- A + B -----------]
speaker turns:  [------ A ------][------ B -----]
result:         [------ A ------][------ B -----]
```

如果 A 在短暂停顿后继续说话：

```text
physical:       [--- A ---]  [--- A ---]
speaker turn:   [------------- A -------------]
subtitle rule:  可在同 speaker 且满足字幕规则时组合
```

### 8.3 边界 collar

`boundary_collar_ms` 默认 80ms：

- speaker turn 边界距离 physical boundary 不超过 collar 时，优先使用物理边界。
- speaker turn 位于连续语音内部时，保留 speaker turn 作为硬换人边界，RMS 只用于校验，不得把换人边界合并掉。
- collar 不能跨越另一个 speaker turn，也不能让事件超出 physical speech region。

## 9. ASR 和字幕映射

### 9.1 有词级时间戳

当 `word_timestamps=true` 时：

1. 对每个词计算与 atomic span 的时间重叠。
2. 将词归入 speaker id 最大重叠的 span。
3. 连续同 speaker 词组成字幕事件。
4. 词组边界不能跨不同 speaker id。
5. 使用原始词时间戳和 physical boundary 共同生成事件 start/end。

### 9.2 没有词级时间戳

如果一个 ASR 结果覆盖多个 speaker turn：

1. 只对该混合区间按 atomic span 局部重识别。
2. 成功时使用局部结果重新映射。
3. 失败时保留 `mixed/unknown`，禁止跨 speaker 合并。

不允许将一整段跨人的文本简单归给持续时间最长的 speaker 作为正常结果。

## 10. 后处理和合并保护

最终顺序：

```text
physical + speaker intersection
  → ASR / word timestamps
  → speaker-aware SubtitleEvent
  → frame seamless stitching（同 speaker）
  → LLM/rule merge（同 speaker）
  → AcousticValidator
  → EndTimePostValidator
  → final dedup
```

具体要求：

- `TimeMapper._merge_gaps()` 在映射前就应获得 speaker id，跨 speaker gap 不得扩展前一个事件。
- `apply_frame_seamless_stitching()` 继续保持不同 speaker 不衔接。
- `_run_llm_merge()` 使用数值 `speaker_id` 作为硬约束，不使用 `speaker_label` 作为 identity key。
- LLM 合并输出必须继承并验证 speaker id；任何合并组包含多个 speaker id 时拒绝应用。
- `_deduplicate_overlapping()` 只允许删除同 speaker 的相似重复事件；不同 speaker 的相似文本必须保留。
- 最终校验后再次检查事件是否跨越 speaker turn；若跨越，拆回原子事件而不是扩展时间范围。

## 11. 配置设计

保持旧配置可解析，并将默认主路径改为：

```yaml
diarization:
  enabled: true
  backend: "pyannote"
  model_ref: "pyannote/speaker-diarization-community-1"
  hf_token: ""
  cache_dir: ""
  offline: false
  min_speakers: null
  max_speakers: null
  use_exclusive: true
  overlap_policy: "exclusive"
  min_turn_duration: 0.20
  same_speaker_merge_gap: 0.12
  boundary_collar_ms: 80
  fallback_backend: "embedding"
```

当视频已知是两个人时，可设置 `min_speakers: 2` 和 `max_speakers: 2`，减少 speaker counting 歧义；未知人数时保持 null，由模型估计。

现有 `speaker_embedding` 配置保留，用于 fallback；现有 `engine: agglomerative` 配置映射为兼容 fallback，不再是默认主路径。

## 12. 降级与错误处理

### 12.1 pyannote 不可用

记录 `diarization_status=degraded` 和结构化失败原因，随后尝试完整音频范围的 embedding fallback。fallback 必须使用全局特征和稳定 cluster identity，不能使用按 gap 递增 speaker id。

### 12.2 embedding 也不可用

保留物理字幕，speaker id 为 unknown/null；不生成“说话人 A/B/C”作为确定识别结果。前端和日志显示降级状态。

### 12.3 无有效 diarization turn

不修改物理边界；将对应事件标记 unknown，并禁止跨不确定区域做会改变时间覆盖的语义合并。

### 12.4 重叠语音

使用 exclusive track 生成唯一字幕归属，同时使用 regular track 统计 overlap。对于无法可靠归属的区间输出 `mixed/unknown`，不伪造确定 speaker。

## 13. 缓存与可观测性

diarization 缓存键必须包含：

- 输入音频内容哈希。
- backend 和 model ref/revision。
- min/max speakers。
- exclusive/overlap policy。
- boundary collar 和 turn filtering 参数。

`PipelineStats` 和日志新增：

```text
diarization_backend
diarization_status
speaker_count
physical_region_count
atomic_span_count
overlap_ratio
mixed_event_count
cross_speaker_merge_blocked
fallback_reason
```

失败日志必须包含 backend、模型和阶段，但不得包含 token。

## 14. 测试和验收

### 14.1 单元测试

新增 `turn_reconciler` 测试：

- 一个 physical region 被 A/B turn 正确切分。
- A→B→A 结果为两个稳定 speaker id。
- A→B→A→A→B→C→B 结果保持 `0,1,0,0,1,2,1` 的身份关系。
- 随机化 0ms、50ms、200ms、500ms、2s 间隙不会改变 speaker identity 逻辑。
- 同 speaker 区间仅按配置允许合并。
- 不同 speaker 即使无间隙也不能合并。
- turn 越界、重叠、空区间和短区间处理正确。

### 14.2 Pipeline 集成测试

使用 mock pyannote backend 和确定的物理 regions，覆盖：

- 单块路径。
- 宏观切块路径。
- ffmpeg skeleton 路径。
- 全局 turns 在三条路径中保持一致。
- 同一 speaker 跨块不产生新 id。
- 一个跨人 ASR 片段被拆分。
- LLM 合并不能重新拼回跨 speaker 事件。
- pyannote/embedding/全部后端失败时的显式降级。

### 14.3 真实音频评估

支持 RTTM 或等价人工标注，统计：

- DER/JER。
- speaker count accuracy。
- speaker purity。
- speaker fragmentation。
- cross-speaker merge count。
- 物理边界误差。
- overlap ratio 和 unknown/mixed ratio。

最低验收目标：

1. 双人 A→B→A 样本输出两个稳定身份。
2. 多人 A→B→A→A→B→C→B 样本中同一角色的 speaker id 保持一致。
3. 已确认 speaker 边界上的跨人合并数为 0。
4. 物理边界误差不劣于当前基线。
5. 后端失败时输出状态可解释，不产生伪造 speaker。

### 14.4 技术边界

对完全重叠、严重噪声、极短且无可辨识声学信息的单通道语音，不能承诺每个词都能正确归属。系统应通过 overlap/mixed/unknown 标记暴露不确定性，而不是把不确定结果当作可靠身份。

## 15. 实施顺序

1. 新增数据结构和 pyannote backend，加入独立加载/缓存/错误处理测试。
2. 新增 turn reconciler，先用 mock turns 完成物理区间交集和 speaker 稳定性测试。
3. 将全局 diarization 接入 pipeline，在宏观切块和 skeleton 之前运行一次。
4. 修改 MergeStrategy、TimeMapper 和后处理，使 speaker turn 成为硬保护边界。
5. 移除默认路径中的事件级固定窗口聚类、speaker offset 和 gap-based speaker 轮换。
6. 接入无词级时间戳的局部重识别和 embedding fallback。
7. 增加三条 pipeline 路径回归测试和真实 RTTM 评估入口。
8. 更新配置、README、说话人嵌入说明和运行诊断文档。

实现完成后，旧的 `SpeakerDiarizer` 和 `FeatureExtractor` 不立即删除，保留为兼容 fallback，待真实音频评估确认后再决定是否移除。
