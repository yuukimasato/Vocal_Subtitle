# 字幕创建逻辑修正设计

日期：2026-07-27

## 背景

当前全局识别链路已经能够保留物理语音边界和全局说话人身份，但展示结果仍可能出现两类问题：

1. diarization 未覆盖的短词被输出为无标签事件；连续 unknown 片段无法通过现有的单片段修复逻辑恢复。
2. 物理语音事件直接接近导出结果，字幕偏碎；人工字幕可以作为问题参考，但不能作为不同音频的条数或参数基准。

另外，音频尾部可能存在物理语音但没有被全局 ASR 分配词的情况。此时不能通过延长上一条字幕或猜测文本来掩盖覆盖缺口。

## 目标

1. 保留全局 diarization 作为确定性 speaker 身份的唯一来源。
2. 在证据充分时，将连续 unknown 短片段继承为相邻同一 speaker；证据不足时继续保留 unknown。
3. 将底层识别事件和最终展示字幕分离，按照通用字幕创建规则合并事件。
4. 识别并恢复物理尾部未覆盖语音；恢复失败时明确输出 degraded 诊断。
5. ASS、SRT、WebUI 预览和诊断报告使用同一套最终事件结果。

## 非目标

- 不以人工 ASS 的字幕条数、断句或时间作为参数拟合目标。
- 不恢复旧版按停顿交替生成 A/B/C 的 speaker fallback。
- 不把 unknown 强制改写成已知 speaker。
- 不在无法取得 ASR 证据时合成缺失文本。
- 不更换当前 ASR 或 diarization 模型。

## 方案

### 1. 事件分层

管道内部保留两类结果：

- `recognition_events`：保留词级时间、物理 envelope、physical spans、source word ids、speaker provenance 和 warning，用于诊断、回溯和后续修复。
- `display_events`：从 recognition events 构建，供 ASS/SRT/VTT 和 WebUI 展示。合并不删除底层来源，必须合并 provenance。

展示层合并的必要条件：

- 两侧具有相同的确定性 `speaker_id`，或两侧均为 unknown；
- 物理 clip/region 所有权兼容；
- 没有 `hard_split`、`speaker_conflict` 或 `discontinuous_physical_boundary`；
- 间隙不超过配置阈值；
- 合并后满足最大时长、最大字数和最多行数；
- 句末强标点和明确 speaker 切换优先阻止合并。

展示层合并使用词边界和真实事件范围，不能使用字符比例重新切开中文词。事件合并后的时间范围使用来源事件的 union，并继续满足物理边界约束。

### 2. 连续 unknown 继承

在正常分段前处理连续 unknown run，而不是只检查单个 unknown 事件。只有同时满足以下条件才继承：

- unknown run 前后最近的确定性 speaker 相同；
- unknown run 每个事件都只包含短词/短片段，且总时长不超过配置上限；
- run 内事件和两侧事件属于同一 physical region/clip；
- 事件间隙不超过配置上限；
- 没有 hard boundary、overlap、speaker conflict 或 discontinuous physical warning；
- 文本不是独立语气词集合中的词，例如“咦”“哎”“哦”“嗯”等。

继承时写入 `speaker_id`、通用 `speaker_label` 和修复原因诊断；不改变原始的物理时间边界。ASS 使用继承后的标签，SRT/VTT 使用相同标签格式。

### 3. 尾部覆盖恢复

在 global word allocation 后比较最后一个有效词和最后一个物理语音区间：

- 记录 `transcript_end`、`last_physical_speech_end`、`tail_gap_seconds`；
- 尾部 gap 小于恢复阈值时只诊断，不额外识别；
- 尾部 gap 达到恢复阈值时，对尾部物理区间加上下文 collar 执行局部 ASR；
- 使用现有 overlap dedup 和 physical allocation 规则合并恢复词；
- 恢复失败或没有可用词时保留已有合法字幕，标记 degraded，不延长上一条字幕、不填充占位文本。

尾部恢复后的事件仍需经过严格分段、物理边界校验和最终格式量化。

### 4. 诊断和前端契约

最终结果增加或保持以下指标：

- recognition event count；
- serialized/display event count；
- unknown event count；
- unknown repair count 与 repair reasons；
- physical bin coverage；
- transcript end、physical speech end、tail gap；
- recovery attempted/status；
- speaker backend、speaker count、speaker status。

WebUI 重建事件时必须保留 speaker、physical、word provenance 和 hard-boundary 字段。前端展示的字幕数必须使用最终序列化后的事件数。

## 错误处理

- 全局 diarization 失败：继续输出字幕，但 speaker 保持 unknown，结果标记 degraded。
- unknown 继承条件不完整：不继承，不使用停顿交替猜测。
- 展示层合并违反 speaker 或 physical ownership：拒绝合并。
- 尾部恢复失败：保留已有结果并暴露缺口，不伪造文本。
- ASS/SRT 量化后发生零长度或重叠：由最终 validator 丢弃或修正，并记录统计。

## 测试与验收

新增或扩展测试覆盖：

1. 单个和连续 unknown run 的同 speaker 继承。
2. 两侧 speaker 不同、物理区域不同、存在 hard boundary 时禁止继承。
3. 独立语气词保持 unknown。
4. 同 speaker 展示层合并保留 source word ids、physical spans 和 speaker label。
5. 跨 speaker、跨 physical region 和超过时长/字数限制时禁止合并。
6. 尾部未覆盖触发恢复诊断；恢复成功去重；恢复失败标记 degraded 且不补造文本。
7. ASS/SRT/VTT 输出与 WebUI 重建使用相同的最终事件数和标签。

本次三文件 fixture 的验收不要求复现人工版条数，重点检查：

- 所有文本来源事件可追溯；
- 连续短 unknown 在证据充分时被修复，独立语气词不被误标；
- 输出不跨 speaker、不跨 physical owner 合并；
- 尾部物理语音缺口不会静默丢失；
- 输出字幕遵守配置的时长、字数、行数和时间边界约束。
