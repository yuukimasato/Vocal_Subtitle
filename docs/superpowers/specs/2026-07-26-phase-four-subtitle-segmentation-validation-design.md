# 阶段四：字幕断句与最终校验设计

日期：2026-07-26

状态：设计已确认，待实现

关联设计：

- [离线高精度字幕系统总设计](../../Vocal_Subtitle-离线高精度字幕系统设计文档.md)
- [阶段三全局转录与词级分配](2026-07-26-phase-three-global-transcription-design.md)

## 1. 目标与范围

阶段四将阶段三的词级全局事件转换为可交付字幕。核心原则是：词边界优先于字符边界，物理范围优先于排版便利，speaker 和真实停顿优先于固定时长。所有 profile 和 ASR 路径共享同一套安全约束；global ASR 仍保持显式启用，旧分段 ASR 继续作为默认兼容路径。

本阶段包含：

- 默认严格断句策略；
- 物理 clip、speech evidence、speaker 和来源词约束下的事件合并/拆分；
- 帧级衔接、微间隙合并和 LLM 合并的统一硬边界；
- LLM 文本优化的来源词、当前事件和语言安全门；
- 字幕导出前的最终物理校验与诊断统计；
- 不依赖模型、GPU、网络的单元和集成测试。

本阶段不包含：

- 切换 global ASR 为默认路径；
- 在词中间切割字符或生成半词；
- 自动翻译、跨条目重写或修改 speaker/物理时间的 LLM 行为；
- 重新设计 VAD、diarization 或 WhisperX 模型。

## 2. 输入与不变量

输入事件可以来自阶段三的 `GlobalSubtitleEvent.to_subtitle_event()`，也可以来自旧分段 `TimeMapper`。旧事件没有词级来源时，校验器只执行可获得的时间、speaker 和物理 envelope 检查，不凭空补造来源词。

阶段四必须维持以下不变量：

1. 事件时间为有限值，`0 <= start < end <= audio_duration`；
2. global 事件的 `start/end` 不得超出其 `physical_start/physical_end`；
3. `physical_spans` 只能来自已有物理 clip，不被排版或 LLM 扩展；
4. 一个事件不得包含不同 speaker、不可连续的物理 clip 或被硬断开的成员；
5. `source_word_ids` 只允许来自事件当前词集合，拆分时按词边界完整分配；
6. LLM 只能改变当前事件的 `text`，不能改变时间、词、speaker、物理 span、来源词和逻辑句 ID；
7. 任一安全门失败时保留 ASR 原文，主流程继续完成；
8. 所有事件重新编号后保持时间顺序，删除空事件不产生编号空洞。

## 3. 严格断句器

文件：`vocal_subtitle/mapping/strict_segmenter.py`

提供纯数据接口：

```python
segment_events(
    events: Sequence[SubtitleEvent],
    *,
    config: StrictSegmentationConfig | None = None,
    audio_duration: float | None = None,
) -> SegmentationResult
```

断句器先按时间排序，再按词边界构造候选组。优先级如下：

1. 不同 speaker：硬断；
2. speaker unknown/mixed 与已知 speaker：硬断，不猜测身份；
3. 不连续 physical clip 或不同 physical owner：硬断；
4. `alignment_warning` 中的不可合并警告：硬断；
5. 明显句末标点：优先断句，缩写和小数点例外；
6. 词间真实静音/换气 gap：默认断句；短 gap 只有在同 speaker、同一连续物理区域且前文不是句末时允许合并；
7. 达到最大时长、最大行数或最大字符数：只在词边界拆分；
8. 仅有旧事件而没有词信息时，使用事件级边界和文本标点，不能伪造词级来源。

断句器不直接修改 `PhysicalClip`。拆分事件时，子事件继承 speaker、物理范围和 warning，并按子事件词集合过滤 `words`、`source_word_ids`；`logical_sentence_id` 保留父句 ID，另增加稳定的子序号元数据供诊断使用。跨物理边界的整词保持完整，不在字符中间拆分。

## 4. 合并约束

### 4.1 统一兼容性判断

新增共享判定逻辑，供 `SubtitleBuilder`、`AcousticValidator`、帧级衔接和 `LLMMergeEngine` 使用。两个事件只有同时满足以下条件才可合并：

- speaker ID 相同；双方 unknown 时才允许按 unknown 合并，unknown 不得吸附到已知 speaker；
- 两侧 physical span 有相同 clip，或 clip 在时间轴上连续且没有未覆盖静音；
- 不存在硬断警告；
- 词序、时间顺序和来源词集合可拼接；
- 合并后仍满足最大时长/字符和 physical envelope。

合并后去重拼接 `words`、`source_word_ids` 和 `physical_spans`，warning 使用稳定去重；不改变已有词时间。

### 4.2 LLM 合并

LLM 只接收允许候选的文本和只读元数据。请求结果必须满足：

- 返回键与候选键完整一致；
- 合并组只能引用当前候选事件；
- 不能跨 speaker、physical owner 或硬断边界；
- 文本只作为显示文本候选，不得带回新的时间或来源词；
- 解析失败、超时、置信度不足或规则校验失败时使用规则回退。

## 5. LLM 文本优化安全门

`Pipeline._run_llm_optimize()` 保留现有相似度和长度限制，并增加事件级来源约束：

1. 对每条 event 保存 `asr_text` 作为不可变基线；
2. 只比较当前 event 的 ASR 文本和候选文本，不允许跨 event 复制内容；
3. 保护数字、单位、专名候选、speaker 边界和语言脚本；
4. 通过安全门后才写入 `llm_text`/`text`，否则恢复 ASR 文本；
5. `physical_*`、`physical_spans`、`words`、`source_word_ids`、`logical_sentence_id` 和 `alignment_warning` 始终保留；
6. 实际没有改变的结果不写入虚假的对比修改记录。

## 6. 最终物理校验器

文件：`vocal_subtitle/mapping/final_validator.py`

提供：

```python
validate_events(
    events: Sequence[SubtitleEvent],
    *,
    audio_duration: float | None = None,
    strict: bool = True,
) -> FinalValidationResult
```

校验器按“诊断、修复、删除”三类结果处理：

- 可安全钳制的显示时间越界：钳制到已有 physical envelope，并记录计数；
- 事件完全落在物理范围外、时间反向、来源词为空且没有 legacy 物理信息：删除并记录原因；
- 来源词不属于事件、物理 span 引用未知 clip、跨 speaker 合并或不可连续 physical span：严格模式拒绝事件/合并结果并记录原因；
- LLM 只改变文本时不触碰音频时间和来源字段。

校验器在以下两个位置运行：

1. `_post_process_events()` 完成声学校验后；
2. clean ASR 导出和 LLM 导出各自完成最后一次事件修改后、传入 `SubtitleBuilder` 前。

`SubtitleBuilder` 内部的短事件合并和长事件拆分也必须调用同一兼容性逻辑，并在返回前执行轻量 envelope 校验，防止排版步骤重新扩大事件范围。

## 7. 配置与可观测性

新增 `StrictSegmentationConfig`，默认启用：

- `enabled: true`；
- `silence_gap`、`max_duration`、`max_chars_cjk`、`max_chars_latin`；
- `split_on_sentence_end: true`；
- `allow_short_same_owner_merge: true`。

保留现有 subtitle 配置作为兼容输入，新的严格断句参数通过 `subtitle.strict_segmentation` 加载；缺省值与当前规则一致，不改变旧 profile 的基本时长/字符上限。

统计至少包含：输入/输出事件数、硬断次数、静音断句次数、speaker 断句次数、物理 owner 断句次数、词级拆分数、物理钳制数、删除数、LLM 安全门拒绝数和最终 warning 数。

## 8. 测试与验收

新增 `tests/test_phase_four.py` 及必要的 mapping/merging 回归测试：

1. 同 speaker 短 gap 可按连续物理 owner 合并；换气 gap 默认断开；
2. 不同 speaker、unknown/known、不同 clip 和硬断 warning 强制断开；
3. 过长事件只在词边界拆分，词文本和 source word ID 不丢失；
4. 跨物理边界整词保持完整，不生成半词；
5. Builder、micro-gap、LLM merge 都不能绕过物理兼容性；
6. LLM 跨条目搬运、翻译、非法键、改时间、改来源词均回退 ASR；
7. final validator 能钳制合法 envelope 越界并拒绝物理范围外事件；
8. clean/LLM 两条导出路径都执行最终校验；
9. 默认配置和旧分段路径保持兼容；
10. 无模型、无网络环境下阶段三和阶段四测试全部通过。

## 9. 完成定义

- strict segmentation 可被 Pipeline 默认路径和 global 事件路径共同调用；
- 所有合并入口使用同一 physical/speaker/source-word 兼容性判断；
- LLM 不能改变音频时间和来源词归属，异常稳定回退；
- 最终导出前的校验有结构化统计和可追溯 warning；
- 阶段三事件元数据在拆分、合并、LLM 优化和导出后仍完整；
- 阶段四测试覆盖上述不变量，且不需要外部模型或网络。
