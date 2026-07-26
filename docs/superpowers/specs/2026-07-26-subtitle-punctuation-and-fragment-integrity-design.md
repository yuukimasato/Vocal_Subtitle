# 字幕标点归属与碎片完整性设计

## 背景

真实测试 `test/中文朗读测试-双人.wav` 的前端生成 ASS 出现了三类问题：

- 字幕文本以 `。`、`,` 或 `,,,` 开头；
- 一个连续词组被拆成孤立后缀，例如 `你中间歇了八` / `回`、`二十分` / `钟`；
- speaker、物理来源和 source word 元数据在边界重建后不完整。

人工 ASS 只作为内容、时间和说话人连续性的参考，不作为事件数量或分段数量的硬约束。

## 目标

1. 标点默认归属于前一个真实内容词，不能成为下一条字幕的首字符。
2. 只在完整词边界分割；相邻短碎片在没有硬边界、speaker 变化或物理证据断裂时合并。
3. 任何事件重建都保留 speaker、physical、source word 和校验元数据。
4. 纯标点事件被过滤，但带真实内容的事件不会因为清理规则丢失。
5. LLM 只能改变文本，不得改变事件时间、speaker 或来源词集合。

## 非目标

- 不强行匹配人工稿的事件条数、标点样式或角色命名。
- 不根据字幕顺序猜测 speaker，也不把真实的多 speaker 结果静默压缩成两人。
- 不通过字符级切片修复长字幕。

## 数据流

```text
global/segmented ASR
  -> physical allocation / event construction
  -> punctuation ownership + short-fragment normalization
  -> strict word-safe segmentation
  -> LLM/acoustic post-processing
  -> speaker-boundary reconstruction
  -> final event normalization and validation
  -> SubtitleBuilder / ASS
```

清理函数必须可重复执行；它在严格断句前和所有后处理完成后各执行一次。重复执行不得改变已经稳定的事件边界或词序。

## 规则

### 标点归属

- 句末标点 `。！？.!?`、软标点 `，、；：,;:` 和右括号/引号跟随最近的前一个内容词。
- 如果 ASR 将标点作为独立词，它与前一个事件合并；若前面不存在内容词，则暂存到后续内容，并在生成事件文本时删除首部标点。
- 事件文本首部连续标点和空白全部移除，但不移除内容词内部或末尾的标点。
- 去除首部标点后没有字母、数字或 CJK 内容的事件直接丢弃。
- 标点不会单独触发 CJK 断句；句末标点只允许在当前内容词已经加入当前组后结束当前组。

### 碎片合并

相邻事件满足全部条件时合并：

- 两侧都有词级来源，词序连续或时间间隔不超过短碎片阈值；
- speaker 相同，或两侧均为 unknown；
- 不存在 `hard_split_before`、speaker conflict 或 discontinuous physical boundary；
- physical clip/source 有交集或物理边界连续；
- 合并后仍满足字幕时长和显示长度限制，除非当前事件是孤立尾部碎片。

优先处理显示长度不超过 2 个 CJK 字符或 4 个字符的短事件，确保 `八回`、`二十分钟` 等完整词组不被留成孤立后缀。

### 元数据继承

从事件派生子事件时复制全部 `SubtitleEvent` 字段，并针对所选词过滤：

- `words` 使用新事件起点的相对时间；
- `source_word_ids` 与所选词一一对应；
- `physical_spans`、`physical_bin_*`、`physical_region_id` 保留来源范围；
- `speaker_id`、`speaker_label`、`logical_sentence_id`、`alignment_warning`、`hard_split_before` 保留；
- 无词级时间戳的混合 speaker 事件保留文本一次并将 speaker 置为 unknown，不猜测归属。

## 实现边界

- 在 `vocal_subtitle/physical/events.py` 修正物理事件文本和 bin 碎片合并，确保标点不会成为下一事件首内容。
- 在 `vocal_subtitle/mapping/strict_segmenter.py` 增加可重复的文本清理、标点事件过滤和短碎片合并；继续禁止字符级拆词。
- 在 `vocal_subtitle/pipeline.py` 的 speaker 边界重建中统一使用完整字段继承，并在最终校验前调用清理逻辑。
- 在 `vocal_subtitle/asr/text_normalizer.py` 补充首部标点清理，避免 segmented 路径绕过共享清理时重新产生问题。

## 测试与验收

单元测试覆盖：

- 标点独立词归并到前一内容事件；首部标点和纯标点事件过滤；
- CJK 短碎片在同 speaker、连续物理来源下合并；硬边界下不合并；
- strict segmenter 派生事件保留全部元数据且词时间相对新起点；
- speaker boundary 重建保留 physical/source/warning 字段；
- 重复执行清理和严格断句结果稳定。

真实回归使用 `test/中文朗读测试-双人.wav`：

- CLI 生成 ASS；
- 启动 WebUI，通过 browser-skill 上传/选择同一音频并导出 ASS；
- 解析输出检查：无首部标点、无纯标点事件、无 `回`/`钟` 这类孤立后缀、speaker 不因重建丢失；
- 内容顺序覆盖人工稿的可听文本，允许生成事件数多于人工稿。

