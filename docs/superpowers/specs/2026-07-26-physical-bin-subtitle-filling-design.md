# 物理字幕仓与 ASR 文字灌装设计

日期：2026-07-26

状态：待用户审阅

## 1. 背景与目标

当前 global ASR 已能产出完整词流，但字幕事件仍受到宏观 `PhysicalClip`、通用声学吸附和最终物理钳制的共同影响，导致中文多人、181 单人和英文视频评测出现事件不足、整秒级边界误差和内容覆盖不稳定。

本设计采用“物理字幕仓优先”的灌装模型：融合 VAD/声学检测生成更接近真实发声范围的物理字幕仓；ASR、WhisperX 对齐和 diarization 负责检测文字、词时间、顺序和说话人；文字以整词为单位灌入字幕仓。物理边界可以成为最终字幕时间，但绝不以物理切刀截断词或字符。

目标素材：

- 中文多人；
- `181` 单人；
- 其他单人素材；
- 英文视频评测。

中文双人和英文多人作为回归素材持续验证。

## 2. 核心时间契约

### 2.1 三类时间

1. `asr_word_time`：ASR/对齐模型给出的词级软时间，用于排序、词归属、仓内多句切分和精修候选。
2. `physical_bin_time`：融合 VAD/声学证据生成的物理字幕仓边界，是可验证时优先采用的字幕时间来源。
3. `display_time`：最终导出的时间。单仓单事件时优先等于物理仓边界；仓内有多句时，按完整词边界切分，不能把整个仓时间复制给每一句。

宏观 `PhysicalClip` 仍只负责长音频上下文和物理所有权，不能直接作为字幕仓。短音频的整段 clip 也不能直接生成整段字幕。

### 2.2 不变量

- 物理仓不重叠、按时间排序，并保留来源、边界分辨率、置信度和证据 ID。
- 文字只能按完整词灌装；不在物理边界、字符位置或 Unicode token 中间切割。
- 跨仓词只保留一个完整文本，按最大时间重叠归属；平局按词中点决定，并记录 `cross_bin_word`。
- 物理边界可以扩展显示时间到仓边界，但不能让字幕跨越相邻仓或不连续静音。
- global 事件的时间修改必须能追溯到物理仓或词级精修结果；不得再由通用 `min_duration`、LLM merge 或固定 padding 任意扩张。
- legacy 分段路径保持现有兼容行为，使用旧事件级时间映射。

## 3. 物理字幕仓生成

新增物理仓构建器，输入现有 `PhysicalTimeline.speech_evidence_spans` 和音频，输出 `PhysicalSubtitleBin` 列表。证据优先级为融合边界、ffmpeg skeleton/RMS、Silero/其他 VAD；重叠证据先合并，再按真实静音和边界置信度确定不重叠区间。

每个仓至少包含：

```text
id, start, end, source, confidence,
evidence_ids, physical_clip_id, boundary_resolution_ms
```

仓边界生成规则：

1. 先使用高置信度融合证据的 start/end；
2. 多来源边界不一致时，在可行区间内用音频能量变化和静音谷值择优；
3. 搜索分辨率配置为毫秒级，默认 5ms，结果记录实际采样点分辨率；
4. 过短噪声区间合并，真实静音/换气区间保持断开；
5. 物理仓超出所属宏观 clip 时裁剪到 clip，并记录裁剪原因；
6. 没有可靠物理仓的词不丢弃，进入 `unowned_asr_bin`，沿用词级 ASR 时间并带 `missing_physical_evidence` warning。

物理仓构建失败时 global 任务仍可输出词级事件，但诊断标记为 `physical_bin_unavailable`；`--require-global` 不因仓构建失败而伪造质量通过。

## 4. ASR 文字灌装与事件生成

新增灌装器，按全局词流顺序处理：

1. 为每个词计算与物理仓的重叠时长、词中点距离、speaker 兼容性和证据覆盖；
2. 选择最大重叠仓，整词写入该仓；跨仓词保留完整词并记录 warning；
3. 对每个仓内词流按 speaker 变化、硬断 warning、真实静音、句末标点和行长/时长上限切分；
4. 事件文本由仓内词按原始顺序拼接，中文不插入多余空格，英文保留词间空格；
5. 单仓单事件使用物理仓 `start/end`；仓内多事件使用首尾完整词时间，必要时只在首尾词边界上做物理证据精修；
6. 输出覆盖报告，确保每个 accepted global word 恰好进入一个事件，或进入明确的 rejected/unowned 诊断集合。

禁止行为：

- 用 1 秒、2 秒等固定物理刻度直接切断词；
- 将跨边界句子的前后半段分别当成不同字符内容；
- 丢弃没有 speech evidence 的 ASR 词；
- 将物理仓边界复制给仓内所有字幕句；
- 用通用声学吸附、LLM 合并、最小时长扩展覆盖 global 物理时间契约。

## 5. 后处理与导出

global 路径后处理顺序调整为：

```text
global words
  -> physical subtitle bins
  -> whole-word filling
  -> strict segmentation
  -> optional text-only LLM guard
  -> final coverage/physical validation
  -> export
```

现有 `_clamp_to_physical_envelopes()` 和 `AcousticValidator` 的通用事件端点吸附不再直接改写 global 事件时间。声学校正只允许产生物理仓边界或首尾词的候选时间，并且必须通过词边界、仓范围、speaker 和内容覆盖校验；失败则保留输入时间。

`SubtitleBuilder` 对 global 事件不得因 `min_duration` 扩张时间。短事件只在同 speaker、同仓、无硬断且词流连续时合并，并重新计算覆盖元数据；所有拆分必须保持 `source_word_ids` 完整且不重复。

LLM 只能改文本，不能改仓、时间、词、speaker 或 source word ID。LLM 失败时保留 ASR/灌装原文。

## 6. 诊断与质量门

每个场景报告：

- 物理仓数量、来源、边界分辨率和置信度；
- ASR 词数量、灌装词数量、跨仓词数量、unowned 词数量；
- accepted word 覆盖率、重复率、丢失率；
- `physical_bin_time`、`asr_word_time`、`display_time` 的 start/end MAE、P95 和校正量分布；
- 事件匹配数、文本相似度/CER proxy、speaker 切换和 unknown/mixed 事件；
- 物理越界、事件重叠、跨 speaker、跨不连续仓、LLM 安全门拒绝数。

质量门按目标四类素材分别判定，中文双人和英文多人作为回归门：

- 词内容覆盖率 `>=99%`；重复词、丢词、物理越界为 `0`；
- 最终事件 Start/End MAE 建议 `<=50ms`，P95 `<=100ms`；
- 同时保留 `<=5ms`、`<=10ms`、`<=20ms`、`<=50ms`、`<=100ms` 分桶，不能以平均值掩盖长尾；
- 所有场景必须实际使用 global，任何 fallback 仍使发布门失败。

“毫秒级”表示输出、边界搜索和误差统计均以毫秒为单位；`<=5ms` 作为理想残差观察项，不在缺乏同等精度人工标注和强制对齐证据时作为发布硬门。

## 7. ASR 路径策略

离线默认路径使用 `auto`，但路由契约固定为：

```text
auto
  -> global ASR
  -> global 成功：asr_path=global
  -> global 失败：仅在可恢复错误下进入 segmented
                 asr_path=legacy_degraded
```

路径行为：

- `global`：强制 global，失败直接报错，不降级；
- `segmented`：保留为内部灾备、回归和故障排查入口，不作为生产推荐路径；
- `streaming`：继续使用 segmented，因为实时任务没有完整音频上下文，结果明确标记为 `legacy`；
- global 失败分类继续使用 `dependency_unavailable`、`resource_unavailable`、`execution_failed` 和 `invalid_result`；
- fallback 结果不得进入 global 质量门，报告必须标记为 `legacy_degraded`；
- 缓存身份区分 global、legacy 和 legacy_degraded，禁止跨路径复用。

对外入口收敛为 global 优先：

- WebUI 移除 segmented 选择项，任务提交默认发送 `auto`；
- CLI 帮助和 README 说明 global 优先与失败降级，不把 `--asr-path segmented` 作为常规用法；
- 兼容参数解析继续保留，但标记 deprecated，仅用于测试和故障恢复，并输出明确警告；
- 质量 benchmark 默认运行 `auto --require-global` 或 `global`，segmented 仅作为单独的历史对照命令；
- API 保留旧请求兼容，但不把 segmented 当作默认值；结果始终返回 `asr_path`、`global_attempted`、`fallback_category` 和 `fallback_reason`；
- streaming 页面和接口单独标记为实时 legacy 路径，不与离线 global 质量指标混合。

## 8. 测试计划

新增无模型单元测试：

1. 多来源 evidence 合并为稳定、不重叠物理仓；
2. 单仓单事件使用物理仓边界；仓内多句不重复使用仓边界；
3. 词跨仓时整词归属，不产生半词和重复词；
4. evidence 缺失时词进入 unowned 诊断而不丢失；
5. speaker、静音、标点和长度限制只能在词边界断句；
6. global 后处理不能被 physical clamp、acoustic snap、LLM merge 或 builder 扩张；
7. 覆盖守恒、物理越界、事件重叠和来源词校验能拒绝非法结果；
8. legacy 路径和已有导出测试保持兼容。

真实素材测试：

- 先运行四类目标场景，分别保存物理仓、灌装 trace、最终字幕和对比报告；
- 再运行中文双人、英文多人回归；
- 每次同时比较 global 原始词时间、物理仓时间和最终显示时间，确认误差改善来自边界来源而非 benchmark 匹配放宽。

## 9. 完成定义

- 融合 VAD/声学证据能生成可追溯的物理字幕仓；
- global 词流完整灌装，跨仓词不被截断，覆盖守恒可审计；
- 单仓事件使用物理边界，仓内多句使用完整词边界；
- global 后处理不再通过物理钳制或声学吸附覆盖词级/物理仓时间；
- 四类目标素材的事件完整性和毫秒级统计门可重复运行；
- legacy、中文双人和英文多人回归通过；
- 真实素材报告清楚区分物理仓、ASR 词时间和最终字幕时间。
- 离线默认只把 global 作为生产路径，segmented 仅作为明确标记的灾备/回归路径。
- WebUI、CLI、API、缓存和 benchmark 对 global 主路径与 legacy 降级状态保持一致。
