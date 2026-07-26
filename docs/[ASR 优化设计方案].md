# ASR 识别准确率与字幕安全优化设计

> 日期：2026-07-24
> 状态：已批准并实现（2026-07-24）
> 范围：语言识别稳定性、混合语言显式开关、ASR 后处理安全性

## 1. 背景与问题

项目当前的离线链路为“人声分离 → VAD/声学骨架 → 片段合并 → ASR → 边界处理 → 字幕输出”，并支持 faster-whisper、whisper.cpp 和 FunASR。现有测试主要覆盖配置对象、模拟结果和边界逻辑，缺少真实语言决策与后处理安全回归。

代码审查发现：

- `ASRConfig.language` 默认为空，默认模板也没有锁定语言。
- Pipeline 已做全局语言检测，但低置信度片段会自动以 `language=None` 重识别，并仅按 `avg_logprob` 选择结果，没有验证候选语言证据。
- whisper.cpp 虽然支持语言参数，Pipeline 创建引擎时没有传入配置语言；其默认语言检测能力也没有纳入统一语言策略。
- `TextNormalizer` 默认执行编号恢复和内置专名替换，可能把本来正确的 ASR 文本改成错误文本。
- LLM 后处理已有键匹配、相似度和跨条目检查，但管道仍会直接应用返回文本，缺少统一的语言脚本、长度和小范围修改门控。
- 片段转写缓存没有完整包含语言模式及关键解码参数，容易复用旧策略生成的结果。

用户确认的产品策略是：默认按单一语言处理；只有用户明确选择多语言混合时，才允许片段级语言切换。LLM 可以基于语言上下文修正错字，但不能翻译、搬运或重写字幕内容。

## 2. 目标与非目标

### 2.1 目标

1. 默认模式在整段音频上确定一次主语言并锁定，避免短 VAD 片段造成语言漂移。
2. 提供 CLI、Web UI、YAML/API 一致的多语言混合开关。
3. 混合模式只在有足够语言概率和识别收益时接受切换。
4. 记录语言代码、语言概率和决策来源，便于日志、缓存和诊断。
5. 默认后处理只做安全规范化；LLM 错误修改自动回退，合规的小范围错字修正可以保留。
6. 让缓存区分会影响识别结果的语言和解码策略。

### 2.2 非目标

- 本轮不引入多模型投票或云端 ASR。
- 本轮不修改人声分离、VAD 和声学边界算法的整体策略。
- 本轮不把字幕翻译成目标语言。
- 本轮不承诺未经真实标注音频评测的 WER/CER 数值提升。

## 3. 外部技术依据

截至 2026-07-24 核对的上游说明：

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)：`transcribe` 返回包含 `language` 和 `language_probability` 的信息对象；转写段是惰性生成器，需要完整迭代才能执行完识别；支持 `language`、`condition_on_previous_text`、词级时间戳和 VAD 参数。
- [OpenAI Whisper](https://github.com/openai/whisper)：多语言模型支持语言检测与语言指定，语言指定和翻译任务是不同决策，项目不应把识别误用为翻译。
- [whisper.cpp CLI](https://github.com/ggml-org/whisper.cpp/tree/master/examples/cli)：当前 CLI 支持 `--language`、`--detect-language`、`--prompt` 和 `--translate`；本项目只使用语言识别相关参数，保持翻译关闭。
- [FunASR](https://github.com/modelscope/FunASR)：当前仓库区分 Paraformer-zh、Fun-ASR-Nano、MLT-Nano、SenseVoice 等不同语言范围的模型；语言能力应由模型配置决定，不能把旧版中文模型当成通用多语言模型。

本地环境核验到 `faster-whisper 1.2.1`，Python 3.12.3；FunASR 当前虚拟环境未安装。

## 4. 设计方案

### 4.1 语言配置

在 `ASRConfig` 增加：

```yaml
asr:
  language: null
  language_mode: single
  language_detection_min_probability: 0.65
  mixed_language_min_probability: 0.85
  mixed_language_min_logprob: -1.5
  mixed_language_min_gain: 0.15
```

`language_mode` 只允许 `single` 或 `mixed`，默认 `single`。

- 显式指定 `language` 时，两个模式都以该语言为主语言。
- `single` 且未指定语言时，对完整人声音频做一次全局检测并锁定返回语言。低于阈值时仍记录低置信度并锁定最佳候选，不进行片段级换语言；若引擎无法返回语言候选，则记录可操作错误，提示使用 `--language`。
- `mixed` 先按主语言识别。片段仅在主语言识别置信度低于阈值、自动检测结果有语言概率、候选语言不同且置信度收益达到 `mixed_language_min_gain` 时切换。
- 语言模式不改变 ASR 的翻译任务；所有引擎都必须保持转写模式。

### 4.2 语言证据与数据流

在 `TranscriptionSegment` 增加默认值为空的元数据：

```python
language: Optional[str] = None
language_probability: float = 0.0
```

增加统一的语言检测结果结构或等价内部返回值，至少包含 `code`、`probability` 和 `source`。保持现有 `detect_language()` 兼容，新增信息接口供 Pipeline 使用。

- faster-whisper 从 `info.language` 和 `info.language_probability` 填充证据。
- whisper.cpp 通过当前 CLI 的语言检测选项实现全局检测；显式语言时直接传 `-l`，不做无依据的片段自动切换。
- FunASR 使用模型声明的语言范围；默认 Paraformer 配置保持中文场景约束，非支持语言记录清晰警告。
- Pipeline 将主语言传给每个 VAD 片段、边界重识别窗口和缓存键。
- 混合模式的自动重识别结果必须携带检测语言证据；没有证据时拒绝切换。

### 4.3 缓存隔离

片段和边界窗口缓存键加入以下影响结果的参数：

- 引擎、模型、语言、`language_mode`；
- `beam_size`、`word_timestamps`、`condition_on_previous_text`、`vad_filter`；
- 混合语言阈值版本或等价策略版本。

完整管道缓存继续使用配置和核心源码指纹。新参数应自动使旧的完整管道缓存失效。

### 4.4 安全文本规范化

Pipeline 默认使用安全模式：

- 清理空白、不可见字符和明显格式噪声；
- 不默认执行“数字词恢复为编号”；
- 不默认执行内置人名/品牌词替换；
- 原始 ASR 文本保留，任何启用的词典纠错都必须是显式配置。

为兼容已有 `TextNormalizer` 直接调用，类级 API 可保留旧行为，但 Pipeline 不再无条件启用语义性纠错；增加安全模式测试，防止正确文本被默认词典改写。

### 4.5 LLM 字幕优化安全门

LLM 仍可基于上下文修正错字，但必须通过逐条安全校验：

1. 输出必须包含完整原始键集合，且每个值为字符串。
2. 禁止翻译或语言脚本突变；输出主要文字脚本应与输入一致。
3. 只允许当前字幕条目内的修改；禁止把相邻条目内容复制、移动、合并或清空。
4. 使用可配置的保守相似度阈值和长度变化上限，允许少量同音/近形字、拼写和标点修正。
5. 保护数字、单位、专名候选和说话人边界；跨说话人内容变化直接回退。
6. 任一条目失败时，该条保留 ASR 原文；批次级 JSON/对齐失败时整批保留原文。

`original_text` 只在实际应用合规修改时记录，前端对比和反馈学习使用同一份原文锚点。

## 5. 接口变化

### 5.1 CLI

`run` 和 `batch` 增加：

```text
--mixed-language    允许片段级语言切换
--single-language   强制单一语言模式
```

两个选项互斥，默认使用配置中的 `single`。已有 `--language` 优先级最高。

### 5.2 Web/API

- `RunRequest.overrides` 支持 `asr.language_mode` 或等价短键。
- 配置摘要和配置字典返回 `language_mode` 与检测阈值。
- ASR 选项区域增加一个默认关闭的“多语言混合”开关，并沿用现有 localStorage 和 overrides 收集机制。

## 6. 错误处理与可观测性

日志至少记录：主语言、概率、模式、每次候选切换的原因、候选概率/置信度收益、最终是否接受，以及缓存命中时的语言策略。

- 全局检测失败且未显式语言：在 `single` 模式下停止当前 ASR 阶段并给出 `--language` 建议，避免静默地产生不可信字幕。
- 混合模式候选不满足门槛：继续使用主语言结果并记录拒绝原因。
- LLM 不可用、超时、结构非法或安全门拒绝：保留 ASR 字幕，不影响主流程完成。
- 任何新增策略异常都应降级到原始结果，而不是修改已有文本或扩大自动切换范围。

## 7. 测试与验收

新增或补充以下测试：

- 配置默认 `language_mode == single`，CLI/API/UI 覆盖值一致。
- 显式语言始终锁定，并传给 faster-whisper、whisper.cpp 和边界重识别。
- 单语言模式拒绝低置信度片段级换语言。
- 混合模式只接受高概率、不同语言且置信度收益足够的候选；低概率、无证据或收益不足时拒绝。
- faster-whisper 的语言概率写入结果；whisper.cpp 命令包含语言参数且不包含翻译参数。
- 缓存键在语言模式、语言和解码参数变化时不同。
- Pipeline 默认安全规范化不改内置词典命中；显式启用时仍可测试词典功能。
- LLM 小范围错字/标点修正可以通过；翻译、跨条目搬运、大幅改写、空结果和非法键集合回退原文。
- 现有全量测试保持通过。

验收以行为为准：默认单语言任务不因短片段自动改变语言；显式混合任务可以保留可信的多语言片段；开启 LLM 时不会因后处理产生新的跨条目文本或整句翻译。

## 8. 实施顺序

1. 配置和识别数据结构：新增字段、加载/校验、语言证据元数据。
2. ASR 引擎与 Pipeline：统一语言检测、模式门控、边界重识别传递和缓存键。
3. CLI、Web API、Web UI：增加混合语言开关并接入现有覆盖机制。
4. 文本规范化和 LLM 安全门：默认安全模式、逐条回退和原文锚定。
5. 回归测试、文档和全量验证。

## 9. 实施结果

本方案已落地。当前实际行为是：

- `language: null`（UI 显示 `AUTO`）时，先对完整人声音频做一次语言检测，
  再将检测结果锁定为本次任务的主语言；默认 `language_mode: single`。
- 显式指定 `language` 时跳过全局检测，直接使用指定语言。
- 只有显式启用 `mixed`/`--mixed-language` 时，低置信度片段才会进行自动检测；
  候选语言必须同时满足语言概率、语言差异和识别收益门槛。
- 全局检测失败不会静默退回短片段自动检测；系统会提示用户指定语言。
- 语言策略、解码参数和混合语言阈值已纳入转录缓存键。

验证结果：全量测试 `459 passed, 8 warnings`；语言策略、Pipeline 和 ASR
相关回归测试通过。警告来自既有依赖弃用提示和空数组数值计算，不是本方案失败。
