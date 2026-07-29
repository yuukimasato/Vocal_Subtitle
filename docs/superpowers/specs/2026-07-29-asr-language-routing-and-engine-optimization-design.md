# ASR 语言路由与引擎质量优化方案

日期：2026-07-29  
状态：设计方案，待用户评审  
关联实测：[字幕识别浏览器三音频三引擎分析报告-2026-07-29.md](../../字幕识别浏览器三音频三引擎分析报告-2026-07-29.md)  
关联审计：[复杂音频ASR代码审计与优化方案-2026-07-29.md](../../复杂音频ASR代码审计与优化方案-2026-07-29.md)

## 1. 目标

将 ASR 默认策略调整为：

1. 先对整段音频做全局语言检测。
2. 只有确认整段音频全程为中文时，自动路由到 FunASR。
3. 英文、其他语言、中英混合、语言不确定或检测失败时，整段路由到 faster-whisper。
4. whisper.cpp 仅在用户明确选择的低资源/手动模式下使用，自动模式不擅自选择 whisper.cpp。
5. FunASR 即使模型加载成功，也必须通过识别质量门禁；失败时自动回退到 faster-whisper。

目标不是让三种引擎在所有场景下都输出同样的结果，而是让每种引擎只处理适合自己的输入，并让错误和降级状态可见、可追踪、可复现。

## 2. 实测依据与问题边界

本轮浏览器实测的 3 个音频 × 3 个引擎结果显示：

| 问题 | 证据 | 设计响应 |
|---|---|---|
| FunASR 对英文产生局部英文短语，任务仍显示完成 | 英文多人 1 条；英国老头 1 条 | 非中文硬门，禁止自动路由到 FunASR |
| FunASR 对中文长音频过度合并，只返回尾部短语 | TTS 双人 157.9s 只输出 1 条 | 中文路由仍可使用，但增加覆盖、密度、异常长事件门禁，失败回退 faster-whisper |
| faster-whisper 英文连续讲解的后半段物理证据缺失 | 英国老头人工参照持续到 53.85s，内部物理末端约 37.6s | 修复尾段 VAD/骨架证据合并和全局 ASR 覆盖审计 |
| whisper.cpp 低资源回退产生碎片或最终化失败 | 两个样本 `display must cover physical end`，英文多人为 `legacy_degraded` | 保留手动入口，增加前置检查、降级质量状态和物理边界契约 |
| 任务状态不足以表达识别质量 | FunASR 3 项均 completed 但不可交付 | 增加 `quality_status` 与质量原因，区分执行完成和结果可用 |

人工修正 ASS 只作为内容、边界和说话人结构的复核参照，不作为唯一字幕真值。严格 CER/WER、边界误差和说话人准确率需要后续建立正式评测集。

## 3. 方案原则

### 3.1 路由优先级

路由优先级从高到低为：

1. 用户显式指定引擎。
2. 用户显式指定语言与自动引擎模式。
3. 全局多窗口语言检测结果。
4. 自动资源策略和引擎可用性检查。

自动策略绝不覆盖用户选择的 `whisper.cpp`，也不会仅凭 CPU/内存状态自动切换到 whisper.cpp。低资源用户通过显式选择进入该路径。引擎缺失、模型缺失或前置检查失败时，必须显示失败原因；只有在自动路由的 FunASR 质量失败场景下，才允许按规则回退到 faster-whisper。

### 3.2 整段路由，不做片段级引擎切换

本阶段不在同一任务内按句子切换 FunASR 和 faster-whisper。原因是两种引擎的时间戳、文本分段和置信度定义不同，片段级混合会增加重复字幕、边界冲突、缓存污染和结果排序风险。

混合语言任务统一使用 faster-whisper。后续如果需要中英混合的片段级路由，应另立设计，先建立词级语言元数据和跨引擎时间轴仲裁协议。

## 4. 架构设计

### 4.1 ASR 自动路由器

新增独立的自动路由边界，建议放在 `vocal_subtitle/asr/` 下，由 `Pipeline` 调用，不把检测和路由条件继续堆入 `_get_asr_engine()`。

路由器输入：

- 完整音频与采样率；
- VAD/ffmpeg 产生的有效语音区间；
- `ASRConfig`；
- 设备、模型缓存和引擎可用性信息。

路由器输出一个不可变决策对象，至少包含：

```text
ASRRouteDecision(
    requested_engine,
    selected_engine,
    selected_model,
    detected_language,
    language_probability,
    window_evidence,
    decision_reason,
    fallback_engine,
)
```

该对象写入 `PipelineStats`、任务结果、日志和缓存键。后续阶段只消费决策对象，不再次自行检测语言或重新猜测引擎。

### 4.2 全局多窗口语言检测

自动模式下，使用轻量 faster-whisper 检测器做语言探测，不直接用 FunASR 判断语言。检测流程：

1. 先使用现有 VAD/ffmpeg 语音证据排除长静音。
2. 从语音区间均匀抽取多个窗口，覆盖开头、中部、尾部；长音频增加窗口数，短音频至少覆盖首尾两个有效窗口。
3. 每个窗口只执行语言检测，不生成正式字幕；默认使用 faster-whisper `tiny` 或已缓存的轻量模型，设备跟随当前 CPU/GPU 配置。
4. 记录每个窗口的语言、概率、有效时长和检测错误。
5. 只有在所有有效窗口均判定为 `zh`，且最低中文概率达到阈值时，才选择 FunASR。
6. 任一窗口检测到 `en` 或其他语言、语言概率低于阈值、窗口数量不足、检测异常或结果冲突时，选择 faster-whisper。

建议初始默认值：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `language_probe_enabled` | true | 自动模式开启全局探测 |
| `language_probe_model` | `tiny` | 只用于探测，不作为正式字幕模型 |
| `language_probe_window_seconds` | 8 | 单个检测窗口的目标时长 |
| `language_probe_max_windows` | 8 | 长音频探测上限，均匀覆盖全程 |
| `zh_min_probability` | 0.85 | 中文窗口最低概率 |
| `zh_required_window_ratio` | 1.0 | 初始阶段要求所有有效窗口为中文 |
| `uncertain_window_policy` | `faster-whisper` | 不确定即走多语引擎 |

这些是可配置的路由参数，不应散落在引擎实现或 WebUI JavaScript 中。

### 4.3 引擎选择与构造

现有 `ASRConfig.engine` 只有具体引擎值，方案将增加 `auto`：

```yaml
asr:
  engine: "auto"
  auto_routing:
    enabled: true
    language_probe_model: "tiny"
    zh_min_probability: 0.85
    zh_required_window_ratio: 1.0
    fallback_on_quality_failure: true
    fallback_engine: "faster-whisper"
```

具体引擎模式的行为：

| 配置 | 行为 |
|---|---|
| `auto` | 语言探测后在 FunASR/faster-whisper 之间选择 |
| `faster-whisper` | 始终使用 faster-whisper，语言只影响识别参数 |
| `funasr` | 始终使用 FunASR；非中文输入给出硬错误或明确警告，不自动吞掉错误 |
| `whisper-cpp` | 始终使用 whisper.cpp；启动前校验二进制和模型 |

`language: zh`、`language: en` 等显式语言锁定在 `engine=auto` 时作为强提示：`zh` 直接选择 FunASR，非中文选择 faster-whisper。用户显式指定具体引擎时，语言与引擎冲突应在任务启动阶段明确提示。

## 5. FunASR 质量与回退设计

### 5.1 FunASR 中文硬门

FunASR 的 `detect_language()` 不能作为非中文保护，因为当前实现本身是中文专用并会返回 `zh`。语言硬门必须在 FunASR 构造和正式推理之前由路由器完成。

自动路由到 FunASR 时，决策对象必须确认：

- 所有有效探测窗口为中文；
- 无非中文冲突窗口；
- FunASR 包和模型已就绪；
- 当前设备和输入采样率满足引擎前置条件。

显式选择 FunASR 时，如果用户将语言锁定为非中文，任务直接失败并给出“FunASR 为中文专用引擎”的可操作提示，不生成中文乱码字幕。

### 5.2 中文分段与物理时间轴

当前 FunASR 能加载模型并输出结果，但在 TTS 双人长音频中将内容压为单条。优化分为三层：

1. 保留 FunASR 原始句级/词级时间戳，禁止用单个整段时间范围覆盖全部文字。
2. 当 FunASR 只返回少量长事件时，使用既有 VAD/ffmpeg 物理语音区间做二次分箱；分箱过程必须保留 `physical_start`、`physical_end` 和源词 ID。
3. 最终化前执行最大时长、异常长事件、重叠和空文本校验；校验失败时不把结果标记为普通完成。

二次分箱不能凭字符数任意切中文文本。优先使用 FunASR 时间戳，其次使用 VAD 的静音边界，最后才使用保守的文本切分规则，并把降级原因写入诊断信息。

### 5.3 FunASR 质量门禁

质量门禁不依赖人工 ASS，使用音频物理证据和引擎自身结果：

- 正式字幕文本非空且通过安全规范化；
- 有效字幕覆盖了主要物理语音区间，末尾不得无理由提前结束；
- 不得存在覆盖长时间但只含极少文字的异常事件；
- 不得存在大量重复事件、重叠事件或无法投影到物理语音区间的文字；
- 字幕最大时长、最小文本密度和长音频单事件阈值可配置；
- 最终化和导出必须成功。

建议将以下信号作为初始门禁指标：

```text
coverage_ratio
text_density
max_event_duration
long_event_count
overlap_ratio
invalid_event_count
```

其中 `text_density` 按中文字符和有效语音时长计算，不使用完整音频时长；阈值应先在三份实测样本和现有 benchmark 上校准。单条字幕不能单独判定失败，只有在“长音频 + 低密度/低文本量/高重叠”组合出现时才触发。

### 5.4 回退策略

自动路由的 FunASR 发生以下情况时，最多回退一次到 faster-whisper：模型准备失败、推理异常、空结果、质量门禁失败、最终化失败、导出失败。

回退时：

- 丢弃 FunASR 的正式事件，保留其原始诊断摘要；
- faster-whisper 使用同一音频、同一物理时间轴和同一语言决策；
- `PipelineStats` 记录 `selected_engine=funasr`、`final_engine=faster-whisper`、`fallback_reason`；
- 禁止 FunASR ↔ faster-whisper 循环回退；
- faster-whisper 也失败时，任务状态为失败并显示两层错误。

这样可以充分利用 FunASR 的中文优势，同时避免当前实测中“FunASR completed 但只有一条字幕”的结果直接交付。

## 6. faster-whisper 路径优化

### 6.1 英文连续讲解的尾段覆盖

针对英国老头样本暴露的尾段问题，检查并调整：

- Silero 与 ffmpeg 语音区间的尾段差异；
- ffmpeg 静音阈值、最小静音时长和骨架合并规则；
- 全局 ASR 输入是否被错误截断为内部物理末端；
- 物理覆盖审计使用的 `last_physical_speech_end` 是否来自完整证据集合；
- 最后一个语音区间是否因边界收缩或声学校验被误删。

当 Silero 和 ffmpeg 在尾部不一致时，不能直接以较短的末端覆盖完整音频。应保留双方证据，并对冲突尾段做一次宽松复检；复检仍不确定时宁可保留待复核区域，也不能静默丢失。

### 6.2 正式输出质量

faster-whisper 仍是非中文和自动回退的默认主引擎，但要增加：

- 语言检测结果和正式识别语言一致性检查；
- 全局 transcript 到物理 bins 的覆盖审计；
- 长音频末尾、长静音后首句和最大字幕时长的专项诊断；
- 对低覆盖结果进行一次受控局部恢复，而不是直接标记完成；
- 保留原始识别段、最终字幕段和物理证据的三者统计。

局部恢复最多执行一次，且必须有明确的 `recovery_category`；恢复失败时保留失败状态和缺口范围。

## 7. whisper.cpp 低资源路径

whisper.cpp 不进入默认自动语言路由，只作为低资源和手动场景：

- WebUI 中保留显式选择；
- CPU/内存不足时由用户主动切换，不由程序仅凭设备类型擅自切换；
- 任务启动前校验 `whisper-cli`、模型路径、模型格式和可执行权限；
- 缺少依赖时直接返回可操作错误，不启动一个必然得到空结果的任务；
- 全局 ASR 失败后如果进入骨架回退，结果必须标记为 `quality_status=degraded`；
- `display must cover physical end` 等时间轴契约错误必须保留为失败/质量失败原因，不得伪装成正常完成。

## 8. 状态、缓存与 WebUI

### 8.1 任务结果状态

在现有 `status=completed|failed` 之外增加结果质量字段，不破坏现有 API：

```text
quality_status: pass | warning | failed | degraded
requested_engine: auto | faster-whisper | funasr | whisper-cpp
selected_engine: faster-whisper | funasr | whisper-cpp
final_engine: faster-whisper | funasr | whisper-cpp
detected_language: zh | en | other | mixed | unknown
language_probability: float
fallback_reason: string | null
quality_diagnostics: object
```

执行成功但质量门禁失败的任务可以保留 `status=completed` 以便下载诊断产物，但 UI 必须显著显示“结果质量失败/降级”，不能只显示“字幕生成完成”。执行异常、没有可用字幕或导出失败使用 `status=failed`。

### 8.2 缓存键

转录缓存和完整管道缓存必须包含：

- 请求引擎和实际引擎；
- 路由版本；
- 探测模型、窗口策略和阈值；
- 检测语言及概率摘要；
- FunASR/faster-whisper 的正式模型；
- 语言模式、VAD/骨架策略和字幕质量门禁版本。

旧缓存不能在缺少这些字段时被误认为自动路由结果。建议增加 `asr_route_version` 和 `quality_gate_version`，变更策略时自然失效旧缓存。

### 8.3 WebUI 交互

- ASR 引擎增加“自动（中文优先）”选项，并保留三个具体引擎选项。
- 选择自动时展示检测模型和路由策略摘要；不把详细解释堆在运行按钮旁。
- 任务进度显示“语言检测”“路由到 FunASR/faster-whisper”“质量校验”“回退”阶段。
- 结果区显示检测语言、选中引擎、最终引擎、质量状态和回退原因。
- FunASR 准备期间禁用开始按钮，并在准备完成后自动恢复，避免用户重复点击。
- 低资源用户可直接选择 whisper.cpp，并在二进制/模型缺失时看到具体路径和修复建议。

## 9. 配置兼容与迁移

兼容现有具体引擎配置：


- 旧配置 `asr.engine=faster-whisper`、`funasr`、`whisper-cpp` 继续表示显式模式；
- 新安装默认配置改为 `asr.engine=auto`；
- 已保存 WebUI 设置若包含具体引擎，不自动改写，避免用户的手动选择被覆盖；
- `language` 显式设置继续优先于探测；
- 旧缓存可读取，但自动路由任务不得复用没有路由版本信息的旧结果；
- 配置加载器对未知路由值、非法阈值和缺少模型字段给出明确校验错误。

## 10. 测试方案

### 10.1 单元测试

新增或扩展以下测试：

1. 所有探测窗口均为高概率 `zh` 时选择 FunASR。
2. 任一窗口为 `en`、其他语言或低概率时选择 faster-whisper。
3. 探测失败、VAD 没有足够有效窗口时选择 faster-whisper。
4. 显式引擎覆盖自动路由；显式 whisper.cpp 不被自动策略替换。
5. 显式 FunASR + 非中文语言在运行前硬失败。
6. FunASR 空结果、低覆盖、低密度、异常长事件和最终化错误触发一次 faster-whisper 回退。
7. 回退不会形成循环，并且统计字段正确记录初始引擎、最终引擎和原因。
8. whisper.cpp 前置检查缺失二进制/模型时任务失败；降级事件带有 `degraded` 标记。
9. 路由版本和语言探测参数改变时缓存键改变。
10. 旧显式配置和新自动配置的配置加载兼容。

### 10.2 浏览器验收矩阵

保留本轮 3 × 3 实测集，并增加自动模式：

| 样本 | 预期自动路由 | 需要验证 |
|---|---|---|
| TTS 中文双人 | FunASR，若质量门禁失败则 faster-whisper | 中文检测、FunASR 分段、回退可见 |
| 英文多人 | faster-whisper | 不调用 FunASR，6/7 角色内容和边界可诊断 |
| 英文单人长讲解 | faster-whisper | 尾段物理覆盖、长字幕和恢复策略 |
| 中英混合新增样本 | faster-whisper | 任一非中文窗口触发整段多语路由 |
| 无语音/静音样本 | 不调用正式 ASR 或明确无语音 | 不生成伪字幕 |

浏览器验收必须记录：任务 ID、检测语言、选中/最终引擎、质量状态、字幕条数、物理覆盖、回退原因和下载结果。人工 ASS 仍只用于人工复核，不作为唯一自动判定依据。

### 10.3 质量通过条件

第一阶段的最低通过条件：

- 英文和其他非中文样本不调用 FunASR；
- 纯中文样本自动优先调用 FunASR；
- FunASR 低质量时能自动回退并显式标注；
- 任何执行异常、空结果或最终化契约错误不显示为无警告的成功；
- faster-whisper 的英文长音频末尾不再因错误物理末端被静默丢弃；
- 同一输入和配置得到可复现的路由及缓存结果。

第二阶段再基于带正式真值的评测集设定 CER/WER、召回率、边界误差和说话人指标，避免用当前人工修正 ASS 的行数直接设定错误阈值。

## 11. 分阶段实施顺序

### 阶段 A：路由基础

- 增加 `engine=auto` 和自动路由配置；
- 提取独立全局语言探测器和路由决策对象；
- 接入多窗口探测、中文硬门和路由统计；
- 更新配置加载、缓存键和单元测试。

### 阶段 B：FunASR 质量闭环

- 修复 FunASR 时间戳/物理分箱和长事件处理；
- 实现质量门禁、一次性 faster-whisper 回退和诊断传播；
- 增加中文双人/多人专项测试和浏览器验证。

### 阶段 C：faster-whisper 物理覆盖

- 修复英文长音频尾段 VAD/骨架证据问题；
- 增加尾段复检和局部恢复；
- 验证全局 transcript、物理 bins 和最终字幕的覆盖一致性。

### 阶段 D：whisper.cpp 与 WebUI

- 完善 whisper.cpp 资源前置检查和降级状态；
- 增加自动模式与具体引擎模式的 UI 展示；
- 显示检测、路由、质量和回退阶段；
- 用低资源 CPU 样本做手动 whisper.cpp 回归。

### 阶段 E：全量验收

- 执行自动路由矩阵和显式引擎矩阵；
- 对比人工参照并记录非严格质量指标；
- 更新质量 manifest、浏览器实测报告和默认配置迁移说明；
- 只有满足执行状态、质量状态和缓存一致性条件后，才将 `auto` 设为默认。

## 12. 非目标

- 本方案不在同一音频内实现中英片段级跨引擎混合。
- 本方案不把 FunASR 作为英文或其他语言的通用引擎。
- 本方案不自动根据 CPU 类型强制切换 whisper.cpp；低资源模式需要用户明确选择。
- 本方案不把人工修正 ASS 自动导入为字幕内容、唯一真值或训练数据。
- 本方案不在本阶段同时重构 VAD、说话人分离和字幕编辑器的全部实现，只修复影响 ASR 路由和质量判断的接口。

## 13. 设计验收

该设计在实现前需要确认以下决策：

- 自动模式是否作为新安装默认值；
- 语言探测轻量模型是否允许首次自动下载；
- FunASR 质量失败时是否默认自动回退 faster-whisper；
- `quality_status=failed/degraded` 是否在 WebUI 中阻止直接导出，还是允许用户手动下载诊断结果；
- 语言检测和质量门禁阈值是否接受配置化并通过样本校准。
