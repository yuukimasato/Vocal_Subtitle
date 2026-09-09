# 字幕编辑器 Agent 化开发文档（汇总版）

- 日期：2026-09-09
- 来源：本文汇总并取代同目录下两份分析稿
  - 《字幕编辑器Agent化方案-CLI与Harness-2026-09-09.md》（原始方案）
  - 《字幕编辑器Agent化-独立分析-2026-09-09.md》（独立分析 v2）
- 冲突裁决原则：凡两稿分歧处，**以独立分析 v2 为准**（独立开源硬约束下修正后的结论），原始方案中仍然成立的部分收编为正文。
- 状态：开发依据文档。配套实施计划见《实施计划-Agent化落地-2026-09-09.md》。

---

## 1. 定位与硬约束

### 1.1 产品定位

subtitle-editor 在全链路中的位置是 **ASR 的功能补充组件**：ASR 引擎产出文本与粗时间 → 本工具做时间轴精修（打轴）。两种被使用方式并存：

- **被人使用**：浏览器打开、手动打轴、导出（现有形态，不改变）；
- **被程序 / Agent 使用**（本方案目标）：上层流水线或 Agent 直接调用其波形分析与字幕变换能力，自动完成粗打轴，人只做终审。

### 1.2 硬约束：独立开源组件

**本工具作为独立开源组件发布，不得强绑定 Vocal_Subtitle。** 这是一条后来追加的硬约束，它反转了原始方案中"Node CLI 子进程调用上层 Python VAD"的复用建议——那会让开源组件的核心能力依赖一个未开源的私有项目，不可接受。

实测结论（独立分析 §1.1）：**代码层面已是零耦合**——所有 import 均为工具目录内相对路径（含 `../../vendor/*`），无任何父仓库引用，无运行时网络。耦合只存在于**命名（4 处）与文档**。因此独立发布的成本是命名清理 + 文档归属划分 + 发布机制，而不是解耦重构。

一句话概括：**组件自治，契约互操作**。编辑器自己拥有一切能独立工作的能力（字幕变换、峰值、边界候选、结构校验）；上层只以"实现了某个契约的可选外部提供者"身份接入，且契约文本由编辑器仓库拥有并版本化。

### 1.3 零依赖原则的边界

- 项目卖点：**零构建、零 npm 依赖、纯本地**（不联网、不上传文件）；
- CLI 与 harness 一律用 **Node 标准库**实现（`node:test` 已开先例；Node v24 全局 `WebSocket` 可裸连 CDP，无需 playwright）；
- `ffmpeg` 是"检测到即用"的可选外部能力，缺失时优雅降级并打印获取指引，绝不静默降质；
- MCP 包装（若做）同样仅标准库（stdio JSON-RPC）。

---

## 2. 现状盘点（含实证）

环境实测基线：Node v24.20.0、Chrome 152、ffmpeg/ffprobe 可用、`node --test` 126 项全绿。

### 2.1 可直接复用的资产

| 资产 | 位置 | 无头可用性（实测） |
|---|---|---|
| 命令层（命名命令 + 撤销 + coalesce） | `js/actions.js` `createActions()` | ✅ 零 shim 直接 import |
| 字幕解析/序列化（SRT/VTT/ASS 双向） | `js/format/index.js` 等 | ✅ |
| Aegisub 对话定时语义（pending→commit） | `js/audio/timing.js`（纯逻辑无 DOM） | ✅ |
| 中央状态 store | `js/state.js` | ✅ |
| 峰值采集数学（跨块均值降采样） | `js/audio/capture.js` `createPeakCollector` | ✅（纯数据部分） |
| 调试句柄 | `js/main.js:179` `window.__editor` | 浏览器内 |
| 自动化加载口 | `?media=<url>&subs=<url>`（与手动打开同路径） | 浏览器内 |
| 测试物料 | `.fixture/`（test.wav / test-video.mp4 / test.ass）、`.smoke/` | 可用作夹具 |

无头全链路实测已跑通：`parseSubtitle` → `loadSubtitle` → `updateCueTimes` → `nudge` → `splitCue` → `undo` → `redo` → `serializeSubtitle` 全部正确。**数据面 CLI 不需要新建工程，它已经存在了大约一半。**

两个已知的签名坑：`parseSubtitle(text, {filename, format})` 收选项对象；`serializeSubtitle(format, cues, doc)` 格式在前。

### 2.2 缺口

1. 波形解码绑死浏览器：主路 `decodeAudioData`、兜底 `HTMLMediaElement` 加速采集，Node 中皆无 → headless 需要另一条解码路径（ffmpeg / WAV 直读）；
2. `__editor` 是调试口不是契约：无文档、无稳定性承诺、字段随 store 内部形态漂移；
3. 波形数值无导出口：峰值只存在于 wavesurfer 实例与渲染循环里，页面无 `getPeaks(t0, t1)` 类读数 API；
4. 无机器可读校验：重叠、间隙、最短时长等检查未成文，无 JSON 报告与退出码；
5. 本地文件加载依赖用户手势：harness 须走 fetch/URL 路径（autoload 已具备，需固化）；
6. **（独立分析补）** 现有 fixture 无静音结构（`silencedetect -35dB/0.4s` 对 `test.wav`、`mono.wav` 均零输出）→ `vad` 与 `check --media` 无法用现有素材验收，必须新增带已知静音间隙的 fixture。

### 2.3 实证技术可行性

- **零依赖 CDP harness**：Node 24 全局 `WebSocket` 完成 `/json/version` → `/json/list` → `Runtime.evaluate` 往返 → `Page.captureScreenshot` 取回 PNG base64，全程无需 playwright；
- **`createPeakCollector` 是均值降采样（低通）**：设计目标是"降采样后波形形状不失真"，刻意抹平起止瞬态——**画波形对，定边界错**。4kHz 的 0.25ms 是桶宽不是边界分辨率。但同时它是**跨解码路径的归一化锚点**（浏览器采集兜底、ffmpeg、WAV 直读三条路都经它归一到 4kHz），是"Agent 看到的数值与人在屏幕上看到的波形同源"的技术基础，必须显式文档化。

---

## 3. 核心决策

### 3.1 总形态：三层薄接入，不是无头编辑器

1. **AGENTS.md（环境理解层）**——Agent 一进目录就知道项目是什么、怎么跑、哪些模块能动；
2. **页面 Agent API + CDP Harness（控制面）**——把 `window.__editor` 冻结为版本化契约 `window.agent`，配一个零依赖 Node 脚本经 Chrome DevTools Protocol 驱动**真实页面**，Agent 在真实 UI 中持续工作、人类同屏监督；
3. **Headless 数据面 CLI（`cli.mjs`）**——字幕读写变换、峰值提取、VAD 候选段、时间轴校验，全部 JSON in/out，作为可组合组件。

**范围守则：CLI 只做数据变换，不做无头编辑器。** 定时"决策权"始终在 Agent + 人，工具只提供观测与验证；UI 级操作一律走 harness / 真实页面。

### 3.2 "视觉模型看波形打轴"的正确定位

**测量用数据，验证用视觉。**

| 用途 | 正确工具 | 精度/成本 |
|---|---|---|
| 与人在屏幕上看到的波形一致（复核、截图、视觉模型输入） | `peaks`（4kHz） | 采样分辨率 0.25ms/桶；文本 token |
| 定位边界、生成候选段、判定"是否在波谷" | VAD（Tier 3/4）的语音/静音区间 | 确定性，无模型推理 |
| 与 ASR 文本对齐、最终裁决 | Agent（文本模型） | — |
| 替代人眼复核（判断题："边界是否落在波谷内？"） | 视觉模型 + 带标定 PNG | ±30–100ms，只答判断题不答定位题 |

两条实现约束：

1. 视觉模型只回答**判断题**（"这条边界是否落在波谷内？"），不回答**定位题**（"边界应该在哪？"）——判断题对 10ms/px 的像素误差不敏感，定位题敏感；
2. `renderWaveform({t0,t1,pxPerSec})` 的标定元数据必须**烧进 PNG 本身**（角落水印 `t0=… pxPerSec=…`），否则 Agent 传图给视觉模型时标定丢失，"读刻度"退化成"猜"。

### 3.3 能力分层模型（独立性的技术证明）

开源组件的最低可用性底线：**Tier 0–3 在只有 Node 标准库时全部可用**。更重的能力一律"检测到即用、缺失即优雅降级"。

| Tier | 能力 | 依赖 | 缺失时行为 |
|---|---|---|---|
| 0 | 字幕读写 / 变换 / 结构校验 | 无（Node 标准库） | — |
| 1 | 波形数据：WAV 直读 → 4kHz 峰值流 | 无（纯 JS WAV 解析） | 非 WAV 报错退出（码 2），提示装 ffmpeg |
| 2 | 任意媒体解码 → 16kHz mono PCM → 4kHz | 可选 ffmpeg（探测到即用） | 退回 Tier 1 |
| 3 | **内置确定性 VAD**（能量 + 滞回） | 无 | 无音频输入时只做结构校验 |
| 4 | 高质量 VAD（Silero 等） | 用户自备 provider 命令 | 退回 Tier 3 |
| 5 | 真实页面驱动 / 截图复核 | Chrome + CDP | 不可用（不影响 0–4） |

三条分层原则：

- **Tier 3 必须存在且够用**——否则组件脱离上层就只剩"改字幕文件"，"ASR 功能补充"的定位落空；Tier 3 的存在本身就是独立性的技术证明；
- **Tier 4 是集成，不是依赖**——上层 VAD 只是"恰好实现了我们契约的一个 provider"，换任何人实现都一样，缺失时组件照常工作；
- **所有解码路径经同一个 `createPeakCollector` 归一到 4kHz**——Agent 从 CLI 拿到的数值与人在浏览器波形上看到的是同一套数据（测量一致性锚点）。

### 3.4 CLI、Harness、MCP 三者的关系

| 形态 | 覆盖场景 | 优势 | 弱点 |
|---|---|---|---|
| A. Headless CLI（数据面） | 流水线组件：批量变换、校验、自闭环 | 确定性、可组合（JSON in/out、退出码）、快、无浏览器 | 无浏览器解码与采集兜底 |
| B. CDP Harness（控制面） | Agent 在真实页面持续工作、人在回路 | 复用页面全部能力（decodeAudioData、加速采集兜底、ASS 预览、撤销/草稿）；人机同屏 | 需起 Chrome；比进程内 CLI 慢 |
| C. MCP Server | ZCode/claude code 原生工具面 | 即插即用、工具描述自解释 | 只是 A/B 的薄包装，不产生新能力 |

**A + B 先行，C 后置可选。** MCP 后置的理由（独立分析分歧 F）：harness 已给 Agent 全部能力，MCP 的增量仅是省掉读文档成本；在 `window.agent` 契约冻结前做 MCP，等于把未冻结的契约再包一层，返工翻倍。

---

## 4. 方案设计

### 4.1 第 0 步成果：`check` 的语义边界（决定）

**结构检查归编辑器，声学判定可委托 provider，两者词汇不混用。**

- 编辑器 `check` **拥有**的词汇（面向打轴问题，本仓库定义并冻结）：
  `overlaps`（行间重叠）/ `gaps`（可疑间隙）/ `tooShort`（低于 `MIN_LEN`）/ `endBeforeStart` / `sorted`（未排序）/ `boundaryInSilence`（边界落入 VAD 静音区，`--media` 时）/ `startAfterSpeech` 类边界-语音关系检查。
- **不得**抄用上层的报告字段名（`snapped_starts` / `rms_overrides` / `skipped_high_confidence` 等）——上层字段描述"吸附算法做了什么"，编辑器字段描述"字幕与音频的偏差是什么"，语义不同，硬套会让 Agent 误解。映射表放集成文档，不放编辑器。
- 声学判定的"最终质量裁决"（如某边界离语音起点该留多少余量）不属于编辑器，属于 provider / 流水线侧；编辑器只提供"边界落在静音区"这类**事实性**声学观察。
- 已知冲突显式化：编辑器 `MIN_LEN = 0.05`（50ms）与上层 `min_fragment_duration = 0.15`（150ms）不是同一个量，`check` 只对前者负责。
- 退出码语义稳定（0/2/3），可被发布检查脚本或任何未来的 CI 消费（**注意：目前没有 CI**，不写"进 CI"）。

### 4.2 AGENTS.md（环境理解层）

放置于 `tools/subtitle-editor/AGENTS.md`，内容清单：

- 一句话定位 + 与上层流水线的关系（上层是消费者之一，不是归属）；
- 目录地图与分层：`state.js`（store）→ `actions.js`（命名命令+撤销）→ `ui/*`（只做显示与输入）；纯逻辑在 `audio/timing.js`、`format/*`、`capture.js`（数学部分）；
- 命令：`python3 -m http.server 8631`、`node --test`、`node build-standalone.mjs`、`node agent/cli.mjs`；
- 能力分层表（§3.3）与 Tier 降级行为说明；
- 页面 Agent API 速查表与 `?media=&subs=` 用法；
- 数据不变量：cues 恒排序、`MIN_LEN` 最短时长、定时改动走 pending→commit 不直写、写操作只经 actions；
- **禁令：禁止 import 上层项目任何模块路径**（独立开源约束）；
- 扩展守则：新逻辑优先纯函数进 `js/` 可单测；UI 进 `js/ui/`；改动跑 `node --test`。

### 4.3 Headless CLI（`agent/cli.mjs`）

全部子命令 JSON 输出（`--format text` 提供人读模式）；退出码：**0 正常 / 2 输入错误 / 3 校验发现问题**。

| 命令 | 功能 | Tier |
|---|---|---|
| `info <subs>` | 字幕格式、行数、时长分布；`info <media>`（ffmpeg/WAV 探测） | 0 / 1–2 |
| `peaks <media> [--rate 4000]` | 能量包络 JSON（ffmpeg→PCM→capture 数学；WAV 直读兜底） | 1–2 |
| `vad <media> [--provider "<cmd>"]` | 语音段候选：默认内置 VAD；`--provider` 走外部契约，失败自动降级并记录 | 3–4 |
| `cues <subs> get/set/shift/split/merge/renumber` | 字幕读写变换，复用 `format/*`，行为与 `actions.js` 一致 | 0 |
| `convert <in> --to srt\|vtt\|ass` | 格式互转 | 0 |
| `check <subs> [--media <m>]` | 结构检查恒有；`--media` 加边界-静音关系检查；JSON 报告 + 退出码 | 0 / 3–4 |

实现分层：纯逻辑进 `agent/*.js` 模块（`cueOps` / `wav` / `ffmpeg` / `peaks` / `vad` / `vadProvider`），`cli.mjs` 只做 argv 解析与输出格式化，全部可被 `node --test` 直测。

流水线组合示例：

```
node agent/cli.mjs vad media.mkv > seg.json
# Agent：seg.json × ASR 草稿 SRT 对齐 → 生成逐行 set 计划
node agent/cli.mjs cues draft.srt set --plan plan.json -o timed.srt
node agent/cli.mjs check timed.srt --media media.mkv   # 退出码 0 才放行
```

### 4.4 页面 Agent API：`window.__editor` → `window.agent` 契约

不新造机制，把现有句柄冻结为版本化契约：`window.agent = { version: 1, ... }`

- **只读**：`getSnapshot()`（媒体时长、cues 全量、选中态、dirty/撤销栈深度、波形就绪）；`getPeaks(t0, t1)`（数值数组 + 采样率）；`renderWaveform({t0, t1, pxPerSec})` → dataURL + 标定元数据烧进 PNG；
- **加载**：`loadMedia(url)` / `loadSubs(url)`——固化现有 autoload 的 fetch 路径；
- **写**：`act(name, args, {coalesceKey})` 转发 `actions` 命名命令——**不绕过 store 直改**，自动获得撤销/草稿/ASS 预览联动。

**契约冻结为 baseline 文件**（沿用仓库既有惯例 `scripts/check_api_contract.py` + baseline JSON 的思路）：`window.agent` 的字段路径清单冻结为 `.smoke/agent-contract.baseline.json`，冒烟脚本对真实页面求值后比对，字段增删即红。

### 4.5 写路径的两个风险与对策（人机同屏的前提）

1. **撤销步爆炸**：`actions.commit()` 每次调用推一个快照，Agent 批量精修 200 行 = 200 个撤销步，人类终审时 Ctrl+Z 形同虚设。对策：`act(name, args, {coalesceKey})` 透传 coalesce 语义，Agent 声明一段工作共用同一 key（`commit()` 已支持，只需在契约中暴露）。
2. **草稿互相覆盖**：`draft.js` 按 `subtitleName || mediaName` 生成 localStorage key，800ms 防抖写入；Agent 与人类同源同 key，Agent 工作 800ms 后人类未导出的草稿即被覆盖。对策：agent 会话使用**独立草稿命名空间**（`?agent=1` 时 key 加后缀）且 agent 模式**默认不写草稿**、只在显式 save 时写。这是人机同屏最容易出的数据事故，必须先于 harness 解决。

### 4.6 Harness（`agent/harness.mjs`，Node ≥ 22 标准库）

**写成通用 CDP 客户端，不绑定 `window.agent`**——只提供 `open / attach / eval / shot / close` 原语，`window.agent` 只是被 `eval` 的目标表达式。这样同一份 harness 能驱动本编辑器与上层 WebUI 两边，也更符合独立发布（harness 是通用工具，不含产品语义）。

```
harness open  --url URL [--media X --subs Y] [--attach]  # 起 Chrome 或连接已开的浏览器
harness eval  "<expression>"              # Runtime.evaluate → stdout JSON
harness shot  --out p.png                 # Page.captureScreenshot
harness close                             # 关闭受管浏览器
```

安全边界：仅监听/连接 127.0.0.1；页面本身纯本地无回传，agent API 不新增网络面。

**人监督模式**（`--attach` 连人已开的浏览器）是"真实场景持续工作"的关键形态：浏览器开着、人随时可看，Agent 每步改动在撤销栈里，整段工作可一键回滚。

### 4.7 内置 VAD（Tier 3）规格

定位：**确定性、可单测、精度够用的边界候选器**，不是重造 Silero。

- 输入：4kHz 单声道流（16kHz 内部同法处理）；
- 短时能量：窗 20ms / 跳 10ms（与打轴领域常用 `grid_resolution=0.01` 收敛一致）；
- 噪声底：帧能量取 20 分位；
- 判决：阈值 = 噪声底 × 3；滞回：能量 > hi 进入语音、< hi/2 退出语音；`min_speech=150ms`、`min_silence=400ms`；
- 输出：`[{start, end, confidence}]`，confidence 由超阈裕度映射；附 `engine: "energy-vad"`；
- 精度声明：干净素材 ±10–30ms；音乐/强噪声素材不保证；
- **定位声明（写进 `--help` 与 JSON）**：这是打轴辅助的边界候选器，不是 ASR 分段器。

约 120 行，可完全单测（合成 fixture 断言误差 ≤ 50ms）。

### 4.8 外部 VAD provider 契约（Tier 4）

契约由**编辑器仓库拥有并版本化**（文档：`docs/vad-provider-contract.md`），实现方（上层 Python、用户脚本、任何语言）只需满足它：

```
# 调用：cli.mjs vad media.mkv --provider "<cmd>"
# 请求（stdin，单行 JSON）
{"schema_version":"vad-provider-request-1","audio_path":"...","sample_rate":16000,
 "threshold":0.5,"min_speech_ms":150,"min_silence_ms":400}
# 响应（stdout，单行 JSON）
{"schema_version":"vad-provider-response-1","engine":"silero",
 "segments":[{"start":1.02,"end":3.00,"confidence":0.91}]}
# 退出码：0 成功；非 0 或非法 JSON 视为 provider 失败 → 自动退回内置 VAD 并在 JSON 中记录降级
```

上层项目只需在**它自己**的仓库里放一个约 20 行的适配脚本（把 `FFmpegSilenceVAD`/`SileroVAD` 的输出映射到上面的响应格式）即可接入。**适配脚本属于集成工作，不属于组件本身**，放上层仓库。

### 4.9 MCP 包装（后置可选）

`agent/mcp-server.mjs`（stdio JSON-RPC，标准库实现）把 CLI/harness 命令暴露为 MCP tools。只是薄包装，优先级最低，在契约冻结之后做。

### 4.10 典型端到端工作流（Agent 打轴 + 人终审）

1. Agent 读 AGENTS.md → 起静态服务 + `harness open --media media.mkv --subs asr-draft.srt`；
2. CLI `peaks` / `vad` 取数值候选段；
3. Agent 将候选段与 ASR 文本逐句对齐，`harness eval` 调 `window.agent.act('setCueTimes', …)` 批量精修（coalesce 合并撤销步）；
4. `cli check` 全绿 → `harness shot` 存档证据；
5. 人打开同一页面复核，疑难行用波形 PNG + 视觉模型辅助**判断**；
6. 满意后导出 SRT/ASS，进入下游。

---

## 5. 常量与字段：所有权与可选映射

### 5.1 编辑器自有的不变量（本仓库定义并冻结）

| 语义 | 取值 | 位置 |
|---|---|---|
| 最短行时长 | `MIN_LEN = 0.05` | `js/actions.js:8` |
| 插入空白行时长 | `BLANK_LEN = 5` | `js/actions.js:9` |
| 采集目标率 | `TARGET_RATE = 4000` | `js/audio/capture.js:13` |
| 草稿 key 前缀 | `vstEditor.draft.` + `subtitleName \|\| mediaName` | `js/draft.js:2` |
| 定时模型 | pending → commit（切行丢弃未提交） | `js/audio/timing.js` |
| 前置/延后 | `LEAD_IN_MS = 200` / `LEAD_OUT_MS = 300` | `js/audio/timing.js:26-27` |
| 微调步长 | `NUDGE_MS = 10` | `js/audio/timing.js:28` |
| cue 排序 | 恒按 `start` 稳定排序（同 start 按 `end`） | `js/format/cue.js` |
| 内置 VAD 参数 | 窗 20ms / 跳 10ms、噪声底 20 分位、阈值 ×3、滞回 hi/2、min_speech 150ms、min_silence 400ms | 待实现（`agent/vad.js`） |

### 5.2 与上层的可选映射（引用，不依赖）

下列是上层 Vocal_Subtitle 的取值，**只作为集成文档里的对照**，不得成为编辑器代码的常量来源：硬静音阈值 `0.4s`、吸附上限 `0.5/0.25`、吸附余量 `0.03/0.01`、高置信跳过 `0.6`、网格 `0.01`、合并 padding `0.10`、`min_silence_gap=0.4`、`max_segment_length=20.0`、`min_fragment_duration=0.15` 等。

**重要：不把上层的报告字段名（`snapped_*` / `rms_overrides` 等）抄进编辑器的 `check` 输出**（理由见 §4.1）。

---

## 6. 独立发布的结构性要求

### 6.1 命名去耦合（4 处，机械改动）

| 位置 | 现状 | 改为 |
|---|---|---|
| `package.json` name | `vocal-subtitle-timing-editor` | `subtitle-timing-editor` |
| `package.json` description | 含 "Vocal Subtitle" | "零构建、零依赖的本地字幕打轴编辑器与 Agent 工具链" |
| `index.html:6,19,170` | 标题/品牌/页脚含 Vocal Subtitle | 中性品牌；页脚可保留"可与任意 ASR 流水线配合" |
| `README.md:1` | 标题含产品名 | 中性标题；新增「与 ASR 流水线集成」一节，把 Vocal_Subtitle 列为**消费者之一** |

保留一处对 Vocal_Subtitle 的**正向引用**（"被用于…"）是合理的生态说明，不是绑定。

### 6.2 防回归：把"不绑定"变成可执行断言

`test/` 下新增单测（随 `node --test` 跑），扫描源码与页面中的禁止字符串：

- 禁止：`vocal_subtitle`、`Vocal_Subtitle`、`Vocal Subtitle`（品牌）、`/home/` 绝对路径、任何越出工具目录的 `../` import；
- 允许：README/docs/LICENSE 中作为"集成对象"的说明性提及（白名单文件列表控制）。

### 6.3 文档与契约的归属划分

| 内容 | 归属仓库 |
|---|---|
| Agent API 契约（`window.agent`）、CLI JSON schema、VAD provider 契约、编辑器不变量 | **编辑器仓库（本仓库）** |
| 集成文档、字段映射表、provider 适配脚本、"如何从流水线调用编辑器" | **上层仓库** |

### 6.4 发布机制（三方案，按推荐排序）

1. **独立仓库 + 目录级同步脚本**（推荐）：新建 `subtitle-editor` 仓库，父仓库保留 `tools/subtitle-editor` 作为开发位，`git subtree split` 或 rsync 单向同步；
2. **直接迁出**：编辑器完全移出父仓库，父仓库经 release 产物或 submodule 引用；
3. **维持现状，只发布单文件 HTML**：成本最低，但源码不独立，"开源组件"叙事不完整。

无论哪种，**命名清理必须先做**（§6.1），否则同步出去的仓库自带父项目名。README 已指向 `github.com/yuukimasato/subtitle-editor/releases`，独立仓库尚未建立——发布路径待决策，不阻塞本开发计划。

---

## 7. 验收标准（可执行）

### 7.1 VAD/声学 fixture 规格（替代对现有 `.fixture/` 的依赖）

现有 `test.wav`（53.9s）与 `mono.wav`（30s）实测**无任何静音结构**，不能用于 VAD 验收。新 fixture：

- 16kHz 单声道 PCM WAV，总长约 20s，测试内**程序化合成**（不提交二进制）；
- 静音段 ≥ 0.5s（高于 `min_silence=400ms`）；
- 结构如 `[静音 0.0–1.0][音 1.0–3.0][静音 3.0–4.0][音 4.0–6.5][静音 6.5–8.0]…`，**每段起止以常量写在测试里**；
- 断言：内置 VAD 输出与常量表误差 ≤ 50ms；`check --media` 能把人为移入静音区的边界标记为问题。

### 7.2 逐阶段验收

| 阶段 | 验收 |
|---|---|
| 第 0 步 | 语义边界决定成文（§4.1）+ 命名清理完成 + 防回归单测通过 |
| 第 1 步 | 新会话 Agent 不读源码即可：起服务、跑测试、用 `?media=&subs=` 打开 fixture、复述能力分层与降级行为 |
| 第 2 步 | 每个子命令 JSON 输出 + `--format text`；`check` 退出码 0/2/3 有单测；`cues` 与 `actions.js` 行为一致性有单测 |
| 第 3 步 | baseline JSON 冻结 `window.agent` 字段；真实页面冒烟通过；Agent 连续 50 次写操作后撤销栈深度可控（coalesce 生效）；Agent 会话不污染人类草稿 key |
| 第 4 步 | 内置 VAD 在合成 fixture 上误差 ≤ 50ms；无 ffmpeg 时 WAV 直读可用、非 WAV 报错码 2；provider 返回非法 JSON 时自动降级且记录原因 |
| 第 5 步 | MCP server 在 ZCode 挂载后可直接调用工具集 |

### 7.3 风险登记

| 风险 | 对策 |
|---|---|
| 契约漂移 | `window.agent` 版本字段 + baseline JSON 冒烟比对 |
| 解码差异（浏览器 vs ffmpeg 边界样本） | 预期 < 1 峰值窗，文档化容忍度；必要时以 harness 路径为准 |
| 两套编辑器、两条自动化路径 | harness 写成通用 CDP 客户端，不绑定产品语义 |
| 撤销步爆炸 / 草稿覆盖 | coalesceKey 透传 + agent 草稿命名空间隔离（先于 harness） |
| 范围蔓延 | CLI 不做无头编辑器；定时决策权在 Agent + 人 |

---

## 8. 落地顺序（两稿合并后的最终序）

| 顺序 | 内容 | 理由 |
|---|---|---|
| 第 0 步 | 定 `check` 语义边界 + 命名去耦合 + 防回归断言 | 先定"什么归编辑器、什么归 provider"；命名越早清理，同步出去的仓库越干净 |
| 第 1 步 | AGENTS.md | 零风险，立刻提升一切后续工作的可理解性 |
| 第 2 步 | CLI 数据面 `info`/`cues`/`convert`/`check`（结构性） | 实测几乎免费，零外部依赖，立刻可单测；Agent 立刻有确定性工具 |
| 第 3 步 | `window.agent` 契约 + 通用 CDP harness | 前置：草稿隔离与 coalesce 透传，否则人机同屏出数据事故 |
| 第 4 步 | Tier 1–3：WAV 直读 + ffmpeg 解码 + 内置 VAD + `peaks`/`vad`/`check --media`；Tier 4 provider 契约 | 依赖合成 fixture；provider 契约先文档后实现 |
| 第 5 步 | MCP 包装（可选） | 最后做；契约冻结后才是低成本封装 |

**为什么 CLI 先于 harness**（对原始方案 P1/P2 的对调）：harness 的价值依赖"人机同屏"，而人机同屏的前提是草稿隔离与事务语义就位；CLI 数据面不依赖任何这些，是纯增量、零风险、可立即验收的部分。先做确定的，把有语义风险的放后面想清楚。
