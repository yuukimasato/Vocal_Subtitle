# 字幕编辑器 Agent 化方案：CLI 与 Harness 接入分析

- 日期：2026-09-09
- 范围：`tools/subtitle-editor`（字幕打轴工作台）及其在 Vocal_Subtitle 全链路中的组件化定位
- 状态：分析稿，未实施

## 0. 结论（TL;DR）

**需要做，但形态不是"为 Agent 复刻一套无头编辑器"，而是三层薄接入：**

1. **AGENTS.md（环境理解层）** —— 让 codex/ZCode/claude code 一进目录就知道项目是什么、怎么跑、哪些模块能动；
2. **页面 Agent API + CDP Harness（控制面）** —— 把已有的 `window.__editor` 调试句柄冻结为文档化契约，配一个零依赖 Node 脚本通过 Chrome DevTools Protocol 驱动**真实页面**，Agent 在真实 UI 中持续工作、人类同屏监督；
3. **Headless 数据面 CLI（`cli.mjs`）** —— 波形峰值提取、VAD 候选段、字幕读写变换、时间轴校验，全部 JSON in/out，作为上层 ASR 流水线（`vocal_subtitle`）的可组合组件。

**关于"让视觉模型识别波形波峰波谷打轴"：方向对，形态要修正。** 波形数据本来就是数字（Float32Array 采样 / `capture.js` 已产出 4kHz 峰值流），让 Agent 直接读 JSON 数值的精度比视觉模型看 PNG 反推像素→时间映射高一个数量级，成本也低得多。**测量用数据，验证用视觉**：视觉模型的正确位置是截图复核（波形叠加、边界合理性），测量主路径应该是数值峰值 + 确定性 VAD。方案同时保留"带标定元数据的时间轴 PNG 渲染"作为视觉模型的兜底输入。

## 1. 背景与定位

subtitle-editor 在全链路中的位置是 **ASR 的功能补充组件**：ASR 引擎产出文本与粗时间 → 本工具做时间轴精修（打轴）。"组件"意味着两种被使用方式并存：

- **被人使用**：当前形态——浏览器打开、手动打轴、导出；
- **被程序/Agent 使用**（本方案目标）：上层流水线或 Agent 直接调用其波形分析与字幕变换能力，自动完成粗打轴，人只做终审。

Agent（codex / ZCode / claude code）要在一个项目里"理解环境、使用工具、持续工作"，需要四类能力，逐项对照现状：

| 能力 | 含义 | 现状 |
|---|---|---|
| 理解环境 | 目录结构、构建/测试命令、架构分层、改动守则 | ❌ 无 AGENTS.md，每次会话需重新推断 |
| 观测 | 读媒体元数据、波形数据、当前字幕文档、选中态 | 半有：`__editor` 控制台句柄存在但无契约；波形数值无导出口 |
| 行动 | 改时间、拆/合行、平移、插入、保存导出 | 半有：`actions.js` 已是命名命令层，但只被 UI 与快捷键调用 |
| 验证 | 时间轴合法性检查、与波形对齐度、UI 截图复核 | ❌ 无机器可读的 check 输出；测试只覆盖解析与定时逻辑 |

## 2. 现状盘点

### 2.1 可直接复用的资产

架构在 `refactor/componentize-large-files` 分支已完成组件化，分层对 Agent 接入非常友好：

- **命令层已存在**：`js/actions.js` 的 `createActions()` 返回一组命名命令（`updateCueTimes` / `updateCueTimesBulk` / `splitCue` / `mergeWithNext` / `insertAtTime` / `nudge` / `duplicateCues` / `removeCues` / `select` …），自带撤销栈与 coalesce 合并——**这就是现成的 Agent action 集合，不需要新发明**。
- **纯逻辑可无头单测**：`js/audio/timing.js`（Aegisub 对话定时语义，pending→commit 模型）、`js/format/*`（SRT/VTT/ASS 双向解析序列化）、`js/audio/capture.js` 的峰值采集数学（`createPeakCollector`，目标 4kHz 单声道）——`node --test` 126 项已在证明 Node 是合法的无头运行时。
- **调试句柄已挂**：`js/main.js:179` `window.__editor = { store, actions, player, waveform, timing, audioCommands, assPreview }`。
- **自动化加载口已挂**：`?media=<url>&subs=<url>` 启动自动加载（`main.js:154`），走的与手动打开文件同一路径。
- **测试物料齐**：`.fixture/`（test.wav / test-video.mp4 / test.ass）、`.smoke/`（含 coi-server.py 与样例字幕）可直接作为 CLI/harness 的验收夹具。

### 2.2 缺口

1. **波形解码绑死浏览器**：主路 `decodeAudioData`，兜底"加速采集"依赖 `HTMLMediaElement`——Node 中两者皆无，headless 取峰值需要另一条解码路径。
2. **`__editor` 是调试口不是契约**：无文档、无稳定性承诺、字段随 store 内部形态漂移；Agent 直接依赖它等于依赖实现细节。
3. **波形数值无导出口**：峰值只存在于 wavesurfer 实例与渲染循环里（`ws.load(mediaUrl, channels, dur)`），页面上没有 `getPeaks(t0, t1)` 这类读数 API。
4. **无机器可读校验**：重叠、间隙、最短时长、边界距静音的距离等检查未成文，更没有 JSON 报告与退出码，无法进 CI 或流水线。
5. **本地文件加载依赖用户手势**：页面通过 `File` 对象打开文件；harness 必须走 fetch/URL 路径（autoload 已具备，需固化）。

## 3. 关键决策分析

### 3.1 "视觉模型看波形打轴"的正确打开方式

用户设想的"最简单场景"：渲染时间轴波形图 → 视觉模型识别波峰波谷 → 输出时间点。分析：

| 路径 | 精度 | 成本 | 可复现性 |
|---|---|---|---|
| 视觉模型读 PNG | 受像素分辨率限制：常见缩放下约 10ms/px，模型标注误差再叠加数像素，实际 ±30–100ms | 每张图一次视觉调用 | 弱（同一图两次结果可能不同） |
| Agent 读峰值 JSON | 采样分辨率 0.25ms（4kHz），文本模型可直接做能量门限/拐点推理 | 纯文本 token | 强（确定性数据） |
| 确定性 VAD 代码 + Agent 裁决 | 同上，且拐点检测不耗模型推理 | 最低 | 强 |

**结论：测量主路径 = 峰值 JSON + 确定性 VAD 候选段，Agent（文本模型）做候选段与 ASR 文本的对齐与裁决；视觉模型只用于验证**（截图复核叠加效果、疑难段粗定位）。若保留视觉输入，必须渲染**带时间刻度网格与标定元数据**（`pxPerSec`、`t0`）的 PNG，把"看图猜"降级为"看图读刻度"。注意上层仓库已有 VAD 相关工作（`docs/方案B-自适应声学与VAD优化方案`、`docs/字幕时间轴精度优化方案`），CLI 的 VAD 命令应与之对齐复用，不另起炉灶。

### 3.2 CLI、浏览器 Harness、MCP 三者不是三选一

| 形态 | 覆盖场景 | 优势 | 弱点 |
|---|---|---|---|
| A. Headless CLI（数据面） | 流水线组件：批量变换、校验、进 CI | 确定性、可组合（JSON in/out、退出码）、快、无需浏览器 | 无浏览器解码与采集兜底；看不见 UI |
| B. CDP Harness（控制面） | Agent 在真实页面持续工作、人在回路 | 复用页面全部能力（decodeAudioData、加速采集兜底、JASSUB 预览、撤销/草稿）；人与 Agent 同屏，改动可整体撤销回滚 | 需起 Chrome；比进程内 CLI 慢 |
| C. MCP Server | 给 ZCode/claude code 原生工具面 | 即插即用、工具描述自解释 | 只是 A/B 的薄包装，不产生新能力 |

**取舍：A + B 先行，C 后置可选。** A 服务"组件被流水线调用"的定位；B 服务"Agent 在真实场景持续工作 + 人类监督"的定位——两者共用同一套 actions/采集/校验代码，互为补充。B 还有一个独特价值：**Agent 的所有写操作都走 actions，天然进入撤销栈与 localStorage 草稿**，人类终审时看到的就是 Agent 的工作现场，Ctrl+Z 可逐步回退。

### 3.3 Headless 下的波形解码策略

- **CLI 侧**：优先探测系统 `ffmpeg`（上层 ASR 流水线本就依赖它）→ 抽取 16kHz/mono PCM → 喂给 `capture.js` 已单测的采集数学产出 4kHz 峰值。无 ffmpeg 时：接受 WAV 直读（`.fixture/test.wav` 可测）或明确报错退出（退出码区分），不静默降质。
- **Harness 侧**：无需解决——页面内两条解码路径照常工作，MKV/AC-3 等覆盖不了的媒体自动走加速采集兜底。
- **一致性**：两路峰值率统一对齐 `capture.js` 的 `TARGET_RATE`（4kHz）；文档化浏览器解码与 ffmpeg 解码在边界样本上的微小差异容忍度（预期 < 1 个峰值窗，对打轴决策无影响）。

### 3.4 零依赖原则的边界

项目卖点是"零构建、零 npm 依赖、纯本地"。守则：

- CLI 与 harness 用 **Node 标准库**实现（`node:test` 已开先例；本机 Node v24，全局 WebSocket 可裸连 CDP，无需 playwright）；
- `ffmpeg` 为"检测到即用"的可选外部能力，缺失时优雅降级并打印获取指引；
- MCP 包装若做，同样仅标准库（stdio JSON-RPC 足够）。

## 4. 方案设计

### 4.1 AGENTS.md（环境理解层）

放置于 `tools/subtitle-editor/AGENTS.md`，内容清单：

- 一句话定位 + 与上层流水线的关系；
- 目录地图与分层：`state.js`（store）→ `actions.js`（命名命令+撤销）→ `ui/*`（只做显示与输入）；纯逻辑在 `audio/timing.js`、`format/*`、`capture.js`（数学部分）；
- 命令：`python3 -m http.server 8631`（模块版）、`node --test`、`node build-standalone.mjs`（单文件版）；
- 页面 Agent API 速查表（见 4.2）与 `?media=&subs=` 用法；
- 数据不变量：cues 恒排序、`MIN_LEN` 最短时长、定时改动走 pending→commit 不直写、写操作只经 actions；
- 扩展守则：新逻辑优先纯函数进 `js/` 可单测；UI 进 `js/ui/`；改动跑 `node --test`。

### 4.2 页面 Agent API：`__editor` → `window.agent` 契约

不新造机制，把现有句柄冻结为版本化契约（`window.agent = { version: 1, ... }`）：

- **只读**：`getSnapshot()`（媒体时长、cues 全量、选中态、dirty/撤销栈深度、波形是否就绪）；`getPeaks(t0, t1)`（数值数组 + 采样率）；`renderWaveform({ t0, t1, pxPerSec })` → dataURL + 标定元数据 `{ t0, pxPerSec }`；
- **加载**：`loadMedia(url)` / `loadSubs(url)` —— 固化现有 autoload 的 fetch 路径；
- **写**：直接转发 `actions` 命名命令（`setCueTimes`、`splitCue`、`mergeWithNext`、`insertAtTime`、`nudge`、`select`、`undo`/`redo`），参数形状与 actions 一致——**不绕过 store 直改，自动获得撤销/草稿/ASS 预览联动**。

契约测试：用 harness 对 `window.agent` 做冒烟（纳入 `.smoke`），字段变更即测试红。

### 4.3 Harness（`tools/agent/harness.mjs`，Node ≥ 22 标准库）

职责：起/连 Chrome（`--remote-debugging-port`，仅 127.0.0.1）→ 打开页面（含 `?media=&subs=`）→ `Runtime.evaluate` 调用 agent API → 截图/取值落盘 JSON/PNG。子命令式：

```
harness open  --media X --subs Y [--attach]   # attach=连接已开的浏览器（人监督模式）
harness state                                    # getSnapshot → stdout JSON
harness peaks --t0 .. --t1 ..                    # getPeaks → stdout JSON
harness act  setCueTimes '{...}'                 # 转发命名命令
harness shot --out p.png [--t0 --t1 --pxPerSec]  # 截图/定区渲染
```

**人监督模式**是"真实场景持续工作"的关键形态：浏览器开着、人随时可看，Agent 每步改动在撤销栈里，整段工作可一键回滚。

### 4.4 Headless CLI（`tools/agent/cli.mjs`）

全部子命令 JSON 输出（`--format text` 提供人读模式），退出码：0 正常 / 2 输入错误 / 3 校验发现问题：

| 命令 | 功能 |
|---|---|
| `info <media\|subs>` | 时长/音轨/编码；字幕格式、行数、时长分布 |
| `peaks <media> [--rate 4000]` | 能量包络 JSON（ffmpeg→PCM→capture 数学；无 ffmpeg 报错或 `--wav` 直读） |
| `vad <media> [--opts]` | 确定性语音段候选（能量门限+滞回+最短段长），对齐上层 VAD 方案 |
| `cues <subs> get/set/shift/split/merge/renumber` | 字幕档位读写变换，复用 `format/*` |
| `convert <in> --to srt\|vtt\|ass` | 格式互转 |
| `check <subs> [--media]` | 重叠/间隙/最短时长/边界距 VAD 静音距离，机器可读报告，可进 CI |

组合进上层流水线示例（Python 侧 `subprocess` 或直接由 Agent 驱动）：

```
cli vad media.mkv --json > seg.json
# Agent：seg.json × ASR 草稿 SRT 对齐 → 生成逐行 set 计划
cli cues draft.srt set --plan plan.json -o timed.srt
cli check timed.srt --media media.mkv   # 退出码 0 才放行
```

JSON 字段命名与上层 `schemas/` 对齐，避免两套词汇。

### 4.5 MCP 包装（后置可选，约 0.5 天）

`tools/agent/mcp-server.mjs`（stdio JSON-RPC，标准库实现）把 4.3/4.4 的命令暴露为 MCP tools（`subtitle_peaks`、`subtitle_vad`、`subtitle_cues_set`、`subtitle_check`、`editor_*` 系列）。ZCode/claude code 原生即插即用；codex 亦支持 MCP。只是薄包装，优先级最低。

### 4.6 典型端到端工作流（Agent 打轴 + 人终审）

1. Agent 读 AGENTS.md → 起静态服务 + `harness open --media media.mkv --subs asr-draft.srt`；
2. `harness peaks` / CLI `vad` 取数值候选段；
3. Agent 将候选段与 ASR 文本逐句对齐，`harness act setCueTimes` 批量精修（连续撤销步）；
4. `cli check` 全绿 → `harness shot` 存档证据；
5. 人打开同一页面（草稿同源）复核，疑难行用波形 PNG + 视觉模型辅助判断；
6. 满意后导出 SRT/ASS，进入下游。

## 5. 风险与边界

- **契约漂移**：`window.agent` 必须带版本字段并有冒烟测试，否则页面重构会静默破坏 Agent 工作流；
- **解码差异**：浏览器 vs ffmpeg 峰值的边界差异需文档化（预期 < 1 峰值窗），必要时以 harness 路径为准；
- **安全边界**：harness 仅监听/连接 127.0.0.1；页面本身纯本地无回传，agent API 不新增网络面；
- **范围蔓延守则**：CLI 只做数据变换，**不做无头编辑器**；定时"决策权"始终在 Agent + 人，工具只提供观测与验证。UI 级操作一律走 harness/真实页面；
- **与既有工作的关系**：VAD/时间轴精度已有成套方案文档与 Python 实现，CLI 侧对齐复用而非重写；本方案只补"Agent 可操作的接口面"。

## 6. 分阶段落地与验收

| 阶段 | 内容 | 估时 | 验收标准 |
|---|---|---|---|
| P0 | AGENTS.md | 0.5 天 | 新会话 Agent 不读源码即可正确启动服务、跑测试、用 harness 打开媒体 |
| P1 | `window.agent` 契约 + `harness.mjs` | 1–2 天 | 脚本闭环：加载 `.fixture` 媒体 → 读峰值 → 改一行时间 → 截图；冒烟进 `.smoke` |
| P2 | `cli.mjs` 数据面（peaks/vad/cues/check/convert） | 2–3 天 | 各命令 JSON in/out + 单测；ffmpeg 缺失降级路径有测试；`check` 退出码可进 CI |
| P3（可选） | MCP 包装 | 0.5 天 | ZCode 中作为 MCP server 挂载后可直接调用工具集 |

P0–P2 合计约一周内可完成，均不引入运行时 npm 依赖，不改变"零构建纯本地"的产品形态。
