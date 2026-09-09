# AGENTS.md —— 字幕打轴工作台（Subtitle Timing Editor）

零构建、零 npm 依赖、纯本地的网页版字幕打轴编辑器，外加一套 headless Agent 工具链。定位是 **ASR 的功能补充组件**：ASR 产出文本与粗时间 → 本工具做时间轴精修。任何 ASR 流水线都可接入（Vocal_Subtitle 是消费者之一，本项目不属于它）。

**硬约束：独立开源组件。禁止 import 上层项目任何模块路径；产品文件中出现 `Vocal_Subtitle` / `/home/` 等绑定会被 `test/independence.test.mjs` 判红。**

## 命令

```bash
python3 -m http.server 8631     # 模块版页面（ESM 需 HTTP 服务）→ http://127.0.0.1:8631/
node --test                     # 全量单测（node:test，无任何依赖）
node build-standalone.mjs       # 重建单文件版 HTML（仅构建期用 npx esbuild）
node agent/cli.mjs --help       # headless CLI（JSON in/out；info/cues/convert/check/peaks/vad）
node agent/harness.mjs --help   # 通用 CDP harness（驱动真实页面）
node agent/mcp-server.mjs       # MCP server（stdio；把 CLI/harness 挂成 tools）
node .smoke/agent-smoke.mjs     # 真实 Chrome 冒烟：契约表面/字段路径与 baseline 比对 + 截图
```

自动化打开页面：`http://127.0.0.1:8631/?media=<url>&subs=<url>`（与手动打开文件同一路径）；加 `&agent=1` 进入 agent 会话（独立草稿命名空间）。

## 目录地图与分层

```
index.html            页面骨架
js/state.js           中央 store（发布订阅）
js/actions.js         命名命令层 + 撤销栈 + coalesce —— 一切写操作的唯一入口
js/history.js         撤销/重做栈
js/draft.js           localStorage 草稿（800ms 防抖）
js/agent-api.js       window.agent 版本化契约（冻结于 .smoke/agent-contract.baseline.json）
js/audio/timing.js    Aegisub 对话定时控制器（纯逻辑，无 DOM）
js/audio/capture.js   波形兜底解码；createPeakCollector 为跨解码路径的峰值归一数学（纯）
js/audio/commands.js  音频命令层
js/format/            SRT/VTT/ASS 解析与序列化（纯逻辑）
js/ui/                只做显示与输入（player/waveform/cue-list/toolbar/…）
agent/                headless 工具链：cli.mjs、harness.mjs、纯逻辑模块（cueOps/wav/ffmpeg/peaks/vad…）
test/                 node --test 单测（含 independence.test.mjs 防回归断言）
.smoke/               冒烟物料与 agent-contract.baseline.json
docs/                 设计文档（开发文档 + 实施计划 + VAD provider 契约）
vendor/               第三方原样产物（勿改）
```

依赖方向：`ui/*` → `actions.js` → `state.js` / `format/*`；纯逻辑不 import UI。

## 能力分层（独立性的技术证明）

Tier 0–3 在只有 Node 标准库时全部可用；更重的能力"检测到即用、缺失即优雅降级"：

| Tier | 能力 | 依赖 | 缺失时行为 |
|---|---|---|---|
| 0 | 字幕读写/变换/结构校验（`cli.mjs cues/convert/check/info`） | 无 | — |
| 1 | WAV 直读 → 4kHz 峰值流 | 无 | 非 WAV 报错退出（码 2），提示装 ffmpeg |
| 2 | 任意媒体解码 → PCM → 4kHz（`peaks`/`vad`） | 可选 ffmpeg | 退回 Tier 1 |
| 3 | 内置确定性 VAD（能量+滞回） | 无 | 无音频输入时只做结构校验 |
| 4 | 高质量 VAD（Silero 等） | 用户自备 provider 命令 | 退回 Tier 3，JSON 记录降级 |
| 5 | 真实页面驱动/截图（`harness.mjs` + `window.agent`） | Chrome + CDP | 不可用（不影响 0–4） |

所有解码路径经同一个 `createPeakCollector` 归一到 4kHz——CLI 拿到的数值与人在浏览器波形上看到的是同一套数据。

## 页面 Agent API 速查（`window.agent`，version 1）

```js
window.agent.getSnapshot()                    // → {media,cues,selection,history,waveform…} 全量只读态
window.agent.getPeaks(t0, t1)                 // → {rate, peaks[]}（与 CLI peaks 同源）
window.agent.renderWaveform({t0,t1,pxPerSec}) // → {dataUrl, t0, pxPerSec}（标定水印烧进 PNG）
window.agent.loadMedia(url); loadSubs(url)    // 固化的 fetch 加载路径
window.agent.act(name, args, {coalesceKey})   // 转发 actions 命名命令（updateCueTimes/splitCue/mergeWithNext/
                                              //   insertAtTime/nudge/duplicateCues/removeCues/select/undo/redo…）
```

字段路径冻结在 `.smoke/agent-contract.baseline.json`，变更须同步更新并说明；冒烟比对脚本 `.smoke/agent-smoke.mjs`。

**测量用数据，验证用视觉**：边界定位走 VAD/峰值数值；视觉模型只回答判断题（截图复核，标定水印在 PNG 内），不做定位。

## 数据不变量（改动前先读）

- cues **恒按 start 稳定排序**（同 start 按 end）；`MIN_LEN = 0.05`s 为最短行时长；`BLANK_LEN = 5`s 为插入空白行时长；
- 音频盒定时改动走 **pending → commit**，不直写 cues；切行丢弃未提交改动；`LEAD_IN_MS=200 / LEAD_OUT_MS=300 / NUDGE_MS=10`；
- **写操作只经 `actions`**（自动获得撤销/草稿/预览联动），绝不绕过 store 直改；
- 批量自动化写操作传 `coalesceKey` 合并撤销步，否则人类终审时撤销栈被步进淹没；
- agent 会话（`?agent=1`）使用独立草稿命名空间且默认不写人类草稿——防覆盖人未导出的工作；
- `check` 的语义边界：**结构检查归编辑器**（overlaps/gaps/tooShort/endBeforeStart/sorted/boundaryInSilence），声学质量裁决归 provider/流水线；不使用上游吸附算法的词汇（snapped_* 等）。

## CLI 约定

子命令一律 JSON 输出（`--format text` 人读模式）；退出码：**0 正常 / 2 输入错误（含缺依赖）/ 3 校验发现问题**。内置 VAD（`engine:"energy-vad"`）是打轴辅助的边界候选器，**不是 ASR 分段器**（`--help` 与 JSON 输出均带此声明）。外部 VAD provider 契约见 `docs/vad-provider-contract.md`（本仓库拥有并版本化）。

## 扩展守则

- 新逻辑优先写成**纯函数**进 `js/` 或 `agent/`（可单测）；UI 代码进 `js/ui/`；
- 不引入任何 npm 运行时依赖；Node 工具链只用标准库；
- 改动后跑 `node --test`；改了 `window.agent` 字段须同步 baseline JSON；
- 改了品牌/命名相关内容后重建 `subtitle-editor-standalone.html`（`test/independence.test.mjs` 会扫提交的构建产物）；
- 详细设计依据与两稿分歧裁决见 `docs/开发文档-字幕编辑器Agent化.md`。
