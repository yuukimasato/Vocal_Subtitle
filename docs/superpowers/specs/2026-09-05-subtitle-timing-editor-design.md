# 字幕打轴工作台（独立网页编辑器）设计

日期：2026-09-05
状态：已批准（2026-09-05 修订：播放器改用 ArtPlayer，新增 ASS 样式预览与小键盘快捷键）
位置：`tools/subtitle-editor/`（完全独立工具目录，不改现有 WebUI/后端代码）

## 1. 目标与范围

从零实现一个网页版字幕打轴编辑器：加载本地音视频、显示音频波形、以波形区块拖拽方式调整字幕时间轴、编辑字幕文本、导入导出 SRT/VTT/ASS。

本阶段包含：

- 媒体加载（本地视频/音频文件）与播放控制：ArtPlayer（MIT）作为播放器外壳承载 `<video>` 与其控制条（播放/进度/音量/全屏）；本工具条另行提供逐帧步进、fps 选择、倍速、循环当前句、试听窗口；
- 波形渲染与字幕区块：每条字幕一个可拖拽区块（拖边缘改入出点、拖中间整体移动、点击空白定位）；
- 字幕列表：单击跳转选中、双击行内编辑（时间+文本）、当前句高亮自动滚动；
- 编辑操作：播放头插入、删除、拆分、合并、全体撤销/重做；
- 时间轴小键盘快捷键（REAPER 打轴式键位，见 §6）；
- SRT/VTT 双向读写；ASS/SSA 基础读写（文本与时间可改，`{\...}` 标签原样保留，其余字段与文档内容原样写回）；
- ASS 样式预览（可开关）：加载 ASS 时经 JASSUB（WASM libass）在画面上按样式渲染；编辑内容防抖后重建预览；关闭或初始化失败时回退为纯文本 overlay；
- 导出下载（Blob）、localStorage 编辑草稿自动保存与恢复提示；
- 音频解码失败时波形区降级为静态刻度+区块视图（点击定位可用，不支持拖拽编辑）。

本阶段不包含：

- ASS 样式的可视化编辑器（样式仅在预览中呈现，源文件字段原样保留）；
- 与现有 FastAPI 后端、任务历史、反馈学习的任何集成（后续阶段可加跳转入口）；
- 波形缩放控件（仅 Ctrl+滚轮缩放）、多人协作、批量时间轴平移、翻译对齐；
- 内置完整 CJK 字体（预览默认字体仅覆盖拉丁字形；中文样式渲染依赖用户系统字体访问能力，不可用时显示回退字体）。

## 2. 洁净室合规措施

参考对象 `vtt-editor-pro-3.1/` 为非商用授权，**不得复制其任何源代码与美术表达**。本设计采用洁净室流程：

1. 开发全程不阅读参考项目的任何 HTML/JS/CSS 源码与截图；实现唯一依据是本文档。
2. 功能需求来自用户书面描述（上传视频→显示波形→拖拽打轴、上方视频下方波形等）与打轴软件行业通用惯例，这些是功能思想，不受版权保护。
3. UI 视觉从零设计，复用本项目自己的 WebUI 颜色 token（`vocal_subtitle/webui/static/styles/app.css`），图标为自绘通用几何符号 + 文字标签。
4. 第三方代码全部为商用友好许可并以 vendor 文件引入，保留声明文件（见 `vendor/NOTICE.md`）：wavesurfer.js 7.12.11（BSD-3-Clause）、ArtPlayer 5.4.0（MIT）、JASSUB 2.5.14（JS 为 MIT；WASM 产物含 libass/freetype/fribidi 等编译库，许可栈为 LGPL-2.1-or-later AND (FTL OR GPL-2.0-or-later) AND MIT AND MIT-Modern-Variant AND ISC AND NTP AND Zlib AND BSL-1.0）。

## 3. 技术形态

零构建：原生 ES Modules + 静态文件，无 npm 依赖、无打包步骤（vendor 产物在开发期一次性生成后入库）。本地预览需经 HTTP（ES Modules 不支持 `file://`），用 `python3 -m http.server` 或任意静态服务器指向本目录。

```text
tools/subtitle-editor/
  index.html              # 唯一入口
  css/editor.css          # 自绘深色主题（复用项目 token）
  js/
    main.js               # 装配与启动
    state.js              # 中央状态 + 发布订阅
    actions.js            # 字幕增删改/拆分/合并/微调（统一走撤销快照）
    history.js            # 撤销/重做快照栈（上限 100）
    shortcuts.js          # 全局键盘（含小键盘键位）
    format/time.js        # 时间解析/格式化（SRT/VTT/ASS/显示/输入）
    format/cue.js         # cue 构造与排序
    format/srt.js         # SRT 解析/序列化
    format/vtt.js         # WebVTT 解析/序列化
    format/ass.js         # ASS/SSA 骨架式解析/序列化 + 最小骨架构造
    format/index.js       # 格式嗅探与分发
    ui/player.js          # ArtPlayer 装配、播放控制条、试听窗口、文本 overlay
    ui/waveform.js        # wavesurfer 封装 + 区块同步 + 降级视图
    ui/ass-preview.js     # JASSUB 装配/重建/销毁（样式预览开关）
    ui/cue-list.js        # 字幕表格与行内编辑
    ui/toolbar.js         # 顶栏、文件打开、拖放、导出
    ui/toast.js           # 轻量提示
  vendor/                 # 第三方产物 + NOTICE.md + 各 LICENSE（见 NOTICE）
  test/                   # node:test 单测（解析器为纯函数）
  package.json            # 仅 type:module 与 test script，零依赖
  README.md               # 用法 + 版权/洁净室声明
```

## 4. 数据模型与数据流

中央状态（`state.js`，发布订阅，高频播放进度不经此）：

```js
{ cues, selectedId, editingId, duration, fps: 25, rate: 1, follow: true,
  loopCue: false, previewOn: true, assPreview: false,
  mediaName, subtitleName, subtitleFormat, subDoc, dirty }
```

`cue = { id, start, end, text }`，秒为单位的 float；`cues` 恒按 start 稳定排序。VTT cue 可携带 `settings` 原样串；ASS 额外携带 `cue.meta`（字段序、其余字段原值）。

数据流：文件打开 → 解析器产出 `{cues, doc}` → store → 订阅方（波形同步区块 / 列表重渲染 / 工具栏状态 / ASS 预览防抖重建）。所有字幕变更只能走 `actions.js`，先推撤销快照再写 store 并派发 `cues` 事件，随后派发 `history` 状态。播放进度由 `player.js` 的 rAF 循环直接驱动时间显示、overlay、列表高亮与试听窗口，不进 store；JASSUB 自行经 requestVideoFrameCallback 与视频同步。

## 5. 交互与布局

上方视频（ArtPlayer，含可开关的字幕文本 overlay 与 ASS 样式预览层）、下方波形轨道 + 播放控制条、右侧字幕列表；<1100px 列表折叠到底部。波形与 ArtPlayer 内部 `<video>` 绑定同一媒体元素。

- 波形：点击空白 seek；区块拖边缘改时间、拖中间移动，拖拽结束一次性写入 store（单条撤销记录）；Ctrl+滚轮缩放；拖选新块不启用（插入统一走播放头插入）。
- 列表：单击行选中并跳转；双击进入行内编辑（开始/结束接受 `mm:ss.mmm`/`ss.mmm`/秒数，Enter 提交、Shift+Enter 换行、Esc 取消）；行内快捷按钮拆分/合并/删除。
- 拆分规则：时间按播放头位置，文本按时间比例的字符边界；合并为与下一句合并（文本以换行连接）。
- 降级：波形解码失败显示错误条 + 静态刻度区块视图（点击 seek、随 cues 重绘），编辑经列表完成；JASSUB 初始化失败自动关闭样式预览并提示。

## 6. 时间轴快捷键（小键盘键位，`e.code` 识别，不受 NumLock 影响）

| 键位 | 动作 |
|---|---|
| NumpadEnter | 提交编辑中的输入并停止试听 |
| Numpad5 | 播放当前字幕，到结束为止 |
| Numpad1 | 试听开始点之前 500ms |
| Numpad3 | 试听结束点之后 500ms |
| Numpad8 | 停止播放 |
| Numpad4 / Numpad6 | 当前句开始点前移 / 后移（默认 ±100ms，Shift ±10ms） |
| Numpad7 / Numpad9 | 缩短 / 加长当前句（结束点 ∓/±100ms，Shift ±10ms） |
| Numpad2 / Numpad0 | 下一句 / 上一句 |

主键盘数字排镜像同一套动作（无小键盘设备）；其余：Space 播放暂停、←/→ seek 5s（Shift 1s）、`,`/`.` 逐帧、N 插入、Delete 删除、Ctrl+D 拆分、Ctrl+M 合并、L 循环当前句、F 跟随、Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y 撤销重做、Ctrl+S 按当前格式导出。输入框聚焦时仅放行编辑器自身按键；ArtPlayer 内置全局热键关闭，快捷键统一由本工具接管。

## 7. 格式解析

- 统一剥 BOM、CRLF 归一化；解析为纯函数，逐行报错（含行号）。
- SRT：序号行可有可无，毫秒分隔符 `,`/`.` 均收；导出 `,`、按 start 排序、UTF-8 无 BOM。
- VTT：识别 `WEBVTT` 头；跳过 NOTE/STYLE/REGION；cue settings 原样保留写回；导出重新编号。
- ASS/SSA：骨架式——非 Dialogue 行原样保留；`[Events] Format:` 定义字段序（缺失时用 v4+ 常见默认序）；Dialogue 按字段拆分，Text 为最后一段（允许含逗号）；`\N`/`\n` ↔ 换行双向转换；删除的句对应行整体移除；新增句以默认字段追加到 `[Events]` 末尾；从 SRT/VTT 导出 ASS 时构造最小合法骨架。
- 嗅探：优先扩展名，失败按内容特征（`WEBVTT` 头 / `-->` 时间行 / `[Script Info]`）。

## 8. 错误处理

媒体加载失败、字幕解析失败、音频解码失败、JASSUB 初始化失败均给出可见且可操作的提示；时间输入非法拒绝提交并还原；草稿写 localStorage 失败静默降级；`beforeunload` 在有未导出修改时提醒。不在本地存储任何敏感信息。

## 9. 测试与验收

- 解析器单测（`node --test`）：time/SRT/VTT/ASS 的解析、容错、错误行号、往返一致性（含 ASS 骨架保真、删除句、文本逗号、`\N` 转换）与嗅探。
- 浏览器 smoke（本地静态服务）：打开媒体出 ArtPlayer 控制条与波形、解析示例字幕、区块拖拽改时间回写列表、小键盘试听/微调、ASS 样式预览开关、导出产物与解析往返一致、解码失败走降级视图。
- 验收：全程零构建可运行；不修改 `tools/subtitle-editor/` 之外任何文件（spec 与本目录除外）；README 含使用方法、第三方许可声明与洁净室声明。
