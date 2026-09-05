# 字幕打轴工作台（Vocal Subtitle Timing Editor）

本地运行的网页版字幕打轴编辑器：视频/音频播放、音频波形、波形区块拖拽调时间轴、字幕列表行内编辑、SRT/VTT/ASS 读写、ASS 样式预览。

**零构建、零 npm 依赖、纯本地运行**——不联网、不上传任何文件。

## 启动

ES Modules 需要 HTTP 服务（不支持 `file://` 直接打开）：

```bash
cd tools/subtitle-editor
python3 -m http.server 8631
# 打开 http://127.0.0.1:8631/
```

或使用任意静态服务器（`npx serve`、nginx 等）指向本目录。

## 使用

1. 点击「打开媒体」或把 视频/音频 文件拖进页面；
2. 点击「打开字幕」或拖入 SRT/VTT/ASS 字幕（也可以从零开始：把播放头移到目标位置按 `N` 逐句新建）；
3. 波形上每条字幕一个色块：拖左右边缘改入/出点，拖中间整体移动，点击空白定位；
4. 双击右侧列表行编辑文本与时间；播放时当前句自动高亮滚动；
5. 「导出 SRT / VTT / ASS」下载结果；编辑过程自动存草稿（localStorage），下次打开同名字幕时可恢复。

### 时间轴快捷键（小键盘方案）

左右手分工：左手打字，右手小键盘打轴。`e.code` 识别，NumLock 任意状态可用；无小键盘时主键盘数字排等效。

| 键位 | 动作 |
|---|---|
| 小键盘 5 | 播放当前字幕，到结束为止 |
| 小键盘 1 | 试听开始点之前 500ms（检查抢拍） |
| 小键盘 3 | 试听结束点之后 500ms（检查拖尾） |
| 小键盘 8 | 停止播放 |
| 小键盘 4 / 6 | 开始点前移 / 后移（±100ms，Shift 微调 ±10ms） |
| 小键盘 7 / 9 | 缩短 / 加长当前句（Shift 微调 ±10ms） |
| 小键盘 2 / 0 | 下一句 / 上一句 |
| 小键盘 Enter | 提交编辑并停止试听 |

其余：`Space` 播放暂停；`←`/`→` 快退快进 5s（Shift 1s）；`,`/`.` 逐帧（帧率在控制条选择）；`N` 插入；`Delete` 删除；`Ctrl+D` 拆分；`Ctrl+M` 合并；`L` 循环当前句；`F` 跟随滚动；`Ctrl+Z` / `Ctrl+Shift+Z` 撤销重做；`Ctrl+S` 按当前格式导出；`Ctrl+滚轮` 缩放波形。完整列表见页面内「快捷键」。

## 字幕格式

- **SRT / WebVTT**：双向读写；VTT 的 cue settings 原样保留。
- **ASS/SSA**：基础读写——文本与时间可编辑，`{\...}` 标签原样保留，其余字段（样式、边距等）与文档其余部分逐字保留；删除的句子对应行整体移除，新建句子以默认字段追加。从 SRT/VTT 导出 ASS 时自动构造最小合法骨架。
- **ASS 样式预览**：经 JASSUB（WASM libass）按样式渲染到画面，可开关；编辑内容约 0.6s 后自动刷新。默认字体为 Liberation Sans（Latin 字形）；中文字幕在无匹配字体时会回退，样式预览初始化失败时自动回退为纯文本预览，不影响编辑。

## 开发

```bash
node --test        # 解析器与动作逻辑单测（39 项）
```

目录结构见 `docs/superpowers/specs/2026-09-05-subtitle-timing-editor-design.md`。浏览器控制台可用 `__editor` 句柄查看内部状态。

## 已知限制

- ASS 样式预览内置字体仅覆盖拉丁字形，中文渲染依赖运行环境字体能力；
- 波形依赖浏览器 `decodeAudioData`，个别编码（或超大文件）无法解码时自动降级为时间刻度视图（点击定位可用，拖拽编辑请用列表）；
- 仅支持浏览器原生可播放的媒体格式。

## 版权与洁净室声明

- 本工具为**独立实现**：开发全程未参考、未复制任何同类商业/非商用编辑器的源代码或美术资源；功能与交互仅依据书面需求描述与打轴软件行业通用惯例设计。
- 第三方组件均在 `vendor/` 内以原始产物形式分发并保留许可声明，详见 **[vendor/NOTICE.md](vendor/NOTICE.md)**：
  - ArtPlayer 5.4.0 —— MIT License © Harvey Zhao（zhw2590582）
  - wavesurfer.js 7.12.11 —— BSD 3-Clause © katspaugh and contributors
  - JASSUB 2.5.14 —— JS 为 MIT；其 WASM 产物内含 libass/freetype/fribidi/harfbuzz 等编译库（LGPL-2.1-or-later AND (FTL OR GPL-2.0-or-later) AND MIT AND MIT-Modern-Variant AND ISC AND NTP AND Zlib AND BSL-1.0，均为商用友好许可）
- `vendor/jassub.esm.js` 为上游 `dist/jassub.js` 与其运行时依赖的 esbuild 打包产物，未修改任何逻辑。
