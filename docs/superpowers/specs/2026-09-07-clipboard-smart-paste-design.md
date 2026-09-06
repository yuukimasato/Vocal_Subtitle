# 剪贴板智能粘贴设计（纯文本按行新建 + 字幕格式解析粘贴）

日期：2026-09-07
状态：已实现
范围：tools/subtitle-editor（零构建网页版打轴编辑器）

## 背景与目标

编辑器已有两条粘贴路径：

- 「粘贴行」（Ctrl+V / 右键）：`actions.pasteCues`，把剪贴板行作为新行插入；
- 「选择性粘贴…」（右键）：`actions.pasteSpecial` + 字段勾选对话框，Aegisub Paste-Over 语义。

外部剪贴板经 `parseSubtitle` 嗅探（SRT/VTT/ASS），嗅探失败即整体失败——**纯多行文本**（逐行歌词、译文）目前粘贴报错「剪贴板中没有可粘贴的字幕行」。此外从 Aegisub 复制出的剪贴板是裸 `Dialogue: ...` 行（无 `[Events]` 段头），`parseAss` 解析出 0 条，同样失败。

目标（对齐 Aegisub 语义并覆盖纯文本场景）：

1. 剪贴板是**纯多行文本**时，粘贴自动**按行新建字幕行**；
2. 剪贴板包含**字幕内容**（SRT / WebVTT / ASS/SSA，含 Aegisub 复制的裸 Dialogue 行）时，解析后照常粘贴（插入或选择性覆盖）；
3. 「选择性粘贴」入口在纯文本时直接新建行（不弹对话框），在字幕格式时保留字段覆盖对话框；
4. 空文档也能粘贴（歌词工作流：开媒体 → 贴歌词 → 逐句打轴）。

## 剪贴板分类

纯函数 `classifyClipboard(text)`（`js/format/index.js`，与 `detectFormat` 同处）：

```text
classifyClipboard(text)
  ├─ 空白文本                          → null（无可粘贴）
  ├─ 嗅探到格式（SRT/VTT/ASS）        → { kind: 'subtitle', cues, format, doc }
  │    └─ 解析抛 ParseError 时向上抛出（嗅探成功但内容损坏 → 报错而非静默）
  ├─ 嗅探到格式但 0 条 cue             → 回落为纯文本
  └─ 其余                              → { kind: 'text', lines }
       lines = 按 \r\n|\n|\r 拆分 → 每行 trim → 跳过空行
```

## 行为定义

### 带时间的剪贴板（kind=subtitle）

维持现状：`pasteCues` 按原时间插入新行；`pasteSpecial` 弹字段对话框，从参考行起逐行覆盖勾选字段，溢出行追加在末尾。

### 纯文本剪贴板（kind=text）

- 新建行时间采用**顺序占位**：第一行从参考行 `end`（无参考行则播放头，再无则 0）起，每行 5 秒（复用 `insertRelativeTo` 的 `BLANK_LEN` 语义），行间首尾相接、互不重叠。用户随后在音频盒逐句打轴。
  （备选的 Aegisub 同款「全部 0 长度堆在 t=0」与「沿用参考行时间互相重叠」被否：N 行叠在一起在波形上不可分辨，不利于逐句试听。）
- 「粘贴行」与「选择性粘贴…」行为一致：直接按行新建。
- 「选择性粘贴…」在 kind=text 时**不弹对话框**（无时间字段可选，无需勾选）。
- 新建行被选中，一次撤销可整体回退。

### 空文档

右键「选择性粘贴…」不再要求 `cues.length > 0`：kind=text 从 0 开始顺序占位；kind=subtitle 走现有溢出追加路径（`from=0`，全部按新行追加）。

### 错误反馈

- 剪贴板无内容 / 权限被拒：现有 toast「剪贴板中没有可粘贴的字幕行」。
- 嗅探成功但解析失败（如残缺 SRT）：toast 展示 `ParseError` 消息（带格式与行号），不静默回落为纯文本——避免把损坏字幕悄悄贴成散文行。

## 组件与改动点

| 单元 | 改动 |
| --- | --- |
| `js/format/index.js` | 新增 `classifyClipboard`（纯函数，单测覆盖） |
| `js/format/ass.js` | `parseAss` 宽容化：`Dialogue:` 行不再要求处于 `[Events]` 段内（修复 Aegisub 互操作），字段缺省用 `DEFAULT_ASS_FIELDS` |
| `js/actions.js` | ① `createActions(store, deps)` 增加可注入 `readClipboardText`（默认 `navigator.clipboard.readText`，测试可注入）；② `loadExternalClipboard` 改用 `classifyClipboard`，纯文本条目为 `{start:null, end:null, text}`，`ParseError` 重新抛出；③ 新增 `clipboardKind()`：确保剪贴板已加载并返回 `'subtitle' | 'text' | null`；④ `pasteCues` 为无时间条目生成顺序占位时间；⑤ `pasteSpecial` 遇纯文本条目委托给 `pasteCues`（防御 + 语义总闸） |
| `js/ui/cue-list.js` | 「选择性粘贴…」改为先 `clipboardKind` 路由：text → 直接 `pasteCues`；subtitle → 原对话框；菜单项取消 `hasCues` 禁用条件；`tryPaste` 增加 ParseError catch 展示解析错误 |
| `js/shortcuts.js` | Ctrl+V 路径 catch `ParseError` 展示消息 |
| `index.html` | 选择性粘贴对话框 hint 补一句纯文本直贴说明；帮助页粘贴键位说明补充 |
| `README.md` | 使用说明第 5 条补充纯文本粘贴行为 |
| 测试 | `test/clipboard.test.mjs`（分类纯函数）、`test/ass.test.mjs`（裸 Dialogue）、`test/actions.test.mjs`（注入剪贴板的端到端：纯文本顺序占位、SRT 粘贴、损坏 SRT 报错、pasteSpecial 委托） |

## 不做的事（YAGNI）

- 不解析 LRC 等其他歌词格式（编辑器本身不支持 LRC 文件，单独给剪贴板加格式不一致）；
- 不跨文档保留 ASS 样式/meta（粘贴仅取时间与文本，与现状一致；字段覆盖对话框也只有时间/文本三项）；
- 不改动内部行剪贴板的复制格式。

## 测试与验收

`node --test test/`（Node 内建 runner，纯 Node 无依赖）。关键断言：

1. 纯文本 3 行 → `pasteCues({refId})` 在参考行后新增 3 行，时间 `ref.end` 起 5s/行、互不重叠，选中新行，一步撤销；
2. 空文档纯文本 → 从 0 开始；
3. SRT 字符串注入剪贴板 → 插入行时间与文本来自 SRT；
4. 裸 `Dialogue:` 三行（Aegisub 复制格式）→ 分类为 subtitle，解析 3 条；
5. 残缺 SRT（时间行缺 `-->`）→ `pasteCues` reject `ParseError`；
6. `clipboardKind` 返回 text 时 `pasteSpecial` 实际执行插入而非覆盖。
