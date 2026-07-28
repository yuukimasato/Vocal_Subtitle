# WebUI 凭据字段密码管理器提示修复设计

## 背景

用户在 WebUI 中双击并保存字幕后，Chrome/Edge 可能弹出“要更新密码吗？”提示。截图中的用户名是 `https://api.deepseek.com`，说明浏览器把 LLM API Key 输入框识别成了登录密码字段。页面同时还有 Hugging Face Token 字段，它们都属于 API 凭据而非网站登录凭据。

## 目标

- 字幕编辑和保存过程中不再触发浏览器的密码更新提示。
- LLM API Key 和 Hugging Face Token 仍以圆点遮罩显示，并保留现有输入、读取和存储行为。
- 不修改字幕编辑 API、字幕文件写入逻辑或后端密钥存储逻辑。

## 方案选择

### 方案 A：普通文本字段 + 视觉遮罩（采用）

将两个 API 凭据字段从 `type="password"` 改为普通文本输入，使用 Chromium 的 `-webkit-text-security` 显示圆点，并设置 `autocomplete="off"`、`data-form-type="other"`、`data-lpignore="true"` 等忽略标记。这样浏览器不会把字段纳入登录密码更新流程，同时用户界面仍保持密钥遮罩。

优点是直接消除触发密码管理器的核心语义，改动集中在静态 WebUI；缺点是视觉遮罩不是安全边界，页面脚本仍能读取凭据，这与当前实现和本地存储模型一致。

### 方案 B：保留 password 类型并补充忽略属性

保留 `type="password"`，改用 `autocomplete="off"` 和更多密码管理器忽略属性。改动最小，但 Chrome/Edge 可能忽略这些属性，仍然触发更新提示，因此不能可靠满足目标。

### 方案 C：独立凭据配置页面

把 API 凭据移到单独页面或弹窗。隔离范围更大，但实现复杂，且 password 字段仍可能触发浏览器提示，不适用于本次窄范围修复。

## 实现设计

1. 在 `renderLLMOptions` 生成的 API Key 输入框上移除 password 语义，保留现有 `value`、输入事件和状态徽章同步逻辑。
2. 在 `renderSpeakerEmbeddingOptions` 生成的 HF Token 输入框上执行相同处理，避免另一个凭据字段在后续操作中触发同类提示。
3. 为两个字段增加统一的非登录字段属性：`autocomplete="off"`、`spellcheck="false"`、`autocapitalize="off"`、`autocorrect="off"`、`data-form-type="other"`、`data-lpignore="true"`；字段名不使用 password/login 语义。
4. 增加 CSS 视觉遮罩规则，仅作用于 API 凭据字段。后端接口、localStorage 键名和现有密钥遮罩值不变。

## 验证设计

- 增加/更新静态 WebUI 测试，确认两个凭据字段不含 `type="password"`，并包含 `autocomplete="off"` 与非登录字段标记。
- 运行现有 WebUI/API 相关测试，确保 LLM 配置、HF Token 下载和字幕编辑行为未回归。
- 使用 Chromium 打开 WebUI，加载含 DeepSeek API Key 的配置，编辑并保存字幕；验收标准是字幕正常保存且不出现密码更新提示，两个凭据字段仍显示为圆点。

## 非目标

- 不清理浏览器已经保存的旧凭据。若浏览器已经显示过提示，用户可能需要在浏览器密码管理器中单独删除已有条目。
- 不改变密钥是否持久化、加密存储方式或 API 请求格式。
