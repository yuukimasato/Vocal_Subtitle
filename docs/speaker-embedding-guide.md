# 说话人嵌入模型配置指南

## 为什么需要说话人嵌入模型？

Vocal Subtitle 的说话人分离默认使用间隙交替方案，能将交替对话中的语句分配给两个说话人。但对于以下场景，需要更精确的声学模型：

- **2 人以上对话**：间隙交替方案仅支持二元分配
- **说话人音色相近**：基于间隙无法区分谁在说话
- **非严格交替对话**：一人连续多句后换人，间隙模式会出错

**说话人嵌入模型**通过深度学习分析声纹特征，能从任何长度的音频中提取说话人身份向量，实现更精确的聚类。

---

## 支持的模型

### 方案 1：pyannote/embedding（推荐）

| 属性 | 值 |
|------|-----|
| 模型 | ECAPA-TDNN |
| 嵌入维度 | 512 |
| 代码协议 | MIT |
| **模型协议** | **需签署** (pyannote-eula) |
| 文件大小 | ~100 MB |
| 需 HF Token | ✅ 是 |

### 方案 2：speechbrain/spkrec-ecapa-voxceleb

| 属性 | 值 |
|------|-----|
| 模型 | ECAPA-TDNN on VoxCeleb |
| 嵌入维度 | 192 |
| 代码协议 | Apache 2.0 |
| 模型协议 | Apache 2.0 |
| 文件大小 | ~80 MB |
| 需 HF Token | ❌ 否（公开模型） |

---

## 配置步骤（pyannote/embedding）

### 第一步：签署模型使用协议

1. 打开 [https://huggingface.co/pyannote/embedding](https://huggingface.co/pyannote/embedding)
2. 登录你的 HuggingFace 账号（没有则注册）
3. 在模型页面点击 **「Agree and access repository」** 按钮
4. 填写使用用途（如 "personal research" 或 "speaker diarization for subtitle generation"）
5. 提交后等待页面刷新，确认显示 **"You have been granted access to this model"**

![示意：点击 Agree and access repository](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)

### 第二步：获取 HF Token

1. 打开 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. 点击 **「Create new token」**
3. Token 类型选择 **`Read`**（只需读取权限）
4. Token 名称填写 `vocal-subtitle`（便于识别）
5. 点击 **「Create token」**
6. **立即复制**生成的 token（格式：`hf_xxxxxxxxxxxxxxxxxxxxxxxxxx`）
   > ⚠️ Token 只显示一次，关闭页面后无法再次查看

### 第三步：在 Vocal Subtitle 中配置

1. 打开 Web GUI → 左侧设置面板
2. 找到 **「🧬 说话人嵌入模型」** 设置组
3. 勾选 **「启用嵌入模型」**
4. 模型选择保持默认 `pyannote/embedding (512维)`
5. 在 **「HF Token」** 输入框中粘贴刚才复制的 token
6. （可选）也可设置环境变量 `HF_TOKEN` 替代 UI 输入

### 第四步：运行验证

1. 上传音频文件，点击「开始处理」
2. 首次运行将自动下载模型（~100 MB），网络良好时约 1-2 分钟
3. 下载完成后，模型缓存于 `cache/speaker_models/`，后续无需重新下载
4. 观察运行日志：应出现 `Speaker embedding engine loaded: pyannote (dim=512)`
5. 检查输出字幕：说话人数量应正确检测，同一说话人标注一致

---

## 配置步骤（speechbrain/ecapa — 无需签署协议）

此方案无需 HuggingFace token，直接在设置中选择即可：

1. 找到 **「🧬 说话人嵌入模型」** 设置组
2. 勾选 **「启用嵌入模型」**
3. 模型选择切换为 `speechbrain/ecapa (192维, Apache 2.0, 无需协议)`
4. HF Token **留空**
5. 运行即可，首次自动下载模型

---

## 环境变量 (可选)

若不希望在 UI 中输入 Token，可设置环境变量：

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
```

程序会按以下优先级查找 Token：
1. 前端 UI 输入 (`speaker_embedding.hf_token`)
2. 环境变量 `HF_TOKEN`
3. 环境变量 `HUGGING_FACE_HUB_TOKEN`

---

## FAQ

### Q: 模型下载失败怎么办？
A: 检查 Token 是否正确、网络是否可访问 huggingface.co。可手动下载模型文件放到 `cache/speaker_models/` 目录。

### Q: 如何确认嵌入模型正在工作？
A: 查看运行日志，应包含 `Speaker embedding engine loaded: pyannote (dim=512)`。如果是 `Gap-based alternation`，说明降级到了间隙方案。

### Q: pyannote 协议会影响商用吗？
A: pyannote/embedding 模型采用 pyannote-eula 协议，商业使用需确认协议条款。如担心合规问题，可使用 speechbrain/ecapa（Apache 2.0）。

### Q: 嵌入模型对 GPU 有要求吗？
A: CPU 即可运行。GPU 会更快，但不是必须。

### Q: 能用自己的模型吗？
A: 可以。`speaker_embedding.model_ref` 支持任何 HuggingFace 上的 speaker embedding 模型，或本地路径。
