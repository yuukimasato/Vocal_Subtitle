# Vocal Subtitle — 人声分离 + 字幕生成全链路工具

从原始音视频文件中自动提取人声并生成精准字幕 (SRT / VTT / ASS)。

> **核心约束**：全链路仅使用 MIT / Apache 2.0 / BSD 等宽松协议的工具，确保可自由商用，零合规风险。
![preview1](https://github.com/yuukimasato/Vocal_Subtitle/blob/main/resources/preview1.png)
![preview2](https://github.com/yuukimasato/Vocal_Subtitle/blob/main/resources/preview2.png)
![preview3 — 字幕打轴工作台 Subtitle Timing Editor](https://github.com/yuukimasato/Vocal_Subtitle/blob/main/resources/preview3.png)
## 功能特性

- **全链路自动化**：人声分离 → 宏观切块 → 音频预处理 → VAD 检测 → 片段合并 → 说话人分离 → ASR 识别 → 边界精修 → 时间轴映射 → 后处理 → 字幕输出
- **多引擎可切换**：
  - 分离：UVR (BS-RoFormer, 默认) / Spleeter / Open-Unmix
  - VAD：Silero VAD (默认) / WebRTC VAD / TEN VAD / ffmpeg silencedetect
  - ASR：FunASR / Qwen3-ASR / faster-whisper / whisper.cpp
- **物理优先架构 (Phase 0-3)**：
  - 全局坐标系 (CoordinateMapper)：将各阶段碎片化时间轴统一映射到原始音频时间轴
  - 物理时间线 IR (PhysicalTimeline)：统一承载分离后的音频、VAD 语音段、ffmpeg 骨干证据
  - 词级物理分配 (WordAllocation)：将 ASR 词级时间戳分配到物理跨度，修复迟到词
  - 物理覆盖审计 (PhysicalCoverageReport)：检测遮盖空白、过分配、欠分配
  - 物理字幕分箱 (PhysicalSubtitleBin)：按物理语音间隔自动分箱，替代启发式事件操作
- **7 层渐进优化方案 (Plan 0-7)**：
  - Plan 0：宏观静音切块（长音频自动分治）
  - Plan 1：ffmpeg silencedetect 并行 VAD
  - Plan 2：Silero + ffmpeg + RMS 三方法边界融合
  - Plan 3：段内静音预切分（减少 ASR 漏识）
  - Plan 4：ASR 词级时间戳双向边界精修 + 冗余 ASR + LLM 语义仲裁
  - Plan 5：三层级联 LLM 语义合并（快规则 → 本地 NLP → 云端 LLM）
  - Plan 6：帧级无缝衔接（消除字幕闪烁）
  - Plan 7：方向感知声学校验门控（ffmpeg 骨架为物理基准，禁止跨静音延长）
- **骨架分段模式 (Skeleton Mode)**：跳过 VAD，以 ffmpeg 声学骨架直接分段处理
- **扬声器分离 (Speaker Diarization)**：声学特征提取 + 凝聚聚类 + LLM 角色标注
- **YAML 配置驱动**：5 种场景模板（通用 / 播客 / 教学 / 综艺 / 音乐现场），24 个 dataclass 配置类，参数可精细调优
- **LLM 后处理优化**：可选的 AI 字幕修正（DeepSeek 默认），支持 12 个 Provider 预设，优化前后对比
- **离线/流式双模式**：离线批量 + 实时流式处理，流式下自动降级全局依赖模块
- **人声/伴奏导出**：分离产出的纯净人声和背景声可单独下载保存
- **批量处理**：支持目录级批量处理，进度条实时显示
- **🖥️ Web GUI 图形界面**：拖拽上传、实时 WebSocket 进度、字幕预览 + 批量编辑（合并/拆分/说话人标注）、LLM 对比视图、多格式导出
- **三级缓存架构**：文件级分离缓存 + 片段级转录缓存 + SQLite 持久化任务历史
- **💾 设置持久化**：所有用户参数自动保存，刷新页面后恢复上次配置；服务端持久化到 cache/
- **🧠 自适应反馈学习 (Phase 5)**：上传修订字幕自动学习用户偏好，音频指纹匹配、参数震荡检测、健康度评分、Few-shot 示例缓存、Shadow Mode 安全试错
- **🛡️ 质量治理与黄金集门禁**：离线生产链预检 (preflight)、运行生命周期管理、决策追踪 (decision trace) 与运行报告；黄金集质量门禁覆盖幻觉保留、真实语音漏删、物理越界、跨静音等指标
- **🎛️ 字幕打轴工作台 (独立工具)**：零构建网页版打轴编辑器，视频/音频播放 + Aegisub 式音频盒（波形/标记拖拽）+ ASS 样式预览，提供双击即用的单文件版（见 [tools/subtitle-editor](tools/subtitle-editor/README.md)）
- **全 MIT 兼容**：所有依赖可自由商用

## 快速开始

### 一键部署 (推荐)

```bash
# 克隆项目
git clone <repo-url>
cd Vocal_Subtitle

# 生产链 + CLI/Web GUI（CPU 示例；GPU 可去掉 --cpu）
bash install.sh --production --cpu --download-review qwen3-asr-1.7b --review-mirror

# 安装完成后，激活环境并启动
source venv/bin/activate
vocal-subtitle-gui              # 打开浏览器图形界面
```

### 手动安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS

# 2. 安装依赖（按需选择）
# CLI only:
pip install -e ".[faster-whisper,silero-vad,uvr]"
# CLI + Web GUI:
pip install -e ".[faster-whisper,silero-vad,uvr,webui]"
# CLI + Web GUI + 所有功能:
pip install -e ".[faster-whisper,silero-vad,uvr,webui,llm,local-nlp,diarization]"
# 全量安装:
pip install -e ".[all]"
# Spleeter 已从全量兼容集合中隔离；需要时请使用独立环境:
python -m venv venv-spleeter
venv-spleeter/bin/pip install -r requirements-spleeter-legacy.txt

# 3. 安装系统依赖
# Ubuntu/Debian
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

生产安装会固定安装 faster-whisper、FunASR、Qwen runtime 和 WebUI，并要求默认
Qwen 本地权重预检通过；模型下载到 `~/.cache/vocal-subtitle/review-models/`。
四种显式配对可通过 `--primary-engine`、`--secondary-engine` 和
`--engine-pair-policy risk_only|full_quality` 设置：
`FunASR -> Qwen`、`Qwen -> FunASR`、`Qwen -> Whisper`、`Whisper -> Qwen`。

### 下载模型（可选，首次运行自动下载）

```bash
vocal-subtitle download-models --all
```

多引擎复核模型（Qwen3-ASR、ForcedAligner、SED）不会在普通安装时自动下载。先安装对应依赖，再按需下载：

```bash
pip install -e ".[review-models,qwen-runtime]"
# qwen-asr 0.0.6 固定使用 Transformers 4.57.6，避免 Transformers 5.x API 漂移。

# 查看已登记的官方地址
python3 scripts/download_review_models.py --list

# 下载全部复核模型，默认保存到 ~/.cache/vocal-subtitle/review-models/
python3 scripts/download_review_models.py --all

# 只下载指定模型；可用 --mirror 切换到 hf-mirror.com
python3 scripts/download_review_models.py --model qwen3-asr-1.7b
python3 scripts/download_review_models.py --model sed-ast-audioset --mirror
```

模型注册表及地址如下：

| 用途 | Hugging Face 仓库 |
| --- | --- |
| Qwen3-ASR 高质量识别 | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| Qwen3-ASR 低显存识别 | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| 强制对齐复核 | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) |
| SED 声事件检测 | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) |

也可以在一键安装时显式下载：

```bash
bash install.sh --all --download-review-models
bash install.sh --download-review qwen3-asr-1.7b
```

### CLI 基本用法

```bash
# 单文件处理（默认配置）
vocal-subtitle run input.mp3 -o output.srt

# 指定场景模板
vocal-subtitle run input.mp3 -o output.srt --profile podcast

# 指定语言和输出格式
vocal-subtitle run input.mp3 --language zh --format vtt

# 指定分离引擎和模型
vocal-subtitle run input.mp3 --separator uvr --uvr-model model_bs_roformer_ep_317_sdr_12.9755.ckpt

# 指定设备和 VAD 阈值
vocal-subtitle run input.mp3 --device cuda --vad-threshold 0.4

# 启用说话人分离和角色标注
vocal-subtitle run input.mp3 --diarization --speaker-role

# 启用骨架分段模式（跳过 VAD，以 ffmpeg 声学骨架分段）
vocal-subtitle run input.mp3 --skeleton-mode

# 关闭人声分离（输入已是纯净人声）
vocal-subtitle run vocals.wav --skip-separation

# 导出骨架分段音频（调试用）
vocal-subtitle run input.mp3 --export-skeleton-segments --export-skeleton-dir ./segments/

# 启用 LLM 优化（需设置 API Key）
vocal-subtitle run input.mp3 --llm-optimize

# 批量处理
vocal-subtitle batch ./inputs/ -o ./outputs/ --profile education --pattern "*.mp4"

# 查看系统信息
vocal-subtitle info

# 查看可用模板及说明
vocal-subtitle profiles

# 自适应反馈学习 — 上传修订字幕自动学习偏好
vocal-subtitle feedback learn -a input.wav -r revised.srt
vocal-subtitle feedback learn -a input.mp3 -r fixed.srt --dry-run  # 仅预览
vocal-subtitle feedback show                                         # 查看学习到的参数
vocal-subtitle feedback rollback                                     # 回滚到上一版本
vocal-subtitle feedback reset                                        # 重置为系统默认
vocal-subtitle feedback fingerprints                                 # 查看音频指纹库
vocal-subtitle feedback export -o my_profile.yaml                    # 导出配置分享
vocal-subtitle feedback import -i friend_profile.yaml                # 导入他人配置
```

### Web GUI 图形界面

```bash
# 启动 Web GUI（自动打开浏览器）
vocal-subtitle-gui

# 或使用 Python 入口
python main_gui.py

# 指定端口，不自动打开浏览器
vocal-subtitle-gui --port 8080 --no-browser

# 开发模式（热重载）
vocal-subtitle-gui --reload
```

GUI 功能：
- 🎵 **拖拽上传** — 直接拖入音频/视频文件（自动提取音轨）
- 📋 **场景模板** — 一键切换播客/教学/综艺等预设，参数即时生效
- ⚡ **实时进度** — WebSocket 推送各阶段实时进度（stage_start/progress/stage_finish/complete/error）
- 📝 **字幕预览 + 批量编辑** — 时间轴表格展示，双击编辑单条，批量合并/拆分/设说话人，编辑后自动写回磁盘文件
- 🔍 **LLM 对比视图** — 优化前后并排显示，变更绿色高亮
- 🎤 **分离音频导出** — 下载纯净人声和背景声 WAV 文件
- 📥 **多格式导出** — 一键下载 SRT / VTT / ASS
- 💾 **设置持久化** — 所有参数自动保存到浏览器 localStorage，刷新恢复；服务端持久化到 `cache/persistence_settings.json`
- 📊 **任务历史** — SQLite 持久化，支持分页查看、筛选、删除
- 🔧 **持久化文件管理** — 任务产出 (字幕/音频) 按 TTL 自动管理
- 🧠 **自适应反馈学习** — 上传修订字幕 + 音频，自动分析差异并调整管道参数；支持参数震荡检测、健康度趋势、Shadow Mode 试错、音频指纹匹配
- 🔄 **FunASR 自动准备** — WebUI 启动时自动检测并安装 FunASR 引擎及模型

### Python API

```python
from pathlib import Path
from vocal_subtitle import Pipeline
from vocal_subtitle.config import ConfigLoader

# 加载配置
config = ConfigLoader().load_profile("podcast")

# 创建管道
pipeline = Pipeline(config)

# 运行
result = pipeline.run(
    input_path=Path("input.mp3"),
    output_path=Path("output.srt"),
)

stats = result["stats"]
print(f"完成! 总耗时: {stats.total_time:.1f}s")
print(f"音频时长: {stats.duration_seconds:.1f}s")
print(f"字幕数: {stats.subtitle_count}")
print(f"发言人数: {stats.speaker_count}")
print(f"人声文件: {result.get('vocals_path')}")
print(f"伴奏文件: {result.get('accompaniment_path')}")
print(f"各阶段耗时: {stats.stage_timings}")

# 使用自定义参数覆盖配置
result = pipeline.run(
    input_path=Path("input.mp3"),
    output_path=Path("output.srt"),
    overrides={
        "vad.threshold": 0.45,
        "asr.language": "zh",
        "merging.min_silence_gap": 0.5,
    },
)
```

## 字幕打轴工作台（独立工具）

`tools/subtitle-editor/` 提供一个**零构建、零 npm 依赖、纯本地运行**的网页版字幕打轴编辑器，与主管道解耦，可单独使用：

- 🎬 视频/音频播放（ArtPlayer）+ ASS 样式实时预览（JASSUB，含 wasm 渲染）
- 🌊 Aegisub 式音频盒：波形 + 时间标尺 + 红/蓝标记拖拽 + 播放头打点（`[` / `]`）
- 📝 字幕列表行内编辑、多格式读写（SRT / VTT / ASS）、格式自动嗅探
- ⌨️ 完整快捷键体系、撤销/重做、草稿自动保存
- 📦 **单文件独立版** `subtitle-editor-standalone.html`：全部 JS/CSS 与 wasm/worker/字体内嵌于一个 HTML，双击即可使用（`file://` 协议、无需服务器、不联网）

```bash
cd tools/subtitle-editor
python3 -m http.server 8631   # 模块版（开发形态）
# 或直接双击 subtitle-editor-standalone.html

npm test                      # 运行测试（node --test，120 项）
node build-standalone.mjs     # 重新生成单文件版
```

详见 [tools/subtitle-editor/README.md](tools/subtitle-editor/README.md)。

## 处理流程

```
[原始音频/视频]
    │
    ▼
Stage 1: 人声分离 — UVR (BS-RoFormer) / Spleeter / Open-Unmix
    │ → vocals.wav + accompaniment.wav
    ▼
Stage 0: 宏观静音切块 (Plan 0) — 长音频 (>3min) 自动在 >2s 静音处分治
    │ → List[AudioChunk] (每个 chunk 独立走完整 Pipeline)
    ▼
Stage 1.5: 音频预处理 — 频谱门降噪 + 突发噪声抑制 (可选)
    │
    ▼
Stage 2: VAD 检测 — Silero VAD (默认)
    │ + Plan 1: ffmpeg silencedetect 并行 VAD (ThreadPoolExecutor)
    │ + Plan 2: 三方法边界融合 (Silero+ffmpeg+RMS, 10ms网格, 2/3共识)
    │ → List[SpeechSegment]
    ▼
Stage 3: 片段合并 — 合并/段内预切分 (Plan 3) + 自适应填充
    │ → List[SpeechSegment]
    ▼
Stage 3.5: 说话人分离 — 87维声学特征 + 凝聚聚类 + 文本降级
    │ + pyannote 全局说话人聚类 (speaker-diarization-3.1)
    │ + speechbrain ECAPA-TDNN 嵌入 (可选)
    │ → speaker_id per segment
    ▼
Stage 4: ASR 识别 — faster-whisper / whisper.cpp / FunASR / WhisperX
    │ + 全局语言检测 (完整音频一次运行)
    │ + 幻觉过滤 + 段内去重 + 回退转录
    │ → List[TranscriptionSegment] + List[WordTimestamp]
    ▼
Stage 4.5: 边界精修 (Plan 4) — 词级时间戳 + 三帧能量斜率双向调校
    │ + 滑动窗口冗余 ASR (BoundaryReASR) + LLM 语义仲裁 (BoundaryArbitration)
    │ + 可选: LLM 说话人角色标注
    ▼
Stage 5: 时间轴映射 + 字幕构建 — pysubs2
    │ → List[SubtitleEvent]
    ▼
物理优先中间层 (Phase 0-3):
    ├── 全局坐标系映射 — CoordinateMapper 统一碎片化时间轴
    ├── 物理时间线 IR — 承载分离后音频、VAD 证据、ASR 词分配
    ├── 词级物理分配 — allocate_words + repair_late_words
    ├── 物理覆盖审计 — audit_physical_coverage 检测遮盖空白
    └── 物理字幕分箱 — build_physical_subtitle_bins 自动分箱
    ▼
后处理优化层 (执行顺序经精心设计):
    ├── 0. 事件级说话人融合 — 全局集合聚类 + 局部嵌入精修 (SpeakerFusion)
    ├── 1. Plan 6: 帧级无缝衔接 — 非句尾字幕扩展 end time 到下一句 start
    ├── 2. Plan 5: 三层级联语义合并（边界变动的模块）
    │   gap <200ms → 规则强制合并
    │   gap 200-600ms → 本地 NLP 语义判断 (sentence-transformers)
    │   gap 600-1200ms → 云端 LLM 裁决 (DeepSeek/OpenAI 等)
    │   gap >1200ms → 强制不合并
    ├── 3. Plan 7: 声学标尺校验 — ffmpeg 骨架为物理基准，最终关卡
    │   + 方向感知查询 (禁止跨静音延长)
    │   + 置信度门控 (保留可靠 ASR 边界)
    │   + 诊断报告 (健康评分/逐决策审计日志)
    └── 4. (可选) LLM 后处理优化 — Agent Loop 优化器 (最多3轮)
    │
    ▼
[SRT / VTT / ASS 字幕文件]
```

## 架构概览

### 核心模块

| 模块 | 路径 | 职责 |
|------|------|------|
| `pipeline.py` | `vocal_subtitle/pipeline.py` | 管道入口（组件化后的薄编排层，实际调度在 `application/`） |
| `application/` | `vocal_subtitle/application/` | 编排层：preflight 预检、run 生命周期、契约协调、阶段/分块/流式执行、运行报告（17 模块） |
| `config/` | `vocal_subtitle/config/` | YAML 配置管理 + dataclass 定义 + 覆盖解析 |
| `contracts/` | `vocal_subtitle/contracts/` | 跨层契约（证据/决策/投影的稳定接口，9 模块） |
| `streaming.py` | `vocal_subtitle/streaming.py` | 流式处理架构（滑动窗口 + 模块降级映射） |
| `macro_chunker.py` | `vocal_subtitle/macro_chunker.py` | 宏观静音切块 (Plan 0) |
| `acoustic/` | `vocal_subtitle/acoustic/` | 声学骨架 + 声学校验（含 Plan 7 方向感知门控） |
| `quality/` | `vocal_subtitle/quality/` | 质量门禁与黄金集校准（10 模块） |
| `governance/` | `vocal_subtitle/governance/` | 发布治理（配对/默认值/发布检查） |
| `reporting/` | `vocal_subtitle/reporting/` | 运行报告与诊断聚合（8 模块） |
| `session_manager.py` | `vocal_subtitle/utils/session_manager.py` | 会话管理 (hash 目录 + 去重 + 输出命名) |
| `feedback/` | `vocal_subtitle/feedback/` | 自适应反馈学习引擎 (Phase 5) — 差异分析、参数学习、健康度评分、音频指纹、Shadow Mode、匿名化与分层采样 |
| `physical/` | `vocal_subtitle/physical/` | 物理优先中间层 (Phase 0-3) — 全局坐标系、物理时间线 IR、词级分配、覆盖审计、决策 IR/投影 |

### 引擎层

| 阶段 | 引擎 | 后端 | 备注 |
|------|------|------|------|
| 人声分离 | UVR | audio-separator (ONNX) | 默认，BS-RoFormer |
| | Spleeter | TensorFlow | Python < 3.12 可用 |
| | Open-Unmix | PyTorch | 低资源场景 |
| VAD | Silero VAD | PyTorch (~1.5MB) | 默认，神经网络 |
| | WebRTC VAD | 信号处理 | 轻量，无 ML 依赖 |
| | TEN VAD | 能量阈值 | 纯回退方案 |
| | ffmpeg silencedetect | ffmpeg subprocess | Plan 1 并行运行 |
| ASR | faster-whisper | CTranslate2 | 默认，支持 GPU/CPU |
| | whisper.cpp | CLI subprocess | 低内存环境 |
| | FunASR | PyTorch | 中文优化，WebUI 自动准备 |
| | WhisperX | PyTorch | 词级时间戳增强 |

### ASR 边界优化子系统 (Plan 4 扩展)

| 模块 | 路径 | 职责 |
|------|------|------|
| `boundary_refiner.py` | `vocal_subtitle/asr/boundary_refiner.py` | 词级时间戳 + 三帧能量斜率双向调校 (~528 行) |
| `boundary_confidence.py` | `vocal_subtitle/asr/boundary_confidence.py` | 边界置信度评估 (5维度评分, ~400 行) |
| `boundary_reasr.py` | `vocal_subtitle/asr/boundary_reasr.py` | 滑动窗口冗余 ASR (3窗口并行识别, ~483 行) |
| `boundary_arbitration.py` | `vocal_subtitle/asr/boundary_arbitration.py` | LLM 语义仲裁器 (争议词归属 + 时间轴, ~681 行) |
| `text_normalizer.py` | `vocal_subtitle/asr/text_normalizer.py` | 文本后处理标准化 (~152 行) |
| `global_transcriber.py` | `vocal_subtitle/asr/global_transcriber.py` | 全局语言检测 + 完整音频转录 (~272 行) |
| `hallucination.py` | `vocal_subtitle/asr/hallucination.py` | 幻觉过滤（训练模板短语检测, ~253 行) |
| `local_recovery.py` | `vocal_subtitle/asr/local_recovery.py` | 局部回退转录恢复 (~434 行) |
| `funasr_manager.py` | `vocal_subtitle/asr/funasr_manager.py` | FunASR 引擎自动安装 + 模型准备 (~205 行) |

### 优化方案层

| 方案 | 功能 | 默认 | 依赖 |
|------|------|------|------|
| Plan 0 | 宏观静音切块 | ✅ | ffmpeg |
| Plan 1 | ffmpeg 并行 VAD | ✅ | ffmpeg |
| Plan 2 | 三方法边界融合 | ❌ | Silero + ffmpeg + RMS |
| Plan 3 | 段内静音预切分 | ✅ | 规则引擎 |
| Plan 4 | ASR 边界双向精修 + 冗余识别 + LLM仲裁 | ✅ | ASR word_timestamps + LLM API |
| Plan 5 | LLM 语义合并 | ✅ (级联) | sentence-transformers + LLM API |
| Plan 6 | 帧级无缝衔接 | ✅ | 规则引擎 |
| Plan 7 | 方向感知声学校验门控 + 诊断报告 | ✅ | ffmpeg |

## 场景模板

| 模板 | 适用场景 | 特点 |
|------|----------|------|
| `default` | 通用 | 平衡配置，适合大多数场景 |
| `podcast` | 播客/访谈 (1-3小时) | 语言=zh，启用角色标注，较高合并阈值 |
| `education` | 教学/演讲/TED | 慢语速 (600ms 最小静音)，长字幕 (24 CJK字符) |
| `variety_show` | 综艺/直播 | 快节奏 (300ms 静音)，噪声抑制，快速字幕切换 |
| `music_live` | 音乐现场/LiveHouse | 最低 VAD 阈值 (0.3)，噪声抑制，宽松 snap 距离 |

模板文件位于 `configs/` 目录，可通过 `--profile` 参数选择，也可基于模板自定义。

### 自定义模板

```bash
# 复制默认模板
cp configs/default.yaml configs/my_custom.yaml

# 编辑参数后使用
vocal-subtitle run input.mp3 --config configs/my_custom.yaml
```

## 运行模式

### 离线模式 (默认)

完整 Pipeline，所有 7 个 Plan 可用，支持宏观切块、三方法融合、批量 LLM 合并。

离线字幕默认使用 `production + risk_only` 复核链：分段主候选先经过 EvidenceDecision 和物理投影，再进入后处理。异质副引擎缺失、语言不匹配、超时或失败时保留主候选，并记录 `production_path=segmented_fallback`；需要回归对比时可在配置中显式设置 `shadow_mode: true` 和 `authoritative_mode: false`。

发布前使用黄金集门禁：

```bash
python scripts/run_golden_quality_gate.py --input golden-report.json --ci
```

门禁报告覆盖幻觉误保留、真实对白误删除、物理越界、跨静音、`unresolved/split/drop`、decision trace 和 raw-event bypass。真实黄金集数据不随默认安装下载。

### 流式模式

滑动窗口处理，自动降级全局依赖模块：

- ✅ Plan 3 (段内预切分)、Plan 4 (边界精修)、Plan 6 (帧无缝衔接) — 窗口内运行
- ⚠️ Plan 1 (ffmpeg VAD)、Plan 5 (LLM 合并) — 降级运行
- ❌ Plan 0 (宏观切块)、Plan 2 (三方法融合)、Plan 7 (声学标尺) — 不可用

配置方式：设置 `pipeline.mode: "streaming"` 或在配置 YAML 中指定流式参数。

### 降级模式

通过 `degradation.mode` 控制应对异常：

| 模式 | 行为 |
|------|------|
| `full` | 所有模块按配置运行 |
| `degraded` | 关闭 LLM 调用，仅本地规则 |
| `minimal` | 仅 VAD + ASR + 规则合并 |

## LLM 优化（可选）

默认使用 **DeepSeek**（降低成本），兼容所有 OpenAI 兼容协议 API。

### 支持的 API 提供商

11 个预设 Provider + 自定义：DeepSeek、OpenAI、Anthropic、Google Gemini、智谱 GLM、阿里百炼 Qwen、腾讯混元、Moonshot Kimi、MiniMax、硅基流动 SiliconFlow、Ollama (本地)。

### 配置方式

```bash
# 设置 API Key
export DEEPSEEK_API_KEY="sk-..."

# 或使用 OpenAI 兼容 API
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.deepseek.com"  # 或任何兼容 API

# CLI 启用 LLM 优化
vocal-subtitle run input.mp3 -o output.srt --llm-optimize
```

### LLM 功能模块

| 功能 | 用途 | 需要 API |
|------|------|----------|
| `llm_optimize` | 字幕后处理优化（修正错字、优化断句） | 是 |
| `merge_decision.llm_tier` | 语义合并裁决（600-1200ms 间隙区间） | 是 (cascading 模式) |
| `speaker_role` | 说话人角色标注（主持人/嘉宾/旁白） | 是 |

### LLM 语义合并（三层级联）

```
gap < 200ms    → 快路径：规则强制合并 (<1ms, 纯 CPU)
gap 200-600ms  → 本地 NLP：sentence-transformers (~30ms, CPU, 零成本)
gap 600-1200ms → 云端 LLM：API 调用裁决 (~1s)
gap > 1200ms   → 硬规则：强制不合并
API 不可用      → 自动降级到规则模式
```

## 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| OS | Ubuntu 20.04+ / macOS 12+ | Ubuntu 22.04 LTS |
| Python | 3.10 | 3.11+ |
| RAM | 8GB | 16GB+ |
| GPU (可选) | NVIDIA GTX 1060 6GB | NVIDIA RTX 3060+ 8GB+ |
| 磁盘 | 5GB (含模型缓存) | 20GB+ |
| ffmpeg | 4.4+ | 6.0+ |

## 项目结构

```
Vocal_Subtitle/
├── vocal_subtitle/                 # 核心包（~61,000 行）
│   ├── pipeline.py                 # 管道入口（组件化后的薄编排层）
│   ├── application/                # 编排层：preflight、run 生命周期、契约协调、
│   │                               #   阶段/分块/流式执行、离线生产链、运行报告 (17 模块)
│   ├── config/                     # YAML 配置 + dataclass 定义 + 覆盖解析
│   ├── contracts/                  # 跨层契约（证据/决策/投影稳定接口, 9 模块）
│   ├── separation/                 # Stage 1: 人声分离 (UVR / Spleeter / Open-Unmix)
│   ├── vad/                        # Stage 2: VAD (Silero/WebRTC/TEN/ffmpeg + 三方法边界融合)
│   ├── merging/                    # Stage 3: 片段合并 + Plan 5/6 语义合并与帧级衔接
│   ├── diarization/                # Stage 3.5: 说话人分离 (嵌入/聚类/角色标注/pyannote)
│   ├── asr/                        # Stage 4: 多引擎 ASR + 边界精修 + 证据/复核调度 (37 模块)
│   ├── acoustic/                   # 声学骨架 + Plan 7 方向感知声学校验门控
│   ├── physical/                   # Phase 0-3 物理中间层：坐标系/时间线 IR/词分配/
│   │                               #   覆盖审计/决策 IR/投影
│   ├── mapping/                    # Stage 5: 时间轴映射 + 字幕构建 + 最终化
│   ├── quality/                    # 质量门禁 + 黄金集校准 (10 模块)
│   ├── governance/                 # 发布治理（引擎配对/生产默认值）
│   ├── reporting/                  # 运行报告与诊断聚合 (8 模块)
│   ├── feedback/                   # Phase 5 自适应反馈学习（含匿名化/分层采样）
│   ├── cli_commands/               # CLI 子命令实现
│   ├── utils/                      # 缓存/会话/任务历史/进度/GPU 检测/日志
│   └── webui/                      # FastAPI Web GUI + 前端工作区 (23 模块)
├── tools/subtitle-editor/          # 字幕打轴工作台（独立零构建网页编辑器）
├── llm_subtitle_optimizer/         # LLM 字幕优化独立包 (~1,200 行)
├── configs/                        # YAML 场景模板 (5 个)
├── scripts/                        # 工具脚本（基准/门禁/复核模型下载/发布检查, 22 个）
├── tests/                          # 测试套件 (126 文件, ~20,500 行)
├── docs/                           # 技术文档 + ADR + 设计/实施报告
├── main.py / main_gui.py           # CLI / GUI 入口
├── install.sh                      # 一键安装脚本
└── pyproject.toml                  # 项目元数据 + 依赖定义
```

**代码规模统计** (核心模块):

| 子系统 | 行数 (约) |
|--------|------|
| `asr/` (37 文件) | 10,054 |
| `application/` (17 文件) | 5,988 |
| `mapping/` | 5,350 |
| `feedback/` | 5,289 |
| `diarization/` | 4,967 |
| `physical/` | 4,837 |
| `webui/` (23 文件) | 4,623 |
| `utils/` | 3,670 |
| `quality/` (10 文件) | 3,156 |
| `merging/` | 2,393 |
| `acoustic/` | 1,278 |
| `governance/` | 1,316 |
| `reporting/` | 1,153 |
| `config/` | 1,506 |
| `contracts/` | 1,004 |
| `cli_commands/` | 950 |
| `vad/` | 1,264 |
| `separation/` | 685 |
| 顶层模块 (pipeline/cli/streaming 等) | 2,015 |
| **核心包总计** | **~61,000** |
| `tests/` (126 文件) | ~20,500 |
| `tools/subtitle-editor/` | ~5,700 |
| `llm_subtitle_optimizer/` | ~1,200 |
| **项目总计** | **~88,000** |

## 缓存架构

三层渐进缓存，最大化重复处理效率：

| 层级 | 存储引擎 | Key | 内容 |
|------|---------|-----|------|
| L1: 分离缓存 | diskcache | 文件哈希 + 引擎 + 模型 | 人声/伴奏 WAV 文件 |
| L2: 转录缓存 | diskcache | 片段参数 + 模型 + 语言 | 逐片段 ASR 结果 |
| L3: 任务历史 | SQLite | 任务 ID | 完整 Pipeline 结果 + 事件日志 |
| 附加: 持久化文件 | 文件系统 | 任务 ID | 字幕/音频产出文件 (TTL 管理) |

```python
from vocal_subtitle.utils.cache_manager import CacheManager
from vocal_subtitle.utils.task_history import TaskHistoryManager

# 管理分离和转录缓存
cache = CacheManager()
cache.get_info()      # 查看缓存统计
cache.clear("asr")    # 清除指定阶段缓存

# 管理任务历史
history = TaskHistoryManager()
tasks = history.list(limit=20, offset=0)
history.delete(task_id)
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 带覆盖率
pytest --cov=vocal_subtitle --cov-report=html

# 代码格式检查
ruff check .
ruff format --check .

# 类型检查
mypy vocal_subtitle/
```

### 运行 Benchmark

```bash
# 单次性能测试
python scripts/benchmark.py --input test/中文多人员测试音频.wav --repeat 3

# 批量 Benchmark (6 个场景)
python scripts/run_benchmarks.py

# 字幕时间轴对比
python scripts/compare_timeline.py --auto output.srt --ground-truth test/中文多人员测试音频字幕.ass
```

## 协议

本项目采用 [MIT](LICENSE) 协议。所有第三方依赖均为 MIT / Apache 2.0 / BSD 类协议。

详见 [NOTICE](NOTICE)。

## AI 辅助开发说明

本项目由 `yuukimasato` 主导开发和维护。开发过程中使用了包括 Claude、GPT 等多种 AI 大模型辅助进行代码分析、方案设计、实现和测试；AI 工具不作为本项目的作者、独立贡献者或维护者。最终的技术决策、代码审查、测试验证、发布和维护责任均由项目作者承担。

提交历史中可能保留 AI 工具生成的协作元数据，这些信息仅用于记录开发过程，不代表 AI 服务或其提供方对本项目的署名、背书或权利主张。

## 致谢

- [BS-RoFormer](https://github.com/Anjok07/ultimatevocalremovergui) — UVR 人声分离引擎
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 加速语音识别
- [Silero VAD](https://github.com/snakers4/silero-vad) — 神经网络语音检测
- [Spleeter](https://github.com/deezer/spleeter) — Deezer 人声分离
- [Open-Unmix](https://github.com/sigsep/open-unmix) — PyTorch 音源分离
- [pysubs2](https://github.com/tkarabela/pysubs2) — 字幕格式处理
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) — Transformer 推理加速
- [FunASR](https://github.com/modelscope/FunASR) — 中文语音识别
- [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) — LLM 优化模块来源
