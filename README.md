# Vocal Subtitle — 人声分离 + 字幕生成全链路工具

从原始音视频文件中自动提取人声并生成精准字幕 (SRT / VTT / ASS)。

> **核心约束**：全链路仅使用 MIT / Apache 2.0 / BSD 等宽松协议的工具，确保可自由商用，零合规风险。

## 功能特性

- **全链路自动化**：人声分离 → VAD 检测 → 片段合并 → 说话人分离 → ASR 识别 → 时间轴映射 → 字幕输出
- **多引擎可切换**：
  - 分离：UVR (BS-RoFormer, 默认) / Spleeter / Open-Unmix
  - VAD：Silero VAD (默认) / WebRTC VAD / TEN VAD / ffmpeg silencedetect
  - ASR：faster-whisper (默认) / whisper.cpp / FunASR
- **7 层渐进优化方案 (Plan 0-7)**：
  - Plan 0：宏观静音切块（长音频自动分治）
  - Plan 1：ffmpeg silencedetect 并行 VAD
  - Plan 2：Silero + ffmpeg + RMS 三方法边界融合
  - Plan 3：段内静音预切分（减少 ASR 漏识）
  - Plan 4：ASR 词级时间戳双向边界精修
  - Plan 5：三层级联 LLM 语义合并（快规则 → 本地 NLP → 云端 LLM）
  - Plan 6：帧级无缝衔接（消除字幕闪烁）
  - Plan 7：声学标尺校验（ffmpeg 骨架为物理基准）
- **骨架分段模式 (Skeleton Mode)**：跳过 VAD，以 ffmpeg 声学骨架直接分段处理
- **扬声器分离 (Speaker Diarization)**：声学特征提取 + 凝聚聚类 + LLM 角色标注
- **YAML 配置驱动**：5 种场景模板（通用 / 播客 / 教学 / 综艺 / 音乐现场），24 个 dataclass 配置类，参数可精细调优
- **LLM 后处理优化**：可选的 AI 字幕修正（DeepSeek 默认），支持优化前后对比
- **离线/流式双模式**：离线批量 + 实时流式处理，流式下自动降级全局依赖模块
- **人声/伴奏导出**：分离产出的纯净人声和背景声可单独下载保存
- **批量处理**：支持目录级批量处理，进度条实时显示
- **🖥️ Web GUI 图形界面**：拖拽上传、实时 WebSocket 进度、字幕预览编辑、LLM 对比视图、多格式导出
- **三级缓存架构**：文件级分离缓存 + 片段级转录缓存 + SQLite 持久化任务历史
- **💾 设置持久化**：所有用户参数自动保存，刷新页面后恢复上次配置
- **🧠 自适应反馈学习 (Phase 5)**：上传修订字幕自动学习用户偏好，音频指纹匹配、参数震荡检测、健康度评分、Few-shot 示例缓存、Shadow Mode 安全试错
- **全 MIT 兼容**：所有依赖可自由商用

## 快速开始

### 一键部署 (推荐)

```bash
# 克隆项目
git clone <repo-url>
cd Vocal_Subtitle

# CLI + Web GUI 一键部署
bash install.sh --gui

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

# 3. 安装系统依赖
# Ubuntu/Debian
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

### 下载模型（可选，首次运行自动下载）

```bash
vocal-subtitle download-models --all
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
- 📝 **字幕预览编辑** — 时间轴表格展示，双击直接编辑，实时保存
- 🔍 **LLM 对比视图** — 优化前后并排显示，变更绿色高亮
- 🎤 **分离音频导出** — 下载纯净人声和背景声 WAV 文件
- 📥 **多格式导出** — 一键下载 SRT / VTT / ASS
- 💾 **设置持久化** — 所有参数自动保存到浏览器 localStorage，刷新恢复；服务端持久化到 `cache/persistence_settings.json`
- 📊 **任务历史** — SQLite 持久化，支持分页查看、筛选、删除
- 🔧 **持久化文件管理** — 任务产出 (字幕/音频) 按 TTL 自动管理
- 🧠 **自适应反馈学习** — 上传修订字幕 + 音频，自动分析差异并调整管道参数；支持参数震荡检测、健康度趋势、Shadow Mode 试错、音频指纹匹配

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
    │ → speaker_id per segment
    ▼
Stage 4: ASR 识别 — faster-whisper / whisper.cpp / FunASR
    │ → List[TranscriptionSegment] + List[WordTimestamp]
    ▼
Stage 4.5: 边界精修 (Plan 4) — 词级时间戳 + 三帧能量斜率双向调校
    │ + 滑动窗口冗余 ASR (BoundaryReASR) + LLM 语义仲裁 (BoundaryArbitration)
    │ + 可选: LLM 说话人角色标注
    ▼
Stage 5: 时间轴映射 + 字幕构建 — pysubs2
    │ → List[SubtitleEvent]
    ▼
后处理优化层 (执行顺序经精心设计):
    ├── 0. 事件级说话人聚类 — 全局集合声学聚类（替代段级 diarization）
    ├── 1. Plan 6: 帧级无缝衔接 — 非句尾字幕扩展 end time 到下一句 start
    ├── 2. Plan 5: 三层级联语义合并（边界变动的模块）
    │   gap <200ms → 规则强制合并
    │   gap 200-600ms → 本地 NLP 语义判断 (sentence-transformers)
    │   gap 600-1200ms → 云端 LLM 裁决 (DeepSeek/OpenAI 等)
    │   gap >1200ms → 强制不合并
    ├── 3. Plan 7: 声学标尺校验 — ffmpeg 骨架为物理基准，最终关卡
    │   + 诊断报告 (健康评分/异常事件统计)
    └── 4. (可选) LLM 后处理优化 — DeepSeek agent loop (最多3轮)
    │
    ▼
[SRT / VTT / ASS 字幕文件]
```

## 架构概览

### 核心模块

| 模块 | 路径 | 职责 |
|------|------|------|
| `pipeline.py` | `vocal_subtitle/pipeline.py` | 管道编排器，调度全部阶段，管理数据流 (~3537 行) |
| `config.py` | `vocal_subtitle/config.py` | YAML 配置管理 + 24 个 dataclass 定义 (~1110 行) |
| `pipeline_context.py` | `vocal_subtitle/pipeline_context.py` | 统一数据上下文 (跨模块数据交换) |
| `streaming.py` | `vocal_subtitle/streaming.py` | 流式处理架构（滑动窗口 + 模块降级映射, ~316 行） |
| `macro_chunker.py` | `vocal_subtitle/macro_chunker.py` | 宏观静音切块 (Plan 0, ~345 行) |
| `audio_preprocessor.py` | `vocal_subtitle/audio_preprocessor.py` | 预 VAD 降噪（频谱门 + 突发噪声抑制, ~477 行） |
| `acoustic_validator.py` | `vocal_subtitle/acoustic_validator.py` | 声学标尺校验 + 诊断报告 (Plan 7, ~823 行) |
| `session_manager.py` | `vocal_subtitle/utils/session_manager.py` | 会话管理 (hash 目录 + 去重 + 输出命名) |
| `feedback/` | `vocal_subtitle/feedback/` | 自适应反馈学习引擎 (Phase 5) — 差异分析、参数学习、健康度评分、音频指纹、Shadow Mode 等 10 个模块 (~4184 行) |

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
| | FunASR | PyTorch | 中文优化 |

### ASR 边界优化子系统 (Plan 4 扩展)

| 模块 | 路径 | 职责 |
|------|------|------|
| `boundary_refiner.py` | `vocal_subtitle/asr/boundary_refiner.py` | 词级时间戳 + 三帧能量斜率双向调校 |
| `boundary_confidence.py` | `vocal_subtitle/asr/boundary_confidence.py` | 边界置信度评估 (5维度评分) |
| `boundary_reasr.py` | `vocal_subtitle/asr/boundary_reasr.py` | 滑动窗口冗余 ASR (3窗口并行识别) |
| `boundary_arbitration.py` | `vocal_subtitle/asr/boundary_arbitration.py` | LLM 语义仲裁器 (争议词归属 + 时间轴) |
| `text_normalizer.py` | `vocal_subtitle/asr/text_normalizer.py` | 文本后处理标准化 |

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
| Plan 7 | 声学标尺校验 + 诊断报告 | ✅ | ffmpeg |

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
├── vocal_subtitle/                 # 核心包
│   ├── pipeline.py                 # 管道编排器 (~3537 行)
│   ├── config.py                   # YAML 配置 + 24 个 dataclass 定义 (~1110 行)
│   ├── cli.py                      # Click CLI 命令行入口 (~817 行)
│   ├── pipeline_context.py         # 统一数据上下文
│   ├── streaming.py                # 流式处理架构 (~316 行)
│   ├── macro_chunker.py            # Plan 0: 宏观静音切块 (~345 行)
│   ├── audio_preprocessor.py       # 预 VAD 降噪 (~477 行)
│   ├── acoustic_validator.py       # Plan 7: 声学标尺校验 + 诊断 (~823 行)
│   ├── separation/                 # Stage 1: 人声分离 (3 engines, ~679 行)
│   ├── vad/                        # Stage 2: VAD 检测 (4 engines + fusion, ~1264 行)
│   ├── merging/                    # Stage 3: 片段合并 + LLM 语义合并 (~1829 行)
│   ├── asr/                        # Stage 4: ASR 识别 + 边界优化子系统 (~3054 行)
│   │   ├── faster_whisper_engine.py    # faster-whisper (CTranslate2)
│   │   ├── whisper_cpp_engine.py       # whisper.cpp (subprocess)
│   │   ├── funasr_engine.py            # FunASR (中文优化)
│   │   ├── boundary_refiner.py         # Plan 4: 双向边界精修
│   │   ├── boundary_confidence.py      # 边界置信度评估 (5维度)
│   │   ├── boundary_reasr.py           # 滑动窗口冗余 ASR (3窗口)
│   │   ├── boundary_arbitration.py     # LLM 语义仲裁器
│   │   └── text_normalizer.py          # 文本后处理
│   ├── diarization/                # Stage 3.5: 说话人分离 (~1976 行)
│   │   ├── speaker_embedding.py        # 声学嵌入提取 (SpeechBrain)
│   │   ├── feature_extractor.py        # 87维声学特征提取
│   │   ├── speaker_clusterer.py        # 凝聚聚类 + 文本降级
│   │   └── role_labeler.py             # LLM 角色标注
│   ├── feedback/                   # Phase 5: 自适应反馈学习引擎 (~4184 行)
│   │   ├── aligner.py                  # 自动版与修订版字幕对齐
│   │   ├── diff_analyzer.py            # 差异分析 + 参数归因
│   │   ├── param_learner.py            # 参数学习 + 梯度累积
│   │   ├── audio_fingerprint.py        # 音频指纹提取与匹配
│   │   ├── health_scorer.py            # 健康度评分 (5维度)
│   │   ├── conflict_detector.py        # 参数震荡检测
│   │   ├── few_shot_builder.py         # Few-shot 示例缓存
│   │   ├── impact_estimator.py         # 变更影响预估
│   │   ├── shadow_mode.py              # Shadow Mode 安全试错
│   │   └── user_profile.py             # 用户配置管理
│   ├── mapping/                    # Stage 5: 时间轴映射 + 字幕构建 (~1408 行)
│   │   ├── time_mapper.py              # 时间轴映射 + 去重
│   │   ├── subtitle_builder.py         # pysubs2 字幕输出
│   │   └── end_time_validator.py       # 结束时间校验
│   ├── utils/                      # 工具层 (~2946 行)
│   │   ├── cache_manager.py            # 磁盘缓存管理 (diskcache)
│   │   ├── task_history.py             # SQLite 任务历史
│   │   ├── persistence_manager.py      # 持久化文件管理
│   │   ├── session_manager.py          # 会话管理 (hash 目录 + 去重)
│   │   ├── audio_utils.py              # 音频加载/转换/重采样
│   │   ├── gpu_detector.py             # CUDA/MPS/CPU 自动检测
│   │   ├── file_hasher.py              # SHA256 文件哈希
│   │   ├── progress.py                 # 进度管理器 (CLI + WebSocket 双路)
│   │   ├── model_loader.py             # 模型加载工具
│   │   └── logger.py                   # structlog 结构化日志
│   └── webui/                      # FastAPI Web GUI (~3071 行)
│       ├── app.py                      # FastAPI 工厂
│       ├── api.py                      # REST API (46 端点)
│       ├── websocket.py                # WebSocket 实时通信
│       ├── models.py                   # Pydantic 数据模型
│       ├── cli_runner.py               # GUI 启动入口
│       └── static/                     # 前端 SPA (index.html)
├── llm_subtitle_optimizer/         # LLM 字幕优化独立包 (~1223 行)
│   ├── optimizer.py                   # Agent Loop 优化器 (最多3轮)
│   ├── llm_client.py                  # OpenAI 兼容 API 客户端 (支持任意兼容协议)
│   ├── aligner.py                     # Diff 对齐修复
│   ├── prompts.py                     # 提示词模板
│   ├── text_utils.py                  # 文本工具函数
│   └── prompts/                       # 提示词模板文件
│       ├── subtitle.md
│       └── speaker_role.md
├── configs/                        # YAML 场景模板 (5个)
│   ├── default.yaml                   # 通用 (平衡配置)
│   ├── podcast.yaml                   # 播客/访谈
│   ├── education.yaml                 # 教学/演讲
│   ├── variety_show.yaml              # 综艺/直播
│   └── music_live.yaml                # 音乐现场
├── scripts/                        # 工具脚本
│   ├── benchmark.py                   # 性能基准测试
│   ├── run_benchmarks.py              # 批量 Benchmark 运行器
│   ├── compare_timeline.py            # 字幕时间轴对比 (ASS/SRT)
│   ├── generate_test_fixtures.py      # 测试数据生成
│   ├── install_deps.sh                # 系统依赖安装脚本
│   └── download_speaker_model.sh      # 说话人模型下载
├── tests/                          # 测试套件
│   ├── test_pipeline.py               # 全链路集成测试
│   ├── test_cli.py                    # CLI 端到端测试
│   ├── test_webui.py                  # Web GUI API 测试
│   ├── test_audio_preprocessor.py     # 降噪模块测试
│   ├── test_acoustic_validator.py     # 声学校验测试
│   ├── test_macro_chunker.py          # 宏观切块测试
│   ├── test_streaming.py              # 流式处理测试
│   ├── test_session_manager.py        # 会话管理器测试
│   ├── test_feedback.py               # 反馈学习模块测试
│   ├── test_end_time_fixes.py         # 结束时间修正测试
│   ├── test_separation/               # 分离引擎测试 (3 文件)
│   ├── test_vad/                      # VAD 引擎测试 (5 文件)
│   ├── test_merging/                  # 合并模块测试 (2 文件)
│   ├── test_asr/                      # ASR 引擎测试 (4 文件)
│   ├── test_mapping/                  # 映射模块测试 (2 文件)
│   ├── test_diarization/              # 说话人分离测试 (3 文件)
│   ├── test_utils/                    # 工具模块测试 (3 文件)
│   ├── benchmarks/                    # Benchmark 场景 (6 目录)
│   └── fixtures/                      # 测试数据 (音频/配置/期望输出)
├── docs/                           # 技术文档
│   ├── ARCHITECTURE.md                # 技术架构文档
│   ├── 人声分离字幕工程化方案.md       # 工程化方案详细说明
│   ├── 字幕时间轴精度优化方案.md       # 时间轴精度优化
│   ├── 统一优化方案.md                 # 统一优化方案概览
│   └── ...等                          # 更多中文技术文档
├── main.py                         # CLI 入口
├── main_gui.py                     # GUI 入口
├── ARCHITECTURE.md                 # 架构文档入口 → docs/ARCHITECTURE.md
├── install.sh                      # 一键安装脚本
└── pyproject.toml                  # 项目元数据 + 依赖定义
```

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
