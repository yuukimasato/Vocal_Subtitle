# 复杂场景下 ASR 识别质量断崖式下降分析报告

**日期**: 2026-07-29
**版本**: v1.0
**分析范围**: Vocal_Subtitle 全链路 Pipeline（VAD → ASR → 时间轴映射）

---

## 一、问题现象总结

| 场景 | 识别质量 | 时间轴准确度 | 典型表现 |
|------|----------|-------------|----------|
| TTS 合成音频 | ★★★★★ 优秀 | 精确 | 静音干净、语速均匀、无背景噪音 |
| 产品评测录音 | ★★☆☆☆ 差 | 偏移 | 语速不均、术语多、半消声室混响 |
| 个人演讲 | ★★★☆☆ 一般 | 局部偏移 | 即兴停顿、口头禅、不均匀音量 |
| 影视剧 | ★☆☆☆☆ 极差 | 严重偏移 | BGM覆盖、多人重叠、环境噪音、音效 |
| 非中文内容 (FunASR) | ☆☆☆☆☆ 乱码 | 无意义 | FunASR 仅支持中文，非中文输出乱码 |

**核心矛盾**：Pipeline 在全链路中大量使用固定阈值和全局参数，假设输入音频具有 TTS 级别的"干净"特征。当遇到真实世界的复杂声学环境时，这些假设全部失效。

---

## 二、根因分析：5 个关键失效点

### 2.1 VAD 检测层：固定阈值导致语音段漏检/误检

**当前状态** ([vad/silero_vad.py](vocal_subtitle/vad/silero_vad.py#L134-L136), [config.py](vocal_subtitle/config.py#L35-L37)):
```python
threshold: float = 0.5              # 固定语音概率阈值
min_speech_duration_ms: int = 150   # 最小语音段 150ms
min_silence_duration_ms: int = 400  # 最小静音段 400ms
```

**失效机制**：
- **背景噪音场景**（影视剧 BGM、街道录音）：Silero VAD 将背景音乐/环境音也判定为语音（概率 > 0.5），产生大量**假阳性语音段**。ASR 在这些段上强行识别，输出噪音文字或幻觉文本，破坏了时间轴。
- **低音量语音场景**（远场录音、轻声细语）：语音概率 < 0.5，Silero 判定为静音，导致**整段语音被丢弃**，字幕出现缺失。
- **快速对话场景**（辩论、群聊）：`min_silence_duration_ms=400` 无法捕捉 < 400ms 的说话人切换间隙，多个说话人的语句被合并为一个 VAD 段，造成时间轴偏移。

**影响程度**：★★★★★（最高，VAD 是后续所有阶段的输入，错误在此层即被放大）

### 2.2 三方法融合默认关闭

**当前状态** ([config.py](vocal_subtitle/config.py#L58-L59)):
```python
fusion.enabled: bool = False  # 三方法边界融合默认关闭
```

**失效机制**：
- BoundaryFusion 通过 Silero + ffmpeg + RMS Energy 三方法投票来提高 VAD 边界的置信度，但**默认关闭**。
- 单靠 Silero VAD，在复杂场景下边界误差可达 ±200ms+，这个误差经 ASR 段时间偏移累积后，会导致明显的字幕时间轴偏移。

**影响程度**：★★★★☆

### 2.3 固定噪声阈值：声学骨架在噪音环境下失效

**当前状态** ([acoustic_validator.py](vocal_subtitle/acoustic_validator.py#L32-L33)):
```python
skeleton_noise_db: float = -40.0   # 骨架提取的噪声阈值，固定 -40dB
skeleton_min_silence: float = 0.1  # 最小静音段 100ms
```

**失效机制**：
- TTS 音频的静音区 RMS 通常在 -60dB ~ -50dB，`-40dB` 能完美区分语音/静音。
- 影视剧/户外录音的背景噪音通常在 -30dB ~ -20dB。此时 `-40dB` 阈值**远低于实际噪音水平**，ffmpeg 将整个音频（包括纯噪音段）都判定为"语音"。
- 结果：声学骨架失效 → `AcousticValidator.validate()` 返回 `"skipped": True, "reason": "skeleton build failed"` → 方案七的物理吸附完全不起作用。

**日志证据**（预期会出现）：
```
WARNING - Failed to build acoustic skeleton, skipping validation
```

**影响程度**：★★★★☆（方案七是整个时间轴的最后防线，它失效意味着没有物理层的兜底修正）

### 2.4 前置降噪默认关闭

**当前状态** ([audio_preprocessor.py](vocal_subtitle/audio_preprocessor.py#L47)):
```python
DenoiseConfig.enabled: bool = False  # 默认关闭
```

**失效机制**：
- 谱减法降噪（spectral_gate）、突发噪音抑制（burst_noise_protection）都是可用的，但默认关闭。
- 嘈杂录音中，关门声、键盘敲击、桌椅移动等 < 200ms 的突发噪音会被 ffmpeg 误检为语音事件，打断正常的语音段。
- 稳态背景噪音（空调、风扇、交通）抬高了全局 RMS，导致 RMS Energy Scan 的语音/静音判定阈值失真。

**影响程度**：★★★☆☆

### 2.5 ASR 分段识别 + 时间偏移累积

**当前状态** ([pipeline.py](vocal_subtitle/pipeline.py#L2056-L2096)):
```python
# 每个 VAD 段独立调用 ASR，语言预检测仅适用于 faster-whisper
for i, seg in enumerate(segments):
    segment_audio = AudioUtils.extract_segment(audio, start_sample, end_sample)
    seg_results = engine.transcribe(segment_audio, sample_rate, language=resolved_language)
```

**失效机制**：
- **上下文断裂**：每个 VAD 段独立识别，ASR 模型无法利用前后文的语义信息。在噪音环境下，缺少上下文使得模型更容易产生幻觉。
- **FunASR 单语言限制**：FunASR 引擎的 `detect_language()` 始终返回 `"zh"`。对于非中文音频，不仅无警告阻止，还会产生乱码中文字幕。
- **时间偏移累积**：VAD 边界的微小误差（±50ms）× N 个段 → 累积偏移。当音频有 100+ 个 VAD 段时，末尾段的绝对时间误差可能达到 5 秒以上。
- **Global ASR 降级路径脆弱**：虽然存在全局转录路径（`_run_global_transcription_path`），但它在 GPU 内存不足、模型加载失败等情况下会静默降级到分段路径。

**影响程度**：★★★★☆

---

## 三、问题链路图

```
复杂场景音频 (噪音/音乐/多人/远场)
        │
        ▼
┌─ 前置降噪: 默认关闭 ─────────────────────────┐
│  噪音原样进入下游                                │
└────────────────────────────────────────────────┘
        │
        ▼
┌─ VAD (Silero): 固定阈值 0.5 ──────────────────┐
│  背景音乐 → 假阳性语音段                          │
│  轻声细语 → 假阴性（被丢弃）                       │
│  快速对话 → 多人合并为一段                         │
└────────────────────────────────────────────────┘
        │
        ▼
┌─ 三方法融合: 默认关闭 ─────────────────────────┐
│  错失了 ffmpeg+RMS 纠正 Silero 错误的机会          │
└────────────────────────────────────────────────┘
        │
        ▼
┌─ ASR 分段识别 ───────────────────────────────┐
│  上下文断裂 → 识别错误增多                         │
│  幻觉文本 → 虚假字幕事件                           │
│  FunASR+非中文 → 乱码                             │
└────────────────────────────────────────────────┘
        │
        ▼
┌─ 声学标尺: 固定 -40dB ─────────────────────────┐
│  噪音场景 → 骨架构建失败 → 物理校验跳过              │
│  时间轴失去最后的兜底保护                           │
└────────────────────────────────────────────────┘
        │
        ▼
   时间轴严重偏移 + 字幕内容错误
```

---

## 四、优化方案

### 方案 A：自适应 VAD 阈值（优先级：★★★★★ 最高）

**目标**：根据音频的噪声特征动态调整 VAD 参数。

**具体改动**：

1. **预处理阶段自动测量噪声基底**（已有基础设施 `noise_profile.py`）：
   - 在 VAD 前调用 `estimate_noise_profile()` 获取 `LocalNoiseProfile`
   - 提取 `noise_db`（估计的本地噪声水平）和 `rms_median`（中位数 RMS 能量）
   - 根据噪声水平自动选择 Silero VAD 的 threshold：
     - 噪声 < -50dB（TTS级）：threshold = 0.5（当前默认）
     - 噪声 -40~-50dB（安静录音）：threshold = 0.4
     - 噪声 -30~-40dB（一般噪音）：threshold = 0.35，同时启用 fusion
     - 噪声 > -30dB（高噪音）：threshold = 0.3，强制启用 fusion + 降噪

2. **根据噪声水平自动调整 ffmpeg 的 noise_db**：
   - 测量到的 noise_db + 6dB 作为 silencedetect 的 noise 参数
   - 例如：测量噪音 = -28dB → silencedetect noise=-22dB

3. **代码实现位置**：在 `_process_chunk_pipeline()` 中，VAD 调用前插入自适应逻辑。

**预期效果**：
- 嘈杂环境下 VAD 召回率提升 20-40%
- 假阳性语音段减少 30-50%

### 方案 B：默认启用三方法融合（优先级：★★★★★ 最高）

**目标**：通过多方法投票提升 VAD 边界精度。

**具体改动**：

1. 将 `FusionConfig.enabled` 默认值从 `False` 改为 `True`
2. 在 `_process_chunk_pipeline()` 中并行运行：
   - Silero VAD（主 VAD）
   - ffmpeg silencedetect（并行）
   - RMS Energy Scan（从音频计算）
3. 三方法投票（10ms 网格，2/3 多数决），取高置信度边界

**代码位置**：
- [vocal_subtitle/config.py:59](vocal_subtitle/config.py#L59) — 修改默认值
- [vocal_subtitle/pipeline.py](vocal_subtitle/pipeline.py#L1900-L1948) — `_run_ffmpeg_vad()` 已存在，需在非 skeleton_mode 下也调用
- [vocal_subtitle/vad/boundary_fusion.py](vocal_subtitle/vad/boundary_fusion.py) — 融合引擎已实现

**预期效果**：
- VAD 边界误差从 ±200ms 降至 ±50ms
- 时间轴偏移显著减少

### 方案 C：自适应声学骨架阈值（优先级：★★★★☆ 高）

**目标**：让声学标尺校验在噪音环境下也能正常工作。

**具体改动**：

1. 在 `AcousticValidator._get_skeleton()` 中，使用 `LocalNoiseProfile` 测量的噪声基底来动态设定 `skeleton_noise_db`：
   ```python
   # 伪代码
   noise_profile = estimate_noise_profile(audio, sample_rate)
   measured_noise_db = noise_profile.intervals[0].noise_db
   adaptive_noise_db = max(measured_noise_db + 6, -50)  # 噪声基底 + 6dB，不低于 -50dB
   adaptive_noise_db = min(adaptive_noise_db, -25)       # 最高不超过 -25dB
   ```

2. 当噪音 > -25dB 时，骨架构建可能完全不可靠。此时应：
   - 标记骨架为 "low_confidence"
   - 放宽 `max_snap_distance`（从 0.25s → 0.5s）
   - 降低 `confidence_threshold`（从 0.6 → 0.4）
   - 在诊断报告中明确标记 "high_noise_degraded"

3. 代码位置：[acoustic_validator.py](vocal_subtitle/acoustic_validator.py#L140-L170)

**预期效果**：
- 嘈杂环境下声学校验不再 100% 跳过
- 骨架覆盖率从 0% 提升到 60-80%

### 方案 D：增强前置降噪并默认启用（优先级：★★★☆☆ 中高）

**目标**：在 VAD 前减少噪音对检测的干扰。

**具体改动**：

1. 将 `DenoiseConfig.enabled` 默认值改为 `True`
2. 噪声水平 < -45dB（安静环境）时自动跳过降噪，避免对纯净音频的副作用
3. 突发噪音保护（`burst_noise_protection`）默认启用
4. 代码位置：[audio_preprocessor.py](vocal_subtitle/audio_preprocessor.py#L47)

**预期效果**：
- 关门声、键盘等瞬时噪音不再打断 VAD 段
- 稳态背景噪音降低 6-12dB

### 方案 E：FunASR 非中文保护 + 多引擎回退（优先级：★★★☆☆ 中）

**目标**：防止非中文音频被 FunASR 错误处理。

**具体改动**：

1. 在 `_run_asr()` 的语言检测阶段，如果引擎是 FunASR 且检测到的语言不是 `zh`：
   - CLI 模式：直接报错退出，提示用户切换引擎
   - WebUI 模式：弹出警告并要求确认，或自动切换到 faster-whisper
2. 实现引擎自动回退链：`funasr → faster-whisper (base)` 作为保底
3. 代码位置：[pipeline.py](vocal_subtitle/pipeline.py#L2112-L2121)

**预期效果**：
- 非中文音频不会产生乱码字幕
- 用户体验改善（不会浪费数分钟等待无效结果）

### 方案 F：场景自适应配置预设（优先级：★★☆☆☆ 中）

**目标**：为用户提供针对不同场景优化的预设配置。

**具体改动**：

在现有 profile 系统上增加以下预设：

```yaml
# 新增 profile: "noisy_environment"
vad:
  engine: silero
  threshold: 0.35          # 降低以捕捉被噪音掩盖的语音
  min_speech_duration_ms: 100
  min_silence_duration_ms: 300
  ffmpeg_enabled: true     # 强制启用

fusion:
  enabled: true            # 强制启用三方法融合

acoustic_validation:
  skeleton_noise_db: -30   # 适配噪音环境
  max_snap_distance: 0.5   # 放大吸附窗口

denoise:
  enabled: true            # 强制启用降噪
  engine: spectral_gate

# 新增 profile: "film_drama"
# (影视剧：BGM、多人、快速对话)
vad:
  threshold: 0.35
  min_silence_duration_ms: 200  # 缩短以捕捉快速对话切换

merging:
  pre_split_threshold: 0.4      # 缩短以在多人切换处切分

# 新增 profile: "live_speech"
# (现场演讲：远场、回声、不均匀音量)
vad:
  threshold: 0.3           # 更敏感（远场音量低）

acoustic_validation:
  skeleton_noise_db: -35   # 中等噪音
```

### 方案 G：时间轴自校准机制（优先级：★★☆☆☆ 长期）

**目标**：在无可靠声学骨架时，通过 ASR 自身的置信度信号做时间轴修正。

**具体改动**：

1. 利用 whisper/faster-whisper 的词级时间戳置信度做加权
2. 对低置信度（confidence < 0.5）的词边界，放宽声学校验的容忍度
3. 生成"时间轴置信度热力图"，标注哪些字幕的时间轴可信、哪些需要人工复核

---

## 五、优先级与实施建议

| 优先级 | 方案 | 改动量 | 风险 | 效果 |
|--------|------|--------|------|------|
| P0 立即 | A: 自适应 VAD 阈值 | 中 (~200行) | 低 | 高 |
| P0 立即 | B: 默认启用三方法融合 | 小 (~20行) | 低 | 高 |
| P1 短期 | C: 自适应声学骨架 | 中 (~150行) | 中 | 高 |
| P1 短期 | D: 启用前置降噪 | 小 (~10行) | 低 | 中 |
| P2 中期 | E: FunASR 非中文保护 | 小 (~50行) | 低 | 中 |
| P2 中期 | F: 场景预设 | 小 (~80行 YAML) | 低 | 中 |
| P3 长期 | G: 时间轴自校准 | 大 (~500行) | 中 | 中 |

**建议实施顺序**：A + B 同时进行（改动独立、互不冲突）→ C → D → E → F → G

---

## 六、验证方法

1. **建立测试基准**：
   - 准备 3-5 个代表性复杂场景音频（影视剧片段、街头采访、会议录音）
   - 手工标注 Ground Truth 字幕（文本 + 时间轴）

2. **量化指标**：
   - WER (Word Error Rate)：文本识别准确率
   - Time F1：时间轴对齐的 F1 分数（边界误差 < 100ms 视为正确）
   - Coverage：ASR 覆盖的语音时长 / 实际语音时长

3. **A/B 对比**：
   - 每个方案实施前后，跑完整的测试集
   - 对比 `diagnostic_report` 中的 `health_score`、`snapped_starts`、`snapped_ends`
