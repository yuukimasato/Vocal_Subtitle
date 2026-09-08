# 实验注册表

**版本**: experiment-registry-v1
**更新日期**: 2026-08-02
**依据**: [ADR-003](../adr/003-feedback-shadow-consumption.md), [ENGINE_LIFECYCLE.md](ENGINE_LIFECYCLE.md)

## 概述

所有实验性引擎、模型、配置组合必须在影响默认输出前在此注册。每个实验记录其预期收益、已知风险、启用范围和撤回条件。

## 模板

```yaml
experiment_id: "exp-{date}-{slug}"
name: "人类可读名称"
category: engine | model | quantization | config | composite
status: proposed | shadow | review | enabled | rolled_back
engines: ["engine_name"]
models: ["model_name"]
languages: ["zh", "en", ...]
expected_benefit: "预期收益描述"
known_risks: ["风险1", "风险2"]
enable_scope: "shadow_only" | "opt_in" | "language_zh" | "all"
withdraw_conditions:
  - "D1/D3 regression on key scenario"
  - "user complaint rate increase"
validation_samples: ["D1-sample-001", "D3-sample-003"]
owner: "researcher_name"
created: "2026-08-02"
```

## 当前注册实验

### EXP-001: Qwen3-ASR 复核 (Shadow → Review)

```yaml
experiment_id: "exp-20260802-qwen-review"
name: "Qwen3-ASR 作为复核引擎从 Shadow 进入 Review"
category: engine
status: shadow
engines: ["qwen-asr"]
models: ["Qwen3-ASR-1.7B"]
languages: ["zh", "en"]
expected_benefit: "为 EvidenceDecision 提供异质引擎的文本候选，提高召回率和文本准确性"
known_risks:
  - "GPU 内存额外占用 ~4GB"
  - "处理时间增加 1.5-2x"
  - "可能因 Qwen 错误导致替换退化"
enable_scope: "opt_in"
withdraw_conditions:
  - "D1 对照运行中文/英文覆盖率下降 > 5%"
  - "替换文本错误率 > 分段主候选"
validation_samples: ["D1-培训测试-双人", "D1-中文多人", "D1-英文多人"]
owner: "yuukimasato"
created: "2026-08-02"
```

### EXP-002: ForcedAligner 词级时间戳

```yaml
experiment_id: "exp-20260802-forced-aligner"
name: "Qwen3-ForcedAligner 提供词级时间戳精修"
category: engine
status: shadow
engines: ["qwen-forced-aligner"]
models: ["Qwen3-ForcedAligner-0.6B"]
languages: ["zh", "en"]
expected_benefit: "改进词级时间戳精度，减少时间 MAE"
known_risks:
  - "GPU 内存额外占用 ~2GB"
  - "仅在 ASR 文本准确时有效，文本错误时可能引入更差的时间戳"
  - "处理时间增加 0.5-1x"
enable_scope: "shadow_only"
withdraw_conditions:
  - "时间 MAE 无显著改善或恶化"
validation_samples: ["D1-中文多人", "D1-英文多人"]
owner: "yuukimasato"
created: "2026-08-02"
```

### EXP-003: SED 非语音检测

```yaml
experiment_id: "exp-20260802-sed-non-speech"
name: "AST-AudioSet SED 检测非语音区域"
category: engine
status: shadow
engines: ["ast-audioset"]
models: ["MIT/ast-finetuned-audioset-10-10-0.4593"]
languages: ["all"]
expected_benefit: "识别音乐、噪声等非语音区域，辅助 EvidenceDecision 的 drop 决策"
known_risks:
  - "可能过度标记（false positive），导致正确的语音被 drop"
  - "GPU 内存额外占用 ~1GB"
enable_scope: "shadow_only"
withdraw_conditions:
  - "D4 过度检测率 > 15%"
  - "导致有效的语音字幕被错误 drop"
validation_samples: ["D4-音乐现场", "D4-高噪声"]
owner: "yuukimasato"
created: "2026-08-02"
```

### EXP-004: 三方法 VAD 融合

```yaml
experiment_id: "exp-20260802-vad-fusion"
name: "Silero + ffmpeg + RMS 三方法 VAD 边界融合"
category: config
status: shadow
engines: ["silero-vad", "ffmpeg-vad", "rms-detector"]
models: []
languages: ["all"]
expected_benefit: "更精确的 VAD 边界，减少 ASR 漏识和过识别"
known_risks:
  - "三方法结果不一致时 2/3 多数决可能选择次优边界"
  - "CPU 开销增加"
enable_scope: "shadow_only"
withdraw_conditions:
  - "D1 对照运行 VAD 边界精度无改善"
  - "引入额外的语音片段碎片化"
validation_samples: ["D1-培训测试-双人"]
owner: "yuukimasato"
created: "2026-08-02"
```

### EXP-005: LLM 后处理优化

```yaml
experiment_id: "exp-20260802-llm-optimize"
name: "LLM 字幕后处理优化（修正错字、优化断句）"
category: composite
status: proposed
engines: ["llm-api"]
models: ["deepseek-v4-pro", "gpt-4o"]
languages: ["zh", "en"]
expected_benefit: "修正 ASR 常见错误（同音字、标点），优化断句可读性"
known_risks:
  - "可能改变语义（无意翻译、改写）"
  - "API 成本"
  - "增加处理延迟"
  - "不同 LLM 产出质量不一致"
enable_scope: "opt_in"
withdraw_conditions:
  - "语义改变率 > 5%"
  - "用户报告不需要的文本修改"
validation_samples: ["D1-培训测试-双人", "D1-英文多人"]
owner: "yuukimasato"
created: "2026-08-02"
```

### EXP-006: 噪声抑制预处理

```yaml
experiment_id: "exp-20260802-noise-reduction"
name: "频谱门降噪 + 突发噪声抑制预处理"
category: config
status: proposed
engines: ["spectral_gate"]
models: []
languages: ["all"]
expected_benefit: "降低背景噪声对 ASR 的影响"
known_risks:
  - "过度降噪可能损伤语音信号"
  - "纯净录音场景下降质"
enable_scope: "opt_in"
withdraw_conditions:
  - "D4 纯净场景质量下降"
validation_samples: ["D4-高噪声", "D4-咖啡厅"]
owner: "yuukimasato"
created: "2026-08-02"
```

## 实验生命周期状态机

```
proposed ──(approve)──► shadow ──(pass validation)──► review ──(human sign-off)──► enabled
                            │                            │                           │
                            ▼                            ▼                           ▼
                       (regression)               (regression)                (regression)
                            │                            │                           │
                            └────────────────────────────┴───────────────────────────┘
                                                         │
                                                         ▼
                                                    rolled_back
```

## 实验对照运行要求

每次实验变更必须进行对照运行：

```bash
# 基线对照
vocal-subtitle run input.wav -o baseline.srt --config configs/default.yaml

# 实验对照
vocal-subtitle run input.wav -o experiment.srt --config configs/default.yaml \
    --override evidence_review.qwen_enabled=true

# 对比报告
python scripts/compare_timeline.py --auto experiment.srt --ground-truth baseline.srt
```

对照报告至少包含：
- 受益场景和受益程度
- 退化场景和退化程度
- 不确定项和需要更多样本的领域
