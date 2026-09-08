# 引擎生命周期与能力治理

**版本**: engine-lifecycle-v1
**更新日期**: 2026-08-02
**依据**: [ADR-001](../adr/001-unified-decision-export.md), [ADR-002](../adr/002-physical-boundary-priority.md)

## 1. 引擎生命周期状态

每个引擎在运行报告中标记以下状态之一：

| 状态 | 含义 | 对输出的影响 | 条件 |
|------|------|-------------|------|
| `unavailable` | 引擎未安装或依赖缺失 | 无影响（不参与） | 对应 optional-dependency 未安装 |
| `model_missing` | 引擎可用但模型未下载 | 无影响（不参与） | 依赖已安装，权重文件缺失 |
| `ready_shadow` | 已就绪但仅影子运行 | 不影响默认输出，产生影子结果用于对比 | 模型已下载，配置为 shadow |
| `ready_review` | 已就绪用于复核 | 可为 EvidenceDecision 提供附加证据 | 模型已下载，启用复核但不权威 |
| `ready_default` | 已启用为默认引擎 | 直接影响默认输出 | 模型已下载，配置为默认 |

### 状态转换路径

```
unavailable ──(install deps)──► model_missing
model_missing ──(download model)──► ready_shadow
ready_shadow ──(enable review, pass shadow validation)──► ready_review
ready_review ──(pass regression test + human check)──► ready_default
ready_default ──(regression detected)──► ready_shadow  (回退)
```

## 2. 引擎注册表

### 人声分离

| 引擎 | 默认状态 | 模型 | 资源需求 | 语言 | 降级策略 |
|------|----------|------|----------|------|----------|
| UVR (BS-RoFormer) | `ready_default` | bs_roformer_ep_317 | CPU/GPU, ~1GB RAM | 通用 | 降级到 Open-Unmix |
| Spleeter | `unavailable` (Python < 3.12) | 2stems/4stems/5stems | TensorFlow, ~2GB RAM | 通用 | N/A |
| Open-Unmix | `ready_shadow` | umxhq | PyTorch, ~1GB RAM | 通用 | N/A |
| skip (none) | `ready_default` (skip_separation=true) | N/A | 0 | 通用 | 输入已是人声 |

### VAD 检测

| 引擎 | 默认状态 | 模型 | 资源需求 | 降级策略 |
|------|----------|------|----------|----------|
| Silero VAD | `ready_default` | silero_vad.onnx (~1.5MB) | CPU, 极低 | 降级到 WebRTC VAD |
| WebRTC VAD | `ready_shadow` | 无 (信号处理) | CPU, 极低 | 降级到 TEN VAD |
| TEN VAD | `ready_shadow` | 无 (能量阈值) | CPU, 极低 | 最终回退 |
| ffmpeg silencedetect | `ready_default` (与 Silero 并行) | 无 (ffmpeg subprocess) | CPU, 低 | 单独使用精度不足 |

### ASR 识别

| 引擎 | 默认状态 | 模型 | 资源需求 | 语言适配 | 降级策略 |
|------|----------|------|----------|----------|----------|
| faster-whisper | `ready_default` | large-v3 (CTranslate2) | GPU 推荐, CPU 可用 | 多语言 (99+) | 降级到 tiny 模型 |
| FunASR | `ready_shadow` | paraformer-zh | GPU 推荐, ~2GB | 中文优化 | 降级到 faster-whisper |
| Qwen3-ASR | `ready_shadow` | Qwen3-ASR-1.7B/0.6B | GPU 推荐, ~4GB | 多语言 | 降级到 faster-whisper |
| whisper.cpp | `ready_shadow` | ggml-medium.bin | CPU, 低内存 | 多语言 | 降级到 tiny 模型 |
| WhisperX | `ready_shadow` | (依赖 faster-whisper) | GPU 推荐 | 多语言 | 降级到 faster-whisper |

### 复核引擎（Review）

| 引擎 | 默认状态 | 模型 | 资源需求 | 参与条件 |
|------|----------|------|----------|----------|
| Global ASR Evidence | `ready_default` (shadow_mode) | faster-whisper | 同 ASR | 默认影子，不直接影响 |
| Context Re-ASR | `unavailable` | 同 ASR 引擎 | 同 ASR | 实验阶段，需显式启用 |
| Qwen3-ASR Review | `model_missing` | Qwen3-ASR-1.7B | GPU ~4GB | 需下载模型并启用 |
| ForcedAligner | `model_missing` | Qwen3-ForcedAligner-0.6B | GPU ~2GB | 需下载模型并启用 |
| SED | `model_missing` | AST-AudioSet | GPU ~1GB | 需下载模型并启用 |
| Semantic Review | `unavailable` | 无 | CPU | 实验阶段 |

### 说话人分离

| 引擎 | 默认状态 | 模型 | 资源需求 | 降级策略 |
|------|----------|------|----------|----------|
| Agglomerative + ECAPA | `ready_default` | speechbrain/ecapa | CPU, ~500MB | 降级到纯文本聚类 |
| pyannote 全局聚类 | `ready_shadow` | speaker-diarization-3.1 | GPU 推荐, ~2GB | 保留 agglomerative 结果 |
| LLM 角色标注 | `ready_shadow` | 云端 API | 网络, API Key | 降级到 SPEAKER_XX 标签 |

## 3. 引擎可用性快照

每次运行开始前生成引擎可用性快照，写入 `cache/reports/{run_id}/engine_availability.json`：

```json
{
  "run_id": "run-offline-20260802-a1b2c3d4-1690972800000",
  "timestamp": "2026-08-02T12:00:00Z",
  "host": {"cpu_count": 8, "ram_gb": 16, "gpu": "NVIDIA RTX 3060", "vram_gb": 12},
  "engines": {
    "separation_uvr": {"status": "ready_default", "model_path": "~/.cache/...", "model_hash": "sha256:..."},
    "vad_silero": {"status": "ready_default", "model_path": "~/.cache/...", "model_hash": "sha256:..."},
    "asr_faster_whisper": {"status": "ready_default", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
    "asr_funasr": {"status": "model_missing", "reason": "Model not found at ~/.cache/..."},
    "asr_qwen": {"status": "ready_shadow", "model": "Qwen3-ASR-1.7B", "device": "cuda"},
    "review_qwen": {"status": "ready_shadow", "model": "Qwen3-ASR-1.7B"},
    "review_forced_aligner": {"status": "unavailable", "reason": "qwen-asr dependency not installed"},
    "review_sed": {"status": "unavailable", "reason": "transformers dependency not installed"},
    "diarization_speechbrain": {"status": "ready_default", "model_path": "~/.cache/...", "model_hash": "sha256:..."}
  }
}
```

## 4. 默认/实验配置分离

### 默认生产配置 (quality-first)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `pipeline.mode` | `offline` | 离线批量模式 |
| `degradation.mode` | `full` | 所有可用模块运行 |
| `vad.engine` | `silero` | Silero VAD |
| `vad.ffmpeg_enabled` | `true` | Plan 1 并行检测 |
| `fusion.enabled` | `false` | 三方法融合（实验） |
| `asr.engine` | `auto` | 自动选择 |
| `asr.global_asr.evidence_enabled` | `true` | shadow 模式 |
| `evidence_review.shadow_mode` | `true` | 不直接影响默认 |
| `evidence_review.authoritative_mode` | `false` | 不权威 |
| `evidence_review.context_reasr_enabled` | `false` | 实验 |
| `evidence_review.qwen_enabled` | `false` | 实验 |
| `evidence_review.forced_aligner_enabled` | `false` | 实验 |
| `evidence_review.sed_enabled` | `false` | 实验 |
| `diarization.enabled` | `true` | 说话人分离 |
| `boundary_refinement.enabled` | `true` | Plan 4 |
| `merge_decision.llm_tier` | `cascading` | 三层级联 |
| `acoustic_validation.enabled` | `true` | Plan 7 |
| `llm_optimize.enabled` | `false` | LLM 后处理 |
| `noise_reduction.enabled` | `false` | 噪声抑制 |

### 实验配置清单

以下配置项标记为实验，启用前需经过 shadow 验证：

| 配置项 | 预期收益 | 已知风险 | 启用条件 |
|--------|----------|----------|----------|
| `fusion.enabled` | 更精确的 VAD 边界 | 三方法结果不一致可能降低精度 | D1 对照运行无退化 |
| `evidence_review.qwen_enabled` | 额外的文本候选提高召回 | 增加 GPU 内存和时间 2-3x | shadow 对比无明显退化 |
| `evidence_review.authoritative_mode` | review 结果影响默认输出 | 可能因 review 错误导致退化 | D1/D3 对照通过 |
| `evidence_review.context_reasr_enabled` | 边界窗口精细识别 | 增加处理时间 | 边界争议窗口命中率 > 60% |
| `evidence_review.forced_aligner_enabled` | 词级时间戳更精确 | 只在有可靠文本候选时有用 | 时间 MAE 显著改善 |
| `evidence_review.sed_enabled` | 识别非语音区域（音乐、噪声） | 可能过度标记 | D4 无过度检测 |
| `fusion.enabled` | 融合 VAD 边界 | GPU 使用增加 | D1 对照无退化 |
| `llm_optimize.enabled` | LLM 后处理优化文本 | 可能改变语义、增加成本 | 仅在用户显式启用 |

## 5. 降级行为矩阵

| 失败场景 | 降级路径 | production_path | 用户提示 |
|----------|----------|-----------------|----------|
| 分离引擎不可用 | 若 skip_separation=true → 跳过; 否则 → failed | `separation_unavailable` | "分离引擎不可用，请使用已分离的人声文件" |
| Silero VAD 不可用 | → WebRTC VAD | `vad_degraded_to_webrtc` | "Silero VAD 不可用，已降级到 WebRTC" |
| VAD 全部不可用 | → failed | N/A | "无可用的 VAD 引擎" |
| 主 ASR 不可用 | → 尝试备选引擎 | `asr_fallback_to_{engine}` | "主 ASR 不可用，切换到 {engine}" |
| 全部 ASR 不可用 | → failed | N/A | "无可用的 ASR 引擎" |
| FunASR 模型缺失 | → faster-whisper | `segmented_fallback` | "FunASR 模型未下载，使用 faster-whisper" |
| Qwen 超时 | → 忽略 review | `review_timeout` | "Qwen 复核超时，继续生成基础字幕" |
| Global ASR 失败 | → 仅分段结果 | `segmented_fallback` | "全局 ASR 失败，使用分段结果" |
| Evidence Review 失败 | → 仅分段结果 | `segmented_fallback` | "复核模块异常，使用主候选" |
| DecisionProjector 失败 | → 直接投影（降级） | `projector_degraded` | "投影器异常，使用降级投影" |
| 后处理失败 | → 原始字幕（无后处理） | `postprocess_degraded` | "后处理异常，输出原始字幕" |
| LLM API 超时 | → 降级到规则模式 | `llm_timeout_rule_fallback` | "LLM 不可用，使用规则合并" |
| 内存不足 | → 切换到 CPU 推理 | `resource_degraded_to_cpu` | "GPU 内存不足，切换 CPU（较慢）" |
| 磁盘空间不足 | → failed | N/A | "磁盘空间不足" |

### 资源与超时边界

| 约束 | 默认值 | 超时后的行为 |
|------|--------|-------------|
| 单引擎超时 | `degradation.per_module_timeout: 60s` | 跳过该引擎，使用降级路径 |
| ffmpeg 超时 | `degradation.ffmpeg_timeout: 30s` | 跳过 ffmpeg VAD，使用 Silero 独立结果 |
| LLM API 超时 | `degradation.llm_api_timeout: 15s` | 跳过 LLM 调用，降级到规则 |
| 长音频分块 | `macro_chunking.target_chunk_duration: 60s` | 自动切分为独立 Chunk |
| Web 任务恢复 | WebUI 重启后 `fixup_stale_running_tasks()` | 标记残留任务为 failed |
