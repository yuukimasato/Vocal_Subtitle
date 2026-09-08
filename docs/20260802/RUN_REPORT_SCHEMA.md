# 统一运行报告 Schema

**版本**: run-report-v1
**更新日期**: 2026-08-02

## 概述

每次离线生产任务完成后生成统一运行报告，包含从输入到输出的完整可观测信息。报告持久化到 `cache/reports/{run_id}/run_report.json`。

## Schema 定义

```json
{
  "$schema": "run-report-v1",

  "run_id": "run-offline-20260802-a1b2c3d4-1690972800000",
  "task_id": "offline-20260802-a1b2c3d4",
  "created_at": "2026-08-02T12:00:00.000Z",

  "input": {
    "path": "/path/to/input.mp3",
    "file_hash": "sha256:abcdef1234567890...",
    "file_size_bytes": 1048576,
    "duration_seconds": 123.45,
    "sample_rate": 44100,
    "channels": 2,
    "format": "mp3"
  },

  "config_snapshot": {
    "profile": "default",
    "config_hash": "sha256:...",
    "config_version": "default-v1",
    "overrides": {},
    "snapshot_path": "cache/reports/{run_id}/config_snapshot.yaml"
  },

  "engine_availability": {
    "separation": {"engine": "uvr", "model": "bs_roformer", "status": "ready_default", "device": "cpu"},
    "vad": {"engine": "silero", "status": "ready_default"},
    "asr_primary": {"engine": "faster-whisper", "model": "large-v3", "status": "ready_default", "device": "cuda"},
    "asr_secondary": {"engine": "funasr", "model": "paraformer-zh", "status": "model_missing"},
    "qwen": {"engine": "qwen-asr", "model": "Qwen3-ASR-1.7B", "status": "ready_shadow"},
    "forced_aligner": {"engine": "qwen-forced-aligner", "model": "Qwen3-ForcedAligner-0.6B", "status": "unavailable"},
    "sed": {"engine": "ast-audioset", "model": "MIT/ast-finetuned-audioset", "status": "unavailable"},
    "diarization": {"engine": "agglomerative", "embedding_model": "speechbrain/ecapa", "status": "ready_default"}
  },

  "pipeline_path": {
    "mode": "offline",
    "production_path": "quality_first",
    "route_version": "asr-route-v1",
    "quality_gate_version": "asr-quality-v1",
    "review_policy_version": "review-policy-v1",
    "risk_policy_version": "risk-policy-v1",
    "decision_policy_version": "decision-policy-v1",
    "evidence_schema_version": "evidence-v1",
    "golden_quality_gate_version": "golden-quality-v1"
  },

  "stages": {
    "separation": {"status": "completed", "duration_seconds": 12.3, "engine": "uvr"},
    "macro_chunk": {"status": "skipped", "reason": "duration < 180s"},
    "vad": {"status": "completed", "duration_seconds": 2.1, "segment_count": 45},
    "ffmpeg_vad": {"status": "completed", "duration_seconds": 1.2, "segment_count": 42},
    "merging": {"status": "completed", "duration_seconds": 0.5, "merged_segment_count": 38},
    "diarization": {"status": "completed", "duration_seconds": 8.7, "raw_speaker_count": 3, "canonical_speaker_count": 2},
    "asr_primary": {"status": "completed", "duration_seconds": 45.2, "engine": "faster-whisper", "model": "large-v3", "segment_count": 38},
    "evidence_review": {"status": "completed", "duration_seconds": 0.0, "windows_reviewed": 0},
    "decision": {"status": "completed", "decision_count": 42, "actions": {"keep": 40, "split": 1, "drop": 1}},
    "projection": {"status": "completed", "event_count": 41, "physical_violation_count": 0, "cross_silence_count": 0, "raw_event_bypass_count": 0},
    "postprocess": {"status": "completed", "duration_seconds": 3.4},
    "export": {"status": "completed", "formats": ["srt", "vtt", "ass"]}
  },

  "output": {
    "subtitle_count": 41,
    "speaker_count": 2,
    "detected_language": "zh",
    "language_probability": 0.95,
    "total_duration_seconds": 120.5,
    "files": {
      "srt": "cache/reports/{run_id}/output/subtitle.srt",
      "vtt": "cache/reports/{run_id}/output/subtitle.vtt",
      "ass": "cache/reports/{run_id}/output/subtitle.ass"
    }
  },

  "quality": {
    "status": "pass",
    "diagnostics": {},
    "coverage_audit": {
      "total_voice_duration": 95.2,
      "covered_duration": 93.1,
      "coverage_ratio": 0.978,
      "over_allocated_segments": 0,
      "under_allocated_segments": 2,
      "gap_segments": 1
    },
    "acoustic_report": {
      "cross_silence_events": 0,
      "boundary_violations": 0,
      "snap_corrections": 5,
      "health_score": 0.92
    }
  },

  "degradation": {
    "overall_mode": "full",
    "events": [],
    "fallback_category": "",
    "fallback_reason": ""
  },

  "errors": [],
  "warnings": [
    {"stage": "asr_secondary", "message": "FunASR model not found, continuing with primary only"}
  ],

  "timing": {
    "total_wall_seconds": 78.5,
    "stage_timings": {
      "separation": 12.3, "vad": 3.3, "merging": 0.5,
      "diarization": 8.7, "asr": 45.2, "postprocess": 3.4,
      "export": 0.5
    }
  }
}
```

## 状态字段枚举

### 任务状态 (`status`)

| 值 | 含义 |
|----|------|
| `pending` | 已创建，等待执行 |
| `preflight` | 预检中（模型可用性、资源检查） |
| `running` | 执行中 |
| `degraded_completed` | 降级完成（部分引擎不可用但主链完成） |
| `completed` | 正常完成 |
| `failed` | 执行失败 |
| `cancelled` | 用户取消 |

### 阶段状态 (`stages.{stage}.status`)

| 值 | 含义 |
|----|------|
| `completed` | 正常完成 |
| `degraded` | 降级运行（使用了回退方案） |
| `skipped` | 因条件不满足而跳过 |
| `failed` | 执行失败 |
| `unavailable` | 引擎不可用 |

### 引擎状态 (`engine_availability.{engine}.status`)

| 值 | 含义 |
|----|------|
| `unavailable` | 未安装或不可用 |
| `model_missing` | 引擎可用但模型未下载 |
| `ready_shadow` | 可用但仅影子模式 |
| `ready_review` | 可用于复核但不影响默认输出 |
| `ready_default` | 已启用为默认引擎 |

### 生产路径 (`production_path`)

| 值 | 含义 |
|----|------|
| `quality_first` | 完整复核链（segmented primary + evidence review + decision） |
| `segmented_fallback` | 降级：仅使用分段主候选 |
| `global_fallback` | 降级：使用全局转录 |
| `minimal` | 最小路径：VAD + ASR + 规则合并 |
| `review_disabled` | 复核已禁用 |
| `empty_input` | 输入无有效语音内容 |

### 决策动作 (`stages.decision.actions`)

| 值 | 含义 |
|----|------|
| `keep` | 保留候选 |
| `replace` | 用证据替换候选 |
| `split` | 拆分候选 |
| `drop` | 删除候选 |
| `unresolved` | 证据不足，保守保留 |

## 报告位置与命名

```
cache/reports/{run_id}/
├── run_report.json           # 本报告
├── config_snapshot.yaml       # 完整配置快照
├── engine_availability.json   # 引擎可用性详情
├── degradation_log.jsonl      # 降级事件日志（每行一个事件）
├── coverage_audit.json        # 物理覆盖审计详情
├── decision_trace.jsonl       # 逐决策追踪（每行一个 EvidenceDecision）
├── stage_timings.json         # 各阶段详细耗时
├── output/                    # 输出文件
└── diagnostics/               # 诊断制品
```

## 使用方式

### CLI

```bash
vocal-subtitle run input.mp3 -o output.srt
# 报告自动生成: cache/reports/run-{task_id}-{ts}/run_report.json
```

### Python API

```python
result = pipeline.run(input_path, output_path)
stats = result["stats"]  # PipelineStats 对象
report = stats.to_dict()  # 符合本 schema 的 dict
```

### WebUI

任务历史页面按 `run_id` 查询，展示报告摘要；点击详情展开完整 JSON。
