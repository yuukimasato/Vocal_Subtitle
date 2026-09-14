"""高精度配置准入与黄金集门禁(高精度方案 Task 7)。

- ``configs/high_precision.yaml`` 显式开启 WhisperX 对齐、global-primary
  路由、时间轴仲裁、受限吸附与风险门控 Context Re-ASR;
- ``configs/default.yaml`` 保持兼容值(新路径默认关闭或维持现状);
- 黄金集门禁包含词级时间覆盖率 ≥95% 与字幕重叠率 = 0。
"""

from pathlib import Path

import yaml

from vocal_subtitle.config import ConfigLoader
from vocal_subtitle.quality.golden_gate import (
    GoldenQualityThresholds,
    evaluate_golden_set,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(name: str) -> dict:
    with open(REPO_ROOT / "configs" / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _pipeline(name: str) -> dict:
    document = _load_yaml(name)
    return document.get("pipeline", document)


def test_high_precision_config_enables_gated_features():
    pipeline = _pipeline("high_precision.yaml")

    asr = pipeline["asr"]
    assert asr["engine"] == "faster-whisper"
    assert asr["model"] == "large-v3"
    assert asr["word_timestamps"] is True
    global_asr = asr["global_asr"]
    assert global_asr["alignment_enabled"] is True
    assert global_asr["routing"] == "global_primary"
    assert global_asr["evidence_enabled"] is True

    evidence_review = pipeline["evidence_review"]
    assert evidence_review["enabled"] is True
    assert evidence_review["context_reasr_enabled"] is True
    assert evidence_review["context_reasr_min_level"] == "high"
    assert evidence_review["global_alternative_enabled"] is True
    # Qwen/FunASR 等辅助证据先保持 shadow / risk_only,不直接准入。
    assert evidence_review["qwen_enabled"] is False
    assert evidence_review["shadow_mode"] is False

    acoustic = pipeline["acoustic_validation"]
    assert acoustic["enabled"] is True
    assert acoustic["timeline_arbitration"] is True
    assert acoustic["max_snap_distance"] <= 0.15
    assert acoustic["max_start_snap_distance"] <= 0.20

    diarization = pipeline["diarization"]
    assert diarization["enabled"] is True
    assert diarization["early_turns"] is True
    # Task 5 修复词表传递后,TTS/高精度配置才允许开启词级后切分。
    assert diarization["word_split_on_turn"] is True


def test_default_config_keeps_compatible_values():
    pipeline = _pipeline("default.yaml")

    assert pipeline["asr"]["global_asr"]["routing"] == "segmented"
    assert pipeline["evidence_review"]["context_reasr_enabled"] is False
    assert pipeline["acoustic_validation"]["timeline_arbitration"] is False
    assert pipeline["diarization"]["word_split_on_turn"] is False


def test_high_precision_profile_loads_into_pipeline_config():
    config = ConfigLoader().load_profile("high_precision")

    assert config.asr.global_asr.alignment_enabled is True
    assert config.asr.global_asr.routing == "global_primary"
    assert config.evidence_review.context_reasr_min_level == "high"
    assert config.evidence_review.global_alternative_enabled is True
    assert config.acoustic_validation.timeline_arbitration is True
    assert config.diarization.word_split_on_turn is True


def test_golden_thresholds_word_coverage_and_overlap():
    # 默认阈值:两项新检查不设限(coverage>=0 恒真、overlap<=1 恒真),
    # 保持基线行为兼容;高精度门禁显式以 0.95/0.0 运行。
    default_thresholds = GoldenQualityThresholds()
    assert default_thresholds.min_word_time_coverage_rate == 0.0
    assert default_thresholds.max_subtitle_overlap_rate == 1.0

    strict_thresholds = GoldenQualityThresholds(
        min_word_time_coverage_rate=0.95,
        max_subtitle_overlap_rate=0.0,
    )
    assert strict_thresholds.min_word_time_coverage_rate == 0.95
    assert strict_thresholds.max_subtitle_overlap_rate == 0.0


def _event(index, start, end, text, words=None):
    payload = {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
        "words": words or [],
    }
    return payload


def _word(start, end):
    return {"word": "x", "start": start, "end": end, "confidence": 0.9}


def test_golden_gate_passes_full_coverage_non_overlapping_events():
    result = evaluate_golden_set([{
        "id": "case-ok",
        "expected_events": [{
            "kind": "speech",
            "start": 1.0,
            "end": 2.0,
            "text": "hello",
        }],
        "predicted_events": [
            _event(1, 1.0, 1.5, "hel", [_word(1.0, 1.2), _word(1.2, 1.5)]),
            _event(2, 1.5, 2.0, "lo", [_word(1.5, 1.8)]),
        ],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }])

    metrics = result["metrics"]
    assert metrics["word_time_coverage_rate"] == 1.0
    assert metrics["subtitle_overlap_rate"] == 0.0
    # 默认阈值下两项新检查不拦截;高精度阈值下同样通过。
    assert result["checks"]["word_time_coverage"] is True
    assert result["checks"]["subtitle_overlap"] is True
    strict = evaluate_golden_set(
        [{
            **{"id": "case-ok"},
            "expected_events": [{
                "kind": "speech", "start": 1.0, "end": 2.0, "text": "hello",
            }],
            "predicted_events": [
                _event(1, 1.0, 1.5, "hel", [_word(1.0, 1.2), _word(1.2, 1.5)]),
                _event(2, 1.5, 2.0, "lo", [_word(1.5, 1.8)]),
            ],
            "diagnostics": {
                "physical_violation_count": 0,
                "cross_silence_count": 0,
                "raw_event_bypass_count": 0,
            },
        }],
        thresholds=GoldenQualityThresholds(
            min_word_time_coverage_rate=0.95,
            max_subtitle_overlap_rate=0.0,
        ),
    )
    assert strict["status"] == "pass"


def test_golden_gate_flags_missing_word_times_and_overlaps():
    case = {
        "id": "case-bad",
        "expected_events": [],
        "predicted_events": [
            _event(1, 1.0, 2.0, "no words", []),
            _event(2, 1.5, 2.5, "overlap", [_word(1.6, 2.4)]),
        ],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }

    result = evaluate_golden_set(
        [case],
        thresholds=GoldenQualityThresholds(
            min_word_time_coverage_rate=0.95,
            max_subtitle_overlap_rate=0.0,
        ),
    )

    metrics = result["metrics"]
    # 无词事件计入未覆盖单元 + 唯一带词事件合法 → 覆盖率 1/2 < 0.95。
    assert metrics["word_time_coverage_rate"] < 0.95
    assert metrics["subtitle_overlap_rate"] > 0.0
    assert result["checks"]["word_time_coverage"] is False
    assert result["checks"]["subtitle_overlap"] is False
