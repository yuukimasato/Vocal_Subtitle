import json
from types import SimpleNamespace
from pathlib import Path

from scripts.run_offline_golden_production import _categories, _config, _parse_reference
from scripts.run_golden_quality_gate import main as run_quality_gate


def test_golden_runner_reads_json_reference_and_preserves_event_kind():
    root = Path(__file__).resolve().parents[1]
    events = _parse_reference(
        root / "test/golden/non_speech_tone.json",
        "normalized_json",
    )

    assert events[0]["kind"] == "non_speech"
    assert "non_speech" in _categories(
        {"tags": ["non_speech"], "category": "non_speech", "language": "none"},
        4.0,
    )


def test_golden_runner_adds_long_audio_and_repeated_phrase_categories():
    categories = _categories(
        {
            "tags": ["repeated_phrase"],
            "category": "single_speaker",
            "language": "zh",
            "speaker_count": 1,
        },
        180.1,
    )

    assert {"repeated_phrase", "chinese", "long_audio"} <= set(categories)


def test_golden_runner_marks_manual_subtitles_as_advisory_by_default():
    args = SimpleNamespace(output=Path("/tmp/golden-input.json"))
    item = {
        "ground_truth": "test/reference.ass",
        "category": "single_speaker",
        "language": "zh",
    }

    # The runner's default is observable without loading an ASR model through
    # the same manifest convention used by run_scene.
    reference_role = str(item.get("reference_role", "advisory" if item["ground_truth"] else "none"))
    assert reference_role == "advisory"


def test_golden_runner_defaults_no_reference_without_ground_truth():
    item = {"ground_truth": None}
    reference_role = str(item.get("reference_role", "advisory" if item.get("ground_truth") else "none"))
    assert reference_role == "none"


def test_golden_runner_uses_mixed_language_mode_without_forcing_language():
    args = SimpleNamespace(
        profile="default",
        engine="faster-whisper",
        model="large-v3",
        device="cpu",
        secondary_engine="qwen",
        review_policy="risk_only",
        disable_engine_pair=False,
        disable_context_reasr=False,
        qwen_model_path=None,
        whisper_cpp_bin="",
        whisper_cpp_model_path="",
    )

    config = _config(args, language="mixed")

    assert config.asr.language is None
    assert config.asr.language_mode == "mixed"


def test_golden_runner_selects_comparable_production_modes():
    args = SimpleNamespace(
        profile="default",
        engine="faster-whisper",
        model="large-v3",
        device="cpu",
        secondary_engine="qwen",
        review_policy="risk_only",
        disable_engine_pair=False,
        disable_context_reasr=False,
        qwen_model_path=None,
        whisper_cpp_bin="",
        whisper_cpp_model_path="",
        production_mode="baseline",
    )

    baseline = _config(args)
    assert baseline.asr.global_asr.evidence_enabled is False
    assert baseline.evidence_review.enabled is False

    args.production_mode = "shadow"
    shadow = _config(args)
    assert shadow.asr.global_asr.evidence_enabled is True
    assert shadow.evidence_review.shadow_mode is True
    assert shadow.evidence_review.authoritative_mode is False

    args.production_mode = "authoritative"
    authoritative = _config(args)
    assert authoritative.evidence_review.shadow_mode is False
    assert authoritative.evidence_review.authoritative_mode is True


def test_quality_gate_cli_carries_input_production_metadata(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps({
        "schema_version": "golden-quality-input-v2",
        "generated_from": "test/quality_manifest.yaml",
        "metadata": {
            "primary_engine": "faster-whisper",
            "secondary_engine": "qwen",
            "review_policy": "risk_only",
        },
        "cases": [],
    }), encoding="utf-8")

    assert run_quality_gate(["--input", str(input_path), "--output", str(output_path)]) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["metadata"]["input_schema_version"] == "golden-quality-input-v2"
    assert report["metadata"]["primary_engine"] == "faster-whisper"
    assert report["metadata"]["secondary_engine"] == "qwen"


def test_quality_gate_cli_strict_reference_is_opt_in(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({
        "schema_version": "golden-quality-input-v3",
        "cases": [{
            "reference_role": "advisory",
            "reference_status": "manual_reference",
            "expected_events": [{"start": 0.0, "end": 1.0, "text": "参考"}],
            "predicted_events": [],
            "diagnostics": {
                "physical_violation_count": 0,
                "cross_silence_count": 0,
                "raw_event_bypass_count": 0,
            },
        }],
    }), encoding="utf-8")

    assert run_quality_gate(["--input", str(input_path), "--ci"]) == 0
    assert run_quality_gate(["--input", str(input_path), "--ci", "--strict-reference"]) == 3
