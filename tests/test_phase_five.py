"""Phase five path selection, diagnostics and cache compatibility tests."""

from pathlib import Path

from click.testing import CliRunner

from vocal_subtitle.cli import main
from vocal_subtitle.config import ConfigLoader, PipelineConfig
from vocal_subtitle.pipeline import Pipeline, PipelineStats


def test_offline_default_prefers_global_path():
    config = ConfigLoader().load_profile("default")
    pipeline = Pipeline(config)

    assert config.asr.global_asr.enabled is True
    assert config.asr.global_asr.routing == "auto"
    assert pipeline._resolve_asr_path() == "global"


def test_explicit_segmented_path_is_legacy():
    config = ConfigLoader().merge_with_overrides(
        PipelineConfig(), asr_path="segmented"
    )
    pipeline = Pipeline(config)

    assert pipeline._resolve_asr_path() == "segmented"


def test_streaming_path_does_not_select_global():
    streaming = PipelineConfig(mode="streaming")
    assert Pipeline(streaming)._resolve_asr_path() == "segmented"


def test_global_failure_categories_are_stable():
    assert Pipeline._classify_global_failure(ImportError("WhisperX is not installed")) == (
        "dependency_unavailable"
    )
    assert Pipeline._classify_global_failure(MemoryError()) == "resource_unavailable"
    assert Pipeline._classify_global_failure(
        RuntimeError("global ASR produced a degraded transcript")
    ) == "invalid_result"
    assert Pipeline._classify_global_failure(RuntimeError("model crashed")) == (
        "execution_failed"
    )


def test_degraded_global_cache_entries_are_not_reused():
    class Transcript:
        status = "degraded"
        words = []

    assert not Pipeline._is_usable_global_transcript(Transcript())

    usable = Transcript()
    usable.status = "ok"
    usable.words = [object()]
    assert Pipeline._is_usable_global_transcript(usable)

    partial = Transcript()
    partial.status = "degraded"
    partial.words = [object()]
    assert Pipeline._is_usable_global_transcript(partial)


def test_aggregated_global_diagnostics_preserve_dependency_category():
    diagnostics = {
        "failed_windows": [
            {
                "window_id": "ctx-1",
                "error": "WhisperX is not installed; install the optional dependency",
            }
        ]
    }

    assert Pipeline._classify_global_diagnostics(diagnostics) == (
        "dependency_unavailable"
    )
    assert Pipeline._global_diagnostic_errors(diagnostics) == [
        "WhisperX is not installed; install the optional dependency"
    ]


def test_failure_reason_redacts_credentials():
    reason = Pipeline._safe_failure_reason(
        RuntimeError("api_key=secret-token token=another-secret")
    )

    assert "secret-token" not in reason
    assert "another-secret" not in reason
    assert "***" in reason


def test_pipeline_stats_round_trip_keeps_path_diagnostics(tmp_path):
    stats = PipelineStats(input_path=tmp_path / "audio.wav", duration_seconds=3.0)
    stats.asr_path = "legacy_degraded"
    stats.global_attempted = True
    stats.fallback_category = "dependency_unavailable"
    stats.fallback_reason = "WhisperX is not installed"
    stats.global_diagnostics = {"fallback": True}

    restored = PipelineStats.from_dict(
        Path("audio.wav"), stats.to_dict(), duration_seconds=0.0
    )

    assert restored.asr_path == "legacy_degraded"
    assert restored.global_attempted is True
    assert restored.fallback_category == "dependency_unavailable"
    assert restored.global_diagnostics == {"fallback": True}


def test_full_pipeline_cache_requires_compatible_path():
    pipeline = Pipeline(PipelineConfig())
    pipeline._requested_asr_path = "global"
    assert pipeline._is_usable_full_pipeline_cache(
        {"stats": {"asr_path": "global"}}
    )
    assert not pipeline._is_usable_full_pipeline_cache(
        {"stats": {"asr_path": "legacy"}}
    )
    assert not pipeline._is_usable_full_pipeline_cache({"stats": {}})

    pipeline._requested_asr_path = "segmented"
    assert pipeline._is_usable_full_pipeline_cache({"stats": {}})


def test_cli_exposes_asr_path_option():
    result = CliRunner().invoke(main, ["run", "--help"])

    assert result.exit_code == 0
    assert "--asr-path" in result.output
    assert "segmented" in result.output
