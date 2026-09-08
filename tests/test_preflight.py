"""Preflight behavior for selected optional engines."""

from types import SimpleNamespace

from vocal_subtitle.application import preflight
from vocal_subtitle.reporting.run_report_schema import EngineStatusEntry
from vocal_subtitle.reporting.engine_availability import EngineAvailabilitySnapshot


class _FakeChecker:
    def __init__(self, config):
        self.config = config

    def check_all(self):
        return EngineAvailabilitySnapshot(
            entries={
                "separation_uvr": EngineStatusEntry(
                    engine="uvr",
                    model="bs_roformer",
                    status="unavailable",
                    reason="audio-separator not installed",
                ),
                "separation_open_unmix": EngineStatusEntry(
                    engine="open-unmix",
                    model="umxhq",
                    status="ready_shadow",
                ),
            }
        )

    def check_critical(self):
        return {"vad_available": True, "asr_available": True}


def _config():
    return SimpleNamespace(
        separation=SimpleNamespace(engine="uvr"),
        degradation=SimpleNamespace(),
    )


def test_preflight_blocks_when_selected_separation_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "_sync_lifecycle_from_availability", lambda config: None)
    monkeypatch.setattr(
        "vocal_subtitle.reporting.engine_availability.EngineAvailabilityChecker",
        _FakeChecker,
    )
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")

    result = preflight.run_preflight_checks(
        input_path, tmp_path / "output.srt", _config()
    )

    separation = next(c for c in result.checks if c.key == "separation_available")
    assert separation.critical is True
    assert separation.passed is False
    assert result.passed is False
    assert preflight._error_category_from_checks(result.failed_critical()) == "separation_unavailable"


def test_preflight_allows_explicit_skip_separation(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "_sync_lifecycle_from_availability", lambda config: None)
    monkeypatch.setattr(
        "vocal_subtitle.reporting.engine_availability.EngineAvailabilityChecker",
        _FakeChecker,
    )
    input_path = tmp_path / "vocals.wav"
    input_path.write_bytes(b"audio")

    result = preflight.run_preflight_checks(
        input_path,
        tmp_path / "output.srt",
        _config(),
        skip_separation=True,
    )

    separation = next(c for c in result.checks if c.key == "separation_available")
    assert separation.critical is False
    assert separation.passed is True
    assert result.passed is True
