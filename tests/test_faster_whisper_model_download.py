import pytest

from vocal_subtitle.asr import model_download


def test_faster_whisper_cache_probe_requires_model_and_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snapshot = (
        tmp_path / "models--Systran--faster-whisper-small" / "snapshots" / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    assert model_download.is_faster_whisper_model_cached("small") is False
    (snapshot / "model.bin").write_bytes(b"model")
    assert model_download.is_faster_whisper_model_cached("small") is True


def test_faster_whisper_cached_model_path_returns_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snapshot = (
        tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.bin").write_bytes(b"model")

    assert model_download.faster_whisper_cached_model_path("tiny") == snapshot


def test_ensure_faster_whisper_model_reports_download(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    calls = []

    class FakeEngine:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def load_model(self):
            snapshot = (
                tmp_path
                / "models--Systran--faster-whisper-tiny"
                / "snapshots"
                / "revision"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.bin").write_bytes(b"model")

    monkeypatch.setattr(
        "vocal_subtitle.asr.faster_whisper_engine.FasterWhisperEngine",
        FakeEngine,
    )

    result = model_download.ensure_faster_whisper_model("tiny")

    assert result["status"] == "downloaded"
    assert result["model_ref"] == "Systran/faster-whisper-tiny"
    assert calls[0]["compute_type"] == "int8"


def test_unsupported_faster_whisper_model_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        model_download.is_faster_whisper_model_cached("base")
