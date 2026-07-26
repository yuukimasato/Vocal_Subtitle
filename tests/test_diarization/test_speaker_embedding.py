"""Speaker embedding loader tests without downloading model weights."""

import sys
import types

from vocal_subtitle.diarization.speaker_embedding import (
    PyannoteEmbeddingEngine,
    is_huggingface_model_cached,
)


def test_pyannote_loader_passes_computed_device(monkeypatch, tmp_path):
    captured = {}

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda value: value
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeInference:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_pyannote_audio = types.ModuleType("pyannote.audio")
    fake_pyannote_audio.Inference = FakeInference
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.model", None)

    engine = PyannoteEmbeddingEngine(cache_dir=tmp_path)
    engine.load_model(model_ref="pyannote/embedding", token="hf_test")

    assert engine.model_loaded
    assert captured["model"] == "pyannote/embedding"
    assert str(captured["device"]) in {"cpu", "cuda"}


def test_huggingface_pipeline_cache_requires_snapshot_config(tmp_path):
    model_ref = "pyannote/speaker-diarization-community-1"
    snapshot = (
        tmp_path
        / "hub"
        / "models--pyannote--speaker-diarization-community-1"
        / "snapshots"
        / "snapshot"
    )
    snapshot.mkdir(parents=True)

    assert not is_huggingface_model_cached(model_ref, tmp_path)
    (snapshot / "config.yaml").write_text("pipeline: test", encoding="utf-8")
    assert is_huggingface_model_cached(model_ref, tmp_path)


def test_pyannote_4_loader_authenticates_model_before_inference(
    monkeypatch, tmp_path
):
    captured = {}

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda value: value
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_ref, **kwargs):
            captured["model_ref"] = model_ref
            captured["model_kwargs"] = kwargs
            return "loaded-model"

    class FakeInference:
        def __init__(self, **kwargs):
            captured["inference_kwargs"] = kwargs

    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Inference = FakeInference
    fake_core = types.ModuleType("pyannote.audio.core")
    fake_model_module = types.ModuleType("pyannote.audio.core.model")
    fake_model_module.Model = FakeModel
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core", fake_core)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.model", fake_model_module)

    engine = PyannoteEmbeddingEngine(cache_dir=tmp_path)
    engine.load_model(model_ref="pyannote/embedding", token="hf_test")

    assert captured["model_ref"] == "pyannote/embedding"
    assert captured["model_kwargs"]["token"] == "hf_test"
    assert captured["inference_kwargs"] == {
        "model": "loaded-model",
        "device": "cpu",
    }
