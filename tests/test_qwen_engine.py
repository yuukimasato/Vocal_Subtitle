"""Contract tests for the optional Qwen primary ASR adapter."""

import importlib.machinery
from types import ModuleType, SimpleNamespace

import numpy as np

from vocal_subtitle.asr.qwen_engine import (
    QwenASREngine,
    default_qwen_model_path,
    qwen_model_path_ready,
)


def test_qwen_model_path_ready_accepts_nested_snapshot_weights(tmp_path):
    model_dir = tmp_path / "qwen3-asr-1.7b"
    (model_dir / "snapshot").mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "snapshot" / "model.safetensors").write_bytes(b"weights")

    assert qwen_model_path_ready(model_dir)


def test_qwen_engine_loads_and_adapts_fake_runtime(tmp_path, monkeypatch):
    model_dir = tmp_path / "qwen3-asr-1.7b"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    calls = []

    class FakeModel:
        def transcribe(self, audio, language=None):
            calls.append((audio, language))
            return [
                SimpleNamespace(
                    text="你好",
                    start=0.1,
                    end=0.8,
                    language="Chinese",
                    words=[
                        SimpleNamespace(word="你", start=0.1, end=0.4, score=0.9),
                        SimpleNamespace(word="好", start=0.4, end=0.8, score=0.95),
                    ],
                )
            ]

    class FakeQwenModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return FakeModel()

    fake_module = ModuleType("qwen_asr")
    fake_module.__spec__ = importlib.machinery.ModuleSpec("qwen_asr", loader=None)
    fake_module.Qwen3ASRModel = FakeQwenModel
    monkeypatch.setitem(__import__("sys").modules, "qwen_asr", fake_module)

    engine = QwenASREngine(model_dir, device="cpu")
    assert engine.availability()["status"] == "ready"
    segments = engine.transcribe(np.zeros(16000, dtype=np.float32), language="zh")

    assert segments[0].text == "你好"
    assert segments[0].language == "zh"
    assert [(word.word, word.start, word.end) for word in segments[0].words] == [
        ("你", 0.1, 0.4),
        ("好", 0.4, 0.8),
    ]
    assert calls[0] == (str(model_dir), {"device_map": "cpu"})
    assert calls[1][1] == "Chinese"


def test_default_qwen_model_path_uses_shared_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("VOCAL_SUBTITLE_REVIEW_MODEL_DIR", str(tmp_path))
    assert default_qwen_model_path() == tmp_path / "qwen3-asr-1.7b"
