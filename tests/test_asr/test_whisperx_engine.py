"""WhisperX adapter regression tests."""

import numpy as np
import sys
from types import SimpleNamespace

from vocal_subtitle.asr.whisperx_engine import WhisperXEngine


def test_load_model_configures_word_timestamps_at_model_level(monkeypatch):
    calls = {}

    def fake_load_model(*args, **kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(load_model=fake_load_model))

    WhisperXEngine(word_timestamps=True).load_model()

    assert calls["asr_options"] == {"word_timestamps": True}


def test_transcribe_does_not_pass_unsupported_word_timestamp_argument():
    calls = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            calls.update(kwargs)
            return {
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello",
                        "words": [
                            {"word": "hello", "start": 0.1, "end": 0.9}
                        ],
                    }
                ],
            }

    engine = WhisperXEngine(word_timestamps=True)
    engine._model = FakeModel()

    segments = engine.transcribe(np.zeros(16000, dtype=np.float32), language="en")

    assert "word_timestamps" not in calls
    assert len(segments) == 1
    assert len(segments[0].words) == 1
