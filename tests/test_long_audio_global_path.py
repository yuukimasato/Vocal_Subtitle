from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vocal_subtitle.asr.base import ASREngine
from vocal_subtitle.asr.global_transcriber import GlobalTranscriber, GlobalTranscriberConfig


@dataclass
class Word:
    word: str
    start: float
    end: float
    confidence: float = 0.8


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word]


class WindowEngine(ASREngine):
    name = "test-window-engine"
    model_name = "test-window-model"

    def load_model(self):
        return None

    def transcribe(self, audio, sample_rate=10, language=None, **kwargs):
        duration = len(audio) / sample_rate
        if duration > 2.1:
            raise RuntimeError("window unexpectedly exceeded bound")
        end = max(0.1, duration)
        return [Segment("hello", 0.0, end, [Word("hello", 0.0, min(0.5, end))])]


def test_long_audio_is_bounded_and_reports_absolute_window_diagnostics():
    transcriber = GlobalTranscriber(
        WindowEngine(),
        GlobalTranscriberConfig(max_window_duration=2.0, window_overlap=0.5),
    )
    result = transcriber.transcribe(np.zeros(50, dtype=np.float32), 10)

    assert result.diagnostics["window_count"] == 3
    assert all(item["end"] - item["start"] <= 2.0 for item in result.diagnostics["windows"])
    assert result.diagnostics["failed_windows"] == []
    assert result.diagnostics["audio_duration"] == 5.0
    assert all(0.0 <= word.raw_start <= word.raw_end <= 5.0 for word in result.transcript.words)


def test_long_audio_failed_window_is_retained_in_diagnostics():
    class FailingEngine(WindowEngine):
        def transcribe(self, audio, sample_rate=10, language=None, **kwargs):
            if len(audio) / sample_rate > 1.4:
                raise RuntimeError("synthetic failure")
            return super().transcribe(audio, sample_rate, language, **kwargs)

    result = GlobalTranscriber(
        FailingEngine(),
        GlobalTranscriberConfig(max_window_duration=1.5, window_overlap=0.2),
    ).transcribe(np.zeros(30, dtype=np.float32), 10)

    assert result.diagnostics["failed_windows"]
    assert any(item["status"] == "failed" for item in result.diagnostics["windows"])
