"""GlobalTranscriber 词级对齐来源(time_source)优先级行为。

- 对齐成功:词时间来自 WhisperX alignment,标记 ``whisperx_alignment``;
- 对齐失败:保留原始 ASR 词时间,标记 ``faster_whisper_word`` 并记录失败;
- 对齐关闭:同原始路径,标记 ``faster_whisper_word``。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vocal_subtitle.asr.base import ASREngine
from vocal_subtitle.asr.global_transcriber import (
    GlobalTranscriber,
    GlobalTranscriberConfig,
)
from vocal_subtitle.config.models import GlobalASRConfig


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
    words: list = field(default_factory=list)


class AligningEngine(ASREngine):
    """transcribe 给出粗粒度词时间,align 给出精修词时间。"""

    name = "fake-align-engine"
    model_name = "fake-align-model"

    def load_model(self):
        return None

    def transcribe(self, audio, sample_rate=10, language=None, **kwargs):
        return [
            Segment(
                "hello world",
                0.0,
                2.0,
                [Word("hello", 0.0, 1.0), Word("world", 1.0, 2.0)],
            )
        ]

    def align(self, audio, *, sample_rate=10, segments=None, language=None):
        return [
            Segment(
                "hello world",
                0.0,
                2.0,
                [Word("hello", 0.10, 0.90), Word("world", 1.05, 1.95)],
            )
        ]


class FailingAlignEngine(AligningEngine):
    def align(self, audio, *, sample_rate=10, segments=None, language=None):
        raise RuntimeError("whisperx alignment model unavailable")


def _word_times(result):
    return {
        word.text: (word.raw_start, word.raw_end)
        for word in result.transcript.words
    }


def test_alignment_applied_uses_alignment_times_and_marks_provenance():
    transcriber = GlobalTranscriber(AligningEngine(), GlobalTranscriberConfig())

    result = transcriber.transcribe(np.zeros(20, dtype=np.float32), 10)

    assert result.diagnostics["alignment_status"] == "applied"
    assert result.diagnostics["alignment_failures"] == []
    assert _word_times(result) == {
        "hello": (0.10, 0.90),
        "world": (1.05, 1.95),
    }
    assert result.transcript.words
    assert all(
        word.metadata.get("time_source") == "whisperx_alignment"
        for word in result.transcript.words
    )


def test_alignment_failure_keeps_asr_words_and_records_provenance():
    transcriber = GlobalTranscriber(FailingAlignEngine(), GlobalTranscriberConfig())

    result = transcriber.transcribe(np.zeros(20, dtype=np.float32), 10)

    failures = result.diagnostics["alignment_failures"]
    assert failures and failures[0]["window_id"] == "window:full"
    assert "whisperx alignment model unavailable" in failures[0]["error"]
    # 原始 ASR 词时间不得被丢弃。
    assert _word_times(result) == {
        "hello": (0.0, 1.0),
        "world": (1.0, 2.0),
    }
    assert all(
        word.metadata.get("time_source") == "faster_whisper_word"
        for word in result.transcript.words
    )


def test_alignment_disabled_keeps_asr_words_and_marks_disabled():
    transcriber = GlobalTranscriber(
        AligningEngine(), GlobalTranscriberConfig(alignment=False)
    )

    result = transcriber.transcribe(np.zeros(20, dtype=np.float32), 10)

    assert result.diagnostics["alignment_status"] == "disabled"
    assert _word_times(result) == {
        "hello": (0.0, 1.0),
        "world": (1.0, 2.0),
    }
    assert all(
        word.metadata.get("time_source") == "faster_whisper_word"
        for word in result.transcript.words
    )


def test_word_time_source_counts_reported_in_diagnostics():
    result = GlobalTranscriber(AligningEngine(), GlobalTranscriberConfig()).transcribe(
        np.zeros(20, dtype=np.float32), 10
    )

    counts = result.diagnostics["word_time_source_counts"]
    assert counts == {"whisperx_alignment": 2}


def test_global_asr_config_keeps_alignment_default_compatible():
    config = GlobalASRConfig()

    # 兼容默认值:与 GlobalTranscriberConfig.alignment=True 的现有行为一致。
    assert config.alignment_enabled is True
