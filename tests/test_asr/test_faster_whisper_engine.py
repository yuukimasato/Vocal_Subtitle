"""测试 faster-whisper 引擎"""

from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
from vocal_subtitle.asr.faster_whisper_engine import FasterWhisperEngine


class TestFasterWhisperEngine:
    """faster-whisper 引擎单元测试"""

    def test_engine_name(self):
        engine = FasterWhisperEngine()
        assert engine.name == "faster-whisper"

    def test_model_name(self):
        engine = FasterWhisperEngine(model="large-v3")
        assert engine.model_name == "large-v3"

    def test_default_parameters(self):
        engine = FasterWhisperEngine()
        assert engine._beam_size == 5
        assert engine._word_timestamps is True
        assert engine._condition_on_previous_text is False
        assert engine._vad_filter is False

    def test_custom_parameters(self):
        engine = FasterWhisperEngine(
            model="medium",
            device="cpu",
            compute_type="int8",
            beam_size=3,
            word_timestamps=False,
            condition_on_previous_text=True,
            vad_filter=True,
        )
        assert engine.model_name == "medium"
        assert engine._device == "cpu"
        assert engine._compute_type == "int8"
        assert engine._beam_size == 3
        assert engine._word_timestamps is False
        assert engine._condition_on_previous_text is True
        assert engine._vad_filter is True

    def test_repr(self):
        engine = FasterWhisperEngine(model="tiny")
        rep = repr(engine)
        assert "FasterWhisperEngine" in rep
        assert "tiny" in rep

    def test_model_not_loaded_initially(self):
        engine = FasterWhisperEngine()
        assert engine._model is None


class TestTranscriptionSegment:
    """TranscriptionSegment 数据结构测试"""

    def test_duration(self):
        seg = TranscriptionSegment(text="测试", start=0.0, end=2.5)
        assert seg.duration == 2.5

    def test_with_words(self):
        words = [
            WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.95),
            WordTimestamp(word="World", start=0.6, end=1.0, confidence=0.92),
        ]
        seg = TranscriptionSegment(
            text="Hello World", start=0.0, end=1.0, words=words
        )
        assert len(seg.words) == 2
        assert seg.words[0].word == "Hello"
        assert seg.words[0].confidence == 0.95

    def test_repr(self):
        seg = TranscriptionSegment(
            text="这是一段测试文本用于验证", start=1.0, end=3.0
        )
        rep = repr(seg)
        assert "TranscriptionSegment" in rep
        assert "1.000" in rep


class TestWordTimestamp:
    """WordTimestamp 数据结构测试"""

    def test_basic(self):
        w = WordTimestamp(word="测试", start=0.5, end=0.8, confidence=0.9)
        assert w.word == "测试"
        assert w.start == 0.5
        assert w.end == 0.8
        assert w.confidence == 0.9

    def test_default_confidence(self):
        w = WordTimestamp(word="A", start=0.0, end=0.1)
        assert w.confidence == 1.0

    def test_repr(self):
        w = WordTimestamp(word="Hello", start=1.0, end=1.5)
        rep = repr(w)
        assert "Hello" in rep
        assert "1.000" in rep
        assert "1.500" in rep
