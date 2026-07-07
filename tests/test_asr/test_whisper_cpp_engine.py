"""测试 whisper.cpp 引擎"""

from vocal_subtitle.asr.whisper_cpp_engine import WhisperCppEngine


class TestWhisperCppEngine:
    """whisper.cpp 引擎单元测试"""

    def test_engine_name(self):
        engine = WhisperCppEngine()
        assert engine.name == "whisper-cpp"

    def test_model_name(self):
        engine = WhisperCppEngine(model="small")
        assert engine.model_name == "small"

    def test_default_parameters(self):
        engine = WhisperCppEngine()
        assert engine._n_threads == 4
        assert engine._language is None

    def test_custom_parameters(self):
        engine = WhisperCppEngine(
            model="medium",
            n_threads=8,
            language="zh",
            whisper_cpp_bin="/usr/local/bin/whisper-cli",
        )
        assert engine.model_name == "medium"
        assert engine._n_threads == 8
        assert engine._language == "zh"
        assert engine._bin == "/usr/local/bin/whisper-cli"

    def test_parse_timestamp(self):
        engine = WhisperCppEngine()
        result = engine._parse_timestamp("00:01:30,500")
        assert result == 90.5

    def test_parse_timestamp_zero(self):
        engine = WhisperCppEngine()
        result = engine._parse_timestamp("00:00:00,000")
        assert result == 0.0

    def test_parse_timestamp_hour(self):
        engine = WhisperCppEngine()
        result = engine._parse_timestamp("01:00:00,000")
        assert result == 3600.0

    def test_repr(self):
        engine = WhisperCppEngine(model="tiny")
        rep = repr(engine)
        assert "WhisperCppEngine" in rep
        assert "tiny" in rep

    def test_model_not_loaded_initially(self):
        engine = WhisperCppEngine()
        assert engine._model_path is None
