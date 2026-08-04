"""测试 FunASR 引擎"""

from vocal_subtitle.asr.funasr_engine import FunASREngine
from vocal_subtitle.asr.funasr_manager import DEFAULT_FUNASR_MODEL


class TestFunASREngine:
    """FunASR 引擎单元测试"""

    def test_engine_name(self):
        engine = FunASREngine()
        assert engine.name == "funasr"

    def test_model_name(self):
        engine = FunASREngine()
        assert "paraformer" in engine.model_name.lower()

    def test_generic_model_name_uses_funasr_default(self):
        engine = FunASREngine(model="large-v3")
        assert engine._model_id == DEFAULT_FUNASR_MODEL

    def test_default_parameters(self):
        engine = FunASREngine()
        assert engine._device == "cuda"
        assert engine._ncpu == 4

    def test_custom_parameters(self):
        engine = FunASREngine(
            model="iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            device="cpu",
            ncpu=8,
        )
        assert engine._device == "cpu"
        assert engine._ncpu == 8

    def test_model_name_extraction(self):
        engine = FunASREngine(
            model="iic/speech_custom_model_name"
        )
        assert engine.model_name == "speech_custom_model_name"

    def test_repr(self):
        engine = FunASREngine()
        rep = repr(engine)
        assert "FunASREngine" in rep
        assert "funasr" in rep

    def test_model_not_loaded_initially(self):
        engine = FunASREngine()
        assert engine._model is None
