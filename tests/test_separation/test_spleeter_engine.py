"""测试 Spleeter 人声分离引擎"""

import pytest

from vocal_subtitle.separation.base import LicenseInfo, SeparationResult
from vocal_subtitle.separation.spleeter_engine import SpleeterEngine


class TestSpleeterEngine:
    """Spleeter 引擎单元测试"""

    def test_engine_name(self):
        engine = SpleeterEngine()
        assert engine.name == "spleeter"

    def test_license_info(self):
        engine = SpleeterEngine()
        info = engine.license_info
        assert isinstance(info, LicenseInfo)
        assert info.code_license == "MIT"
        assert info.model_license == "MIT"
        assert "spleeter" in info.source_url.lower()

    def test_default_model_name(self):
        engine = SpleeterEngine()
        assert engine.DEFAULT_MODEL == "2stems"

    def test_repr(self):
        engine = SpleeterEngine()
        rep = repr(engine)
        assert "SpleeterEngine" in rep
        assert "spleeter" in rep
