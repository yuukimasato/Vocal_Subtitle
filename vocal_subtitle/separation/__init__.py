"""Stage 1: 人声分离模块

提供多种人声分离引擎抽象接口和实现。

引擎列表:
- UVREngine: Ultimate Vocal Remover (audio-separator + BS-RoFormer), MIT, 默认推荐
- OpenUnmixEngine: Open-Unmix (UMX), MIT, 轻量备选
- SpleeterEngine: Deezer Spleeter, MIT, 仅 Python < 3.12
"""

from .base import LicenseInfo, SeparationEngine, SeparationResult
from .openunmix_engine import OpenUnmixEngine
from .spleeter_engine import SpleeterEngine
from .uvr_engine import UVREngine

__all__ = [
    "SeparationEngine",
    "SeparationResult",
    "LicenseInfo",
    "UVREngine",
    "SpleeterEngine",
    "OpenUnmixEngine",
]
