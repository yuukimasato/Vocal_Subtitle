"""Stage 5: 时间轴映射与字幕输出模块

负责将 ASR 分段识别结果映射回全局时间轴，并构建标准格式字幕。
"""

from .subtitle_builder import SubtitleBuilder, SubtitleRule
from .time_mapper import TimeMapper

__all__ = [
    "TimeMapper",
    "SubtitleBuilder",
    "SubtitleRule",
]
