"""Stage 3: 片段合并模块

负责 VAD 检测后片段的合并、切分和填充。
"""

from .merge_strategy import MergeConfig, MergeStrategy

__all__ = [
    "MergeStrategy",
    "MergeConfig",
]
