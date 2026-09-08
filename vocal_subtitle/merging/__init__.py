"""Stage 3: 片段合并模块

负责 VAD 检测后片段的合并、切分和填充。
"""

from .merge_strategy import MergeConfig, MergeStrategy
from .merge_engine import LLMMergeEngine, MergeDecisionConfig
from .llm_decider import LLMMergeDecider
from .local_decider import LocalMergeDecider
from .semantic_window import SemanticWindowInput, SemanticWindowOutput, build_semantic_windows
from .layout import SUBTITLE_LAYOUT_RULES

__all__ = [
    "MergeStrategy",
    "MergeConfig",
    "LLMMergeEngine",
    "MergeDecisionConfig",
    "LLMMergeDecider",
    "LocalMergeDecider",
    "SemanticWindowInput",
    "SemanticWindowOutput",
    "build_semantic_windows",
    "SUBTITLE_LAYOUT_RULES",
]
