"""LLM Subtitle Optimizer

使用大语言模型(LLM)优化和修正字幕内容，支持 agent loop 自动验证和修正。

核心特性:
- Agent loop: LLM → 验证 → 反馈 → 重试（最多 3 轮）
- 并发批量处理
- 自动对齐修复
- 改动幅度验证（防止过度修改）
- 支持任意 OpenAI 兼容 API

Usage:
    >>> from llm_subtitle_optimizer import SubtitleOptimizer
    >>>
    >>> optimizer = SubtitleOptimizer(model="deepseek-v4-pro")
    >>>
    >>> subtitles = {
    ...     "1": "大家好啊今天呢我们来讲一下机器学习的基础只是",
    ...     "2": "那么它其实就是嗯人工治能的一个重要份支",
    ... }
    >>>
    >>> result = optimizer.optimize(subtitles)
    >>> print(result["1"])
    '大家好，今天我们来学习机器学习的基础知识'
"""

from .optimizer import SubtitleOptimizer
from .llm_client import call_llm, get_llm_client
from .prompts import get_prompt, list_prompts

__all__ = [
    "SubtitleOptimizer",
    "call_llm",
    "get_llm_client",
    "get_prompt",
    "list_prompts",
]

__version__ = "0.2.0"
