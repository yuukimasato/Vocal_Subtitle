"""Vocal Subtitle — 人声分离 + 字幕生成全链路工具

构建从原始音频到 SRT/VTT/ASS 字幕的完整处理管道。

核心模块:
- separation: 人声分离引擎 (UVR-BS-RoFormer / Open-Unmix / Spleeter)
- vad: 语音活动检测 (Silero VAD / TEN VAD / WebRTC VAD)
- merging: 片段合并策略
- asr: 语音识别引擎 (faster-whisper / whisper.cpp / FunASR)
- mapping: 时间轴映射与字幕构建
- pipeline: 管道编排器
- config: YAML 配置管理
- utils: 工具函数 (音频处理 / 缓存 / GPU检测 / 日志 / 模型加载)

Usage:
    >>> from vocal_subtitle import Pipeline
    >>> from vocal_subtitle.config import ConfigLoader
    >>>
    >>> config = ConfigLoader().load_profile("podcast")
    >>> pipeline = Pipeline(config)
    >>> result = pipeline.run("input.mp3", "output.srt")
"""

# ------------------------------------------------------------------
# ★ 关键：包初始化时强制离线模式
#
# huggingface_hub / transformers 在首次导入时将
# os.environ 缓存为模块级常量（HF_HUB_OFFLINE / _is_offline_mode），
# 后续修改 os.environ 不会生效。
# 因此必须在包的最早加载点设置这些环境变量，确保默认优先使用本地缓存。
# ------------------------------------------------------------------
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from .pipeline import Pipeline, PipelineStats
from .config import ConfigLoader, PipelineConfig

__all__ = [
    "Pipeline",
    "PipelineStats",
    "ConfigLoader",
    "PipelineConfig",
]

__version__ = "0.2.0"
__author__ = "vocal-subtitle contributors"
__license__ = "MIT"
