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
- governance: 引擎生命周期管理 / 实验注册表 / 发布治理
- reporting: 统一运行报告 / 引擎可用性检查 / 降级日志
- quality: 质量运营 (问题分类 / 场景切片 / 趋势报告)

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

from .config import ConfigLoader, PipelineConfig

# governance public API
from .governance import (
    EngineLifecycle,
    EngineRegistry,
    EngineStatus,
    LifecycleManager,
)
from .pipeline import Pipeline, PipelineStats

# quality public API (key classes only)
from .quality import (
    IssueCategory,
    IssueClassifier,
    IssueSeverity,
    QualityIssue,
)

# reporting public API
from .reporting import (
    DegradationLogger,
    EngineAvailabilityChecker,
    EngineAvailabilitySnapshot,
    RunReport,
    RunReportBuilder,
    StageInfo,
)

__all__ = [
    "Pipeline",
    "PipelineStats",
    "ConfigLoader",
    "PipelineConfig",
    # governance
    "EngineLifecycle",
    "EngineStatus",
    "EngineRegistry",
    "LifecycleManager",
    # reporting
    "RunReport",
    "RunReportBuilder",
    "StageInfo",
    "EngineAvailabilityChecker",
    "EngineAvailabilitySnapshot",
    "DegradationLogger",
    # quality
    "IssueCategory",
    "IssueSeverity",
    "QualityIssue",
    "IssueClassifier",
]

__version__ = "0.3.0"
__author__ = "vocal-subtitle contributors"
__license__ = "MIT"
