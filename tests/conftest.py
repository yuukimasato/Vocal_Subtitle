"""pytest 共享 fixtures"""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """临时目录"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_audio_mono() -> np.ndarray:
    """生成 1 秒的 16kHz 单声道测试音频（正弦波）"""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz 纯音
    return audio


@pytest.fixture
def sample_audio_silence() -> np.ndarray:
    """生成 1 秒的静音"""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def sample_audio_speech_silence_mix() -> np.ndarray:
    """生成含语音和静音的测试音频
    0.0-0.5s: 440Hz 正弦波（模拟语音）
    0.5-0.7s: 静音
    0.7-1.0s: 880Hz 正弦波（模拟语音）
    """
    sample_rate = 16000
    total = np.zeros(int(sample_rate * 1.0), dtype=np.float32)

    # 语音段 1
    t1 = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
    total[: len(t1)] = np.sin(2 * np.pi * 440 * t1).astype(np.float32)

    # 语音段 2
    t2 = np.linspace(0, 0.3, int(sample_rate * 0.3), endpoint=False)
    start2 = int(sample_rate * 0.7)
    total[start2 : start2 + len(t2)] = (
        np.sin(2 * np.pi * 880 * t2).astype(np.float32) * 0.5
    )

    return total


@pytest.fixture
def sample_wav_file(temp_dir: Path, sample_audio_mono) -> Path:
    """生成临时 WAV 文件"""
    from vocal_subtitle.utils.audio_utils import AudioUtils

    path = temp_dir / "test_audio.wav"
    AudioUtils.save_audio(sample_audio_mono, path)
    return path


@pytest.fixture
def default_config():
    """默认管道配置"""
    from vocal_subtitle.config import ConfigLoader

    return ConfigLoader().load_profile("default")
