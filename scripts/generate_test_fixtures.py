"""生成测试用音频 fixtures

创建用于单元测试和集成测试的合成音频文件。

输出:
    tests/fixtures/audio/
    ├── sample_1min_speech.wav       # 1分钟模拟语音
    ├── sample_silence_only.wav      # 纯静音
    ├── sample_multi_speaker.wav     # 多人对话（不同频率模拟）
    └── sample_empty.wav             # 极短空文件
"""

import argparse
import struct
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
OUTPUT_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "audio"


def generate_speech_wav(
    output_path: Path,
    duration: float = 60.0,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """生成模拟语音的 WAV 文件

    使用 440Hz 正弦波（模拟元音）配合静音段，模拟真实语音的节奏。
    """
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    # 模拟句子：每 2 秒一个语音片段，间隔 0.5-1.5 秒静音
    speech_dur = 1.5  # 每段语音 1.5 秒
    silence_dur_min = 0.3
    silence_dur_max = 1.0

    pos = 0
    freqs = [220, 330, 440, 550, 660, 880]  # 不同频率模拟不同音节
    freq_idx = 0

    while pos < num_samples:
        # 添加语音段
        speech_samples = int(speech_dur * sample_rate)
        end = min(pos + speech_samples, num_samples)
        t = np.arange(end - pos) / sample_rate
        freq = freqs[freq_idx % len(freqs)]
        audio[pos:end] = 0.5 * np.sin(2 * np.pi * freq * t)
        freq_idx += 1

        # 添加随机"音调"变化
        if end - pos > sample_rate // 4:
            chirp_start = pos + (end - pos) // 3
            chirp_end = pos + 2 * (end - pos) // 3
            chirp_len = chirp_end - chirp_start
            chirp_t = np.arange(chirp_len) / sample_rate
            audio[chirp_start:chirp_end] += (
                0.2 * np.sin(2 * np.pi * freq * 1.5 * chirp_t)
            )

        pos = end

        # 添加静音段
        silence_dur = np.random.uniform(silence_dur_min, silence_dur_max)
        silence_samples = int(silence_dur * sample_rate)
        pos += silence_samples

    # 归一化
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    _write_wav(output_path, audio, sample_rate)
    print(f"  ✓ {output_path.name} ({duration:.0f}s)")


def generate_silence_wav(
    output_path: Path,
    duration: float = 5.0,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """生成纯静音 WAV 文件"""
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32) + 1e-6  # 微小值避免全零
    _write_wav(output_path, audio, sample_rate)
    print(f"  ✓ {output_path.name} ({duration:.0f}s)")


def generate_multi_speaker_wav(
    output_path: Path,
    duration: float = 30.0,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """生成多人对话模拟音频

    使用不同频率和交替模式模拟不同说话人。
    """
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    # 说话人频率
    speakers = {
        "speaker_1": {"freq": 180, "amplitude": 0.5},   # 低频（男声）
        "speaker_2": {"freq": 350, "amplitude": 0.4},   # 中频（女声）
        "speaker_3": {"freq": 260, "amplitude": 0.35},  # 中低频
    }
    speaker_names = list(speakers.keys())

    segment_dur = 2.0
    gap_dur = 0.3
    pos = 0
    speaker_idx = 0

    while pos < num_samples:
        seg_samples = int(segment_dur * sample_rate)
        end = min(pos + seg_samples, num_samples)
        t = np.arange(end - pos) / sample_rate

        spk = speakers[speaker_names[speaker_idx % len(speaker_names)]]
        audio[pos:end] = spk["amplitude"] * np.sin(
            2 * np.pi * spk["freq"] * t
        )

        speaker_idx += 1
        pos = end + int(gap_dur * sample_rate)

    # 归一化
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    _write_wav(output_path, audio, sample_rate)
    print(f"  ✓ {output_path.name} ({duration:.0f}s, {len(speakers)} speakers)")


def generate_empty_wav(
    output_path: Path,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """生成极短（几乎空）的 WAV 文件"""
    # 0.01 秒，约 160 个样本
    audio = np.zeros(int(sample_rate * 0.01), dtype=np.float32) + 1e-6
    _write_wav(output_path, audio, sample_rate)
    print(f"  ✓ {output_path.name} (0.01s)")


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """写入 16-bit PCM WAV 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)

    # 转为 int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="生成测试用音频 fixtures"
    )
    parser.add_argument(
        "--all", action="store_true", default=True,
        help="生成所有 test fixtures"
    )
    args = parser.parse_args()

    print(f"生成测试音频 fixtures → {OUTPUT_DIR}")
    print()

    generate_speech_wav(OUTPUT_DIR / "sample_1min_speech.wav", duration=60.0)
    generate_multi_speaker_wav(OUTPUT_DIR / "sample_multi_speaker.wav", duration=30.0)
    generate_silence_wav(OUTPUT_DIR / "sample_silence_only.wav", duration=5.0)
    generate_empty_wav(OUTPUT_DIR / "sample_empty.wav")

    print()
    print("完成！")


if __name__ == "__main__":
    main()
