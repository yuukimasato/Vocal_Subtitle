#!/usr/bin/env python3
"""性能基准测试脚本

测试各引擎在不同模型配置下的推理速度。
"""

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np

from vocal_subtitle.utils.gpu_detector import GPUDetector


def generate_test_audio(duration_seconds: float = 30.0, sample_rate: int = 16000) -> np.ndarray:
    """生成测试用音频（白噪声 + 正弦波混合，模拟语音）"""
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)

    # 模拟语音：440Hz 正弦波 + 白噪声
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t)
        + 0.02 * np.random.randn(num_samples)
    ).astype(np.float32)

    # 归一化
    audio /= np.max(np.abs(audio)) + 1e-8
    return audio


def benchmark_asr_models() -> Dict[str, float]:
    """测试 ASR 模型加载和推理速度"""
    print("=" * 60)
    print("ASR 模型基准测试")
    print("=" * 60)

    device = GPUDetector.get_best_device().value
    print(f"设备: {device}")
    print()

    test_audio = generate_test_audio(duration_seconds=10.0)
    models = ["tiny", "small", "medium", "large-v3"]
    results = {}

    for model_name in models:
        try:
            from vocal_subtitle.asr.faster_whisper_engine import FasterWhisperEngine

            print(f"测试模型: {model_name}...")

            engine = FasterWhisperEngine(
                model=model_name,
                device=device,
                compute_type="float16" if device == "cuda" else "int8",
            )

            # 加载计时
            t0 = time.time()
            engine.load_model()
            load_time = time.time() - t0
            print(f"  加载耗时: {load_time:.2f}s")

            # 推理计时
            t1 = time.time()
            result = engine.transcribe(test_audio, sample_rate=16000)
            inference_time = time.time() - t1

            speed_ratio = 10.0 / inference_time  # 10秒音频
            print(f"  推理耗时: {inference_time:.2f}s (速度比: {speed_ratio:.1f}x)")

            if result:
                print(f"  识别文本: {result[0].text[:50]}...")

            results[model_name] = inference_time
            print()

        except ImportError as e:
            print(f"  ⚠ 跳过: {e}")
            print()
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            print()

    return results


def benchmark_vad() -> Dict[str, float]:
    """测试 VAD 引擎性能"""
    print("=" * 60)
    print("VAD 引擎基准测试")
    print("=" * 60)

    test_audio = generate_test_audio(duration_seconds=60.0)
    results = {}

    # Silero VAD
    try:
        from vocal_subtitle.vad.silero_vad import SileroVAD

        print("测试 Silero VAD...")
        engine = SileroVAD()
        engine.load_model()

        t0 = time.time()
        segments = engine.detect_on_array(test_audio, 16000)
        elapsed = time.time() - t0

        print(f"  耗时: {elapsed:.3f}s")
        print(f"  检测到 {len(segments)} 个语音片段")
        results["silero"] = elapsed
    except Exception as e:
        print(f"  ⚠ 跳过: {e}")

    print()

    # WebRTC VAD
    try:
        from vocal_subtitle.vad.webrtc_vad import WebRTCVAD

        print("测试 WebRTC VAD...")
        engine = WebRTCVAD()
        engine.load_model()

        t0 = time.time()
        segments = engine.detect_on_array(test_audio, 16000)
        elapsed = time.time() - t0

        print(f"  耗时: {elapsed:.3f}s")
        print(f"  检测到 {len(segments)} 个语音片段")
        results["webrtc"] = elapsed
    except Exception as e:
        print(f"  ⚠ 跳过: {e}")

    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="性能基准测试")
    parser.add_argument(
        "--asr", action="store_true", help="测试 ASR 模型性能"
    )
    parser.add_argument(
        "--vad", action="store_true", help="测试 VAD 引擎性能"
    )
    parser.add_argument(
        "--all", action="store_true", help="运行所有测试"
    )
    args = parser.parse_args()

    run_all = args.all or (not args.asr and not args.vad)

    if run_all or args.asr:
        benchmark_asr_models()

    if run_all or args.vad:
        benchmark_vad()

    print("=" * 60)
    print("基准测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
