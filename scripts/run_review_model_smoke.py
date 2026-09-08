#!/usr/bin/env python3
"""Run a bounded smoke test for the configured review/ASR model runtimes.

The command is intentionally local-first. Missing dependencies or model
snapshots are reported as ``blocked`` instead of triggering an implicit large
download or being counted as a successful model run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_subtitle.asr.faster_whisper_engine import FasterWhisperEngine
from vocal_subtitle.asr.funasr_engine import FunASREngine
from vocal_subtitle.asr.funasr_manager import find_local_model, normalize_model_id
from vocal_subtitle.asr.model_download import (
    faster_whisper_cached_model_path,
)
from vocal_subtitle.asr.optional_adapters import LazyQwenASR
from vocal_subtitle.asr.review_scheduler import ReviewWindow
from vocal_subtitle.asr.review_engines import WindowedASREngine
from vocal_subtitle.asr.review_telemetry import resource_snapshot
from vocal_subtitle.asr.window_execution import CancellationToken, WindowExecutionCoordinator
from vocal_subtitle.asr.whisper_cpp_engine import WhisperCppEngine


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(np.ceil(len(ordered) * percentile / 100)) - 1))
    return round(ordered[index], 6)


def _latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50_seconds": _percentile(values, 50),
        "p95_seconds": _percentile(values, 95),
        "min_seconds": round(min(values), 6) if values else None,
        "max_seconds": round(max(values), 6) if values else None,
        "mean_seconds": round(statistics.mean(values), 6) if values else None,
    }


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"smoke audio must be 16-bit PCM: {path}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def _environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "dependencies": {},
    }
    for name in ("torch", "qwen_asr", "funasr", "faster_whisper", "transformers"):
        result["dependencies"][name] = importlib.util.find_spec(name) is not None
    try:
        import torch

        result["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    except (ImportError, AttributeError, RuntimeError) as exc:
        result["torch"] = {"status": "unavailable", "reason": str(exc)}
    result["resources_before"] = resource_snapshot()
    return result


def _blocked(engine: str, model: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "engine": engine,
        "model": model,
        "status": "blocked",
        "reason": reason,
        "latency": _latency_summary([]),
        **extra,
    }


def _run_iterations(
    transcribe: Callable[[], Any],
    iterations: int,
) -> tuple[list[float], list[int], str | None]:
    latencies: list[float] = []
    result_sizes: list[int] = []
    for _ in range(iterations):
        started = time.perf_counter()
        try:
            result = transcribe()
        except Exception as exc:  # model-specific errors belong in the report
            return latencies, result_sizes, f"{type(exc).__name__}: {exc}"
        latencies.append(time.perf_counter() - started)
        result_sizes.append(len(result or ()))
    return latencies, result_sizes, None


def _window_smoke(engine: Any, audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> dict[str, Any]:
    windows = (
        ReviewWindow("smoke-0", 0.0, min(1.0, len(audio) / sample_rate), ()),
        ReviewWindow("smoke-1", min(1.0, len(audio) / sample_rate), min(2.0, len(audio) / sample_rate), ()),
    )
    windows = tuple(item for item in windows if item.end > item.start)
    coordinator = WindowExecutionCoordinator(
        max_workers=args.max_workers,
        timeout_seconds=args.timeout_seconds,
    )
    candidates, diagnostics = coordinator.execute(
        audio,
        sample_rate,
        windows,
        engine,
        language=args.language,
    )
    cancel_token = CancellationToken()
    cancel_token.cancel("smoke_pre_cancel")
    _, cancel_diagnostics = coordinator.execute(
        audio,
        sample_rate,
        windows,
        engine,
        language=args.language,
        token=cancel_token,
    )
    return {
        "candidate_count": len(candidates),
        "diagnostics": diagnostics,
        "timeout_control": diagnostics.get("timeout_seconds") == args.timeout_seconds,
        "concurrency_control": diagnostics.get("max_workers") == args.max_workers,
        "cancel_control": "cancelled" in cancel_diagnostics.get("status_counts", {}),
        "cancel_diagnostics": cancel_diagnostics,
    }


def _smoke_engine(
    engine_name: str,
    model: str,
    load: Callable[[], Any],
    transcribe: Callable[[], Any],
    audio: np.ndarray,
    sample_rate: int,
    args: argparse.Namespace,
    *,
    window_engine: Any = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        loaded = load()
    except Exception as exc:
        return _blocked(
            engine_name,
            model,
            f"load_failed: {type(exc).__name__}: {exc}",
            load_seconds=round(time.perf_counter() - started, 6),
        )
    latencies, result_sizes, error = _run_iterations(transcribe, args.iterations)
    if error:
        return {
            "engine": engine_name,
            "model": model,
            "status": "failed",
            "reason": error,
            "loaded": True,
            "latency": _latency_summary(latencies),
            "result_sizes": result_sizes,
        }
    result: dict[str, Any] = {
        "engine": engine_name,
        "model": model,
        "status": "pass",
        "loaded": True,
        "latency": _latency_summary(latencies),
        "result_sizes": result_sizes,
        "resources_after": resource_snapshot(),
    }
    if window_engine is not None:
        try:
            window_smoke = _window_smoke(
                window_engine, audio, sample_rate, args
            )
            result["window_smoke"] = window_smoke
            window_diagnostics = window_smoke.get("diagnostics", {})
            if window_diagnostics.get("degraded"):
                result["status"] = "failed"
                result["reason"] = "window_execution_degraded"
        except Exception as exc:
            result["window_smoke"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            result["status"] = "failed"
            result["reason"] = "window_execution_failed"
    return result


def _smoke_whisper_cpp(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> dict[str, Any]:
    engine = WhisperCppEngine(
        model=args.whisper_cpp_model,
        whisper_cpp_bin=args.whisper_cpp_bin,
        model_path=args.whisper_cpp_model_path,
        n_threads=args.threads,
    )
    window_engine = WindowedASREngine(
        lambda: engine,
        name="whisper-cpp",
        source="context_reasr",
        family="whisper",
        model_name=args.whisper_cpp_model,
    )
    return _smoke_engine(
        "whisper-cpp",
        args.whisper_cpp_model,
        engine.load_model,
        lambda: engine.transcribe(audio, sample_rate, language=args.language),
        audio,
        sample_rate,
        args,
        window_engine=window_engine,
    )


def _smoke_faster_whisper(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> dict[str, Any]:
    model = args.faster_whisper_model
    if not importlib.util.find_spec("faster_whisper"):
        return _blocked("faster-whisper", model, "dependency_unavailable: faster_whisper")
    model_path = Path(model).expanduser() if Path(model).expanduser().exists() else None
    if model_path is None and not args.allow_download:
        model_path = faster_whisper_cached_model_path(model)
    if not args.allow_download and model_path is None:
        return _blocked("faster-whisper", model, "local_model_path_required", download_allowed=False)
    resolved_model = str(model_path) if model_path is not None else model
    engine = FasterWhisperEngine(
        model=resolved_model,
        device=args.device,
        compute_type="int8" if args.device == "cpu" else "float16",
    )
    return _smoke_engine(
        "faster-whisper",
        resolved_model,
        engine.load_model,
        lambda: list(engine.transcribe(audio, sample_rate, language=args.language)),
        audio,
        sample_rate,
        args,
    )


def _smoke_funasr(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> dict[str, Any]:
    model = normalize_model_id(args.funasr_model)
    if not importlib.util.find_spec("funasr"):
        return _blocked("funasr", model, "dependency_unavailable: funasr")
    local_model = find_local_model(model)
    if local_model is None and not args.allow_download:
        return _blocked("funasr", model, "local_model_snapshot_missing", download_allowed=False)
    resolved_model = str(local_model) if local_model is not None else model
    engine = FunASREngine(model=resolved_model, device=args.device, ncpu=args.threads)
    return _smoke_engine(
        "funasr",
        resolved_model,
        engine.load_model,
        lambda: list(engine.transcribe(audio, sample_rate, language=args.language)),
        audio,
        sample_rate,
        args,
    )


def _smoke_qwen(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> dict[str, Any]:
    model = args.qwen_model_path
    adapter = LazyQwenASR(model, device=args.device, allow_remote=args.allow_download)
    availability = adapter.availability()
    if availability.get("status") != "ready":
        return _blocked("qwen", model or "qwen3-asr", availability.get("reason", "unavailable"), availability=availability)
    window = ReviewWindow("smoke-qwen", 0.0, min(2.0, len(audio) / sample_rate), ())
    return _smoke_engine(
        "qwen",
        model or "qwen3-asr",
        lambda: True,
        lambda: list(adapter.review(audio, sample_rate, window, language=args.language)),
        audio,
        sample_rate,
        args,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=ROOT / "test/golden/repeated_phrase_me.wav")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--whisper-cpp-model", default="tiny")
    parser.add_argument("--whisper-cpp-bin", default=str(ROOT / "cache/whisper_cpp/build/bin/whisper-cli"))
    parser.add_argument("--whisper-cpp-model-path", default=str(ROOT / "cache/whisper_cpp/models/ggml-tiny.bin"))
    parser.add_argument("--qwen-model-path", default=os.environ.get("QWEN_ASR_MODEL_PATH", ""))
    parser.add_argument("--funasr-model", default=os.environ.get("FUNASR_MODEL", ""))
    parser.add_argument("--faster-whisper-model", default=os.environ.get("FASTER_WHISPER_MODEL", "large-v3"))
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    audio_path = args.audio if args.audio.is_absolute() else ROOT / args.audio
    output = args.output if args.output.is_absolute() else ROOT / args.output
    audio, sample_rate = _load_wav(audio_path)
    results = [
        _smoke_whisper_cpp(audio, sample_rate, args),
        _smoke_faster_whisper(audio, sample_rate, args),
        _smoke_funasr(audio, sample_rate, args),
        _smoke_qwen(audio, sample_rate, args),
    ]
    report = {
        "schema_version": "review-model-smoke-v1",
        "audio": str(audio_path.relative_to(ROOT)),
        "sample_rate": sample_rate,
        "duration_seconds": round(len(audio) / sample_rate, 6),
        "allow_download": args.allow_download,
        "environment": _environment(),
        "execution_contract": {
            "max_workers": args.max_workers,
            "timeout_seconds": args.timeout_seconds,
            "cooperative_cancel_supported": True,
            "real_window_concurrency_exercised_by": [
                item["engine"] for item in results
                if item.get("window_smoke", {}).get("diagnostics", {}).get("window_count", 0) > 1
            ],
        },
        "results": results,
        "status": "pass" if all(item["status"] == "pass" for item in results) else "blocked",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "results": results}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
