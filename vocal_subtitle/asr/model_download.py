"""Explicit model readiness and download helpers for faster-whisper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUPPORTED_FASTER_WHISPER_MODELS = ("tiny", "small", "medium", "large-v3")


def faster_whisper_model_ref(model: str) -> str:
    if model not in SUPPORTED_FASTER_WHISPER_MODELS:
        choices = ", ".join(SUPPORTED_FASTER_WHISPER_MODELS)
        raise ValueError(
            f"unsupported faster-whisper model {model!r}; choose {choices}"
        )
    return f"Systran/faster-whisper-{model}"


def faster_whisper_cache_dir() -> Path:
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home if hf_home.name == "hub" else hf_home / "hub"


def is_faster_whisper_model_cached(model: str) -> bool:
    """Check for a complete local snapshot without importing the runtime."""
    model_dir = faster_whisper_cache_dir() / (
        "models--" + faster_whisper_model_ref(model).replace("/", "--")
    )
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(
        (snapshot / "config.json").is_file() and (snapshot / "model.bin").is_file()
        for snapshot in snapshots.iterdir()
        if snapshot.is_dir()
    )


def faster_whisper_cached_model_path(model: str) -> Path | None:
    """Return a complete local snapshot path for ``model`` when available."""
    model_dir = faster_whisper_cache_dir() / (
        "models--" + faster_whisper_model_ref(model).replace("/", "--")
    )
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(
        snapshot
        for snapshot in snapshots.iterdir()
        if snapshot.is_dir()
        and (snapshot / "config.json").is_file()
        and (snapshot / "model.bin").is_file()
    )
    return candidates[0] if candidates else None


def ensure_faster_whisper_model(
    model: str,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a model, downloading it when absent, and return readiness details."""
    cached_before = is_faster_whisper_model_cached(model)
    from .faster_whisper_engine import FasterWhisperEngine

    compute_type = "int8" if device == "cpu" else "float16"
    engine = FasterWhisperEngine(
        model=model,
        device=device,
        compute_type=compute_type,
        word_timestamps=False,
    )
    engine.load_model()
    if not is_faster_whisper_model_cached(model):
        raise RuntimeError(
            f"faster-whisper model loaded but cache snapshot is incomplete: {model}"
        )
    return {
        "status": "ready" if cached_before else "downloaded",
        "model": model,
        "model_ref": faster_whisper_model_ref(model),
        "cache_dir": str(faster_whisper_cache_dir()),
    }


__all__ = [
    "SUPPORTED_FASTER_WHISPER_MODELS",
    "ensure_faster_whisper_model",
    "faster_whisper_cached_model_path",
    "faster_whisper_cache_dir",
    "faster_whisper_model_ref",
    "is_faster_whisper_model_cached",
]
