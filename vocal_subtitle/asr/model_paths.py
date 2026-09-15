"""Stable local paths for offline review models."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_QWEN_MODEL = "qwen3-asr-1.7b"
REVIEW_MODEL_DIR_ENV = "VOCAL_SUBTITLE_REVIEW_MODEL_DIR"

_QWEN_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "cmn": "Chinese",
    "chinese": "Chinese",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "en": "English",
    "english": "English",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "es": "Spanish",
    "spanish": "Spanish",
    "ru": "Russian",
    "russian": "Russian",
    "it": "Italian",
    "italian": "Italian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ar": "Arabic",
    "arabic": "Arabic",
    "th": "Thai",
    "thai": "Thai",
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
}


def qwen_language_name(language: str | None) -> str | None:
    """Convert internal ISO-style language codes to Qwen's language names."""
    if language is None:
        return None
    value = str(language).strip()
    if not value or value.casefold() in {"auto", "mixed", "multilingual"}:
        return None
    return _QWEN_LANGUAGE_NAMES.get(value.casefold(), value)


def internal_language_name(language: str | None) -> str | None:
    """Normalize Qwen language names back to the pipeline's short codes."""
    if language is None:
        return None
    value = str(language).strip()
    for code, qwen_name in _QWEN_LANGUAGE_NAMES.items():
        if len(code) <= 3 and value.casefold() == qwen_name.casefold():
            return code
    return value or None


def review_model_cache_dir() -> Path:
    configured = os.environ.get(REVIEW_MODEL_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "vocal-subtitle" / "review-models"


def default_qwen_model_path(model: str = DEFAULT_QWEN_MODEL) -> Path:
    return review_model_cache_dir() / model


__all__ = [
    "DEFAULT_QWEN_MODEL",
    "REVIEW_MODEL_DIR_ENV",
    "default_qwen_model_path",
    "internal_language_name",
    "qwen_language_name",
    "review_model_cache_dir",
]
