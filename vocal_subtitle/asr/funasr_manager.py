"""FunASR dependency and model preparation helpers.

The manager deliberately keeps local-cache inspection separate from model
download so WebUI status checks never cause network access.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_FUNASR_MODEL = (
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)
_GENERIC_MODEL_NAMES = {"", "large-v3", "large-v2", "medium", "small", "tiny"}
_MODEL_MARKERS = {
    "configuration.json",
    "config.yaml",
    "configuration.yaml",
    "model.pt",
    "model.pth",
    "model.bin",
}
_PREPARE_LOCK = threading.Lock()


class FunASRPrepareError(RuntimeError):
    """Raised when FunASR cannot be made ready for use."""


def normalize_model_id(model: str | None) -> str:
    """Map generic Whisper names to the model expected by FunASR."""
    value = (model or "").strip()
    return DEFAULT_FUNASR_MODEL if value in _GENERIC_MODEL_NAMES else value


def funasr_package_installed() -> bool:
    """Check package availability without importing FunASR or loading models."""
    return importlib.util.find_spec("funasr") is not None


def _cache_roots(cache_dir: Optional[Path | str] = None) -> list[Path]:
    if cache_dir:
        return [Path(cache_dir).expanduser()]

    roots: list[Path] = []
    for key in ("FUNASR_MODEL_CACHE_DIR", "MODELSCOPE_CACHE"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value).expanduser())
    roots.extend(
        [
            Path.home() / ".cache" / "modelscope" / "hub",
            Path.home() / ".cache" / "modelscope",
        ]
    )

    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any((path / marker).is_file() for marker in _MODEL_MARKERS):
        return True
    return any(
        item.is_file() and item.suffix in {".pt", ".pth", ".bin", ".onnx"}
        for item in path.iterdir()
    )


def find_local_model(model: str | None, cache_dir: Optional[Path | str] = None) -> Optional[Path]:
    """Find a complete local ModelScope snapshot without network access."""
    model_id = normalize_model_id(model)
    if not model_id:
        return None

    leaf = model_id.rsplit("/", 1)[-1]
    relative_candidates = (
        Path(model_id),
        Path(model_id.replace("/", "_")),
        Path("models") / model_id,
        Path("models") / model_id.replace("/", "_"),
    )
    for root in _cache_roots(cache_dir):
        for relative in relative_candidates:
            candidate = root / relative
            if _looks_like_model_dir(candidate):
                return candidate
        if root.is_dir():
            # ModelScope cache layouts have changed between releases; limit the
            # fallback search to directories named after this model only.
            try:
                for candidate in root.rglob(leaf):
                    if _looks_like_model_dir(candidate):
                        return candidate
            except OSError:
                continue
    return None


def _install_funasr_package() -> None:
    logger.info("FunASR package is missing; installing with the active Python")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "funasr"],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FunASRPrepareError(
            "FunASR 自动安装失败，请检查 pip、网络或手动安装 funasr"
        ) from exc
    if result.returncode != 0:
        logger.error("FunASR installation failed with exit code %s", result.returncode)
        raise FunASRPrepareError(
            "FunASR 自动安装失败，请检查 pip、网络或手动安装 funasr"
        )
    importlib.invalidate_caches()
    if not funasr_package_installed():
        raise FunASRPrepareError(
            "FunASR 安装完成但当前 Python 无法加载，请重启 WebUI"
        )


def _download_model(model_id: str, cache_dir: Optional[Path | str] = None) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise FunASRPrepareError(
            "缺少 ModelScope 下载依赖，请先安装 funasr 或手动安装 modelscope"
        ) from exc

    root = _cache_roots(cache_dir)[0]
    root.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = snapshot_download(model_id, cache_dir=str(root))
    except Exception as exc:
        logger.error("FunASR model download failed: %s", type(exc).__name__)
        raise FunASRPrepareError("FunASR 模型下载失败，请检查网络后重试") from exc

    downloaded_path = Path(downloaded).expanduser() if downloaded else None
    if downloaded_path and _looks_like_model_dir(downloaded_path):
        return downloaded_path
    local_path = find_local_model(model_id, root)
    if local_path is None:
        raise FunASRPrepareError("FunASR 模型下载完成但本地缓存不完整，请重试")
    return local_path


def funasr_status(
    model: str | None = None,
    cache_dir: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Return local-only readiness information."""
    model_id = normalize_model_id(model)
    local_path = find_local_model(model_id, cache_dir)
    package_installed = funasr_package_installed()
    return {
        "engine": "funasr",
        "model": model_id,
        "package_installed": package_installed,
        "model_cached": local_path is not None,
        "model_path": str(local_path) if local_path else None,
        "ready": package_installed and local_path is not None,
    }


def ensure_funasr_ready(
    model: str | None = None,
    cache_dir: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Install missing package and download only a missing local model."""
    model_id = normalize_model_id(model)
    with _PREPARE_LOCK:
        if not funasr_package_installed():
            _install_funasr_package()

        local_path = find_local_model(model_id, cache_dir)
        if local_path is None:
            local_path = _download_model(model_id, cache_dir)

        return {
            "engine": "funasr",
            "model": model_id,
            "package_installed": True,
            "model_cached": True,
            "model_path": str(local_path),
            "ready": True,
        }
