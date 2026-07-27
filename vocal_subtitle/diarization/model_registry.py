"""Speaker model catalog, cache inspection, and explicit downloads."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from .pyannote_engine import PyannoteDiarizationEngine
from .speaker_embedding import (
    DEFAULT_CACHE_DIR,
    PyannoteEmbeddingEngine,
    is_huggingface_model_cached,
)


MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "speechbrain-ecapa": {
        "model_id": "speechbrain-ecapa",
        "kind": "embedding",
        "model_ref": "speechbrain/spkrec-ecapa-voxceleb",
        "name": "SpeechBrain ECAPA",
        "description": "VoxCeleb ECAPA speaker embedding, default local backend",
        "size_mb": 80,
        "requires_token": False,
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
    },
    "pyannote-embedding": {
        "model_id": "pyannote-embedding",
        "kind": "embedding",
        "model_ref": "pyannote/embedding",
        "name": "pyannote/embedding",
        "description": "512-dim pyannote speaker embedding",
        "size_mb": 100,
        "requires_token": True,
        "license": "pyannote model terms",
        "license_url": "https://huggingface.co/pyannote/embedding",
    },
    "community-1": {
        "model_id": "community-1",
        "kind": "global",
        "model_ref": "pyannote/speaker-diarization-community-1",
        "name": "Community-1",
        "description": "Global speaker diarization pipeline",
        "size_mb": 500,
        "requires_token": True,
        "license": "pyannote model terms",
        "license_url": "https://huggingface.co/pyannote/speaker-diarization-community-1",
    },
    "diarization-3.1": {
        "model_id": "diarization-3.1",
        "kind": "global",
        "model_ref": "pyannote/speaker-diarization-3.1",
        "name": "Diarization 3.1",
        "description": "pyannote speaker diarization 3.1 pipeline",
        "size_mb": 500,
        "requires_token": True,
        "license": "pyannote model terms",
        "license_url": "https://huggingface.co/pyannote/speaker-diarization-3.1",
    },
}


def _cache_root(cache_dir: Optional[Path | str] = None) -> Path:
    return Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR


def _speechbrain_cached(model_ref: str, cache_dir: Path) -> bool:
    savedir = cache_dir / model_ref.replace("/", "_")
    return (savedir / "hyperparams.yaml").is_file() and any(
        item.stat().st_size > 200
        for item in savedir.glob("*.ckpt")
        if item.is_file()
    )


def is_model_cached(model_id: str, cache_dir: Optional[Path | str] = None) -> bool:
    spec = MODEL_CATALOG.get(model_id)
    if spec is None:
        raise KeyError(f"Unknown speaker model: {model_id}")
    root = _cache_root(cache_dir)
    if spec["kind"] == "embedding" and spec["model_ref"].startswith("speechbrain/"):
        return _speechbrain_cached(spec["model_ref"], root)
    return is_huggingface_model_cached(spec["model_ref"], root)


def model_status(model_id: str, cache_dir: Optional[Path | str] = None) -> dict[str, Any]:
    spec = MODEL_CATALOG.get(model_id)
    if spec is None:
        raise KeyError(f"Unknown speaker model: {model_id}")
    root = _cache_root(cache_dir)
    cached = is_model_cached(model_id, root)
    result = dict(spec)
    result.update({
        "cached": cached,
        "status": "ready" if cached else "not_cached",
        "cache_integrity": "complete" if cached else "incomplete_or_missing",
        "cache_dir": str(root),
    })
    return result


def list_model_status(cache_dir: Optional[Path | str] = None) -> list[dict[str, Any]]:
    return [model_status(model_id, cache_dir) for model_id in MODEL_CATALOG]


@contextmanager
def _network_enabled_for_hf_download():
    """Temporarily override the package-wide offline defaults for downloads."""
    previous_env = {
        key: os.environ.get(key)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    }
    saved_hf_offline = None
    saved_transformers_offline = None
    try:
        for key in previous_env:
            os.environ[key] = "0"
        try:
            import huggingface_hub.constants as hf_constants

            saved_hf_offline = hf_constants.HF_HUB_OFFLINE
            hf_constants.HF_HUB_OFFLINE = False
        except (ImportError, AttributeError):
            pass
        try:
            import transformers.utils.hub as transformers_hub

            saved_transformers_offline = transformers_hub._is_offline_mode
            transformers_hub._is_offline_mode = False
        except (ImportError, AttributeError):
            pass
        yield
    finally:
        if saved_hf_offline is not None:
            try:
                import huggingface_hub.constants as hf_constants

                hf_constants.HF_HUB_OFFLINE = saved_hf_offline
            except (ImportError, AttributeError):
                pass
        if saved_transformers_offline is not None:
            try:
                import transformers.utils.hub as transformers_hub

                transformers_hub._is_offline_mode = saved_transformers_offline
            except (ImportError, AttributeError):
                pass
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _download_snapshot(**kwargs):
    """Lazy wrapper to keep model catalog/status operations offline-safe."""
    from huggingface_hub import snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_ENDPOINT")
    if endpoint:
        kwargs.setdefault("endpoint", endpoint.rstrip("/"))
    # A longer metadata timeout avoids false failures on slow but working links.
    kwargs.setdefault("etag_timeout", 30)
    return snapshot_download(**kwargs)


def download_model(
    model_id: str,
    *,
    token: Optional[str] = None,
    cache_dir: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Download one model using the Hugging Face cache used by the engines."""
    spec = MODEL_CATALOG.get(model_id)
    if spec is None:
        raise KeyError(f"Unknown speaker model: {model_id}")
    root = _cache_root(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    if is_model_cached(model_id, root):
        return model_status(model_id, root)

    normalized_token = (token or "").strip()
    if normalized_token == "***":
        normalized_token = ""
    resolved_token = normalized_token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    if not resolved_token:
        try:
            from ..utils.hf_token_store import load_hf_token

            resolved_token = load_hf_token()
        except (ImportError, OSError, ValueError):
            resolved_token = None
    download_kwargs = {
        "repo_id": spec["model_ref"],
        "token": resolved_token,
    }
    if spec["kind"] == "embedding" and spec["model_ref"].startswith("speechbrain/"):
        download_kwargs["local_dir"] = str(root / spec["model_ref"].replace("/", "_"))
    else:
        download_kwargs["cache_dir"] = str(root / "hub")
    try:
        with _network_enabled_for_hf_download():
            _download_snapshot(**download_kwargs)
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download speaker models"
        ) from exc
    result = model_status(model_id, root)
    if not result["cached"]:
        raise RuntimeError(
            f"Downloaded files for {spec['model_ref']} are incomplete; "
            "the local cache did not pass integrity validation"
        )
    return result


def resolve_global_model_ref(selection: str) -> Optional[str]:
    if not selection or selection in ("none", "disabled"):
        return None
    if selection == "community-1":
        return MODEL_CATALOG["community-1"]["model_ref"]
    if selection == "diarization-3.1":
        return MODEL_CATALOG["diarization-3.1"]["model_ref"]
    if selection in MODEL_CATALOG and MODEL_CATALOG[selection]["kind"] == "global":
        return MODEL_CATALOG[selection]["model_ref"]
    return None
