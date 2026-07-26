"""Local encrypted storage for the Hugging Face access token."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Optional


DEFAULT_TOKEN_DIR = Path(__file__).parent.parent.parent / "cache"
DEFAULT_KEY_PATH = DEFAULT_TOKEN_DIR / ".hf_token.key"
DEFAULT_TOKEN_PATH = DEFAULT_TOKEN_DIR / "hf_token.enc"

_STORE_LOCK = threading.Lock()


def _secure_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _get_fernet(key_path: Path):
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise ImportError(
            "cryptography is required for encrypted HF Token storage. "
            "Install the WebUI dependency."
        ) from exc

    if not key_path.exists():
        _secure_write(key_path, Fernet.generate_key())
    else:
        os.chmod(key_path, 0o600)
    return Fernet(key_path.read_bytes())


def store_hf_token(
    token: str,
    *,
    key_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> None:
    """Encrypt and persist a non-empty HF Token with user-only permissions."""
    normalized = (token or "").strip()
    if not normalized or normalized == "***":
        raise ValueError("A real HF Token is required")

    resolved_key = Path(key_path or DEFAULT_KEY_PATH).expanduser()
    resolved_token = Path(token_path or DEFAULT_TOKEN_PATH).expanduser()
    with _STORE_LOCK:
        fernet = _get_fernet(resolved_key)
        _secure_write(resolved_token, fernet.encrypt(normalized.encode("utf-8")))


def load_hf_token(
    *,
    key_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> Optional[str]:
    """Return the decrypted HF Token, or ``None`` when no valid token exists."""
    resolved_key = Path(key_path or DEFAULT_KEY_PATH).expanduser()
    resolved_token = Path(token_path or DEFAULT_TOKEN_PATH).expanduser()
    if not resolved_key.is_file() or not resolved_token.is_file():
        return None

    try:
        from cryptography.fernet import InvalidToken
    except ImportError:
        # Token storage is optional; a missing crypto extra must not break the UI.
        return None

    try:
        with _STORE_LOCK:
            os.chmod(resolved_token, 0o600)
            fernet = _get_fernet(resolved_key)
            return fernet.decrypt(resolved_token.read_bytes()).decode("utf-8")
    except (InvalidToken, OSError, UnicodeError, ValueError):
        return None


def has_hf_token(
    *,
    key_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> bool:
    return bool(load_hf_token(key_path=key_path, token_path=token_path))


def delete_hf_token(
    *,
    key_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> bool:
    """Delete encrypted Token material and report whether anything existed."""
    resolved_key = Path(key_path or DEFAULT_KEY_PATH).expanduser()
    resolved_token = Path(token_path or DEFAULT_TOKEN_PATH).expanduser()
    removed = False
    with _STORE_LOCK:
        for path in (resolved_token, resolved_key):
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                continue
    return removed
