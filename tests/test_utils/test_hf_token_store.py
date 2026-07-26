"""Tests for local encrypted Hugging Face Token storage."""

from vocal_subtitle.utils.hf_token_store import (
    has_hf_token,
    load_hf_token,
    store_hf_token,
)


def test_hf_token_round_trip_uses_restricted_files(tmp_path):
    key_path = tmp_path / ".hf_token.key"
    token_path = tmp_path / "hf_token.enc"

    store_hf_token(
        "hf_test_secret",
        key_path=key_path,
        token_path=token_path,
    )

    assert load_hf_token(key_path=key_path, token_path=token_path) == "hf_test_secret"
    assert has_hf_token(key_path=key_path, token_path=token_path)
    assert token_path.read_bytes() != b"hf_test_secret"
    assert key_path.stat().st_mode & 0o777 == 0o600
    assert token_path.stat().st_mode & 0o777 == 0o600


def test_hf_token_store_returns_none_when_missing(tmp_path):
    assert load_hf_token(
        key_path=tmp_path / ".missing.key",
        token_path=tmp_path / "missing.enc",
    ) is None
