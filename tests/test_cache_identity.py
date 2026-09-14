"""缓存身份契约(2026-09-15 重构计划 Task 7)。

- 同内容不同路径 → 相同键(命中);
- 内容或版本/模型/引擎/配置变化 → 不同键(失效);
- 参数规范化:排序无关、None 剔除。
"""

import hashlib

import pytest

from vocal_subtitle.utils.cache_key import (
    SCHEMA_VERSION,
    build_cache_key,
    normalize_params,
)
from vocal_subtitle.utils.file_hasher import compute_file_hash


def _content(tmp_path, text: str) -> str:
    target = tmp_path / f"{hashlib.md5(text.encode()).hexdigest()}.wav"
    target.write_text(text, encoding="utf-8")
    return compute_file_hash(target)


def test_same_content_different_path_hits(tmp_path):
    first = tmp_path / "a.wav"
    second = tmp_path / "sub" / "b.wav"
    second.parent.mkdir()
    first.write_text("same-bytes", encoding="utf-8")
    second.write_text("same-bytes", encoding="utf-8")

    key_a = build_cache_key(
        content_hash=compute_file_hash(first),
        stage="asr", stage_version="v1", engine="faster-whisper", model="large-v3",
    )
    key_b = build_cache_key(
        content_hash=compute_file_hash(second),
        stage="asr", stage_version="v1", engine="faster-whisper", model="large-v3",
    )

    assert key_a == key_b


def test_changed_content_misses(tmp_path):
    old_hash = _content(tmp_path, "old")
    new_hash = _content(tmp_path, "new")

    key_old = build_cache_key(content_hash=old_hash, stage="asr", stage_version="v1")
    key_new = build_cache_key(content_hash=new_hash, stage="asr", stage_version="v1")

    assert key_old != key_new


def test_model_or_stage_version_change_invalidates():
    base = dict(content_hash="abc", stage="asr")

    key_v1 = build_cache_key(**base, stage_version="v1", model="large-v3")
    key_v2 = build_cache_key(**base, stage_version="v2", model="large-v3")
    key_model = build_cache_key(**base, stage_version="v1", model="large-v2")

    assert len({key_v1, key_v2, key_model}) == 3


def test_config_change_invalidates():
    key_a = build_cache_key(
        content_hash="abc", stage="asr", stage_version="v1",
        config={"beam_size": 5},
    )
    key_b = build_cache_key(
        content_hash="abc", stage="asr", stage_version="v1",
        config={"beam_size": 1},
    )

    assert key_a != key_b


def test_param_normalization_is_order_and_none_insensitive():
    assert normalize_params({"b": 1, "a": 2}) == normalize_params({"a": 2, "b": 1})
    assert normalize_params({"a": 1, "b": None}) == normalize_params({"a": 1})

    key_a = build_cache_key(
        content_hash="abc", stage="asr", stage_version="v1", params={"a": 1, "b": 2},
    )
    key_b = build_cache_key(
        content_hash="abc", stage="asr", stage_version="v1", params={"b": 2, "a": 1},
    )
    key_c = build_cache_key(
        content_hash="abc", stage="asr", stage_version="v1",
        params={"a": 1, "b": None},
    )
    key_a_only = build_cache_key(
        content_hash="abc", stage="asr", stage_version="v1", params={"a": 1},
    )

    assert key_a == key_b
    # None 参数等价于缺省。
    assert key_c == key_a_only


def test_schema_version_change_invalidates_everything():
    key_old = build_cache_key(
        content_hash="abc", stage="asr", stage_version="v1",
        schema_version="cache-key-v0",
    )
    key_new = build_cache_key(
        content_hash="abc", stage="asr", stage_version="v1",
        schema_version=SCHEMA_VERSION,
    )

    assert key_old != key_new


def test_bounded_cleanup_of_normalized_params():
    params = {f"p{index}": index for index in range(200)}
    normalized = normalize_params(params)

    assert len(normalized) == 200
    key = build_cache_key(
        content_hash="abc", stage="asr", stage_version="v1", params=params,
    )
    assert len(key) == 64  # sha256 hex 长度有界
