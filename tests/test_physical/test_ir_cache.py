import pytest

from vocal_subtitle.physical.ir_cache import (
    decode_ir_value,
    encode_ir_value,
    fingerprint_ir,
    make_ir_cache_key,
    load_ir_value,
    persist_ir_value,
)


def _key(**overrides):
    params = {
        "artifact_type": "context_windows",
        "schema_version": "physical-timeline-v1",
        "producer_version": "p2.4",
        "input_sha256": "a" * 64,
        "audio_duration": 10.0,
        "timeline_fingerprint": "b" * 64,
        "coordinate_policy": "absolute-v1",
        "context_policy": "left1-right1",
        "additional_params": {"profile": "default"},
    }
    params.update(overrides)
    return make_ir_cache_key(**params)


def test_fingerprint_is_order_independent_and_content_sensitive():
    assert fingerprint_ir({"b": 2, "a": [1, 2]}) == fingerprint_ir({"a": [1, 2], "b": 2})
    assert fingerprint_ir({"a": [1, 2]}) != fingerprint_ir({"a": [1, 3]})


def test_ir_key_isolated_by_contract_fields_and_excludes_paths():
    base = _key()
    assert len(base) == 64
    assert base != _key(schema_version="other")
    assert base != _key(input_sha256="c" * 64)
    assert base != _key(context_policy="left2-right2")
    assert base != _key(additional_params={"profile": "other"})
    with pytest.raises(ValueError):
        _key(additional_params={"path": "/secret/file"})
    assert "/secret/file" not in base


def test_none_duration_is_distinct_and_cache_value_is_json_safe():
    assert _key(audio_duration=None) != _key(audio_duration=0.0)
    value = encode_ir_value({"schema_version": "global-ir-v1", "words": []})
    assert value == {"schema_version": "global-ir-v1", "words": []}
    assert decode_ir_value(value, lambda payload: payload["words"], expected_schema_version="global-ir-v1") == []


def test_damaged_or_wrong_schema_cache_value_is_a_miss():
    assert decode_ir_value(None, lambda payload: payload) is None
    assert decode_ir_value({"schema_version": "old"}, lambda payload: payload, expected_schema_version="new") is None
    assert decode_ir_value({"schema_version": "new"}, lambda payload: 1 / 0, expected_schema_version="new") is None


def test_persist_and_load_use_json_safe_cache_boundary():
    class FakeCache:
        def __init__(self):
            self.values = {}

        def set(self, stage, key, value, ttl=None):
            self.values[(stage, key)] = value

        def get(self, stage, key):
            return self.values.get((stage, key))

    cache = FakeCache()
    assert persist_ir_value(cache, "ir", "key", {"schema_version": "global-ir-v1", "words": []})
    assert load_ir_value(
        cache, "ir", "key", lambda payload: payload["words"],
        expected_schema_version="global-ir-v1",
    ) == []
