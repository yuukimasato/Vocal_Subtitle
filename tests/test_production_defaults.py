from vocal_subtitle.config import ConfigLoader


def test_default_offline_review_is_production_risk_only_with_fallback():
    config = ConfigLoader().load_profile("default")

    assert config.evidence_review.enabled is True
    assert config.asr.global_asr.routing == "segmented"
    assert config.asr.global_asr.evidence_enabled is True
    # EvidenceReview runs in shadow_mode by default so it annotates risk
    # without changing the release event until the golden gate passes.
    assert config.evidence_review.authoritative_mode is False
    assert config.evidence_review.shadow_mode is True
    assert config.evidence_review.fallback_to_segmented is True
    assert config.asr.engine_pair.policy == "risk_only"
    assert config.evidence_review.golden_quality_gate_version == "golden-quality-v1"


def test_explicit_shadow_remains_a_valid_rollback_configuration(tmp_path):
    config_path = tmp_path / "shadow.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  evidence_review:\n"
        "    authoritative_mode: false\n"
        "    shadow_mode: true\n",
        encoding="utf-8",
    )

    config = ConfigLoader.load_file(config_path)

    assert config.evidence_review.authoritative_mode is False
    assert config.evidence_review.shadow_mode is True


def test_legacy_config_without_review_mode_uses_safe_shadow_defaults(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  evidence_review:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    config = ConfigLoader.load_file(config_path)

    assert config.evidence_review.shadow_mode is True
    assert config.evidence_review.authoritative_mode is False
    assert config.evidence_review.context_reasr_enabled is False
