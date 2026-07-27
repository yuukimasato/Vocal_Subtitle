from vocal_subtitle.config import ConfigLoader
from vocal_subtitle.config import (
    AcousticValidationConfig,
    ASRConfig,
    BoundaryRefinementConfig,
    GapHandlingConfig,
    LLMOptimizeConfig,
    MergingConfig,
    PipelineConfig,
    VADConfig,
)
from vocal_subtitle.pipeline import Pipeline


def test_default_profile_uses_lightweight_runtime():
    config = ConfigLoader().load_profile("default")

    assert config.separation.engine == "uvr"
    assert config.vad.engine == "silero"
    assert config.asr.model == "large-v3"
    assert config.asr.device == "auto"
    assert config.asr.compute_type == "float16"
    assert config.asr.global_asr.enabled is True
    assert config.asr.global_asr.routing == "auto"
    assert config.diarization.enabled is True
    assert config.diarization.backend == "auto"
    assert config.diarization.fusion_mode == "auto"
    assert config.diarization.global_model == "auto"
    assert config.diarization.diarization_scope == "hierarchical"
    assert config.diarization.local_refinement == "embedding"
    assert config.speaker_embedding.enabled is True


def test_none_separator_is_forwarded_as_skip_separation(tmp_path):
    config = ConfigLoader().load_profile("default")
    config.separation.engine = "none"
    config.mode = "streaming"
    pipeline = Pipeline(config)
    input_path = tmp_path / "voice.wav"
    input_path.write_bytes(b"wav")
    captured = {}

    def fake_streaming(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    pipeline.run_streaming = fake_streaming
    result = pipeline.run(
        input_path=input_path,
        output_path=tmp_path / "voice.srt",
        skip_separation=True,
    )

    assert result == {"ok": True}
    assert captured["skip_separation"] is True


def test_yaml_omitted_values_match_dataclass_defaults(tmp_path):
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text("pipeline: {}\n", encoding="utf-8")

    loaded = ConfigLoader.load_file(config_path)
    expected = PipelineConfig()

    assert loaded.asr.model == ASRConfig().model == expected.asr.model
    assert loaded.vad.min_speech_duration_ms == VADConfig().min_speech_duration_ms
    assert loaded.merging.pre_split_threshold == MergingConfig().pre_split_threshold
    assert loaded.subtitle.gap_handling.seamless_threshold == GapHandlingConfig().seamless_threshold
    assert loaded.acoustic_validation.snap_start_margin == AcousticValidationConfig().snap_start_margin
    assert loaded.acoustic_validation.skeleton_mode == AcousticValidationConfig().skeleton_mode
    assert loaded.boundary_refinement.max_shrink_ms == BoundaryRefinementConfig().max_shrink_ms
    assert loaded.llm_optimize.batch_num == LLMOptimizeConfig().batch_num


def test_default_degradation_is_loaded_at_pipeline_level():
    config = ConfigLoader().load_profile("default")

    assert config.degradation.mode == "full"
    assert config.degradation.per_module_timeout == 60
    assert config.diarization.expected_speakers is None
    assert config.feedback.active_profile == "user_default"


def test_speaker_overrides_are_loaded_into_diarization_config():
    config = ConfigLoader().load_profile("default")
    overridden = ConfigLoader().merge_with_overrides(
        config,
        expected_speakers=2,
        speaker_fusion="dual",
        global_diarization_model="community-1",
        speaker_diarization_scope="hierarchical",
        local_speaker_refinement="full",
    )
    assert overridden.diarization.expected_speakers == 2
    assert overridden.diarization.fusion_mode == "dual"
    assert overridden.diarization.global_model == "community-1"
    assert overridden.diarization.local_refinement == "full"


def test_degradation_accepts_legacy_nested_and_top_level_yaml(tmp_path):
    nested_path = tmp_path / "nested.yaml"
    nested_path.write_text(
        "pipeline:\n  degradation:\n    mode: minimal\n",
        encoding="utf-8",
    )
    assert ConfigLoader.load_file(nested_path).degradation.mode == "minimal"

    top_level_path = tmp_path / "top-level.yaml"
    top_level_path.write_text(
        "degradation:\n  mode: degraded\n"
        "pipeline:\n  degradation:\n    mode: minimal\n",
        encoding="utf-8",
    )
    assert ConfigLoader.load_file(top_level_path).degradation.mode == "degraded"


def test_all_builtin_profiles_expose_global_and_feedback_settings():
    loader = ConfigLoader()
    for profile in ("podcast", "education", "variety_show", "music_live"):
        config = loader.load_profile(profile)
        assert config.asr.global_asr.enabled is True
        assert config.asr.global_asr.routing == "auto"
        assert config.boundary_redundancy.enabled is True
        assert config.feedback.active_profile == "user_default"
