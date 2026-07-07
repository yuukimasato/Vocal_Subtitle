"""管道集成测试"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vocal_subtitle.config import ConfigLoader, PipelineConfig
from vocal_subtitle.pipeline import Pipeline


class TestPipelineConfig:
    """配置加载测试"""

    def test_load_default_config(self):
        config = ConfigLoader().load_profile("default")
        assert isinstance(config, PipelineConfig)
        assert config.separation.engine == "uvr"
        assert config.vad.engine == "silero"
        assert config.asr.engine == "faster-whisper"

    def test_load_podcast_config(self):
        config = ConfigLoader().load_profile("podcast")
        assert config.separation.engine == "uvr"
        # podcast 配置默认使用自动语言检测（language=None），
        # 用户可通过 --language 参数或取消配置文件中 language 行的注释来锁定语言
        assert config.asr.language is None

    def test_load_education_config(self):
        config = ConfigLoader().load_profile("education")
        assert config.merging.min_silence_gap == 0.6

    def test_list_profiles(self):
        loader = ConfigLoader()
        profiles = loader.list_profiles()
        assert "default" in profiles
        assert "podcast" in profiles
        assert "education" in profiles
        assert "variety_show" in profiles
        assert "music_live" in profiles

    def test_merge_overrides(self):
        loader = ConfigLoader()
        config = loader.load_profile("default")
        config = loader.merge_with_overrides(
            config,
            separator="uvr",
            language="ja",
            vad_threshold=0.3,
        )
        assert config.separation.engine == "uvr"
        assert config.asr.language == "ja"
        assert config.vad.threshold == 0.3

    def test_load_nonexistent_profile(self):
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_profile("nonexistent")


class TestPipeline:
    """管道集成测试"""

    def test_pipeline_creation(self):
        """管道创建"""
        pipeline = Pipeline()
        assert pipeline.config is not None
        assert pipeline.config.separation.engine == "uvr"

    def test_pipeline_with_config(self):
        config = ConfigLoader().load_profile("podcast")
        pipeline = Pipeline(config)
        assert pipeline.config.separation.engine == "uvr"

    def test_pipeline_stats(self):
        from vocal_subtitle.pipeline import PipelineStats

        stats = PipelineStats(
            input_path=Path("test.mp3"),
            duration_seconds=120.0,
            stage_timings={"separation": 10.0, "vad": 0.5, "asr": 8.0},
        )
        d = stats.to_dict()
        assert d["duration_seconds"] == 120.0
        assert d["segment_count"] == 0


class TestPipelineDiarizationConfig:
    """管道配置 — 说话人分离与角色标注"""

    def test_diarization_enabled_by_default(self):
        """回归验证：默认配置中 diarization 已开启（说话人分离），speaker_role 保持关闭"""
        config = ConfigLoader().load_profile("default")
        assert config.diarization.enabled is True, (
            "Diarization must be ON by default for speaker detection"
        )
        assert config.speaker_role.enabled is False, (
            "Speaker role labeling (LLM) must be OFF by default"
        )

    def test_diarization_config_fields_parsed(self):
        """验证所有 diarization 配置字段正确解析"""
        config = ConfigLoader().load_profile("podcast")
        assert config.diarization.enabled is True
        assert config.diarization.engine == "agglomerative"
        assert config.diarization.distance_threshold == 0.5
        assert config.diarization.min_speakers == 1
        assert config.diarization.max_speakers == 10
        assert config.diarization.use_pca is True
        assert config.diarization.pca_variance == 0.95

    def test_speaker_role_config_fields_parsed(self):
        """验证所有 speaker_role 配置字段正确解析"""
        config = ConfigLoader().load_profile("podcast")
        assert config.speaker_role.enabled is True
        assert config.speaker_role.model == "deepseek-v4-pro"
        assert config.speaker_role.temperature == 0.2

    def test_variety_show_diarization_enabled(self):
        """综艺模板默认启用说话人分离和角色标注"""
        config = ConfigLoader().load_profile("variety_show")
        assert config.diarization.enabled is True
        assert config.speaker_role.enabled is True

    def test_education_diarization_enabled(self):
        """教育模板默认启用说话人分离（多人讨论/问答也常见于教学场景）"""
        config = ConfigLoader().load_profile("education")
        assert config.diarization.enabled is True

    def test_merge_overrides_diarization(self):
        """CLI --diarization 和 --speaker-role 覆盖生效"""
        loader = ConfigLoader()
        config = loader.load_profile("default")

        # 模拟 CLI 传入 --diarization --speaker-role
        overridden = loader.merge_with_overrides(
            config, diarization=True, speaker_role=True,
        )
        assert overridden.diarization.enabled is True
        assert overridden.speaker_role.enabled is True

    def test_pipeline_stats_speaker_fields(self):
        """PipelineStats 包含 speaker_count 和 diarization_silhouette 字段"""
        from vocal_subtitle.pipeline import PipelineStats

        stats = PipelineStats(
            input_path=Path("test.mp3"),
            duration_seconds=60.0,
        )
        assert hasattr(stats, "speaker_count")
        assert stats.speaker_count == 0
        assert hasattr(stats, "diarization_silhouette")
        assert stats.diarization_silhouette is None

    def test_diarization_config_defaults(self):
        """DiarizationConfig 默认值正确"""
        from vocal_subtitle.config import DiarizationConfig

        cfg = DiarizationConfig()
        assert cfg.enabled is True  # Phase 3: 默认启用说话人分离
        assert cfg.engine == "agglomerative"
        assert cfg.distance_threshold == 0.5
        assert cfg.min_speakers == 1
        assert cfg.max_speakers == 10

    def test_speaker_role_config_defaults(self):
        """SpeakerRoleConfig 默认值正确"""
        from vocal_subtitle.config import SpeakerRoleConfig

        cfg = SpeakerRoleConfig()
        assert cfg.enabled is False
        assert cfg.model == "deepseek-v4-pro"
        assert cfg.temperature == 0.2
        assert cfg.context_hint is None
