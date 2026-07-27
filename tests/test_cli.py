"""测试 CLI 命令层"""

import pytest
from click.testing import CliRunner

from vocal_subtitle.cli import main


class TestCLI:
    """CLI 命令测试"""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_version(self, runner):
        """--version 选项"""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_profiles(self, runner):
        """profiles 命令"""
        result = runner.invoke(main, ["profiles"])
        assert result.exit_code == 0
        assert "default" in result.output
        assert "podcast" in result.output
        assert "education" in result.output
        assert "variety_show" in result.output
        assert "music_live" in result.output

    def test_info(self, runner):
        """info 命令"""
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "系统信息" in result.output or "System" in result.output or "GPU" in result.output or "操作系统" in result.output

    def test_download_models_no_args(self, runner):
        """download-models 无参数"""
        result = runner.invoke(main, ["download-models"])
        assert result.exit_code == 0

    def test_download_models_all(self, runner, monkeypatch):
        """download-models --all"""
        import vocal_subtitle.diarization.model_registry as registry

        monkeypatch.setattr(
            registry,
            "list_model_status",
            lambda: [{"model_id": "speechbrain-ecapa", "model_ref": "fake/ref"}],
        )
        monkeypatch.setattr(
            registry,
            "download_model",
            lambda model_id, token=None: {
                "status": "ready",
                "model_ref": "fake/ref",
            },
        )
        result = runner.invoke(main, ["download-models", "--all"])
        assert result.exit_code == 0

    def test_download_models_with_asr(self, runner):
        """download-models --asr-model"""
        result = runner.invoke(main, ["download-models", "--asr-model", "large-v3"])
        assert result.exit_code == 0

    def test_run_missing_file(self, runner):
        """run 命令 — 文件不存在"""
        result = runner.invoke(main, ["run", "/nonexistent/file.mp3"])
        assert result.exit_code != 0

    def test_run_help_shows_options(self, runner):
        """run --help 显示所有选项"""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--profile" in result.output
        assert "--separator" in result.output
        assert "--language" in result.output
        assert "--format" in result.output
        assert "--skip-separation" in result.output
        assert "--expected-speakers" in result.output
        assert "--speaker-fusion" in result.output
        assert "--global-diarization-model" in result.output
        assert "--speaker-diarization-scope" in result.output

    def test_batch_help(self, runner):
        """batch --help"""
        result = runner.invoke(main, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--profile" in result.output
        assert "--pattern" in result.output

    def test_run_with_profile(self, runner):
        """run 指定 profile 但文件不存在"""
        result = runner.invoke(
            main, ["run", "/nonexistent/test.mp3", "--profile", "podcast", "--language", "zh"]
        )
        assert result.exit_code != 0  # 文件不存在会报错

    def test_run_with_format_option(self, runner):
        """run 指定输出格式"""
        result = runner.invoke(
            main, ["run", "/nonexistent/test.mp3", "--format", "vtt"]
        )
        assert result.exit_code != 0  # 文件不存在

    def test_batch_nonexistent_dir(self, runner):
        """batch 目录不存在"""
        result = runner.invoke(main, ["batch", "/nonexistent/dir/"])
        assert result.exit_code != 0

    def test_main_group_help(self, runner):
        """主命令帮助信息"""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "batch" in result.output
        assert "profiles" in result.output
        assert "info" in result.output
        assert "download-models" in result.output

    def test_download_models_lists_speaker_models(self, runner):
        result = runner.invoke(main, ["download-models", "--list-speaker-models"])
        assert result.exit_code == 0
        assert "speechbrain-ecapa" in result.output
        assert "community-1" in result.output
