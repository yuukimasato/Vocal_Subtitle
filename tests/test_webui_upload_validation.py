"""WebUI 上传校验回归（2026-09-15 修复）。

用户把 .ass 字幕文件直接上传到 /api/run，服务端无格式校验，
预检 report 模式只告警放行，最终 pydub 在无音频流文件上抛出
晦涩的 IndexError。修复后：

- 服务端提交阶段拒绝非音频/视频扩展名（HTTP 400 + 明确中文提示）；
- AudioUtils.load_audio 对无音频流文件抛出可读错误而非 IndexError。
"""

import asyncio
from pathlib import Path

import pytest

from vocal_subtitle.utils.audio_utils import AudioUtils
from vocal_subtitle.webui import api, pipeline_tasks
from vocal_subtitle.webui.pipeline_tasks import PipelineSubmissionError


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def submit_env(monkeypatch, tmp_path, isolated_home):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(api, "_task_store", {})
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(api, "_task_history", _HistorySpy())
    return upload_dir


class _HistorySpy:
    def __init__(self):
        self.created = None

    def find_by_hash(self, file_hash, config_hash):
        return None

    def create(self, **kwargs):
        self.created = kwargs


def _submit(service, contents, filename):
    return asyncio.run(
        service.submit(
            contents,
            filename,
            profile="default",
            output_format="srt",
            skip_separation=True,
            overrides_text="{}",
            thread_target=lambda *args: None,
        )
    )


def test_submit_rejects_subtitle_file(submit_env):
    """.ass 字幕文件上传 → 400 明确报错，不创建任务、不启动线程"""
    service = pipeline_tasks.PipelineTaskService()

    with pytest.raises(PipelineSubmissionError) as exc_info:
        _submit(
            service,
            b"[Script Info]\nDialogue: 0,0:00:00.19,0:00:01.74,Default,,0,0,0,,test\n",
            "中文朗读测试.ass",
        )

    assert exc_info.value.status_code == 400
    assert ".ass" in exc_info.value.detail
    assert not api._task_store, "被拒绝的文件不应创建任务条目"
    assert api._task_history.created is None, "被拒绝的文件不应写入历史记录"


def test_submit_rejects_srt_and_text_files(submit_env):
    service = pipeline_tasks.PipelineTaskService()

    for filename in ("subtitles.srt", "notes.txt"):
        with pytest.raises(PipelineSubmissionError) as exc_info:
            _submit(service, b"1\n00:00:00,000 --> 00:00:01,000\ntest\n", filename)
        assert exc_info.value.status_code == 400


def test_submit_still_accepts_audio_and_video(submit_env, monkeypatch, tmp_path):
    """合法音频/视频扩展名不被误伤（线程目标被调用 = 校验放行）"""
    launched = []
    service = pipeline_tasks.PipelineTaskService()

    def _spy_thread(task_id, *args):
        launched.append(task_id)

    def _fake_extract(video_path, output_dir, output_name="extracted_audio"):
        extracted = output_dir / f"{output_name}.wav"
        extracted.write_bytes(b"RIFF")
        return extracted

    monkeypatch.setattr(AudioUtils, "extract_audio_from_video", _fake_extract)

    asyncio.run(
        service.submit(
            b"RIFF",
            "clip.wav",
            profile="default",
            output_format="srt",
            skip_separation=True,
            overrides_text="{}",
            thread_target=_spy_thread,
        )
    )
    asyncio.run(
        service.submit(
            b"\x1aE\xdf\xa3",
            "movie.mkv",
            profile="default",
            output_format="srt",
            skip_separation=True,
            overrides_text="{}",
            thread_target=_spy_thread,
        )
    )

    assert len(launched) == 2


def test_load_audio_raises_clear_error_for_text_file(tmp_path):
    """无音频流的文件 → 可读 ValueError，而非 pydub 的 IndexError"""
    text_file = tmp_path / "input.ass"
    text_file.write_text("[Script Info]\nDialogue: line\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        AudioUtils.load_audio(text_file)

    assert "音频" in str(exc_info.value)
