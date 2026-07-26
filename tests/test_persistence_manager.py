"""Persistence output regression tests."""

import json
from pathlib import Path

from vocal_subtitle.utils.persistence_manager import (
    PersistenceManager,
    PersistenceSettings,
)


def test_persist_task_generates_final_subtitle_files(tmp_path: Path):
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    manager = PersistenceManager(
        settings_path=tmp_path / "settings.json",
        files_dir=tmp_path / "files",
    )

    manifest = manager.persist_task(
        "task-1",
        {
            "subtitle_path": str(source),
            "events": [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                }
            ],
        },
        PersistenceSettings(
            persist_asr_subtitle=False,
            persist_llm_subtitle=False,
            persist_final_ass=True,
            persist_final_srt=True,
            persist_vocals=False,
            persist_accompaniment=False,
        ),
    )

    labels = {item["label"] for item in manifest["files"]}
    assert {"最终SRT字幕", "最终ASS字幕"} <= labels
    assert (tmp_path / "files" / "task-1" / "final.srt").read_text(encoding="utf-8")
    assert (tmp_path / "files" / "task-1" / "final.ass").read_text(encoding="utf-8")
    persisted_manifest = json.loads(
        (tmp_path / "files" / "task-1" / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(persisted_manifest["files"]) == 2
