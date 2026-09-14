"""TTS 清晰人声场景 profile(高精度方案 Task 8 / 优化方案 §10)。

- ``configs/tts_clean.yaml`` 启用骨架优先、时间轴仲裁、重投影与双向修正;
- 骨架优先模式下 cue 的合法范围由物理骨架段决定,词级只做段内细化,
  骨架间静音是硬边界不得跨越;
- 通用配置(不带 ``skeleton_priority``)行为保持不变。
"""

from pathlib import Path

import yaml

from vocal_subtitle.config import ConfigLoader

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pipeline(name):
    with open(REPO_ROOT / "configs" / name, encoding="utf-8") as fh:
        document = yaml.safe_load(fh)
    return document.get("pipeline", document)


def test_tts_clean_profile_enables_skeleton_priority():
    pipeline = _pipeline("tts_clean.yaml")

    acoustic = pipeline["acoustic_validation"]
    assert acoustic["enabled"] is True
    assert acoustic["skeleton_priority"] is True
    assert acoustic["skeleton_mode"] is True
    assert acoustic["timeline_arbitration"] is True
    assert acoustic["reproject_grouped_windows"] is True
    assert acoustic["allow_start_pull_earlier"] is True
    assert acoustic["allow_end_shorten"] is True
    assert acoustic["allow_end_extend"] is True
    assert acoustic["max_snap_distance"] <= 0.15
    assert acoustic["max_start_snap_distance"] <= 0.20

    diarization = pipeline["diarization"]
    assert diarization["single_speaker_shortcut"] is True


def test_tts_clean_profile_loads_into_pipeline_config():
    config = ConfigLoader().load_profile("tts_clean")

    assert config.acoustic_validation.skeleton_priority is True
    assert config.acoustic_validation.timeline_arbitration is True
    assert config.acoustic_validation.reproject_grouped_windows is True


def test_default_config_keeps_skeleton_priority_off():
    config = ConfigLoader().load_profile("default")

    assert config.acoustic_validation.skeleton_priority is False
