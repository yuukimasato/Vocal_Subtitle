#!/usr/bin/env python3
"""高精度时间轴与识别文本优化 — 基线回归快照。

运行两组基线测试并把 HEAD、工作树 diff stat、ASR/声学校验关键配置值、
测试摘要和退出码写入 ``test/baselines/high_precision_baseline.json``。
任何一组测试失败时以非零退出码结束。

用法::

    python scripts/capture_high_precision_baseline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "test" / "baselines" / "high_precision_baseline.json"

BASELINE_TEST_GROUPS = {
    "release_gate_group": [
        "tests/test_release_check.py",
        "tests/test_golden_quality_gate.py",
        # 实施计划中的 tests/test_global_pipeline_contract.py 尚不存在，
        # 以现有的全局链路契约测试代替。
        "tests/test_global_asr_path.py",
    ],
    "asr_physical_mapping_group": [
        "tests/test_asr",
        "tests/test_physical",
        "tests/test_mapping",
    ],
}

CONFIG_SNAPSHOT_KEYS = {
    "asr.engine": ("asr", "engine"),
    "asr.model": ("asr", "model"),
    "asr.word_timestamps": ("asr", "word_timestamps"),
    "asr.global_asr.enabled": ("asr", "global_asr", "enabled"),
    "asr.global_asr.routing": ("asr", "global_asr", "routing"),
    "asr.global_asr.evidence_enabled": ("asr", "global_asr", "evidence_enabled"),
    "evidence_review.enabled": ("evidence_review", "enabled"),
    "evidence_review.shadow_mode": ("evidence_review", "shadow_mode"),
    "evidence_review.context_reasr_enabled": (
        "evidence_review",
        "context_reasr_enabled",
    ),
    "acoustic_validation.enabled": ("acoustic_validation", "enabled"),
    "acoustic_validation.timeline_arbitration": (
        "acoustic_validation",
        "timeline_arbitration",
    ),
    "acoustic_validation.max_snap_distance": (
        "acoustic_validation",
        "max_snap_distance",
    ),
    "acoustic_validation.max_start_snap_distance": (
        "acoustic_validation",
        "max_start_snap_distance",
    ),
    "diarization.early_turns": ("diarization", "early_turns"),
    "diarization.word_split_on_turn": ("diarization", "word_split_on_turn"),
}


def _load_pipeline_config() -> dict:
    import yaml

    with open(REPO_ROOT / "configs" / "default.yaml", encoding="utf-8") as fh:
        document = yaml.safe_load(fh)
    return document.get("pipeline", document)


def _dig(config: dict, path: tuple[str, ...]):
    node = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _run_pytest(group: str, paths: list[str]) -> dict:
    command = [sys.executable, "-m", "pytest", *paths, "-q"]
    result = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    stdout_lines = (result.stdout or "").splitlines()
    summary = stdout_lines[-1] if stdout_lines else ""
    # 记录到仓库的快照不应携带本机绝对路径：家目录前缀统一脱敏为 ~
    home = os.path.expanduser("~")
    command_repr = " ".join(command).replace(home, "~", 1)
    return {
        "command": command_repr,
        "exit_code": result.returncode,
        "summary": summary,
        "passed": result.returncode == 0,
    }


def main() -> int:
    config = _load_pipeline_config()
    config_snapshot = {
        name: _dig(config, path) for name, path in CONFIG_SNAPSHOT_KEYS.items()
    }

    groups = {
        name: _run_pytest(name, paths) for name, paths in BASELINE_TEST_GROUPS.items()
    }

    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": "d3d49d1",
        "head": _git(["rev-parse", "HEAD"]),
        "worktree_diff_stat": _git(["diff", "--stat"]),
        "worktree_status": _git(["status", "--short"]),
        "config_snapshot": config_snapshot,
        "test_groups": groups,
        "all_passed": all(group["passed"] for group in groups.values()),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"baseline snapshot written to {OUTPUT_PATH}")
    for name, group in groups.items():
        print(f"  {name}: exit={group['exit_code']} {group['summary']}")
    return 0 if snapshot["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
