"""Regression tests for the source and Debian installation entry points."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_runtime_target_cpu_probe_is_machine_readable() -> None:
    result = run("python3", "scripts/runtime_target.py", "--line")
    assert result.returncode == 0, result.stdout
    target, reason, platform_name = result.stdout.strip().split("|")
    assert target in {"cpu", "cuda"}
    assert reason
    assert platform_name


def test_runtime_target_can_simulate_nvidia_smi() -> None:
    env = os.environ.copy()
    env["VOCAL_SUBTITLE_NVIDIA_SMI"] = "python3 -c 'print(\"GPU 0\")'"
    result = run("python3", "scripts/runtime_target.py", "--line", env=env)
    assert result.returncode == 0, result.stdout
    assert result.stdout.startswith("cuda|nvidia_smi|")


def test_runtime_target_failed_probe_falls_back_to_cpu() -> None:
    env = os.environ.copy()
    env["VOCAL_SUBTITLE_NVIDIA_SMI"] = "python3 -c 'raise SystemExit(1)'"
    result = run("python3", "scripts/runtime_target.py", "--line", env=env)
    assert result.returncode == 0, result.stdout
    assert result.stdout.startswith("cpu|nvidia_smi_failed|")


def test_install_script_is_noninteractive_for_help() -> None:
    result = run("bash", "install.sh", "--help")
    assert result.returncode == 0, result.stdout
    assert "--cpu" in result.stdout
    assert "--no-torch" in result.stdout


def test_runtime_metadata_schema_is_json_compatible(tmp_path: Path) -> None:
    metadata = {
        "runtime": "whisperx",
        "target": "cpu",
        "reason": "nvidia_smi_missing",
        "platform": "linux-x86_64",
    }
    path = tmp_path / ".vocal-subtitle-runtime.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["runtime"] == "whisperx"
