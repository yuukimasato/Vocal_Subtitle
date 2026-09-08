"""Regression tests for release-check command construction."""

import subprocess
import sys
from pathlib import Path

from scripts import release_check

ROOT = Path(__file__).resolve().parents[1]


def test_run_tests_uses_only_available_pytest_options(monkeypatch):
    observed = {}

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return _Completed()

    monkeypatch.setattr(release_check.subprocess, "run", fake_run)

    result = release_check.run_tests()

    assert result["passed"] is True
    assert "--timeout=120" not in observed["command"]
    assert observed["kwargs"]["timeout"] == 180


def test_public_compatibility_script_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/check_public_compatibility.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
