#!/usr/bin/env python3
"""Select the PyTorch runtime used by the offline WhisperX installer.

The probe intentionally runs before PyTorch is installed.  It only relies on
the operating system and ``nvidia-smi`` so it is safe in a fresh virtualenv
and easy to exercise in CI.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys


def platform_name() -> str:
    """Return a stable, human-readable platform identifier."""

    system = sys.platform
    if system.startswith("linux"):
        system = "linux"
    elif system == "darwin":
        system = "darwin"
    elif system.startswith("win"):
        system = "windows"
    else:
        system = "other"
    return f"{system}-{platform.machine().lower() or 'unknown'}"


def nvidia_smi_available() -> tuple[bool, str]:
    """Probe the NVIDIA driver without importing torch."""

    override = os.environ.get("VOCAL_SUBTITLE_NVIDIA_SMI")
    if override:
        try:
            command = shlex.split(override)
        except ValueError:
            return False, "nvidia_smi_failed"
        if not command:
            return False, "nvidia_smi_failed"
    else:
        executable = shutil.which("nvidia-smi")
        if not executable:
            return False, "nvidia_smi_missing"
        command = [executable]

    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "nvidia_smi_failed"
    if result.returncode != 0:
        return False, "nvidia_smi_failed"
    return True, "nvidia_smi"


def detect_runtime() -> tuple[str, str, str]:
    """Return ``(target, reason, platform)`` for the installer."""

    current_platform = platform_name()
    if current_platform.startswith("linux-"):
        available, reason = nvidia_smi_available()
        if available:
            return "cuda", reason, current_platform
        return "cpu", reason, current_platform
    if current_platform.startswith("darwin-"):
        return "cpu", "macos_mps_uses_pytorch_cpu_wheel", current_platform
    return "cpu", "unsupported_platform", current_platform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", action="store_true", help="print pipe-delimited output")
    parser.add_argument("--platform", action="store_true", help="print platform only")
    args = parser.parse_args()

    target, reason, current_platform = detect_runtime()
    if args.platform:
        print(current_platform)
    elif args.line:
        print(f"{target}|{reason}|{current_platform}")
    else:
        print(f"target={target} reason={reason} platform={current_platform}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
