#!/usr/bin/env python3
"""发布检查单自动化脚本

自动检查 RELEASE_GOVERNANCE.md §3 定义的发布前检查项。
生成 release-check-report.json。

使用:
    python scripts/release_check.py
    python scripts/release_check.py --check preflight,regression
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check(msg: str) -> tuple[bool, str]:
    """轻量级检查包装。"""
    return True, msg


def check_pyproject_lock() -> dict:
    """检查依赖锁定状态。"""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    uv_lock = PROJECT_ROOT / "uv.lock"
    issues = []
    if not pyproject.exists():
        issues.append("pyproject.toml not found")
    if not uv_lock.exists():
        issues.append("uv.lock not found")
    return {
        "name": "Dependency lock",
        "passed": len(issues) == 0,
        "issues": issues,
    }


def check_configs_exist() -> dict:
    """检查配置文件。"""
    config_dir = PROJECT_ROOT / "configs"
    required = ["default.yaml"]
    missing = [f for f in required if not (config_dir / f).exists()]
    return {
        "name": "Config files present",
        "passed": len(missing) == 0,
        "issues": missing,
    }


def check_install_script() -> dict:
    """检查安装脚本存在。"""
    script = PROJECT_ROOT / "install.sh"
    return {
        "name": "Install script present",
        "passed": script.exists(),
        "issues": [] if script.exists() else ["install.sh not found"],
    }


def check_models_status() -> dict:
    """检查模型可用性（faster-whisper, Silero VAD）。"""
    issues = []
    # faster-whisper
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        issues.append("faster-whisper not installed")

    # Silero VAD
    try:
        import onnxruntime  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        issues.append("torch/onnxruntime not installed (Silero VAD requires)")

    return {
        "name": "Model availability (core)",
        "passed": len(issues) == 0,
        "issues": issues,
    }


def check_python_version() -> dict:
    """检查 Python 版本。"""
    v = sys.version_info
    ok = v >= (3, 10)
    issues = [] if ok else [f"Python {v.major}.{v.minor} < 3.10"]
    return {
        "name": "Python >= 3.10",
        "passed": ok,
        "issues": issues,
        "version": f"{v.major}.{v.minor}.{v.micro}",
    }


def check_output_dirs() -> dict:
    """检查输出目录可写。"""
    test_dirs = [
        PROJECT_ROOT / "cache",
        PROJECT_ROOT / "logs",
    ]
    issues = []
    for d in test_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.write_text("ok")
            probe.unlink()
        except (OSError, PermissionError):
            issues.append(f"Directory not writable: {d}")
    return {
        "name": "Output directories writable",
        "passed": len(issues) == 0,
        "issues": issues,
    }


def run_tests(test_pattern: str = "") -> dict:
    """运行 pytest。"""
    try:
        cmd = [sys.executable, "-m", "pytest", test_pattern, "-x", "--tb=short"]
        if test_pattern:
            pass  # will fail without pattern
        else:
            # Run core tests only (skip slow)
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(PROJECT_ROOT / "tests" / "test_cli.py"),
                str(PROJECT_ROOT / "tests" / "test_asr" / "test_router.py"),
                "-x",
                "--tb=short",
            ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180, cwd=str(PROJECT_ROOT)
        )
        passed = result.returncode == 0
        return {
            "name": f"Core tests ({test_pattern or 'test_cli + test_router'})",
            "passed": passed,
            "issues": []
            if passed
            else [result.stdout[-500:] if result.stdout else result.stderr[-500:]],
        }
    except subprocess.TimeoutExpired:
        return {
            "name": "Core tests",
            "passed": False,
            "issues": ["Test timeout (180s)"],
        }
    except Exception as e:
        return {"name": "Core tests", "passed": False, "issues": [str(e)]}


def check_cli_smoke() -> dict:
    """检查 CLI 基础功能。"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "vocal_subtitle.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        return {
            "name": "CLI smoke (--help)",
            "passed": result.returncode == 0,
            "issues": [] if result.returncode == 0 else [result.stderr[:500]],
        }
    except Exception as e:
        return {"name": "CLI smoke", "passed": False, "issues": [str(e)]}


def check_readme() -> dict:
    """检查 README 存在。"""
    readme = PROJECT_ROOT / "README.md"
    return {
        "name": "README present",
        "passed": readme.exists(),
        "issues": [] if readme.exists() else ["README.md not found"],
    }


def check_architecture_state() -> dict:
    """检查架构状态文档。"""
    state = PROJECT_ROOT / "docs" / "20260802" / "ARCHITECTURE_STATE.md"
    return {
        "name": "ARCHITECTURE_STATE.md present",
        "passed": state.exists(),
        "issues": []
        if state.exists()
        else ["docs/20260802/ARCHITECTURE_STATE.md not found"],
    }


def run_all_checks() -> dict[str, Any]:
    """运行所有发布前检查。"""
    checks = []

    checks.append(check_python_version())
    checks.append(check_pyproject_lock())
    checks.append(check_configs_exist())
    checks.append(check_install_script())
    checks.append(check_models_status())
    checks.append(check_output_dirs())
    checks.append(check_readme())
    checks.append(check_architecture_state())

    # 需要网络/时间的检查
    checks.append(check_cli_smoke())

    try:
        checks.append(run_tests())
    except Exception:
        checks.append(
            {"name": "Core tests", "passed": False, "issues": ["Test runner error"]}
        )

    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)
    all_pass = passed == total

    report = {
        "version": "release-check-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "all_pass": all_pass,
        },
        "recommendation": ("production-usable" if all_pass else "issues-found"),
        "checks": checks,
    }

    return report


def main() -> None:
    """运行检查并输出报告。"""
    print("=" * 60)
    print("  Vocal Subtitle — Release Checklist")
    print("=" * 60)
    print()

    report = run_all_checks()

    for check_item in report["checks"]:
        status = "✅" if check_item["passed"] else "❌"
        print(f"  {status} {check_item['name']}")
        for issue in check_item.get("issues", []):
            print(f"     → {issue}")
        print()

    print("-" * 60)
    summary = report["summary"]
    print(f"  Result: {summary['passed']}/{summary['total']} passed")
    print(f"  Recommendation: {report['recommendation']}")
    print("-" * 60)

    # 持久化
    output_path = PROJECT_ROOT / "cache" / "release_check_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Report saved: {output_path}")

    sys.exit(0 if report["summary"]["all_pass"] else 1)


if __name__ == "__main__":
    main()
