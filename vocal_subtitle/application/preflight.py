"""Preflight checks for offline pipeline runs.

Implements the 7 checks defined in TASK_STATE_MACHINE.md §预检清单.
Used by both the pipeline runtime and the CLI ``preflight`` command.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_SUFFIXES = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".aac",
        ".ogg",
        ".opus",
        ".wma",
        ".webm",
        ".mp4",
        ".mkv",
        ".mov",
    }
)

MIN_FREE_DISK_BYTES = 512 * 1024 * 1024  # 512 MB
DISK_OVERHEAD_FACTOR = 3.0  # per TASK_STATE_MACHINE.md: 输出预估 × 3


@dataclass
class PreflightCheck:
    """A single preflight check result."""

    key: str
    label: str
    critical: bool
    passed: bool
    reason: str = ""


@dataclass
class PreflightResult:
    """Aggregated preflight check result.

    ``passed`` is True only when ALL critical checks passed.
    Separation is non-critical only when the caller explicitly skips it.
    """

    passed: bool
    checks: list[PreflightCheck] = field(default_factory=list)
    engine_snapshot: dict = field(default_factory=dict)

    def failed_critical(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.critical and not c.passed]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "key": c.key,
                    "label": c.label,
                    "critical": c.critical,
                    "passed": c.passed,
                    "reason": c.reason,
                }
                for c in self.checks
            ],
        }


class PreflightError(RuntimeError):
    """Raised when preflight fails in enforce mode."""

    def __init__(self, result: PreflightResult) -> None:
        failed = result.failed_critical()
        reasons = "; ".join(f"{c.label}: {c.reason}" for c in failed)
        super().__init__(f"Preflight failed ({len(failed)} critical): {reasons}")
        self.result = result


def _error_category_from_checks(checks: list[PreflightCheck]) -> str:
    """Derive an ErrorCategory from the first failed critical check."""
    category_map = {
        "input_exists": "input_missing",
        "format_supported": "format_unsupported",
        "separation_available": "separation_unavailable",
        "vad_available": "vad_unavailable",
        "asr_available": "asr_unavailable",
        "disk_space": "disk_space",
        "output_writable": "output_unwritable",
    }
    for c in checks:
        if c.critical and not c.passed:
            return category_map.get(c.key, "unrecoverable_failure")
    return "unrecoverable_failure"


def run_preflight_checks(
    input_path: Path,
    output_path: Path,
    config,  # PipelineConfig (lazy to avoid circular import)
    skip_separation: bool = False,
) -> PreflightResult:
    """Run the 7 preflight checklist checks.

    Args:
        input_path: Path to the input audio file.
        output_path: Expected output path (for disk/writability checks).
        config: PipelineConfig instance.

    Returns:
        PreflightResult with pass/fail for each check.
    """
    from ..reporting.engine_availability import EngineAvailabilityChecker

    checks: list[PreflightCheck] = []

    # ---- Sync EngineRegistry lifecycle states from real availability ----
    _sync_lifecycle_from_availability(config)

    # ---- 1. input_exists ----
    exists = input_path.exists() and os.access(input_path, os.R_OK)
    checks.append(
        PreflightCheck(
            key="input_exists",
            label="输入文件存在且可读",
            critical=True,
            passed=exists,
            reason="" if exists else f"文件不存在或不可读: {input_path}",
        )
    )

    # ---- 2. format_supported ----
    suffix = input_path.suffix.lower()
    fmt_ok = suffix in SUPPORTED_AUDIO_SUFFIXES
    checks.append(
        PreflightCheck(
            key="format_supported",
            label="音频格式支持",
            critical=True,
            passed=fmt_ok,
            reason=""
            if fmt_ok
            else f"不支持的音频格式: {suffix}，支持 {sorted(SUPPORTED_AUDIO_SUFFIXES)}",
        )
    )

    # ---- 3-5. engine availability ----
    checker = EngineAvailabilityChecker(config)
    snapshot = checker.check_all()
    critical = checker.check_critical()

    # Separation is optional only when the caller explicitly says that the
    # input is already a vocals track.  Otherwise the configured engine must
    # be ready; an unrelated ready engine must not mask a bad configuration.
    sep_key = {
        "uvr": "separation_uvr",
        "openunmix": "separation_open_unmix",
        "spleeter": "separation_spleeter",
    }.get(getattr(config.separation, "engine", ""))
    selected_sep = snapshot.entries.get(sep_key) if sep_key else None
    sep_ready = bool(selected_sep and selected_sep.status.startswith("ready"))
    sep_reason = ""
    if skip_separation:
        sep_ready = True
        sep_reason = "已按请求跳过人声分离"
    elif selected_sep is None:
        sep_reason = f"未知的分离引擎: {getattr(config.separation, 'engine', '')}"
    else:
        sep_reason = selected_sep.reason or f"状态: {selected_sep.status}"
    checks.append(
        PreflightCheck(
            key="separation_available",
            label="分离引擎可用",
            critical=not skip_separation,
            passed=sep_ready,
            reason="" if sep_ready else sep_reason,
        )
    )

    # critical: VAD
    checks.append(
        PreflightCheck(
            key="vad_available",
            label="VAD 引擎可用",
            critical=True,
            passed=critical["vad_available"],
            reason=""
            if critical["vad_available"]
            else "缺少 VAD 引擎（Silero VAD 或 WebRTC VAD）",
        )
    )

    # critical: ASR
    checks.append(
        PreflightCheck(
            key="asr_available",
            label="至少一个 ASR 引擎可用",
            critical=True,
            passed=critical["asr_available"],
            reason="" if critical["asr_available"] else "缺少可用的 ASR 引擎",
        )
    )

    # ---- 6. disk_space ----
    try:
        input_size = input_path.stat().st_size
    except OSError:
        input_size = 0
    required = max(MIN_FREE_DISK_BYTES, int(input_size * DISK_OVERHEAD_FACTOR))
    try:
        usage = shutil.disk_usage(output_path.parent)
        disk_ok = usage.free >= required
    except OSError:
        disk_ok = True  # can't check; assume ok
    checks.append(
        PreflightCheck(
            key="disk_space",
            label="磁盘空间充足",
            critical=True,
            passed=disk_ok,
            reason=""
            if disk_ok
            else f"磁盘空间不足（需要 ≥ {required // 1024**2} MB）",
        )
    )

    # ---- 7. output_writable ----
    try:
        output_parent = output_path.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(output_parent, os.W_OK)
    except OSError:
        writable = False
    checks.append(
        PreflightCheck(
            key="output_writable",
            label="输出目录可写",
            critical=True,
            passed=writable,
            reason="" if writable else f"输出目录不可写: {output_parent}",
        )
    )

    all_critical_pass = all(c.passed for c in checks if c.critical)
    return PreflightResult(
        passed=all_critical_pass,
        checks=checks,
        engine_snapshot={k: e.status for k, e in snapshot.entries.items()},
    )


def _sync_lifecycle_from_availability(config) -> None:
    """根据实际的引擎可用性检查结果，同步 EngineRegistry 中的生命周期状态。

    确保 EngineRegistry 反映运行时实际状态，而非仅代码中的默认值。
    仅用于日志/报告，失败时静默降级。
    """
    try:
        from ..governance.engine_lifecycle import (
            EngineLifecycle,
            EngineRegistry,
            LifecycleManager,
        )
        from ..reporting.engine_availability import EngineAvailabilityChecker

        checker = EngineAvailabilityChecker(config)
        snapshot = checker.check_all()

        # 引擎键名映射：EngineAvailabilityChecker key → EngineRegistry key
        key_map = {
            "separation_uvr": "uvr",
            "separation_spleeter": "spleeter",
            "separation_open_unmix": "open-unmix",
            "vad_silero": "silero",
            "vad_webrtc": "webrtc",
            "asr_faster_whisper": "faster-whisper",
            "asr_funasr": "funasr",
            "asr_qwen": "qwen-asr",
            "asr_whisper_cpp": "whisper.cpp",
            "review_global_evidence": "global-asr-evidence",
            "review_context_reasr": "context-reasr",
            "review_qwen": "qwen-review",
            "review_forced_aligner": "forced-aligner",
            "review_sed": "sed",
            "review_semantic": "semantic-review",
            "diarization_speechbrain": "speechbrain-ecapa",
            "diarization_pyannote": "pyannote",
        }

        registry = EngineRegistry()
        lm = LifecycleManager(registry)

        for checker_key, entry in snapshot.entries.items():
            registry_key = key_map.get(checker_key)
            if not registry_key:
                continue
            eng = registry.get(registry_key)
            if eng is None:
                continue

            # 映射到生命周期状态
            status_map = {
                "unavailable": EngineLifecycle.UNAVAILABLE,
                "model_missing": EngineLifecycle.MODEL_MISSING,
                "ready_shadow": EngineLifecycle.READY_SHADOW,
                "ready_review": EngineLifecycle.READY_REVIEW,
                "ready_default": EngineLifecycle.READY_DEFAULT,
            }
            target = status_map.get(entry.status)
            if target is None:
                continue

            if eng.status != target:
                try:
                    lm.transition(
                        registry_key,
                        target,
                        reason=f"Synced from availability check: {entry.reason}"
                        if entry.reason
                        else "Synced from availability check",
                        force=True,
                    )
                except ValueError:
                    pass  # 跳过不合法转换
    except Exception:
        pass  # 非致命操作
