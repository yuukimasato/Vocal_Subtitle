"""Local noise profile — rolling-window acoustic noise floor estimation.

Computes per-interval noise statistics from the physical timeline using
robust quantile-based estimators. Produces stable noise intervals with
hysteresis gating. When estimation is unreliable, falls back to configured
thresholds and marks affected candidates with reduced confidence.

This module only calibrates physical candidates; it never rewrites raw
acoustic evidence.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseInterval:
    """A stable noise-floor estimate for one time interval."""

    start: float
    end: float
    rms_median: float
    rms_mad: float
    noise_db: float  # estimated dB SPL floor
    stable: bool  # True if derived from enough clean data
    sample_count: int = 0


@dataclass
class LocalNoiseProfile:
    """Rolling-window noise floor statistics for the entire audio.

    Attributes:
        intervals: Ordered list of noise intervals covering the audio.
        fallback_db: Configured fallback noise threshold in dB.
        fallback_applied: Whether any interval used the fallback value.
        warnings: Diagnostic warnings produced during estimation.
        metadata: Additional metadata for caching/diagnostics.
    """

    intervals: List[NoiseInterval] = field(default_factory=list)
    fallback_db: float = -35.0
    fallback_applied: bool = False
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def interval_at(self, time: float) -> NoiseInterval | None:
        """Return the stable interval containing ``time``."""
        for interval in self.intervals:
            if interval.start <= time <= interval.end:
                return interval
        if self.intervals:
            return min(
                self.intervals,
                key=lambda item: min(abs(time - item.start), abs(time - item.end)),
            )
        return None

    def candidate_features(self, time: float) -> dict[str, float]:
        """Expose normalized noise features for boundary candidate auditing."""
        interval = self.interval_at(time)
        if interval is None:
            return {
                "noise_db": self.fallback_db,
                "noise_stability": 0.0,
                "noise_fallback": 1.0,
            }
        return {
            "noise_db": float(interval.noise_db),
            "noise_stability": 1.0 if interval.stable else 0.0,
            "noise_fallback": 0.0 if interval.stable else 1.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervals": [
                {
                    "start": iv.start,
                    "end": iv.end,
                    "rms_median": iv.rms_median,
                    "rms_mad": iv.rms_mad,
                    "noise_db": iv.noise_db,
                    "stable": iv.stable,
                    "sample_count": iv.sample_count,
                }
                for iv in self.intervals
            ],
            "fallback_db": self.fallback_db,
            "fallback_applied": self.fallback_applied,
            "warnings": list(self.warnings),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LocalNoiseProfile":
        return cls(
            intervals=[
                NoiseInterval(
                    start=iv["start"],
                    end=iv["end"],
                    rms_median=iv["rms_median"],
                    rms_mad=iv["rms_mad"],
                    noise_db=iv["noise_db"],
                    stable=iv.get("stable", True),
                    sample_count=iv.get("sample_count", 0),
                )
                for iv in payload.get("intervals", [])
            ],
            fallback_db=payload.get("fallback_db", -35.0),
            fallback_applied=payload.get("fallback_applied", False),
            warnings=payload.get("warnings", []),
            metadata=payload.get("metadata", {}),
        )


def _window_rms(audio: np.ndarray, win_samples: int, hop_samples: int) -> np.ndarray:
    """Compute rolling RMS energy over an audio array."""
    if len(audio) < win_samples:
        return np.array([np.sqrt(np.mean(audio.astype(np.float64) ** 2))])

    n_frames = 1 + (len(audio) - win_samples) // hop_samples
    rms_values = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_samples
        frame = audio[start:start + win_samples].astype(np.float64)
        rms_values[i] = np.sqrt(np.mean(frame ** 2))
    return np.maximum(rms_values, 1e-12)  # avoid log(0)


def _median_absolute_deviation(values: np.ndarray, scale: float = 1.4826) -> float:
    """Robust estimator of scale, scaled to match std for normal data."""
    median = np.median(values)
    return float(np.median(np.abs(values - median)) * scale)


def _rms_to_db(rms: float, ref: float = 1.0) -> float:
    """Convert linear RMS to dB (20*log10 ratio)."""
    return float(20.0 * np.log10(max(rms, 1e-12) / ref))


def estimate_noise_profile(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fallback_db: float = -35.0,
    window_sec: float = 2.0,
    hop_sec: float = 0.5,
    max_intervals: int = 128,
    min_stable_samples: int = 4,
    hysteresis_db: float = 3.0,
    min_interval_duration: float = 2.0,
    rms_floor: float = 1e-6,
    rms_ceiling: float = 0.5,
) -> LocalNoiseProfile:
    """Build a local noise profile from the audio waveform.

    Uses rolling quantile+MAD statistics to detect noise-floor changes
    with hysteresis gating. Falls back to ``fallback_db`` for intervals
    where the local sample count is insufficient.

    Args:
        audio: float32 audio array, [-1, 1].
        sample_rate: Sample rate in Hz.
        fallback_db: Fallback noise threshold when estimation is unreliable.
        window_sec: RMS window duration in seconds.
        hop_sec: RMS hop duration in seconds.
        max_intervals: Maximum number of noise intervals to produce.
        min_stable_samples: Minimum samples per interval for "stable" flag.
        hysteresis_db: Hysteresis band to suppress oscillation (±hysteresis_db).
        min_interval_duration: Minimum interval duration in seconds.
        rms_floor: Clamp RMS below this value (prevents log of zero).
        rms_ceiling: Clamp RMS above this value.

    Returns:
        A LocalNoiseProfile with ordered intervals.
    """
    warnings: List[str] = []
    duration = len(audio) / max(sample_rate, 1)

    if duration < min_interval_duration:
        warnings.append(
            f"audio too short ({duration:.1f}s < {min_interval_duration:.1f}s), "
            "using fallback noise profile"
        )
        return LocalNoiseProfile(
            intervals=[
                NoiseInterval(
                    start=0.0, end=duration,
                    rms_median=rms_floor, rms_mad=0.0,
                    noise_db=fallback_db, stable=False, sample_count=0,
                )
            ],
            fallback_db=fallback_db,
            fallback_applied=True,
            warnings=warnings,
            metadata={"duration": duration, "sample_rate": sample_rate},
        )

    win_samples = max(128, int(window_sec * sample_rate))
    hop_samples = max(32, int(hop_sec * sample_rate))

    try:
        rms = _window_rms(audio, win_samples, hop_samples)
    except Exception:
        warnings.append("RMS computation failed, using fallback")
        return LocalNoiseProfile(
            intervals=[
                NoiseInterval(
                    start=0.0, end=duration,
                    rms_median=rms_floor, rms_mad=0.0,
                    noise_db=fallback_db, stable=False, sample_count=0,
                )
            ],
            fallback_db=fallback_db,
            fallback_applied=True,
            warnings=warnings,
        )

    rms = np.clip(rms, rms_floor, rms_ceiling)
    rms_db = 20.0 * np.log10(rms)

    # Use robust quantile to find the noise floor
    noise_quantile = 0.25  # lower quartile tends to be noise
    def _noise_percentile(arr: np.ndarray) -> float:
        return float(np.percentile(arr, noise_quantile * 100))

    # Compute per-window noise stats
    window_size = max(1, int(min_interval_duration * sample_rate / hop_samples))
    intervals: List[NoiseInterval] = []
    applied_fallback = False

    for win_start in range(0, len(rms_db), max(1, window_size // 2)):
        win_end = min(win_start + window_size, len(rms_db))
        segment = rms_db[win_start:win_end]

        if len(segment) < min_stable_samples:
            db_val = fallback_db
            stable = False
            applied_fallback = True
        else:
            db_val = _noise_percentile(segment)
            stable = True

        t_start = win_start * hop_samples / sample_rate
        t_end = win_end * hop_samples / sample_rate
        rms_segment = rms[win_start:win_end]
        med = float(np.median(rms_segment)) if len(rms_segment) > 0 else rms_floor
        mad = _median_absolute_deviation(rms_segment) if len(rms_segment) > 1 else 0.0

        intervals.append(NoiseInterval(
            start=float(t_start),
            end=float(t_end),
            rms_median=med,
            rms_mad=mad,
            noise_db=float(db_val),
            stable=stable,
            sample_count=int(len(segment)),
        ))

        if len(intervals) >= max_intervals:
            break

    # Merge adjacent intervals with similar noise floors (hysteresis-gated)
    merged: List[NoiseInterval] = []
    for iv in intervals:
        if not merged:
            merged.append(iv)
            continue
        last = merged[-1]
        if abs(iv.noise_db - last.noise_db) < hysteresis_db:
            # Merge: extend the last interval
            merged[-1] = NoiseInterval(
                start=last.start,
                end=iv.end,
                rms_median=(
                    last.rms_median * last.sample_count + iv.rms_median * iv.sample_count
                ) / max(last.sample_count + iv.sample_count, 1),
                rms_mad=max(last.rms_mad, iv.rms_mad),
                noise_db=(last.noise_db * last.sample_count + iv.noise_db * iv.sample_count)
                / max(last.sample_count + iv.sample_count, 1),
                stable=last.stable and iv.stable,
                sample_count=last.sample_count + iv.sample_count,
            )
        else:
            if iv.end - last.end >= min_interval_duration or not merged or iv is intervals[-1]:
                merged.append(iv)

    if applied_fallback:
        warnings.append(
            "one or more intervals used fallback noise floor — "
            "boundary candidates in those intervals will have reduced confidence"
        )

    return LocalNoiseProfile(
        intervals=merged,
        fallback_db=fallback_db,
        fallback_applied=applied_fallback,
        warnings=warnings,
        metadata={"duration": duration, "sample_rate": sample_rate},
    )
