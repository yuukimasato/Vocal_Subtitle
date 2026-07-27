"""说话人变更声学检测

基于声学特征差异检测潜在说话人变更点。

设计契约：
- RMS 均值突变只能产生 SPK_CHANGE_CANDIDATE 弱信号，不能单独形成硬边界。
- MFCC/F0 可在 embedding 不可用时补强。
- 马氏距离仅在样本足以稳定估计协方差时使用。
- 候选经过 diarization change point 或多源声学一致性确认后才升级为 speaker 硬边界。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_VALID_SIGNAL_TYPES = frozenset(
    {
        "rms_surge",
        "rms_stable",
        "embedding_distance",
        "feature_divergence",
        "mfcc_divergence",
        "pitch_divergence",
        "formant_divergence",
        "insufficient_data",
        "no_change",
    }
)


# ── 数据结构 ──────────────────────────────────────────────────────────


@dataclass
class SpeakerChangeSignal:
    """单个声学特征的说话人变更信号。

    signal_type 说明：
    - rms_surge：RMS 能量突变（弱信号，不单独形成硬边界）。
    - embedding_distance：说话人嵌入距离（强信号）。
    - feature_divergence：多特征差异。
    - mfcc_divergence：MFCC 分布差异。
    - pitch_divergence：基频变化。
    - formant_divergence：共振峰变化。
    - insufficient_data：样本不足。
    - no_change：无明显变化。
    """

    start: float
    end: float
    signal_type: str
    confidence: float
    source: str = "acoustic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"signal must have 0 <= start <= end: {self.start=}, {self.end=}")
        if not self.signal_type or self.signal_type not in _VALID_SIGNAL_TYPES:
            raise ValueError(
                f"signal_type must be one of {sorted(_VALID_SIGNAL_TYPES)}, got {self.signal_type!r}"
            )
        self.start = float(self.start)
        self.end = float(self.end)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def is_candidate(self) -> bool:
        """是否为弱候选信号（需多源确认才可升级为硬边界）。"""
        return self.signal_type in (
            "rms_surge",
            "mfcc_divergence",
            "pitch_divergence",
            "formant_divergence",
        )

    @property
    def is_strong(self) -> bool:
        """是否为强信号（可直接参与硬边界判定）。"""
        return self.signal_type in ("embedding_distance", "feature_divergence")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass
class SpeakerChangeResult:
    """多特征融合后的说话人变更判断。"""

    signals: Tuple[SpeakerChangeSignal, ...] = ()
    is_hard_boundary: bool = False
    confidence: float = 0.0
    signal_type: str = "insufficient_data"
    evidence_ids: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    _change_time: Optional[float] = None

    @property
    def before_start(self) -> float | None:
        """Start time of the region before the change point."""
        if not self.signals:
            return None
        return min(s.start for s in self.signals)

    @property
    def after_end(self) -> float | None:
        """End time of the region after the change point."""
        if not self.signals:
            return None
        return max(s.end for s in self.signals)

    @property
    def change_time(self) -> float | None:
        """The time of the speaker change (transition between before and after)."""
        return self._change_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals": [s.to_dict() for s in self.signals],
            "is_hard_boundary": self.is_hard_boundary,
            "confidence": self.confidence,
            "signal_type": self.signal_type,
            "evidence_ids": list(self.evidence_ids),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class SpeakerChangeConfig:
    """说话人变更检测配置。

    rms_surge_ratio：RMS 能量比超过此值产生候选信号。
    candidate_threshold：单特征置信度达到此值产生候选。
    hard_boundary_threshold：融合置信度达到此值产生硬边界。
    min_signal_duration：检测需要的最小音频时长（秒）。
    min_samples_for_mahalanobis：使用马氏距离所需的最小帧数。
    """

    rms_surge_ratio: float = 2.5
    candidate_threshold: float = 0.3
    hard_boundary_threshold: float = 0.7
    min_signal_duration: float = 0.15
    min_samples_for_mahalanobis: int = 20

    def __post_init__(self) -> None:
        if self.rms_surge_ratio < 1.0:
            raise ValueError("rms_surge_ratio must be >= 1.0")
        if self.candidate_threshold < 0.0 or self.candidate_threshold > 1.0:
            raise ValueError("candidate_threshold must be in [0, 1]")
        if self.hard_boundary_threshold < 0.5 or self.hard_boundary_threshold > 1.0:
            raise ValueError("hard_boundary_threshold must be in [0.5, 1]")
        if self.candidate_threshold >= self.hard_boundary_threshold:
            raise ValueError("candidate_threshold must be less than hard_boundary_threshold")
        if self.min_signal_duration <= 0:
            raise ValueError("min_signal_duration must be > 0")
        if self.min_samples_for_mahalanobis < 5:
            raise ValueError("min_samples_for_mahalanobis must be >= 5")


# ── 音量突变检测 ──────────────────────────────────────────────────────



def detect_volume_surge(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    *,
    sample_rate: int = 16000,
    surge_ratio: float = 2.5,
) -> SpeakerChangeSignal:
    """检测音量突变。

    RMS 均值突变只产生弱候选信号，不能单独形成 speaker 硬边界。
    """
    min_samples = int(sample_rate * 0.05)  # 至少 50ms
    if len(audio_before) < min_samples or len(audio_after) < min_samples:
        return SpeakerChangeSignal(
            start=0.0,
            end=0.0,
            signal_type="insufficient_data",
            confidence=0.0,
            source="rms",
            metadata={"reason": "audio too short"},
        )

    rms_before = float(np.sqrt(np.mean(audio_before.astype(np.float64) ** 2)))
    rms_after = float(np.sqrt(np.mean(audio_after.astype(np.float64) ** 2)))

    # 防止除零
    if rms_before < 1e-10:
        if rms_after > 1e-8:
            # 从静音到有声的"突变"不视为 speaker 变更信号
            return SpeakerChangeSignal(
                start=0.0, end=0.0,
                signal_type="rms_surge",
                confidence=0.15,
                source="rms",
                metadata={"rms_before": rms_before, "rms_after": rms_after,
                          "ratio": float("inf"), "note": "silence_to_speech"},
            )
        return SpeakerChangeSignal(
            start=0.0, end=0.0,
            signal_type="rms_stable",
            confidence=0.0,
            source="rms",
            metadata={"rms_before": rms_before, "rms_after": rms_after, "ratio": 1.0},
        )

    ratio = rms_after / max(rms_before, 1e-10)
    if ratio >= surge_ratio:
        confidence = min(0.4, (ratio - surge_ratio) / (surge_ratio * 3))
        return SpeakerChangeSignal(
            start=0.0, end=0.0,
            signal_type="rms_surge",
            confidence=round(confidence, 3),
            source="rms",
            metadata={"rms_before": rms_before, "rms_after": rms_after, "ratio": ratio},
        )

    return SpeakerChangeSignal(
        start=0.0, end=0.0,
        signal_type="rms_stable",
        confidence=0.0,
        source="rms",
        metadata={"rms_before": rms_before, "rms_after": rms_after, "ratio": ratio},
    )


# ── 特征差异检测 ──────────────────────────────────────────────────────


def detect_speaker_change_from_features(
    features_before: np.ndarray,
    features_after: np.ndarray,
    *,
    time1: Tuple[float, float] = (0.0, 0.0),
    time2: Tuple[float, float] = (0.0, 0.0),
    config: Optional[SpeakerChangeConfig] = None,
) -> SpeakerChangeResult:
    """基于声学特征向量检测说话人变更。

    Args:
        features_before：变更前的特征矩阵 (n_frames × n_features)。
        features_after：变更后的特征矩阵 (n_frames × n_features)。
        time1：(before_start, before_end) 时间范围。
        time2：(after_start, after_end) 时间范围。
        config：检测配置。

    Returns:
        SpeakerChangeResult 包含融合信号和硬边界判定。
    """
    cfg = config or SpeakerChangeConfig()

    if features_before.size == 0 or features_after.size == 0:
        return SpeakerChangeResult(
            signals=(),
            is_hard_boundary=False,
            confidence=0.0,
            signal_type="insufficient_data",
            diagnostics={"reason": "empty feature array"},
            _change_time=time1[1],
        )

    if features_before.ndim != 2 or features_after.ndim != 2:
        return SpeakerChangeResult(
            signals=(),
            is_hard_boundary=False,
            confidence=0.0,
            signal_type="insufficient_data",
            diagnostics={"reason": "expected 2D arrays"},
            _change_time=time1[1],
        )

    if features_before.shape[1] != features_after.shape[1]:
        return SpeakerChangeResult(
            signals=(),
            is_hard_boundary=False,
            confidence=0.0,
            signal_type="insufficient_data",
            diagnostics={"reason": "feature dimension mismatch"},
            _change_time=time1[1],
        )

    change_time = time1[1]

    # 计算均值向量的余弦距离
    mean_before = np.mean(features_before, axis=0)
    mean_after = np.mean(features_after, axis=0)

    norm_before = float(np.linalg.norm(mean_before))
    norm_after = float(np.linalg.norm(mean_after))

    if norm_before < 1e-10 or norm_after < 1e-10:
        return SpeakerChangeResult(
            signals=(),
            is_hard_boundary=False,
            confidence=0.0,
            signal_type="insufficient_data",
            diagnostics={"reason": "zero-norm feature mean"},
            _change_time=change_time,
        )

    cosine_sim = float(np.dot(mean_before, mean_after) / (norm_before * norm_after))
    cosine_distance = max(0.0, 1.0 - cosine_sim)
    cosine_confidence = min(1.0, cosine_distance / 0.3)  # 0.3 余弦距离 → 1.0 置信度

    signals: List[SpeakerChangeSignal] = []

    if cosine_confidence >= cfg.candidate_threshold:
        signal_type = "embedding_distance" if features_before.shape[1] >= 50 else "feature_divergence"
        signals.append(
            SpeakerChangeSignal(
                start=time1[0],
                end=time2[1],
                signal_type=signal_type,
                confidence=round(cosine_confidence, 3),
                source="acoustic",
                metadata={
                    "cosine_distance": round(cosine_distance, 4),
                    "cosine_similarity": round(cosine_sim, 4),
                    "n_frames_before": features_before.shape[0],
                    "n_frames_after": features_after.shape[0],
                },
            )
        )
    else:
        signals.append(
            SpeakerChangeSignal(
                start=time1[0],
                end=time2[1],
                signal_type="no_change",
                confidence=0.0,
                source="acoustic",
                metadata={"cosine_distance": round(cosine_distance, 4)},
            )
        )

    # 融合判定
    has_strong = any(s.is_strong and s.confidence >= cfg.hard_boundary_threshold for s in signals)
    has_candidate = any(s.confidence >= cfg.candidate_threshold for s in signals)

    if has_strong:
        confidence = max(s.confidence for s in signals if s.is_strong)
        signal_type = max(
            (s for s in signals if s.is_strong),
            key=lambda s: s.confidence,
        ).signal_type
        return SpeakerChangeResult(
            signals=tuple(signals),
            is_hard_boundary=True,
            confidence=confidence,
            signal_type=signal_type,
            diagnostics={"fusion": "single_strong_signal"},
            _change_time=change_time,
        )

    if has_candidate:
        confidence = max(s.confidence for s in signals)
        signal_type = max(signals, key=lambda s: s.confidence).signal_type
        return SpeakerChangeResult(
            signals=tuple(signals),
            is_hard_boundary=False,
            confidence=confidence,
            signal_type=signal_type,
            diagnostics={"fusion": "candidate_only"},
            _change_time=change_time,
        )

    return SpeakerChangeResult(
        signals=tuple(signals),
        is_hard_boundary=False,
        confidence=0.0,
        signal_type="no_change",
        _change_time=change_time,
    )


def compute_mahalanobis_distance(
    features_before: np.ndarray,
    features_after: np.ndarray,
    *,
    min_samples: int = 20,
    regularization: float = 1e-4,
) -> Optional[float]:
    """计算两组特征的马氏距离（当样本充足时）。

    仅在每个集合的帧数 >= min_samples 时计算，
    否则返回 None。

    Args:
        features_before：变更前的特征矩阵。
        features_after：变更后的特征矩阵。
        min_samples：每组最少需要的帧数。
        regularization：协方差矩阵的正则化项。

    Returns:
        马氏距离或 None（样本不足时）。
    """
    if features_before.shape[0] < min_samples or features_after.shape[0] < min_samples:
        return None

    try:
        mean_before = np.mean(features_before, axis=0)
        mean_after = np.mean(features_after, axis=0)
        diff = mean_before - mean_after

        # 使用合并的协方差估计
        cov_before = np.cov(features_before, rowvar=False)
        cov_after = np.cov(features_after, rowvar=False)
        pooled_cov = (cov_before + cov_after) / 2.0

        # 正则化
        n_features = pooled_cov.shape[0]
        pooled_cov += np.eye(n_features) * regularization

        inv_cov = np.linalg.inv(pooled_cov)
        mahalanobis = float(np.sqrt(diff @ inv_cov @ diff))
        return mahalanobis
    except (np.linalg.LinAlgError, ValueError) as exc:
        logger.debug("Mahalanobis distance computation failed: %s", exc)
        return None
