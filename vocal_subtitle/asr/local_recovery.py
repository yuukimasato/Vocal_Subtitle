"""局部再识别引擎

对有疑问的音频片段执行受限的重新识别，为 LLM 发起的复核请求、
物理覆盖率缺失区间和低置信度词提供新的声学证据。

设计契约：
- 每个区间有明确的尝试次数上限（max_attempts_per_range）。
- 候选词必须与请求区间有正重叠才能被接受。
- 低于 min_confidence 的候选被静默拒绝。
- 可选的备用 ASR 引擎在主引擎无结果时被调用。
- 每次请求产生结构化的 LocalRecoveryResult，包含 outcome 和 candidates。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..asr.base import ASREngine, TranscriptionSegment
from ..physical.ir import GlobalWord

logger = logging.getLogger(__name__)

_VALID_OUTCOMES = frozenset(
    {
        "recovered",
        "no_candidates",
        "max_attempts_exceeded",
        "asr_error",
        "skipped",
    }
)


# ── 数据结构 ──────────────────────────────────────────────────────────


@dataclass
class RecoveryCandidate:
    """从再识别中产生的候选词，携带声学置信度。"""

    word_id: str
    text: str
    start: float
    end: float
    confidence: float = 0.0
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.word_id, str) or not self.word_id.strip():
            raise ValueError("word_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(self.start, (int, float)) or self.start < 0:
            raise ValueError("start must be non-negative")
        if not isinstance(self.end, (int, float)) or self.end <= self.start:
            raise ValueError(f"end must be greater than start: {self.start=}, {self.end=}")
        self.start = float(self.start)
        self.end = float(self.end)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_id": self.word_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "language": self.language,
            "metadata": dict(self.metadata),
        }

    def to_global_word(
        self,
        *,
        source_window_id: str = "recovery",
        segment_id: str = "recovery-seg",
    ) -> GlobalWord:
        """将候选词转换为 GlobalWord，保留溯源信息。"""
        return GlobalWord(
            id=self.word_id,
            text=self.text,
            raw_start=self.start,
            raw_end=self.end,
            confidence=self.confidence,
            source_window_id=source_window_id,
            segment_id=segment_id,
            language=self.language,
            metadata={"source": "local_recovery", **self.metadata},
        )


@dataclass
class LocalRecoveryRequest:
    """对有限音频范围的再识别请求。

    reasons 记录发起原因，如 "uncovered"、"low_conf"、"review_request"。
    """

    start: float
    end: float
    reasons: Sequence[str] = field(default_factory=lambda: ["uncovered"])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"recovery request must have 0 <= start < end: {self.start=}, {self.end=}")
        self.start = float(self.start)
        self.end = float(self.end)
        if not self.reasons:
            raise ValueError("at least one reason is required")
        self.reasons = tuple(dict.fromkeys(
            str(reason).strip().lower() for reason in self.reasons
        ))

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class LocalRecoveryResult:
    """单个请求的再识别结果。

    outcome 判定：
    - "recovered"：找到了可接受的候选词。
    - "no_candidates"：ASR 无输出或全部被置信度过滤器拒绝。
    - "max_attempts_exceeded"：尝试次数达到上限仍无结果。
    - "asr_error"：ASR 引擎抛出异常。
    - "skipped"：请求因参数无效或音频不足被跳过。
    """

    request: LocalRecoveryRequest
    candidates: List[RecoveryCandidate] = field(default_factory=list)
    attempt_count: int = 0
    outcome: str = "skipped"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in _VALID_OUTCOMES:
            raise ValueError(
                f"outcome must be one of {sorted(_VALID_OUTCOMES)}, got {self.outcome!r}"
            )
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be >= 1")

    @property
    def success(self) -> bool:
        return self.outcome == "recovered" and len(self.candidates) > 0


@dataclass
class LocalRecoveryConfig:
    """局部再识别的配置。

    max_attempts_per_range：每个区间最多尝试多少次再识别。
    min_confidence：接受候选的最低词级置信度。
    context_window：在请求区间两侧扩展的上下文字秒数。
    """

    max_attempts_per_range: int = 3
    min_confidence: float = 0.5
    context_window: float = 0.5

    def __post_init__(self) -> None:
        if self.max_attempts_per_range < 1:
            raise ValueError("max_attempts_per_range must be >= 1")
        if self.max_attempts_per_range > 5:
            raise ValueError("max_attempts_per_range must be <= 5")
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.context_window < 0.0:
            raise ValueError("context_window must be non-negative")


# ── 引擎 ──────────────────────────────────────────────────────────────


class LocalRecoveryEngine:
    """限次局部 ASR 再识别引擎。

    用法：
        engine = LocalRecoveryEngine(asr_engine, config=LocalRecoveryConfig())
        results = engine.process_requests(requests, audio, sample_rate=16000)
        for result in results:
            if result.success:
                for candidate in result.candidates:
                    word = candidate.to_global_word()
    """

    def __init__(
        self,
        asr_engine: ASREngine,
        *,
        config: Optional[LocalRecoveryConfig] = None,
        fallback_asr: Optional[ASREngine] = None,
        word_id_prefix: str = "rec",
        language: Optional[str] = None,
    ):
        self._asr = asr_engine
        self._fallback = fallback_asr
        self._config = config or LocalRecoveryConfig()
        self._word_id_prefix = word_id_prefix
        self._language = language

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def process_requests(
        self,
        requests: Sequence[LocalRecoveryRequest],
        audio: np.ndarray,
        *,
        sample_rate: int = 16000,
    ) -> List[LocalRecoveryResult]:
        """处理一批再识别请求。

        每个请求单独尝试，失败不会阻塞后续请求。
        """
        if not requests:
            return []

        results: List[LocalRecoveryResult] = []
        for request in requests:
            result = self._process_one(request, audio, sample_rate=sample_rate)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # 单请求处理
    # ------------------------------------------------------------------

    def _process_one(
        self,
        request: LocalRecoveryRequest,
        audio: np.ndarray,
        *,
        sample_rate: int,
    ) -> LocalRecoveryResult:
        cfg = self._config
        total_samples = len(audio)
        total_duration = total_samples / sample_rate

        context_start = max(0.0, request.start - cfg.context_window)
        context_end = min(total_duration, request.end + cfg.context_window)
        start_sample = int(context_start * sample_rate)
        end_sample = int(context_end * sample_rate)

        if end_sample - start_sample < sample_rate * 0.05:  # < 50ms
            return LocalRecoveryResult(
                request=request,
                outcome="skipped",
                attempt_count=1,
                diagnostics={"reason": "audio too short"},
            )

        segment_audio = audio[start_sample:end_sample]
        last_error: Optional[str] = None

        for attempt in range(1, cfg.max_attempts_per_range + 1):
            try:
                candidates = self._try_transcribe(
                    segment_audio, sample_rate, context_start, request
                )
            except Exception as exc:
                logger.warning("Recovery ASR call failed (attempt %d): %s", attempt, exc)
                last_error = str(exc)
                continue

            if candidates:
                return LocalRecoveryResult(
                    request=request,
                    candidates=candidates,
                    attempt_count=attempt,
                    outcome="recovered",
                    diagnostics={"context_start": context_start, "context_end": context_end},
                )

            if attempt < cfg.max_attempts_per_range:
                logger.debug(
                    "Recovery attempt %d/%d returned no candidates for [%.2f, %.2f]",
                    attempt, cfg.max_attempts_per_range, request.start, request.end,
                )

        if last_error is not None:
            return LocalRecoveryResult(
                request=request,
                candidates=[],
                attempt_count=cfg.max_attempts_per_range,
                outcome="asr_error",
                diagnostics={"context_start": context_start, "context_end": context_end,
                             "last_error": last_error},
            )

        return LocalRecoveryResult(
            request=request,
            candidates=[],
            attempt_count=cfg.max_attempts_per_range,
            outcome="max_attempts_exceeded",
            diagnostics={"context_start": context_start, "context_end": context_end},
        )

    def _try_transcribe(
        self,
        segment_audio: np.ndarray,
        sample_rate: int,
        time_offset: float,
        request: LocalRecoveryRequest,
    ) -> List[RecoveryCandidate]:
        """尝试一次转录，返回通过过滤的候选词。"""

        def _transcribe(engine: ASREngine) -> List[TranscriptionSegment]:
            result = engine.transcribe(
                segment_audio, sample_rate, language=self._language
            )
            if isinstance(result, TranscriptionSegment):
                return [result]
            if isinstance(result, list):
                return result
            return []

        try:
            segments = _transcribe(self._asr)
        except Exception as exc:
            logger.warning("Recovery ASR call failed: %s", exc)
            raise

        # 主引擎无结果时尝试备用引擎
        if not segments and self._fallback is not None:
            logger.debug("Primary ASR returned empty, trying fallback")
            try:
                segments = _transcribe(self._fallback)
            except Exception as exc:
                logger.warning("Fallback ASR call failed: %s", exc)

        if not segments:
            return []

        candidates = self._segments_to_candidates(segments, time_offset, request)
        return candidates

    def _segments_to_candidates(
        self,
        segments: List[TranscriptionSegment],
        time_offset: float,
        request: LocalRecoveryRequest,
    ) -> List[RecoveryCandidate]:
        """将 ASR 转录段转为候选词，过滤不可接受的候选。"""
        cfg = self._config
        candidates: List[RecoveryCandidate] = []
        candidate_index = 0

        for seg in segments:
            words = getattr(seg, "words", []) or []
            if not words:
                # 无词级信息时，使用段级信息创建单个候选
                g_start = time_offset + float(getattr(seg, "start", 0.0))
                g_end = time_offset + float(getattr(seg, "end", 0.0))
                seg_confidence = float(getattr(seg, "avg_logprob", 0.5))
                # avg_logprob 是负数对数概率，转换为 0-1 置信度
                if seg_confidence < 0:
                    seg_confidence = max(0.0, 1.0 + seg_confidence)
                seg_confidence = max(0.0, min(1.0, seg_confidence))

                if seg_confidence < cfg.min_confidence:
                    continue
                if not self._overlaps_request(g_start, g_end, request):
                    continue

                candidate = RecoveryCandidate(
                    word_id=f"{self._word_id_prefix}:{candidate_index:04d}",
                    text=str(getattr(seg, "text", "")).strip(),
                    start=g_start,
                    end=g_end,
                    confidence=seg_confidence,
                )
                candidates.append(candidate)
                candidate_index += 1
                continue

            for word in words:
                g_start = time_offset + float(getattr(word, "start", 0.0))
                g_end = time_offset + float(getattr(word, "end", 0.0))
                confidence = float(getattr(word, "confidence", 0.9))

                if confidence < cfg.min_confidence:
                    continue
                if not self._overlaps_request(g_start, g_end, request):
                    continue

                candidate = RecoveryCandidate(
                    word_id=f"{self._word_id_prefix}:{candidate_index:04d}",
                    text=str(getattr(word, "word", "")).strip(),
                    start=g_start,
                    end=g_end,
                    confidence=confidence,
                )
                candidates.append(candidate)
                candidate_index += 1

        return candidates

    @staticmethod
    def _overlaps_request(
        word_start: float,
        word_end: float,
        request: LocalRecoveryRequest,
        *,
        epsilon: float = 0.01,
    ) -> bool:
        """检查词与请求区间是否有正重叠。"""
        return word_start < request.end - epsilon and word_end > request.start + epsilon


# ── 便捷函数 ──────────────────────────────────────────────────────────


def make_recovery_requests_from_coverage(
    recovery_ranges: Sequence[Any],
    reason: str = "uncovered",
) -> List[LocalRecoveryRequest]:
    """从物理覆盖率审计的 recovery_ranges 构建请求。"""
    requests = []
    for rec_range in recovery_ranges:
        start = float(getattr(rec_range, "start", 0.0))
        end = float(getattr(rec_range, "end", 0.0))
        if end > start:
            requests.append(LocalRecoveryRequest(start=start, end=end, reasons=[reason]))
    return requests
