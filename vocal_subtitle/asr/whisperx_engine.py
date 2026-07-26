"""Optional WhisperX adapter with lazy dependency loading."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ..physical.ir import GlobalTranscript, adapt_transcription_segments
from .base import ASREngine, TranscriptionSegment, WordTimestamp

logger = logging.getLogger(__name__)


class WhisperXUnavailableError(RuntimeError):
    """WhisperX is not installed or cannot be loaded."""


class WhisperXEngine(ASREngine):
    """WhisperX ASR adapter; importing this class does not import WhisperX."""

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        batch_size: int = 16,
        word_timestamps: bool = True,
    ) -> None:
        self._model_size = model
        self._device = device
        self._compute_type = compute_type
        self._batch_size = batch_size
        self._word_timestamps = word_timestamps
        self._model = None
        self._alignment_model = None
        self._alignment_metadata = None
        self._alignment_language = None

    @property
    def name(self) -> str:
        return "whisperx"

    @property
    def model_name(self) -> str:
        return self._model_size

    @property
    def supports_alignment(self) -> bool:
        return True

    def load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import whisperx
        except ImportError as exc:
            raise WhisperXUnavailableError(
                "WhisperX is not installed; install the optional whisperx dependency"
            ) from exc
        try:
            self._model = whisperx.load_model(
                self._model_size,
                self._device,
                compute_type=self._compute_type,
                asr_options={"word_timestamps": self._word_timestamps},
            )
        except Exception as exc:
            raise WhisperXUnavailableError(
                f"failed to load WhisperX model: {exc}"
            ) from exc

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
        **kwargs,
    ) -> list[TranscriptionSegment]:
        self.load_model()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        result = self._model.transcribe(
            audio,
            batch_size=kwargs.pop("batch_size", self._batch_size),
            language=language,
            **kwargs,
        )
        detected_language = (
            result.get("language", language)
            if isinstance(result, Mapping)
            else language
        )
        raw_segments = (
            result.get("segments", []) if isinstance(result, Mapping) else result
        )
        return normalize_whisperx_segments(raw_segments, language=detected_language)

    def align(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 16000,
        segments: Iterable[Any],
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        """Run WhisperX forced alignment for already decoded segments.

        The alignment model is loaded lazily and reused per language. A
        caller may fall back to the original ASR timestamps when alignment is
        unavailable, so this method keeps failure handling explicit.
        """
        if self._model is None:
            self.load_model()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        raw_segments = list(segments or [])
        if not raw_segments:
            return []
        resolved_language = language or _value(raw_segments[0], "language", None)
        if not resolved_language:
            raise WhisperXUnavailableError(
                "WhisperX alignment requires a detected or configured language"
            )

        try:
            import whisperx
        except ImportError as exc:
            raise WhisperXUnavailableError(
                "WhisperX is not installed; alignment is unavailable"
            ) from exc

        if self._alignment_language != resolved_language:
            try:
                self._alignment_model, self._alignment_metadata = (
                    whisperx.load_align_model(
                        language_code=resolved_language,
                        device=self._device,
                    )
                )
            except Exception as exc:
                self._alignment_model = None
                self._alignment_metadata = None
                self._alignment_language = None
                raise WhisperXUnavailableError(
                    f"failed to load WhisperX alignment model: {exc}"
                ) from exc
            self._alignment_language = resolved_language

        payload = [
            _segment_payload(segment)
            for segment in raw_segments
            if _value(segment, "start", None) is not None
        ]
        aligned = whisperx.align(
            payload,
            self._alignment_model,
            self._alignment_metadata,
            audio,
            self._device,
            return_char_alignments=False,
        )
        aligned_segments = (
            aligned.get("segments", []) if isinstance(aligned, Mapping) else aligned
        )
        return normalize_whisperx_segments(
            aligned_segments,
            language=resolved_language,
        )


def normalize_whisperx_segments(
    raw_segments: Iterable[Any],
    *,
    language: str | None = None,
) -> list[TranscriptionSegment]:
    """Convert WhisperX dict/object segments without applying an offset."""
    normalized: list[TranscriptionSegment] = []
    for raw in raw_segments or []:
        text = _value(raw, "text", "")
        start = _finite_time(_value(raw, "start", None), "segment.start")
        end = _finite_time(_value(raw, "end", None), "segment.end")
        if not text or start is None or end is None or end <= start:
            continue
        segment_speaker = _speaker(
            _value(raw, "speaker", _value(raw, "speaker_id", None))
        )
        words: list[WordTimestamp] = []
        raw_words = _value(raw, "words", []) or []
        seen = set()
        for index, raw_word in enumerate(raw_words):
            word_text = str(
                _value(raw_word, "word", _value(raw_word, "text", "")) or ""
            ).strip()
            word_start = _finite_time(_value(raw_word, "start", None), "word.start")
            word_end = _finite_time(_value(raw_word, "end", None), "word.end")
            if (
                not word_text
                or word_start is None
                or word_end is None
                or word_end <= word_start
            ):
                continue
            if word_start < start or word_end > end:
                continue
            identity = (word_text, round(word_start, 6), round(word_end, 6))
            if identity in seen:
                continue
            seen.add(identity)
            words.append(
                WordTimestamp(
                    word=word_text,
                    start=word_start,
                    end=word_end,
                    confidence=_confidence(
                        _value(
                            raw_word,
                            "score",
                            _value(
                                raw_word,
                                "probability",
                                _value(raw_word, "confidence", 1.0),
                            ),
                        )
                    ),
                    speaker_id=_speaker(
                        _value(
                            raw_word,
                            "speaker",
                            _value(raw_word, "speaker_id", segment_speaker),
                        )
                    ),
                )
            )
        normalized.append(
            TranscriptionSegment(
                text=str(text).strip(),
                start=start,
                end=end,
                words=words,
                avg_logprob=float(_value(raw, "avg_logprob", 0.0) or 0.0),
                language=_value(raw, "language", language),
                language_probability=float(
                    _value(raw, "language_probability", 0.0) or 0.0
                ),
                no_speech_prob=_optional_float(_value(raw, "no_speech_prob", None)),
                compression_ratio=_optional_float(
                    _value(raw, "compression_ratio", None)
                ),
                speaker_id=segment_speaker,
            )
        )
    return normalized


def normalize_whisperx_transcript(
    raw_segments: Iterable[Any],
    *,
    source_window_id: str,
    segment_id_prefix: str,
    time_offset: float = 0.0,
    language: str | None = None,
    audio_duration: float | None = None,
) -> GlobalTranscript:
    """Normalize one window directly into absolute global IR."""
    segments = normalize_whisperx_segments(raw_segments, language=language)
    return adapt_transcription_segments(
        segments,
        source_window_id=source_window_id,
        segment_id_prefix=segment_id_prefix,
        time_offset=time_offset,
        language=language,
        audio_duration=audio_duration,
    )


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _segment_payload(segment: Any) -> dict[str, Any]:
    """Build the minimal WhisperX alignment input from normalized segments."""
    payload: dict[str, Any] = {
        "text": str(_value(segment, "text", "") or ""),
        "start": _value(segment, "start", 0.0),
        "end": _value(segment, "end", 0.0),
    }
    speaker = _value(segment, "speaker_id", None)
    if speaker is not None:
        payload["speaker"] = f"SPEAKER_{int(speaker):02d}"
    return payload


def _finite_time(value: Any, name: str) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0:
        return None
    return result


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _confidence(value: Any) -> float:
    result = _optional_float(value)
    if result is None:
        return 1.0
    return max(0.0, min(1.0, result))


def _speaker(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().upper().startswith("SPEAKER_"):
        suffix = value.strip().split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None
