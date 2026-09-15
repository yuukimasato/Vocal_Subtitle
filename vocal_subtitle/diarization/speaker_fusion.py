"""融合声纹身份与层级式全局/局部 diarization。"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import DiarizationResult, SpeakerTurn
from .canonicalizer import canonicalize_diarization_result
from .model_registry import is_model_cached, resolve_global_model_ref
from .speaker_embedding import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingEvidence:
    labels: list[int] = field(default_factory=list)
    spans: list[tuple[float, float]] = field(default_factory=list)
    centroids: dict[int, np.ndarray] = field(default_factory=dict)
    model: str = ""
    silhouette: float | None = None
    status: str = "unavailable"


# ---- 说话人标签国际化（与 application/stage_runner 的映射保持一致） ----
_SPEAKER_LABEL_MAP = {"zh": "说话人", "ja": "話者", "ko": "화자"}
_SPEAKER_LABEL_DEFAULT = "Speaker"


def _speaker_label(language: str | None, speaker_id: int) -> str:
    """按语言生成说话人标签（"说话人A" / "Speaker A" / "話者A"）。"""
    letter = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[speaker_id]
        if 0 <= speaker_id < 26
        else str(speaker_id)
    )
    if language:
        base = str(language).split("-")[0].lower()
        prefix = _SPEAKER_LABEL_MAP.get(base)
        if prefix is not None:
            # CJK 语言：标签与字母之间无空格（如 "说话人A"）
            return f"{prefix}{letter}"
    return f"{_SPEAKER_LABEL_DEFAULT} {letter}"


# 回退聚类质量下限：silhouette 低于该值说明簇分离不可信（对 MFCC 特征
# 尤其容易碎分成假说话人），收敛为单说话人比给出错误的多人划分更诚实。
_FALLBACK_MIN_SILHOUETTE = 0.15


def _fallback_cluster_labels(
    events: list,
    audio: np.ndarray,
    sample_rate: int,
    diar_cfg: Any,
) -> dict | None:
    """声学嵌入与全局分离均不可用时的轻量聚类回退。

    复用仓库内置的 MFCC+音高凝聚聚类（SpeakerDiarizer，仅依赖
    sklearn/librosa，无需 speechbrain/pyannote），给事件级分配说话人，
    避免最小依赖环境下全部事件 unknown。返回 {事件下标: speaker_id}。
    """
    if not events or not _has_audio_signal(audio):
        return None
    try:
        from types import SimpleNamespace

        from .speaker_clusterer import SpeakerDiarizer

        segments = [
            SimpleNamespace(start=event.start, end=event.end) for event in events
        ]
        diarizer = SpeakerDiarizer(
            distance_threshold=getattr(diar_cfg, "distance_threshold", 0.5),
            min_speakers=getattr(diar_cfg, "min_speakers", 1),
            max_speakers=getattr(diar_cfg, "max_speakers", 10),
            expected_speakers=getattr(diar_cfg, "expected_speakers", None),
            use_pca=getattr(diar_cfg, "use_pca", True),
            pca_variance=getattr(diar_cfg, "pca_variance", 0.95),
        )
        diarizer.load_model()
        ids = diarizer.diarize(segments, audio, sample_rate)
    except Exception as exc:  # noqa: BLE001 - 回退失败保持 unknown
        logger.warning("Fallback speaker clustering failed: %s", exc)
        return None
    if len(ids) != len(events):
        return None
    silhouette = getattr(diarizer, "last_silhouette_", None)
    if len(set(ids)) > 1 and (
        silhouette is None or silhouette < _FALLBACK_MIN_SILHOUETTE
    ):
        logger.info(
            "Fallback speaker clustering quality low (silhouette=%.3f) "
            "→ collapsing to single speaker",
            float(silhouette) if silhouette is not None else -1.0,
        )
        ids = [0] * len(ids)
    logger.info(
        "Fallback speaker clustering assigned %d events across %d speakers",
        len(ids),
        len(set(ids)),
    )
    return {index: int(sid) for index, sid in enumerate(ids)}


@dataclass
class SpeakerFusionResult:
    events: list
    speaker_count: int = 0
    backend: str = "unknown"
    status: str = "unknown"
    model_ref: str = ""
    local_split_count: int = 0
    conflict_count: int = 0
    unknown_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _word_text(word: Any) -> str:
    return str(_value(word, "word", ""))


def _word_start(word: Any) -> float:
    return float(_value(word, "start", 0.0))


def _word_end(word: Any) -> float:
    return float(_value(word, "end", _word_start(word)))


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return max(0.0, min(2.0, 1.0 - similarity))


def _audio_slice(
    audio: np.ndarray, start: float, end: float, sample_rate: int
) -> np.ndarray:
    start_sample = max(0, min(len(audio), int(start * sample_rate)))
    end_sample = max(start_sample, min(len(audio), int(end * sample_rate)))
    return np.asarray(audio[start_sample:end_sample], dtype=np.float32)


def _has_audio_signal(audio: np.ndarray, floor: float = 1e-4) -> bool:
    """Return whether a slice contains enough finite energy for embeddings."""
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return False
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(finite, dtype=np.float64))))
    return np.isfinite(rms) and rms > floor


def _window_spans(duration: float) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    if duration <= 3.0:
        return [(0.0, duration)]
    spans = []
    start = 0.0
    while start + 3.0 <= duration:
        spans.append((start, start + 3.0))
        start += 1.0
    if not spans or spans[-1][1] < duration:
        spans.append((max(0.0, duration - 3.0), duration))
    return list(dict.fromkeys(spans))


def _nearest_centroid(
    feature: np.ndarray, centroids: dict[int, np.ndarray]
) -> int | None:
    if not centroids:
        return None
    return min(
        centroids,
        key=lambda label: _cosine_distance(feature, centroids[label]),
    )


def _extract_embedding_evidence(
    events: list,
    audio: np.ndarray,
    sample_rate: int,
    embedding_engine: Any,
    diar_cfg: Any,
) -> EmbeddingEvidence:
    if embedding_engine is None or not getattr(embedding_engine, "model_loaded", False):
        return EmbeddingEvidence(status="unavailable")

    duration = len(audio) / max(sample_rate, 1)
    spans = _window_spans(duration)
    features: list[np.ndarray] = []
    valid_spans: list[tuple[float, float]] = []
    for start, end in spans:
        snippet = _audio_slice(audio, start, end, sample_rate)
        if len(snippet) < max(1, int(sample_rate * 0.15)) or not _has_audio_signal(
            snippet
        ):
            continue
        try:
            feature = np.asarray(
                embedding_engine.extract_embedding(snippet, sample_rate),
                dtype=np.float64,
            ).reshape(-1)
        except Exception as exc:
            logger.warning("Speaker embedding extraction failed: %s", exc)
            continue
        if (
            feature.size == 0
            or not np.any(np.isfinite(feature))
            or not np.any(np.abs(feature) > 1e-8)
        ):
            continue
        features.append(np.nan_to_num(feature))
        valid_spans.append((start, end))

    if not features:
        return EmbeddingEvidence(
            model=getattr(embedding_engine, "name", ""), status="failed"
        )

    matrix = np.vstack(features)
    if matrix.ndim != 2 or matrix.shape[0] != len(valid_spans):
        return EmbeddingEvidence(
            model=getattr(embedding_engine, "name", ""),
            status="failed",
        )
    expected = getattr(diar_cfg, "expected_speakers", None)
    identical_embeddings = (
        len(features) > 1 and float(np.max(np.ptp(matrix, axis=0))) <= 1e-8
    )
    if expected is not None and int(expected) > 1 and identical_embeddings:
        # A known multi-speaker constraint must not manufacture clusters from
        # identical embeddings. Let global diarization or unknown carry this
        # case until there is discriminating acoustic evidence.
        return EmbeddingEvidence(
            model=getattr(embedding_engine, "name", ""),
            status="failed",
        )
    if identical_embeddings:
        return EmbeddingEvidence(
            labels=[0] * len(events),
            spans=valid_spans,
            centroids={0: matrix[0]},
            model=getattr(embedding_engine, "name", ""),
            silhouette=1.0,
            status="ok",
        )

    from .speaker_clusterer import SpeakerDiarizer

    min_speakers = int(expected or max(1, getattr(diar_cfg, "min_speakers", 1)))
    max_speakers = int(expected or getattr(diar_cfg, "max_speakers", 10))
    diarizer = SpeakerDiarizer(
        distance_threshold=getattr(diar_cfg, "distance_threshold", 0.5),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        expected_speakers=expected,
        use_pca=False,
    )
    try:
        labels = np.asarray(diarizer._cluster(matrix), dtype=np.int64).reshape(-1)
        if labels.size != matrix.shape[0] or labels.size == 0:
            raise ValueError("embedding cluster labels do not match feature rows")
        silhouette = diarizer._evaluate_clustering(matrix, labels)
    except Exception as exc:
        logger.warning("Embedding clustering failed: %s", exc)
        return EmbeddingEvidence(
            model=getattr(embedding_engine, "name", ""), status="failed"
        )

    centroids = {}
    for label in sorted(set(labels.tolist())):
        members = matrix[labels == label]
        if len(members):
            centroids[int(label)] = members.mean(axis=0)
    if not centroids:
        return EmbeddingEvidence(
            model=getattr(embedding_engine, "name", ""), status="failed"
        )
    event_labels: list[int] = []
    for event in events:
        best_index = max(
            range(len(valid_spans)),
            key=lambda index: max(
                0.0,
                min(float(event.end), valid_spans[index][1])
                - max(float(event.start), valid_spans[index][0]),
            ),
        )
        event_labels.append(int(labels[best_index]))

    return EmbeddingEvidence(
        labels=event_labels,
        spans=valid_spans,
        centroids=centroids,
        model=getattr(embedding_engine, "name", ""),
        silhouette=float(silhouette),
        status="ok",
    )


def _select_global_model(global_model: str, cache_dir: str | None) -> str | None:
    if global_model in ("none", "disabled"):
        return None
    if global_model == "auto":
        for candidate in ("community-1", "diarization-3.1"):
            if is_model_cached(candidate, cache_dir):
                return resolve_global_model_ref(candidate)
        return None
    return resolve_global_model_ref(global_model)


def _run_global_pass(
    audio: np.ndarray, sample_rate: int, config: Any
) -> tuple[DiarizationResult | None, str, str]:
    diar_cfg = config.diarization
    if getattr(diar_cfg, "backend", "auto") == "legacy":
        return None, "", "disabled"
    if getattr(diar_cfg, "diarization_scope", "hierarchical") not in (
        "global",
        "hierarchical",
    ):
        return None, "", "disabled"

    emb_cfg = config.speaker_embedding
    # Keep model discovery and runtime loading on the same cache root. The
    # empty config value means the project's shared speaker-model directory.
    cache_dir = getattr(emb_cfg, "cache_dir", "") or str(DEFAULT_CACHE_DIR)
    model_ref = _select_global_model(
        getattr(diar_cfg, "global_model", "auto"), cache_dir
    )
    if not model_ref:
        return None, "", "unavailable"

    try:
        from .pyannote_engine import PyannoteDiarizationEngine

        engine = PyannoteDiarizationEngine(
            model_ref=model_ref,
            token=getattr(emb_cfg, "hf_token", "") or None,
            cache_dir=cache_dir,
        )
        expected = getattr(diar_cfg, "expected_speakers", None)
        result = engine.diarize(
            audio=audio,
            sample_rate=sample_rate,
            min_speakers=expected,
            max_speakers=expected or getattr(diar_cfg, "max_speakers", 10),
        )
        result = canonicalize_diarization_result(
            result,
            max_speakers=expected or getattr(diar_cfg, "max_speakers", 10),
            audio=audio,
            sample_rate=sample_rate,
        )
        if expected is None:
            result, model_ref = _verify_single_speaker_auto(
                audio,
                sample_rate,
                diar_cfg,
                model_ref,
                result,
                token=getattr(emb_cfg, "hf_token", "") or None,
                cache_dir=cache_dir,
            )
        return result, model_ref, "ok"
    except Exception as exc:
        logger.warning("Global diarization unavailable: %s", exc)
        return None, model_ref, "failed"


# auto 模式单说话人复核：首个候选模型把多人对话塌缩为单说话人时（真实案例：
# community-1 对相近声线的双人对话返回 1 人，导致全片标成说话人A），用下一个
# auto 候选模型自然重跑一次；第二模型发现更多说话人则采纳其 turns。不设嵌入
# 仲裁门控——实测 turn 级 silhouette 无法区分"真双人"与"被强拆的独白"
# （真双人可低至 0.04），而两个候选模型一致认为单说话人时不会进入采纳分支。
def _verify_single_speaker_auto(
    audio: np.ndarray,
    sample_rate: int,
    diar_cfg: Any,
    first_model_ref: str,
    result: DiarizationResult,
    *,
    token: str | None,
    cache_dir: str,
) -> tuple[DiarizationResult, str]:
    turns = list(result.turns or [])
    speakers = {turn.speaker_id for turn in turns}
    if len(speakers) > 1 or len(turns) < 2:
        return result, first_model_ref
    if getattr(diar_cfg, "global_model", "auto") != "auto":
        return result, first_model_ref

    for candidate in ("diarization-3.1", "community-1"):
        ref = resolve_global_model_ref(candidate)
        if (
            not ref
            or ref == first_model_ref
            or not is_model_cached(candidate, cache_dir)
        ):
            continue
        try:
            from .pyannote_engine import PyannoteDiarizationEngine

            engine = PyannoteDiarizationEngine(
                model_ref=ref,
                token=token,
                cache_dir=cache_dir,
            )
            verify_result = engine.diarize(
                audio=audio,
                sample_rate=sample_rate,
                min_speakers=None,
                max_speakers=getattr(diar_cfg, "max_speakers", 10),
            )
            verify_result = canonicalize_diarization_result(
                verify_result,
                max_speakers=getattr(diar_cfg, "max_speakers", 10),
                audio=audio,
                sample_rate=sample_rate,
            )
        except Exception as exc:
            logger.warning("Single-speaker verify pass failed (%s): %s", ref, exc)
            continue
        verify_speakers = {turn.speaker_id for turn in (verify_result.turns or [])}
        if len(verify_speakers) > 1:
            logger.info(
                "Single-speaker verify: %s found %d speakers "
                "(first model %s reported 1) — adopting multi-speaker turns",
                ref,
                len(verify_speakers),
                first_model_ref,
            )
            return verify_result, ref
        return result, first_model_ref
    return result, first_model_ref


def _global_speaker_at(turns: list[SpeakerTurn], point: float) -> int | None:
    active = [turn for turn in turns if turn.start <= point < turn.end]
    if not active:
        return None
    return max(active, key=lambda turn: turn.end - turn.start).speaker_id


def _global_boundaries(event: Any, turns: list[SpeakerTurn]) -> list[float]:
    relevant = [
        turn for turn in turns if turn.end > event.start and turn.start < event.end
    ]
    speaker_ids = {turn.speaker_id for turn in relevant}
    if len(speaker_ids) < 2:
        return []
    points = []
    for turn in relevant:
        for point in (turn.start, turn.end):
            if event.start + 1e-4 < point < event.end - 1e-4:
                points.append(point)
    return sorted(set(points))


def _embedding_boundaries(
    event: Any,
    audio: np.ndarray,
    sample_rate: int,
    embedding_engine: Any,
    context_seconds: float,
    threshold: float,
) -> list[float]:
    words = sorted(list(event.words or []), key=_word_start)
    if len(words) < 2 or embedding_engine is None:
        return []
    points = []
    for left, right in zip(words, words[1:]):
        boundary_start = _word_end(left)
        boundary_end = _word_start(right)
        left_start = max(float(event.start), boundary_start - context_seconds)
        right_end = min(float(event.end), boundary_end + context_seconds)
        left_audio = _audio_slice(audio, left_start, boundary_start, sample_rate)
        right_audio = _audio_slice(audio, boundary_end, right_end, sample_rate)
        if (
            len(left_audio) < sample_rate * 0.12
            or len(right_audio) < sample_rate * 0.12
        ):
            continue
        try:
            left_feature = embedding_engine.extract_embedding(left_audio, sample_rate)
            right_feature = embedding_engine.extract_embedding(right_audio, sample_rate)
        except Exception:
            continue
        distance = _cosine_distance(left_feature, right_feature) / 2.0
        gap_score = min(1.0, max(0.0, boundary_end - boundary_start) / 0.25)
        score = 0.8 * distance + 0.2 * gap_score
        if score >= threshold:
            points.append(
                boundary_start if boundary_end > boundary_start else boundary_end
            )
    return sorted(set(points))


def _local_global_boundaries(
    event: Any,
    audio: np.ndarray,
    sample_rate: int,
    engine: Any,
    expected_speakers: int | None,
    context_seconds: float,
) -> list[float]:
    if engine is None or len(event.words or []) < 2:
        return []
    start = max(0.0, float(event.start) - context_seconds)
    end = min(len(audio) / max(sample_rate, 1), float(event.end) + context_seconds)
    snippet = _audio_slice(audio, start, end, sample_rate)
    if len(snippet) < max(1, int(sample_rate * 0.25)):
        return []
    try:
        result = engine.diarize(
            audio=snippet,
            sample_rate=sample_rate,
            min_speakers=expected_speakers,
            max_speakers=expected_speakers or 10,
        )
    except Exception:
        return []
    labels = {turn.speaker_id for turn in result.turns}
    if len(labels) < 2:
        return []
    points = []
    for turn in result.turns:
        for point in (turn.start + start, turn.end + start):
            if event.start + 1e-4 < point < event.end - 1e-4:
                points.append(point)
    return sorted(set(points))


def _copy_event_part(
    event: Any,
    words: list[Any],
    start: float,
    end: float,
    index: int,
    source_word_ids: list[str] | None = None,
) -> Any:
    part = copy.deepcopy(event)
    part.index = index
    part.words = words
    part.start = start
    part.end = end
    text = "".join(_word_text(word) for word in words).strip()
    if text:
        part.text = text
    part.original_text = text or part.original_text
    part.source_word_ids = list(source_word_ids or [])
    if part.physical_start is not None:
        part.physical_start = max(part.physical_start, start)
    if part.physical_end is not None:
        part.physical_end = min(part.physical_end, end)
    return part


def _split_event(
    event: Any,
    points: list[float],
    min_part_duration: float = 0.0,
) -> list[Any]:
    original_words = list(event.words or [])
    words = sorted(original_words, key=_word_start)
    points = sorted({point for point in points if event.start < point < event.end})
    if len(words) < 2 or not points:
        return [event]
    boundaries = [float(event.start), *points, float(event.end)]
    groups: list[list[Any]] = [[] for _ in range(len(boundaries) - 1)]
    for word in words:
        midpoint = (_word_start(word) + _word_end(word)) / 2.0
        group_index = min(
            len(groups) - 1,
            max(
                0,
                next(
                    (
                        index
                        for index in range(len(boundaries) - 1)
                        if midpoint < boundaries[index + 1]
                    ),
                    len(groups) - 1,
                ),
            ),
        )
        groups[group_index].append(word)
    groups = [group for group in groups if group]
    if len(groups) < 2:
        return [event]
    parts = []
    original_positions = {
        id(word): position for position, word in enumerate(original_words)
    }
    original_source_ids = list(getattr(event, "source_word_ids", []) or [])
    for index, group in enumerate(groups):
        start = max(float(event.start), _word_start(group[0]))
        end = min(float(event.end), _word_end(group[-1]))
        if end - start < 1e-4:
            continue
        if min_part_duration > 0 and end - start < min_part_duration:
            return [event]
        group_positions = {
            original_positions[id(word)]
            for word in group
            if id(word) in original_positions
        }
        part_source_ids = (
            [
                source_id
                for position, source_id in enumerate(original_source_ids)
                if position in group_positions
            ]
            if len(original_source_ids) == len(original_words)
            else []
        )
        parts.append(
            _copy_event_part(
                event,
                group,
                start,
                end,
                event.index + index,
                part_source_ids,
            )
        )
    return parts or [event]


def _map_global_to_embedding(
    result: DiarizationResult | None, events: list[Any], labels: list[int]
) -> dict[int, int]:
    if result is None or not labels:
        return {}
    scores: dict[tuple[int, int], float] = {}
    for event, label in zip(events, labels):
        for turn in result.turns:
            overlap = max(0.0, min(event.end, turn.end) - max(event.start, turn.start))
            if overlap:
                scores[(turn.speaker_id, label)] = (
                    scores.get((turn.speaker_id, label), 0.0) + overlap
                )
    mapping: dict[int, int] = {}
    used_labels: set[int] = set()
    candidates = sorted(
        ((score, global_id, label) for (global_id, label), score in scores.items()),
        reverse=True,
    )
    for score, global_id, label in candidates:
        if global_id in mapping or label in used_labels:
            continue
        mapping[global_id] = label
        used_labels.add(label)
    global_ids = {turn.speaker_id for turn in result.turns}
    if len(mapping) != len(global_ids):
        # A partial mapping would silently collapse a global identity into a
        # different embedding cluster. Keep the global IDs authoritative.
        return {}
    return mapping


def run_speaker_fusion(
    events: list,
    audio: np.ndarray,
    sample_rate: int,
    config: Any,
    *,
    embedding_engine: Any = None,
    language: str | None = None,
) -> SpeakerFusionResult:
    """Run both speaker lines and return final events with provenance."""
    if not events or not getattr(config.diarization, "enabled", True):
        return SpeakerFusionResult(events=events, status="disabled", backend="disabled")

    diar_cfg = config.diarization
    embedding = _extract_embedding_evidence(
        events,
        audio,
        sample_rate,
        embedding_engine,
        diar_cfg,
    )
    fusion_mode = getattr(diar_cfg, "fusion_mode", "auto")
    if fusion_mode == "embedding":
        global_result, global_model_ref, global_status = None, "", "disabled"
    else:
        global_result, global_model_ref, global_status = _run_global_pass(
            audio,
            sample_rate,
            config,
        )
    use_global = global_result is not None
    if getattr(diar_cfg, "fusion_mode", "auto") == "dual" and global_result is None:
        global_status = "degraded"

    # 双声学线均不可用（如未安装 speechbrain/pyannote 的最小环境）时，
    # 退回内置 MFCC+音高凝聚聚类，至少给出粗粒度的说话人区分。
    # 仅在引擎根本不可用（unavailable）时触发；引擎存在但提取失败
    # （failed）说明音频本身有问题，保持 unknown 更诚实。
    fallback_labels: dict | None = None
    if embedding.status == "unavailable" and not use_global:
        fallback_labels = _fallback_cluster_labels(
            events,
            audio,
            sample_rate,
            diar_cfg,
        )

    global_turns = (
        list(global_result.exclusive_turns or global_result.turns) if use_global else []
    )
    global_map = _map_global_to_embedding(
        global_result if use_global else None, events, embedding.labels
    )
    global_engine = None
    if use_global and getattr(diar_cfg, "local_refinement", "embedding") == "full":
        try:
            from .pyannote_engine import PyannoteDiarizationEngine

            global_engine = PyannoteDiarizationEngine(
                model_ref=global_model_ref,
                token=getattr(config.speaker_embedding, "hf_token", "") or None,
                cache_dir=getattr(config.speaker_embedding, "cache_dir", "")
                or str(DEFAULT_CACHE_DIR),
            )
            global_engine.load_model()
        except Exception:
            global_engine = None

    final_events: list[Any] = []
    local_split_count = 0
    conflict_count = 0
    expected = getattr(diar_cfg, "expected_speakers", None)
    local_mode = getattr(diar_cfg, "local_refinement", "embedding")
    if getattr(diar_cfg, "diarization_scope", "hierarchical") == "global":
        local_mode = "off"

    for event_index, event in enumerate(events):
        points = _global_boundaries(event, global_turns)
        if local_mode in ("embedding", "full"):
            points.extend(
                _embedding_boundaries(
                    event,
                    audio,
                    sample_rate,
                    embedding_engine,
                    getattr(diar_cfg, "local_context_seconds", 0.6),
                    getattr(diar_cfg, "min_change_confidence", 0.70),
                )
            )
        if local_mode == "full":
            points.extend(
                _local_global_boundaries(
                    event,
                    audio,
                    sample_rate,
                    global_engine,
                    expected,
                    getattr(diar_cfg, "local_context_seconds", 0.6),
                )
            )
        parts = _split_event(
            event,
            points,
            getattr(diar_cfg, "min_local_segment_seconds", 0.0),
        )
        if len(parts) > 1:
            local_split_count += len(parts) - 1

        for part in parts:
            midpoint = (part.start + part.end) / 2.0
            global_id = _global_speaker_at(global_turns, midpoint)
            embedding_id = None
            if embedding.status == "ok" and embedding_engine is not None:
                try:
                    snippet = _audio_slice(
                        audio,
                        max(
                            0.0,
                            part.start
                            - getattr(diar_cfg, "local_context_seconds", 0.6),
                        ),
                        min(
                            len(audio) / max(sample_rate, 1),
                            part.end + getattr(diar_cfg, "local_context_seconds", 0.6),
                        ),
                        sample_rate,
                    )
                    feature = embedding_engine.extract_embedding(snippet, sample_rate)
                    embedding_id = _nearest_centroid(feature, embedding.centroids)
                except Exception:
                    embedding_id = None
            if (
                embedding_id is None
                and embedding.status == "ok"
                and event_index < len(embedding.labels)
            ):
                embedding_id = embedding.labels[event_index]

            if global_id is not None and embedding_id is not None:
                mapped = global_map.get(global_id)
                if mapped is not None and mapped != embedding_id:
                    # 两条声学线归属冲突。过去直接置空(unknown),把可裁决的
                    # 证据丢弃了——嵌入线是 3s 粗窗口聚类,短事件/跨turn事件
                    # 频繁误标,是冲突的主要来源;全局 diarization 是专门的
                    # "谁在何时说话"模型,证据更强。冲突时择优回退到全局线,
                    # 并保留冲突计数供诊断。
                    conflict_count += 1
                    speaker_id = mapped
                    source = "global_conflict_fallback"
                elif mapped is not None:
                    speaker_id = mapped
                    source = "fused"
                else:
                    speaker_id = global_id
                    source = "global"
            elif global_id is not None:
                speaker_id = global_id
                source = "global"
            elif embedding_id is not None:
                speaker_id = embedding_id
                source = "embedding"
            elif fallback_labels is not None and event_index in fallback_labels:
                speaker_id = fallback_labels[event_index]
                source = "fallback_cluster"
            else:
                speaker_id = None
                source = "unknown"

            part.speaker_id = int(speaker_id) if speaker_id is not None else None
            part.speaker_source = source
            part.speaker_status = "confirmed" if speaker_id is not None else "unknown"
            if source == "global_conflict_fallback":
                part.speaker_repair_reason = "speaker_evidence_conflict_resolved_global"
            elif source == "unknown":
                part.speaker_repair_reason = "speaker_evidence_unavailable"
            else:
                part.speaker_repair_reason = ""
            model_refs = [ref for ref in (embedding.model, global_model_ref) if ref]
            part.speaker_model = "+".join(dict.fromkeys(model_refs)) or None
            # Engines currently expose no calibrated posterior. Keep this field
            # optional instead of presenting a fabricated numeric confidence.
            part.speaker_confidence = None
            final_events.append(part)

    # Compact IDs and ensure labels remain stable after local splitting.
    unique = sorted(
        {event.speaker_id for event in final_events if event.speaker_id is not None}
    )
    remap = {old: new for new, old in enumerate(unique)}
    for index, event in enumerate(final_events, start=1):
        if event.speaker_id is not None:
            event.speaker_id = remap[event.speaker_id]
            event.speaker_label = _speaker_label(language, event.speaker_id)
        else:
            event.speaker_label = None
        event.index = index

    if use_global and embedding.status == "ok":
        backend = "fused"
    elif use_global:
        backend = "pyannote"
    elif embedding.status == "ok":
        backend = "embedding"
    elif fallback_labels is not None:
        # 声学嵌入/全局分离不可用时的 MFCC+音高凝聚聚类回退
        backend = "agglomerative"
    else:
        backend = "unknown"
    status = (
        "ok"
        if final_events and any(e.speaker_id is not None for e in final_events)
        else "degraded"
    )
    # 回退线已经给出说话人时，嵌入线缺失不再把整体状态压回 degraded
    if global_status in ("failed", "degraded") or (
        embedding.status in ("failed", "unavailable") and fallback_labels is None
    ):
        status = "degraded" if status == "ok" else status
    unknown_count = sum(event.speaker_id is None for event in final_events)
    diagnostics = {
        "embedding_model": embedding.model,
        "embedding_status": embedding.status,
        "embedding_silhouette": embedding.silhouette,
        "global_model": global_model_ref,
        "global_status": global_status,
        "global_turn_count": len(global_turns),
        "local_split_count": local_split_count,
        "conflict_count": conflict_count,
        "unknown_count": unknown_count,
        "expected_speakers": expected,
    }
    return SpeakerFusionResult(
        events=final_events,
        speaker_count=len(unique),
        backend=backend,
        status=status,
        model_ref=global_model_ref or embedding.model,
        local_split_count=local_split_count,
        conflict_count=conflict_count,
        unknown_count=unknown_count,
        diagnostics=diagnostics,
    )
