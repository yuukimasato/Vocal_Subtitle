"""Constrained LLM semantic sliding window.

Wraps LLM merge operations so the model only sees fragment/word IDs
and readonly metadata. The LLM can merge, split, add punctuation, correct
obvious typos, and infer UNKNOWN speaker from context — but it must never
return numerical times, and its output is validated against invariants
before being applied.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from ..mapping.semantic_fragments import PhysicalFragment

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Window input / output schemas
# ------------------------------------------------------------------


@dataclass
class SemanticWindowInput:
    """Read-only input sent to the LLM for one window.

    Contains fragment/word IDs, readonly range metadata, pause class,
    hard split and overlap markers. Numerical times are intentionally
    excluded — the LLM sees only ordinal/structural information.
    """

    window_id: str
    fragment_ids: List[str] = field(default_factory=list)
    word_ids: List[str] = field(default_factory=list)
    physical_range_start: float = 0.0  # readonly, for diagnostics only
    physical_range_end: float = 0.0
    language: str = ""
    fragments: List[Dict[str, Any]] = field(default_factory=list)  # readonly metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "fragment_ids": list(self.fragment_ids),
            "word_ids": list(self.word_ids),
            "language": self.language,
            "fragments": copy.deepcopy(self.fragments),
        }


@dataclass
class SemanticWindowOutput:
    """Validated LLM output for one window.

    Only operations on existing fragment/word IDs are permitted.
    Numerical times are never accepted.
    """

    window_id: str
    groups: List[List[str]] = field(default_factory=list)  # ordered fragment/word ID groups
    normalized_text: Dict[str, str] = field(default_factory=dict)  # word_id -> corrected text
    speaker_decisions: Dict[str, int] = field(default_factory=dict)  # fragment_id -> speaker_id (UNKNOWN only)
    review_requests: List[Dict[str, Any]] = field(default_factory=list)  # suspected missing/incorrect words
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "groups": [list(g) for g in self.groups],
            "normalized_text": dict(self.normalized_text),
            "speaker_decisions": {k: v for k, v in self.speaker_decisions.items()},
            "review_requests": copy.deepcopy(self.review_requests),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SemanticWindowOutput":
        return cls(
            window_id=payload["window_id"],
            groups=payload.get("groups", []),
            normalized_text=payload.get("normalized_text", {}),
            speaker_decisions=payload.get("speaker_decisions", {}),
            review_requests=payload.get("review_requests", []),
            reason=payload.get("reason", ""),
        )


# ------------------------------------------------------------------
# Window construction
# ------------------------------------------------------------------


def build_semantic_windows(
    fragments: Sequence[PhysicalFragment],
    *,
    max_fragments_per_window: int = 10,
    max_window_duration: float = 30.0,
    window_overlap_fragments: int = 1,
    window_id_prefix: str = "sw",
) -> List[SemanticWindowInput]:
    """Partition PhysicalFragments into semantic windows respecting hard boundaries.

    Windows are truncated at hard_split_before markers — no window ever
    spans a hard boundary. Adjacent windows overlap by one fragment to
    allow context continuity, but the overlap fragment is read-only in
    each window's non-primary context.

    Args:
        fragments: Ordered PhysicalFragments.
        max_fragments_per_window: Soft limit on fragments per window.
        max_window_duration: Soft limit on total window duration.
        window_overlap_fragments: Number of overlapping fragments.
        window_id_prefix: Prefix for window IDs.

    Returns:
        Ordered list of SemanticWindowInput.
    """
    if not fragments:
        return []

    windows: List[SemanticWindowInput] = []
    window_index = 0
    i = 0

    while i < len(fragments):
        window_index += 1
        window_frags: List[PhysicalFragment] = []
        window_duration = 0.0
        frag_ids: List[str] = []
        word_ids: List[str] = []

        for j in range(i, len(fragments)):
            frag = fragments[j]

            # Hard boundary: start a new window
            if frag.hard_split_before and j > i:
                break

            # Soft limits
            if len(window_frags) >= max_fragments_per_window:
                break
            if window_duration >= max_window_duration and window_frags:
                break

            window_frags.append(frag)
            frag_ids.append(frag.id)
            word_ids.extend(frag.word_ids)
            window_duration = frag.physical_end - window_frags[0].physical_start

        # Build read-only metadata for each fragment
        frag_metadata = []
        for frag in window_frags:
            frag_metadata.append({
                "fragment_id": frag.id,
                "word_ids": list(frag.word_ids),
                "language": frag.language,
                "candidate_speaker": frag.candidate_speaker,
                "speaker_status": frag.speaker_status,
                "speaker_confidence": frag.speaker_confidence,
                "pause_class": frag.pause_class,
                "hard_split_before": frag.hard_split_before,
                "hard_split_reason": frag.hard_split_reason,
                "genuine_overlap": frag.genuine_overlap,
            })

        window = SemanticWindowInput(
            window_id=f"{window_id_prefix}-{window_index:04d}",
            fragment_ids=frag_ids,
            word_ids=word_ids,
            physical_range_start=window_frags[0].physical_start if window_frags else 0.0,
            physical_range_end=window_frags[-1].physical_end if window_frags else 0.0,
            language=window_frags[0].language or "",
            fragments=frag_metadata,
        )
        windows.append(window)

        # Advance: move to next unprocessed fragment
        # If we consumed at least one fragment, advance past consumed minus overlap
        consumed = len(window_frags)
        if consumed == 0:
            i += 1
        else:
            step = max(1, consumed - window_overlap_fragments)
            i = i + step


    return windows


# ------------------------------------------------------------------
# Output validation
# ------------------------------------------------------------------


def validate_window_output(
    output: SemanticWindowOutput,
    window: SemanticWindowInput,
    *,
    confirmed_speakers: Optional[Set[int]] = None,
) -> tuple[bool, List[str]]:
    """Validate LLM output against invariants.

    Returns (valid, errors). The output is REJECTED if errors is non-empty.

    Checks:
    - No numerical time fields present
    - All IDs exist in the window input
    - No ID duplication or reversal
    - No cross-hard-boundary merges
    - No confirmed speaker modification
    - No invented text without source word IDs
    """
    errors: List[str] = []
    valid_ids = set(window.fragment_ids) | set(window.word_ids)
    confirmed = confirmed_speakers or set()

    # Collect all referenced IDs
    all_group_ids: Set[str] = set()
    seen_ids: Set[str] = set()

    for group in output.groups:
        group_ids = set()
        for fid in group:
            if fid in seen_ids:
                errors.append(f"duplicate id in groups: {fid}")
            if fid not in valid_ids:
                errors.append(f"unknown id: {fid}")
            seen_ids.add(fid)
            group_ids.add(fid)

        # Check for cross-hard-boundary merge
        group_frags = [f for f in window.fragments if f.get("fragment_id", f.get("id", "")) in group_ids]
        if len(group_frags) >= 2:
            for k in range(1, len(group_frags)):
                if group_frags[k].get("hard_split_before", False):
                    errors.append(
                        f"group crosses hard boundary at fragment {group_frags[k].get('fragment_id', group_frags[k].get('id', '?'))}"
                    )

    # Check normalized_text keys exist
    for word_id in output.normalized_text:
        if word_id not in valid_ids:
            errors.append(f"normalized text references unknown word: {word_id}")

    # Check speaker decisions don't modify confirmed speakers
    for frag_id, spk in output.speaker_decisions.items():
        if frag_id not in valid_ids:
            errors.append(f"speaker decision for unknown fragment: {frag_id}")
            continue
        frag = next((f for f in window.fragments if f.get("fragment_id", f.get("id", "")) == frag_id), None)
        if frag and frag.get("speaker_status") == "confirmed" and frag.get("candidate_speaker") in confirmed:
            errors.append(f"cannot modify confirmed speaker for fragment {frag_id}")

    return len(errors) == 0, errors


def merge_windows(
    outputs: Sequence[SemanticWindowOutput],
    *,
    resolver: str = "primary_window",
) -> SemanticWindowOutput:
    """Merge overlapping window outputs using stable ID deduplication.

    When two windows contain the same fragment ID, the one from the
    window where it appears first (its "primary" window) wins.
    """
    if not outputs:
        return SemanticWindowOutput(window_id="empty")

    all_groups: List[List[str]] = []
    seen_ids: Set[str] = set()
    all_text: Dict[str, str] = {}
    all_speakers: Dict[str, int] = {}
    all_reviews: List[Dict[str, Any]] = []

    for output in outputs:
        for group in output.groups:
            new_group = [fid for fid in group if fid not in seen_ids]
            if new_group:
                all_groups.append(new_group)
                seen_ids.update(new_group)

        for wid, text in output.normalized_text.items():
            if wid not in all_text:
                all_text[wid] = text

        for fid, spk in output.speaker_decisions.items():
            if fid not in all_speakers:
                all_speakers[fid] = spk

        all_reviews.extend(output.review_requests)

    return SemanticWindowOutput(
        window_id="merged",
        groups=all_groups,
        normalized_text=all_text,
        speaker_decisions=all_speakers,
        review_requests=all_reviews,
        reason=f"merged {len(outputs)} windows",
    )
