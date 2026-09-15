"""Local rule and semantic merge decision service."""

from __future__ import annotations

import logging
from typing import Any

from . import merge_constraints, merge_policy

logger = logging.getLogger(__name__)


class LocalMergeDecider:
    """Own local model lifecycle and deterministic local merge decisions."""

    def __init__(self, config: Any):
        self.config = config
        self._model = None
        self._model_attempted = False

    def load_model(self):
        if self._model is not None:
            return self._model
        if self._model_attempted:
            return None
        self._model_attempted = True
        from ..utils.model_loader import load_sentence_transformer

        self._model = load_sentence_transformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        if self._model is not None:
            logger.info("Local NLP model loaded for merge decisions")
        return self._model

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        model = self.load_model()
        if model is None or not text_a or not text_b:
            return 0.5
        try:
            import numpy as np

            embeddings = model.encode(
                [text_a, text_b],
                convert_to_numpy=True,
            )
            dot = float(np.dot(embeddings[0], embeddings[1]))
            norm_a = float(np.linalg.norm(embeddings[0]))
            norm_b = float(np.linalg.norm(embeddings[1]))
            return dot / max(norm_a * norm_b, 1e-8)
        except Exception as exc:
            logger.debug("Similarity computation failed: %s", exc)
            return 0.5

    def decide(self, candidates: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return local decisions and candidates requiring cloud review."""
        comma_endings = {",", "，", "、", ";", "；"}
        sentence_endings = {".", "!", "?", "。", "！", "？"}
        decided_groups = []
        unresolved = []

        for frag in candidates:
            gap = frag.get("gap_to_next_sec", 999)
            if gap is None or gap > 10 or gap < -0.02:
                continue

            text = frag.get("text", "").rstrip()
            frag_id = frag.get("id", 0)
            next_id = frag_id + 1
            next_frag = next(
                (other for other in candidates if other.get("id") == next_id),
                None,
            )
            next_text = next_frag.get("text", "").strip() if next_frag else ""
            next_speaker = next_frag.get("speaker", "") if next_frag else ""
            curr_speaker = frag.get("speaker", "")

            if curr_speaker and next_speaker and curr_speaker != next_speaker:
                continue
            if (
                next_frag is not None
                and not merge_constraints.physical_owner_compatible(frag, next_frag)
            ):
                continue
            if next_text and merge_policy.detect_semantic_boundary(text, next_text):
                continue

            if text and text[-1] in comma_endings:
                decided_groups.append(
                    {
                        "ids": [frag_id, next_id],
                        "reason": "[local] comma/clause continuation -> merge",
                    }
                )
                continue
            if text and text[-1] in sentence_endings and gap > 0.35:
                continue

            local_min, local_max = self.config.local_nlp_gap_range
            if local_min <= gap <= local_max:
                similarity = (
                    self.compute_similarity(text, next_text) if next_text else 0.5
                )
                if similarity > 0.8:
                    decided_groups.append(
                        {
                            "ids": [frag_id, next_id],
                            "reason": f"[local] high similarity ({similarity:.2f}) -> merge",
                        }
                    )
                    continue
                if similarity < 0.3:
                    continue
            unresolved.append(frag)

        if decided_groups:
            logger.info(
                "Local NLP: %d candidates -> %d merged, %d -> cloud LLM",
                len(candidates),
                len(decided_groups),
                len(unresolved),
            )
        return decided_groups, unresolved


__all__ = ["LocalMergeDecider"]
