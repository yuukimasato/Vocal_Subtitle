"""Cloud LLM merge decision and deterministic fallback service."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from . import merge_constraints, merge_policy

logger = logging.getLogger(__name__)

MERGE_DECISION_PROMPT = """You are a subtitle merging expert.
Given a sequence of subtitle fragments with precise timestamps,
decide which adjacent fragments should be merged into a single subtitle line.

Hard Rules (these OVERRIDE any semantic judgment):
1. If silence gap > 1.2 seconds -> ABSOLUTELY FORBIDDEN to merge.
2. If speakers are different -> ABSOLUTELY FORBIDDEN to merge.
3. If combined duration > 5.0 seconds -> FORBIDDEN to merge.
4. If either fragment ends with terminal punctuation and gap > 400ms -> default to NOT merging.

Semantic Rules (only apply when Hard Rules allow):
- Merge ONLY if the combined text forms a COMPLETE semantic unit.
- Do NOT merge if the gap represents a natural sentence boundary.
- A trailing comma or incomplete clause strongly suggests merging with the next fragment.

Input format: JSON array of fragments with id, start, end, speaker, text, gap_to_next_sec, gap_is_silent.
Output format: JSON with merge_groups array.
"""


class LLMMergeDecider:
    """Own cloud request, response parsing and rule fallback."""

    def __init__(self, config: Any):
        self.config = config

    def decide(self, candidates: list[dict]) -> list[dict]:
        try:
            return self.decide_core(candidates)
        except Exception:
            if self.config.llm_fallback_to_rules:
                return self.fallback_rule_decisions(candidates)
            raise

    def decide_core(self, candidates: list[dict]) -> list[dict]:
        cfg = self.config
        api_url = (
            f"{cfg.llm_base_url}/v1/chat/completions" if cfg.llm_base_url else None
        )
        if not api_url:
            logger.warning("No LLM API URL configured, using fallback rules")
            return self.fallback_rule_decisions(candidates)

        llm_input = []
        for frag in candidates:
            gap_is_silent = frag.get("gap_is_silent")
            llm_input.append(
                {
                    "id": frag.get("id", 0),
                    "start": round(float(frag.get("start", 0)), 2),
                    "end": round(float(frag.get("end", 0)), 2),
                    "speaker": str(frag.get("speaker", "unknown")),
                    "text": str(frag.get("text", "")),
                    "gap_to_next_sec": (
                        round(float(frag.get("gap_to_next_sec", 0)), 3)
                        if frag.get("gap_to_next_sec") is not None
                        else None
                    ),
                    "gap_is_silent": (
                        bool(gap_is_silent) if gap_is_silent is not None else None
                    ),
                }
            )

        messages = [
            {"role": "system", "content": MERGE_DECISION_PROMPT},
            {
                "role": "user",
                "content": json.dumps(llm_input, ensure_ascii=False, indent=2),
            },
        ]
        payload = {
            "model": cfg.llm_model,
            "messages": messages,
            "temperature": cfg.llm_temperature,
            "max_tokens": 2000,
        }
        headers = {"Content-Type": "application/json"}
        if cfg.llm_api_key:
            headers["Authorization"] = f"Bearer {cfg.llm_api_key}"

        import requests

        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=cfg.llm_timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if "```json" in content:
            start = content.index("```json") + 7
            end = content.index("```", start)
            json_str = content[start:end].strip()
        elif "{" in content:
            json_str = content[content.index("{") : content.rindex("}") + 1]
        else:
            json_str = content
        groups = json.loads(json_str).get("merge_groups", [])
        logger.info(
            "LLM merge: %d candidates -> %d groups", len(candidates), len(groups)
        )
        return groups

    def fallback_rule_decisions(self, candidates: Sequence[dict]) -> list[dict]:
        groups = []
        sentence_endings = {".", "!", "?", "。", "！", "？"}
        comma_endings = {",", "，", ";", "；"}

        for frag in candidates:
            gap = frag.get("gap_to_next_sec", 999)
            if gap is None or gap < -0.02:
                continue
            text = frag.get("text", "").rstrip()
            next_id = frag.get("id", 0) + 1
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
            if text and text[-1] in comma_endings and gap < 0.6:
                groups.append(
                    {
                        "ids": [frag["id"], next_id],
                        "reason": "[fallback] comma + moderate gap -> merge",
                    }
                )
            elif text and text[-1] in sentence_endings and gap > 0.4:
                continue
            elif gap < 0.4:
                groups.append(
                    {
                        "ids": [frag["id"], next_id],
                        "reason": "[fallback] short gap -> merge",
                    }
                )
        return groups


__all__ = ["LLMMergeDecider", "MERGE_DECISION_PROMPT"]
