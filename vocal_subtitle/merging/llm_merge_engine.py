"""LLM 语义合并引擎 (方案五) — backward-compat re-export hub.

The original implementation has been split into focused submodules:

- merging/merge_engine.py   — LLMMergeEngine class (thin entry point)
- merging/merge_policy.py   — Fast-Slow Path, local NLP, LLM call decisions
- merging/local_decider.py  — sentence-transformers lifecycle and similarity
- merging/llm_decider.py    — cloud LLM request, timeout, fallback
- merging/layout.py         — frame seamless, line break, layout suggestions
- merging/merge_constraints.py — physical owner, speaker, gap hard constraints
"""

# Re-export public surface for backward compatibility
from .merge_engine import (  # noqa: F401
    LLMMergeEngine,
    MergeDecisionConfig,
)
from .layout import (  # noqa: F401
    apply_frame_seamless_stitching,
    apply_layout_suggestions,
    auto_layout_events,
    auto_line_break_fallback,
)
from .merge_constraints import (  # noqa: F401
    _detect_semantic_boundary,
    _physical_owner_compatible,
    _physical_owner_compatible_for_events,
)
