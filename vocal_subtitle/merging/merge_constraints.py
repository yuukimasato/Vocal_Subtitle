"""Hard merge constraints shared by rule, local and cloud decisions."""

from __future__ import annotations


def physical_owner_compatible(left: dict, right: dict) -> bool:
    left_bin = left.get("physical_bin_id")
    right_bin = right.get("physical_bin_id")
    if left_bin is not None or right_bin is not None:
        return left_bin is not None and right_bin is not None and left_bin == right_bin
    left_spans = left.get("physical_spans", []) or []
    right_spans = right.get("physical_spans", []) or []
    if not left_spans or not right_spans:
        return True
    left_clips = {span.get("physical_clip_id") or span.get("clip_id") for span in left_spans}
    right_clips = {span.get("physical_clip_id") or span.get("clip_id") for span in right_spans}
    return bool(left_clips & right_clips)


def physical_owner_compatible_for_events(left: object, right: object) -> bool:
    left_bin = getattr(left, "physical_bin_id", None)
    right_bin = getattr(right, "physical_bin_id", None)
    if left_bin is not None or right_bin is not None:
        return left_bin is not None and right_bin is not None and left_bin == right_bin
    left_spans = list(getattr(left, "physical_spans", []) or [])
    right_spans = list(getattr(right, "physical_spans", []) or [])
    if not left_spans or not right_spans:
        return True

    def clip_id(span: object):
        if isinstance(span, dict):
            return span.get("physical_clip_id") or span.get("clip_id")
        return getattr(span, "clip_id", None) or getattr(span, "physical_clip_id", None)

    return bool({clip_id(span) for span in left_spans} & {clip_id(span) for span in right_spans})


def can_merge_gap(gap: float, max_gap: float, *, allow_overlap: bool = False) -> bool:
    """Apply the hard gap guard before semantic decisions."""
    if not allow_overlap and gap < -0.02:
        return False
    return gap < max_gap


# Historical private spellings.
_physical_owner_compatible = physical_owner_compatible
_physical_owner_compatible_for_events = physical_owner_compatible_for_events
