"""Composable Click command groups for the public CLI facade."""

from .administration import register as register_administration
from .feedback_commands import register as register_feedback

__all__ = ["register_administration", "register_feedback"]
