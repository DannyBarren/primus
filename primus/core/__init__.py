"""Primus core layer: directives, personality, prompts, and command guardrails.

Everything here is plain Python and fully editable — no locks, no read-only enforcement.
Extracted verbatim from the original admin_assistant.py so behavior is identical.
"""
from .obedience import GLOBAL_OBEDIENCE
from .personality import CORE_OPERATOR_PROFILE
from .guardrails import (
    DANGEROUS_PATTERNS,
    SAFE_COMMAND_LEADERS,
    SAFE_LEADING_RE,
    UNSAFE_IN_SAFE,
)
from .prompts import (
    FORGE_PLANNER_PROMPT,
    PLANNER_PROMPT,
    SINGLE_STEP_DIRECTIVE,
    STEP_PROMPT,
    SUMMARY_PROMPT,
    FORGE_SYSTEM,
    FAST_CHAT_SYSTEM,
    PRIMUS_SYSTEM,
)

__all__ = [
    "GLOBAL_OBEDIENCE", "CORE_OPERATOR_PROFILE",
    "DANGEROUS_PATTERNS", "SAFE_COMMAND_LEADERS", "SAFE_LEADING_RE", "UNSAFE_IN_SAFE",
    "FORGE_PLANNER_PROMPT", "PLANNER_PROMPT", "SINGLE_STEP_DIRECTIVE", "STEP_PROMPT",
    "SUMMARY_PROMPT",
    "FORGE_SYSTEM", "FAST_CHAT_SYSTEM", "PRIMUS_SYSTEM",
]
