"""Shell risk classification — the gate ``run_shell`` and launch helpers consult.

Patterns live in ``guardrails``. This module is the decision procedure: dangerous
(always queue), review (queue in Suggest; Execute may run), or safe (auto-run when
configured). Designed to be imported without the host module so unit tests stay hermetic.
"""
from __future__ import annotations

import re

from primus.core.guardrails import (
    DANGEROUS_PATTERNS,
    SAFE_LEADING_RE,
    UNSAFE_IN_SAFE,
)

# Command substitution, backticks, and ${} expansion that can hide a payload.
_SUBSTITUTION_RE = re.compile(r"\$\(|`|\$\{")
# Background job / lone ampersand (not &&, not 2>&1, not &>).
_BACKGROUND_RE = re.compile(r"(?:^|[\s;])&(?:\s|$)|(?:^|[\s;])&$")
# ``env <prog>`` / ``env VAR=val <prog>`` — env is not a read-only inspector here.
_ENV_EXEC_RE = re.compile(r"^\s*env\b(?:\s+\w+=\S+)*\s+\S+", re.I)

# git <flags> push --force / git -C repo push -f  (flags may sit between git and push)
_GIT_FORCE_PUSH_RE = re.compile(
    r"\bgit\b(?:\s+\S+)*\s+push\b.*(?:\s(?:-f|--force|--force-with-lease)\b)",
    re.I,
)


def classify_command(command: str) -> tuple[bool, list[str]]:
    """Return ``(is_dangerous, reasons)`` from the static dangerous-pattern list."""
    reasons = [label for pattern, label in DANGEROUS_PATTERNS if pattern.search(command)]
    return bool(reasons), reasons


def split_shell_pipeline(command: str) -> list[str]:
    """Split on pipes for per-segment safety checks."""
    return [seg.strip() for seg in command.split("|") if seg.strip()]


def split_shell_chain(command: str) -> list[str]:
    """Split on ``&&`` or ``;`` for chained commands. Does not treat ``&`` as a chain."""
    return [p.strip() for p in re.split(r"\s*&&\s*|\s*;\s*", command.strip()) if p.strip()]


def _is_safe_command_part(part: str) -> bool:
    part = part.strip()
    if not part or UNSAFE_IN_SAFE.search(part):
        return False
    if re.search(r"(?<![<\\])>>?(?!>)", part):
        return False
    for segment in split_shell_pipeline(part):
        if not SAFE_LEADING_RE.match(segment):
            return False
    return True


def assess_command_risk(command: str) -> tuple[str, list[str]]:
    """Return risk tier: ``safe`` | ``review`` | ``dangerous``.

    Elevations that the old leader/block-list missed:
    - newlines (a second command after a "safe" first line)
    - ``$()``, backticks, ``${}``
    - lone ``&`` (background)
    - ``env <prog>`` (env is an executor, not ``printenv``)
    - ``git … push --force`` with flags between ``git`` and ``push``
    """
    command = (command or "").strip()
    if not command:
        return "review", []

    extra: list[str] = []
    if "\n" in command or "\r" in command:
        extra.append("embedded newline")
    if _SUBSTITUTION_RE.search(command):
        extra.append("command substitution")
    if _BACKGROUND_RE.search(command) or command.rstrip().endswith("&"):
        extra.append("background execution")
    if _ENV_EXEC_RE.search(command):
        extra.append("env executing a program")
    if _GIT_FORCE_PUSH_RE.search(command):
        extra.append("force git push")

    is_dangerous, reasons = classify_command(command)
    reasons = list(reasons) + extra

    # Substitution / newlines hide a second command — treat as dangerous, not review.
    if extra and any(
        tag in extra
        for tag in ("command substitution", "embedded newline", "force git push")
    ):
        return "dangerous", reasons
    if is_dangerous:
        return "dangerous", reasons

    for part in split_shell_chain(command):
        if not _is_safe_command_part(part):
            return "review", reasons + [f"needs review: {part[:60]}"]
    if extra:
        return "review", reasons
    return "safe", []
