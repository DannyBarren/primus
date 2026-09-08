"""FileMaid safety helpers — block system roots and junk segments.

Labels and gates only. No quarantine executor, no DryRunContext this pass.
``_guard_path`` on the host remains the write gate for ``write_file`` / ``move_path``.
Library modules reach the host via ``_host`` — do not import admin_assistant.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

_host = sys.modules.get("admin_assistant") or sys.modules["__main__"]

PathLike = Union[str, Path]

# Linux system roots. ``/`` is special-cased below — putting it here would
# block HOME because ``/`` is a parent of every absolute path.
# Do not block all of ``/var``; only ``/var/lib``.
BLOCKED_PATHS: tuple[Path, ...] = (
    Path("/etc"),
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/boot"),
    Path("/sys"),
    Path("/proc"),
    Path("/dev"),
    Path("/root"),
    Path("/var/lib"),
    Path("/lib"),
    Path("/lib64"),
)

BLOCKED_SEGMENTS: frozenset[str] = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
})


def normalize_path(path: PathLike) -> Path:
    """``expanduser().resolve()``. Raises on failure so callers can refuse."""
    return Path(path).expanduser().resolve()


def is_path_blocked(path: PathLike) -> tuple[bool, str]:
    """Return ``(blocked, reason)``. Never raises."""
    try:
        resolved = normalize_path(path)
    except Exception as exc:  # noqa: BLE001 — refuse unresolvable paths
        return True, f"Cannot resolve path: {exc}"

    # Block the filesystem root only when the path *is* ``/``.
    # ``/`` must not be in the parent-check list (that would block HOME).
    if resolved == Path("/"):
        return True, "Blocked filesystem root: /"

    for root in BLOCKED_PATHS:
        try:
            blocked_root = root.expanduser().resolve()
        except Exception:  # noqa: BLE001
            blocked_root = root
        if resolved == blocked_root or blocked_root in resolved.parents:
            return True, f"Blocked system path: {blocked_root}"

    for part in resolved.parts:
        if part.lower() in BLOCKED_SEGMENTS:
            return True, f"Blocked segment: {part}"

    return False, ""


def is_write_blocked(path: PathLike) -> tuple[bool, str]:
    """``is_path_blocked`` or host ``_guard_path`` refusal. No host → first check only."""
    blocked, reason = is_path_blocked(path)
    if blocked:
        return True, reason
    guard = getattr(_host, "_guard_path", None)
    if not callable(guard):
        return False, ""
    try:
        _ok, err = guard(str(path))
    except Exception:  # noqa: BLE001 — unit tests / missing host stay on is_path_blocked
        return False, ""
    if err:
        return True, str(err)
    return False, ""
