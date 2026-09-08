"""FileMaid heuristic analyzer — labels only. No LLM. No moves. No deletes.

Not registered as @tools. ``analyze_path`` returns a proposal dict or None.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from primus.core.extractors import detect_file_type

PathLike = Union[str, Path]

CLUTTER_EXTS = frozenset({
    ".tmp", ".temp", ".bak", ".old", ".log", ".cache",
    ".ds_store", ".crdownload", ".part", ".download",
})
INSTALLER_EXTS = frozenset({
    ".exe", ".msi", ".dmg", ".pkg", ".deb", ".rpm",
})
LARGE_BYTES = 100 * 1024 * 1024
OLD_PLUS_SIZE = 10 * 1024 * 1024
AGE_SHORT = 14
AGE_DESKTOP = 30
AGE_MID = 90
AGE_LONG = 180

ACTIONS = frozenset({"quarantine", "archive", "move_to_folder", "keep"})


@dataclass
class ScannedFile:
    path: Path
    filename: str
    suffix: str
    size_bytes: int
    mtime: float
    file_type: str

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.mtime) / 86400.0)


@dataclass
class ActionProposal:
    action: str
    reason: str
    confidence: int
    path: str
    filename: str

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "path": self.path,
            "filename": self.filename,
        }


def _is_clutter(sf: ScannedFile) -> bool:
    if sf.suffix.lower() in CLUTTER_EXTS:
        return True
    return sf.filename.lower() in {".ds_store", "ds_store"}


def _is_installer(sf: ScannedFile) -> bool:
    return sf.suffix.lower() in INSTALLER_EXTS


def _parent_has(sf: ScannedFile, token: str) -> bool:
    needle = token.lower()
    return any(part.lower() == needle or needle in part.lower() for part in sf.path.parts[:-1])


def _name_junk(sf: ScannedFile) -> bool:
    name = sf.filename
    return name.startswith("~") or name.startswith("Copy of")


def _scan(path: PathLike) -> Optional[ScannedFile]:
    try:
        p = Path(path).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return None
    if not p.exists() or not p.is_file():
        return None
    try:
        st = p.stat()
    except Exception:  # noqa: BLE001
        return None
    return ScannedFile(
        path=p,
        filename=p.name,
        suffix=p.suffix,
        size_bytes=st.st_size,
        mtime=st.st_mtime,
        file_type=detect_file_type(p),
    )


def _propose(sf: ScannedFile, action: str, reason: str, confidence: int) -> ActionProposal:
    act = action if action in ACTIONS else "keep"
    return ActionProposal(
        action=act,
        reason=reason,
        confidence=confidence,
        path=str(sf.path),
        filename=sf.filename,
    )


def _rules_free_space(sf: ScannedFile) -> Optional[ActionProposal]:
    if _is_installer(sf):
        return _propose(sf, "quarantine", "Installer package", 82)
    if _is_clutter(sf):
        return _propose(sf, "quarantine", "Clutter / leftover file", 88)
    if sf.size_bytes >= LARGE_BYTES:
        return _propose(sf, "quarantine", "File is ≥100MB", 75)
    if sf.age_days > AGE_MID and _parent_has(sf, "download"):
        return _propose(sf, "archive", "Older than 90 days in a Downloads folder", 72)
    if sf.age_days > AGE_MID:
        return _propose(sf, "archive", "Older than 90 days", 65)
    return None


def _rules_archive_old(sf: ScannedFile) -> Optional[ActionProposal]:
    if sf.age_days >= AGE_LONG:
        return _propose(sf, "archive", "Older than 180 days", 80)
    if sf.age_days >= AGE_MID:
        return _propose(sf, "archive", "Older than 90 days", 62)
    return None


def _rules_deep_clean(sf: ScannedFile) -> Optional[ActionProposal]:
    if _is_clutter(sf):
        return _propose(sf, "quarantine", "Clutter / leftover file", 85)
    if _parent_has(sf, "desktop") and sf.age_days > AGE_DESKTOP:
        return _propose(sf, "move_to_folder", "Desktop file older than 30 days", 68)
    if _name_junk(sf):
        return _propose(sf, "quarantine", "Temp or duplicate name (~ / Copy of)", 78)
    return None


def _rules_organize_project(sf: ScannedFile) -> Optional[ActionProposal]:
    if sf.file_type in {"pdf", "office", "text"} and sf.age_days < AGE_SHORT:
        return _propose(sf, "move_to_folder", "Recent project document", 60)
    return None


def _rules_custom(sf: ScannedFile) -> Optional[ActionProposal]:
    if _is_clutter(sf):
        return _propose(sf, "quarantine", "Clutter / leftover file", 75)
    if sf.age_days > AGE_MID and sf.size_bytes > OLD_PLUS_SIZE:
        return _propose(sf, "archive", "Older than 90 days and larger than 10MB", 58)
    return None


_GOAL_RULES = {
    "free_space": _rules_free_space,
    "archive_old": _rules_archive_old,
    "deep_clean": _rules_deep_clean,
    "organize_project": _rules_organize_project,
    "custom": _rules_custom,
}


def analyze_path(path: PathLike, goal_type: str = "custom") -> dict | None:
    """Heuristic label only. ``{"action","reason","confidence","path","filename"}`` or None."""
    sf = _scan(path)
    if sf is None:
        return None
    rules = _GOAL_RULES.get(goal_type, _rules_custom)
    proposal = rules(sf)
    return proposal.to_dict() if proposal else None
