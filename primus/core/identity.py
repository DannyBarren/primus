"""Operator identity — who Primus serves. Resolved ONCE at import time.

The repo ships a generic identity ("the operator") so a fresh clone never leaks a real
person's profile. A deployment personalizes Primus WITHOUT editing the repo by dropping
JSON at ``$PRIMUS_CONFIG_DIR/operator.json`` (default ``~/.config/primus/operator.json``)::

    {
      "name": "Ada",
      "full_name": "Ada Lovelace",
      "projects": ["Analytical Engine"],
      "creator_lock": false,
      "context_block": "Ada writes notes on the Analytical Engine.",
      "project_example": "the Analytical Engine",
      "signoff": "Ada"
    }

Every field is optional; anything omitted falls back to the generic default. Derived
forms (possessive, vocative, sentence-initial capitalization) are computed from ``name``
unless explicitly overridden. ``PRIMUS_OPERATOR_NAME`` / ``PRIMUS_OPERATOR_FULL_NAME``
env vars win over the file for the two name fields.

This module is stdlib-only on purpose: it is imported by prompts/personality at package
import time, before the host config exists, so it must not import ``primus.config``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _config_dir() -> Path:
    """Mirror primus.config's CONFIG_DIR resolution (duplicated to avoid an import cycle)."""
    raw = os.environ.get("PRIMUS_CONFIG_DIR", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".config" / "primus"


CONFIG_DIR = _config_dir()
OPERATOR_FILE = CONFIG_DIR / "operator.json"
PERSONALITY_OVERRIDE_FILE = CONFIG_DIR / "personality_override.md"
SEED_KNOWLEDGE_FILE = CONFIG_DIR / "seed_knowledge.json"
BUSINESS_CONTEXT_FILE = CONFIG_DIR / "business_context.json"

_GENERIC_CONTEXT_BLOCK = "The operator runs a Linux-first, automation-heavy workflow."

_DEFAULTS: dict[str, Any] = {
    "name": "the operator",
    "full_name": "",            # default: same as name
    "name_cap": "",             # default: "The operator" / the custom name
    "label": "",                # default: "Operator" / the custom name (transcript labels)
    "possessive": "",           # default: "the operator's" / "<name>'s"
    "vocative": "",             # default: "" / ", <name>"
    "signoff": "",              # default: "" / the custom name (email draft signature)
    "projects": [],             # known project names (auto-tagging, KB routing)
    "projects_paren": "",       # default: " (A, B, C)" derived from projects
    "projects_dev_paren": "",   # default: same as projects_paren
    "context_block": _GENERIC_CONTEXT_BLOCK,
    "project_example": "the portal rebuild",
    "project_example2": "the inventory tool",
    "creator_lock": False,      # enforce canonical operator-name spelling in memory writes
    "tag_keywords_personal": [],   # extra auto-tag keywords (memory classification)
    "tag_keywords_business": [],
}


def _load() -> dict[str, Any]:
    ident = dict(_DEFAULTS)
    custom = False
    try:
        if OPERATOR_FILE.exists():
            data = json.loads(OPERATOR_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in ident:
                        ident[key] = value
                custom = bool(str(data.get("name", "")).strip())
    except (json.JSONDecodeError, OSError):
        pass
    env_name = os.environ.get("PRIMUS_OPERATOR_NAME", "").strip()
    if env_name:
        ident["name"] = env_name
        custom = True
    env_full = os.environ.get("PRIMUS_OPERATOR_FULL_NAME", "").strip()
    if env_full:
        ident["full_name"] = env_full
        custom = True

    ident["custom"] = custom
    name = str(ident["name"]).strip() or _DEFAULTS["name"]
    ident["name"] = name
    if not ident["full_name"]:
        ident["full_name"] = name
    if not ident["name_cap"]:
        ident["name_cap"] = name if custom else "The operator"
    if not ident["label"]:
        ident["label"] = name if custom else "Operator"
    if not ident["possessive"]:
        ident["possessive"] = f"{name}'s" if custom else "the operator's"
    if not ident["vocative"]:
        ident["vocative"] = f", {name}" if custom else ""
    if not ident["signoff"] and custom:
        ident["signoff"] = name
    projects = ident.get("projects")
    if not isinstance(projects, list):
        projects = [str(projects)] if projects else []
    ident["projects"] = [str(p) for p in projects if str(p).strip()]
    if not ident["projects_paren"] and ident["projects"]:
        ident["projects_paren"] = f" ({', '.join(ident['projects'])})"
    if not ident["projects_dev_paren"]:
        ident["projects_dev_paren"] = ident["projects_paren"]
    for key in ("tag_keywords_personal", "tag_keywords_business"):
        val = ident.get(key)
        if not isinstance(val, list):
            val = [str(val)] if val else []
        ident[key] = [str(w) for w in val if str(w).strip()]
    return ident


OPERATOR: dict[str, Any] = _load()

# Flat token map for prompt rendering. Prompts are written with {op}-style tokens and
# rendered with plain str.replace (NOT .format) so literal JSON braces in few-shot
# examples can never crash rendering.
_TOKENS: dict[str, str] = {
    "op": OPERATOR["name"],
    "op_cap": OPERATOR["name_cap"],
    "op_label": OPERATOR["label"],
    "op_full": OPERATOR["full_name"],
    "op_pos": OPERATOR["possessive"],
    "op_voc": OPERATOR["vocative"],
    "op_signoff": OPERATOR["signoff"],
    "op_context": OPERATOR["context_block"],
    "op_projects_paren": OPERATOR["projects_paren"],
    "op_projects_dev_paren": OPERATOR["projects_dev_paren"],
    "op_project_example": OPERATOR["project_example"],
    "op_project_example2": OPERATOR["project_example2"],
}


def render(text: str) -> str:
    """Substitute operator tokens in ``text``. Safe on stray braces (plain replace)."""
    if not text:
        return text
    for token, value in _TOKENS.items():
        needle = "{" + token + "}"
        if needle in text:
            text = text.replace(needle, str(value))
    return text


def email_signoff() -> str:
    """Signature block for drafted emails: 'Best,\\n<name>' when personalized, else 'Best,'."""
    return f"Best,\n{OPERATOR['signoff']}" if OPERATOR["signoff"] else "Best,"


def load_personality_override() -> str | None:
    """Full replacement for CORE_OPERATOR_PROFILE from the live config dir, if present."""
    try:
        if PERSONALITY_OVERRIDE_FILE.exists():
            text = PERSONALITY_OVERRIDE_FILE.read_text(encoding="utf-8").strip()
            if text:
                return text
    except OSError:
        pass
    return None


def load_json_override(path: Path) -> Any | None:
    """Parsed JSON from a config-dir override file, or None when absent/invalid."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return None
