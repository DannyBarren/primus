"""Local capability snapshot — tools, connections, and credentials.

Reads files and config on this machine only. No model call, no HTTP, and no
secret values (token files are presence checks; vault labels are not decrypted).

Library modules reach the host via ``_host`` — do not import admin_assistant.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import sys as _sys

_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

# Whole-message matchers. Real mail/file actions ("check my inbox") do not fit.
_ACCESS_RE = re.compile(
    r"""(?ix)
    ^\s*
    (?:please\s+|hey\s+|primus[,:]?\s+)?
    (?:can\s+you\s+|could\s+you\s+|would\s+you\s+)?
    (?:tell\s+me\s+|show\s+me\s+|list\s+)?
    (?:
        what\s+access\s+do\s+you\s+have
        | what(?:'s|\s+is)\s+your\s+access
        | what\s+are\s+you\s+allowed\s+to\s+do
        | what\s+(?:still\s+)?requires\s+setup
        | what\s+(?:tools|tool\s+packs|credentials|tokens|services|accounts)
            (?:\s+do\s+you\s+have|\s+are\s+(?:available|connected|missing))?
        | which\s+(?:tools|tool\s+packs|services|accounts)
            (?:\s+are\s+(?:available|connected)|\s+do\s+you\s+have)
        | are\s+you\s+connected\s+to\s+(?:my\s+)?(?:e-?mail|gmail|inbox|calendar|slack)
        | (?:is|are)\s+(?:my\s+)?(?:e-?mail|gmail|inbox|calendar|slack)\s+connected
        | do\s+you\s+have\s+(?:access|a\s+token|credentials)
            (?:\s+to\s+(?:my\s+)?(?:e-?mail|gmail|inbox|calendar|slack))?
        | (?:access|capability)\s+(?:report|summary)
    )
    \s*\??\s*$
    """
)
_SECRETISH_RE = re.compile(r"(xox[baprs]-|sk-|ya29\.|AIza|-----BEGIN)", re.I)
_PACK_ORDER = ("minimal", "files", "shell", "knowledge", "web", "mail", "brain", "default")
_FIXTURE = Path(__file__).resolve().parents[2] / "examples" / "gmail" / "fixture_inbox.json"


def _app_dir() -> Path:
    try:
        raw = getattr(_host, "APP_DIR", None)
        if raw:
            return Path(raw)
    except Exception:  # noqa: BLE001
        pass
    from primus.config import APP_DIR  # noqa: PLC0415

    return APP_DIR


def _cfg() -> dict:
    try:
        cfg = getattr(_host, "CFG", None)
        if isinstance(cfg, dict):
            return cfg
    except Exception:  # noqa: BLE001
        pass
    return {}


def _exists(name: str) -> bool:
    try:
        return (_app_dir() / name).is_file()
    except OSError:
        return False


def _vault_labels() -> dict[str, str]:
    """id → label from the index. Never reads the encrypted blob."""
    path = _app_dir() / "vault_index.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return {}
    out: dict[str, str] = {}
    for item in entries:
        if isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = _safe_label(item.get("label") or item["id"])
    return out


def _safe_label(text: object) -> str:
    label = " ".join(str(text).split())[:80]
    if not label or _SECRETISH_RE.search(label):
        return "(label hidden)"
    return label


def _vault_state() -> str:
    try:
        from primus.core.vault import vault_locked  # noqa: PLC0415

        return "locked" if vault_locked() else "unlocked"
    except Exception:  # noqa: BLE001
        return "unavailable"


def _local_only() -> bool:
    try:
        from primus.core.brain_governor import local_only_enabled  # noqa: PLC0415

        return bool(local_only_enabled())
    except Exception:  # noqa: BLE001
        return bool(_cfg().get("local_only", True))


def _xai_key_state() -> str:
    """'set', 'vault label', or 'missing'. Never returns the key."""
    if os.environ.get("XAI_API_KEY", "").strip():
        return "set"
    if "grok" in _vault_labels():
        return "vault label"
    return "missing"


def _focus(message: str) -> str:
    low = (message or "").lower()
    if re.search(r"\b(e-?mail|gmail|inbox)\b", low):
        return "gmail"
    if re.search(r"\bcalendar\b", low):
        return "calendar"
    if re.search(r"\bslack\b", low):
        return "slack"
    if re.search(r"\b(tool\s+packs?|tools)\b", low):
        return "tools"
    if re.search(r"\b(credentials?|tokens?|vault|api\s+key)\b", low):
        return "creds"
    return "all"


def _gmail_lines() -> tuple[str, bool]:
    """Lead sentence, and whether a real token is present."""
    connected = _exists("gmail_token.json")
    if connected:
        return "Gmail is connected (gmail_token.json is present).", True
    line = "Gmail is not connected (no gmail_token.json)."
    fixture_on = bool(_cfg().get("gmail_fixture_mode", True)) and _FIXTURE.is_file()
    if fixture_on:
        line += " A demo inbox is on. That is not a live mailbox."
    return line, False


def _calendar_line() -> tuple[str, bool]:
    if _exists("calendar_token.json"):
        return "Calendar is connected (calendar_token.json is present).", True
    return "Calendar is not connected (no calendar_token.json).", False


def _slack_line() -> tuple[str, bool]:
    if _exists("slack_token.json"):
        return "Slack is connected (slack_token.json is present).", True
    bot = bool(os.environ.get("SLACK_BOT_TOKEN", "").strip())
    user = bool(os.environ.get("SLACK_USER_TOKEN", "").strip())
    if bot or user:
        which = "SLACK_BOT_TOKEN" if bot else "SLACK_USER_TOKEN"
        return f"Slack is connected ({which} is set).", True
    return "Slack is not connected (no slack_token.json and no Slack env token).", False


def _pack_lines() -> list[str]:
    from primus.core.tool_packs import PACKS  # noqa: PLC0415

    lines = []
    for name in _PACK_ORDER:
        tools = PACKS.get(name) or ()
        lines.append(f"  {name} — {', '.join(tools)}")
    return lines


def access_report(message: str = "") -> str:
    """Plain-text snapshot. Safe to show in chat."""
    gmail, gmail_ok = _gmail_lines()
    calendar, cal_ok = _calendar_line()
    slack, slack_ok = _slack_line()
    secret = _exists("gmail_credentials.json")
    local_only = _local_only()
    key = _xai_key_state()
    mode = str(_cfg().get("execution_mode", "execute") or "execute").strip().lower()
    if mode not in ("execute", "suggest"):
        mode = "execute"
    daily = str(_cfg().get("model") or "qwen2.5:7b")
    heavy = str(_cfg().get("forge_model") or "qwen2.5-coder:32b")
    labels = _vault_labels()
    label_txt = ", ".join(f"{i} ({labels[i]})" for i in sorted(labels)[:12]) or "none"
    focus = _focus(message)

    if key == "set":
        key_line = "xAI key — set (value hidden)."
    elif key == "vault label":
        key_line = "xAI key — vault label present (value hidden)."
    else:
        key_line = "xAI key — missing."
    if local_only:
        brain_line = "External brain — off (local-only). Ollama only."
    elif key == "missing":
        brain_line = "External brain — unavailable until an xAI key is set."
    else:
        brain_line = "External brain — available when a turn asks. Local-only is off."

    allowed = [
        "Local files, shell, knowledge search, and date/time.",
        f"Models — {daily} (daily), {heavy} (heavy).",
        "Weather, news, and web search — no stored token.",
    ]
    if mode == "suggest":
        allowed.append("Mode — suggest. Actions stay preview until you switch to execute.")
    else:
        allowed.append("Mode — execute.")
    if gmail_ok:
        allowed.append("Gmail — connected.")
    if cal_ok:
        allowed.append("Calendar — connected.")
    if slack_ok:
        allowed.append("Slack — connected.")

    needs: list[str] = []
    if not gmail_ok:
        needs.append("Gmail token (gmail_token.json).")
    if not secret:
        needs.append("OAuth client secret (gmail_credentials.json) for Gmail and Calendar sign-in.")
    if not cal_ok:
        needs.append("Calendar token (calendar_token.json).")
    if not slack_ok:
        needs.append("Slack token file or SLACK_BOT_TOKEN / SLACK_USER_TOKEN.")
    if local_only:
        needs.append("Turn local-only off before the external brain can run.")
    elif key == "missing":
        needs.append("xAI key (XAI_API_KEY or vault label grok).")

    lead = ""
    if focus == "gmail":
        lead = gmail
    elif focus == "calendar":
        lead = calendar
    elif focus == "slack":
        lead = slack

    parts: list[str] = ["Local access"]
    if lead:
        parts.append(lead)
    parts.append("")
    parts.append("Tool packs (one pack is bound per turn; the full registry stays registered):")
    parts.extend(_pack_lines())
    parts.append("")
    parts.append("Services:")
    parts.append(f"  {gmail}")
    parts.append(
        "  OAuth client secret — present."
        if secret
        else "  OAuth client secret — missing (gmail_credentials.json)."
    )
    parts.append(f"  {calendar}")
    parts.append(f"  {slack}")
    parts.append("")
    parts.append("Credentials:")
    parts.append(f"  {key_line}")
    parts.append(f"  {brain_line}")
    parts.append(f"  Vault — {_vault_state()}. Index labels: {label_txt}.")
    parts.append("")
    parts.append("Allowed without more setup:")
    parts.extend(f"  {line}" for line in allowed)
    parts.append("")
    parts.append("Needs setup:" if needs else "Needs setup: nothing in this checklist.")
    parts.extend(f"  {line}" for line in needs)
    return "\n".join(parts).rstrip() + "\n"


def access_answer(message: str) -> str | None:
    """Return the local report for an access question, else None."""
    msg = (message or "").strip()
    if not msg or len(msg) > 160 or msg.startswith("/"):
        return None
    if not _ACCESS_RE.match(msg):
        return None
    return access_report(msg)
