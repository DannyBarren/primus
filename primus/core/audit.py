"""Shared ``{APP_DIR}/audit.jsonl`` writer. Do not fork a second log.

Schema: ``{ts, action, ok, detail, turn_id?}``. Detail never holds secrets or a
mail body. ``_fs_audit`` in the registry is a thin wrapper over ``write_audit``.

Library modules reach the host via ``_host`` — do not import admin_assistant.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime
from typing import Any, Optional

import sys as _sys

_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

_LOCK = threading.Lock()
_LOCAL = threading.local()
_DETAIL_LIMIT = 240
_RECENT_DEFAULT = 12

# UI slash handlers own these — never steal them for the why-ask path.
_SKIP_SLASH_RE = re.compile(
    r"^\s*/(?:good|bad|\N{THUMBS UP SIGN}|\N{THUMBS DOWN SIGN})(?:\s|$)"
    r"|^\s*/feedback\s+(?:good|bad)\b",
    re.I,
)

_WHY_RE = re.compile(
    r"(?is)^\s*(?:(?:can|could)\s+you\s+|please\s+)?"
    r"(?:"
    r"why\s+did\s+you\s+(?:just\s+)?do\s+that"
    r"|what\s+did\s+you\s+just\s+do"
    r"|what\s+did\s+you\s+do(?:\s+just\s+now)?"
    r"|why['’]d\s+you\s+do\s+that"
    r"|explain\s+what\s+you\s+(?:just\s+)?did"
    r"|explain\s+your\s+(?:reasoning|process)"
    r"(?:\s+and\s+(?:your\s+)?(?:reasoning|process))?"
    r"|walk\s+me\s+through"
    r"(?:\s+(?:that|it|this|what\s+you\s+(?:just\s+)?did|"
    r"your\s+(?:reasoning|process|steps)))?"
    r"|how\s+did\s+you\s+do\s+that"
    r")\s*[?.!]?\s*$"
)

_SECRET_RE = re.compile(
    r"(?:gmail_token|gmail_credentials|calendar_token)(?:\.json)?"
    r"|ya29\.[A-Za-z0-9._-]+"
    r"|xox[baprs]-[A-Za-z0-9-]+"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r"|sk-[A-Za-z0-9_-]{10,}"
    r"|xai-[A-Za-z0-9_-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|[Bb]earer\s+[A-Za-z0-9._\-+/=]{8,}",
    re.I,
)
_PASSWORD_RE = re.compile(r"(?i)\bpassword\s*=\s*\S+")
_MAIL_BODY_RE = re.compile(
    r"(?is)(?:^|\n)(?:From|Subject|To|Date):[^\n]+\n(?:(?:From|Subject|To|Date|Cc):[^\n]+\n)*"
)
_DROP_ARG_KEYS = frozenset({
    "content", "body", "briefing", "message", "text", "html", "payload",
    "password", "token", "secret", "key", "api_key", "credentials",
})

_ACTION_ENGLISH = {
    "tool_start": "started",
    "tool_end": "finished",
    "queued": "queued",
    "executed": "executed",
    "ask_grok_cap": "stopped ask_grok at a budget cap",
    "ask_grok_loop": "broke an ask_grok loop",
    "vault_resolve": "resolved a vault id",
    "export": "exported a backup zip",
    "import": "imported a backup zip",
    "write": "wrote",
    "append": "appended",
    "overwrite": "overwrote",
    "mkdir": "created a directory",
    "move": "moved",
    "copy": "copied",
}


def _audit_path():
    try:
        from primus.config import AUDIT_FILE  # noqa: PLC0415

        return AUDIT_FILE
    except Exception:
        return _host.APP_DIR / "audit.jsonl"


def begin_turn(turn_id: Optional[str] = None) -> str:
    """Stamp this thread's optional ``turn_id``. Never raises."""
    tid = (turn_id or uuid.uuid4().hex[:8]).strip() or uuid.uuid4().hex[:8]
    _LOCAL.turn_id = tid
    return tid


def current_turn_id() -> Optional[str]:
    tid = getattr(_LOCAL, "turn_id", None)
    return str(tid) if tid else None


def _safe_detail(detail: Any) -> str:
    text = " ".join(str(detail or "").split())
    text = _SECRET_RE.sub("[redacted]", text)
    text = _PASSWORD_RE.sub("password=[redacted]", text)
    if _MAIL_BODY_RE.search(text) and len(text) > 80:
        head = text.split("\n", 1)[0]
        text = (head[:80] + " [mail body omitted]").strip()
    low = text.lower()
    if any(k in low for k in ("\nfrom:", "subject:", "message-id:")) and len(text) > 80:
        text = text[:80] + " [mail body omitted]"
    return text[:_DETAIL_LIMIT]


def tool_detail(name: str, args: Any = None) -> str:
    """Tool name + safe args. Drops bodies, secrets, and mail text."""
    label = (name or "tool").strip() or "tool"
    if not isinstance(args, dict) or not args:
        return label
    mailish = label.startswith("gmail_") or "mail" in label
    parts: list[str] = [label]
    for key, raw in args.items():
        k = str(key)
        if k.lower() in _DROP_ARG_KEYS:
            continue
        if mailish and k.lower() not in {"query", "message_id", "max_results", "to", "subject"}:
            continue
        val = " ".join(str(raw or "").split())
        if mailish and k.lower() in {"to", "subject"}:
            val = val[:60]
        parts.append(f"{k}={val[:80]}")
    return _safe_detail(" ".join(parts))


def write_audit(
    action: str,
    *,
    ok: bool = True,
    detail: str = "",
    turn_id: Optional[str] = None,
) -> None:
    """Append one line to ``audit.jsonl``. Never raises. Never forks the file."""
    try:
        tid = turn_id if turn_id is not None else current_turn_id()
        rec: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "action": str(action or "")[:80],
            "ok": bool(ok),
            "detail": _safe_detail(detail),
        }
        if tid:
            rec["turn_id"] = str(tid)[:32]
        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — auditing must never break the caller
        pass


def read_recent(limit: int = _RECENT_DEFAULT) -> list[dict[str, Any]]:
    """Last ``limit`` parseable lines. Empty list on any failure."""
    try:
        path = _audit_path()
        if not path.is_file():
            return []
        n = max(1, int(limit))
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()[-n:]
        rows: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("action"):
                rows.append(rec)
        return rows
    except Exception:  # noqa: BLE001
        return []


def _english(rec: dict[str, Any]) -> str:
    action = str(rec.get("action") or "").strip()
    detail = _safe_detail(rec.get("detail") or "")
    ok = rec.get("ok", True)
    if action == "queued":
        return f"Queued {detail or 'a command'} for approval — not executed."
    if action == "executed":
        return f"Executed {detail or 'a command'}."
    if action == "tool_start":
        return f"Started {detail or 'a tool'}."
    if action == "tool_end":
        verb = "Finished" if ok else "Failed"
        return f"{verb} {detail or 'a tool'}."
    if action == "ask_grok_cap":
        return f"Stopped ask_grok at a budget cap{f' ({detail})' if detail else ''}."
    if action == "ask_grok_loop":
        return f"Broke an ask_grok loop{f' ({detail})' if detail else ''}."
    if action == "vault_resolve":
        return f"Resolved vault id {detail or '(id)'}."
    if action == "export":
        return f"Exported a backup zip{f' ({detail})' if detail else ''}."
    if action == "import":
        return f"Imported a backup zip{f' ({detail})' if detail else ''}."
    label = _ACTION_ENGLISH.get(action, action.replace("_", " "))
    if detail:
        return f"{label} {detail}.".replace("..", ".")
    return f"{label}.".capitalize() if label else ""


def is_why_ask(message: str) -> bool:
    """True for 'why did you do that' / 'explain your reasoning' / 'how did you do that'.

    Leaves /good /bad alone. On a hit the caller must return ``explain_recent()`` —
    no graph, no tools, no re-run.
    """
    text = (message or "").strip()
    if not text or _SKIP_SLASH_RE.match(text):
        return False
    return bool(_WHY_RE.match(text))


def explain_recent(*, limit: int = _RECENT_DEFAULT) -> str:
    """English from the last audit lines. Never dumps JSON."""
    rows = read_recent(limit)
    if not rows:
        return "I haven't recorded any actions yet."
    lines = [s for s in (_english(r) for r in rows) if s]
    if not lines:
        return "I took a few steps, but there's nothing useful to quote from the log."
    return "Here's what I just did:\n\n" + "\n".join(f"• {s}" for s in lines)


def attach_audit(tool: Any) -> Any:
    """Log tool start/end around an existing StructuredTool ``.func``. Class tools log via run_shell."""
    if tool is None or getattr(tool, "_primus_audit", False):
        return tool
    inner = getattr(tool, "func", None)
    if not callable(inner):
        return tool
    name = getattr(tool, "name", None) or getattr(inner, "__name__", "tool")

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        write_audit("tool_start", ok=True, detail=tool_detail(str(name), kwargs))
        try:
            out = inner(*args, **kwargs)
            text = str(out) if out is not None else ""
            ok = not text.startswith(("✗", "Refusing", "Error"))
            write_audit("tool_end", ok=ok, detail=str(name))
            return out
        except Exception:
            write_audit("tool_end", ok=False, detail=str(name))
            raise

    try:
        object.__setattr__(tool, "func", _wrapped)
        object.__setattr__(tool, "_primus_audit", True)
    except Exception:  # noqa: BLE001
        try:
            tool.func = _wrapped
            tool._primus_audit = True
        except Exception:  # noqa: BLE001
            return tool
    return tool
