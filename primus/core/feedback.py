"""Append-only correction log. Classify locally. Never patch runtime. Never print JSON.

Library modules reach the host via ``_host`` — do not import admin_assistant.
A failed write must not fail the turn. /good and /bad stay on MetricsTracker.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from typing import Optional

_host = sys.modules.get("admin_assistant") or sys.modules["__main__"]

_CLASSES = ("routing", "extraction", "timeout", "safety", "other")

# UI slash handlers own these — never steal them for feedback.jsonl.
_SKIP_METRICS_RE = re.compile(
    r"^\s*/(?:good|bad|\N{THUMBS UP SIGN}|\N{THUMBS DOWN SIGN})(?:\s|$)"
    r"|^\s*/feedback\s+(?:good|bad)\b",
    re.I,
)

_EXPLICIT_RE = re.compile(
    r"(?:^|\s)(?:log\s+that\b|remember\s+this\s+miss\b|feedback\s*:|/feedback\b)",
    re.I,
)
_EXPLICIT_STRIP_RE = re.compile(
    r"^\s*(?:/feedback\b|feedback\s*:|log\s+that\s*:?|remember\s+this\s+miss\s*:?)\s*",
    re.I,
)

_PHRASE_RE = re.compile(
    r"\bthat\s+wasn['’]t\b"
    r"|\bi\s+meant\b"
    r"|\bfrom\s+\S+\s+not\s+"
    r"|\bwrong\s+sender\b"
    r"|\bsummarize\s+means\s+plaintext\b"
    r"|\bwrong\b",
    re.I,
)

_ROUTING_RE = re.compile(
    r"wrong\s+sender|from\s+\S+\s+not\s+|unrelated\s+mail|"
    r"i\s+meant\s+from|that\s+wasn['’]t\s+from",
    re.I,
)
_EXTRACTION_RE = re.compile(r"summarize\s+means\s+plaintext|\bplaintext\b|\bhtml\b", re.I)
_TIMEOUT_RE = re.compile(r"\btimeout\b|timed\s+out|too\s+slow|took\s+too\s+long", re.I)
_SAFETY_RE = re.compile(
    r"\b(?:shouldn['’]t\s+have|unsafe|safety|delete_file|queued)\b|\brm\s+-",
    re.I,
)

_SECRET_RE = re.compile(
    r"(?:gmail_token|gmail_credentials|calendar_token)(?:\.json)?"
    r"|ya29\.[A-Za-z0-9._-]+"
    r"|xox[baprs]-[A-Za-z0-9-]+"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r"|sk-[A-Za-z0-9]{20,}",
    re.I,
)

_PREV: dict[str, str] = {"path": "", "answer": ""}
_USER_LIMIT = 240
_HAPPENED_LIMIT = 160


def _feedback_path():
    try:
        from primus.config import FEEDBACK_FILE  # noqa: PLC0415

        return FEEDBACK_FILE
    except Exception:
        return _host.APP_DIR / "feedback.jsonl"


def _redact(text: str, limit: int) -> str:
    t = " ".join((text or "").split())
    t = _SECRET_RE.sub("[redacted]", t)
    if re.search(r"(?i)\b(?:from|subject|to):\s*\S+", t) and len(t) > 80:
        t = t[:80]
    return t[:limit]


def classify(text: str) -> str:
    """Local regex only — never call Ollama."""
    raw = text or ""
    if _ROUTING_RE.search(raw):
        return "routing"
    if _EXTRACTION_RE.search(raw):
        return "extraction"
    if _TIMEOUT_RE.search(raw):
        return "timeout"
    if _SAFETY_RE.search(raw):
        kind = "safety"
    else:
        kind = "other"
    return kind if kind in _CLASSES else "other"


def detect(user_text: str) -> Optional[tuple[str, str]]:
    """Return (source, correction) or None. Leaves /good /bad to metrics.json."""
    text = (user_text or "").strip()
    if not text or _SKIP_METRICS_RE.match(text):
        return None
    if _EXPLICIT_RE.search(text):
        return ("explicit", _EXPLICIT_STRIP_RE.sub("", text, count=1).strip() or text)
    if _PHRASE_RE.search(text):
        return ("phrase", text)
    return None


def _append(record: dict) -> None:
    try:
        path = _feedback_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — log write must never fail the turn
        pass


def capture_after_turn(user_text: str, answer: str = "") -> None:
    """Best-effort append if this turn is a correction. Never raises. Returns None."""
    try:
        hit = detect(user_text)
        if hit:
            source, correction = hit
            prev_path = _PREV.get("path") or ""
            prev_answer = _PREV.get("answer") or ""
            happened = f"{prev_path}: {prev_answer}".strip(": ").strip()
            _append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "class": classify(user_text),
                "user_text": _redact(user_text, _USER_LIMIT),
                "what_happened": _redact(happened, _HAPPENED_LIMIT),
                "correction": _redact(correction, _USER_LIMIT),
                "source": source,
            })
    except Exception:  # noqa: BLE001
        pass
    try:
        _PREV["path"] = str(getattr(_host.PrimusSession, "last_path", "") or "")
        _PREV["answer"] = _redact(answer or "", _HAPPENED_LIMIT)
    except Exception:  # noqa: BLE001
        pass
