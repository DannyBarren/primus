"""Frontier-brain budget and loop governor.

``ask_grok`` (the only new tool this pass) must pass both weekly $ and per-turn
call caps, then the loop detector, before any HTTP. Fast winners never call here.
Library modules reach the host via ``_host`` — do not import admin_assistant.
"""
from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import sys as _sys

_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

from primus.config import BRAIN_BUDGET_FILE  # noqa: E402

BUDGET_FILE = BRAIN_BUDGET_FILE
GROK_CHAT_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-4"
HTTP_TIMEOUT_SEC = 45
NOT_CONFIGURED = "Brain is not configured (missing XAI_API_KEY / vault:grok)."
LOOP_STUCK = "I kept circling on this — here's where I got stuck."
REFUSE_PRIMUS_PATCH = "ask_grok will not patch primus/ source."

DEFAULTS: dict[str, Any] = {
    "weekly_usd": 10.0,
    "per_turn_calls": 3,
    "override_grant_max": 3,
    "usd_per_call_estimate": 0.02,
    "week_start": "",
    "spent_usd": 0.0,
    "calls_this_week": 0,
}

_OVERRIDE_RE = re.compile(
    r"(?i)\b(?:yes,?\s+(\d+)\s+more|try\s+(\d+)\s+more)\b"
)
_PATCH_PRIMUS_RE = re.compile(
    r"(?is)(?:\b(?:patch|write|overwrite|modify|edit|update|replace|apply)\b.{0,100}\bprimus/)"
    r"|(?:\bprimus/.{0,80}\b(?:patch|write|overwrite|modify|edit|update|replace)\b)"
    r"|apply_pending_self_edit"
)
_LOCK = threading.Lock()


def _iso_monday(now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    monday = now.date() - timedelta(days=now.weekday())
    return monday.isoformat()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _near_identical(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= 0.92


def parse_override(message: str) -> Optional[int]:
    """``yes, 2 more`` / ``try 3 more`` → n, else None."""
    m = _OVERRIDE_RE.search(message or "")
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def refuses_primus_patch(question: str, briefing: str = "") -> bool:
    return bool(_PATCH_PRIMUS_RE.search(f"{question or ''}\n{briefing or ''}"))


@dataclass
class _Turn:
    calls: int = 0
    limit: int = 3
    proposals: list[str] = field(default_factory=list)
    api_texts: list[str] = field(default_factory=list)
    rejected: set[str] = field(default_factory=set)
    last_draft: str = ""
    override_applied: bool = False
    message: str = ""


_TURN = _Turn()


def current_turn() -> _Turn:
    return _TURN


def budget_path() -> Path:
    return BUDGET_FILE


def _coerce_budget(raw: Any) -> dict[str, Any]:
    data = dict(DEFAULTS)
    if isinstance(raw, dict):
        for key in DEFAULTS:
            if key in raw and raw[key] is not None:
                data[key] = raw[key]
    try:
        data["weekly_usd"] = float(data["weekly_usd"])
        data["per_turn_calls"] = int(data["per_turn_calls"])
        data["override_grant_max"] = int(data["override_grant_max"])
        data["usd_per_call_estimate"] = float(data["usd_per_call_estimate"])
        data["spent_usd"] = float(data["spent_usd"])
        data["calls_this_week"] = int(data["calls_this_week"])
    except (TypeError, ValueError):
        data = dict(DEFAULTS)
    data["week_start"] = str(data.get("week_start") or "")
    return data


def _roll_week(data: dict[str, Any], *, now: Optional[datetime] = None) -> dict[str, Any]:
    monday = _iso_monday(now)
    stored = str(data.get("week_start") or "")
    if stored != monday:
        data["week_start"] = monday
        data["spent_usd"] = 0.0
        data["calls_this_week"] = 0
    return data


def load_budget(*, now: Optional[datetime] = None) -> dict[str, Any]:
    """Load ``brain_budget.json``, creating defaults if missing. Rolls on local Monday."""
    with _LOCK:
        if not BUDGET_FILE.is_file():
            data = _roll_week(dict(DEFAULTS), now=now)
            _write_budget_unlocked(data)
            return dict(data)
        try:
            raw = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        data = _roll_week(_coerce_budget(raw), now=now)
        if str(raw.get("week_start") or "") != data["week_start"] or not BUDGET_FILE.is_file():
            _write_budget_unlocked(data)
        return dict(data)


def _write_budget_unlocked(data: dict[str, Any]) -> None:
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = _coerce_budget(data)
    BUDGET_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(BUDGET_FILE, 0o600)
    except OSError:
        pass


def save_budget(data: dict[str, Any]) -> None:
    with _LOCK:
        _write_budget_unlocked(data)


def begin_turn(message: str = "") -> None:
    """Reset per-turn counters. Override in this message replaces the turn cap."""
    budget = load_budget()
    _TURN.calls = 0
    _TURN.proposals = []
    _TURN.api_texts = []
    _TURN.rejected = set()
    _TURN.last_draft = ""
    _TURN.message = message or ""
    n = parse_override(_TURN.message)
    if n is not None:
        _TURN.limit = min(n, max(0, int(budget.get("override_grant_max", 3))))
        _TURN.override_applied = True
    else:
        _TURN.limit = max(0, int(budget.get("per_turn_calls", 3)))
        _TURN.override_applied = False


def grant_override(n: int) -> int:
    """This-turn extra calls: ``min(n, override_grant_max)``. Still counts against weekly $.

    Clears the proposal / consecutive-text ledger so a grant is a fresh try, not a loop.
    """
    budget = load_budget()
    granted = min(max(0, int(n)), max(0, int(budget.get("override_grant_max", 3))))
    _TURN.limit += granted
    _TURN.override_applied = True
    _TURN.proposals = []
    _TURN.api_texts = []
    return granted


def apply_override_from_text(text: str) -> Optional[int]:
    n = parse_override(text or "")
    if n is None:
        return None
    return grant_override(n)


def note_proposal(name: str) -> None:
    """Record a proposed tool/action name for the loop detector."""
    label = (name or "").strip()
    if label:
        _TURN.proposals.append(label)


def note_api_text(text: str) -> None:
    blob = (text or "").strip()
    if blob:
        _TURN.api_texts.append(blob)
        _TURN.last_draft = blob


def note_risky_rejected(action: str) -> None:
    label = _norm(action)
    if label:
        _TURN.rejected.add(label)


def note_risky_proposed(action: str) -> bool:
    """True when this risky action was already rejected this turn (re-propose)."""
    label = _norm(action)
    return bool(label) and label in _TURN.rejected


def last_useful_draft() -> str:
    return (_TURN.last_draft or "").strip()


def stuck_message() -> str:
    draft = last_useful_draft()
    if draft:
        return f"{LOOP_STUCK}\n\n{draft}"
    return LOOP_STUCK


def loop_block_reason(action: str = "") -> Optional[str]:
    """Loop detector — no extra API call. Any trip → stuck message (caller formats)."""
    counts = Counter(_TURN.proposals)
    if any(c >= 3 for c in counts.values()):
        return LOOP_STUCK
    texts = _TURN.api_texts
    if len(texts) >= 2 and _near_identical(texts[-1], texts[-2]):
        return LOOP_STUCK
    if action and note_risky_proposed(action):
        return LOOP_STUCK
    return None


def cap_block_reason(budget: Optional[dict[str, Any]] = None) -> Optional[str]:
    data = budget if budget is not None else load_budget()
    estimate = float(data.get("usd_per_call_estimate", 0.02))
    weekly = float(data.get("weekly_usd", 10.0))
    spent = float(data.get("spent_usd", 0.0))
    if spent + estimate > weekly + 1e-9:
        return f"Brain weekly budget reached (${weekly:.2f})."
    if _TURN.calls >= _TURN.limit:
        return f"Brain per-turn call limit reached ({_TURN.limit})."
    return None


def allow_http(action: str = "") -> tuple[bool, str]:
    """Both caps and the loop detector must pass. Either trip → no HTTP."""
    cap = cap_block_reason()
    if cap:
        try:
            from primus.core.audit import write_audit  # noqa: PLC0415

            write_audit("ask_grok_cap", ok=False, detail=cap)
        except Exception:  # noqa: BLE001
            pass
        return False, cap
    loop = loop_block_reason(action)
    if loop:
        try:
            from primus.core.audit import write_audit  # noqa: PLC0415

            write_audit("ask_grok_loop", ok=False, detail="loop-break")
        except Exception:  # noqa: BLE001
            pass
        return False, stuck_message()
    return True, ""


def record_success(text: str = "", *, estimate: Optional[float] = None) -> None:
    """Persist weekly spend after a successful HTTP call."""
    data = load_budget()
    cost = float(estimate if estimate is not None else data.get("usd_per_call_estimate", 0.02))
    data["spent_usd"] = float(data.get("spent_usd", 0.0)) + cost
    data["calls_this_week"] = int(data.get("calls_this_week", 0)) + 1
    save_budget(data)
    _TURN.calls += 1
    if text:
        note_api_text(text)
    elif not _TURN.last_draft:
        _TURN.last_draft = ""


def grok_api_key() -> str:
    """Env ``XAI_API_KEY``, else ``resolve("vault:grok")`` after the tool is chosen."""
    key = os.environ.get("XAI_API_KEY", "").strip()
    if key:
        return key
    try:
        from primus.core.vault import resolve  # noqa: PLC0415

        val = resolve("vault:grok", dest="api")
    except Exception:  # noqa: BLE001
        val = None
    return (val or "").strip()


def _redact_payload(question: str, briefing: str, key: str = "") -> tuple[str, str]:
    from primus.core.redact import redact_for_outbound  # noqa: PLC0415

    safe_q = redact_for_outbound(question or "")
    safe_b = redact_for_outbound(briefing or "")
    if key:
        if key in safe_q:
            safe_q = safe_q.replace(key, "[redacted]")
        if key in safe_b:
            safe_b = safe_b.replace(key, "[redacted]")
    return safe_q, safe_b


def _post_chat(question: str, briefing: str, key: str, *, timeout: float = HTTP_TIMEOUT_SEC) -> str:
    messages: list[dict[str, str]] = []
    if briefing:
        messages.append({"role": "system", "content": briefing})
    messages.append({"role": "user", "content": question})
    body = json.dumps({
        "model": GROK_MODEL,
        "messages": messages,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        GROK_CHAT_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Primus/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return f"Brain request failed ({exc.code})."
    except TimeoutError:
        return "Brain request timed out."
    except urllib.error.URLError:
        return "Brain request failed (network)."
    except Exception:  # noqa: BLE001 — never crash the leftover turn
        return "Brain request failed."
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "Brain request failed (bad response)."
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return "Brain request failed (empty response)."
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        content = "".join(parts)
    text = str(content or "").strip()
    return text or "Brain request failed (empty response)."


def leftover_may_ask_grok(question: str, briefing: str = "") -> Optional[str]:
    """Leftover / need_brain / heavy may call the frontier. Fast winners never call this."""
    try:
        from primus.core.receptionist import need_brain_stashed, stashed_slots  # noqa: PLC0415

        slots = stashed_slots()
        if not need_brain_stashed() and not (slots and slots.bucket == "heavy"):
            return None
    except Exception:  # noqa: BLE001
        return None
    return ask_frontier(question, briefing)


def ask_frontier(question: str, briefing: str = "") -> str:
    """Governor + redact + HTTP. Used by ``ask_grok`` and leftover/need_brain/heavy."""
    q = question or ""
    brief = briefing or ""
    if refuses_primus_patch(q, brief):
        return REFUSE_PRIMUS_PATCH

    n = parse_override(q)
    if n is not None and not _TURN.override_applied:
        grant_override(n)

    fingerprint = _norm(q) or "ask_grok"
    ok, reason = allow_http(q)
    if not ok:
        return reason
    if _TURN.proposals.count(fingerprint) >= 3:
        return stuck_message()

    key = grok_api_key()
    if not key:
        return NOT_CONFIGURED

    note_proposal(fingerprint)
    safe_q, safe_b = _redact_payload(q, brief, key)
    text = _post_chat(safe_q, safe_b, key)
    failed = text.startswith("Brain request failed") or text.startswith("Brain request timed")
    if failed:
        return text
    record_success(text)
    return text
