"""Leftover-utterance receptionist — fast-model JSON slots, never shown in chat.

Runs only after the green gates miss and ``route_prompt`` falls through to
``path=primus`` / the plan→execute graph. One bounded ``primus_fast_model`` call.
Missing model / timeout / bad JSON → ``None`` (caller keeps the current graph).
``need_brain`` may call ``ask_grok`` (governor + redact). Fast winners never do.

Library modules reach the host via ``_host`` — do not import admin_assistant.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import sys as _sys

_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

BUCKETS = frozenset({"fact", "work", "pulse", "heavy", "unknown"})
_INVENTED_MAIL = frozenset({
    "gmail_get_unread_email", "gmail_get_latest", "get_unread_email",
})

# Short dispatcher prompt — do not rewrite PRIMUS_SYSTEM.
DISPATCHER_SYSTEM = """You fill JSON slots for leftover operator utterances. Reply with ONLY this object — no markdown, no prose, no explanation:

{"bucket":"fact|work|pulse|heavy|unknown","tools":["existing_tool_name"],"args":{},"need_brain":false,"path_mode":"keep"}

bucket: fact=lookup, work=files/mail/shell, pulse=greeting, heavy=long/hard, unknown=unsure.
tools: zero or more names from the allow-list below. Drop anything not listed. Never invent gmail_get_unread_email, gmail_get_latest, or get_unread_email.
args: flat object for the first tool if obvious; else {}.
need_brain: true only if a stronger model is required after local tools. When true, leftover/heavy may call ask_grok.
path_mode: always "keep". Do not plan, reorder, or invent steps.
If the operator is on user_path: slot-normalize only.

Allow-list:
{allow_list}
"""

_STASH: dict[str, Any] = {}
_TOOL_NAME_CACHE: Optional[set[str]] = None


@dataclass
class ReceptionistSlots:
    bucket: str = "unknown"
    tools: list[str] = field(default_factory=list)
    args: dict[str, Any] = field(default_factory=dict)
    need_brain: bool = False
    path_mode: str = "keep"
    dropped: list[str] = field(default_factory=list)


def clear_stash() -> None:
    _STASH.clear()


def stashed_slots() -> Optional[ReceptionistSlots]:
    slots = _STASH.get("slots")
    return slots if isinstance(slots, ReceptionistSlots) else None


def need_brain_stashed() -> bool:
    """True when the last leftover classification asked for a stronger model.

    Leftover / heavy may call ``ask_grok``. Fast winners never reach this stash.
    """
    return bool(_STASH.get("need_brain"))


def stash_slots(slots: ReceptionistSlots) -> None:
    _STASH["slots"] = slots
    _STASH["need_brain"] = bool(slots.need_brain)


def known_tool_names() -> set[str]:
    """Real ``build_tools()`` names. Unknown names are dropped, never invented."""
    global _TOOL_NAME_CACHE
    if _TOOL_NAME_CACHE is not None:
        return _TOOL_NAME_CACHE
    names: set[str] = set()
    try:
        tools = getattr(_host, "build_tools", None)
        raw = tools() if callable(tools) else None
        if raw is None:
            from primus.tools.registry import build_tools  # noqa: PLC0415

            raw = build_tools()
        names = {getattr(t, "name", "") for t in (raw or [])} - {""}
    except Exception:  # noqa: BLE001
        names = set()
    _TOOL_NAME_CACHE = names
    return names


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _first_json_object(text: str) -> Optional[str]:
    """Pull the first top-level JSON object out of model text (fences / prose)."""
    blob = (text or "").strip()
    if not blob:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, re.S)
    if fence:
        blob = fence.group(1)
    start = blob.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(blob[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return blob[start : i + 1]
    return None


def parse_slots(raw: Any, allowed: Optional[set[str]] = None) -> Optional[ReceptionistSlots]:
    """Validate receptionist JSON. Drop unknown tool names. Never invent mail tools."""
    if raw is None:
        return None
    if isinstance(raw, ReceptionistSlots):
        return raw
    text = raw if isinstance(raw, str) else str(raw)
    blob = _first_json_object(text)
    if not blob:
        return None
    try:
        obj = json.loads(blob)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None

    allow = allowed if allowed is not None else known_tool_names()
    bucket = str(obj.get("bucket") or "unknown").strip().lower()
    if bucket not in BUCKETS:
        bucket = "unknown"

    raw_tools = obj.get("tools") or []
    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]
    if not isinstance(raw_tools, list):
        raw_tools = []

    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for item in raw_tools:
        name = str(item or "").strip()
        if not name:
            continue
        if name in _INVENTED_MAIL or name not in allow:
            dropped.append(name)
            continue
        if name not in seen:
            seen.add(name)
            kept.append(name)

    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    path_mode = str(obj.get("path_mode") or "keep").strip().lower()
    if path_mode != "keep":
        path_mode = "keep"

    return ReceptionistSlots(
        bucket=bucket,
        tools=kept,
        args=dict(args),
        need_brain=_as_bool(obj.get("need_brain", False)),
        path_mode=path_mode,
        dropped=dropped,
    )


def _timeout_sec() -> int:
    try:
        bound = int(_host.CFG.get("fast_chat_invoke_timeout_sec", 15))
    except Exception:  # noqa: BLE001
        bound = 15
    if bound <= 0:
        bound = 12
    return max(8, min(15, bound))


def _classify(message: str) -> Optional[ReceptionistSlots]:
    """One bounded fast-model call. Redacts model-bound input. Never returns chat text."""
    from primus.agents.system import (  # noqa: PLC0415 — lazy; host already loaded
        _fast_model_available,
        coerce_message_text,
        make_chat_ollama,
    )
    from primus.core.redact import redact_model_input  # noqa: PLC0415

    if not _fast_model_available():
        return None
    model = _host.CFG.get("primus_fast_model") or ""
    if not model:
        return None
    allow = known_tool_names()
    if not allow:
        return None

    safe_msg = redact_model_input(message)
    allow_list = ", ".join(sorted(allow))
    system = redact_model_input(DISPATCHER_SYSTEM.replace("{allow_list}", allow_list))
    llm = make_chat_ollama(model, temperature=0.0, num_predict=256)
    SystemMessage = _host.SystemMessage
    HumanMessage = _host.HumanMessage
    msgs = [
        SystemMessage(content=system),
        HumanMessage(content=safe_msg),
    ]
    timeout = _timeout_sec()
    box: dict[str, Any] = {}
    err: list[Exception] = []

    def _run() -> None:
        try:
            box["resp"] = llm.invoke(msgs)
        except Exception as exc:  # noqa: BLE001 — surfaced below; never hang the turn
            err.append(exc)

    worker = threading.Thread(target=_run, name="primus-receptionist", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return None
    if err:
        return None
    resp = box.get("resp")
    if resp is None:
        return None
    text = coerce_message_text(getattr(resp, "content", resp))
    return parse_slots(text, allowed=allow)


def receive_leftover(message: str) -> Optional[ReceptionistSlots]:
    """Classify a leftover utterance. Return slots or None.

    Never prints JSON. Never answers the operator. The leftover hook in
    ``system.py`` maps a valid ``tools[]`` into ``_apply_turn_pack`` (around
    the ``receive_leftover`` call site). ``user_path``: slot-normalize only —
    never plan or reorder.
    """
    clear_stash()
    msg = (message or "").strip()
    if not msg:
        return None
    try:
        from primus.core.path_mode import current_path_mode  # noqa: PLC0415

        user_path = current_path_mode() == "user_path"
    except Exception:  # noqa: BLE001
        user_path = False

    try:
        slots = _classify(msg)
    except Exception:  # noqa: BLE001 — missing model / timeout / bad JSON → leftover graph
        return None
    if slots is None:
        return None
    slots.path_mode = "keep"
    if user_path:
        # Slot-normalize only. Never plan, reorder, or run a narrow agent this pass.
        slots.path_mode = "keep"
    stash_slots(slots)
    return slots
