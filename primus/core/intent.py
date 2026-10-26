"""Local intent step — one short 7B call before fast paths and the main agent.

Ollama only (``CFG['model']``, the daily 7B). The reply is JSON, never shown in chat.
A timeout or bad JSON leaves routing on the existing rules.

Library modules reach the host via ``_host`` — do not import admin_assistant.
"""
from __future__ import annotations

import contextvars
import json
import re
import threading
from typing import NamedTuple, Optional

import sys as _sys

_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

INTENTS = frozenset({
    "chat", "file_op", "gmail", "knowledge", "code", "multi_step", "status", "other",
})

# Shown to the 7B. Keep it short — the call is capped at a few dozen tokens.
INTENT_SYSTEM = """You classify one operator message for a local assistant. Reply with ONLY this JSON object — no markdown, no prose:

{"intent":"chat|file_op|gmail|knowledge|code|multi_step|status|other","entities":["name or path"],"simple":true}

intent:
- chat: greeting, thanks, small talk, or a question that needs no tool
- file_op: list, read, write, move, copy, or delete files or folders
- gmail: read, search, draft, or send email
- knowledge: recall, remember, or search the knowledge base
- code: write, fix, debug, or explain code
- multi_step: two or more actions, or a plan ("then", "after that", "and also")
- status: time, date, weather, battery, disk, network, or git/system status
- other: unclear or mixed

simple: true for one obvious action. A short example, a single function, or a basic explanation is simple:true even when it names one file. simple:false for debug, refactor, a traceback, multi-step work, or more than one file.
entities: short strings copied from the message (paths, people, repos, places). [] if none."""

_INTENT: contextvars.ContextVar[Optional["TurnIntent"]] = contextvars.ContextVar(
    "primus_turn_intent", default=None
)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_ALIASES = {
    "email": "gmail",
    "mail": "gmail",
    "file": "file_op",
    "files": "file_op",
    "coding": "code",
    "multistep": "multi_step",
    "multi_step": "multi_step",
}
_CLASSIFY_TIMEOUT_SEC = 8


class TurnIntent(NamedTuple):
    intent: str
    entities: tuple[str, ...]
    simple: bool

    def prefers_fast_path(self) -> bool:
        """Clear one-action turns should try the existing in-process fast paths."""
        if not self.simple:
            return False
        return self.intent in {"chat", "file_op", "gmail", "knowledge", "code", "status"}


def set_turn_intent(intent: Optional[TurnIntent]) -> None:
    _INTENT.set(intent)


def clear_turn_intent() -> None:
    _INTENT.set(None)


def current_intent() -> Optional[TurnIntent]:
    return _INTENT.get()


def heavy_from_intent() -> Optional[bool]:
    """32B only when the label is clearly complex.

    ``None`` means this turn was not classified, or the label is ``other`` —
    the regex router decides. Chat, files, mail, knowledge, and status stay
    on the 7B even when ``simple`` is false.
    """
    hint = current_intent()
    if hint is None:
        return None
    if hint.intent == "multi_step":
        return True
    if hint.intent == "code" and not hint.simple:
        return True
    if hint.intent == "other":
        return None
    return False


def parse_intent(text: str) -> Optional[TurnIntent]:
    """Parse the classifier JSON. Unknown labels become ``other``."""
    raw = (text or "").strip()
    if not raw:
        return None
    fenced = _FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    label = str(obj.get("intent") or "").strip().lower().replace("-", "_").replace(" ", "_")
    label = _ALIASES.get(label, label)
    if label not in INTENTS:
        label = "other"

    entities: list[str] = []
    raw_ents = obj.get("entities") if isinstance(obj.get("entities"), list) else []
    if isinstance(obj.get("entities"), str):
        raw_ents = [obj["entities"]]
    for item in raw_ents[:6]:
        piece = str(item).strip()
        if piece:
            entities.append(piece[:80])

    simple_raw = obj.get("simple", None)
    if isinstance(simple_raw, bool):
        simple = simple_raw
    elif isinstance(simple_raw, str):
        simple = simple_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        simple = label not in {"code", "multi_step", "other"}
    if label in {"multi_step", "other"}:
        simple = False
    return TurnIntent(label, tuple(entities), simple)


def interpret_intent(message: str) -> Optional[TurnIntent]:
    """One bounded 7B call. Returns None when Ollama is down, slow, or unreadable."""
    msg = (message or "").strip()
    if not msg or msg.startswith("/"):
        return None
    try:
        ok, _health = _host.check_ollama_health()
    except Exception:  # noqa: BLE001
        return None
    if not ok:
        return None
    model = str(_host.CFG.get("model") or getattr(_host, "DEFAULT_MODEL", "") or "")
    if not model:
        return None
    try:
        from primus.agents.system import coerce_message_text, make_chat_ollama  # noqa: PLC0415
        from primus.core.redact import redact_model_input  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None

    safe = redact_model_input(msg[:1500])
    system = redact_model_input(INTENT_SYSTEM)
    llm = make_chat_ollama(model, temperature=0.0, num_predict=96)
    msgs = [
        _host.SystemMessage(content=system),
        _host.HumanMessage(content=safe),
    ]
    box: dict = {}
    err: list[Exception] = []

    def _run() -> None:
        try:
            box["resp"] = llm.invoke(msgs)
        except Exception as exc:  # noqa: BLE001
            err.append(exc)

    worker = threading.Thread(target=_run, name="primus-intent", daemon=True)
    worker.start()
    worker.join(_CLASSIFY_TIMEOUT_SEC)
    if worker.is_alive() or err or "resp" not in box:
        return None
    text = coerce_message_text(getattr(box["resp"], "content", box["resp"]))
    return parse_intent(text)
