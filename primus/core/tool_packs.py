"""Named tool packs for the inner ``create_agent``.

``build_tools()`` still registers every tool (120). The react agent only sees the
bound pack (~10 schemas). Fast paths never consult this module.

Library modules reach the host via ``_host`` — do not import admin_assistant.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Iterable, Optional

import sys as _sys

_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

# Real ``build_tools()`` names only — do not invent.
PACKS: dict[str, tuple[str, ...]] = {
    "mail": (
        "gmail_auth",
        "gmail_status",
        "gmail_list_messages",
        "gmail_read_message",
        "gmail_create_draft",
        "gmail_send_email",
    ),
    "files": (
        "list_directory",
        "read_file",
        "write_file",
        "move_path",
        "copy_path",
        "delete_file",
        "search_files",
        "create_directory",
    ),
    "shell": ("terminal", "git"),
    "knowledge": (
        "search_knowledge",
        "remember_fact",
        "remember_preference",
        "recall_memory",
    ),
    "brain": ("ask_grok",),
    "default": ("list_directory", "get_datetime", "get_news", "ask_grok"),
}

PACK_ORDER = ("mail", "files", "shell", "knowledge", "brain", "default")

_TOOL_TO_PACK: dict[str, str] = {
    name: pack for pack in PACK_ORDER for name in PACKS[pack]
}

_MAIL_RE = re.compile(
    r"\b(?:e-?mails?|gmail|inbox|mailbox|draft|compose)\b|\bmail\b(?!\s*box)",
    re.I,
)
_FILES_RE = re.compile(
    r"\b(?:files?|folders?|director(?:y|ies)|list_directory|read_file|write_file|"
    r"move_path|copy_path|delete_file|search_files|create_directory|"
    r"ls|cat|mkdir|mv|cp|rm)\b"
    r"|\b(?:list|show|read|write|move|copy|delete|remove|search|create)\b"
    r"[^.\n]{0,40}\b(?:file|folder|directory|dir|path|notes?|pdf|docx?|xlsx?)\b",
    re.I,
)
_SHELL_RE = re.compile(
    r"\b(?:terminal|bash|zsh|shell|command line|cli)\b"
    r"|\bgit\s+(?:status|log|diff|branch|remote|add|commit|push|pull|clone)\b"
    r"|^\s*(?:git|pwd|df|du|free|whoami|uptime|ps|kill)\b",
    re.I,
)
_KNOWLEDGE_RE = re.compile(
    r"\b(?:search_knowledge|remember_fact|remember_preference|recall_memory|"
    r"remember (?:that|this)|recall|knowledge base|\bkb\b)\b",
    re.I,
)
_BRAIN_RE = re.compile(r"\b(?:ask_grok|ask grok|\bgrok\b|need_brain|frontier)\b", re.I)
_SNAKE_RE = re.compile(r"\b([a-z][a-z0-9_]{2,})\b")
_INVENTED_MAIL = frozenset({
    "gmail_get_unread_email", "gmail_get_latest", "get_unread_email",
})

_TLS = threading.local()
_REGISTRY_NAMES: Optional[set[str]] = None
_REGISTRY_LOCK = threading.Lock()


def registry_names() -> set[str]:
    """Real ``build_tools()`` names. Cached. Does not shrink the registry."""
    global _REGISTRY_NAMES
    if _REGISTRY_NAMES is not None:
        return _REGISTRY_NAMES
    with _REGISTRY_LOCK:
        if _REGISTRY_NAMES is not None:
            return _REGISTRY_NAMES
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
        _REGISTRY_NAMES = names
        return names


def pack_names(pack: str) -> tuple[str, ...]:
    return PACKS.get(pack, PACKS["default"])


def pack_for_tool(name: str) -> Optional[str]:
    return _TOOL_TO_PACK.get(name)


def named_tools_in_text(text: str, *, allowed: Optional[set[str]] = None) -> list[str]:
    """Snake_case tokens that are real registry names, longest first, de-duped."""
    blob = text or ""
    allow = allowed if allowed is not None else registry_names()
    found: list[str] = []
    seen: set[str] = set()
    # Longest names first so gmail_list_messages wins over a stray shorter token.
    for name in sorted(allow, key=len, reverse=True):
        if name in seen:
            continue
        if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", blob):
            found.append(name)
            seen.add(name)
    return found


def invented_tool_in_text(text: str, *, allowed: Optional[set[str]] = None) -> Optional[str]:
    """First tool-shaped token that is not in the registry (invented mail, gmail_*, …)."""
    allow = allowed if allowed is not None else registry_names()
    for tok in _SNAKE_RE.findall(text or ""):
        if tok in allow or tok in _TOOL_TO_PACK:
            continue
        if tok in _INVENTED_MAIL or tok.startswith(("gmail_", "calendar_", "slack_")):
            return tok
    return None


def select_pack(text: str = "", tools: Optional[Iterable[str]] = None) -> str:
    """One pack. Receptionist ``tools[]`` wins when it names a known pack tool."""
    named = [str(t).strip() for t in (tools or []) if str(t).strip()]
    if named:
        # Prefer a concrete work pack over brain/default when mixed.
        hits: list[str] = []
        for t in named:
            p = pack_for_tool(t)
            if p and p not in hits:
                hits.append(p)
        for prefer in ("mail", "files", "shell", "knowledge", "brain"):
            if prefer in hits:
                return prefer
        if hits:
            return hits[0]

    blob = text or ""
    mentioned = named_tools_in_text(blob)
    if mentioned:
        for t in mentioned:
            p = pack_for_tool(t)
            if p:
                return p

    if _MAIL_RE.search(blob):
        return "mail"
    if _FILES_RE.search(blob):
        return "files"
    if _SHELL_RE.search(blob):
        return "shell"
    if _KNOWLEDGE_RE.search(blob):
        return "knowledge"
    if _BRAIN_RE.search(blob):
        return "brain"
    if re.search(
        r"\b(?:code|script|python|function|class|refactor|implement)\b", blob, re.I
    ):
        return "files"
    return "default"


def extras_from_named(
    text: str,
    pack: str,
    *,
    tools: Optional[Iterable[str]] = None,
    allowed: Optional[set[str]] = None,
) -> tuple[list[str], Optional[str]]:
    """Named tools outside the pack.

    In-registry extras are allowed (one or more explicit names — still not 120).
    Unknown / invented names → refuse one-liner (no bind).
    """
    allow = allowed if allowed is not None else registry_names()
    pack_set = set(pack_names(pack))
    extras: list[str] = []
    seen: set[str] = set()

    invented = invented_tool_in_text(text, allowed=allow)
    if invented:
        return [], f"I don't have a tool named `{invented}`."

    candidates: list[str] = []
    for t in tools or []:
        n = str(t).strip()
        if n:
            candidates.append(n)
    for n in named_tools_in_text(text, allowed=allow):
        if n not in candidates:
            candidates.append(n)

    for n in candidates:
        if n in _INVENTED_MAIL or n not in allow:
            return [], f"I don't have a tool named `{n}`."
        if n not in pack_set and n not in seen:
            extras.append(n)
            seen.add(n)
    return extras, None


def bind_tools(
    registry: list,
    pack: str,
    extra: Optional[Iterable[str]] = None,
) -> list:
    """Filter a ``build_tools()`` list to one pack (+ optional named extras).

    Never returns the full registry. Unknown pack → default.
    """
    if pack not in PACKS:
        pack = "default"
    wanted = set(PACKS[pack])
    allow = {getattr(t, "name", "") for t in (registry or [])} - {""}
    for name in extra or []:
        n = str(name).strip()
        if n and n in allow:
            wanted.add(n)
    bound = [t for t in (registry or []) if getattr(t, "name", "") in wanted]
    # Safety: a bug must not silently pass every schema.
    if registry and len(bound) >= len(registry) and len(registry) > 20:
        wanted = set(PACKS["default"])
        bound = [t for t in registry if getattr(t, "name", "") in wanted]
    return bound


def bound_names(tools: list) -> list[str]:
    return [getattr(t, "name", "") for t in (tools or []) if getattr(t, "name", "")]


def set_turn_pack(pack: str, extra: Optional[Iterable[str]] = None) -> None:
    _TLS.pack = pack if pack in PACKS else "default"
    _TLS.extra = [str(x).strip() for x in (extra or []) if str(x).strip()]


def clear_turn_pack() -> None:
    _TLS.pack = "default"
    _TLS.extra = []


def current_pack() -> str:
    pack = getattr(_TLS, "pack", None)
    return pack if pack in PACKS else "default"


def current_extra() -> list[str]:
    extra = getattr(_TLS, "extra", None)
    return list(extra) if extra else []


def bind_for_text(
    registry: list,
    text: str,
    *,
    tools: Optional[Iterable[str]] = None,
    pack: Optional[str] = None,
    allow_named_extra: bool = False,
) -> tuple[list, str, list[str], Optional[str]]:
    """Resolve pack + extras and bind. Returns (tools, pack, extra, refuse)."""
    chosen = pack if pack in PACKS else select_pack(text, tools=tools)
    extra: list[str] = []
    refuse: Optional[str] = None
    if allow_named_extra:
        extra, refuse = extras_from_named(text, chosen, tools=tools)
        if refuse:
            return [], chosen, [], refuse
    elif tools:
        extra, refuse = extras_from_named("", chosen, tools=tools)
        if refuse:
            extra = []
            refuse = None
        extra = [n for n in extra if n in {getattr(t, "name", "") for t in registry}]
    bound = bind_tools(registry, chosen, extra)
    return bound, chosen, extra, None
