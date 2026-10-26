"""Per-turn session state and the rules for what memory is worth reading or writing.

Pure Python. No model call and no network. The live copy sits on PrimusSession;
``apply_turn_state`` is the only function that touches the host.
"""
from __future__ import annotations

import re
import sys as _sys
from dataclasses import dataclass

_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

# Words that are not a topic, an entity, or a retrieval token.
_STOP = frozenset({
    "the", "a", "an", "this", "that", "it", "its", "me", "my", "you", "your",
    "please", "can", "could", "would", "should", "do", "does", "did", "is",
    "are", "was", "were", "what", "whats", "how", "why", "when", "where", "who",
    "and", "or", "for", "to", "with", "about", "from", "into", "onto", "than",
    "then", "just", "like", "want", "need", "tell", "show", "give", "make",
    "list", "open", "read", "write", "fix", "help", "some", "any", "there",
    "here", "have", "has", "been", "will", "shall", "them", "they", "their",
    "our", "ours", "also", "only", "very", "really", "thing", "stuff", "something",
    "please", "thanks", "thank", "hello", "okay", "yes", "yeah", "sure",
})

_FOLLOWUP_RE = re.compile(
    r"^(?:yes|no|ok|okay|sure|thanks|thank you|please|do it|go ahead|continue|"
    r"that one|the same|and|also|more|again)\b",
    re.I,
)
_LOOKUP_RE = re.compile(
    r"\b(weather|forecast|time|date|battery|wifi|bluetooth|disk|headlines?|"
    r"news|git status|what time|list|show me|how many)\b",
    re.I,
)
_GREET_RE = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|ok|okay|got it|cool|yes|no|sure|yo)[.!\s]*$",
    re.I,
)
_DURABLE_RE = re.compile(
    r"\b(remember|don'?t forget|do not forget|from now on|i prefer|i'?d prefer|"
    r"my preference|always use|never use|call me|my name is|we decided|decision:)\b",
    re.I,
)
_MEM_CUE_RE = re.compile(
    r"\b(remember|recall|forgot|memory|memor(?:y|ies)|last time|we discussed|"
    r"earlier|yesterday|prefer|preference|my name|who am i)\b",
    re.I,
)
_KB_CUE_RE = re.compile(
    r"\b(how does|what is|explain|documentation|knowledge base|our (?:code|project|repo))\b",
    re.I,
)
_CONV_CUE_RE = re.compile(
    r"\b(last time|we discussed|earlier|yesterday|remember when|previously|before that)\b",
    re.I,
)
_PATH_RE = re.compile(
    r"(?:~?/|\./)[^\s,;:\"']+|[\w.-]+\.(?:py|js|ts|tsx|md|json|txt|yml|yaml)\b"
)
_QUOTED_RE = re.compile(r"[\"']([^\"']{2,60})[\"']")
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?\b")
_CAP_SKIP = frozenset({
    "What", "When", "Where", "Why", "How", "Please", "The", "This", "That",
    "Can", "Could", "Would", "Should", "Show", "List", "Tell", "Give", "Open",
    "Read", "Make", "Fix", "Write", "Help", "Primus", "Thanks", "Hello",
})
_PREF_RES = (
    re.compile(r"\bi prefer(?: that)?\s+(.{3,80})", re.I),
    re.compile(r"\bmy preference is\s+(.{3,80})", re.I),
    re.compile(r"\b(?:from now on|always use|never use)\b[:,]?\s*(.{3,80})", re.I),
    re.compile(r"\bcall me\s+(.{2,40})", re.I),
    re.compile(r"\bremember(?: that)?\s+(.{3,80})", re.I),
)
_SUMMARY_MARK = "[Earlier in this chat:"


@dataclass(frozen=True)
class SessionState:
    """Topic, a few recent entities, and preferences stated this session."""

    topic: str = ""
    entities: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()


def _tokens(text: str) -> set[str]:
    found = {
        w for w in re.findall(r"[a-z0-9][\w-]{3,}", (text or "").lower())
        if w not in _STOP
    }
    for path in _PATH_RE.findall(text or ""):
        token = path.lower().rstrip(".,)")
        if len(token) >= 3:
            found.add(token)
    return found


def _clip_phrase(text: str, limit: int = 80) -> str:
    phrase = re.split(r"[.!?\n]", (text or "").strip(), maxsplit=1)[0]
    phrase = re.sub(r"\s+", " ", phrase).strip(" -,:;")
    return phrase[:limit].strip()


def _extract_entities(text: str) -> list[str]:
    found: list[str] = []
    for match in _PATH_RE.findall(text or ""):
        found.append(match.rstrip(".,)"))
    for match in _QUOTED_RE.findall(text or ""):
        found.append(match.strip())
    for match in _CAP_RE.findall(text or ""):
        if match.split()[0] not in _CAP_SKIP:
            found.append(match)
    return found


def _extract_preferences(text: str) -> list[str]:
    out: list[str] = []
    for pattern in _PREF_RES:
        match = pattern.search(text or "")
        if not match:
            continue
        phrase = _clip_phrase(match.group(1))
        if len(phrase) >= 3:
            out.append(phrase)
    return out


def _topic_from(text: str) -> str:
    lookup = _LOOKUP_RE.search(text or "")
    if lookup and lookup.group(1).lower() not in {"show me", "list", "how many", "what time"}:
        return lookup.group(1).lower()[:32]
    quoted = _QUOTED_RE.search(text or "")
    if quoted:
        return quoted.group(1)[:48]
    path = _PATH_RE.search(text or "")
    if path:
        token = path.group(0).rstrip(".,)")
        return token.rsplit("/", 1)[-1][:48]
    words = [w for w in re.findall(r"[A-Za-z][\w-]{3,}", text or "") if w.lower() not in _STOP]
    caps = [w for w in words if w[:1].isupper() and w not in _CAP_SKIP]
    if caps:
        return caps[0][:48]
    return words[0][:48] if words else ""


def _is_followup(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if _FOLLOWUP_RE.match(stripped):
        return True
    # A short reply with no new noun stays on the current topic.
    return len(stripped) <= 24 and not _tokens(stripped)


def update_session_state(
    prev: SessionState,
    message: str,
    *,
    entities: tuple[str, ...] | list[str] = (),
) -> SessionState:
    """Fold one user message into the session state. Keeps the topic on follow-ups."""
    text = (message or "").strip()
    merged: list[str] = []
    seen: set[str] = set()
    for raw in list(_extract_entities(text)) + [str(e) for e in entities] + list(prev.entities):
        item = re.sub(r"\s+", " ", (raw or "").strip())[:80]
        key = item.lower()
        if not item or key in seen or key in _STOP:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= 8:
            break
    prefs = list(prev.preferences)
    have = {p.lower() for p in prefs}
    for phrase in _extract_preferences(text):
        if phrase.lower() in have:
            continue
        prefs.insert(0, phrase[:120])
        have.add(phrase.lower())
    prefs = prefs[:6]
    if _is_followup(text) and prev.topic:
        topic = prev.topic
    else:
        topic = _topic_from(text) or prev.topic
    return SessionState(topic=topic[:60], entities=tuple(merged), preferences=tuple(prefs))


def recall_layers(message: str) -> frozenset[str]:
    """Which memory layers a turn should search.

    Empty means skip the embedding round-trip. Lookups and greetings stay empty
    so fast answers are not wrapped in unrelated notes.
    """
    text = (message or "").strip()
    if not text or _GREET_RE.match(text):
        return frozenset()
    mem = bool(_MEM_CUE_RE.search(text))
    conv = bool(_CONV_CUE_RE.search(text))
    if _LOOKUP_RE.search(text) and not mem and not conv and len(text) < 180:
        return frozenset()
    layers: set[str] = set()
    if mem or conv:
        layers.add("ltm")
    if conv:
        layers.add("conv")
    words = _tokens(text)
    if _KB_CUE_RE.search(text) or (len(text) > 100 and len(words) >= 3):
        layers.add("kb")
    if len(words) >= 3 and not _LOOKUP_RE.search(text):
        layers.add("ltm")
        layers.add("summary")
    elif layers and len(text) > 40:
        layers.add("summary")
    return frozenset(layers)


def persistence_for(user_msg: str) -> str:
    """How much of a finished turn is worth writing down.

    ``skip`` — nothing durable (lookups, greetings, short questions).
    ``session`` — one short note for this run, not the knowledge base.
    ``durable`` — long-term memory. Explicit remember / preference / decision lines.
    """
    text = (user_msg or "").strip()
    if _DURABLE_RE.search(text) and len(text) >= 8:
        return "durable"
    if len(text) < 12 or _GREET_RE.match(text):
        return "skip"
    if _LOOKUP_RE.search(text) and len(text) < 180:
        return "skip"
    if len(text) >= 160 or (len(text) >= 100 and not text.endswith("?")):
        return "session"
    return "skip"


def hit_is_relevant(
    content: str,
    query: str,
    *,
    topic: str = "",
    entities: tuple[str, ...] | list[str] = (),
) -> bool:
    """True when a stored note shares a distinctive token with this turn."""
    if not (content or "").strip():
        return False
    blob = (content or "").lower()
    for ent in list(_extract_entities(query)) + [str(e) for e in entities]:
        ent = ent.strip().lower()
        if len(ent) >= 3 and ent in blob:
            return True
    wanted = _tokens(query) | _tokens(topic) | _tokens(" ".join(str(e) for e in entities))
    if not wanted:
        return False
    overlap = wanted & _tokens(content)
    if any(len(token) >= 5 or "." in token or "/" in token for token in overlap):
        return True
    return len(overlap) >= 2


def clip_history(messages: list[dict], *, full: int = 4, clip: int = 240) -> list[dict]:
    """Copy `messages`, leaving the newest `full` intact and shortening the rest."""
    items = [m for m in messages if isinstance(m, dict)]
    if len(items) <= full:
        return [dict(m) for m in items]
    head, tail = items[:-full], items[-full:]
    out: list[dict] = []
    for msg in head:
        content = str(msg.get("content") or "")
        if len(content) > clip:
            content = content[:clip].rstrip() + "…"
        copied = dict(msg)
        copied["content"] = content
        out.append(copied)
    out.extend(dict(m) for m in tail)
    return out


def extractive_summary(messages: list[dict], *, limit: int = 500) -> str:
    """Local stand-in for a model summary: the latest older user lines, clipped."""
    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = re.sub(r"\s+", " ", str(msg.get("content") or "")).strip()
        if len(text) < 12:
            continue
        lines.append(text[:140])
    if not lines:
        return ""
    chosen = lines[-6:]
    while len(chosen) > 1 and len(" · ".join(chosen)) > limit:
        chosen = chosen[1:]
    return " · ".join(chosen)[:limit]


def render_focus(
    state: SessionState,
    message: str,
    recent: list[tuple[str, str]] | None = None,
) -> str:
    """Short system block. The transcript carries the older turns."""
    lines = ["## CURRENT FOCUS — stay on this thread"]
    if state.topic:
        lines.append(f"Topic: {state.topic}")
    if state.entities:
        lines.append("Entities: " + ", ".join(state.entities[:6]))
    if state.preferences:
        lines.append("Preferences: " + " | ".join(state.preferences[:4]))
    if recent:
        lines.append("Just before:")
        for who, text in recent[-2:]:
            snippet = re.sub(r"\s+", " ", (text or "")).strip()[:140]
            if snippet:
                lines.append(f"  {who}: {snippet}")
    if (message or "").strip():
        lines.append(f"Current request: {message.strip()[:220]}")
    lines.append(
        "Finish this request. Continue the thread when it matches. "
        "If the subject changes, follow the new request."
    )
    return "\n".join(lines)


def keep_summary_prefix(messages: list, limit: int) -> list:
    """Keep a leading summary message when a caller takes only the last N turns."""
    if limit <= 0 or len(messages) <= limit:
        return messages
    first = messages[0]
    content = str(getattr(first, "content", "") or "")
    if content.startswith(_SUMMARY_MARK):
        return [first] + list(messages[-(limit - 1):])
    return list(messages[-limit:])


_HYDRATED = False


def _hydrate_from_disk() -> None:
    """Load the last saved state once, so a follow-up after restart keeps its topic."""
    global _HYDRATED
    if _HYDRATED:
        return
    _HYDRATED = True
    try:
        saved = _host.get_memory_system().load_turn_state()
    except Exception:  # noqa: BLE001 — state is optional
        return
    if not isinstance(saved, dict):
        return
    sess = _host.PrimusSession
    if not getattr(sess, "current_topic", "") and saved.get("topic"):
        sess.current_topic = str(saved.get("topic") or "")[:60]
    if not list(getattr(sess, "recent_entities", []) or []) and saved.get("entities"):
        sess.recent_entities = [str(e)[:80] for e in list(saved.get("entities") or [])[:8]]
    if not list(getattr(sess, "active_preferences", []) or []) and saved.get("preferences"):
        sess.active_preferences = [str(p)[:120] for p in list(saved.get("preferences") or [])[:6]]


def apply_turn_state(message: str, *, entities: tuple[str, ...] | list[str] = ()) -> SessionState:
    """Update PrimusSession for this turn and persist only when the state changed."""
    _hydrate_from_disk()
    sess = _host.PrimusSession
    prev = SessionState(
        topic=str(getattr(sess, "current_topic", "") or ""),
        entities=tuple(getattr(sess, "recent_entities", []) or []),
        preferences=tuple(getattr(sess, "active_preferences", []) or []),
    )
    nxt = update_session_state(prev, message, entities=entities)
    sess.current_topic = nxt.topic
    sess.recent_entities = list(nxt.entities)
    sess.active_preferences = list(nxt.preferences)
    if nxt != prev:
        try:
            _host.get_memory_system().save_turn_state(
                nxt.topic, list(nxt.entities), list(nxt.preferences)
            )
        except Exception:  # noqa: BLE001 — a state write must not break the turn
            pass
    return nxt


def reset_hydration_for_tests() -> None:
    """Test hook. Production turns hydrate once per process."""
    global _HYDRATED
    _HYDRATED = False
