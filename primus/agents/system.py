"""Primus agent system (Phase 6 extraction).

The whole agent/orchestration layer lives here, moved VERBATIM from admin_assistant.py:
deterministic routing (RoutingDecision / ResponseCache / ModelRouter), the LangGraph builders
(build_agent_graph / build_primus_graph / build_forge_graph / init_agent_graphs), model plumbing
(make_chat_ollama, warm-up, enforce_sequential_models, DynamicModelManager), the fast paths
(instant_answer / fast_tool_answer / classify_complexity / route_prompt / _fast_chat_answer /
_fast_coding_answer), invoke_primus + the turn watchdog, and the background + scheduled agent
systems (BackgroundAgentManager / BackgroundTaskManager / ScheduledTaskManager).

The ONLY mechanical change vs. the original single-file code is that references to admin_assistant
module globals are reached through the live module object as ``_host.<name>`` — this keeps the
import-cycle safe and preserves runtime-rebound config/model globals (``_host.CFG``,
``_host.DEFAULT_MODEL``/``FORGE_MODEL``) and forward-declared helpers. ``_agent_graphs`` stays
owned by the host (it is rebound from both sides) and is reached as ``_host._agent_graphs``.
Nothing here is locked — edit routing, prompts wiring, fast paths, and agents exactly as before.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, NamedTuple, Optional, TypedDict

import sys as _sys

# Resolve the host module without importing it by name (admin_assistant runs as "__main__").
# Returns the already-loaded, live module object — see primus/tools/registry.py for the rationale.
_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

from primus.core.identity import OPERATOR as _OPERATOR, render as _render_op  # noqa: E402

# Regex alternation of the operator's project names ("" in a stock clone) — lets routing /
# recall / code-reference heuristics key off personalized projects without hard-coding them.
_OP_PROJECT_ALT = (
    "|" + "|".join(re.escape(p.lower()) for p in _OPERATOR["projects"]) if _OPERATOR["projects"] else ""
)
_OP_PROJECT_WORDS: tuple[str, ...] = tuple(p.lower() for p in _OPERATOR["projects"])

# langchain / langgraph surface bound from the host (the exact real objects or graceful stubs that
# admin_assistant resolved at startup). These exist well before this module is imported.
AIMessage = _host.AIMessage
BaseMessage = _host.BaseMessage
HumanMessage = _host.HumanMessage
SystemMessage = _host.SystemMessage
ChatOllama = _host.ChatOllama
StateGraph = _host.StateGraph
END = _host.END
MemorySaver = _host.MemorySaver
add_messages = _host.add_messages
create_agent = _host.create_agent

# === Stability substrate: retries, context pruning, no-progress stop, disk checkpoints =======
# Official LangChain 1.x middleware (no site-packages patching). Every react agent Primus/Forge
# builds gets the same stack so a tool/model hiccup becomes a message, not a crashed turn.
from langchain.agents.middleware import (  # noqa: E402
    AgentMiddleware,
    ContextEditingMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.context_editing import ClearToolUsesEdit  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402


class NoProgressMiddleware(AgentMiddleware):
    """Short-circuit exact-repeat tool calls (same tool + same args) inside one run.

    A model stuck re-issuing an identical failing call burns the recursion budget without
    changing anything. Repeats up to the limit are allowed (a legit retry can follow a real
    change); hitting the limit returns an error ToolMessage telling the model to change
    approach or report blocked — the loop then exits cleanly instead of hitting the
    recursion cap. State is thread-local: each graph invoke runs in its own worker thread,
    so streaks never leak across turns.
    """

    def __init__(self, max_identical: int = 3) -> None:
        super().__init__()
        self.max_identical = max(2, int(max_identical))
        self._local = threading.local()

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        call = getattr(request, "tool_call", None) or {}
        try:
            key = (str(call.get("name")), json.dumps(call.get("args") or {}, sort_keys=True, default=str))
        except (TypeError, ValueError):
            key = (str(call.get("name")), repr(call.get("args")))
        streak = getattr(self._local, "streak", None)
        count = streak[1] + 1 if streak and streak[0] == key else 1
        self._local.streak = (key, count)
        if count < self.max_identical:
            return handler(request)
        _host.log.warning("No-progress stop: %s called %dx with identical args", key[0], count)
        return ToolMessage(
            content=(
                f"No-progress stop: `{key[0]}` was called {count} times with identical "
                "arguments and nothing changed. Do NOT call it again with the same args — "
                "change the approach (different tool or different arguments) or report the "
                "step blocked with the exact error."
            ),
            tool_call_id=str(call.get("id") or "no-progress"),
            name=key[0],
            status="error",
        )


# Mutating filesystem tools are EXCLUDED from tool-retry: they report failures as strings
# (never raise), and retrying a mutation that partially succeeded is a double-write.
_NO_RETRY_TOOLS = frozenset({
    "write_file", "delete_file", "move_path", "copy_path", "create_directory",
    "batch_file_operations", "edit_docx", "edit_xlsx",
    "organize_downloads", "clean_temp_files", "sort_client_folders", "organize_by_extension",
})


def _agent_middleware(tools: list) -> list:
    """The standard middleware stack for every Primus/Forge react agent."""
    retryable = [getattr(t, "name", "") for t in tools if getattr(t, "name", "") not in _NO_RETRY_TOOLS]
    stack: list = [
        # Transient model errors (Ollama timeout/5xx/connection drop) retry with short backoff
        # before a step is allowed to fail.
        ModelRetryMiddleware(
            max_retries=int(_host.CFG.get("model_retry_max", 2)),
            initial_delay=1.0, backoff_factor=2.0, max_delay=8.0, jitter=True,
            on_failure="continue",
        ),
        # Transient tool errors retry too; past the cap the error becomes a ToolMessage the
        # model can react to — never an uncaught exception out of the graph. Mutating file
        # tools are not on the retry list (see _NO_RETRY_TOOLS).
        ToolRetryMiddleware(
            max_retries=int(_host.CFG.get("tool_retry_max", 2)),
            tools=retryable or None,
            initial_delay=0.5, backoff_factor=2.0, max_delay=6.0, jitter=True,
            on_failure="continue",
        ),
        NoProgressMiddleware(int(_host.CFG.get("no_progress_repeat_limit", 3))),
    ]
    if bool(_host.CFG.get("prune_tool_traces", True)):
        # Context prune WITHOUT an extra model call: old tool outputs are replaced by a short
        # placeholder once the window estimate crosses the trigger, keeping the original user
        # goal, system prompt, and the most recent messages (latest errors) intact.
        stack.append(
            ContextEditingMiddleware(
                edits=[ClearToolUsesEdit(
                    trigger=int(_host.CFG.get("prune_trigger_tokens", 3000)),
                    keep=int(_host.CFG.get("prune_keep_messages", 6)),
                    placeholder="[older tool output cleared — re-run the tool if you need it again]",
                )]
            )
        )
    return stack


_CHECKPOINTER: Any = None
_CHECKPOINTER_LOCK = threading.Lock()


def _get_checkpointer() -> Any:
    """Agent-thread checkpointer. OPTIONAL SQLite backend (~/.primus/checkpoints.sqlite) so
    thread state can survive a restart — but nothing depends on it: the in-memory MemorySaver
    is the default fallback whenever the sqlite extra isn't installed, the file can't be
    opened, or CFG["use_sqlite_checkpointer"] is false. Chat correctness, project-chat
    isolation, and Suggest/Execute never touch this file. Never raises.
    """
    global _CHECKPOINTER
    with _CHECKPOINTER_LOCK:
        if _CHECKPOINTER is None:
            if not bool(_host.CFG.get("use_sqlite_checkpointer", True)):
                _CHECKPOINTER = MemorySaver()
            else:
                try:
                    import sqlite3

                    from langgraph.checkpoint.sqlite import SqliteSaver

                    path = _host.APP_DIR / "checkpoints.sqlite"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    conn = sqlite3.connect(str(path), check_same_thread=False)
                    _CHECKPOINTER = SqliteSaver(conn)
                    _host.log.info("Agent checkpoints → %s", path)
                except Exception as exc:  # noqa: BLE001
                    _host.log.warning("SQLite checkpointer unavailable (%s) — using in-memory", exc)
                    _CHECKPOINTER = MemorySaver()
    return _CHECKPOINTER


# === Fast paths (instant/tool/coding answers, complexity + routing classifiers) ===
# Conjunctions / phrasing that signal a genuinely multi-step request.
_MULTI_STEP_RE = re.compile(
    r"\b(then|after that|afterwards|followed by|next,|and then|step by step|"
    r"first.*second|finally|once .* done)\b",
    re.I,
)

# Requests that map cleanly to a single tool action — execute immediately, no planning LLM.
_DIRECT_ACTION_PATTERNS: tuple[str, ...] = (
    r"^\s*(open|launch|start|run|fire up)\s+[\w.\- ]{1,40}$",
    r"^\s*(list|show|ls|display)\b.*\b(file|files|folder|folders|director|desktop|download|document|home|content)",
    r"^\s*(check|show|what'?s|whats|tell me)\b.*\b(battery|wifi|wi-?fi|bluetooth|disk|space|memory|ram|cpu|uptime|network|ip|time|date|status)\b",
    r"^\s*(ls|pwd|cat|tree|df|du|free|whoami|date|uptime|cd|head|tail|stat|file|which)\b",
    r"\bgit\s+(status|log|diff|branch|remote)\b",
    r"^\s*(lock|suspend|hibernate)\b.*\b(screen|session|laptop|computer|system|machine)?\s*$",
    r"^\s*(open|show me|take me to)\b.*\b(downloads|desktop|documents|home folder)\b",
    r"^\s*(open|launch|go to|browse|search|google|find)\b.*\b(google|youtube|chrome|firefox|chromium|brave|browser|web|reddit|github|maps|gmail|bing|duckduckgo|\.com)\b",
    r"^\s*search\s+(google|youtube|the web|online)\b",
    r"\b(weather|forecast|temperature outside|how (?:hot|cold|warm) is it|is it (?:going to )?(?:rain|snow))\b",
    r"\b(what time|current time|the time right now|today'?s date|what day is it|what'?s the date)\b",
)
_DIRECT_ACTION_RE = re.compile("|".join(_DIRECT_ACTION_PATTERNS), re.I)


# Instant local answers — no LLM, no RAG, no network. Pure speed for the most common asks.
_INSTANT_TIME_RE = re.compile(
    r"\b(what(?:'?s| is) the time|what time is it|current time|time right now|"
    r"tell me the time|got the time)\b",
    re.I,
)
_INSTANT_DATE_RE = re.compile(
    r"\b(what(?:'?s| is) (?:the |today'?s )?date|today'?s date|what day is it|"
    r"what(?:'?s| is) the day|current date|what(?:'?s| is) today)\b",
    re.I,
)


# "date and time" asked together — both must be answered (a date-only reply is a miss).
_INSTANT_BOTH_RE = re.compile(
    r"\b(?:date\s+and\s+time|time\s+and\s+date|day\s+and\s+time|date\s*&\s*time)\b",
    re.I,
)


def instant_answer(message: str) -> Optional[str]:
    """Return an instant local answer (time/date) with no model or network call, else None."""
    msg = (message or "").strip()
    if not msg or len(msg) > 64 or msg.startswith("/"):
        return None
    if _MULTI_STEP_RE.search(msg):
        return None
    low = msg.lower()
    want_time = bool(_INSTANT_TIME_RE.search(low))
    want_date = bool(_INSTANT_DATE_RE.search(low))
    if _INSTANT_BOTH_RE.search(low):
        want_time = want_date = True
    if not (want_time or want_date):
        return None
    now = datetime.now()
    day = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
    clock = now.strftime("%I:%M %p").lstrip("0")
    if want_time and want_date:
        return f"It's {day}, {clock}."
    if want_time:
        return f"It's {clock} right now."
    return f"Today is {day}."


# ---------------------------------------------------------------------------
# Deterministic TOOL fast-path
#
# Why: routing a request like "what's the weather" through the full qwen2.5:7b
# ReAct agent just so the model *decides* to call one obvious tool is brutally
# slow (observed ~130s on CPU/limited GPU). When the intent maps unambiguously
# to a single tool, we run that tool directly in Python — no planning LLM, no
# RAG, no embeddings — and return its (already clean) output. This turns
# multi-second waits into the tool's own latency (e.g. ~1s network for weather,
# instant for app launches). Precision is favoured over coverage: every matcher
# is conservative so normal conversation never gets hijacked.
# ---------------------------------------------------------------------------

_FT_WEATHER_LOC_IN_RE = re.compile(
    r"\bweather\s+(?:in|for|at|near|around)\s+(.+)$", re.I
)
_FT_WEATHER_LOC_PRE_RE = re.compile(
    r"^(?:what'?s|what is|how'?s|hows|the)?\s*(?:the\s+)?(.+?)\s+weather\b", re.I
)
_FT_WEATHER_INTENT_RE = re.compile(
    r"\b(weather|forecast|how (?:hot|cold|warm) is it|is it (?:going to )?(?:rain|snow))\b",
    re.I,
)
_FT_OPEN_RE = re.compile(r"^\s*(?:open|launch|start|run|fire up|bring up)\s+(.+?)\s*$", re.I)
_FT_OPEN_SKIP_RE = re.compile(r"\b(and|then|search|google for)\b|\bto\s+\w+\s+and\b", re.I)
_FT_FILLER_RE = re.compile(r"\b(today|right now|currently|like|please|now|outside|here)\b", re.I)

# Headless web-search intent — "search the web for X", "look up X", "google X", "latest on X".
_FT_WEB_PREFIX = r"^\s*(?:please\s+|can you\s+|could you\s+|hey\s+)?"
_FT_WEB_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(_FT_WEB_PREFIX + r"(?:do a |run a )?(?:web |internet |online )?search\s+(?:the\s+)?(?:web|internet|online)\s+(?:for\s+)?(.+)$", re.I),
    re.compile(_FT_WEB_PREFIX + r"search\s+(?:for\s+)?(.+?)\s+(?:online|on the (?:web|internet))\s*$", re.I),
    re.compile(_FT_WEB_PREFIX + r"(?:look up|google|research|find out(?: about)?)\s+(.+)$", re.I),
    re.compile(_FT_WEB_PREFIX + r"(?:what'?s|what is)\s+the latest\s+(?:news\s+)?(?:on|about)\s+(.+)$", re.I),
)
# Don't treat local/file look-ups as web searches.
_FT_WEB_SKIP_RE = re.compile(
    r"\b(file|files|folder|directory|on my (?:computer|machine|disk|drive|laptop|pc)|"
    r"locally|in my (?:home|downloads|documents|desktop))\b",
    re.I,
)


def _ft_websearch_query(msg: str) -> str:
    """Extract a clean web-search query from an explicit search request, else ''."""
    for pat in _FT_WEB_PATTERNS:
        m = pat.match(msg)
        if m:
            q = re.sub(r"[?!.]+$", "", m.group(1)).strip()
            q = re.sub(r"\s+(online|on the (?:web|internet))\s*$", "", q, flags=re.I).strip()
            return q
    return ""


def _ft_tool_text(fn: Any, **kwargs: Any) -> str:
    """Invoke a @tool function directly and return clean text, recording meta-learning stats."""
    name = getattr(fn, "name", None) or getattr(fn, "__name__", "tool")
    t0 = time.time()
    ok = True
    already = bool(getattr(fn, "_primus_audit", False))
    if not already:
        try:
            from primus.core.audit import tool_detail, write_audit  # noqa: PLC0415

            write_audit("tool_start", ok=True, detail=tool_detail(str(name), kwargs))
        except Exception:  # noqa: BLE001 — audit must never break the fast path
            pass
    try:
        out = coerce_message_text(fn.invoke(kwargs) if hasattr(fn, "invoke") else fn(**kwargs))
        # Heuristic success: a clear failure phrase at the start counts as a miss.
        low = out.strip().lower()
        if low.startswith(("i couldn't", "i can't", "failed", "error", "couldn't")):
            ok = False
        if not already:
            try:
                from primus.core.audit import write_audit  # noqa: PLC0415

                write_audit("tool_end", ok=ok, detail=str(name))
            except Exception:  # noqa: BLE001
                pass
        return out
    except Exception:
        ok = False
        if not already:
            try:
                from primus.core.audit import write_audit  # noqa: PLC0415

                write_audit("tool_end", ok=False, detail=str(name))
            except Exception:  # noqa: BLE001
                pass
        raise
    finally:
        try:
            _host.get_metrics().record_tool(name, ok, (time.time() - t0) * 1000.0)
        except Exception:  # noqa: BLE001
            pass


_FT_LOC_STOPWORDS = {
    "the", "a", "an", "my", "me", "current", "today", "tell", "give", "show",
    "get", "please", "now", "is", "it", "like", "whats", "what", "hows", "how",
    "local", "outside", "here", "right", "currently",
}


def _ft_weather_location(low: str) -> str:
    m = _FT_WEATHER_LOC_IN_RE.search(low)
    loc = m.group(1) if m else ""
    if not loc:
        m2 = _FT_WEATHER_LOC_PRE_RE.search(low)
        loc = m2.group(1) if m2 else ""
    loc = _FT_FILLER_RE.sub("", loc)
    loc = re.sub(r"[?!.,']+", " ", loc)
    # Drop leading/trailing filler words ("tell me the …", "… please") so only a real place remains.
    tokens = [t for t in loc.split() if t.lower() not in _FT_LOC_STOPWORDS]
    return " ".join(tokens).strip()


def _clean_mail_lines(text: str, max_lines: int = 8) -> list[str]:
    """Plain-text lines from a mail body: HTML/CSS stripped, tracking URLs, zero-width chars,
    and 'View in browser' boilerplate removed. Deterministic — no LLM, no timeout risk."""
    import html as _html

    t = re.sub(r"[​‌‍﻿­]", "", text)
    t = re.sub(r"(?is)<(style|script)\b[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = _html.unescape(t)
    out: list[str] = []
    for raw in t.splitlines():
        ln = re.sub(r"https?://\S+", "", raw).strip()
        ln = re.sub(r"\s{2,}", " ", ln)
        if len(ln) < 2:
            continue
        if re.match(
            r"(?i)(?:view (?:in|this) (?:browser|online)|unsubscribe|manage (?:your )?preferences|"
            r"privacy policy|terms of (?:service|use)|update your preferences)\b",
            ln,
        ):
            continue
        out.append(ln)
        if len(out) >= max_lines:
            break
    return out


def _summarize_mail(full: str) -> str:
    """From/Subject/Date + 4–8 lines of plain body text from a gmail_read_message result."""
    def _hdr(name: str) -> str:
        m = re.search(rf"^{name}:\s*(.+)$", full, re.M | re.I)
        return m.group(1).strip() if m else "?"

    body = full
    subj_m = re.search(r"^Subject:.*$", full, re.M | re.I)
    if subj_m:
        body = full[subj_m.end():]
    lines = _clean_mail_lines(body, max_lines=8)
    head = f"From: {_hdr('From')}\nSubject: {_hdr('Subject')}\nDate: {_hdr('Date')}"
    return head + "\n\n" + ("\n".join(lines) if lines else "(no readable body text)")


# Location words after from/in — not a person. "from my inbox" / "from gmail" is latest-mail.
_MAIL_LOCATION_TOKENS = frozenset({
    "inbox", "gmail", "mail", "email", "e-mail", "mailbox",
    "account", "folder", "home", "device", "phone", "laptop",
})


def _fast_mail_answer(reg: Any, low: str, msg: str = "") -> str:
    """In-process latest-mail answer using ONLY the real Gmail tools (gmail_list_messages →
    gmail_read_message). Never invents a tool name, never raises, never shows a timeout shrug.

    "email from NAME" narrows the list query with from:NAME; a from-ask that matches nothing
    says so — it never silently substitutes the newest unrelated message. Location tokens
    (inbox, gmail, mail, …) are not senders: "from my inbox" is the same path as latest email.
    """
    want_unread = bool(re.search(r"\bunread\b|\bnew\b", low))
    from_m = re.search(r"\b(?:from|by)\s+(?:my\s+)?([A-Za-z][\w.&'-]{0,40})", msg or low)
    sender = (from_m.group(1).strip(".?!,'\"") if from_m else "") or ""
    if sender.lower() in ("me", "you", "the", "a", "an") or sender.lower() in _MAIL_LOCATION_TOKENS:
        sender = ""
    try:
        connected = reg._GMAIL_TOKEN_PATH.exists() or reg._gmail_fixture_active()
    except Exception:  # noqa: BLE001 — let the tool's own error text speak instead
        connected = True
    if not connected:
        return (
            "Gmail isn't connected yet — it's a one-time Desktop OAuth setup. Put the Google "
            "client-secret JSON at `~/.primus/gmail_credentials.json`, then open Menu → Inbox "
            "and approve access (or ask me to run gmail_auth). After that I'll read your mail "
            "directly."
        )
    query_parts: list[str] = []
    if sender:
        query_parts.append(f"from:{sender}")
    if want_unread:
        query_parts.append("is:unread")
    query = " ".join(query_parts)
    listing = _ft_tool_text(reg.gmail_list_messages, query=query, max_results=1)
    # Ids are anchored to the "• [id] date — sender" bullet both the API and the fixture emit
    # (fixture ids are short, real Gmail ids are long — don't gate on length).
    m = re.search(r"(?m)^\s*•\s*\[([^\]]+)\]", listing)
    # Never substitute an unrelated message for a from-ask: no id, an explicit no-match, or a
    # listing whose sender line doesn't mention NAME all mean "no email from NAME".
    if (
        not m
        or (sender and "no messages" in listing.lower())
        or (sender and sender.lower() not in listing.lower())
    ):
        if sender and not re.search(r"⚠|🔐|gmail_auth|not connected", listing, re.I):
            return f"No email from {sender} in the recent inbox."
        if not m:
            # Empty mailbox or the tool's own clean auth/error text — already human-readable.
            return listing
    full = _ft_tool_text(reg.gmail_read_message, message_id=m.group(1))
    if re.search(r"\b(summari[sz]e|summary|tl;?dr)\b", low):
        return _summarize_mail(full)
    return full


def _ft_count_target(low: str) -> str:
    """Resolve the directory named in a count/list-folders ask to a guarded path (default ~)."""
    named = {
        "home": "~", "downloads": "~/Downloads", "desktop": "~/Desktop",
        "documents": "~/Documents", "docs": "~/Documents", "pictures": "~/Pictures",
    }
    # Explicit path wins ("in ~/Downloads", "in /home/operator/projects").
    m = re.search(r"(~[\w/.-]*|/[\w/.-]+)", low)
    if m:
        return m.group(1)
    # "in <place>" — a tight 1–3 word window so "in it and give me the total" can't swallow
    # the clause; trailing "folder/directory" words are decoration.
    m = re.search(
        r"\bin\s+(?:my\s+)?([\w-]+(?:\s+[\w-]+){0,2}?)(?:\s+(?:folder|directory|dir))?\s*(?:[?!.,]|$)",
        low,
    )
    cand = (m.group(1) if m else "").strip().strip("/")
    if cand and cand.lower() not in ("it", "there", "that", "them", "total"):
        return named.get(cand.lower(), cand)
    # Named location mentioned anywhere ("Look through my Home folder …").
    for word, path in named.items():
        if re.search(rf"\b{word}\b", low):
            return path
    return "~"


def fast_tool_answer(message: str) -> Optional[str]:
    """Run the single obvious tool for a clear request, returning clean text — else None.

    No LLM, no RAG, no embeddings. Keeps everyday actions snappy and works even
    when Ollama is unavailable (apps still open, weather still fetches).
    """
    if not _host.CFG.get("fast_tool_path", True):
        return None
    msg = (message or "").strip()
    if not msg or msg.startswith("/"):
        return None
    low = msg.lower()

    # --- Compound fact turn: leading hello/please is decoration. Time/date AND news and/or
    # weather in one message = Fact, never Work. Run in-process: get_datetime, then get_news
    # if asked, then get_weather only if asked. No plan, no RAG, no agent, no tool graph.
    # If news/weather fails, the time is still returned. ---
    dt_intent = bool(
        _INSTANT_TIME_RE.search(low) or _INSTANT_DATE_RE.search(low)
        or re.search(r"\b(?:the\s+)?(?:date|time|day)\b", low)
    )
    news_intent = bool(re.search(r"\b(?:news|headlines?|top stor(?:y|ies)|big stor(?:y|ies))\b", low))
    wx_intent = bool(_FT_WEATHER_INTENT_RE.search(low))
    if dt_intent and (news_intent or wx_intent) and not re.search(
        r"\b(?:file|files|folder|directory|e-?mail|gmail|inbox|write|create|delete|organize|"
        r"download|upload|script|code|install)\b",
        low,
    ):
        reg = _registry_mod()
        if reg is not None:
            _host.PrimusSession.emit_think("Fast tool", "compound fact — direct", "running")
            parts: list[str] = []
            try:
                parts.append(f"It's {_ft_tool_text(reg.get_datetime)}.")
            except Exception:  # noqa: BLE001 — datetime is local; should never fail
                pass
            if news_intent:
                try:
                    news = _ft_tool_text(reg.get_news, topic="")
                    if news and not news.strip().lower().startswith(("i couldn't", "couldn't", "error")):
                        parts.append(news)
                    else:
                        parts.append("The news feed didn't answer — the time above is still good.")
                except Exception:  # noqa: BLE001 — news failure must not eat the time
                    parts.append("The news feed didn't answer — the time above is still good.")
            if wx_intent:
                try:
                    parts.append(_ft_tool_text(reg.get_weather, location=""))
                except Exception:  # noqa: BLE001
                    pass
            if parts:
                return "\n\n".join(parts)

    # --- Latest mail / inbox check → real Gmail tools, in-process (never an invented tool) ---
    if not re.match(r"^\s*(?:send|draft|compose|write|reply|forward|delete|archive)\b", low) and re.search(
        r"\b(?:most recent|latest|last|newest|unread)\b[^?\n]{0,40}\b(?:e-?mails?|mail|messages?|inbox)\b"
        r"|\b(?:e-?mails?|inbox)\b[^?\n]{0,40}\b(?:most recent|latest|last|newest|unread)\b"
        r"|\b(?:fetch|get|check|read|show)\b[^?\n]{0,30}\b(?:my\s+)?(?:e-?mails?|mail|inbox|gmail)\b"
        r"|\bcheck\s+(?:my\s+)?gmail\b|\bany new (?:e-?mails?|mail|messages?)\b",
        low,
    ):
        reg = _registry_mod()
        if reg is not None:
            _host.PrimusSession.emit_think("Fast tool", "gmail latest — direct", "running")
            return _fast_mail_answer(reg, low, msg)

    # --- Count folders/files in a directory → list_directory (a FILE ask, never a code dump) ---
    count_m = re.search(
        r"\b(?:how many|count|total(?:\s+count)?(?:\s+number)?(?:\s+of)?|number of)\b[^?\n]{0,50}"
        r"\b(folders?|directories|files)\b",
        low,
    )
    if count_m and not re.search(r"\b(?:code|script|python|function|program)\b", low):
        reg = _registry_mod()
        if reg is None:
            return None
        kind = count_m.group(1).lower()
        target = _ft_count_target(low)
        _host.PrimusSession.emit_think("Fast tool", f"list_directory({target}) — direct", "running")
        listing = _ft_tool_text(reg.list_directory, path=target)
        if listing.startswith(("✗", "Refusing", "Not found", "No matches")):
            return listing
        want_dir = kind.startswith(("folder", "director"))
        mark = "[dir]" if want_dir else "[file]"
        names = [
            ln.split("] ", 1)[1].rsplit(" (", 1)[0]
            for ln in listing.splitlines()
            if ln.startswith(mark)
        ]
        noun = "folders" if want_dir else "files"
        if not names:
            return f"There are no {noun} in `{target}`."
        if len(names) == 1:
            noun = noun[:-1]  # "1 folder", not "1 folders"
        shown = ", ".join(names[:8]) + (f", … and {len(names) - 8} more" if len(names) > 8 else "")
        return f"There are **{len(names)} {noun}** in `{target}`: {shown}"

    # Multi-step still leaves the fast path (compound/mail/count already ran above).
    if _MULTI_STEP_RE.search(msg):
        return None

    # --- List a directory → list_directory (empty is 0 files, 0 folders — not "No matches") ---
    list_m = re.match(r"^\s*(?:list|ls)\s+(.+?)\s*$", msg, re.I)
    if list_m:
        reg = _registry_mod()
        if reg is not None:
            raw = list_m.group(1).strip()
            raw = re.sub(r"^(?:the\s+)?(?:folder|directory|dir)\s+", "", raw, flags=re.I)
            target = _ft_count_target(raw.lower()) if raw else "~"
            if raw.startswith("~") or raw.startswith("/"):
                target = raw.split()[0]
            _host.PrimusSession.emit_think("Fast tool", f"list_directory({target}) — direct", "running")
            listing = _ft_tool_text(reg.list_directory, path=target)
            if listing.startswith(("✗", "Refusing", "Not found")):
                return listing
            if (
                (not listing.strip())
                or listing.startswith("No matches")
                or listing.startswith("0 files, 0 folders")
            ):
                return f"0 files, 0 folders in `{target}`."
            return listing

    # --- Weather → get_weather (network-bound, ~1s) ---
    # Only treat as a *current weather* request: skip when it's really an action
    # ("open a ticket about the weather") or a discussion ("weather patterns in history").
    # Uncapped: a long "weather in …" ask is still weather, not a planner job.
    if _FT_WEATHER_INTENT_RE.search(low):
        weather_blocked = bool(
            re.match(r"^(open|launch|start|run|make|create|file|send|write|add|build|install|fix|debug)\b", low)
            or re.search(r"\b(history|patterns|climate change|why (?:is|does)|explain|how does weather)\b", low)
        )
        if not weather_blocked:
            loc = _ft_weather_location(low)
            _host.PrimusSession.emit_think("Fast tool", f"weather({loc or 'local'}) — direct", "running")
            return _ft_tool_text(_host.get_weather, location=loc)

    # --- Open app / folder / site → open_application or browse_web (instant) ---
    m = _FT_OPEN_RE.match(msg)
    if m and not _FT_OPEN_SKIP_RE.search(low):
        target = m.group(1).strip().strip("'\"")
        target = re.sub(r"^(my|the)\s+", "", target, flags=re.I)
        target = re.sub(r"\s+(app|application|folder|window|up)$", "", target, flags=re.I).strip()
        if 1 <= len(target.split()) <= 3:
            key = target.lower()
            _host.PrimusSession.emit_think("Fast tool", f"open({target}) — direct", "running")
            if key in _host._SITE_HOMES or (("." in key) and (" " not in key)):
                # "open" is an explicit visibility verb → show a real browser window.
                return _ft_tool_text(
                    _host.browse_web,
                    query_or_url=target if "." in key else "",
                    site=key if key in _host._SITE_HOMES else "",
                    visible=True,
                )
            return _ft_tool_text(_host.open_application, target=target)

    # --- News / headlines → get_news (RSS, headless) ---
    # Singular "headline" and "biggest story" count too — "what is the biggest headline in
    # tech today?" is NEWS, not a knowledge question. Uncapped so a long headline ask still
    # hits get_news instead of falling through the 90-char bail-out.
    if re.search(
        r"\b(latest news|headlines?|news (?:about|on)|in the news|what'?s happening (?:with|in)|"
        r"(?:biggest|top|latest) stor(?:y|ies))\b",
        low,
    ) and not re.match(r"^(open|launch|start|run|make|create|file|write|add)\b", low):
        topic_m = re.search(r"\bnews (?:about|on)\s+(.+)$|\bhappening (?:with|in)\s+(.+)$", low)
        topic = ""
        if topic_m:
            topic = (topic_m.group(1) or topic_m.group(2) or "").strip(" ?.!")
        if not topic:
            # "headline(s) in/on tech", "tech headlines", "biggest story in tech" → topic word.
            tm = re.search(
                r"\b(?:headlines?|stor(?:y|ies)|news)\s+(?:in|on|about|for)\s+(\w+)"
                r"|\b(tech|technology|world|business|science|sports|politics|ai)\b",
                low,
            )
            if tm:
                topic = (tm.group(1) or tm.group(2) or "").strip()
        _host.PrimusSession.emit_think("Fast tool", f"get_news({topic or 'top'}) — direct", "running")
        return _ft_tool_text(_host.get_news, topic=topic)

    # Length cap keeps the remaining fast path for short commands — but allow long messages
    # that carry a URL (so "read this article https://very/long/url" still routes to read_article).
    if len(msg) > 90 and not re.search(r"https?://", msg):
        return None

    # --- Simple status checks → device/disk tools (fast, local) ---
    is_query = bool(re.match(r"^\s*(what'?s|whats|what is|how'?s|hows|how much|check|show|is|do i have)\b", low))
    if is_query and len(low) <= 60:
        if re.search(r"\bbatter(y|ies)\b", low):
            _host.PrimusSession.emit_think("Fast tool", "battery — direct", "running")
            return _ft_tool_text(_host.device_management, action="battery")
        if re.search(r"\bwi-?fi\b", low):
            _host.PrimusSession.emit_think("Fast tool", "wifi — direct", "running")
            return _ft_tool_text(_host.device_management, action="wifi")
        if re.search(r"\bbluetooth\b", low):
            _host.PrimusSession.emit_think("Fast tool", "bluetooth — direct", "running")
            return _ft_tool_text(_host.device_management, action="bluetooth")
        if re.search(r"\b(disk space|free space|disk usage|space left|storage left|how much (?:space|storage))\b", low):
            _host.PrimusSession.emit_think("Fast tool", "disk — direct", "running")
            return _ft_tool_text(_host.disk_cleanup, mode="report")

    # --- Read/summarize a URL → headless article extraction (Trafilatura) ---
    url_m = re.search(r"https?://\S+", msg)
    if url_m and re.search(r"\b(read|summar(?:y|ise|ize)|extract|article|what does (?:it|this|the page) say)\b", low):
        _host.PrimusSession.emit_think("Fast tool", "read_article — direct", "running")
        return _ft_tool_text(_host.read_article, url=url_m.group(0).rstrip(").,"))

    # --- System resource monitor → psutil snapshot (fast, local) ---
    if re.search(
        r"\b(system monitor|resource (?:usage|monitor)|cpu and (?:ram|memory)|memory and cpu|"
        r"what'?s using my (?:cpu|ram|memory)|how(?:'?s| is) my system|system (?:load|resources))\b",
        low,
    ):
        detail = "full" if re.search(r"\b(full|detailed|everything)\b", low) else "summary"
        _host.PrimusSession.emit_think("Fast tool", "system_monitor — direct", "running")
        return _ft_tool_text(_host.system_monitor, detail=detail)

    # --- Explicit desktop GUI control → desktop_control (needs X11; guarded) ---
    if re.match(r"^(?:take |grab )?(?:a )?screenshot\b", low):
        _host.PrimusSession.emit_think("Fast tool", "desktop_control(screenshot) — direct", "running")
        return _ft_tool_text(_host.desktop_control, action="screenshot")
    m = re.match(r"^type\s+(.+)$", msg, re.I)
    if m:
        _host.PrimusSession.emit_think("Fast tool", "desktop_control(type) — direct", "running")
        return _ft_tool_text(_host.desktop_control, action="type", value=m.group(1).strip("'\""))
    m = re.match(r"^(?:press|hit)\s+(.+)$", low)
    if m:
        val = re.sub(r"\b(the|key|button)\b", "", m.group(1)).strip()
        val = val.replace(" plus ", "+")
        act = "hotkey" if "+" in val else "press"
        _host.PrimusSession.emit_think("Fast tool", f"desktop_control({act}) — direct", "running")
        return _ft_tool_text(_host.desktop_control, action=act, value=val)

    # --- Reddit / community search → search_reddit (headless JSON) ---
    rm = re.match(r"^(?:search\s+)?reddit(?:\s+for)?\s+(.+)$", low) or \
        re.match(r"^what (?:does|do) reddit (?:think|say)(?:\s+about)?\s+(.+)$", low)
    if rm and not re.match(r"^(open|launch)\b", low):
        rq = rm.group(1).strip(" ?.!")
        _host.PrimusSession.emit_think("Fast tool", f"search_reddit({rq[:40]}) — direct", "running")
        return _ft_tool_text(_host.search_reddit, query=rq)

    # --- Finance / markets / prices → precise web_search (NOT generic news, avoids drift) ---
    if re.search(
        r"\b(stock|stocks|stock market|share price|shares?|ticker|nasdaq|s&p|dow|"
        r"crypto|bitcoin|btc|ethereum|eth|price of|how much is|exchange rate|"
        r"market(?:s)? (?:doing|today|up|down)|earnings|dividend)\b",
        low,
    ) and not re.match(r"^(open|launch|start|run|make|create|file|write|add)\b", low):
        _host.PrimusSession.emit_think("Fast tool", "finance → web_search (precise)", "running")
        return _host._run_web_search(re.sub(r"[?!.]+$", "", msg).strip(), queue_kb=False)

    # --- Explicit internet search → headless web_search (no visible browser) ---
    if not _FT_WEB_SKIP_RE.search(low):
        wq = _ft_websearch_query(msg)
        if wq:
            _host.PrimusSession.emit_think("Fast tool", f"web_search({wq[:40]}) — direct", "running")
            return _host._run_web_search(wq, queue_kb=True)
    return None


def is_direct_action(message: str) -> bool:
    """True when the request is a simple, single tool-mappable action that should run immediately."""
    msg = (message or "").strip()
    if not msg or msg.startswith("/"):
        return False
    if len(msg) > 140:
        return False
    if _MULTI_STEP_RE.search(msg):
        return False
    # Multiple imperative clauses joined by "and" usually means >1 action.
    if re.search(r"\band\b", msg, re.I) and len(msg.split()) > 12:
        return False
    return bool(_DIRECT_ACTION_RE.search(msg))


# Signals that a request is heavier work worth a "working on it…" ack.
_COMPLEX_RE = re.compile(
    r"\b(write|build|create|implement|refactor|debug|fix|design|set up|setup|configure|"
    r"automate|migrate|analyze|audit|review|organize|sort|backup|deploy|integrate|"
    r"script|pipeline|workflow|scrape|generate)\b",
    re.I,
)


def classify_complexity(message: str) -> str:
    """Deterministic simple|complex split for response pacing (no LLM)."""
    msg = (message or "").strip()
    if not msg:
        return "simple"
    if is_direct_action(msg):
        return "simple"
    if _MULTI_STEP_RE.search(msg) or _COMPLEX_RE.search(msg):
        return "complex"
    if len(msg) > 160 or "```" in msg:
        return "complex"
    return "simple"


# ---------------------------------------------------------------------------
# Turn contract — the "pulse" bucket
#
# Greetings, thanks, and identity statements ("I am the operator, you are Primus") are
# conversation, not work. They must never reach the planner: a one-step plan on a
# twelve-word identity lock is exactly what produced "Progress: 0/1 steps complete",
# a Chinese "Step 1", and a timeout notice as the visible reply.
#
# Detection is deterministic and deliberately narrow. Anything carrying a tool hint,
# a work verb, a multi-step marker, or real length falls through to normal routing,
# so this can only ever make a chat turn cheaper — never swallow real work.
# ---------------------------------------------------------------------------
# The operator's first name (or a generic "operator" fallback) anchors "I am <name>" detection.
_OP_FIRST_NAME = (
    re.escape(_OPERATOR["name"].split()[0]) if _OPERATOR["custom"] else r"(?:the\s+)?operator"
)
_PULSE_IDENTITY_RE = re.compile(
    rf"\bi(?:'m| am) {_OP_FIRST_NAME}\b|\byou(?:'re| are) primus\b|\byour name is primus\b"
    r"|\bwho are you\b|\bwhat are you\b|\bwhat'?s your name\b"
    r"|\b(?:do|can|did|will) you (?:understand|know|get|remember) (?:that|this|who|me|it)\b",
    re.I,
)
_PULSE_THANKS_RE = re.compile(
    r"^\s*(?:thanks|thank you|thx|ty|appreciate it|nice work|good work|well done|"
    r"perfect|nice one)\b"
    r"|^\s*great[.!]?\s*$",
    re.I,
)
_PULSE_SIGNOFF_RE = re.compile(r"\b(?:good ?night|bye|goodbye|later|see you|talk later)\b", re.I)
_PULSE_RE = re.compile(
    r"^\s*(?:hi|hey|hello|yo|sup|howdy|greetings|good (?:morning|afternoon|evening|day))\b"
    r"|\bhow (?:are|r) (?:you|u)\b|\bhow'?s it going\b|\bhow are things\b|\byou good\b"
    r"|\bhow have you been\b|\bare you (?:there|online|up|awake|ready|with me)\b"
    r"|" + _PULSE_IDENTITY_RE.pattern
    + r"|" + _PULSE_THANKS_RE.pattern
    + r"|" + _PULSE_SIGNOFF_RE.pattern,
    re.I,
)
# Any of these means real work is being asked for, even inside a friendly sentence — they veto
# the pulse bucket so the turn routes normally. ("make sure" is excluded: it's conversational.)
_PULSE_VETO_RE = re.compile(
    r"\b(?:weather|forecast|temperature|time|date|news|headlines?|e-?mail|gmail|inbox|mail|"
    r"calendar|slack|file|files|folder|directory|disk|battery|wifi|bluetooth|cpu|ram|process|"
    r"git|repo|code|script|app|website|write|build|create|fix|debug|refactor|install|update|"
    r"upgrade|backup|research|search|look up|google|summari[sz]e|analy[sz]e|read|open|launch|"
    r"run|execute|list|show|schedule|remind|todo|note|organize|clean|delete|remove|send|draft|"
    r"find|locate|download|ingest|upload|move|copy|rename|edit|check|test|deploy|commit|push|"
    r"pull|clone|scan|monitor|query|connect|disconnect|mount|unmount)\b"
    r"|\bmake\b(?!\s+sure)",
    re.I,
)

IDENTITY_ACK = _render_op("You're {op_full}. I'm Primus. What do you need?")


def is_pulse_turn(message: str) -> bool:
    """True for a pulse turn (greeting, thanks, identity, 'you there?'): chat only.

    A pulse turn gets one bounded chat call — zero tools, zero plan, zero step machinery.
    """
    msg = (message or "").strip()
    if not msg or msg.startswith("/"):
        return False
    if len(msg) > 160 or len(msg.split()) > 26 or "```" in msg:
        return False
    if _MULTI_STEP_RE.search(msg) or _PULSE_VETO_RE.search(msg):
        return False
    return bool(_PULSE_RE.search(msg))


def pulse_reply(message: str) -> str:
    """Deterministic pulse answer — used when no model call is available or finishes in time.

    Never a plan, never a resume pointer: a greeting that times out still gets a real sentence.
    """
    msg = (message or "").strip()
    if _PULSE_IDENTITY_RE.search(msg):
        return IDENTITY_ACK
    if _PULSE_THANKS_RE.search(msg):
        return _render_op("Anytime{op_voc}. What's next?")
    if _PULSE_SIGNOFF_RE.search(msg):
        return _render_op("Good night{op_voc} — I'll be here.")
    return _render_op("I'm up and running{op_voc}. What do you need?")


# One calm sentence for a WORK-graph overrun. Deliberately not a partial-plan ledger on a chat
# turn: there is no plan on a chat turn, so there is nothing to resume. Never "ask me
# again" — the turn is over; the next message is a fresh turn by definition.
CHAT_TIMEOUT_REPLY = (
    "That ran too long on my side, so I stopped it — nothing was changed or sent."
)
# Chat/knowledge-tier overrun: the single bounded call outran its limit. A question was never
# going to change or send anything, so the work-graph line above is wrong here — one or two
# calm sentences about the only real fact: the model was slow.
CHAT_SLOW_REPLY = "I couldn't get an answer out in time — the model was slow on that one."


def acknowledgment_text(message: str) -> str:
    """Short, natural 'I'm on it' line shown immediately before the real answer streams in."""
    if is_pulse_turn(message):
        return "_Thinking…_"
    if is_direct_action(message):
        return "_On it — working in the background…_"
    if classify_complexity(message) == "complex":
        # Forge-aware: if this will be delegated to the coding specialist, say so naturally.
        try:
            if ModelRouter.analyze(message).agent_id == "forge":
                return "_Got it — handing this to Forge. One moment…_"
        except Exception:
            pass
        return "_Got it — working on that in the background. I'll have it for you in a moment…_"
    return "_Thinking…_"


# === Dual-model orchestration: routing, LangGraph builders, invoke_primus, watchdog ===
class RoutingDecision(NamedTuple):
    """Result of deterministic routing analysis."""
    agent_id: str          # "primus" | "forge"
    model_name: str
    reason: str
    intent: str            # admin | memory | chat | coding | automation | technical
    confidence: float      # 0–1 routing confidence
    forge_score: float
    primus_score: float
    reasoning: str         # human-readable chain for thinking panel


class ResponseCache:
    """TTL cache for trivial Primus responses — avoids redundant LLM calls."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def _key(self, message: str, mode: str) -> str:
        norm = re.sub(r"\s+", " ", message.strip().lower())[:200]
        return hashlib.sha256(f"{mode}:{norm}".encode()).hexdigest()[:24]

    def get(self, message: str, mode: str) -> Optional[str]:
        ttl = int(_host.CFG.get("response_cache_ttl_sec", 300))
        if ttl <= 0:
            return None
        key = self._key(message, mode)
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            text, ts = entry
            if time.time() - ts > ttl:
                del self._store[key]
                return None
            return text

    def put(self, message: str, mode: str, response: str) -> None:
        max_n = int(_host.CFG.get("response_cache_max", 64))
        ttl = int(_host.CFG.get("response_cache_ttl_sec", 300))
        if ttl <= 0 or len(response) > 800:
            return
        key = self._key(message, mode)
        with self._lock:
            self._store[key] = (response, time.time())
            if len(self._store) > max_n:
                oldest = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest]


_response_cache = ResponseCache()


class ModelRouter:
    """Weighted intent router — deterministic, explainable, Forge-conservative.

    Scoring model:
      - Each pattern match adds weighted points to forge_score or primus_score
      - Intent phrases (build a script, automate, debug) add bonus forge weight
      - Primus fast-path patterns veto Forge unless forge_score is very high
      - Delegates only when forge_score >= router_forge_threshold (default 3.0)
    """

    # --- Forge signal groups (pattern, weight) ---
    FORGE_CODE: list[tuple[str, float]] = [
        (r"\b(write|fix|debug|refactor|implement|review|rewrite|patch|port)\b.*\b(code|script|function|class|module|snippet|program)\b", 3.0),
        (r"\b(code|script|function|class|module|snippet)\b.*\b(write|fix|debug|refactor|implement|review)\b", 3.0),
        # High-signal coding imperatives — precise verbs + objects.
        (r"\bwrite a (function|class|method|script|decorator|generator|test|wrapper|cli)\b", 3.0),
        (r"\b(implement|add) a (function|class|method|endpoint|feature|tool|command|test)\b", 2.8),
        (r"\b(unit ?tests?|pytest|unittest|test cases?|assert\b|mock\b)\b", 2.2),
        (r"\b(ruff|black|mypy|flake8|pylint|type ?hints?|type ?annotation|lint)\b", 2.0),
        (r"\b(async|await|asyncio|threading|subprocess|dataclass|regex pattern|list comprehension)\b", 1.5),
        (r"\b(debug|fix|refactor|implement|review|rewrite)\b", 1.5),
        # Common Python exceptions / runtime failure signals.
        (r"\b(traceback|stack trace|syntax error|indentationerror|typeerror|nameerror|attributeerror|"
         r"importerror|modulenotfound|keyerror|valueerror|indexerror|recursionerror|zerodivisionerror|"
         r"assertionerror|runtimeerror|segfault|exit code \d|non-?zero exit)\b", 2.5),
        (r"```", 2.0),
        (r"\.(py|sh|ts|js|tsx|jsx|yaml|yml|toml|json|cfg|ini|sql)\b", 1.5),
        (r"\badmin_assistant\.py\b|\bprimus\b.*\b(code|script|router|graph|tool|prompt)\b", 2.0),
    ]

    FORGE_STACK: list[tuple[str, float]] = [
        (r"\b(python|bash|shell|zsh|regex|sql|typescript|javascript|pytest|unittest|dockerfile)\b", 1.2),
        (r"\b(langchain|langgraph|crewai|chromadb|gradio|fastapi|pydantic|ollama|langgraph)\b", 1.5),
        (r"\b(pip install|uv pip|import error|requirements\.txt|pyproject\.toml)\b", 1.8),
        (r"\b(git (diff|commit|rebase|merge)|merge conflict|pull request|\.git)\b", 1.2),
    ]

    FORGE_AUTOMATION: list[tuple[str, float]] = [
        (r"\b(build|create|make|generate)\b.{0,40}\b(script|automation|bot|agent|pipeline|workflow|cron|systemd unit)\b", 3.5),
        (r"\b(automate|automation)\b", 2.5),
        (r"\bhow (?:do i|to|can i)\b.{0,50}\b(implement|build|write|script|automate|deploy|integrate)\b", 3.0),
        (r"\bcan you\b.{0,40}\b(write|fix|debug|implement|build|create|script)\b", 2.5),
        (r"\b(api endpoint|rest api|webhook|data pipeline|etl|schema migration|scraper)\b", 2.0),
        (r"\b(multi-?step|step by step)\b.{0,30}\b(script|workflow|pipeline|deploy)\b", 2.5),
        (r"\b(crewai|langgraph|langchain) (agent|crew|graph|node|tool|chain)\b", 2.5),
        (r"\b(poll(?:ing)?|retry|backoff|rate.?limit|cron job|systemd (service|unit|timer))\b", 1.8),
        (r"\b(integrat\w+|wire up|hook up)\b.{0,30}\b(api|service|tool|model|endpoint)\b", 2.0),
    ]

    FORGE_TERMINAL_COMPLEX: list[tuple[str, float]] = [
        (r"\b(complex|advanced|one-?liner|pipe|awk|sed|find|xargs|jq)\b.{0,40}\b(command|script|bash)\b", 2.0),
        (r"\bgenerate\b.{0,30}\b(terminal|shell|bash|command)\b", 2.5),
        (r"\b(systemd|cron|docker compose|dockerfile|nginx config)\b", 1.8),
    ]

    # --- Primus fast-path (pattern, weight) — high score keeps Forge off ---
    PRIMUS_ADMIN: list[tuple[str, float]] = [
        (r"^/(help|memory|status|kb|recall|learn|clear|tasks|queue|export|setup|primus|forge)\b", 5.0),
        (r"\b(organize downloads|clean temp|disk space|disk usage|free space|du -sh)\b", 3.0),
        (r"\b(wifi|battery|bluetooth|lock screen|system health|cpu|ram usage|uptime)\b", 2.5),
        (r"\b(rclone|backup|onedrive|list remotes)\b", 2.0),
        (r"\b(troubleshoot slow|why is my (system|laptop|machine) slow)\b", 2.5),
        (r"\b(install|apt install|package)\b.{0,30}\b(package|tool)\b", 1.5),
    ]

    PRIMUS_MEMORY: list[tuple[str, float]] = [
        (r"\b(what did we|remind me|recall|remember when|last time we|from memory)\b", 3.5),
        (r"\b(my preference|my business" + _OP_PROJECT_ALT + r")\b", 1.5),
        (r"\b(search knowledge|knowledge base|what do you know about)\b", 2.5),
    ]

    PRIMUS_CHAT: list[tuple[str, float]] = [
        (r"^(hi|hello|hey|thanks|thank you|good morning|good evening)\b", 4.0),
        (r"\b(what can you do|who are you|help me|how do i use primus)\b", 3.0),
        (r"\b(list todos|show todos|show notes|export chat)\b", 3.0),
        # High-confidence non-coding chit-chat — keep it snappy on Primus.
        (r"^(yo|sup|howdy|good ?night|goodnight|gm|gn)\b", 4.0),
        (r"\b(just (saying|chatting|kidding)|never ?mind|nvm|lol|haha|nice|cool)\b", 3.0),
    ]

    PRIMUS_SIMPLE_CMD: list[tuple[str, float]] = [
        (r"^(show|list|check|get)\b.{0,40}\b(status|git status|processes|disk|memory|network)\b", 2.5),
        (r"\bgit status\b", 2.0),
        (r"\bps aux\b|\btop\b|\bdf -h\b", 1.5),
    ]

    @classmethod
    def _score_groups(cls, msg: str, groups: list[tuple[str, float]]) -> tuple[float, list[str]]:
        total = 0.0
        hits: list[str] = []
        for pattern, weight in groups:
            if re.search(pattern, msg, re.I | re.DOTALL):
                total += weight
                hits.append(f"+{weight:.1f} {pattern[:40]}")
        return total, hits

    @classmethod
    def _detect_intent(cls, forge: float, primus: float, msg: str) -> str:
        low = msg.lower()
        # Precise current-data intents first — these must hit a live tool, never drift
        # into unrelated KB/news. Kept on Primus (it owns web_search/get_weather).
        if re.search(
            r"\b(stock|stocks|share price|ticker|nasdaq|s&p|dow|crypto|bitcoin|ethereum|"
            r"price of|exchange rate|earnings|dividend|market(?:s)? (?:today|doing|up|down))\b",
            low,
        ):
            return "finance"
        if re.search(r"\b(weather|forecast|temperature outside|how hot|how cold)\b", low):
            return "weather"
        if re.search(r"\b(latest news|headlines|news (?:about|on)|what'?s happening)\b", low):
            return "news"
        if primus >= 4 and forge < 3:
            if any(re.search(p, low) for p in (r"recall|memory|remember", r"knowledge")):
                return "memory"
            if re.search(r"^/(help|status|kb)", low):
                return "admin"
            if re.search(r"^(hi|hello|thanks)", low):
                return "chat"
            return "admin"
        if re.search(r"\b(automate|automation|pipeline|workflow)\b", low):
            return "automation"
        if forge >= 3:
            return "coding"
        if re.search(r"\b(api|docker|systemd|nginx)\b", low):
            return "technical"
        return "chat"

    @classmethod
    def coding_subtype(cls, msg: str) -> str:
        """Classify a coding request so Forge gets the right planning bias + reasoning trace.

        Returns: new_script | debug | refactor | add_feature | general.
        Order matters: debug (errors) → new_script (write/create) → add_feature → refactor.
        """
        low = msg.lower()
        if re.search(r"\b(traceback|stack trace|error|exception|fails?|failing|broken|bug|crash|"
                     r"doesn'?t work|not working|exit code)\b", low):
            return "debug"
        # New self-contained code: a write/create verb + a concrete code object.
        if re.search(
            r"\b(write|create|generate|make|build|give me|need|code up)\b.{0,45}\b("
            r"script|function|class|method|tool|cli|snippet|program|loop|utility|wrapper|scraper|"
            r"parser|bot|helper|one-?liner|regex|decorator|generator|endpoint|module|test|code)\b",
            low,
        ):
            return "new_script"
        if re.search(r"\b(add|extend|integrate|wire up|hook up|support for)\b.{0,40}"
                     r"\b(feature|tool|command|endpoint|option|to (the|my|primus|forge|existing))\b", low):
            return "add_feature"
        if re.search(r"\b(refactor|clean up|simplify|restructure|deduplicate|tidy)\b"
                     r"|\brename\b.{0,20}\b(function|variable|class|method|module|param)\b", low):
            return "refactor"
        return "general"

    @classmethod
    def _build_reasoning(
        cls,
        *,
        intent: str,
        forge_score: float,
        primus_score: float,
        forge_hits: list[str],
        primus_hits: list[str],
        delegate: bool,
        reason: str,
    ) -> str:
        lines = [
            f"Intent: {intent}",
            f"Scores — Forge {forge_score:.1f} vs Primus {primus_score:.1f}",
        ]
        if forge_hits:
            lines.append("Forge signals: " + "; ".join(forge_hits[:4]))
        if primus_hits:
            lines.append("Primus signals: " + "; ".join(primus_hits[:4]))
        if delegate:
            lines.append(f"→ This is a {intent} task → delegating to Forge")
        else:
            lines.append(f"→ {reason} → staying on Primus (fast path)")
        return "\n".join(lines)

    @classmethod
    def analyze(
        cls,
        message: str,
        *,
        force_forge: bool = False,
        force_primus: bool = False,
    ) -> RoutingDecision:
        """Full routing analysis with reasoning trace."""
        primus_model = _host.CFG.get("model", _host.DEFAULT_MODEL)
        forge_model = _host.CFG.get("forge_model", _host.FORGE_MODEL)
        threshold = float(_host.CFG.get("router_forge_threshold", 3.0))
        strong = float(_host.CFG.get("router_forge_strong_threshold", 5.0))

        msg = message.strip()
        if force_forge:
            return RoutingDecision(
                "forge", forge_model, "manual /forge override", "coding", 1.0, 99.0, 0.0,
                "Intent: coding (forced)\n→ Manual /forge override → delegating to Forge",
            )
        if force_primus:
            return RoutingDecision(
                "primus", primus_model, "primus-only path", "admin", 1.0, 0.0, 99.0,
                "Intent: admin (forced)\n→ Primus-only path",
            )
        if not msg:
            return RoutingDecision(
                "primus", primus_model, "empty message", "chat", 1.0, 0.0, 1.0,
                "Intent: chat\n→ Empty → Primus",
            )
        if not _host.CFG.get("auto_delegate_forge", True):
            return RoutingDecision(
                "primus", primus_model, "auto-delegate disabled", "admin", 1.0, 0.0, 5.0,
                "Intent: admin\n→ auto_delegate_forge=false → Primus",
            )

        forge_score = 0.0
        primus_score = 0.0
        forge_hits: list[str] = []
        primus_hits: list[str] = []

        for groups in (cls.FORGE_CODE, cls.FORGE_STACK, cls.FORGE_AUTOMATION, cls.FORGE_TERMINAL_COMPLEX):
            s, h = cls._score_groups(msg, groups)
            forge_score += s
            forge_hits.extend(h)

        for groups in (cls.PRIMUS_ADMIN, cls.PRIMUS_MEMORY, cls.PRIMUS_CHAT, cls.PRIMUS_SIMPLE_CMD):
            s, h = cls._score_groups(msg, groups)
            primus_score += s
            primus_hits.extend(h)

        # Length heuristics: long technical messages lean Forge; very short lean Primus
        if len(msg) > 120 and forge_score >= 1.5:
            forge_score += 0.8
            forge_hits.append("+0.8 long technical message")
        if len(msg) < 45 and primus_score >= 1.0:
            primus_score += 1.0
            primus_hits.append("+1.0 short admin/chat message")

        intent = cls._detect_intent(forge_score, primus_score, msg)

        # --- Adaptive routing: apply a bounded, feedback-learned nudge for this intent ---
        # Positive nudge favors Forge, negative favors Primus. Learned from 👍/👎 on prior
        # routes (see MetricsTracker.record_feedback). Bounded so it can refine but never
        # override the deterministic rules below.
        try:
            nudge = _host.get_metrics().route_nudge(intent)
        except Exception:  # noqa: BLE001
            nudge = 0.0
        if nudge:
            forge_score += nudge
            (forge_hits if nudge > 0 else primus_hits).append(f"{nudge:+.2f} learned feedback ({intent})")

        # Decision logic — conservative about Forge
        delegate = False
        reason = "general orchestration"

        if forge_score >= strong:
            delegate = True
            reason = f"strong coding/automation signal ({forge_score:.1f} ≥ {strong})"
        elif forge_score >= threshold and forge_score > primus_score + 1.0:
            delegate = True
            reason = f"coding beats admin ({forge_score:.1f} > {primus_score:.1f})"
        elif forge_score >= threshold and primus_score < 2.5:
            delegate = True
            reason = f"technical task, low admin signal ({forge_score:.1f})"
        elif primus_score >= 4.0 and forge_score < threshold + 1.0:
            delegate = False
            reason = f"admin/memory fast-path ({primus_score:.1f})"
        elif intent in ("memory", "chat", "admin") and forge_score < threshold:
            delegate = False
            reason = f"{intent} intent — Primus handles this"
        else:
            delegate = False
            reason = "Primus orchestration (Forge threshold not met)"

        agent_id = "forge" if delegate else "primus"
        model_name = forge_model if delegate else primus_model
        confidence = min(1.0, abs(forge_score - primus_score) / max(forge_score + primus_score, 1.0) + 0.3)
        # Surface the coding sub-type in the trace when Forge is engaged — aids the thinking panel
        # and tells the planner which playbook to use (new script vs debug vs refactor vs feature).
        if delegate and intent in ("coding", "automation", "technical"):
            subtype = cls.coding_subtype(msg)
            if subtype != "general":
                reason = f"{reason} · {subtype.replace('_', ' ')}"
                forge_hits.append(f"subtype: {subtype}")
        reasoning = cls._build_reasoning(
            intent=intent,
            forge_score=forge_score,
            primus_score=primus_score,
            forge_hits=forge_hits,
            primus_hits=primus_hits,
            delegate=delegate,
            reason=reason,
        )
        return RoutingDecision(
            agent_id, model_name, reason, intent, confidence, forge_score, primus_score, reasoning
        )

    @classmethod
    def route(
        cls,
        message: str,
        *,
        force_forge: bool = False,
        force_primus: bool = False,
    ) -> tuple[str, str, str]:
        """Backward-compatible: (agent_id, model_name, reason)."""
        d = cls.analyze(message, force_forge=force_forge, force_primus=force_primus)
        return d.agent_id, d.model_name, d.reason

    @classmethod
    def cacheable(cls, decision: RoutingDecision, message: str) -> bool:
        """Only cache trivial Primus chat — never actions or when memory/KB recall likely needed."""
        if decision.agent_id != "primus":
            return False
        if decision.intent not in ("chat",):
            return False
        # Never cache something that triggers a tool action — it must run every time.
        if is_direct_action(message):
            return False
        if len(message.strip()) > 60:
            return False
        if decision.forge_score >= 1.0:
            return False
        low = message.lower()
        if any(w in low for w in ("recall", "remember", "project", "last time") + _OP_PROJECT_WORDS):
            return False
        # Knowledge-subject questions need retrieval, not a cached/trivial-chat answer.
        if _FAST_CHAT_KB_HINT_RE.search(message):
            return False
        return True


def is_forge_model_ready() -> bool:
    details = _host.get_ollama_details()
    return bool(details.get("reachable") and details.get("forge_ready"))


def _invoke_graph_with_timeout(
    graph: Any, payload: dict, timeout_sec: int, *, poll_sec: float = 0.5
) -> dict:
    """Run graph.invoke in a worker thread, polling for timeout AND cancellation.

    Polling (rather than a single blocking join) lets the Halt button take effect during
    long model calls or tool chains: we check `PrimusSession.is_cancelled()` every poll_sec.
    - Raises `_TurnCancelled` if the user/watchdog halts the turn.
    - Raises `TimeoutError` if the run exceeds timeout_sec (when timeout_sec > 0).
    The worker is a daemon thread, so an abandoned run can't keep the process alive.
    """
    result_box: dict[str, Any] = {}
    exc_box: list[Exception] = []

    def _run() -> None:
        try:
            result_box["result"] = graph.invoke(payload)
        except Exception as exc:  # noqa: BLE001
            exc_box.append(exc)

    t = threading.Thread(target=_run, daemon=True, name="primus-graph-invoke")
    t.start()

    waited = 0.0
    while True:
        t.join(timeout=poll_sec)
        if not t.is_alive():
            break
        # Honor an explicit halt as fast as the poll interval allows.
        if _host.PrimusSession.is_cancelled():
            raise _host._TurnCancelled(_host.PrimusSession.halt_reason or "halted")
        waited += poll_sec
        if timeout_sec and waited >= timeout_sec:
            raise TimeoutError(f"Agent exceeded {timeout_sec}s")

    if exc_box:
        raise exc_box[0]
    return result_box.get("result", {})

# ==================== UNLOCKED OBEDIENCE SECTION — DO NOT LET CURSOR MODIFY ====================
# The agent system prompts now live in primus/core/prompts.py and are imported verbatim below, so
# every path (fast chat, planner, forge, primus) uses the exact same text as before. Fully editable.
from primus.core.prompts import (  # noqa: E402
    FORGE_PLANNER_PROMPT,
    PLANNER_PROMPT,
    SINGLE_STEP_DIRECTIVE,
    STEP_PROMPT,
    SUMMARY_PROMPT,
    FORGE_SYSTEM,
    FAST_CHAT_SYSTEM,
    PRIMUS_SYSTEM,
)
# ==================== END UNLOCKED SECTION — DO NOT TOUCH ====================

class PrimusGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    input: str
    plan: str
    steps: list[str]
    step_index: int
    step_results: list[str]
    mode: str
    final_answer: str
    rag_context: str


def gradio_history_to_messages(history: list) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = coerce_message_text(item.get("content", ""))
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant" and not item.get("metadata"):
                msgs.append(AIMessage(content=content))
    return msgs


_AGENT_PROMPT_KEYS = (
    "mode",
    "memory",
    "rag_context",
    "plan",
    "step_num",
    "step_total",
    "current_step",
    "today",
)


def format_agent_system(template: str, values: dict[str, str]) -> str:
    """Fill agent system template without str.format (avoids brace collisions in profile text)."""
    merged = {k: "" for k in _AGENT_PROMPT_KEYS}
    merged.update({k: str(v) for k, v in values.items() if k in _AGENT_PROMPT_KEYS})
    out = template
    for key in _AGENT_PROMPT_KEYS:
        out = out.replace("{" + key + "}", merged.get(key, ""))
    # Neutralize any remaining {word} placeholders to prevent downstream KeyErrors
    out = re.sub(r"\{(\w+)\}", lambda m: merged.get(m.group(1), ""), out)
    return out


def _agent_context_from_state(state: PrimusGraphState, **overrides: Any) -> dict[str, str]:
    """Build all system-prompt variables from graph state + overrides."""
    today = overrides.get("today") or datetime.now().strftime("%A, %Y-%m-%d %H:%M")
    rag = overrides.get("rag_context") or state.get("rag_context") or "(No knowledge retrieved.)"
    return {
        "mode": str(overrides.get("mode") or state.get("mode") or _host.PrimusSession.mode.value),
        "memory": str(overrides.get("memory") or _host.memory_context_block()),
        "rag_context": str(rag),
        "plan": str(overrides.get("plan") or state.get("plan") or ""),
        "step_num": str(overrides.get("step_num") or "1"),
        "step_total": str(overrides.get("step_total") or "1"),
        "current_step": str(overrides.get("current_step") or "Task"),
        "today": str(today),
    }


_PACK_AGENT_CACHE: dict[tuple, Any] = {}
_PACK_AGENT_LOCK = threading.Lock()


def _packed_step_agent(
    llm: Any,
    pack: str,
    extra: Optional[list[str]],
    agent_id: str,
) -> Any:
    """``create_agent`` with one pack (+ named extras). Never the full 120-schema list."""
    from primus.core.tool_packs import bind_tools, bound_names  # noqa: PLC0415

    extra_t = tuple(extra or ())
    key = (id(llm), pack, extra_t, agent_id)
    with _PACK_AGENT_LOCK:
        cached = _PACK_AGENT_CACHE.get(key)
        if cached is not None:
            return cached
    registry = _host.build_tools()
    bound = bind_tools(registry, pack, extra_t)
    names = bound_names(bound)
    _host.log.info("create_agent pack=%s bound=%s", pack, names)
    try:
        _host.PrimusSession.emit_think(
            "Tool pack",
            f"{pack} · {len(names)}: {', '.join(names)}",
            "done",
        )
    except Exception:  # noqa: BLE001
        pass
    if not bound:
        bound = bind_tools(registry, "default")
        names = bound_names(bound)
        _host.log.info("create_agent pack=%s bound=%s (empty→default)", "default", names)
    agent = create_agent(
        llm,
        bound,
        system_prompt=None,
        middleware=_agent_middleware(bound),
        checkpointer=_get_checkpointer(),
        name=f"{agent_id}-react-{pack}",
    )
    with _PACK_AGENT_LOCK:
        _PACK_AGENT_CACHE[key] = agent
    return agent


def _resolve_invoke_pack(
    *,
    user_content: str,
    current_step: str,
    pack: Optional[str],
    extra_tools: Optional[list[str]],
    allow_named_extra: bool,
) -> tuple[str, list[str], Optional[str]]:
    """Pick the pack for this invoke. user_path may switch and may add one named extra."""
    from primus.core.tool_packs import (  # noqa: PLC0415
        current_extra,
        current_pack,
        extras_from_named,
        select_pack,
    )
    from primus.core.path_mode import current_path_mode  # noqa: PLC0415

    blob = current_step or user_content
    if pack:
        chosen = pack
        extra = list(extra_tools or [])
    elif current_path_mode() == "user_path":
        chosen = select_pack(blob)
        extra, refuse = extras_from_named(blob, chosen)
        if refuse:
            return chosen, [], refuse
    else:
        chosen = current_pack()
        extra = list(extra_tools if extra_tools is not None else current_extra())
    if allow_named_extra and extra_tools:
        extra = list(dict.fromkeys([*(extra or []), *extra_tools]))
    return chosen, extra, None


def invoke_step_agent(
    step_agent: Any,
    *,
    system_template: str,
    state: PrimusGraphState,
    user_content: str,
    thread_id: str,
    llm: Any = None,
    pack: Optional[str] = None,
    extra_tools: Optional[list[str]] = None,
    agent_id: str = "primus",
    **ctx: Any,
) -> dict[str, Any]:
    """Invoke LangChain agent with a fully formatted system prompt (messages-only state).

    Binds one tool pack on ``create_agent`` — never the full ``build_tools()`` list.
    """
    values = _agent_context_from_state(state, **ctx)
    system = format_agent_system(system_template, values)
    prior = list(state.get("messages") or [])
    from primus.core.redact import redact_model_input  # noqa: PLC0415

    system = redact_model_input(system)
    user_content = redact_model_input(user_content)
    prior = [redact_model_input(m) for m in prior]
    messages = [SystemMessage(content=system)] + prior + [HumanMessage(content=user_content)]

    chosen, extra, refuse = _resolve_invoke_pack(
        user_content=user_content,
        current_step=str(ctx.get("current_step") or state.get("input") or ""),
        pack=pack,
        extra_tools=extra_tools,
        allow_named_extra=True,
    )
    if refuse:
        _host.log.info("create_agent refuse pack=%s: %s", chosen, refuse)
        return {"messages": [AIMessage(content=refuse)]}

    agent = step_agent
    if llm is not None:
        agent = _packed_step_agent(llm, chosen, extra, agent_id)
    elif agent is None:
        raise RuntimeError("invoke_step_agent needs llm or a prebuilt step_agent")

    _host.log.debug(
        "Agent invoke thread=%s mode=%s pack=%s rag_chars=%d plan_chars=%d",
        thread_id,
        values["mode"],
        chosen,
        len(values["rag_context"]),
        len(values["plan"]),
    )
    return agent.invoke(
        {"messages": messages},
        config={
            "configurable": {"thread_id": thread_id},
            # Hard cap on react-loop steps: a runaway tool chain exits with a clean
            # GraphRecursionError (caught + retried by the caller) instead of spinning forever.
            "recursion_limit": int(_host.CFG.get("agent_recursion_limit", 30)),
        },
    )


def make_chat_ollama(
    model_name: str,
    *,
    temperature: float = 0.2,
    num_predict: Optional[int] = None,
    fast: bool = True,
) -> Any:
    """Centralized ChatOllama factory with speed + GPU defaults (keep_alive, bounded ctx/output).

    GPU-first: by default we let Ollama auto-offload as many layers as fit on the GPU
    (`gpu_layers = -1` → num_gpu unset → Ollama's own auto-detect, which is ideal for a
    shared-memory iGPU). Setting `prefer_gpu = false` or `gpu_backend = cpu` forces
    `num_gpu = 0` (CPU-only). An explicit positive `gpu_layers` pins that layer count.
    If the GPU can't be used at runtime, Ollama falls back to CPU automatically.
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "base_url": _host.CFG.get("ollama_url", _host.OLLAMA_URL),
        "keep_alive": _host.CFG.get("ollama_keep_alive", "30m"),
    }
    if fast:
        kwargs["num_ctx"] = int(_host.CFG.get("ollama_num_ctx", 4096))
        kwargs["top_k"] = int(_host.CFG.get("ollama_top_k", 30))
        kwargs["top_p"] = float(_host.CFG.get("ollama_top_p", 0.9))
        if num_predict is not None:
            kwargs["num_predict"] = int(num_predict)

    # --- GPU offload policy ---
    prefer_gpu = bool(_host.CFG.get("prefer_gpu", True))
    cpu_only = (_host.gpu_backend_value() == "cpu") or not prefer_gpu
    gpu_layers = int(_host.CFG.get("gpu_layers", -1))
    if cpu_only:
        kwargs["num_gpu"] = 0          # force CPU
    elif gpu_layers >= 0:
        kwargs["num_gpu"] = gpu_layers  # explicit offload count
    # else: gpu_layers < 0 → leave num_gpu unset so Ollama auto-offloads all fitting layers.

    try:
        llm = ChatOllama(**kwargs)
    except TypeError:
        # Older langchain-ollama without one of the kwargs (e.g. num_gpu) — drop the GPU hint
        # and retry, then fall back to the bare essentials. CPU/GPU still resolves server-side.
        kwargs.pop("num_gpu", None)
        try:
            llm = ChatOllama(**kwargs)
        except TypeError:
            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=_host.CFG.get("ollama_url", _host.OLLAMA_URL),
            )
    return _attach_outbound_redact(llm)


def _attach_outbound_redact(llm: Any) -> Any:
    """Redact strings that enter this ChatOllama. Does not touch the operator-facing reply."""
    if llm is None or getattr(llm, "_primus_redact", False):
        return llm
    try:
        from primus.core.redact import redact_model_input  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return llm

    def _wrap(method: Any) -> Any:
        def wrapped(input: Any, *args: Any, **kwargs: Any) -> Any:
            return method(redact_model_input(input), *args, **kwargs)

        return wrapped

    for name in ("invoke", "stream", "_generate", "_stream"):
        fn = getattr(llm, name, None)
        if callable(fn):
            try:
                object.__setattr__(llm, name, _wrap(fn))
            except Exception:  # noqa: BLE001
                try:
                    setattr(llm, name, _wrap(fn))
                except Exception:  # noqa: BLE001
                    pass
    try:
        object.__setattr__(llm, "_primus_redact", True)
    except Exception:  # noqa: BLE001
        try:
            llm._primus_redact = True
        except Exception:  # noqa: BLE001
            pass
    return llm


def _maybe_unwrap_struct_string(s: str) -> Any:
    """If a string is actually a stringified content-block dict/list
    (e.g. "{'text': '…'}" or "[{'type':'text','text':'…'}]"), parse it back.

    These come from models/tools whose content blocks got str()'d, sometimes nested
    many layers deep. Returns the parsed object, or the original string if it's plain text.
    """
    t = s.strip()
    if not t or t[0] not in "{[":
        return s
    # Only attempt parsing when it carries content-block markers — avoids touching
    # legitimate JSON/code the user may have asked for.
    if not any(tok in t for tok in ("'text'", '"text"', "'type'", '"type"', "'content'", '"content"')):
        return s
    import ast

    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(t)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(parsed, str):
            return parsed
    return s


def coerce_message_text(content: Any, _depth: int = 0) -> str:
    """Flatten any LangChain message content into clean human-readable text.

    Models/tools sometimes return content as a list of blocks ([{'type':'text','text':'…'}]),
    a dict, or — worst case — a deeply nested *stringified* dict ({'text': "{'text': …}"}).
    Naive str() would leak {'text': …} into the chat. This recursively extracts the real text.
    """
    if content is None or _depth > 16:
        return "" if content is None else (content if isinstance(content, str) else "")
    if isinstance(content, str):
        unwrapped = _maybe_unwrap_struct_string(content)
        if not isinstance(unwrapped, str):
            return coerce_message_text(unwrapped, _depth + 1)
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "output", "value"):
            if key in content and isinstance(content[key], (str, list, dict)):
                return coerce_message_text(content[key], _depth + 1)
        return ""
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(coerce_message_text(block, _depth + 1))
            elif isinstance(block, dict):
                # Skip non-text blocks (tool_use, images) — only surface readable text.
                if block.get("type") in (None, "text", "output_text"):
                    parts.append(coerce_message_text(block, _depth + 1))
                elif "text" in block:
                    parts.append(coerce_message_text(block, _depth + 1))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p).strip()
    return str(content)


# Debug/prefix artifacts that must never reach the user.
_ARTIFACT_PREFIX_RE = re.compile(
    r"^\s*(?:```(?:json)?\s*)?(?:AI|Assistant|Primus|Forge|System|Human|Tool|Observation|Thought|"
    r"Final Answer)\s*[:>]\s*",
    re.I,
)
_PY_REPR_BLOCK_RE = re.compile(
    r"^\s*\[?\s*\{['\"](?:type|text|content)['\"]\s*:\s*.*\}\s*\]?\s*$",
    re.S,
)

# --- Scaffolding scrubbers: keep internal reasoning / tool plumbing out of the chat ---
# ReAct-style keyword lines some models emit as plain text instead of acting silently.
_REACT_LINE_RE = re.compile(
    r"^[ \t]*(?:Thought|Action(?:\s*Input)?|Observation|Reasoning|Tool(?:\s*Call|\s*Result)?|"
    r"Function(?:\s*Call)?|Invoking(?:\s+tool)?|Calling(?:\s+tool)?)\s*:[ \t].*$",
    re.I | re.M,
)
# Internal multi-step headers produced by the execute loop ("**Step 1:** …"). Targets the bold
# form the graph generates — NOT user-facing tutorials that legitimately use "Step 1:".
_INTERNAL_STEP_RE = re.compile(r"^[ \t]*\*\*Step\s+\d+:\*\*.*$", re.M)
# Plan/step machinery that must never reach the chat window. The progress ledger and resume
# pointer are internal recovery copy — fine in the thinking panel and the log, never a reply.
# Bold "**Step N**" headers are the machine form; a plain "Step 1: …" line is left alone so a
# tutorial the operator actually asked for survives.
_PLAN_CHROME_RE = re.compile(
    r"^[ \t]*\**Progress:\**[ \t]*\d+[ \t]*/[ \t]*\d+.*$"
    # Ledger rows only — the "Step" requirement keeps a real "Remaining: 3 items" line in a
    # report the operator actually asked for.
    r"|^[ \t]*(?:[-*•▶◻◼☐][ \t]*)*(?:In progress|Remaining)[ \t]*:.*\bStep\b.*$"
    r"|^[ \t]*_*Say[ \t]+\**continue\b.*$"
    r"|^[ \t]*_*Partial answer due to timeout.*$"
    r"|^[ \t]*\*\*Step[ \t]+\d+(?:[ \t]*(?:of|/)[ \t]*\d+)?[ \t]*:?\*\*.*$"
    # The executor's own report header ("Step 1 of 1 — done"), which is written for the
    # summarizer, not for the operator. Shaped tightly enough that a "Step 1: install uv" line
    # in a walkthrough they asked for is untouched.
    r"|^[ \t]*\**Step[ \t]+\d+[ \t]*(?:of|/)[ \t]*\d+\**[ \t]*[—:-]?[ \t]*(?:done|blocked)\b.*$",
    re.M | re.I,
)
# Turn-status theater: the machine-readable status object lives in last_task_status.json and the
# thinking panel — never in the chat window. Strip the footer shape and its ledger/resume cousins
# wherever they come from (an old session, a model imitating the pattern, a pasted log).
_STATUS_THEATER_RE = re.compile(
    # Footer: optional hrule line followed by the bold status line. The bold may close after
    # the value (**Status: done**) or right after the label (**Status:** done) — both are the
    # same footer. A "- **Status:** done" list row inside a report the operator asked for is
    # left alone: these alternatives only match at the start of a line.
    r"^[ \t]*-{3,}[ \t]*\n[ \t]*\*\*Status:(?:\s*(?:done|blocked|needs_input)\*\*"
    r"|\*\*[ \t]*(?:done|blocked|needs_input)\b).*$"
    r"|^[ \t]*\*\*Status:(?:\s*(?:done|blocked|needs_input)\*\*"
    r"|\*\*[ \t]*(?:done|blocked|needs_input)\b).*$"
    r"|^[ \t]*Status:\s*(?:done|blocked|needs_input)\b.*$"
    # Debug-object mentions the operator never asked for.
    r"|^[ \t]*.*\b(?:files_touched|last_task_status\.json)\b.*$"
    # Ledger / resume-bait copy.
    r"|^[ \t]*.*\bstill owed\b.*$"
    r"|^[ \t]*(?:Got|Completed)[ \t]+\d+[ \t]+of[ \t]+\d+[ \t]+done\b.*$"
    r"|^[ \t]*.*\btell me to keep going\b.*$",
    re.M | re.I,
)
# Canned shrug lines — from old sessions in chat history, a model parroting them, or a pasted
# log. A turn that produced nothing gets ONE calm sentence (BAD_FORMAT_REPLY / CHAT_TIMEOUT_REPLY
# / pulse_reply), never these. Also the invented Gmail tool names: the real tools are
# gmail_status / gmail_list_messages / gmail_read_message — anything else is mapped or dropped.
_CANNED_SHRUG_RE = re.compile(
    r"^[ \t]*.*\b(?:caught a bad format|ask again in one line|took longer than it should have|"
    r"ask me again and i'?ll answer straight|i'?ll answer directly)\b.*$"
    r"|^[ \t]*.*\b(?:gmail_get_unread_email|gmail_get_latest|get_unread_email)\b.*$",
    re.M | re.I,
)
# Governor / budget lines must survive polish — the UI cap hint is appended after polish,
# but a graph reply that *is* the cap/loop message must not be stripped to BAD_FORMAT_REPLY.
_BRAIN_KEEP_RE = re.compile(
    r"(?im)^[ \t]*.*(?:"
    r"Brain weekly budget reached"
    r"|Brain per-turn call limit reached"
    r"|I kept circling on this"
    r"|Primus used \d+ brain calls and is stuck"
    r").*$"
)


def _strip_invented_tool_lines(text: str) -> str:
    """Drop lines that are ONLY a snake_case identifier naming no real tool — the model printing
    a (usually invented) tool name as if it were the answer. Real tool names survive; prose
    survives; fenced code blocks are never touched (checked by the caller).
    """
    known = _known_tool_names()
    if not known:
        return text  # can't tell invented from real — leave the text alone

    def _drop(m: "re.Match[str]") -> str:
        name = m.group(0).strip().strip("`").removesuffix("()")
        return m.group(0) if name in known else ""

    return re.sub(r"(?m)^[ \t]*`?[a-z][a-z0-9]*(?:_[a-z0-9]+)+`?(?:\(\))?[ \t]*$", _drop, text)

# Addressing the operator as Primus ("How about you, Primus?", "…for you, Primus.") — an identity
# slip. polish_response replaces the whole reply with IDENTITY_ACK deterministically; the
# clause-strip in _strip_tool_scaffolding stays as defense-in-depth for any other caller.
_ADDRESSED_AS_PRIMUS_RE = re.compile(
    r"(?:^|(?<=[.!?\n]))[^.!?\n]*\b(?:how about|what about|and)\s+you\s*,?\s*primus\b[^.!?\n]*[.!?]?"
    r"|\byou\s*,\s*primus\b",
    re.I,
)
# Claiming the operator is the agent, or Primus speaking as the operator. Not a slip — the
# identity is inverted, so the whole reply is replaced deterministically (no LLM round-trip).
_OP_INV_NAME = (
    re.escape(_OPERATOR["name"].split()[0].lower()) if _OPERATOR["custom"] else r"the\s+operator"
)
_IDENTITY_INVERSION_RE = re.compile(
    r"\byou(?:'re| are)\s+primus\b|\byour name is\s+primus\b"
    rf"|\bi(?:'m| am)\s+{_OP_INV_NAME}\b"
    rf"|\bmy name is\s+{_OP_INV_NAME}\b"
    r"|\bi(?:'m| am)\s+your\s+(?:operator|boss|owner|master)\b",
    re.I,
)
_OP_CLAIM_NAMES = [
    re.escape(_OPERATOR["name"]) if _OPERATOR["custom"] else r"the\s+operator",
]
if _OPERATOR["custom"]:
    _op_full_claim = str(_OPERATOR.get("full_name") or "").strip()
    if _op_full_claim and _op_full_claim.lower() != str(_OPERATOR["name"]).lower():
        _OP_CLAIM_NAMES.append(re.escape(_op_full_claim))
_OP_CLAIM_ALT = "|".join(_OP_CLAIM_NAMES)
_OPERATOR_CLAIM_RE = re.compile(
    rf"(?im)^(?:{_OP_CLAIM_ALT})\s+here\b[^\n]*\n?"
    rf"|^i(?:'m| am)\s+(?:{_OP_CLAIM_ALT})\b[^\n]*\n?"
)
# Shown when stripping machinery leaves nothing behind — better than an empty bubble, and it
# never tells the operator to type "continue".
# Used ONLY when a reply is empty after stripping AND no tool produced content — one calm
# sentence, never "caught a bad format", never "ask again".
BAD_FORMAT_REPLY = "I came up empty on that one — nothing was run, changed, or sent."
# Raw tool-call JSON the model "speaks" instead of invoking ({"name": "list_tools", ...}).
# Matches a flat object or one level of nesting (the typical arguments/parameters block).
_TOOL_CALL_JSON_RE = re.compile(
    r"\{[^{}]*?['\"](?:name|tool|function|action)['\"]\s*:\s*['\"][\w.\-]+['\"]"
    r"(?:[^{}]*\{[^{}]*\})?[^{}]*\}",
    re.S,
)
_TOOL_NAME_CACHE: Optional[set[str]] = None


def _known_tool_names() -> set[str]:
    """Real tool names (cached) — used to recognize leaked tool-call JSON without nuking the
    user's own JSON (e.g. {"name": "Ada", "age": 28} must survive)."""
    global _TOOL_NAME_CACHE
    if _TOOL_NAME_CACHE is None:
        try:
            _TOOL_NAME_CACHE = {getattr(t, "name", "") for t in _host.build_tools()} - {""}
        except Exception:  # noqa: BLE001
            _TOOL_NAME_CACHE = set()
    return _TOOL_NAME_CACHE


# Keys that, taken together, mean a JSON object IS a tool call (not user data). If an object's
# keys are a subset of these, it's scaffolding even when the tool name is one the model invented.
_TOOLCALL_KEYS = frozenset({
    "name", "tool", "function", "action", "arguments", "args", "parameters",
    "params", "tool_input", "input", "type", "id", "tool_call_id",
})


def _strip_tool_scaffolding(text: str) -> str:
    """Remove leaked tool-call JSON, ReAct keyword lines, and internal step headers.

    Conservative by design: a JSON object is only stripped when it clearly IS a tool call — its
    keys are all tool-call keys (e.g. {"name": "list_tools"}), it carries an arguments block, or its
    name matches a real Primus tool. Genuine JSON the user asked for ({"name": "Ada", "age": 28})
    has extra data keys, so it's always preserved.
    """
    if not text:
        return text
    import ast

    out = _REACT_LINE_RE.sub("", text)
    out = _INTERNAL_STEP_RE.sub("", out)
    out = _PLAN_CHROME_RE.sub("", out)
    out = _STATUS_THEATER_RE.sub("", out)
    out = _CANNED_SHRUG_RE.sub("", out)
    out = _ADDRESSED_AS_PRIMUS_RE.sub("", out)
    if "```" not in out:
        # A bare snake_case line naming no real tool is the model printing a tool name as the
        # answer — never content. Skipped inside fenced code (real code has fences).
        out = _strip_invented_tool_lines(out)

    if any(k in out for k in ('"name"', "'name'", '"tool"', "'tool'",
                              '"function"', "'function'", '"action"', "'action'")):
        tool_names = _known_tool_names()

        def _maybe_strip(m: "re.Match[str]") -> str:
            blob = m.group(0)
            name_m = re.search(
                r"['\"](?:name|tool|function|action)['\"]\s*:\s*['\"]([\w.\-]+)['\"]", blob
            )
            if not name_m:
                return blob
            parsed: Any = None
            for parser in (ast.literal_eval, json.loads):
                try:
                    parsed = parser(blob)
                    break
                except Exception:  # noqa: BLE001
                    continue
            if isinstance(parsed, dict) and {str(k).lower() for k in parsed} <= _TOOLCALL_KEYS:
                return ""  # pure tool-call shape — drop it
            has_args = re.search(
                r"['\"](?:arguments|args|parameters|params|tool_input|input)['\"]\s*:", blob
            )
            if has_args or name_m.group(1) in tool_names:
                return ""  # leaked tool call (carries args, or names a real tool)
            return blob

        out = _TOOL_CALL_JSON_RE.sub(_maybe_strip, out)
        # Remove code fences left empty after their tool-call payload was stripped.
        out = re.sub(r"```(?:json|tool_call|tool_code|python)?[ \t]*\n?\s*```", "", out, flags=re.I)
    return out


# Robotic closings/filler to strip — concise, human, no-fluff replies only.
_ROBOTIC_CLOSING_RE = re.compile(
    r"(?:^|(?<=[.!?\n]))\s*("
    r"(?:please\s+)?(?:do\s+)?(?:feel free|don'?t hesitate)\s+to\s+(?:ask|reach out|let me know)[^.!?\n]*[.!?]?|"
    r"(?:please\s+)?let me know\s+(?:if|how|whenever|should)[^.!?\n]*[.!?]?|"
    r"is there (?:anything|something) else[^?\n]*\??|"
    r"if you (?:have|need)\s+(?:any|more|further)?[^.!?\n]*(?:question|help|assist|need)[^.!?\n]*[.!?]?|"
    r"i(?:'m| am)\s+(?:here|happy|glad)\s+to\s+(?:help|assist)[^.!?\n]*[.!?]?|"
    r"(?:i\s+)?hope (?:this|that|it)\s+helps[^.!?\n]*[.!?]?|"
    r"happy to help[^.!?\n]*[.!?]?|"
    r"glad (?:i could|to) (?:help|assist)[^.!?\n]*[.!?]?"
    r")\s*$",
    re.I,
)
# Filler openers ("Certainly!", "Of course!", "Great question!", …) at the very start.
_FILLER_OPENER_RE = re.compile(
    r"^\s*(certainly|sure(?: thing)?|of course|absolutely|great question|good question|"
    r"no problem|got it|understood|i'?d be happy to[^.!?\n]*|happy to[^.!?\n]*)[!.,:]*\s+",
    re.I,
)
# Abrupt robotic closings ("Task complete.", "Done.", "Finished.", "All done.", …) when they
# stand alone as the LAST sentence. Anchored so a natural sentence like "The migration is now
# complete." is preserved — only the clipped machine-style sign-off is removed.
_ABRUPT_CLOSING_RE = re.compile(
    r"(?:(?<=[.!?\n])|^)\s*"
    r"(?:task\s+(?:is\s+)?(?:now\s+)?(?:complete|completed|done|finished)|"
    r"complete|completed|done|finished|all done|all set|that'?s (?:it|all|done))"
    r"[.!]*\s*$",
    re.I,
)


def _strip_robotic(text: str) -> str:
    """Remove robotic openers, trailing filler sign-offs, and abrupt machine-style closings."""
    out = text.strip()
    # Strip up to two stacked filler openers (keep the substantive remainder).
    for _ in range(2):
        nxt = _FILLER_OPENER_RE.sub("", out, count=1).strip()
        if nxt == out or not nxt:
            break
        out = nxt
    # Strip up to three trailing filler closings + abrupt robotic sign-offs, but only when
    # there is substantive content before them (never blank out a one-word reply).
    for _ in range(3):
        nxt = _ROBOTIC_CLOSING_RE.sub("", out).strip()
        nxt2 = _ABRUPT_CLOSING_RE.sub("", nxt).strip()
        if nxt2 and nxt2 != out:
            out = nxt2
            continue
        out = nxt if (nxt and nxt != out) else out
        break
    return out.strip()


# Obsolete/manual Gmail-auth guidance the model must never emit — Primus uses OAuth via its own tools.
_OBSOLETE_GMAIL_AUTH_RE = re.compile(
    r"app[\s-]?password|less[\s-]?secure app|allow less secure|myaccount\.google\.com|"
    r"account\.google\.com/apppasswords",
    re.IGNORECASE,
)
# Flat refusal about email access (only corrected when it's a dead-end, not an actionable step).
_EMAIL_REFUSAL_RE = re.compile(
    r"\b(?:can'?t|cannot|can not|unable to|don'?t have access to|do not have access to|not able to)\b"
    r"[^.\n]{0,40}\b(?:access|read|open|check|connect to|see)\b[^.\n]{0,30}\b(?:e-?mail|gmail|inbox)\b",
    re.IGNORECASE,
)
_GMAIL_AUTH_CTA = (
    "\n\n→ Actually, I can connect this directly — run `ensure_gmail_access` and I'll link your Gmail "
    "(no App Passwords or manual Google settings needed)."
)


def _redirect_gmail_auth_guidance(out: str) -> str:
    """Convert obsolete manual-auth guidance / flat email refusals into an actionable tool step.

    Safety net for when a reply slips past the system-prompt steering: it only APPENDS a concrete
    next action (never removes content), and only when the text shows obsolete Gmail auth advice
    (App Passwords / "less secure apps") or a dead-end "can't access your email" AND doesn't already
    point at the right tool. Backward-compatible: no-op for every other response.
    """
    if not out or "ensure_gmail_access" in out or "gmail_auth" in out:
        return out
    if _OBSOLETE_GMAIL_AUTH_RE.search(out) or _EMAIL_REFUSAL_RE.search(out):
        return out + _GMAIL_AUTH_CTA
    return out


def polish_response(text: Any) -> str:
    """Final post-processing layer — strip technical artifacts, robotic filler, normalize.

    Guarantees the chat only ever shows clean, human-readable text in the operator's preferred
    concise, no-fluff voice.
    """
    out = coerce_message_text(text).strip()
    if not out:
        return out

    # Re-coerce until stable — defends against deeply nested stringified content blocks.
    for _ in range(6):
        nxt = coerce_message_text(out).strip()
        if nxt == out:
            break
        out = nxt

    # Last-resort: if residual content-block markers remain, pull out readable text spans.
    if out and out[0] in "{[" and any(tok in out for tok in ("'text'", '"text"', "'type'", '"type"')):
        spans = re.findall(r"['\"]text['\"]\s*:\s*['\"](.*?)['\"]", out, re.S)
        if spans:
            recovered = max(spans, key=len)
            try:
                recovered = recovered.encode().decode("unicode_escape")
            except Exception:  # noqa: BLE001
                pass
            if recovered.strip():
                out = recovered.strip()

    # Identity lock, before anything else can cosmetically hide it: if the reply claims the operator
    # is the agent (or speaks as the operator), the content is wrong, not just badly worded. Replace
    # it outright — deterministically, no second model call. Same when the reply addresses the
    # operator AS Primus ("How about you, Primus?") — a reply that confused about who is who gets
    # the canonical identity line, not a cosmetic trim.
    if _IDENTITY_INVERSION_RE.search(out) or _ADDRESSED_AS_PRIMUS_RE.search(out):
        return IDENTITY_ACK
    claimed = _OPERATOR_CLAIM_RE.sub("", out).strip()
    if claimed:
        out = claimed

    # Scrub leaked scaffolding FIRST (whole ReAct/tool lines), so the prefix pass below can't
    # orphan their text (e.g. turning "Thought: list them" into a bare "list them").
    keep_brain = [ln for ln in out.splitlines() if _BRAIN_KEEP_RE.search(ln)]
    had_content = bool(out)
    out = _strip_tool_scaffolding(out).strip()
    if had_content and not out:
        if keep_brain:
            return "\n\n".join(ln.strip() for ln in keep_brain if ln.strip())
        return BAD_FORMAT_REPLY

    # Drop leading role/debug prefixes ("AI:", "Final Answer:", "Assistant>").
    prev = None
    while prev != out:
        prev = out
        out = _ARTIFACT_PREFIX_RE.sub("", out, count=1).strip()

    # Strip robotic openers/closings unless the whole reply IS a short clarifying question.
    if not (out.endswith("?") and len(out) < 120):
        out = _strip_robotic(out) or out

    # Collapse 3+ blank lines and trailing spaces.
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = out.strip()

    # Anti-refusal safety net: turn obsolete manual Gmail-auth advice / dead-end email refusals into
    # a concrete tool step (append-only; no-op for everything else).
    out = _redirect_gmail_auth_guidance(out)
    out = out.strip()
    for ln in keep_brain:
        if ln.strip() and ln.strip() not in out:
            out = f"{out}\n\n{ln.strip()}".strip() if out else ln.strip()
    # Machinery was all there was: say so plainly rather than showing an empty reply.
    return out if out or not had_content else BAD_FORMAT_REPLY


def _forge_shape_plan(steps: list[str]) -> list[str]:
    """Keep Forge plans focused (≤ forge_max_plan_steps) and guarantee a final verification step.

    This is what makes the 7B coder punch above its weight: a tight plan + an explicit verify
    step (verify_python / quick run) closes the loop so generated code is checked, not assumed.
    """
    max_steps = max(1, int(_host.CFG.get("forge_max_plan_steps", 4)))
    steps = [s for s in steps if s.strip()][:max_steps]
    if bool(_host.CFG.get("coding_verify_on_generate", True)) and steps:
        joined = " ".join(steps).lower()
        if not re.search(
            r"\b(verify|verify_python|lint|ruff|black|test|pytest|compile|py_compile|run the|run it)\b",
            joined,
        ):
            if len(steps) >= max_steps:
                steps = steps[: max_steps - 1]  # make room for the verify step
            steps.append(
                "Verify the result: run verify_python on the code (and a quick test/run if "
                "applicable), then fix anything it reports."
            )
    return steps or ["Complete the coding task."]


# Signals that a step DESCRIBED / offered / hallucinated a path instead of actually completing.
# Used by the completion nudge to force one real attempt before accepting a non-answer.
_PLACEHOLDER_PATH_RE = re.compile(
    r"/path/to/|/path/to/your|<[^>\n]{0,30}(?:path|file|dir)[^>\n]{0,30}>|"
    r"\byour[-_/ ]?(?:file|pdf|document)[-_/ ]?(?:path|name)\b|/\.\.\./|/…/",
    re.IGNORECASE,
)
_OFFER_RE = re.compile(
    r"(I can continue if you|let me know if you.?d like|would you like me to|shall I proceed|"
    r"do you want me to|if you.?d like,?\s+I can|I would (?:now |then )?(?:proceed|run|call|use)|"
    r"I'?ll go ahead and|next,? I would)",
    re.IGNORECASE,
)


# Handing the work back to the operator — "I'd need the contents of that file", "please share the
# output", "if you can provide the file". Primus has read_file and search_files; asking the
# operator to paste their own file is a non-answer, and it comes with invented guesses about the content.
_HANDBACK_RE = re.compile(
    r"\b(?:i(?:'d| would)? (?:need|require)(?: to (?:see|read|inspect|view|access|have))?"
    r"|(?:i )?don'?t have|without) (?:direct )?"
    r"(?:access to |the )?(?:actual |specific |exact )?(?:contents?|texts?|files?|outputs?|"
    r"details?|code|data)\b"
    r"|\bwithout (?:direct )?access to\b"
    r"|\b(?:if you (?:can|could)|please|you could) (?:provide|share|paste|give me|send me)\b"
    r"|\b(?:share|paste|provide) (?:the|its|those|these) (?:content|contents|file|output|code)\b",
    re.IGNORECASE,
)


def _looks_incomplete(text: str) -> bool:
    """Heuristic: did a step merely describe/offer, or emit a placeholder path, instead of finishing?

    Conservative on purpose — placeholder paths always flag (a fake path is never a real result);
    'offer to continue' phrasing only flags on short outputs, so we never re-run substantial work.
    """
    t = (text or "").strip()
    if not t:
        return True
    # A placeholder path OR an example/placeholder URL is never a real deliverable — always flag.
    if _PLACEHOLDER_PATH_RE.search(t) or _has_placeholder_url(t):
        return True
    # Asking the operator to hand over something Primus can fetch itself is never a finished step,
    # however long the answer is — length here is padding, not work.
    if _HANDBACK_RE.search(t):
        return True
    return bool(_OFFER_RE.search(t)) and len(t) < 1000


# Characters that legitimately end a finished message (sentence punctuation, closing quotes/brackets).
_SENTENCE_TERMINATORS = ('.', '!', '?', '…', '"', "'", '`', ')', ']', '}', '»', '”', '’', ':')


def _looks_truncated(text: str) -> bool:
    """Heuristic: was the answer cut off mid-thought (model hit its token cap)?

    Deliberately conservative to avoid false positives: only long-ish outputs are considered, and
    lines that legitimately end without sentence punctuation (code fences, tables, list items,
    headings, numbers) are treated as complete. An UNBALANCED code fence always flags.
    """
    t = (text or "").rstrip()
    if not t:
        return False
    if t.count("```") % 2 == 1:  # an opened code block was never closed → definitely truncated
        return True
    if len(t) < 200:  # short replies (acks, quick answers, clarifying questions) are intentional
        return False
    last = t.splitlines()[-1].strip()
    if not last:
        return False
    # Structural lines that don't need sentence punctuation to be "done".
    if last.endswith(("```", ":", "|")) or last[:1] in ("|", "-", "*", "#", ">") or last[-1:].isdigit():
        return False
    return not t.endswith(_SENTENCE_TERMINATORS)


def _complete_if_truncated(llm: Any, base_messages: list, answer: str, *, max_rounds: int = 2) -> str:
    """If `answer` looks cut off, ask the model to continue from where it stopped and append.

    Bounded (≤ max_rounds) and cancellation-aware; each round appends only genuinely new text.
    Guarded by CFG['complete_truncated_answers'] (default on). Never raises — returns best effort.
    """
    if not bool(_host.CFG.get("complete_truncated_answers", True)):
        return answer
    rounds = 0
    while _looks_truncated(answer) and rounds < max_rounds and not _host.PrimusSession.is_cancelled():
        rounds += 1
        _host.PrimusSession.emit_think("Summary", f"Answer looked cut off — continuing ({rounds})", "running")
        try:
            cont = llm.invoke(
                base_messages
                + [
                    AIMessage(content=answer),
                    HumanMessage(content=(
                        "Your previous message was cut off. Continue from EXACTLY where you stopped — "
                        "do NOT repeat any text already written, just append the remainder and finish "
                        "the thought. If it was actually already complete, reply with only: DONE"
                    )),
                ]
            )
            extra = coerce_message_text(cont.content).strip()
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("truncation continuation failed: %s", exc)
            break
        if not extra or extra.upper().rstrip(".") == "DONE":
            break
        joiner = "" if answer.endswith((" ", "\n")) or extra.startswith((" ", "\n")) else " "
        answer = (answer + joiner + extra).strip()
    return answer


def _registry_mod() -> Any:
    """The loaded tools registry module (for the filesystem audit trail). Never raises."""
    return _sys.modules.get("primus.tools.registry")


def _reset_turn_file_tracking() -> None:
    try:
        reg = _registry_mod()
        if reg is not None:
            reg._fs_turn_reset()
    except Exception:  # noqa: BLE001
        pass


def _finalize_turn_status(state: "PrimusGraphState", answer: str, agent_id: str) -> str:
    """Persist the end-of-turn status object to ~/.primus/last_task_status.json (debug aid)
    and the thinking panel. The CHAT NEVER sees it — no footer, no JSON, no status line.

    status: done | blocked | needs_input
      - blocked: a step reported failure/blocked, the no-progress stop fired, or the agent errored
      - needs_input: the reply ends in a direct question to the operator (clarifier)
      - done: otherwise — tool results already verified the work on disk
    files_touched comes from the filesystem audit trail, so it only lists REAL mutations.
    Returns the answer unchanged. Never raises.
    """
    try:
        reg = _registry_mod()
        files = reg._fs_turn_files() if reg is not None else []
        results = state.get("step_results") or []
        blob = "\n".join(str(r) for r in results) + "\n" + answer
        low = blob.lower()
        if ("**agent error**" in blob or "could not complete" in low
                or "blocked:" in low or "no-progress stop" in low):
            status = "blocked"
        elif answer.rstrip().endswith("?"):
            status = "needs_input"
        else:
            status = "done"
        errors = [str(r)[:200] for r in results
                  if "could not complete" in str(r).lower() or "blocked" in str(r).lower()]
        obj = {
            "status": status,
            "summary": answer[:600],
            "files_touched": files,
            "errors": errors,
            "task": str(state.get("input", ""))[:300],
            "agent": agent_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            _host._atomic_write_text(
                _host.APP_DIR / "last_task_status.json",
                json.dumps(obj, indent=2, ensure_ascii=False),
            )
        except Exception:  # noqa: BLE001
            pass
        _host.PrimusSession.emit_think(
            "Status", f"{status} · {len(files)} file(s) touched",
            "done" if status == "done" else "error",
        )
    except Exception:  # noqa: BLE001
        pass
    return answer


def build_agent_graph(
    model_name: str,
    system_prompt: str,
    *,
    agent_id: str = "primus",
    planner_prompt: Optional[str] = None,
) -> Any:
    """Build LangGraph plan→execute loop for Primus or Forge."""
    temp_key = "forge_temperature" if agent_id == "forge" else "primus_temperature"
    temperature = float(_host.CFG.get(temp_key, 0.2 if agent_id == "forge" else 0.25))
    num_predict = int(
        _host.CFG.get("forge_num_predict", 2048) if agent_id == "forge" else _host.CFG.get("primus_num_predict", 768)
    )
    planner_prompt = planner_prompt or (
        FORGE_PLANNER_PROMPT if agent_id == "forge" else PLANNER_PROMPT
    )
    llm = make_chat_ollama(model_name, temperature=temperature, num_predict=num_predict)
    # Registry stays 120. Inner create_agent binds one pack in invoke_step_agent.
    _ = _host.build_tools()

    def plan_node(state: PrimusGraphState) -> dict:
        user_input = state["input"]
        _reset_turn_file_tracking()  # fresh audit window for this turn's status object

        # Bail out immediately if the turn was halted before planning even started.
        if _host.PrimusSession.is_cancelled():
            _host.PrimusSession.emit_think("Halted", "Stopped before planning", "error")
            return {"plan": "halted", "steps": [], "step_index": 0, "step_results": []}

        # --- Direct-action fast path: simple, tool-mappable request → execute now ---
        if is_direct_action(user_input):
            _host.PrimusSession.emit_think(
                "Direct action",
                "Simple request — executing immediately (no planning).",
                "done",
            )
            return {
                "plan": f"Direct execution: {user_input}",
                "steps": [user_input],
                "step_index": 0,
                "step_results": [],
            }

        try:
            _host.PrimusSession.emit_think("Planning", f"{agent_id} analyzing…", "running")
            prior = state.get("messages") or []
            ctx = "\n".join(f"{m.type}: {str(m.content)[:200]}" for m in prior[-4:])
            resp = llm.invoke(
                [
                    SystemMessage(content=planner_prompt),
                    HumanMessage(content=f"Context:\n{ctx}\n\nTask:\n{user_input}"),
                ]
            )
            plan = str(resp.content)
            steps = _host.parse_plan_steps(plan)
            # Forge: cap to a focused plan and ensure a verification step closes the loop.
            if agent_id == "forge":
                steps = _forge_shape_plan(steps)
            _host.save_plan(user_input, plan)
            _host.PrimusSession.emit_think("Planning", plan[:400], "done")
            for i, s in enumerate(steps, 1):
                _host.PrimusSession.emit_think(f"Step {i}", s, "pending")
            return {"plan": plan, "steps": steps, "step_index": 0, "step_results": []}
        except Exception as exc:
            _host.log.exception("Plan node failed (%s)", agent_id)
            _host.PrimusSession.emit_think("Planning", str(exc)[:200], "error")
            return {"plan": "Single-step fallback.", "steps": [state["input"]], "step_index": 0, "step_results": []}

    def execute_step_node(state: PrimusGraphState) -> dict:
        idx = state.get("step_index", 0)
        steps = state.get("steps") or ["Complete task"]
        if idx >= len(steps):
            return {}
        if _host.PrimusSession.is_cancelled():
            _host.PrimusSession.emit_think("Halted", "Stopped before next step", "error")
            results = list(state.get("step_results") or [])
            results.append("_(Halted by user before completing all steps.)_")
            return {"step_results": results, "step_index": len(steps)}

        current = steps[idx]
        _host.PrimusSession.emit_think(f"Step {idx + 1}/{len(steps)}", current, "running")
        mem = _host.memory_context_block()
        today = datetime.now().strftime("%A, %Y-%m-%d %H:%M")
        rag = state.get("rag_context") or "(No knowledge retrieved.)"
        task = f"Execute step {idx + 1}/{len(steps)}: {current}\nOriginal task: {state['input']}\n\n{STEP_PROMPT}"
        thread = f"{agent_id}-step-{idx}"

        def _attempt(content: str, suffix: str) -> str:
            result = invoke_step_agent(
                None,
                system_template=system_prompt,
                state=state,
                user_content=content,
                thread_id=f"{thread}{suffix}",
                llm=llm,
                agent_id=agent_id,
                mode=state.get("mode", _host.PrimusSession.mode.value),
                memory=mem,
                rag_context=rag,
                plan=state.get("plan", ""),
                step_num=str(idx + 1),
                step_total=str(len(steps)),
                current_step=current,
                today=today,
            )
            return coerce_message_text(result["messages"][-1].content)

        try:
            out = _attempt(task, "")
        except Exception as exc:
            _host.log.warning("Step %s first attempt failed (%s): %s", idx + 1, agent_id, exc)
            _host.PrimusSession.emit_think(f"Step {idx + 1}", f"Recovering from: {str(exc)[:120]}", "running")
            try:
                out = _attempt(
                    f"Retry step {idx + 1}: {current}\nKeep it simple and use the most direct tool.",
                    "-retry",
                )
                _host.PrimusSession.emit_think(f"Step {idx + 1}", "Recovered on retry", "done")
            except Exception as exc2:
                _host.log.exception("Step %s failed after retry (%s)", idx + 1, agent_id)
                out = f"Step could not complete ({exc2}). Continuing with remaining steps…"
                _host.PrimusSession.emit_think(f"Step {idx + 1}", str(exc2)[:200], "error")

        # Completion nudge: if the step only described/offered or used a placeholder path, force ONE
        # real attempt with the correct tool + real paths (config-gated, single retry, never loops).
        if (
            bool(_host.CFG.get("enforce_step_completion", True))
            and not _host.PrimusSession.is_cancelled()
            and _looks_incomplete(out)
        ):
            _host.PrimusSession.emit_think(
                f"Step {idx + 1}", "Output looks incomplete — enforcing real action", "running"
            )
            try:
                out = _attempt(
                    f"Step {idx + 1} is NOT done. Do not describe, offer, or use placeholder paths — "
                    f"actually perform it NOW with the most specific correct tool and REAL paths "
                    f"(e.g. ~/.primus/knowledge/downloads/). Step: {current}",
                    "-force",
                )
            except Exception:  # noqa: BLE001 — nudge is best-effort; keep the original output
                pass
        results = list(state.get("step_results") or [])
        results.append(f"**Step {idx + 1}:** {current}\n{out}")
        _host.PrimusSession.emit_think(f"Step {idx + 1}/{len(steps)}", out[:300], "done")
        return {"step_results": results, "step_index": idx + 1, "messages": [AIMessage(content=out)]}

    def summarize_node(state: PrimusGraphState) -> dict:
        results = state.get("step_results") or []
        body = "\n\n".join(results)

        # Fast path: short results don't need an extra LLM round-trip — present them directly.
        skip_limit = int(_host.CFG.get("skip_summary_max_chars", 900))
        if len(body) <= skip_limit:
            _host.PrimusSession.emit_think("Summary", "Short result — returning directly (fast path).", "done")
            answer = body or "That's handled — nothing else needed on this one."
            answer = _finalize_turn_status(state, answer, agent_id)
            _host.log_past_task(state["input"], answer[:500])
            return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

        # Skip the extra summary round-trip if the user halted — hand back the raw results.
        if _host.PrimusSession.is_cancelled():
            _host.PrimusSession.emit_think("Halted", "Returning results without summary", "error")
            answer = body or "_(Halted before completion.)_"
            return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

        _host.PrimusSession.emit_think("Summary", "Composing final response…", "running")
        # Give the final synthesis extra token headroom so long answers aren't clipped at the
        # step cap, then a continuation safety-net catches any remaining mid-sentence cut-off.
        summary_cap = max(num_predict, int(_host.CFG.get("summary_num_predict", 1536)))
        summary_msgs = [
            SystemMessage(content=SUMMARY_PROMPT),
            HumanMessage(content=f"Agent: {agent_id}\nTask: {state['input']}\n\nStep results:\n{body}"),
        ]
        try:
            summary_llm = make_chat_ollama(model_name, temperature=temperature, num_predict=summary_cap)
            resp = summary_llm.invoke(summary_msgs)
            answer = coerce_message_text(resp.content)
            answer = _complete_if_truncated(summary_llm, summary_msgs, answer)
        except Exception as exc:
            _host.log.warning("Summary LLM failed (%s) — returning raw results", exc)
            answer = body
        answer = _finalize_turn_status(state, answer, agent_id)
        _host.log_past_task(state["input"], answer[:500])
        _host.PrimusSession.emit_think("Summary", "Done", "done")
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    def route_after_plan(state: PrimusGraphState) -> str:
        steps = state.get("steps") or []
        return "multi_step" if len(steps) > 1 else "single_step"

    def route_after_step(state: PrimusGraphState) -> str:
        idx = state.get("step_index", 0)
        steps = state.get("steps") or []
        if idx < len(steps):
            return "continue"
        return "summarize"

    def single_step_node(state: PrimusGraphState) -> dict:
        if _host.PrimusSession.is_cancelled():
            _host.PrimusSession.emit_think("Halted", "Stopped before execution", "error")
            return {"final_answer": "⏹ Halted before I started — nothing was run."}
        _host.PrimusSession.emit_think("Executing", state["input"][:120], "running")
        mem = _host.memory_context_block()
        today = datetime.now().strftime("%A, %Y-%m-%d %H:%M")
        rag = state.get("rag_context") or "(No knowledge retrieved.)"
        current = (state.get("steps") or ["Task"])[0]
        # Single-step turns used to run on the bare user message, so no execution discipline
        # reached the path that answers most requests — hence "I'd need the contents of that
        # file" on a file Primus can read. STEP_PROMPT is the wrong fit here (it asks for a
        # "Step X of Y — done" report for the summarizer to consume, and this path has no
        # summarizer), so single-step gets the same tool discipline without the step protocol.
        task = f"{state['input']}\n\n{SINGLE_STEP_DIRECTIVE}"

        def _attempt(suffix: str, content: str = "") -> str:
            result = invoke_step_agent(
                None,
                system_template=system_prompt,
                state=state,
                user_content=content or task,
                thread_id=f"{agent_id}-single{suffix}",
                llm=llm,
                agent_id=agent_id,
                mode=state.get("mode", _host.PrimusSession.mode.value),
                memory=mem,
                rag_context=rag,
                plan=state.get("plan", ""),
                step_num="1",
                step_total="1",
                current_step=current,
                today=today,
            )
            return coerce_message_text(result["messages"][-1].content)

        try:
            answer = _attempt("")
        except Exception as exc:
            _host.log.warning("Single-step first attempt failed (%s): %s", agent_id, exc)
            _host.PrimusSession.emit_think("Executing", f"Recovering from: {str(exc)[:120]}", "running")
            try:
                answer = _attempt("-retry")
                _host.PrimusSession.emit_think("Executing", "Recovered on retry", "done")
            except Exception as exc2:
                _host.log.exception("Single-step agent failed after retry (%s)", agent_id)
                _host.PrimusSession.emit_think("Executing", str(exc2)[:200], "error")
                answer = (
                    f"**Agent error** — `{exc2}`\n\n"
                    "Check Ollama, open Menu → Status, or try a simpler request."
                )

        # Completion nudge (same as multi-step): a lone offer/placeholder → one forced real attempt.
        if (
            bool(_host.CFG.get("enforce_step_completion", True))
            and not _host.PrimusSession.is_cancelled()
            and _looks_incomplete(answer)
        ):
            _host.PrimusSession.emit_think("Executing", "Looks incomplete — enforcing real action", "running")
            try:
                answer = _attempt(
                    "-force",
                    content=(
                        f"That was not completed. Do not describe, offer, or use placeholder paths — "
                        f"actually perform the request NOW with the most specific correct tool and REAL "
                        f"paths (e.g. ~/.primus/knowledge/downloads/). Request: {state['input']}"
                    ),
                )
            except Exception:  # noqa: BLE001
                pass
        answer = _finalize_turn_status(state, answer, agent_id)
        _host.log_past_task(state["input"], answer[:500])
        _host.PrimusSession.emit_think("Executing", "Complete", "done")
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    graph = StateGraph(PrimusGraphState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("single_step", single_step_node)
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"multi_step": "execute_step", "single_step": "single_step"},
    )
    graph.add_conditional_edges(
        "execute_step",
        route_after_step,
        {"continue": "execute_step", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)
    graph.add_edge("single_step", END)
    return graph.compile()


def build_primus_graph(model_name: str) -> Any:
    return build_agent_graph(model_name, PRIMUS_SYSTEM, agent_id="primus")


def build_forge_graph(model_name: str) -> Any:
    return build_agent_graph(model_name, FORGE_SYSTEM, agent_id="forge")


def init_agent_graphs(*, force: bool = False) -> dict[str, Any]:
    """Lazy-build Primus + Forge LangGraph agents."""
    if not _host.HAS_AI_STACK:
        _host.log.warning("Agent graphs unavailable — AI stack not installed: %s", _host.AI_STACK_ERROR[:120])
        return {}
    if _host._agent_graphs and not force:
        return _host._agent_graphs
    _host._agent_graphs = {
        "primus": build_primus_graph(_host.CFG.get("model", _host.DEFAULT_MODEL)),
        "forge": build_forge_graph(_host.CFG.get("forge_model", _host.FORGE_MODEL)),
    }
    # Optional small-model Primus graph for dynamic switching (same tools/prompt, faster model).
    if bool(_host.CFG.get("dynamic_switching_enabled", True)) and _fast_model_available():
        try:
            _host._agent_graphs["primus_fast"] = build_primus_graph(
                _host.CFG.get("fast_fallback_model") or _host.CFG.get("primus_fast_model")
            )
            _host.log.info("Built fast-model Primus graph: %s", _host.CFG.get("fast_fallback_model"))
        except Exception as exc:  # noqa: BLE001
            _host.log.warning("Fast-model Primus graph unavailable (%s) — using full model only", exc)
    _host.PrimusSession.graphs = _host._agent_graphs
    _host.PrimusSession.graph = _host._agent_graphs["primus"]
    warm_up_models_async()
    return _host._agent_graphs


def warm_up_models_async() -> None:
    """Preload the Primus model into memory so the first real request is fast."""
    if not bool(_host.CFG.get("warm_up_models", True)):
        return

    def _warm() -> None:
        try:
            ok, _ = _host.check_ollama_health()
            if not ok:
                return
            models = [_host.CFG.get("model", _host.DEFAULT_MODEL)]
            if bool(_host.CFG.get("fast_chat_tier", True)) and _fast_model_available():
                models.append(_host.CFG.get("primus_fast_model"))
            for m in models:
                try:
                    make_chat_ollama(m, temperature=0, num_predict=1, fast=True).invoke(
                        [HumanMessage(content="ok")]
                    )
                    _host.log.info("Model warm-up complete: %s", m)
                except Exception as exc:  # noqa: BLE001
                    _host.log.debug("Warm-up skipped for %s: %s", m, exc)
            # Now that a model is resident, report where it actually landed (GPU vs CPU).
            # This is how the automatic GPU→CPU fallback becomes visible in the logs.
            try:
                _host._gpu.bust_status_cache()  # force a fresh `ollama ps` read
                mode, detail = _host.gpu_effective_mode()
                if mode == "cpu" and _host.gpu_backend_value() != "cpu" and bool(_host.CFG.get("prefer_gpu", True)):
                    _host.log.warning("GPU preferred but model is on CPU — %s. See Setup / `/gpu`.", detail)
                else:
                    _host.log.info("Acceleration mode: %s — %s", mode.upper(), detail)
            except Exception as exc:  # noqa: BLE001
                _host.log.debug("GPU mode check skipped: %s", exc)
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("Model warm-up skipped: %s", exc)

    threading.Thread(target=_warm, name="primus-warmup", daemon=True).start()


# ---------------------------------------------------------------------------
# Sequential model policy — keep only ONE heavy model resident at a time.
#
# On the Ryzen laptop, having Primus (qwen2.5:7b) and Forge (qwen2.5-coder:7b) both
# loaded means two llama-server processes competing for RAM/CPU → heat + slowdowns.
# We evict the idle heavy model when switching agents. The light fast-chat model
# (llama3.2:3b) is intentionally kept loaded as the always-on lightweight fallback.
# ---------------------------------------------------------------------------

def _light_models_to_keep() -> set[str]:
    """Heavy-policy exemptions: the light fast-chat model stays resident as fallback."""
    keep: set[str] = set()
    if bool(_host.CFG.get("fast_chat_tier", True)) or bool(_host.CFG.get("dynamic_switching_enabled", True)):
        fast = _host.CFG.get("primus_fast_model") or _host.CFG.get("fast_fallback_model")
        if fast:
            keep.add(fast)
    return keep


def _model_matches(loaded: str, want: str) -> bool:
    """True if a loaded model name refers to the same model (handles :tag variants)."""
    if not loaded or not want:
        return False
    if loaded == want:
        return True
    base = want.split(":")[0]
    return loaded == base or loaded.startswith(f"{base}:")


def warm_model_async(name: str) -> None:
    """Reload a model into memory in the background (tiny no-op generation)."""
    if not name or not bool(_host.CFG.get("warm_up_models", True)):
        return

    def _warm() -> None:
        try:
            make_chat_ollama(name, temperature=0, num_predict=1, fast=True).invoke(
                [HumanMessage(content="ok")]
            )
            _host.log.info("Reloaded model into memory: %s", name)
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("Warm reload skipped for %s: %s", name, exc)

    threading.Thread(target=_warm, name="primus-reload", daemon=True).start()


def enforce_sequential_models(target_model: str, *, label: str = "") -> None:
    """Evict every resident HEAVY model except `target_model` (and the light fallback).

    Cheap: checks /api/ps first and only unloads models that are actually loaded.
    No-op unless `sequential_models` is enabled.
    """
    if not bool(_host.CFG.get("sequential_models", True)):
        return
    keep = _light_models_to_keep()
    keep.add(target_model)
    try:
        loaded = _host.ollama_loaded_models()
    except Exception:  # noqa: BLE001
        return
    for name in loaded:
        if any(_model_matches(name, k) for k in keep):
            continue
        _host.PrimusSession.emit_think(
            "Sequential models",
            f"Unloading {name} → freeing RAM for {label or target_model}",
            "running",
        )
        _host.ollama_unload_model(name)


def release_forge_reload_primus() -> None:
    """After a Forge turn: evict Forge and warm Primus back up for the next turn."""
    if not bool(_host.CFG.get("sequential_models", True)):
        return
    forge_model = _host.CFG.get("forge_model", _host.FORGE_MODEL)
    primus_model = _host.CFG.get("model", _host.DEFAULT_MODEL)
    if any(_model_matches(m, forge_model) for m in _host.ollama_loaded_models()):
        _host.PrimusSession.emit_think("Sequential models", f"Forge done → unloading {forge_model}", "done")
        _host.ollama_unload_model(forge_model)
    if bool(_host.CFG.get("sequential_warm_reload", True)):
        warm_model_async(primus_model)


_FAST_CHAT_TOOL_HINT_RE = re.compile(
    r"\b(run|execute|open|launch|start|list|show me|ls|cat|create|make|write|delete|remove|"
    r"install|update|upgrade|backup|organize|sort|clean|kill|restart|status|disk|battery|"
    r"wifi|bluetooth|network|process|git|index|search|recall|remember|learn|weather|forecast|"
    r"temperature|find|locate|download|ingest|upload|move|copy|rename|edit|check|test|deploy|"
    r"commit|push|pull|clone|scan|monitor|connect|disconnect|mount|unmount|e-?mail|gmail|inbox|"
    r"calendar|slack|file|files|folder|directory|summari[sz]e|analy[sz]e|read|schedule|remind|"
    r"send|draft|research|look up|google|news|headlines?)\b",
    re.I,
)
# Knowledge-subject veto: the fast-chat tier runs with NO retrieval, so a question whose
# answer lives in the KB / memory (documents, proposals, prices, operator preferences or
# projects) must route to the full path instead — otherwise the small model answers from
# parametric guesswork and invents facts (e.g. a wrong proposal price).
_FAST_CHAT_KB_HINT_RE = re.compile(
    r"\b(proposals?|quotes?|invoices?|contracts?|agreements?|documents?|docs?|notes?|reports?|"
    r"spreadsheet|presentation|deck|whitepaper|knowledge\s*base|kb|"
    r"price|pricing|cost|budget|rates?" + _OP_PROJECT_ALT + r")\b"
    r"|\baccording to\b"
    r"|\bwhat does\b[^?\n]{0,60}\bsay\b"
    r"|\bshould you (?:never|always)\b"
    r"|\bwhat did i (?:ask|tell|say)\b"
    r"|\b(?:your|my) (?:rules|instructions|standing orders|preferences?)\b",
    re.I,
)
# "Tell me about X" / "what is X" / "who is X" — a knowledge question with no tool verb.
# These must answer on the chat tier (fast_chat), never enter the 119-tool planning graph.
# Leading hello/please is decoration and must not block the match.
_KNOWLEDGE_Q_RE = re.compile(
    r"^\s*(?:(?:hi|hello|hey|yo|greetings|good\s(?:morning|afternoon|evening))\b[\s,!]*"
    r"(?:primus\b[\s,!]*)?)?(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:tell me about|tell me what|what(?:'?s| is| are| was| were)\b|who(?:'?s| is| was| are)\b|"
    r"explain\b|describe\b|define\b|how does\b|how do\b|why is\b|why are\b)",
    re.I,
)
# A "what is / tell me about" ask whose payload is really news, mail, files, weather, or time is
# NOT knowledge — it belongs to fast_tool_answer's news/mail/file branches. ("headline" singular
# is the live hole: the tool-hint veto only carried the plural.)
_KNOWLEDGE_TOOL_VETO_RE = re.compile(
    r"\b(?:headlines?|news|top stor(?:y|ies)|big stor(?:y|ies)|stor(?:y|ies) of the day|"
    r"e-?mails?|inbox|gmail|mail from|"
    r"folders?|director(?:y|ies)|home folder|"
    r"weather|temperature|forecast)\b"
    r"|\bwhat(?:'?s| is) the (?:time|date)\b|\bwhat (?:time|date)\b",
    re.I,
)

# "Give me the file" intent — the user wants the document itself, not an answer about it.
_FETCH_FILE_RE = re.compile(
    r"\b(?:give|get|fetch|send|hand|show)\s+me\b[^?\n]{0,60}"
    r"\b(?:file|document|doc|pdf|proposal|report|notes?|spreadsheet|deck)\b"
    r"|\b(?:download|export)\b[^?\n]{0,40}"
    r"\b(?:file|document|doc|pdf|proposal|report)\b",
    re.I,
)


def _kb_fetch_file_reply(mem_block: str) -> str:
    """Deterministic 'give me the file' handover: copy KB source files into exports + cite.

    Asked for the document itself, the model tends to paste the retrieved text (sometimes
    paraphrasing it wrong). A fetch request wants the FILE, so handle it without an LLM
    call: copy each cited source into ~/.primus/exports (the Gradio-download allowlisted
    dir) and answer with title + exact path + [KB-N] tag.
    """
    if not mem_block:
        return ""
    entries: list[tuple[str, str, str, bool]] = []  # (kb_tag, title, path, in_exports)
    seen: set[str] = set()
    for m in re.finditer(r"\[(KB-\d+)\][^\n]*?source=`file:([^`]+)`", mem_block):
        tag, raw = m.group(1), m.group(2).strip()
        src = Path(raw)
        if raw in seen or not src.is_file():
            continue
        seen.add(raw)
        title = src.name
        if src.suffix.lower() in (".md", ".markdown", ".txt", ".rst", ".csv", ".log"):
            try:
                for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip().lstrip("#").strip()
                    if line:
                        title = f"{src.name} — {line[:80]}"
                        break
            except OSError:
                pass
        dest = src
        try:
            _host.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            candidate = _host.EXPORT_DIR / src.name
            if candidate.resolve() != src.resolve():
                n = 1
                while candidate.exists() and candidate.stat().st_size != src.stat().st_size:
                    candidate = _host.EXPORT_DIR / f"{src.stem}-{n}{src.suffix}"
                    n += 1
                shutil.copy2(src, candidate)
                dest = candidate
        except OSError:
            pass
        entries.append((tag, title, str(dest), dest.parent == _host.EXPORT_DIR))
        if len(entries) >= 3:
            break
    if not entries:
        return ""
    in_exports = any(e[3] for e in entries)
    lines = [
        "Here you go — a copy is in your exports folder:" if in_exports else "Here you go:",
        "",
    ]
    for tag, title, path, _ in entries:
        lines.append(f"- [{tag}] **{title}**")
        lines.append(f"  `{path}`")
    lines.append("")
    lines.append('Say "print it" if you want the full text pasted here.')
    return "\n".join(lines)


def _kb_answer_eligible(decision: "RoutingDecision", message: str) -> bool:
    """Knowledge question: the answer lives in the KB/memory → retrieve + one bounded call.

    Same gates as the fast-chat tier (primus chat intent, no tool verbs, not a direct action)
    plus a knowledge-subject match — but unlike fast chat this path injects retrieval and does
    NOT require the small model. Without it, doc questions fell into a tool-less tier (and
    invented answers) or into the planner (which went searching the filesystem for minutes).
    """
    if decision.agent_id != "primus" or decision.intent != "chat":
        return False
    if decision.forge_score >= 1.0:
        return False
    if is_direct_action(message):
        return False
    if _FAST_CHAT_TOOL_HINT_RE.search(message) or _PULSE_VETO_RE.search(message):
        return False
    return bool(_FAST_CHAT_KB_HINT_RE.search(message))


def _fast_model_available() -> bool:
    fast = _host.CFG.get("primus_fast_model") or ""
    if not fast or fast == _host.CFG.get("model", _host.DEFAULT_MODEL):
        return False
    try:
        installed = list(_host.get_ollama_details().get("models") or [])
    except Exception:
        return False
    return any(fast == m or m.startswith(fast.split(":")[0]) for m in installed)


def _fast_chat_eligible(decision: "RoutingDecision", message: str) -> bool:
    """Pure conversational turns that need no tools → answer with the light model in one call."""
    if not bool(_host.CFG.get("fast_chat_tier", True)):
        return False
    if decision.agent_id != "primus" or decision.intent != "chat":
        return False
    if decision.forge_score >= 1.0:
        return False
    if is_direct_action(message):
        return False
    # Two vetoes, same purpose: this tier has no tools, so anything naming real work (a tool
    # verb, or a subject like mail/calendar/files/code) must not land here. The router calls
    # plenty of those turns "chat" — "summarize unread mail" scored chat/primus — and answering
    # them from a tool-less model is how Primus ends up inventing instead of fetching.
    if _FAST_CHAT_TOOL_HINT_RE.search(message) or _PULSE_VETO_RE.search(message):
        return False
    # Knowledge questions need retrieval; this tier has none (would answer from guesswork).
    if _FAST_CHAT_KB_HINT_RE.search(message):
        return False
    if not _fast_model_available():
        return False
    return True


# Signals that a coding task touches EXISTING code/projects → keep full RAG + planning.
_CODING_CONTEXT_RE = re.compile(
    r"\b(my|our|this|existing|current|already|above|previous)\b.{0,30}"
    r"\b(code|script|file|module|function|class|project|repo|app|agent|tool|pipeline)\b"
    r"|\b(admin_assistant|primus|ai-?workshop" + _OP_PROJECT_ALT + r")\b"
    r"|\.(py|sh|ts|js|jsx|tsx|yaml|yml|toml|json|sql)\b"
    r"|```",
    re.I,
)


def _coding_needs_context(message: str) -> bool:
    """True when a coding request references the operator's existing code/projects/files (needs RAG)."""
    return bool(_CODING_CONTEXT_RE.search(message or ""))


def _is_fast_coding(decision: "RoutingDecision", message: str) -> bool:
    """Eligible for the single-shot fast coding path: a small, self-contained NEW-code ask.

    Deliberately conservative — only the clear 'write a little script/function' case bypasses
    the plan→step→summarize graph. Anything touching existing code, pasted snippets, multi-step
    work, or long prompts goes through the full Forge graph (with its verification loop).
    """
    if not bool(_host.CFG.get("fast_coding_enabled", True)):
        return False
    if decision.agent_id != "forge":
        return False
    msg = (message or "").strip()
    if not msg or len(msg) > int(_host.CFG.get("fast_coding_max_chars", 320)):
        return False
    if "```" in msg or _MULTI_STEP_RE.search(msg):
        return False
    if _coding_needs_context(msg):
        return False
    if not is_forge_model_ready():
        return False
    return ModelRouter.coding_subtype(msg) == "new_script"


def route_prompt(decision: "RoutingDecision", message: str) -> dict[str, Any]:
    """Smart prompt ingestion — pick the cheapest correct execution path in one decision.

    Returns: {path, agent_id, needs_rag, reason, no_plan}
      path ∈ fast_chat | fast_coding | direct | forge | primus
    This is the single source of truth that drives RAG skipping + tier selection for speed.

    `no_plan=True` is a hard contract, not a hint: invoke_primus must answer that turn with a
    single bounded call and is forbidden from entering the planning graph, whatever happens.
    """
    # Knowledge / explain question FIRST ("tell me about X", "what is X", "who is X") with no
    # file/mail/shell/code verb → one bounded chat call. This must match before pulse (leading
    # "hello" is decoration, not a greeting turn) and before the eligibility gates that used to
    # let these slip into the 119-tool graph. FORBIDDEN here: plan→step, tools, watchdog shrug.
    # Identity questions ("who are you") stay pulse — they have a deterministic answer.
    if (
        _KNOWLEDGE_Q_RE.search(message)
        and not _PULSE_IDENTITY_RE.search(message)
        and decision.forge_score < 1.0
        and "```" not in message
        and len(message) <= 200
        and not _MULTI_STEP_RE.search(message)
        and not _FAST_CHAT_TOOL_HINT_RE.search(message)
        and not _PULSE_VETO_RE.search(message)
        and not _FAST_CHAT_KB_HINT_RE.search(message)
        and not _KNOWLEDGE_TOOL_VETO_RE.search(message)
    ):
        return {"path": "fast_chat", "agent_id": "primus", "needs_rag": False, "no_plan": True,
                "knowledge": True,
                "reason": "knowledge/explain question → one bounded chat call, no tools, no plan"}
    # Pulse: a greeting or identity lock is conversation, so it can never be worth a plan.
    if is_pulse_turn(message):
        return {"path": "fast_chat", "agent_id": "primus", "needs_rag": False, "no_plan": True,
                "reason": "pulse (greeting/identity/thanks) → one chat call, no tools, no plan"}
    if decision.agent_id == "forge":
        # Small self-contained coding ask → one focused coder-model call (no graph).
        if _is_fast_coding(decision, message):
            return {"path": "fast_coding", "agent_id": "forge", "needs_rag": False,
                    "no_plan": False,
                    "reason": "small self-contained coding task → single coder-model call"}
        # Lean context: skip the heavy memory dump for self-contained coding (cleaner, faster).
        needs_rag = True
        if bool(_host.CFG.get("forge_lean_context", True)) and not _coding_needs_context(message):
            needs_rag = False
        return {"path": "forge", "agent_id": "forge", "needs_rag": needs_rag, "no_plan": False,
                "reason": f"coding/technical → Forge ({decision.reason})"}
    if is_direct_action(message):
        # Tool action (open app, ls, browse…) — no embeddings needed, run immediately.
        return {"path": "direct", "agent_id": "primus", "needs_rag": False, "no_plan": False,
                "reason": "direct tool action — fast path, no RAG"}
    if _kb_answer_eligible(decision, message):
        return {"path": "kb_answer", "agent_id": "primus", "needs_rag": True, "no_plan": True,
                "reason": "knowledge question → retrieve + one bounded answer, no plan"}
    if _fast_chat_eligible(decision, message):
        return {"path": "fast_chat", "agent_id": "primus", "needs_rag": False, "no_plan": True,
                "reason": "pure conversation → fast model, no RAG"}
    if ModelRouter.cacheable(decision, message) and not _PULSE_VETO_RE.search(message):
        # Trivial chatter (short, no tool hint, no work subject, no recall): a plan can only
        # make it worse, so this stays on the chat path even when the small model isn't
        # installed — the single call just runs on the main model instead.
        return {"path": "fast_chat", "agent_id": "primus", "needs_rag": False, "no_plan": True,
                "reason": "trivial chatter — one chat call, no plan"}
    # Leftover / mixed — primus_path: Primus decides order and tools (plan→execute graph).
    return {"path": "primus", "agent_id": "primus", "needs_rag": True, "no_plan": False,
            "reason": f"primus_path orchestration ({decision.reason})"}


# ---------------------------------------------------------------------------
# Intelligent dynamic model switching
#
# Watches how long Primus turns take. If a *simple* (non-coding) turn runs slower
# than slow_threshold_sec, it flips into "fast mode" for a cooldown window — routing
# simple Primus turns through the small fallback model for snappier replies. Coding /
# Forge / deep-reasoning turns always keep the full model. A manual override
# (/model fast | large | auto) takes precedence. Fully optional and never breaks
# existing models or fast paths.
# ---------------------------------------------------------------------------

# Intents that are safe to answer with the small model when things get slow.
_DYNAMIC_SIMPLE_INTENTS = {"chat", "admin", "memory"}


class DynamicModelManager:
    """Session-scoped policy that may swap Primus's model for the small one when slow."""

    override: str = "auto"      # auto | fast | large
    fast_until: float = 0.0     # epoch; while now < this, prefer the fast model
    last_reason: str = ""

    @classmethod
    def set_override(cls, mode: str) -> str:
        mode = (mode or "auto").lower().strip()
        if mode not in ("auto", "fast", "large"):
            return "Usage: `/model fast` · `/model large` · `/model auto`"
        cls.override = mode
        if mode != "auto":
            cls.fast_until = 0.0  # manual choice clears any auto-trip
        labels = {
            "fast": f"Pinned to the fast model (`{_host.CFG.get('fast_fallback_model')}`) for simple turns.",
            "large": f"Pinned to the full model (`{_host.CFG.get('model', _host.DEFAULT_MODEL)}`). No auto-switching.",
            "auto": "Dynamic model switching is back on (auto).",
        }
        return labels[mode]

    @classmethod
    def fast_mode_active(cls) -> bool:
        if cls.override == "fast":
            return True
        if cls.override == "large":
            return False
        return bool(_host.CFG.get("dynamic_switching_enabled", True)) and time.time() < cls.fast_until

    @classmethod
    def prefer_fast(cls, decision: "RoutingDecision") -> bool:
        """True when this Primus turn should use the small model."""
        if decision.agent_id != "primus":
            return False
        if decision.intent not in _DYNAMIC_SIMPLE_INTENTS or decision.forge_score >= 2.0:
            return False
        if not _fast_model_available():
            return False
        return cls.fast_mode_active()

    @classmethod
    def note_result(cls, *, agent_id: str, intent: str, path: str, elapsed_sec: float) -> None:
        """After a turn: trip fast mode if a simple full-model turn was too slow."""
        if not bool(_host.CFG.get("dynamic_switching_enabled", True)) or cls.override != "auto":
            return
        # Only the heavy graph path on the full model is worth reacting to.
        if agent_id != "primus" or path not in ("primus", "direct"):
            return
        if intent not in _DYNAMIC_SIMPLE_INTENTS:
            return
        threshold = float(_host.CFG.get("slow_threshold_sec", 5.0))
        if elapsed_sec >= threshold and _fast_model_available():
            cooldown = int(_host.CFG.get("fast_model_switch_cooldown", 300))
            cls.fast_until = time.time() + cooldown
            cls.last_reason = f"{intent} turn took {elapsed_sec:.1f}s ≥ {threshold:.0f}s"
            _host.log.info("Dynamic switch → fast model for %ds (%s)", cooldown, cls.last_reason)
            _host.PrimusSession.emit_think(
                "Dynamic model",
                f"Slow ({elapsed_sec:.1f}s) → fast model for {cooldown // 60}m",
                "done",
            )

    @classmethod
    def status(cls) -> str:
        fast = _host.CFG.get("fast_fallback_model") or _host.CFG.get("primus_fast_model")
        full = _host.CFG.get("model", _host.DEFAULT_MODEL)
        if cls.override == "fast":
            return f"pinned fast (`{fast}`)"
        if cls.override == "large":
            return f"pinned full (`{full}`)"
        if cls.fast_mode_active():
            rem = max(0, int(cls.fast_until - time.time()))
            return f"auto · fast mode {rem}s left (`{fast}`)"
        return f"auto · full (`{full}`)"


def _fast_chat_answer(
    message: str,
    history: list,
    mem_block: str,
    mode: str,
    *,
    prefer_main: bool = False,
    timeout_sec: Optional[int] = None,
) -> tuple[str, str]:
    """Single bounded conversational reply — no tools, no graph (RAG + memory still injected).

    Runs on the small model when it is actually installed, otherwise on the main model: the
    chat path has to be able to answer on its own, because escalating a greeting to the
    planning graph is what leaked plan chrome onto identity turns. `prefer_main` forces the
    main model (knowledge answers need citation fidelity, not just speed).

    Hard-capped like the fast coding call. A stall here returns TimeoutError to the caller,
    which answers plainly instead of falling through to a planner.
    """
    model = (
        _host.CFG.get("primus_fast_model")
        if (_fast_model_available() and not prefer_main)
        else _host.CFG.get("model", _host.DEFAULT_MODEL)
    )
    llm = make_chat_ollama(
        model,
        temperature=float(_host.CFG.get("primus_temperature", 0.2)),
        num_predict=int(_host.CFG.get("primus_num_predict", 768)),
    )
    from primus.core.redact import redact_model_input  # noqa: PLC0415

    message = redact_model_input(message)
    mem_block = redact_model_input(mem_block)
    values = _agent_context_from_state(
        {},  # type: ignore[arg-type]
        mode=mode,
        memory=_host.memory_context_block(),
        rag_context=mem_block,
        plan="",
        step_num="1",
        step_total="1",
        current_step=message,
    )
    system = redact_model_input(format_agent_system(FAST_CHAT_SYSTEM, values))
    prior = [redact_model_input(m) for m in _host.prepare_agent_messages(history)[-6:]]
    msgs = [SystemMessage(content=system)] + prior + [
        HumanMessage(content=message)
    ]
    timeout = timeout_sec or int(_host.CFG.get("fast_chat_invoke_timeout_sec", 15))
    if timeout <= 0:
        return polish_response(llm.invoke(msgs).content), model
    box: dict[str, Any] = {}
    err: list[Exception] = []

    def _run() -> None:
        try:
            box["resp"] = llm.invoke(msgs)
        except Exception as exc:  # noqa: BLE001 — surfaced to the caller below
            err.append(exc)

    worker = threading.Thread(target=_run, name="primus-fast-chat", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"fast chat exceeded {timeout}s on {model}")
    if err:
        raise err[0]
    return polish_response(box["resp"].content), model


def _fast_coding_answer(message: str, history: list, mem_block: str, mode: str) -> tuple[str, str]:
    """Single-call coding reply on the Forge (coder) model — clean, focused, no plan/step overhead.

    Used for small self-contained asks. Gives qwen2.5-coder a tight context window so it produces
    high-quality code fast, then tells it to include a verify command in the output.
    """
    model = _host.CFG.get("forge_model", _host.FORGE_MODEL)
    llm = make_chat_ollama(
        model,
        temperature=float(_host.CFG.get("forge_temperature", 0.15)),
        num_predict=int(_host.CFG.get("forge_num_predict", 2048)),
    )
    values = _agent_context_from_state(
        {},  # type: ignore[arg-type]
        mode=mode,
        memory="",                       # lean: no full memory dump for a self-contained snippet
        rag_context=mem_block or "",
        plan="",
        step_num="1",
        step_total="1",
        current_step=message,
    )
    system = format_agent_system(FORGE_FAST_SYSTEM, values)
    msgs = [SystemMessage(content=system)] + _host.prepare_agent_messages(history)[-4:] + [
        HumanMessage(content=message)
    ]
    # Bound the single call so a stall degrades cleanly to the full Forge graph instead of hanging.
    timeout = int(_host.CFG.get("forge_fast_invoke_timeout_sec", 100))
    box: dict[str, Any] = {}
    err: list[Exception] = []

    def _run() -> None:
        try:
            box["resp"] = llm.invoke(msgs)
        except Exception as exc:  # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=_run, name="primus-fast-coding", daemon=True)
    t.start()
    t.join(timeout=timeout if timeout > 0 else None)
    if t.is_alive():
        raise TimeoutError(f"fast coding exceeded {timeout}s")
    if err:
        raise err[0]
    return polish_response(box["resp"].content), model


# Fenced Python blocks in a Forge answer (```python … ``` or ```py … ```).
_PY_BLOCK_RE = re.compile(r"```(?:python|py)[^\n]*\n(.*?)```", re.S | re.I)

_FORGE_REPAIR_SYSTEM = (
    "You are Forge, a senior Python engineer. The code you just delivered FAILED static "
    "verification (compile and/or lint). Return the COMPLETE corrected answer in the same shape: a "
    "one-line note on what you fixed, the full corrected code in a single ```python block (never "
    "abbreviate with '...'), then a `Verify:` line with the command to run it. Fix every reported "
    "error and re-check mentally before answering. Output the answer only — no apologies, no preamble."
)


def _extract_python_blocks(text: str) -> list[str]:
    return [m.group(1) for m in _PY_BLOCK_RE.finditer(text or "")]


def _forge_autocorrect(message: str, answer: str, model_name: str) -> str:
    """Verify Forge's delivered Python and self-repair once if it has real errors.

    This is the closed loop that pushes the local 7B coder toward frontier reliability: instead of
    trusting the model's claim that code works, we actually compile + lint it. If a block has a hard
    error, we hand the diagnostics back for ONE bounded repair pass and re-verify. Static-only and
    fully local — it never executes the code and never edits files.
    """
    if not bool(_host.CFG.get("forge_autocorrect", True)) or not answer:
        return answer
    blocks = _extract_python_blocks(answer)
    if not blocks:
        return answer

    def _problems(blks: list[str]) -> list[str]:
        out: list[str] = []
        for i, code in enumerate(blks, 1):
            if len(code.strip()) < 12:  # skip trivial one-liners / pseudo-snippets
                continue
            ok, report = _host._verify_python_text(code)
            if not ok:
                out.append(f"Code block {i}:\n{report}")
        return out

    try:
        problems = _problems(blocks)
    except Exception as exc:  # noqa: BLE001 — verification must never break a turn
        _host.log.debug("Forge autocorrect verify skipped: %s", exc)
        return answer
    if not problems:
        _host.PrimusSession.emit_think("Verify", "Forge code compiles + lints cleanly", "done")
        return answer

    _host.PrimusSession.emit_think(
        "Self-correct", f"{len(problems)} code issue(s) found — repairing once", "running"
    )
    try:
        llm = make_chat_ollama(
            model_name,
            temperature=0.1,
            num_predict=int(_host.CFG.get("forge_num_predict", 2048)),
        )
        repair_input = (
            f"Original request:\n{message}\n\n"
            f"Your previous answer:\n{answer}\n\n"
            f"Static verification found these errors that must be fixed:\n"
            + "\n\n".join(problems)
        )
        msgs = [
            SystemMessage(content=_FORGE_REPAIR_SYSTEM),
            HumanMessage(content=repair_input),
        ]
        timeout = int(_host.CFG.get("forge_autocorrect_timeout_sec", 90))
        box: dict[str, Any] = {}
        err: list[Exception] = []

        def _run() -> None:
            try:
                box["resp"] = llm.invoke(msgs)
            except Exception as exc:  # noqa: BLE001
                err.append(exc)

        t = threading.Thread(target=_run, name="primus-forge-repair", daemon=True)
        t.start()
        t.join(timeout=timeout if timeout > 0 else None)
        if t.is_alive() or err or "resp" not in box:
            raise (err[0] if err else TimeoutError("repair pass timed out"))

        repaired = polish_response(box["resp"].content)
        new_blocks = _extract_python_blocks(repaired)
        # Only accept the repair if it actually has code AND verifies at least as well.
        if new_blocks and len(_problems(new_blocks)) < len(problems):
            _host.PrimusSession.emit_think("Self-correct", "Repaired and re-verified", "done")
            return repaired
        _host.PrimusSession.emit_think("Self-correct", "Kept original (repair not cleaner)", "done")
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Forge autocorrect repair skipped: %s", exc)
        _host.PrimusSession.emit_think("Self-correct", "Skipped — kept original", "done")
    return answer


def _response_path_label() -> str:
    """Normalized label for the path that answered the last turn (for latency metrics)."""
    return _host.PrimusSession.last_path or "primus"


# Patterns that make a proactive follow-up genuinely useful (not noise).
_FOLLOWUP_RULES: tuple[tuple[str, str], ...] = (
    (r"\b(disk|storage|space)\b.*\b(full|low|\d{2,3}%)\b",
     "Want me to run a cleanup report to free space?"),
    (r"\b(installed|opened|launched)\b",
     "Want me to pin it or set it to auto-start?"),
    (r"\b(error|failed|traceback|exception)\b",
     "Want me to dig into the cause and propose a fix?"),
    (r"\bweb_search|searched|look(ed)? up\b",
     "Want me to save these findings to your knowledge base?"),
)


def append_followup(message: str, answer: str) -> str:
    """Proactive anticipation: append ONE smart, learned next-step suggestion when it adds value.

    Lightweight + gated: skips short/error answers, slash output, and anything that already
    asks a question. Uses learned signals (open todos, recurring topics) plus a few high-value
    rules so Primus anticipates needs without nagging.
    """
    if not _host.CFG.get("proactive_followups", True):
        return answer
    # A greeting, a thank-you, or an identity confirmation is complete on its own. Bolting an
    # offer onto it is the padding the operator asked to be rid of.
    if is_pulse_turn(message):
        return answer
    a = (answer or "").strip()
    newsish = bool(re.search(r"\b(?:news|headlines?|top stor(?:y|ies))\b", message or "", re.I))
    if newsish:
        a = re.sub(
            r"(?:\n+\s*)?_?Want me to pin it or set it to auto-start\??_?\s*$",
            "",
            a,
            flags=re.I,
        ).strip()
        answer = a
    if len(a) < 40 or a.endswith("?") or a.startswith(("**Primus error", "_")):
        return answer
    if a in (IDENTITY_ACK, BAD_FORMAT_REPLY, CHAT_TIMEOUT_REPLY, CHAT_SLOW_REPLY):
        return answer
    if "Want me to" in a or "Should I" in a:  # already proactive
        return answer

    blob = f"{message}\n{a}".lower()
    for pattern, suggestion in _FOLLOWUP_RULES:
        if newsish and "pin it" in suggestion:
            continue
        if re.search(pattern, blob):
            return f"{answer}\n\n_{suggestion}_"

    # Learned nudge: if the operator signals a task is done and there are open todos, surface the next one.
    try:
        if re.search(r"\b(done|finished|completed|next|what now)\b", message.lower()):
            raw = _host.TODOS_FILE.read_text(encoding="utf-8") if _host.TODOS_FILE.exists() else ""
            pending = [ln for ln in raw.splitlines() if "- [ ]" in ln]
            if pending:
                nxt = pending[0].split("- [ ]", 1)[1].split("_(")[0].strip()
                if nxt:
                    return f"{answer}\n\n_Next on your list: {nxt[:80]}. Want me to start it?_"
    except Exception:  # noqa: BLE001
        pass
    return answer


# --- Fast meta-instruction path -------------------------------------------------------------
# Preference/style directives ("be more concise", "always provide complete info",
# "remember I prefer…") are cheap to honor: store them in memory + confirm. Routing them
# through the full plan→execute→summarize loop is slow and was prone to getting stuck, so we
# short-circuit them here. Patterns are deliberately tight to avoid hijacking real requests.

_META_INSTRUCTION_PATTERNS = (
    r"^\s*remember (that |this[:,]? )?(i|my|to|that)\b",
    r"\b(i'?d|i would|i'?ll)?\s*(prefer|always want|never want)\b",
    r"\bi (prefer|like it when|want you to (always|never))\b",
    r"\b(from now on|going forward|in (the )?future|always|never)\b.{0,60}\b"
    r"(provide|include|give|answer|respond|reply|use|format|list|add|keep|be|make|"
    r"avoid|skip|complete|full|detail|brief|concise|verbose)\b",
    r"\bbe (more|less) (concise|brief|detailed|thorough|verbose|formal|casual|wordy|technical)\b",
    r"\b(keep|make) (it|your (responses|answers|replies|messages))\b.{0,40}"
    r"(short|shorter|brief|concise|long|longer|detailed|complete|thorough)\b",
    r"\b(give|provide|want) (me )?(longer|shorter|complete|full|detailed|brief|concise)\b.{0,30}"
    r"(responses|answers|information|replies|detail)\b",
    r"\bprovide all (of )?the (information|info|details|data)\b",
    r"\b(your )?(responses|answers|replies) should (always|never|be)\b",
)

# If the user is clearly asking for work/analysis (not just setting a preference), let the full
# agent handle it even if it contains preference-ish words.
_META_DEEP_ANALYSIS_RE = re.compile(
    r"\b(analy[sz]e|investigate|debug|research|write (me )?(a|the|some)|build|create|generate|"
    r"refactor|implement|fix|explain|describe|summari[sz]e|why (does|is|are|did)|"
    r"how (do|does|can) i|deep dive|walk me through|"
    r"(run|execute|launch) (this|that|the|it|my|a )|this command)\b",
    re.IGNORECASE,
)


def is_meta_instruction(message: str) -> bool:
    """True when the message is a standalone preference/style directive we can honor instantly."""
    if not _host.CFG.get("fast_meta_path", True):
        return False
    text = (message or "").strip()
    # Keep it to short, single-intent directives — long messages are usually real tasks.
    if not text or len(text) > 240:
        return False
    if _META_DEEP_ANALYSIS_RE.search(text):
        return False
    low = text.lower()
    return any(re.search(p, low) for p in _META_INSTRUCTION_PATTERNS)


def handle_meta_instruction(message: str) -> str:
    """Persist a preference/style directive (memory + session + prefs) and confirm briefly.

    Stores in three places so it takes effect immediately AND survives restarts:
      • long-term memory (high importance, tagged 'preferences') for semantic recall,
      • the active session facts (injected into context this turn onward),
      • a structured `style_directives` list in preferences.json for quick reference.
    """
    directive = " ".join((message or "").strip().split())
    _host.PrimusSession.last_path = "meta"
    _host.PrimusSession.active_agent = "primus"
    _host.PrimusSession.active_model = "meta"
    stored_ok = True
    try:
        _host.get_memory_system().add_memory(
            directive,
            tags=["preferences", "style"],
            importance=0.95,
            source="user:meta",
            promote_session=True,  # also adds a session fact → applied right away
        )
    except Exception as exc:  # noqa: BLE001 — never fail the confirmation on storage hiccups
        _host.log.warning("Meta-instruction LTM store failed: %s", exc)
        stored_ok = False
    try:
        mem = _host.load_memory()
        prefs = mem.setdefault("preferences", {})
        directives = prefs.setdefault("style_directives", [])
        if directive not in directives:
            directives.append(directive)
            prefs["style_directives"] = directives[-25:]  # keep the most recent
        _host.save_memory(mem)
    except Exception as exc:  # noqa: BLE001
        _host.log.warning("Meta-instruction preference store failed: %s", exc)

    tail = "Saved to memory and active now." if stored_ok else "Active this session (note: long-term save hit an issue)."
    return f"Got it — I'll keep that in mind: \"{directive}\". {tail}"


def _progress_from_thinking() -> dict[str, Any]:
    """Reconstruct step progress from the live thinking trace so a partial answer can show a clear
    done/remaining ledger. Planner emits 'Step N' (pending, with the step text); the executor emits
    'Step N/M' (running→done). We stitch both into {planned, done, total, running}."""
    planned: dict[int, str] = {}
    done: set[int] = set()
    total = 0
    running = 0
    for step in _host.PrimusSession.thinking_steps:
        title = str(step.get("title", ""))
        detail = str(step.get("detail", "")).strip()
        m_plan = re.match(r"^Step (\d+)$", title)
        if m_plan:
            planned[int(m_plan.group(1))] = detail
            continue
        m_exec = re.match(r"^Step (\d+)/(\d+)$", title)
        if m_exec:
            n, tot = int(m_exec.group(1)), int(m_exec.group(2))
            total = max(total, tot)
            if step.get("status") == "done":
                done.add(n)
            elif step.get("status") == "running":
                running = n
    if planned and not total:
        total = max(planned)
    return {"planned": planned, "done": sorted(done), "total": total, "running": running}


def _is_chat_turn() -> bool:
    """True when the turn that just ran was conversation rather than work.

    Recovery copy branches on this: a chat turn has no plan, so there is no progress to report
    and nothing to resume, and offering either is how plan chrome reached a greeting.

    Keyed on the PATH, not the intent. The router calls plenty of tool work "chat" intent
    ("summarize unread mail" scores chat/primus), and those turns do run a plan — telling
    the operator "ask me again" instead of handing them the work done so far would lose it.
    """
    return _host.PrimusSession.last_path in ("fast_chat", "instant", "fast_tool")


def _partial_answer_from_thinking(note: str, *, chat_turn: Optional[bool] = None) -> str:
    """Assemble a graceful PARTIAL answer from whatever step output was captured before a
    timeout/halt. Pulls completed 'Step N' details from the live thinking trace so the user
    still gets the work done so far — plus a compact progress ledger (done / in-progress /
    remaining) and a clear resume pointer, instead of a bare "partial answer" note.

    On a chat turn the ledger is suppressed entirely: `note` is the whole reply.
    """
    if chat_turn is None:
        chat_turn = _is_chat_turn()
    if chat_turn:
        return note
    parts: list[str] = []
    for step in _host.PrimusSession.thinking_steps:
        title = str(step.get("title", ""))
        detail = str(step.get("detail", "")).strip()
        if step.get("status") == "done" and detail and re.match(r"(Step \d|Executing|Summary)", title):
            parts.append(detail)
    body = "\n\n".join(p for p in parts if p)

    # Where it stopped, in prose. The old version drew a "Progress: a/b / Remaining: Step N /
    # Say continue" ledger; that is step machinery, so it reads as scaffolding to the operator and is
    # stripped on the way out anyway. Same information, written like a person.
    prog = _progress_from_thinking()
    status = ""
    if prog["total"]:
        remaining = [n for n in range(1, prog["total"] + 1) if n not in prog["done"]]
        next_up = prog["running"] or (remaining[0] if remaining else 0)
        detail = prog["planned"].get(next_up, "") if next_up else ""
        status = f"Got {len(prog['done'])} of {prog['total']} done"
        status += f" — still owed: {detail}." if detail else "."

    joined = "\n\n".join(s for s in (body, status) if s)
    return f"{joined}\n\n_{note}_" if joined else f"_{note}_"


# --- Background watchdog + auto-recovery -----------------------------------------------------
# A lightweight daemon that watches the active turn. If a turn runs past the hard limit (a
# model/tool chain hung, network stalled, etc.) it forces a halt so the invocation path returns
# a partial answer and the UI becomes responsive again. It never blocks normal operation —
# it only acts on turns that have clearly overrun.

_WATCHDOG_STARTED = False
_WATCHDOG_LOCK = threading.Lock()


def _log_watchdog_incident(runtime_sec: float) -> None:
    """Append a small record of a force-recovered turn for later review (best-effort)."""
    try:
        incidents: list = []
        if _host.WATCHDOG_FILE.exists():
            incidents = json.loads(_host.WATCHDOG_FILE.read_text(encoding="utf-8")) or []
        incidents.append(
            {
                "at": _host._now_iso(),
                "runtime_sec": round(runtime_sec, 1),
                "agent": _host.PrimusSession.active_agent,
                "model": _host.PrimusSession.active_model,
                "path": _host.PrimusSession.last_path,
            }
        )
        _host.WATCHDOG_FILE.write_text(json.dumps(incidents[-100:], indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — incident logging must never crash the watchdog
        _host.log.debug("Watchdog incident log failed: %s", exc)


def _turn_watchdog_loop() -> None:
    """Guardian loop: force-recover any turn that exceeds the configured hard limit."""
    last_incident_start = 0.0
    while True:
        try:
            time.sleep(3.0)
            if not bool(_host.CFG.get("turn_watchdog_enabled", True)):
                continue
            if not _host.PrimusSession.turn_active:
                continue
            limit = int(_host.CFG.get("turn_hard_limit_sec", 420))
            if limit <= 0:
                continue
            runtime = _host.PrimusSession.turn_runtime()
            started = _host.PrimusSession.turn_started_at
            # Fire once per stuck turn (keyed on its start time), and only if not already halting.
            if (
                runtime >= limit
                and started
                and started != last_incident_start
                and not _host.PrimusSession.is_cancelled()
            ):
                last_incident_start = started
                _host.log.warning(
                    "WATCHDOG: turn exceeded %ss (ran %.0fs) — forcing recovery", limit, runtime
                )
                _log_watchdog_incident(runtime)
                _host.PrimusSession.request_halt(f"watchdog:{int(runtime)}s")
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("Watchdog loop error: %s", exc)


def start_turn_watchdog() -> None:
    """Start the background watchdog exactly once (idempotent)."""
    global _WATCHDOG_STARTED
    with _WATCHDOG_LOCK:
        if _WATCHDOG_STARTED:
            return
        _WATCHDOG_STARTED = True
    threading.Thread(target=_turn_watchdog_loop, name="primus-watchdog", daemon=True).start()
    _host.log.info("Turn watchdog active (hard limit %ss)", _host.CFG.get("turn_hard_limit_sec", 420))


def _apply_turn_pack(
    text: str,
    tools: Optional[list[str]] = None,
    *,
    need_brain: bool = False,
) -> str:
    """One pack per leftover turn. Receptionist ``tools[]`` may select it."""
    from primus.core.tool_packs import (  # noqa: PLC0415
        extras_from_named,
        pack_names,
        select_pack,
        set_turn_pack,
    )

    pack = select_pack(text, tools=tools)
    extra, _refuse = extras_from_named("", pack, tools=tools or [])
    extra = list(extra)
    if need_brain and "ask_grok" not in pack_names(pack) and "ask_grok" not in extra:
        extra.append("ask_grok")
    set_turn_pack(pack, extra)
    _host.log.info("turn pack=%s extra=%s", pack, extra)
    return pack


def _user_path_in_process(step: str) -> bool:
    """True when path_mode can run this step without create_agent."""
    from primus.core import path_mode as pm  # noqa: PLC0415

    low = step.lower().strip()
    if pm._SLACK_RE.search(low) and not re.search(r"\b(?:list|read|status)\b", low):
        return True
    if pm._SEND_MAIL_RE.search(low):
        return True
    if pm._SUMMARIZE_RE.search(low):
        return True
    if pm._COUNT_RE.search(low) and not re.search(r"\b(?:code|script|python)\b", low):
        return True
    if pm._LIST_RE.match(step) or re.match(r"^\s*(?:list|ls)\b", low):
        return True
    if pm._MOVE_RE.match(step):
        return True
    if pm._READ_RE.match(step):
        return True
    if (
        pm.is_write_with_step(step)
        or pm._WRITE_RE.match(step)
        or pm.is_create_file_step(step)
        or pm.is_create_dir_step(step)
    ):
        return True
    if pm._DELETE_RE.match(step) or re.search(r"\bdelete\b.*\bdraft\b", low):
        return True
    if pm._KNOWLEDGE_STEP_RE.search(step):
        return True
    return False


def _invoke_packed_user_step(step: str, pack: str, extra: list[str], mode: str) -> str:
    """One user_path step through create_agent with that step's pack."""
    if not _host.HAS_AI_STACK:
        return _host._agent_unavailable_message()
    ok, health = _host.check_ollama_health()
    if not ok:
        return f"**Primus paused** — {health}"
    model_name = _host.CFG.get("model", _host.DEFAULT_MODEL)
    llm = make_chat_ollama(
        model_name,
        temperature=float(_host.CFG.get("primus_temperature", 0.25)),
        num_predict=int(_host.CFG.get("primus_num_predict", 768)),
    )
    state: PrimusGraphState = {
        "messages": [],
        "input": step,
        "plan": "",
        "steps": [step],
        "step_index": 0,
        "step_results": [],
        "mode": mode,
        "final_answer": "",
        "rag_context": "(No knowledge retrieved.)",
    }
    result = invoke_step_agent(
        None,
        system_template=PRIMUS_SYSTEM,
        state=state,
        user_content=f"{step}\n\n{SINGLE_STEP_DIRECTIVE}",
        thread_id=f"user-path-{pack}",
        llm=llm,
        pack=pack,
        extra_tools=extra,
        agent_id="primus",
        mode=mode,
        current_step=step,
        today=datetime.now().strftime("%A, %Y-%m-%d %H:%M"),
    )
    return coerce_message_text(result["messages"][-1].content)


def _run_user_path_with_packs(
    steps: list[str],
    *,
    answer_knowledge: Callable[[str], str],
    mode: str,
) -> str:
    """In-process mappings stay in-process. Other steps bind one pack (may switch)."""
    from primus.core.path_mode import run_user_path  # noqa: PLC0415
    from primus.core.tool_packs import extras_from_named, select_pack  # noqa: PLC0415

    parts: list[str] = []
    i = 0
    n = len(steps)
    while i < n:
        if _user_path_in_process(steps[i]):
            j = i + 1
            while j < n and _user_path_in_process(steps[j]):
                j += 1
            parts.append(run_user_path(steps[i:j], answer_knowledge=answer_knowledge))
            i = j
            continue

        step = steps[i]
        i += 1
        inst = instant_answer(step)
        if inst is not None:
            parts.append(inst)
            continue
        try:
            ft = fast_tool_answer(step)
        except Exception:  # noqa: BLE001
            ft = None
        if ft is not None:
            parts.append(ft)
            continue
        if _KNOWLEDGE_Q_RE.search(step) and not _KNOWLEDGE_TOOL_VETO_RE.search(step):
            parts.append(answer_knowledge(step))
            continue

        pack = select_pack(step)
        extra, refuse = extras_from_named(step, pack)
        if refuse:
            _host.log.info("user_path refuse pack=%s: %s", pack, refuse)
            parts.append(refuse)
            continue
        _host.log.info("user_path pack=%s extra=%s step=%s", pack, extra, step[:80])
        parts.append(_invoke_packed_user_step(step, pack, extra, mode))
    return "\n\n".join(p for p in parts if p)


def invoke_primus(
    graphs: dict[str, Any],
    message: str,
    history: list,
    mode: str,
    *,
    force_forge: bool = False,
    force_primus: bool = False,
    scope: str = "",
) -> tuple[str, list[dict]]:
    """Route to Primus or Forge with reasoning trace, cache, and fallback.

    `scope` is an optional project-chat retrieval hint passed through to build_memory_context;
    "" (General Chat) preserves the original behavior exactly.
    """
    answer, steps = _invoke_primus_impl(
        graphs, message, history, mode,
        force_forge=force_forge, force_primus=force_primus, scope=scope,
    )
    try:
        from primus.core.feedback import capture_after_turn  # noqa: PLC0415

        capture_after_turn(message, answer)
    except Exception:  # noqa: BLE001 — feedback.jsonl must never fail the turn
        pass
    return answer, steps


def _invoke_primus_impl(
    graphs: dict[str, Any],
    message: str,
    history: list,
    mode: str,
    *,
    force_forge: bool = False,
    force_primus: bool = False,
    scope: str = "",
) -> tuple[str, list[dict]]:
    """Inner turn body. The JSONL correction log hooks after ``invoke_primus`` returns."""
    _host.PrimusSession.clear_thinking()
    _host.PrimusSession.begin_turn()
    try:
        from primus.core.tool_packs import clear_turn_pack  # noqa: PLC0415

        clear_turn_pack()
    except Exception:  # noqa: BLE001
        pass
    try:
        from primus.core.brain_governor import begin_turn as _brain_begin_turn  # noqa: PLC0415

        _brain_begin_turn(message)
    except Exception:  # noqa: BLE001 — governor must never fail the turn
        pass
    try:
        from primus.core.audit import begin_turn as _audit_begin_turn  # noqa: PLC0415

        _audit_begin_turn()
    except Exception:  # noqa: BLE001 — audit must never fail the turn
        pass
    _host.PrimusSession.emit_think("Received", message[:100], "done")

    from primus.core.path_mode import (  # noqa: PLC0415 — keep path_mode off the hot import
        current_path_mode,
        has_file_write_steps,
        has_list_steps,
        inspect_path_message,
        set_path_mode,
    )

    path_ins = inspect_path_message(message)
    if path_ins.toggle_off:
        set_path_mode("primus_path")
    elif path_ins.toggle_on or path_ins.prefix_user:
        set_path_mode("user_path")

    try:
        from primus.core.audit import explain_recent, is_why_ask  # noqa: PLC0415

        if is_why_ask(message):
            _host.PrimusSession.active_agent = "primus"
            _host.PrimusSession.active_model = "audit"
            _host.PrimusSession.last_path = "audit"
            _host.PrimusSession.emit_think("Audit", "English from last actions — no JSON", "done")
            why = explain_recent()
            claimed = _OPERATOR_CLAIM_RE.sub("", why).strip()
            return (claimed or why), list(_host.PrimusSession.thinking_steps)
    except Exception:  # noqa: BLE001 — why-ask must never break routing
        pass

    # Fast paths still win for a single time/news/mail/count/knowledge step unless `my path:`.
    skip_fast = bool(
        path_ins.prefix_user
        or path_ins.extra_ask
        or (path_ins.toggle_on and path_ins.steps)
        or (current_path_mode() == "user_path" and len(path_ins.steps) > 1)
    )

    # --- Step 0: instant local answers (time/date) — no model, no RAG, no network ---
    if not skip_fast:
        instant = instant_answer(message)
        if instant is not None:
            _host.PrimusSession.active_agent = "primus"
            _host.PrimusSession.active_model = "instant"
            _host.PrimusSession.last_path = "instant"
            _host.PrimusSession.emit_think("Instant", "Answered locally — no model call", "done")
            return instant, list(_host.PrimusSession.thinking_steps)

    # --- Step 0.5: deterministic tool fast-path — run the obvious single tool now,
    #     skipping the planning LLM, RAG, and embeddings entirely (huge speed win). ---
    if not skip_fast:
        try:
            tool_ans = fast_tool_answer(message)
        except Exception as exc:  # noqa: BLE001 — never let the fast path break normal routing
            _host.log.warning("Fast-tool path error (%s) — falling through to full routing", exc)
            tool_ans = None
        if tool_ans is not None:
            _host.PrimusSession.active_agent = "primus"
            _host.PrimusSession.active_model = "fast-tool"
            _host.PrimusSession.last_path = "fast_tool"
            _host.PrimusSession.emit_think("Fast tool", "Done — direct tool, no model", "done")
            return tool_ans, list(_host.PrimusSession.thinking_steps)

    # --- user_path: execute the operator's listed order (after instant/fast_tool). ---
    if path_ins.toggle_off and not path_ins.steps and not path_ins.extra_ask:
        _host.PrimusSession.active_agent = "primus"
        _host.PrimusSession.active_model = "path-mode"
        _host.PrimusSession.last_path = "primus_path"
        _host.PrimusSession.emit_think("Path mode", "primus_path", "done")
        return "Back on primus_path — I'll decide the order.", list(_host.PrimusSession.thinking_steps)

    run_as_user = bool(
        path_ins.prefix_user
        or (path_ins.toggle_on and path_ins.steps)
        or (current_path_mode() == "user_path" and len(path_ins.steps) > 1)
        or has_file_write_steps(path_ins.steps)
        or has_list_steps(path_ins.steps)
    )
    if path_ins.toggle_on and not path_ins.steps and not path_ins.extra_ask:
        _host.PrimusSession.active_agent = "primus"
        _host.PrimusSession.active_model = "path-mode"
        _host.PrimusSession.last_path = "user_path"
        _host.PrimusSession.emit_think("Path mode", "user_path — empty", "done")
        return "What's the first step?", list(_host.PrimusSession.thinking_steps)

    if run_as_user and path_ins.steps:
        def _path_knowledge(step: str) -> str:
            try:
                ans, _used = _fast_chat_answer(
                    step,
                    history,
                    "",
                    mode,
                    prefer_main=True,
                    timeout_sec=max(45, int(_host.CFG.get("fast_chat_invoke_timeout_sec", 15))),
                )
                return ans
            except Exception:
                return CHAT_SLOW_REPLY

        _host.PrimusSession.active_agent = "primus"
        _host.PrimusSession.active_model = "user-path"
        _host.PrimusSession.last_path = "user_path"
        _host.PrimusSession.emit_think("Path mode", f"user_path · {len(path_ins.steps)} step(s)", "running")
        path_out = _run_user_path_with_packs(
            path_ins.steps, answer_knowledge=_path_knowledge, mode=mode,
        )
        if path_ins.extra_ask:
            extra = path_ins.extra_ask
            extra_out = None
            inst = instant_answer(extra)
            if inst is not None:
                extra_out = inst
            else:
                try:
                    extra_out = fast_tool_answer(extra)
                except Exception:
                    extra_out = None
            if extra_out is None and _KNOWLEDGE_Q_RE.search(extra):
                extra_out = _path_knowledge(extra)
            if extra_out is None:
                extra_out, _st = invoke_primus(
                    graphs, extra, history, mode,
                    force_forge=force_forge, force_primus=force_primus, scope=scope,
                )
            path_out = path_out + "\n\n" + extra_out
        _host.PrimusSession.emit_think("Path mode", "user_path done", "done")
        return path_out, list(_host.PrimusSession.thinking_steps)

    if not _host.HAS_AI_STACK:
        _host.PrimusSession.emit_think("Setup", "AI stack missing", "error")
        return _host._agent_unavailable_message(), list(_host.PrimusSession.thinking_steps)

    ok, health = _host.check_ollama_health()
    _host.PrimusSession.last_health = health
    if not ok:
        _host.PrimusSession.emit_think("Ollama", health, "error")
        return (
            f"**Primus paused** — {health}\n\n"
            "Start Ollama (`ollama serve`) and pull models, then retry.",
            list(_host.PrimusSession.thinking_steps),
        )

    # --- Step 1: deterministic routing + internal reasoning (no LLM) ---
    decision = ModelRouter.analyze(
        message, force_forge=force_forge, force_primus=force_primus
    )
    _host.PrimusSession.emit_think("Reasoning", decision.reasoning, "running")
    _host.PrimusSession.emit_think("Reasoning", "Route decided", "done")

    agent_id = decision.agent_id
    model_name = decision.model_name
    fallback_note = ""

    # --- Step 2: Forge availability / fallback ---
    if agent_id == "forge" and _host.CFG.get("forge_fallback_to_primus", True):
        if not is_forge_model_ready():
            fallback_note = (
                f"\n\n*(Forge model `{_host.CFG.get('forge_model', _host.FORGE_MODEL)}` not available — "
                f"using Primus `{_host.CFG.get('model', _host.DEFAULT_MODEL)}`. Run: "
                f"`ollama pull {_host.CFG.get('forge_model', _host.FORGE_MODEL)}`)*"
            )
            agent_id = "primus"
            model_name = _host.CFG.get("model", _host.DEFAULT_MODEL)
            _host.PrimusSession.emit_think(
                "Fallback",
                "Forge unavailable → Primus",
                "error",
            )

    _host.PrimusSession.active_agent = agent_id
    _host.PrimusSession.active_model = model_name
    _host.PrimusSession.delegation_reason = decision.reason
    agent_label = "Forge" if agent_id == "forge" else "Primus"

    # --- Smart prompt ingestion: one decisive router picks the cheapest correct path ---
    route = route_prompt(decision, message)
    _host.PrimusSession.last_path = route["path"]
    _host.PrimusSession.last_intent = decision.intent
    _host.PrimusSession.emit_think("Route", f"{route['path']} — {route['reason']}", "done")
    # Leftover receptionist: 3b JSON slots, internal only. Green gates + route_prompt
    # knowledge / pulse / forge / kb_answer / fast_coding already won — do not call those.
    # Missing model / timeout / bad JSON → current primus graph. need_brain may
    # call ask_grok (tool is in build_tools; governor + redact). Never print JSON.
    if route["path"] == "primus" and agent_id == "primus":
        try:
            from primus.core.receptionist import receive_leftover  # noqa: PLC0415

            slots = receive_leftover(message)
            if slots is not None:
                if slots.need_brain or slots.bucket == "heavy":
                    if "ask_grok" not in slots.tools:
                        slots.tools.append("ask_grok")
                    _host.PrimusSession.emit_think(
                        "Brain",
                        "need_brain — ask_grok available",
                        "done",
                    )
                _apply_turn_pack(
                    message, slots.tools,
                    need_brain=bool(slots.need_brain or slots.bucket == "heavy"),
                )
                _host.PrimusSession.emit_think(
                    "Receptionist",
                    f"{slots.bucket} · {len(slots.tools)} tool(s)",
                    "done",
                )
            else:
                _apply_turn_pack(message)
        except Exception as exc:  # noqa: BLE001 — receptionist must not fail the leftover graph
            _host.log.debug("Receptionist skipped: %s", exc)
            _apply_turn_pack(message)
    else:
        # Forge / direct still hit create_agent — bind a pack from the message.
        _apply_turn_pack(message)
    # Meta-learning: remember the chosen route so feedback can adapt future routing.
    try:
        _host.get_metrics().record_route(agent_id, decision.intent)
    except Exception:  # noqa: BLE001
        pass

    # --- Step 3: unified memory + RAG retrieval (BEFORE cache & graph — both agents) ---
    retries = int(_host.CFG.get("ollama_retries", 2))
    # Skip the embedding/RAG round-trip whenever the router says it isn't needed (speed).
    if not route["needs_rag"]:
        mem_block, mem_sources = "", []
        _host.log.debug("Retrieval skipped (path=%s): %s", route["path"], message[:60])
    else:
        mem_block, mem_sources = _host.build_memory_context(
            message, history=history, agent_id=agent_id, scope=scope
        )

    # Always prepend the short-term conversation focus (free, no embeddings) so Primus stays
    # on the current thread on EVERY path — fast chat included. Highest-priority context.
    focus_block = _host.build_conversation_focus(message, history)
    mem_block = focus_block + ("\n\n" + mem_block if mem_block else "")
    if _host.PrimusSession.current_topic:
        _host.PrimusSession.emit_think("Focus", f"Topic: {_host.PrimusSession.current_topic}", "done")
    _host.log.info(
        "Retrieval agent=%s sources=%d chars=%d",
        agent_id,
        len(mem_sources),
        len(mem_block),
    )
    if mem_sources:
        layers = {s.get("layer", "?") for s in mem_sources}
        _host.PrimusSession.emit_think(
            "Memory+RAG",
            f"{len(mem_sources)} hits ({', '.join(sorted(layers))}) — retrieval injected",
            "done",
        )
        conv_hits = sum(1 for s in mem_sources if s.get("layer") == "conversation")
        if conv_hits:
            _host.PrimusSession.emit_think(
                "Recall",
                f"{conv_hits} past conversation(s) — cite as [CONV-N] if used",
                "done",
            )
    else:
        _host.PrimusSession.emit_think(
            "Memory+RAG",
            "No KB/memory/conv hits — anti-hallucination: state lack of coverage",
            "pending",
        )

    _host.PrimusSession.emit_think(
        "Router",
        f"{agent_label} · {model_name} — {decision.reason} (intent={decision.intent})",
        "done",
    )

    # --- Step 4: response cache (Primus trivial only; never when retrieval has hits) ---
    if ModelRouter.cacheable(decision, message) and agent_id == "primus" and not mem_sources:
        cached = _response_cache.get(message, mode)
        if cached:
            _host.PrimusSession.emit_think("Cache", "Hit — skipped LLM", "done")
            return cached, list(_host.PrimusSession.thinking_steps)

    # --- Step 4.5: conversational tier — one bounded call, no tools, no graph ---------------
    # BINDING when route["no_plan"] is set. This used to be advisory: any hiccup here (small
    # model missing, cold model, slow first token) fell through to build_primus_graph, which
    # dutifully planned one step for a greeting and then reported plan progress and a timeout
    # as the visible reply. A chat turn now answers on the chat path or answers plainly —
    # it never escalates to a planner.
    if route["path"] in ("fast_chat", "kb_answer"):
        pulse = is_pulse_turn(message)
        kb_turn = route["path"] == "kb_answer"
        knowledge_turn = bool(route.get("knowledge"))
        try:
            _host.PrimusSession.emit_think(
                "Chat tier",
                "Knowledge question → retrieved answer (no plan)" if kb_turn
                else "Knowledge question → single call (no plan, no tools)" if knowledge_turn
                else "Conversational → single call (no plan)",
                "running",
            )
            kb_block = mem_block
            if kb_turn and kb_block:
                if _FETCH_FILE_RE.search(message):
                    fetch_reply = _kb_fetch_file_reply(kb_block)
                    if fetch_reply:
                        _host.PrimusSession.emit_think(
                            "Chat tier", "Handed over source file(s) → exports", "done"
                        )
                        return fetch_reply, list(_host.PrimusSession.thinking_steps)
                kb_block += (
                    "\n\nAnswer from the knowledge-base excerpts above. Cite the excerpt marker "
                    "with its number (e.g. [KB-1]) or the source filename for anything you use. "
                    "If the excerpts don't contain the answer, say so. If the user asked for the "
                    "file or document itself (give me / fetch / send me the file), do NOT paste "
                    "its contents — give a one-line summary plus the exact source path from the "
                    "excerpt's source= field."
                )
            answer, used_model = _fast_chat_answer(
                message,
                history,
                kb_block,
                mode,
                # Knowledge answers need the main model's fidelity (same as kb_turn); when the
                # fast model is missing _fast_chat_answer falls back to the main model anyway.
                prefer_main=kb_turn or knowledge_turn,
                # A cold model's first token can blow past the 15s chat bound — knowledge and
                # KB turns get the same longer leash so the answer actually arrives.
                timeout_sec=(
                    max(45, int(_host.CFG.get("fast_chat_invoke_timeout_sec", 15)))
                    if (kb_turn or knowledge_turn) else None
                ),
            )
            _host.PrimusSession.active_model = used_model
            _host.PrimusSession.emit_think("Chat tier", "Replied", "done")
            if not answer.strip():
                answer = pulse_reply(message) if pulse else BAD_FORMAT_REPLY
            if (
                bool(_host.CFG.get("show_model_badge_in_chat", False))
                and not answer.startswith("**Primus")
                and not answer.startswith("**Forge")
            ):
                answer = f"**Primus** · `{used_model}`\n\n" + answer
            if ModelRouter.cacheable(decision, message) and not mem_sources:
                _response_cache.put(message, mode, answer)
            return answer, list(_host.PrimusSession.thinking_steps)
        except Exception as exc:  # noqa: BLE001
            if route.get("no_plan"):
                # Forbidden from planning: one calm sentence (or the deterministic pulse
                # answer), never a partial-plan ledger for a turn that had no plan.
                # CHAT_TIMEOUT_REPLY is for work-graph overruns only — a question was never
                # going to change or send anything, so the chat tier says the one true thing.
                _host.log.warning("Chat tier failed (%s) — answering directly, no graph", exc)
                _host.PrimusSession.emit_think("Chat tier", "No graph on a chat turn", "error")
                answer = pulse_reply(message) if pulse else CHAT_SLOW_REPLY
                return answer, list(_host.PrimusSession.thinking_steps)
            _host.log.warning("Fast tier failed (%s) — falling back to full graph", exc)
            _host.PrimusSession.emit_think("Fast tier", "Fell back to full path", "error")

    # --- Step 4.6: fast coding tier — one focused coder-model call for small self-contained asks.
    #     Skips plan→step→summarize entirely (big latency win) while keeping Forge code quality. ---
    if route["path"] == "fast_coding" and agent_id == "forge" and is_forge_model_ready():
        try:
            enforce_sequential_models(_host.CFG.get("forge_model", _host.FORGE_MODEL), label="Forge")
            _host.PrimusSession.emit_think(
                "Fast coding", "Self-contained task → single Forge call (no graph)", "running"
            )
            answer, used_model = _fast_coding_answer(message, history, mem_block, mode)
            answer = _forge_autocorrect(message, answer, used_model)
            _host.PrimusSession.active_agent = "forge"
            _host.PrimusSession.active_model = used_model
            _host.PrimusSession.last_path = "fast_coding"
            _host.PrimusSession.emit_think("Fast coding", "Delivered", "done")
            if (
                bool(_host.CFG.get("show_model_badge_in_chat", False))
                and not answer.startswith(("**Forge", "**Primus"))
            ):
                answer = f"**Forge** · `{used_model}`\n\n" + answer
            # Sequential policy: release Forge + warm Primus back up in the background.
            if bool(_host.CFG.get("sequential_models", True)):
                threading.Thread(
                    target=release_forge_reload_primus, name="primus-seq-release", daemon=True
                ).start()
            return answer, list(_host.PrimusSession.thinking_steps)
        except Exception as exc:  # noqa: BLE001
            _host.log.warning("Fast coding tier failed (%s) — falling back to full Forge graph", exc)
            _host.PrimusSession.emit_think("Fast coding", "Fell back to full Forge path", "error")

    graph = graphs.get(agent_id) or graphs.get("primus")
    if graph is None:
        return "Agent graph unavailable.", list(_host.PrimusSession.thinking_steps)

    # --- Intelligent dynamic switching: use the small model for simple, slow-prone
    #     Primus turns (keeps all tools). Coding/Forge and deep turns keep the full model. ---
    try:
        if agent_id == "primus" and graphs.get("primus_fast") and DynamicModelManager.prefer_fast(decision):
            graph = graphs["primus_fast"]
            model_name = _host.CFG.get("fast_fallback_model") or _host.CFG.get("primus_fast_model")
            _host.PrimusSession.active_model = model_name
            _host.PrimusSession.emit_think("Dynamic model", f"Fast model → {model_name}", "done")
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Dynamic switch skipped: %s", exc)

    payload = {
        "input": message,
        "messages": _host.prepare_agent_messages(history),
        "plan": "",
        "steps": [],
        "step_index": 0,
        "step_results": [],
        "mode": mode,
        "final_answer": "",
        "rag_context": mem_block,
    }
    # Hard timeout for BOTH agents now (previously Primus ran uncapped and could hang the UI).
    # The thread-based wrapper also polls for Halt, so this bounds every model/tool chain.
    if agent_id == "forge":
        timeout = int(_host.CFG.get("forge_invoke_timeout_sec", 180))
    else:
        timeout = int(_host.CFG.get("primus_invoke_timeout_sec", 240))

    # --- Sequential model policy: evict the OTHER heavy model before running this one,
    #     so only one of Primus/Forge is resident (the light fast model stays as fallback). ---
    forge_engaged = agent_id == "forge"
    try:
        enforce_sequential_models(model_name, label=agent_label)
    except Exception as exc:  # noqa: BLE001 — model housekeeping must never break a turn
        _host.log.debug("Sequential enforce skipped: %s", exc)

    def _seq_cleanup() -> None:
        """If Forge ran, unload it and warm Primus back up (in background, no added latency)."""
        if forge_engaged and bool(_host.CFG.get("sequential_models", True)):
            threading.Thread(
                target=release_forge_reload_primus, name="primus-seq-release", daemon=True
            ).start()

    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            if timeout > 0:
                result = _invoke_graph_with_timeout(graph, payload, timeout)
            else:
                result = graph.invoke(payload)
            answer = coerce_message_text(result.get("final_answer") or "")
            if not answer:
                msgs = result.get("messages") or []
                if msgs:
                    answer = coerce_message_text(msgs[-1].content)
            if not answer:
                if has_file_write_steps(path_ins.steps):
                    answer = "✗ Write did not run."
                else:
                    answer = "All taken care of — let me know where you want to go next."

            # Closed-loop verification: if Forge produced code, compile/lint it and self-repair once.
            if agent_id == "forge":
                answer = _forge_autocorrect(message, answer, model_name)

            if (
                bool(_host.CFG.get("show_model_badge_in_chat", False))
                and not answer.startswith("**Forge")
                and not answer.startswith("**Primus")
            ):
                answer = f"**{agent_label}** · `{model_name}`\n\n" + answer
            if fallback_note:
                answer += fallback_note

            if ModelRouter.cacheable(decision, message) and agent_id == "primus":
                _response_cache.put(message, mode, answer)

            _seq_cleanup()
            return answer, list(_host.PrimusSession.thinking_steps)
        except _host._TurnCancelled as exc:
            # User pressed Halt (or the watchdog fired). Return whatever was done — instantly.
            _host.log.info("Turn halted during %s graph (%s)", agent_id, exc)
            _host.PrimusSession.emit_think("Halted", "Stopped on request", "error")
            _seq_cleanup()
            if _is_chat_turn():
                return ("⏹ Stopped there. Ask again when you want it.",
                        list(_host.PrimusSession.thinking_steps))
            partial = _partial_answer_from_thinking(
                "Halted — stopped at your request. Ask me to pick it up again when you're ready."
            )
            return ("⏹ " + partial, list(_host.PrimusSession.thinking_steps))
        except TimeoutError as exc:
            last_exc = exc
            if agent_id == "forge" and _host.CFG.get("forge_fallback_to_primus", True):
                _host.log.warning("Forge timeout: %s — falling back to Primus", exc)
                _host.PrimusSession.emit_think("Fallback", "Forge slow → Primus", "error")
                agent_id = "primus"
                model_name = _host.CFG.get("model", _host.DEFAULT_MODEL)
                graph = graphs.get("primus")
                _host.PrimusSession.active_agent = agent_id
                _host.PrimusSession.active_model = model_name
                mem_block, mem_sources = _host.build_memory_context(
                    message, history=history, agent_id=agent_id, scope=scope
                )
                payload["rag_context"] = mem_block
                fallback_note = "\n\n*(Forge timed out — completed on Primus.)*"
                # Keep the fallback Primus run bounded too, so we never hang.
                timeout = int(_host.CFG.get("primus_invoke_timeout_sec", 240))
                continue
            # Primus itself timed out (or Forge with no fallback): return a graceful partial.
            _host.log.warning("%s timeout: %s — returning partial answer", agent_id, exc)
            _host.PrimusSession.emit_think("Timeout", "Returning partial answer", "error")
            _seq_cleanup()
            if _is_chat_turn():
                return (CHAT_TIMEOUT_REPLY, list(_host.PrimusSession.thinking_steps))
            partial = _partial_answer_from_thinking(
                "That's how far I got before the time cap — everything above is done and real."
            )
            return (partial, list(_host.PrimusSession.thinking_steps))
        except Exception as exc:
            last_exc = exc
            _host.log.exception("Graph error (%s attempt %s/%s)", agent_id, attempt + 1, retries + 1)
            if (
                agent_id == "forge"
                and attempt == 0
                and _host.CFG.get("forge_fallback_to_primus", True)
                and graphs.get("primus")
            ):
                _host.PrimusSession.emit_think("Fallback", f"Forge error → Primus", "error")
                agent_id = "primus"
                model_name = _host.CFG.get("model", _host.DEFAULT_MODEL)
                graph = graphs["primus"]
                _host.PrimusSession.active_agent = agent_id
                _host.PrimusSession.active_model = model_name
                mem_block, mem_sources = _host.build_memory_context(
                    message, history=history, agent_id=agent_id, scope=scope
                )
                payload["rag_context"] = mem_block
                fallback_note = f"\n\n*(Forge failed — fallback to Primus: `{exc}`)*"
                timeout = int(_host.CFG.get("primus_invoke_timeout_sec", 240))
                continue
            if attempt < retries:
                _host.PrimusSession.emit_think("Recovery", f"Retry {attempt + 2}…", "running")
                continue

    err = str(last_exc) if last_exc else "unknown error"
    _host.PrimusSession.emit_think("Error", err[:200], "error")
    _seq_cleanup()
    return (
        f"Primus hit an error but remains online.\n\n`{err[:500]}`\n\n"
        "Try a simpler request, check Ollama, or restart Primus.",
        list(_host.PrimusSession.thinking_steps),
    )


def build_thought_messages(steps: list[dict]) -> list[dict]:
    """Gradio chat thought bubbles — metadata.status must be 'pending' or 'done' only."""
    out: list[dict[str, Any]] = []
    for s in steps:
        if s.get("title") in ("Received",):
            continue
        status = _host.normalize_gradio_status(s.get("status", "done"))
        title = str(s.get("title", "Thinking"))
        detail = str(s.get("detail", ""))
        content = detail if detail else title
        if status == "pending" and title and detail and title != detail:
            content = f"**{title}** — {detail}"
        msg = _host.sanitize_chat_message(
            {
                "role": "assistant",
                "content": content,
                "metadata": {
                    "title": title,
                    "status": status,
                    "log": s.get("ts", ""),
                },
            }
        )
        if msg:
            out.append(msg)
    return out


# === Background + scheduled agent systems ===
# ---------------------------------------------------------------------------
# Background agent system
#
# Spins off long-running tasks on the heavy (Forge) model in daemon threads. Each task
# runs its OWN isolated ReAct executor (create_agent + one pack from the job text) and
# deliberately does NOT touch PrimusSession turn/thinking/cancel state, so the main chat
# stays responsive and uncorrupted. Tasks are not bound by the chat's streaming timeout.
# ---------------------------------------------------------------------------

_BG_STATUS_ICONS = {
    "queued": "◷", "running": "▶", "completed": "✓", "failed": "✗", "cancelled": "⊘",
}


# ---------------------------------------------------------------------------
# Background task completion-gate
#
# Root cause of "stalls after 1-2 tool calls": a background task runs as ONE ReAct pass that
# self-terminates the moment the model emits a message with no tool call. Small local models
# tend to do their exploratory calls (read_own_code + search_own_codebase) and then write an
# "analysis" — never reaching propose_code_change / read_office_file / verify. No amount of
# prompt text reliably prevents that, because stopping is a valid ReAct transition.
#
# Fix: detect the task KIND up front, inject an explicit execution contract, and — after the
# agent stops — check whether the mandatory finishing tool actually ran. If not, force a bounded
# continuation on the SAME thread (the checkpointer keeps context, so it resumes rather than
# restarting). Non-gated tasks behave exactly as before (single pass).
# ---------------------------------------------------------------------------

_SELF_IMPROVE_RE = re.compile(
    r"\b(self[- ]?improve|improve\s+yourself|improve\s+your\s+(?:own\s+)?code|edit\s+your\s+(?:own\s+)?code|"
    r"refactor\s+your|fix\s+(?:the\s+|this\s+|a\s+|your\s+)?(?:\w+\s+){0,3}bug|fix\s+yourself|"
    r"propose_code_change|propose\s+a\s+(?:code\s+)?change|apply\s+(?:a\s+)?(?:fix|patch)|patch\s+your)\b",
    re.IGNORECASE,
)
_DOC_SUMMARY_RE = re.compile(
    r"(summari[sz]e|summary\s+of|digest|extract|go\s+through|read)\b.{0,60}"
    r"\b(pdf|pdfs|document|documents|downloads?|ingested|uploaded|report|file|files)\b"
    r"|\b(pdf|pdfs|documents?)\b.{0,60}(summari[sz]e|summary|digest|extract)",
    re.IGNORECASE,
)
# Research: gather real external info. Ingest: acquire web content INTO the knowledge base.
_RESEARCH_RE = re.compile(
    r"\b(research|deep[- ]?dive|deep[- ]?research|find\s+(?:out|sources|articles|papers|studies)|"
    r"look\s+up|latest\s+on|gather\s+(?:info|sources|data|evidence)|survey\s+the|literature\s+review)\b",
    re.IGNORECASE,
)
_INGEST_RE = re.compile(
    r"\b(ingest\b|download\b|save\s+(?:it\s+|them\s+|the\s+\w+\s+)?(?:to|into)\s+(?:the\s+)?(?:kb|knowledge)|"
    r"add\s+(?:to|into)\s+(?:the\s+)?(?:kb|knowledge\s+base)|build\s+(?:a\s+)?knowledge\s+base|"
    r"pull\s+(?:down\s+)?(?:the\s+)?(?:pdf|pdfs|papers?|docs?))\b",
    re.IGNORECASE,
)

# Placeholder/example URLs — a real research or ingestion result must NEVER cite these. Detecting
# them lets the completion-gate reject "downloaded example.com" and force a real search + action.
_PLACEHOLDER_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:example\.(?:com|org|net|edu)|examples?\.\w+|placeholder[\w.-]*|"
    r"your-?(?:site|domain|url|website)[\w.-]*|yourdomain[\w.-]*|test\.(?:com|example)|"
    r"(?:some|the)-?(?:site|url|page)\.\w+|domain\.(?:com|tld)|url\.here|link\.here)",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s)>\]\"'}]+")


def _has_placeholder_url(text: str) -> bool:
    """True if the text contains an obvious placeholder/example URL (never valid for real research)."""
    return bool(_PLACEHOLDER_URL_RE.search(text or ""))


def _extract_real_urls(text: str) -> list[str]:
    """Pull concrete, non-placeholder http(s) URLs out of agent output (for progress persistence)."""
    out: list[str] = []
    for u in _URL_RE.findall(text or ""):
        u = u.rstrip(".,;:)]}\"'")
        if u and not _PLACEHOLDER_URL_RE.search(u) and u not in out:
            out.append(u)
    return out


def _detect_bg_task_kind(task: str) -> dict[str, bool]:
    """Classify a background goal so we can enforce the right finishing action. Cheap, no LLM."""
    low = (task or "").lower()
    return {
        "self_improve": bool(_SELF_IMPROVE_RE.search(low)),
        "doc_summary": bool(_DOC_SUMMARY_RE.search(low)),
        "research": bool(_RESEARCH_RE.search(low)),
        "ingest": bool(_INGEST_RE.search(low)),
    }


def _bg_task_directive(kind: dict[str, bool]) -> tuple[str, set[str]]:
    """Return (execution_contract_text, mandatory_tool_names) for a detected kind. ("", set()) if none.

    The mandatory tool set is what the completion-gate checks actually ran before accepting "done".
    """
    if kind.get("self_improve"):
        directive = (
            "\n\n[MANDATORY EXECUTION CONTRACT — self-improvement]\n"
            "This is a multi-step task. Do NOT stop after reading or analyzing code.\n"
            "1) Locate the code: read_own_code / search_own_codebase (do this ONCE, then move on).\n"
            "2) REQUIRED: call propose_code_change(description, find, replace) with the EXACT existing\n"
            "   text to change (or propose_new_tool for a brand-new capability). Analysis is NOT the deliverable.\n"
            "3) REQUIRED: call verify_python on the changed file/snippet.\n"
            "You are NOT finished until propose_code_change (or propose_new_tool) has been called. If you\n"
            "already searched the codebase, your NEXT action is propose_code_change — do not re-analyze."
        )
        return directive, {"propose_code_change", "propose_new_tool"}
    if kind.get("doc_summary"):
        directive = (
            "\n\n[DOCUMENT TASK — correct tools + paths]\n"
            f"Downloaded/ingested docs live in {_host.KB_DOWNLOADS_DIR} and uploads in {_host.KB_UPLOADS_DIR}.\n"
            "REQUIRED: list that folder (terminal 'ls'), then READ each PDF/office file with\n"
            "read_office_file(path). Do NOT use batch_file_operations to merely list files as a substitute\n"
            "for reading them. Summarize the actual extracted text and cite each file by name."
        )
        return directive, {"read_office_file"}
    if kind.get("ingest"):
        directive = (
            "\n\n[INGESTION TASK — mandatory sequence, real sources only]\n"
            "Stay STRICTLY on the exact topic requested above — never drift to 'current events' or an\n"
            "unrelated subject. Complete this full sequence, do not stop after searching:\n"
            "1) SEARCH for real resources on the exact topic: web_search or deep_web_search. Copy the\n"
            "   ACTUAL result URLs — never invent or use example.com / placeholder links.\n"
            "2) SELECT the 3–5 best real URLs from those results.\n"
            "3) DOWNLOAD each: download_pdf_from_url for a hosted PDF, else download_webpage_as_pdf.\n"
            "4) INGEST each downloaded source with ingest_web_document(url).\n"
            "5) SUMMARIZE exactly what was added (titles + real URLs). Do NOT use batch_file_operations to\n"
            "   'list' as a stand-in for ingesting. If you have no concrete URL yet, your NEXT action is a search."
        )
        return directive, {"ingest_web_document", "download_webpage_as_pdf", "download_pdf_from_url"}
    if kind.get("research"):
        directive = (
            "\n\n[RESEARCH TASK — stay on topic, read real sources]\n"
            "Stay STRICTLY on the exact topic requested above — never drift to 'current events' or an\n"
            "unrelated subject. Do not stop after a single search:\n"
            "1) SEARCH for real sources on the exact topic: web_search / deep_web_search / deep_research.\n"
            "   Use the ACTUAL result URLs — never example.com or made-up links.\n"
            "2) READ at least one of the best real URLs with read_article(url) (or research_topic for a full\n"
            "   cycle) — searching alone is NOT enough.\n"
            "3) Summarize findings WITH the real source URLs cited. Offer to ingest the best sources.\n"
            "Do not answer from memory alone."
        )
        # Gate requires actually CONSUMING a real source (read/deep-research), not merely searching.
        return directive, {
            "read_article", "research_topic", "deep_research",
            "ingest_web_document", "download_webpage_as_pdf", "download_pdf_from_url",
        }
    return "", set()


def _forced_continuation_msg(kind: dict[str, bool], goal: str = "") -> str:
    """The 'you stopped early — do the required action NOW' nudge sent when the gate isn't satisfied.

    `goal` (the original request) is restated for research/ingest tasks so the agent stays anchored to
    the exact topic instead of drifting to current events or an unrelated subject on the retry.
    """
    anchor = (
        f" Stay STRICTLY on the original topic — do not change the subject: \"{goal.strip()[:200]}\"."
        if goal.strip() else ""
    )
    if kind.get("self_improve"):
        return (
            "You stopped early — the required change was never staged. Do NOT re-read or re-analyze the "
            "code. Your NEXT action MUST be a call to propose_code_change(description, find, replace) with "
            "the exact existing text to change (or propose_new_tool). After it is staged, call verify_python. "
            "Do it now."
        )
    if kind.get("doc_summary"):
        return (
            "You stopped early — the documents were never actually read. Your NEXT action MUST be to list "
            f"{_host.KB_DOWNLOADS_DIR} and call read_office_file(path) on each PDF/office file, then summarize "
            "the extracted text. Do it now — do not just list files."
        )
    if kind.get("ingest"):
        return (
            "You stopped before actually downloading + ingesting a real source (or used a placeholder/example "
            "URL). Do a REAL web_search / deep_web_search now, pick 3–5 concrete result URLs (never "
            "example.com), then call download_pdf_from_url or download_webpage_as_pdf on each, followed by "
            "ingest_web_document(url). Finish by summarizing what was added." + anchor
        )
    if kind.get("research"):
        return (
            "You stopped after searching (or used a placeholder/example URL) without reading a real source. "
            "Do a REAL web_search / deep_web_search now, then read_article(url) on the actual result URLs "
            "(never example.com), and summarize with the real sources cited." + anchor
        )
    return "Continue and complete the remaining required steps now. Do not restart from the beginning."


def _stream_agent_turn(
    agent: Any, payload: dict[str, Any], cfg: dict[str, Any], on_update: Callable[[str], None]
) -> tuple[str, set[str], list[str]]:
    """Run one agent turn; return (last_text, tools_used, discovered_urls). Streams to on_update.

    `tools_used` collects the name of every tool actually invoked (the completion-gate uses it to
    tell a genuine finish from a premature stop). `discovered_urls` collects concrete, non-placeholder
    URLs seen in any message this turn, so a resumed task can act on real links instead of re-searching.
    """
    last_text = ""
    tools_used: set[str] = set()
    urls: list[str] = []
    try:
        for chunk in agent.stream(payload, config=cfg, stream_mode="updates"):
            for node_out in (chunk or {}).values():
                for m in (node_out or {}).get("messages", []) or []:
                    # AIMessage.tool_calls → tools the model requested this step.
                    for tc in (getattr(m, "tool_calls", None) or []):
                        nm = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                        if nm:
                            tools_used.add(str(nm))
                    # ToolMessage carries the producing tool's name (belt-and-suspenders).
                    if getattr(m, "type", "") == "tool" and getattr(m, "name", ""):
                        tools_used.add(str(m.name))
                    txt = coerce_message_text(getattr(m, "content", "")).strip()
                    if txt:
                        last_text = txt
                        for u in _extract_real_urls(txt):  # remember real source URLs for resume
                            if u not in urls:
                                urls.append(u)
                        on_update(txt[:200])
    except TypeError:
        # Older agent object without streaming support — single invoke fallback (behavior preserved).
        result = agent.invoke(payload, config=cfg)
        last_text = coerce_message_text(result["messages"][-1].content)
        urls = _extract_real_urls(last_text)
    return last_text, tools_used, urls


def _background_bind(task: str) -> tuple[list, str, list[str]]:
    """Pack + extras from the job text. Never the full 120. No receptionist."""
    from primus.core.tool_packs import (  # noqa: PLC0415
        bind_tools,
        extras_from_named,
        named_tools_in_text,
        select_pack,
    )

    pack = select_pack(task)
    extra, refuse = extras_from_named(task, pack)
    extra = [] if refuse else list(extra)
    named = named_tools_in_text(task)
    want_brain = pack == "brain" or "ask_grok" in extra or "ask_grok" in named
    if want_brain and "ask_grok" not in extra:
        extra.append("ask_grok")
    registry = _host.build_tools()
    bound = bind_tools(registry, pack, extra)
    if not want_brain:
        bound = [t for t in bound if getattr(t, "name", "") != "ask_grok"]
    if not bound:
        bound = bind_tools(registry, "default")
        bound = [t for t in bound if getattr(t, "name", "") != "ask_grok"]
        pack = "default"
        extra = []
    return bound, pack, extra


def _packed_background_agent(llm: Any, task: str) -> Any:
    """``create_agent`` for a background/scheduled job. Cached like ``_packed_step_agent``."""
    from primus.core.tool_packs import bound_names  # noqa: PLC0415

    bound, pack, extra = _background_bind(task)
    extra_t = tuple(extra)
    key = (id(llm), pack, extra_t, "background")
    with _PACK_AGENT_LOCK:
        cached = _PACK_AGENT_CACHE.get(key)
        if cached is not None:
            return cached
    names = bound_names(bound)
    _host.log.info("background create_agent pack=%s bound=%s", pack, names)
    agent = create_agent(
        llm,
        bound,
        system_prompt=None,
        middleware=_agent_middleware(bound),
        checkpointer=_get_checkpointer(),
        name=f"background-react-{pack}",
    )
    with _PACK_AGENT_LOCK:
        _PACK_AGENT_CACHE[key] = agent
    return agent


def _run_background_agent(
    task: str,
    on_update: Callable[[str], None],
    scope: str = "",
    *,
    on_state: Optional[Callable[[dict[str, Any]], None]] = None,
    task_memory: Optional[dict[str, Any]] = None,
) -> str:
    """Execute one task with an isolated Forge ReAct agent bound to one pack.

    Self-contained: builds its own LLM + agent and reuses FORGE_SYSTEM (unchanged) as the
    system prompt. Streams intermediate steps to `on_update` for live progress. Never reads
    or writes PrimusSession turn state. `scope` (optional) is a project-chat hint prepended to
    the task prompt to bias the agent toward that project's context.

    For multi-step goals (self-improvement / document summarization) a completion-gate forces the
    agent to actually reach the mandatory tool (propose_code_change / read_office_file) instead of
    stopping after exploratory calls. `on_state` (optional) receives lightweight task-memory
    updates (tools used, partial result, step) for persistence; `task_memory` (optional) resumes a
    previous run from its last step instead of restarting.
    """
    if not _host.HAS_AI_STACK:
        return "AI stack not installed — background agents need Ollama + LangChain."
    model_name = _host.CFG.get("forge_model", _host.FORGE_MODEL)
    try:
        enforce_sequential_models(model_name, label="Background")
    except Exception:  # noqa: BLE001
        pass
    llm = make_chat_ollama(
        model_name,
        temperature=float(_host.CFG.get("forge_temperature", 0.2)),
        num_predict=int(_host.CFG.get("forge_num_predict", 2048)),
    )
    agent = _packed_background_agent(llm, task)
    kind = _detect_bg_task_kind(task)
    directive, mandatory = _bg_task_directive(kind)

    # Continuation preamble when resuming a persisted task: reload the goal + last step so the
    # agent picks up where it left off instead of re-running the whole analysis.
    resume_note = ""
    if task_memory:
        prior_tools = ", ".join(task_memory.get("tools_used") or []) or "(none recorded)"
        resume_note = (
            "\n\n[CONTINUATION] This is a RESUMED task — the goal above is unchanged. Already-used "
            f"tools: {prior_tools}. Pick up from the last completed step; do NOT restart the analysis.\n"
            f"Last progress: {(task_memory.get('partial_result') or '')[:400]}"
        )
        # Hand back the REAL URLs found last time so a research/ingest resume acts on them directly.
        prior_urls = task_memory.get("discovered_urls") or []
        if prior_urls:
            resume_note += (
                "\nReal source URLs already found (act on THESE concrete links — do not re-search "
                "from scratch, and never substitute example.com): " + ", ".join(prior_urls[:8])
            )

    system = format_agent_system(
        FORGE_SYSTEM,
        {
            "mode": "execute",
            "memory": _host.memory_context_block(),
            "rag_context": "(Background task — gather context with your tools as needed.)",
            "plan": "",
            "step_num": "1", "step_total": "1",
            "current_step": task,
            "today": datetime.now().strftime("%A, %Y-%m-%d %H:%M"),
        },
    )
    user_content = (f"[{scope}]\n\n{task}" if scope else task) + directive + resume_note
    cfg = {"configurable": {"thread_id": f"bg-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 60}
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=user_content)]

    last_text = ""
    tools_used: set[str] = set()
    discovered_urls: list[str] = list(task_memory.get("discovered_urls") or []) if task_memory else []
    # Research/ingest results must be built from REAL sources: a placeholder/example URL in the
    # output means the gate is NOT satisfied even if a mandatory tool "ran" (it ran on a fake URL).
    needs_real_sources = bool(kind.get("research") or kind.get("ingest"))
    # Gated tasks get up to 3 passes to reach their mandatory tool; everything else is single-pass.
    max_rounds = 3 if mandatory else 1
    for round_i in range(max_rounds):
        text, used, urls = _stream_agent_turn(agent, {"messages": messages}, cfg, on_update)
        if text:
            last_text = text
        tools_used |= used
        for u in urls:
            if u not in discovered_urls:
                discovered_urls.append(u)
        if on_state:  # persist lightweight task-memory so a later "continue" can resume real work
            try:
                on_state({
                    "tools_used": sorted(tools_used),
                    "partial_result": last_text[:1000],
                    "current_step": f"round {round_i + 1}/{max_rounds}",
                    "discovered_urls": discovered_urls[:20],
                })
            except Exception:  # noqa: BLE001
                pass
        # Completion-gate: satisfied when no mandatory tool is required, OR a mandatory tool ran AND
        # (for research/ingest) the output isn't leaning on a placeholder/example URL.
        gate_ok = (not mandatory) or (
            bool(tools_used & mandatory)
            and not (needs_real_sources and _has_placeholder_url(last_text))
        )
        if gate_ok:
            break
        # Premature/placeholder stop → force the required real action and continue on the SAME thread,
        # restating the original topic so the retry can't drift to an unrelated subject.
        reason = "real sources" if needs_real_sources else (" / ".join(sorted(mandatory)))
        on_update(f"↻ Enforcing next step: {reason}")
        messages = [HumanMessage(content=_forced_continuation_msg(kind, goal=task))]

    # Be honest if the mandatory step still never happened (rather than reporting a false success).
    if mandatory and not (tools_used & mandatory):
        last_text = (last_text or "").rstrip() + (
            f"\n\n⚠ This run did not reach the required {' / '.join(sorted(mandatory))} step "
            "(the model kept stopping early). Say 'continue' / `/continue` to resume from here."
        )
    return last_text or "Completed with no textual output."


class BackgroundAgentManager:
    """Registry + scheduler for background tasks (in-memory, daemon threads)."""

    _tasks: dict[str, dict[str, Any]] = {}
    _callbacks: dict[str, Callable[[dict[str, Any]], None]] = {}
    _lock = threading.Lock()
    _MAX_KEEP = 30

    @classmethod
    def submit(
        cls,
        task: str,
        *,
        scope: str = "",
        meta: Optional[dict[str, Any]] = None,
        on_done: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        """Run a task in the background. Optional `scope` biases recall; `meta` tags the entry
        (e.g. scheduled task info); `on_done(entry)` fires after completion/failure. All optional —
        existing callers behave exactly as before."""
        task = (task or "").strip()
        if not task:
            return {}
        tid = "bg-" + uuid.uuid4().hex[:8]
        entry = {
            "id": tid, "task": task, "status": "queued",
            "last_update": "Queued…", "result": "",
            "scope": scope, "meta": dict(meta or {}),
            "created": time.time(), "started": 0.0, "finished": 0.0,
        }
        with cls._lock:
            cls._tasks[tid] = entry
            if on_done:
                cls._callbacks[tid] = on_done
            # Trim oldest finished tasks so the registry stays small.
            if len(cls._tasks) > cls._MAX_KEEP:
                for k in sorted(cls._tasks, key=lambda x: cls._tasks[x]["created"])[: len(cls._tasks) - cls._MAX_KEEP]:
                    if cls._tasks[k]["status"] in {"completed", "failed", "cancelled"}:
                        cls._tasks.pop(k, None)
                        cls._callbacks.pop(k, None)
        threading.Thread(target=cls._worker, args=(tid,), name=f"primus-{tid}", daemon=True).start()
        _host.log.info("Background task submitted %s: %s", tid, task[:80])
        return entry

    @classmethod
    def _set(cls, tid: str, **fields: Any) -> None:
        with cls._lock:
            if tid in cls._tasks:
                cls._tasks[tid].update(fields)

    @classmethod
    def _worker(cls, tid: str) -> None:
        with cls._lock:
            entry = cls._tasks.get(tid)
        if not entry:
            return
        cls._set(tid, status="running", started=time.time(), last_update="Starting…")
        try:
            result = _run_background_agent(
                entry["task"], lambda msg: cls._set(tid, last_update=msg),
                scope=entry.get("scope", ""),
            )
            with cls._lock:
                if cls._tasks.get(tid, {}).get("status") == "cancelled":
                    cls._callbacks.pop(tid, None)
                    return
            cls._set(
                tid, status="completed", finished=time.time(),
                result=result, last_update=(result[:200] or "Done."),
            )
            _host.log.info("Background task completed %s", tid)
        except Exception as exc:  # noqa: BLE001
            _host.log.exception("Background task failed %s", tid)
            cls._set(tid, status="failed", finished=time.time(), last_update=f"Failed: {exc}", result=str(exc))
        # Fire the completion hook (used by the scheduler to write output documents).
        cb = cls._callbacks.pop(tid, None)
        if cb:
            try:
                cb(cls.get(tid))
            except Exception:  # noqa: BLE001
                _host.log.exception("Background on_done callback failed %s", tid)

    @classmethod
    def cancel(cls, tid: str) -> bool:
        """Cooperative cancel: marks the task so its result is discarded (thread is daemon)."""
        with cls._lock:
            e = cls._tasks.get(tid)
            if e and e["status"] in {"queued", "running"}:
                e["status"] = "cancelled"
                e["finished"] = time.time()
                e["last_update"] = "Cancelled by user."
                return True
        return False

    @classmethod
    def clear_finished(cls) -> int:
        with cls._lock:
            done = [k for k, v in cls._tasks.items() if v["status"] in {"completed", "failed", "cancelled"}]
            for k in done:
                cls._tasks.pop(k, None)
        return len(done)

    @classmethod
    def list(cls) -> list[dict[str, Any]]:
        with cls._lock:
            return sorted((dict(v) for v in cls._tasks.values()), key=lambda x: x["created"], reverse=True)

    @classmethod
    def get(cls, tid: str) -> dict[str, Any]:
        with cls._lock:
            return dict(cls._tasks.get(tid, {}))

    @classmethod
    def active_count(cls) -> int:
        with cls._lock:
            return sum(1 for v in cls._tasks.values() if v["status"] in {"queued", "running"})


def _bg_age(entry: dict[str, Any]) -> str:
    end = entry.get("finished") or time.time()
    secs = max(0, int(end - (entry.get("created") or end)))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    return f"{secs // 3600}h{(secs % 3600) // 60}m"


def _bg_esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_background_agents_html() -> str:
    """Compact monitor for Menu → Activity: one row per task with status + last progress."""
    tasks = BackgroundAgentManager.list()
    if not tasks:
        return '<div class="bg-panel empty">No background agents running. Use <code>/bg &lt;task&gt;</code>.</div>'
    rows = []
    for t in tasks[:8]:
        icon = _BG_STATUS_ICONS.get(t["status"], "•")
        title = _bg_esc(t["task"][:70])
        last = _bg_esc((t.get("last_update") or "")[:120])
        rows.append(
            f'<div class="bg-task {t["status"]}">'
            f'<span class="bg-status">{icon} {t["status"]}</span>'
            f'<span class="bg-age">{_bg_age(t)}</span>'
            f'<div class="bg-title">{title}</div>'
            f'<div class="bg-last">{last}</div>'
            f"</div>"
        )
    return '<div class="bg-panel">' + "".join(rows) + "</div>"


# ---------------------------------------------------------------------------
# Automatic background-task system
#
# Detects long/heavy requests up front and runs them in isolated, bounded-concurrency
# daemon threads (reusing _run_background_agent → Forge + a packed bind) so the main chat
# turn NEVER blocks or times out. Tasks persist to background_tasks.json so they survive
# restarts (running tasks resume as 'interrupted'). On completion the result is injected
# back into the originating chat's history via the existing save pattern.
#
# This sits alongside BackgroundAgentManager (the explicit /bg + scheduled flow) but owns
# the auto-detected lifecycle, persistence, and chat result-injection the user asked for.
# ---------------------------------------------------------------------------

_BG_TASK_ICONS = {
    "queued": "◷", "running": "▶", "completed": "✓", "failed": "✗",
    "cancelled": "⊘", "interrupted": "⚠",
}

# Heuristic phrases that strongly signal a heavy, multi-step, or long-running request.
_LONG_TASK_PATTERNS = re.compile(
    r"\b("
    r"deep(?:\s|-)?(?:dive|research)|research\s+(?:this\s+)?(?:deeply|thoroughly|in\s+depth)|"
    r"deeply|thorough(?:ly)?|comprehensive(?:ly)?|exhaustive|in\s+depth|end[- ]to[- ]end|"
    r"full\s+(?:report|analysis|automation|app|application|project|audit|review|breakdown)|"
    r"complete\s+(?:report|analysis|app|application|project|guide|overhaul)|"
    r"build\s+(?:a\s+|an\s+|the\s+)?(?:full|complete|entire|whole)|"
    r"analy[sz]e\s+(?:all|every|the\s+entire|the\s+whole)|"
    r"(?:go\s+through|review|scan|process)\s+(?:all|every|the\s+entire)|"
    r"all\s+(?:the\s+)?files\s+in|every\s+file|entire\s+(?:folder|directory|codebase|repo)|"
    r"generate\s+a\s+(?:complete|full|comprehensive|detailed)|"
    r"take\s+your\s+time|overnight|when\s+you\s+have\s+time|in\s+the\s+background|as\s+a\s+background"
    r")\b",
    re.IGNORECASE,
)

# Heavy action verbs that, combined with broad scope, suggest a long job.
_HEAVY_VERBS = re.compile(
    r"\b(build|create|generate|analy[sz]e|research|refactor|migrate|audit|scrape|compile|"
    r"implement|design|automate|summari[sz]e)\b", re.IGNORECASE)
_BROAD_SCOPE = re.compile(
    r"\b(all|every|entire|whole|multiple|many|several|complete|full|across|each)\b", re.IGNORECASE)


def _bg_task_cfg(key: str, default: Any) -> Any:
    """Read a key from the CFG['background_tasks'] section with a safe default."""
    section = _host.CFG.get("background_tasks")
    if isinstance(section, dict) and key in section:
        return section[key]
    return default


def classify_long_task(message: str, *, route_agent: str = "") -> tuple[bool, str]:
    """Decide whether a request should run as a background task. Returns (is_long, task_type).

    Deterministic + cheap (no LLM call): explicit phrasing ("research deeply", "build a full app",
    "analyze all files"), or a heavy action verb combined with broad scope on a sufficiently large
    request. Conservative by design — short questions and quick asks stay in the live chat.
    """
    if not _bg_task_cfg("auto_background_long_tasks", True):
        return False, ""
    text = (message or "").strip()
    if not text or text.startswith("/"):
        return False, ""
    low = text.lower()

    # Never background a question that's clearly meant to be answered right now.
    quick_signals = ("?", "what is", "who is", "when ", "how do i", "quick", "real quick", "tldr")
    is_quick = text.endswith("?") or any(s in low for s in quick_signals)

    explicit = bool(_LONG_TASK_PATTERNS.search(low))
    min_chars = int(_bg_task_cfg("min_chars_for_auto", 120))
    heavy_combo = (
        len(text) >= min_chars
        and bool(_HEAVY_VERBS.search(low))
        and bool(_BROAD_SCOPE.search(low))
    )
    # Forge-bound work with broad scope on a long message is also a strong signal.
    forge_combo = (
        route_agent == "forge"
        and len(text) >= min_chars
        and bool(_BROAD_SCOPE.search(low))
    )

    if explicit and not (is_quick and len(text) < min_chars):
        ttype = _long_task_type(low)
        return True, ttype
    if heavy_combo or forge_combo:
        return True, _long_task_type(low)
    return False, ""


def _long_task_type(low: str) -> str:
    """Coarse task type from the request text (for display/grouping)."""
    if re.search(r"research|web|sources|news", low):
        return "research"
    if re.search(r"build|app|project|code|refactor|implement|script", low):
        return "build"
    if re.search(r"analy[sz]e|audit|review|files|folder|codebase|report", low):
        return "analysis"
    if re.search(r"automat", low):
        return "automation"
    return "general"


class BackgroundTaskManager:
    """Persistent, bounded-concurrency manager for auto-detected / queued long tasks.

    Each task runs in its own daemon thread but only executes once it acquires a slot from a
    bounded semaphore (config: max_concurrent_background_tasks), so heavy work can't overwhelm
    the machine. Tasks persist to background_tasks.json across restarts. On completion the full
    output is written to background_results/ and a concise result is injected into the chat that
    created it. Cooperative cancellation (daemon threads can't be force-killed) discards results.
    """

    _lock = threading.Lock()
    _sem: Optional[threading.BoundedSemaphore] = None
    # chat_path -> list of assistant messages waiting to be merged into that live chat.
    _pending_msgs: dict[str, list[dict[str, Any]]] = {}
    _dirty_chats: set[str] = set()

    # ---- persistence -------------------------------------------------------
    @classmethod
    def _load(cls) -> list[dict[str, Any]]:
        try:
            data = json.loads(_host.BACKGROUND_TASKS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    @classmethod
    def _save(cls, tasks: list[dict[str, Any]]) -> None:
        try:
            _host.ensure_app_dirs()
            keep = int(_bg_task_cfg("keep_tasks", 100))
            _host.BACKGROUND_TASKS_FILE.write_text(json.dumps(tasks[-keep:], indent=2), encoding="utf-8")
        except OSError as exc:
            _host.log.debug("Could not save background tasks: %s", exc)

    @classmethod
    def _update(cls, tid: str, **fields: Any) -> None:
        with cls._lock:
            tasks = cls._load()
            for t in tasks:
                if t.get("id") == tid:
                    t.update(fields)
                    break
            cls._save(tasks)

    @classmethod
    def _semaphore(cls) -> threading.BoundedSemaphore:
        if cls._sem is None:
            n = max(1, int(_bg_task_cfg("max_concurrent_background_tasks", 2)))
            cls._sem = threading.BoundedSemaphore(n)
        return cls._sem

    # ---- submission --------------------------------------------------------
    @classmethod
    def submit(
        cls,
        user_request: str,
        *,
        description: str = "",
        task_type: str = "general",
        scope: str = "",
        chat_path: str = "",
        mode: str = "execute",
        task_memory: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Queue a task and start its worker thread. Returns the persisted task entry.

        `task_memory` (optional) carries a previous run's state (tools used / partial result / step)
        so a resumed task picks up instead of restarting — see `resume()`.
        """
        user_request = (user_request or "").strip()
        if not user_request:
            return {}
        tid = "task-" + uuid.uuid4().hex[:6]
        entry = {
            "id": tid,
            "user_request": user_request,
            # original_intent is the durable goal we reload on resume (user_request may be rewritten
            # by callers; original_intent stays the north star for "continue").
            "original_intent": user_request,
            "description": (description or user_request)[:200],
            "status": "queued",
            "task_type": task_type or "general",
            "scope": scope,
            "chat_path": chat_path,
            "mode": mode,
            "created_at": _host._now_iso(),
            "started_at": "",
            "completed_at": "",
            "last_update": "Queued…",
            "result_summary": "",
            "output_path": "",
            "error": "",
            # --- lightweight task memory (updated live by the worker; used by resume) ---
            "tools_used": (task_memory or {}).get("tools_used", []),
            "partial_result": (task_memory or {}).get("partial_result", ""),
            "discovered_urls": (task_memory or {}).get("discovered_urls", []),
            "current_step": "",
            "task_memory": task_memory or None,
        }
        with cls._lock:
            tasks = cls._load()
            tasks.append(entry)
            cls._save(tasks)
        threading.Thread(target=cls._worker, args=(tid,), name=f"primus-{tid}", daemon=True).start()
        _host.log.info("Background task queued %s (%s): %s", tid, task_type, user_request[:80])
        return entry

    # ---- execution ---------------------------------------------------------
    @classmethod
    def _status_of(cls, tid: str) -> str:
        with cls._lock:
            for t in cls._load():
                if t.get("id") == tid:
                    return t.get("status", "")
        return ""

    @classmethod
    def _worker(cls, tid: str) -> None:
        # Block here (as 'queued') until a concurrency slot frees up.
        with cls._semaphore():
            if cls._status_of(tid) == "cancelled":  # cancelled while waiting in the queue
                return
            with cls._lock:
                entry = next((t for t in cls._load() if t.get("id") == tid), None)
            if not entry:
                return
            cls._update(tid, status="running", started_at=_host._now_iso(), last_update="Starting…")
            try:
                result = _run_background_agent(
                    entry["user_request"],
                    lambda msg: cls._update(tid, last_update=msg[:200]),
                    scope=entry.get("scope", ""),
                    # Persist step/tool/partial/URL state after every pass so a later "continue"
                    # resumes real work (including the concrete source URLs already discovered).
                    on_state=lambda st: cls._update(
                        tid,
                        tools_used=st.get("tools_used", []),
                        partial_result=st.get("partial_result", ""),
                        current_step=st.get("current_step", ""),
                        discovered_urls=st.get("discovered_urls", []),
                    ),
                    task_memory=entry.get("task_memory") or None,
                )
                if cls._status_of(tid) == "cancelled":
                    return
                summary = cls._summarize(result)
                out_path = cls._write_output(entry, result)
                cls._update(
                    tid, status="completed", completed_at=_host._now_iso(),
                    result_summary=summary, output_path=out_path,
                    last_update=summary[:200] or "Done.",
                )
                cls._notify_chat(entry, summary, out_path)
                _host.log.info("Background task completed %s", tid)
            except Exception as exc:  # noqa: BLE001
                _host.log.exception("Background task failed %s", tid)
                cls._update(
                    tid, status="failed", completed_at=_host._now_iso(),
                    error=str(exc), last_update=f"Failed: {exc}",
                )
                cls._notify_chat(entry, f"⚠ Background task failed: {exc}", "")

    @staticmethod
    def _summarize(result: str) -> str:
        """A short, clean summary line for lists/notifications (keeps full text in the output file)."""
        text = (result or "").strip()
        if not text:
            return "Completed with no textual output."
        first = text.split("\n\n", 1)[0].strip()
        return (first[:400] + "…") if len(first) > 400 else first

    @classmethod
    def _write_output(cls, entry: dict[str, Any], result: str) -> str:
        """Persist the full output to background_results/ and return the path."""
        try:
            _host.BACKGROUND_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = _host.BACKGROUND_RESULTS_DIR / f"{entry['id']}_{stamp}.md"
            doc = (
                f"# Background task {entry['id']}\n\n"
                f"- **Request:** {entry.get('user_request', '')}\n"
                f"- **Type:** {entry.get('task_type', 'general')}\n"
                f"- **Started:** {entry.get('started_at', '')}\n"
                f"- **Completed:** {_host._now_iso()}\n\n## Result\n\n{result or '(no output)'}\n"
            )
            path.write_text(doc, encoding="utf-8")
            return str(path)
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("Could not write background result: %s", exc)
            return ""

    @classmethod
    def _notify_chat(cls, entry: dict[str, Any], summary: str, out_path: str) -> None:
        """Inject the result into the originating chat history + queue it for the live UI."""
        if not _bg_task_cfg("notify_in_chat", True):
            return
        tail = f"\n\n_Full output: `{out_path}`_" if out_path else ""
        msg = {
            "role": "assistant",
            "content": (
                f"✅ **Background task {entry['id']} complete** "
                f"— {entry.get('description', '')[:120]}\n\n{summary}{tail}"
            ),
        }
        chat_path = entry.get("chat_path") or ""
        # 1) Persist into the chat history file (survives even if the UI is closed).
        if chat_path:
            try:
                with cls._lock:
                    hist = _host.load_chat_history(Path(chat_path))
                    hist.append(msg)
                    _host.save_chat_history(hist, Path(chat_path))
            except Exception as exc:  # noqa: BLE001
                _host.log.debug("Could not inject background result into chat file: %s", exc)
        # 2) Stage for the live UI to surface on its next refresh / next turn.
        with cls._lock:
            cls._pending_msgs.setdefault(chat_path, []).append(msg)
            cls._dirty_chats.add(chat_path)

    # ---- live-UI bridge ----------------------------------------------------
    @classmethod
    def has_updates_for(cls, chat_path: str) -> bool:
        with cls._lock:
            return (chat_path or "") in cls._dirty_chats

    @classmethod
    def consume_updates(cls, chat_path: str) -> None:
        """Mark a chat's completions as seen (call after refreshing it from the file)."""
        with cls._lock:
            cls._dirty_chats.discard(chat_path or "")
            cls._pending_msgs.pop(chat_path or "", None)

    @classmethod
    def drain_pending(cls, chat_path: str) -> list[dict[str, Any]]:
        """Return + clear assistant messages waiting for a live chat (for in-turn merging)."""
        with cls._lock:
            msgs = cls._pending_msgs.pop(chat_path or "", [])
            cls._dirty_chats.discard(chat_path or "")
            return msgs

    @classmethod
    def inject_message(cls, chat_path: str, content: str) -> None:
        """Persist an assistant message into a chat history file + stage it for the live-UI bridge.

        Shared delivery path for any async producer that needs to surface a result in chat — used by
        background-task completions and by scheduled agents (so their deliverables actually land in
        the chat, not just an output file). Safe to call from any thread; never raises.
        """
        content = (content or "").strip()
        if not content:
            return
        msg = {"role": "assistant", "content": content}
        chat_path = chat_path or ""
        if chat_path:
            try:
                with cls._lock:
                    hist = _host.load_chat_history(Path(chat_path))
                    hist.append(msg)
                    _host.save_chat_history(hist, Path(chat_path))
            except Exception as exc:  # noqa: BLE001
                _host.log.debug("inject_message file write failed: %s", exc)
        with cls._lock:
            cls._pending_msgs.setdefault(chat_path, []).append(msg)
            cls._dirty_chats.add(chat_path)

    # ---- queries + control -------------------------------------------------
    @classmethod
    def list(cls) -> list[dict[str, Any]]:
        return list(reversed(cls._load()))

    @classmethod
    def get(cls, tid: str) -> dict[str, Any]:
        for t in cls._load():
            if t.get("id") == tid:
                return t
        return {}

    @classmethod
    def active_count(cls) -> int:
        return sum(1 for t in cls._load() if t.get("status") in ("queued", "running"))

    @classmethod
    def cancel(cls, tid: str) -> bool:
        """Cooperative cancel: queued tasks never start; running tasks discard their result."""
        with cls._lock:
            tasks = cls._load()
            for t in tasks:
                if t.get("id") == tid and t.get("status") in ("queued", "running"):
                    t["status"] = "cancelled"
                    t["completed_at"] = _host._now_iso()
                    t["last_update"] = "Cancelled by user."
                    cls._save(tasks)
                    return True
        return False

    @classmethod
    def resume(cls, tid: str = "") -> dict[str, Any]:
        """Resume a finished/interrupted/failed task from its last step (reloads goal + task memory).

        With no id, resumes the most recent resumable task. Re-submits the ORIGINAL intent carrying
        the stored task memory so the agent continues (see `_run_background_agent`) rather than
        restarting from scratch. Returns the new task entry ({} if nothing to resume).
        """
        prev: dict[str, Any] = {}
        if tid:
            prev = cls.get(tid)
        else:
            for t in cls.list():  # list() is newest-first
                if t.get("status") in ("interrupted", "failed", "completed", "cancelled"):
                    prev = t
                    break
        if not prev:
            return {}
        mem = {
            "tools_used": prev.get("tools_used", []),
            "partial_result": prev.get("partial_result", ""),
            "current_step": prev.get("current_step", ""),
            "discovered_urls": prev.get("discovered_urls", []),
        }
        goal = prev.get("original_intent") or prev.get("user_request", "")
        return cls.submit(
            goal,
            description="↻ continued: " + (prev.get("description") or goal)[:180],
            task_type=prev.get("task_type", "general"),
            scope=prev.get("scope", ""),
            chat_path=prev.get("chat_path", ""),
            mode=prev.get("mode", "execute"),
            task_memory=mem,
        )

    @classmethod
    def reconcile_on_start(cls) -> None:
        """Mark tasks that were mid-flight at last shutdown as 'interrupted' (threads don't survive)."""
        with cls._lock:
            tasks = cls._load()
            changed = False
            for t in tasks:
                if t.get("status") in ("queued", "running"):
                    t["status"] = "interrupted"
                    t["last_update"] = "Interrupted by restart — re-issue if still needed."
                    changed = True
            if changed:
                cls._save(tasks)


def render_background_tasks_md(limit: int = 12) -> str:
    """Markdown list of auto/queued background tasks for the UI + `/tasks`."""
    all_tasks = BackgroundTaskManager.list()
    tasks = all_tasks[:limit]
    if not tasks:
        return ("_No background tasks yet. Heavy requests (deep research, full builds, "
                "'analyze all files in X') run here automatically, or use `/bg <task>`._")
    # Counts header for at-a-glance health (running / done / failed / interrupted).
    counts: dict[str, int] = {}
    for t in all_tasks:
        counts[t.get("status", "")] = counts.get(t.get("status", ""), 0) + 1
    summary_bits = []
    for key, label in (("running", "running"), ("queued", "queued"), ("completed", "done"),
                       ("failed", "failed"), ("interrupted", "interrupted"), ("cancelled", "cancelled")):
        if counts.get(key):
            summary_bits.append(f"{_BG_TASK_ICONS.get(key, '•')} {counts[key]} {label}")
    header = "**Background tasks** — " + (" · ".join(summary_bits) if summary_bits else "none active")
    lines = [header, "", "| # | Status | Type | Request | Updated |", "|---|--------|------|---------|---------|"]
    for t in tasks:
        icon = _BG_TASK_ICONS.get(t.get("status", ""), "•")
        req = (t.get("description") or t.get("user_request", ""))[:48].replace("|", "/")
        upd = (t.get("last_update") or "")[:40].replace("|", "/")
        lines.append(
            f"| `{t.get('id', '')}` | {icon} {t.get('status', '')} | {t.get('task_type', '')} "
            f"| {req} | {upd} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scheduled task agents
#
# A scheduling layer ON TOP of BackgroundAgentManager. Schedules persist to
# scheduled_tasks.json; a single daemon loop dispatches due tasks as background agents
# (with optional project scope) and saves a clean output document per run. Fully isolated
# from the main chat and existing background agents.
# ---------------------------------------------------------------------------

SCHEDULED_TASKS_FILE = _host.APP_DIR / "scheduled_tasks.json"
AGENT_OUTPUTS_DIR = _host.APP_DIR / "agent_outputs"
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _scope_hint_for(project_id: str) -> str:
    """Project scope hint for an arbitrary project id ("" for General / unknown)."""
    if not project_id or project_id == _host.GENERAL_CHAT_ID:
        return ""
    p = _host.ChatProjects.get(project_id)
    if not p or p.get("id") == _host.GENERAL_CHAT_ID:
        return ""
    bits = [f"Project focus: {p.get('name', '')}"]
    if (p.get("description") or "").strip():
        bits.append(p["description"].strip())
    if p.get("tags"):
        bits.append("Keywords: " + ", ".join(p["tags"]))
    return " — ".join(bits)[:240]


def _schedule_label(task: dict[str, Any]) -> str:
    st = task.get("schedule_type", "once")
    if st == "once":
        return f"Once at {task.get('run_at', '?')}"
    if st == "daily":
        return f"Daily at {task.get('time_of_day', '09:00')}"
    if st == "weekly":
        wd = task.get("weekday", 0)
        name = _WEEKDAYS[wd] if 0 <= wd < 7 else "?"
        return f"Weekly on {name} at {task.get('time_of_day', '09:00')}"
    if st == "interval":
        return f"Every {task.get('interval_minutes', 60)} min"
    return st


def _parse_hh_mm(text: str, default=(9, 0)) -> tuple[int, int]:
    try:
        h, m = (text or "").strip().split(":")
        return max(0, min(int(h), 23)), max(0, min(int(m), 59))
    except Exception:  # noqa: BLE001
        return default


def _write_agent_output(task: dict[str, Any], entry: dict[str, Any]) -> str:
    """Persist a clean markdown report of a completed scheduled run. Returns the file path."""
    try:
        AGENT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^\w\-]+", "_", task.get("name", "task"))[:50].strip("_") or "task"
        path = AGENT_OUTPUTS_DIR / f"{safe}_{stamp}.md"
        status = entry.get("status", "")
        started, finished = entry.get("started", 0) or 0, entry.get("finished", 0) or 0
        dur = int(finished - started) if finished and started else 0
        proj = _host.ChatProjects.get(task.get("project_id", _host.GENERAL_CHAT_ID)).get("name", "General Chat")
        result = entry.get("result", "") or "_(No output produced.)_"
        doc = (
            f"# {task.get('name', 'Scheduled task')}\n\n"
            f"- **Generated:** {datetime.now().strftime('%A, %Y-%m-%d %H:%M:%S')}\n"
            f"- **Status:** {status}\n"
            f"- **Schedule:** {_schedule_label(task)}\n"
            f"- **Project scope:** {proj}\n"
            f"- **Duration:** {dur}s\n"
            f"- **Background task ID:** {entry.get('id', '')}\n\n"
            f"## Instructions\n{task.get('instructions', '')}\n\n"
            f"## Result / Findings\n{result}\n\n"
            f"## Notes\n"
            f"Any files created or actions taken are described in the result above (the agent "
            f"reports the tools it used and outputs it produced).\n\n"
            f"---\n_Generated automatically by a Primus scheduled agent._\n"
        )
        path.write_text(doc, encoding="utf-8")
        _host.log.info("Scheduled output saved: %s", path)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        _host.log.exception("Could not write scheduled output")
        return ""


class ScheduledTaskManager:
    """Persistent scheduler that dispatches due tasks as background agents."""

    _lock = threading.Lock()
    _scheduler_started = False
    _inflight: set[str] = set()

    # ---- persistence ----
    @classmethod
    def _load(cls) -> list[dict[str, Any]]:
        _host.ensure_app_dirs()
        try:
            data = json.loads(SCHEDULED_TASKS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    @classmethod
    def _save(cls, tasks: list[dict[str, Any]]) -> None:
        try:
            _host.ensure_app_dirs()
            SCHEDULED_TASKS_FILE.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        except OSError as exc:
            _host.log.warning("Could not save scheduled tasks: %s", exc)

    # ---- schedule math ----
    @classmethod
    def compute_next_run(cls, task: dict[str, Any], after: Optional[float] = None) -> float:
        """Next run as an epoch timestamp (0.0 if none / disabled / one-time already past-dispatched)."""
        now = after if after is not None else time.time()
        st = task.get("schedule_type", "once")
        if st == "once":
            try:
                dt = datetime.strptime(task.get("run_at", "").strip(), "%Y-%m-%d %H:%M")
                return dt.timestamp()
            except Exception:  # noqa: BLE001
                return now + 300  # default 5 min out if unparseable
        if st == "interval":
            mins = max(1, int(task.get("interval_minutes", 60)))
            return now + mins * 60
        # daily / weekly
        h, m = _parse_hh_mm(task.get("time_of_day", "09:00"))
        base = datetime.fromtimestamp(now)
        cand = base.replace(hour=h, minute=m, second=0, microsecond=0)
        if st == "weekly":
            target = int(task.get("weekday", 0)) % 7
            days_ahead = (target - cand.weekday()) % 7
            cand = cand + timedelta(days=days_ahead)
            if cand.timestamp() <= now:
                cand = cand + timedelta(days=7)
        else:  # daily
            if cand.timestamp() <= now:
                cand = cand + timedelta(days=1)
        return cand.timestamp()

    # ---- CRUD ----
    @classmethod
    def create(cls, **fields: Any) -> dict[str, Any]:
        name = (fields.get("name") or "").strip()
        instructions = (fields.get("instructions") or "").strip()
        if not name or not instructions:
            return {}
        task = {
            "id": "sched-" + uuid.uuid4().hex[:8],
            "name": name[:80],
            "instructions": instructions,
            "project_id": fields.get("project_id") or _host.GENERAL_CHAT_ID,
            "schedule_type": fields.get("schedule_type", "once"),
            "run_at": (fields.get("run_at") or "").strip(),
            "time_of_day": (fields.get("time_of_day") or "09:00").strip(),
            "weekday": int(fields.get("weekday", 0)),
            "interval_minutes": int(fields.get("interval_minutes", 60)),
            "enabled": bool(fields.get("enabled", True)),
            "created_at": _host._now_iso(),
            "last_run": "", "last_status": "", "last_output_path": "",
            "run_count": 0,
        }
        task["next_run_ts"] = cls.compute_next_run(task)
        tasks = cls._load()
        tasks.append(task)
        cls._save(tasks)
        _host.log.info("Scheduled task created %s (%s)", task["id"], _schedule_label(task))
        return task

    @classmethod
    def update(cls, tid: str, **fields: Any) -> bool:
        tasks = cls._load()
        for t in tasks:
            if t.get("id") == tid:
                for k in ("name", "instructions", "project_id", "schedule_type", "run_at",
                          "time_of_day", "weekday", "interval_minutes", "enabled"):
                    if k in fields and fields[k] is not None:
                        t[k] = fields[k]
                t["next_run_ts"] = cls.compute_next_run(t) if t.get("enabled", True) else 0.0
                cls._save(tasks)
                return True
        return False

    @classmethod
    def delete(cls, tid: str) -> bool:
        tasks = cls._load()
        new = [t for t in tasks if t.get("id") != tid]
        if len(new) != len(tasks):
            cls._save(new)
            return True
        return False

    @classmethod
    def toggle(cls, tid: str) -> bool:
        tasks = cls._load()
        for t in tasks:
            if t.get("id") == tid:
                t["enabled"] = not t.get("enabled", True)
                t["next_run_ts"] = cls.compute_next_run(t) if t["enabled"] else 0.0
                cls._save(tasks)
                return t["enabled"]
        return False

    @classmethod
    def list(cls) -> list[dict[str, Any]]:
        return cls._load()

    @classmethod
    def get(cls, tid: str) -> dict[str, Any]:
        for t in cls._load():
            if t.get("id") == tid:
                return t
        return {}

    # ---- dispatch + completion ----
    @classmethod
    def run_now(cls, tid: str) -> bool:
        t = cls.get(tid)
        if not t:
            return False
        cls._dispatch(t)
        return True

    @classmethod
    def _dispatch(cls, task: dict[str, Any]) -> None:
        tid = task["id"]
        with cls._lock:
            if tid in cls._inflight:
                return
            cls._inflight.add(tid)
        scope = _scope_hint_for(task.get("project_id", _host.GENERAL_CHAT_ID))
        # If submission itself fails, we must still release the in-flight lock and mark the task
        # failed — otherwise the id stays "running" forever and never re-dispatches.
        dispatched = True
        try:
            BackgroundAgentManager.submit(
                task.get("instructions", ""),
                scope=scope,
                meta={"scheduled_id": tid, "scheduled_name": task.get("name", "")},
                on_done=lambda entry, _tid=tid: cls._on_complete(_tid, entry),
            )
        except Exception as exc:  # noqa: BLE001
            dispatched = False
            _host.log.exception("Scheduled dispatch failed %s", tid)
            with cls._lock:
                cls._inflight.discard(tid)
        # Advance the schedule now so a long run can't double-trigger.
        tasks = cls._load()
        for t in tasks:
            if t.get("id") == tid:
                t["last_status"] = "running" if dispatched else "failed"
                t["last_run"] = _host._now_iso()
                if t.get("schedule_type") == "once" and dispatched:
                    t["enabled"] = False
                    t["next_run_ts"] = 0.0
                else:
                    # Recurring tasks always re-arm; a failed one-time task re-arms to retry later.
                    t["next_run_ts"] = cls.compute_next_run(t)
                break
        cls._save(tasks)
        _host.log.info("Dispatched scheduled task %s (ok=%s)", tid, dispatched)

    @classmethod
    def _on_complete(cls, tid: str, entry: dict[str, Any]) -> None:
        task = cls.get(tid)
        out_path = _write_agent_output(task, entry) if task else ""
        status = entry.get("status", "completed")
        tasks = cls._load()
        for t in tasks:
            if t.get("id") == tid:
                t["last_status"] = status
                t["last_output_path"] = out_path
                t["run_count"] = int(t.get("run_count", 0)) + 1
                break
        cls._save(tasks)
        with cls._lock:
            cls._inflight.discard(tid)
        # Deliver the result into the task's chat so scheduled agents (e.g. Morning Briefing)
        # actually surface in the conversation — not just as an output file on disk.
        if task:
            try:
                cls._deliver_to_chat(task, entry, out_path, status)
            except Exception as exc:  # noqa: BLE001
                _host.log.debug("Scheduled chat delivery failed %s: %s", tid, exc)

    @staticmethod
    def _deliver_to_chat(task: dict[str, Any], entry: dict[str, Any], out_path: str, status: str) -> None:
        """Post a completed/failed scheduled run into its project chat (via the live-UI bridge)."""
        result = (entry.get("result") or "").strip()
        name = task.get("name", "Scheduled task")
        if status == "completed" and result:
            # Deliver the full deliverable (generously capped); the output file holds the complete copy.
            body = result if len(result) <= 8000 else result[:8000] + "\n\n…(truncated — see full report)"
            icon = "🗓️"
        elif status == "completed":
            body, icon = "_(Completed with no textual output.)_", "🗓️"
        else:
            body, icon = f"_{entry.get('last_update', 'Run failed.')}_", "⚠"
        tail = f"\n\n_Full report: `{out_path}`_" if out_path else ""
        content = f"{icon} **{name}** — scheduled run {status}\n\n{body}{tail}"
        chat_path = str(_host.chat_history_path(task.get("project_id", _host.GENERAL_CHAT_ID)))
        BackgroundTaskManager.inject_message(chat_path, content)

    # ---- scheduler loop ----
    @classmethod
    def ensure_scheduler_started(cls) -> None:
        with cls._lock:
            if cls._scheduler_started:
                return
            cls._scheduler_started = True
        threading.Thread(target=cls._loop, name="primus-scheduler", daemon=True).start()
        _host.log.info("Scheduled-task scheduler started")

    @classmethod
    def _loop(cls) -> None:
        while True:
            try:
                now = time.time()
                for t in cls._load():
                    if not t.get("enabled", True):
                        continue
                    nrt = float(t.get("next_run_ts", 0) or 0)
                    if 0 < nrt <= now and t["id"] not in cls._inflight:
                        cls._dispatch(t)
            except Exception:  # noqa: BLE001
                _host.log.exception("Scheduler loop iteration failed")
            time.sleep(20)


def list_agent_outputs(limit: int = 20) -> list[Path]:
    """Recent scheduled-agent output documents, newest first."""
    try:
        if not AGENT_OUTPUTS_DIR.exists():
            return []
        files = [p for p in AGENT_OUTPUTS_DIR.glob("*.md") if p.is_file()]
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    except OSError:
        return []


def render_scheduled_tasks_md() -> str:
    """Human-readable list of scheduled tasks with next/last run + status."""
    tasks = ScheduledTaskManager.list()
    if not tasks:
        return "_No scheduled tasks yet. Create one above._"
    lines = ["| Task | Schedule | Scope | Next run | Last status | On |",
             "|------|----------|-------|----------|-------------|----|"]
    for t in sorted(tasks, key=lambda x: x.get("next_run_ts", 0) or 9e18):
        proj = _host.ChatProjects.get(t.get("project_id", _host.GENERAL_CHAT_ID)).get("name", "General")
        nrt = float(t.get("next_run_ts", 0) or 0)
        nxt = datetime.fromtimestamp(nrt).strftime("%Y-%m-%d %H:%M") if nrt else "—"
        on = "✓" if t.get("enabled", True) else "○"
        status = t.get("last_status", "") or "—"
        lines.append(
            f"| **{t.get('name','?')}** <br><sub>`{t.get('id','')}`</sub> "
            f"| {_schedule_label(t)} | {proj} | {nxt} | {status} | {on} |"
        )
    return "\n".join(lines)


def render_recent_outputs_md() -> str:
    outs = list_agent_outputs()
    if not outs:
        return "_No scheduled-agent outputs yet._"
    lines = ["**Recent outputs** (`~/.primus/agent_outputs/`):"]
    for p in outs:
        ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"- `{p.name}` · {ts}")
    return "\n".join(lines)
