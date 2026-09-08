"""user_path vs primus_path — parse an operator-ordered sequence and run it in order.

No 7b. Gap-fill A–D only. Never reorder, skip, collapse, or add helpful extras.
Library modules reach the host via ``_host`` — do not import admin_assistant.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import sys as _sys

_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

# Real names we may speak when a step cannot be mapped. Never invent others.
_REAL_TOOL_NAMES = (
    "terminal", "git", "gmail_list_messages", "gmail_read_message",
    "list_directory", "write_file", "move_path", "read_file", "delete_file",
)
_UNKNOWN_Q = (
    "Which tool should I use for that? I can use terminal, git, "
    "gmail_list_messages, gmail_read_message, list_directory, write_file, "
    "move_path, read_file, or delete_file."
)

_MY_PATH_PREFIX_RE = re.compile(r"^\s*my\s+path\s*:\s*", re.I)
_TOGGLE_ON_RE = re.compile(
    r"^\s*(?:follow\s+my\s+path|do\s+these\s+steps\s+in\s+order|exactly\s+this\s*:?)"
    r"\s*(?:[,:]|\s+then\b)?\s*",
    re.I,
)
_TOGGLE_OFF_LEAD_RE = re.compile(
    r"^\s*(?:your\s+path|handle\s+it|you\s+decide|primus\s+path)\b",
    re.I,
)
_TOGGLE_OFF_CHUNK_RE = re.compile(
    r"^\s*,?\s*(?:your\s+path|handle\s+it|you\s+decide|primus\s+path)\s*",
    re.I,
)
_EXTRA_ASK_RE = re.compile(
    r"\s+\+\s+(?=(?:tell me about|what(?:'?s| is| are)|who(?:'?s| is)|explain|describe)\b)",
    re.I,
)
_THEN_SPLIT_RE = re.compile(r"\s+(?:and\s+then|then)\s+", re.I)
_NUMBERED_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")

_COUNT_RE = re.compile(
    r"\b(?:how many|count|total(?:\s+count)?(?:\s+number)?(?:\s+of)?|number of)\b[^?\n]{0,50}"
    r"\b(folders?|directories|files)\b",
    re.I,
)
_LIST_RE = re.compile(
    r"^\s*(?:list|ls|show)\s+(?:the\s+)?(?:contents\s+(?:of\s+)?)?(.+?)\s*$",
    re.I,
)
_MOVE_RE = re.compile(r"^\s*(?:move|mv)\s+(.+?)\s+to\s+(.+?)\s*$", re.I)
_READ_RE = re.compile(r"^\s*(?:read|cat|open)\s+(.+?)\s*$", re.I)
_WRITE_RE = re.compile(r"^\s*(?:write|save|put)\s+(.+?)\s+to\s+(.+?)\s*$", re.I)
# write|create|put PATH with TEXT — each destination is its own write_file step.
_PATH_TOKEN = (
    r"(?:~(?:/[\w./-]*)?|/[\w./-]+|(?:[\w.-]+/)+[\w.-]+|[\w.-]+\.\w{1,12})"
)
_WRITE_WITH_RE = re.compile(
    rf"^\s*(?:create|write|put)\s+({_PATH_TOKEN})\s+with\s+(.+?)\s*$",
    re.I,
)
_WRITE_WITH_FIND_RE = re.compile(
    rf"(?:create|write|put)\s+{_PATH_TOKEN}\s+with\s+",
    re.I,
)
_CREATE_DIR_RE = re.compile(
    r"^\s*create\s+(~(?:/[\w./-]*)?|/[\w./-]+|[\w.-]+(?:/[\w.-]+)*)\s*$",
    re.I,
)
# create|write|put PATH with a file suffix and no `with` text → write_file (empty body).
_CREATE_FILE_RE = re.compile(
    rf"^\s*(?:create|write|put)\s+({_PATH_TOKEN})\s*$",
    re.I,
)
_FILE_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]{1,12}$")
_DELETE_RE = re.compile(r"^\s*(?:delete|remove|rm)\s+(.+?)\s*$", re.I)
_SEND_MAIL_RE = re.compile(r"\bsend\b.*\b(?:e-?mails?|mail)\b|\b(?:e-?mails?|mail)\b.*\bsend\b", re.I)
_SLACK_RE = re.compile(r"\bslack\b|\bpost\s+to\s+slack\b", re.I)
_SUMMARIZE_RE = re.compile(r"^\s*(?:summari[sz]e|summary|tl;?dr)\b", re.I)
_KNOWLEDGE_STEP_RE = re.compile(
    r"^\s*(?:tell me about|what(?:'?s| is| are)|who(?:'?s| is)|explain|describe)\b",
    re.I,
)
_NAMED_DIRS = {
    "home": "~", "downloads": "~/Downloads", "desktop": "~/Desktop",
    "documents": "~/Documents", "docs": "~/Documents", "pictures": "~/Pictures",
}


@dataclass
class PathInspect:
    prefix_user: bool = False
    toggle_on: bool = False
    toggle_off: bool = False
    steps: list[str] = field(default_factory=list)
    extra_ask: str = ""
    body: str = ""


@dataclass
class _RunCtx:
    last_list_path: str = ""
    last_listing: str = ""
    last_files: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


def _has_file_suffix(path: str) -> bool:
    name = (path or "").rstrip("/").rsplit("/", 1)[-1]
    return bool(_FILE_SUFFIX_RE.search(name))


def _split_write_destinations(text: str) -> list[str]:
    """Each create|write|put PATH with TEXT is its own step. Do not merge destinations."""
    raw = (text or "").strip(" \t,.")
    if not raw:
        return []
    matches = list(_WRITE_WITH_FIND_RE.finditer(raw))
    if len(matches) <= 1:
        return [raw]
    parts: list[str] = []
    if matches[0].start() > 0:
        lead = raw[: matches[0].start()].strip(" \t,.")
        if lead:
            parts.append(lead)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        chunk = raw[m.start():end].strip(" \t,.")
        if chunk:
            parts.append(chunk)
    return parts


def parse_user_path(text: str) -> list[str]:
    """Split on numbered lists, line breaks, or then / and then. No model.

    Each create|write|put PATH with TEXT stays its own write_file step.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    if _NUMBERED_RE.search(raw):
        chunks = [p.strip(" \t,.") for p in _NUMBERED_RE.split(raw) if p.strip()]
    else:
        lines = [ln.strip(" \t,") for ln in raw.splitlines() if ln.strip()]
        if len(lines) > 1:
            chunks = lines
        else:
            chunks = [p.strip(" \t,.") for p in _THEN_SPLIT_RE.split(raw) if p.strip()]
    out: list[str] = []
    for chunk in chunks:
        out.extend(_split_write_destinations(chunk))
    return out


def is_write_with_step(step: str) -> bool:
    return bool(_WRITE_WITH_RE.match((step or "").strip()))


def is_create_dir_step(step: str) -> bool:
    """create DIR with no ``with`` text and no file suffix → create_directory."""
    m = _CREATE_DIR_RE.match((step or "").strip())
    if not m:
        return False
    return not _has_file_suffix(m.group(1))


def is_create_file_step(step: str) -> bool:
    """create|write|put PATH with a file suffix and no ``with`` text → write_file."""
    m = _CREATE_FILE_RE.match((step or "").strip())
    if not m:
        return False
    return _has_file_suffix(m.group(1))


def is_list_step(step: str) -> bool:
    """Bare list/ls (not 'show battery')."""
    return bool(re.match(r"^\s*(?:list|ls)\b", (step or "").strip(), re.I))


def has_file_write_steps(steps: list[str]) -> bool:
    """True when any step is a real write_file / create_directory mapping."""
    for step in steps or []:
        if (
            is_write_with_step(step)
            or is_create_dir_step(step)
            or is_create_file_step(step)
            or _WRITE_RE.match(step)
        ):
            return True
    return False


def has_list_steps(steps: list[str]) -> bool:
    return any(is_list_step(s) for s in (steps or []))


def inspect_path_message(message: str) -> PathInspect:
    """Detect toggle / `my path:` prefix, listed steps, and a trailing extra ask."""
    msg = (message or "").strip()
    out = PathInspect()
    if not msg:
        return out

    rest = msg
    if _MY_PATH_PREFIX_RE.match(rest):
        out.prefix_user = True
        out.toggle_on = True
        rest = _MY_PATH_PREFIX_RE.sub("", rest, count=1).strip()
    elif _TOGGLE_ON_RE.match(rest):
        out.toggle_on = True
        rest = _TOGGLE_ON_RE.sub("", rest, count=1).strip()
    elif _TOGGLE_OFF_LEAD_RE.match(rest):
        out.toggle_off = True
        rest = rest
        while True:
            nxt = _TOGGLE_OFF_CHUNK_RE.sub("", rest, count=1)
            if nxt == rest:
                break
            rest = nxt.strip()
        rest = rest.lstrip(",").strip()

    extra = ""
    em = _EXTRA_ASK_RE.search(rest)
    if em:
        extra = rest[em.end():].strip()
        rest = rest[:em.start()].strip()
    rest = re.sub(r"^\s*(?:and\s+then|then)\s+", "", rest, flags=re.I).strip()

    out.body = rest
    out.extra_ask = extra
    out.steps = parse_user_path(rest)
    return out


def set_path_mode(mode: str) -> None:
    """Persist path_mode on the live CFG (existing config.json — no new store)."""
    if mode not in ("primus_path", "user_path"):
        mode = "primus_path"
    try:
        _host.CFG["path_mode"] = mode
        _host.save_config_file()
    except Exception:  # noqa: BLE001 — toggle must not crash a turn
        try:
            _host.CFG["path_mode"] = mode
        except Exception:
            pass


def current_path_mode() -> str:
    try:
        mode = str(_host.CFG.get("path_mode") or "primus_path")
    except Exception:
        return "primus_path"
    return mode if mode in ("primus_path", "user_path") else "primus_path"


def _registry() -> Any:
    return _sys.modules.get("primus.tools.registry")


def _invoke(tool: Any, **kwargs: Any) -> str:
    if tool is None:
        return _UNKNOWN_Q
    try:
        out = tool.invoke(kwargs) if hasattr(tool, "invoke") else tool(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return f"✗ {exc}"
    return str(out)


def _count_target(low: str) -> str:
    m = re.search(r"(~[\w/.-]*|/[\w/.-]+)", low)
    if m:
        return m.group(1)
    m = re.search(
        r"\bin\s+(?:my\s+)?([\w-]+(?:\s+[\w-]+){0,2}?)(?:\s+(?:folder|directory|dir))?\s*(?:[?!.,]|$)",
        low,
    )
    cand = (m.group(1) if m else "").strip().strip("/")
    if cand and cand.lower() not in ("it", "there", "that", "them", "total"):
        return _NAMED_DIRS.get(cand.lower(), cand)
    for word, path in _NAMED_DIRS.items():
        if re.search(rf"\b{word}\b", low):
            return path
    return "~"


def _count_from_listing(listing: str, kind: str, target: str) -> str:
    want_dir = kind.lower().startswith(("folder", "director"))
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
        noun = noun[:-1]
    shown = ", ".join(names[:8]) + (f", … and {len(names) - 8} more" if len(names) > 8 else "")
    return f"There are **{len(names)} {noun}** in `{target}`: {shown}"


def _files_from_listing(listing: str, suffix: str = "") -> list[str]:
    names: list[str] = []
    for ln in listing.splitlines():
        if not ln.startswith("[file]"):
            continue
        name = ln.split("] ", 1)[1].rsplit(" (", 1)[0]
        if suffix and not name.lower().endswith(suffix.lower()):
            continue
        names.append(name)
    return names


def _run_count(step: str, ctx: _RunCtx, reg: Any) -> str:
    kind_m = _COUNT_RE.search(step)
    kind = kind_m.group(1) if kind_m else "folders"
    target = _count_target(step.lower())
    listing = _invoke(reg.list_directory, path=target)
    ctx.last_list_path = target
    ctx.last_listing = listing
    ctx.last_files = _files_from_listing(listing)
    if listing.startswith(("✗", "Refusing", "Not found", "No matches")):
        return listing
    return _count_from_listing(listing, kind, target)


_LIST_CHAT_CAP = 20


def _is_lock_name(name: str) -> bool:
    n = (name or "").lower()
    return n.endswith(".lock") or ".~lock" in n or n.startswith(".~lock")


def _looks_like_listing(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    marked = [ln for ln in lines if ln.startswith(("[dir]", "[file]"))]
    if not marked:
        return False
    rest = [ln for ln in lines if not ln.startswith(("[dir]", "[file]"))]
    if rest and all(re.search(r"\band\s+\d+\s+more", ln, re.I) for ln in rest):
        return True
    return len(marked) >= max(1, len(lines) - 1)


def _cap_listing_for_chat(listing: str, limit: int = _LIST_CHAT_CAP) -> str:
    if listing.startswith(("✗", "Refusing", "Not found", "No matches")):
        return listing
    lines = listing.splitlines()
    if len(lines) <= limit:
        return listing
    extra = len(lines) - limit
    return "\n".join(lines[:limit]) + f"\n… and {extra} more"


def _listing_inventory(listing: str, path: str = "") -> str:
    files: list[str] = []
    dirs: list[str] = []
    for ln in (listing or "").splitlines():
        if ln.startswith("[file]"):
            name = ln.split("] ", 1)[1].rsplit(" (", 1)[0]
            if _is_lock_name(name):
                continue
            files.append(name)
        elif ln.startswith("[dir]"):
            name = ln.split("] ", 1)[1].rsplit(" (", 1)[0]
            dirs.append(name)
    ext_counts: dict[str, int] = {}
    for name in files:
        if "." not in name[1:]:
            continue
        ext = "." + name.rsplit(".", 1)[-1].lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    top = sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
    ext_s = ", ".join(f"{e} ({n})" for e, n in top) if top else "none"
    names = (files + dirs)[:5]
    leftover = max(0, len(files) + len(dirs) - len(names))
    shown = ", ".join(names) if names else "(empty)"
    if leftover:
        shown = f"{shown}, and {leftover} more"
    where = f" in `{path}`" if path else ""
    nf, nd = len(files), len(dirs)
    if nf == 0 and nd == 0:
        return f"0 files, 0 folders{where}."
    return (
        f"{nf} file{'s' if nf != 1 else ''}, "
        f"{nd} folder{'s' if nd != 1 else ''}{where}. "
        f"Top extensions: {ext_s}.\n"
        f"{shown}"
    )


def _run_list(step: str, ctx: _RunCtx, reg: Any) -> str:
    m = _LIST_RE.match(step)
    raw = (m.group(1) if m else step).strip()
    raw = re.sub(r"^(?:the\s+)?(?:folder|directory|dir)\s+", "", raw, flags=re.I)
    target = _count_target(raw.lower()) if raw else "~"
    if raw.startswith("~") or raw.startswith("/"):
        target = raw.split()[0]
    listing = _invoke(reg.list_directory, path=target)
    ctx.last_list_path = target
    if listing.startswith(("✗", "Refusing", "Not found")):
        ctx.last_listing = listing
        ctx.last_files = []
        return listing
    if (
        (not listing.strip())
        or listing.startswith("No matches")
        or listing.startswith("0 files, 0 folders")
    ):
        ctx.last_listing = ""
        ctx.last_files = []
        return _listing_inventory("", target)
    ctx.last_listing = listing
    ctx.last_files = _files_from_listing(listing)
    return _cap_listing_for_chat(listing)


def _run_move(step: str, ctx: _RunCtx, reg: Any) -> str:
    m = _MOVE_RE.match(step)
    if not m:
        return _UNKNOWN_Q
    src_raw, dst_raw = m.group(1).strip(), m.group(2).strip()
    dst = dst_raw.split()[0]
    # Gap B: "pdfs" / glob implied by the previous list root.
    if re.search(r"pdfs?\b", src_raw, re.I) and "/" not in src_raw and not src_raw.startswith("~"):
        root = ctx.last_list_path or "~"
        listing = ctx.last_listing or _invoke(reg.list_directory, path=root, pattern="*.pdf")
        pdfs = _files_from_listing(listing, ".pdf")
        if not pdfs and ctx.last_listing:
            listing = _invoke(reg.list_directory, path=root, pattern="*.pdf")
            pdfs = _files_from_listing(listing, ".pdf")
        if not pdfs:
            return f"No PDF files to move from `{root}`."
        parts: list[str] = []
        for name in pdfs:
            parts.append(_invoke(reg.move_path, source=f"{root}/{name}", destination=f"{dst}/{name}"))
        return "\n".join(parts)
    # Gap B: bare name from previous listing.
    src = src_raw.split()[0]
    if ctx.last_list_path and "/" not in src and not src.startswith("~"):
        src = f"{ctx.last_list_path}/{src}"
    return _invoke(reg.move_path, source=src, destination=dst)


def _run_delete(step: str, ctx: _RunCtx, reg: Any) -> str:
    m = _DELETE_RE.match(step)
    raw = (m.group(1) if m else "draft").strip()
    path = raw.split()[0]
    # Gap B: previous listing root + a bare name ("draft").
    if ctx.last_list_path and "/" not in path and not path.startswith("~"):
        listed = f"{ctx.last_list_path}/{path}"
        trial = _invoke(reg.delete_file, path=listed)
        if trial.startswith("✗") and "Not found" in trial:
            pass
        else:
            return trial
    out = _invoke(reg.delete_file, path=path)
    if "Not found" in out:
        # Explicit delete step still goes through the rm approval queue (same as delete_file).
        from pathlib import Path

        lexical = Path(path).expanduser()
        if not lexical.is_absolute():
            lexical = _host.HOME / lexical
        return _host.run_shell(f"rm {shlex.quote(str(lexical))}")
    return out


def _run_send_email(reg: Any) -> str:
    # Gap C: Suggest/Execute wraps send. Missing args → preview in Suggest, one question in Execute.
    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return (
            "[Suggest] Would send email:\n"
            "  To: (not specified)\n"
            "  Subject: (not specified)\n"
            "  Body: (not specified)\n"
            "Switch to Execute (or approve) to actually send."
        )
    return "Who should I send it to, and what's the subject and body?"


def _run_write(step: str, ctx: _RunCtx, reg: Any) -> str:
    wm = _WRITE_WITH_RE.match(step)
    if wm:
        dest, content = wm.group(1).strip(), wm.group(2).strip()
    else:
        wm = _WRITE_RE.match(step)
        if not wm:
            return _UNKNOWN_Q
        content, dest = wm.group(1).strip(), wm.group(2).strip()
        dest = dest.split()[0]
    if content.lower() in ("that", "it", "this", "the output") and ctx.outputs:
        content = ctx.outputs[-1]
    return _invoke(reg.write_file, path=dest.split()[0], content=content)


def _run_create_dir(step: str, reg: Any) -> str:
    m = _CREATE_DIR_RE.match(step)
    if not m:
        return _UNKNOWN_Q
    return _invoke(reg.create_directory, path=m.group(1).strip())


def _run_summarize(ctx: _RunCtx) -> str:
    if not ctx.outputs:
        return "(nothing to summarize yet)"
    prev = ctx.outputs[-1]
    listed = bool(ctx.last_list_path) and not (ctx.last_listing or "").startswith(
        ("✗", "Refusing", "Not found")
    )
    if _looks_like_listing(prev) or listed:
        blob = ctx.last_listing if ctx.last_list_path else prev
        return _listing_inventory(blob, ctx.last_list_path)
    blobs = []
    for chunk in ctx.outputs:
        lines = [ln for ln in chunk.splitlines() if ln.strip()][:8]
        blobs.append("\n".join(lines) if lines else chunk[:400])
    return "Summary:\n" + "\n\n".join(blobs)


def run_user_path(
    steps: list[str],
    *,
    answer_knowledge: Optional[Callable[[str], str]] = None,
) -> str:
    """Execute listed steps in order. Gap-fill A–D only. Stop on an unknown-tool question."""
    if not steps:
        return "What's the first step?"
    reg = _registry()
    if reg is None:
        return _UNKNOWN_Q
    ctx = _RunCtx()
    parts: list[str] = []
    for step in steps:
        low = step.lower().strip()
        if _SLACK_RE.search(low) and not re.search(r"\b(?:list|read|status)\b", low):
            # Gap D / A: don't invent a tool name; required args are not supplied.
            parts.append("What should I post, and to which channel?")
            break
        if _SEND_MAIL_RE.search(low):
            out = _run_send_email(reg)
        elif _SUMMARIZE_RE.search(low):
            out = _run_summarize(ctx)
        elif _COUNT_RE.search(low) and not re.search(r"\b(?:code|script|python)\b", low):
            out = _run_count(step, ctx, reg)
        elif _LIST_RE.match(step) or re.match(r"^\s*(?:list|ls)\b", low):
            out = _run_list(step, ctx, reg)
        elif _MOVE_RE.match(step):
            out = _run_move(step, ctx, reg)
        elif _READ_RE.match(step):
            path = _READ_RE.match(step).group(1).strip()
            if ctx.last_list_path and "/" not in path and not path.startswith("~"):
                path = f"{ctx.last_list_path}/{path.split()[0]}"  # Gap B
            out = _invoke(reg.read_file, path=path.split()[0] if path else path)
        elif is_write_with_step(step) or _WRITE_RE.match(step):
            out = _run_write(step, ctx, reg)
        elif is_create_file_step(step):
            dest = _CREATE_FILE_RE.match(step).group(1).strip()
            out = _invoke(reg.write_file, path=dest.split()[0], content="")
        elif is_create_dir_step(step):
            out = _run_create_dir(step, reg)
        elif _DELETE_RE.match(step) or re.search(r"\bdelete\b.*\bdraft\b", low):
            out = _run_delete(step if _DELETE_RE.match(step) else "delete draft", ctx, reg)
        elif _KNOWLEDGE_STEP_RE.search(step) and answer_knowledge is not None:
            out = answer_knowledge(step)
        else:
            out = _UNKNOWN_Q
            parts.append(out)
            break
        ctx.outputs.append(out)
        parts.append(out)
        if out.startswith("✗") and (
            is_write_with_step(step)
            or _WRITE_RE.match(step)
            or is_create_file_step(step)
            or is_create_dir_step(step)
        ):
            break
    return "\n\n".join(parts)
