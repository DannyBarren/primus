"""Primus tool system (Phase 4 extraction).

The entire tool layer — TerminalTool/GitTool, every ``@tool`` function, the self-introspection and
self-improvement tools, and ``build_tools()`` — lives here. It was moved VERBATIM from
admin_assistant.py; the only mechanical change is that references to admin_assistant module globals
are reached through the live module object as ``_host.<name>``. That keeps the circular import safe
and preserves runtime-rebound config (``_host.CFG``/model globals) and forward-declared helpers
(``_host.make_chat_ollama`` etc.) without changing any behavior. Nothing here is locked or
read-only — add new tools here and register them in ``build_tools()`` exactly as before.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import sys as _sys

# Resolve the host application module WITHOUT importing it by name. admin_assistant.py is normally
# run as the program entrypoint, so it is registered in sys.modules as "__main__" — a plain
# ``import admin_assistant`` would then RE-EXECUTE the whole file as a second module object and
# trigger a circular import (this package is imported mid-way through admin_assistant). Looking it
# up in sys.modules returns the already-loaded module (preferring the named module when present,
# e.g. when admin_assistant is imported as a library), with live CFG / model globals intact.
_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

# langchain / pydantic surface — bound from the host so we inherit the exact objects (real or
# graceful stubs) that admin_assistant resolved at startup. These exist well before this module is
# imported (top-of-file in admin_assistant), so binding them here is safe.
tool = _host.tool
BaseTool = _host.BaseTool
BaseModel = _host.BaseModel
Field = _host.Field
HumanMessage = _host.HumanMessage
SystemMessage = _host.SystemMessage
AIMessage = _host.AIMessage

# Shared filename sanitizer (downloads / project + app scaffolding). Imported directly rather than
# via the host so it never depends on a re-export.
from primus.utils.text import safe_filename as _safe_filename  # noqa: E402

# Operator identity (generic "the operator" unless ~/.config/primus/operator.json personalizes it).
from primus.core.identity import OPERATOR as _OPERATOR, email_signoff as _email_signoff  # noqa: E402


class ShellInput(BaseModel):
    command: str = Field(..., description="Shell command to preview or execute.")


class TerminalTool(BaseTool):
    name: str = "terminal"
    description: str = (
        "Run shell commands on the operator's Linux machine under $HOME. "
        "Safe read-only commands (ls, pwd, cat, df, git status, find, grep, etc.) "
        "auto-execute and return real output. Risky commands (rm, sudo, mv) queue for approval. "
        "Always use this tool instead of guessing command output."
    )
    args_schema: type[BaseModel] = ShellInput

    def _run(self, command: str) -> str:
        from primus.core.vault import expand_vault_tokens  # noqa: PLC0415

        return _host.run_shell(expand_vault_tokens(command))


class GitInput(BaseModel):
    repo_path: str = Field(default=".", description="Git repo under home.")
    git_args: str = Field(..., description="Git subcommand + args.")


class GitTool(BaseTool):
    name: str = "git"
    description: str = "Git operations in repos under home."
    args_schema: type[BaseModel] = GitInput

    def _run(self, git_args: str, repo_path: str = ".") -> str:
        from primus.core.vault import expand_vault_tokens  # noqa: PLC0415

        git_args = expand_vault_tokens(git_args)
        repo_path = expand_vault_tokens(repo_path)
        repo, err = _host._guard_path(repo_path, must_exist=True)
        if err:
            return err
        if not (repo / ".git").exists():
            return f"Not a git repo: {repo}"
        cmd = f"git -C {shlex.quote(str(repo))} {git_args}"
        return _host.run_shell(cmd, cwd=repo)


# --- Filesystem tool substrate: audit trail -------------------------------------------------
# Mutating file tools append to ~/.primus/audit.jsonl (best-effort — a failed log write never
# fails the tool) so a turn can answer "what did you change?" from disk. Tool OBSERVATIONS stay
# short human strings (✓/✗ family); the structured record lives only in the audit log.

_FS_TURN_FILES: set[str] = set()  # paths mutated this turn — read by the final status object


def _fs_turn_reset() -> None:
    _FS_TURN_FILES.clear()


def _fs_turn_files() -> list[str]:
    return sorted(_FS_TURN_FILES)


def _fs_audit(action: str, path: Any, detail: str = "") -> None:
    """Append one line to the shared audit.jsonl + track the file for the turn status. Never raises."""
    try:
        _FS_TURN_FILES.add(str(path))
        from primus.core.audit import write_audit  # noqa: PLC0415

        extra = str(path)
        if detail:
            extra = f"{path}: {detail}"
        write_audit(action, ok=True, detail=extra)
    except Exception:  # noqa: BLE001 — auditing must never break the tool itself
        pass


@tool
def read_file(path: str, offset: int = 0, max_lines: int = 200) -> str:
    """Read a text file under home or a safe system read zone (/usr/share, /opt, app folders).

    `offset` skips that many lines first (0-based) and `max_lines` caps how many are returned,
    so large files are read in windows instead of all at once.
    """
    target, err = _host._guard_path(path, must_exist=True, allow_system_read=True)
    if err:
        return f"✗ {err}"
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"✗ Could not read {target}: {exc}"
    start = max(0, int(offset))
    window = lines[start:start + max(1, int(max_lines))]
    body = "\n".join(window)
    remaining = len(lines) - (start + len(window))
    note = f"\n… (+{remaining} more lines — use offset={start + len(window)} to continue)" if remaining > 0 else ""
    prefix = f"(lines {start + 1}–{start + len(window)} of {len(lines)})\n" if start else ""
    return prefix + body + note


def _filemaid_write_block(path) -> Optional[str]:
    """FileMaid layer after ``_guard_path``. Empty = allowed. Never a ``@tool``."""
    from primus.core.filemaid_safety import is_path_blocked  # noqa: PLC0415

    blocked, reason = is_path_blocked(path)
    if blocked:
        return f"✗ Refusing blocked path: {reason}"
    return None


@tool
def write_file(path: str, content: str, append: bool = False, overwrite: bool = False) -> str:
    """Write/append a text file under home.

    Atomic: content lands via temp-file + replace, so a crash never leaves a truncated file.
    Overwriting an existing file first saves a timestamped .bak next to it (best-effort).
    Suggest mode writes nothing — it previews what would happen. `overwrite` is accepted for
    backward compatibility but is NOT required: Execute-mode writes simply proceed.
    """
    del overwrite  # compatibility kwarg — the Suggest/Execute mode is the gate, not a flag
    target, err = _host._guard_path(path)
    if err:
        return f"✗ {err}"
    blocked = _filemaid_write_block(target)
    if blocked:
        return blocked
    exists = target.exists()
    action = "append" if append else ("overwrite" if exists else "write")
    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return f"[Suggest] Would {action} {len(content)} chars into `{target}`. Switch to Execute to apply."
    try:
        if append:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(content)
        else:
            if exists:
                try:
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    shutil.copy2(target, target.with_name(f"{target.name}.{stamp}.bak"))
                except OSError:
                    pass  # the backup is best-effort — it never blocks the write
            _host._atomic_write_text(target, content)
    except OSError as exc:
        return f"✗ Could not write {target}: {exc}"
    _fs_audit(action, target, f"{len(content)} chars")
    return f"✓ {'Appended' if append else 'Wrote'} `{target}`"


@tool
def delete_file(path: str, confirm: bool = False) -> str:
    """Delete a file or directory under home — destructive.

    Goes through the EXACT same approval path as `rm` in the terminal: the deletion is queued
    for one-click approval in both Suggest and Execute mode; it never deletes directly,
    whatever `confirm` says (the flag is accepted only so older call sites don't break).
    Paths outside home are always refused.
    """
    del confirm  # not a safety gate — the approval queue is
    target, err = _host._guard_path(path, must_exist=True)
    if err:
        return f"✗ {err}"
    cmd = f"rm -rf {shlex.quote(str(target))}" if target.is_dir() else f"rm {shlex.quote(str(target))}"
    return _host.run_shell(cmd)


@tool
def verify_python(path: str = "", code: str = "") -> str:
    """Statically verify Python code: compile-check (py_compile) + ruff lint if installed.

    Pass EITHER a file `path` under home OR a `code` snippet. This is SAFE — it never runs the
    code, it only compiles (catches syntax/indentation errors) and lints it (catches likely bugs
    and style issues). Use it right after writing or editing Python so the code actually works
    before you hand it to the operator. Returns the diagnostics to fix, or a clean pass.
    """
    import py_compile

    target: Optional[Path] = None
    cleanup = False
    if path.strip():
        target, err = _host._guard_path(path, must_exist=True, allow_system_read=True)
        if err:
            return err
    elif code.strip():
        tmp_dir = _host.APP_DIR / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_dir / f"verify_{uuid.uuid4().hex[:8]}.py"
        target.write_text(code, encoding="utf-8")
        cleanup = True
    else:
        return "verify_python: provide a file `path` or a `code` snippet."

    findings: list[str] = []
    try:
        # 1) Compile check — catches syntax/indentation errors WITHOUT executing the code.
        try:
            py_compile.compile(str(target), doraise=True)
        except py_compile.PyCompileError as exc:
            return f"✗ Compile error:\n{str(exc).strip()[:1500]}"
        findings.append("✓ Compiles cleanly (py_compile)")

        # 2) Ruff lint (static, fast) when available — high-signal bug/style checks.
        ruff = shutil.which("ruff")
        if ruff:
            try:
                proc = subprocess.run(
                    [ruff, "check", "--quiet", str(target)],
                    capture_output=True, text=True, timeout=30,
                )
                out = (proc.stdout + proc.stderr).strip()
                findings.append("✓ ruff: clean" if proc.returncode == 0
                                else "⚠ ruff findings:\n" + out[:1800])
            except Exception as exc:  # noqa: BLE001
                findings.append(f"(ruff skipped: {exc})")
        else:
            findings.append("(ruff not installed — `uv pip install ruff` enables lint checks)")
    finally:
        if cleanup and target and target.exists():
            try:
                target.unlink()
            except OSError:
                pass
    return "\n".join(findings)


def _verify_python_text(code: str) -> tuple[bool, str]:
    """Static-verify a Python snippet → (ok, report). Compile-check + ruff; never executes code.

    `ok` is False only for REAL errors (syntax/compile failures, or high-signal ruff codes like
    undefined names / redefinitions), not mere style nits — so the auto-repair loop fires on
    things that actually break, not cosmetic warnings.
    """
    import py_compile

    tmp_dir = _host.APP_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / f"verify_{uuid.uuid4().hex[:8]}.py"
    target.write_text(code, encoding="utf-8")
    findings: list[str] = []
    ok = True
    try:
        try:
            py_compile.compile(str(target), doraise=True)
            findings.append("✓ Compiles cleanly")
        except py_compile.PyCompileError as exc:
            return False, f"✗ Compile error:\n{str(exc).strip()[:1500]}"
        ruff = shutil.which("ruff")
        if ruff:
            try:
                proc = subprocess.run(
                    [ruff, "check", "--quiet", str(target)],
                    capture_output=True, text=True, timeout=30,
                )
                out = (proc.stdout + proc.stderr).strip()
                if proc.returncode == 0:
                    findings.append("✓ ruff: clean")
                else:
                    findings.append("⚠ ruff findings:\n" + out[:1800])
                    # Treat syntax (E999) and pyflakes bug codes (undefined name, redefinition,
                    # unused import in __all__, etc.) as real failures worth repairing.
                    if re.search(r"\b(E999|F8\d\d|F4\d\d|F6\d\d|F7\d\d)\b", out):
                        ok = False
            except Exception as exc:  # noqa: BLE001
                findings.append(f"(ruff skipped: {exc})")
    finally:
        try:
            target.unlink()
        except OSError:
            pass
    return ok, "\n".join(findings)


# ---------------------------------------------------------------------------
# Office / OnlyOffice document tools — read/edit .docx, .xlsx, .pptx
#
# Safety model: reads are sandboxed to home + system read zones; edits are confined to
# home, make a timestamped .bak backup before touching the file (never destructive), and
# in Suggest mode only PREVIEW the change instead of applying it.
# ---------------------------------------------------------------------------

def _office_backup(path: Path) -> Optional[Path]:
    """Timestamped sibling backup so an edit is always recoverable. Best-effort."""
    try:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = path.with_name(f"{path.name}.{ts}.bak")
        shutil.copy2(path, bak)
        return bak
    except Exception as exc:  # noqa: BLE001
        _host.log.warning("Office backup failed for %s: %s", path, exc)
        return None


def _docx_replace_text(document: Any, find: str, replace: str) -> int:
    """Replace `find` with `replace` across paragraphs + table cells. Returns hit count."""
    count = 0

    def _do(paragraphs: Any) -> None:
        nonlocal count
        for p in paragraphs:
            if find and find in p.text:
                count += p.text.count(find)
                new_text = p.text.replace(find, replace)
                if p.runs:  # preserve the first run's formatting, clear the rest
                    p.runs[0].text = new_text
                    for r in p.runs[1:]:
                        r.text = ""
                else:
                    p.text = new_text

    _do(document.paragraphs)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                _do(cell.paragraphs)
    return count


@tool
def read_office_file(path: str, sheet_name: Optional[str] = None, max_rows: int = 100) -> str:
    """Read and summarize Office documents — .docx (Word), .xlsx (Excel), .pptx (PowerPoint).

    Returns clean, usable text: Word paragraphs + tables, Excel cell grids (pass sheet_name
    to target one tab; max_rows caps output), PowerPoint slide text + tables. Read-only and
    safe. Also falls back to PDF/ODT/text extraction for other document types.
    """
    target, err = _host._guard_path(path, must_exist=True, allow_system_read=True)
    if err:
        return err
    ext = target.suffix.lower()
    try:
        if ext == ".xlsx":
            text, e = _host._extract_xlsx_text(target, sheet_name=sheet_name, max_rows=max_rows)
        elif ext == ".pptx":
            text, e = _host._extract_pptx_text(target)
        elif ext == ".docx":
            text, e = _host._extract_docx_text(target)
        else:
            text, e = _host.load_document_text(target)  # pdf/odt/txt/etc.
        if e:
            return f"Could not read `{target.name}`: {e}"
        if not text.strip():
            return f"`{target.name}` has no extractable text."
        return text[:10_000] + ("\n… (truncated)" if len(text) > 10_000 else "")
    except Exception as exc:  # noqa: BLE001
        return f"Failed to read `{target.name}`: {str(exc)[:120]}"


@tool
def edit_docx(
    path: str,
    action: Literal["replace", "append", "add_heading", "add_table"],
    content: str = "",
    find: str = "",
    replace: str = "",
    heading_text: str = "",
    table_data: str = "",  # e.g. "Header1,Header2\nRow1Col1,Row1Col2"
) -> str:
    """Edit a Word .docx safely. Actions:
      • replace      — replace every `find` with `replace` across paragraphs & tables
      • append       — add `content` (newlines become separate paragraphs) at the end
      • add_heading  — add `heading_text` (or `content`) as a level-1 heading
      • add_table    — add a table from `table_data` ("H1,H2\\nr1c1,r1c2")
    Makes a timestamped .bak backup before changing anything (never destructive). In Suggest
    mode it previews the change instead of applying it. Creates the file for add_* if missing.
    """
    target, err = _host._guard_path(path)  # writes confined to home
    if err:
        return err
    try:
        import docx
    except ImportError:
        return "python-docx not installed. Run: uv pip install python-docx"

    if action == "replace" and not target.exists():
        return f"Not found: {target} (replace needs an existing document)."

    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        previews = {
            "replace": f"replace '{(find or content)[:30]}' → '{replace[:30]}'",
            "append": f"append {len((content or replace).splitlines()) or 1} paragraph(s)",
            "add_heading": f"add heading '{(heading_text or content)[:40]}'",
            "add_table": f"add a table from {len((table_data or content).splitlines())} row(s)",
        }
        return (
            f"[Suggest] Would {previews.get(action, action)} in `{target.name}`. "
            "Switch to Execute (or approve) to apply — a .bak backup is made first."
        )

    try:
        document = docx.Document(str(target)) if target.exists() else docx.Document()
        backup = _office_backup(target) if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)

        if action == "replace":
            needle = find or content
            if not needle:
                return "replace needs `find` (text to find) and `replace`."
            n = _docx_replace_text(document, needle, replace)
            if n == 0:
                return f"No occurrences of '{needle[:40]}' in `{target.name}` — nothing changed."
            msg = f"Replaced {n} occurrence(s) of '{needle[:30]}'"
        elif action == "append":
            text = content or replace
            if not text.strip():
                return "append needs `content`."
            for line in text.split("\n"):
                document.add_paragraph(line)
            msg = "Appended paragraph(s)"
        elif action == "add_heading":
            htext = heading_text or content
            if not htext.strip():
                return "add_heading needs `heading_text`."
            document.add_heading(htext, level=1)
            msg = f"Added heading '{htext[:40]}'"
        elif action == "add_table":
            data = table_data or content
            grid = [[c.strip() for c in r.split(",")] for r in data.split("\n") if r.strip()]
            if not grid:
                return "add_table needs `table_data` (CSV-like: \"H1,H2\\nr1,r2\")."
            cols = max(len(r) for r in grid)
            table = document.add_table(rows=len(grid), cols=cols)
            try:
                table.style = "Light Grid Accent 1"
            except Exception:  # noqa: BLE001 — style may not exist in the template
                pass
            for ri, row in enumerate(grid):
                for ci in range(cols):
                    table.rows[ri].cells[ci].text = row[ci] if ci < len(row) else ""
            msg = f"Added {len(grid)}×{cols} table"
        else:
            return f"Unknown action '{action}'. Use replace|append|add_heading|add_table."

        document.save(str(target))
        out = f"✓ {msg} → `{target.name}`"
        if backup:
            out += f"  (backup: {backup.name})"
        return out
    except Exception as exc:  # noqa: BLE001
        return f"DOCX edit failed for `{target.name}`: {str(exc)[:120]}"


@tool
def edit_xlsx(
    path: str,
    sheet_name: str = "Sheet1",
    updates: str = "",  # e.g. "A1:New Value,B2:=SUM(C1:C10)"
    new_data: Optional[dict] = None,
) -> str:
    """Update cells or append rows in an Excel .xlsx (openpyxl).

    updates: comma-separated CELL:VALUE pairs, e.g. "A1:Revenue,B2:=SUM(C1:C10)". A value
             starting with '=' is written as a live formula.
    new_data: optional dict {"headers": [...], "rows": [[...], ...]} appended to the sheet
              (also accepts a JSON string).
    Creates the file/sheet if missing. Makes a .bak backup before changes; previews in
    Suggest mode. Never destructive.
    """
    target, err = _host._guard_path(path)
    if err:
        return err
    try:
        import openpyxl
    except ImportError:
        return "openpyxl not installed. Run: uv pip install openpyxl"

    pairs: list[tuple[str, str]] = []
    for chunk in updates.split(","):
        if ":" in chunk:
            cell, _, val = chunk.partition(":")
            if cell.strip():
                pairs.append((cell.strip().upper(), val))

    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        bits = ", ".join(f"{c}={v[:20]}" for c, v in pairs) or "no cells"
        extra = " + append rows" if new_data else ""
        return (
            f"[Suggest] Would set [{sheet_name}] {bits}{extra} in `{Path(path).name}`. "
            "Switch to Execute (or approve) to apply — a .bak backup is made first."
        )

    try:
        if target.exists():
            wb = openpyxl.load_workbook(str(target))
            backup = _office_backup(target)
        else:
            wb = openpyxl.Workbook()
            wb.active.title = sheet_name
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = None
        ws = wb[sheet_name] if sheet_name in wb.sheetnames else wb.create_sheet(sheet_name)

        changed = 0
        for cell, val in pairs:
            try:
                ws[cell] = val  # openpyxl auto-treats a leading '=' as a formula
                changed += 1
            except Exception as exc:  # noqa: BLE001
                return f"Bad cell reference '{cell}': {str(exc)[:60]}"

        added_rows = 0
        nd = new_data
        if isinstance(nd, str) and nd.strip():
            try:
                nd = json.loads(nd)
            except json.JSONDecodeError:
                nd = None
        if isinstance(nd, dict):
            headers = nd.get("headers")
            if headers:
                ws.append([str(h) for h in headers])
                added_rows += 1
            for row in nd.get("rows", []) or []:
                ws.append(list(row))
                added_rows += 1

        if not changed and not added_rows:
            return "Nothing to do — provide `updates` (e.g. \"A1:Hello\") and/or `new_data`."

        wb.save(str(target))
        out = f"✓ Updated {changed} cell(s)"
        if added_rows:
            out += f", appended {added_rows} row(s)"
        out += f" in `{target.name}` [{sheet_name}]"
        if backup:
            out += f"  (backup: {backup.name})"
        return out
    except Exception as exc:  # noqa: BLE001
        return f"XLSX edit failed for `{Path(path).name}`: {str(exc)[:120]}"


@tool
def open_in_onlyoffice(path: str) -> str:
    """Open a document in OnlyOffice Desktop Editors (falls back to the system default app)."""
    target, err = _host._guard_path(path, must_exist=True, allow_system_read=True)
    if err:
        return err
    candidates = ["onlyoffice-desktopeditors", "DesktopEditors", "desktopeditors", "onlyoffice"]
    binary = next((b for b in candidates if shutil.which(b)), None)
    try:
        if binary:
            subprocess.Popen(
                [binary, str(target)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
            )
            return f"Opening `{target.name}` in OnlyOffice…"
        if shutil.which("xdg-open"):
            subprocess.Popen(
                ["xdg-open", str(target)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
            )
            return (
                f"OnlyOffice not found — opened `{target.name}` in the default app instead.\n"
                "Install OnlyOffice: `sudo snap install onlyoffice-desktopeditors` "
                "(or `flatpak install flathub org.onlyoffice.desktopeditors`)."
            )
        return (
            "OnlyOffice isn't installed and `xdg-open` is unavailable.\n"
            "Install: `sudo snap install onlyoffice-desktopeditors`."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Couldn't open `{target.name}`: {str(exc)[:120]}"


@tool
def create_directory(path: str) -> str:
    """Create directory under home."""
    target, err = _host._guard_path(path)
    if err:
        return f"✗ {err}"
    blocked = _filemaid_write_block(target)
    if blocked:
        return blocked
    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return f"[Suggest] Would create `{target}`. Switch to Execute to apply."
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"✗ Could not create {target}: {exc}"
    _fs_audit("mkdir", target)
    return f"✓ Created `{target}`"


@tool
def move_path(source: str, destination: str) -> str:
    """Move/rename file or folder under home."""
    src, err = _host._guard_path(source, must_exist=True)
    if err:
        return f"✗ {err}"
    dst, err = _host._guard_path(destination)
    if err:
        return f"✗ {err}"
    blocked = _filemaid_write_block(dst)
    if blocked:
        return blocked
    cmd = f"mv {shlex.quote(str(src))} {shlex.quote(str(dst))}"
    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return _host.suggest_command(cmd)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except OSError as exc:
        return f"✗ Could not move {src} → {dst}: {exc}"
    _fs_audit("move", dst, detail=f"from {src}")
    return f"✓ Moved → `{dst}`"


@tool
def copy_path(source: str, destination: str) -> str:
    """Copy file or directory under home."""
    src, err = _host._guard_path(source, must_exist=True)
    if err:
        return f"✗ {err}"
    dst, err = _host._guard_path(destination)
    if err:
        return f"✗ {err}"
    blocked = _filemaid_write_block(dst)
    if blocked:
        return blocked
    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return _host.suggest_command(f"cp -r {shlex.quote(str(src))} {shlex.quote(str(dst))}")
    try:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except OSError as exc:
        return f"✗ Could not copy {src} → {dst}: {exc}"
    _fs_audit("copy", dst, detail=f"from {src}")
    return f"✓ Copied → `{dst}`"


@tool
def list_directory(path: str = "~", pattern: str = "*") -> str:
    """List directory entries. Works in home and safe system read zones (/usr/bin, /opt, app dirs)."""
    target, err = _host._guard_path(path, must_exist=True, allow_system_read=True)
    if err:
        return err
    entries = sorted(target.glob(pattern))[:200]
    if not entries:
        return f"0 files, 0 folders in `{target}`"
    return "\n".join(
        f"[{'dir' if e.is_dir() else 'file'}] {e.name} ({e.stat().st_size if e.is_file() else 0} B)"
        for e in entries
    )


@tool
def search_files(query: str, search_root: str = "~", max_results: int = 40) -> str:
    """Find files by name under home or a safe system read zone (e.g. /usr/share/applications, /opt).

    Use search_root="all" (or "device") to sweep home + the common system app/binary locations.
    """
    roots: list[Path] = []
    if search_root.strip().lower() in ("all", "device", "everywhere", "system"):
        roots = [_host.HOME] + [d for d in _host.READONLY_SYSTEM_DIRS if d.exists()]
    else:
        root, err = _host._guard_path(search_root, must_exist=True, allow_system_read=True)
        if err:
            return err
        roots = [root]
    q = query if "*" in query else f"*{query}*"
    per_root = max(5, min(max_results, _host.FIND_MAX_RESULTS) // max(1, len(roots)))
    quoted_roots = " ".join(shlex.quote(str(r)) for r in roots)
    cmd = (
        f"find {quoted_roots} -iname {shlex.quote(q)} "
        r"-not -path '*/.*' 2>/dev/null "
        f"| head -n {min(max_results, _host.FIND_MAX_RESULTS)}"
    )
    _ = per_root  # find's global head bound is sufficient
    return _host.run_shell(cmd, force=True)


@tool
def organize_downloads(dry_run: bool = True) -> str:
    """Organize ~/Downloads by file type (pdf, images, archives, etc.)."""
    prefs = _host.load_memory().get("preferences", {})
    dl = Path(prefs.get("downloads_folder", str(_host.DOWNLOADS))).expanduser()
    if not dl.exists():
        return f"Downloads folder not found: {dl}"
    type_map = {
        "documents": {".pdf", ".doc", ".docx", ".txt", ".md", ".odt"},
        "images": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"},
        "archives": {".zip", ".tar", ".gz", ".bz2", ".7z", ".rar"},
        "media": {".mp4", ".mkv", ".mp3", ".wav"},
        "code": {".py", ".js", ".ts", ".json", ".yaml", ".toml"},
    }
    moves: list[str] = []
    for item in sorted(dl.iterdir()):
        if not item.is_file() or item.name.startswith("."):
            continue
        ext = item.suffix.lower()
        folder = "other"
        for name, exts in type_map.items():
            if ext in exts:
                folder = name
                break
        dest = dl / folder / item.name
        moves.append(f"{item.name} → {folder}/")
    if not moves:
        return "Downloads already tidy."
    preview = "Downloads organize plan:\n" + "\n".join(moves[:50])
    if dry_run or _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return preview + "\n\nSet dry_run=false and Execute mode to apply."
    for item in sorted(dl.iterdir()):
        if not item.is_file() or item.name.startswith("."):
            continue
        ext = item.suffix.lower()
        folder = "other"
        for name, exts in type_map.items():
            if ext in exts:
                folder = name
                break
        dest_dir = dl / folder
        dest_dir.mkdir(exist_ok=True)
        shutil.move(str(item), str(dest_dir / item.name))
    _host.log_past_task("Organize Downloads", f"Sorted {len(moves)} files", "completed")
    return f"Organized {len(moves)} files in {dl}"


@tool
def clean_temp_files(dry_run: bool = True) -> str:
    """Analyze and optionally clean temp/cache locations."""
    targets = [
        _host.HOME / ".cache",
        Path("/tmp"),
        _host.HOME / ".local" / "share" / "Trash",
    ]
    report_lines = []
    total = 0
    for t in targets:
        if t.exists():
            size = _host.run_shell(f"du -sh {shlex.quote(str(t))} 2>/dev/null | cut -f1", force=True)
            report_lines.append(f"{t}: {size.strip()}")
    preview = "Temp/cache report:\n" + "\n".join(report_lines)
    preview += "\n\nSuggested safe cleanup (preview):\n"
    preview += "- rm -rf ~/.cache/thumbnails/*\n"
    preview += "- find ~/.cache -type f -atime +30 -delete  (stale cache)\n"
    preview += "- rm -rf ~/.local/share/Trash/files/*\n"
    if dry_run or _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return preview + "\nUse Execute + approval to run cleanup commands."
    return preview


@tool
def sort_client_folders(root: str = "", dry_run: bool = True) -> str:
    """List or create client folder structure under ~/Clients."""
    prefs = _host.load_memory().get("preferences", {})
    base = _host._resolve_path(root or prefs.get("client_folders_root", str(_host.HOME / "Clients")))
    if not _host._path_in_home(base):
        return f"Outside home: {base}"
    base.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base.iterdir() if d.is_dir()]
    report = f"Client root: {base}\nFolders: {', '.join(existing) or '(none)'}\n"
    report += "Tip: create per-client dirs like Clients/Acme/{docs,invoices,assets}."
    if dry_run or _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return report
    return report


@tool
def organize_by_extension(source_dir: str, dry_run: bool = True) -> str:
    """Organize any folder by file extension."""
    root, err = _host._guard_path(source_dir, must_exist=True)
    if err:
        return err
    plan = []
    for item in sorted(root.iterdir()):
        if item.is_file() and not item.name.startswith("."):
            ext = item.suffix.lstrip(".").lower() or "no_ext"
            plan.append(f"{item.name} → {ext}/")
    if not plan:
        return "Nothing to organize."
    if dry_run or _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return "Plan:\n" + "\n".join(plan[:40])
    for item in sorted(root.iterdir()):
        if item.is_file() and not item.name.startswith("."):
            ext = item.suffix.lstrip(".").lower() or "no_ext"
            folder = root / ext
            folder.mkdir(exist_ok=True)
            shutil.move(str(item), str(folder / item.name))
    return f"Organized {len(plan)} files."


@tool
def package_management(action: str, target: str = "") -> str:
    """Linux packages. action: list|search|updates|install|remove|info. Uses apt."""
    if action == "list":
        return _host.run_shell("apt list --installed 2>/dev/null | head -40", force=True)
    if action == "search" and target:
        return _host.run_shell(f"apt search {shlex.quote(target)} 2>/dev/null | head -25", force=True)
    if action == "updates":
        return _host.run_shell("apt list --upgradable 2>/dev/null", force=True)
    if action == "info" and target:
        return _host.run_shell(f"apt show {shlex.quote(target)} 2>/dev/null | head -30", force=True)
    if action == "install" and target:
        return _host.suggest_command(f"sudo apt install -y {shlex.quote(target)}")
    if action == "remove" and target:
        return _host.suggest_command(f"sudo apt remove {shlex.quote(target)}")
    return "Actions: list|search|updates|install|remove|info"


@tool
def service_control(action: str, service: str = "") -> str:
    """systemd services. action: list|status|start|stop|restart|enabled."""
    if action == "list":
        return _host.run_shell("systemctl list-units --type=service --state=running --no-pager | head -25", force=True)
    if not service:
        return "Provide service name."
    if action == "status":
        return _host.run_shell(f"systemctl status {shlex.quote(service)} --no-pager | head -20", force=True)
    if action in ("start", "stop", "restart"):
        return _host.suggest_command(f"sudo systemctl {action} {shlex.quote(service)}")
    if action == "enabled":
        return _host.run_shell(f"systemctl is-enabled {shlex.quote(service)} 2>&1", force=True)
    return "Actions: list|status|start|stop|restart|enabled"


@tool
def backup_advisor(target_path: str = "~", remote: str = "") -> str:
    """Suggest backup strategy (rsync/rclone/tar) with preview commands."""
    prefs = _host.load_memory().get("preferences", {})
    remote = remote or prefs.get("backup_remote_hint", "onedrive:")
    target = shlex.quote(str(_host._resolve_path(target_path)))
    lines = [
        f"Backup target: {target}",
        "",
        "**Suggested workflow:**",
        "1. Dry-run rsync to external or cloud folder",
        "2. Verify with diff/checksum",
        "3. Schedule via cron or systemd timer",
        "",
        "Preview commands:",
        f"rsync -avhn --delete {target}/ ~/Backups/home-snapshot/",
        f"rclone sync {target} {remote}Backup/home --dry-run",
        f"tar -czvf ~/Backups/home-$(date +%F).tar.gz {target}",
    ]
    return "\n".join(lines)


@tool
def performance_tune(mode: str = "report") -> str:
    """Performance report and tuning suggestions."""
    parts = [_host.build_status_bar(), ""]
    parts.append(_host.run_shell("cat /proc/sys/vm/swappiness 2>/dev/null; echo 'swappiness ^'", force=True))
    parts.append(_host.run_shell("systemd-analyze blame 2>/dev/null | head -12", force=True))
    parts.append(_host.run_shell("systemctl list-units --type=service --state=running --no-pager | wc -l", force=True))
    if mode == "full":
        parts.append(_host.run_shell("free -h; echo '---'; df -h", force=True))
    parts.append(
        "\nTuning tips: reduce startup services, check swappiness (10-60), "
        "clear stale caches, monitor top CPU processes."
    )
    return "\n".join(parts)


@tool
def troubleshoot(issue: str = "general") -> str:
    """Collect diagnostic info for common Linux issues."""
    sections = [f"Issue context: {issue}", ""]
    sections.append("## Disk & memory")
    sections.append(_host.run_shell("df -h; free -h", force=True))
    sections.append("## Recent errors (journal)")
    sections.append(_host.run_shell("journalctl -p 3 -b --no-pager | tail -20", force=True))
    sections.append("## dmesg tail")
    sections.append(_host.run_shell("dmesg 2>/dev/null | tail -15", force=True))
    sections.append("## Network")
    sections.append(_host.run_shell("ping -c 2 1.1.1.1 2>&1 | tail -3", force=True))
    sections.append("## Top CPU")
    sections.append(_host.run_shell("ps aux --sort=-%cpu | head -8", force=True))
    return "\n\n".join(sections)


@tool
def device_management(action: str, value: str = "") -> str:
    """Device control: lock|suspend|wifi|bluetooth|battery|wifi_connect."""
    if action == "lock":
        for cmd in (
            "loginctl lock-session",
            "xdg-screensaver lock",
            "gnome-screensaver-command -l",
        ):
            if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
                return _host.suggest_command(cmd)
            r = _host.run_shell(cmd, force=True)
            if "exit_code=0" in r:
                return "Screen locked."
        return _host.suggest_command("loginctl lock-session")
    if action == "suspend":
        return _host.suggest_command("systemctl suspend")
    if action == "hibernate":
        return _host.suggest_command("systemctl hibernate")
    if action == "battery":
        return _host.run_shell(
            "upower -i $(upower -e | grep BAT | head -1) 2>/dev/null; "
            "cat /sys/class/power_supply/BAT*/capacity 2>/dev/null",
            force=True,
        )
    if action == "wifi":
        return _host.run_shell("nmcli -t -f ACTIVE,SSID,SIGNAL,SECURITY dev wifi | head -10", force=True)
    if action == "wifi_on":
        return _host.suggest_command("nmcli radio wifi on")
    if action == "wifi_off":
        return _host.suggest_command("nmcli radio wifi off")
    if action == "wifi_connect" and value:
        return _host.suggest_command(f"nmcli dev wifi connect {shlex.quote(value)}")
    if action == "bluetooth":
        return _host.run_shell("bluetoothctl show 2>/dev/null | head -15; bluetoothctl devices 2>/dev/null | head -10", force=True)
    if action == "bt_on":
        return _host.suggest_command("bluetoothctl power on")
    if action == "bt_off":
        return _host.suggest_command("bluetoothctl power off")
    return (
        "Actions: lock|suspend|hibernate|battery|wifi|wifi_on|wifi_off|"
        "wifi_connect (value=SSID)|bluetooth|bt_on|bt_off"
    )


# Common desktop apps → ordered launch candidates (native binary first, then flatpak).
COMMON_APP_LAUNCHERS: dict[str, list[str]] = {
    "zoom": ["zoom", "flatpak run us.zoom.Zoom"],
    "chrome": ["google-chrome-stable", "google-chrome", "flatpak run com.google.Chrome"],
    "google chrome": ["google-chrome-stable", "google-chrome"],
    "chromium": ["chromium-browser", "chromium", "flatpak run org.chromium.Chromium"],
    "firefox": ["firefox", "flatpak run org.mozilla.firefox"],
    "slack": ["slack", "flatpak run com.slack.Slack"],
    "discord": ["discord", "flatpak run com.discordapp.Discord"],
    "code": ["code", "flatpak run com.visualstudio.code"],
    "vscode": ["code"],
    "cursor": ["cursor"],
    "obsidian": ["obsidian", "flatpak run md.obsidian.Obsidian"],
    "spotify": ["spotify", "flatpak run com.spotify.Client"],
    "files": ["nautilus", "nemo", "dolphin", "xdg-open ."],
    "file manager": ["nautilus", "nemo", "dolphin"],
    "terminal": ["gnome-terminal", "x-terminal-emulator", "konsole", "xterm"],
    "calculator": ["gnome-calculator", "kcalc"],
    "settings": ["gnome-control-center", "systemsettings"],
    "text editor": ["gnome-text-editor", "gedit", "kate"],
    "thunderbird": ["thunderbird", "flatpak run org.mozilla.Thunderbird"],
    "libreoffice": ["libreoffice"],
}

# Folder shortcuts so "open my downloads" maps to a real path.
_FOLDER_SHORTCUTS: dict[str, str] = {
    "downloads": "~/Downloads",
    "desktop": "~/Desktop",
    "documents": "~/Documents",
    "home": "~",
    "home folder": "~",
    "pictures": "~/Pictures",
    "music": "~/Music",
    "videos": "~/Videos",
}


def _spawn_detached(command: str) -> str:
    """Launch a GUI app/file in the background, fully detached from Primus."""
    if _host.PrimusSession.is_cancelled():
        return "⏹ Halted — launch skipped."
    # Don't open the same app/URL twice for one request unless explicitly asked.
    if _host.PrimusSession.already_did("launch", command):
        return f"Already opened that (`{command}`) a moment ago — skipping the duplicate."
    _host.PrimusSession.push_history(command)
    try:
        subprocess.Popen(  # noqa: S602 - launching trusted local apps
            command,
            shell=True,
            cwd=str(_host.HOME),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ},
        )
        _host.PrimusSession.mark_done("launch", command)
        return f"▶ Launched: `{command}`"
    except Exception as exc:  # noqa: BLE001
        return f"Failed to launch `{command}`: {exc}"


@tool
def open_application(target: str) -> str:
    """Open/launch a desktop app, folder, file, or URL immediately.

    Use for requests like "open Zoom", "launch Chrome", "open my Downloads folder".
    Accepts an app name (zoom, chrome, firefox, slack, code, files, terminal, ...),
    a folder shortcut (downloads, desktop, documents, home), a path, or a URL.
    Apps launch in the background and the function returns right away.
    """
    raw = (target or "").strip()
    if not raw:
        return "Provide an app name, folder, path, or URL to open."
    key = raw.lower()

    # URLs → default browser
    if re.match(r"^https?://", raw, re.I):
        if shutil.which("xdg-open"):
            return _spawn_detached(f"xdg-open {shlex.quote(raw)}")
        return "No xdg-open available to open URLs."

    # Known folder shortcut
    if key in _FOLDER_SHORTCUTS:
        path = Path(_FOLDER_SHORTCUTS[key]).expanduser()
        opener = "xdg-open" if shutil.which("xdg-open") else "nautilus"
        return _spawn_detached(f"{opener} {shlex.quote(str(path))}")

    # Known app → try candidates in order
    candidates = list(COMMON_APP_LAUNCHERS.get(key, []))
    candidates.append(raw)  # fall back to the literal name as a binary
    for cand in candidates:
        leader = cand.split()[0]
        if leader == "flatpak":
            if shutil.which("flatpak"):
                return _spawn_detached(cand)
        elif shutil.which(leader):
            return _spawn_detached(cand)

    # Existing path on disk
    maybe_path = Path(raw).expanduser()
    if maybe_path.exists() and shutil.which("xdg-open"):
        return _spawn_detached(f"xdg-open {shlex.quote(str(maybe_path))}")

    # Last resort: .desktop launcher lookup
    if shutil.which("gtk-launch"):
        return _spawn_detached(f"gtk-launch {shlex.quote(raw)}")
    if shutil.which("xdg-open"):
        return _spawn_detached(f"xdg-open {shlex.quote(raw)}")
    return (
        f"Couldn't find a launcher for '{raw}'. "
        "Try the exact binary name, or tell me the full command to run."
    )


# Browser binaries (native first, then flatpak) for explicit browser control.
_BROWSER_BINS: dict[str, list[str]] = {
    "firefox": ["firefox", "flatpak run org.mozilla.firefox"],
    "chrome": ["google-chrome-stable", "google-chrome", "flatpak run com.google.Chrome"],
    "chromium": ["chromium-browser", "chromium", "flatpak run org.chromium.Chromium"],
    "brave": ["brave-browser", "brave", "flatpak run com.brave.Browser"],
}

# Search-engine query templates ({q} → url-encoded terms).
_SEARCH_ENGINES: dict[str, str] = {
    "google": "https://www.google.com/search?q={q}",
    "youtube": "https://www.youtube.com/results?search_query={q}",
    "duckduckgo": "https://duckduckgo.com/?q={q}",
    "ddg": "https://duckduckgo.com/?q={q}",
    "bing": "https://www.bing.com/search?q={q}",
    "github": "https://github.com/search?q={q}&type=repositories",
    "reddit": "https://www.reddit.com/search/?q={q}",
    "maps": "https://www.google.com/maps/search/{q}",
    "amazon": "https://www.amazon.com/s?k={q}",
}

# Bare site homepages for "open X to youtube" style requests.
_SITE_HOMES: dict[str, str] = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "maps": "https://www.google.com/maps",
    "drive": "https://drive.google.com",
    "calendar": "https://calendar.google.com",
}


def _resolve_browser_cmd(browser: str) -> Optional[str]:
    """Return a runnable browser command, or xdg-open for the system default."""
    browser = (browser or "default").strip().lower()
    if browser in ("", "default", "browser", "system"):
        return "xdg-open" if shutil.which("xdg-open") else (
            next((b.split()[0] for cands in _BROWSER_BINS.values() for b in cands
                  if shutil.which(b.split()[0])), None)
        )
    for cand in _BROWSER_BINS.get(browser, [browser]):
        leader = cand.split()[0]
        if leader == "flatpak":
            if shutil.which("flatpak"):
                return cand
        elif shutil.which(leader):
            return cand
    return "xdg-open" if shutil.which("xdg-open") else None


# Keep Selenium driver references alive so the launched window isn't garbage-collected.
_selenium_drivers: list[Any] = []


def _selenium_firefox_open(url: str, *, headless: bool = False) -> Optional[str]:
    """Open a URL in Firefox via Selenium as a reliable fallback. Returns None if unavailable.

    Selenium 4 ships Selenium Manager, which auto-resolves geckodriver — so this needs only
    a Firefox install. Lazy-imported and fully guarded: if anything is missing it quietly
    returns None and the caller falls back to its normal error message.
    """
    try:
        from selenium import webdriver  # type: ignore
        from selenium.webdriver.firefox.options import Options  # type: ignore
    except Exception:  # noqa: BLE001 — selenium not installed
        return None
    try:
        opts = Options()
        if headless:
            opts.add_argument("-headless")
        driver = webdriver.Firefox(options=opts)
        driver.get(url)
        _selenium_drivers.append(driver)
        return f"Opened Firefox (Selenium) → {url}"
    except Exception as exc:  # noqa: BLE001 — no geckodriver/Firefox/display
        _host.log.warning("Selenium Firefox fallback failed: %s", exc)
        return None


def _build_browser_url(query_or_url: str, site: str, search: str) -> str:
    """Turn loose intent (URL / search terms / site) into a single navigable URL."""
    from urllib.parse import quote_plus

    text = (query_or_url or "").strip()
    site = (site or "").strip().lower()
    search = (search or "").strip()

    if re.match(r"^https?://", text, re.I):
        return text
    if re.match(r"^[\w-]+(\.[\w-]+)+(/.*)?$", text):  # bare domain like youtube.com/foo
        return "https://" + text

    query = search or (text if site else text)
    if query and (site in _SEARCH_ENGINES or site == "" or site in _SITE_HOMES):
        engine = _SEARCH_ENGINES.get(site, _SEARCH_ENGINES["google"])
        return engine.format(q=quote_plus(query))
    if site in _SITE_HOMES:
        return _SITE_HOMES[site]
    if site:
        return f"https://{site}" if "." in site else f"https://www.{site}.com"
    if text:
        return _SEARCH_ENGINES["google"].format(q=quote_plus(text))
    return "https://www.google.com"


@tool
def browse_web(
    query_or_url: str = "",
    browser: str = "default",
    site: str = "",
    search: str = "",
    visible: bool = False,
) -> str:
    """Get web info silently in the background, OR open a visible browser when asked.

    DEFAULT IS SILENT/HEADLESS (visible=False): no window opens, nothing steals focus —
    Primus fetches the page or search results in the background and returns a clean summary.

    Set **visible=True ONLY** when the operator explicitly wants to see it ("open", "show me",
    "make it visible", "in the browser", "pop it up").

    Args:
      query_or_url: a full URL ("https://…") OR search terms ("cats").
      browser: which browser for visible mode — default | firefox | chrome | chromium | brave.
      site: optional target — google | youtube | bing | duckduckgo | github | reddit | maps | gmail.
      search: explicit search terms (use when query_or_url is empty or is a site).
      visible: True = open a real browser window; False (default) = silent background fetch.

    Examples:
      "look up the latest on Mars rovers"  → browse_web(query_or_url="latest on Mars rovers")  (silent)
      "read https://example.com/post"      → browse_web(query_or_url="https://example.com/post")  (silent)
      "open YouTube"                       → browse_web(site="youtube", visible=True)
      "show me Google in Firefox"          → browse_web(site="google", browser="firefox", visible=True)
    """
    raw = (query_or_url or "").strip()
    terms = (search or "").strip()
    site_l = (site or "").strip().lower()

    # --- Silent background mode (default): fetch info, no window, no focus steal ---
    if not visible:
        is_url = bool(
            re.match(r"^https?://", raw, re.I)
            or re.match(r"^[\w-]+(\.[\w-]+)+(/.*)?$", raw)
        )
        if is_url:
            return _read_article(raw)
        query = terms or raw
        if site_l and site_l not in _SITE_HOMES and not query:
            query = site_l
        elif site_l in ("youtube", "github", "reddit", "maps") and query:
            query = f"{query} {site_l}"
        if not query:
            return "Tell me what to look up (search terms or a URL), or say 'open …' to see it."
        _host.PrimusSession.emit_think("Browse", f"Silent web fetch: {query[:50]}", "running")
        return _host._run_web_search(query, queue_kb=True)

    # --- Visible mode (explicit request): open a real browser window ---
    url = _build_browser_url(query_or_url, site, search)
    cmd = _resolve_browser_cmd(browser)
    if not cmd:
        # Last-resort fallback: drive Firefox via Selenium (auto-manages geckodriver).
        sel = _selenium_firefox_open(url)
        if sel:
            return sel
        return (
            "I can't find a browser to open. Install Firefox or Chrome, or the xdg-utils "
            "package (provides xdg-open), and I'll handle it next time."
        )
    result = _spawn_detached(f"{cmd} {shlex.quote(url)}")
    if result.startswith("Failed"):
        sel = _selenium_firefox_open(url)
        if sel:
            return sel
        return result
    bname = browser if browser and browser.lower() not in ("", "default", "system", "browser") else "your browser"
    return f"Opened {bname} → {url}"


@tool
def list_installed_apps(name_filter: str = "") -> str:
    """List installed desktop applications (from .desktop entries in standard app folders).

    Use for "list all installed applications", "show my desktop apps", "is Zoom installed?".
    Optional name_filter narrows results (case-insensitive).
    """
    flt = (name_filter or "").strip().lower()
    found: dict[str, str] = {}
    for d in _host.APPLICATION_DIRS:
        if not d.exists():
            continue
        for entry in sorted(d.glob("*.desktop")):
            try:
                text = entry.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            name: Optional[str] = None
            hidden = False
            for line in text.splitlines():
                if line.startswith("Name=") and name is None:
                    name = line.split("=", 1)[1].strip()
                elif line.strip() in ("NoDisplay=true", "Hidden=true"):
                    hidden = True
            if not name or hidden:
                continue
            if flt and flt not in name.lower() and flt not in entry.stem.lower():
                continue
            found[name] = entry.stem
    if not found:
        return f"No applications found{f' matching {name_filter!r}' if name_filter else ''}."
    names = sorted(found, key=str.lower)
    shown = names[:120]
    body = "\n".join(f"- {n}" for n in shown)
    extra = f"\n…and {len(names) - 120} more." if len(names) > 120 else ""
    head = f"Installed apps ({len(names)}{', matching ' + repr(name_filter) if name_filter else ''}):"
    return f"{head}\n{body}{extra}"


@tool
def system_info(detail: str = "summary") -> str:
    """System overview."""
    parts = [_host.build_status_bar()]
    parts.append(_host.run_shell("ps aux --sort=-%mem | head -n 8", force=True))
    if detail == "full":
        parts.append(_host.run_shell("df -h; ip -br a", force=True))
    return "\n".join(parts)


@tool
def get_datetime(query: str = "") -> str:
    """Current local date and time — instant, no network. Use for the time, date, or day of week."""
    now = datetime.now()
    day = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
    clock = now.strftime("%I:%M %p").lstrip("0")
    return f"{day} — {clock}"


@tool
def get_weather(location: str = "") -> str:
    """Current weather for a location (defaults to your local area, auto-detected by IP).

    Use for "what's the weather", "weather in Paris", "is it going to rain", "how hot is it".
    Uses wttr.in (free, no API key). Offline-safe — says so cleanly if there's no connection.
    """
    import urllib.parse
    import urllib.request

    loc = (location or "").strip()
    fmt = "%l: %c %t (feels %f), %h humidity, wind %w"
    url = f"https://wttr.in/{urllib.parse.quote(loc)}?format={urllib.parse.quote(fmt)}&m"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310 (trusted host)
            text = resp.read().decode("utf-8", "replace").strip()
    except Exception as exc:  # noqa: BLE001
        return (
            "I couldn't reach the weather service — looks like I'm offline "
            f"({str(exc)[:80]}). I'll grab it once you're connected."
        )
    if not text or "Unknown location" in text or "<" in text or "Sorry" in text:
        return f"I couldn't find weather for {loc or 'your area'} right now."
    return f"Weather — {text}"


@tool
def network_status() -> str:
    """Network interfaces and connectivity."""
    parts = []
    if shutil.which("nmcli"):
        parts.append(_host.run_shell("nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device", force=True))
    parts.append(_host.run_shell("ip -br addr 2>/dev/null", force=True))
    parts.append(_host.run_shell("ping -c 1 -W 2 1.1.1.1 2>&1 | tail -2", force=True))
    return "\n\n".join(parts)


@tool
def disk_cleanup(mode: str = "report") -> str:
    """Disk usage report."""
    if mode == "large":
        return _host.run_shell(
            f"find {shlex.quote(str(_host.HOME))} -xdev -type f -size +500M 2>/dev/null | head -20",
            force=True,
        )
    usage = shutil.disk_usage(_host.HOME)
    return (
        f"Home: {usage.used // (1024**3)}G / {usage.total // (1024**3)}G\n"
        + _host.run_shell("du -sh ~/* 2>/dev/null | sort -hr | head -12", force=True)
    )


@tool
def process_management(action: str = "list", target: str = "") -> str:
    """Processes: list|top|kill."""
    if action == "list":
        return _host.run_shell("ps aux --sort=-%cpu | head -n 12", force=True)
    if action == "top":
        return _host.run_shell("top -bn1 | head -18", force=True)
    if action == "kill" and target:
        return _host.suggest_command(f"kill {shlex.quote(target)}")
    return "Use list|top|kill."


@tool
def rclone_status(remote: str = "") -> str:
    """Rclone remotes and usage."""
    if not shutil.which("rclone"):
        return "rclone not installed."
    parts = [_host.run_shell("rclone listremotes", force=True)]
    if remote:
        r = remote if remote.endswith(":") else remote + ":"
        parts.append(_host.run_shell(f"rclone about {shlex.quote(r)}", force=True))
    return "\n\n".join(parts)


@tool
def manage_todos(action: str, text: str = "") -> str:
    """Todos: list|add|done|clear."""
    _host.ensure_app_dirs()
    lines = _host.TODOS_FILE.read_text(encoding="utf-8").splitlines()
    if action == "list":
        return _host.TODOS_FILE.read_text(encoding="utf-8").strip() or "(empty)"
    if action == "add" and text.strip():
        lines.append(f"- [ ] {text.strip()}  _({datetime.now():%Y-%m-%d})_")
        _host.TODOS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"Added: {text.strip()}"
    if action == "done" and text.strip().isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(lines):
            lines[idx] = lines[idx].replace("- [ ]", "- [x]", 1)
            _host.TODOS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return "Marked done."
    if action == "clear":
        _host.TODOS_FILE.write_text("# Primus Todos\n\n", encoding="utf-8")
        return "Cleared."
    return "Use list|add|done|clear."


@tool
def manage_notes(action: str, text: str = "") -> str:
    """Notes: list|add|search."""
    _host.ensure_app_dirs()
    content = _host.NOTES_FILE.read_text(encoding="utf-8")
    if action == "list":
        return content.strip() or "(empty)"
    if action == "add" and text.strip():
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with _host.NOTES_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"\n## {stamp}\n{text.strip()}\n")
        return f"Saved ({stamp})."
    if action == "search" and text.strip():
        hits = [ln for ln in content.splitlines() if text.lower() in ln.lower()]
        return "\n".join(hits[:50]) or "No matches."
    return "Use list|add|search."


@tool
def remember_fact(fact: str) -> str:
    """Save a durable fact to long-term tagged memory."""
    fact = fact.strip()
    if not fact:
        return "Empty fact."
    mem = _host.load_memory()
    facts: list = mem.setdefault("facts", [])
    if fact not in facts:
        facts.append(fact)
        facts[:] = facts[-100:]
        _host.save_memory(mem)
    try:
        return _host.get_memory_system().add_memory(
            fact, tags=_host.detect_memory_tags(fact), importance=0.75, source="agent:fact"
        )
    except Exception:
        return f"Remembered: {fact}"


@tool
def remember_preference(key: str, value: str) -> str:
    """Save a user preference (e.g. downloads_folder, default_mode)."""
    mem = _host.load_memory()
    prefs: dict = mem.setdefault("preferences", {})
    prefs[key.strip()] = value.strip()
    _host.save_memory(mem)
    return f"Preference saved: {key} = {value}"


@tool
def log_completed_task(task: str, summary: str, outcome: str = "completed") -> str:
    """Record a completed task in persistent memory."""
    _host.log_past_task(task, summary, outcome)
    mem = _host.load_memory()
    mem.setdefault("past_tasks", []).append(
        {"task": task[:500], "summary": summary[:500], "outcome": outcome, "date": _host._now_iso()}
    )
    mem["past_tasks"] = mem["past_tasks"][-80:]
    _host.save_memory(mem)
    return f"Task logged: {task[:80]}"


@tool
def recall_memory(query: str = "") -> str:
    """Search long-term memory and knowledge base. Pass a topic or leave empty for recent memories."""
    ms = _host.get_memory_system()
    if not query.strip():
        return ms.display_recent_memories()
    ltm = ms.recall_ltm(query, k=5)
    try:
        kb_docs = _host.get_kb().search(query, k=5, exclude_kinds={"ltm"})
    except Exception as exc:
        return f"Recall error: {exc}"
    return ms.format_recall_results(ltm, kb_docs)


@tool
def search_knowledge(query: str, limit: int = 5) -> str:
    """Semantic search across all KB collections. Use before answering domain/technical questions."""
    try:
        docs = _host.get_kb().search(query, k=min(max(limit, 1), 12))
    except Exception as exc:
        return f"Knowledge base error: {exc}"
    if not docs:
        return "No matching entries. Prefer stating KB has no coverage over guessing."
    body = _host.get_kb().format_results(docs)
    return body + "\n\n" + _host.get_kb().format_sources_block(docs)


@tool
def web_search(query: str) -> str:
    """Search the internet for real-time, current, or unfamiliar information — headless, no browser window.

    Reach for this whenever a good answer needs fresh or external facts you don't already
    know: current events and news, sports scores, prices, release dates, "what is X",
    "latest on Y", definitions, research, and general look-ups. Returns a concise,
    source-cited summary and offers to save useful findings to the knowledge base.

    Prefer this when the operator just wants the answer. Both this and `browse_web` (default visible=False)
    fetch quietly in the background; only set `browse_web(visible=True)` when he wants a window opened.
    """
    # Prefer the multi-engine fusion path (richer, source + confidence tagged); fall back to the
    # original single-engine search on any error or empty result so behavior is fully backward
    # compatible (same signature, same string return, same KB-queue offer).
    query = (query or "").strip()
    if query:
        try:
            hits = _multi_engine_search(query, max_results=5)
            if hits:
                engines = sorted({e for h in hits for e in h.get("engines", [])})
                header = f"_Sources: {', '.join(engines)}_\n\n" if len(engines) > 1 else ""
                return header + _format_fused_hits(query, hits, queue_kb=True, show_meta=True)
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("Multi-engine web_search failed (%s) — single-engine fallback", exc)
    return _host._run_web_search(query, queue_kb=True)


@tool
def search_web_for_kb(query: str) -> str:
    """Search the internet and queue useful snippets for `/approve learn` into the knowledge base."""
    return _host._run_web_search(query, queue_kb=True)


# ---------------------------------------------------------------------------
# Multi-engine web search + result fusion + lightweight reranking
#
# Runs several independent search backends in parallel — DuckDuckGo (via the `ddgs` package),
# an optional self-hosted SearxNG JSON instance (CFG['searxng_url'] / $SEARXNG_URL), Brave
# (official API if CFG['brave_api_key']/$BRAVE_API_KEY is set, else HTML), and the host's
# DuckDuckGo/Bing/Brave HTML fallback — then FUSES them: dedupe by normalized URL, merge the
# richest snippet, and rerank by a cheap heuristic (cross-engine agreement + domain authority +
# snippet quality + recency hint + original rank). Fully headless; every layer degrades
# gracefully, so search keeps working with zero optional deps installed.
# ---------------------------------------------------------------------------

# Gentle ranking nudge only (never a filter): a small allow-list of high-signal domains.
_AUTHORITY_DOMAINS = {
    "wikipedia.org": 3.0, "github.com": 2.5, "stackoverflow.com": 2.2, "arxiv.org": 2.5,
    "nature.com": 2.4, "python.org": 2.2, "docs.python.org": 2.4, "readthedocs.io": 1.8,
    "developer.mozilla.org": 2.4, "reuters.com": 2.2, "apnews.com": 2.2, "bbc.co.uk": 2.0,
    "bbc.com": 2.0, "nytimes.com": 2.0, "arstechnica.com": 1.8, "theverge.com": 1.6,
}

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid",
    "ref", "ref_src", "spm", "_hsenc", "_hsmi", "mc_cid", "mc_eid", "igshid",
}

_RECENCY_RE = re.compile(
    r"\b(\d+\s+(?:minute|hour|day|week)s?\s+ago|today|yesterday|20(?:2[3-9]|[3-9]\d))\b", re.I
)


def _normalize_url(url: str) -> str:
    """Canonicalize a URL for de-duplication (lowercase host, drop tracking params + fragment + trailing /)."""
    try:
        p = urllib.parse.urlsplit(url.strip())
    except Exception:  # noqa: BLE001
        return url.strip().lower()
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=False)
         if k.lower() not in _TRACKING_PARAMS]
    return urllib.parse.urlunsplit((
        (p.scheme or "https").lower(), host, (p.path or "").rstrip("/"),
        urllib.parse.urlencode(sorted(q)), "",
    ))


def _domain_of(url: str) -> str:
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return ""
    return host[4:] if host.startswith("www.") else host


def _authority_boost(url: str) -> float:
    d = _domain_of(url)
    for dom, boost in _AUTHORITY_DOMAINS.items():
        if d == dom or d.endswith("." + dom):
            return boost
    return 0.0


def _recency_hint(text: str) -> float:
    """Cheap recency signal from snippet/title text (relative dates or a recent year mention)."""
    m = _RECENCY_RE.search(text or "")
    if not m:
        return 0.0
    tok = m.group(0).lower()
    if any(k in tok for k in ("minute", "hour", "today", "yesterday")):
        return 1.5
    if "day" in tok or "week" in tok:
        return 1.0
    return 0.5  # a recent year appears


def _hit_quality(hit: dict) -> float:
    title = (hit.get("title") or "").strip()
    body = (hit.get("body") or hit.get("snippet") or "").strip()
    score = 0.0
    if title:
        score += min(len(title), 80) / 80.0
    if body:
        score += min(len(body), 300) / 300.0 * 1.5
    return score


def _engine_ddgs(query: str, max_results: int) -> list[dict]:
    """DuckDuckGo via the `ddgs` (or legacy `duckduckgo_search`) package. [] if unavailable."""
    DDGS = None
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return []
    out: list[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append({
                    "title": r.get("title", ""),
                    "href": r.get("href") or r.get("link", ""),
                    "body": r.get("body") or r.get("snippet", ""),
                    "engine": "duckduckgo",
                })
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("ddgs engine failed: %s", exc)
    return out


def _engine_searxng(query: str, max_results: int) -> list[dict]:
    """Query a self-hosted SearxNG JSON API (CFG['searxng_url'] / $SEARXNG_URL). [] if not configured."""
    base = (_host.CFG.get("searxng_url") or os.environ.get("SEARXNG_URL") or "").strip().rstrip("/")
    if not base:
        return []
    try:
        qs = urllib.parse.urlencode({"q": query, "format": "json", "safesearch": 1})
        raw = _host._http_get(f"{base}/search?{qs}", timeout=10)
        data = json.loads(raw) if raw else {}
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("SearxNG engine failed: %s", exc)
        return []
    return [
        {"title": r.get("title", ""), "href": r.get("url", ""),
         "body": r.get("content", ""), "engine": "searxng"}
        for r in (data.get("results") or [])[:max_results]
    ]


def _engine_brave(query: str, max_results: int) -> list[dict]:
    """Brave Search — official API when a key is configured, else HTML scrape. [] on failure."""
    key = (_host.CFG.get("brave_api_key") or os.environ.get("BRAVE_API_KEY") or "").strip()
    if key:
        try:
            qs = urllib.parse.urlencode({"q": query, "count": max_results})
            req = urllib.request.Request(
                f"https://api.search.brave.com/res/v1/web/search?{qs}",
                headers={"Accept": "application/json", "X-Subscription-Token": key,
                         "User-Agent": "Primus/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8", "replace"))
            out = [
                {"title": r.get("title", ""), "href": r.get("url", ""),
                 "body": r.get("description", ""), "engine": "brave"}
                for r in (data.get("web", {}).get("results") or [])[:max_results]
            ]
            if out:
                return out
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("Brave API failed (%s) — falling back to HTML", exc)
    try:
        html = _host._http_get(
            f"https://search.brave.com/search?q={urllib.parse.quote(query)}", timeout=10
        )
        hits = _host._parse_search_html("brave", html, max_results) if html else []
        for h in hits:
            h["engine"] = "brave"
        return hits
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Brave HTML failed: %s", exc)
        return []


def _engine_html_fallback(query: str, max_results: int) -> list[dict]:
    """Host's DuckDuckGo/Bing/Brave HTML fallback → normalized hits with an engine tag."""
    try:
        hits = _host._html_search_fallback(query, max_results) or []
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("HTML fallback failed: %s", exc)
        return []
    for h in hits:
        h.setdefault("engine", "html")
    return hits


def _fuse_results(hitlists: list[list[dict]], max_results: int) -> list[dict]:
    """Merge results from several engines: dedupe by normalized URL, merge snippets, rerank.

    Confidence rises with cross-engine agreement and result quality, normalized to 0-1.
    """
    fused: dict[str, dict] = {}
    for hits in hitlists:
        for rank, h in enumerate(hits or []):
            url = (h.get("href") or h.get("link") or "").strip()
            if not url.lower().startswith(("http://", "https://")):
                continue
            key = _normalize_url(url)
            entry = fused.get(key)
            if entry is None:
                entry = fused[key] = {
                    "title": (h.get("title") or "").strip(), "href": url,
                    "body": (h.get("body") or h.get("snippet") or "").strip(),
                    "engines": set(), "best_rank": rank,
                }
            entry["engines"].add(h.get("engine", "unknown"))
            body = (h.get("body") or h.get("snippet") or "").strip()
            if len(body) > len(entry["body"]):
                entry["body"] = body
            if not entry["title"] and h.get("title"):
                entry["title"] = h["title"].strip()
            entry["best_rank"] = min(entry["best_rank"], rank)

    scored: list[dict] = []
    for entry in fused.values():
        score = (
            len(entry["engines"]) * 2.0
            + _authority_boost(entry["href"])
            + _hit_quality(entry)
            + _recency_hint(f"{entry['title']} {entry['body']}")
            + max(0.0, 1.0 - entry["best_rank"] * 0.1)
        )
        entry["score"] = round(score, 3)
        scored.append(entry)
    scored.sort(key=lambda e: e["score"], reverse=True)

    top = scored[:max_results]
    max_score = max((e["score"] for e in top), default=1.0) or 1.0
    for e in top:
        e["confidence"] = round(min(1.0, 0.35 + 0.65 * (e["score"] / max_score)), 2)
        e["engines"] = sorted(e["engines"])
    return top


def _multi_engine_search(
    query: str, max_results: int = 10, *, per_engine: int = 8, timeout: float = 18.0
) -> list[dict]:
    """Run all available search engines in parallel and return fused, reranked results.

    Partial results are used if some engines are slow (bounded by `timeout`); a serial fallback
    runs if a thread pool is unavailable. Never raises.
    """
    query = (query or "").strip()
    if not query:
        return []
    engines = (_engine_ddgs, _engine_searxng, _engine_brave, _engine_html_fallback)
    results: list[list[dict]] = []
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from concurrent.futures import TimeoutError as _FTimeout
        ex = ThreadPoolExecutor(max_workers=len(engines))
        futs = {ex.submit(fn, query, per_engine): fn for fn in engines}
        try:
            for fut in as_completed(futs, timeout=timeout):
                try:
                    results.append(fut.result() or [])
                except Exception as exc:  # noqa: BLE001
                    _host.log.debug("Search engine %s failed: %s", futs[fut].__name__, exc)
        except _FTimeout:
            _host.log.debug("Some search engines timed out — proceeding with partial results")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Parallel search unavailable (%s) — running serially", exc)
        for fn in engines:
            try:
                results.append(fn(query, per_engine) or [])
            except Exception:  # noqa: BLE001
                pass
    return _fuse_results(results, max_results)


def _format_fused_hits(query: str, hits: list[dict], *, queue_kb: bool, show_meta: bool) -> str:
    """Render fused hits as a clean, source-cited summary; optionally queue snippets for `/approve learn`."""
    lines = [f"Here's what I found for **{query}**:", ""]
    for i, h in enumerate(hits, 1):
        title = (h.get("title") or "Result").strip()
        body = re.sub(r"\s+", " ", (h.get("body") or "")).strip()[:240]
        if body and body[-1] not in ".!?…":
            body += "…"
        url = h.get("href", "")
        meta = ""
        if show_meta:
            eng = ", ".join(h.get("engines") or []) or "web"
            conf = h.get("confidence")
            meta = f"   _[{eng} · confidence {int(conf * 100)}%]_" if conf is not None else f"   _[{eng}]_"
        lines.append(f"{i}. **{title}** — {body}\n   ↳ {url}{meta}" if url else f"{i}. **{title}** — {body}")
        if queue_kb and body:
            _host.PrimusSession.queue_kb_learn(
                f"{title}\n{h.get('body', '')}", title=title, source_url=url,
                collection="learned", importance=0.6,
            )
    if queue_kb and _host.PrimusSession.pending_kb_learn:
        n = len(_host.PrimusSession.pending_kb_learn)
        lines.append(
            f"\nWant me to save {'these' if n > 1 else 'this'} to your knowledge base? "
            "Say `/approve learn` (or `/reject learn`)."
        )
    return "\n".join(lines)


@tool
def deep_web_search(query: str, max_results: int = 10) -> str:
    """Multi-engine web search with result fusion + reranking — the most thorough quick search.

    Queries several independent engines in parallel (DuckDuckGo, optional SearxNG, Brave, plus
    HTML fallbacks), de-duplicates by URL, merges the best snippets, and reranks by cross-engine
    agreement, domain authority, snippet quality and recency. Returns richer, source-cited results
    that show which engines found each hit and a confidence score. Use for research-grade look-ups,
    or when a single-engine `web_search` seems thin. Headless; offers to save findings to the KB.
    """
    query = (query or "").strip()
    if not query:
        return "Tell me what you'd like me to research."
    _host.PrimusSession.emit_think("Deep search", f"Fusing multi-engine results: {query}", "running")
    try:
        hits = _multi_engine_search(query, max_results=max(1, min(int(max_results or 10), 20)))
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("deep_web_search failed (%s) — single-engine fallback", exc)
        return _host._run_web_search(query, queue_kb=True)
    if not hits:
        _host.PrimusSession.emit_think("Deep search", "No fused results — single-engine fallback", "done")
        return _host._run_web_search(query, queue_kb=True)
    _host.PrimusSession.emit_think("Deep search", f"{len(hits)} fused result(s)", "done")
    engines = sorted({e for h in hits for e in h.get("engines", [])})
    header = f"_Fused from: {', '.join(engines)}_\n\n" if engines else ""
    return header + _format_fused_hits(query, hits, queue_kb=True, show_meta=True)


def _selenium_fetch_html(url: str, *, timeout: int = 15) -> str:
    """Headless Firefox render → page HTML, for JS-heavy pages. '' if unavailable. Never raises."""
    try:
        from selenium import webdriver  # type: ignore
        from selenium.webdriver.firefox.options import Options  # type: ignore
    except Exception:  # noqa: BLE001
        return ""
    driver = None
    try:
        opts = Options()
        opts.add_argument("-headless")
        driver = webdriver.Firefox(options=opts)
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        return driver.page_source or ""
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Selenium headless fetch failed: %s", exc)
        return ""
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:  # noqa: BLE001
                pass


def _strip_html_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|nav|header|footer|aside).*?</\1>", " ", raw)
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", raw)).strip()


def _extract_trafilatura(url: str) -> str:
    try:
        import trafilatura  # type: ignore

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(
                downloaded, include_comments=False, include_tables=False, favor_recall=True
            ) or ""
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Trafilatura failed: %s", exc)
    return ""


def _extract_newspaper(url: str) -> str:
    try:
        from newspaper import Article  # type: ignore

        art = Article(url)
        art.download()
        art.parse()
        return (art.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Newspaper3k failed: %s", exc)
    return ""


def _extract_bs4(html: str) -> str:
    """Structured extraction with BeautifulSoup: prefer <article>/<main>, else paragraphs."""
    soup = _host._soup(html)
    if soup is None:
        return ""
    try:
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        node = soup.find("article") or soup.find("main") or soup.body or soup
        paras = [p.get_text(" ", strip=True) for p in node.find_all("p")]
        text = "\n\n".join(p for p in paras if len(p) > 40)
        return text.strip() or node.get_text(" ", strip=True).strip()
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("BeautifulSoup extraction failed: %s", exc)
    return ""


# ---------------------------------------------------------------------------
# Stronger structured extraction: Playwright (JS render) + tables→markdown, code blocks with
# language detection, headings hierarchy, JSON-LD metadata, readability scoring, and optional
# archive.is / 12ft.io bypass. All layers are optional and degrade gracefully; never raise.
# ---------------------------------------------------------------------------

_CODE_LANG_RE = re.compile(r"(?:language|lang|highlight|brush:)[-\s:]?([a-z0-9+#]+)", re.I)

_BYPASS_SERVICES = {
    "archive.ph": "https://archive.ph/newest/{url}",
    "12ft.io": "https://12ft.io/proxy?q={url}",
}


def _playwright_fetch_html(url: str, *, timeout_ms: int = 20000) -> str:
    """Render a JS-heavy page with headless Playwright (Chromium) → HTML. '' if unavailable. Never raises.

    A stronger, faster fallback than Selenium for modern SPA/JS sites. Optional dependency:
    ``uv pip install playwright && playwright install chromium``.
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:  # noqa: BLE001
        return ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent="Mozilla/5.0 (X11; Linux x86_64) Primus/1.0")
                page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                try:
                    page.wait_for_load_state("networkidle", timeout=4000)
                except Exception:  # noqa: BLE001
                    pass
                return page.content() or ""
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Playwright fetch failed: %s", exc)
        return ""


def _detect_code_language(node: Any) -> str:
    """Best-effort code language from a <pre>/<code> element's class attributes."""
    try:
        classes = " ".join(node.get("class") or [])
        parent = getattr(node, "parent", None)
        if parent is not None:
            classes += " " + " ".join(parent.get("class") or [])
        m = _CODE_LANG_RE.search(classes)
        if m:
            lang = m.group(1).lower()
            return "" if lang in ("hljs", "highlight", "prettyprint", "sourcecode") else lang
    except Exception:  # noqa: BLE001
        pass
    return ""


def _table_to_markdown(table: Any) -> str:
    """Convert a BeautifulSoup <table> into a GitHub-flavored markdown table. '' on failure."""
    try:
        md_rows: list[str] = []
        header_done = False
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            vals = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)).replace("|", "\\|") for c in cells]
            md_rows.append("| " + " | ".join(vals) + " |")
            if not header_done:
                md_rows.append("| " + " | ".join("---" for _ in vals) + " |")
                header_done = True
        return "\n".join(md_rows) if len(md_rows) > 1 else ""
    except Exception:  # noqa: BLE001
        return ""


def _extract_jsonld(soup: Any) -> list[dict]:
    """Collect JSON-LD structured-data blocks from a page (max 5). [] on failure."""
    out: list[dict] = []
    try:
        for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                data = json.loads(tag.string or tag.get_text() or "")
            except Exception:  # noqa: BLE001
                continue
            if isinstance(data, list):
                out.extend(d for d in data if isinstance(d, dict))
            elif isinstance(data, dict):
                out.append(data)
    except Exception:  # noqa: BLE001
        pass
    return out[:5]


def _extract_structured(html: str) -> dict:
    """Rich structured extraction: main text + tables(md) + code + headings + JSON-LD + readability.

    Returns a dict with keys: text, tables, code, headings, jsonld, readability. Falls back to a
    plain text strip when BeautifulSoup isn't available. Never raises.
    """
    result: dict[str, Any] = {
        "text": "", "tables": [], "code": [], "headings": [], "jsonld": [], "readability": 0.0,
    }
    soup = _host._soup(html)
    if soup is None:
        result["text"] = _strip_html_text(html)
        return result
    try:
        full_len = len(soup.get_text(" ", strip=True))
        result["jsonld"] = _extract_jsonld(soup)
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            txt = re.sub(r"\s+", " ", h.get_text(" ", strip=True))
            if txt:
                result["headings"].append({"level": int(h.name[1]), "text": txt[:160]})
        for pre in soup.find_all("pre"):
            code_el = pre.find("code") or pre
            code_txt = code_el.get_text("\n", strip=False).strip("\n")
            if code_txt and len(code_txt) > 8:
                result["code"].append({"lang": _detect_code_language(code_el), "code": code_txt[:4000]})
        for table in soup.find_all("table"):
            md = _table_to_markdown(table)
            if md:
                result["tables"].append(md[:4000])
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()
        node = soup.find("article") or soup.find("main") or soup.body or soup
        paras = [p.get_text(" ", strip=True) for p in node.find_all(["p", "li"])]
        main_text = "\n\n".join(p for p in paras if len(p) > 40)
        if len(main_text) < 200:
            main_text = node.get_text(" ", strip=True)
        result["text"] = main_text.strip()
        result["readability"] = round(min(1.0, len(result["text"]) / max(1, full_len)), 2)
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Structured extraction failed: %s", exc)
        if not result["text"]:
            result["text"] = _strip_html_text(html)
    return result


def _render_structured_extras(struct: dict) -> str:
    """Append outline/code/tables/metadata sections to an article read (structured mode)."""
    extras: list[str] = []
    headings = struct.get("headings") or []
    if headings:
        extras.append("\n\n### Outline\n" + "\n".join(
            f"{'  ' * (int(h['level']) - 1)}- {h['text']}" for h in headings[:20]))
    for i, code in enumerate((struct.get("code") or [])[:5], 1):
        extras.append(f"\n\n### Code block {i}\n```{code.get('lang', '')}\n{code['code']}\n```")
    for i, tbl in enumerate((struct.get("tables") or [])[:5], 1):
        extras.append(f"\n\n### Table {i}\n{tbl}")
    jsonld = struct.get("jsonld") or []
    if jsonld:
        try:
            extras.append("\n\n### Metadata (JSON-LD)\n```json\n"
                          + json.dumps(jsonld[0], ensure_ascii=False)[:800] + "\n```")
        except Exception:  # noqa: BLE001
            pass
    rd = struct.get("readability")
    if rd:
        extras.append(f"\n\n_Readability (main-content density): {int(rd * 100)}%_")
    return "".join(extras)


def _read_article(url: str, *, structured: bool = False, allow_bypass: bool = False) -> str:
    """Core headless article extraction with a multi-layer pipeline:

    Trafilatura → Newspaper3k → (HTTP fetch) structured BeautifulSoup → plain strip → Playwright
    (JS render) → headless Selenium → optional archive.is/12ft.io bypass (only when allow_bypass).
    Each layer is optional and degrades gracefully; the first one that yields real text wins. When
    ``structured`` is set, tables (as markdown), code blocks (with language), a heading outline and
    JSON-LD metadata are appended.
    """
    url = (url or "").strip()
    if not re.match(r"^https?://", url, re.I):
        if re.match(r"^[\w-]+(\.[\w-]+)+(/.*)?$", url):
            url = "https://" + url
        else:
            return "Give me a valid URL to read (e.g. https://example.com/article)."
    _host.PrimusSession.emit_think("Read article", f"Fetching {url[:80]} (headless)", "running")

    text = _extract_trafilatura(url)
    if len(text) < 200:
        text = _extract_newspaper(url) or text

    struct: dict = {}
    if len(text) < 200 or structured:  # fetch raw HTML once → structured + plain extraction
        html = _host._http_get(url, timeout=12)
        if html:
            struct = _extract_structured(html)
            if len(struct.get("text", "")) > len(text):
                text = struct["text"]
            if len(text) < 200:
                text = _strip_html_text(html) or text

    if len(text) < 160:  # stronger JS render: Playwright (preferred) → Selenium
        html = _playwright_fetch_html(url) or _selenium_fetch_html(url)
        if html:
            struct = _extract_structured(html) or struct
            text = struct.get("text") or _extract_bs4(html) or _strip_html_text(html) or text

    if len(text) < 120 and allow_bypass:  # explicit opt-in reader/paywall bypass
        for name, tpl in _BYPASS_SERVICES.items():
            html = _host._http_get(tpl.format(url=url), timeout=15)
            if html:
                cand = _extract_structured(html)
                if len(cand.get("text", "")) > 200:
                    struct, text = cand, cand["text"]
                    _host.PrimusSession.emit_think("Read article", f"Recovered via {name}", "running")
                    break

    if not text:
        _host.PrimusSession.emit_think("Read article", "No readable content", "done")
        return (
            "I couldn't fetch that page — it may be offline, blocked, or require a login. "
            "Say 'open it' / 'show me' if you'd like it in a visible browser instead."
        )

    _host.PrimusSession.emit_think("Read article", f"Extracted {len(text)} chars", "done")
    clipped = text[:6000]
    suffix = "\n\n…(truncated — ask me to continue if you need more.)" if len(text) > 6000 else ""
    body = f"**Read from {url} (in the background):**\n\n{clipped}{suffix}"
    if structured and struct:
        body += _render_structured_extras(struct)
    return body


@tool
def read_article(url: str, structured: bool = False, allow_bypass: bool = False) -> str:
    """Fetch a web page and extract its clean main text — headless, silent, no browser window.

    Use to "read", "summarize", or "check" an article/page when you have a URL. Runs entirely in
    the background through a multi-layer pipeline (Trafilatura → Newspaper3k → BeautifulSoup →
    HTTP → Playwright → headless Selenium). Returns clean text to summarize.

    Set ``structured=True`` to also extract tables (as markdown), code blocks (with detected
    language), a heading outline, and JSON-LD metadata. Set ``allow_bypass=True`` ONLY when the operator
    explicitly asks to get past a paywall/blocked page (tries archive.is / 12ft.io mirrors).
    """
    return _read_article(url, structured=structured, allow_bypass=allow_bypass)


@tool
def research_topic(topic: str, max_sources: int = 4, ingest_offer: bool = True) -> str:
    """Smart research: multi-engine search → read the top sources → summarize with citations.

    Runs the `deep_web_search` fusion, extracts the top `max_sources` articles, then synthesizes a
    concise, cited markdown briefing (Overview / Key findings / Details / Open questions + Sources).
    When `ingest_offer` is set, the best sources are staged so the operator can save them to the knowledge
    base with `/approve learn`. This backs the `/research` command and background research runs.
    Headless; never invents sources. `max_sources` is clamped to 1-6.
    """
    topic = (topic or "").strip()
    if not topic:
        return "Give me a topic to research."
    _host.PrimusSession.emit_think("Research", f"Multi-engine search: {topic}", "running")
    try:
        hits = _multi_engine_search(topic, max_results=max(2, min(int(max_sources or 4) * 2, 16)))
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("research_topic search failed: %s", exc)
        hits = []
    if not hits:
        return f"I couldn't find sources for '{topic}' right now (offline or every engine is busy)."

    n = max(1, min(int(max_sources or 4), 6))
    top = hits[:n]
    extracts: list[str] = []
    for h in top:
        url = h.get("href", "")
        _host.PrimusSession.emit_think("Research", f"Reading {url[:70]}", "running")
        try:
            art = _read_article(url)
        except Exception as exc:  # noqa: BLE001
            art = f"(couldn't read: {exc})"
        extracts.append(f"### Source: {h.get('title') or url}\nURL: {url}\n{art[:2500]}")
        if ingest_offer and len(art) > 300:
            _host.PrimusSession.queue_kb_learn(
                art[:4000], title=h.get("title") or url, source_url=url,
                collection="learned", importance=0.7,
            )

    corpus = "\n\n".join(extracts)[:11000]
    sources_md = "\n".join(
        f"- [{h.get('title') or h.get('href')}]({h.get('href')})  _(conf {int(h.get('confidence', 0) * 100)}%)_"
        for h in top
    )
    try:
        llm = _host.make_chat_ollama(
            _host.CFG.get("model", _host.DEFAULT_MODEL), temperature=0.2, num_predict=1100
        )
        prompt = (
            f"Synthesize these researched sources into a clear markdown briefing on '{topic}'. "
            "Use exactly these sections: '## Overview', '## Key findings' (bullets), '## Details', "
            "'## Open questions'. Be factual and concise; reference source titles/URLs where "
            "relevant; never invent sources or facts not in the notes.\n\n" + corpus
        )
        report = _host.coerce_message_text(llm.invoke([HumanMessage(content=prompt)]).content)
    except Exception as exc:  # noqa: BLE001
        report = f"(Synthesis model unavailable: {exc})\n\n{corpus}"

    out = f"# 🔬 Research: {topic}\n\n{report}\n\n## Sources\n{sources_md}"
    if ingest_offer and _host.PrimusSession.pending_kb_learn:
        out += (
            "\n\n---\n*Save the best sources to your knowledge base?* "
            "Say `/approve learn` to ingest them, or `/reject learn` to skip."
        )
    _host.PrimusSession.emit_think("Research", "Briefing ready", "done")
    return out


# ---------------------------------------------------------------------------
# News / RSS, community (Reddit), and social search — all headless & background
# ---------------------------------------------------------------------------

_DEFAULT_NEWS_FEEDS = {
    "world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "tech": "https://feeds.arstechnica.com/arstechnica/index",
    "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
}


def _parse_feed(url: str, limit: int) -> list[dict]:
    """Parse an RSS/Atom feed via feedparser (preferred) or a regex fallback. Never raises."""
    entries: list[dict] = []
    try:
        import feedparser  # type: ignore

        parsed = feedparser.parse(url)
        for e in parsed.entries[:limit]:
            entries.append({
                "title": getattr(e, "title", "").strip(),
                "link": getattr(e, "link", "").strip(),
                "published": getattr(e, "published", "") or getattr(e, "updated", ""),
                "summary": re.sub(r"<[^>]+>", "", getattr(e, "summary", ""))[:240].strip(),
            })
        if entries:
            return entries
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("feedparser failed (%s) — regex fallback", exc)

    xml = _host._http_get(url, timeout=10)
    if xml:
        for m in re.findall(r"<item>(.*?)</item>", xml, re.S | re.I)[:limit]:
            def _tag(t: str) -> str:
                mm = re.search(rf"<{t}[^>]*>(.*?)</{t}>", m, re.S | re.I)
                return re.sub(r"<!\[CDATA\[|\]\]>|<[^>]+>", "", mm.group(1)).strip() if mm else ""
            entries.append({
                "title": _tag("title"),
                "link": _tag("link"),
                "published": _tag("pubDate"),
                "summary": _tag("description")[:240],
            })
    return entries


@tool
def get_news(topic: str = "", limit: int = 6) -> str:
    """Real-time news headlines via RSS — headless, fast, source-linked.

    Use for "latest news", "news about X", "what's happening with Y", "headlines".
    With a topic, pulls Google News results for it; without one, aggregates top world/tech/
    business/science headlines. Degrades gracefully offline.

    Args:
      topic: optional subject to focus on (e.g. "AI regulation", "Tesla").
      limit: max headlines (default 6).
    """
    import urllib.parse

    limit = max(1, min(int(limit or 6), 15))
    topic = (topic or "").strip()
    _host.PrimusSession.emit_think("News", f"Fetching headlines{' on ' + topic if topic else ''}", "running")

    entries: list[dict] = []
    if topic:
        q = urllib.parse.quote(topic)
        entries = _parse_feed(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en", limit)
    else:
        per = max(2, limit // len(_DEFAULT_NEWS_FEEDS) + 1)
        for cat, feed in _DEFAULT_NEWS_FEEDS.items():
            for e in _parse_feed(feed, per):
                e["category"] = cat
                entries.append(e)
        entries = entries[:limit]

    if not entries:
        _host.PrimusSession.emit_think("News", "No items / offline", "error")
        return (
            "I couldn't pull news right now — I may be offline, or the feed is down. "
            "Install `feedparser` (`uv pip install feedparser`) for the most reliable parsing."
        )

    head = f"Latest news on **{topic}**:" if topic else "Top headlines right now:"
    lines = [head, ""]
    for i, e in enumerate(entries, 1):
        when = f" _{e['published'][:22]}_" if e.get("published") else ""
        cat = f"[{e['category']}] " if e.get("category") else ""
        lines.append(f"{i}. {cat}**{e['title']}**{when}")
        if e.get("summary"):
            lines.append(f"   {e['summary']}")
        if e.get("link"):
            lines.append(f"   ↳ {e['link']}")
    _host.PrimusSession.emit_think("News", f"{len(entries)} headline(s)", "done")
    return "\n".join(lines)


@tool
def search_reddit(query: str, subreddit: str = "", limit: int = 6) -> str:
    """Search Reddit for community discussion — headless, via Reddit's public JSON.

    Use for "what does Reddit think about X", "reddit discussions on Y", community opinions,
    real-world experiences. Returns top posts with subreddit, score, comments, and links.

    Args:
      query: search terms.
      subreddit: optional subreddit to restrict to (without the r/).
      limit: max posts (default 6).
    """
    import urllib.parse

    query = (query or "").strip()
    if not query:
        return "Tell me what to search Reddit for."
    limit = max(1, min(int(limit or 6), 15))
    sub = subreddit.strip().lstrip("r/").strip("/")
    q = urllib.parse.quote(query)
    if sub:
        url = f"https://www.reddit.com/r/{sub}/search.json?q={q}&restrict_sr=1&sort=relevance&limit={limit}"
    else:
        url = f"https://www.reddit.com/search.json?q={q}&sort=relevance&limit={limit}"
    _host.PrimusSession.emit_think("Reddit", f"Searching r/{sub or 'all'} for: {query}", "running")

    raw = _host._http_get(url, timeout=10)
    if not raw:
        return "I couldn't reach Reddit just now (offline or rate-limited). Try again shortly."
    try:
        data = json.loads(raw)
        children = data.get("data", {}).get("children", [])
    except Exception as exc:  # noqa: BLE001
        return f"Reddit returned something I couldn't parse ({str(exc)[:80]})."
    if not children:
        return f'No Reddit posts found for "{query}".'

    lines = [f"Reddit discussions for **{query}**:", ""]
    for i, c in enumerate(children[:limit], 1):
        d = c.get("data", {})
        title = (d.get("title") or "").strip()
        sr = d.get("subreddit_name_prefixed") or f"r/{d.get('subreddit', '?')}"
        score = d.get("score", 0)
        ncom = d.get("num_comments", 0)
        link = f"https://reddit.com{d.get('permalink', '')}"
        body = re.sub(r"\s+", " ", (d.get("selftext") or ""))[:160].strip()
        lines.append(f"{i}. **{title}** ({sr} · ▲{score} · 💬{ncom})")
        if body:
            lines.append(f"   {body}{'…' if len(d.get('selftext','')) > 160 else ''}")
        lines.append(f"   ↳ {link}")
    _host.PrimusSession.emit_think("Reddit", f"{len(children)} post(s)", "done")
    return "\n".join(lines)


@tool
def social_search(query: str, platform: str = "reddit") -> str:
    """Search social/community platforms for real-time chatter — headless.

    platform:
      "reddit"             → Reddit search (community opinions & experiences).
      "twitter" | "x"      → recent X/Twitter-style posts (via web search of x.com/twitter.com + nitter).
      "hackernews" | "hn"  → Hacker News discussions (tech, via Algolia API).
    Falls back to a general web search if a platform is unavailable.
    """
    query = (query or "").strip()
    if not query:
        return "Tell me what to look for."
    p = (platform or "reddit").strip().lower()

    if p in ("reddit", "r"):
        return search_reddit.func(query) if hasattr(search_reddit, "func") else _host._run_web_search(f"{query} site:reddit.com")
    if p in ("hackernews", "hn", "ycombinator"):
        import urllib.parse

        raw = _host._http_get(
            f"https://hn.algolia.com/api/v1/search?query={urllib.parse.quote(query)}&tags=story&hitsPerPage=6",
            timeout=10,
        )
        if raw:
            try:
                hits = json.loads(raw).get("hits", [])
                if hits:
                    lines = [f"Hacker News on **{query}**:", ""]
                    for i, h in enumerate(hits[:6], 1):
                        title = h.get("title") or h.get("story_title") or "(untitled)"
                        pts = h.get("points", 0)
                        ncom = h.get("num_comments", 0)
                        oid = h.get("objectID", "")
                        lines.append(f"{i}. **{title}** (▲{pts} · 💬{ncom})")
                        if h.get("url"):
                            lines.append(f"   ↳ {h['url']}")
                        lines.append(f"   ↳ https://news.ycombinator.com/item?id={oid}")
                    return "\n".join(lines)
            except Exception:  # noqa: BLE001
                pass
        return _host._run_web_search(f"{query} site:news.ycombinator.com")
    if p in ("twitter", "x"):
        # No stable public API; surface recent posts via search across X/Twitter + nitter mirrors.
        return _host._run_web_search(f"{query} (site:x.com OR site:twitter.com OR site:nitter.net)")
    return _host._run_web_search(query)


def _get_webdriver(visible: bool = False, *, timeout: int = 20) -> Any:
    """Create a Firefox WebDriver (headless unless visible). Returns driver or raises."""
    from selenium import webdriver  # type: ignore
    from selenium.webdriver.firefox.options import Options  # type: ignore

    opts = Options()
    if not visible:
        opts.add_argument("-headless")
    driver = webdriver.Firefox(options=opts)
    driver.set_page_load_timeout(timeout)
    return driver


@tool
def browser_automate(url: str, actions: str = "", visible: bool = False) -> str:
    """Drive a real browser to handle dynamic pages, logins, forms, and clicks — Selenium/Firefox.

    HEADLESS by default (silent, no window). Set visible=True only when the operator wants to watch.
    Use this when `read_article`/`web_search` aren't enough: JS-rendered content, login walls,
    multi-step forms, clicking through pagination, etc.

    Args:
      url: the page to open first.
      actions: optional newline- or ';'-separated steps. Supported verbs:
        - `type <css_selector> = <value>`   type into a field
        - `fill <name_attr> = <value>`      type into <input name="…">
        - `click <css_selector>`            click an element
        - `submit <css_selector>`           submit a form/field
        - `wait <seconds>`                  pause for JS/navigation
        - `text <css_selector>`             extract text from matching elements
        - `screenshot`                      save a PNG to ~/.primus and report the path
      visible: True opens a visible window; default False (headless background).

    Example:
      browser_automate("https://example.com/login",
        "fill username=me\\nfill password=secret\\nclick button[type=submit]\\nwait 2\\ntext .dashboard")
    """
    url = (url or "").strip()
    if not re.match(r"^https?://", url, re.I):
        if re.match(r"^[\w-]+(\.[\w-]+)+(/.*)?$", url):
            url = "https://" + url
        else:
            return "Give me a valid URL to start from."
    try:
        from selenium.webdriver.common.by import By  # type: ignore
    except Exception:  # noqa: BLE001
        return (
            "Browser automation needs Selenium + Firefox. Install with `uv pip install selenium` "
            "and ensure Firefox + geckodriver are available. Meanwhile I can `read_article` or `web_search`."
        )

    _host.PrimusSession.emit_think("Browser", f"Automating {url[:60]} ({'visible' if visible else 'headless'})", "running")
    driver = None
    report: list[str] = []
    try:
        driver = _get_webdriver(visible=visible)
        driver.get(url)
        steps = [s.strip() for s in re.split(r"[;\n]", actions or "") if s.strip()]
        for step in steps:
            if _host.PrimusSession.is_cancelled():
                report.append("(halted)")
                break
            low = step.lower()
            try:
                if low.startswith("wait"):
                    secs = float(re.search(r"[\d.]+", step).group()) if re.search(r"[\d.]+", step) else 1.0
                    time.sleep(min(secs, 10))
                elif low.startswith("type ") and "=" in step:
                    sel, val = step[5:].split("=", 1)
                    driver.find_element(By.CSS_SELECTOR, sel.strip()).send_keys(val.strip())
                elif low.startswith("fill ") and "=" in step:
                    name, val = step[5:].split("=", 1)
                    driver.find_element(By.NAME, name.strip()).send_keys(val.strip())
                elif low.startswith("click "):
                    driver.find_element(By.CSS_SELECTOR, step[6:].strip()).click()
                elif low.startswith("submit "):
                    driver.find_element(By.CSS_SELECTOR, step[7:].strip()).submit()
                elif low.startswith("text "):
                    els = driver.find_elements(By.CSS_SELECTOR, step[5:].strip())
                    txt = "\n".join(e.text for e in els if e.text.strip())[:1500]
                    report.append(f"**{step[5:].strip()}**:\n{txt or '(no text found)'}")
                elif low == "screenshot":
                    _host.ensure_app_dirs()
                    shot = _host.APP_DIR / f"browser_{datetime.now():%Y%m%d-%H%M%S}.png"
                    driver.save_screenshot(str(shot))
                    report.append(f"Screenshot saved: {shot}")
            except Exception as exc:  # noqa: BLE001
                report.append(f"(step failed: {step[:50]} — {str(exc)[:80]})")

        title = (driver.title or "").strip()
        final_url = driver.current_url
        if not any(r.startswith("**") for r in report):
            # No explicit text extraction → return a clean snapshot of the page body.
            try:
                body = driver.find_element(By.TAG_NAME, "body").text
                report.append(_strip_html_text(body)[:1500] if body else "(page had no visible text)")
            except Exception:  # noqa: BLE001
                pass
        _host.PrimusSession.emit_think("Browser", "Automation complete", "done")
        header = f"**{title}** — {final_url}" if title else final_url
        return header + "\n\n" + "\n\n".join(report) if report else header
    except Exception as exc:  # noqa: BLE001
        _host.log.warning("Browser automation failed: %s", exc)
        _host.PrimusSession.emit_think("Browser", "Failed", "error")
        return (
            f"Browser automation couldn't complete ({str(exc)[:120]}). "
            "Firefox/geckodriver may be missing. I can try `read_article` or `web_search` instead."
        )
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:  # noqa: BLE001
                pass


@tool
def system_monitor(detail: str = "summary") -> str:
    """Live system resource snapshot — CPU, memory, swap, top processes, uptime (psutil-powered).

    Use for "system monitor", "resource usage", "what's using my CPU/RAM", "how's my system".
    detail="full" adds per-core CPU, disk usage, and temperatures. Falls back gracefully
    to a lightweight report if psutil isn't installed.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return system_info.func(detail) if hasattr(system_info, "func") else _host.build_status_bar()

    lines: list[str] = []
    try:
        cpu = psutil.cpu_percent(interval=0.2)
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        boot = datetime.fromtimestamp(psutil.boot_time())
        up = datetime.now() - boot
        hrs, rem = divmod(int(up.total_seconds()), 3600)
        mins = rem // 60
        lines.append(f"CPU: {cpu:.0f}%   RAM: {vm.percent:.0f}% ({vm.used // 1024**2} / {vm.total // 1024**2} MB)")
        lines.append(f"Swap: {sw.percent:.0f}% ({sw.used // 1024**2} / {sw.total // 1024**2} MB)   Uptime: {hrs}h {mins}m")
        procs = sorted(
            psutil.process_iter(["name", "cpu_percent", "memory_percent"]),
            key=lambda p: (p.info.get("memory_percent") or 0.0),
            reverse=True,
        )[:6]
        lines.append("\nTop memory consumers:")
        for p in procs:
            nm = (p.info.get("name") or "?")[:24]
            lines.append(f"  {nm:<24} mem {p.info.get('memory_percent') or 0:.1f}%  cpu {p.info.get('cpu_percent') or 0:.0f}%")
        if detail == "full":
            per_core = psutil.cpu_percent(interval=0.2, percpu=True)
            lines.append("\nPer-core CPU: " + " ".join(f"{c:.0f}%" for c in per_core))
            du = psutil.disk_usage(str(_host.HOME))
            lines.append(f"Disk ({_host.HOME}): {du.percent:.0f}% used ({du.used // 1024**3}G / {du.total // 1024**3}G)")
            try:
                temps = psutil.sensors_temperatures() or {}
                for name, entries in list(temps.items())[:3]:
                    cur = entries[0].current if entries else None
                    if cur is not None:
                        lines.append(f"Temp {name}: {cur:.0f}°C")
            except Exception:  # noqa: BLE001 — sensors not available everywhere
                pass
    except Exception as exc:  # noqa: BLE001
        return f"Couldn't read system stats ({str(exc)[:90]})."
    return "\n".join(lines)


# Cache the desktop-control availability check so we don't re-import on every call.
_PYAUTOGUI_STATE: dict[str, Any] = {}


def _get_pyautogui() -> tuple[Any, Optional[str]]:
    """Lazily import pyautogui. Returns (module, None) or (None, reason). Never raises."""
    if "mod" in _PYAUTOGUI_STATE:
        return _PYAUTOGUI_STATE["mod"], _PYAUTOGUI_STATE.get("reason")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        _PYAUTOGUI_STATE.update(mod=None, reason="no active graphical session (no DISPLAY)")
        return None, _PYAUTOGUI_STATE["reason"]
    try:
        os.environ.setdefault("PYAUTOGUI_FAILSAFE", "1")
        import pyautogui  # type: ignore

        _PYAUTOGUI_STATE.update(mod=pyautogui, reason=None)
        return pyautogui, None
    except Exception as exc:  # noqa: BLE001 — missing display libs / Wayland
        _PYAUTOGUI_STATE.update(mod=None, reason=str(exc)[:100])
        return None, _PYAUTOGUI_STATE["reason"]


@tool
def desktop_control(action: str, value: str = "") -> str:
    """Control the desktop GUI via PyAutoGUI: type text, press keys, hotkeys, or screenshot.

    actions:
      - "type"      → type `value` as keystrokes
      - "press"     → press a single key (e.g. "enter", "tab", "esc")
      - "hotkey"    → press a combo, '+'-separated (e.g. "ctrl+c", "alt+tab")
      - "screenshot"→ capture the screen to ~/Pictures and return the path
      - "position"  → report the current mouse position

    Requires an active graphical session (X11 works best). Returns a clean message if the
    desktop can't be controlled in the current environment — it never crashes Primus.
    """
    if _host.PrimusSession.is_cancelled():
        return "⏹ Halted — desktop action skipped."
    pg, reason = _get_pyautogui()
    if pg is None:
        return (
            "I can't control the desktop GUI right now — "
            f"{reason or 'PyAutoGUI is unavailable'}. This needs an active X11 session."
        )
    act = (action or "").strip().lower()
    try:
        if act == "type":
            if not value:
                return "Give me text to type."
            pg.typewrite(value, interval=0.01)
            return f"Typed {len(value)} characters."
        if act == "press":
            pg.press((value or "enter").strip())
            return f"Pressed {value or 'enter'}."
        if act == "hotkey":
            keys = [k.strip() for k in re.split(r"[+\s]+", value) if k.strip()]
            if not keys:
                return "Give me a key combo like 'ctrl+c'."
            pg.hotkey(*keys)
            return f"Pressed {'+'.join(keys)}."
        if act == "screenshot":
            shots = _host.HOME / "Pictures"
            shots.mkdir(parents=True, exist_ok=True)
            out = shots / f"primus-screenshot-{datetime.now():%Y%m%d-%H%M%S}.png"
            pg.screenshot(str(out))
            return f"Saved screenshot → {out}"
        if act == "position":
            x, y = pg.position()
            return f"Mouse is at ({x}, {y})."
    except Exception as exc:  # noqa: BLE001
        return f"Desktop action failed ({str(exc)[:90]})."
    return "Unknown action. Use: type | press | hotkey | screenshot | position."


@tool
def learn_knowledge(text: str, source: str = "agent") -> str:
    """Teach Primus a durable fact — stored in long-term tagged memory + vector index."""
    if not text.strip():
        return "Empty text — nothing stored."
    ms = _host.get_memory_system()
    result = ms.add_memory(
        text.strip(),
        tags=_host.detect_memory_tags(text),
        importance=0.8,
        source=source or "agent:tool",
    )
    try:
        _host.get_kb().learn_text(
            text.strip(),
            kind="tool",
            source=source or "agent:tool",
            importance=0.8,
            tags=_host.detect_memory_tags(text),
        )
    except Exception:
        pass
    return result


@tool
def kb_status() -> str:
    """Show knowledge base statistics and collection counts."""
    return _host.render_kb_dashboard()


@tool
def index_knowledge_folder(folder_path: str) -> str:
    """Index text files in a folder into the knowledge base (background)."""
    path, err = _host._guard_path(folder_path, must_exist=True)
    if err:
        return err
    if not path.is_dir():
        return "Path must be a directory."
    threading.Thread(
        target=lambda: _host.get_kb().ingest_folder(path, incremental=True),
        daemon=True,
        name="primus-index-tool",
    ).start()
    return f"Background indexing started for `{path}`."


# ---------------------------------------------------------------------------
# Self-improvement — controlled, human-gated access to Primus's own codebase
#
# SAFETY MODEL (read this before changing anything here):
#   • The introspection tools (read_own_code / analyze_self) are READ-ONLY.
#   • The proposal tools (propose_code_change / propose_new_tool) NEVER write to
#     disk — they only STAGE an edit on PrimusSession.pending_edit and show a preview.
#   • Edits are applied ONLY by apply_pending_self_edit(), which is reachable solely
#     through the user typing `/approve edit`. There is no autonomous code path that
#     applies an edit. The agent cannot call apply itself (it's not a tool).
#   • Before writing, we back up the current file, run the edit against an in-memory
#     copy, ast.parse() it to verify it still compiles, and only then write — reverting
#     automatically if anything fails. Critical guardrail markers cannot be deleted.
# ---------------------------------------------------------------------------

# If a proposed edit would remove any of these, it is refused — they are the
# safety rails for self-modification and must never be weakened by a self-edit.
_SELF_EDIT_CRITICAL_MARKERS = (
    "def apply_pending_self_edit",
    "/approve edit",
    "PrimusSession.pending_edit",
    "_SELF_EDIT_CRITICAL_MARKERS",
)

_SELF_EDIT_MAX_REPLACE = 24000  # cap per-op replacement size (defense against huge rewrites)


def _read_source() -> str:
    """Read Primus's own source file. Never raises."""
    try:
        return _host.SCRIPT_PATH.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _host.log.warning("Could not read own source: %s", exc)
        return ""


# Maps tool names → business-relevant capability buckets, so self-analysis can report
# performance by the work that matters (coding, research, client/comms, background, etc.).
_TOOL_CATEGORIES: dict[str, list[str]] = {
    "Coding & projects": [
        "create_full_project", "build_application_from_spec", "iterative_code_improver",
        "verify_python", "write_file", "read_file", "git", "create_directory", "move_path",
        "propose_code_change", "propose_new_tool", "design_and_propose_tool", "propose_changes_batch",
    ],
    "Research & web": [
        "deep_research", "search_web_for_kb", "browse_web", "web_search",
        "download_webpage_as_pdf", "download_pdf_from_url", "ingest_web_document",
    ],
    "Client & comms": ["compose_email", "draft_document", "summarize", "calendar", "schedule"],
    "Background & automation": [
        "long_running_command", "batch_file_operations", "terminal", "run_shell",
    ],
    "Memory & knowledge": [
        "learn_knowledge", "kb_status", "index_knowledge_folder", "recall_memory",
        "remember", "search_memory",
    ],
    "System & files": [
        "list_directory", "system_info", "manage_process", "open_application", "find_files",
    ],
    "Self-improvement": [
        "read_own_code", "analyze_self", "propose_code_change", "propose_new_tool",
        "design_and_propose_tool", "propose_changes_batch", "search_own_codebase",
        "explain_my_architecture", "remember_self_lesson", "recall_self_lessons",
    ],
}


def _tool_category(name: str) -> str:
    for cat, names in _TOOL_CATEGORIES.items():
        if name in names:
            return cat
    return "Other"


class SelfImprovementLog:
    """Persistent record of self-improvement proposals and their outcomes.

    Lets Primus learn what kinds of changes he's attempted and which were actually approved,
    so self-analysis can reference real history instead of guessing. Pure logging — it never
    applies anything and has no bearing on the approval gate.
    """

    _lock = threading.Lock()
    _MAX = 100

    @classmethod
    def _load(cls) -> list[dict[str, Any]]:
        try:
            data = json.loads(_host.SELF_IMPROVE_LOG.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    @classmethod
    def _save(cls, items: list[dict[str, Any]]) -> None:
        try:
            _host.ensure_app_dirs()
            _host.SELF_IMPROVE_LOG.write_text(json.dumps(items[-cls._MAX:], indent=2), encoding="utf-8")
        except OSError as exc:
            _host.log.debug("Could not save self-improvement log: %s", exc)

    @classmethod
    def record_proposal(cls, description: str, kind: str, op_count: int, *, tool_name: str = "") -> str:
        """Record a freshly staged proposal. Returns its id (store it on the pending edit)."""
        entry_id = "si-" + uuid.uuid4().hex[:8]
        with cls._lock:
            items = cls._load()
            items.append({
                "id": entry_id,
                "description": description,
                "kind": kind,                 # "code" | "new_tool" | "batch"
                "op_count": int(op_count),
                "tool_name": tool_name,
                "status": "proposed",
                "created_at": _host._now_iso(),
                "resolved_at": "",
            })
            cls._save(items)
        return entry_id

    @classmethod
    def mark_outcome(cls, entry_id: str, status: str) -> None:
        """Update a proposal's status to 'applied' or 'rejected'. No-op if id is missing."""
        if not entry_id:
            return
        with cls._lock:
            items = cls._load()
            for it in items:
                if it.get("id") == entry_id:
                    it["status"] = status
                    it["resolved_at"] = _host._now_iso()
                    break
            cls._save(items)

    @classmethod
    def recent(cls, limit: int = 8) -> list[dict[str, Any]]:
        return list(reversed(cls._load()))[:limit]

    @classmethod
    def summary(cls) -> str:
        items = cls._load()
        if not items:
            return ("_No self-improvement attempts logged yet. Use `design_and_propose_tool(...)` "
                    "or `propose_code_change(...)`; each proposal is tracked here with its outcome._")
        applied = sum(1 for i in items if i.get("status") == "applied")
        rejected = sum(1 for i in items if i.get("status") == "rejected")
        proposed = sum(1 for i in items if i.get("status") == "proposed")
        lines = [
            "## Self-improvement history",
            f"**{len(items)}** proposals — ✅ {applied} applied · ❌ {rejected} rejected · "
            f"⏳ {proposed} awaiting approval",
            "",
            "**Recent:**",
        ]
        icon = {"applied": "✅", "rejected": "❌", "proposed": "⏳"}
        for it in cls.recent(8):
            tag = icon.get(it.get("status", ""), "•")
            when = (it.get("resolved_at") or it.get("created_at") or "")[:10]
            lines.append(f"- {tag} ({when}) {it.get('description', '')[:90]}")
        return "\n".join(lines)


def _infer_lesson_category(text: str) -> str:
    """Best-effort category from free text (used when auto-logging outcomes)."""
    t = (text or "").lower()
    rules = [
        ("background", ("background agent", "scheduled", "overnight", "daemon")),
        ("tooling", ("tool", "register", "build_tools", "@tool")),
        ("coding", ("project", "code", "compile", "refactor", "forge", "app", "python")),
        ("research", ("research", "web", "ingest", "download", "pdf")),
        ("memory", ("memory", "recall", "rag", "embedding", "knowledge")),
        ("routing", ("route", "router", "intent", "model")),
        ("performance", ("latency", "slow", "fast-path", "cache", "speed", "timeout")),
    ]
    for cat, kws in rules:
        if any(k in t for k in kws):
            return cat
    return "process"


class SelfImprovementMemory:
    """A dedicated, structured long-term memory about Primus's own growth — separate from normal
    user memory, session memory, and the general MemorySystem.

    Each entry captures: what was ATTEMPTED (new tool, UI change, refactor…), the OUTCOME (success
    metrics / problems found), the LESSON learned, a category, tags, and an effectiveness note. It's
    purely advisory — it informs self-analysis and planning, never applies anything.

    Efficiency: retrieval is lazy. Semantic ranking reuses the existing (already-cached) KB
    embeddings over ONLY this small lesson set, with an in-memory per-entry vector cache, and is
    invoked solely when lessons are explicitly recalled (planning / `recall_self_lessons`). Normal
    chat turns never touch it, so there is zero added cost on the hot path. Falls back to fast
    keyword scoring whenever embeddings are unavailable.
    """

    _lock = threading.Lock()
    _MAX = 300
    _CATEGORIES = ("tooling", "background", "performance", "routing", "coding",
                   "research", "memory", "ui", "process", "general")
    # Lazy in-memory vector cache: entry-id -> (content_hash, vector). Never persisted.
    _vec_cache: dict[str, tuple[str, list[float]]] = {}

    @classmethod
    def _load(cls) -> list[dict[str, Any]]:
        try:
            data = json.loads(_host.SELF_IMPROVE_MEMORY.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    @classmethod
    def _save(cls, items: list[dict[str, Any]]) -> None:
        try:
            _host.ensure_app_dirs()
            _host.SELF_IMPROVE_MEMORY.write_text(json.dumps(items[-cls._MAX:], indent=2), encoding="utf-8")
        except OSError as exc:
            _host.log.debug("Could not save self-improvement memory: %s", exc)

    @staticmethod
    def _entry_text(entry: dict[str, Any]) -> str:
        """Combined searchable/embeddable text for an entry."""
        parts = [
            entry.get("lesson", ""),
            entry.get("attempt", ""),
            entry.get("outcome", ""),
            entry.get("category", ""),
            " ".join(entry.get("tags", []) or []),
        ]
        return " · ".join(p for p in parts if p).strip()

    @classmethod
    def add(cls, lesson: str, category: str = "general", effectiveness: str = "",
            tags: Optional[list[str]] = None, attempt: str = "", outcome: str = "") -> dict[str, Any]:
        lesson = (lesson or "").strip()
        if not lesson:
            return {}
        cat = (category or "general").strip().lower()
        if cat not in cls._CATEGORIES:
            cat = "general"
        entry = {
            "id": "sim-" + uuid.uuid4().hex[:8],
            "lesson": lesson[:600],
            "attempt": (attempt or "").strip()[:300],
            "outcome": (outcome or "").strip()[:300],
            "category": cat,
            "effectiveness": (effectiveness or "").strip()[:120],
            "tags": [t.strip().lower() for t in (tags or []) if t.strip()][:8],
            "created_at": _host._now_iso(),
        }
        with cls._lock:
            items = cls._load()
            # De-dupe near-identical lessons (case-insensitive exact match).
            if not any(i.get("lesson", "").strip().lower() == lesson.lower() for i in items):
                items.append(entry)
                cls._save(items)
        return entry

    # --- retrieval -----------------------------------------------------------

    @classmethod
    def _keyword_rank(cls, items: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
        terms = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
        scored: list[tuple[int, dict[str, Any]]] = []
        for it in items:
            blob = cls._entry_text(it).lower()
            score = sum(blob.count(t) for t in terms)
            if score:
                scored.append((score, it))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [it for _, it in scored[:limit]]

    @classmethod
    def _semantic_rank(cls, items: list[dict[str, Any]], query: str, limit: int) -> Optional[list[dict[str, Any]]]:
        """Cosine-rank entries via the cached KB embeddings. Returns None if embeddings unavailable."""
        if not _host.CFG.get("self_improve_semantic", True):
            return None
        try:
            emb = _host.get_kb()._get_embeddings()
        except Exception:  # noqa: BLE001
            return None
        try:
            # Embed only entries whose text changed since last cache (lazy + bounded).
            to_embed: list[tuple[str, str]] = []
            for it in items:
                eid = it.get("id", "")
                text = cls._entry_text(it)
                h = _host._content_hash(text)
                cached = cls._vec_cache.get(eid)
                if not cached or cached[0] != h:
                    to_embed.append((eid, text))
            if to_embed:
                vecs = emb.embed_documents([t for _, t in to_embed])
                for (eid, text), v in zip(to_embed, vecs):
                    cls._vec_cache[eid] = (_host._content_hash(text), v)
            qvec = emb.embed_query(query)

            def _cos(a: list[float], b: list[float]) -> float:
                dot = sum(x * y for x, y in zip(a, b))
                na = sum(x * x for x in a) ** 0.5
                nb = sum(y * y for y in b) ** 0.5
                return dot / (na * nb) if na and nb else 0.0

            scored = []
            for it in items:
                v = cls._vec_cache.get(it.get("id", ""))
                if v:
                    scored.append((_cos(qvec, v[1]), it))
            scored.sort(key=lambda kv: kv[0], reverse=True)
            # Keep only meaningfully-similar hits.
            return [it for s, it in scored[:limit] if s > 0.18]
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("Self-improvement semantic rank failed: %s", exc)
            return None

    @classmethod
    def search(cls, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """Top-k relevant lessons. Semantic when embeddings are ready (lazy), else keyword."""
        items = cls._load()
        limit = max(1, min(int(limit or 6), 20))
        q = (query or "").strip()
        if not q:
            return list(reversed(items))[:limit]
        if len(items) >= 3:
            sem = cls._semantic_rank(items, q, limit)
            if sem is not None:
                return sem or cls._keyword_rank(items, q, limit)
        return cls._keyword_rank(items, q, limit)

    @classmethod
    def recent(cls, limit: int = 6) -> list[dict[str, Any]]:
        return list(reversed(cls._load()))[:limit]

    @classmethod
    def summary(cls) -> str:
        items = cls._load()
        if not items:
            return ("_No self-improvement lessons recorded yet. As I learn what works, I'll save "
                    "lessons with `remember_self_lesson(...)` and draw on them when planning._")
        by_cat: dict[str, int] = {}
        for it in items:
            by_cat[it.get("category", "general")] = by_cat.get(it.get("category", "general"), 0) + 1
        cat_line = " · ".join(f"{c}: {n}" for c, n in sorted(by_cat.items(), key=lambda kv: -kv[1]))
        lines = [
            "## Self-improvement lessons",
            f"**{len(items)}** lessons — {cat_line}",
            "",
            "**Most recent:**",
        ]
        for it in cls.recent(8):
            eff = f" _(impact: {it['effectiveness']})_" if it.get("effectiveness") else ""
            extra = ""
            if it.get("outcome"):
                extra = f" — outcome: {it['outcome'][:60]}"
            lines.append(f"- [{it.get('category', 'general')}] {it.get('lesson', '')[:110]}{extra}{eff}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Codebase introspection: structured, offline "semantic" search over Primus's
# own source (functions, classes, methods, @tools). Deterministic + fast, so it
# works well inside background/scheduled agents with no embedding dependency.
# ---------------------------------------------------------------------------

_SYMBOL_INDEX_CACHE: dict[str, Any] = {"mtime": 0.0, "symbols": []}


def _index_own_symbols() -> list[dict[str, Any]]:
    """AST-index Primus's own source: classes, functions, methods, and @tools. Cached by mtime."""
    try:
        mtime = _host.SCRIPT_PATH.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _SYMBOL_INDEX_CACHE["symbols"] and _SYMBOL_INDEX_CACHE["mtime"] == mtime:
        return _SYMBOL_INDEX_CACHE["symbols"]

    src = _read_source()
    symbols: list[dict[str, Any]] = []
    if src:
        import ast
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None

        def _decorators(node: Any) -> list[str]:
            out: list[str] = []
            for d in getattr(node, "decorator_list", []) or []:
                if isinstance(d, ast.Name):
                    out.append(d.id)
                elif isinstance(d, ast.Attribute):
                    out.append(d.attr)
                elif isinstance(d, ast.Call) and isinstance(d.func, ast.Name):
                    out.append(d.func.id)
            return out

        def _sig(node: Any) -> str:
            try:
                args = [a.arg for a in node.args.args]
                return f"{node.name}({', '.join(args)})"
            except Exception:  # noqa: BLE001
                return getattr(node, "name", "?") + "(...)"

        def _doc1(node: Any) -> str:
            try:
                d = ast.get_docstring(node) or ""
            except Exception:  # noqa: BLE001
                d = ""
            return d.strip().splitlines()[0][:160] if d.strip() else ""

        if tree is not None:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decs = _decorators(node)
                    symbols.append({
                        "kind": "tool" if "tool" in decs else "function",
                        "name": node.name, "qualname": node.name, "sig": _sig(node),
                        "doc": _doc1(node), "lineno": node.lineno,
                        "is_tool": "tool" in decs, "decorators": decs,
                    })
                elif isinstance(node, ast.ClassDef):
                    symbols.append({
                        "kind": "class", "name": node.name, "qualname": node.name,
                        "sig": f"class {node.name}", "doc": _doc1(node),
                        "lineno": node.lineno, "is_tool": False, "decorators": _decorators(node),
                    })
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            symbols.append({
                                "kind": "method", "name": sub.name,
                                "qualname": f"{node.name}.{sub.name}", "sig": _sig(sub),
                                "doc": _doc1(sub), "lineno": sub.lineno,
                                "is_tool": False, "decorators": _decorators(sub),
                            })
    _SYMBOL_INDEX_CACHE["symbols"] = symbols
    _SYMBOL_INDEX_CACHE["mtime"] = mtime
    return symbols


def _rank_symbols(query: str, *, tools_only: bool = False, limit: int = 8) -> list[dict[str, Any]]:
    """Rank indexed symbols by token overlap with the query (name > tag > doc)."""
    symbols = _index_own_symbols()
    if tools_only:
        symbols = [s for s in symbols if s.get("is_tool")]
    q = (query or "").strip().lower()
    terms = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 1]
    wants_tool = any(w in q for w in ("tool", "tools", "capability", "capabilities"))
    scored: list[tuple[int, dict[str, Any]]] = []
    for s in symbols:
        name = s["name"].lower()
        doc = s.get("doc", "").lower()
        score = 0
        for t in terms:
            if t == name:
                score += 8
            elif t in name:
                score += 4
            if t in doc:
                score += 2
            if t in s.get("qualname", "").lower():
                score += 1
        if wants_tool and s.get("is_tool"):
            score += 3
        if score:
            scored.append((score, s))
    scored.sort(key=lambda kv: (kv[0], kv[1].get("is_tool", False)), reverse=True)
    if not scored and terms:  # graceful fallback: substring scan
        for s in symbols:
            if any(t in (s["name"] + s.get("doc", "")).lower() for t in terms):
                scored.append((1, s))
    return [s for _, s in scored[:limit]]


@tool
def search_own_codebase(query: str, limit: int = 8, tools_only: bool = False) -> str:
    """Semantically search Primus's OWN source for relevant functions, classes, methods, and tools.

    Ask things like "what tools do I have for downloading files", "background agent scheduling", or
    "memory recall". Returns the most relevant symbols with their signature, a one-line description,
    and line number so you can then `read_own_code` the exact area. READ-ONLY.

    Args:
      query: what you're looking for (a capability, subsystem, or concept).
      limit: max results (default 8, capped at 20).
      tools_only: if true, only return @tool definitions (useful for "what tools do I have for X").
    """
    query = (query or "").strip()
    if not query:
        return "Tell me what to look for (a capability, subsystem, or symbol)."
    limit = max(1, min(int(limit or 8), 20))
    hits = _rank_symbols(query, tools_only=bool(tools_only), limit=limit)
    if not hits:
        return f"No matching symbols for {query!r} in my source. Try different keywords."
    icon = {"tool": "🛠", "class": "🏗", "method": "▸", "function": "ƒ"}
    lines = [f"**{len(hits)}** matches for {query!r} in admin_assistant.py:"]
    for s in hits:
        marker = icon.get(s["kind"], "•")
        doc = f" — {s['doc']}" if s.get("doc") else ""
        lines.append(f"- {marker} `{s['sig']}` (L{s['lineno']}){doc}")
    lines.append("\n_Use `read_own_code(query=...)` or `explain_my_architecture(topic=...)` to go deeper._")
    return "\n".join(lines)


@tool
def explain_my_architecture(topic: str) -> str:
    """Explain how a part of Primus's own architecture works, grounded in the actual source.

    Give a subsystem or concept (e.g. "background agent system", "memory and RAG", "model routing",
    "self-improvement", "knowledge ingestion"). Primus finds the relevant code and explains how it's
    structured and how the pieces fit together. READ-ONLY — it reads source, never modifies it.
    """
    topic = (topic or "").strip()
    if not topic:
        return "Name a subsystem or concept to explain (e.g. 'background agents', 'model routing')."
    hits = _rank_symbols(topic, limit=10)
    if not hits:
        return f"I couldn't find code clearly related to {topic!r}. Try `search_own_codebase`."

    overview = "\n".join(
        f"- {s['kind']} `{s['sig']}` (L{s['lineno']})" + (f" — {s['doc']}" if s.get("doc") else "")
        for s in hits
    )
    # Build a bounded source context from the top matches for a grounded explanation.
    src_lines = _read_source().splitlines()
    context_blocks: list[str] = []
    for s in hits[:5]:
        lo = max(0, s["lineno"] - 1)
        hi = min(len(src_lines), lo + 40)
        context_blocks.append(f"# {s['qualname']} (L{s['lineno']})\n" + "\n".join(src_lines[lo:hi]))
    context = "\n\n".join(context_blocks)[:6000]

    if not _host.HAS_AI_STACK:
        return f"## Architecture: {topic}\n\nRelevant components:\n{overview}"
    try:
        llm = _host.make_chat_ollama(_host.CFG.get("model", _host.DEFAULT_MODEL), temperature=0.2, num_predict=900)
        prompt = (
            f"You are explaining your own codebase. Based ONLY on the source excerpts below, explain "
            f"how '{topic}' works in this system: its key components, how they fit together, and any "
            f"notable design choices. Be concrete and reference the function/class names. Do not invent "
            f"behavior that isn't in the code.\n\nRelevant symbols:\n{overview}\n\nSource excerpts:\n{context}"
        )
        explanation = _host.coerce_message_text(llm.invoke([HumanMessage(content=prompt)]).content)
    except Exception as exc:  # noqa: BLE001
        return f"## Architecture: {topic}\n\nRelevant components:\n{overview}\n\n(Synthesis unavailable: {exc})"
    return f"## How '{topic}' works\n\n{explanation}\n\n**Key components:**\n{overview}"


@tool
def read_own_code(query: str = "", start_line: int = 0, num_lines: int = 80) -> str:
    """Read Primus's own source code (admin_assistant.py) — READ-ONLY, for self-analysis.

    Use this to understand or critique your own implementation before suggesting improvements.

    Args:
      query: optional keyword/symbol to locate (e.g. "def browse_web", "ModelRouter").
             Returns matching line ranges with a little context.
      start_line: 1-based line to start reading from (used when query is empty).
      num_lines: how many lines to return (default 80, capped at 200).
    """
    src = _read_source()
    if not src:
        return "I couldn't read my own source file."
    lines = src.splitlines()
    total = len(lines)
    num_lines = max(1, min(int(num_lines or 80), 200))

    if query.strip():
        q = query.strip().lower()
        matches = [i for i, ln in enumerate(lines) if q in ln.lower()]
        if not matches:
            return f"No lines matching {query!r} in admin_assistant.py ({total} lines total)."
        out: list[str] = [f"admin_assistant.py — {len(matches)} match(es) for {query!r} ({total} lines total):"]
        shown = 0
        for idx in matches[:6]:
            lo = max(0, idx - 3)
            hi = min(total, idx + 12)
            out.append(f"\n— lines {lo + 1}-{hi} —")
            for j in range(lo, hi):
                marker = ">>" if j == idx else "  "
                out.append(f"{marker}{j + 1:5} | {lines[j]}")
            shown += 1
        if len(matches) > 6:
            out.append(f"\n…and {len(matches) - 6} more matches. Narrow the query to see them.")
        return "\n".join(out)[:8000]

    start = max(1, int(start_line or 1))
    end = min(total, start + num_lines - 1)
    body = "\n".join(f"{i:5} | {lines[i - 1]}" for i in range(start, end + 1))
    return f"admin_assistant.py lines {start}-{end} of {total}:\n{body}"[:8000]


def _usage_patterns_report() -> str:
    """Build a usage-pattern report grouped by business-relevant capability area."""
    try:
        data = _host.get_metrics()._data  # same-module read of the metrics snapshot
    except Exception as exc:  # noqa: BLE001
        return f"Usage patterns unavailable: {exc}"
    tools = data.get("tools", {})
    if not tools:
        return ("## Usage patterns\n_No tool usage recorded yet — patterns appear here as I work "
                "(coding, research, client tasks, background jobs)._")

    # Aggregate per category.
    cats: dict[str, dict[str, float]] = {}
    for name, t in tools.items():
        cat = _tool_category(name)
        c = cats.setdefault(cat, {"calls": 0, "ok": 0, "fail": 0, "ms": 0.0})
        c["calls"] += t.get("calls", 0)
        c["ok"] += t.get("ok", 0)
        c["fail"] += t.get("fail", 0)
        c["ms"] += t.get("ms_total", 0.0)

    lines = ["## Usage patterns by area"]
    for cat, c in sorted(cats.items(), key=lambda kv: kv[1]["calls"], reverse=True):
        calls = int(c["calls"])
        if not calls:
            continue
        rate = c["ok"] / max(1, calls)
        avg = c["ms"] / max(1, calls)
        lines.append(f"- **{cat}** — {calls} calls · {rate:.0%} ok · {avg / 1000:.2f}s avg")

    # Most-used tools overall.
    top = sorted(tools.items(), key=lambda kv: kv[1].get("calls", 0), reverse=True)[:5]
    if top:
        lines.append("\n**Most-used tools:**")
        for name, t in top:
            calls = t.get("calls", 0)
            lines.append(f"- `{name}` ({_tool_category(name)}) — {calls} calls")

    # Common failure modes.
    fails = [
        (name, t) for name, t in tools.items()
        if t.get("calls", 0) >= 3 and (t.get("ok", 0) / max(1, t.get("calls", 1))) < 0.8
    ]
    if fails:
        lines.append("\n**Common failure modes:**")
        for name, t in sorted(fails, key=lambda kv: kv[1].get("fail", 0), reverse=True)[:5]:
            calls = t.get("calls", 0)
            rate = t.get("ok", 0) / max(1, calls)
            lines.append(f"- `{name}` — {rate:.0%} success over {calls} calls ({_tool_category(name)})")

    # Response paths + routing context.
    paths = data.get("paths", {})
    if paths:
        lines.append("\n**Response paths** (count · avg):")
        for path, p in sorted(paths.items(), key=lambda kv: kv[1].get("n", 0), reverse=True)[:5]:
            n = p.get("n", 0)
            avg = p.get("ms_total", 0.0) / max(1, n)
            lines.append(f"- {path} — {n} · {avg / 1000:.2f}s")
    return "\n".join(lines)


def _self_improvement_plan() -> str:
    """Synthesize a concrete, prioritized self-improvement roadmap from all available signals."""
    # Gather raw signals.
    try:
        weak = _host.get_metrics().weaknesses()
    except Exception:  # noqa: BLE001
        weak = []
    try:
        gap = _capability_gap_hint()
    except Exception:  # noqa: BLE001
        gap = ""
    try:
        reflections = _host.get_memory_system().recent_self_improvements(limit=4)
    except Exception:  # noqa: BLE001
        reflections = []
    # Pull lessons RELEVANT to the current weaknesses/gap (semantic), not just the most recent.
    lesson_query = " ".join(weak + ([gap] if gap else [])).strip()
    lessons = (SelfImprovementMemory.search(lesson_query, limit=6)
               if lesson_query else SelfImprovementMemory.recent(6))
    history = SelfImprovementLog.recent(5)

    # Deterministic, prioritized skeleton (always available, even offline).
    p1 = [w for w in weak if "succeeds only" in w]
    p2 = [w for w in weak if "slow" in w]
    p3 = [w for w in weak if "Negative feedback" in w]
    skeleton: list[str] = ["## Self-improvement plan", ""]
    n = 1
    for w in p1:
        skeleton.append(f"{n}. **[P1 reliability]** {w}\n   → `propose_code_change(...)` to harden it.")
        n += 1
    for w in p2:
        skeleton.append(f"{n}. **[P2 latency]** {w}\n   → add a fast-path/cache via `propose_code_change(...)`.")
        n += 1
    if gap:
        skeleton.append(f"{n}. **[Capability]** {gap}\n   → `design_and_propose_tool(purpose=...)`.")
        n += 1
    for w in p3:
        skeleton.append(f"{n}. **[P3 quality]** {w}\n   → tighten routing; adaptive nudge is learning.")
        n += 1
    for r in reflections:
        skeleton.append(f"{n}. **[Reflection]** {r}")
        n += 1
    if n == 1:
        skeleton.append("1. No pressing weaknesses detected — consider a capability you wish you had "
                        "and draft it with `design_and_propose_tool(...)`.")

    context_bits = []
    if lessons:
        context_bits.append("Lessons learned:\n" + "\n".join(
            f"- [{ls.get('category')}] {ls.get('lesson')}" for ls in lessons))
    if history:
        icon = {"applied": "✅", "rejected": "❌", "proposed": "⏳"}
        context_bits.append("Recent proposal outcomes:\n" + "\n".join(
            f"- {icon.get(h.get('status'), '•')} {h.get('description', '')[:80]}" for h in history))
    context = "\n\n".join(context_bits)

    base = "\n".join(skeleton) + (("\n\n---\n" + context) if context else "")
    if not _host.HAS_AI_STACK:
        base += ("\n\n_All changes remain human-gated: I draft them, you apply with `/approve edit`._")
        return base

    # Forge-polish into a tighter narrative plan, grounded in the deterministic signals above.
    try:
        llm = _host.make_chat_ollama(_host.CFG.get("model", _host.DEFAULT_MODEL), temperature=0.3, num_predict=900)
        prompt = (
            "You are planning improvements to YOURSELF (an AI assistant). Turn the signals below into "
            "a concise, prioritized self-improvement plan: 3-6 numbered items, each with a clear action "
            "and the tool to use (propose_code_change, design_and_propose_tool, or propose_changes_batch). "
            "Be specific and grounded ONLY in these signals — don't invent problems. Keep it tight.\n\n"
            f"{base}"
        )
        plan = _host.coerce_message_text(llm.invoke([HumanMessage(content=prompt)]).content).strip()
    except Exception:  # noqa: BLE001
        plan = ""
    if not plan:
        plan = base
    return (f"{plan}\n\n_All changes remain human-gated: I draft them, you apply with `/approve edit`._")


def _capability_gap_hint() -> str:
    """If a heavily-used area is underserved or failing, suggest a new tool to fill the gap."""
    try:
        data = _host.get_metrics()._data
    except Exception:  # noqa: BLE001
        return ""
    tools = data.get("tools", {})
    if not tools:
        return ""
    # Surface the busiest tool that's also unreliable — a strong candidate for a purpose-built helper.
    worst = None
    for name, t in tools.items():
        calls = t.get("calls", 0)
        if calls >= 5:
            rate = t.get("ok", 0) / max(1, calls)
            if rate < 0.75 and (worst is None or calls > worst[1]):
                worst = (name, calls, rate)
    if worst:
        return (f"`{worst[0]}` is used a lot ({worst[1]} calls) but only succeeds {worst[2]:.0%} of "
                f"the time — a dedicated, more robust tool for this workflow could help.")
    return ""


@tool
def analyze_self(aspect: str = "overview") -> str:
    """Analyze Primus's own architecture, tools, and recent performance — READ-ONLY.

    aspect:
      "overview"     — file size, counts of tools/classes/functions, top sections.
      "tools"        — the full registered tool list with one-line descriptions.
      "performance"  — recent self-reflection scores + improvement notes (why I drifted, etc.).
      "metrics"      — meta-learning stats: tool success/latency, routing, feedback.
      "patterns"     — usage patterns grouped by business area (coding, research, client work,
                       background) with most-used tools, hot/slow paths, and failure modes.
      "bottlenecks"  — concrete weak spots (unreliable/slow tools, slow paths, 👎 trend).
      "suggestions"  — prioritized, actionable improvement proposals derived from real usage,
                       each tagged with the exact next step (and which proposal tool to use).
      "history"      — past self-improvement proposals and whether they were applied/rejected.
      "lessons"      — lessons learned (self-improvement memory) grouped by category.
      "plan"         — a concrete, prioritized self-improvement roadmap synthesizing weaknesses,
                       gaps, usage patterns, past outcomes, and learned lessons.
      "capabilities" — high-level summary of what Primus can do.
    """
    aspect = (aspect or "overview").strip().lower()
    src = _read_source()
    lines = src.splitlines()

    if aspect in ("history", "log", "self-improvements", "improvements-log"):
        return SelfImprovementLog.summary()

    if aspect in ("lessons", "memory", "learned"):
        return SelfImprovementMemory.summary()

    if aspect in ("plan", "roadmap", "improvement-plan"):
        return _self_improvement_plan()

    if aspect in ("patterns", "usage", "usage-patterns"):
        return _usage_patterns_report()

    if aspect in ("metrics", "stats"):
        try:
            return _host.get_metrics().report()
        except Exception as exc:  # noqa: BLE001
            return f"Metrics unavailable: {exc}"

    if aspect in ("bottlenecks", "weaknesses", "weak"):
        try:
            weak = _host.get_metrics().weaknesses()
        except Exception as exc:  # noqa: BLE001
            return f"Metrics unavailable: {exc}"
        return "## Bottlenecks & weaknesses\n" + (
            "\n".join(f"- {w}" for w in weak) if weak else "- None detected yet — metrics still warming up."
        )

    if aspect in ("suggestions", "improve", "improvements"):
        bits: list[str] = ["## Prioritized improvement proposals", ""]
        try:
            weak = _host.get_metrics().weaknesses()
        except Exception:  # noqa: BLE001
            weak = []

        # Bucket weaknesses by priority so the most impactful fixes surface first.
        p1_fail, p2_slow, p3_fb, p_other = [], [], [], []
        for w in weak:
            if "succeeds only" in w:
                p1_fail.append(f"- **[P1 reliability]** {w}\n  → `propose_code_change(...)` to harden inputs/error handling.")
            elif "slow" in w:
                p2_slow.append(f"- **[P2 latency]** {w}\n  → Add a fast-path/cache via `propose_code_change(...)`.")
            elif "Negative feedback" in w:
                p3_fb.append(f"- **[P3 quality]** {w}\n  → Tighten routing for the intent (the adaptive nudge is already learning).")
            else:
                p_other.append(f"- {w}")
        ordered = p1_fail + p2_slow + p3_fb + p_other
        if ordered:
            bits += ordered
        else:
            bits.append("- No metric-driven weaknesses yet — reliability, latency, and feedback look healthy.")

        # Capability-gap nudge from usage patterns: a heavily-used area with no dedicated tool.
        try:
            gap = _capability_gap_hint()
            if gap:
                bits.append("")
                bits.append(f"- **[Capability gap]** {gap}\n  → Draft one with `design_and_propose_tool(purpose=...)`.")
        except Exception:  # noqa: BLE001
            pass

        try:
            improvements = _host.get_memory_system().recent_self_improvements(limit=4)
        except Exception:  # noqa: BLE001
            improvements = []
        if improvements:
            bits.append("")
            bits.append("**From self-reflection:**")
            for n in improvements:
                bits.append(f"- {n}")

        # Past-outcome context so Primus learns from what was accepted/rejected before.
        recent = SelfImprovementLog.recent(3)
        if recent:
            bits.append("")
            bits.append("**Recent self-improvement history:**")
            icon = {"applied": "✅", "rejected": "❌", "proposed": "⏳"}
            for it in recent:
                bits.append(f"- {icon.get(it.get('status'), '•')} {it.get('description', '')[:80]}")

        # Relevant lessons learned, so suggestions build on what actually worked.
        _lq = " ".join(weak).strip()
        lessons = (SelfImprovementMemory.search(_lq, limit=3) if _lq
                   else SelfImprovementMemory.recent(3))
        if lessons:
            bits.append("")
            bits.append("**Lessons to apply:**")
            for ls in lessons:
                bits.append(f"- [{ls.get('category', 'general')}] {ls.get('lesson', '')[:90]}")

        bits.append("")
        bits.append(
            "**Next step:** I draft a concrete change — `propose_code_change(...)`, "
            "`propose_changes_batch(...)` for several related edits, or `design_and_propose_tool(...)` "
            "for a brand-new tool — and it waits for your **`/approve edit`**."
        )
        return "\n".join(bits)

    if aspect in ("tools", "tool", "capabilities"):
        try:
            tools = build_tools()
        except Exception as exc:  # noqa: BLE001
            return f"Couldn't enumerate tools: {exc}"
        rows = []
        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", t.__class__.__name__)
            desc = (getattr(t, "description", None) or (getattr(t, "__doc__", "") or "")).strip()
            rows.append(f"- **{name}** — {desc.splitlines()[0][:100] if desc else 'no description'}")
        head = f"Primus has **{len(tools)}** registered tools:" if aspect != "capabilities" else \
            f"Primus capabilities ({len(tools)} tools across shell, files, web, system, memory, KB):"
        return head + "\n" + "\n".join(rows)

    if aspect in ("performance", "reflection", "reflections"):
        try:
            return _host.get_memory_system().reflection_report(limit=12)
        except Exception as exc:  # noqa: BLE001
            return f"No reflection data available ({exc})."

    # overview (default)
    n_tools = src.count("@tool")
    n_def = src.count("\ndef ")
    n_class = src.count("\nclass ")
    return (
        "## Primus self-overview\n"
        f"- Source: `{_host.SCRIPT_PATH}` — **{len(lines)}** lines\n"
        f"- ~**{n_tools}** @tool definitions · **{n_def}** functions · **{n_class}** classes\n"
        f"- Dual-model: Primus (chat/admin) + Forge (coding), deterministic ModelRouter\n"
        f"- Memory: session facts → LTM → KB (Chroma) + conversation archive + learning digests\n"
        f"- Continuous learning: per-turn self-reflection, periodic digests, core-knowledge promotion\n\n"
        "Use `read_own_code(query=...)` to inspect a specific area, or `analyze_self('tools')` "
        "for the tool list. To improve something, call `propose_code_change(...)` — I'll show the "
        "diff and apply it only after you type `/approve edit`."
    )


def _self_edit_make_op(find: str, replace: str) -> dict[str, Any]:
    return {"find": find, "replace": replace}


def _stage_self_edit(description: str, ops: list[dict[str, Any]]) -> str:
    """Validate a set of find/replace ops against the source and stage them. No disk writes.

    Returns a human-readable preview (or an error explaining why it can't be staged).
    """
    src = _read_source()
    if not src:
        return "I can't read my own source right now, so I can't propose a change."

    working = src
    previews: list[str] = []
    for n, op in enumerate(ops, 1):
        find = op.get("find", "")
        replace = op.get("replace", "")
        if not find:
            return "Proposal rejected: every change needs the exact existing text to locate (`find`)."
        if len(replace) > _SELF_EDIT_MAX_REPLACE:
            return f"Proposal rejected: change #{n} is too large ({len(replace)} chars). Break it up."
        count = working.count(find)
        if count == 0:
            return (
                f"Proposal rejected: I couldn't find the exact text for change #{n} in the current "
                "source. Re-read the relevant section with `read_own_code` and copy it verbatim."
            )
        if count > 1:
            return (
                f"Proposal rejected: the text for change #{n} appears {count} times (ambiguous). "
                "Include more surrounding context so it matches exactly one place."
            )
        # Guardrail: never let a self-edit delete a critical safety marker.
        for marker in _SELF_EDIT_CRITICAL_MARKERS:
            if marker in find and marker not in replace:
                return (
                    f"Proposal rejected: change #{n} would remove a safety guardrail "
                    f"(`{marker}`). Self-modification cannot weaken its own approval gate."
                )
        working = working.replace(find, replace, 1)
        # Unified diff for a clear, reviewable view of exactly what changes.
        import difflib
        diff = "".join(
            difflib.unified_diff(
                find.splitlines(keepends=True),
                (replace or "").splitlines(keepends=True),
                fromfile=f"change{n}/before", tofile=f"change{n}/after", n=2,
            )
        )[:1400]
        previews.append(
            f"### Change {n}: {description if len(ops) == 1 else ''}\n"
            + (f"```diff\n{diff}\n```\n" if diff.strip() else "")
            + f"<details><summary>full before/after</summary>\n\n"
            f"**Remove:**\n```python\n{find[:600]}\n```\n"
            f"**Add:**\n```python\n{(replace or '(deletes the above)')[:600]}\n```\n</details>"
        )

    # Final safety net: the edited file must still parse as valid Python.
    try:
        import ast
        ast.parse(working)
    except SyntaxError as exc:
        return (
            "Proposal rejected: applying this would break Python syntax "
            f"(line {exc.lineno}: {exc.msg}). I won't stage an edit that doesn't compile."
        )

    kind = "new_tool" if description.startswith("Add new tool") else ("batch" if len(ops) > 1 else "code")
    log_id = SelfImprovementLog.record_proposal(description, kind, len(ops))
    _host.PrimusSession.pending_edit = {
        "description": description,
        "ops": ops,
        "created_at": _host._now_iso(),
        "new_size": len(working),
        "old_size": len(src),
        "log_id": log_id,
    }
    body = "\n\n".join(previews)
    return (
        f"**Proposed self-edit:** {description}\n\n{body}\n\n"
        "I have **not** changed anything. Review the diff above, then type **`/approve edit`** to "
        "apply it (I'll back up the file and verify it compiles first), or **`/reject edit`** to discard."
    )


@tool
def propose_code_change(description: str, find: str, replace: str = "") -> str:
    """Propose a change to Primus's OWN source code — STAGES ONLY, never applies.

    Use after analyzing yourself with `read_own_code`/`analyze_self`. Present the change; it is
    applied only when the operator types `/approve edit`. You cannot apply it yourself.

    Args:
      description: short explanation of what the change does and why it helps.
      find: the EXACT existing source text to replace (copy it verbatim, include enough context
            that it appears exactly once in the file).
      replace: the new text. Leave empty only to delete the matched block (rarely appropriate).
    """
    return _stage_self_edit(description.strip() or "self-improvement", [_self_edit_make_op(find, replace)])


def _stage_new_tool(code: str, registry_anchor: str = "index_knowledge_folder,") -> str:
    """Validate + stage a new @tool (insert before build_tools + register). No disk writes."""
    code = (code or "").strip()
    if "@tool" not in code or "def " not in code:
        return "That doesn't look like a @tool function. Include the `@tool` decorator and a `def`."
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
    if not m:
        return "I couldn't find the function name in that tool code."
    fn_name = m.group(1)

    # Pre-validate the snippet parses on its own (catches obvious syntax errors early).
    try:
        import ast
        ast.parse(code)
    except SyntaxError as exc:
        return f"That tool code doesn't parse (line {exc.lineno}: {exc.msg}). Fix it and re-propose."

    insert_op = _self_edit_make_op(
        "def build_tools() -> list:\n    return [",
        f"{code}\n\n\ndef build_tools() -> list:\n    return [",
    )
    anchor = registry_anchor.strip()
    if not anchor.endswith(","):
        anchor += ","
    registry_op = _self_edit_make_op(
        f"        {anchor}",
        f"        {anchor}\n        {fn_name},",
    )
    return _stage_self_edit(
        f"Add new tool `{fn_name}` and register it",
        [insert_op, registry_op],
    )


@tool
def propose_new_tool(tool_code: str, registry_anchor: str = "index_knowledge_folder,") -> str:
    """Propose adding a NEW @tool to Primus — STAGES ONLY, applied via `/approve edit`.

    Provide the full Python for a new `@tool`-decorated function. It will be inserted just before
    `build_tools()` and registered in the tool list. Nothing is written until the operator approves.

    Args:
      tool_code: complete source of the new tool, e.g.
                 '@tool\\ndef my_tool(x: str) -> str:\\n    \"\"\"Docs.\"\"\"\\n    return x'
      registry_anchor: an existing tool name (with trailing comma) in build_tools() to insert after.
                       Defaults to the last entry. Must match exactly one registry line.
    """
    return _stage_new_tool(tool_code, registry_anchor)


def _tool_pattern_examples(max_chars: int = 2400) -> str:
    """Pull a couple of real @tool definitions from the source as style/pattern exemplars."""
    src = _read_source()
    if not src:
        return ""
    marker = "\n@tool\n"
    starts = [m.start() for m in re.finditer(re.escape(marker), src)]
    examples: list[str] = []
    for i in starts:
        block_start = i + 1  # skip the leading newline so the snippet begins at "@tool"
        nxt = src.find(marker, i + len(marker))  # next @tool block bounds this one
        end = nxt if nxt != -1 else block_start + 900
        snippet = src[block_start:end].strip()
        if 120 < len(snippet):
            examples.append(snippet[:1100])
        if len(examples) >= 2:
            break
    return ("\n\n# ---\n\n".join(examples))[:max_chars]


@tool
def design_and_propose_tool(
    purpose: str, suggested_name: str = "", inputs: str = "", notes: str = ""
) -> str:
    """Autonomously DESIGN and write a complete new @tool from a description — STAGES ONLY.

    Describe what the tool should do; Primus drafts a full, idiomatic `@tool` function (typed args,
    a clear docstring, and error handling) modeled on the existing tools in this codebase, then
    stages it for review. Nothing is written until the operator types `/approve edit`. Ideal for background
    self-improvement: turn an observed need into ready-to-approve code.

    Args:
      purpose: what the tool should accomplish and when it should be used.
      suggested_name: optional snake_case function name (otherwise the model picks one).
      inputs: optional description of the arguments the tool should take.
      notes: optional extra constraints (libraries to use/avoid, output format, safety).
    """
    purpose = (purpose or "").strip()
    if not purpose:
        return "Describe the tool's purpose so I can design it."
    if not _host.HAS_AI_STACK:
        return "Tool design needs the Forge model (Ollama + LangChain), which isn't available."
    examples = _tool_pattern_examples()
    prompt = (
        "Write ONE new tool for this Python codebase. Output ONLY the function source — no prose, "
        "no markdown fences. It MUST:\n"
        "- start with the `@tool` decorator on its own line,\n"
        "- be a top-level `def` with type-annotated args and a `-> str` return,\n"
        "- have a clear docstring (one-line summary, then Args), \n"
        "- handle errors defensively and return a helpful string (never raise),\n"
        "- only use the standard library or modules already imported in this file.\n\n"
        f"PURPOSE: {purpose}\n"
        f"SUGGESTED NAME: {suggested_name or '(you choose a clear snake_case name)'}\n"
        f"INPUTS: {inputs or '(infer sensible typed arguments)'}\n"
        f"NOTES: {notes or '(none)'}\n\n"
        f"Follow the style of these existing tools:\n\n{examples}"
    )
    try:
        llm = _forge_code_llm(int(_host.CFG.get("forge_num_predict", 2048)))
        code = _host.coerce_message_text(llm.invoke([HumanMessage(content=prompt)]).content)
    except Exception as exc:  # noqa: BLE001
        return f"Tool design failed: {exc}"
    code = re.sub(r"^```[\w]*\n?|```$", "", code.strip(), flags=re.MULTILINE).strip()
    staged = _stage_new_tool(code)
    if staged.startswith("Proposal rejected") or staged.startswith("That") or staged.startswith("I couldn't"):
        return f"Drafted a tool but it couldn't be staged:\n{staged}\n\n--- Draft ---\n```python\n{code[:1500]}\n```"
    return (
        f"**Designed a new tool for:** {purpose}\n\n"
        f"```python\n{code[:1800]}\n```\n\n"
        f"{staged}"
    )


@tool
def propose_changes_batch(description: str, changes_json: str) -> str:
    """Propose SEVERAL related source edits together as ONE reviewable change — STAGES ONLY.

    Use when an improvement spans multiple spots (e.g. a new helper + its call sites). All edits are
    staged as a single proposal and applied together only when the operator types `/approve edit`.

    Args:
      description: what the combined change accomplishes and why.
      changes_json: a JSON array of objects, each {"find": "<exact existing text>",
                    "replace": "<new text>"}. Each `find` must occur exactly once in the source.
    """
    try:
        raw = json.loads(changes_json)
    except Exception as exc:  # noqa: BLE001
        return f"changes_json must be valid JSON array of {{find, replace}} objects: {exc}"
    if not isinstance(raw, list) or not raw:
        return "Provide a non-empty JSON array of {find, replace} changes."
    ops: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or "find" not in item:
            return "Each change needs at least a `find` field (the exact existing text)."
        ops.append(_self_edit_make_op(str(item.get("find", "")), str(item.get("replace", ""))))
    return _stage_self_edit(description.strip() or "batched self-improvement", ops)


@tool
def remember_self_lesson(
    lesson: str, category: str = "general", effectiveness: str = "", tags: str = "",
    attempt: str = "", outcome: str = "",
) -> str:
    """Save a structured LESSON about your own improvement to dedicated self-improvement memory.

    Use this to record what works and what doesn't for improving yourself — e.g. "background agents
    perform better when given explicit success criteria", "the fast-coding path cut latency a lot",
    or "I struggle with large multi-file refactors". Stored separately from normal memory and drawn
    on when you plan future improvements (`analyze_self('plan')`). Background/scheduled agents should
    call this to log learnings from long-running self-improvement work.

    Args:
      lesson: the insight to remember (one or two sentences).
      category: one of tooling, background, performance, routing, coding, research, memory, ui, process, general.
      effectiveness: optional short impact note (e.g. "high impact", "minor", "backfired").
      tags: optional comma-separated keywords for retrieval.
      attempt: optional — what was attempted (e.g. "added deep_research tool").
      outcome: optional — the result, success metrics, or problems found.
    """
    tag_list = [t for t in re.split(r"[,;]", tags or "") if t.strip()]
    entry = SelfImprovementMemory.add(lesson, category, effectiveness, tag_list, attempt, outcome)
    if not entry:
        return "Give me a non-empty lesson to remember."
    return (f"Logged self-improvement lesson [{entry['category']}]: \"{entry['lesson'][:140]}\""
            + (f" (impact: {entry['effectiveness']})" if entry.get("effectiveness") else "")
            + ". I'll factor this into future self-improvement plans.")


@tool
def recall_self_lessons(query: str = "", limit: int = 6) -> str:
    """Recall lessons from self-improvement memory, optionally filtered by a query.

    Use before planning changes to yourself so you build on what already worked (or avoid what
    didn't). With no query, returns the most recent lessons. READ-ONLY.
    """
    limit = max(1, min(int(limit or 6), 20))
    hits = SelfImprovementMemory.search(query, limit=limit)
    if not hits:
        return ("No matching self-improvement lessons yet." if query
                else "No self-improvement lessons recorded yet.")
    head = f"Self-improvement lessons{(' for ' + repr(query)) if query else ''}:"
    lines = [head]
    for it in hits:
        eff = f" _(impact: {it['effectiveness']})_" if it.get("effectiveness") else ""
        lines.append(f"- [{it.get('category', 'general')}] {it.get('lesson', '')}{eff}")
    return "\n".join(lines)


def apply_pending_self_edit() -> str:
    """Apply the staged self-edit — invoked ONLY by the `/approve edit` command.

    Backs up the current file, applies every op to an in-memory copy, re-validates that it
    compiles, writes it, and reverts on any failure. This is the single, human-gated write path.
    """
    edit = _host.PrimusSession.pending_edit
    if not edit:
        return "No self-edit is staged. Ask me to `propose_code_change(...)` first."

    src = _read_source()
    if not src:
        return "I can't read my own source, so I won't write to it."

    working = src
    for n, op in enumerate(edit.get("ops", []), 1):
        find = op.get("find", "")
        if not find or working.count(find) != 1:
            _host.PrimusSession.pending_edit = None
            return (
                f"Apply aborted: change #{n} no longer matches uniquely (the file changed). "
                "Discarded the stale proposal — please re-propose."
            )
        working = working.replace(find, op.get("replace", ""), 1)

    # Must still compile.
    try:
        import ast
        ast.parse(working)
    except SyntaxError as exc:
        _host.PrimusSession.pending_edit = None
        return f"Apply aborted: edited source fails to compile (line {exc.lineno}: {exc.msg}). Nothing was written."

    # Back up, then write.
    try:
        _host.ensure_app_dirs()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = _host.SELF_EDIT_BACKUP_DIR / f"admin_assistant.{stamp}.py"
        backup.write_text(src, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _host.PrimusSession.pending_edit = None
        return f"Apply aborted: couldn't create a backup ({exc}). Nothing was written."

    try:
        _host.SCRIPT_PATH.write_text(working, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        # Best-effort restore from the backup we just made.
        try:
            _host.SCRIPT_PATH.write_text(src, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        _host.PrimusSession.pending_edit = None
        return f"Apply failed while writing ({exc}). Restored the original from backup."

    desc = edit.get("description", "self-edit")
    SelfImprovementLog.mark_outcome(edit.get("log_id", ""), "applied")
    # Close the learning loop: record the applied change as a structured lesson.
    try:
        SelfImprovementMemory.add(
            lesson=f"Applied self-edit: {desc}.",
            category=_infer_lesson_category(desc),
            effectiveness="applied",
            attempt=desc,
            outcome=f"Approved & applied; {edit.get('old_size', '?')}→{edit.get('new_size', '?')} chars, compiled OK.",
        )
    except Exception:  # noqa: BLE001
        pass
    _host.PrimusSession.pending_edit = None
    _host.log.info("Applied self-edit: %s (backup: %s)", desc, backup.name)
    return (
        f"Applied: **{desc}**.\n\n"
        f"- Backup saved: `{backup}`\n"
        f"- Size: {edit.get('old_size', '?')} → {edit.get('new_size', '?')} chars · verified it compiles.\n\n"
        "Restart Primus to load the new code. If anything misbehaves, restore the backup above."
    )


# ---------------------------------------------------------------------------
# High-ROI tools for autonomous / long-running (background) work.
# Usable by both the main Primus chat and background agents.
# ---------------------------------------------------------------------------

_RESEARCH_FACETS = [
    "overview and definition",
    "latest developments and news",
    "key facts, figures and data",
    "advantages, risks and trade-offs",
    "expert analysis and best practices",
    "future outlook",
]


@tool
def deep_research(topic: str, angles: int = 4) -> str:
    """Run extended, multi-angle web research on a topic and return a structured, cited report.

    Searches several focused angles (overview, latest developments, key data, trade-offs, expert
    analysis, outlook), aggregates the findings, then synthesizes a clean markdown briefing with
    sections: Overview, Key findings, Details, Sources, Open questions. Built for longer/background
    runs where you want a thorough briefing rather than a single quick answer. `angles` (1-6)
    controls breadth. Stays headless — no browser window. Never invents sources.
    """
    topic = (topic or "").strip()
    if not topic:
        return "Provide a research topic."
    n = max(1, min(int(angles or 4), len(_RESEARCH_FACETS)))
    blocks: list[str] = []
    for facet in _RESEARCH_FACETS[:n]:
        q = f"{topic} — {facet}"
        try:
            res = _host._run_web_search(q, max_results=4, queue_kb=False)
        except Exception as exc:  # noqa: BLE001
            res = f"(search failed: {exc})"
        blocks.append(f"### Angle: {facet}\n_Query: {q}_\n{res}")
    corpus = "\n\n".join(blocks)[:9000]
    try:
        llm = _host.make_chat_ollama(
            _host.CFG.get("model", _host.DEFAULT_MODEL), temperature=0.2, num_predict=1200
        )
        prompt = (
            f"Synthesize the research notes below into a clear, structured markdown report on "
            f"'{topic}'. Use exactly these sections: '## Overview', '## Key findings' (bullets), "
            f"'## Details', '## Sources' (list the URLs found in the notes), '## Open questions'. "
            f"Be factual and concise; do NOT invent sources or facts not present in the notes.\n\n"
            f"Research notes:\n{corpus}"
        )
        report = _host.coerce_message_text(llm.invoke([HumanMessage(content=prompt)]).content)
    except Exception as exc:  # noqa: BLE001
        report = f"# Research: {topic}\n\n(Synthesis model unavailable: {exc})\n\n{corpus}"
    return report or f"No findings for '{topic}'."


_BATCH_TYPE_FOLDERS = {
    "Documents": {".pdf", ".doc", ".docx", ".odt", ".txt", ".md", ".rtf", ".tex"},
    "Spreadsheets": {".xls", ".xlsx", ".ods", ".csv"},
    "Images": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".tiff"},
    "Audio": {".mp3", ".wav", ".flac", ".ogg", ".m4a"},
    "Video": {".mp4", ".mkv", ".mov", ".avi", ".webm"},
    "Archives": {".zip", ".tar", ".gz", ".bz2", ".7z", ".rar"},
    "Code": {".py", ".js", ".ts", ".sh", ".json", ".yaml", ".yml", ".html", ".css"},
}


@tool
def batch_file_operations(
    operation: str,
    folder: str,
    pattern: str = "*",
    destination: str = "",
    dry_run: bool = True,
) -> str:
    """Perform bulk file operations across a folder with safety checks and a progress report.

    operation: 'list' | 'organize_by_type' | 'move' | 'copy' | 'rename_sequential' | 'delete_empty'.
    folder: source directory (must be inside your home folder). pattern: glob like '*.pdf' (default '*').
    destination: target directory for 'move'/'copy'. dry_run: TRUE by default — previews the plan
    without changing anything; set dry_run=false to actually apply. Returns a clear before/after
    summary with counts. Safe by design: never touches paths outside home and skips on errors.
    """
    op = (operation or "").strip().lower()
    src = _host._resolve_path(folder)
    if not src.exists() or not src.is_dir():
        return f"Folder not found: {src}"
    if not _host._path_in_home(src):
        return f"For safety, batch operations are limited to your home folder. ({src})"

    try:
        matches = sorted(p for p in src.glob(pattern) if p.is_file())
    except Exception as exc:  # noqa: BLE001
        return f"Bad pattern '{pattern}': {exc}"

    tag = "PREVIEW (dry run)" if dry_run else "APPLIED"
    if op == "list":
        lines = [f"{p.name}  ·  {p.stat().st_size // 1024} KB" for p in matches[:200]]
        return f"**{len(matches)} file(s)** matching `{pattern}` in `{src}`:\n" + "\n".join(lines or ["(none)"])

    done = 0
    skipped = 0
    detail: list[str] = []

    def _safe_dest(target_dir: Path, name: str) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / name
        i = 1
        while dest.exists():
            dest = target_dir / f"{Path(name).stem}_{i}{Path(name).suffix}"
            i += 1
        return dest

    try:
        if op == "organize_by_type":
            for p in matches:
                cat = next((c for c, exts in _BATCH_TYPE_FOLDERS.items() if p.suffix.lower() in exts), "Other")
                dest = _safe_dest(src / cat, p.name)
                detail.append(f"{p.name} → {cat}/")
                if not dry_run:
                    shutil.move(str(p), str(dest))
                done += 1
        elif op in {"move", "copy"}:
            if not destination:
                return f"'{op}' needs a destination folder."
            dst_dir = _host._resolve_path(destination)
            if not _host._path_in_home(dst_dir):
                return f"Destination must be inside your home folder. ({dst_dir})"
            for p in matches:
                dest = _safe_dest(dst_dir, p.name)
                detail.append(f"{p.name} → {dst_dir.name}/")
                if not dry_run:
                    (shutil.copy2 if op == "copy" else shutil.move)(str(p), str(dest))
                done += 1
        elif op == "rename_sequential":
            prefix = src.name.replace(" ", "_")
            for i, p in enumerate(matches, 1):
                new_name = f"{prefix}_{i:04d}{p.suffix.lower()}"
                detail.append(f"{p.name} → {new_name}")
                if not dry_run:
                    p.rename(_safe_dest(src, new_name))
                done += 1
        elif op == "delete_empty":
            for d in sorted((q for q in src.rglob("*") if q.is_dir()), reverse=True):
                try:
                    if not any(d.iterdir()):
                        detail.append(f"remove empty dir {d.relative_to(src)}/")
                        if not dry_run:
                            d.rmdir()
                        done += 1
                except OSError:
                    skipped += 1
        else:
            return (
                "Unknown operation. Use: list, organize_by_type, move, copy, "
                "rename_sequential, delete_empty."
            )
    except Exception as exc:  # noqa: BLE001
        return f"Batch '{op}' stopped after {done} item(s): {exc}"

    head = f"**Batch {op}** — {tag}\nFolder: `{src}` · pattern `{pattern}`\n"
    head += f"Processed **{done}**" + (f", skipped {skipped}" if skipped else "") + ".\n\n"
    preview = "\n".join(f"- {d}" for d in detail[:60])
    if len(detail) > 60:
        preview += f"\n- … and {len(detail) - 60} more"
    if dry_run and done:
        preview += "\n\n_Set dry_run=false to apply these changes._"
    return head + (preview or "(nothing matched)")


@tool
def long_running_command(command: str, timeout_sec: int = 1800, workdir: str = "") -> str:
    """Run a long-running shell command (builds, tests, data jobs) with an extended timeout.

    Use for processes that exceed the normal terminal limit — e.g. `npm run build`, `pytest`,
    large `rsync`/data jobs. Honors the same safety gate as the terminal tool (dangerous commands
    still require approval). `timeout_sec` is clamped to 60–7200s. `workdir` (optional) runs the
    command in that directory. Returns exit code plus combined stdout/stderr (tail).
    """
    command = (command or "").strip()
    if not command:
        return "No command provided."
    risk, reasons = _host.assess_command_risk(command)
    if risk == "dangerous" and not _host.PrimusSession.allow_dangerous_once:
        return _host.suggest_command(command, risk="dangerous", reasons=reasons)
    cwd = None
    if workdir:
        cwd = _host._resolve_path(workdir)
        if not cwd.exists() or not _host._path_in_home(cwd):
            return f"Working directory not found or outside home: {cwd}"
    timeout = max(60, min(int(timeout_sec or 1800), 7200))
    header = f"▶ **Long-running** (timeout {timeout}s):\n```bash\n{command}\n```\n\n"
    return header + _host._execute_shell_core(command, cwd=cwd, timeout=timeout)


# ---------------------------------------------------------------------------
# Autonomous project / application builders.
#
# These drive the Forge model to emit a strict JSON file manifest, then write the
# files to disk via the EXISTING file tools (home-confined, no '..' traversal).
# Usable by the main chat AND background agents, so a full project can be built
# autonomously (even overnight via scheduled agents). No system prompts touched.
# ---------------------------------------------------------------------------

PROJECTS_ROOT = _host.HOME / "Projects"

# Lightweight best-practice nudges per stack keyword (biases generation, never hard-codes output).
_STACK_HINTS: dict[str, str] = {
    "fastapi": (
        "Python + FastAPI: use app/ package, main.py with FastAPI() app, routers/, models/ "
        "(pydantic), services/, requirements.txt (fastapi, uvicorn[standard], pydantic), a .env.example, "
        "and `uvicorn app.main:app --reload` run instructions. Include Dockerfile."
    ),
    "flask": (
        "Python + Flask: app factory pattern (create_app), blueprints/, templates/, static/, "
        "requirements.txt (flask, python-dotenv), config.py, and `flask run` instructions."
    ),
    "django": (
        "Python + Django: standard manage.py layout, settings split, one starter app, "
        "requirements.txt (django), and migration/run instructions."
    ),
    "react": (
        "React + TypeScript (Vite): package.json (react, react-dom, vite, typescript, @types), "
        "tsconfig.json, vite.config.ts, index.html, src/main.tsx, src/App.tsx, components/, "
        "and `npm install && npm run dev` instructions."
    ),
    "next": (
        "Next.js + TypeScript (app router): package.json (next, react, react-dom, typescript), "
        "tsconfig.json, next.config.js, app/layout.tsx, app/page.tsx, components/, "
        "and `npm install && npm run dev` instructions."
    ),
    "cli": (
        "Python CLI: src/ package with a main entry, argparse or click, pyproject.toml with a "
        "console_scripts entry point, requirements.txt, and usage examples in the README."
    ),
    "node": (
        "Node.js: package.json with scripts, src/index.js (or .ts), .gitignore, and run instructions."
    ),
    "python": (
        "Python: src/ package layout, pyproject.toml (or requirements.txt), a clear entry point, "
        "type hints, docstrings, and a tests/ folder with a couple of pytest stubs."
    ),
}


def _stack_hint(tech_stack: str) -> str:
    blob = (tech_stack or "").lower()
    hints = [h for key, h in _STACK_HINTS.items() if key in blob]
    if not hints:  # default to general Python guidance
        hints = [_STACK_HINTS["python"]]
    return "  ".join(hints[:3])


def _forge_code_llm(num_predict: int = 4096) -> Any:
    """Forge model tuned for code generation (low temp, larger output budget)."""
    model_name = _host.CFG.get("forge_model", _host.FORGE_MODEL)
    try:
        _host.enforce_sequential_models(model_name, label="Coding")
    except Exception:  # noqa: BLE001
        pass
    return _host.make_chat_ollama(
        model_name,
        temperature=float(_host.CFG.get("forge_temperature", 0.2)),
        num_predict=int(num_predict),
        fast=True,
    )


def _extract_json_blob(text: str) -> Any:
    """Pull the first JSON object/array out of an LLM reply (tolerates code fences + preamble)."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:  # noqa: BLE001
        pass
    # Fall back to locating the outermost {...} or [...] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:  # noqa: BLE001
                continue
    return None


def _normalize_manifest(parsed: Any) -> list[dict[str, str]]:
    """Coerce a parsed manifest into a clean [{path, content}] list."""
    files = parsed.get("files") if isinstance(parsed, dict) else parsed
    out: list[dict[str, str]] = []
    if not isinstance(files, list):
        return out
    for item in files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("filename") or "").strip()
        content = item.get("content")
        if isinstance(content, list):
            content = "\n".join(str(c) for c in content)
        if path and content is not None:
            out.append({"path": path, "content": str(content)})
    return out


def _safe_rel_path(rel: str) -> Optional[str]:
    """Reject absolute paths and '..' traversal; return a clean POSIX relative path or None."""
    rel = (rel or "").strip().lstrip("/").replace("\\", "/")
    if not rel:
        return None
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts) or not parts:
        return None
    return "/".join(parts)


def _forge_generate_manifest(prompt: str, *, num_predict: int = 4096) -> tuple[list[dict[str, str]], str]:
    """Ask Forge for a JSON file manifest. Returns (files, raw_note)."""
    if not _host.HAS_AI_STACK:
        return [], "AI stack not installed — project builders need Ollama + LangChain."
    instruction = (
        "You are an expert software engineer. Output ONLY a single JSON object — no prose, no code "
        'fences — of the form {"files": [{"path": "relative/path", "content": "FULL FILE CONTENTS"}], '
        '"post_setup": ["shell command", ...], "notes": "short summary"}. '
        "Paths are RELATIVE to the project root (never absolute, never use '..'). Include every file "
        "needed for a clean, runnable project: config/build files, source, and a README.md with setup "
        "and run instructions. Write complete, idiomatic, well-commented code following clean "
        "architecture. Do not truncate files.\n\n" + prompt
    )
    try:
        llm = _forge_code_llm(num_predict)
        raw = _host.coerce_message_text(llm.invoke([HumanMessage(content=instruction)]).content)
    except Exception as exc:  # noqa: BLE001
        return [], f"Generation failed: {exc}"
    parsed = _extract_json_blob(raw)
    files = _normalize_manifest(parsed)
    if not files:
        return [], "Model did not return a usable JSON file manifest."
    note = ""
    if isinstance(parsed, dict):
        post = parsed.get("post_setup") or []
        notes = str(parsed.get("notes") or "").strip()
        if notes:
            note += notes + "\n"
        if isinstance(post, list) and post:
            note += "Suggested setup:\n" + "\n".join(f"  $ {c}" for c in post[:12])
    return files, note


def _write_project_files(root: Path, files: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """Write manifest files under `root`. Returns (written_rel_paths, skipped_messages)."""
    written: list[str] = []
    skipped: list[str] = []
    for f in files:
        rel = _safe_rel_path(f.get("path", ""))
        if not rel:
            skipped.append(f"unsafe path skipped: {f.get('path')!r}")
            continue
        target, err = _host._guard_path(str(root / rel))
        if err:
            skipped.append(f"{rel}: {err}")
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.get("content", ""), encoding="utf-8")
            written.append(rel)
        except OSError as exc:
            skipped.append(f"{rel}: {exc}")
    return written, skipped


def _project_tree(written: list[str]) -> str:
    return "\n".join(f"  {p}" for p in sorted(written)[:80])


def _verify_python_files(root: Path, written: list[str]) -> str:
    """Compile-check any written .py files and summarize. Best-effort, never raises."""
    py = [p for p in written if p.endswith(".py")]
    if not py:
        return ""
    import py_compile

    issues: list[str] = []
    for rel in py[:40]:
        try:
            py_compile.compile(str(root / rel), doraise=True)
        except py_compile.PyCompileError as exc:
            issues.append(f"  ✗ {rel}: {str(exc).strip()[:200]}")
        except Exception:  # noqa: BLE001
            pass
    if not issues:
        return f"✓ All {len(py)} Python file(s) compile cleanly."
    return f"⚠ {len(issues)} Python file(s) have syntax errors:\n" + "\n".join(issues[:15])


@tool
def create_full_project(
    project_name: str, description: str, tech_stack: str = "Python", features: str = ""
) -> str:
    """Create a complete, well-structured project folder for a chosen tech stack.

    Generates a proper directory layout, all config/build files (requirements.txt / pyproject.toml /
    package.json / Dockerfile as appropriate), core source files, and a README with setup + run
    instructions — following modern best practices for the stack. `tech_stack` examples:
    'Python + FastAPI', 'React + TypeScript', 'Next.js', 'Python CLI', 'Flask'. `features` is a
    comma-separated list of key features. Writes everything under ~/Projects/<name> and returns the
    file tree plus setup steps. Build on this for real, runnable projects.
    """
    name = (project_name or "").strip()
    if not name:
        return "Provide a project_name."
    safe = _safe_filename(name).replace(" ", "_") or "project"
    root = PROJECTS_ROOT / safe
    prompt = (
        f"Create a complete {tech_stack} project named '{name}'.\n"
        f"Description: {description}\n"
        f"Key features: {features or '(use sensible defaults for the description)'}\n\n"
        f"Stack guidance: {_stack_hint(tech_stack)}\n"
        "Produce a production-quality starting point: correct config/build files, a sensible "
        "directory structure, working core source files with comments, and a thorough README.md."
    )
    files, note = _forge_generate_manifest(prompt, num_predict=int(_host.CFG.get("forge_num_predict", 4096)))
    if not files:
        return f"⚠ Could not generate '{name}': {note}"
    if not any(p["path"].lower().endswith("readme.md") for p in files):
        files.append({
            "path": "README.md",
            "content": f"# {name}\n\n{description}\n\n## Tech stack\n{tech_stack}\n\n"
                       f"## Features\n{features or '-'}\n",
        })
    written, skipped = _write_project_files(root, files)
    if not written:
        return f"⚠ Nothing written for '{name}'. " + ("; ".join(skipped[:5]) or note)
    out = [f"✓ Created project **{name}** at `{root}` ({len(written)} files)."]
    out.append("```\n" + _project_tree(written) + "\n```")
    verify = _verify_python_files(root, written)
    if verify:
        out.append(verify)
    if skipped:
        out.append("Skipped: " + "; ".join(skipped[:5]))
    if note:
        out.append(note)
    return "\n\n".join(out)


@tool
def build_application_from_spec(app_name: str, detailed_spec: str, tech_stack: str = "Python") -> str:
    """Build an entire working application from a natural-language spec.

    Turns `detailed_spec` (a full description of what the app should do) into a wired, multi-file
    application in `tech_stack`, with proper structure, basic error handling, and a README. Writes
    under ~/Projects/<app_name>, then compile-checks any Python so issues can be fixed in a follow-up.
    Use for building real, usable tools and small-to-medium apps.
    """
    name = (app_name or "").strip()
    if not name:
        return "Provide an app_name."
    if not (detailed_spec or "").strip():
        return "Provide a detailed_spec describing what the app should do."
    safe = _safe_filename(name).replace(" ", "_") or "app"
    root = PROJECTS_ROOT / safe
    prompt = (
        f"Build a complete, working application named '{name}' in {tech_stack}.\n\n"
        f"SPECIFICATION:\n{detailed_spec}\n\n"
        f"Stack guidance: {_stack_hint(tech_stack)}\n"
        "Requirements: split logic across appropriately-named files and wire them together so the "
        "app actually runs; add input validation and basic error handling; include config/build "
        "files and a README.md with exact setup and run commands. Prefer a clean, layered structure."
    )
    files, note = _forge_generate_manifest(prompt, num_predict=int(_host.CFG.get("forge_num_predict", 4096)))
    if not files:
        return f"⚠ Could not build '{name}': {note}"
    written, skipped = _write_project_files(root, files)
    if not written:
        return f"⚠ Nothing written for '{name}'. " + ("; ".join(skipped[:5]) or note)
    out = [f"✓ Built application **{name}** at `{root}` ({len(written)} files)."]
    out.append("```\n" + _project_tree(written) + "\n```")
    verify = _verify_python_files(root, written)
    if verify:
        out.append(verify)
        if verify.startswith("⚠"):
            out.append("_Tip: run `iterative_code_improver` on this path with goal 'fix errors'._")
    if skipped:
        out.append("Skipped: " + "; ".join(skipped[:5]))
    if note:
        out.append(note)
    return "\n\n".join(out)


_CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".c", ".cpp",
             ".h", ".hpp", ".sh", ".css", ".html", ".sql"}


@tool
def iterative_code_improver(code_or_project_path: str, goals: str = "readability, correctness") -> str:
    """Analyze existing code and make targeted improvements across one file or a whole project.

    `code_or_project_path` is a file or directory under home. `goals` describes what to improve
    (e.g. 'performance', 'readability', 'security', 'add tests', 'fix errors', 'features'). For a
    directory, it improves the main code files (bounded). Originals are backed up to `*.bak` before
    being overwritten. Returns a per-file summary of what changed. Safe to call in a loop (e.g. from
    a background agent) to refine code over multiple steps.
    """
    target, err = _host._guard_path(code_or_project_path, must_exist=True)
    if err:
        return err
    if target.is_dir():
        candidates = sorted(
            p for p in target.rglob("*")
            if p.is_file() and p.suffix.lower() in _CODE_EXTS
            and ".bak" not in p.name and "node_modules" not in p.parts
            and ".git" not in p.parts and "__pycache__" not in p.parts
        )[:8]
    elif target.is_file():
        candidates = [target]
    else:
        return f"Not a file or directory: {target}"
    if not candidates:
        return f"No code files found to improve under {target}."

    results: list[str] = []
    for fpath in candidates:
        try:
            original = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            results.append(f"  ✗ {fpath.name}: {exc}")
            continue
        if len(original) > 24000:
            results.append(f"  • {fpath.name}: skipped (too large to refine safely)")
            continue
        prompt = (
            f"Improve the following {fpath.suffix.lstrip('.') or 'code'} file. Goals: {goals}.\n"
            "Return ONLY the complete improved file contents (no prose, no code fences). Preserve "
            "behavior unless a goal requires changing it; keep it runnable and well-commented.\n\n"
            f"FILE: {fpath.name}\n-----\n{original}"
        )
        try:
            llm = _forge_code_llm(int(_host.CFG.get("forge_num_predict", 4096)))
            improved = _host.coerce_message_text(llm.invoke([HumanMessage(content=prompt)]).content)
        except Exception as exc:  # noqa: BLE001
            results.append(f"  ✗ {fpath.name}: model error: {exc}")
            continue
        improved = re.sub(r"^```[\w]*\n?|```$", "", improved.strip(), flags=re.MULTILINE).strip()
        if not improved or improved == original.strip():
            results.append(f"  • {fpath.name}: no change suggested")
            continue
        try:
            fpath.with_suffix(fpath.suffix + ".bak").write_text(original, encoding="utf-8")
            fpath.write_text(improved + "\n", encoding="utf-8")
            delta = len(improved.splitlines()) - len(original.splitlines())
            results.append(f"  ✓ {fpath.name}: updated ({delta:+d} lines; backup .bak saved)")
        except OSError as exc:
            results.append(f"  ✗ {fpath.name}: write failed: {exc}")

    head = f"**Code improvement** — `{target}`\nGoals: {goals}\n\n"
    return head + "\n".join(results)


# ---------------------------------------------------------------------------
# Autonomous web-document acquisition → knowledge base.
# Headless download (PDF or webpage→PDF) + ingest via the EXISTING pipeline.
# Usable by the main chat and background agents.
# ---------------------------------------------------------------------------

_BLOCKED_HOST_RE = re.compile(
    r"^(localhost|127\.|10\.|192\.168\.|169\.254\.|0\.0\.0\.0|::1|metadata\.google)",
    re.IGNORECASE,
)

# Placeholder/example hosts that agents hallucinate instead of using a real search result. These are
# never valid ingestion/download targets — reject them so we don't "download example.com" and force a
# real web_search first. (RFC 2606 reserves example.*; the rest are common LLM filler.)
_PLACEHOLDER_HOST_RE = re.compile(
    r"^(?:www\.)?(?:example\.(?:com|org|net|edu)|examples?\.\w+|placeholder[\w.-]*|"
    r"your-?(?:site|domain|url|website)[\w.-]*|yourdomain[\w.-]*|test\.(?:com|example)|"
    r"(?:some|the)-?(?:site|url|page)\.\w+|domain\.(?:com|tld)|url\.here|link\.here)$",
    re.IGNORECASE,
)


def _is_placeholder_url(url: str) -> bool:
    """True if `url`'s host is an obvious placeholder/example (not a real, ingestable source)."""
    try:
        host = (urllib.parse.urlparse((url or "").strip()).hostname or "").strip()
    except Exception:  # noqa: BLE001
        return False
    return bool(host) and bool(_PLACEHOLDER_HOST_RE.match(host))


_PLACEHOLDER_URL_MSG = (
    "⚠ That looks like a placeholder/example URL, not a real source. Run `web_search` or "
    "`deep_web_search` first, then pass the ACTUAL result URL here (never example.com)."
)


def _validate_web_url(url: str) -> tuple[bool, str]:
    """Allow only public http/https URLs; block local/private/metadata hosts."""
    url = (url or "").strip()
    if not url:
        return False, "No URL provided."
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not parse URL: {exc}"
    if parsed.scheme not in ("http", "https"):
        return False, "Only http/https URLs are allowed."
    host = (parsed.hostname or "").strip()
    if not host or _BLOCKED_HOST_RE.match(host) or host.endswith(".local"):
        return False, f"Blocked or invalid host: {host or '(none)'}"
    return True, ""


def _download_name_from_url(url: str, *, suffix: str) -> str:
    """Derive a safe filename (with the given suffix) from a URL."""
    base = os.path.basename(urllib.parse.urlparse(url).path) or "document"
    base = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", base)  # drop any existing extension
    base = _safe_filename(base) or "document"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{stamp}{suffix}"


def _selenium_print_pdf(url: str, dest: Path, *, timeout: int = 30) -> tuple[bool, str]:
    """Render a page in headless Firefox (Chrome fallback) and write it as a PDF. Never raises."""
    # --- Firefox print_page() (Selenium 4) ---
    try:
        from selenium import webdriver  # type: ignore
        from selenium.webdriver.firefox.options import Options as FxOptions  # type: ignore
        from selenium.webdriver.common.print_page_options import PrintOptions  # type: ignore

        driver = None
        try:
            opts = FxOptions()
            opts.add_argument("-headless")
            driver = webdriver.Firefox(options=opts)
            driver.set_page_load_timeout(timeout)
            driver.get(url)
            time.sleep(1.0)  # let late JS/layout settle
            po = PrintOptions()
            po.page_ranges = ["1-"]
            b64 = driver.print_page(po)
            dest.write_bytes(base64.b64decode(b64))
            if dest.exists() and dest.stat().st_size > 800:
                return True, "firefox"
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("Firefox print_page failed: %s", exc)
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Firefox PDF path unavailable: %s", exc)

    # --- Chrome/Chromium CDP Page.printToPDF fallback ---
    try:
        from selenium import webdriver  # type: ignore
        from selenium.webdriver.chrome.options import Options as ChOptions  # type: ignore

        driver = None
        try:
            opts = ChOptions()
            for arg in ("--headless=new", "--disable-gpu", "--no-sandbox", "--window-size=1280,1696"):
                opts.add_argument(arg)
            driver = webdriver.Chrome(options=opts)
            driver.set_page_load_timeout(timeout)
            driver.get(url)
            time.sleep(1.0)
            res = driver.execute_cdp_cmd("Page.printToPDF", {"printBackground": True})
            dest.write_bytes(base64.b64decode(res["data"]))
            if dest.exists() and dest.stat().st_size > 800:
                return True, "chrome"
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("Chrome printToPDF failed: %s", exc)
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("Chrome PDF path unavailable: %s", exc)
    return False, ""


def _url_looks_like_pdf(url: str) -> bool:
    """Best-effort check whether a URL points at a PDF (extension or content-type)."""
    if urllib.parse.urlparse(url).path.lower().endswith(".pdf"):
        return True
    try:
        req = urllib.request.Request(
            url, method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Primus/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return "application/pdf" in (resp.headers.get("Content-Type") or "").lower()
    except Exception:  # noqa: BLE001
        return False


@tool
def download_webpage_as_pdf(url: str, filename: str = "") -> str:
    """Load a web page in a headless browser (incl. JS-rendered content) and save it as a clean PDF.

    Saves to ~/.primus/knowledge/downloads/ and returns the local file path. Use this to capture
    a full webpage (articles, docs, dashboards) as a durable PDF. Only http/https URLs are allowed.
    """
    if _is_placeholder_url(url):
        return _PLACEHOLDER_URL_MSG
    ok, reason = _validate_web_url(url)
    if not ok:
        return f"⚠ {reason}"
    _host.KB_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe_filename(filename) if filename else _download_name_from_url(url, suffix=".pdf")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    dest = _host.KB_DOWNLOADS_DIR / name
    success, engine = _selenium_print_pdf(url, dest)
    if success:
        return f"✓ Saved webpage as PDF ({engine}): `{dest}` ({dest.stat().st_size // 1024} KB)"
    return (
        "⚠ Could not render the page to PDF — a headless browser (Selenium + Firefox or Chrome) "
        "isn't available. Install with `uv pip install selenium` and a browser/driver, or use "
        "`ingest_web_document` which falls back to saving the page's text."
    )


@tool
def download_pdf_from_url(url: str, filename: str = "") -> str:
    """Download a hosted PDF file directly from a URL, validating that it is really a PDF.

    Saves to ~/.primus/knowledge/downloads/ and returns the local file path. Only http/https URLs
    are allowed. Use this when the URL points straight at a `.pdf` document.
    """
    if _is_placeholder_url(url):
        return _PLACEHOLDER_URL_MSG
    ok, reason = _validate_web_url(url)
    if not ok:
        return f"⚠ {reason}"
    _host.KB_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe_filename(filename) if filename else _download_name_from_url(url, suffix=".pdf")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    dest = _host.KB_DOWNLOADS_DIR / name
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Primus/1.0"},
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            data = resp.read(40 * 1024 * 1024)  # cap at 40 MB
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Download failed: {exc}"
    is_pdf = data[:5] == b"%PDF-" or "application/pdf" in ctype
    if not is_pdf:
        return (
            f"⚠ That URL didn't return a PDF (content-type `{ctype or 'unknown'}`). "
            "Use `download_webpage_as_pdf` for regular web pages."
        )
    try:
        dest.write_bytes(data)
    except OSError as exc:
        return f"⚠ Could not save file: {exc}"
    return f"✓ Downloaded PDF: `{dest}` ({len(data) // 1024} KB)"


def _project_download_tags(project: str) -> tuple[str, list[str]]:
    """Resolve a project chat (by id or name) → (collection_project_name, extra_tags)."""
    proj = (project or "").strip()
    if not proj or proj == _host.GENERAL_CHAT_ID:
        return "", []
    match = None
    for p in _host.ChatProjects.list():
        if p.get("id") == proj or p.get("name", "").lower() == proj.lower():
            match = p
            break
    if match and match.get("id") != _host.GENERAL_CHAT_ID:
        return match.get("name", proj), [f"project:{match['id']}", *(match.get("tags") or [])]
    return proj, [proj.lower()]


@tool
def ingest_web_document(url: str, category: str = "", project: str = "", allow_bypass: bool = False) -> str:
    """Find, download, and ingest a web document into the knowledge base for future retrieval.

    If the URL is a PDF it is downloaded directly; if it's a web page it is rendered to PDF with a
    headless browser (falling back to saving the page's structured extracted text — preserving
    tables, code blocks and a heading outline — if no browser is available). The file is then
    ingested through the normal pipeline (chunk → embed → vector store). Optional `category` (a KB
    category label) and `project` (a project-chat name or id) tag the document; set
    `allow_bypass=True` only when the operator explicitly asks to get past a paywall/blocked page.
    Returns a summary of what was downloaded and ingested.
    """
    if _is_placeholder_url(url):
        return _PLACEHOLDER_URL_MSG
    ok, reason = _validate_web_url(url)
    if not ok:
        return f"⚠ {reason}"
    _host.KB_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Acquire the document as a local file.
    saved_path = ""
    note = ""
    if _url_looks_like_pdf(url):
        res = download_pdf_from_url.func(url) if hasattr(download_pdf_from_url, "func") else download_pdf_from_url(url)
        if res.startswith("✓"):
            saved_path = res.split("`", 2)[1]
            note = "Downloaded hosted PDF."
    if not saved_path:
        # Try webpage → PDF.
        name = _download_name_from_url(url, suffix=".pdf")
        dest = _host.KB_DOWNLOADS_DIR / name
        success, engine = _selenium_print_pdf(url, dest)
        if success:
            saved_path = str(dest)
            note = f"Rendered webpage to PDF ({engine})."
        else:
            # Fallback: save structured extracted text (tables/code/outline preserved) so ingestion
            # still works without a browser.
            text = _read_article(url, structured=True, allow_bypass=allow_bypass)
            if text and len(text.strip()) > 80:
                md = _host.KB_DOWNLOADS_DIR / _download_name_from_url(url, suffix=".md")
                md.write_text(f"# Source: {url}\n\n{text}", encoding="utf-8")
                saved_path = str(md)
                note = "Saved page text (headless PDF unavailable)."
    if not saved_path:
        return f"⚠ Couldn't acquire a document from {url} (no PDF, no browser, no readable text)."

    # 2) Ingest via the existing upload pipeline.
    category_label = category if category in _host.KB_CATEGORY_LABELS else "Learned / New Information"
    proj_name, extra_tags = _project_download_tags(project)
    is_projects = _host.KB_CATEGORY_TO_COLLECTION.get(category_label) == "projects"
    try:
        summary = _host.ingest_uploaded_files(
            [saved_path], category_label, proj_name if is_projects else "",
            extra_tags=["web", "downloaded", *extra_tags],
        )
    except Exception as exc:  # noqa: BLE001
        return f"Saved `{saved_path}` but ingestion failed: {exc}"
    return f"✓ {note} Saved `{saved_path}`.\n\n{summary}"


@tool
def recover_ollama_gpu(max_attempts: int = 8, model: str = "qwen2.5:7b") -> str:
    """Self-heal local Ollama GPU acceleration on this AMD Radeon 890M (gfx1150) laptop.

    The 890M's ROCm path is flaky on Linux and Ollama silently falls back to CPU. This aggressively
    restarts `ollama serve` with the Vulkan settings that work best on this iGPU
    (OLLAMA_VULKAN=1, GGML_VK_VISIBLE_DEVICES=0, flash attention, 30m keep-alive), force-loads a
    model to trigger GPU layer placement, and checks `ollama ps`. It retries up to `max_attempts`
    times (1–10; the Vulkan path often needs several tries) and stops as soon as the PROCESSOR
    column shows GPU. Returns the full attempt log plus the final `ollama ps`. Local-laptop oriented;
    safe to call from background/scheduled agents.
    """
    final = ""
    for log_text, _done, _success in _host._gpu.recover_ollama_gpu_stream(int(max_attempts), str(model)):
        final = log_text
    return final or "GPU recovery produced no output."


# ===========================================================================
# Gmail integration (official Gmail API, OAuth2)
# ---------------------------------------------------------------------------
# Read/search/list/read-full/draft/send over the operator's Gmail. Credentials + token live under
# ~/.primus (never in the repo). Google libraries are optional — every tool degrades gracefully
# with clear install/setup instructions if they're missing or auth hasn't been done. Sending is
# gated by the same safety model as edit_docx/edit_xlsx: in Suggest mode it only previews.
# ===========================================================================

# Least-privilege scopes: readonly (list/get/search) + compose (drafts) + send (messages.send).
_GMAIL_SCOPES = [
    "https://mail.google.com/",                          # full mailbox access (read, modify, send)
    "https://www.googleapis.com/auth/gmail.modify",      # read/label/modify without full-delete
    "https://www.googleapis.com/auth/gmail.compose",     # create drafts / compose
    "https://www.googleapis.com/auth/gmail.readonly",    # read-only access (metadata + bodies)
]
_GMAIL_CREDS_PATH = _host.APP_DIR / "gmail_credentials.json"  # OAuth client secret (from GCP console)
_GMAIL_TOKEN_PATH = _host.APP_DIR / "gmail_token.json"        # cached user token (auto-created)
_GMAIL_ATTACHMENTS_DIR = _host.APP_DIR / "gmail_attachments"  # ~/.primus/gmail_attachments/

_GMAIL_MISSING_LIBS = (
    "⚠ Gmail libraries not installed. Run:\n"
    "  uv pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
)

# Works in EVERY environment (desktop, headless, Docker, SSH): no local callback server is required.
# The user opens the URL in ANY browser (even on another device), approves, then pastes the code
# back. This is the modern OAuth flow — NEVER tell the user to use App Passwords / "less secure
# apps" / IMAP-SMTP settings; those are obsolete and don't apply here.
_GMAIL_MANUAL_TOKEN_HELP = (
    "This flow works anywhere — desktop, headless, Docker, or over SSH:\n"
    "  1. Run `gmail_get_auth_url` (or `gmail_auth`) to get a Google sign-in link.\n"
    "  2. Open it in ANY browser, on any device, and approve access.\n"
    "  3. You'll land on `http://localhost/?code=XXXX…` (a browser 'can't connect' page is normal — "
    "no server needs to be running).\n"
    "  4. Copy the `code` value and run `gmail_auth_code <code>` (you can paste the whole URL).\n"
    f"Client secret lives at `{_GMAIL_CREDS_PATH}`; the token is cached at `{_GMAIL_TOKEN_PATH}`."
)


def _gmail_b64decode(data: str) -> str:
    """Decode a Gmail API base64url body part to text (never raises)."""
    try:
        return base64.urlsafe_b64decode((data or "").encode("utf-8")).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _gmail_build_raw(to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
    """Build a base64url-encoded RFC-2822 message for the Gmail API from plain fields."""
    from email.mime.text import MIMEText

    message = MIMEText(body or "", _charset="utf-8")
    message["to"] = to or ""
    message["subject"] = subject or ""
    if cc:
        message["cc"] = cc
    if bcc:
        message["bcc"] = bcc
    return base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")


def _gmail_extract_parts(payload: dict) -> tuple[str, list[dict]]:
    """Walk a Gmail message payload → (best-effort text body, attachment metadata list).

    Prefers text/plain; falls back to stripped text/html. Attachment entries carry the
    attachmentId needed to download the bytes separately.
    """
    text_chunks: list[str] = []
    html_chunks: list[str] = []
    attachments: list[dict] = []

    def walk(part: dict) -> None:
        mime = part.get("mimeType", "") or ""
        body = part.get("body", {}) or {}
        filename = part.get("filename", "") or ""
        if filename and body.get("attachmentId"):
            attachments.append({
                "filename": filename,
                "attachmentId": body["attachmentId"],
                "mimeType": mime,
                "size": int(body.get("size", 0) or 0),
            })
        data = body.get("data")
        if data and mime == "text/plain":
            text_chunks.append(_gmail_b64decode(data))
        elif data and mime == "text/html":
            html_chunks.append(_gmail_b64decode(data))
        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload or {})
    text = "\n".join(c for c in text_chunks if c).strip()
    if not text and html_chunks:
        text = _strip_html_text("\n".join(html_chunks))
    return text, attachments


def _get_gmail_service() -> tuple[Any, str]:
    """Return (gmail_service, "") on success, else (None, user-facing error/instructions).

    Loads the cached token, silently refreshes it if expired, and builds a Gmail API client.
    Handles missing libraries, missing/invalid token, and refresh failures gracefully.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        return None, _GMAIL_MISSING_LIBS

    if not _GMAIL_TOKEN_PATH.exists():
        return None, (
            "⚠ Gmail is NOT authorized yet — there is no live mailbox to read. Fix it now by calling "
            "`ensure_gmail_access` (or `gmail_auth`) — do NOT give manual Google settings/App-Password "
            "steps.\n" + _GMAIL_MANUAL_TOKEN_HELP + "\nIMPORTANT: relay this to the user — do NOT "
            "invent, sample, or guess any emails."
        )
    try:
        creds = Credentials.from_authorized_user_file(str(_GMAIL_TOKEN_PATH), _GMAIL_SCOPES)
    except Exception as exc:  # noqa: BLE001
        return None, (
            f"⚠ Could not load Gmail token ({str(exc)[:120]}). Reconnect with `ensure_gmail_access`. "
            f"Token file: `{_GMAIL_TOKEN_PATH}`. Do NOT fabricate email data — report this to the user."
        )

    # Silent token refresh (Google access tokens expire ~hourly; refresh token is long-lived).
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _GMAIL_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
            _host.log.debug("Gmail token refreshed silently")
        except Exception as exc:  # noqa: BLE001
            _host.log.warning("Gmail token refresh failed: %s", exc)
            return None, (
                f"⚠ Gmail token refresh failed ({str(exc)[:120]}). Reconnect with `ensure_gmail_access`. "
                "Do NOT fabricate email data — report this error to the user."
            )
    if not creds or not creds.valid:
        return None, (
            "⚠ Gmail credentials are invalid or revoked. Reconnect with `ensure_gmail_access` "
            "(never App Passwords / manual settings). Do NOT fabricate email data — report this to the user."
        )

    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return service, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"⚠ Could not build Gmail service: {str(exc)[:140]}"


# --- Fixture mailbox (demo mode) ---------------------------------------------------------
# A fresh clone has no Gmail token, so the inbox tools would only ever show an auth error.
# When `gmail_fixture_mode` is on (default) and no real token exists, the list/read/status
# tools serve examples/gmail/fixture_inbox.json instead — clearly labelled "Demo inbox" —
# so the Inbox UI and agent flows are demoable end-to-end without Google. Draft/send in
# fixture mode are preview-only and always say to connect Gmail. The moment a real
# gmail_token.json exists, every fixture path switches itself off.
_GMAIL_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "gmail" / "fixture_inbox.json"
)


def _gmail_fixture_active() -> bool:
    """True when the demo mailbox should be served: flag on, no real token, fixture present."""
    try:
        return (
            bool(_host.CFG.get("gmail_fixture_mode", True))
            and not _GMAIL_TOKEN_PATH.exists()
            and _GMAIL_FIXTURE_PATH.exists()
        )
    except Exception:  # noqa: BLE001 — never let demo mode break the real path
        return False


def _gmail_fixture_load() -> list[dict]:
    """Load the fixture messages (list of dicts); [] on any problem."""
    try:
        data = json.loads(_GMAIL_FIXTURE_PATH.read_text(encoding="utf-8"))
        msgs = data.get("messages") if isinstance(data, dict) else data
        return [m for m in (msgs or []) if isinstance(m, dict) and str(m.get("id", "")).strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _gmail_fixture_query(messages: list[dict], query: str) -> list[dict]:
    """Tiny Gmail-query subset for the fixture: is:unread, from:, subject:, newer_than:Nd."""
    q = (query or "").strip().lower()
    if not q or q == "all":
        return messages
    out = messages
    if "is:unread" in q:
        out = [m for m in out if m.get("unread")]
    from_m = re.search(r"from:(\S+)", q)
    if from_m:
        out = [m for m in out if from_m.group(1) in str(m.get("from", "")).lower()]
    subj_m = re.search(r"subject:(\S+)", q)
    if subj_m:
        out = [m for m in out if subj_m.group(1) in str(m.get("subject", "")).lower()]
    newer_m = re.search(r"newer_than:(\d+)d", q)
    if newer_m:
        try:
            from email.utils import parsedate_to_datetime

            cutoff = datetime.now().astimezone() - timedelta(days=int(newer_m.group(1)))
            kept = []
            for m in out:
                try:
                    dt = parsedate_to_datetime(str(m.get("date", ""))).astimezone()
                    if dt >= cutoff:
                        kept.append(m)
                except (TypeError, ValueError):
                    kept.append(m)  # unparseable fixture date → keep rather than drop
            out = kept
        except (TypeError, ValueError):
            pass
    return out


# --- Non-blocking Google OAuth (Gmail + Calendar): return the URL to chat immediately, complete
#     the token exchange on a background loopback server, and support manual code entry. ---
_PENDING_OAUTH: dict[str, dict] = {}
_PENDING_OAUTH_LOCK = threading.Lock()


def _set_pending_status(service_key: str, status: str) -> None:
    with _PENDING_OAUTH_LOCK:
        if service_key in _PENDING_OAUTH:
            _PENDING_OAUTH[service_key]["status"] = status


def _google_oauth_config(service_key: str) -> tuple[list[str], Path, str]:
    """(scopes, default token path, human label) for a Google service key ('gmail' | 'calendar')."""
    if service_key == "calendar":
        return _CALENDAR_SCOPES, _CALENDAR_TOKEN_PATH, "Calendar"
    return _GMAIL_SCOPES, _GMAIL_TOKEN_PATH, "Gmail"


def _finish_google_oauth(service_key: str, code: str) -> tuple[bool, str]:
    """Exchange an authorization code for a token and save it (chmod 600). Returns (ok, message).

    Uses the in-session pending flow when present, but — crucially — REBUILDS the flow from the
    client secret + the fixed redirect URI when there's no live pending flow (e.g. the app was
    restarted between issuing the URL and pasting the code). Without this, manual code entry would
    fail with "no pending authorization" and no token would ever be written. Scope checking is
    relaxed so the extra scopes Google grants don't abort the exchange.
    """
    with _PENDING_OAUTH_LOCK:
        pend = _PENDING_OAUTH.get(service_key)
    if pend and pend.get("status") == "done":
        return True, "already authorized"

    scopes, default_token_path, label = _google_oauth_config(service_key)
    token_path = (pend or {}).get("token_path") or default_token_path
    flow = (pend or {}).get("flow")

    # Rebuild the flow if we don't have a live one, so completion works across restarts/sessions.
    if flow is None:
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            return False, "google-auth-oauthlib is not installed"
        if not _GMAIL_CREDS_PATH.exists():
            return False, f"missing OAuth client secret at {_GMAIL_CREDS_PATH}"
        try:
            flow = InstalledAppFlow.from_client_secrets_file(str(_GMAIL_CREDS_PATH), scopes)
            flow.redirect_uri = _google_redirect_uri()
        except Exception as exc:  # noqa: BLE001
            _host.log.exception("%s OAuth: could not rebuild flow for token exchange", label)
            return False, f"could not rebuild OAuth flow: {str(exc)[:120]}"

    # Google routinely grants MORE scopes than requested; without relaxing this, fetch_token raises.
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    try:
        flow.fetch_token(code=code)
        creds = flow.credentials
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        try:
            token_path.chmod(0o600)  # token is a secret — owner-only
        except OSError:
            pass
    except Exception as exc:  # noqa: BLE001
        _host.log.exception("%s OAuth: token exchange failed", label)
        decoded = _gmail_decode_oauth_error(str(exc))
        if decoded:
            return False, decoded
        return False, f"token exchange failed: {str(exc)[:140]}"

    # Verify the token file actually landed on disk — never report success without a real token.
    if not token_path.exists():
        return False, f"token exchange returned no persisted token at {token_path}"

    _set_pending_status(service_key, "done")
    try:
        _conn_note_auth(service_key)  # register with the central Connection Manager (non-fatal)
    except Exception:  # noqa: BLE001
        pass
    _host.log.info("%s OAuth: token saved to %s", label, token_path)
    return True, "authorized"


def _complete_google_oauth_manual(service_key: str, code_or_url: str) -> str:
    """Manual fallback: user pastes the authorization code (or full redirect URL) to finish sign-in."""
    raw = (code_or_url or "").strip()
    label = service_key.title()
    with _PENDING_OAUTH_LOCK:
        pend = _PENDING_OAUTH.get(service_key)
    if pend:
        label = pend.get("label", label)
    if not raw:
        return (f"⚠ Paste the authorization code after the command, e.g. "
                f"`{service_key}_auth_code 4/0Ab...`. You can paste the whole redirect URL too.")
    # Accept a pasted redirect URL and pull the ?code= out of it.
    code = raw
    if "code=" in raw:
        try:
            query = urllib.parse.urlparse(raw).query or raw.split("?", 1)[-1]
            code = urllib.parse.parse_qs(query).get("code", [raw])[0]
        except Exception:  # noqa: BLE001
            pass
    code = code.strip().strip("/")
    if pend and pend.get("status") == "done":
        return f"✓ {label} is already authorized. Try `{service_key}_status`."
    if not pend:
        # No live pending flow (e.g. the app restarted). We can STILL finish: _finish_google_oauth
        # rebuilds the flow from the client secret, so don't hard-fail here.
        _host.log.info("%s manual auth: no in-memory pending flow — rebuilding from client secret", label)
    # Stop the background server so it doesn't compete for the (single-use) code.
    srv = (pend or {}).get("server")
    if srv is not None:
        try:
            srv.server_close()
        except Exception:  # noqa: BLE001
            pass
    ok, msg = _finish_google_oauth(service_key, code)
    if ok:
        return f"✓ {label} authorized! Token saved (owner-only). Try `{service_key}_status`."
    return f"⚠ {label} authorization failed: {msg}. Re-run `{service_key}_auth` for a fresh link."


def _google_redirect_uri() -> str:
    """The redirect URI put in the OAuth request.

    MUST exactly match a redirect URI registered on the OAuth client in Google Cloud, otherwise
    Google returns ``400 invalid_request`` (redirect_uri_mismatch). We standardize on the bare
    loopback ``http://localhost`` (no port, no trailing slash) — the value the user registers on the
    client and the single value the manual code-copy flow relies on. Override via
    ``CFG['gmail_redirect_uri']`` only if you registered a different one (e.g. with a port).
    """
    val = str(_host.CFG.get("gmail_redirect_uri", "")).strip()
    return val or "http://localhost"


def _spawn_browser_open(url: str, service_label: str) -> None:
    """Best-effort, non-blocking browser launch for an OAuth URL.

    Runs in a daemon thread because ``webbrowser.open`` can block on some Linux setups (a ``$BROWSER``
    handler that waits). Off the critical path it can never delay returning the URL to the chat.
    """
    def _open() -> None:
        try:
            import webbrowser
            if webbrowser.open(url, new=1, autoraise=True):
                _host.log.info("%s OAuth: opened a browser", service_label)
            else:
                _host.log.info("%s OAuth: no browser could be opened — use the URL manually", service_label)
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("%s OAuth: webbrowser.open raised: %s", service_label, exc)

    try:
        threading.Thread(target=_open, name="oauth-browser", daemon=True).start()
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("%s OAuth: could not spawn browser thread: %s", service_label, exc)


def _load_google_client_id() -> tuple[str, str, str]:
    """Read the OAuth client-secret JSON → (client_id, auth_uri, error_message).

    Supports both Desktop ('installed') and Web ('web') client types. Returns a clear error string if
    the file is missing/unreadable or doesn't contain a client_id.
    """
    try:
        data = json.loads(_GMAIL_CREDS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "", "", f"could not read client secret JSON at {_GMAIL_CREDS_PATH}: {str(exc)[:100]}"
    if "installed" not in data and "web" in data:
        return "", "", (
            f"`{_GMAIL_CREDS_PATH}` is a **Web application** OAuth client — Primus needs a "
            "**Desktop app** client instead. In the Google Cloud Console → APIs & Services → "
            "Credentials, create a NEW OAuth client ID with application type 'Desktop app', "
            "download that JSON, and replace the file. (Web clients need redirect URIs registered "
            "per port and break the local sign-in flow.)"
        )
    block = data.get("installed") or data.get("web") or {}
    client_id = (block.get("client_id") or "").strip()
    auth_uri = (block.get("auth_uri") or "https://accounts.google.com/o/oauth2/auth").strip()
    if not client_id or "example.com" in client_id or ".apps.googleusercontent.com" not in client_id:
        return "", "", (
            "the credentials JSON has no valid client_id — make sure "
            f"`{_GMAIL_CREDS_PATH}` is the real 'Desktop app' OAuth client you downloaded from the "
            "Google Cloud Console (it must contain a *.apps.googleusercontent.com client_id)."
        )
    return client_id, auth_uri, ""


def _gmail_decode_oauth_error(text: str) -> str:
    """Translate the common Google OAuth failure strings into actionable one-liners.

    Google's raw errors ("redirect_uri_mismatch", "access_denied", disabled API, …) mean
    nothing to most operators; each maps to one concrete fix. Returns "" when nothing matched.
    """
    low = (text or "").lower()
    if not low:
        return ""
    if "redirect_uri_mismatch" in low:
        return (
            "redirect_uri_mismatch — the OAuth client's registered redirect URI doesn't match "
            f"`{_google_redirect_uri()}`. Edit the Desktop client in Google Cloud Console → "
            "Credentials and add that exact URI (or set gmail_redirect_uri in config to the "
            "registered one)."
        )
    if "access_denied" in low:
        return (
            "access_denied — sign-in was declined, OR the consent screen is in 'Testing' and "
            "your Google account isn't listed under OAuth consent screen → Test users. Add your "
            "account there and try again."
        )
    if "has not completed the google verification" in low or "app is not verified" in low:
        return (
            "App not verified — the consent screen is unpublished. Either publish it, or add "
            "your Google account under OAuth consent screen → Test users and retry."
        )
    if "api has not been used" in low or "api is disabled" in low or "access not configured" in low:
        return (
            "Gmail API is not enabled — in Google Cloud Console → APIs & Services → Library, "
            "enable the **Gmail API** for this project, wait a minute, and retry."
        )
    if "invalid_client" in low:
        return (
            "invalid_client — the credentials JSON doesn't match a live OAuth client (deleted or "
            "wrong project). Download a fresh Desktop-app client secret from the Console."
        )
    if "invalid_grant" in low:
        return (
            "invalid_grant — the authorization code expired or was already used (codes are "
            "single-use and live ~10 minutes). Run gmail_auth again for a fresh link."
        )
    return ""


def _manual_auth_url(client_id: str, auth_uri: str, redirect_uri: str, scopes: list[str]) -> str:
    """Deterministically build a correct Google OAuth2 authorization URL (all required params).

    Built by hand (not via flow.authorization_url) so the URL is ALWAYS well-formed: it includes
    client_id, redirect_uri, response_type=code, a space-separated + URL-encoded scope list,
    access_type=offline and prompt=consent. Spaces/slashes are percent-encoded (%20/%2F).
    """
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    return f"{auth_uri}?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"


def _display_available() -> bool:
    """True when a real browser can plausibly open on this machine (not Docker/headless/SSH)."""
    if os.name == "nt" or _sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _start_loopback_listener(service_key: str, scopes: list[str], token_path: Path,
                             service_label: str) -> bool:
    """Host sessions: run a real loopback listener so consent completes with ZERO pasting.

    Uses its own flow instance (the pending manual flow keeps the fixed ``http://localhost``
    redirect, so ``<service>_auth_code`` still works as the headless fallback).
    ``run_local_server(port=0)`` binds an ephemeral port and sets the redirect to match —
    Desktop-app clients are exempt from per-port redirect registration (RFC 8252 §7.3), so
    no Google Cloud Console change is needed. Returns True when the listener thread started.
    """
    if not bool(_host.CFG.get("gmail_oauth_local_server", True)):
        return False
    if not _display_available():
        return False
    with _PENDING_OAUTH_LOCK:
        pend = _PENDING_OAUTH.get(service_key) or {}
        if pend.get("listener") == "waiting":
            return True  # one listener at a time — a second tab race helps nobody

    def _serve() -> None:
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow

            flow = InstalledAppFlow.from_client_secrets_file(str(_GMAIL_CREDS_PATH), scopes)
            os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
            creds = flow.run_local_server(
                port=0,
                open_browser=True,
                success_message=(
                    f"Primus is now connected to {service_label} — you can close this tab."
                ),
                timeout_seconds=300,
            )
        except Exception as exc:  # noqa: BLE001
            _host.log.warning(
                "%s loopback listener ended without a token: %s", service_label, exc
            )
            with _PENDING_OAUTH_LOCK:
                if service_key in _PENDING_OAUTH:
                    _PENDING_OAUTH[service_key]["listener"] = ""
            return
        if not creds or not creds.valid:
            return
        try:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")
            try:
                token_path.chmod(0o600)  # token is a secret — owner-only
            except OSError:
                pass
            _set_pending_status(service_key, "done")
            try:
                _conn_note_auth(service_key)
            except Exception:  # noqa: BLE001
                pass
            _host.log.info("%s OAuth: token saved via loopback listener → %s", service_label, token_path)
            try:
                _host.PrimusSession.emit_think(service_label, "Authorized — token saved", "done")
            except Exception:  # noqa: BLE001
                pass
        except OSError as exc:
            _host.log.warning("%s OAuth: could not save token: %s", service_label, exc)

    try:
        threading.Thread(target=_serve, name=f"oauth-loopback-{service_key}", daemon=True).start()
    except Exception:  # noqa: BLE001
        return False
    with _PENDING_OAUTH_LOCK:
        if service_key in _PENDING_OAUTH:
            _PENDING_OAUTH[service_key]["listener"] = "waiting"
    return True


def _build_google_auth_url(scopes: list[str], service_label: str, token_path: Path, service_key: str,
                           open_browser: bool = False) -> str:
    """Build the Google OAuth2 authorization URL and return it — no browser, no local server.

    Registers a pending flow so the matching ``<service>_auth_code`` tool can finish the exchange
    with the code the user copies from the (loopback) redirect page. Dead-simple and side-effect free
    beyond storing the pending flow. Never raises.
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        return _GMAIL_MISSING_LIBS
    if not _GMAIL_CREDS_PATH.exists():
        return (
            f"⚠ Missing OAuth client secret at `{_GMAIL_CREDS_PATH}`.\n\n"
            "Set it up once in the Google Cloud Console (enable the API, configure the OAuth consent "
            "screen with your account as a Test user, create a **Desktop app** OAuth client ID), then "
            f"save the JSON as `{_GMAIL_CREDS_PATH}` and run this again."
        )
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(_GMAIL_CREDS_PATH), scopes)
    except Exception as exc:  # noqa: BLE001
        _host.log.exception("%s auth-url: could not load client secret", service_label)
        return f"⚠ {service_label} auth URL failed — bad client-secret JSON: {str(exc)[:120]}"

    # Fixed loopback redirect that EXACTLY matches what's registered on the OAuth client. No server is
    # started here — after the user approves, the ?code=... simply appears in the address bar and they
    # paste it into `<service>_auth_code`. Using the bare `http://localhost` avoids the classic
    # `400 invalid_request` (redirect_uri_mismatch) that ephemeral ports cause on Web clients.
    redirect_uri = _google_redirect_uri()
    flow.redirect_uri = redirect_uri
    client_id, auth_uri, cid_err = _load_google_client_id()
    if cid_err:
        return f"⚠ {service_label} auth URL failed — {cid_err}"
    auth_url = _manual_auth_url(client_id, auth_uri, redirect_uri, scopes)

    # Register the pending flow so <service>_auth_code can complete the exchange.
    with _PENDING_OAUTH_LOCK:
        old = _PENDING_OAUTH.get(service_key)
        if old and old.get("server") is not None:
            try:
                old["server"].server_close()
            except Exception:  # noqa: BLE001
                pass
        _PENDING_OAUTH[service_key] = {
            "flow": flow, "token_path": token_path, "label": service_label,
            "redirect_uri": redirect_uri, "auth_url": auth_url, "server": None, "status": "waiting",
        }
    _host.log.info("%s OAuth URL (manual/no-server): %s", service_label, auth_url)

    # Loud terminal banner + optional browser launch (both best-effort, never on the critical path).
    try:
        print(
            "\n" + "=" * 74 + "\n"
            f"  PRIMUS · {service_label} authorization needed\n"
            + "-" * 74 + "\n"
            "  Open this URL in a browser and approve access:\n\n"
            f"    {auth_url}\n"
            + "=" * 74 + "\n",
            flush=True,
        )
    except Exception:  # noqa: BLE001
        pass
    # Host sessions: a loopback listener catches the redirect so consent completes on its own
    # (no "can't connect" page, no code pasting). Headless/Docker falls back to the manual URL.
    listener_live = False
    if open_browser:
        listener_live = _start_loopback_listener(service_key, scopes, token_path, service_label)
        if not listener_live:
            _spawn_browser_open(auth_url, service_label)

    if listener_live:
        return (
            f"🔐 **{service_label} authorization — a browser tab just opened on this machine.**\n\n"
            "Approve access there; the local listener completes sign-in automatically — "
            "you don't need to copy anything. Then confirm with `" + service_key + "_status`.\n\n"
            "_On a headless/remote box instead? Open this URL anywhere, approve, and paste the "
            "resulting `code` back with `" + service_key + "_auth_code <code>`:_\n\n"
            f"{auth_url}"
        )

    browser_line = "A browser will also try to open automatically.\n\n" if open_browser else ""
    return (
        f"🔐 **{service_label} authorization — open this link:**\n\n"
        f"{auth_url}\n\n"
        f"{browser_line}"
        f"Approve access, then you'll be redirected to a page at `{redirect_uri}/?code=XXXXX…` "
        "(it may show a browser connection error — that's expected, no server is running). "
        "Copy the `code` value from that address bar and run:\n\n"
        f"`{service_key}_auth_code <code>`\n\n"
        f"(You can paste the whole redirect URL too.) Then confirm with `{service_key}_status`."
    )


def _google_oauth_authorize(scopes: list[str], service_label: str, token_path: Path, service_key: str) -> str:
    """Kick off a Google OAuth2 sign-in and return the authorization URL for the chat.

    Thin wrapper over :func:`_build_google_auth_url` so ``gmail_auth`` / ``calendar_auth`` and
    ``gmail_get_auth_url`` all produce the *exact same* deterministically-built URL with a fixed
    ``redirect_uri=http://localhost`` (the value registered on the OAuth client). Two completion
    paths: on a host with a display, a background loopback listener (``run_local_server(port=0)``,
    ephemeral port — Desktop-app clients are exempt from per-port redirect registration, RFC 8252)
    catches the redirect and finishes automatically; headless/Docker/SSH falls back to the manual
    ``<service>_auth_code`` step with the fixed redirect URL. Never raises.
    """
    return _build_google_auth_url(scopes, service_label, token_path, service_key, open_browser=True)


@tool
def gmail_auth() -> str:
    """Authorize Primus to access the operator's Gmail (one-time OAuth2 browser sign-in).

    SETUP (do this once in the Google Cloud Console — https://console.cloud.google.com):
      1. Create/select a project.
      2. APIs & Services → Library → enable the **Gmail API**.
      3. APIs & Services → OAuth consent screen → set it up (External is fine); add your own
         Google account under **Test users** so it can sign in without app verification.
      4. APIs & Services → Credentials → Create credentials → **OAuth client ID** →
         Application type **Desktop app**.
      5. Download the JSON and save it as `~/.primus/gmail_credentials.json`.
      6. Run this tool. A browser window opens; approve the requested read/compose/send scopes.
         The resulting token is cached at `~/.primus/gmail_token.json` and refreshed automatically.

    Safe to re-run: if a valid token already exists it just confirms; an expired one is refreshed.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        return _GMAIL_MISSING_LIBS

    if not _GMAIL_CREDS_PATH.exists():
        # Convenience: the Console download lands in ~/Downloads as client_secret_*.json — if
        # it's sitting there, copy (never move) the newest one into place and continue.
        try:
            downloads = sorted(
                Path.home().glob("Downloads/client_secret_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            downloads = []
        for candidate in downloads[:1]:
            try:
                if "installed" in json.loads(candidate.read_text(encoding="utf-8")):
                    _GMAIL_CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, _GMAIL_CREDS_PATH)
                    _host.log.info("Gmail: copied Desktop client secret from %s", candidate)
                    break
            except (OSError, json.JSONDecodeError):
                continue
        if not _GMAIL_CREDS_PATH.exists():
            downloads_note = ""
            if downloads:
                downloads_note = (
                    f"\n\nFound a downloaded client secret at `{downloads[0]}` — it isn't a "
                    "Desktop-app client, so it wasn't copied. Create a Desktop app client, or copy "
                    f"it manually: `cp {downloads[0]} {_GMAIL_CREDS_PATH}`"
                )
            return (
                f"⚠ Missing OAuth client secret at `{_GMAIL_CREDS_PATH}`.\n\n"
                "One-time Google Cloud Console setup (this is the ONLY manual step — no App Passwords, "
                "no 'less secure apps', no IMAP/SMTP settings):\n"
                "  1. Enable the Gmail API for a project.\n"
                "  2. Configure the OAuth consent screen and add your Google account as a Test user.\n"
                "  3. Create an OAuth client ID of type 'Desktop app'.\n"
                f"  4. Download the JSON and save it as `{_GMAIL_CREDS_PATH}`.\n"
                "Then run `gmail_auth` (or `ensure_gmail_access`) again — it works on desktop, headless, "
                "Docker, and SSH." + downloads_note
            )
        # Copied from Downloads — fall through and start the flow with the new file.

    # Reuse an existing valid/refreshable token instead of forcing a new browser flow.
    if _GMAIL_TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(_GMAIL_TOKEN_PATH), _GMAIL_SCOPES)
            if creds and creds.valid:
                return "✓ Gmail is already authorized. Use gmail_status to see the connected account."
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                _GMAIL_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
                return "✓ Gmail token refreshed — you're still authorized."
        except Exception:  # noqa: BLE001 — fall through to a fresh flow
            pass

    # Non-blocking flow: returns the authorization URL to the chat immediately; the token exchange
    # completes on a background loopback server (or via gmail_auth_code as a manual fallback).
    return _google_oauth_authorize(_GMAIL_SCOPES, "Gmail", _GMAIL_TOKEN_PATH, "gmail")


@tool
def gmail_auth_code(code: str = "") -> str:
    """Finish Gmail authorization by pasting the code from the browser (manual fallback for gmail_auth).

    Use this only if `gmail_auth` didn't complete on its own (e.g. the browser couldn't reach the
    local callback). After approving access in the browser, copy the `code` value from the page you
    land on — the address bar looks like `http://localhost:PORT/?code=XXXX…` — and pass it here. You
    may paste either just the code or the entire redirect URL. Requires a `gmail_auth` started in
    this session.
    """
    return _complete_google_oauth_manual("gmail", code)


@tool
def ensure_gmail_access() -> str:
    """Make sure Gmail is connected — the one-shot tool for ANY "connect / fix / set up my Gmail" request.

    Checks the live connection; if Gmail is already authorized it confirms the account, otherwise it
    starts the OAuth sign-in and returns the EXACT next step (a link to open + `gmail_auth_code`).
    Prefer this over giving manual Google settings — Primus never uses App Passwords, "less secure
    apps", or IMAP/SMTP. Works on desktop, headless, Docker, and SSH. Safe/read-only; nothing is sent.
    """
    _host.PrimusSession.emit_think("Gmail", "Checking access", "running")

    # Libraries present?
    try:
        import google.auth  # noqa: F401
    except ImportError:
        _host.PrimusSession.emit_think("Gmail", "Libraries missing", "error")
        return _GMAIL_MISSING_LIBS

    # Client secret present? If not, that's the only real manual (one-time) setup step.
    if not _GMAIL_CREDS_PATH.exists():
        _host.PrimusSession.emit_think("Gmail", "Client secret missing", "error")
        return _run_tool(gmail_auth)  # returns the one-time Console setup instructions

    # Already connected? Verify by actually hitting the API (proves the token is live, not just present).
    service, err = _get_gmail_service()
    if service is not None and not err:
        try:
            profile = service.users().getProfile(userId="me").execute()
            _host.PrimusSession.emit_think("Gmail", "Connected", "done")
            return (
                f"✓ Gmail is connected as **{profile.get('emailAddress', '?')}** "
                f"({profile.get('messagesTotal', '?')} messages). You're all set — I can read, search, "
                "draft, and send from here."
            )
        except Exception as exc:  # noqa: BLE001 — token exists but the call failed → re-auth
            _host.log.warning("ensure_gmail_access: token present but API call failed: %s", exc)

    # Not connected (or token stale) → start authorization and hand back the exact next step.
    _host.PrimusSession.emit_think("Gmail", "Starting authorization", "running")
    auth_msg = _run_tool(gmail_auth)
    return (
        "Gmail isn't connected yet — I'm starting authorization for you (no App Passwords or manual "
        "settings needed).\n\n" + auth_msg
    )


@tool
def gmail_get_auth_url() -> str:
    """Return the Gmail OAuth2 authorization URL immediately — no browser, no local server needed.

    The simplest, most reliable way to authorize: this just prints the full Google sign-in URL. Open
    it in ANY browser (even on another device), approve access, then copy the `code` from the page you
    land on and finish with `gmail_auth_code <code>`. Uses the existing client secret + Gmail scopes.
    """
    if _GMAIL_TOKEN_PATH.exists():
        try:
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_file(str(_GMAIL_TOKEN_PATH), _GMAIL_SCOPES)
            if creds and creds.valid:
                return "✓ Gmail is already authorized. Use gmail_status to see the connected account."
        except Exception:  # noqa: BLE001
            pass
    return _build_google_auth_url(_GMAIL_SCOPES, "Gmail", _GMAIL_TOKEN_PATH, "gmail")


@tool
def gmail_status() -> str:
    """Check the Gmail connection and show the authorized account + mailbox totals.

    Returns the connected email address and total message/thread counts, or clear guidance if
    the Google libraries are missing or authorization hasn't been completed (run gmail_auth).
    Use this first to confirm Gmail is actually connected before reporting any inbox contents.
    """
    if _gmail_fixture_active():
        msgs = _gmail_fixture_load()
        unread = sum(1 for m in msgs if m.get("unread"))
        return (
            "📬 **Demo inbox (fixture)** — not connected to a real Gmail account.\n"
            f"Messages: {len(msgs)} · Unread: {unread}\n"
            "This is sample data from `examples/gmail/fixture_inbox.json`. "
            "Run `gmail_auth` to connect a real mailbox."
        )
    service, err = _get_gmail_service()
    if err:
        return err
    try:
        profile = service.users().getProfile(userId="me").execute()
        return (
            f"✓ Gmail connected as **{profile.get('emailAddress', '?')}**\n"
            f"Messages: {profile.get('messagesTotal', '?')} · Threads: {profile.get('threadsTotal', '?')}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Gmail status check failed: {str(exc)[:160]}"


@tool
def gmail_list_messages(query: str = "", max_results: int = 10) -> str:
    """List/search recent Gmail messages, returning id, date, sender, subject and a snippet.

    `query` uses standard Gmail search syntax (e.g. "is:unread", "from:boss@x.com newer_than:7d",
    "subject:invoice has:attachment"); empty lists the most recent messages. `max_results` is
    clamped to 1–50. Use the returned [id] with gmail_read_message to read a full message.

    This returns REAL data from the live connected mailbox. If it returns an authorization error,
    relay that error to the user and ask them to run gmail_auth — never invent or sample emails.
    When no mailbox is connected but gmail_fixture_mode is on, results come from the demo
    fixture mailbox and are clearly labelled as such.
    """
    max_results = max(1, min(int(max_results or 10), 50))
    if _gmail_fixture_active():
        msgs = _gmail_fixture_query(_gmail_fixture_load(), query)[:max_results]
        if not msgs:
            return f"📬 Demo inbox (fixture): no messages match{f' query: {query}' if query else ''}."
        lines = []
        for m in msgs:
            mark = "🔵 " if m.get("unread") else ""
            lines.append(
                f"• [{m['id']}] {str(m.get('date', '?'))[:25]} — {str(m.get('from', '?'))[:45]}\n"
                f"    {mark}{str(m.get('subject', '(no subject)'))[:80]}\n"
                f"    {str(m.get('snippet', ''))[:120]}"
            )
        return (
            f"📬 {len(msgs)} message(s) — **demo inbox (fixture)**:\n\n" + "\n\n".join(lines)
        )
    service, err = _get_gmail_service()
    if err:
        return err
    _host.PrimusSession.emit_think("Gmail", f"Listing messages{f' · {query}' if query else ''}", "running")
    try:
        resp = service.users().messages().list(
            userId="me", q=query or "", maxResults=max_results
        ).execute()
        msgs = resp.get("messages", []) or []
        if not msgs:
            _host.PrimusSession.emit_think("Gmail", "No messages", "done")
            return f"No messages found{f' for query: {query}' if query else ''}."
        lines: list[str] = []
        for m in msgs:
            meta = service.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            hdr = {h["name"]: h["value"] for h in meta.get("payload", {}).get("headers", [])}
            lines.append(
                f"• [{m['id']}] {hdr.get('Date', '?')[:25]} — {hdr.get('From', '?')[:45]}\n"
                f"    {hdr.get('Subject', '(no subject)')[:80]}\n"
                f"    {meta.get('snippet', '')[:120]}"
            )
        _host.PrimusSession.emit_think("Gmail", f"{len(msgs)} message(s)", "done")
        return f"📬 {len(msgs)} message(s):\n\n" + "\n\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        _host.PrimusSession.emit_think("Gmail", "List failed", "error")
        return f"⚠ Gmail list failed: {str(exc)[:160]}"


@tool
def gmail_read_message(message_id: str, include_attachments: bool = False) -> str:
    """Read one full Gmail message: headers, body text, and attachment info.

    Provide a `message_id` from gmail_list_messages. Returns From/To/Date/Subject plus the plain
    text body (HTML is stripped to text when that's all a message has). When `include_attachments`
    is True, each attachment is downloaded to ~/.primus/gmail_attachments/ and its saved path is
    reported; otherwise attachments are only listed.

    This returns REAL data from the live connected mailbox. If it returns an authorization error,
    relay that error to the user and ask them to run gmail_auth — never invent or sample content.
    """
    if not (message_id or "").strip():
        return "⚠ Provide a message_id (get one from gmail_list_messages)."
    mid = message_id.strip()
    if _gmail_fixture_active():
        for m in _gmail_fixture_load():
            if str(m.get("id")) == mid:
                return "\n".join([
                    "📬 Demo inbox (fixture) — sample message, not real mail.",
                    "",
                    f"From: {m.get('from', '?')}",
                    f"To: {m.get('to', '?')}",
                    f"Date: {m.get('date', '?')}",
                    f"Subject: {m.get('subject', '(no subject)')}",
                    "",
                    str(m.get("body", "(no body)"))[:4000],
                ])
        return f"⚠ No fixture message with id {mid} — run gmail_list_messages for valid ids."
    service, err = _get_gmail_service()
    if err:
        return err
    try:
        msg = service.users().messages().get(userId="me", id=mid, format="full").execute()
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Could not read message {mid}: {str(exc)[:140]}"

    payload = msg.get("payload", {}) or {}
    hdr = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    body_text, attachments = _gmail_extract_parts(payload)

    saved: list[str] = []
    if include_attachments and attachments:
        try:
            _GMAIL_ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            saved.append(f"(could not create attachments dir: {str(exc)[:60]})")
        for att in attachments:
            try:
                a = service.users().messages().attachments().get(
                    userId="me", messageId=mid, id=att["attachmentId"],
                ).execute()
                raw = base64.urlsafe_b64decode((a.get("data", "") or "").encode("utf-8"))
                fname = _safe_filename(att["filename"]) or f"attachment-{att['attachmentId'][:8]}"
                dest = _GMAIL_ATTACHMENTS_DIR / fname
                dest.write_bytes(raw)
                saved.append(f"`{dest}` ({len(raw) // 1024} KB)")
            except Exception as exc:  # noqa: BLE001
                saved.append(f"{att['filename']} — failed: {str(exc)[:60]}")

    out = [
        f"From: {hdr.get('from', '?')}",
        f"To: {hdr.get('to', '?')}",
        f"Date: {hdr.get('date', '?')}",
        f"Subject: {hdr.get('subject', '(no subject)')}",
        "",
        (body_text or "(no text body)")[:4000],
    ]
    if attachments:
        out.append(
            "\n📎 Attachments: "
            + ", ".join(f"{a['filename']} ({a['size'] // 1024} KB)" for a in attachments)
        )
    if saved:
        out.append("Saved: " + "; ".join(saved))
    return "\n".join(out)


@tool
def gmail_create_draft(to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
    """Create a Gmail draft (not sent) addressed to `to` with `subject` and `body`.

    Optional `cc`/`bcc` accept comma-separated addresses. Creating a draft is non-destructive — it
    just saves to Drafts for review; send it later with gmail_send_email(draft_id=...) or from Gmail.
    Returns the new draft id.
    """
    if not (to or "").strip():
        return "⚠ Provide at least one recipient (to)."
    if _gmail_fixture_active():
        return (
            "[Demo inbox] Draft preview only — nothing was saved:\n"
            f"  To: {to}\n  Subject: {(subject or '')[:60]}\n"
            f"  Body: {(body or '')[:160]}{'…' if len(body or '') > 160 else ''}\n"
            "Connect Gmail (`gmail_auth`) to create real drafts."
        )
    service, err = _get_gmail_service()
    if err:
        return err
    try:
        raw = _gmail_build_raw(to, subject, body, cc, bcc)
        draft = service.users().drafts().create(
            userId="me", body={"message": {"raw": raw}}
        ).execute()
        return (
            f"✓ Draft created (id: {draft.get('id')}) — To: {to}, Subject: '{(subject or '')[:60]}'. "
            "Review in Gmail or send it with gmail_send_email(draft_id=...)."
        )
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Draft creation failed: {str(exc)[:160]}"


@tool
def gmail_send_email(
    to: str = "", subject: str = "", body: str = "", draft_id: str = "", cc: str = "", bcc: str = ""
) -> str:
    """Send an email via Gmail — either a new message (to/subject/body) or an existing `draft_id`.

    SAFETY: like edit_docx/edit_xlsx, this only PREVIEWS in Suggest mode and actually sends in
    Execute mode (or once approved). Optional `cc`/`bcc` are comma-separated. Provide either a
    `draft_id` (to send a saved draft) or at least `to` (+ subject/body) for a fresh message.
    """
    if not (to or "").strip() and not (draft_id or "").strip():
        return "⚠ Provide recipients (to) + subject + body, or a draft_id to send."

    # Fixture demo mode: never pretend a send happened — preview only, with the way out.
    if _gmail_fixture_active():
        return (
            "[Demo inbox] Not sent — the fixture mailbox can't send email.\n"
            f"  Would send to: {to or f'draft {draft_id.strip()}'}\n"
            f"  Subject: {(subject or '')[:60]}\n"
            "Connect Gmail (`gmail_auth`) to send for real."
        )

    # Preview-only in Suggest mode (mirrors edit_docx / edit_xlsx exactly). Sending is a risky
    # action, so it must not go out until the operator is in Execute mode or explicitly approves.
    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        if (draft_id or "").strip():
            return (
                f"[Suggest] Would send existing draft {draft_id.strip()}. "
                "Switch to Execute (or approve) to send."
            )
        preview = (body or "")[:200] + ("…" if len(body or "") > 200 else "")
        return (
            "[Suggest] Would send email:\n"
            f"  To: {to}\n"
            + (f"  Cc: {cc}\n" if cc else "")
            + (f"  Bcc: {bcc}\n" if bcc else "")
            + f"  Subject: {subject}\n"
            f"  Body: {preview}\n"
            "Switch to Execute (or approve) to actually send."
        )

    service, err = _get_gmail_service()
    if err:
        return err
    try:
        if (draft_id or "").strip():
            sent = service.users().drafts().send(
                userId="me", body={"id": draft_id.strip()}
            ).execute()
            return f"✓ Sent draft {draft_id.strip()} (message id: {sent.get('id')})."
        if not (to or "").strip():
            return "⚠ Provide recipients (to) to send a new email."
        raw = _gmail_build_raw(to, subject, body, cc, bcc)
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"✓ Email sent to {to} (message id: {sent.get('id')}), subject '{(subject or '')[:60]}'."
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Send failed: {str(exc)[:160]}"


# ===========================================================================
# Google Calendar integration (official Calendar API, OAuth2)
# ---------------------------------------------------------------------------
# List/read/create events + find free time on the operator's primary calendar. Reuses the SAME OAuth
# client secret as Gmail (~/.primus/gmail_credentials.json) — no extra Google Cloud setup — but
# keeps its own token (~/.primus/calendar_token.json) so it never clobbers the Gmail token or
# forces a mixed-scope re-consent. Google libraries are optional; every tool degrades gracefully.
# Creating events is gated by the same safety model as gmail_send_email (preview in Suggest mode).
# ===========================================================================

# Minimal but powerful: readonly (list/get/freebusy) + events (create/update).
_CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]
_CALENDAR_TOKEN_PATH = _host.APP_DIR / "calendar_token.json"  # cached calendar token (auto-created)
_CALENDAR_WORK_START = 9   # free-slot search window: 09:00 local …
_CALENDAR_WORK_END = 17    # … to 17:00 local


def _cal_parse_dt(value: str) -> Optional[datetime]:
    """Parse a user/API datetime string → timezone-aware datetime (never raises).

    Accepts ISO 8601 (with offset or trailing 'Z'), "YYYY-MM-DD HH:MM", and plain "YYYY-MM-DD".
    Naive inputs are interpreted as LOCAL time; 'Z'/offset inputs keep their real zone. None on failure.
    """
    s = (value or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt: Optional[datetime] = None
    try:
        dt = datetime.fromisoformat(s)  # handles offsets + date-only
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    return dt.astimezone() if dt.tzinfo is None else dt  # naive => local, aware => keep zone


def _cal_event_when(ev: dict) -> str:
    """Human-friendly start time for an event (handles all-day 'date' and timed 'dateTime')."""
    start = ev.get("start", {}) or {}
    when = start.get("dateTime") or start.get("date") or "?"
    return when[:16].replace("T", " ")


def _cal_fmt_event_line(ev: dict) -> str:
    """One-line event summary: [id] when — title (location)."""
    loc = ev.get("location", "")
    tail = f"  @ {loc[:40]}" if loc else ""
    return f"[{ev.get('id', '?')}] {_cal_event_when(ev)} — {ev.get('summary', '(no title)')[:60]}{tail}"


def _get_calendar_service() -> tuple[Any, str]:
    """Return (calendar_service, "") on success, else (None, user-facing error/instructions).

    Loads the cached calendar token, silently refreshes it if expired, and builds a Calendar API
    client. Handles missing libraries, missing/invalid token, and refresh failures gracefully —
    mirrors _get_gmail_service exactly but uses the calendar token + scopes.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        return None, _GMAIL_MISSING_LIBS

    if not _CALENDAR_TOKEN_PATH.exists():
        return None, (
            "⚠ Google Calendar isn't authorized yet. Run the `calendar_auth` tool once to sign in "
            "(it reuses your Gmail credentials file — see calendar_auth for details)."
        )
    try:
        creds = Credentials.from_authorized_user_file(str(_CALENDAR_TOKEN_PATH), _CALENDAR_SCOPES)
    except Exception as exc:  # noqa: BLE001
        return None, f"⚠ Could not load Calendar token: {str(exc)[:120]}. Re-run calendar_auth."

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _CALENDAR_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return None, f"⚠ Calendar token refresh failed: {str(exc)[:120]}. Re-run calendar_auth."
    if not creds or not creds.valid:
        return None, "⚠ Calendar credentials are invalid or revoked. Re-run calendar_auth."

    try:
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        return service, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"⚠ Could not build Calendar service: {str(exc)[:140]}"


@tool
def calendar_auth() -> str:
    """Authorize Primus to access the operator's Google Calendar (one-time OAuth2 browser sign-in).

    Reuses the SAME OAuth client secret as Gmail — if you've already set up
    `~/.primus/gmail_credentials.json` for Gmail there's NOTHING new to download; just make sure the
    **Google Calendar API** is enabled for the same Google Cloud project:
      1. Google Cloud Console → APIs & Services → Library → enable **Google Calendar API**
         (the Gmail API can stay enabled too — one project, one client works for both).
      2. Ensure `~/.primus/gmail_credentials.json` exists (from the Gmail setup / a Desktop OAuth
         client ID). No separate credentials file is needed.
      3. Run this tool. A browser window opens; approve the calendar read + events scopes. The token
         is cached separately at `~/.primus/calendar_token.json` and refreshed automatically, so it
         never interferes with your Gmail authorization.

    Safe to re-run: a valid token is confirmed; an expired one is refreshed.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        return _GMAIL_MISSING_LIBS

    if not _GMAIL_CREDS_PATH.exists():
        return (
            f"⚠ Missing OAuth client secret at `{_GMAIL_CREDS_PATH}`.\n\n"
            "Calendar reuses the Gmail credentials file. Set it up once in the Google Cloud Console:\n"
            "  1. Enable the Google Calendar API (and Gmail API) for a project.\n"
            "  2. Configure the OAuth consent screen and add your Google account as a Test user.\n"
            "  3. Create an OAuth client ID of type 'Desktop app'.\n"
            f"  4. Download the JSON and save it as `{_GMAIL_CREDS_PATH}`.\n"
            "Then run calendar_auth again."
        )

    # Reuse an existing valid/refreshable calendar token instead of forcing a new browser flow.
    if _CALENDAR_TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(_CALENDAR_TOKEN_PATH), _CALENDAR_SCOPES)
            if creds and creds.valid:
                return "✓ Google Calendar is already authorized. Use calendar_status to see upcoming events."
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                _CALENDAR_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
                return "✓ Calendar token refreshed — you're still authorized."
        except Exception:  # noqa: BLE001 — fall through to a fresh flow
            pass

    # Non-blocking flow (shared with Gmail): returns the URL to chat; completes in the background
    # or via calendar_auth_code as a manual fallback.
    return _google_oauth_authorize(_CALENDAR_SCOPES, "Google Calendar", _CALENDAR_TOKEN_PATH, "calendar")


@tool
def calendar_auth_code(code: str = "") -> str:
    """Finish Google Calendar authorization by pasting the code from the browser (manual fallback).

    Use this only if `calendar_auth` didn't complete on its own. After approving in the browser,
    copy the `code` from the redirect page (address bar `http://localhost:PORT/?code=XXXX…`) and pass
    it here — just the code or the whole URL. Requires a `calendar_auth` started in this session.
    """
    return _complete_google_oauth_manual("calendar", code)


@tool
def calendar_status() -> str:
    """Check the Google Calendar connection and summarize the next few upcoming events.

    Returns the primary calendar id (your email) + timezone and a short list of upcoming events, or
    clear guidance if the Google libraries are missing or authorization hasn't run (calendar_auth).
    """
    service, err = _get_calendar_service()
    if err:
        return err
    try:
        cal = service.calendars().get(calendarId="primary").execute()
        now = datetime.now().astimezone()
        resp = service.events().list(
            calendarId="primary", timeMin=now.isoformat(), maxResults=5,
            singleEvents=True, orderBy="startTime",
        ).execute()
        items = resp.get("items", []) or []
        lines = [
            f"✓ Google Calendar connected — primary: **{cal.get('id', '?')}** "
            f"(timezone: {cal.get('timeZone', '?')})"
        ]
        if items:
            lines.append(f"Next {len(items)} event(s):")
            lines.extend("  • " + _cal_fmt_event_line(ev) for ev in items)
        else:
            lines.append("No upcoming events.")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Calendar status check failed: {str(exc)[:160]}"


@tool
def calendar_list_events(max_results: int = 10, time_min: Optional[str] = None) -> str:
    """List upcoming Google Calendar events (soonest first) with id, time, title and location.

    `max_results` is clamped to 1–50. `time_min` optionally sets the earliest start to list (ISO or
    "YYYY-MM-DD [HH:MM]"); it defaults to now. Use the returned [id] with calendar_get_event for
    full details, or with calendar_create_event's flow to schedule around them.
    """
    service, err = _get_calendar_service()
    if err:
        return err
    max_results = max(1, min(int(max_results or 10), 50))
    start = _cal_parse_dt(time_min) if time_min else None
    start = start or datetime.now().astimezone()
    _host.PrimusSession.emit_think("Calendar", "Listing upcoming events", "running")
    try:
        resp = service.events().list(
            calendarId="primary", timeMin=start.isoformat(), maxResults=max_results,
            singleEvents=True, orderBy="startTime",
        ).execute()
        items = resp.get("items", []) or []
        if not items:
            _host.PrimusSession.emit_think("Calendar", "No events", "done")
            return "No upcoming events found."
        lines = []
        for ev in items:
            atts = ev.get("attendees", []) or []
            extra = f"  ({len(atts)} attendee(s))" if atts else ""
            lines.append("• " + _cal_fmt_event_line(ev) + extra)
        _host.PrimusSession.emit_think("Calendar", f"{len(items)} event(s)", "done")
        return f"📅 {len(items)} upcoming event(s):\n\n" + "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        _host.PrimusSession.emit_think("Calendar", "List failed", "error")
        return f"⚠ Calendar list failed: {str(exc)[:160]}"


@tool
def calendar_get_event(event_id: str) -> str:
    """Read the full details of one Google Calendar event by id.

    Provide an `event_id` from calendar_list_events. Returns title, start/end, location, description,
    attendees (with RSVP status) and the event's Google Calendar link.
    """
    if not (event_id or "").strip():
        return "⚠ Provide an event_id (get one from calendar_list_events)."
    service, err = _get_calendar_service()
    if err:
        return err
    try:
        ev = service.events().get(calendarId="primary", eventId=event_id.strip()).execute()
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Could not read event {event_id}: {str(exc)[:140]}"

    start = (ev.get("start", {}) or {}).get("dateTime") or (ev.get("start", {}) or {}).get("date", "?")
    end = (ev.get("end", {}) or {}).get("dateTime") or (ev.get("end", {}) or {}).get("date", "?")
    out = [
        f"Title: {ev.get('summary', '(no title)')}",
        f"When: {start} → {end}",
    ]
    if ev.get("location"):
        out.append(f"Location: {ev['location']}")
    attendees = ev.get("attendees", []) or []
    if attendees:
        out.append("Attendees: " + ", ".join(
            f"{a.get('email', '?')} ({a.get('responseStatus', 'needsAction')})" for a in attendees
        ))
    if ev.get("description"):
        out.append(f"\n{ev['description'][:2000]}")
    if ev.get("htmlLink"):
        out.append(f"\nLink: {ev['htmlLink']}")
    return "\n".join(out)


@tool
def calendar_find_free_slots(start_date: str, end_date: str, duration_minutes: int = 30) -> str:
    """Find open time blocks (≥ duration_minutes) between two dates, within 09:00–17:00 local.

    `start_date`/`end_date` are "YYYY-MM-DD" (time optional). Uses the Calendar free/busy API to
    subtract existing events from each working day and reports the remaining free windows — great
    for "when am I free this week for a 1-hour call?". `duration_minutes` filters out windows too
    short to fit (default 30). Range is capped at 31 days to keep output readable.
    """
    service, err = _get_calendar_service()
    if err:
        return err
    start = _cal_parse_dt(start_date)
    end = _cal_parse_dt(end_date)
    if not start or not end:
        return "⚠ Provide start_date and end_date as YYYY-MM-DD (time optional)."
    if end < start:
        return "⚠ end_date must be on or after start_date."
    dur = max(5, int(duration_minutes or 30))
    if (end.date() - start.date()).days > 31:
        return "⚠ Range too large — please keep start_date..end_date within 31 days."

    # Query busy intervals across the whole window (end of the last day inclusive).
    window_end = end.replace(hour=23, minute=59, second=59, microsecond=0)
    try:
        fb = service.freebusy().query(body={
            "timeMin": start.isoformat(),
            "timeMax": window_end.isoformat(),
            "items": [{"id": "primary"}],
        }).execute()
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Free/busy lookup failed: {str(exc)[:160]}"
    busy_raw = fb.get("calendars", {}).get("primary", {}).get("busy", []) or []
    busy = []
    for b in busy_raw:
        bs, be = _cal_parse_dt(b.get("start")), _cal_parse_dt(b.get("end"))
        if bs and be:
            busy.append((bs, be))

    out_days: list[str] = []
    days = (end.date() - start.date()).days
    for offset in range(days + 1):
        day = start.date() + timedelta(days=offset)
        day_start = datetime(day.year, day.month, day.day, _CALENDAR_WORK_START, 0).astimezone()
        day_end = datetime(day.year, day.month, day.day, _CALENDAR_WORK_END, 0).astimezone()
        # Clip the working window to the requested overall range.
        day_start = max(day_start, start)
        day_end = min(day_end, window_end)
        if day_end <= day_start:
            continue
        # Busy intervals overlapping this day's working window, clipped + sorted.
        day_busy = sorted(
            (max(bs, day_start), min(be, day_end))
            for bs, be in busy if be > day_start and bs < day_end
        )
        free: list[tuple[datetime, datetime]] = []
        cursor = day_start
        for bs, be in day_busy:
            if bs - cursor >= timedelta(minutes=dur):
                free.append((cursor, bs))
            cursor = max(cursor, be)
        if day_end - cursor >= timedelta(minutes=dur):
            free.append((cursor, day_end))
        if free:
            slots = ", ".join(
                f"{s.strftime('%H:%M')}–{e.strftime('%H:%M')} "
                f"({int((e - s).total_seconds() // 60)}m)"
                for s, e in free
            )
            out_days.append(f"{day.isoformat()}: {slots}")

    if not out_days:
        return (
            f"No free blocks ≥ {dur}m found between {start.date()} and {end.date()} "
            f"(within {_CALENDAR_WORK_START:02d}:00–{_CALENDAR_WORK_END:02d}:00)."
        )
    return f"🕒 Free blocks ≥ {dur}m ({_CALENDAR_WORK_START:02d}:00–{_CALENDAR_WORK_END:02d}:00 local):\n\n" + "\n".join(out_days)


@tool
def calendar_create_event(
    summary: str, start_time: str, end_time: str, description: str = "", attendees: str = ""
) -> str:
    """Create a Google Calendar event on the primary calendar.

    `start_time`/`end_time` are "YYYY-MM-DD HH:MM" (or ISO); naive times are treated as local.
    Optional `description` and comma-separated `attendees` emails (attendees get an invite email).
    SAFETY: like gmail_send_email, this only PREVIEWS in Suggest mode and actually creates the event
    in Execute mode (or once approved) — scheduling is a real-world action, so it never fires silently.
    """
    if not (summary or "").strip():
        return "⚠ Provide a summary (event title)."
    start = _cal_parse_dt(start_time)
    end = _cal_parse_dt(end_time)
    if not start or not end:
        return "⚠ Provide start_time and end_time as 'YYYY-MM-DD HH:MM' (or ISO 8601)."
    if end <= start:
        return "⚠ end_time must be after start_time."
    attendee_list = [e.strip() for e in (attendees or "").split(",") if e.strip()]

    # Preview-only in Suggest mode (mirrors gmail_send_email / edit_docx exactly).
    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        who = f"\n  Attendees: {', '.join(attendee_list)}" if attendee_list else ""
        desc = f"\n  Description: {description[:120]}" if description else ""
        return (
            "[Suggest] Would create event:\n"
            f"  Title: {summary}\n"
            f"  When: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')}"
            f"{who}{desc}\n"
            "Switch to Execute (or approve) to actually create it."
        )

    service, err = _get_calendar_service()
    if err:
        return err
    body: dict[str, Any] = {
        "summary": summary,
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": end.isoformat()},
    }
    if description:
        body["description"] = description
    if attendee_list:
        body["attendees"] = [{"email": e} for e in attendee_list]
    try:
        ev = service.events().insert(
            calendarId="primary", body=body,
            sendUpdates="all" if attendee_list else "none",
        ).execute()
        link = ev.get("htmlLink", "")
        return (
            f"✓ Event created: '{summary}' on {start.strftime('%Y-%m-%d %H:%M')} "
            f"(id: {ev.get('id')})." + (f"\nLink: {link}" if link else "")
        )
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Event creation failed: {str(exc)[:160]}"


# ===========================================================================
# Gmail ↔ Calendar bridge (turn emails into meetings, and propose times back)
# ---------------------------------------------------------------------------
# Deterministic helpers extract meeting intent from an email (title/duration/times/attendees), then
# reuse the Calendar free/busy logic to suggest concrete open slots. Actions that create an event or
# send mail preview in Suggest mode; drafting a reply is always safe (saved unsent to Drafts).
#
# Example workflows:
#   • "Book the call from that email":   gmail_list_messages → create_meeting_from_email(id)
#   • "When am I free for this?":         suggest_meeting_times_from_email(id, 30, 7)
#   • "Reply with a few times":           propose_times_in_reply(id, 3)   (draft only, unsent)
#   • Inspect what was parsed:            gmail_extract_meeting_request(id)
# ===========================================================================

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_WEEKDAY_IDX = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}


def _extract_emails(text: str) -> list[str]:
    """All email addresses in a blob, de-duplicated, order preserved."""
    out: list[str] = []
    for e in _EMAIL_RE.findall(text or ""):
        if e not in out:
            out.append(e)
    return out


def _parse_duration_minutes(text: str) -> int:
    """Best-effort meeting duration from free text ('30 min', '1 hour', 'quick sync'). 0 if unknown."""
    t = (text or "").lower()
    m = re.search(r"(\d+)\s*[-\s]?\s*(?:hours?|hrs?|h)\b", t)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*[-\s]?\s*(?:minutes?|mins?)\b", t)
    if m:
        return int(m.group(1))
    if re.search(r"\bhalf\s*(?:an\s*)?hour\b", t):
        return 30
    if re.search(r"\b(?:an|one)\s*hour\b", t):
        return 60
    if re.search(r"\bquick\s*(?:call|chat|sync|catch[- ]?up)\b", t):
        return 15
    return 0


def _parse_datetimes_from_text(text: str, now: datetime) -> list[datetime]:
    """Best-effort future datetimes from natural language ('Tuesday at 2pm', 'July 5 10:00', '7/5').

    Combines detected day references with detected times; falls back to sensible defaults when only
    one is present. Returns up to 4 timezone-aware, future datetimes. Purely heuristic — never raises.
    """
    low = (text or "").lower()

    # Times: 12-hour (am/pm) first, then 24-hour on text with am/pm times stripped (avoid dupes).
    times: list[tuple[int, int]] = []
    for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", low):
        h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
        times.append((h, int(m.group(2) or 0)))
    low_wo = re.sub(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", " ", low)
    for m in re.finditer(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", low_wo):
        times.append((int(m.group(1)), int(m.group(2))))
    seen: set = set()
    times = [hm for hm in times if not (hm in seen or seen.add(hm))]

    # Days.
    days: list = []
    if re.search(r"\btomorrow\b", low):
        days.append((now + timedelta(days=1)).date())
    if re.search(r"\btoday\b", low):
        days.append(now.date())
    for name, idx in _WEEKDAY_IDX.items():
        if re.search(r"\b" + name + r"\b", low):
            ahead = (idx - now.weekday()) % 7 or 7
            days.append((now + timedelta(days=ahead)).date())
    for m in re.finditer(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})\b", low):
        try:
            cand = datetime(now.year, _MONTHS[m.group(1)], int(m.group(2))).date()
            days.append(cand if cand >= now.date() else cand.replace(year=now.year + 1))
        except ValueError:
            pass
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})\b", low):
        try:
            cand = datetime(now.year, int(m.group(1)), int(m.group(2))).date()
            days.append(cand if cand >= now.date() else cand.replace(year=now.year + 1))
        except ValueError:
            pass
    seen = set()
    days = [d for d in days if not (d in seen or seen.add(d))]

    if not days and not times:
        return []
    if not days:
        days = [now.date(), (now + timedelta(days=1)).date()]
    if not times:
        times = [(10, 0)]

    out: list[datetime] = []
    for d in days:
        for h, mm in times:
            try:
                dt = datetime(d.year, d.month, d.day, h, mm).astimezone()
            except ValueError:
                continue
            if dt > now and dt not in out:
                out.append(dt)
            if len(out) >= 4:
                return out
    return out


def _extract_meeting_details(
    subject: str, body: str, sender: str, to_hdr: str, default_duration: int = 30
) -> dict:
    """Deterministically extract meeting intent → {title, duration, suggested_times, attendees, description}."""
    title = re.sub(r"^\s*(?:re|fwd|fw)\s*:\s*", "", subject or "", flags=re.IGNORECASE).strip() or "Meeting"
    duration = _parse_duration_minutes(f"{subject}\n{body}") or default_duration
    suggested = _parse_datetimes_from_text(body or "", datetime.now().astimezone())
    attendees = _extract_emails(f"{sender}\n{to_hdr}\n{body}")
    description = re.sub(r"\s+", " ", body or "").strip()[:240]
    return {
        "title": title, "duration": duration, "suggested_times": suggested,
        "attendees": attendees, "description": description,
    }


def _read_email_fields(service: Any, mid: str) -> tuple[Optional[dict], str]:
    """Fetch one message → ({subject, from, to, cc, body}, "") or (None, error)."""
    try:
        msg = service.users().messages().get(userId="me", id=mid, format="full").execute()
    except Exception as exc:  # noqa: BLE001
        return None, f"⚠ Could not read message {mid}: {str(exc)[:140]}"
    payload = msg.get("payload", {}) or {}
    hdr = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    body, _ = _gmail_extract_parts(payload)
    return {
        "subject": hdr.get("subject", ""), "from": hdr.get("from", ""),
        "to": hdr.get("to", ""), "cc": hdr.get("cc", ""), "body": body,
    }, ""


def _free_windows(service: Any, start_dt: datetime, end_dt: datetime, duration_minutes: int) -> list[tuple]:
    """Structured free windows (≥ duration) within work hours across [start,end]. Reuses freebusy.

    Shares the same free/busy algorithm as calendar_find_free_slots but returns (start, end) datetime
    tuples so the meeting-bridge tools can pick concrete slots. Returns [] on any error.
    """
    window_end = end_dt.replace(hour=23, minute=59, second=59, microsecond=0)
    try:
        fb = service.freebusy().query(body={
            "timeMin": start_dt.isoformat(), "timeMax": window_end.isoformat(),
            "items": [{"id": "primary"}],
        }).execute()
    except Exception:  # noqa: BLE001
        return []
    busy = []
    for b in fb.get("calendars", {}).get("primary", {}).get("busy", []) or []:
        bs, be = _cal_parse_dt(b.get("start")), _cal_parse_dt(b.get("end"))
        if bs and be:
            busy.append((bs, be))
    windows: list[tuple] = []
    days = (end_dt.date() - start_dt.date()).days
    for off in range(min(days, 31) + 1):
        day = start_dt.date() + timedelta(days=off)
        ds = max(datetime(day.year, day.month, day.day, _CALENDAR_WORK_START, 0).astimezone(), start_dt)
        de = min(datetime(day.year, day.month, day.day, _CALENDAR_WORK_END, 0).astimezone(), window_end)
        if de <= ds:
            continue
        day_busy = sorted((max(bs, ds), min(be, de)) for bs, be in busy if be > ds and bs < de)
        cursor = ds
        for bs, be in day_busy:
            if bs - cursor >= timedelta(minutes=duration_minutes):
                windows.append((cursor, bs))
            cursor = max(cursor, be)
        if de - cursor >= timedelta(minutes=duration_minutes):
            windows.append((cursor, de))
    return windows


def _meeting_options(windows: list[tuple], duration: int, num: int) -> list[tuple]:
    """Pick up to `num` concrete (start, end) options — one at the start of each free window."""
    opts: list[tuple] = []
    for ws, we in windows:
        end = ws + timedelta(minutes=duration)
        if end <= we:
            opts.append((ws, end))
        if len(opts) >= num:
            break
    return opts


@tool
def gmail_extract_meeting_request(message_id: str) -> str:
    """Analyze an email and extract proposed meeting details (heuristic, read-only).

    Reads the message and pulls out a likely title, duration, any dates/times mentioned in the body,
    attendee emails, and a short description. Nothing is created or sent — use this to see what
    Primus parsed before calling suggest_meeting_times_from_email / create_meeting_from_email.
    Provide a `message_id` from gmail_list_messages.
    """
    if not (message_id or "").strip():
        return "⚠ Provide a message_id (get one from gmail_list_messages)."
    service, err = _get_gmail_service()
    if err:
        return err
    fields, ferr = _read_email_fields(service, message_id.strip())
    if ferr:
        return ferr
    det = _extract_meeting_details(fields["subject"], fields["body"], fields["from"], fields["to"])
    times = "; ".join(dt.strftime("%a %b %-d %H:%M") for dt in det["suggested_times"]) or "none detected in email"
    return (
        "📋 Meeting request extracted:\n"
        f"  Title: {det['title']}\n"
        f"  Duration: {det['duration']} min\n"
        f"  Times mentioned in email: {times}\n"
        f"  Attendees: {', '.join(det['attendees']) or '(none found)'}\n"
        f"  Description: {det['description'][:200] or '(none)'}\n"
        "Next: suggest_meeting_times_from_email to find open slots, or create_meeting_from_email to book it."
    )


@tool
def suggest_meeting_times_from_email(message_id: str, duration_minutes: int = 30, days_ahead: int = 7) -> str:
    """Suggest 3 good open meeting times based on an email + your calendar (read-only).

    Combines the email (for title + any duration hint) with your calendar free/busy to return up to
    three concrete slots (≥ duration_minutes, within 09:00–17:00 local) over the next `days_ahead`
    days. Great for "when am I free to meet about this?". Provide a `message_id` from
    gmail_list_messages. Nothing is created or sent.
    """
    if not (message_id or "").strip():
        return "⚠ Provide a message_id (get one from gmail_list_messages)."
    gsvc, gerr = _get_gmail_service()
    if gerr:
        return gerr
    fields, ferr = _read_email_fields(gsvc, message_id.strip())
    if ferr:
        return ferr
    det = _extract_meeting_details(
        fields["subject"], fields["body"], fields["from"], fields["to"], default_duration=duration_minutes
    )
    dur = det["duration"] or int(duration_minutes or 30)
    csvc, cerr = _get_calendar_service()
    if cerr:
        return cerr
    now = datetime.now().astimezone()
    end = now + timedelta(days=max(1, min(int(days_ahead or 7), 31)))
    opts = _meeting_options(_free_windows(csvc, now, end, dur), dur, 3)
    if not opts:
        return f"No free {dur}-min slots in the next {days_ahead} day(s) for “{det['title']}”."
    lines = [f"🕒 Suggested times for “{det['title']}” ({dur} min):"]
    lines += [
        f"  {i}. {s.strftime('%A %b %-d, %H:%M')}–{e.strftime('%H:%M')}"
        for i, (s, e) in enumerate(opts, 1)
    ]
    lines.append(
        "Next: propose_times_in_reply to draft these to the sender, or "
        "create_meeting_from_email(start_time=...) to book one."
    )
    return "\n".join(lines)


@tool
def create_meeting_from_email(message_id: str, start_time: Optional[str] = None, duration_minutes: int = 30) -> str:
    """Create a calendar event from an email — auto-picking a free slot unless you give `start_time`.

    Reads the email for title/attendees/description, uses `start_time` ("YYYY-MM-DD HH:MM") if given
    or otherwise the next free slot this week, creates the event (inviting the sender + any attendees)
    and drafts a confirmation reply (unsent). SAFETY: like calendar_create_event / gmail_send_email,
    this only PREVIEWS in Suggest mode and acts in Execute mode (or once approved). Provide a
    `message_id` from gmail_list_messages.
    """
    if not (message_id or "").strip():
        return "⚠ Provide a message_id (get one from gmail_list_messages)."
    gsvc, gerr = _get_gmail_service()
    if gerr:
        return gerr
    fields, ferr = _read_email_fields(gsvc, message_id.strip())
    if ferr:
        return ferr
    det = _extract_meeting_details(
        fields["subject"], fields["body"], fields["from"], fields["to"], default_duration=duration_minutes
    )
    dur = det["duration"] or int(duration_minutes or 30)

    # Resolve the start time: explicit arg, else the next free slot this week.
    if start_time and str(start_time).strip():
        start = _cal_parse_dt(str(start_time))
        if not start:
            return "⚠ Could not parse start_time — use 'YYYY-MM-DD HH:MM' (or ISO 8601)."
    else:
        csvc, cerr = _get_calendar_service()
        if cerr:
            return cerr
        now = datetime.now().astimezone()
        opts = _meeting_options(_free_windows(csvc, now, now + timedelta(days=7), dur), dur, 1)
        if not opts:
            return f"No free {dur}-min slot found this week — pass start_time explicitly."
        start = opts[0][0]
    end = start + timedelta(minutes=dur)
    attendees = ",".join(det["attendees"])

    # Combined safety preview (mirrors calendar_create_event / gmail_send_email).
    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return (
            "[Suggest] Would create meeting + draft a confirmation:\n"
            f"  Title: {det['title']}\n"
            f"  When: {start.strftime('%Y-%m-%d %H:%M')}–{end.strftime('%H:%M')}\n"
            f"  Attendees: {attendees or '(none)'}\n"
            "Switch to Execute (or approve) to create the event and draft the confirmation."
        )

    ev_res = _run_tool(
        calendar_create_event, summary=det["title"],
        start_time=start.strftime("%Y-%m-%d %H:%M"), end_time=end.strftime("%Y-%m-%d %H:%M"),
        description=det["description"], attendees=attendees,
    )
    # Draft a confirmation reply to the sender (unsent — always safe).
    draft_note = ""
    sender_emails = _extract_emails(fields["from"])
    if sender_emails:
        body = (
            f"Hi,\n\nConfirming our meeting: {det['title']} on "
            f"{start.strftime('%A %b %-d at %H:%M')} ({dur} min). A calendar invite has been sent.\n\n"
            + _email_signoff()
        )
        draft_note = "\n" + _run_tool(
            gmail_create_draft, to=sender_emails[0],
            subject=f"Re: {fields['subject'] or det['title']}", body=body,
        )
    return f"{ev_res}{draft_note}"


@tool
def propose_times_in_reply(message_id: str, num_options: int = 3) -> str:
    """Draft (unsent) a reply proposing your free times for a meeting email.

    Finds up to `num_options` open slots (≥ 30 min, or the email's hinted duration) over the next
    week and writes a polite reply to the sender listing them, saved to Drafts for review — nothing
    is sent. Provide a `message_id` from gmail_list_messages. Send it later with
    gmail_send_email(draft_id=...) or from Gmail.
    """
    if not (message_id or "").strip():
        return "⚠ Provide a message_id (get one from gmail_list_messages)."
    gsvc, gerr = _get_gmail_service()
    if gerr:
        return gerr
    fields, ferr = _read_email_fields(gsvc, message_id.strip())
    if ferr:
        return ferr
    det = _extract_meeting_details(fields["subject"], fields["body"], fields["from"], fields["to"])
    dur = det["duration"] or 30
    sender = _extract_emails(fields["from"])
    if not sender:
        return "⚠ Could not determine the sender's email to reply to."
    csvc, cerr = _get_calendar_service()
    if cerr:
        return cerr
    now = datetime.now().astimezone()
    n = max(1, min(int(num_options or 3), 5))
    opts = _meeting_options(_free_windows(csvc, now, now + timedelta(days=7), dur), dur, n)
    if not opts:
        return f"No free {dur}-min slots in the next week to propose."
    times_txt = "\n".join(
        f"  • {s.strftime('%A %b %-d, %H:%M')}–{e.strftime('%H:%M')}" for s, e in opts
    )
    body = (
        f"Hi,\n\nThanks for reaching out about {det['title']}. Here are a few times that work for me "
        f"({dur} minutes):\n\n{times_txt}\n\nLet me know which works best and I'll send an invite.\n\n"
        + _email_signoff()
    )
    draft = _run_tool(
        gmail_create_draft, to=sender[0],
        subject=f"Re: {fields['subject'] or det['title']}", body=body,
    )
    return f"✍ Draft reply proposing {len(opts)} time(s) created (unsent):\n{draft}"


# ===========================================================================
# Morning Briefing (scheduled executive summary built from Gmail + Calendar)
# ---------------------------------------------------------------------------
# A deterministic composer that pulls today's calendar, unread email, weather and headlines and
# assembles a concise, executive-style briefing — WITHOUT relying on the model (so it's fast and
# never derails). It takes NO actions (never sends email or creates events), so it's identical in
# Suggest and Execute mode. schedule_morning_briefing registers it as a daily ScheduledTaskManager
# task that runs through the normal background-agent path and posts to General Chat.
# ===========================================================================

_MORNING_BRIEFING_NAME = "Morning Briefing"

# Emails worth flagging for a reply vs. automated noise to ignore.
_REPLY_HINT_RE = re.compile(
    r"\?|\b(please|can you|could you|would you|let me know|thoughts|confirm|review|approve|"
    r"available|schedule|when (?:are|can|is)|rsvp|get back|follow ?up|reply|respond|question|"
    r"need(?:ed)? (?:your|by)|waiting on|deadline|urgent|asap)\b",
    re.IGNORECASE,
)
_AUTOMATED_RE = re.compile(
    r"\b(no-?reply|do-?not-?reply|notification|newsletter|unsubscribe|receipt|"
    r"order (?:confirmation|shipped)|verify your|password reset|digest)\b",
    re.IGNORECASE,
)


def _run_tool(tool_obj: Any, **kwargs: Any) -> str:
    """Invoke a registered @tool from inside Python, robust across langchain / stub modes."""
    try:
        if hasattr(tool_obj, "invoke"):
            return str(tool_obj.invoke(kwargs))
        fn = getattr(tool_obj, "func", None)
        if fn is not None:
            return str(fn(**kwargs))
        return str(tool_obj(**kwargs))
    except Exception as exc:  # noqa: BLE001
        return f"(unavailable: {str(exc)[:80]})"


def _sender_display(sender: str) -> str:
    """Best-effort human name from an RFC From header ('Jane Doe <jane@x.com>' → 'Jane Doe')."""
    m = re.match(r'\s*"?([^"<]+?)"?\s*<', sender or "")
    return (m.group(1).strip() if m else (sender or "").strip())


def _email_needs_reply(sender: str, subject: str, snippet: str) -> bool:
    """Heuristic: does this unread email likely need a human reply (vs. automated noise)?"""
    if _AUTOMATED_RE.search(f"{sender}\n{subject}\n{snippet}"):
        return False
    return bool(_REPLY_HINT_RE.search(f"{subject} {snippet}"))


def _reply_context(sender: str) -> str:
    """One-line hint about prior interactions with a sender, from long-term memory (best-effort)."""
    try:
        name = _sender_display(sender)
        if not name:
            return ""
        hits = _host.get_memory_system().recall_ltm(name, k=1)
        if not hits:
            return ""
        h = hits[0]
        txt = ""
        for attr in ("page_content", "text", "content"):
            txt = getattr(h, attr, "") or (h.get(attr, "") if isinstance(h, dict) else "")
            if txt:
                break
        txt = re.sub(r"\s+", " ", txt or str(h)).strip()
        return f"prior context: {txt[:80]}" if txt else ""
    except Exception:  # noqa: BLE001
        return ""


def _briefing_detect_conflicts(timed: list[tuple]) -> list[str]:
    """From [(start, end, title)] find overlapping events and uncomfortably tight (<15m) gaps."""
    out: list[str] = []
    ev = sorted(timed, key=lambda x: x[0])
    for i in range(1, len(ev)):
        ps, pe, pn = ev[i - 1]
        cs, ce, cn = ev[i]
        if cs < pe:
            out.append(f"“{pn[:30]}” overlaps “{cn[:30]}” at {cs.strftime('%H:%M')}")
        else:
            gap = int((cs - pe).total_seconds() // 60)
            if gap < 15:
                out.append(
                    f"Only {gap}m between “{pn[:25]}” and “{cn[:25]}” "
                    f"({pe.strftime('%H:%M')}→{cs.strftime('%H:%M')})"
                )
    return out


def _briefing_first_free_block(timed: list[tuple], day, now: datetime, min_minutes: int = 30) -> str:
    """First free block today ≥ min_minutes within work hours, starting no earlier than now."""
    ws = datetime(day.year, day.month, day.day, _CALENDAR_WORK_START, 0).astimezone()
    we = datetime(day.year, day.month, day.day, _CALENDAR_WORK_END, 0).astimezone()
    cursor = max(ws, now)
    if cursor >= we:
        return ""
    busy = sorted((max(s, ws), min(e, we)) for s, e, _ in timed if e > ws and s < we)
    for bs, be in busy:
        if bs - cursor >= timedelta(minutes=min_minutes):
            return f"{cursor.strftime('%H:%M')}–{bs.strftime('%H:%M')}"
        cursor = max(cursor, be)
    if we - cursor >= timedelta(minutes=min_minutes):
        return f"{cursor.strftime('%H:%M')}–{we.strftime('%H:%M')}"
    return ""


def _comms_health_states() -> list[tuple[str, bool, str]]:
    """(label, ok, detail) health for each CONFIGURED comms account (Gmail/Calendar/Slack).

    Only reports on services whose token file exists, so a service you never set up isn't flagged
    as broken. Used by the Morning Briefing banner and the UI status dashboard.
    """
    out: list[tuple[str, bool, str]] = []
    for label, ctype, path in (
        ("Gmail", "gmail", _GMAIL_TOKEN_PATH),
        ("Calendar", "calendar", _CALENDAR_TOKEN_PATH),
        ("Slack", "slack", _SLACK_TOKEN_PATH),
    ):
        if not path.exists():
            continue
        try:
            ok, detail = _conn_health({"type": ctype, "token_path": str(path)})
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, str(exc)[:80]
        out.append((label, ok, detail))
    return out


def _briefing_health_banner() -> list[str]:
    """Warning block for any configured account that's currently expired/failed (empty if all OK)."""
    try:
        states = _comms_health_states()
    except Exception:  # noqa: BLE001
        return []
    bad = [(label, detail) for label, ok, detail in states if not ok]
    if not bad:
        return []
    lines = ["**⚠ Account health — action needed**"]
    lines += [f"  • {label}: {detail}" for label, detail in bad]
    lines += [
        "  _Sections below may be incomplete. Run `connections_refresh`, or re-auth "
        "(gmail_auth / calendar_auth / slack_auth)._",
        "",
    ]
    return lines


def _compose_morning_briefing(max_emails: int = 8, include_news: bool = True) -> str:
    """Assemble the full executive briefing deterministically from the Gmail + Calendar APIs.

    Every section degrades gracefully: an auth/API problem shows a short note instead of aborting.
    Any expired/failed account is flagged in a banner at the very top so nothing fails silently.
    """
    now = datetime.now().astimezone()
    today = now.date()
    _host.PrimusSession.emit_think("Briefing", "Composing morning briefing", "running")
    lines: list[str] = [f"# ☀️ Morning Briefing — {now.strftime('%A, %B %-d, %Y')}", ""]
    lines += _briefing_health_banner()  # flag any expired/failed accounts up top

    # --- Weather (text tool) ---
    weather = _run_tool(get_weather).strip()
    if weather and not weather.startswith("("):
        lines += ["**Weather**", weather, ""]

    # --- Today's calendar (structured, for conflict + free-slot intelligence) ---
    cal_service, cal_err = _get_calendar_service()
    timed: list[tuple] = []
    if cal_err:
        lines += ["**Today's schedule**", cal_err, ""]
    else:
        try:
            day_start = datetime(today.year, today.month, today.day, 0, 0).astimezone()
            day_end = datetime(today.year, today.month, today.day, 23, 59, 59).astimezone()
            resp = _api_retry(
                lambda: cal_service.events().list(
                    calendarId="primary", timeMin=day_start.isoformat(), timeMax=day_end.isoformat(),
                    singleEvents=True, orderBy="startTime",
                ).execute(),
                label="briefing.calendar",
            )
            todays = resp.get("items", []) or []
        except Exception as exc:  # noqa: BLE001
            todays = []
            lines += ["**Today's schedule**", f"⚠ Could not load events: {str(exc)[:100]}", ""]
        else:
            if not todays:
                lines += ["**Today's schedule**", "Nothing scheduled — a clear day.", ""]
            else:
                ev_lines: list[str] = []
                for ev in todays:
                    s, e = ev.get("start", {}) or {}, ev.get("end", {}) or {}
                    title = ev.get("summary", "(no title)")
                    if s.get("date") and not s.get("dateTime"):
                        ev_lines.append(f"  • All day — {title[:60]}")
                        continue
                    sdt, edt = _cal_parse_dt(s.get("dateTime")), _cal_parse_dt(e.get("dateTime"))
                    when = f"{sdt.strftime('%H:%M') if sdt else '?'}–{edt.strftime('%H:%M') if edt else '?'}"
                    loc = ev.get("location", "")
                    ev_lines.append(f"  • {when} {title[:55]}" + (f"  @ {loc[:30]}" if loc else ""))
                    if sdt and edt:
                        timed.append((sdt, edt, title))
                lines += [f"**Today's schedule** ({len(todays)} event(s))", *ev_lines, ""]
                conflicts = _briefing_detect_conflicts(timed)
                if conflicts:
                    lines += ["**⚠ Conflicts / tight timing**", *[f"  • {c}" for c in conflicts], ""]
                free = _briefing_first_free_block(timed, today, now)
                if free:
                    lines += [f"**First free block:** {free} (local)", ""]

    # --- Unread email in the last 24h (structured, with reply-need flagging) ---
    gmail_service, gm_err = _get_gmail_service()
    if gm_err:
        lines += ["**Inbox**", gm_err, ""]
    else:
        max_emails = max(1, min(int(max_emails or 8), 20))
        try:
            resp = _api_retry(
                lambda: gmail_service.users().messages().list(
                    userId="me", q="is:unread newer_than:1d", maxResults=max_emails,
                ).execute(),
                label="briefing.gmail",
            )
            msgs = resp.get("messages", []) or []
        except Exception as exc:  # noqa: BLE001
            msgs = []
            lines += ["**Inbox**", f"⚠ Could not load email: {str(exc)[:100]}", ""]
        else:
            if not msgs:
                lines += ["**Inbox**", "No unread email in the last 24h — inbox zero. 🎉", ""]
            else:
                mail_lines: list[str] = []
                needs_reply: list[str] = []
                for m in msgs:
                    try:
                        meta = gmail_service.users().messages().get(
                            userId="me", id=m["id"], format="metadata",
                            metadataHeaders=["From", "Subject"],
                        ).execute()
                    except Exception:  # noqa: BLE001
                        continue
                    hdr = {h["name"]: h["value"] for h in meta.get("payload", {}).get("headers", [])}
                    frm, subj = hdr.get("From", "?"), hdr.get("Subject", "(no subject)")
                    snippet = meta.get("snippet", "")
                    mail_lines.append(f"  • {_sender_display(frm)[:32]} — {subj[:55]}")
                    if _email_needs_reply(frm, subj, snippet) and len(needs_reply) < 3:
                        ctx = _reply_context(frm)
                        needs_reply.append(
                            f"{_sender_display(frm)[:28]} — “{subj[:45]}”" + (f"  ({ctx})" if ctx else "")
                        )
                lines += [f"**Unread email** ({len(msgs)} in last 24h)", *mail_lines, ""]
                if needs_reply:
                    lines += ["**✍ Likely needs a reply**", *[f"  • {r}" for r in needs_reply], ""]

    # --- Top headlines ---
    if include_news:
        news = _run_tool(get_news, limit=5).strip()
        if news and not news.startswith("("):
            lines += ["**Top headlines**", news, ""]

    lines += ["---", f"_Have a productive day{_OPERATOR['vocative']}._"]
    _host.PrimusSession.emit_think("Briefing", "Briefing ready", "done")
    return "\n".join(lines)


@tool
def morning_briefing(include_news: bool = True, max_emails: int = 8) -> str:
    """Generate the operator's executive Morning Briefing right now (read-only — safe in any mode).

    Pulls today's calendar, unread email from the last 24h, current weather and top headlines, then
    flags schedule conflicts / tight gaps, the first free block of the day, and emails that likely
    need a reply (with any prior context from long-term memory). Built deterministically from the
    Gmail + Calendar APIs so it's fast and consistent — it does NOT rely on the model and takes NO
    actions (never sends email or creates events), so output is identical in Suggest and Execute
    mode. Requires gmail_auth + calendar_auth; missing pieces degrade gracefully with a short note.
    Use schedule_morning_briefing to run this automatically every morning.
    """
    try:
        return _compose_morning_briefing(max_emails=max_emails, include_news=include_news)
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Could not build morning briefing: {str(exc)[:160]}"


@tool
def schedule_morning_briefing(time: str = "07:00", enabled: bool = True) -> str:
    """Schedule the Morning Briefing to run automatically every day at `time` (HH:MM 24h, default 07:00).

    Registers a daily scheduled task (via Primus's ScheduledTaskManager) that runs the
    morning_briefing tool and posts the result to General Chat, saving a copy under
    ~/.primus/agent_outputs/. Re-running just updates the existing briefing's time/enabled state
    instead of creating duplicates. Requires Gmail + Calendar authorization (gmail_auth /
    calendar_auth). Setup: authorize both once, then call this. Pass enabled=False to pause it.
    """
    try:
        parts = (time or "07:00").strip().split(":")
        h = max(0, min(int(parts[0]), 23))
        m = max(0, min(int(parts[1]), 59))
    except Exception:  # noqa: BLE001
        return "⚠ Provide time as HH:MM (24-hour), e.g. 07:00."

    mgr = _host.ScheduledTaskManager
    instructions = (
        "Call the morning_briefing tool (include_news=True) and post its output VERBATIM as the operator's "
        "executive briefing for today. Do not add commentary, do not call any other tools, and do "
        "not take any actions."
    )
    hhmm = f"{h:02d}:{m:02d}"

    # Update the existing briefing instead of creating duplicates.
    existing = next((t for t in mgr.list() if t.get("name") == _MORNING_BRIEFING_NAME), None)
    if existing:
        mgr.update(
            existing["id"], schedule_type="daily", time_of_day=hhmm,
            enabled=enabled, instructions=instructions,
        )
        mgr.ensure_scheduler_started()
        state = "" if enabled else " (currently paused)"
        return f"✓ Updated Morning Briefing → daily at {hhmm}{state}. (task {existing['id']})"

    task = mgr.create(
        name=_MORNING_BRIEFING_NAME, instructions=instructions, schedule_type="daily",
        time_of_day=hhmm, enabled=enabled, project_id=_host.GENERAL_CHAT_ID,
    )
    if not task:
        return "⚠ Could not create the scheduled briefing."
    mgr.ensure_scheduler_started()
    return (
        f"✓ Morning Briefing scheduled — daily at {hhmm}. It posts to General Chat and saves to "
        f"~/.primus/agent_outputs/. Make sure Gmail + Calendar are authorized (gmail_auth / "
        f"calendar_auth). (task {task['id']})"
    )


# ===========================================================================
# Slack integration (Slack Web API via slack-sdk)
# ---------------------------------------------------------------------------
# Read channels/threads, summarize + extract action items, and post messages. Tokens live locally
# (~/.primus/slack_token.json: {"bot_token": "xoxb-…", "user_token": "xoxp-…"}) or via the
# SLACK_BOT_TOKEN / SLACK_USER_TOKEN env vars. The slack-sdk library is optional — every tool
# degrades gracefully with setup instructions if it's missing or unauthorized. Posting is gated by
# the same safety model as gmail_send_email (preview in Suggest mode). Summaries/action items are
# extracted deterministically (no model dependency) for speed + consistency.
# ===========================================================================

_SLACK_TOKEN_PATH = _host.APP_DIR / "slack_token.json"
_SLACK_MISSING_LIBS = "⚠ Slack SDK not installed. Run:\n  uv pip install slack-sdk"

# Message lines that look like commitments / requests → candidate action items.
_SLACK_ACTION_RE = re.compile(
    r"\b(action item|todo|to-?do|follow ?up|need(?:s)? to|have to|please|can you|could you|"
    r"would you|will (?:you|we|i)\b|i'?ll\b|we'?ll\b|let'?s\b|make sure|don'?t forget|assign(?:ed)? to|"
    r"by (?:eod|eow|cob|tomorrow|monday|tuesday|wednesday|thursday|friday|next week|\d{1,2}/\d{1,2}))\b",
    re.IGNORECASE,
)


def _load_slack_tokens() -> dict:
    """Slack tokens from ~/.primus/slack_token.json, with SLACK_BOT_TOKEN/SLACK_USER_TOKEN env fallback."""
    toks: dict = {}
    try:
        if _SLACK_TOKEN_PATH.exists():
            data = json.loads(_SLACK_TOKEN_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                toks.update(data)
    except (OSError, json.JSONDecodeError):
        pass
    if not toks.get("bot_token") and os.environ.get("SLACK_BOT_TOKEN"):
        toks["bot_token"] = os.environ["SLACK_BOT_TOKEN"]
    if not toks.get("user_token") and os.environ.get("SLACK_USER_TOKEN"):
        toks["user_token"] = os.environ["SLACK_USER_TOKEN"]
    return toks


def _get_slack_client(prefer: str = "bot") -> tuple[Any, str]:
    """Return (WebClient, "") on success, else (None, user-facing error/instructions).

    Prefers the bot token (`prefer="bot"`, default) or the user token (`prefer="user"`), falling back
    to the other if only one is configured. Mirrors _get_gmail_service's graceful-degradation style.
    """
    try:
        from slack_sdk import WebClient
    except ImportError:
        return None, _SLACK_MISSING_LIBS
    toks = _load_slack_tokens()
    if prefer == "user":
        token = toks.get("user_token") or toks.get("bot_token")
    else:
        token = toks.get("bot_token") or toks.get("user_token")
    if not token:
        return None, (
            "⚠ Slack isn't authorized yet. Run the `slack_auth` tool for setup, then save your token "
            f"to `{_SLACK_TOKEN_PATH}` as {{\"bot_token\": \"xoxb-…\"}} (or set SLACK_BOT_TOKEN)."
        )
    try:
        return WebClient(token=token), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"⚠ Could not create Slack client: {str(exc)[:140]}"


def _slack_err(exc: Exception) -> str:
    """Best-effort Slack API error string (unwraps SlackApiError's response['error'])."""
    try:
        resp = getattr(exc, "response", None)
        if resp is not None and resp.get("error"):
            return str(resp["error"])
    except Exception:  # noqa: BLE001
        pass
    return str(exc)[:140]


def _slack_fmt_ts(ts: str) -> str:
    """Slack message ts ('1720000000.001') → 'YYYY-MM-DD HH:MM' local (best-effort)."""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError, OSError):
        return "?"


def _slack_user_name(client: Any, uid: str, cache: dict) -> str:
    """Resolve a Slack user id → display name, cached per call to limit API hits."""
    if not uid:
        return "?"
    if uid in cache:
        return cache[uid]
    name = uid
    try:
        u = client.users_info(user=uid).get("user", {}) or {}
        name = u.get("real_name") or u.get("name") or uid
    except Exception:  # noqa: BLE001
        pass
    cache[uid] = name
    return name


def _slack_resolve_channel(client: Any, ref: str) -> tuple[str, str]:
    """Resolve a channel id or '#name' → (channel_id, "") or ("", error)."""
    ref = (ref or "").strip().lstrip("#")
    if not ref:
        return "", "⚠ Provide a channel id or name."
    if re.match(r"^[CGD][A-Z0-9]{6,}$", ref):
        return ref, ""
    try:
        cursor = None
        for _ in range(10):  # paginate up to ~2000 channels
            resp = client.conversations_list(
                types="public_channel,private_channel", limit=200, cursor=cursor,
            )
            for c in resp.get("channels", []) or []:
                if c.get("name") == ref:
                    return c["id"], ""
            cursor = (resp.get("response_metadata", {}) or {}).get("next_cursor")
            if not cursor:
                break
    except Exception as exc:  # noqa: BLE001
        return "", f"⚠ Could not resolve channel '{ref}': {_slack_err(exc)}"
    return "", f"⚠ Channel '{ref}' not found (or the bot isn't a member of it)."


def _slack_action_items(messages: list[dict]) -> list[str]:
    """Extract candidate action items from messages ([{user_name, text}]) → ['name: line', …]."""
    items: list[str] = []
    for msg in messages:
        who = msg.get("user_name", "?")
        for line in re.split(r"[\n.!?]+", msg.get("text", "") or ""):
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) >= 4 and _SLACK_ACTION_RE.search(line):
                entry = f"{who}: {line[:140]}"
                if entry not in items:
                    items.append(entry)
    return items[:20]


@tool
def slack_auth() -> str:
    """Set up Slack access for Primus (create a Slack app + Bot token — one-time).

    SETUP (https://api.slack.com/apps):
      1. **Create New App** → From scratch → pick your workspace.
      2. **OAuth & Permissions** → under *Bot Token Scopes* add:
         `channels:read`, `groups:read`, `channels:history`, `groups:history`,
         `chat:write`, `users:read`  (add `search:read` on a *User* token if you want search).
      3. Click **Install to Workspace** and authorize.
      4. Copy the **Bot User OAuth Token** (starts with `xoxb-`).
      5. Save it locally as `~/.primus/slack_token.json`:
             {"bot_token": "xoxb-your-token"}
         (optionally add "user_token": "xoxp-…" for user-scoped reads). Or set the SLACK_BOT_TOKEN
         environment variable instead.
      6. **Invite the bot to channels** you want it to read/post in: `/invite @YourApp`.
      7. Run this tool again to confirm, then use slack_status.

    Safe to re-run: if a token is already present it just validates the connection.
    """
    try:
        from slack_sdk import WebClient  # noqa: F401
    except ImportError:
        return _SLACK_MISSING_LIBS
    toks = _load_slack_tokens()
    if not (toks.get("bot_token") or toks.get("user_token")):
        return (
            "⚠ No Slack token found. Follow the setup in this tool's description:\n"
            "  1. Create a Slack app at https://api.slack.com/apps\n"
            "  2. Add bot scopes: channels:read, groups:read, channels:history, groups:history, "
            "chat:write, users:read\n"
            "  3. Install to workspace and copy the Bot User OAuth Token (xoxb-…)\n"
            f"  4. Save it as `{_SLACK_TOKEN_PATH}` → {{\"bot_token\": \"xoxb-…\"}} "
            "(or set SLACK_BOT_TOKEN)\n"
            "  5. Invite the bot to your channels, then run slack_auth again."
        )
    client, err = _get_slack_client()
    if err:
        return err
    try:
        info = client.auth_test()
        _conn_note_auth("slack")  # register with the central Connection Manager (non-fatal)
        return (
            f"✓ Slack authorized — workspace **{info.get('team', '?')}** as "
            f"{info.get('user', '?')}. Use slack_status or slack_list_channels."
        )
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Slack token present but validation failed: {_slack_err(exc)}. Check scopes/reinstall."


@tool
def slack_status() -> str:
    """Check the Slack connection and show workspace + bot identity.

    Returns the authorized workspace name, URL and bot user, or clear guidance if the SDK is missing
    or no token is configured (run slack_auth).
    """
    client, err = _get_slack_client()
    if err:
        return err
    try:
        info = client.auth_test()
        return (
            f"✓ Slack connected — workspace **{info.get('team', '?')}** "
            f"({info.get('url', '?')}) as {info.get('user', '?')} (id: {info.get('user_id', '?')})."
        )
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Slack status check failed: {_slack_err(exc)}"


@tool
def slack_list_channels(limit: int = 20) -> str:
    """List Slack channels the bot can see (public + private), with id, name and member count.

    `limit` is clamped to 1–200. Use a returned channel id (or #name) with slack_read_channel /
    slack_post_message. Only channels the bot is a member of are readable — `/invite @YourApp` first.
    """
    client, err = _get_slack_client()
    if err:
        return err
    limit = max(1, min(int(limit or 20), 200))
    try:
        resp = client.conversations_list(
            types="public_channel,private_channel", limit=limit, exclude_archived=True,
        )
        chans = resp.get("channels", []) or []
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Slack channel list failed: {_slack_err(exc)}"
    if not chans:
        return "No channels visible to the bot. Invite it with `/invite @YourApp` in a channel."
    lines = []
    for c in chans:
        topic = (c.get("topic", {}) or {}).get("value", "")
        priv = "🔒" if c.get("is_private") else "#"
        lines.append(
            f"• [{c.get('id')}] {priv}{c.get('name', '?')} ({c.get('num_members', '?')} members)"
            + (f" — {topic[:50]}" if topic else "")
        )
    return f"💬 {len(chans)} channel(s):\n\n" + "\n".join(lines)


@tool
def slack_read_channel(channel_id_or_name: str, limit: int = 20) -> str:
    """Read the most recent messages in a Slack channel (newest last), with sender + timestamp.

    Accepts a channel id or '#name'. `limit` is clamped to 1–100. Thread replies aren't expanded —
    use slack_read_thread on a message's ts for the full thread. The bot must be a member.
    """
    client, err = _get_slack_client()
    if err:
        return err
    cid, cerr = _slack_resolve_channel(client, channel_id_or_name)
    if cerr:
        return cerr
    limit = max(1, min(int(limit or 20), 100))
    _host.PrimusSession.emit_think("Slack", f"Reading {channel_id_or_name}", "running")
    try:
        resp = client.conversations_history(channel=cid, limit=limit)
        msgs = list(reversed(resp.get("messages", []) or []))  # oldest → newest for readability
    except Exception as exc:  # noqa: BLE001
        _host.PrimusSession.emit_think("Slack", "Read failed", "error")
        return f"⚠ Could not read channel: {_slack_err(exc)}"
    if not msgs:
        return "No messages in that channel."
    cache: dict = {}
    lines = []
    for m in msgs:
        who = _slack_user_name(client, m.get("user") or m.get("bot_id", ""), cache)
        text = re.sub(r"\s+", " ", m.get("text", "") or "").strip()
        reply_n = m.get("reply_count", 0)
        tail = f"  ↳ {reply_n} repl{'y' if reply_n == 1 else 'ies'} (thread ts {m.get('ts')})" if reply_n else ""
        lines.append(f"[{_slack_fmt_ts(m.get('ts', ''))}] {who}: {text[:200]}{tail}")
    _host.PrimusSession.emit_think("Slack", f"{len(msgs)} message(s)", "done")
    return f"💬 #{channel_id_or_name.lstrip('#')} — last {len(msgs)} message(s):\n\n" + "\n".join(lines)


@tool
def slack_read_thread(channel_id: str, thread_ts: str) -> str:
    """Read a full Slack thread (all replies) given the channel and the thread's root ts.

    Get `thread_ts` from slack_read_channel (shown as 'thread ts …' on messages with replies).
    Returns every message in order with sender + timestamp.
    """
    if not (thread_ts or "").strip():
        return "⚠ Provide the thread_ts (root timestamp of the thread)."
    client, err = _get_slack_client()
    if err:
        return err
    cid, cerr = _slack_resolve_channel(client, channel_id)
    if cerr:
        return cerr
    try:
        resp = client.conversations_replies(channel=cid, ts=thread_ts.strip(), limit=200)
        msgs = resp.get("messages", []) or []
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Could not read thread: {_slack_err(exc)}"
    if not msgs:
        return "No messages found in that thread."
    cache: dict = {}
    lines = []
    for m in msgs:
        who = _slack_user_name(client, m.get("user") or m.get("bot_id", ""), cache)
        text = re.sub(r"\s+", " ", m.get("text", "") or "").strip()
        lines.append(f"[{_slack_fmt_ts(m.get('ts', ''))}] {who}: {text[:300]}")
    return f"🧵 Thread ({len(msgs)} message(s)):\n\n" + "\n".join(lines)


@tool
def slack_summarize_thread(channel_id: str, thread_ts: str) -> str:
    """Summarize a Slack thread: participants, the opening + latest message, and action items.

    Deterministic (no model needed): reads the full thread, lists participants, quotes the root and
    most recent messages, and extracts likely action items. Provide the channel (id or #name) and the
    thread's root ts (from slack_read_channel).
    """
    if not (thread_ts or "").strip():
        return "⚠ Provide the thread_ts (root timestamp of the thread)."
    client, err = _get_slack_client()
    if err:
        return err
    cid, cerr = _slack_resolve_channel(client, channel_id)
    if cerr:
        return cerr
    try:
        resp = client.conversations_replies(channel=cid, ts=thread_ts.strip(), limit=200)
        msgs = resp.get("messages", []) or []
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Could not read thread: {_slack_err(exc)}"
    if not msgs:
        return "No messages found in that thread."
    cache: dict = {}
    for m in msgs:
        m["user_name"] = _slack_user_name(client, m.get("user") or m.get("bot_id", ""), cache)
    participants = list(dict.fromkeys(m["user_name"] for m in msgs))
    root, last = msgs[0], msgs[-1]
    root_text = re.sub(r"\s+", " ", root.get("text", "") or "")[:180]
    last_text = re.sub(r"\s+", " ", last.get("text", "") or "")[:180]
    lines = [
        f"🧵 Thread summary — {len(msgs)} message(s), {len(participants)} participant(s): "
        f"{', '.join(participants[:8])}",
        f"Started by {root['user_name']}: “{root_text}”",
    ]
    if len(msgs) > 1:
        lines.append(f"Latest from {last['user_name']}: “{last_text}”")
    items = _slack_action_items(msgs)
    if items:
        lines += ["", "**Action items:**", *[f"  • {it}" for it in items]]
    else:
        lines.append("No clear action items detected.")
    return "\n".join(lines)


@tool
def slack_extract_action_items(channel_id: str, thread_ts: Optional[str] = None) -> str:
    """Extract action items from a Slack thread (if thread_ts given) or a channel's recent messages.

    Scans messages for commitments/requests ("please", "can you", "I'll", "TODO", "by Friday", …)
    and lists them with who said them. Deterministic — no model needed. Provide the channel (id or
    #name); pass thread_ts to focus on one thread, or omit it to scan the last ~50 channel messages.
    """
    client, err = _get_slack_client()
    if err:
        return err
    cid, cerr = _slack_resolve_channel(client, channel_id)
    if cerr:
        return cerr
    try:
        if thread_ts and thread_ts.strip():
            resp = client.conversations_replies(channel=cid, ts=thread_ts.strip(), limit=200)
        else:
            resp = client.conversations_history(channel=cid, limit=50)
        msgs = resp.get("messages", []) or []
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Could not read messages: {_slack_err(exc)}"
    if not msgs:
        return "No messages to scan."
    cache: dict = {}
    for m in msgs:
        m["user_name"] = _slack_user_name(client, m.get("user") or m.get("bot_id", ""), cache)
    items = _slack_action_items(msgs)
    if not items:
        return "No action items detected in the scanned messages."
    return f"✅ {len(items)} action item(s) found:\n\n" + "\n".join(f"  • {it}" for it in items)


@tool
def slack_post_message(channel_id_or_name: str, text: str, thread_ts: Optional[str] = None) -> str:
    """Post a message to a Slack channel (or as a thread reply if `thread_ts` is given).

    SAFETY: like gmail_send_email, this only PREVIEWS in Suggest mode and actually posts in Execute
    mode (or once approved). Accepts a channel id or '#name'; the bot must be a member and have the
    chat:write scope. Set `thread_ts` to reply within an existing thread.
    """
    if not (text or "").strip():
        return "⚠ Provide the message text to post."

    # Preview-only in Suggest mode (mirrors gmail_send_email exactly).
    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        where = f"#{channel_id_or_name.lstrip('#')}" + (f" (thread {thread_ts})" if thread_ts else "")
        preview = text[:200] + ("…" if len(text) > 200 else "")
        return (
            f"[Suggest] Would post to {where}:\n  {preview}\n"
            "Switch to Execute (or approve) to actually post."
        )

    client, err = _get_slack_client()
    if err:
        return err
    cid, cerr = _slack_resolve_channel(client, channel_id_or_name)
    if cerr:
        return cerr
    try:
        kwargs: dict[str, Any] = {"channel": cid, "text": text}
        if thread_ts and thread_ts.strip():
            kwargs["thread_ts"] = thread_ts.strip()
        resp = client.chat_postMessage(**kwargs)
        return f"✓ Posted to #{channel_id_or_name.lstrip('#')} (ts {resp.get('ts', '?')})."
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Slack post failed: {_slack_err(exc)}"


# ===========================================================================
# Auto-ingest — pull Gmail / Calendar / Slack into the Knowledge Base
# ---------------------------------------------------------------------------
# Read-only w.r.t. the external services (never sends/posts): scans each source, scores/filters what
# matters, tags it (source:*, project:*, channel:*, from:*), and stores it via the existing KB
# pipeline (get_kb().learn_text for message/event text, ingest_uploaded_files for attachments).
# Safe to run anytime and from the Morning Briefing / scheduled agents.
#
# How to use auto-ingest:
#   • "ingest my important emails":     auto_ingest_gmail("is:unread", 10, 0.6)
#   • "save this week's calendar":       auto_ingest_calendar(7)
#   • "capture #client-acme":            auto_ingest_slack(["#client-acme"], 20)
#   • "sync everything into my KB":      auto_ingest_all()
# Requires the relevant integration to be authorized (gmail_auth / calendar_auth / slack_auth);
# any unauthorized source is skipped gracefully.
# ===========================================================================

# The operator's known projects/clients — used (plus live ChatProjects) to auto-tag ingested
# content. Comes from ~/.config/primus/operator.json ("projects"); empty in a stock clone.
_KNOWN_PROJECTS = list(_OPERATOR["projects"])
_IMPORTANCE_KEYWORD_RE = re.compile(
    r"\b(urgent|asap|important|priority|deadline|contract|invoice|proposal|quote|nda|agreement|"
    r"payment|signed|legal|renewal|onboarding|kickoff)\b",
    re.IGNORECASE,
)


def _detect_project_tags(text: str) -> list[str]:
    """Detect known project/client names (built-in list + live ChatProjects) → ['project:name', …]."""
    low = (text or "").lower()
    names = set(_KNOWN_PROJECTS)
    try:
        for p in _host.ChatProjects.list():
            if p.get("name"):
                names.add(p["name"])
    except Exception:  # noqa: BLE001
        pass
    tags = []
    for name in names:
        nl = name.lower().strip()
        if nl and re.search(r"\b" + re.escape(nl) + r"\b", low):
            tags.append(f"project:{nl}")
    return sorted(set(tags))


def _kb_ingest_text(text: str, source: str, extra_tags: list[str], importance: float = 0.6) -> bool:
    """Store a block of text in the KB with smart tags (source + projects + auto memory tags)."""
    text = (text or "").strip()
    if not text:
        return False
    tags = [f"source:{source}", *extra_tags, *_detect_project_tags(text)]
    try:
        tags += _host.detect_memory_tags(text)
    except Exception:  # noqa: BLE001
        pass
    tags = list(dict.fromkeys(t for t in tags if t))  # de-dupe, keep order
    try:
        _host.get_kb().learn_text(
            text, kind="tool", source=f"auto-ingest:{source}",
            importance=max(0.1, min(float(importance), 0.95)), tags=tags,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("auto-ingest learn_text failed: %s", exc)
        return False


def _email_importance(sender: str, subject: str, snippet: str, has_attachment: bool) -> float:
    """Heuristic 0..1 importance score for an email (reply-needed + project + attachment + urgency)."""
    blob = f"{subject} {snippet}"
    score = 0.3
    if _email_needs_reply(sender, subject, snippet):
        score += 0.25
    if _detect_project_tags(f"{sender} {blob}"):
        score += 0.25
    if has_attachment:
        score += 0.15
    if _IMPORTANCE_KEYWORD_RE.search(blob):
        score += 0.2
    if _is_vip_sender(sender):  # sender importance: the operator's flagged VIPs
        score += 0.2
    return min(score, 1.0)


def _gmail_download_attachments(service: Any, mid: str, attachments: list[dict]) -> list[str]:
    """Download an email's attachments to ~/.primus/gmail_attachments/ → list of saved paths."""
    saved: list[str] = []
    if not attachments:
        return saved
    try:
        _GMAIL_ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return saved
    for att in attachments:
        try:
            a = service.users().messages().attachments().get(
                userId="me", messageId=mid, id=att["attachmentId"],
            ).execute()
            raw = base64.urlsafe_b64decode((a.get("data", "") or "").encode("utf-8"))
            fname = _safe_filename(att["filename"]) or f"attachment-{att['attachmentId'][:8]}"
            dest = _GMAIL_ATTACHMENTS_DIR / fname
            dest.write_bytes(raw)
            saved.append(str(dest))
        except Exception:  # noqa: BLE001
            continue
    return saved


# ---------------------------------------------------------------------------
# Communications reliability layer — shared by auto-ingest, triage and briefing.
#   • _api_retry:  retry transient API errors (rate limits, 5xx, timeouts) with backoff.
#   • ingest ledger: content-hash de-duplication so nothing is ingested into the KB twice.
#   • VIP scoring: boost importance for senders the operator flags in CFG['vip_senders'].
# All are best-effort and never raise into the caller.
# ---------------------------------------------------------------------------

# Substrings that mark a transient (worth-retrying) failure across Google/Slack/HTTP libs.
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "temporarily", "rate limit", "ratelimited", "too many requests",
    "429", "500", "502", "503", "504", "connection reset", "connection aborted",
    "connection error", "ssl", "broken pipe", "service unavailable", "backenderror",
)


def _is_transient_error(exc: Exception) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    return any(m in s for m in _TRANSIENT_MARKERS)


def _api_retry(fn: Callable[[], Any], *, attempts: int = 3, base_delay: float = 0.6, label: str = "api") -> Any:
    """Call fn(); retry on transient errors with exponential backoff. Re-raises the last error."""
    last: Optional[Exception] = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_transient_error(exc) or i == attempts - 1:
                raise
            _host.log.debug("retry %s (%d/%d): %s", label, i + 1, attempts, str(exc)[:120])
            time.sleep(base_delay * (2 ** i))
    if last:
        raise last
    return None


_INGEST_LEDGER_PATH = _host.APP_DIR / "ingest_ledger.json"
_INGEST_LEDGER_MAX = 5000  # keep the most recent N content hashes


def _content_hash(text: str) -> str:
    """Stable hash of normalized content (used only for ingest de-duplication)."""
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha1(norm.encode("utf-8", "ignore")).hexdigest()  # noqa: S324 - dedup, not security


def _load_ingest_ledger() -> list[str]:
    try:
        data = json.loads(_INGEST_LEDGER_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [h for h in data if isinstance(h, str)]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _persist_ingest_ledger(entries: list[str]) -> None:
    entries = list(dict.fromkeys(entries))[-_INGEST_LEDGER_MAX:]
    try:
        _host._atomic_write_text(_INGEST_LEDGER_PATH, json.dumps(entries))
    except Exception:  # noqa: BLE001
        pass


def _is_vip_sender(sender: str) -> bool:
    """True if the sender matches an entry in CFG['vip_senders'] (name or address substring)."""
    s = (sender or "").lower()
    try:
        vips = _host.CFG.get("vip_senders", []) or []
    except Exception:  # noqa: BLE001
        vips = []
    return any(v and str(v).lower() in s for v in vips)


@tool
def auto_ingest_gmail(query: str = "is:unread", max_messages: int = 10, importance_threshold: float = 0.6) -> str:
    """Read important emails (+ attachments) and ingest them into the Knowledge Base (read-only).

    Scans up to `max_messages` matching `query` (Gmail search syntax), scores each for importance
    (reply-needed + project/client match + attachments + urgency keywords), and ingests those at or
    above `importance_threshold` (0..1) with smart tags (source:gmail, from:*, project:*). PDF/office
    attachments are downloaded and ingested through the normal pipeline. Never sends or modifies
    anything in Gmail. Requires gmail_auth.
    """
    service, err = _get_gmail_service()
    if err:
        return err
    max_messages = max(1, min(int(max_messages or 10), 50))
    try:
        thr = float(importance_threshold)
    except (TypeError, ValueError):
        thr = 0.6
    _host.PrimusSession.emit_think("Auto-ingest", "Scanning Gmail", "running")
    try:
        resp = _api_retry(
            lambda: service.users().messages().list(
                userId="me", q=query or "", maxResults=max_messages,
            ).execute(),
            label="gmail.list",
        )
        ids = [m["id"] for m in resp.get("messages", []) or []]
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Gmail scan failed: {str(exc)[:140]}"
    if not ids:
        return f"No emails matched query '{query}'."

    ledger = set(_load_ingest_ledger())
    new_hashes: list[str] = []
    ingested = skipped = duplicate = att_count = 0
    notes: list[str] = []
    for mid in ids:
        try:
            msg = _api_retry(
                lambda mid=mid: service.users().messages().get(
                    userId="me", id=mid, format="full",
                ).execute(),
                label="gmail.get",
            )
        except Exception:  # noqa: BLE001
            continue
        payload = msg.get("payload", {}) or {}
        hdr = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
        subject, sender = hdr.get("subject", "(no subject)"), hdr.get("from", "?")
        body, attachments = _gmail_extract_parts(payload)
        snippet = msg.get("snippet", "")
        score = _email_importance(sender, subject, snippet, bool(attachments))
        if score < thr:
            skipped += 1
            continue
        text = (
            f"Email from {sender}\nSubject: {subject}\nDate: {hdr.get('date', '?')}\n\n"
            f"{(body or snippet)[:4000]}"
        )
        chash = _content_hash(text)
        if chash in ledger:  # hash-based skip — already ingested
            duplicate += 1
            continue
        proj_tags = _detect_project_tags(f"{subject}\n{body}")
        if _kb_ingest_text(text, "gmail", [f"from:{_sender_display(sender)[:40]}", "email", *proj_tags],
                           importance=min(score, 0.9)):
            ingested += 1
            ledger.add(chash)
            new_hashes.append(chash)
            notes.append(f"“{subject[:50]}” ({score:.2f})")
        if attachments:
            saved = _gmail_download_attachments(service, mid, attachments)
            if saved:
                try:
                    _host.ingest_uploaded_files(
                        saved, "Learned / New Information", "",
                        extra_tags=["gmail", "attachment", *proj_tags],
                    )
                    att_count += len(saved)
                except Exception as exc:  # noqa: BLE001
                    _host.log.debug("attachment ingest failed: %s", exc)
    if new_hashes:
        _persist_ingest_ledger(list(ledger))
    _host.PrimusSession.emit_think("Auto-ingest", f"Gmail: {ingested} ingested", "done")
    lines = [
        f"📥 Gmail auto-ingest: {ingested} email(s) ingested, {skipped} below threshold "
        f"({thr:.2f}), {duplicate} already-ingested (skipped), {att_count} attachment(s)."
    ]
    if notes:
        lines += ["Ingested:", *[f"  • {n}" for n in notes]]
    return "\n".join(lines)


@tool
def auto_ingest_calendar(days_ahead: int = 7) -> str:
    """Ingest upcoming calendar events (today → +days_ahead) into the Knowledge Base (read-only).

    Stores each event's title, time, attendees, location and description with smart tags
    (source:calendar, event, project:*) so Primus can recall your schedule and meeting context later.
    Never modifies the calendar. Requires calendar_auth.
    """
    service, err = _get_calendar_service()
    if err:
        return err
    days = max(1, min(int(days_ahead or 7), 31))
    now = datetime.now().astimezone()
    end = now + timedelta(days=days)
    _host.PrimusSession.emit_think("Auto-ingest", "Scanning Calendar", "running")
    try:
        resp = _api_retry(
            lambda: service.events().list(
                calendarId="primary", timeMin=now.isoformat(), timeMax=end.isoformat(),
                singleEvents=True, orderBy="startTime", maxResults=100,
            ).execute(),
            label="calendar.list",
        )
        events = resp.get("items", []) or []
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Calendar scan failed: {str(exc)[:140]}"
    if not events:
        return f"No events in the next {days} day(s) to ingest."
    ledger = set(_load_ingest_ledger())
    new_hashes: list[str] = []
    ingested = duplicate = 0
    for ev in events:
        s, e = ev.get("start", {}) or {}, ev.get("end", {}) or {}
        when = s.get("dateTime") or s.get("date") or "?"
        end_when = e.get("dateTime") or e.get("date") or "?"
        atts = ", ".join(a.get("email", "") for a in ev.get("attendees", []) or [])
        text = f"Calendar event: {ev.get('summary', '(no title)')}\nWhen: {when} → {end_when}\n"
        if ev.get("location"):
            text += f"Location: {ev['location']}\n"
        if atts:
            text += f"Attendees: {atts}\n"
        if ev.get("description"):
            text += f"\n{ev['description'][:2000]}"
        chash = _content_hash(text)
        if chash in ledger:
            duplicate += 1
            continue
        proj = _detect_project_tags(f"{ev.get('summary', '')} {ev.get('description', '')}")
        if _kb_ingest_text(text, "calendar", ["event", *proj], importance=0.55):
            ingested += 1
            ledger.add(chash)
            new_hashes.append(chash)
    if new_hashes:
        _persist_ingest_ledger(list(ledger))
    _host.PrimusSession.emit_think("Auto-ingest", f"Calendar: {ingested}", "done")
    return (
        f"📥 Calendar auto-ingest: {ingested} event(s) ingested into the KB "
        f"(next {days} day(s)), {duplicate} already-ingested (skipped)."
    )


@tool
def auto_ingest_slack(channels: Optional[list] = None, limit: int = 20) -> str:
    """Ingest recent messages + action items from Slack channels into the Knowledge Base (read-only).

    `channels` is a list of ids or '#names' (default ['#general']); a comma-separated string is also
    accepted. For each channel it stores a digest of the last `limit` messages plus any detected
    action items, tagged (source:slack, channel:*, project:*). Never posts anything. Requires
    slack_auth and the bot to be a member of each channel.
    """
    if channels is None:
        channels = ["#general"]
    if isinstance(channels, str):
        channels = [c.strip() for c in channels.split(",") if c.strip()]
    client, err = _get_slack_client()
    if err:
        return err
    limit = max(1, min(int(limit or 20), 100))
    _host.PrimusSession.emit_think("Auto-ingest", "Scanning Slack", "running")
    ledger = set(_load_ingest_ledger())
    new_hashes: list[str] = []
    total = duplicate = 0
    per: list[str] = []
    cache: dict = {}
    for ch in channels:
        cid, cerr = _slack_resolve_channel(client, ch)
        if cerr:
            per.append(f"{ch}: {cerr}")
            continue
        try:
            resp = _api_retry(
                lambda cid=cid: client.conversations_history(channel=cid, limit=limit),
                label="slack.history",
            )
            msgs = list(reversed(resp.get("messages", []) or []))
        except Exception as exc:  # noqa: BLE001
            per.append(f"{ch}: {_slack_err(exc)}")
            continue
        if not msgs:
            per.append(f"{ch}: no messages")
            continue
        for m in msgs:
            m["user_name"] = _slack_user_name(client, m.get("user") or m.get("bot_id", ""), cache)
        convo_lines = []
        for m in msgs:
            t = re.sub(r"\s+", " ", m.get("text", "") or "").strip()
            convo_lines.append(f"{m['user_name']}: {t[:200]}")
        items = _slack_action_items(msgs)
        text = f"Slack digest — {ch} ({len(msgs)} messages)\n\n" + "\n".join(convo_lines)[:3500]
        if items:
            text += "\n\nAction items:\n" + "\n".join(f"- {it}" for it in items)
        chash = _content_hash(text)
        if chash in ledger:
            duplicate += 1
            per.append(f"{ch}: unchanged since last ingest (skipped)")
            continue
        proj = _detect_project_tags(text)
        if _kb_ingest_text(text, "slack", ["slack", f"channel:{ch.lstrip('#')}", *proj], importance=0.55):
            total += 1
            ledger.add(chash)
            new_hashes.append(chash)
            per.append(f"{ch}: ingested ({len(msgs)} msgs, {len(items)} action item(s))")
    if new_hashes:
        _persist_ingest_ledger(list(ledger))
    _host.PrimusSession.emit_think("Auto-ingest", f"Slack: {total}", "done")
    return (
        f"📥 Slack auto-ingest: {total} channel digest(s) ingested, {duplicate} unchanged (skipped).\n"
        + "\n".join(f"  • {p}" for p in per)
    )


@tool
def auto_ingest_all(full: bool = False, importance_threshold: float = 0.7) -> str:
    """Run all auto-ingest sources (Gmail + Calendar + Slack) into the Knowledge Base (read-only, safe).

    By DEFAULT this is high-signal only: it ingests important unread email (score ≥
    `importance_threshold`, default 0.7), the next 7 days of calendar events, and skips the broad
    Slack sweep — so the KB stays clean. Pass `full=True` for a deep sweep: a lower email bar (0.4),
    14 days of calendar, and the default Slack channels. De-duplication (content hashing) means
    re-running never creates duplicate KB entries. Any source that isn't authorized is skipped with
    a clear note (graceful degradation). Ideal to attach to a scheduled/morning agent.
    """
    try:
        thr = float(importance_threshold)
    except (TypeError, ValueError):
        thr = 0.7
    if full:
        parts = [
            _run_tool(auto_ingest_gmail, query="newer_than:2d", max_messages=25, importance_threshold=min(thr, 0.4)),
            _run_tool(auto_ingest_calendar, days_ahead=14),
            _run_tool(auto_ingest_slack),
        ]
        header = "🔄 Auto-ingest (FULL sweep — Gmail + Calendar + Slack):"
    else:
        parts = [
            _run_tool(auto_ingest_gmail, query="is:unread", max_messages=15, importance_threshold=thr),
            _run_tool(auto_ingest_calendar, days_ahead=7),
        ]
        header = (
            f"🔄 Auto-ingest (high-signal only, importance ≥ {thr:.2f}). "
            "Call with full=True for a deep sweep incl. Slack:"
        )
    return header + "\n\n" + "\n\n".join(parts)


# ===========================================================================
# Connection Manager — central registry for connected accounts
# ---------------------------------------------------------------------------
# A thin, secure registry over the existing per-service token files. Each connection is a small JSON
# under ~/.primus/connections/ (chmod 600, dir 700) that points at the service's real token/creds
# (e.g. _GMAIL_TOKEN_PATH, _SLACK_TOKEN_PATH) — so it interoperates with, and never breaks, the
# existing Gmail/Calendar/Slack code. Already-authorized services are auto-detected even without an
# explicit "add". Supports OAuth tokens (Google) and raw API keys (Slack/Discord/etc.). The Gmail,
# Calendar and Slack auth tools register here automatically on success (via _conn_note_auth).
# ===========================================================================

# Known services → their real token file (the source of truth the service code already reads).
_KNOWN_CONN_TOKENS = {
    "gmail": _GMAIL_TOKEN_PATH,
    "calendar": _CALENDAR_TOKEN_PATH,
    "slack": _SLACK_TOKEN_PATH,
}


def _chmod600(path: Path) -> None:
    """Best-effort owner-only permissions on a secret file (never raises)."""
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _write_secret_json(path: Path, data: dict) -> None:
    """Write JSON to a secret file (parent created, chmod 600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _chmod600(path)


def _connections_dir() -> Path:
    """~/.primus/connections/, created with owner-only (700) permissions."""
    d = _host.APP_DIR / "connections"
    try:
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)
    except OSError:
        pass
    return d


def _conn_record_path(name: str) -> Path:
    return _connections_dir() / f"{_safe_filename(name) or 'conn'}.json"


def _conn_save(rec: dict) -> Path:
    p = _conn_record_path(rec["name"])
    _write_secret_json(p, rec)
    return p


def _conn_load_all() -> list[dict]:
    """All explicitly-registered connection records."""
    out: list[dict] = []
    try:
        for f in sorted(_connections_dir().glob("*.json")):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(r, dict) and r.get("name"):
                    out.append(r)
            except (OSError, json.JSONDecodeError):
                continue
    except OSError:
        pass
    return out


def _conn_detected() -> list[dict]:
    """Known services whose token file already exists (so status works without an explicit add)."""
    recs = []
    for t, path in _KNOWN_CONN_TOKENS.items():
        if path.exists():
            recs.append({
                "name": t, "type": t,
                "kind": "oauth" if t in ("gmail", "calendar") else "api",
                "token_path": str(path), "detected": True,
            })
    return recs


def _conn_merged() -> list[dict]:
    """Registered records + auto-detected known services (registered wins on name clash)."""
    persisted = _conn_load_all()
    names = {r["name"] for r in persisted}
    return persisted + [d for d in _conn_detected() if d["name"] not in names]


def _conn_health(rec: dict) -> tuple[bool, str]:
    """(ok, detail) for one connection — validates live for Gmail/Calendar/Slack, presence otherwise."""
    t = rec.get("type")
    if t == "gmail":
        _, err = _get_gmail_service()
        return (not err, err or "connected")
    if t == "calendar":
        _, err = _get_calendar_service()
        return (not err, err or "connected")
    if t == "slack":
        client, err = _get_slack_client()
        if err:
            return False, err
        try:
            info = client.auth_test()
            return True, f"workspace {info.get('team', '?')} as {info.get('user', '?')}"
        except Exception as exc:  # noqa: BLE001
            return False, _slack_err(exc)
    tp = rec.get("token_path")
    if tp and Path(tp).exists():
        return True, "token present (not validated)"
    return False, "no token configured"


def _conn_note_auth(conn_type: str) -> None:
    """Register/refresh a known connection after a successful auth flow. Non-fatal by design."""
    try:
        tp = _KNOWN_CONN_TOKENS.get(conn_type)
        _conn_save({
            "name": conn_type, "type": conn_type,
            "kind": "oauth" if conn_type in ("gmail", "calendar") else "api",
            "token_path": str(tp) if tp else "", "added": _host._now_iso(),
        })
    except Exception:  # noqa: BLE001
        pass


@tool
def connections_list() -> str:
    """List all connected accounts (registered + auto-detected) with type and token presence.

    Read-only overview — does not hit the network. Use connections_status for live health checks.
    """
    conns = _conn_merged()
    if not conns:
        return "No connections yet. Add one with connections_add, or run gmail_auth / slack_auth."
    lines = [f"🔌 {len(conns)} connection(s):"]
    for c in conns:
        tp = c.get("token_path")
        present = bool(tp and Path(tp).exists())
        src = " (auto-detected)" if c.get("detected") else ""
        lines.append(
            f"  • {c['name']} [{c['type']}/{c.get('kind', '?')}]{src} — "
            f"token {'present' if present else 'missing'}"
        )
    return "\n".join(lines)


@tool
def connections_status() -> str:
    """Detailed live health for every connection (validates Gmail/Calendar/Slack against their APIs).

    Refreshes Google tokens as a side effect of the health check. Use this to confirm everything is
    actually working (vs. connections_list which is a quick offline overview).
    """
    conns = _conn_merged()
    if not conns:
        return "No connections configured. Use connections_add or gmail_auth / calendar_auth / slack_auth."
    lines = ["🔌 Connection status:"]
    for c in conns:
        ok, detail = _conn_health(c)
        src = " (auto-detected)" if c.get("detected") else ""
        lines.append(f"  {'✓' if ok else '⚠'} {c['name']} [{c['type']}/{c.get('kind', '?')}]{src} — {detail}")
    return "\n".join(lines)


@tool
def connections_add(name: str, type: str = "gmail", token_or_creds_path: str = "") -> str:
    """Add or update a connection. Stores the secret in the right place and registers it centrally.

    - `type="gmail"` / `"calendar"`: pass the path to your Google OAuth client-secret JSON (Desktop
      app). It's copied to ~/.primus/gmail_credentials.json; then run gmail_auth / calendar_auth.
    - `type="slack"`: pass the Bot token (xoxb-…) / User token (xoxp-…) directly, or a path to a
      token JSON. Merged into ~/.primus/slack_token.json.
    - `type="discord"` (or any other): pass the API token/key; stored at ~/.primus/<type>_token.json.
    `name` is a friendly label for the connection. Secrets are written owner-only (chmod 600).
    """
    name = (name or "").strip()
    ctype = (type or "").strip().lower()
    val = (token_or_creds_path or "").strip()
    if not name:
        return "⚠ Provide a name for the connection."
    if ctype not in ("gmail", "calendar", "slack", "discord"):
        return "⚠ Unsupported type. Use gmail, calendar, slack, or discord."

    rec: dict = {"name": name, "type": ctype, "added": _host._now_iso()}
    try:
        if ctype in ("gmail", "calendar"):
            if not val:
                return "⚠ Provide the path to your Google OAuth client-secret JSON (Desktop app)."
            src = Path(val).expanduser()
            if not src.exists():
                return f"⚠ File not found: {src}"
            _GMAIL_CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _GMAIL_CREDS_PATH.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            _chmod600(_GMAIL_CREDS_PATH)
            tok = _KNOWN_CONN_TOKENS[ctype]
            rec.update({"kind": "oauth", "creds_path": str(_GMAIL_CREDS_PATH), "token_path": str(tok)})
            detail = f"Credentials stored. Run {'gmail_auth' if ctype == 'gmail' else 'calendar_auth'} to sign in."
        elif ctype == "slack":
            if not val:
                return "⚠ Provide the Slack Bot token (xoxb-…) or a path to a token JSON."
            if val.startswith("xox"):
                toks = {("bot_token" if val.startswith("xoxb") else "user_token"): val}
            else:
                sp = Path(val).expanduser()
                if not sp.exists():
                    return f"⚠ File not found: {sp}"
                try:
                    toks = json.loads(sp.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    return f"⚠ Could not read token JSON: {str(exc)[:100]}"
            existing = _load_slack_tokens()
            existing.update({k: v for k, v in toks.items() if v})
            _write_secret_json(_SLACK_TOKEN_PATH, existing)
            rec.update({"kind": "api", "token_path": str(_SLACK_TOKEN_PATH)})
            detail = "Slack token stored. Run slack_status to verify."
        else:  # discord / generic API key
            if not val:
                return "⚠ Provide the API token/key for this service."
            tok = _host.APP_DIR / f"{ctype}_token.json"
            _write_secret_json(tok, {"token": val})
            rec.update({"kind": "api", "token_path": str(tok)})
            detail = f"{ctype} token stored at {tok}."
    except OSError as exc:
        return f"⚠ Could not store credentials: {str(exc)[:120]}"

    _conn_save(rec)
    return f"✓ Connection '{name}' ({ctype}) added. {detail}"


@tool
def connections_remove(name: str) -> str:
    """Remove a connection: deletes its registry entry and its local token (revokes access locally).

    Shared Google client-secrets are preserved (removing 'gmail' won't break 'calendar'). This does
    not revoke tokens server-side — do that in the provider's app settings if needed.
    """
    name = (name or "").strip()
    if not name:
        return "⚠ Provide the connection name to remove (see connections_list)."
    p = _conn_record_path(name)
    rec = None
    if p.exists():
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rec = None
        try:
            p.unlink()
        except OSError as exc:
            return f"⚠ Could not remove record: {str(exc)[:100]}"
    else:
        rec = {d["name"]: d for d in _conn_detected()}.get(name.lower())
        if not rec:
            return f"No connection named '{name}'. Use connections_list to see them."

    removed_token = False
    tp = (rec or {}).get("token_path")
    if tp:
        tpp = Path(tp)
        if tpp.exists() and tpp != _GMAIL_CREDS_PATH:  # keep the shared OAuth client secret
            try:
                tpp.unlink()
                removed_token = True
            except OSError:
                pass
    return f"✓ Removed connection '{name}'." + (" Local token deleted." if removed_token else "")


@tool
def connections_refresh() -> str:
    """Refresh/validate all connections — renews Google OAuth tokens and re-checks Slack.

    Runs a live health check on each connection; for Google services this triggers a silent token
    refresh (and re-saves the token). Reports the result per connection.
    """
    conns = _conn_merged()
    if not conns:
        return "No connections to refresh. Add one with connections_add."
    _host.PrimusSession.emit_think("Connections", "Refreshing tokens", "running")
    lines = ["🔄 Connection refresh:"]
    for c in conns:
        ok, detail = _conn_health(c)
        lines.append(f"  {'✓' if ok else '⚠'} {c['name']} ({c['type']}): {detail}")
    _host.PrimusSession.emit_think("Connections", "Refresh complete", "done")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Host/UI-facing bridges (plain functions, not tools) so the Setup tab and status
# dashboard can render/manage connections directly via ``_host.<fn>``. These reuse the exact
# same helpers as the @tools, so behavior is identical to running the tools in chat.
# ---------------------------------------------------------------------------

def connections_status_markdown() -> str:
    """Rich markdown for the UI 'Connected Accounts' panel — live health for every connection."""
    conns = _conn_merged()
    if not conns:
        return (
            "**No accounts connected yet.**\n\n"
            "Add one below, or run `gmail_auth` / `calendar_auth` / `slack_auth` in chat."
        )
    lines = ["| Account | Type | Health |", "| --- | --- | --- |"]
    for c in conns:
        try:
            ok, detail = _conn_health(c)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, str(exc)[:80]
        src = " · auto" if c.get("detected") else ""
        icon = "✅" if ok else "⚠️"
        lines.append(f"| {c['name']}{src} | {c.get('type', '?')} | {icon} {detail[:90]} |")
    return "\n".join(lines)


def connections_health_line() -> str:
    """One-line health summary for the main status panel (empty string if nothing configured)."""
    try:
        states = _comms_health_states()
    except Exception:  # noqa: BLE001
        return ""
    if not states:
        return ""
    parts = [f"{label} {'✓' if ok else '⚠'}" for label, ok, _ in states]
    return "**Accounts:** " + " · ".join(parts)


def connections_refresh_markdown() -> str:
    """Run a live refresh/validation of all connections, then return the updated status markdown."""
    try:
        _run_tool(connections_refresh)
    except Exception as exc:  # noqa: BLE001
        _host.log.debug("connections_refresh (ui) failed: %s", exc)
    return connections_status_markdown()


def connections_add_ui(name: str, ctype: str, secret: str) -> str:
    """UI bridge for connections_add. Returns the result message; empty inputs are reported clearly."""
    if not (name or "").strip():
        return "⚠ Enter a friendly name for the account."
    return _run_tool(connections_add, name=name, type=ctype or "gmail", token_or_creds_path=secret or "")


def connections_remove_ui(name: str) -> str:
    """UI bridge for connections_remove."""
    if not (name or "").strip():
        return "⚠ Enter the account name to remove."
    return _run_tool(connections_remove, name=name)


# ---------------------------------------------------------------------------
# Inbox drawer bridges (plain functions, not @tools). The agent-facing Gmail tools keep their
# string API; these return structured dicts for the Gradio inbox and reuse the exact same
# service/fixture helpers, so live and demo behavior match chat behavior.
# ---------------------------------------------------------------------------

def _gmail_list_structured(query: str, max_results: int = 25) -> tuple[list[dict], str]:
    """(rows, note) for the Inbox UI. row = {id, date, from, subject, snippet, unread}."""
    max_results = max(1, min(int(max_results or 25), 50))
    if _gmail_fixture_active():
        rows = []
        for m in _gmail_fixture_query(_gmail_fixture_load(), query)[:max_results]:
            rows.append({
                "id": str(m.get("id", "")),
                "date": str(m.get("date", "")),
                "from": str(m.get("from", "")),
                "subject": str(m.get("subject", "(no subject)")),
                "snippet": str(m.get("snippet", "")),
                "unread": bool(m.get("unread")),
            })
        return rows, "Demo inbox (fixture) — sample data"
    service, err = _get_gmail_service()
    if err:
        return [], err.splitlines()[0].lstrip("⚠ ").strip()[:140]
    try:
        resp = service.users().messages().list(
            userId="me", q=query or "", maxResults=max_results
        ).execute()
        rows = []
        for m in resp.get("messages", []) or []:
            meta = service.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            hdr = {h["name"]: h["value"] for h in meta.get("payload", {}).get("headers", [])}
            rows.append({
                "id": m["id"],
                "date": hdr.get("Date", ""),
                "from": hdr.get("From", ""),
                "subject": hdr.get("Subject", "(no subject)"),
                "snippet": meta.get("snippet", ""),
                "unread": "UNREAD" in (meta.get("labelIds") or []),
            })
        return rows, ""
    except Exception as exc:  # noqa: BLE001
        decoded = _gmail_decode_oauth_error(str(exc))
        return [], (decoded or f"Gmail list failed: {str(exc)[:140]}")


def _gmail_read_structured(message_id: str) -> tuple[dict, str]:
    """(message, note) for the read pane. message = {id, from, to, date, subject, body}."""
    mid = (message_id or "").strip()
    if not mid:
        return {}, "No message selected."
    if _gmail_fixture_active():
        for m in _gmail_fixture_load():
            if str(m.get("id")) == mid:
                return {
                    "id": mid,
                    "from": str(m.get("from", "?")),
                    "to": str(m.get("to", "?")),
                    "date": str(m.get("date", "?")),
                    "subject": str(m.get("subject", "(no subject)")),
                    "body": str(m.get("body", "")),
                    "unread": bool(m.get("unread")),
                }, "Demo inbox (fixture)"
        return {}, f"No fixture message with id {mid}."
    service, err = _get_gmail_service()
    if err:
        return {}, err.splitlines()[0].lstrip("⚠ ").strip()[:140]
    try:
        msg = service.users().messages().get(userId="me", id=mid, format="full").execute()
        payload = msg.get("payload", {}) or {}
        hdr = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
        body_text, _attachments = _gmail_extract_parts(payload)
        return {
            "id": mid,
            "from": hdr.get("from", "?"),
            "to": hdr.get("to", "?"),
            "date": hdr.get("date", "?"),
            "subject": hdr.get("subject", "(no subject)"),
            "body": body_text or "(no text body)",
            "unread": "UNREAD" in (msg.get("labelIds") or []),
        }, ""
    except Exception as exc:  # noqa: BLE001
        return {}, f"Could not read message: {str(exc)[:140]}"


def inbox_status_markdown() -> str:
    """One-line Inbox status for the drawer: fixture / not configured / connected account."""
    if _gmail_fixture_active():
        n = len(_gmail_fixture_load())
        return (
            f"**Demo inbox (fixture)** — {n} sample messages. "
            "Connect Gmail below for your real mail."
        )
    if not _GMAIL_CREDS_PATH.exists():
        return "⚠ **Not configured** — no OAuth client secret yet. See setup below."
    service, err = _get_gmail_service()
    if err or service is None:
        return "⚠ **Not connected** — click **Connect** to authorize Gmail."
    try:
        profile = service.users().getProfile(userId="me").execute()
        return f"✓ **Connected as {profile.get('emailAddress', '?')}**"
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Status check failed: {str(exc)[:120]}"


def inbox_setup_markdown() -> str:
    """The collapsed 6-step Desktop-OAuth setup help for the Inbox drawer."""
    return (
        "**One-time Gmail setup (Desktop OAuth):**\n"
        "1. Open [Google Cloud Console](https://console.cloud.google.com) → create/select a project.\n"
        "2. APIs & Services → Library → enable the **Gmail API**.\n"
        "3. OAuth consent screen → configure it; add your Google account under **Test users**.\n"
        "4. Credentials → Create credentials → **OAuth client ID** → type **Desktop app**.\n"
        f"5. Download the JSON and save it as `~/.primus/gmail_credentials.json`.\n"
        "6. Click **Connect** above and approve in the browser that opens on this machine.\n\n"
        "_Running in Docker/headless? Do the first Connect from the host UI — the token lands in "
        "the bind-mounted `~/.primus` and the container reuses it._"
    )


def inbox_connect_ui() -> str:
    """Connect button → run the existing gmail_auth tool (opens a browser on this machine)."""
    return _run_tool(gmail_auth)


def inbox_list_ui(query: str) -> tuple[list[list], list[dict], str]:
    """Refresh button → (dataframe rows, state rows, status note)."""
    rows, note = _gmail_list_structured(query)
    table = [
        ["🔵" if r["unread"] else "", r["date"][:16], r["from"][:38], r["subject"][:70]]
        for r in rows
    ]
    if not rows and not note:
        note = "No messages match that query."
    return table, rows, note


def inbox_read_ui(rows: list[dict], index: Any) -> tuple[str, str]:
    """Open a selected list row → (read-pane markdown, message id)."""
    if not rows or index is None:
        return "_Select a message above._", ""
    try:
        i = int(index[0] if isinstance(index, (list, tuple)) else index)
        row = rows[i]
    except (ValueError, TypeError, IndexError):
        return "_Select a message above._", ""
    msg, note = _gmail_read_structured(row.get("id", ""))
    if not msg:
        return f"⚠ {note}", ""
    head = (
        f"### {msg['subject']}\n"
        f"**From:** {msg['from']}  \n"
        f"**To:** {msg['to']}  \n"
        f"**Date:** {msg['date']}\n"
    )
    if note:
        head = f"_{note}_\n\n" + head
    return head + "\n" + (msg["body"] or "(no text body)"), msg["id"]


def inbox_reply_prefill_ui(message_id: str) -> str:
    """Prefill the reply box with the deterministic smart-reply draft (not sent)."""
    msg, _note = _gmail_read_structured(message_id)
    if not msg:
        return ""
    ai = _comms_action_items(f"{msg['subject']}. {msg['body'][:400]}")
    return _suggest_email_reply(_sender_display(msg["from"]), msg["subject"], ai)


def inbox_draft_reply_ui(message_id: str, body: str) -> str:
    """Create a Gmail draft reply — preview in Suggest mode, real draft in Execute mode.

    In fixture mode this is always a preview (nothing is saved anywhere).
    """
    if not (body or "").strip():
        return "⚠ Write a reply first."
    msg, note = _gmail_read_structured(message_id)
    if not msg:
        return f"⚠ {note or 'No message selected.'}"
    to_addr = (_extract_emails(msg["from"]) or [""])[0]
    if not to_addr:
        return "⚠ Could not determine the reply address from that message."
    subject = msg["subject"]
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    if _gmail_fixture_active():
        return (
            f"[Demo inbox] Draft preview only — nothing saved. Connect Gmail to create drafts.\n"
            f"To: {to_addr} · Subject: {subject}"
        )
    if _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST:
        return (
            f"[Suggest] Would create a draft reply to {to_addr} ({subject[:50]}). "
            "Switch to Execute to save it."
        )
    return _run_tool(gmail_create_draft, to=to_addr, subject=subject, body=body)


# ===========================================================================
# Auto-triage + Smart Reply Drafting (Gmail + Slack + Calendar)
# ---------------------------------------------------------------------------
# Scans all connected comms, scores priority deterministically (reusing _email_importance,
# _slack_action_items, project tagging), and turns them into action: todos + ready-to-send reply
# drafts. Every source degrades gracefully — an unauthorized/failed connection is skipped with a
# note (never aborts). Drafting/posting is preview-only in Suggest mode. The last scan is cached so
# create_todos_from_comms / draft_smart_replies act on the same items without rescanning.
#
# How to run daily triage:
#   • "triage my inbox and channels":   triage_communications(1)
#   • "turn those into todos":           create_todos_from_comms()
#   • "draft replies to the urgent ones":draft_smart_replies()   (preview in Suggest)
#   • "brief me with priorities":        triage_and_brief()
# ===========================================================================

# Cache of the most recent triage so the follow-up tools reuse it (avoids rescanning every call).
_LAST_TRIAGE: dict = {"ts": 0.0, "items": [], "errors": []}


def _comms_action_items(text: str) -> list[str]:
    """Extract commitment/request lines from any comms text (reuses the shared action-item regex)."""
    out: list[str] = []
    for line in re.split(r"[\n.!?]+", text or ""):
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) >= 4 and _SLACK_ACTION_RE.search(line) and line not in out:
            out.append(line)
    return out[:5]


def _triage_reason(score: float, needs_reply: bool, has_attach: bool) -> str:
    bits = []
    if needs_reply:
        bits.append("needs reply")
    if has_attach:
        bits.append("has attachment")
    if score >= 0.8:
        bits.append("high importance")
    return ", ".join(bits) or "flagged"


def _suggest_email_reply(name: str, subject: str, action_items: list[str]) -> str:
    """Deterministic, professional acknowledgement draft (not sent)."""
    who = name.split()[0] if name and name != "?" else "there"
    first = action_items[0].rstrip(".") if action_items else ""
    line = f" I'll take care of {first}." if first else " I'll review and follow up shortly."
    return f"Hi {who},\n\nThanks for your email regarding \"{subject[:60]}\".{line}\n\n" + _email_signoff()


def _suggest_slack_reply(name: str, action_items: list[str]) -> str:
    first = action_items[0].rstrip(".") if action_items else ""
    tail = f" I'll handle {first} and follow up." if first else " On it — I'll follow up shortly."
    return f"Thanks{f', {name}' if name and name != '?' else ''} —{tail}"


def _triage_gather(days_back: int = 1) -> tuple[list[dict], list[str]]:
    """Scan Gmail + Slack + Calendar → (prioritized items, degraded-source notes). Never raises."""
    items: list[dict] = []
    errors: list[str] = []
    days = max(1, min(int(days_back or 1), 14))

    # --- Gmail: unread in the window, kept if important enough ---
    gsvc, gerr = _get_gmail_service()
    if _gmail_fixture_active():
        # Demo mode: triage the fixture mailbox with the same scoring, noted as demo data.
        try:
            for m in _gmail_fixture_query(_gmail_fixture_load(), "is:unread")[:25]:
                subject = str(m.get("subject", "(no subject)"))
                sender = str(m.get("from", "?"))
                snippet = str(m.get("snippet", ""))
                body = str(m.get("body", ""))
                score = _email_importance(sender, subject, snippet, False)
                if score < 0.55:
                    continue
                needs = _email_needs_reply(sender, subject, snippet)
                ai = _comms_action_items(f"{subject}. {body or snippet}")
                sender_email = (_extract_emails(sender) or [""])[0]
                reply = None
                if needs and sender_email:
                    reply = {
                        "kind": "gmail", "to": sender_email, "subject": f"Re: {subject}",
                        "suggested": _suggest_email_reply(_sender_display(sender), subject, ai),
                    }
                items.append({
                    "source": "gmail (fixture)", "priority": round(score, 2), "title": subject[:80],
                    "who": _sender_display(sender), "reason": _triage_reason(score, needs, False),
                    "action_items": ai, "ref": str(m.get("id", "")), "reply": reply,
                    "tags": _detect_project_tags(f"{subject} {body}"),
                })
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Gmail fixture scan: {str(exc)[:80]}")
    elif gerr:
        # Keep the degraded note short (the full guidance/hint is surfaced separately).
        errors.append("Gmail: " + (gerr.splitlines()[0].lstrip("⚠ ").strip()[:80] or "unavailable"))
    else:
        try:
            resp = _api_retry(
                lambda: gsvc.users().messages().list(
                    userId="me", q=f"is:unread newer_than:{days}d", maxResults=25,
                ).execute(),
                label="triage.gmail.list",
            )
            for m in resp.get("messages", []) or []:
                try:
                    msg = _api_retry(
                        lambda m=m: gsvc.users().messages().get(
                            userId="me", id=m["id"], format="full",
                        ).execute(),
                        label="triage.gmail.get",
                    )
                except Exception:  # noqa: BLE001
                    continue
                payload = msg.get("payload", {}) or {}
                hdr = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
                subject, sender = hdr.get("subject", "(no subject)"), hdr.get("from", "?")
                body, attachments = _gmail_extract_parts(payload)
                snippet = msg.get("snippet", "")
                score = _email_importance(sender, subject, snippet, bool(attachments))
                if score < 0.55:
                    continue
                needs = _email_needs_reply(sender, subject, snippet)
                ai = _comms_action_items(f"{subject}. {body or snippet}")
                sender_email = (_extract_emails(sender) or [""])[0]
                reply = None
                if needs and sender_email:
                    reply = {
                        "kind": "gmail", "to": sender_email, "subject": f"Re: {subject}",
                        "suggested": _suggest_email_reply(_sender_display(sender), subject, ai),
                    }
                items.append({
                    "source": "gmail", "priority": round(score, 2), "title": subject[:80],
                    "who": _sender_display(sender), "reason": _triage_reason(score, needs, bool(attachments)),
                    "action_items": ai, "ref": m["id"], "reply": reply,
                    "tags": _detect_project_tags(f"{subject} {body}"),
                })
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Gmail scan: {str(exc)[:80]}")

    # --- Slack: mentions of me (high) + action items across recent channel messages ---
    client, serr = _get_slack_client()
    if serr:
        errors.append(f"Slack: {serr}")
    else:
        me = ""
        try:
            me = _api_retry(client.auth_test, label="triage.slack.auth").get("user_id", "")
        except Exception:  # noqa: BLE001
            pass
        try:
            chans = _api_retry(
                lambda: client.conversations_list(
                    types="public_channel,private_channel", limit=20, exclude_archived=True,
                ),
                label="triage.slack.list",
            ).get("channels", []) or []
        except Exception as exc:  # noqa: BLE001
            chans = []
            errors.append(f"Slack channels: {_slack_err(exc)}")
        cache: dict = {}
        for c in chans[:8]:  # bound the scan
            cid, cname = c.get("id"), c.get("name", "?")
            try:
                msgs = list(reversed(_api_retry(
                    lambda cid=cid: client.conversations_history(channel=cid, limit=15),
                    label="triage.slack.history",
                ).get("messages", []) or []))
            except Exception:  # noqa: BLE001
                continue
            for m in msgs:
                text = m.get("text", "") or ""
                mentioned = bool(me) and f"<@{me}>" in text
                ai = _comms_action_items(text)
                if not mentioned and not ai:
                    continue
                who_name = _slack_user_name(client, m.get("user") or m.get("bot_id", ""), cache)
                thread_ts = m.get("thread_ts") or m.get("ts")
                reply = None
                if mentioned:
                    reply = {
                        "kind": "slack", "channel": cid, "channel_name": cname,
                        "thread_ts": thread_ts, "suggested": _suggest_slack_reply(who_name, ai),
                    }
                items.append({
                    "source": "slack", "priority": 0.85 if mentioned else 0.6,
                    "title": (re.sub(r"\s+", " ", text)[:80] or "(message)"),
                    "who": f"#{cname} · {who_name}",
                    "reason": "you were mentioned" if mentioned else "action item in channel",
                    "action_items": ai, "ref": m.get("ts"), "reply": reply,
                    "tags": _detect_project_tags(text),
                })

    # --- Calendar: imminent events (next 3h) as prep items ---
    csvc, cerr = _get_calendar_service()
    if cerr:
        errors.append(f"Calendar: {cerr}")
    else:
        try:
            now = datetime.now().astimezone()
            evs = _api_retry(
                lambda: csvc.events().list(
                    calendarId="primary", timeMin=now.isoformat(),
                    timeMax=(now + timedelta(days=2)).isoformat(),
                    singleEvents=True, orderBy="startTime", maxResults=25,
                ).execute(),
                label="triage.calendar",
            ).get("items", []) or []
            for ev in evs:
                s = ev.get("start", {}) or {}
                sdt = _cal_parse_dt(s.get("dateTime")) if s.get("dateTime") else None
                if not sdt:
                    continue
                mins = (sdt - now).total_seconds() / 60
                if 0 <= mins <= 180:
                    items.append({
                        "source": "calendar", "priority": 0.6, "title": ev.get("summary", "(event)")[:80],
                        "who": ", ".join(a.get("email", "") for a in ev.get("attendees", []) or [])[:60],
                        "reason": f"starts in {int(mins)} min — prep", "action_items": [],
                        "ref": ev.get("id"), "reply": None,
                        "tags": _detect_project_tags(ev.get("summary", "")),
                    })
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Calendar scan: {str(exc)[:80]}")

    items.sort(key=lambda x: x["priority"], reverse=True)
    return items, errors


def _triage_get(days_back: int = 1, max_age: float = 600.0) -> tuple[list[dict], list[str]]:
    """Return the cached triage if fresh (<max_age s), else run a new scan and cache it."""
    if _LAST_TRIAGE["items"] and (time.time() - _LAST_TRIAGE["ts"]) < max_age:
        return _LAST_TRIAGE["items"], _LAST_TRIAGE["errors"]
    items, errors = _triage_gather(days_back)
    _LAST_TRIAGE.update({"ts": time.time(), "items": items, "errors": errors})
    return items, errors


def _comms_reconnect_hint(errors: list[str]) -> str:
    """One actionable reconnect line for degraded comms sources — always Primus tools, never manual.

    Turns "some sources are down" into a concrete next step so Primus never dead-ends on a comms
    failure. Deliberately points at the OAuth tools (ensure_gmail_access / calendar_auth / slack_auth),
    NOT App Passwords or manual Google/Slack settings.
    """
    low = " ".join(errors).lower()
    cmds: list[str] = []
    if "gmail" in low:
        cmds.append("`ensure_gmail_access`")
    if "calendar" in low:
        cmds.append("`calendar_auth`")
    if "slack" in low:
        cmds.append("`slack_auth`")
    if not cmds:
        return ""
    return "→ Reconnect with " + " / ".join(cmds) + " (never App Passwords or manual settings)."


def _fmt_triage_items(items: list[dict], limit: int = 10) -> list[str]:
    icons = {"gmail": "📧", "slack": "💬", "calendar": "📅"}
    lines = []
    for it in items[:limit]:
        lines.append(
            f"{icons.get(it['source'], '•')} [{it['priority']:.2f}] {it['title']} "
            f"— {it['who']} ({it['reason']})"
        )
        for ai in it.get("action_items", [])[:3]:
            lines.append(f"      ↳ {ai[:100]}")
    return lines


@tool
def triage_communications(days_back: int = 1) -> str:
    """Scan Gmail + Slack + Calendar, score priority, and surface what needs attention (read-only).

    Pulls important/unread email, Slack @-mentions + action items, and imminent meetings from the
    last `days_back` day(s), scores each deterministically, and returns a prioritized list with
    detected action items. Unavailable connections are skipped with a note (graceful degradation).
    Results are cached for the follow-ups create_todos_from_comms / draft_smart_replies.
    """
    _host.PrimusSession.emit_think("Triage", "Scanning communications", "running")
    items, errors = _triage_gather(days_back)
    _LAST_TRIAGE.update({"ts": time.time(), "items": items, "errors": errors})
    _host.PrimusSession.emit_think("Triage", f"{len(items)} priority item(s)", "done")
    if not items and not errors:
        return "✓ Nothing urgent — no high-priority items found across your comms."
    lines = [f"🧭 Communications triage — {len(items)} priority item(s):", ""]
    lines += _fmt_triage_items(items, 12)
    if errors:
        lines += ["", "⚠ Some sources were unavailable (degraded):", *[f"  • {e}" for e in errors]]
        hint = _comms_reconnect_hint(errors)
        if hint:
            lines += ["", hint]
    lines += ["", "Next: create_todos_from_comms · draft_smart_replies · triage_and_brief"]
    return "\n".join(lines)


@tool
def create_todos_from_comms() -> str:
    """Push action items detected by the latest triage into your todo list (via manage_todos).

    Uses the cached triage (runs one if none is fresh), de-duplicates action items, and adds each as
    a todo tagged with its source + who. Safe and non-destructive.
    """
    items, _ = _triage_get()
    if not items:
        return "No triaged items. Run triage_communications first."
    added: list[str] = []
    seen: set = set()
    for it in items:
        for ai in it.get("action_items", []):
            key = ai.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            label = f"{ai[:110]} [{it['source']}: {it['who'][:30]}]"
            _run_tool(manage_todos, action="add", text=label)
            added.append(label)
    if not added:
        return "No action items detected in the current triage to add as todos."
    return f"✅ Added {len(added)} todo(s) from communications:\n" + "\n".join(f"  • {a}" for a in added[:20])


@tool
def draft_smart_replies() -> str:
    """Draft replies for high-priority triaged items (preview in Suggest, create/post in Execute).

    For each high-priority item with a reply target: Gmail → an unsent draft via gmail_create_draft;
    Slack → a threaded reply via slack_post_message. SAFETY: in Suggest mode this only PREVIEWS the
    proposed replies (nothing is created or posted); in Execute mode it creates the Gmail drafts and
    posts the Slack thread replies. Uses the cached triage (runs one if none is fresh).
    """
    items, _ = _triage_get()
    if not items:
        return "No triaged items. Run triage_communications first."
    candidates = [it for it in items if it.get("reply") and it["priority"] >= 0.7]
    if not candidates:
        return "No high-priority items with a reply target to draft."

    suggest = _host.PrimusSession.mode == _host.ExecutionMode.SUGGEST
    made: list[str] = []
    previews: list[str] = []
    for it in candidates:
        r = it["reply"]
        if r["kind"] == "gmail":
            if suggest:
                previews.append(f"📧 To {r['to']} — “{r['subject']}”:\n    {r['suggested'][:160]}")
            else:
                res = _run_tool(gmail_create_draft, to=r["to"], subject=r["subject"], body=r["suggested"])
                made.append(f"📧 {r['to']}: {res}")
        else:  # slack
            if suggest:
                previews.append(f"💬 #{r.get('channel_name', '?')} (thread) reply:\n    {r['suggested'][:160]}")
            else:
                res = _run_tool(
                    slack_post_message, channel_id_or_name=r["channel"],
                    text=r["suggested"], thread_ts=r.get("thread_ts"),
                )
                made.append(f"💬 #{r.get('channel_name', '?')}: {res}")
    if suggest:
        return (
            "[Suggest] Would draft these replies (switch to Execute to create/post):\n\n"
            + "\n\n".join(previews)
        )
    return f"✍ Created {len(made)} draft/reply action(s):\n\n" + "\n".join(made)


@tool
def triage_and_brief() -> str:
    """Full Morning Briefing PLUS a prioritized 'Priority items needing attention' section.

    Composes the standard briefing (calendar/email/weather/news), then runs a fresh triage and
    appends the top priority items + a pointer to ready reply drafts and todo capture. Read-only:
    nothing is sent or posted. Ideal to run each morning or attach to a scheduled agent.
    """
    brief = _compose_morning_briefing()
    items, errors = _triage_gather(days_back=1)
    _LAST_TRIAGE.update({"ts": time.time(), "items": items, "errors": errors})
    extra = ["", "---", "## ⚡ Priority items needing attention"]
    if items:
        extra += _fmt_triage_items(items, 10)
        drafts = [it for it in items if it.get("reply") and it["priority"] >= 0.7]
        if drafts:
            extra += [
                "",
                f"✍ {len(drafts)} reply draft(s) ready — run `draft_smart_replies` "
                "(preview in Suggest, create/post in Execute).",
            ]
        extra += ["", "Run `create_todos_from_comms` to capture the action items as todos."]
    else:
        extra.append("Nothing urgent right now.")
    if errors:
        extra += ["", "_Degraded sources: " + "; ".join(errors) + "_"]
        hint = _comms_reconnect_hint(errors)
        if hint:
            extra += ["", hint]
    return brief + "\n" + "\n".join(extra)


@tool
def ask_grok(question: str, briefing: str = "") -> str:
    """Frontier brain (xAI Grok) for leftover / heavy / need_brain work after local tools miss.

    Do not use for time, date, news, headlines, folder/file counts, knowledge/explain,
    pulse greetings, mail-from-NAME, or listed user_path steps. Never patch primus/ source.
    Budget and loop governed; payloads are redacted before send.
    Key is read after this tool is chosen (XAI_API_KEY, else vault:grok).
    """
    from primus.core.brain_governor import ask_frontier  # noqa: PLC0415

    return ask_frontier(question, briefing)


def build_tools() -> list:
    from primus.core.audit import attach_audit  # noqa: PLC0415
    from primus.core.vault import attach_resolve  # noqa: PLC0415

    return [attach_audit(attach_resolve(t)) for t in [
        TerminalTool(),
        GitTool(),
        read_file,
        write_file,
        delete_file,
        verify_python,
        read_office_file,
        edit_docx,
        edit_xlsx,
        open_in_onlyoffice,
        create_directory,
        move_path,
        copy_path,
        list_directory,
        search_files,
        organize_downloads,
        clean_temp_files,
        sort_client_folders,
        organize_by_extension,
        package_management,
        service_control,
        backup_advisor,
        performance_tune,
        troubleshoot,
        device_management,
        open_application,
        browse_web,
        list_installed_apps,
        system_info,
        system_monitor,
        desktop_control,
        get_datetime,
        get_weather,
        network_status,
        disk_cleanup,
        process_management,
        rclone_status,
        manage_todos,
        manage_notes,
        remember_fact,
        remember_preference,
        log_completed_task,
        recall_memory,
        search_knowledge,
        web_search,
        deep_web_search,
        read_article,
        research_topic,
        get_news,
        search_reddit,
        social_search,
        browser_automate,
        search_web_for_kb,
        learn_knowledge,
        kb_status,
        index_knowledge_folder,
        read_own_code,
        analyze_self,
        search_own_codebase,
        explain_my_architecture,
        propose_code_change,
        propose_new_tool,
        design_and_propose_tool,
        propose_changes_batch,
        remember_self_lesson,
        recall_self_lessons,
        deep_research,
        batch_file_operations,
        long_running_command,
        download_webpage_as_pdf,
        download_pdf_from_url,
        ingest_web_document,
        gmail_auth,
        gmail_auth_code,
        gmail_get_auth_url,
        ensure_gmail_access,
        gmail_status,
        gmail_list_messages,
        gmail_read_message,
        gmail_create_draft,
        gmail_send_email,
        calendar_auth,
        calendar_auth_code,
        calendar_status,
        calendar_list_events,
        calendar_get_event,
        calendar_find_free_slots,
        calendar_create_event,
        gmail_extract_meeting_request,
        suggest_meeting_times_from_email,
        create_meeting_from_email,
        propose_times_in_reply,
        slack_auth,
        slack_status,
        slack_list_channels,
        slack_read_channel,
        slack_read_thread,
        slack_summarize_thread,
        slack_extract_action_items,
        slack_post_message,
        auto_ingest_gmail,
        auto_ingest_calendar,
        auto_ingest_slack,
        auto_ingest_all,
        connections_list,
        connections_status,
        connections_add,
        connections_remove,
        connections_refresh,
        triage_communications,
        create_todos_from_comms,
        draft_smart_replies,
        triage_and_brief,
        morning_briefing,
        schedule_morning_briefing,
        create_full_project,
        build_application_from_spec,
        iterative_code_improver,
        recover_ollama_gpu,
        ask_grok,
    ]]
