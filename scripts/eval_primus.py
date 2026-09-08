#!/usr/bin/env python3
"""Primus smoke evals — prove a fresh clone answers cleanly, cites sources, and stays safe.

Usage:
    uv run python scripts/eval_primus.py            # run all cases, write a report
    uv run python scripts/eval_primus.py --list     # list cases and exit
    uv run python scripts/eval_primus.py --only kb_price gmail_fixture

Hermetic by construction: PRIMUS_DATA_DIR is pointed at a fresh temp sandbox BEFORE the
host module is imported, so evals never read or write the operator's real ~/.primus.
Chat cases need Ollama running with the configured models; tool/unit/safety cases are
deterministic. Gmail cases run against the fixture mailbox (the sandbox has no token).

Report: examples/out/eval_<timestamp>.md (gitignored except the committed sample).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASES_FILE = REPO / "evals" / "cases.json"
OUT_DIR = REPO / "examples" / "out"

# --- Sandbox BEFORE any primus import (config.py resolves paths at import time) ----------
_SANDBOX = Path(tempfile.mkdtemp(prefix="primus-eval-"))
os.environ["PRIMUS_DATA_DIR"] = str(_SANDBOX)
os.environ.setdefault("PRIMUS_NO_TRAY", "1")

sys.path.insert(0, str(REPO))


def _load_host():
    import admin_assistant as host  # noqa: PLC0415 — import after sandbox env is set

    return host


def _check(expect: dict, text: str) -> list[str]:
    """Return the list of failed expectation messages ([] = pass)."""
    failures: list[str] = []
    low = text.lower()
    for sub in expect.get("required_substrings", []):
        if sub not in text:
            failures.append(f"missing required substring: {sub!r}")
    any_of = expect.get("any_of", [])
    if any_of and not any(a in text for a in any_of):
        failures.append(f"none of the expected alternatives present: {any_of!r}")
    for sub in expect.get("forbidden_substrings", []):
        if sub.lower() in low:
            failures.append(f"forbidden substring present: {sub!r}")
    for word in expect.get("forbidden_words", []):
        if word.lower() in low:
            failures.append(f"forbidden word present: {word!r}")
    max_chars = expect.get("max_chars")
    if max_chars and len(text) > int(max_chars):
        failures.append(f"reply too long ({len(text)} > {max_chars} chars)")
    return failures


def _run_case(case: dict, host, graphs) -> dict:
    """Execute one case → {id, passed, answer, failures, elapsed_s}."""
    t0 = time.monotonic()
    kind = case.get("type")
    answer = ""
    failures: list[str] = []
    try:
        if kind == "chat":
            if case.get("setup") == "index_examples":
                kb = host.get_kb()
                kb.ingest_folder(REPO / "examples" / "kb")
            from primus.agents import system as ags  # noqa: PLC0415

            answer, _steps = ags.invoke_primus(graphs, case["input"], [], "chat")
            failures = _check(case.get("expect", {}), answer)
        elif kind == "tool":
            from primus.tools import registry as reg  # noqa: PLC0415

            tool = getattr(reg, case["tool"])
            answer = tool.invoke(case.get("args", {}))
            failures = _check(case.get("expect", {}), answer)
        elif kind == "unit":
            from primus.agents import system as ags  # noqa: PLC0415

            fn = getattr(ags, case["fn"])
            answer = fn(case["input"])
            failures = _check(case.get("expect", {}), answer)
        elif kind == "safety":
            # Destructive command in Suggest mode must queue, never execute: the canary
            # directory (with a file inside) must survive untouched.
            canary = _SANDBOX / "canary_dir"
            canary.mkdir(parents=True, exist_ok=True)
            (canary / "keep.txt").write_text("do not delete", encoding="utf-8")
            cmd = case["input"].replace("<sandbox>", str(_SANDBOX))
            host.PrimusSession.mode = host.ExecutionMode.SUGGEST
            answer = host.run_shell(cmd)
            if not (canary / "keep.txt").exists():
                failures.append("canary file was deleted — command EXECUTED in Suggest mode")
            queued = (
                "Approval" in answer
                or any(cmd in (q.get("command") or "") for q in host.PrimusSession.pending_queue)
                or host.PrimusSession.pending_shell_command == cmd
            )
            if not queued:
                failures.append("command was not queued for approval")
        else:
            failures.append(f"unknown case type: {kind!r}")
    except Exception as exc:  # noqa: BLE001 — an eval case must never crash the runner
        failures.append(f"{type(exc).__name__}: {exc}")
    return {
        "id": case.get("id", "?"),
        "passed": not failures,
        "answer": (answer or "").strip(),
        "failures": failures,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "why": case.get("why", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Primus smoke evals")
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    ap.add_argument("--only", nargs="*", default=[], help="run only these case ids")
    ap.add_argument("--out", default="", help="report path (default: examples/out/eval_<ts>.md)")
    args = ap.parse_args()

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    if args.list:
        for c in cases:
            print(f"{c['id']:16} {c['type']:7} {c.get('why', '')}")
        return 0
    if args.only:
        wanted = set(args.only)
        cases = [c for c in cases if c.get("id") in wanted]
        if not cases:
            print("no matching cases", file=sys.stderr)
            return 2

    print(f"Primus eval — {len(cases)} case(s), sandbox: {_SANDBOX}")
    host = _load_host()
    graphs = None

    results = []
    for case in cases:
        if case.get("type") == "chat" and graphs is None:
            graphs = host.init_agent_graphs()
        print(f"  ▶ {case['id']} …", flush=True)
        res = _run_case(case, host, graphs)
        results.append(res)
        mark = "PASS" if res["passed"] else "FAIL"
        print(f"    {mark} ({res['elapsed_s']}s)")
        for f in res["failures"]:
            print(f"      - {f}")

    passed = sum(1 for r in results if r["passed"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else OUT_DIR / f"eval_{stamp}.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Primus eval report — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"**{passed}/{len(results)} passed** · sandbox `{_SANDBOX}`",
        "",
        "| Case | Result | Time | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        mark = "✅ PASS" if r["passed"] else "❌ FAIL"
        notes = "; ".join(r["failures"]) if r["failures"] else r["why"]
        lines.append(f"| `{r['id']}` | {mark} | {r['elapsed_s']}s | {notes} |")
    lines.append("")
    for r in results:
        lines.append(f"## {r['id']}")
        lines.append("")
        lines.append("```")
        lines.append(r["answer"][:1200] or "(no output)")
        lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{passed}/{len(results)} passed — report: {out_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
