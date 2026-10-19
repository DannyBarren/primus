# Primus

<p align="center">
  <img src="docs/header.png" alt="Primus — local-first operator" width="100%">
</p>

**v0.9.4** — a local-first Linux operator agent. Files, mail, and shell run as
you. Destructive work hits one approval gate. Fast facts never leave the
laptop. A leftover planner is last resort, not the front door.

The long tool/slash/config reference lives in
[docs/operator-manual.md](docs/operator-manual.md). Security policy:
[SECURITY.md](SECURITY.md).

---

## Problem

A 7B local model cannot be a reliable sysadmin if it sees 120 tool schemas, a
5k-token standing prompt, and a live shell in the same turn. Primus splits the
job:

1. **Pulse / fact** — greetings, time, fixture mail, KB lookup. No planner.
2. **Fast tool** — one obvious tool (`list_directory`, `get_weather`, allowlisted
   `open_application`) in-process.
3. **Leftover graph** — plan → one **tool pack** (~8–14 schemas) → summarize.
   A frontier call (`ask_grok`) is budget-gated and last.

The product question is not “can the model do everything?” It is “can a
careless turn delete `~` or bind a shell to the LAN?” Those are engineering
gates, not prompt text.

---

## Architecture

```mermaid
flowchart TD
  msg[User message] --> slash{Slash command?}
  slash -->|yes| direct[Handler: /approve /help /forge …]
  slash -->|no| pulse[instant_answer / pulse]
  pulse -->|miss| fast[fast_tool_answer]
  fast -->|miss| recv[Receptionist + select_pack]
  recv --> graph[Outer LangGraph: plan → step → summarize]
  graph --> pack[Inner create_agent<br/>one pack + named extras]
  pack --> tools[Terminal / files / mail / web / …]
  tools --> risk[assess_command_risk + _guard_path]
  risk -->|safe| run[Execute]
  risk -->|review / dangerous| queue[Approval queue]
  run --> polish[polish_response]
  queue --> polish
  polish --> ui[Gradio UI on 127.0.0.1]
```

**120 tools** are registered (`TerminalTool`, `GitTool`, plus 118 `@tool`
functions). The inner agent never sees all of them. Packs live in
`primus/core/tool_packs.py` and are stored in a process-wide turn slot (plus
graph state) so the worker thread that runs the graph sees the same pack the
receptionist chose.

---

## Design trade-offs

| Choice | Why | Cost |
| --- | --- | --- |
| Tool packs, not 120 schemas | A 7B planner drowns in schema tokens | Tools not in the chosen pack are invisible unless named |
| Suggest mode default | Mutating tools preview; `rm` always queues | Existing `config.json` that saved `execute` still wins |
| Classifier + allowlist, not a sandbox | No extra VM; works on a laptop HOME | Block lists have gaps; tests encode the ones we closed |
| Shared-secret Gradio auth | Enough to stop an open cloud port | Not SSO, not per-tool ACLs |
| `num_ctx=4096` | Snappy local 7B | `PRIMUS_SYSTEM` + profile is several thousand tokens; retrieval can be evicted. Not rewritten in this release |
| Outer graph uncheckpointed | Avoids cross-turn state bleed | A killed leftover turn does not resume mid-plan |
| Fixture Gmail | Clone-ready inbox without OAuth | Demo data only; never send from fixtures |

---

## How to run

```bash
git clone <this-repo> primus && cd primus

# 1. Python deps (pinned in requirements.txt)
uv sync || uv pip install -r requirements.txt

# 2. Ollama + models (skip what you already have)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:7b             # orchestrator
ollama pull qwen2.5-coder:7b       # Forge
ollama pull llama3.2:3b            # fast chat
ollama pull nomic-embed-text       # KB embeddings

# 3. Check, then launch on loopback (no auth required)
uv run python admin_assistant.py --check-deps
uv run python admin_assistant.py --local     # http://127.0.0.1:7860
```

Docker (local, bind-mounted state, host port `127.0.0.1:7860`):

```bash
cp .env.example .env                 # or: make up writes a token for you
# PRIMUS_AUTH_TOKEN is required — the container process binds 0.0.0.0
make up
```

Sign in as user `primus` with the token from `.env`. Cloud / GPU hosts:
`make cloud` (publishes `:7860`, still requires the token). Front a public
box with a proxy or SSH tunnel anyway.

**Do not** pass `--share` or `--host 0.0.0.0` without `PRIMUS_AUTH_TOKEN`.
Primus exits on startup if you do.

Hermetic tests (no Ollama, no Gmail, no network):

```bash
pip install pytest ruff
make test          # path guards, risk classifier, routing, auth policy
make lint
```

CI runs ruff, pytest, and `docker build` — see `.github/workflows/ci.yml`.

---

## Security model

Full policy: [SECURITY.md](SECURITY.md). Short version:

- **Auth.** `PRIMUS_AUTH_TOKEN` / `PRIMUS_PASSWORD` (user `PRIMUS_AUTH_USER`,
  default `primus`). Remote bind or `--share` without it → refuse to start.
- **Bind.** Default `127.0.0.1`. Docker publishes loopback on the host; the
  process inside the container is `0.0.0.0` and therefore requires the token.
- **Mode.** Default **Suggest**. Review-tier shell and mutating file/project
  tools preview or queue. Dangerous shell always queues.
- **Launch path.** `open_application` and the `run <words>` fast path do not
  spawn raw user text. `run rm -rf ~` is classified and queued.
- **Classifier.** `find ~ -delete` is no longer a false safe (the old `\b-delete`
  never matched). `$()`, backticks, newlines, lone `&`, `env <prog>`,
  `printenv`, and `git -C … push --force` are covered. See `tests/test_risk.py`.
- **Paths.** Writes stay under `$HOME`. FileMaid blocks `/etc`, `/usr`, … and
  junk segments (`.git`, `.venv`). `is_write_blocked` is the write gate.
- **Gmail scopes.** `gmail.readonly`, `gmail.compose`, `gmail.send` only — not
  `gmail.modify` or `https://mail.google.com/`.
- **Approvals.** Bare `/approve` runs the queued command (same as `/approve
  run`). It does not leave a dangling one-shot allow flag.

Anyone who passes Gradio auth is you. This is not a multi-tenant product.

---

## Known limitations / failure modes

- **Linux only.** Assumes `apt`, `xdg-open`, `wmctrl`. No macOS/Windows support.
- **Local 7B quality.** Routing and packs compensate; they do not make the
  model senior-engineer-smart. Forge is a second 7B coder, not GPT-class.
- **System prompt vs context.** `PRIMUS_SYSTEM` plus the operator profile is
  several thousand tokens against `ollama_num_ctx=4096`. Leftover turns can
  drop retrieval. Shrinking the standing prompt is deferred (behavior risk).
- **90-character fast-path cap.** After the high-value open/news/weather
  routes, `fast_tool_answer` returns `None` for messages longer than 90
  characters (unless a URL is present). Long “run …” lines still hit the
  opener *before* that cap.
- **Pack coverage.** Not every registered tool is in a pack. Named extras can
  be bound; invented names (`gmail_get_latest`) are refused.
- **Classifier is a list.** New dangerous idioms will exist. Prefer Suggest
  mode and do not expose the UI past loopback without a token.
- **Outer graph is not checkpointed.** Only the inner `create_agent` has a
  checkpointer. A killed leftover turn does not resume mid-plan.
- **Ollama required for chat.** No Ollama → no leftover/Forge/fast-chat.
  Fast tools and the fixture inbox still work.
- **Optional deps degrade.** Missing `selenium` / `faster-whisper` / Google
  libraries disable that tool, they do not crash the process.
- **Gmail send / Slack post / calendar create** need a real account. This
  repo’s tests never call those live APIs.

---

## Evals

`evals/cases.json` is a six-case smoke suite (`scripts/eval_primus.py`). Chat
cases need a running Ollama with the configured models. Tool / unit / safety
cases are deterministic.

**Measured in this change:** not measured against a live model. The hermetic
pytest suite (path guards, risk classification, pack/auth routing) is what CI
runs. Do not treat `examples/out/eval_sample.md` as a current score — it is a
committed sample from an earlier run.

To run the smoke eval yourself (sandboxed `$PRIMUS_DATA_DIR`, no real mailbox):

```bash
make eval    # all cases; chat cases fail closed if Ollama is down
# uv run python scripts/eval_primus.py --only polish_json safety_rm gmail_fixture
```

---

## License

Copyright (c) 2026 Daniel Lee Barren (Danny Barren). Personal, educational,
and other non-commercial use is welcome. Commercial use needs his written
permission. Terms: [LICENSE](LICENSE) (PolyForm Noncommercial 1.0.0).
Attribution: [NOTICE](NOTICE).
