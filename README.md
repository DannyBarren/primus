# Primus v1.0

<p align="center">
  <img src="docs/header.png" alt="Primus — local-first operator" width="100%">
</p>

A roommate for your Linux box: files, mail, and shell as you, with one approval gate for anything destructive. Fast facts never leave the laptop. Hard leftover work can call a frontier model under a cap you set — the planner is last resort, not the front door.

---

## Quickstart — clone to chatting in ~2 minutes

```bash
git clone <this-repo> ai-workshop && cd ai-workshop

# 1. Python deps (uv: https://docs.astral.sh/uv/)
uv sync || uv pip install -r requirements.txt

# 2. Ollama + models (skip what you already have)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                     # leave running
ollama pull qwen2.5:7b             # Primus orchestrator
ollama pull qwen2.5-coder:7b       # Forge coding agent
ollama pull llama3.2:3b            # fast chat tier
ollama pull nomic-embed-text       # embeddings for the knowledge base

# 3. Verify, then launch
uv run python admin_assistant.py --check-deps
uv run python admin_assistant.py --local     # http://127.0.0.1:7860
```

Or with Docker (local, bind-mounted state, `127.0.0.1:7860`):

```bash
make up          # builds, pulls models into a volume, starts the stack
```

First things to try in the UI:

1. **Say hello** — pulse turns answer instantly, no planning machinery.
2. **Open the ≡ drawer → Inbox** — without Google credentials you get the **demo fixture
   inbox** so you can explore; connect real Gmail later (see below).
3. **Index the example docs** — `uv run python admin_assistant.py --index examples/kb`,
   then ask *"What is the Acme Q3 proposal price?"* — the answer cites its source.
4. **Personalize (optional)** — copy `examples/operator.example.md`'s JSON into
   `~/.config/primus/operator.json` so Primus knows your name and projects. Nothing
   personal ever lives in this repo.

**Gmail (real inbox):** create a **Desktop app** OAuth client in Google Cloud, save the
JSON as `~/.primus/gmail_credentials.json`, then click **Connect** in the drawer → Inbox
(or ask Primus to "connect my Gmail"). Do the first sign-in on the **host** UI — it opens
a real browser; Docker/headless reuses the token via the bind-mounted `~/.primus`.
See [SECURITY.md](SECURITY.md) before exposing anything beyond localhost.

**Prove it works:** `make eval` runs the smoke suite (`evals/cases.json`) against a
sandboxed data dir — pulse turns stay clean, the Acme price comes back cited, the fixture
inbox answers, and a destructive command in Suggest mode is queued, never executed.
A committed sample run lives at `examples/out/eval_sample.md`.

## Operator tour (this tree)

Menu drawer holds **Mode** (Suggest / Execute), **Brain budget**, **Inbox**, and Status.
Chat is a transcript: no Status footer, no leaked tool JSON. `num_ctx` stays 4096.

Try these in order (fixture mailbox, no live xAI required):

1. **Date + time + headlines** — local clock, then `get_news`. Not a knowledge paragraph.
2. **Last email from NAME + summarize** — fixture `gmail_list_messages` / `gmail_read_message`.
   A miss says so; it does not swap in someone else.
3. **How many folders in ~** — a `[dir]` count sentence, not a code dump.
4. **What is gravity** — one knowledge paragraph (~45s leash). Not the work-timeout shrug.
5. **`my path:` list a HOME folder then summarize** — that order. Suggest still queues `rm`.
6. **Suggest + create a folder** — preview only; the folder is not created until Execute.
7. **Why did you do that** — English from `~/.primus/audit.jsonl`, not a JSON dump.
   `/good` `/bad` stay ratings.

The inner agent sees one **pack** (files / mail / shell / …), not 120 schemas.
Fast facts (1–3) answer in-process with no model and no planner; a `my path:` list runs
your steps in your order; anything left over goes to the graph. Only that leftover tier
can reach `ask_grok` — it is last in `build_tools()`, budget-governed, and leftover
`need_brain` merely *offers* it. Fast winners never call it. Brain budget dials write the
same `~/.primus/brain_budget.json` the governor reads.

Secrets live in the **vault** (`~/.primus/vault/blob`, encrypted). You reference one as
`vault:<id>`, never as plaintext. While the vault is locked, that token resolves to
nothing and the miss is audited by id — the plaintext is only ever decrypted inside a
tool's `_run`, and outbound payloads are redacted before send. Gmail OAuth is *not* in
the vault — it stays at `~/.primus/gmail_credentials.json` + `gmail_token.json`.

Backup is **not** a chat tool: `make export` (or
`uv run python admin_assistant.py --export`) writes
`~/.primus/exports/primus-backup-*.zip`. Tokens and vault blobs stay out unless
`--include-secrets`. Restore: `make import FILE=…` (prints overwrites first).

`make eval` is the clone proof: pulse, Acme price, fixture inbox, polish JSON, Suggest `rm`.

The full operator manual follows below.

---

Primus is a self-hosted AI assistant that runs entirely on your own machine. It talks to you in natural language, runs real system commands, manages your files, searches the internet silently in the background, remembers what matters across sessions, learns from every interaction, and can even read and propose improvements to its own code — all gated behind your approval. Nothing leaves your computer except the web searches you ask for.

It is built around a **dual-model brain**:

- **Primus** — the fast orchestrator and conversationalist (default `qwen2.5:7b`). Handles chat, system administration, file work, memory, and tool use.
- **Forge** — the coding/reasoning specialist (default `qwen2.5-coder:7b`). The router hands off programming, debugging, and heavy reasoning tasks to Forge automatically.
- **Fast / fallback model** — a small, quantized model (default `llama3.2:3b`) that answers pure conversation in a single call with no tools, **and** is used by **intelligent dynamic model switching** (see [section 19](#19-amd-gpu-acceleration--dynamic-model-switching)) to keep simple turns snappy when the big model is slow.

On **AMD Ryzen AI** hardware, Primus auto-configures GPU acceleration (Vulkan/ROCm, flash attention, HSA gfx override) at startup — see [section 19](#19-amd-gpu-acceleration--dynamic-model-switching).

Everything runs through **Ollama** locally. The UI is a dark, cyberpunk-themed **Gradio** web app served only on `127.0.0.1`.

---

## Table of Contents

1. [How Primus Works (Architecture)](#1-how-primus-works-architecture)
2. [Requirements & Dependencies](#2-requirements--dependencies)
3. [Installation](#3-installation)
4. [Launching Primus](#4-launching-primus)
5. [The User Interface](#5-the-user-interface)
6. [Command Safety & Execution Modes](#6-command-safety--execution-modes)
7. [Complete Tool Reference](#7-complete-tool-reference)
8. [Internet Capabilities](#8-internet-capabilities)
9. [Voice Input (Speech-to-Text)](#9-voice-input-speech-to-text)
10. [Memory & Learning Systems](#10-memory--learning-systems)
11. [Self-Improvement](#11-self-improvement)
12. [Slash Command Reference](#12-slash-command-reference)
13. [How to Use Primus Properly So He Improves](#13-how-to-use-primus-properly-so-he-improves)
14. [Configuration Reference](#14-configuration-reference)
15. [Data & File Locations](#15-data--file-locations)
16. [Abilities & Limitations](#16-abilities--limitations)
17. [Troubleshooting](#17-troubleshooting)
18. [Privacy](#18-privacy)
19. [AMD GPU Acceleration & Dynamic Model Switching](#19-amd-gpu-acceleration--dynamic-model-switching)

---

## 1. How Primus Works (Architecture)

Every message you send flows through a layered pipeline designed to answer as fast as possible while still being able to do heavy work when needed.

```
Your message
    │
    ▼
[1] Slash command?  ── yes ──▶ handled directly (/help, /forge, /metrics, …)
    │ no
    ▼
[2] instant_answer()      ── matches time/date/etc ──▶ instant reply (no LLM)
    │ no match
    ▼
[3] fast_tool_answer()    ── clear single-tool request ──▶ run tool directly in code
    │   (weather, open app, web search, news, reddit, system monitor, desktop)
    │ no match
    ▼
[4] ModelRouter.analyze() ── deterministic, regex-scored, NO LLM call
    │   decides: Primus vs Forge · intent · confidence
    ▼
[5] fast_chat_tier?       ── pure conversation ──▶ one light-model call, no tools/RAG
    │ no
    ▼
[6] build_memory_context()── hybrid KB (vector+keyword, scoped) + long-term memory + recent convos + reflection notes
    ▼
[7] LangGraph agent loop  ── plan → step (tool calls) → summarize
    ▼
[8] polish_response()     ── scrub leaked scaffolding (tool-call JSON, Thought:/Action: lines,
                              **Step N** headers) + robotic filler → clean, concise human text
    ▼
[9] append_followup()     ── optionally offer ONE smart next step
    ▼
Clean reply  +  background: learning digest · self-reflection · metrics
```

**Key design principles:**

- **Speed first.** Layers 2–5 answer the majority of everyday requests without ever invoking the heavy agent loop. The deterministic router (layer 4) makes routing decisions with regex scoring, not an LLM, so it costs nothing.
- **Silent background operation.** All internet and research tools run *headless* by default — no browser window pops up, no focus is stolen. A visible browser opens only when you explicitly say "open", "show me", or "pop it up".
- **Clean chat.** Internal reasoning, routing traces, and JSON artifacts are stripped from the chat window. `polish_response()` also scrubs any **leaked tool-call JSON** (e.g. `{"name": "list_tools"}`), `Thought:`/`Action:`/`Observation:` lines, and internal `**Step N**` headers a model might emit as text — so you only ever see the final, synthesized answer. Genuine JSON you ask for (objects with real data keys) is preserved. Reasoning stays in the side "thinking" panel and the console log.
- **Human-gated risk.** Read-only/safe actions run automatically. Destructive or sensitive actions (delete, sudo, install, send) are queued for one-click approval. Primus never modifies its own code without your explicit `/approve edit`.
- **Compounds with use.** Memory, a learning digest, self-reflection, and usage metrics all update in background threads after each turn, so Primus gets more useful the more you use him.
- **Concise, grounded, human voice.** An always-injected operator profile + voice rules (operator background, projects, banned-phrase list, "stay on-topic / no filler" rules — personalized via `~/.config/primus/operator.json`, see `examples/operator.example.md`) ground every reply, and `polish_response()` strips robotic openers/closings so answers read like a sharp human colleague — never corporate boilerplate.
- **On-topic by design.** Retrieval is scoped to high-signal collections (`core_business`/`projects`) for business/engineering queries, and precise intents (finance, weather, news) route to live tools so a stocks question never drifts into unrelated headlines.

### The Router

`ModelRouter.analyze()` scores each message against weighted regex patterns for coding/technical intent (favoring Forge) vs administrative/chat/memory intent (favoring Primus). Thresholds (`router_forge_threshold`, `router_forge_strong_threshold`) decide delegation. A small, **bounded** feedback nudge (≤ `feedback_routing_max_nudge`, default 1.5) learned from your `/good` and `/bad` feedback refines routing over time without overriding the deterministic rules. If Forge is selected but unavailable or too slow, Primus transparently falls back (`forge_fallback_to_primus`).

---

## 2. Requirements & Dependencies

### Platform
- **Linux** (developed/tested on Ubuntu-family; uses `apt`, `wmctrl`, `upower`, `xdg-open`).
- **Python 3.10+** (3.12 recommended; the project ships a `.venv`).
- Recommended: a discrete GPU for faster models and voice, but everything works on CPU.

### Required: Ollama + models
Primus needs [Ollama](https://ollama.com) running locally and at least the orchestrator model pulled.

| Model | Role | Default | Required? |
|-------|------|---------|-----------|
| `qwen2.5:7b` | **Primus** orchestrator (daily driver) | yes | **Required** |
| `qwen2.5-coder:7b` | **Forge** coding/debugging specialist | yes | Required for code delegation |
| `llama3.2:3b` | **Fast chat** tier (max-speed conversation) | yes | Optional but recommended |
| `nomic-embed-text` | Embeddings for the knowledge base / RAG | yes | Required for KB/memory recall |

Alternatives you can swap in (Setup tab lists them): `qwen2.5:14b` (smarter, slower Primus), `mistral:7b`.

### Required Python packages (`PYTHON_PACKAGES`)
The core AI + UI stack:

```
gradio>=5.0
langchain>=0.3, langchain-core>=0.3, langchain-ollama>=0.3,
langchain-community>=0.3, langchain-chroma>=0.2,
langchain-text-splitters>=0.3
langgraph>=0.2, langgraph-prebuilt>=0.1.8
chromadb
pydantic>=2
```

### Optional Python packages (`OPTIONAL_PYTHON_PACKAGES`) — what each unlocks

| Package | Capability it enables |
|---------|----------------------|
| `psutil` | System monitor (CPU/RAM/disk/battery/temps), process management |
| `ddgs` | Web search (DuckDuckGo). Bing/Brave HTML fallback works without it |
| `trafilatura` | Primary clean article/webpage text extraction |
| `newspaper3k` | Secondary article extraction (news articles) |
| `beautifulsoup4` + `lxml` | Structured HTML scraping; search-result + RSS parsing fallback |
| `feedparser` | News / RSS aggregation (`get_news`) |
| `selenium` | Headless browser automation (`browser_automate`), JS-heavy page rendering |
| `pyautogui` | Desktop control (mouse/keyboard/screenshots) — needs an X11 session |
| `pypdf` + `pdfplumber` | Ingest PDF files into the knowledge base (multi-backend, robust) |
| `python-docx` | Read/edit Word `.docx` (tools + KB ingestion) |
| `openpyxl` | Read/edit Excel `.xlsx` (tools + KB ingestion) |
| `python-pptx` | Read/edit PowerPoint `.pptx` (tools + KB ingestion) |
| `faster-whisper` | **Voice input** (local speech-to-text) |
| `sentence-transformers` + `langchain-huggingface` | Local embedding fallback when Ollama embeddings are unavailable |
| `pystray` + `pillow` | System tray icon |
| `piper-tts` / `pyttsx3` | Spoken launch sound ("Primus.") |

### Optional system packages
- APT (recommended): `wmctrl`, `network-manager`, `upower`, `bluetooth`, `bluez`
- APT (optional): `rclone` (cloud status), `chromium` / `google-chrome-stable` (GPU-safe app window)

> Primus **degrades gracefully**: any missing optional package simply disables that one capability. The Setup tab and the startup "Tool & capability check" show exactly what is live (`✓`) and what is missing (`○`) with the exact install command.

---

## 3. Installation

### Quick install (one block)
The Setup tab generates a tailored version of this, but the canonical sequence is:

```bash
# 1. System packages
sudo apt update && sudo apt install -y python3 python3-venv git curl wmctrl network-manager upower bluetooth bluez

# 2. Ollama + models
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                     # leave running
ollama pull qwen2.5:7b             # Primus
ollama pull qwen2.5-coder:7b       # Forge
ollama pull llama3.2:3b            # fast chat
ollama pull nomic-embed-text       # embeddings

# 3. Python deps (from the project folder)
cd ~/ai-workshop
uv pip install gradio>=5.0 langchain>=0.3 langchain-core>=0.3 langchain-ollama>=0.3 \
  langchain-community>=0.3 langchain-chroma>=0.2 langgraph>=0.2 langgraph-prebuilt>=0.1.8 \
  langchain-text-splitters>=0.3 chromadb pydantic>=2
uv pip install psutil pypdf pdfplumber python-docx openpyxl python-pptx \
  ddgs trafilatura selenium pyautogui \
  newspaper3k beautifulsoup4 lxml feedparser faster-whisper \
  sentence-transformers langchain-huggingface pystray pillow

# 4. (Optional) desktop launcher + first run
uv run python admin_assistant.py --install-desktop
uv run python admin_assistant.py
```

### Check your install
```bash
uv run python admin_assistant.py --check-deps   # prints dependency status and exits
uv run python admin_assistant.py --setup        # full setup guide + checks
```

On every normal startup Primus prints a **Tool & capability check** block (e.g. `14/19 capabilities live`) and logs a one-line summary plus warnings if Ollama is unreachable or the Primus model isn't pulled.

---

## 4. Launching Primus

```bash
uv run python admin_assistant.py [options]
```

### Common launch profiles
| Goal | Command |
|------|---------|
| Simple (browser tab) | `uv run python admin_assistant.py` |
| Desktop app window + tray | `uv run python admin_assistant.py --browser --tray` |
| Compact, pinned window | `uv run python admin_assistant.py --browser --tray --compact` |
| Background (tray only) | `uv run python admin_assistant.py --tray` |
| Start on login | `uv run python admin_assistant.py --install-autostart` |

### All command-line flags
| Flag | Description |
|------|-------------|
| `--port N` | Server port (default 7860; auto-bumps if busy) |
| `--host ADDR` | Bind address (default `127.0.0.1`, local-only) |
| `--model NAME` | Override the Primus orchestrator model |
| `--forge-model NAME` | Override the Forge coding model |
| `--browser` | Open an app window on start |
| `--app-window` | Open in a dedicated Chromium app window (GPU-safe) |
| `--tray` / `--no-tray` | Enable / disable the system tray |
| `--compact` | Compact UI mode |
| `--verbose` / `--debug` | Verbose logs (`--debug` also shows Gradio errors in the UI) |
| `--install-desktop` | Install a `.desktop` launcher |
| `--install-autostart` / `--uninstall-autostart` | Manage login autostart |
| `--install-all` | Install desktop + autostart |
| `--setup` | Print setup guide and check deps |
| `--check-deps` | Check dependencies and exit |
| `--index [PATH]` | Index a folder into the knowledge base and exit |
| `--share` / `--share=true/false` | Create a public Gradio link (default: off) |

### Launch reliability (built in)
- **Local-only by default** — keeps `localhost`/`127.0.0.1` out of any HTTP proxy so Gradio's startup self-check can't wrongly fail.
- **Resilient binding** — tries the requested host/port, then bumps the port up to 7 times, then forces a `127.0.0.1` local-only retry.
- **Graceful kwarg fallback** — if your Gradio version rejects a cosmetic launch option, Primus retries with a minimal, always-supported set so the UI still comes up.
- **"Aw, Snap!" guidance** — if the browser crashes on GPU (common on Linux + AMD GPUs), the server is fine; open the printed URL in Firefox, or launch Chromium with `--disable-gpu --disable-software-rasterizer`.

---

## 20. Project structure (modularization complete)

Primus has been fully modularized from the single `admin_assistant.py` file into a clean `primus/` package.
All seven planned slices are extracted and run-verified. At every step the extracted code is imported back into
`admin_assistant.py` so the module-level namespace and runtime behavior stay **100% identical** — no
features, prompts, personality, or guardrails are changed, weakened, or locked.

**Recent additions (v1.0):**
- New core modules: `access.py`, `intent.py`, `present.py`, `session_state.py`
- Communications triage tools in `primus/tools/registry.py`: `triage_communications`, `create_todos_from_comms`, `draft_smart_replies`, `triage_and_brief`
- Docker local + cloud compose + Makefile one-command workflows
- Env-overridable paths in `primus/config.py`

### How to run (unchanged)

```bash
cd ~/ai-workshop
uv run python admin_assistant.py        # launch the UI
uv run python admin_assistant.py --setup   # dependency check / setup guide
```

`admin_assistant.py` is still the single entrypoint. It adds its own directory to `sys.path` and imports
the extracted modules from the `primus/` package, so launching is exactly the same as before.

### Package layout

```
ai-workshop/
├── admin_assistant.py        # entrypoint — launches everything; imports primus.core.* verbatim
└── primus/
    ├── __init__.py           # package metadata + migration roadmap
    ├── core/                 # ✅ migrated (Phase 1) + v1.0 additions
    │   ├── __init__.py        #    re-exports all core symbols
    │   ├── obedience.py       #    GLOBAL_OBEDIENCE
    │   ├── personality.py     #    CORE_OPERATOR_PROFILE
    │   ├── prompts.py         #    FORGE_SYSTEM / PRIMUS_SYSTEM / FAST_CHAT_SYSTEM / planner+step+summary
    │   ├── guardrails.py      #    DANGEROUS_PATTERNS, SAFE_COMMAND_LEADERS, SAFE_LEADING_RE, UNSAFE_IN_SAFE
    │   ├── access.py          #    v1.0 — access control helpers
    │   ├── intent.py          #    v1.0 — intent classification
    │   ├── present.py         #    v1.0 — presentation / polish helpers
    │   └── session_state.py   #    v1.0 — session state management
    ├── config.py             # ✅ migrated (Phase 2) — env-overridable paths + DEFAULT_CONFIG + load/save
    ├── utils/                # 🟡 partial (Phase 3) — gpu.py + patches.py done; more helpers to come
    │   ├── __init__.py        #    re-exports gpu + patches
    │   ├── gpu.py             #    AMD GPU detect/accel/status (reads live CFG via bind_cfg provider)
    │   └── patches.py         #    gradio localhost + schema patches, no-proxy helper, free-port finder
    ├── tools/                # ✅ migrated (Phase 4) — full tool system
    │   ├── __init__.py        #    re-exports build_tools + tools referenced elsewhere
    │   └── registry.py        #    TerminalTool/GitTool + every @tool + build_tools() (via _host.*)
    ├── memory/               # ✅ migrated (Phase 5) — memory + RAG system
    │   ├── __init__.py        #    re-exports singleton accessors + KB/RAG helpers + constants
    │   └── system.py          #    LTM, KnowledgeBase (Chroma), MemorySystem, metrics, reflections
    ├── agents/               # ✅ migrated (Phase 6) — routing, graphs, fast paths, background/scheduled agents
    │   ├── __init__.py        #    re-exports routing/graph API + managers used elsewhere
    │   └── system.py          #    ModelRouter, LangGraph builders, invoke_primus, watchdog, BG/Scheduled mgrs
    ├── ui/                   # ✅ migrated (Phase 7) — the entire Gradio UI
    │   ├── __init__.py        #    re-exports build_ui + prewarm_whisper + run_fallback_setup_server
    │   └── builder.py         #    CYBER_CSS, tabs/panels, handlers, voice (whisper), typing FX, fallback server
    └── self_improve/         # ➰ folded into tools/ — placeholder package (nothing to import)
```

`admin_assistant.py` is now the **host module + entrypoint** (≈3.6k lines, down from the original
≈16.5k): it imports every subsystem back, keeps the shared host surface that modules reach via
`_host.<name>` (config/CFG + model globals, `PrimusSession`, chat/projects, web/shell/system
helpers, the desktop window/tray/sound layer), and owns the launch orchestration (`parse_args`/`main`).

### Editing core directives

Everything in `primus/core/` is plain, fully-editable Python — there are **no read-only files, no
locks, and no new protection layers**. To tweak Primus's obedience, personality, prompts, or command
guardrails, edit the corresponding file in `primus/core/`; the change flows everywhere automatically
because `admin_assistant.py` imports these names back into its own namespace.

> Note: the existing `***CURSOR DO NOT TOUCH***` / "UNLOCKED OBEDIENCE SECTION" comments were preserved
> verbatim (moved alongside their code). They are comments only — nothing is technically enforced, and
> Danny can edit any of it freely.

### Migration status

- **Phase 1 (done + run-verified):** `primus/core/` — obedience, personality, prompts, guardrails
  extracted verbatim and re-imported. Verified: `admin_assistant.py` compiles, `primus.core` imports
  cleanly, every extracted constant is byte-for-byte identical (including prompt composition and the
  `FAST_CHAT_SYSTEM` override order), the full module imports with the AI stack live
  (`HAS_GRADIO=True`, `HAS_AI_STACK=True`), and **the whole app boots — UI binds on
  `http://127.0.0.1:7860`, all models/RAG/scheduler/watchdog initialize.**
- **Phase 2 (done + run-verified):** `primus/config.py` — env-overridable path constants,
  `DEFAULT_CONFIG`, `DEFAULT_INGEST_PATHS`, and the pure `load_config_file` / `save_config_file` /
  `apply_cli_and_sync` logic. The live `CFG` dict, the model globals, `SCRIPT_PATH`, and the global
  rebinding in `init_config()` deliberately stay in `admin_assistant.py` (thin same-named wrappers
  keep every call site unchanged). Verified: both files compile, `primus.config` imports, env
  overrides redirect paths/URL, `init_config()` rebinds globals correctly, and the full app boots.

  New environment variables (all optional; unset ⇒ identical local behavior):

  | Variable | Effect | Default |
  | --- | --- | --- |
  | `PRIMUS_DATA_DIR` | base data dir (`APP_DIR` and everything under it) | `~/.primus` |
  | `PRIMUS_CONFIG_DIR` | config + log dir | `~/.config/primus` |
  | `PRIMUS_DOWNLOADS_DIR` | downloads dir | `~/Downloads` |
  | `OLLAMA_URL` | Ollama endpoint (alias of `OLLAMA_HOST`) | `http://localhost:11434` |

  Precedence is unchanged: env vars set **defaults**; a saved `config.json` still wins (path vars,
  which aren't stored in `config.json`, always apply — ideal for Docker volume mounts).
- **Phase 3 (done + run-verified):** `primus/utils/` — `gpu.py` (AMD GPU detection, acceleration env
  setup, status, and effective-mode reporting) and `patches.py` (`_ensure_localhost_direct`, the two
  gradio compatibility patches, and `_find_free_port`). The GPU functions read the **live** `CFG`
  through a tiny provider: `admin_assistant.py` calls `gpu.bind_cfg(lambda: CFG)` once, so the lambda
  resolves the current (rebound) `CFG` at call time — letting the functions keep their **exact
  original signatures** with zero wrappers and zero call-site changes. The GPU status cache moved into
  `gpu.py`; the three external cache-busting sites now call `gpu.bust_status_cache()`. Verified: all
  files compile, `gpu._cfg() is CFG` (live identity), every GPU/patch function works, `set_gpu_backend`
  still mutates the live config, and the **full app boots** — GPU env applied, AMD Strix detected, the
  GPU→CPU fallback warning fires, and the UI binds on `http://127.0.0.1:7860`.
- **Phase 4 (done + run-verified):** `primus/tools/` — the **entire tool system** (`TerminalTool`,
  `GitTool`, all 68+ `@tool` functions: file ops, system/admin, memory, web/research, knowledge-base,
  self-introspection + self-improvement, background/coding tools, and `build_tools()`). Moved verbatim
  into `registry.py`; the only mechanical change is that references to `admin_assistant` module
  globals are reached through the live host module as `_host.<name>`. This was chosen after an AST
  dependency analysis showed the block references 55 host globals (none shadowed), with 3 rebound
  (`CFG`, `DEFAULT_MODEL`, `FORGE_MODEL`) and 3 forward-declared (`make_chat_ollama`,
  `coerce_message_text`, `enforce_sequential_models`) — all handled uniformly and lazily via `_host`,
  so the circular import is trivially safe and behavior is identical. `admin_assistant.py` imports the
  22 names it still references (incl. `build_tools`) back from `primus.tools`. The host module is
  resolved from `sys.modules` (preferring `"admin_assistant"`, falling back to `"__main__"`) so running
  as the entrypoint never re-executes the file. Verified: F821-clean (no undefined names introduced),
  both files compile, **72 tools register**, individual tools execute (`get_datetime`, `system_info`),
  routing is unchanged (forge↔primus), a real `create_agent` run **issues and executes a tool call**
  (`get_datetime`), and the **full app boots** on `http://127.0.0.1:7860`.

  > Pre-existing note: `FORGE_FAST_SYSTEM` (referenced once at `admin_assistant.py:7623`) is undefined
  > in the original file too (present identically in pre-Phase-4 backups) — a latent issue in an
  > apparently-unhit path. Left untouched to preserve 100% identical behavior.
- **Phase 5 (done + run-verified):** `primus/memory/` — the **entire memory system**: long-term
  memory (`load_memory`/`save_memory`/`sync_core_memory`), the multi-collection Chroma **Knowledge
  Base** (ingestion, search, RAG) with its caching-embeddings wrapper and document extraction, the
  `MemorySystem` (session + structured LTM + learning digest), `MetricsTracker`, reflections/
  interactions, and KB/recall helpers (`build_memory_context`, `retrieve_rag_context`,
  `init_knowledge_base`, `index_projects`, …). Moved verbatim into `system.py` (three contiguous
  ranges); host globals are reached via `_host.<name>`. An AST analysis confirmed the only def-time
  externals are stdlib/typing (imported locally) and **nothing references memory names at module-load
  outside memory**, so the import-back sits cleanly at the first memory range. The cached singletons
  (`KnowledgeBase`/`MemorySystem`/`MetricsTracker`, the KB lock + index status) live in `system.py`;
  `admin_assistant.py` re-imports the 19 accessors/helpers/constants used elsewhere, so call sites
  (and the tools' `_host.get_kb`/`_host.get_memory_system` references) are unchanged. Verified:
  F821-clean (no new undefined names), both files compile, `memory._host.CFG is CFG` (live), the KB /
  MemorySystem / Metrics singletons build, `build_memory_context` and the `recall_memory` + `kb_status`
  tools run, and the **full app boots** — KB embeddings init, Chroma RAG live, models warm, UI on
  `http://127.0.0.1:7860`.
- **Phase 6 (done + run-verified):** `primus/agents/` — the **entire agent/orchestration layer**:
  deterministic routing (`RoutingDecision`/`ResponseCache`/`ModelRouter`), the LangGraph builders
  (`build_agent_graph`/`build_primus_graph`/`build_forge_graph`/`init_agent_graphs`), model plumbing
  (`make_chat_ollama`, warm-up, `enforce_sequential_models`, `DynamicModelManager`), the fast paths
  (`instant_answer`/`fast_tool_answer`/`classify_complexity`/`route_prompt`/`_fast_chat_answer`/
  `_fast_coding_answer`), `invoke_primus` + the turn watchdog, and the **background + scheduled agent
  systems** (`BackgroundAgentManager`/`BackgroundTaskManager`/`ScheduledTaskManager` + their renderers).
  Moved verbatim into `system.py` (three clusters) with host globals reached via `_host.<name>`. An
  extra cross-reference pass drove three decisions: (1) **no shadowing** exists, so blanket `_host.`
  prefixing of the 63 host globals is safe; (2) `_agent_graphs` is rebound from **both** sides
  (`set_config_models` and `init_agent_graphs`), so it stays **host-owned** — its definition remains in
  `admin_assistant.py`, moved code uses `_host._agent_graphs`, and the lone `global _agent_graphs` is
  dropped; (3) the system-prompt `import` block sits inside the moved region, so those 7 prompts resolve
  locally in `system.py` (not prefixed) and the langchain/langgraph surface is bound from `_host` at the
  top. `admin_assistant.py` re-imports the **30 names** referenced by the host or reached via `_host`
  by tools/memory (incl. `make_chat_ollama`, `enforce_sequential_models`, `gradio_history_to_messages`).
  Verified: both files compile, the only F821 is the pre-existing `FORGE_FAST_SYSTEM` (now relocated
  verbatim — no new undefined names), `system._host is admin_assistant`, routing classifies, real
  graphs build (`forge`/`primus`/`primus_fast`) and set `PrimusSession.graph`, fast paths run, the
  BG/Scheduled managers instantiate + render, and the **full app boots** — scheduler + watchdog start,
  models warm, agent stack/RAG live, UI on `http://127.0.0.1:7860`.

  > Pre-existing note: `FORGE_FAST_SYSTEM` (now at `primus/agents/system.py:1921`) is undefined in the
  > original file too — kept bare so the exact original NameError-on-that-path behavior is preserved.
- **Phase 7 (done + run-verified — the finale):** `primus/ui/` — the **entire Gradio presentation
  layer**: the cyberpunk theme (`CYBER_CSS`) + keyboard JS, the chat/project sidebar, the right-side
  status / background-agent / scheduled-task panels, the Setup / Knowledge / Appearance tabs, voice
  input (faster-whisper: `get_whisper_model`/`transcribe_audio`/`prewarm_whisper`), the typing effect,
  **every Gradio event handler/closure**, `build_ui` itself, and the no-Gradio fallback HTTP setup
  server. Moved verbatim into `builder.py`; host globals are reached via `_host.<name>` and `gr` (the
  gradio module, or `None` when absent) is bound from the host. A cross-reference pass confirmed
  **no shadowing** (safe blanket `_host.` prefixing of 104 host names across 399 lines), **no def-time
  host dependencies**, only **3 need-back** names (`build_ui`, `prewarm_whisper`,
  `run_fallback_setup_server` — used by `main()`), and that the rebound whisper-cache globals are
  UI-internal. Verified: both files compile, **F821-clean** (zero undefined names), `builder._host is
  admin_assistant`, `gr` binds to the live gradio module, and the **full app boots** — the complete
  Blocks UI constructs and serves on `http://127.0.0.1:7860` (gradio startup-events `200`), whisper
  STT loads, scheduler + watchdog start, models warm, RAG live.

### ✅ Modularization complete

All seven planned slices are extracted and run-verified. `admin_assistant.py` shrank from ≈16,500
lines to ≈3,640 (host surface + launch orchestration); the subsystems live in `primus/` and are
imported back so every call site is unchanged and behavior is **100% identical**. Launch is still
just `uv run python admin_assistant.py`. See **Cloud / Docker** below for containerized deployment.

> Run note: launch with **`uv run python admin_assistant.py`** (the project venv is uv-managed). Direct
> `.venv/bin/python` is currently broken (a corrupted `exec_prefix` in the uv-managed CPython), but
> `uv run` patches the environment correctly and all dependencies are present.

### Cloud / Docker

The same project runs in two Docker modes from one image. The repo ships a multi-stage `Dockerfile`,
two compose files (`docker-compose.yml` for **local**, `docker-compose.cloud.yml` for **cloud**), a
`.dockerignore`, a `requirements.txt` (core server deps), and a `Makefile` for one-command use.

| | Local (`make up`) | Cloud (`make cloud`) |
|---|---|---|
| Data/config | bind-mount host `~/.primus` + `~/.config/primus` | named volumes `primus-data` / `primus-config` |
| Port | `127.0.0.1:7860` (loopback only) | `7860` on all interfaces (front with auth/proxy) |
| User | host UID/GID (writable bind mounts) | image's non-root `primus` (uid 1000) |
| GPU | off (CPU) | NVIDIA block ready to uncomment |
| Compose file | `docker-compose.yml` | `docker-compose.cloud.yml` (standalone) |

**Shared design (both modes):**

- **Three services.** `ollama` runs the LLM server (models persist in a volume); `ollama-init` is a
  one-shot that **auto-pulls the models** (`qwen2.5:7b`, `qwen2.5-coder:7b`, `llama3.2:3b`,
  `nomic-embed-text`) on first run; `primus` builds this repo and talks to Ollama via
  `OLLAMA_URL=http://ollama:11434`.
- **Ordered, health-gated startup.** `primus` waits for `ollama` to be healthy **and** for
  `ollama-init` to finish (`service_completed_successfully`), so the UI never comes up before its
  models exist. `primus` has its own healthcheck on `/gradio_api/startup-events`.
- **Multi-stage, non-root image.** Build tools stay in the builder stage; the runtime is slim, runs
  as a non-root `primus` user, and caches models (HF/whisper) under `/data/.cache` so they persist and
  stay writable whatever UID the container runs as.
- **Env-driven paths.** The container sets `PRIMUS_DATA_DIR=/data` and `PRIMUS_CONFIG_DIR=/config`;
  state survives rebuilds and is identical to a host run.
- **Headless extras.** Desktop-only features (tray, app-window, launch sound, `pyautogui`) are
  gracefully skipped; voice STT (`faster-whisper`) is included and works.

#### Local Docker

Behaves exactly like `uv run python admin_assistant.py`: it shares `~/.primus` and `~/.config/primus`
with the host, so chats/knowledge/config are the same files either way.

```bash
make up            # build + start (ollama → auto-pull models → primus), data bind-mounted
make logs          # follow startup
xdg-open http://127.0.0.1:7860
make down          # stop (data is kept on the host)
```

`make` exports your `UID`/`GID` automatically so the bind-mounted dirs stay writable. (Without `make`:
`export UID=$(id -u) GID=$(id -g) && docker compose up -d --build`.)

#### Cloud Deployment (Vast.ai · RunPod · VPS)

Production stack with named volumes, exposed port, and a ready-to-enable GPU block. Run it standalone
(don't combine it with the local file):

```bash
make cloud         # docker compose -f docker-compose.cloud.yml up -d --build
make cloud-logs    # follow startup
make cloud-down    # stop (named volumes persist)
```

- **Enable the GPU.** On a GPU host with `nvidia-container-toolkit` (Vast.ai/RunPod GPU instances have
  it), uncomment the `deploy.resources.reservations.devices` (NVIDIA) block under the `ollama` service
  in `docker-compose.cloud.yml`. AMD/ROCm needs a different Ollama image/runtime; the in-app AMD Ryzen
  AI acceleration applies to the **host** launcher, not the container.
- **Vast.ai / RunPod.** Provision an Ubuntu + Docker (+ NVIDIA) template, then:

```bash
git clone <your-repo> primus && cd primus
# Edit docker-compose.cloud.yml: uncomment the NVIDIA `deploy:` block under `ollama` to use the GPU.
make cloud && make cloud-logs
# Reach the UI via the provider's port mapping for 7860, an SSH tunnel
# (ssh -L 7860:localhost:7860 user@host), or a reverse proxy with auth.
```

- **⚠ No built-in auth.** Port 7860 is exposed on all interfaces; only reach it through the provider's
  tunnel, an SSH tunnel, a firewalled network, or a reverse proxy with auth (Caddy/Nginx). Never put it
  on the open internet unprotected.
- **Persistence.** Knowledge base, memories, chats, and config live in the `primus-data` /
  `primus-config` named volumes. `make cloud-down` keeps them; `make cloud-clean` deletes them.

**Validated:** `docker build --check` reports no warnings and both compose files pass
`docker compose config`. Host launch is unchanged — `uv run python admin_assistant.py` (or `make run`)
still works identically.

---

*Primus v1.0 — your private, local executive assistant. Built to be fast, decisive, and to get better every time you use it. Now GPU-accelerated and self-tuning on AMD Ryzen AI.*
