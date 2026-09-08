# Primus

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

## 5. The User Interface

The UI is a dark, cyberpunk-themed Gradio web app with three tabs.

### Chat tab
- **Mode selector** — `Suggest` vs `Execute` (see [Command Safety](#6-command-safety--execution-modes)).
- **Compact** / **Pin** checkboxes, **Tray** and **Win** buttons for window control.
- **Chat window** — clean, message-style conversation. No JSON, no reasoning dumps.
- **Thinking panel** — a collapsible side view showing Primus's live steps (routing, tool calls, status) without cluttering the chat.
- **Command queue row** (appears when a risky command is queued):
  - **Execute** — run the highlighted risky command once
  - **Approve All** — run the queued safe batch
  - **Modify** — drop the command into the input box to edit
  - **Dismiss** — discard it
- **Input box** + **Send** + **■ Halt** (red stop button — immediately cancels ongoing agent activity).
- **Voice row** — 🎤 mic (tap to record, tap again to transcribe & send) + **Auto-send** toggle + a live voice status line. (Hidden if voice is disabled or `faster-whisper` isn't installed.)
- **Workflows dropdown** — common multi-step suggestions, plus quick buttons: history `↑`, clear `Clr`, export `Exp`, refresh `↻`.
- **Shortcuts & modes** accordion with the full hotkey/slash reference.

**Keyboard shortcuts:** `Ctrl+K` focus input · `Ctrl+L` clear chat · `Ctrl+S` Suggest mode · `Ctrl+E` Execute mode · `Ctrl+H` / `Esc` hide to tray · `↑` previous command.

### Setup / Status tab
- Live **dashboard** of Ollama status, models present, and every capability (`✓`/`○`).
- One-click **install scripts** for whatever is missing.
- **Voice settings** — enable/disable, model size (`tiny`/`base`/`small`/`medium`), and language. "Apply voice" reloads the speech model.
- Model selection and other settings.

### Knowledge tab
- **Drag-and-drop** documents (PDF, DOCX, XLSX, PPTX, ODT, Markdown, code, text, etc.) to ingest into the knowledge base.
- Choose a **category** and add metadata; files are copied to `~/.primus/knowledge/uploads/` and indexed into Chroma.
- View KB status and manage indexed content.

---

## 6. Command Safety & Execution Modes

Primus can run real shell commands. Safety is layered:

**Two modes (toggle in the Chat tab or with `Ctrl+S` / `Ctrl+E`):**
- **Execute** (default) — safe, read-only commands auto-run; risky ones still queue for approval.
- **Suggest** — even review-tier commands are queued; nothing runs until you approve.

**Auto-run (safe):** `ls`, `pwd`, `cat`, `tree`, `git status`, and similar read-only inspection commands.

**Always queued for approval (risky):** `rm`, `sudo`, `mv`, package installs, service changes, sending — anything destructive, privileged, or irreversible.

Approve via the queue buttons or slash commands: `/approve run` (the highlighted one), `/approve all` (safe batch), `/reject` (discard). The **■ Halt** button stops any in-progress activity instantly.

---

## 7. Complete Tool Reference

Primus exposes **57 tools** to the agent. They run silently in the background by default. Grouped by purpose:

### Terminal & version control
- **TerminalTool** — run shell commands (with the safety model above).
- **GitTool** — git operations (status, diff, commit, etc.).

### File system
- **read_file**, **write_file** — read/write file contents.
- **verify_python** — static code check used by Forge: compiles (`py_compile`, no execution) and lints (`ruff` if installed) a file or snippet, returning the exact errors to fix. Safe — it never runs the code.
- **create_directory**, **move_path**, **copy_path** — directory and path operations.
- **list_directory** — list folder contents (Desktop, Downloads, anywhere).
- **search_files** — find files by name/pattern across the filesystem.

### Office documents (OnlyOffice / Microsoft Office)
First-class read/write/edit for `.docx`, `.xlsx`, `.pptx` — built for reports, client deliverables, templates, job docs, etc. **Edits are never destructive:** a timestamped `.bak` backup is written before any change, edits are confined to your home folder, and in **Suggest** mode tools only *preview* the change instead of applying it.

- **read_office_file** — read/summarize `.docx` (paragraphs + tables), `.xlsx` (cell grids; pass `sheet_name` to target a tab, `max_rows` to cap output), and `.pptx` (slide text + tables). Read-only; also falls back to PDF/ODT/text. Example: *"Read ~/Clients/acme/proposal.docx and summarize it."*
- **edit_docx** — edit Word docs with `action` = `replace` (swap `find`→`replace` across paragraphs & tables), `append` (add paragraphs), `add_heading`, or `add_table` (from CSV-like `table_data`, e.g. `"Metric,Value\nRevenue,135k"`). Creates the file for `add_*` if missing. Example: *"Add a heading 'Q2 Report' and a revenue table to report.docx."*
- **edit_xlsx** — update cells or append rows in Excel via `updates` (`"A1:Revenue,B2:=SUM(C1:C10)"` — a leading `=` becomes a live formula) and/or `new_data` (`{"headers": [...], "rows": [[...]]}`). Creates the file/sheet if missing.
- **open_in_onlyoffice** — launch a document in **OnlyOffice Desktop Editors** (auto-detected); falls back to the system default app with an install hint if OnlyOffice isn't present.

> Requires `python-docx`, `openpyxl`, and `python-pptx`. Install with: `uv pip install python-docx openpyxl python-pptx`. If a package is missing, the matching tool returns a clear, actionable message instead of failing. These formats are also ingestible into the Knowledge Base.

### File organization
- **organize_downloads** — tidy the Downloads folder.
- **clean_temp_files** — remove temporary/junk files.
- **sort_client_folders** — organize client/project folders.
- **organize_by_extension** — group files by type.

### System administration
- **package_management** — inspect/manage system packages.
- **service_control** — manage systemd services.
- **backup_advisor** — backup guidance/checks.
- **performance_tune** — performance tuning suggestions/actions.
- **troubleshoot** — guided system troubleshooting.
- **device_management** — devices (Bluetooth, network adapters, etc.).
- **network_status** — connectivity, interfaces, Wi-Fi.
- **disk_cleanup** — reclaim disk space.
- **process_management** — list/inspect/kill processes.
- **rclone_status** — cloud sync (rclone) status.

### System information & monitoring
- **system_info** — OS, hardware, environment summary.
- **system_monitor** — live CPU, RAM, disk, battery, temperatures (via `psutil`).

### Apps & desktop
- **open_application** — launch local apps by name.
- **list_installed_apps** — enumerate installed applications.
- **desktop_control** — mouse/keyboard/screenshot automation (via `pyautogui`, needs X11).
- **browse_web** — silent web fetch by default; opens a **visible** browser only when asked (`visible=True`).

### Everyday utilities
- **get_datetime** — current date/time (also served instantly without the LLM).
- **get_weather** — current weather / forecast.

### Memory, notes & knowledge tools
- **manage_todos** — to-do list (stored as Markdown in `todos.md`).
- **manage_notes** — freeform notes (`notes.md`).
- **remember_fact** — store a durable fact in long-term memory.
- **remember_preference** — store a user preference.
- **log_completed_task** — record a finished task (task awareness).
- **recall_memory** — retrieve from long-term memory.
- **search_knowledge** — semantic search across the knowledge base.
- **learn_knowledge** — add content to the knowledge base.
- **kb_status** — knowledge base size/health.
- **index_knowledge_folder** — ingest a folder into the KB.
- **search_web_for_kb** — search the web and queue results for KB ingestion.

### Internet (all headless/silent by default) — see [section 8](#8-internet-capabilities)
- **web_search**, **read_article**, **get_news**, **search_reddit**, **social_search**, **browser_automate**.

### Self-improvement (human-gated) — see [section 11](#11-self-improvement)
- **read_own_code**, **analyze_self**, **propose_code_change**, **propose_new_tool**.

---

## 8. Internet Capabilities

All internet tools are **headless and silent** — they run in the background, no window pops up, and your focus is never stolen. Primus picks the best tool for the request:

| Need | Tool | How it works |
|------|------|--------------|
| Facts, prices, "what is X", "latest on Y", research | **web_search** | DuckDuckGo via `ddgs`, with a multi-engine HTML fallback (DuckDuckGo HTML → Bing → Brave). Source-cited summary. Offers to save to the KB. |
| Read/summarize a specific URL | **read_article** | Multi-layer extraction: Trafilatura → Newspaper3k → BeautifulSoup → raw HTTP → headless Selenium (for JS-heavy pages). |
| News / headlines / "what's happening with X" | **get_news** | RSS aggregation via `feedparser` (incl. Google News topics). Fast, linked. |
| "What does Reddit think about X" | **search_reddit** | Reddit's public JSON API. |
| Community / tech opinions | **social_search** | Router across Reddit, Hacker News (Algolia), and X/Twitter (web search). |
| JS-heavy pages, logins, forms, clicking | **browser_automate** | Selenium (headless by default; `visible=True` to watch). Supports type/fill/click/submit/screenshot/wait actions. |
| Open a real browser window | **browse_web** with `visible=True` | Triggered when you say "open", "show me", "pop it up", "in the browser". |

**Chaining:** Primus will, for example, `web_search` to find a link → `read_article` to read it → summarize with the source. It always cites sources and never fabricates current data — if a tool is needed, it uses one instead of guessing.

---

## 9. Voice Input (Speech-to-Text)

Primus has **local, offline-first voice input** powered by `faster-whisper`. Nothing is sent to the cloud.

**How to use it:** Tap the 🎤 mic in the Chat tab, speak, then tap again to stop. Your speech is transcribed locally. With **Auto-send** on, it's submitted immediately; with it off, the text drops into the input box for you to edit first.

**Settings (Setup tab or config):**
- `stt_model_size` — `tiny` | `base` (default) | `small` | `medium`. Bigger = more accurate but slower.
- `stt_device` — `auto` (detects GPU, falls back to CPU) | `cpu` | `cuda`.
- `stt_compute_type` — `auto` | `int8` | `float16` | `float32`.
- `stt_vad` — voice-activity detection (trims silence/noise), default on.
- `stt_language` — empty = auto-detect, or e.g. `en`.
- `stt_autosubmit` — auto-send transcription (default on).

The whisper model is **prewarmed in a background thread at startup** so the first voice input is instant. If `faster-whisper` isn't installed or no mic is found, voice degrades gracefully with a clear message and the rest of the app is unaffected.

---

## 10. Memory & Learning Systems

Primus is designed to **compound with use**. Several systems work together, all updating in background threads so they never slow your reply.

### Knowledge Base (RAG)
- A **Chroma** vector database at `~/.primus/knowledge/chroma`, embedded with `nomic-embed-text` (local HuggingFace fallback available).
- Ingest documents via the **Knowledge tab** (drag-and-drop), `learn_knowledge`, `index_knowledge_folder`, or `--index PATH`.
- Supports PDF (`pypdf` + `pdfplumber`), Word `.docx` (`python-docx`), Excel `.xlsx` (`openpyxl`), PowerPoint `.pptx` (`python-pptx`), ODT, Markdown, code, and text.
- Semantic search (`search_knowledge`, `/kb`, `/recall`) injects relevant chunks into the agent's context.

### Long-term memory (LTM)
- Durable **facts** (`remember_fact`) and **preferences** (`remember_preference`) stored in `memories.json`.
- Recalled automatically into context and on demand via `recall_memory` / `/recall`.

### Conversation memory & continuity
- **Session memory** (`session_memory.json`) — current-topic facts weighted heavily in each reply.
- **Chat summary** (`chat_summary.json`) — long chats are summarized once they pass a threshold, keeping recent turns verbatim.
- **Conversation archive** (`conversation_archive.json`) — past sessions are archived and periodically consolidated, referenced naturally as `[CONV-N]` / `[MEM-N]`.

### Continuous learning
- **Learning digest** — every few user turns (`learning_digest_turns`, default 6), Primus distils a structured digest of key facts, preferences, working style, wins/failures, and open threads. Trigger on demand with `/learn now` or `/digest`.
- **Core-knowledge promotion** — high-value project/business knowledge is promoted to durable memory.

### Self-reflection
- After each substantive turn, Primus quietly self-reflects (was it on-task? helpful? natural? what to improve?) and logs it to `reflections.json`.
- The most actionable improvement notes are injected back into future prompts under an "Apply these (learned from self-reflection)" block — closing the loop so behavior actually changes.
- See the report with `/reflect`.

### Meta-learning (usage metrics)
- A `MetricsTracker` (`metrics.json`) records **tool success rates & latency**, **response latency by path**, **routing tallies per intent**, and your **👍/👎 feedback** — persisted and debounced.
- Feedback applies a **bounded nudge** to future routing automatically.
- View with `/metrics` or `analyze_self("metrics")`; weak spots with `analyze_self("bottlenecks")`.

---

## 11. Self-Improvement

Primus has **controlled, human-gated read access to its own codebase** (`admin_assistant.py`). This lets it explain its own behavior and *propose* improvements — but it can **never** change itself without your explicit approval.

**Inspection (read-only, runs freely):**
- `read_own_code(query=…)` — read its own source by keyword or line range.
- `analyze_self(aspect)` — `overview` | `tools` | `performance` | `metrics` | `bottlenecks` | `suggestions`. Summarizes architecture, tools, performance, weak spots, and concrete improvement ideas. View a snapshot any time with `/self`.

**Proposing changes (staged only — nothing is written):**
- `propose_code_change(description, find, replace)` — stages an exact find/replace edit and shows you a diff.
- `propose_new_tool(tool_code, registry_anchor)` — stages a brand-new tool plus its registration.

**Applying changes (you, and only you):**
- `/approve edit` — applies the staged edit. `/reject edit` — discards it.

**Guardrails (cannot be bypassed by the model):**
- Edits touching **critical safety markers** (the approval logic, the self-edit machinery itself) are **rejected before staging**.
- A **timestamped backup** is written to `~/.primus/self_edits/` before any approved edit.
- The edited file must **compile** (syntax-checked); if it doesn't, the change is reverted automatically.
- Primus will **never claim** it changed its own code unless `/approve edit` actually ran. No autonomous self-modification, ever.

> If you ask Primus to "fix yourself", the correct flow is: analyze → propose → show diff → wait for your `/approve edit`.

---

## 12. Slash Command Reference

Type these in the chat input.

| Command | Action |
|---------|--------|
| `/help` | Show shortcuts, command safety, and slash reference |
| `/forge <task>` | Force the task to the Forge coding model |
| `/primus` | Force the next turn through Primus |
| `/model [name]` | Show or change the active model |
| `/model fast` · `/model large` · `/model auto` | Pin the small model · pin the full model · dynamic switching (default) |
| `/gpu status` | AMD GPU / acceleration report (backend, placement, utilization) |
| `/status` | Current system/agent status (incl. active model policy) |
| `/setup` | Setup guide and dependency checks |
| **Memory & learning** | |
| `/memory` | Show stored memory |
| `/recall <query>` / `/search <query>` | Search memory + knowledge base |
| `/kb status` | Knowledge base status |
| `/learn <text>` | Teach a fact (`/learn important <key fact>` to prioritize) |
| `/learn <~/file.md>` | Ingest a file into the KB |
| `/learn now` · `/digest` · `/summarize` | Distil this session into a learning digest now |
| `/forget <query>` | Remove matching memory |
| `/index [path]` | Index a folder into the KB |
| `/reflect` | Show the self-reflection report |
| **Meta-learning & feedback** | |
| `/metrics` · `/stats` · `/performance` | Usage statistics report |
| `/good [note]` · `/👍` | Positive feedback (tunes routing) |
| `/bad [note]` · `/👎` | Negative feedback (tunes routing + logs a session note) |
| **Command queue** | |
| `/queue` | Show queued commands |
| `/approve run` | Run the highlighted risky command |
| `/approve all` | Run the queued safe batch |
| `/reject` | Discard the queue |
| **Self-improvement** | |
| `/self` | Self-overview + any staged edit |
| `/approve edit` · `/reject edit` | Apply / discard a staged self-edit |
| `/approve learn` · `/reject learn` | Apply / discard staged knowledge |
| **Session** | |
| `/clear` | Clear the chat |
| `/export` | Export the conversation to Markdown (`~/.primus/exports/`) |
| `/tasks` | Show task history |
| `/index_projects` | Index known project folders |

---

## 13. How to Use Primus Properly So He Improves

Primus gets smarter the more deliberately you use these habits:

1. **Give feedback.** After a good answer, type `/good` (or `/good loved the concise summary`). After a bad one, `/bad it was off-topic`. This is the single biggest lever: feedback tunes routing and is logged as a concrete signal for next time.

2. **Teach him facts and preferences explicitly.** Use `/learn important I run a small consultancy; prioritize automation tasks` or just tell him in chat ("remember that I prefer concise replies"). Durable facts and preferences resurface automatically in future sessions.

3. **Feed the knowledge base.** Drag documents into the **Knowledge tab** (SOPs, project docs, client info), or `/learn ~/path/to/file.pdf`. Anything in the KB becomes searchable context for every future answer.

4. **Run `/learn now` at the end of a meaningful session.** This distils the session into a structured digest (facts, preferences, wins/failures, open threads) and consolidates memory — so the next session starts warm.

5. **Let it self-reflect (it's automatic).** Substantive turns are reflected on in the background; the actionable notes are reapplied to future prompts. Periodically check `/reflect` to see what it's learning about serving you.

6. **Use `/metrics` to spot weak points.** If a tool is slow or failing, `analyze_self("bottlenecks")` and `analyze_self("suggestions")` will surface it — and you can then ask Primus to `propose_code_change` to fix it, reviewing the diff before `/approve edit`.

7. **Be explicit about visibility.** Say "open YouTube" to get a real browser window; otherwise research stays silent in the background. This keeps Primus unobtrusive while still powerful.

8. **Confirm risky actions deliberately.** Stay in **Execute** mode for fluid daily use; the queue + approval still protects destructive commands. Switch to **Suggest** when you want to review everything.

9. **Keep models warm.** Leave `ollama serve` running and `ollama_keep_alive` at `30m` so the first reply each session is fast.

The compounding loop in one line: **use it → give feedback → teach facts → `/learn now` → it reflects → next time it's better.**

---

## 14. Configuration Reference

Config lives at `~/.config/primus/config.json` and is created from defaults on first run. Edit it directly or via the Setup tab. Selected keys (defaults shown):

### Models & routing
| Key | Default | Meaning |
|-----|---------|---------|
| `model` | `qwen2.5:7b` | Primus orchestrator model |
| `forge_model` | `qwen2.5-coder:7b` | Forge coding model |
| `primus_fast_model` | `llama3.2:3b` | Fast chat tier model |
| `auto_delegate_forge` | `true` | Let the router hand off to Forge |
| `router_forge_threshold` / `_strong_threshold` | `3.0` / `5.0` | Forge delegation score thresholds |
| `forge_fallback_to_primus` | `true` | Fall back to Primus if Forge is down |
| `forge_invoke_timeout_sec` | `180` | Forge hard timeout — on expiry, falls back to Primus, then returns a partial answer |
| `primus_invoke_timeout_sec` | `240` | Hard timeout for the main Primus graph path (0 = uncapped). On expiry Primus returns a graceful **partial answer** instead of hanging |
| `fast_coding_enabled` | `true` | Small, self-contained "write a little script/function" asks answer in ONE focused coder-model call (skips plan→step→summarize) |
| `fast_coding_max_chars` | `320` | Max prompt length for the fast coding path |
| `forge_fast_invoke_timeout_sec` | `100` | Tighter cap for the single-shot fast coding call (degrades to the full Forge graph on timeout) |
| `forge_max_plan_steps` | `4` | Keep Forge plans focused (2–4 steps) |
| `coding_verify_on_generate` | `true` | Forge always ends with a verification step and is told to run `verify_python` on generated code |
| `forge_lean_context` | `true` | Skip the heavy memory/RAG dump for self-contained coding (cleaner context → faster, sharper code) |
| `forge_autocorrect` | `true` | **Closed-loop self-correction.** After Forge answers, every Python block is automatically compiled + linted; if there's a real error, Forge gets the diagnostics and does ONE bounded repair pass, then re-verifies. Static-only — never executes code. (Install `ruff` for undefined-name/bug detection on top of syntax checks.) |
| `forge_autocorrect_timeout_sec` | `90` | Bound for the single repair round-trip so it can't hang a turn |

### Resilience / responsiveness
| Key | Default | Meaning |
|-----|---------|---------|
| `fast_meta_path` | `true` | Preference/style directives ("be more concise", "always provide complete info", "remember I prefer…") are saved + confirmed **instantly**, skipping the full agent loop |
| `turn_watchdog_enabled` | `true` | Background guardian that force-recovers any turn that overruns the hard limit so the Halt button + UI stay responsive |
| `turn_hard_limit_sec` | `420` | Force-recover a turn that exceeds this (~7 min); incidents are logged to `~/.primus/watchdog_incidents.json` |
| `ollama_keep_alive` | `30m` | Keep model resident in RAM/VRAM |
| `sequential_models` | `true` | Keep only ONE heavy model (Primus *or* Forge) loaded at a time — unloads the idle one to cut RAM/CPU/heat on the Ryzen laptop |
| `sequential_warm_reload` | `true` | After a Forge turn, warm Primus back up in the background |
| `ollama_num_ctx` | `4096` | Context window (smaller = faster) |
| `primus_num_predict` / `forge_num_predict` | `768` / `2048` | Output token caps |
| `warm_up_models` | `true` | Preload Primus model at startup |
| `fast_chat_tier` | `true` | Light model (`llama3.2:3b`) answers pure chat in one call — kept loaded as the always-on lightweight fallback even under the sequential policy |
| `fast_tool_path` | `true` | Clear single-tool requests run directly in code |
| `response_cache_ttl_sec` / `_max` | `300` / `64` | Cache for trivial replies |

### Clean chat / UI
| Key | Default | Meaning |
|-----|---------|---------|
| `show_thoughts_in_chat` | `false` | Keep reasoning in the side panel, not the chat |
| `show_model_badge_in_chat` | `false` | Model shown in status bar, not per message |
| `theme` | `cyberpunk` | UI theme |
| `compact_mode` | `false` | Compact layout |
| `execution_mode` | `execute` | `execute` or `suggest` |
| `auto_execute_safe_commands` | `true` | Auto-run read-only commands |
| `gmail_fixture_mode` | `true` | No Gmail token → list/read/status tools serve the demo mailbox from `examples/gmail/fixture_inbox.json` (clearly labelled, drafts/sends preview-only). Switches itself off the moment a real `gmail_token.json` exists |

### Knowledge / memory / RAG performance
| Key | Default | Meaning |
|-----|---------|---------|
| `embedding_model` | `nomic-embed-text` | KB embeddings. `auto` = prefer light `bge-small` if `sentence-transformers` is installed; `bge-small` / `BAAI/bge-small-en-v1.5` = CPU-light local model (needs reindex); else an Ollama model name |
| `embedding_fallback` | `BAAI/bge-small-en-v1.5` | Local HF fallback embeddings |
| `embedding_cache_size` | `256` | In-memory LRU cache of query embeddings (perf; floored at 16) |
| `rag_top_k` / `kb_search_k` / `memory_recall_k` | `3` / `4` / `4` | Retrieval breadth (tuned low for the Ryzen laptop) |
| `rag_chunk_size` / `rag_chunk_overlap` | `600` / `80` | Smaller chunks = sharper, cheaper retrieval |
| `hybrid_retrieval` | `true` | BM25-style keyword rerank fused with vector results |
| `rag_scope_by_intent` | `true` | Search `core_business`/`projects`/`technical` first for business/eng queries |
| `memory_consolidate_hours` | `6` | Memory consolidation cadence |
| `chat_summarize_threshold` / `chat_recent_keep` | `24` / `12` | Long-chat summarization |

> **Embedding model note:** switching `embedding_model` changes the vector dimension, so an existing Chroma store must be **reindexed** (the old vectors won't match). To use the lighter `bge-small` (great for the Ryzen laptop — CPU-only, lower heat):
> ```bash
> uv pip install sentence-transformers langchain-huggingface
> # set embedding_model to "bge-small" in Setup/config
> rm -rf ~/.primus/knowledge/chroma          # clear the old (nomic) vectors
> uv run python admin_assistant.py --index ~/Documents   # re-ingest your docs (seeds reseed on launch)
> ```
> If those packages aren't installed, Primus automatically falls back to Ollama `nomic-embed-text` so RAG keeps working — no crash.

### Continuous learning & meta-learning
| Key | Default | Meaning |
|-----|---------|---------|
| `self_reflection` | `true` | Reflect after each substantive turn |
| `learning_digest_turns` | `6` | Auto learning-digest cadence |
| `meta_learning` | `true` | Track tool/route/latency/feedback stats |
| `adaptive_routing` | `true` | Let feedback nudge routing (bounded) |
| `feedback_routing_max_nudge` | `1.5` | Max routing nudge from feedback |
| `proactive_followups` | `true` | Offer a smart next step when useful |

### Voice (STT)
`stt_enabled` (true), `stt_model_size` (`base`), `stt_device` (`auto`), `stt_compute_type` (`auto`), `stt_vad` (true), `stt_language` (`""` = auto), `stt_autosubmit` (true), `stt_beam_size` (5).

### AMD GPU acceleration & dynamic switching
| Key | Default | Meaning |
|-----|---------|---------|
| `prefer_gpu` | `true` | GPU-first: offload to the GPU when usable. CPU fallback is automatic |
| `gpu_layers` | `-1` | Layers to offload: `-1` = all that fit (best for iGPU, lets Ollama auto-detect) · `0` = CPU-only · `N` = explicit count (→ `num_gpu`) |
| `gpu_backend` | `auto` | `auto` \| `rocm` \| `vulkan` \| `cpu` (canonical; legacy `ollama_gpu_backend` is kept in sync) |
| `ollama_flash_attention` | `true` | Sets `OLLAMA_FLASH_ATTENTION=1` (faster, less VRAM) |
| `hsa_override_gfx_version` | `auto` | `""` = off · `auto` = detect Ryzen AI iGPU (→ `11.0.2`) · or explicit e.g. `11.0.2` |
| `gpu_apply_env` | `true` | Apply `OLLAMA_*`/`HSA_*` env at startup |
| `fast_fallback_model` | `llama3.2:3b` | Small/quantized model for snappy simple turns |
| `dynamic_switching_enabled` | `true` | Auto-switch to the fast model when the big model is slow |
| `slow_threshold_sec` | `5.0` | A simple turn slower than this trips fast mode |
| `fast_model_switch_cooldown` | `300` | Seconds to stay in fast mode after tripping |

### Environment variable overrides
`PRIMUS_MODEL`, `PRIMUS_FORGE_MODEL`, `PRIMUS_FAST_MODEL`, `OLLAMA_HOST`, `PRIMUS_HOST`, `PRIMUS_PORT`.

---

## 15. Data & File Locations

| Path | Contents |
|------|----------|
| `~/.config/primus/config.json` | Configuration |
| `~/.config/primus/primus.log` | Application log |
| `~/.primus/` | All app data (below) |
| `~/.primus/knowledge/chroma/` | Vector database (RAG) |
| `~/.primus/knowledge/uploads/` | Drag-and-drop originals (for re-ingest) |
| `~/.primus/knowledge/manifest.json` | KB manifest |
| `~/.primus/memories.json` | Long-term facts & preferences |
| `~/.primus/session_memory.json` | Current-topic memory |
| `~/.primus/chat_history.json` · `chat_summary.json` | Chat + summaries |
| `~/.primus/conversation_archive.json` | Archived past sessions |
| `~/.primus/reflections.json` | Self-reflection log |
| `~/.primus/metrics.json` | Meta-learning statistics |
| `~/.primus/todos.md` · `notes.md` | To-dos and notes |
| `~/.primus/task_history.json` | Completed-task log |
| `~/.primus/exports/` | Exported conversations |
| `~/.primus/self_edits/` | Timestamped backups before approved self-edits |

**To start fresh** (wipe memory/learning), stop Primus and delete the relevant files in `~/.primus/` (e.g. `memories.json`, `session_memory.json`, `conversation_archive.json`, `reflections.json`, `metrics.json`, and the `knowledge/chroma/` folder).

---

## 16. Abilities & Limitations

### What Primus can do well
- Natural conversation and fast answers for everyday requests.
- Run and manage your Linux system (files, processes, services, devices, disk, network) with safety gating.
- Organize files, clean junk, sort folders.
- Real-time internet research, news, community opinions, and page summarization — silently.
- Coding and debugging via Forge.
- Remember facts, preferences, and whole conversations; recall them naturally later.
- Ingest your documents and answer from them (RAG).
- Voice input, desktop automation, and browser automation.
- Inspect and *propose* improvements to its own code (with your approval).

### Limitations (be aware)
- **Linux-focused.** Assumes `apt`, `wmctrl`, `xdg-open`, X11 for desktop control. Other OSes are unsupported.
- **Local model quality.** Reasoning is bounded by the 7B-class local models; not GPT-4-class. Heavier models (`qwen2.5:14b`) are smarter but slower.
- **Requires Ollama running.** No Ollama → no chat. The startup check warns you.
- **Optional features degrade silently.** Missing `psutil`/`selenium`/`faster-whisper`/etc. disables only that feature.
- **Internet tools depend on third-party sites.** Searches/extractions can fail if a site is down, blocked, or login-walled; Primus says so and offers a visible-browser fallback.
- **Not autonomous over itself.** It cannot modify its own code, run destructive commands, or send anything without explicit approval — by design.
- **Voice/desktop need hardware/session.** A mic for STT; an X11 session for `pyautogui`.

---

## 17. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Chat says agent unavailable | Ensure `ollama serve` is running and `ollama pull qwen2.5:7b` is done; check `/status`. |
| Browser shows "Aw, Snap!" / SIGILL | Browser GPU crash (Linux + AMD common). Server is fine — open the URL in Firefox, or Chromium with `--disable-gpu --disable-software-rasterizer`. |
| "localhost is not accessible…" launch error | A proxy is routing localhost. Run with `NO_PROXY=localhost,127.0.0.1` or `--local`; Primus already patches this automatically. |
| Port already in use | Primus auto-bumps the port; or pass `--port 7870`. |
| Voice mic missing | Install `faster-whisper`; check `stt_enabled` and that a mic is connected. |
| Web search returns nothing | Install `ddgs` (DuckDuckGo); Bing/Brave HTML fallback covers gaps. |
| KB/recall empty | Pull `nomic-embed-text` and install `chromadb` + `langchain-chroma`; ingest docs via the Knowledge tab. |
| Verbose diagnostics | `uv run python admin_assistant.py --debug` (logs + in-UI errors). |

Useful diagnostics: `uv run python admin_assistant.py --check-deps`, the startup **Tool & capability check** block, and `~/.config/primus/primus.log`.

---

## 18. Privacy

Primus is **local-first and private by design**:

- The LLMs run on your machine via Ollama — **prompts never leave your computer**.
- The web UI binds to `127.0.0.1` only (no public exposure unless you pass `--share`).
- Memory, knowledge, and metrics are stored as plain files under `~/.primus/` on your disk.
- Voice transcription is fully local (`faster-whisper`) — **no cloud STT**.
- The **only** outbound traffic is the web searches, article fetches, and news/social lookups *you* ask for. Gradio/HuggingFace telemetry is disabled at startup.

---

---

## 19. AMD GPU Acceleration & Dynamic Model Switching

Primus is tuned to run **blazing fast on AMD Ryzen AI hardware** (e.g. Ryzen AI 9 HX 375 with a Radeon 890M iGPU) while keeping every model and capability intact.

### 19.1 GPU acceleration (AMD-specific)

At startup Primus runs `setup_gpu_acceleration()`, which:

- **Detects your AMD GPU** via `lspci` and recognizes Ryzen AI iGPUs (Radeon 880M/890M, Phoenix/Hawk Point/Strix family).
- **Sets Ollama acceleration env vars** according to `ollama_gpu_backend`:
  - `OLLAMA_VULKAN=1` (preferred first for Ryzen AI; used on `auto`/`vulkan`)
  - `OLLAMA_FLASH_ATTENTION=1` (faster attention, less VRAM)
  - `HSA_OVERRIDE_GFX_VERSION` — set automatically to `11.0.2` when a Ryzen AI iGPU is detected and `hsa_override_gfx_version` is `auto`, or to whatever explicit version you configure. This is the override most commonly needed to get ROCm working on these APUs.
- **Logs** exactly what it applied, and surfaces a ready-to-paste export block.

> **Important:** Ollama runs as a **separate server process**. Env vars Primus sets apply to its embedded client and to any `ollama serve` it launches as a child — but if your Ollama server is **already running**, restart it with the recommended export block (shown in `/gpu status` and the Setup tab) for the settings to take effect:
>
> ```bash
> OLLAMA_FLASH_ATTENTION=1 OLLAMA_VULKAN=1 HSA_OVERRIDE_GFX_VERSION=11.0.2 ollama serve
> ```

**GPU-first with automatic CPU fallback:** `prefer_gpu` (default **on**) tells Primus to offload to the GPU whenever it's usable. `gpu_layers = -1` lets Ollama auto-offload all layers that fit (ideal for a shared-memory iGPU); set a positive number to pin a layer count, or `0` (or `gpu_backend = cpu`, or `prefer_gpu = off`) to force CPU. If the GPU can't be used at runtime, **Ollama falls back to CPU automatically** and Primus surfaces it as `Mode: 🟡 running on CPU — GPU unavailable, fell back automatically`.

**Monitoring & verification:**
- `/gpu status` (or the Setup tab → "GPU / acceleration") shows a **Mode** line (`🟢 GPU active` / `🟡 CPU` / `⚪ unknown`), the detected GPU, backend, prefer-GPU + offload settings, flash/HSA/Vulkan flags, where loaded models are running (parsed from `ollama ps` — look for `100% GPU`), and GPU utilization (via `rocm-smi` if installed).
- The **Setup / Status** tab now embeds the full GPU panel, and the **status bar** shows `GPU <util>%` / `GPU✓` when acceleration is active.
- After warm-up the **log** prints `Acceleration mode: GPU — …` (or a `GPU preferred but model is on CPU` warning), so the fallback is visible without opening the UI.
- `uv run python admin_assistant.py --check-deps` prints the full GPU report.

**One-click ROCm help:** Setup tab → "GPU / acceleration" → **Install ROCm for AMD GPU** prints tailored install + verification commands (auto-includes the Ryzen AI `HSA_OVERRIDE_GFX_VERSION` when an 880M/890M-class iGPU is detected).

**Recommended AMD tooling** (optional, for full ROCm acceleration + monitoring):
```bash
sudo apt install -y rocm-smi radeontop vulkan-tools
# For full ROCm compute support, see AMD's amdgpu-install / ROCm docs for your distro.
```

**Config:** `prefer_gpu`, `gpu_layers`, `gpu_backend` (`auto`/`rocm`/`vulkan`/`cpu`), `ollama_flash_attention`, `hsa_override_gfx_version` (`auto`/blank/explicit), `gpu_apply_env`. All editable in the Setup tab ("Apply GPU") or `config.json`.

### 19.2 Smaller / quantized fallback model

Primus ships with a small fast model (`fast_fallback_model`, default `llama3.2:3b`) used both for the fast-chat tier and for dynamic switching. Pull it once:

```bash
ollama pull llama3.2:3b      # or: ollama pull qwen2.5:3b
```

Pick it in **Setup → Fast model** (the "Apply models" button rebuilds the fast graph live), or set `fast_fallback_model` in config. If you don't set a separate fast-chat model, Primus derives it from `fast_fallback_model` automatically.

### 19.3 Intelligent dynamic model switching

The `DynamicModelManager` keeps simple turns snappy without ever degrading coding or deep-reasoning work:

- It **times every turn**. If a *simple* turn (chat / admin / memory intent, low coding signal) on the full model exceeds `slow_threshold_sec` (default 5s), it flips into **fast mode** for `fast_model_switch_cooldown` seconds (default 300).
- While in fast mode, simple Primus turns run through a **fast-model Primus graph** — same system prompt and **all the same tools**, just the small model — so you keep full capability at higher speed.
- **Coding / Forge / deep-reasoning turns always use the full model.** Switching never touches them.
- Decisions are **logged in the thinking panel and metrics** (`/metrics` shows the current model policy).

**Manual control (overrides auto):**
- `/model fast` — pin the small fast model for simple turns.
- `/model large` (or `/model full`) — pin the full model; no auto-switching.
- `/model auto` — back to dynamic switching (default).

Current policy is shown in `/status`, `/metrics`, and the status bar. Everything is bounded, reversible, and degrades gracefully — if the fast model isn't installed, Primus simply stays on the full model.

---

## 20. Project structure (modularization in progress)

Primus is being refactored from the single `admin_assistant.py` file into a clean `primus/` package
**incrementally**, one verified slice at a time. At every step the extracted code is imported back into
`admin_assistant.py` so the module-level namespace and runtime behavior stay **100% identical** — no
features, prompts, personality, or guardrails are changed, weakened, or locked.

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
    ├── core/                 # ✅ migrated (Phase 1)
    │   ├── __init__.py        #    re-exports all core symbols
    │   ├── obedience.py       #    GLOBAL_OBEDIENCE (verbatim)
    │   ├── personality.py     #    CORE_OPERATOR_PROFILE (verbatim)
    │   ├── prompts.py         #    FORGE_SYSTEM / PRIMUS_SYSTEM / FAST_CHAT_SYSTEM / planner+step+summary
    │   └── guardrails.py      #    DANGEROUS_PATTERNS, SAFE_COMMAND_LEADERS, SAFE_LEADING_RE, UNSAFE_IN_SAFE
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
> you can edit any of it freely.

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

## License

Copyright (c) 2026 Daniel Lee Barren (Danny Barren). Personal, educational, and other non-commercial use is welcome. Commercial use needs his written permission — commercial rights stay with him. If you publish, demo, write about, or reuse Primus, credit is required. Terms: [LICENSE](LICENSE) (PolyForm Noncommercial 1.0.0). Attribution: [NOTICE](NOTICE).
