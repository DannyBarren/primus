# STATE CARD

Primus as it exists in this tree at **v0.9.3 / `136a30f`**. Every claim below was re-read from
source in this pass, and every `file:line` was re-derived against that commit. The tree wins:
where an older note or a docstring disagreed with the code, the code
is recorded and the disagreement is called out. Nothing was implemented, refactored, or
"quickly fixed" while writing this.

## 1. STACK

**Framework (versions from `.venv/lib/python3.12/site-packages`; `requirements.txt` is unpinned):**

- Gradio **5.20.0** (+ `gradio_client` 1.7.2) — UI
- LangChain **1.3.9**, `langchain-core` **1.4.7**, `langchain-ollama` **1.1.0**,
  `langchain-text-splitters` **1.1.2**, `langchain-chroma` **1.1.0**
- LangGraph **1.2.5**, `langgraph-prebuilt` **1.1.0**, `langgraph-checkpoint` **4.1.1**,
  `langgraph-checkpoint-sqlite` **3.1.1**
- Pydantic **2.13.4**, ChromaDB **1.5.9**, cryptography **49.0.0**
- Retrieval / ingest: `pypdf` 6.13.2, `python-docx` 1.2.0, `docx2txt` 0.9, `trafilatura` 2.1.0,
  `beautifulsoup4` 4.15.0, `feedparser` 6.0.12, `duckduckgo-search` 8.1.1
- Optional: `google-api-python-client` 2.198.0, `faster-whisper` 1.2.1, `psutil` 7.2.2,
  `pyautogui` 0.9.54
- Listed in `requirements.txt` but **no dist-info in this venv**: `langchain-community`,
  `openpyxl`, `python-pptx`, `slack-sdk`. Code paths that need them degrade instead of crashing.
- Python **3.12.13** (host venv; `python:3.12-slim` in the Dockerfile)

**Entrypoint:** `admin_assistant.py` — `main()` `admin_assistant.py:3495`, `if __name__`
`:3787`. `HAS_AI_STACK` is set at import `:171` / `:195`; a missing stack degrades to a message
instead of a traceback.

**Boot commands:**

- Host (daily driver): `make run` → `uv run python admin_assistant.py` (`Makefile` `run:` target)
- Docker local: `make up`; image CMD is
  `python admin_assistant.py --host 0.0.0.0 --port 7860 --no-tray` (`Dockerfile`, final `CMD`)
- Cloud: `make cloud` → `docker-compose.cloud.yml`
- Eval: `make eval` → `uv run python scripts/eval_primus.py`
- Backup: `make export` (`INCLUDE_SECRETS=1` opt-in) / `make import FILE=…`
- CLI flags live at `admin_assistant.py:3442-3494` (`--host`, `--port`, `--share` off by default,
  `--no-tray`, `--export`, `--import`, `--index`, …). Default bind host is `127.0.0.1`
  (`primus/config.py:158`, re-forced to localhost at `:348`).

**Graph owner:** `build_agent_graph()` `primus/agents/system.py:2313`. Wrappers:
`build_primus_graph` `:2607`, `build_forge_graph` `:2611`, `init_agent_graphs` `:2615`
(returns `{"primus", "primus_fast", "forge"}`).

**Graph shape:** outer LangGraph `StateGraph(PrimusGraphState)` `:2586` with four nodes —
`plan` `:2587`, `execute_step` `:2588`, `summarize` `:2589`, `single_step` `:2590`. Conditional
edges at `:2592` and `:2597`; `summarize → END` `:2602`; `single_step → END` `:2603`;
`graph.compile()` `:2604`. Each step then invokes a LangChain **`create_agent`** react loop.

**Model factory:** `make_chat_ollama()` `:1579` — `ChatOllama`, `num_ctx` from
`CFG["ollama_num_ctx"]`, with `redact_model_input` wrapped around the model's input `:1639-1645`.

**Checkpointer — inner vs outer (unchanged, worth knowing):**

- `_get_checkpointer()` `:165` tries `SqliteSaver` at `{APP_DIR}/checkpoints.sqlite`; any failure
  or `CFG["use_sqlite_checkpointer"]=False` falls back to `MemorySaver()`.
- It is passed **only to the inner `create_agent`** (`_packed_step_agent` `:1469-1474`, and the
  background agent `:4699-4704`).
- `graph.compile()` `:2604` takes **no checkpointer**. The outer plan graph is not checkpointed.
  Chat correctness does not depend on the sqlite file.

## 2. FILE MAP

| path | job |
| --- | --- |
| `admin_assistant.py` | Host + entrypoint. CFG/model globals, `PrimusSession` `:1554`, `ExecutionMode` `:1545`, `_guard_path` `:2595`, `run_shell` `:2749` + approval queue `:1717`, chat/projects/history, tray, `main()` `:3495`. Library modules must never `import admin_assistant`. |
| `primus/config.py` | Paths + `DEFAULT_CONFIG` + load/save. Env: `PRIMUS_DATA_DIR` → `APP_DIR` `:47`, `PRIMUS_CONFIG_DIR` → `CONFIG_DIR` `:48`, `PRIMUS_DOWNLOADS_DIR` `:94`. |
| `primus/agents/system.py` | Routing, fast paths, graphs, packs bind, timeouts, watchdog, polish, background/scheduled agents. |
| `primus/tools/registry.py` | Every `@tool`, `TerminalTool` `:64` / `GitTool` `:85`, Gmail/Calendar/Slack, `ask_grok` `:8914`, `build_tools()` `:8927`. |
| `primus/ui/builder.py` | The entire Gradio chrome: chat-first stage, `primus-drawer`, Inbox, Brain budget, Suggest/Execute, composer. `build_ui` `:1338`. |
| `primus/core/prompts.py` | `PRIMUS_SYSTEM`, `FORGE_SYSTEM`, `FAST_CHAT_SYSTEM`, planner/step/summary prompts. |
| `primus/core/obedience.py` | `GLOBAL_OBEDIENCE` block. |
| `primus/core/personality.py` | `CORE_OPERATOR_PROFILE` / voice; override via `personality_override.md`. |
| `primus/core/identity.py` | Operator identity from `operator.json` / env. Stdlib-only; does not import `primus.config`. |
| `primus/core/guardrails.py` | `DANGEROUS_PATTERNS` `:7`, `SAFE_COMMAND_LEADERS` `:31`, `SAFE_LEADING_RE` `:48`, `UNSAFE_IN_SAFE` `:52`. |
| `primus/core/path_mode.py` | `user_path` vs `primus_path`. `parse_user_path` `:141`, `inspect_path_message` `:205`, `set_path_mode` `:243`, `current_path_mode` `:257`, `run_user_path` `:535`. Persists on `CFG["path_mode"]`. |
| `primus/core/feedback.py` | Append-only `feedback.jsonl` via `capture_after_turn` `:125`; local `classify` `:87` / `detect` `:103`. `/good` `/bad` stay on `MetricsTracker`. |
| `primus/core/filemaid_safety.py` | `normalize_path` `:44`, `is_path_blocked` `:49`, `is_write_blocked` `:76` (defined, no callers). Linux roots + junk segments. Labels/gates only — no quarantine executor. |
| `primus/core/extractors.py` | `detect_file_type`, `ExtractionOptions`, `ContentExtractor`, `extract_text`. Calls `is_path_blocked` `:94`. Libraries, not `@tool`s. |
| `primus/core/analyzer.py` | Heuristic `ScannedFile` / `ActionProposal` / `analyze_path`. Labels only. |
| `primus/core/vault.py` | Encrypted local secrets. `vault_locked` `:54`, `store_secret` `:247`, `resolve` `:281`, `known_secret_pairs` `:233`. Blob name is `blob` `:34` under `VAULT_DIR`. |
| `primus/core/redact.py` | `redact_for_outbound` `:84` / `redact_model_input` `:128` — known vault values + regex (`sk-`, `xai-`, `AKIA`, bearer, PEM, `password=`, OAuth JSON fields); home → `~/`. |
| `primus/core/receptionist.py` | Leftover-only 3b JSON slots on `primus_fast_model`. `_classify` `:208`, `receive_leftover` `:260`, bounded timeout `_timeout_sec` `:198`. Internal; never printed in chat. |
| `primus/core/brain_governor.py` | Weekly $ + per-turn caps, Monday roll, loop detector, override grant, `ask_frontier`. `BUDGET_FILE = BRAIN_BUDGET_FILE` `:26-28`; `GROK_MODEL = "grok-4"` `:30`; `HTTP_TIMEOUT_SEC = 45` `:31`. |
| `primus/core/tool_packs.py` | Named packs `:19-47` built from real `build_tools()` names. `bind_tools` `:227` filters the inner `create_agent` list. Does **not** shrink `build_tools()`. Fast paths never consult it. |
| `primus/core/audit.py` | Shared `{APP_DIR}/audit.jsonl` writer. `write_audit` `:164`, `read_recent` `:191`, `is_why_ask` `:300`, `explain_recent` `:312`, `attach_audit` `:354`. One log, not two. |
| `primus/core/backup.py` | `make export` / `--export` zip under `{APP_DIR}/exports/`; `export_bundle` `:89`, `preview_import` `:161`, `import_bundle` `:185`. No `@tool`. |
| `primus/memory/system.py` | KB/Chroma RAG (`KnowledgeBase` `:463`), LTM, session memory, metrics, reflections, `build_conversation_focus` `:1447`. |
| `primus/utils/gpu.py` · `patches.py` · `text.py` | AMD/ROCm/Vulkan env + Ollama GPU recover; Gradio launch helpers; `safe_filename`. |
| `primus/__init__.py` | Package map. Its docstring still says "68+ @tools" — stale, left alone on purpose. |
| `scripts/eval_primus.py` | Sandboxed smoke runner; sets `PRIMUS_DATA_DIR` to a temp dir **before** importing the host `:32-40`. |
| `evals/cases.json` | Six cases: `pulse_hello`, `pulse_identity`, `kb_price`, `gmail_fixture`, `polish_json`, `safety_rm`. |
| `examples/gmail/fixture_inbox.json` | Demo mailbox used when no token + `gmail_fixture_mode`. |
| `Makefile` / `Dockerfile` / `docker-compose*.yml` | Host + local/cloud Docker boot. |
| `SECURITY.md` | Bind model, export allowlist, OAuth paths, secrets hygiene. |

## 3. ROUTING TABLE

| ask | function | graph? |
| --- | --- | --- |
| hi / who are you | `route_prompt` pulse branch (`is_pulse_turn` `:862`); "who are you" is `_PULSE_IDENTITY_RE` `:823` and is excluded from the knowledge branch. One `_fast_chat_answer`, `no_plan=True`. | **no** |
| date + time | `instant_answer` `:239` (`_INSTANT_BOTH_RE` `:233`). Local clock, no model, no network. | **no** |
| compound time + news | `fast_tool_answer` compound-fact branch `:521-559`. In-process `get_datetime`, then `get_news`, then `get_weather` only if asked. News failure still returns the time. | **no** |
| biggest headline (short **and** >90 chars) | `fast_tool_answer` news branch `:664-687`, placed **before** the 90-char bail-out `:691`. A long headline ask still reaches `get_news`. Never consults `route_prompt` or the receptionist. | **no** |
| last email (no sender) | `fast_tool_answer` mail branch `:561-572` → `_fast_mail_answer` `:430` (`gmail_list_messages` → `gmail_read_message`). Location tokens after from/in (`inbox`, `gmail`, `mail`, `email`, `e-mail`, `mailbox`, `account`, `folder`, `home`, `device`, `phone`, `laptop`) are not senders — "from my inbox" / "from gmail" is the same path as latest email. | **no** |
| last email from NAME + summarize | Same fast path. `_fast_mail_answer` adds `from:NAME` only when NAME is a person; summarize triggers `_summarize_mail` `:408` at `:478`. | **no** |
| count folders in ~ | `fast_tool_answer` count branch `:574-593` → `list_directory`, counts `[dir]` lines. `_ft_count_target` `:482` maps "home" → `~`. | **no** |
| tell me about X / what is X | `route_prompt` knowledge branch `:2988-3001` (`_KNOWLEDGE_Q_RE`) → `_fast_chat_answer(prefer_main=True)` with a `max(45, fast_chat_invoke_timeout_sec)` leash `:4176-4183`. Vetoed by `_KNOWLEDGE_TOOL_VETO_RE` (news/mail/folders/weather/time). Receptionist never called. | **no** |
| file write / move / read | `create\|write\|put PATH with TEXT` is its own in-process `write_file` step (`path_mode.py` `_WRITE_WITH_RE` / `_run_write` `:496`); two destinations are never merged (`parse_user_path` `:141` + `_split_write_destinations` `:120`). `create PATH` with a file suffix and no `with` text is also `write_file` (`is_create_file_step` `:175`) so `create A then create B` runs both. `create DIR` with no `with` text and no file suffix is `create_directory` (`is_create_dir_step` `:167`). Bare write (no `my path:`) still sets `run_as_user` via `has_file_write_steps` (`system.py` `:3932`) and calls `write_file` **before** leftover chat. Reply is `✓ Wrote` / `[Suggest]` / `✗` from the real tool — never `All taken care of` when a listed write did not run (`:4304`). Bare `list` / `ls` is `has_list_steps` `:3933` plus a single-list fast-tool after the multi-step gate (`:609`). `list_directory` empty return is `0 files, 0 folders in path` (`registry.py` `:700`); summarize uses the same inventory (`_listing_inventory` `:369`) and does not say `No matches`. | **no** for write/create-dir/list; **yes** (typical) for other file verbs |
| `my path:` | `inspect_path_message` `:3867` sets `run_as_user` `:3928-3934`; `_run_user_path_with_packs` `:3961` runs steps in the operator's order. In-process steps batch into `run_user_path`; others bind their own pack per step. After a `[dir]`/`[file]` listing (or an empty dir), `summarize` is a short inventory (file/dir counts, top extensions, 5 names + N more) — not a reprint. Listings shown in chat are capped at ~20 lines. | mixed (only for steps that need an agent) |
| leftover / mixed | Default `route_prompt` return `:3035-3038` → `path=primus`, `needs_rag=True`, `no_plan=False`. Then `receive_leftover` `:4045`, then the plan→execute graph. Listed `write\|create\|put PATH with TEXT` never reaches this path. | **yes** |
| why did you do that / explain your reasoning | `is_why_ask` `:300` (`audit.py` `_WHY_RE` `:33`) also matches explain your reasoning/process, walk me through, how did you do that, what did you just do. Route unchanged: `explain_recent()` `:312` from `system.py` `:3874-3885`. One English bullet per action (`Pulled today's headlines`, `Wrote path`) via `_TOOL_ENGLISH` `:87` — no Started/Finished, no coffee persona, no re-run. Operator-claim openers ("Danny Barren here") are stripped. Not a `@tool`; does not steal `/good` `/bad`. | **no** |
| forge / code | `ModelRouter.analyze` → `agent_id="forge"`; small self-contained → `fast_coding` (`_is_fast_coding` `:2949`), else the Forge graph `:3006-3016`. | **no** if `fast_coding`, else **yes** |
| `ask_grok` | Never a route. It is a registered tool that only enters a turn's pack when the receptionist says `need_brain` / `heavy` `:4047-4053`, or when the message itself names it. Fast winners return long before that point. | **yes** (leftover graph only) |

### Invoke order (`_invoke_primus_impl` `primus/agents/system.py:3826`)

`invoke_primus` `:3798` is a thin wrapper: it calls the impl, then `capture_after_turn` inside a
`try/except` `:3817-3821`, so feedback logging can never kill a turn.

1. `clear_turn_pack()` `:3842`, brain-governor `begin_turn` `:3848`, audit `begin_turn` `:3854` — each in its own `except`.
2. `inspect_path_message` `:3867` → apply the `path_mode` toggle.
3. why-ask: `is_why_ask` `:3876` → `explain_recent()` returns immediately (audit English only).
4. `skip_fast` `:3888` — the only fast-tier suppressor, and only for `my path:` prefix / toggle-with-steps / multi-step `user_path`.
5. `instant_answer` `:3897`.
6. `fast_tool_answer` `:3909` (wrapped; an exception falls through to full routing).
7. path-mode toggles answer inline `:3921-3940`; a listed path **or** any `has_file_write_steps` / `has_list_steps` message (bare `write ~/… with …` or `list ~/…` included) runs `_run_user_path_with_packs` **before** leftover, and a trailing `extra_ask` re-enters instant → fast-tool → knowledge → `invoke_primus`.
8. `HAS_AI_STACK` `:3986`, then `check_ollama_health` `:3990`.
9. `ModelRouter.analyze` `:4001`; Forge availability fallback `:4012-4025`.
10. `route_prompt` `:4033`.
11. Receptionist, fenced to `route["path"] == "primus" and agent_id == "primus"` `:4043` → `receive_leftover` `:4045` → `ask_grok` appended on `need_brain`/`heavy` `:4047-4053` → `_apply_turn_pack(message, slots.tools, …)` `:4055`. Every other branch binds a pack from the text `:4065` / `:4068` / `:4071`.
12. RAG skipped when `route["needs_rag"]` is false `:4081-4087`; response cache `:4129-4133`.
13. `fast_chat` / `kb_answer` single bounded call `:4141-4197`; otherwise the graph.

### `fast_tool_answer` branch order (`:508`) — first match wins

1. `fast_tool_path` flag + slash-command guard `:514-518`
2. **compound fact** — date/time + news and/or weather, no file/mail/code verb `:521-559`
3. **latest mail** — last/latest/unread/check + email/inbox/gmail, not send/draft/… `:562-572`. Location tokens are not `from:` senders.
4. **count folders/files** — how many / count / number of `:575-603`
5. **multi-step bail** (`_MULTI_STEP_RE`) → `None` `:606-607`
6. **weather** (`_FT_WEATHER_INTENT_RE`, uncapped) `:631-643`
7. **open / launch** → `open_application` / `browse_web` `:645-662`
8. **news / headlines** (uncapped, singular "headline" counts) `:664-687`
9. **length cap** — `len(msg) > 90` and no URL → `None` `:691-692`
10. **battery / wifi / bluetooth / disk** `:694-709`
11. **read a URL** → `read_article` `:710-715`
12. **system monitor** `:716-725`
13. **screenshot / type / press** → `desktop_control` `:726-741`
14. **reddit** `:742-749`
15. **finance / markets** → `_run_web_search`, not `get_news` `:750-759`
16. **explicit web search** `:760-768`, else `None`

### `route_prompt` branch order (`:2972`) — first match wins

1. **knowledge / explain** `_KNOWLEDGE_Q_RE` `:2988`, vetoed by pulse-identity, forge score, multi-step, `_FAST_CHAT_TOOL_HINT_RE`, `_PULSE_VETO_RE`, `_FAST_CHAT_KB_HINT_RE`, `_KNOWLEDGE_TOOL_VETO_RE` → `fast_chat` + `knowledge=True`
2. **pulse** `is_pulse_turn` `:3003` → `fast_chat`, `no_plan`
3. **forge** `:3006` → `fast_coding` or the `forge` graph
4. **direct action** `is_direct_action` `:3018` → `direct` (graph single-step, RAG off)
5. **kb_answer** `_kb_answer_eligible` `:3022` → retrieve + one bounded call
6. **fast_chat eligible** `:3025`
7. **cacheable chatter** `:3028`
8. **else** → `primus` graph `:3035`

## 4. TOOLS

**Exact `build_tools()` count: 120** — the returned literal list runs
`primus/tools/registry.py:8932` (`TerminalTool()`) through `:9051` (`ask_grok`), one entry per
line, 120 entries. `ask_grok` is **last** `:9051`. Every entry is wrapped by
`attach_audit(attach_resolve(t))` `:8931`.

Two class tools (`terminal`, `git`) + 118 `@tool` callables. `apply_pending_self_edit` is an
`@tool` at `:3960` but is **not** in `build_tools()`; it is slash-gated behind `/approve edit`
(`primus/ui/builder.py:2469-2471`).

**Mail tools in `build_tools()`:** `gmail_auth`, `gmail_auth_code`, `gmail_get_auth_url`,
`ensure_gmail_access`, `gmail_status`, `gmail_list_messages`, `gmail_read_message`,
`gmail_create_draft`, `gmail_send_email` `:9004-9012`, plus
`gmail_extract_meeting_request` / `suggest_meeting_times_from_email` /
`create_meeting_from_email` / `propose_times_in_reply` `:9020-9023`.

**File tools + return type — short human strings, never JSON envelopes:**

- `list_directory(path="~", pattern="*")` `:693` → `[dir]`/`[file] name (N B)` lines; a guard
  refusal returns the raw `_guard_path` string (no `✗` prefix)
- `read_file` `:135` → a text window; errors `✗ …`
- `write_file` `:168` → `✓ Wrote/Appended \`path\`` or a Suggest preview; errors `✗`
- `move_path` `:643` / `copy_path` `:667` → `✓ Moved/Copied …` or a Suggest-queued shell
- `create_directory` `:624` → `[Suggest] Would create \`path\`. Switch to Execute to apply.`; Execute → `✓ Created …`
- `search_files` `:708` → `run_shell(find …, force=True)`
- `delete_file` `:207` → `✗ Refusing blocked path: …` when FileMaid blocks the target `:219`;
  otherwise the `run_shell("rm …")` queue text, not a `✓`

**News / datetime / weather / frontier:** `get_datetime(query="")` `:1325`,
`get_weather(location="")` `:1334` (wttr.in), `get_news(topic="", limit=6)` `:2308` (RSS; topic →
Google News RSS), `ask_grok(question, briefing="")` `:8914` → `brain_governor.ask_frontier`.

**Shell / git:** `TerminalTool.name = "terminal"` `:64` → `run_shell`. `GitTool.name = "git"`
`:85` → `git -C <repo> …` via `run_shell`.

**How `delete_file` works:** it never unlinks in-process. `confirm=` is accepted and immediately
discarded (`del confirm` `:215` — "not a safety gate — the approval queue is"), `_guard_path`
resolves the target `:216`, `_filemaid_write_block` refuses a blocked destination `:219-221`
**before** anything is queued, then it builds `rm` / `rm -rf` and hands it to `run_shell` `:222-223`.
`run_shell` matches `rm` in `DANGEROUS_PATTERNS` (`guardrails.py:10`) and queues it for approval in
**both** modes unless `force` / `allow_dangerous_once` (`admin_assistant.py:2764-2765`).

**Suggest vs Execute.** `ExecutionMode` is `suggest` | `execute` (`admin_assistant.py:1545`);
default config `execution_mode: "execute"` (`config.py:160`). `run_shell` `:2749` grades every
command: **safe** auto-runs when `auto_execute_safe_commands` `:2761`, `:2767-2773`; **review** is queued in
Suggest and runs in Execute `:2775-2779`; **dangerous** is always queued `:2764-2765`. File writes
and `gmail_send_email` preview in Suggest (`registry.py:5950`) and apply in Execute. This is the
only operator safety gate — there is no second `confirm=` system.

**Packs and how `create_agent` is bound.** `PACKS` (`tool_packs.py:19-47`): **mail** (6), **files**
(8), **shell** (`terminal`, `git`), **knowledge** (4), **brain** (`ask_grok`), **default**
(`list_directory`, `get_datetime`, `get_news`, `ask_grok`). `select_pack` `:147` picks one from the
message (or from the receptionist's `tools[]`); `extras_from_named` `:188` allows a named
in-registry extra and refuses an invented name; `bind_tools` `:227` filters the registry list and
has a safety net that falls back to `default` if a bug ever tried to pass everything `:245-248`.
`_packed_step_agent` `:1438` calls `create_agent(llm, bound, …)` `:1469` with ~2–8 schemas and
caches the agent per (llm, pack, extras, agent_id). **`build_tools()` is still 120** — packs only
narrow what the react loop sees.

Background / scheduled jobs use the same pack helpers: `_background_bind` `:4658` runs
`select_pack` + `extras_from_named` + `bind_tools` on the job text (unclear → default).
`ask_grok` is kept only if named or the pack is `brain`. `_packed_background_agent` `:4686`
caches like `_packed_step_agent`. Never binds 120. No receptionist on background.

## 5. SAFETY

**`_guard_path` (`admin_assistant.py:2595-2623`)** is the write gate.

- Resolves both a symlink-resolved and a lexically-normalized path `:2608-2609` (so `..` is
  collapsed and blocked while legitimate symlinks still work)
- Writes are confined to `HOME`; outside → `Refusing path outside home: …` `:2618`
- `allow_system_read=True` additionally permits `READONLY_SYSTEM_DIRS` `:2553` and refuses with
  `Refusing path outside home and allowed read zones: …` `:2614-2617`
- `must_exist` → `Not found: …` `:2621-2622`

**FileMaid `is_path_blocked` vs `_guard_path` on write.** They are **not** the same gate.
`_guard_path` remains the HOME write gate. After it resolves, `write_file` `:180` /
`delete_file` `:219` / `create_directory` `:629` / `move_path` `:651` / `copy_path` `:675`
all call `_filemaid_write_block` `:157` → `is_path_blocked` `:49` on the destination. Blocked →
`✗ Refusing blocked path: {reason}`. `is_path_blocked` blocks `/` itself, `BLOCKED_PATHS`
(`/etc`, `/usr`, `/bin`, `/sbin`, `/boot`, `/sys`, `/proc`, `/dev`, `/root`, `/var/lib`,
`/lib`, `/lib64`) `:20-33` and junk segments (`.git`, `node_modules`, `__pycache__`,
`.venv`, `venv`) `:35-41`. `is_write_blocked` `:76` is defined but has **no callers** —
`extractors.py:94` calls `is_path_blocked` directly, so there is no separate ingest-only
write gate. `list_directory` / `read_file` are not FileMaid-blocked (a successful list/read of a
`.git` path still returns). `/etc/shadow` still dies at `_guard_path` (outside HOME)
before FileMaid runs. FileMaid is a library of labels and gates, not an executor, and
there is no quarantine mover.

**Leftover `confirm=`:** exactly one — `delete_file(path, confirm: bool = False)` `:207`, deleted
on the first line of the body. No other tool in `primus/` or `admin_assistant.py` takes a `confirm`
flag. `DEFAULT_PREFERENCES` (`admin_assistant.py:445`) has a `confirm_dangerous` *memory
preference*, which is not a tool argument.

**Human confirm on send / destructive:** `gmail_send_email` previews in Suggest
(`registry.py:5950`) and the fixture never sends `:5934`. Destructive shell and `delete_file` land
in the approval bar above the composer (`builder.py:2021-2023`) via `PrimusSession.queue_command`
(`admin_assistant.py:1717`). `sudo` is always dangerous (`guardrails.py:8`); `rclone sync` is not a
safe leader (only `listremotes|about|lsd|ls|size|ncdu|version` `:38`) so it is review-tier.

**`polish_response` (`primus/agents/system.py:2038`) strips or replaces:**

- Identity inversion / "you, Primus" → `IDENTITY_ACK` (`_IDENTITY_INVERSION_RE` `:1841`)
- ReAct lines `:1744-1748`, `**Step N:**` `:1751`, plan/progress/"say continue"/timeout chrome `:1756-1769`
- **Status theater** `_STATUS_THEATER_RE` `:1773-1790`: an optional `---` hrule plus the bold
  status line in **both** bold placements — `**Status: done**` *and* `**Status:** done` — the bare
  `Status: done|blocked|needs_input` line, and `files_touched` / `last_task_status.json` mentions,
  plus ledger/resume bait ("still owed", "Got N of M done", "tell me to keep going"). A
  `- **Status:** done` row inside a report the operator asked for survives, because every
  alternative anchors at line start.
- Canned shrug lines and invented mail names `gmail_get_unread_email` / `gmail_get_latest` /
  `get_unread_email` (`_CANNED_SHRUG_RE` `:1795`)
- Leaked tool-call JSON `_TOOL_CALL_JSON_RE` `:1867`; robotic openers/closers `:1920-1964`;
  empty-after-strip → `BAD_FORMAT_REPLY` `:1864`
- **Kept on purpose:** `_BRAIN_KEEP_RE` `:1803-1810` preserves "Brain weekly budget reached",
  "Brain per-turn call limit reached", "I kept circling on this", and
  "Primus used N brain calls and is stuck" (re-applied at `:2080`).

`_finalize_turn_status` `:2262` writes `~/.primus/last_task_status.json` and the thinking-panel
status; the chat transcript never receives a Status footer.

## 6. GMAIL

**OAuth paths:** `_GMAIL_CREDS_PATH = {APP_DIR}/gmail_credentials.json` `registry.py:4914`,
`_GMAIL_TOKEN_PATH = {APP_DIR}/gmail_token.json` `:4915`, attachments
`{APP_DIR}/gmail_attachments/` `:4916` — i.e. `~/.primus/…` locally. Scopes `_GMAIL_SCOPES` `:4908`
(`gmail.readonly`, `gmail.compose`, `gmail.send` per `SECURITY.md`). Calendar reuses the same
credentials file with its own `calendar_token.json`. These files are **not** ingested into the
vault or the KB, and they are **not** in the default export zip.

**Tool names:** `gmail_auth` `:5558`, `gmail_status` `:5718`, `gmail_list_messages` `:5748`,
`gmail_read_message` `:5808`, `gmail_send_email` `:5919` (+ draft/auth-url/meeting helpers). The
real names are these — not `gmail_get_latest` / `get_unread_email`.

**Fixture mailbox:** `examples/gmail/fixture_inbox.json` `:5057`, served when
`_gmail_fixture_active()` `:5061` — `gmail_fixture_mode` (default `True`, `config.py:194`) **and**
no real token. Every fixture path switches itself off the moment `gmail_token.json` exists `:5055`.
Output is labelled as demo data `:5729`. Senders include Amira Haddad, Northwind Billing,
Priya Raman (eval marker `GMAIL-FIXTURE-ALPHA-4421`), Changelog Weekly, Sam Okafor.

**Is `from:NAME` honoured in source today? Yes.** `_fast_mail_answer`
(`agents/system.py:430`) builds a `from:{sender}` query; the fixture filter is
`_gmail_fixture_query` `registry.py:5083`; the live API passes `q` through. A from-ask that
matches nothing returns `No email from {sender} in the recent inbox.` rather than substituting
the newest unrelated message. Location tokens (`inbox`, `gmail`, `mail`, `email`, `e-mail`,
`mailbox`, `account`, `folder`, `home`, `device`, `phone`, `laptop`) are dropped, so
"most recent email from my inbox" does not become `from:inbox`.

**Is HTML stripped in source today? Yes.** `_summarize_mail` `:408` → `_clean_mail_lines` `:381`
removes style/script/tags, unescapes entities and drops tracking URLs. Live `gmail_read_message`
prefers `text/plain` (`registry.py:4963`, `:4982`) and falls back to `_strip_html_text` `:1902`
via `:4992`. Fixture bodies are already plaintext.

## 7. MODELS + TIMEOUTS

**Names (`primus/config.py` `DEFAULT_CONFIG`):**

- Main / Primus: `model` = `qwen2.5:7b` (`PRIMUS_MODEL`) `:98`
- Forge: `forge_model` = `qwen2.5-coder:7b` (`PRIMUS_FORGE_MODEL`) `:99`
- Fast: `primus_fast_model` = `llama3.2:3b` (`PRIMUS_FAST_MODEL`) `:147`; `fast_fallback_model`
  `:278` (the two are cross-filled at `:364-367`)
- Embeddings: Ollama `nomic-embed-text`

**Timeouts:**

- `fast_chat_invoke_timeout_sec` = **15** `:150` — the default `_fast_chat_answer` bound `:3179`
- Knowledge / KB chat: `max(45, fast_chat_invoke_timeout_sec)` → **~45s**
  (`agents/system.py:4181`, and the same leash for `my path:` knowledge steps `:3951`)
- Receptionist: `max(8, min(15, fast_chat_invoke_timeout_sec))` → **8–15s**
  (`receptionist.py:198-205`), enforced by a joined worker thread `:248`
- `ask_grok`: `HTTP_TIMEOUT_SEC = 45` (`brain_governor.py:31`)
- Graph watchdog wrapper: `primus_invoke_timeout_sec` = **240** (`config.py:122`, used
  `agents/system.py:4271`, `:4352`, `:4384`)
- Forge: `forge_invoke_timeout_sec` = **180** `:106`, `forge_fast_invoke_timeout_sec` = **100**
  `:112`, `forge_autocorrect_timeout_sec` = **90** `:119`
- Turn watchdog hard limit: `turn_hard_limit_sec` = **420** (`config.py:130`; loop
  `agents/system.py:3610-3650`, halts the turn and logs an incident)
- `slow_threshold_sec` = 5.0 `:280` (dynamic model switch)

**As they exist — observe only:** `ollama_num_ctx` = **4096** (`config.py:138`, applied in
`make_chat_ollama`), `agent_recursion_limit` = **30** (`config.py:204`),
`use_sqlite_checkpointer` = **True** (`config.py:211`).

## 8. UI

**Stack:** Gradio 5 `gr.Blocks` inside `build_ui(graphs, model_name)`
(`primus/ui/builder.py:1338`), opened at `:1422`. **`build_ui` returns `(demo, theme, CYBER_CSS)`**
`:3854` — a 3-tuple for `launch()`, not a mounted app.

Layout is chat-first: a top bar (Menu + wordmark/status dot) → a centred `primus-chat`
transcript `:2001` → a dock holding the approval bar and composer. Everything else lives in a
CSS-only slide-over drawer `elem_id="primus-drawer"` `:1433`, toggled by the
`body.primus-drawer-open` class from load-JS `:514-516` — no Gradio remount, so the chat never
loses state.

**Frozen chrome (do not move):**

- Suggest / Execute radio in Menu → **Mode** accordion `:1472-1478`
- Hidden Suggest/Execute hotkey buttons `:2062-2063` (`Ctrl+S` / `Ctrl+E`)
- Approval bar: Execute / **Approve all** `:2021` / **Modify** `:2022` / **Dismiss** `:2023`
- Composer: attach `:2027-2031`, textbox, **Send** `:2044`, **Halt** `:2045`
- Drawer sections in order: Chats `:1439`, Mode `:1472`, **Brain budget** `:1506`, Status `:1555`,
  Knowledge `:1649`, Scheduled `:1710`, Voice & window `:1769`, Connections `:1846`,
  **Inbox** `:1877`, Activity `:1925`
- No Status footer in the transcript (`show_thoughts_in_chat` defaults false; status lives in
  Menu → Status)

**Inbox present?** Yes — a drawer accordion `:1877`, fixture-capable, driven by `inbox_*_ui`
bridges over the same Gmail tools.

**Brain budget accordion present?** Yes — drawer accordion `:1506`, collapsed by default, sitting
next to Mode. Its dials call `save_budget` `:1277` and re-read via `load_budget` `:1278`, both
imported from `brain_governor` `:1224`; a live calculator previews before Save `:1325-1330`. It is
an accordion, not a page or route.

## 9. MEMORY / LOGS / STORES

**`~/.primus/` (`APP_DIR`, `config.py:47`) — referenced from code:** `memory.json` `:54`,
`chat_history.json` `:55`, `projects.json` `:57`, `project_chats/` `:58`, `todos.md` `:60`,
`notes.md` `:61`, `task_history.json` `:62`, `settings.json` `:63` (legacy, migrated),
`exports/` `:64` (chat markdown **and** `primus-backup-*.zip`), `plans/` `:65`,
`knowledge/` `:66` (`chroma/`, `manifest.json`, `uploads/`, `downloads/`, `interactions.json`),
`memories.json` `:72`, `session_memory.json` `:73`, `chat_summary.json` `:74`,
`conversation_archive.json` `:75`, `reflections.json` `:76`, `metrics.json` `:77`,
**`feedback.jsonl`** `:78`, **`audit.jsonl`** `:79`, **`brain_budget.json`** `:80`,
**`vault/`** `:81` (the encrypted `blob`), `vault_index.json` `:82` (ids + labels only),
`vault_access.jsonl` `:83`, `watchdog_incidents.json` `:84`, `audio/` `:85`, `piper/` `:86`,
`self_edits/` `:88`, `self_improvements.json` `:89`, `self_improvement_memory.json` `:90`,
`background_tasks.json` `:91`, `background_results/` `:92`, plus `checkpoints.sqlite`
(`agents/system.py:165`), `gmail_credentials.json` / `gmail_token.json` / `gmail_attachments/`
(`registry.py:4889-4891`), `calendar_token.json`, and `last_task_status.json`
(`agents/system.py:2217`).

**`~/.config/primus/` (`CONFIG_DIR` `:48`):** `config.json` `:49`, `primus.log` `:50`,
`operator.json`, `personality_override.md` (plus optional `seed_knowledge.json` /
`business_context.json` per `SECURITY.md`).

**One log, not two:** `_fs_audit` and `attach_audit` both write `AUDIT_FILE`
(`audit.py:164`). Actions are recorded with an English mapping `_ACTION_ENGLISH` `:70-84`
(`tool_start`, `tool_end`, `queued`, `executed`, `ask_grok_cap`, `ask_grok_loop`,
`vault_resolve`, `export`, `import`, `write`, `mkdir`, `move`, …), which is what "why did you
do that" reads back `:312-352`.

**Still not referenced (do not invent):** a FileMaid state store, an `ask_grok` response cache, a
pack manifest file (packs are code in `tool_packs.py`), and any quarantine directory.

## 10. GREEN vs HOLES

**GREEN — reachable today:**

- Instant date/time; compound time+news `:521-559`; HOME folder count `:575-603`
- Headlines before the 90-char cap `:664-687`, so a long headline ask still hits `get_news`
- Last email, last-email-from-NAME + summarize, and last-email-from-inbox (location token ≠
  sender), with HTML stripped `:430`, `:408`
- Knowledge / "tell me about X" on the main model with a ~45s leash, no planner
- Pulse hello / identity; why-ask (including "explain your reasoning and process") in English
  from `audit.jsonl` — one bullet per action, no Started/Finished, no graph, no re-run
- `my path:` executes the operator's order, switching packs per step `:3748-3795`; summarize
  after a listing is counts + 5 names, not a second dump. Empty list → `0 files, 0 folders`
  (`list_directory` `:700` and `_run_list` `:406`), not `No matches`
- `create\|write\|put PATH with TEXT` → real `write_file` per destination, including a bare
  write with no `my path:` (`has_file_write_steps` `system.py` `:3932`). `create FILE` (suffix,
  no `with`) → `write_file` so two creates both run. `create DIR` (no `with`, no suffix) →
  `create_directory`. Suggest previews; Execute writes. `✗` is shown and success is not claimed
- Suggest previews `create_directory` and queues `rm`; Execute applies
- Fixture Gmail list/read/status with no token
- Leftover receptionist → pack selection → optional `ask_grok` under the governor
- Packs bind the inner `create_agent` to ~2–8 schemas while `build_tools()` stays 120
- Background / scheduled jobs pack from the job text (`_background_bind` `:4658`); never 120
- FileMaid `is_path_blocked` after `_guard_path` on every destructive verb — `write_file` `:180`,
  `delete_file` `:219`, `create_directory` `:629`, `move_path` `:651`, `copy_path` `:675`.
  A blocked `delete_file` refuses before the rm queue. `list_directory` / `read_file` stay unblocked
- Vault + `redact_for_outbound` on every model-bound payload
- `make export` / `make import`, tokens and vault blob excluded by default
- `make eval` 6/6

**HOLES — leftover on purpose. Do not fix them.**

- Post-90-char fast-tool branches (weather, open/launch, reddit, finance, system monitor,
  desktop control) still sit **after** the `:661` early return, so a long ask of that kind misses
  them and falls to routing. News and mail and count are already above the cap `:691`.
- The outer plan graph is **not** checkpointed — only the inner react agent is (`:2604`).
- File-tool return types are not uniformly `✓/✗`: `list_directory` guard errors are bare,
  `read_file` success is a text window, `delete_file` returns queue text.
- `primus/__init__.py` still says "68+ @tools"; the real number is 120.
- Non-knowledge, non-KB chat still uses the 15s cap; an overrun becomes `CHAT_SLOW_REPLY`.
- Invented mail tool names survive only as polish-drop targets, not as a routing rule.
- `apply_pending_self_edit` is deliberately outside `build_tools()`; there is no self-edit loop.

## 11. SUBSYSTEMS

| subsystem | status | path |
| --- | --- | --- |
| `path_mode` | **present** | `primus/core/path_mode.py`; `DEFAULT_CONFIG["path_mode"]` `config.py:163`; wired `agents/system.py:3867`, `:3921` |
| `feedback.jsonl` | **present** | `primus/core/feedback.py` (`capture_after_turn` `:125`); `FEEDBACK_FILE` `config.py:78`; hooked `agents/system.py:3817-3821` |
| `filemaid_*` | **present as libraries** | `primus/core/filemaid_safety.py`, `extractors.py`, `analyzer.py`. Not `@tool`s. No quarantine executor. |
| `vault` | **present** | `primus/core/vault.py`; `VAULT_DIR` / `VAULT_INDEX` / `VAULT_ACCESS_LOG` `config.py:81-83` |
| `redact` | **present** | `primus/core/redact.py`; hooked on the chat model `system.py:1639`, step agent `:1536`, `_fast_chat_answer` `:3160`, receptionist `receptionist.py:226`, `ask_grok` payload `brain_governor.py:329` |
| `receptionist` | **present** | `primus/core/receptionist.py`; leftover-only hook `system.py:4045` |
| `ask_grok` | **present** | `registry.py:8914`; last in `build_tools()` `:9051` |
| `brain_governor` | **present** | `primus/core/brain_governor.py`; `{APP_DIR}/brain_budget.json` |
| `tool_packs` | **present** | `primus/core/tool_packs.py`; bound at `_packed_step_agent` `system.py:1469` |
| backup export | **present** | `primus/core/backup.py`; `make export` / `--export` → `{APP_DIR}/exports/`. No `@tool`. |
| brain budget UI | **present** | drawer accordion `builder.py:1506`; writes the governor's file |
| OCR / spreadsheets / OneDrive / self-edit loop / LiteLLM / quarantine executor | **absent by design** | do not add |

## 12. TAILORING NOTES

- Library modules resolve the host with
  `_host = sys.modules.get("admin_assistant") or sys.modules["__main__"]` (`registry.py:39`,
  `agents/system.py:36`, `memory/system.py:32`, `ui/builder.py:39`, `path_mode.py:15`,
  `feedback.py:14`, `filemaid_safety.py:13`, `receptionist.py:20`, `tool_packs.py:16`,
  `audit.py:19`, `backup.py:19`, `brain_governor.py:24`). **Never**
  `from admin_assistant import …` inside `primus/`.
- Real tool names: shell is **`terminal`**, git is **`git`**, mail is
  **`gmail_list_messages` / `gmail_read_message` / `gmail_auth` / `gmail_status`** — never
  `gmail_get_latest` or `get_unread_email`.
- `build_tools()` is **120**, `ask_grok` last. Packs filter the inner `create_agent` list only;
  they never shrink the registry.
- Pack extra-allow: `extras_from_named` `tool_packs.py:188` permits **one named tool that really
  exists in the registry** outside the chosen pack; an invented name is refused with a one-liner
  (`invented_tool_in_text` `:136`, `_INVENTED_MAIL` `:80`).
- Vault token shape is **`vault:<id>`**. `resolve()` is called only inside a tool's `_run` (via
  `attach_resolve`), returns `None` while locked, and audits by id only — never the value.
- Export is `make export` / `--export`, never a chat tool. `--include-secrets` (or
  `INCLUDE_SECRETS=1`) is the only way tokens and the vault blob enter a zip, and
  `import_bundle` will only restore them if the zip was built that way `backup.py:194`.
- Eval command is `uv run python scripts/eval_primus.py` (or `make eval`); it sandboxes
  `PRIMUS_DATA_DIR` **before** importing the host `scripts/eval_primus.py:32-36`, so the real
  `~/.primus` is never touched. Six cases in `evals/cases.json`.
- `delete_file` = a `run_shell("rm …")` queue. Do not make it delete in-process, and do not treat
  its vestigial `confirm=` as a gate.
- Suggest/Execute (`PrimusSession.mode` / `ExecutionMode`) is the only safety fork. Do not add a
  second one.
- Knowledge answers use `prefer_main=True` and ~45s — not the 15s greeting cap.
- Compound fact, folder count, and mail-from are **in-process Python**, not planner tools. They
  must never reach the receptionist, the graph, or `ask_grok`.
- File observations are short strings; `list_directory`'s `[dir] name (N B)` prefix is what the
  folder-count fast path counts. Do not JSON-ify it.
- Inbox and Brain budget are **drawer accordions**, not pages or routes. Do not remount the chat.
- `path_mode` lives on `CFG["path_mode"]` in the existing `config.json` — not a separate store.
- `audit.jsonl` is shared; `_fs_audit` reuses it. Why-ask is English from the last lines, not a
  `@tool`, and `/good` `/bad` stay on `MetricsTracker`.
- `num_ctx=4096`, `agent_recursion_limit=30`, and the sqlite checkpointer flag are observe-only.

## 13. SEAM AUDIT

The eight designed seams, re-proved in source this pass.

| # | seam | verdict | proof |
| --- | --- | --- | --- |
| 1 | Receptionist `tools[]` → pack | **already wired** | `system.py:4055` passes `slots.tools` into `_apply_turn_pack` `:3650`, which calls `select_pack(text, tools=…)` + `extras_from_named` `:3664-3669`. `receive_leftover` `receptionist.py:260-267` documents that mapping. |
| 2 | `need_brain` → `ask_grok` under the governor, leftover/heavy only | **already wired** | `system.py:4047-4053` appends `ask_grok` only on `need_brain` or `bucket == "heavy"`, then `_apply_turn_pack(..., need_brain=True)` `:4055-4058` adds it as a pack extra `:3667-3668`. `ask_grok` `registry.py:8914` goes through `ask_frontier`. Fast winners return at `:3903` / `:3918`, before `route_prompt` `:4033`. |
| 3 | UI Brain-budget Save === `BRAIN_BUDGET_FILE` === governor read path | **already wired** | `builder.py:1224` imports `load_budget`/`save_budget` from `brain_governor`; save `:1277`, re-read `:1278`, `budget_path()` for the raw blob `:1080-1083`. `brain_governor.py:28` sets `BUDGET_FILE = BRAIN_BUDGET_FILE` (`config.py:80`). One file, both directions. |
| 4 | `redact_for_outbound` on chat / step agent / 3b input / `ask_grok` payload | **already wired** | Chat model wrapper `system.py:1639-1645`; step agent `:1536-1540`; `_fast_chat_answer` `:3160-3175`; receptionist 3b input `receptionist.py:226-228`; `ask_grok` payload `brain_governor.py:329-339` (`_redact_payload`, which also scrubs the live key). |
| 5 | `user_path` still wraps Suggest/Execute and the rm-queue | **already wired** | `path_mode.py:479` routes a delete step to `_host.run_shell("rm …")` (and `_run_delete` `:459` inherits the `delete_file` FileMaid refusal); `_run_send_email` `:483-493` returns a Suggest preview instead of sending; move/write go through the guarded registry tools `:456`, `:508`. |
| 6 | Default export zip excludes tokens and the vault blob | **already wired** | `backup.py:27-31` `SECRET_BASENAMES` + any `*token.json`; `:32` `VAULT_BLOB_NAMES = {"blob"}`, which is the real blob name (`vault.py:34`); filtered at `:100` (APP_DIR) and `:114` (CONFIG_DIR); `exports/` itself is skipped `:106`. |
| 7 | `polish_response` does not eat `Primus used N brain calls…` | **already wired** | `_BRAIN_KEEP_RE` `system.py:1803-1810` re-applied at `:2080`; covers both cap lines, the loop-break line, and the stuck line. |
| 8 | Fast-path turns must not crash in `capture_after_turn` / audit / redact | **already wired** | `capture_after_turn` in `try/except` `system.py:3817-3821`; governor `begin_turn` `:3846-3850` and audit `begin_turn` `:3852-3856` each in their own `except`; `fast_tool_answer` itself is wrapped `:3907-3912`. |

The last product change in this area was v0.9.3 (`136a30f`): `delete_file` now calls
`_filemaid_write_block` `registry.py:219-221` after `_guard_path`, so a blocked path is refused
before the rm queue. Nothing else in the seam set was restyled.

---

TOOL COUNT: **120** (`registry.py:8932-9051`; `ask_grok` last at `:9051`; `apply_pending_self_edit`
defined `:3960` but not registered)
EVAL: **6/6** (`make eval`, run this pass — `pulse_hello`, `pulse_identity`, `kb_price`,
`gmail_fixture`, `polish_json`, `safety_rm`)
FILES READ: `admin_assistant.py`, `Makefile`, `Dockerfile`, `requirements.txt`, `primus/config.py`,
`primus/__init__.py`, `primus/agents/system.py`, `primus/tools/registry.py`, `primus/ui/builder.py`,
`primus/core/{__init__,prompts,obedience,personality,identity,guardrails,path_mode,feedback,
filemaid_safety,extractors,analyzer,vault,redact,receptionist,brain_governor,tool_packs,audit,
backup}.py`, `primus/memory/system.py`, `scripts/eval_primus.py`, `evals/cases.json`, `README.md`,
`SECURITY.md`, the previous state card
FILES WRITTEN: **2** — `artifacts/primus-current-state.md` and `docs/primus-current-state.md`
(every `file:line` re-derived against `136a30f`; FileMaid verb list, `delete_file` refusal order,
`build_tools()` span, and the dead `is_write_blocked` claim corrected)
IMPLEMENTATION EDITS: **0** — docs only this pass. The last product edit was v0.9.3 `136a30f`
(`registry.py` `delete_file` → `_filemaid_write_block`)
