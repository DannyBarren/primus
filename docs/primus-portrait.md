# Primus — a portrait

A plain description of what this system is, what it actually does, and where it stops. Everything
here was checked against the source in this tree. Where a capability is partial or degraded, it
says so.

---

## What Primus is

Primus is a local-first executive assistant that runs on the operator's own Ubuntu machine. It is a
single Python application — a host module (`admin_assistant.py`) plus a `primus/` package — that
serves a Gradio web UI on `127.0.0.1:7860` and drives local language models through Ollama. There
is no account, no tenant, no server-side session: the models, the knowledge base, the memory files,
the secret vault, and the audit log all live under `~/.primus` on the machine you are sitting at.
It runs three local models in two roles — `qwen2.5:7b` as the orchestrator, `qwen2.5-coder:7b` as
the coding specialist, and a small `llama3.2:3b` for cheap conversational turns and internal
classification. A fourth, optional path calls xAI's Grok over HTTP, but only for leftover work that
the local stack could not finish, and only under a weekly dollar cap and a per-turn call cap that
the operator sets in the UI.

The important architectural claim is the inverse of the usual one: this is not a cloud agent with a
local skin. The default answer path never leaves the machine. Asking the time is arithmetic on the
system clock. Counting folders is a directory listing. Reading your last email is two Gmail API
calls and a text cleanup, with no model in the loop at all. The language models are used where
language is actually needed, and the planner graph — the expensive part — is the last resort rather
than the front door. Primus can run real shell commands and edit real files as your user, which is
the point and also the risk, so every destructive action goes through one explicit approval gate
rather than a confidence threshold.

---

## What he can do today

Each of these was exercised against this tree, not inferred from the code.

**Facts, instantly and locally**

- Date and time from the system clock, with no model call.
- Compound asks — "what's the date and time, and what's in the news" — answered in one turn by
  running the clock and then the news reader in-process.
- News headlines over RSS, including a single "what's the biggest headline in tech today", which is
  treated as a news request rather than a knowledge question.
- Weather via wttr.in, system/battery/wifi/disk status, and a `psutil` resource snapshot.

**Email**

- Lists, reads, drafts, and sends Gmail through the official Google API once you authenticate.
- Understands "the last email from Priya" — it narrows the query by sender, and if nobody matches
  it says so instead of quietly handing you the newest unrelated message.
- "Summarize the last email from X" returns sender, subject, date and a short plaintext body, with
  HTML markup, styles, and tracking URLs stripped out.
- With no token, it serves a labelled demo mailbox from `examples/gmail/fixture_inbox.json`, so a
  fresh clone is demonstrable without anyone's real mail.

**Files and the shell**

- Lists, reads, writes, moves, copies, and searches files under your home directory.
- Runs shell commands and git through a three-tier risk grader: read-only commands auto-run, review
  commands queue in Suggest mode, and destructive commands (`rm`, `sudo`, `dd`, force-push, …)
  always queue for one-click approval.
- Counts things properly — "how many folders are in my home directory" comes back as a sentence
  with a number, not a directory dump.

**Knowledge and memory**

- Retrieval-augmented answers over a local Chroma knowledge base with source citations; the smoke
  test proves it quotes a real figure from an indexed proposal and cites the file rather than
  inventing a number.
- General knowledge questions ("what is the Quran", "tell me about gravity") get a single bounded
  call on the main model with a ~45-second budget — a real paragraph, not a timeout apology.
- Layered memory: long-term facts, session memory, conversation archive, project scoping, and a
  metrics/reflection loop.

**Operator control**

- `my path:` mode, where you list the steps and Primus executes them in your order instead of
  planning its own.
- "Why did you do that" answers in English from the action log, without dumping JSON.
- A local encrypted vault for secrets, referenced as `vault:<id>` tokens that resolve only inside a
  tool call.
- Backup and restore: `make export` writes a zip of your data; `make import FILE=…` previews what
  it would overwrite before restoring.
- Coding work routes to the Forge model, with small self-contained tasks answered in one shot.

---

## What he will not do

- **Will not act destructively without you.** Deletion is not an in-process file operation — it is
  compiled into an `rm` command and pushed through the same approval queue as any other dangerous
  shell command, in both Suggest and Execute mode.
- **Will not send email on its own.** Sending previews in Suggest and requires Execute.
- **Will not write outside your home directory.** The path guard collapses `..`, confines writes to
  `$HOME`, and refuses everything else. Read-only tools may additionally see a small allowlist of
  system binary/app directories. `/etc/shadow` is refused on both read and write, and the refusal
  does not take the process down with it.
- **Will not edit its own source autonomously.** It can read and analyse its own code and stage a
  proposed diff, but the change is applied only when you type `/approve edit`. The apply tool is
  deliberately not in the tool registry, and the frontier brain explicitly refuses to patch
  `primus/` source.
- **Will not quietly move your files "for safety".** The FileMaid layer labels and blocks paths; it
  has no quarantine executor.
- **Will not invent a tool.** Names the model makes up are validated against the real registry and
  dropped, and a handful of historical hallucinations (`gmail_get_unread_email`, `gmail_get_latest`)
  are stripped from output on sight.
- **Will not spend without a ceiling.** The Grok path is capped weekly in dollars and per-turn in
  calls, with a loop detector that stops it circling.
- **Has no authentication.** It binds to localhost on purpose. Anyone who can reach the UI can run
  commands as you. It is a shell on your own machine, and it is documented as one.

---

## How a turn runs, in plain English

When you send a message, Primus asks a series of cheap questions before it asks an expensive one.

First it checks whether you are steering: did you switch into `my path:` mode, or are you asking
what it just did? Those are answered immediately, the second one straight from the action log.

Then it tries to answer without a model at all. If you asked for the time or the date, it reads the
clock and replies. If you asked for something a single tool can answer — headlines, your last
email, a folder count, the weather — it runs that tool in-process and replies. These are ordinary
Python, not planner decisions, which is why they come back in under a second and why they can never
accidentally escalate into a multi-step plan.

If you gave it a list of steps, it runs your list in your order. Steps it can handle directly
(list, read, move, count, summarize, delete) run in-process; anything else gets its own small tool
set for that one step.

Only now does it involve a model. A deterministic router classifies the request: a greeting, an
identity question, or a general-knowledge question gets exactly one bounded chat call with no tools
and no plan. A question whose answer lives in your documents retrieves the relevant excerpts first
and then answers once, citing them. Coding goes to Forge.

Whatever is left over — genuinely mixed or multi-part work — is the only thing that reaches the
planner graph. Before it does, a small fast model reads the request and fills in a short internal
form: what kind of work is this, which tools are relevant, does this need the frontier brain? That
form is never shown to you. Its main effect is to choose one small tool pack, so the reasoning
agent sees roughly half a dozen tool schemas instead of all 120. If and only if that form says the
work needs more than the local models can give, the frontier brain becomes available for that
turn — still under the budget, still with the payload redacted.

Finally, whatever came back is cleaned before you see it: internal reasoning traces, plan
scaffolding, status footers, leaked tool-call JSON, and canned apologies are stripped, while
genuine budget and cap messages are deliberately preserved. The turn is then recorded to the
append-only feedback and audit logs, and every one of those steps is individually wrapped so that
logging can never be the thing that breaks your answer.

---

## How to talk to him

Short and literal works better than elaborate.

- **Ask for a fact directly.** "date and time", "biggest headline in tech", "how many folders in
  my home directory", "last email from Priya, summarize it". These take the fast path.
- **Prefix `my path:` when the order matters.** `my path: list ~/Downloads then summarize it` runs
  those two steps in that sequence rather than letting Primus decide.
- **Use Suggest mode when you want to look before you leap.** In Suggest, creating a folder shows
  you a preview and creates nothing; switch to Execute (or approve from the bar) to apply. `Ctrl+S`
  and `Ctrl+E` toggle the two modes.
- **Ask "why did you do that"** and you get an English recap of the last actions, not a log dump.
  `/good` and `/bad` remain separate rating commands.
- **Reference secrets as `vault:<id>`,** never as plaintext.
- Slash commands cover the rest: `/forge`, `/bg`, `/research`, `/queue`, `/approve run`,
  `/approve all`, `/kb status`, `/memory`, `/approve edit`, `/metrics`.

---

## Privacy model

The default is that nothing leaves the machine. Models run locally through Ollama; embeddings and
the vector store are local; chat history, memories, and the knowledge base are files under
`~/.primus`.

Four things do cross the network, and each is something you asked for: explicit web searches and
article fetches, the RSS news reader, the weather lookup, and the Gmail/Calendar APIs once you have
authenticated. A fifth, `ask_grok`, calls xAI — but only on leftover work, only within the budget,
and only when a key is present.

Everything model-bound is passed through a redaction layer first. It replaces known vault values,
matches common secret shapes (`sk-…`, `xai-…`, AWS keys, bearer tokens, PEM private keys,
`password=`, OAuth JSON fields), and rewrites your home directory to `~`. It runs even when the
vault is locked, and it is idempotent — redacting twice does not corrupt the text, and a
`vault:test` reference passes through untouched because the reference is not itself a secret.

Secrets live in a Fernet-encrypted blob under `~/.primus/vault/`, with a separate index that holds
only ids and labels. Plaintext is decrypted inside a tool call and nowhere else; while the vault is
locked, a token resolves to nothing and the miss is recorded by id. Google OAuth files stay at
`~/.primus/gmail_credentials.json` and `gmail_token.json`, outside the vault and outside the repo.
Backup zips exclude every token file and the vault blob unless you explicitly pass
`--include-secrets`. Gradio's download allowlist is narrowed to the exports directory alone.

Every tool call, queue-versus-execute decision, budget stop, vault resolution, and export is
appended to a single shared `audit.jsonl` — one log, not a scattering of them — which is what the
"why did you do that" answer reads back.

---

## Hardware and boot

Primus targets Ubuntu with Ollama running locally. On AMD Ryzen AI hardware it auto-configures GPU
acceleration (Vulkan/ROCm, flash attention, HSA override) at startup; in this tree the small model
reports 100% GPU residency. Python is 3.12.

```
make run       # host, the daily driver: uv run python admin_assistant.py
make up        # local Docker, binds 127.0.0.1:7860, shares ~/.primus with the host
make cloud     # GPU/VPS stack (exposed port — put an authenticated proxy in front)
make eval      # six-case smoke suite in a sandboxed data dir
make export    # backup zip under ~/.primus/exports (INCLUDE_SECRETS=1 to include tokens)
```

The UI binds to `127.0.0.1` by default and the public-share flag is off. State lives in
`~/.primus` (data) and `~/.config/primus` (config, logs, operator profile); in Docker these are
`/data` and `/config` volumes.

---

## What landed across the eleven-step build

This tree has no git history, so the following is reconstructed from the modules themselves and
from the running state card — it is a dependency order, not a commit log.

1. **Modularisation.** The monolith became a host module plus a `primus/` package, with library
   code reaching the host through a live `_host` reference instead of importing it.
2. **Voice and directives.** Prompts, personality, operator identity, and the global obedience
   block moved into `primus/core/`, alongside the command guardrails.
3. **Routing and fast paths.** Instant answers, the deterministic tool fast path, and a cheap
   router — so facts stop being planner jobs.
4. **The polish layer.** One place that strips ReAct traces, plan scaffolding, status theatre, and
   leaked tool JSON before anything reaches the transcript.
5. **The safety gate.** Suggest/Execute, the three-tier shell risk grader, the approval queue, and
   the home-confined path guard as the single write authority.
6. **The receptionist.** A leftover-only classifier on the small model that fills an internal slot
   form — never shown in chat — instead of letting every mixed request hit the planner blind.
7. **Operator steering.** `my path:` ordered execution, plus an append-only correction log that
   records turns without stealing the rating commands.
8. **FileMaid libraries.** Path blocking, content extraction, and heuristic analysis as callable
   helpers — labels and gates, deliberately not tools, and with no quarantine mover.
9. **Vault and redaction.** Encrypted local secrets behind `vault:<id>` tokens, and a redaction
   layer on every model-bound payload.
10. **Budgeted frontier + packs + provenance.** `ask_grok` under a weekly/per-turn governor with a
    Brain-budget drawer, named tool packs so the reasoning agent sees a handful of schemas instead
    of 120, one shared audit log, and backup export/import as a Make target rather than a tool.
11. **Integration.** A seam-by-seam audit of the previous ten layers, one reproducible polish
    defect fixed (a status footer variant that slipped through), and an operator tour — with the
    tool count and the eval suite held constant.

---

## Honest limits

- **The local models are 7B.** `qwen2.5:7b` and `qwen2.5-coder:7b` are competent, not brilliant.
  Long-horizon reasoning, subtle code review, and unusual domain knowledge are where the frontier
  call earns its budget.
- **The context window is 4096 tokens** and the react loop is capped at 30 recursions. Large
  documents are handled by retrieval and chunking, not by stuffing the window.
- **The outer plan graph is not checkpointed.** Only the inner react agent gets a checkpointer, so
  a mid-plan crash does not resume from the middle of the plan.
- **No OCR.** Scanned images are not read. Text extraction covers text-bearing formats only.
- **No self-improvement loop.** Primus can propose a patch to itself; applying it is a manual,
  explicit command, by design.
- **Some optional integrations degrade rather than work.** `langchain-community`, `openpyxl`,
  `python-pptx`, and `slack-sdk` are listed in `requirements.txt` but are not installed in this
  venv, so the tools that depend on them report a missing dependency instead of running.
- **A few long-form phrasings miss the fast path.** Weather, "open X", Reddit, and finance lookups
  sit behind a 90-character cut-off, so an unusually wordy version of those asks falls through to
  the slower route. News, mail, and counting were deliberately moved above that line; the rest were
  left as they are.
- **Background and scheduled agents see the full tool registry**, unlike interactive turns, which
  are narrowed to a pack.
- **It is single-user and unauthenticated.** There is no login, no role model, and no audit of
  *who* asked — only of what was done.
