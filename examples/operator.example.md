# Personalizing Primus (optional)

Primus ships generic: the agent calls the user "the operator" and knows nothing about
you. Everything below lives in your **config dir** (`~/.config/primus/` by default,
`$PRIMUS_CONFIG_DIR` if set) — never in the repo — so your identity is never committed.

## `operator.json` — identity tokens used across prompts, routing, and drafts

```json
{
  "name": "Ada",
  "full_name": "Ada Lovelace",
  "projects": ["Analytical Engine", "Notes"],
  "context_block": "Ada runs a small research consultancy.\nKey projects: Analytical Engine, Notes. Ubuntu workstation; ~/work.",
  "project_example": "the Analytical Engine",
  "project_example2": "Notes",
  "signoff": "Ada",
  "creator_lock": false,
  "tag_keywords_personal": ["ada"],
  "tag_keywords_business": ["lovelace"]
}
```

All fields optional. Derived forms (possessive, vocative, sentence-initial caps) are
computed from `name` unless you set them explicitly. `creator_lock: true` makes Primus
enforce the exact spelling of `full_name` in long-term memory writes.

Env vars `PRIMUS_OPERATOR_NAME` / `PRIMUS_OPERATOR_FULL_NAME` override the file.

## `personality_override.md` — full agent personality

If present, its contents replace the built-in standing-orders/personality block that is
injected into every agent system prompt (Primus, Forge, fast-chat). Write it as plain
markdown; keep the "no leaked machinery" rules — the chat polish layer assumes them.

## `seed_knowledge.json` — first-run knowledge-base seeds

```json
[
  {"source": "primus:seed:operator", "text": "Ada Lovelace is the operator. Primus serves her exclusively."},
  {"source": "primus:seed:toolstack", "text": "Ada's tools: Linux, uv, Ollama, git."}
]
```

Seeds sync into Chroma on first boot (or when `SEED_KNOWLEDGE_VERSION` bumps). Sources
containing a project name route to the `projects` collection; `toolstack`/`hardware`/
`specialt*` route to `technical`; everything else lands in `core_business`.

## `business_context.json` — long-term memory seed facts

A JSON list of fact strings merged into `~/.primus/memory.json` on first run (existing
facts are never overwritten).

## Try the demo corpus

```bash
uv run python admin_assistant.py --index examples/kb
```

Then ask: "What is the Acme Q3 proposal price?" — the answer should cite
`examples/kb/sample_proposal.md` with the `$41,500` figure.
