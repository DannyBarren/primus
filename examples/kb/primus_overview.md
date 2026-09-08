# Primus Overview

Primus is a local-first personal admin agent for Linux. It combines a Gradio chat
UI, local Ollama models, a LangGraph tool runtime, and a Chroma knowledge base.
Everything runs on the operator's own machine: no cloud LLM calls, no telemetry,
no external data leaves the box unless the operator explicitly connects an
integration such as Gmail.

## Components

- **Chat UI** — Gradio app on `127.0.0.1:7860` with a chat-first shell: top bar,
  centered transcript, pinned composer, and a drawer for chats, knowledge,
  scheduled jobs, and connections.
- **Agent runtime** — a router picks the cheapest correct path per turn: instant
  answers, fast chat, direct tool actions, or the full planning graphs.
- **Tools** — terminal, files, git, web search, Gmail, Slack, calendar, and more,
  all gated by Suggest/Execute approval for risky actions.
- **Knowledge base** — Chroma collections (`learned`, `core_business`, `projects`,
  `uploads`) with hybrid retrieval and citations.

## Privacy model

Primus binds to localhost only. Exports are confined to `~/.primus/exports`.
OAuth tokens live under `~/.primus` and are never sent anywhere but the provider
they belong to.
