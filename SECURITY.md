# Security Model

Primus is a **local-first personal agent**. Treat anyone who can pass Gradio
auth as you — they can run tools as the OS user.

## Authentication

- Shared-secret auth from the environment: `PRIMUS_AUTH_TOKEN` or `PRIMUS_PASSWORD`
  (username `PRIMUS_AUTH_USER`, default `primus`).
- Loopback binds (`127.0.0.1`, `localhost`, `::1`) may start without a secret.
- A remote bind (`0.0.0.0`, a LAN/public address, or `--share`) **refuses to
  start** if no secret is set. This is a hard fail, not a warning.
- Docker images bind `0.0.0.0` *inside* the container (so port mapping works).
  Compose therefore requires `PRIMUS_AUTH_TOKEN`. `make up` writes one to `.env`
  when missing. The local compose file still publishes only `127.0.0.1:7860`.

## Bind & exposure

- Default bind is `127.0.0.1`. Keep it there for a laptop install.
- Never launch with `--share` unless you also set a token. `--share` is off by
  default.
- `docker-compose.cloud.yml` publishes `7860` on all interfaces **and** requires
  the token. Still put a reverse proxy or SSH tunnel in front of a public GPU
  host.

## Execution mode

- Default mode is **Suggest**: review-tier commands and mutating file tools
  preview or queue; they do not apply until you switch to Execute or approve.
- Dangerous shell (`rm`, `sudo`, substitutions, `find -delete`, force-push, …)
  always queues, in both modes.
- `open_application` and the `run <words>` fast path do **not** spawn raw user
  text. Shell-shaped input is risk-classified and queued. Only allowlisted apps,
  folder shortcuts, quoted URLs, and existing guarded paths launch.

## Export allowlist

`GRADIO_ALLOWED_PATHS` is set to exactly `~/.primus/exports` (or a repo-local
`examples/out/` for demo artifacts) and nothing else. Do not widen it to
`~/.primus`, the knowledge base, the config dir, or `$HOME`.

## Gmail / Google OAuth

- OAuth client must be a **Desktop app** client (the JSON has an `"installed"`
  top-level key). A "Web application" client (`"web"`) will fail with
  `redirect_uri_mismatch`.
- Live credentials live at `~/.primus/gmail_credentials.json`; the issued token
  at `~/.primus/gmail_token.json`. Both are git-ignored. **Never commit them.**
- First sign-in must run on the **host UI** (it opens a real browser via
  `run_local_server`). Docker/headless: authenticate on the host with `~/.primus`
  bind-mounted so the container reuses the token.
- Scopes requested: `gmail.readonly`, `gmail.compose`, `gmail.send` only.
  Sending is always Suggest-preview / Execute-send — the agent never auto-sends.
  Do not re-add `gmail.modify` or `https://mail.google.com/` (full mailbox).

## Secrets hygiene

- `.gitignore` blocks `*_token.json`, `*_credentials.json`, `client_secret*.json`,
  `.env`, Chroma data, and `~/.primus` dumps. Keep it that way.
- `examples/` contains only synthetic fixtures (fake mailbox, fake proposal).
  Never put real mailbox exports, real client documents, or real tokens in
  `examples/`.
- The operator identity override files (`operator.json`,
  `personality_override.md`, `seed_knowledge.json`, `business_context.json`)
  live in `~/.config/primus/` — outside the repo — by design.

## Agent safety rails

- Risky shell commands queue for one-click approval. Bare `/approve` runs the
  queued command (same as `/approve run`); it does not leave a dangling
  one-shot allow flag.
- Self-edit tools (`propose_code_change` et al.) only stage diffs; nothing is
  applied without `/approve edit`.
- Unit tests in `tests/` cover path guards, the risk classifier, launch
  routing, and the auth bind policy. They do not talk to Gmail or send mail.

## Reporting

This is a personal-project distro. If you find a security issue, open a private
report with the maintainer rather than a public issue with exploit details.
