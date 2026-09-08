# Security Model

Primus is a **local-first personal agent with no built-in authentication**. Treat it like
a shell on your own machine — because it is one.

## Bind & exposure

- The UI binds to `127.0.0.1` by default. Keep it there.
- Never launch with `--share` (Gradio public link). It is off by default; do not enable it.
- Never bind `0.0.0.0` unless Primus sits behind an authenticated reverse proxy (TLS +
  strong auth). `docker-compose.cloud.yml` exposes the port for cloud GPUs — front it
  with a proxy before you trust it.
- Anyone who can reach the UI can run tools as your user, read what the agent can read,
  and download anything in the export directory.

## Export allowlist

`GRADIO_ALLOWED_PATHS` is set to exactly `~/.primus/exports` (or a repo-local
`examples/out/` for demo artifacts) and nothing else. This is intentional and narrow:
it is what makes the Export button work without exposing your whole data dir. Do not
widen it to `~/.primus`, the knowledge base, the config dir, or `$HOME`. If a file
fails to download, move it into the exports dir — don't widen the list.

## Gmail / Google OAuth

- OAuth client must be a **Desktop app** client (the JSON has an `"installed"` top-level
  key). A "Web application" client (`"web"`) will fail with `redirect_uri_mismatch`.
- Live credentials live at `~/.primus/gmail_credentials.json`; the issued token at
  `~/.primus/gmail_token.json`. Both are git-ignored. **Never commit them.**
- First sign-in must run on the **host UI** (it opens a real browser via
  `run_local_server`). Docker/headless: authenticate on the host with `~/.primus`
  bind-mounted so the container reuses the token.
- Scopes requested: `gmail.readonly`, `gmail.compose`, `gmail.send`. Sending is always
  Suggest-preview / Execute-send — the agent never auto-sends.

## Secrets hygiene

- `.gitignore` blocks `*_token.json`, `*_credentials.json`, `client_secret*.json`,
  `.env`, Chroma data, and `~/.primus` dumps. Keep it that way.
- `examples/` contains only synthetic fixtures (fake mailbox, fake proposal). Never put
  real mailbox exports, real client documents, or real tokens in `examples/`.
- The operator identity override files (`operator.json`, `personality_override.md`,
  `seed_knowledge.json`, `business_context.json`) live in `~/.config/primus/` — outside
  the repo — by design. Personal profiles do not belong in version control.

## Agent safety rails

- Risky shell commands (rm, sudo, mv, installs, kills) queue for one-click approval in
  Suggest mode; `gmail_send_email` is preview-then-send.
- Self-edit tools (`propose_code_change` et al.) only stage diffs; nothing is applied
  without `/approve edit`.

## Reporting

This is a personal-project distro. If you find a security issue, open a private report
with the maintainer rather than a public issue with exploit details.
