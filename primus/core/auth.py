"""UI bind + shared-secret auth policy.

Loopback binds may start without credentials. A remote bind (0.0.0.0, a LAN/public
address, or Gradio ``--share``) refuses to start unless ``PRIMUS_AUTH_TOKEN`` or
``PRIMUS_PASSWORD`` is set. Gradio then requires that username/password at the UI.
"""
from __future__ import annotations

import os
from typing import Optional

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "https://example.net/id/garnet"})

# Either env var is accepted; TOKEN is the documented name.
_TOKEN_VARS = ("PRIMUS_AUTH_TOKEN", "PRIMUS_PASSWORD")


def is_loopback_host(host: str) -> bool:
    """True for IPv4/IPv6 loopback names. Empty host is treated as loopback (safe default)."""
    h = (host or "").strip().lower()
    if not h:
        return True
    if "%" in h:  # strip IPv6 zone id
        h = h.split("%", 1)[0]
    return h in LOOPBACK_HOSTS


def is_remote_bind(host: str, *, share: bool = False) -> bool:
    """True when the UI would be reachable beyond this machine's loopback interface."""
    if share:
        return True
    return not is_loopback_host(host)


def auth_credentials() -> Optional[tuple[str, str]]:
    """``(username, password)`` from the environment, or ``None`` when unset."""
    token = ""
    for var in _TOKEN_VARS:
        token = os.environ.get(var, "").strip()
        if token:
            break
    if not token:
        return None
    user = (os.environ.get("PRIMUS_AUTH_USER") or "primus").strip() or "primus"
    return user, token


def require_auth_for_bind(
    host: str,
    *,
    share: bool = False,
) -> Optional[tuple[str, str]]:
    """Return Gradio ``auth`` credentials, or raise ``SystemExit`` if a remote bind has none.

    Loopback binds return credentials when they are set (optional local lock) and ``None``
    otherwise. Never logs the secret.
    """
    creds = auth_credentials()
    if is_remote_bind(host, share=share) and creds is None:
        raise SystemExit(
            "Refusing to start: the UI is bound beyond loopback "
            f"({host or 'share link'}) and no shared secret is set.\n"
            "Set PRIMUS_AUTH_TOKEN (or PRIMUS_PASSWORD) in the environment, "
            "or bind to 127.0.0.1 / pass --local.\n"
            "See SECURITY.md and .env.example."
        )
    return creds
