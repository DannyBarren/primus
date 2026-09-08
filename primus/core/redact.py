"""Redact secrets from strings that leave the device toward a model / API.

Always runs — even when the vault is locked. Replaces known vault values with
``vault:<id>`` when unlocked. Never logs the secret.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from primus.config import HOME
from primus.core.vault import known_secret_pairs, oauth_plaintext_values

# Known patterns — apply even if the vault is locked.
_AKIA_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}")
_XAI_RE = re.compile(r"\bxai-[A-Za-z0-9_-]{10,}")
_BEARER_RE = re.compile(r"\b[Bb]earer\s+[A-Za-z0-9._\-+/=]{8,}")
_PEM_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.S,
)
_PASSWORD_RE = re.compile(r"(?i)\bpassword\s*=\s*\S+")
_GMAIL_JSON_FIELD_RE = re.compile(
    r'(?i)("(?:access_token|refresh_token|token|client_secret)"\s*:\s*")([^"]+)(")'
)
_MAIL_HEADER_RE = re.compile(
    r"(?im)^From:\s.+\n(?:.*\n){0,12}^Subject:\s",
)

_HOME_STRS: tuple[str, ...] = tuple(
    p for p in {
        str(HOME),
        str(Path.home()),
        str(Path.home().resolve()) if Path.home().exists() else "",
    }
    if p and p != "/"
)


def _audit_redact(ident: str, *, ok: bool = True) -> None:
    try:
        from primus.core.vault import _audit  # noqa: PLC0415

        _audit("redact", id=ident, dest="model", ok=ok)
    except Exception:  # noqa: BLE001
        pass


def _redact_home(text: str) -> str:
    out = text
    for home in sorted(_HOME_STRS, key=len, reverse=True):
        if home and home in out:
            out = out.replace(home, "~")
    return out


def _regex_redact(text: str) -> tuple[str, bool]:
    hit = False

    def _mark(repl: str):
        nonlocal hit

        def _sub(m: re.Match[str]) -> str:
            nonlocal hit
            hit = True
            return repl if isinstance(repl, str) else repl(m)

        return _sub

    out = _AKIA_RE.sub(_mark("[redacted]"), text)
    out = _SK_RE.sub(_mark("[redacted]"), out)
    out = _XAI_RE.sub(_mark("[redacted]"), out)
    out = _BEARER_RE.sub(_mark("[redacted]"), out)
    out = _PEM_RE.sub(_mark("[redacted]"), out)
    out = _PASSWORD_RE.sub(_mark("password=[redacted]"), out)
    if _GMAIL_JSON_FIELD_RE.search(out):
        hit = True
        out = _GMAIL_JSON_FIELD_RE.sub(r"\1[redacted]\3", out)
    return out, hit


def redact_for_outbound(text: str) -> str:
    """Known secrets + regex. Home prefixes become ``~/``. Safe when the vault is locked."""
    if not text:
        return text
    original = text
    out = text
    ids: list[str] = []

    pairs = known_secret_pairs()
    # Longest plaintext first so a prefix of another secret cannot win.
    for token, secret in sorted(pairs, key=lambda p: len(p[1]), reverse=True):
        if secret and secret in out:
            out = out.replace(secret, token)
            ids.append(token[6:] if token.startswith("vault:") else token)

    for secret in oauth_plaintext_values():
        if secret and secret in out:
            out = out.replace(secret, "[redacted]")
            ids.append("oauth")

    out, regex_hit = _regex_redact(out)
    out = _redact_home(out)

    if out != original:
        _audit_redact(",".join(ids) if ids else ("regex" if regex_hit else "home"))
    return out


def _looks_like_mail(text: str) -> bool:
    return bool(_MAIL_HEADER_RE.search(text or ""))


def _maybe_summarize_mail(text: str) -> str:
    """Reuse the mail cleaner when a body would enter ChatOllama. Never used on operator chat."""
    if not _looks_like_mail(text):
        return text
    try:
        from primus.agents.system import _summarize_mail  # noqa: PLC0415

        return _summarize_mail(text)
    except Exception:  # noqa: BLE001
        return text


def redact_model_input(obj: Any) -> Any:
    """Redact (and mail-summarize) anything about to become model input. Copies; no mutate."""
    if obj is None:
        return obj
    if isinstance(obj, str):
        return redact_for_outbound(_maybe_summarize_mail(obj))
    if isinstance(obj, list):
        return [redact_model_input(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(redact_model_input(x) for x in obj)
    if isinstance(obj, dict):
        out = dict(obj)
        if "text" in out and isinstance(out["text"], str):
            out["text"] = redact_for_outbound(_maybe_summarize_mail(out["text"]))
        if "content" in out:
            out["content"] = redact_model_input(out["content"])
        return out
    content = getattr(obj, "content", None)
    if content is None:
        return obj
    new_c = redact_model_input(content)
    if new_c == content:
        return obj
    try:
        return obj.model_copy(update={"content": new_c})
    except Exception:  # noqa: BLE001
        try:
            clone = type(obj)(content=new_c)
            return clone
        except Exception:  # noqa: BLE001
            return obj
