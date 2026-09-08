"""Local secret vault — encrypted blob on device. No @tool. Passphrase never hits disk.

Key comes from PRIMUS_VAULT_PASS only. Tokens are vault:<id>. resolve() is for
existing tool _run (after the model chose the tool) and for later callers.
OAuth files stay at {APP_DIR}/gmail_*.json — they are never copied here.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from primus.config import APP_DIR, VAULT_ACCESS_LOG, VAULT_DIR, VAULT_INDEX

_LOCK = threading.Lock()
_TOKEN_RE = re.compile(r"^vault:([A-Za-z0-9_.-]+)$")
_EMBEDDED_TOKEN_RE = re.compile(r"vault:([A-Za-z0-9_.-]+)")
_ID_OK_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

OAUTH_BASENAMES = frozenset({
    "gmail_credentials.json",
    "gmail_token.json",
    "calendar_token.json",
})
_VAULT_META_NAMES = frozenset({
    "vault_index.json",
    "vault_access.jsonl",
})

_BLOB_NAME = "blob"
_SALT_LEN = 16
_ARGON2 = {"iterations": 2, "lanes": 1, "memory_cost": 19456}
_PBKDF2_ITERS = 480_000

try:
    from cryptography.fernet import Fernet, InvalidToken  # type: ignore[import-untyped]
    from cryptography.hazmat.primitives import hashes  # type: ignore[import-untyped]
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # type: ignore[import-untyped]

    _HAS_CRYPTO = True
except ImportError:  # pragma: no cover — requirements list cryptography
    Fernet = InvalidToken = hashes = PBKDF2HMAC = None  # type: ignore[misc,assignment]
    _HAS_CRYPTO = False


def _passphrase() -> str:
    return os.environ.get("PRIMUS_VAULT_PASS", "").strip()


def vault_locked() -> bool:
    """True when there is no passphrase or crypto is unavailable."""
    return not _HAS_CRYPTO or not _passphrase()


def _blob_path() -> Path:
    return VAULT_DIR / _BLOB_NAME


def _audit(action: str, *, id: str = "", dest: str = "api", ok: bool = True) -> None:
    """Append one vault_access.jsonl line. Never logs the secret or the passphrase."""
    try:
        VAULT_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "action": action,
            "id": id,
            "dest": dest,
            "ok": bool(ok),
        }
        with VAULT_ACCESS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        try:
            os.chmod(VAULT_ACCESS_LOG, 0o600)
        except OSError:
            pass
    except Exception:  # noqa: BLE001 — audit must never fail the caller
        pass


def _make_id(label: str) -> str:
    text = (label or "").strip()
    if text.lower().startswith("vault:"):
        text = text[6:].strip()
    if _ID_OK_RE.fullmatch(text):
        return text
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return slug or "secret"


def _token(ident: str) -> str:
    return f"vault:{ident}"


def _derive_key(password: bytes, salt: bytes, meta: dict[str, Any]) -> bytes:
    import base64

    kdf_name = str(meta.get("kdf") or "pbkdf2")
    if kdf_name == "argon2id":
        try:
            from cryptography.hazmat.backends.openssl.backend import backend
            from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

            if backend.argon2_supported():
                params = meta.get("argon2") or _ARGON2
                raw = Argon2id(
                    salt=salt,
                    length=32,
                    iterations=int(params.get("iterations", _ARGON2["iterations"])),
                    lanes=int(params.get("lanes", _ARGON2["lanes"])),
                    memory_cost=int(params.get("memory_cost", _ARGON2["memory_cost"])),
                ).derive(password)
                return base64.urlsafe_b64encode(raw)
        except Exception:  # noqa: BLE001 — fall through to PBKDF2
            pass
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=int(meta.get("pbkdf2_iters") or _PBKDF2_ITERS),
    )
    return base64.urlsafe_b64encode(kdf.derive(password))


def _new_kdf_meta(salt: bytes) -> dict[str, Any]:
    import base64

    meta: dict[str, Any] = {
        "v": 1,
        "salt": base64.b64encode(salt).decode("ascii"),
        "kdf": "pbkdf2",
        "pbkdf2_iters": _PBKDF2_ITERS,
    }
    if not _HAS_CRYPTO:
        return meta
    try:
        from cryptography.hazmat.backends.openssl.backend import backend

        if backend.argon2_supported():
            meta["kdf"] = "argon2id"
            meta["argon2"] = dict(_ARGON2)
    except Exception:  # noqa: BLE001
        pass
    return meta


def _fernet(meta: dict[str, Any]) -> Any:
    import base64

    pw = _passphrase()
    if not pw or not _HAS_CRYPTO:
        return None
    salt = base64.b64decode(meta["salt"])
    return Fernet(_derive_key(pw.encode("utf-8"), salt, meta))


def _read_wrapper() -> Optional[dict[str, Any]]:
    path = _blob_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("salt") and data.get("blob") else None


def _write_wrapper(meta: dict[str, Any], blob: str) -> None:
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(VAULT_DIR, 0o700)
    except OSError:
        pass
    payload = dict(meta)
    payload["blob"] = blob
    path = _blob_path()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _decrypt_map(wrapper: dict[str, Any]) -> Optional[dict[str, str]]:
    f = _fernet(wrapper)
    if f is None:
        return None
    try:
        raw = f.decrypt(wrapper["blob"].encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001 — wrong pass / corrupt → locked, no crash
        return None
    secrets = data.get("secrets") if isinstance(data, dict) else None
    if not isinstance(secrets, dict):
        return {}
    return {str(k): str(v) for k, v in secrets.items()}


def _load_index() -> dict[str, str]:
    """id → label. Never contains plaintext secrets."""
    if not VAULT_INDEX.is_file():
        return {}
    try:
        data = json.loads(VAULT_INDEX.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return {}
    out: dict[str, str] = {}
    for item in entries:
        if isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = str(item.get("label") or item["id"])
    return out


def _write_index(labels: dict[str, str]) -> None:
    VAULT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "version": 1,
        "entries": [{"id": i, "label": labels[i]} for i in sorted(labels)],
    }
    VAULT_INDEX.write_text(json.dumps(body, indent=2), encoding="utf-8")
    try:
        os.chmod(VAULT_INDEX, 0o600)
    except OSError:
        pass


def known_secret_pairs() -> list[tuple[str, str]]:
    """``[(vault:id, plaintext), ...]`` when unlocked; else []. Values stay in-process."""
    if vault_locked():
        return []
    with _LOCK:
        wrapper = _read_wrapper()
        if wrapper is None:
            return []
        secrets = _decrypt_map(wrapper)
        if secrets is None:
            return []
        return [(_token(i), val) for i, val in secrets.items() if val]


def store_secret(label: str, value: str) -> str:
    """Encrypt ``value`` under ``label``. Returns ``vault:<id>``. Locked → token, no persist."""
    ident = _make_id(label)
    token = _token(ident)
    if vault_locked():
        _audit("locked", id=ident, dest="api", ok=False)
        return token
    with _LOCK:
        wrapper = _read_wrapper()
        secrets: dict[str, str] = {}
        meta: dict[str, Any]
        if wrapper is not None:
            loaded = _decrypt_map(wrapper)
            if loaded is None:
                _audit("locked", id=ident, dest="api", ok=False)
                return token
            secrets = loaded
            meta = {k: v for k, v in wrapper.items() if k != "blob"}
        else:
            meta = _new_kdf_meta(os.urandom(_SALT_LEN))
        secrets[ident] = value
        f = _fernet(meta)
        if f is None:
            _audit("locked", id=ident, dest="api", ok=False)
            return token
        blob = f.encrypt(json.dumps({"secrets": secrets}).encode("utf-8")).decode("ascii")
        _write_wrapper(meta, blob)
        labels = _load_index()
        labels[ident] = (label or ident).strip() or ident
        _write_index(labels)
        _audit("store", id=ident, dest="api", ok=True)
    return token


def resolve(token: str, *, dest: str = "tool") -> Optional[str]:
    """Return the plaintext for ``vault:<id>``, or None if locked / unknown / not a token."""
    raw = (token or "").strip()
    m = _TOKEN_RE.fullmatch(raw)
    if not m:
        return None
    ident = m.group(1)

    def _shared_resolve(ok: bool) -> None:
        try:
            from primus.core.audit import write_audit  # noqa: PLC0415

            write_audit("vault_resolve", ok=ok, detail=ident)
        except Exception:  # noqa: BLE001 — shared audit must never fail resolve
            pass

    if vault_locked():
        _audit("locked", id=ident, dest=dest, ok=False)
        _shared_resolve(False)
        return None
    with _LOCK:
        wrapper = _read_wrapper()
        if wrapper is None:
            _audit("resolve", id=ident, dest=dest, ok=False)
            _shared_resolve(False)
            return None
        secrets = _decrypt_map(wrapper)
        if secrets is None:
            _audit("locked", id=ident, dest=dest, ok=False)
            _shared_resolve(False)
            return None
        value = secrets.get(ident)
        _audit("resolve", id=ident, dest=dest, ok=value is not None)
        _shared_resolve(value is not None)
        return value


def expand_vault_tokens(text: str) -> str:
    """Replace embedded ``vault:<id>`` tokens. Unresolved tokens stay as-is."""
    if not text or "vault:" not in text:
        return text

    def _repl(m: re.Match[str]) -> str:
        val = resolve(m.group(0), dest="tool")
        return val if val is not None else m.group(0)

    return _EMBEDDED_TOKEN_RE.sub(_repl, text)


def _expand_value(val: Any) -> Any:
    if isinstance(val, str):
        return expand_vault_tokens(val)
    if isinstance(val, dict):
        return {k: _expand_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_expand_value(v) for v in val]
    if isinstance(val, tuple):
        return tuple(_expand_value(v) for v in val)
    return val


def attach_resolve(tool: Any) -> Any:
    """Expand ``vault:<id>`` inside an existing tool's ``_run`` (via ``.func``).

    StructuredTool._run keeps its LangChain signature (``config=`` is keyword-only);
    tokens are resolved in ``.func``, which ``_run`` calls after the model chose the tool.
    TerminalTool / GitTool expand inside their own ``_run`` bodies.
    """
    if tool is None or getattr(tool, "_primus_vault_resolve", False):
        return tool
    func = getattr(tool, "func", None)
    if not callable(func):
        return tool

    def _func(*args: Any, **kwargs: Any) -> Any:
        return func(*_expand_value(args), **_expand_value(kwargs))

    try:
        object.__setattr__(tool, "func", _func)
        object.__setattr__(tool, "_primus_vault_resolve", True)
    except Exception:  # noqa: BLE001
        try:
            tool.func = _func
            tool._primus_vault_resolve = True
        except Exception:  # noqa: BLE001
            return tool
    return tool


def is_ingest_blocked(path: Path) -> bool:
    """True for OAuth files and vault stores — never ingest, never copy into the vault."""
    try:
        resolved = path.expanduser().resolve()
    except Exception:  # noqa: BLE001
        resolved = Path(path)
    name = resolved.name
    if name in OAUTH_BASENAMES or name in _VAULT_META_NAMES:
        return True
    try:
        vault_root = VAULT_DIR.expanduser().resolve()
        if resolved == vault_root or vault_root in resolved.parents:
            return True
    except Exception:  # noqa: BLE001
        if "vault" in resolved.parts:
            return True
    return False


def oauth_plaintext_values() -> list[str]:
    """In-memory OAuth JSON field values for redaction. Files stay put — not vaulted."""
    keys = {"token", "access_token", "refresh_token", "client_secret"}
    found: list[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in keys and isinstance(v, str) and v.strip():
                    found.append(v)
                else:
                    _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    for name in OAUTH_BASENAMES:
        fp = APP_DIR / name
        if not fp.is_file():
            continue
        try:
            _walk(json.loads(fp.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return found
