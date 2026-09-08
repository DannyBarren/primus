"""Backup export / import. No @tool. Operator CLI / ``make export`` only.

Default zip excludes ``gmail_token.json``, ``gmail_credentials.json``, and vault
blobs. Import never overwrites live tokens unless the zip **and** the operator
both passed ``--include-secrets``.

Library modules reach the host via ``_host`` — do not import admin_assistant.
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import sys as _sys

_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

from primus.config import APP_DIR, CONFIG_DIR, EXPORT_DIR  # noqa: E402

MANIFEST_NAME = "primus-backup.json"
APP_PREFIX = "app/"
CONFIG_PREFIX = "config/"

SECRET_BASENAMES = frozenset({
    "gmail_token.json",
    "gmail_credentials.json",
    "calendar_token.json",
})
VAULT_BLOB_NAMES = frozenset({"blob"})
SKIP_DIR_NAMES = frozenset({"exports", "__pycache__"})


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_secret_path(path: Path, *, root: Optional[Path] = None) -> bool:
    name = path.name
    if name in SECRET_BASENAMES or name.endswith("token.json"):
        return True
    if name in VAULT_BLOB_NAMES:
        return True
    try:
        parts = path.parts if root is None else path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    if "vault" in parts and name in VAULT_BLOB_NAMES:
        return True
    return False


def _is_secret_arcname(arcname: str) -> bool:
    p = Path(arcname)
    name = p.name
    if name in SECRET_BASENAMES or name.endswith("token.json"):
        return True
    if "vault" in p.parts and name in VAULT_BLOB_NAMES:
        return True
    return False


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part in SKIP_DIR_NAMES for part in rel.parts):
            continue
        yield path


def _manifest(include_secrets: bool, files: list[str]) -> dict:
    return {
        "version": 1,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "include_secrets": bool(include_secrets),
        "files": files,
    }


def export_bundle(*, include_secrets: bool = False, dest_dir: Optional[Path] = None) -> Path:
    """Write a zip under ``{APP_DIR}/exports/``. Logs one audit line. Never a @tool."""
    from primus.core.audit import write_audit  # noqa: PLC0415

    out_dir = Path(dest_dir) if dest_dir else EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"primus-backup-{_now_stamp()}.zip"
    members: list[str] = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in _iter_files(APP_DIR):
            if not include_secrets and _is_secret_path(src, root=APP_DIR):
                continue
            try:
                rel = src.relative_to(APP_DIR)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] == "exports":
                continue
            arc = APP_PREFIX + rel.as_posix()
            zf.write(src, arc)
            members.append(arc)
        for src in _iter_files(CONFIG_DIR):
            if src.name == "primus.log":
                continue
            if not include_secrets and _is_secret_path(src, root=CONFIG_DIR):
                continue
            try:
                rel = src.relative_to(CONFIG_DIR)
            except ValueError:
                continue
            arc = CONFIG_PREFIX + rel.as_posix()
            zf.write(src, arc)
            members.append(arc)
        zf.writestr(
            MANIFEST_NAME,
            json.dumps(_manifest(include_secrets, members), indent=2) + "\n",
        )

    write_audit(
        "export",
        ok=True,
        detail=f"{zip_path.name} secrets={'yes' if include_secrets else 'no'} files={len(members)}",
    )
    return zip_path


def _read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        raw = zf.read(MANIFEST_NAME).decode("utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _dest_for_arc(arcname: str) -> Optional[Path]:
    if arcname == MANIFEST_NAME:
        return None
    if arcname.startswith(APP_PREFIX):
        rel = arcname[len(APP_PREFIX):]
        if not rel or rel.endswith("/"):
            return None
        return APP_DIR / rel
    if arcname.startswith(CONFIG_PREFIX):
        rel = arcname[len(CONFIG_PREFIX):]
        if not rel or rel.endswith("/"):
            return None
        return CONFIG_DIR / rel
    return None


def preview_import(zip_path: Path, *, include_secrets: bool = False) -> tuple[list[str], list[str], bool]:
    """Return (will_overwrite, skipped_secrets, zip_had_secrets)."""
    zpath = Path(zip_path).expanduser()
    overwrite: list[str] = []
    skipped: list[str] = []
    with zipfile.ZipFile(zpath, "r") as zf:
        manifest = _read_manifest(zf)
        zip_had = bool(manifest.get("include_secrets"))
        allow_secrets = zip_had and include_secrets
        for info in zf.infolist():
            if info.is_dir() or info.filename == MANIFEST_NAME:
                continue
            dest = _dest_for_arc(info.filename)
            if dest is None:
                continue
            secret = _is_secret_arcname(info.filename)
            if secret and not allow_secrets:
                skipped.append(str(dest))
                continue
            if dest.exists():
                overwrite.append(str(dest))
    return overwrite, skipped, zip_had


def import_bundle(zip_path: str | Path, *, include_secrets: bool = False) -> str:
    """Printable restore: lists overwrites, then writes. Tokens need both flags."""
    from primus.core.audit import write_audit  # noqa: PLC0415

    zpath = Path(zip_path).expanduser()
    if not zpath.is_file():
        return f"No backup zip at `{zpath}`."

    overwrite, skipped, zip_had = preview_import(zpath, include_secrets=include_secrets)
    allow_secrets = zip_had and include_secrets
    lines = ["Restore preview:"]
    if overwrite:
        lines.append("Will overwrite:")
        lines.extend(f"  {p}" for p in overwrite)
    else:
        lines.append("Will overwrite: (nothing already on disk)")
    if skipped:
        why = (
            "zip was not exported with --include-secrets"
            if not zip_had
            else "operator did not pass --include-secrets"
        )
        lines.append(f"Skipping secrets ({why}):")
        lines.extend(f"  {p}" for p in skipped)
    if include_secrets and not zip_had:
        lines.append("Note: --include-secrets on import is ignored — this zip has no secrets.")

    restored = 0
    with zipfile.ZipFile(zpath, "r") as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename == MANIFEST_NAME:
                continue
            dest = _dest_for_arc(info.filename)
            if dest is None:
                continue
            if _is_secret_arcname(info.filename) and not allow_secrets:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(info.filename))
            restored += 1

    write_audit(
        "import",
        ok=True,
        detail=f"{zpath.name} restored={restored} secrets={'yes' if allow_secrets else 'no'}",
    )
    lines.append(f"Restored {restored} file(s) from `{zpath}`.")
    return "\n".join(lines)
