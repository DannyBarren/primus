"""Gradio / launch compatibility patches and small launch helpers.

Extracted verbatim from admin_assistant.py in Phase 3. None of these depend on the live CFG, so
they move cleanly. Imported back into admin_assistant.py; call sites are unchanged.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("primus")


def _ensure_localhost_direct() -> None:
    """Keep localhost OUT of any HTTP proxy so Gradio's startup self-check can reach the server.

    The common Gradio launch crash ("When localhost is not accessible, a shareable link must be
    created") happens when http_proxy/https_proxy route 127.0.0.1 through a proxy that can't reach it.
    """
    locals_ = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        entries = [e.strip() for e in current.split(",") if e.strip()]
        for host in locals_:
            if host not in entries:
                entries.append(host)
        os.environ[var] = ",".join(entries)
    # Gradio analytics calls aren't needed locally and slow startup.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _patch_gradio_localhost_check() -> None:
    """Make Gradio's localhost self-check robust so local launches never wrongly fail.

    Gradio's url_ok() sends an httpx HEAD to '/', which some setups answer with 405 or that
    races the server startup — triggering the bogus "localhost not accessible, must create a
    shareable link" ValueError. We replace it with a lenient GET that ignores env proxies and
    trusts our own local server.
    """
    try:
        from gradio import networking  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.debug("Skipping gradio localhost patch: %s", exc)
        return

    _orig_url_ok = getattr(networking, "url_ok", None)

    def _lenient_url_ok(url: str) -> bool:
        # We start and own this server. For local URLs, trust it WITHOUT an HTTP probe —
        # the probe (httpx GET/HEAD) can hang against our own single-threaded server and
        # block launch() forever. Returning True lets launch() proceed; the server is up.
        if any(h in (url or "") for h in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]")):
            return True
        if _orig_url_ok:
            try:
                return _orig_url_ok(url)
            except Exception:  # noqa: BLE001
                return False
        return False

    networking.url_ok = _lenient_url_ok
    log.debug("Patched gradio.networking.url_ok — local URLs trusted without HTTP probe")



def _patch_gradio_schema_bug() -> None:
    """Work around a gradio_client bug that 500s the UI on load.

    When a component's JSON schema has a boolean `additionalProperties` (True/False),
    gradio_client.utils._json_schema_to_python_type recurses with that bool and then does
    `"const" in schema`, raising: TypeError: argument of type 'bool' is not iterable.
    This crashes /get_api_info (and the page that depends on it). We wrap the function to
    return "Any" for boolean schemas, which is harmless for local use.
    """
    try:
        from gradio_client import utils as _gc_utils  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.debug("Skipping gradio_client schema patch: %s", exc)
        return

    if getattr(_gc_utils, "_primus_schema_patched", False):
        return

    _orig = _gc_utils._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema: Any, defs: Any = None) -> str:
        if isinstance(schema, bool):
            return "Any"
        try:
            return _orig(schema, defs)
        except Exception:  # noqa: BLE001
            return "Any"

    _gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

    # get_type() is also called directly with the raw schema in places.
    _orig_get_type = getattr(_gc_utils, "get_type", None)
    if _orig_get_type:
        def _safe_get_type(schema: Any) -> Any:
            if isinstance(schema, bool):
                return "Any"
            try:
                return _orig_get_type(schema)
            except Exception:  # noqa: BLE001
                return "Any"

        _gc_utils.get_type = _safe_get_type

    _gc_utils._primus_schema_patched = True
    log.debug("Patched gradio_client schema parser (boolean additionalProperties bug)")



def _find_free_port(host: str, start_port: int, span: int = 20) -> int:
    """Return the first bindable port at/after start_port (so the UI comes up cleanly
    even when the preferred port is taken). Falls back to start_port if none are free."""
    import socket

    bind_host = "127.0.0.1" if host in ("", "0.0.0.0") else host
    for offset in range(span):
        port = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
                return port
            except OSError:
                continue
    return start_port
