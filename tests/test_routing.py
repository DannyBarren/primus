"""Launch-target routing, tool-pack selection, and auth bind policy."""
from __future__ import annotations

import os
import re
import threading

import pytest

from primus.core.auth import (
    auth_credentials,
    is_loopback_host,
    is_remote_bind,
    require_auth_for_bind,
)
from primus.core.launch import has_shell_meta, is_launch_url, looks_like_app_name
from primus.core.tool_packs import (
    PACKS,
    bind_tools,
    clear_turn_pack,
    current_pack,
    select_pack,
    set_turn_pack,
)

# Mirrors the fast-path opener in primus.agents.system — imported here as a regex so
# this test does not load the agent host (Ollama / langchain).
_FT_OPEN_RE = re.compile(r"^\s*(?:open|launch|start|run|fire up|bring up)\s+(.+?)\s*$", re.I)


@pytest.mark.parametrize(
    "msg, expected",
    [
        ("run rm -rf ~", "rm -rf ~"),
        ("run find ~ -delete", "find ~ -delete"),
        ("open firefox", "firefox"),
        ("launch Zoom", "Zoom"),
    ],
)
def test_run_fast_path_captures_raw_words(msg, expected):
    m = _FT_OPEN_RE.match(msg)
    assert m is not None
    assert m.group(1).strip() == expected


def test_shell_shaped_run_is_not_an_app_name():
    assert has_shell_meta("rm -rf ~")
    assert has_shell_meta("find ~ -delete")
    assert not looks_like_app_name("rm -rf ~")
    assert not looks_like_app_name("find")
    assert not looks_like_app_name("env")


def test_known_app_token_is_allowed():
    assert looks_like_app_name("firefox")
    assert looks_like_app_name("google-chrome-stable")
    assert not has_shell_meta("firefox")


def test_url_is_recognized():
    assert is_launch_url("https://example.com")
    assert not is_launch_url("firefox")


@pytest.mark.parametrize(
    "text, pack",
    [
        ("list my unread gmail", "mail"),
        ("read ~/notes.md", "files"),
        ("git status", "shell"),
        ("remember that I prefer dark mode", "knowledge"),
        ("search the web for langgraph checkpointer", "web"),
        ("what's using my CPU", "system"),
        ("list slack channels", "comms"),
        ("hello there", "default"),
    ],
)
def test_select_pack(text, pack):
    assert select_pack(text) == pack


def test_bind_tools_never_returns_full_registry():
    class T:
        def __init__(self, name):
            self.name = name

    registry = [T(f"tool_{i}") for i in range(30)]
    registry += [T("list_directory"), T("get_datetime"), T("get_news"), T("ask_grok")]
    bound = bind_tools(registry, "default")
    assert bound
    assert len(bound) < len(registry)
    assert {t.name for t in bound} <= set(PACKS["default"])


def test_more_than_four_tools_live_in_packs():
    names = {n for pack in PACKS.values() for n in pack}
    assert len(names) >= 40
    assert "long_running_command" in PACKS["shell"]
    assert "web_search" in PACKS["web"]


def test_turn_pack_visible_on_another_thread():
    """The leftover graph runs on a worker thread; TLS-only storage lost the pack."""
    clear_turn_pack()
    set_turn_pack("mail", ["gmail_status"])
    seen: list[str] = []

    def _worker():
        seen.append(current_pack())

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert seen == ["mail"]
    clear_turn_pack()


def test_loopback_hosts():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.1.10")


def test_remote_bind_requires_auth(monkeypatch):
    monkeypatch.delenv("PRIMUS_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PRIMUS_PASSWORD", raising=False)
    with pytest.raises(SystemExit) as exc:
        require_auth_for_bind("0.0.0.0")
    assert "Refusing to start" in str(exc.value)


def test_share_requires_auth(monkeypatch):
    monkeypatch.delenv("PRIMUS_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PRIMUS_PASSWORD", raising=False)
    with pytest.raises(SystemExit):
        require_auth_for_bind("127.0.0.1", share=True)


def test_loopback_without_auth_is_ok(monkeypatch):
    monkeypatch.delenv("PRIMUS_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PRIMUS_PASSWORD", raising=False)
    assert require_auth_for_bind("127.0.0.1") is None


def test_auth_credentials_from_env(monkeypatch):
    monkeypatch.setenv("PRIMUS_AUTH_TOKEN", "s3cret")
    monkeypatch.setenv("PRIMUS_AUTH_USER", "op")
    assert auth_credentials() == ("op", "s3cret")
    assert require_auth_for_bind("0.0.0.0") == ("op", "s3cret")


def test_is_remote_bind():
    assert is_remote_bind("0.0.0.0")
    assert is_remote_bind("127.0.0.1", share=True)
    assert not is_remote_bind("127.0.0.1")
