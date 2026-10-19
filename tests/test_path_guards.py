"""Path authorization: FileMaid blocked roots/segments and write-gate composition."""
from __future__ import annotations

from pathlib import Path

import pytest

from primus.core.filemaid_safety import (
    BLOCKED_PATHS,
    is_path_blocked,
    is_write_blocked,
    normalize_path,
)


def test_normalize_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # expanduser uses the process home; we only assert resolve() is absolute.
    got = normalize_path(tmp_path / "a" / ".." / "b")
    assert got == (tmp_path / "b").resolve()


def test_filesystem_root_blocked():
    blocked, reason = is_path_blocked("/")
    assert blocked
    assert "root" in reason.lower() or reason.endswith(": /")


@pytest.mark.parametrize("root", [str(p) for p in BLOCKED_PATHS])
def test_system_roots_blocked(root):
    blocked, reason = is_path_blocked(root)
    assert blocked, f"{root} should be blocked"
    assert "Blocked" in reason


def test_child_of_system_root_blocked():
    blocked, reason = is_path_blocked("/etc/shadow")
    assert blocked
    assert "/etc" in reason


def test_home_file_not_blocked_by_filemaid(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("ok", encoding="utf-8")
    blocked, reason = is_path_blocked(target)
    assert blocked is False
    assert reason == ""


@pytest.mark.parametrize("segment", [".git", "node_modules", "__pycache__", ".venv", "venv"])
def test_junk_segments_blocked(tmp_path, segment):
    target = tmp_path / segment / "x.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x", encoding="utf-8")
    blocked, reason = is_path_blocked(target)
    assert blocked
    assert segment in reason


def test_is_write_blocked_uses_path_block():
    blocked, reason = is_write_blocked("/etc/passwd")
    assert blocked
    assert reason


def test_is_write_blocked_without_host_allows_home(tmp_path):
    target = tmp_path / "ok.txt"
    target.write_text("x", encoding="utf-8")
    blocked, reason = is_write_blocked(target)
    assert blocked is False
    assert reason == ""


def test_unresolvable_path_is_blocked():
    # A path that cannot be resolved should refuse, not pass.
    blocked, reason = is_path_blocked("\x00not-a-path")
    assert blocked
    assert "Cannot resolve" in reason or "Blocked" in reason
