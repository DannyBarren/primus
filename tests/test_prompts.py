"""Prompt + package metadata that used to drift."""
from __future__ import annotations

import primus
from primus.core.prompts import FORGE_FAST_SYSTEM, PRIMUS_SYSTEM
from primus.core.tool_packs import PACKS


def test_version_is_semver():
    parts = primus.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
    assert tuple(int(p) for p in parts) >= (0, 9, 4)


def test_forge_fast_system_is_defined():
    assert isinstance(FORGE_FAST_SYSTEM, str)
    assert "Forge" in FORGE_FAST_SYSTEM
    assert "{mode}" in FORGE_FAST_SYSTEM


def test_pack_names_are_documented_counts():
    unique = {n for tools in PACKS.values() for n in tools}
    assert len(unique) >= 40
    # default pack stays small so a 4096-token leftover turn is not schema-flooded
    assert 3 <= len(PACKS["default"]) <= 8


def test_primus_system_mentions_mode_gate():
    assert "{mode}" in PRIMUS_SYSTEM
    assert "suggest" in PRIMUS_SYSTEM.lower()
