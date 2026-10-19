"""Shell risk classifier: safe leaders, dangerous patterns, and documented gaps."""
from __future__ import annotations

import pytest

from primus.core.risk import assess_command_risk, classify_command


@pytest.mark.parametrize(
    "cmd",
    [
        "ls -la ~",
        "pwd",
        "cat ~/notes.md",
        "git status",
        "git -C ~/ai-workshop status",
        "df -h",
        "head -n 20 ~/.primus/notes.md",
    ],
)
def test_read_only_is_safe(cmd):
    risk, _ = assess_command_risk(cmd)
    assert risk == "safe", cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf ~",
        "sudo apt install cowsay",
        "dd if=/dev/zero of=/dev/sda",
        "git push --force",
        "git -C ~/ai-workshop push origin main --force",
        "git -C ~/repo push -f",
        "find ~ -delete",
        "find ~/Downloads -type f -delete",
        "echo $(rm -rf /tmp/x)",
        "echo `reboot`",
        "ls\nrm -rf /",
        "env rm -rf /tmp/primus-canary",
    ],
)
def test_destructive_or_hidden_is_dangerous(cmd):
    risk, reasons = assess_command_risk(cmd)
    assert risk == "dangerous", f"{cmd!r} → {risk} {reasons}"


def test_find_delete_regex_no_longer_false_safe():
    """Historical bug: ``\\b-delete`` never matched because ``-`` is a non-word char."""
    risk, reasons = assess_command_risk("find ~ -delete")
    assert risk != "safe"
    assert risk == "dangerous" or any("delete" in r.lower() for r in reasons)


def test_printenv_is_not_auto_safe():
    """printenv dumps the process environment (tokens, keys). Review, not auto-run."""
    risk, _ = assess_command_risk("printenv")
    assert risk != "safe"


def test_env_alone_is_not_auto_safe():
    risk, _ = assess_command_risk("env")
    assert risk != "safe"


def test_lone_ampersand_is_not_safe():
    risk, reasons = assess_command_risk("sleep 30 &")
    assert risk != "safe"
    assert any("background" in r for r in reasons) or risk == "review"


def test_git_force_with_lease_is_dangerous():
    risk, _ = assess_command_risk("git push --force-with-lease")
    assert risk == "dangerous"


def test_empty_command_is_review():
    risk, _ = assess_command_risk("   ")
    assert risk == "review"


def test_classify_command_sudo():
    dangerous, reasons = classify_command("sudo reboot")
    assert dangerous
    assert reasons
