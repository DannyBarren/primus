"""Allowlist checks for ``open_application`` / the ``run <words>`` fast path.

The fast path used to pass arbitrary user text to ``subprocess.Popen(shell=True)``.
Only known desktop apps, folder shortcuts, quoted URLs, and existing home paths
are launchable. Shell-shaped input is refused and must go through ``run_shell``.
"""
from __future__ import annotations

import re

# Single desktop-app token: ``firefox``, ``google-chrome-stable``, not ``rm -rf``.
_APP_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_SHELL_META_RE = re.compile(r"""[;&|`$()<>\n\r\\]|&&|\|\||\s""")

# Binaries that exist on $PATH but must never be treated as "open this app".
LAUNCH_DENY = frozenset({
    "rm", "rmdir", "sudo", "su", "dd", "mkfs", "find", "xargs",
    "env", "printenv", "bash", "sh", "zsh", "dash", "ksh", "fish",
    "python", "python3", "perl", "ruby", "node", "php",
    "curl", "wget", "nc", "ncat", "netcat", "ssh", "scp",
    "chmod", "chown", "kill", "killall", "pkill",
    "apt", "apt-get", "dpkg", "systemctl", "journalctl",
    "git", "rclone", "docker", "podman",
})


def looks_like_app_name(raw: str) -> bool:
    """True for a single token that is not a denied interpreter/tool."""
    name = (raw or "").strip()
    if not name or not _APP_TOKEN_RE.fullmatch(name):
        return False
    return name.lower() not in LAUNCH_DENY


def has_shell_meta(raw: str) -> bool:
    """True when ``raw`` is not a single safe token (spaces or shell metacharacters)."""
    return bool(_SHELL_META_RE.search((raw or "").strip()))


def is_launch_url(raw: str) -> bool:
    return bool(re.match(r"^https?://", (raw or "").strip(), re.I))
