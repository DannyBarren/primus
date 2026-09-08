"""Command-safety guardrails — extracted VERBATIM from admin_assistant.py.

Approval/auto-run command patterns. Unchanged and fully editable.
"""
import re

DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsudo\b", re.I), "elevated privileges (sudo)"),
    (re.compile(r"\bsu\b(?:\s|$|-)", re.I), "switch user (su)"),
    (re.compile(r"\brm\b(?:\s|$|-)", re.I), "file deletion (rm)"),
    (re.compile(r"\brmdir\b", re.I), "directory removal"),
    (re.compile(r"\bdd\b(?:\s|$|=)", re.I), "disk write (dd)"),
    (re.compile(r"\bmkfs\b", re.I), "filesystem format"),
    (re.compile(r"\bshutdown\b|\breboot\b|\bpoweroff\b|\bhalt\b", re.I), "system power control"),
    (re.compile(r"\bsystemctl\s+(suspend|hibernate|poweroff|reboot)", re.I), "power management"),
    (re.compile(r"\bkill\s+-9\b|\bkillall\b", re.I), "force kill processes"),
    (re.compile(r"\bchmod\s+777\b|\bchown\b", re.I), "permission changes"),
    (re.compile(r">\s*/dev/[a-z]", re.I), "direct device write"),
    (re.compile(r"\|\s*(ba)?sh\b", re.I), "pipe to shell"),
    (re.compile(r"curl[^\n|]*\|\s*(ba)?sh", re.I), "remote script execution"),
    (re.compile(r"wget[^\n|]*\|\s*(ba)?sh", re.I), "remote script execution"),
    (re.compile(r"\bgit\s+push\s+.*(-f|--force)\b", re.I), "force git push"),
    (re.compile(r"\bgit\s+reset\s+--hard\b", re.I), "hard git reset"),
    (re.compile(r"\bapt\s+(purge|remove|autoremove)", re.I), "package removal"),
    (re.compile(r"\bdpkg\s+-r\b", re.I), "package removal"),
    (re.compile(r"\bmv\b", re.I), "move/rename (mv)"),
    (re.compile(r"\bcp\s+-r\b", re.I), "recursive copy"),
]

# Read-only / low-risk command leaders — auto-run when auto_execute_safe_commands is enabled
SAFE_COMMAND_LEADERS = (
    r"ls\b", r"pwd\b", r"cd\b", r"cat\b", r"head\b", r"tail\b", r"less\b", r"more\b",
    r"tree\b", r"du\b", r"df\b", r"free\b", r"whoami\b", r"id\b", r"echo\b", r"file\b",
    r"stat\b", r"wc\b", r"grep\b", r"rg\b", r"lsblk\b", r"mount\b", r"uptime\b", r"date\b",
    r"which\b", r"type\b", r"hostname\b", r"uname\b", r"lscpu\b", r"lsusb\b", r"lsmem\b",
    r"printenv\b", r"env\b", r"basename\b", r"dirname\b", r"realpath\b", r"readlink\b",
    r"git\s+(?:status|log|diff|show|branch|remote|rev-parse|describe)\b",
    r"rclone\s+(?:listremotes|about|lsd|ls|size|ncdu|version)\b",
    r"ps\b", r"top\s+-bn1\b", r"pgrep\b", r"pidof\b",
    r"nmcli\s+(?:-t\s+)?(?:dev|device|connection|radio|general)\b",
    r"systemctl\s+(?:status|list-units|is-enabled|is-active|show)\b",
    r"ping\s+-c\b", r"ip\s+(?:-br\s+)?(?:addr|route|link)\b",
    r"journalctl\s+(?:--no-pager\s+)?(?:-n|-p|-u)\b",
    r"find\b", r"locate\b", r"whereis\b",
    r"apt\s+list\b", r"apt\s+search\b", r"apt\s+show\b", r"dpkg\s+-l\b",
    r"python3?\s+--version\b", r"uv\s+--version\b", r"ollama\s+list\b",
)
SAFE_LEADING_RE = re.compile(
    r"^\s*(?:" + "|".join(SAFE_COMMAND_LEADERS) + r")",
    re.I,
)
UNSAFE_IN_SAFE = re.compile(
    r"\b(rm\b|rmdir\b|sudo\b|mkfs\b|dd\b|>\s|>>\s|\|\s*(?:ba)?sh\b|-delete\b|-exec\b|chmod\b|chown\b)",
    re.I,
)
