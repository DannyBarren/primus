"""Wording for replies the operator actually reads.

Facts stay as the tools reported them. This module only drops question-words
that leaked into labels, shell-command echoes, and stiff prefixes.
"""
from __future__ import annotations

import re

# Words that show up in "what will the weather be" and are not place names.
_NOT_PLACE = frozenset({
    "a", "an", "the", "my", "me", "our", "your", "you", "it", "its",
    "what", "whats", "what's", "how", "hows", "how's", "when", "where", "why", "who", "which",
    "will", "would", "wont", "won't", "can", "could", "should", "shall", "may", "might",
    "be", "been", "being", "is", "are", "was", "were", "do", "does", "did",
    "have", "has", "had", "going", "gonna", "to", "for", "of", "on", "at", "in",
    "and", "or", "but", "with", "from", "about", "around", "near", "into",
    "there", "here", "that", "this", "just", "really", "actually", "any", "some",
    "today", "tonight", "tomorrow", "now", "currently", "please", "right",
    "like", "look", "looks", "looking", "seem", "seems",
    "good", "bad", "nice", "out", "up", "down", "outside", "local", "current",
    "tell", "give", "show", "get", "let",
    "weather", "forecast", "temperature", "temp", "rain", "snow",
    "morning", "afternoon", "evening", "weekend", "later", "soon", "next", "last",
    "latest", "top", "biggest", "headline", "headlines", "news", "story", "stories",
})

_SHELL_ECHO_RE = re.compile(
    r"▶\s*\*\*(?:Auto-ran|Running|Executed)\*\*[^\n]*\n```(?:bash)?\n.*?\n```\s*",
    re.S | re.I,
)
_LAUNCH_ECHO_RE = re.compile(r"(?m)^[ \t]*▶\s*Launched:\s*`[^`]+`[ \t]*$")


def clean_place(text: str) -> str:
    """Place name from a question. Question words drop out; 'new york' stays."""
    raw = re.sub(r"[?!.,'\"`]+", " ", text or "")
    kept = [tok for tok in raw.split() if tok.lower() not in _NOT_PLACE]
    return " ".join(kept).strip()


def clean_topic(text: str) -> str:
    """News/search topic. Same filler list so a header cannot become 'on will'."""
    return clean_place(text)


def _temp(token: str) -> str:
    token = token.replace(" ", "")
    return re.sub(r"^\+", "", token)


def voice_weather(text: str) -> str:
    """Turn a wttr one-liner (or an old 'Weather — Place:' label) into a sentence."""
    raw = (text or "").strip()
    if not raw:
        return "I don't have a weather reading right now."
    low = raw.lower()
    if low.startswith(("i couldn't", "i could not", "the weather service", "it's ", "its ")):
        return raw
    raw = re.sub(r"^weather\s*[—:-]\s*", "", raw, flags=re.I).strip()
    match = re.match(r"^(?P<place>[^:\n]{1,60}):\s*(?P<rest>.+)$", raw)
    if not match:
        return raw
    place = match.group("place").strip()
    if clean_place(place) == "":
        place = ""
    rest = match.group("rest").strip()
    temp_m = re.search(r"([+-]?\d+(?:\.\d+)?)(\s*°[CF])", rest)
    feels_m = re.search(r"feels\s+([+-]?\d+(?:\.\d+)?)(\s*°[CF])", rest, re.I)
    hum_m = re.search(r"(\d+)\s*%", rest)
    wind_m = re.search(r"wind\s+(.+?)\s*$", rest, re.I)
    cond = ""
    first = rest.split()[0] if rest.split() else ""
    if first and not re.search(r"\d|°", first):
        cond = first
    if not temp_m:
        body = rest.rstrip(".")
        return f"{body} in {place}." if place else f"{body}."
    temp = _temp(temp_m.group(1) + temp_m.group(2))
    line = f"It's {temp}"
    if cond:
        line += f" {cond}"
    if place:
        line += f" in {place}"
    if feels_m:
        line += f", feels like {_temp(feels_m.group(1) + feels_m.group(2))}"
    line += "."
    extra: list[str] = []
    if hum_m:
        extra.append(f"humidity {hum_m.group(1)}%")
    if wind_m:
        extra.append(f"wind {wind_m.group(1).strip().rstrip('.')}")
    if extra:
        tail = extra[0][0].upper() + extra[0][1:]
        if len(extra) > 1:
            tail += f", {extra[1]}"
        line += f" {tail}."
    return line


def place_label(target: str) -> str:
    """Short folder name for a sentence. A custom path is left as given."""
    raw = (target or "").strip().rstrip("/")
    low = raw.lower().replace("\\", "/")
    named = {
        "~": "Home",
        "": "Home",
        "~/downloads": "Downloads",
        "~/desktop": "Desktop",
        "~/documents": "Documents",
        "~/pictures": "Pictures",
        "~/music": "Music",
        "~/videos": "Videos",
    }
    if low in named:
        return named[low]
    for suffix, label in (
        ("/downloads", "Downloads"),
        ("/desktop", "Desktop"),
        ("/documents", "Documents"),
        ("/pictures", "Pictures"),
    ):
        if low.endswith(suffix):
            return label
    if raw.startswith("~/"):
        return raw.split("/")[-1] or "Home"
    return raw or "That folder"


def count_sentence(names: list[str], noun: str, target: str) -> str:
    """'Downloads has 2 folders: a, b.' `noun` is the plural ('folders' or 'files')."""
    place = place_label(target)
    n = len(names)
    if n == 0:
        return f"No {noun} in {place}."
    word = noun[:-1] if n == 1 and noun.endswith("s") else noun
    shown = ", ".join(names[:8])
    if n > 8:
        shown += f", and {n - 8} more"
    return f"{place} has {n} {word}: {shown}."


def present_listing(listing: str, target: str = "") -> str:
    """Turn `[dir]` / `[file]` lines into a short inventory. Errors pass through."""
    text = listing or ""
    if text.startswith(("✗", "Refusing", "Not found", "No matches")):
        return text
    dirs: list[str] = []
    files: list[str] = []
    for line in text.splitlines():
        if line.startswith("[dir] "):
            dirs.append(line.split("] ", 1)[1].rsplit(" (", 1)[0])
        elif line.startswith("[file] "):
            files.append(line.split("] ", 1)[1].rsplit(" (", 1)[0])
    if not dirs and not files:
        if "[dir]" in text or "[file]" in text or "0 files" in text:
            return f"{place_label(target)} is empty."
        return text
    place = place_label(target)
    bits: list[str] = []
    if dirs:
        bits.append(f"{len(dirs)} folder" + ("" if len(dirs) == 1 else "s"))
    if files:
        bits.append(f"{len(files)} file" + ("" if len(files) == 1 else "s"))
    names = dirs + files
    shown = ", ".join(names[:12])
    if len(names) > 12:
        shown += f", and {len(names) - 12} more"
    return f"{place} has {' and '.join(bits)}.\n{shown}"


def soften_listings(text: str) -> str:
    """Rewrite a reply that is only a raw directory dump."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return text
    marked = [ln for ln in lines if ln.startswith(("[dir]", "[file]"))]
    if marked and len(marked) >= max(1, len(lines) - 1):
        return present_listing(text)
    return text


def strip_shell_echo(text: str) -> str:
    """Remove the '▶ **Executed** ```bash```' wrapper. Keep the command's output."""
    out = _SHELL_ECHO_RE.sub("", text or "")
    out = _LAUNCH_ECHO_RE.sub("", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def present_battery(raw: str) -> str:
    """One sentence from upower / sysfs output. Raw text if it isn't a battery dump."""
    body = strip_shell_echo(raw)
    pct = re.search(r"percentage:\s*(\d+)\s*%", body, re.I)
    if not pct:
        lone = re.search(r"(?m)^\s*(\d{1,3})\s*$", body)
        pct_s = lone.group(1) if lone else ""
    else:
        pct_s = pct.group(1)
    if not pct_s:
        return body or "I couldn't read the battery from this machine."
    state_m = re.search(r"(?m)^\s*state:\s*(\S+)", body, re.I)
    state = (state_m.group(1).lower() if state_m else "").strip(".,")
    state_word = {
        "discharging": "and discharging",
        "charging": "and charging",
        "fully-charged": "and fully charged",
        "pending-charge": "and about to charge",
    }.get(state, "")
    line = f"Battery is at {pct_s}%"
    if state_word:
        line += f", {state_word}"
    left = re.search(r"time to empty:\s*(.+)", body, re.I)
    full = re.search(r"time to full:\s*(.+)", body, re.I)
    if left:
        line += f", about {left.group(1).strip().rstrip('.')} left"
    elif full:
        line += f", about {full.group(1).strip().rstrip('.')} until full"
    return line + "."


def present_bluetooth(raw: str) -> str:
    """One or two sentences from `bluetoothctl show`. Raw text if it isn't that."""
    body = strip_shell_echo(raw)
    powered = re.search(r"(?m)^Powered:\s*(\w+)", body)
    name = re.search(r"(?m)^Name:\s*(.+)", body)
    devices = re.findall(r"(?m)^Device\s+\S+\s+(.+)$", body)
    if not powered and not devices:
        return body or "I couldn't read Bluetooth from this machine."
    on = bool(powered and powered.group(1).lower() == "yes")
    if on and name:
        line = f"Bluetooth is on ({name.group(1).strip()})."
    else:
        line = "Bluetooth is on." if on else "Bluetooth is off."
    if devices:
        shown = ", ".join(d.strip() for d in devices[:5])
        line += f" Paired: {shown}."
    return line


def present_wifi(raw: str) -> str:
    """One sentence from `nmcli -t` wifi rows. Raw text if it isn't that table."""
    body = strip_shell_echo(raw)
    connected: tuple[str, str] | None = None
    seen = False
    for line in body.splitlines():
        parts = line.strip().split(":")
        if len(parts) < 3 or parts[0] not in ("yes", "no"):
            continue
        seen = True
        ssid = parts[1].strip() or "a hidden network"
        signal = parts[2].strip()
        if parts[0] == "yes" and connected is None:
            connected = (ssid, signal)
    if connected:
        ssid, signal = connected
        if signal.isdigit():
            return f"You're on {ssid} ({signal}% signal)."
        return f"You're on {ssid}."
    if seen:
        return "Wi-Fi is on, but you're not connected to a network."
    return body or "I couldn't see a Wi-Fi connection."


def polish_tool_text(text: str) -> str:
    """Last pass for fast paths and chat: echoes, listings, weather labels, status dumps."""
    if not text:
        return text
    out = strip_shell_echo(text)
    if not out:
        return text.strip()
    dumped = out.strip()
    if re.search(r"percentage:\s*\d+\s*%|native-path:", dumped, re.I):
        voiced = present_battery(dumped)
        if voiced and "percentage:" not in voiced.lower():
            return voiced
    if re.search(r"(?m)^(yes|no):", dumped):
        voiced = present_wifi(dumped)
        if voiced and not re.search(r"(?m)^(yes|no):", voiced):
            return voiced
    if re.search(r"(?m)^Powered:\s*\w+", dumped):
        voiced = present_bluetooth(dumped)
        if voiced and "Powered:" not in voiced:
            return voiced
    if re.match(r"(?i)^weather\s*[—:-]", dumped) or (
        "\n" not in dumped and re.search(r"°[CF]", dumped) and ":" in dumped.split("\n", 1)[0]
    ):
        return voice_weather(dumped)
    return soften_listings(out)
