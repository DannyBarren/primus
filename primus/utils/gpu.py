"""AMD GPU acceleration (Ryzen AI / Radeon iGPU) — detection, env setup, and status.

Extracted from admin_assistant.py in Phase 3. Behavior is identical; the only change is that the
functions which read the live runtime config now reach it through a provider that the main module
registers with ``bind_cfg(lambda: CFG)``. The authoritative CFG dict (and its rebinding in
init_config) stays in admin_assistant.py — this keeps every call site unchanged and the rebind
fully effective. Nothing here is locked or read-only.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Optional

try:  # psutil is the preferred precise process tool; we degrade to filtered pgrep without it.
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore[assignment]

log = logging.getLogger("primus")

# Provider that returns the live CFG dict (set by admin_assistant.py at import via bind_cfg()).
_cfg_provider: Optional[Any] = None


def bind_cfg(provider: Any) -> None:
    """Register a zero-arg callable returning the live CFG dict.

    admin_assistant.py calls ``bind_cfg(lambda: CFG)`` so these helpers always see the current,
    rebound configuration without importing the (rebindable) global directly.
    """
    global _cfg_provider
    _cfg_provider = provider


def _cfg() -> dict:
    """Live config dict (empty dict if not yet bound — defensive, never raises)."""
    return _cfg_provider() if _cfg_provider is not None else {}


def bust_status_cache() -> None:
    """Invalidate the cached GPU status so the next get_gpu_status() re-reads `ollama ps`."""
    global _gpu_status_cache_at
    _gpu_status_cache_at = 0.0


# ---------------------------------------------------------------------------
# AMD GPU acceleration (Ryzen AI / Radeon iGPU) — detection + env setup
#
# Ollama is a separate server process. Env vars set here apply to (a) the embedded
# ollama Python client and (b) any `ollama serve` Primus launches as a child. If the
# server is already running, restart it with the recommended export block (shown in
# the Setup tab and `/gpu status`) for the settings to take effect. Everything below
# is best-effort and never raises — Primus runs fine on CPU if detection fails.
# ---------------------------------------------------------------------------

# Common ROCm gfx overrides for AMD APUs that ship without official ROCm targets.
# Radeon 880M/890M (RDNA 3.5, gfx1150/gfx1151) are typically run as gfx1100 → "11.0.0",
# but "11.0.2" is the most widely reported working override for Ryzen AI 300 iGPUs.
_RYZEN_AI_HSA_DEFAULT = "11.0.2"
_gpu_status_cache: dict[str, Any] = {}
_gpu_status_cache_at: float = 0.0


def _detect_amd_gpu_name() -> str:
    """Best-effort AMD GPU model string via lspci; '' if none/undetectable."""
    exe = shutil.which("lspci")
    if not exe:
        return ""
    try:
        out = subprocess.run(
            [exe], capture_output=True, text=True, timeout=4
        ).stdout
    except Exception:  # noqa: BLE001
        return ""
    for line in out.splitlines():
        if re.search(r"\b(VGA|Display|3D)\b", line) and re.search(r"\b(AMD|ATI|Radeon)\b", line, re.I):
            # Trim "00:02.0 VGA compatible controller: " prefix.
            return re.sub(r"^\S+\s+[^:]+:\s*", "", line).strip()
    return ""


def _is_ryzen_ai_igpu(name: str) -> bool:
    """True for RDNA3.5 Ryzen AI iGPUs (Radeon 880M/890M and family) needing an HSA override."""
    return bool(re.search(r"\b(8\d0M|Radeon 8\d0M|Ryzen AI|Phoenix|Hawk Point|Strix)\b", name, re.I))


def gpu_backend_value() -> str:
    """Canonical GPU backend (auto | rocm | vulkan | cpu).

    Reads the new `gpu_backend` key, falling back to the legacy `ollama_gpu_backend`
    so older config files keep working. When the user has effectively disabled the GPU
    (`prefer_gpu` = false), this reports `cpu` so every downstream consumer agrees.
    """
    if not bool(_cfg().get("prefer_gpu", True)):
        return "cpu"
    v = str(_cfg().get("gpu_backend") or _cfg().get("ollama_gpu_backend") or "auto").lower().strip()
    return v or "auto"


def set_gpu_backend(value: str) -> None:
    """Set the GPU backend and keep the new + legacy keys in sync (single source of truth)."""
    v = (str(value or "auto").lower().strip()) or "auto"
    _cfg()["gpu_backend"] = v
    _cfg()["ollama_gpu_backend"] = v


def setup_gpu_acceleration() -> dict[str, Any]:
    """Detect AMD GPU + set Ollama acceleration env vars per config. Returns a status dict."""
    backend = gpu_backend_value()
    gpu_name = _detect_amd_gpu_name()
    applied: dict[str, str] = {}
    recommended: dict[str, str] = {}

    def _set(key: str, val: str) -> None:
        recommended[key] = val
        if _cfg().get("gpu_apply_env", True):
            os.environ[key] = val
            applied[key] = val

    is_ryzen_igpu = bool(gpu_name and _is_ryzen_ai_igpu(gpu_name))
    amd_present = bool(gpu_name)  # _detect_amd_gpu_name() only returns AMD GPUs

    if backend == "cpu":
        _set("OLLAMA_LLM_LIBRARY", "cpu")
    else:
        # OLLAMA_IGPU_ENABLE=1 is the must-have flag for AMD Ryzen AI iGPUs (Radeon 880M/890M,
        # gfx1150/RDNA3.5): without it Ollama discovers the integrated GPU then DROPS it
        # ("dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1") and silently runs on CPU.
        # Harmless on other AMD GPUs; gated to AMD / explicit AMD backends so NVIDIA/Intel systems
        # (which use CUDA/other defaults) are left untouched for backward compatibility.
        if amd_present or backend in ("rocm", "vulkan"):
            _set("OLLAMA_IGPU_ENABLE", "1")
        # Flash attention helps on all backends that support it.
        if bool(_cfg().get("ollama_flash_attention", True)):
            _set("OLLAMA_FLASH_ATTENTION", "1")

        hsa = str(_cfg().get("hsa_override_gfx_version", "auto")).strip()
        hsa_explicit = hsa.lower() != "auto" and bool(hsa)

        if is_ryzen_igpu and backend == "auto":
            # PREFERRED default for Ryzen AI 300-series (e.g. HX 370 + Radeon 890M, gfx1150). After
            # extensive testing the most reliable config is simply OLLAMA_IGPU_ENABLE=1 + flash
            # attention, letting Ollama use native ROCm gfx1150 via its bundled libs. We intentionally
            # DO NOT auto-apply HSA_OVERRIDE_GFX_VERSION (forces gfx1102 — harmful here) or
            # OLLAMA_VULKAN (was inconsistent). Power users can still force either via the GPU backend
            # dropdown or an explicit hsa_override_gfx_version in config.
            if hsa_explicit:
                _set("HSA_OVERRIDE_GFX_VERSION", hsa)
        else:
            # Backward-compatible path for non-Ryzen hardware and explicit backend choices.
            if backend in ("auto", "vulkan"):
                _set("OLLAMA_VULKAN", "1")
            if hsa_explicit:
                _set("HSA_OVERRIDE_GFX_VERSION", hsa)
            elif is_ryzen_igpu and backend == "rocm":
                # Explicit ROCm on a Ryzen iGPU: keep the legacy gfx override as a fallback option.
                _set("HSA_OVERRIDE_GFX_VERSION", _RYZEN_AI_HSA_DEFAULT)

    status = {
        "backend": backend,
        "gpu_name": gpu_name or "(no AMD GPU detected)",
        "applied": applied,
        "recommended": recommended,
        "rocm_smi": bool(shutil.which("rocm-smi")),
        "radeontop": bool(shutil.which("radeontop")),
        "vulkaninfo": bool(shutil.which("vulkaninfo")),
        "rocminfo": bool(shutil.which("rocminfo")),
    }
    if applied:
        log.info("GPU accel (%s) env set: %s", backend, ", ".join(f"{k}={v}" for k, v in applied.items()))
    else:
        log.info("GPU accel: backend=%s, no env applied (gpu_apply_env off or CPU).", backend)
    return status


def _ollama_ps_gpu() -> str:
    """Parse `ollama ps` to report how loaded models are running (GPU/CPU split). '' if none."""
    exe = shutil.which("ollama")
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "ps"], capture_output=True, text=True, timeout=4).stdout
    except Exception:  # noqa: BLE001
        return ""
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return ""
    # Header has a PROCESSOR column (e.g. "100% GPU" / "100% CPU" / "48%/52% CPU/GPU").
    proc = [re.search(r"(\d+%[^\s].*?(?:GPU|CPU)[^\s]*)", ln) for ln in lines[1:]]
    tags = [m.group(1) for m in proc if m]
    return "; ".join(tags) if tags else lines[1].strip()


def _rocm_smi_util() -> str:
    """GPU utilization % via rocm-smi; '' if unavailable."""
    exe = shutil.which("rocm-smi")
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "--showuse"], capture_output=True, text=True, timeout=4).stdout
    except Exception:  # noqa: BLE001
        return ""
    m = re.search(r"GPU use \(%\)\s*:?\s*(\d+)", out)
    return f"{m.group(1)}%" if m else ""


def get_gpu_status(*, ttl: float = 5.0) -> dict[str, Any]:
    """Cached GPU/acceleration snapshot for status bar, /gpu status, and Setup tab."""
    global _gpu_status_cache, _gpu_status_cache_at
    now = time.time()
    if _gpu_status_cache and (now - _gpu_status_cache_at) < ttl:
        return _gpu_status_cache
    backend = gpu_backend_value()
    ps = _ollama_ps_gpu()
    mode, mode_detail = gpu_effective_mode(ps)
    snap = {
        "backend": backend,
        "prefer_gpu": bool(_cfg().get("prefer_gpu", True)),
        "gpu_layers": int(_cfg().get("gpu_layers", -1)),
        "gpu_name": _detect_amd_gpu_name() or "(no AMD GPU detected)",
        "ollama_ps": ps,
        "util": _rocm_smi_util(),
        "flash": bool(_cfg().get("ollama_flash_attention", True)),
        "hsa": os.environ.get("HSA_OVERRIDE_GFX_VERSION", ""),
        "vulkan": os.environ.get("OLLAMA_VULKAN", ""),
        "mode": mode,                # gpu | cpu | unknown
        "mode_detail": mode_detail,  # human-readable explanation
    }
    _gpu_status_cache, _gpu_status_cache_at = snap, now
    return snap


def gpu_effective_mode(ps: Optional[str] = None) -> tuple[str, str]:
    """Resolve where models are ACTUALLY running, from config + live `ollama ps`.

    Returns (mode, detail):
      • 'cpu'     — CPU-only (configured, or GPU unavailable and Ollama fell back)
      • 'gpu'     — at least partly offloaded to the GPU
      • 'unknown' — GPU preferred but no model is loaded yet (can't tell)
    This is the basis for the Setup/`/gpu` "GPU active" vs "→ CPU" line. The fallback is
    automatic: Ollama silently runs on CPU when the GPU can't be used, and we surface that.
    """
    backend = gpu_backend_value()
    if backend == "cpu":
        return "cpu", "CPU-only (configured)"
    if ps is None:
        ps = _ollama_ps_gpu()
    if ps:
        has_gpu = "GPU" in ps
        has_cpu = "CPU" in ps
        if has_gpu and has_cpu:
            return "gpu", f"GPU+CPU split ({ps})"
        if has_gpu:
            return "gpu", f"GPU active ({ps})"
        if has_cpu:
            # GPU was preferred but Ollama is running on CPU → GPU unavailable / not offloading.
            return "cpu", f"running on CPU ({ps}) — GPU unavailable, fell back automatically"
    return "unknown", "GPU preferred — load a model (send a message) to confirm placement"


def gpu_status_text() -> str:
    """Human-readable GPU/acceleration report for `/gpu status` and the Setup tab."""
    s = get_gpu_status()
    recommended = setup_gpu_acceleration().get("recommended", {})
    mode_icon = {"gpu": "🟢", "cpu": "🟡", "unknown": "⚪"}.get(s.get("mode", "unknown"), "⚪")
    layers = s.get("gpu_layers", -1)
    layers_txt = "all that fit" if layers < 0 else ("CPU-only" if layers == 0 else f"{layers} layers")
    lines = [
        "## GPU / acceleration",
        f"- **Mode:** {mode_icon} {s.get('mode_detail', '')}",
        f"- **GPU:** {s['gpu_name']}",
        f"- **Backend:** `{s['backend']}`  ·  Prefer GPU: {'yes' if s.get('prefer_gpu', True) else 'no'}  ·  "
        f"Offload: {layers_txt}  ·  Flash attention: {'on' if s['flash'] else 'off'}",
    ]
    if s.get("hsa"):
        lines.append(f"- **HSA override:** `{s['hsa']}` (Ryzen AI iGPU)")
    if s.get("vulkan"):
        lines.append("- **Vulkan:** enabled")
    if s.get("ollama_ps"):
        lines.append(f"- **Loaded model placement:** {s['ollama_ps']}")
    else:
        lines.append("- **Loaded model placement:** (no model loaded — send a message first)")
    if s.get("util"):
        lines.append(f"- **GPU utilization:** {s['util']}")
    tools = []
    for name, key in (("rocm-smi", "rocm-smi"), ("radeontop", "radeontop"), ("vulkaninfo", "vulkaninfo")):
        tools.append(f"{name} {'✓' if shutil.which(key) else '○'}")
    lines.append(f"- **Tools:** {' · '.join(tools)}")
    if recommended:
        env_block = " ".join(f"{k}={v}" for k, v in recommended.items())
        lines.append("")
        lines.append("**To apply on the Ollama server, restart it with:**")
        lines.append(f"```bash\n{env_block} ollama serve\n```")
    if not shutil.which("rocm-smi") and "no AMD GPU" not in s["gpu_name"]:
        lines.append("")
        lines.append(
            "_ROCm tools not found. For full AMD acceleration: "
            "`sudo apt install rocm-smi radeontop` or run amdgpu-install (see Menu → Status)._"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-healing GPU recovery (Radeon 890M / gfx1150 on Ubuntu)
#
# Root cause on this machine: Ollama is started by systemd (`ollama.service`) with its OWN, minimal
# environment, so the OLLAMA_VULKAN / HSA_OVERRIDE / flash-attention vars Primus exports are ignored
# and models silently load at 100% CPU. The 890M's ROCm path is also flaky and falls back to CPU.
#
# Recovery strategy (precise + safe + non-blocking):
#   • PREFERRED — if a purpose-built `ollama-gpu.service` exists (a unit that already carries the
#     Vulkan env, normally the user unit ~/.config/systemd/user/ollama-gpu.service), just
#     `systemctl --user restart` it and verify. No sudo, no manual process management — cleanest.
#   • FALLBACK — otherwise stop the stock `ollama.service` (so it stops ignoring our env and frees
#     the port; may need sudo), precisely terminate ONLY `ollama serve` processes (psutil — never
#     Primus, never clients), then launch our own `ollama serve` with the Vulkan env block.
#   • Force-load the model via the HTTP API (bounded, never hangs on a TTY) and confirm GPU via
#     `ollama ps`. Retry with backoff up to max_attempts.
#
# The whole loop runs in a background daemon thread (recover_ollama_gpu_threaded); the UI handler
# only drains a queue, so it can NEVER block or crash the Gradio/Primus process. Nothing here raises.
# ---------------------------------------------------------------------------

# THE decisive flag for this hardware. Evidence (Ollama 0.30.7 server log on the Radeon 890M):
#   "dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1" ... type=iGPU compute=gfx1150
# Ollama discovers the 890M (both ROCm gfx1150 and Vulkan RADV) but DROPS it because it's integrated,
# unless OLLAMA_IGPU_ENABLE=1 is set. With it set, `ollama ps` shows 100% GPU (29/29 layers offloaded).
# Also verified: HSA_OVERRIDE_GFX_VERSION=11.0.2 is HARMFUL here — it forces gfx1102 and is unneeded
# because Ollama 0.30.7 supports native gfx1150 via its bundled rocm_v7_2 libs.
_IGPU_BASE: dict[str, str] = {
    "OLLAMA_IGPU_ENABLE": "1",     # ← the fix: stop Ollama from dropping the integrated GPU
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KEEP_ALIVE": "30m",
}

# Candidate env blocks tried in order on the manual `ollama serve` fallback. Block #1 (native ROCm +
# iGPU enable, NO HSA override) is the proven winner on this machine; the Vulkan and HSA variants are
# kept as fallbacks for other driver / ollama-build combinations. Every block sets OLLAMA_IGPU_ENABLE.
OLLAMA_ENV_CANDIDATES: list[tuple[str, dict[str, str]]] = [
    ("rocm-native (iGPU enable, no HSA)", dict(_IGPU_BASE)),
    ("vulkan (iGPU enable)", {**_IGPU_BASE, "OLLAMA_VULKAN": "1", "GGML_VK_VISIBLE_DEVICES": "0"}),
    ("rocm + HSA gfx1102 override", {**_IGPU_BASE, "HSA_OVERRIDE_GFX_VERSION": "11.0.2"}),
]

# Default env for a plain start_ollama_serve() — the proven-best block (#1).
OLLAMA_VULKAN_ENV: dict[str, str] = dict(OLLAMA_ENV_CANDIDATES[0][1])

_OLLAMA_PORT = 11434
# A user-provided unit that bakes in the GPU env; preferred over manual serve if present.
_GPU_UNIT = "ollama-gpu.service"
_STOCK_UNIT = "ollama.service"


def _ollama_exe() -> str:
    """Resolve the ollama binary (falls back to the bare name so PATH lookup still applies)."""
    return shutil.which("ollama") or "ollama"


def _quiet(cmd: list[str], timeout: float = 15.0) -> subprocess.CompletedProcess | None:
    """Run a command, swallow every failure, return the CompletedProcess or None. Never raises."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001 - best-effort process control on a laptop
        return None


def _port_busy(port: int = _OLLAMA_PORT) -> bool:
    """True if something is listening on 127.0.0.1:port (i.e. an Ollama server is up)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def _pid_alive(pid: int) -> bool:
    """True if a PID is still alive (signal 0 probe)."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _server_ready(exe: str) -> bool:
    """True once `ollama ps` succeeds (the server is up and answering)."""
    r = _quiet([exe, "ps"], timeout=8)
    return bool(r and r.returncode == 0)


def _wait_ready(exe: str, timeout: float = 25.0) -> bool:
    """Poll until the Ollama server answers `ollama ps`, or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_ready(exe):
            return True
        time.sleep(0.7)
    return False


def _wait_port_free(timeout: float = 8.0) -> bool:
    """Wait until port 11434 is free (no server listening)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _port_busy():
            return True
        time.sleep(0.4)
    return not _port_busy()


def _systemd_unit_state(unit: str) -> tuple[bool, bool]:
    """Return (exists, active) for a systemd unit. (False, False) if systemctl is unavailable."""
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False, False
    active = False
    a = _quiet([systemctl, "is-active", unit], timeout=6)
    if a and a.stdout.strip() == "active":
        active = True
    exists = active
    if not exists:
        lf = _quiet([systemctl, "list-unit-files", unit], timeout=6)
        if lf and unit in (lf.stdout or ""):
            exists = True
    return exists, active


def _systemctl(action: str, unit: str, *, user: bool = False) -> tuple[bool, str]:
    """Run `systemctl [--user] <action> <unit>`. Returns (ok, detail).

    User-scope units (``systemctl --user``) NEVER need sudo — that's why the recovery prefers a
    user ``ollama-gpu.service``. System units need root, so we try a direct call first then a
    non-interactive ``sudo -n`` (which fails cleanly without passwordless sudo, so the caller can
    surface guidance instead of hanging on a password prompt).
    """
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False, "systemctl not found"
    if user:
        r = _quiet([systemctl, "--user", action, unit], timeout=25)
        if r and r.returncode == 0:
            return True, f"systemctl --user {action} {unit} → ok"
        return False, f"systemctl --user {action} {unit} → failed"
    for cmd in ([systemctl, action, unit], ["sudo", "-n", systemctl, action, unit]):
        r = _quiet(cmd, timeout=25)
        if r and r.returncode == 0:
            return True, f"systemctl {action} {unit} → ok"
    return False, f"systemctl {action} {unit} → failed (passwordless sudo not available?)"


def _gpu_unit_scope() -> Optional[str]:
    """Where the purpose-built ``ollama-gpu.service`` lives: 'user', 'system', or None if absent.

    Checks the documented user unit file first
    (``~/.config/systemd/user/ollama-gpu.service``), then falls back to ``systemctl --user cat`` and
    ``systemctl cat`` so a unit installed in either scope is detected. The returned scope tells the
    recovery whether to drive it with ``systemctl --user`` (no sudo) or system ``systemctl``.
    """
    if (Path.home() / ".config" / "systemd" / "user" / _GPU_UNIT).exists():
        return "user"
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return None
    r = _quiet([systemctl, "--user", "cat", _GPU_UNIT], timeout=6)
    if r and r.returncode == 0:
        return "user"
    r = _quiet([systemctl, "cat", _GPU_UNIT], timeout=6)
    if r and r.returncode == 0:
        return "system"
    return None


def _ollama_serve_pids() -> list[int]:
    """PIDs of ACTUAL `ollama serve` server processes only.

    Precise by design: matches the ollama binary running with a `serve` argument and never returns
    our own PID, Primus, or transient `ollama run`/`ollama pull` clients. Prefers psutil; falls back
    to a tightly-filtered ``pgrep -f "ollama serve"`` when psutil is unavailable.
    """
    me = os.getpid()
    pids: list[int] = []
    if psutil is not None:
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if p.pid == me:
                    continue
                cmdline = p.info.get("cmdline") or []
                if len(cmdline) < 2:
                    continue
                exe0 = os.path.basename(str(cmdline[0])).lower()
                name = (p.info.get("name") or "").lower()
                is_ollama = exe0 == "ollama" or name == "ollama"
                if is_ollama and "serve" in cmdline[1:]:
                    pids.append(p.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return pids
    # Fallback: exact-phrase pgrep (still avoids a broad `pkill ollama`).
    r = _quiet(["pgrep", "-f", "ollama serve"], timeout=6)
    if r and r.stdout:
        for tok in r.stdout.split():
            if tok.strip().isdigit() and int(tok) != me:
                pids.append(int(tok))
    return pids


def kill_ollama_processes(wait: float = 6.0) -> list[str]:
    """Precisely stop ONLY `ollama serve` processes (graceful TERM, then KILL). Returns log lines.

    This NEVER uses a broad ``pkill ollama`` and never targets the current Python/Gradio process —
    it operates on the exact PIDs returned by :func:`_ollama_serve_pids`. A systemd-managed server
    would respawn after this, so the recovery flow stops the unit via systemctl first.
    """
    out: list[str] = []
    pids = _ollama_serve_pids()
    if not pids:
        out.append("no standalone `ollama serve` process found")
        return out

    if psutil is not None:
        procs: list[Any] = []
        for pid in pids:
            try:
                procs.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                continue
        for p in procs:
            try:
                p.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        _gone, alive = psutil.wait_procs(procs, timeout=max(1.0, wait * 0.6))
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        psutil.wait_procs(alive, timeout=max(1.0, wait * 0.4))
    else:
        for pid in pids:
            _quiet(["kill", "-TERM", str(pid)], timeout=5)
        time.sleep(min(wait * 0.6, 4.0))
        for pid in pids:
            if _pid_alive(pid):
                _quiet(["kill", "-9", str(pid)], timeout=5)

    still = [pid for pid in pids if _pid_alive(pid)]
    if still:
        out.append(f"requested stop of `ollama serve` PIDs {pids} — still alive: {still} (may be systemd-managed)")
    else:
        out.append(f"stopped `ollama serve` PIDs: {', '.join(str(p) for p in pids)}")
    return out


def _serve_log_path() -> Path:
    """Where the background `ollama serve` stdout/stderr is captured (under the data dir)."""
    data_dir = Path(os.environ.get("PRIMUS_DATA_DIR") or (Path.home() / ".primus"))
    log_dir = data_dir / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path("/tmp/ollama_serve.log")  # noqa: S108 - fallback only
    return log_dir / "ollama_serve.log"


def start_ollama_serve(env_block: Optional[dict[str, str]] = None) -> Optional[subprocess.Popen]:
    """Launch `ollama serve` in the background with a GPU env block. None on failure.

    ``env_block`` defaults to the proven-best combo (:data:`OLLAMA_VULKAN_ENV`); the recovery passes
    different candidate blocks while cycling. To avoid stale flags leaking in from Primus' own
    environment, the candidate keys are reset first so each block is applied cleanly. Detached into
    its own session so it keeps running independently of Primus; output goes to the data-dir log.
    """
    exe = _ollama_exe()
    block = dict(env_block) if env_block is not None else dict(OLLAMA_VULKAN_ENV)
    env = os.environ.copy()
    # Clear any GPU flags inherited from Primus' env so the chosen block is authoritative.
    for key in ("OLLAMA_VULKAN", "GGML_VK_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
                "OLLAMA_IGPU_ENABLE", "OLLAMA_LLM_LIBRARY"):
        env.pop(key, None)
    env.update(block)
    try:
        fh = open(_serve_log_path(), "ab")  # noqa: SIM115 - handed to the child process
    except OSError:
        fh = subprocess.DEVNULL  # type: ignore[assignment]
    try:
        return subprocess.Popen(
            [exe, "serve"],
            stdout=fh, stderr=fh, stdin=subprocess.DEVNULL,
            env=env, start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to start `ollama serve`: %s", exc)
        return None


def _model_present(exe: str, model: str) -> bool:
    """True if `model` already appears in `ollama list`."""
    r = _quiet([exe, "list"], timeout=15)
    if not r or r.returncode != 0:
        return False
    base = model.split(":", 1)[0]
    return model in r.stdout or base in r.stdout


def _force_load_model(model: str, timeout: float = 150.0) -> tuple[bool, str]:
    """Force the server to load `model` (placing layers) via the HTTP API. Bounded; never hangs.

    Uses /api/generate with a tiny prompt + keep_alive so the model stays resident for the
    subsequent `ollama ps` check. Far more reliable than driving interactive `ollama run` on a
    non-TTY stdin (which is what previously blocked the recovery).
    """
    url = f"http://127.0.0.1:{_OLLAMA_PORT}/api/generate"
    payload = json.dumps(
        {"model": model, "prompt": "hi", "stream": False, "keep_alive": "30m"}
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return True, f"loaded {model} via API"
    except urllib.error.HTTPError as exc:
        return False, f"API load HTTP {exc.code} for {model}"
    except Exception as exc:  # noqa: BLE001
        return False, f"API load failed for {model}: {exc}"


def _ps_placement(raw: str) -> tuple[str, str]:
    """Classify `ollama ps` PROCESSOR column → (mode, detail).

    mode ∈ {'gpu', 'mixed', 'cpu', 'none'}. Parses the actual PROCESSOR token (e.g. '100% GPU',
    '100% CPU', '48%/52% CPU/GPU') from data rows, so a model NAME containing 'gpu' can't trigger a
    false positive. 'mixed' (partial GPU offload) still counts as a GPU win for the recovery.
    """
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return "none", "no model loaded"
    tokens: list[str] = []
    for ln in lines[1:]:
        m = re.search(r"(\d+%(?:/\d+%)?\s*(?:CPU|GPU)(?:/(?:CPU|GPU))?)", ln)
        if m:
            tokens.append(m.group(1).strip())
    if not tokens:
        return "none", "could not parse PROCESSOR column"
    joined = "; ".join(tokens)
    has_gpu, has_cpu = "GPU" in joined, "CPU" in joined
    if has_gpu and has_cpu:
        return "mixed", joined
    if has_gpu:
        return "gpu", joined
    return "cpu", joined


def _ps_has_gpu(raw: str) -> bool:
    """True if any loaded model is (at least partly) on the GPU."""
    return _ps_placement(raw)[0] in ("gpu", "mixed")


def _ps_raw() -> str:
    """Raw `ollama ps` text ('' on failure)."""
    r = _quiet([_ollama_exe(), "ps"], timeout=10)
    return r.stdout if (r and r.returncode == 0) else ""


def _user_gpu_unit_path() -> Path:
    """Path to the user ollama-gpu.service unit file."""
    return Path.home() / ".config" / "systemd" / "user" / _GPU_UNIT


def ensure_gpu_unit_env() -> tuple[bool, list[str]]:
    """Repair the USER ``ollama-gpu.service`` so it actually offloads to the 890M. (changed, notes).

    The real-world bug on this machine: the unit lacked ``OLLAMA_IGPU_ENABLE=1`` (so Ollama dropped
    the integrated GPU) and carried a harmful ``HSA_OVERRIDE_GFX_VERSION=11.0.2`` (forces gfx1102).
    This adds the enable flag and comments out the override, backs up the old unit, and runs
    ``systemctl --user daemon-reload`` — all without sudo (it's a user unit). Best-effort; idempotent.
    """
    notes: list[str] = []
    path = _user_gpu_unit_path()
    try:
        if not path.exists():
            return False, []
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, [f"could not read {path.name}: {exc}"]

    text = original
    changed = False

    if "OLLAMA_IGPU_ENABLE" not in text:
        new_line = 'Environment="OLLAMA_IGPU_ENABLE=1"\n'
        if "ExecStart=" in text:
            text = text.replace("ExecStart=", new_line + "ExecStart=", 1)
        elif "[Service]" in text:
            text = text.replace("[Service]", "[Service]\n" + new_line, 1)
        else:
            text = text.rstrip() + "\n" + new_line
        changed = True
        notes.append("added OLLAMA_IGPU_ENABLE=1 (stops Ollama dropping the integrated GPU)")

    text_no_hsa = re.sub(
        r'(?m)^\s*Environment="HSA_OVERRIDE_GFX_VERSION=[^"]*"\s*$',
        lambda m: "# " + m.group(0).strip() + "   # disabled: native gfx1150 needs no override",
        text,
    )
    if text_no_hsa != text:
        text = text_no_hsa
        changed = True
        notes.append("commented out HSA_OVERRIDE_GFX_VERSION (forces gfx1102 — harmful on the 890M)")

    if changed:
        try:
            path.with_suffix(path.suffix + ".bak").write_text(original, encoding="utf-8")
            path.write_text(text, encoding="utf-8")
            notes.append(f"backed up old unit → {path.name}.bak")
        except OSError as exc:
            return False, [f"could not write {path.name}: {exc}"]
        systemctl = shutil.which("systemctl")
        if systemctl:
            _quiet([systemctl, "--user", "daemon-reload"], timeout=15)
            notes.append("ran `systemctl --user daemon-reload`")
    return changed, notes


def _dropping_warning(scope: Optional[str]) -> str:
    """Most recent GPU-discovery warning from Ollama logs (esp. 'dropping integrated GPU'). '' if none."""
    blobs: list[str] = []
    journalctl = shutil.which("journalctl")
    if journalctl:
        if scope == "user":
            r = _quiet([journalctl, "--user", "-u", _GPU_UNIT, "--no-pager", "-n", "60"], timeout=8)
        else:
            r = _quiet([journalctl, "-u", _STOCK_UNIT, "--no-pager", "-n", "60"], timeout=8)
        if r and r.stdout:
            blobs.append(r.stdout)
    try:
        p = _serve_log_path()
        if p.exists():
            blobs.append(p.read_text(encoding="utf-8", errors="replace")[-4000:])
    except OSError:
        pass
    needles = ("dropping integrated GPU", "no compatible GPUs", "unsupported", "overrode visible devices")
    hits = [ln.strip() for blob in blobs for ln in blob.splitlines() if any(n in ln for n in needles)]
    return hits[-1] if hits else ""


def recover_ollama_gpu_stream(
    max_attempts: int = 8,
    model: str = "qwen2.5:7b",
) -> Iterator[tuple[str, bool, bool]]:
    """Self-heal Ollama GPU acceleration on the Radeon 890M, yielding live progress.

    Yields ``(cumulative_log, done, success)`` tuples so a UI/queue can show a live, growing log and
    a tool can read the final entry. Precise and defensive: prefers restarting a purpose-built
    ``ollama-gpu.service`` (carries the env), otherwise stops the stock service and runs a fresh
    ``ollama serve`` with the Vulkan env, force-loads ``model`` via the API, and verifies GPU via
    ``ollama ps`` — retrying with backoff. Never raises; never broad-kills; never blocks the caller
    beyond its own ``time.sleep`` backoffs (run it via :func:`recover_ollama_gpu_threaded` for UIs).
    """
    max_attempts = max(1, min(int(max_attempts or 8), 10))
    lines: list[str] = []

    def emit(msg: str, *, done: bool = False, success: bool = False) -> tuple[str, bool, bool]:
        lines.append(msg)
        try:
            log.info("recover_ollama_gpu: %s", msg.strip().splitlines()[0][:200])
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(lines), done, success

    exe = shutil.which("ollama")
    if not exe:
        yield emit("✗ `ollama` is not on PATH — install Ollama first, then retry.", done=True, success=False)
        return

    yield emit(f"▶ Recovering Ollama GPU on the Radeon 890M (gfx1150), up to {max_attempts} attempts.")
    yield emit("   key fix for this hardware: OLLAMA_IGPU_ENABLE=1 (Ollama drops the iGPU without it).")

    gpu_scope = _gpu_unit_scope()  # 'user' | 'system' | None
    manual_mode = gpu_scope is None       # no service → go straight to manual serve
    service_failures = 0                  # GPU-not-confirmed count on the service path
    manual_idx = 0                        # which candidate env block to try next (manual path)

    if gpu_scope:
        scope_flag = "--user " if gpu_scope == "user" else ""
        yield emit(f"   detected {_GPU_UNIT} ({gpu_scope} scope) → preferring `systemctl {scope_flag}restart` (no manual kill).")
        # Repair the user unit so the restart can actually offload (add OLLAMA_IGPU_ENABLE, drop HSA).
        if gpu_scope == "user":
            try:
                changed, notes = ensure_gpu_unit_env()
            except Exception as exc:  # noqa: BLE001
                changed, notes = False, [f"unit check failed: {exc}"]
            for n in notes:
                yield emit(f"     • {n}")
            if changed:
                yield emit("     ✓ repaired ollama-gpu.service environment.")
        _stock_exists0, stock_active0 = _systemd_unit_state(_STOCK_UNIT)
        if stock_active0:
            yield emit(
                f"   ⚠ note: stock {_STOCK_UNIT} is also active and may hold port {_OLLAMA_PORT}. "
                "If recovery keeps showing CPU, disable it once: `sudo systemctl disable --now ollama`."
            )

    # Ensure the model is available once up front (a missing model would fail every attempt).
    if not _model_present(exe, model):
        yield emit(f"   {model} not installed — pulling once (may take a while)…")
        r = _quiet([exe, "pull", model], timeout=1800)
        yield emit(f"   {'✓ pulled ' + model if (r and r.returncode == 0) else '⚠ could not pull ' + model}.")

    for attempt in range(1, max_attempts + 1):
        backoff = min(2 + attempt, 8)
        yield emit(f"\n── Attempt {attempt}/{max_attempts} ──")

        if not manual_mode:
            # Path A (preferred): restart the purpose-built ollama-gpu.service (now carrying the env).
            yield emit(f"   Restarting {_GPU_UNIT} ({gpu_scope} scope)…")
            ok, detail = _systemctl("restart", _GPU_UNIT, user=(gpu_scope == "user"))
            yield emit(f"     • {detail}")
            if not ok:
                service_failures += 1
                yield emit("     ⚠ restart failed.")
                if service_failures >= 2:
                    manual_mode = True
                    yield emit("   → switching to manual `ollama serve` with env cycling.")
                time.sleep(backoff)
                continue
        else:
            # Path B (smart fallback): stop the stock service, kill leftovers, then run our own serve
            # with the next candidate env block (cycled across attempts).
            block_name, block_env = OLLAMA_ENV_CANDIDATES[manual_idx % len(OLLAMA_ENV_CANDIDATES)]
            manual_idx += 1
            yield emit(f"   [manual] env block: {block_name}")
            yield emit(f"            {' '.join(f'{k}={v}' for k, v in block_env.items())}")
            _stock_exists, stock_active = _systemd_unit_state(_STOCK_UNIT)
            if stock_active:
                yield emit(f"   Stopping stock {_STOCK_UNIT} (it ignores our env)…")
                ok, detail = _systemctl("stop", _STOCK_UNIT)
                yield emit(f"     • {detail}")
                if not ok:
                    yield emit(
                        "     ✗ stock ollama.service is active but couldn't be stopped without a password.\n"
                        "       Run once, then click Recover again:  sudo systemctl stop ollama\n"
                        "       (or:  sudo systemctl disable --now ollama).",
                        done=True, success=False,
                    )
                    return
            for line in kill_ollama_processes():
                yield emit(f"     • {line}")
            if not _wait_port_free(8):
                yield emit(f"     ⚠ port {_OLLAMA_PORT} still busy — retrying.")
                time.sleep(backoff)
                continue
            proc = start_ollama_serve(block_env)
            if proc is None:
                yield emit("     ✗ failed to launch serve — retrying.")
                time.sleep(backoff)
                continue
            yield emit(f"     • launched serve (pid {proc.pid})")

        if not _wait_ready(exe, 25):
            yield emit("     ⚠ server didn't answer within 25s — retrying.")
            if not manual_mode:
                service_failures += 1
                if service_failures >= 2:
                    manual_mode = True
                    yield emit("   → switching to manual `ollama serve` with env cycling.")
            time.sleep(backoff)
            continue
        yield emit("     ✓ server is up.")

        yield emit(f"   Force-loading {model} via API to place layers on the GPU…")
        _ok, note = _force_load_model(model, timeout=150)
        yield emit(f"     • {note}")

        raw = _ps_raw()
        mode, detail = _ps_placement(raw)
        if mode in ("gpu", "mixed"):
            bust_status_cache()
            yield emit(f"   ✅ GPU CONFIRMED — {detail}")
            yield emit("\n" + raw.strip(), done=True, success=True)
            return

        # Still on CPU — surface the exact Ollama discovery warning (e.g. "dropping integrated GPU").
        yield emit(f"   ✗ still on CPU ({detail}).")
        warn = _dropping_warning(None if manual_mode else gpu_scope)
        if warn:
            yield emit(f"     ↳ ollama log: {warn}")
        if not manual_mode:
            service_failures += 1
            if service_failures >= 2:
                manual_mode = True
                yield emit("   → service restart didn't yield GPU twice; switching to manual serve + env cycling.")
        yield emit(f"   Retrying in {backoff}s…")
        time.sleep(backoff)

    bust_status_cache()
    last_warn = _dropping_warning(None if manual_mode else gpu_scope)
    tail = f"\n   Last Ollama GPU warning: {last_warn}" if last_warn else ""
    yield emit(
        f"\n✗ Gave up after {max_attempts} attempts — Ollama is still on CPU.{tail}\n"
        "   Things to check: that OLLAMA_IGPU_ENABLE=1 is set, that no stock ollama.service is "
        f"holding the port, and the serve log at `{_serve_log_path()}`. Primus keeps working on CPU.",
        done=True, success=False,
    )


def recover_ollama_gpu_threaded(
    max_attempts: int = 8,
    model: str = "qwen2.5:7b",
) -> "queue.Queue":
    """Run :func:`recover_ollama_gpu_stream` in a background daemon thread.

    Returns a Queue that receives ``(cumulative_log, done, success)`` tuples followed by a ``None``
    sentinel when finished. The heavy work (process control, sleeps, model load) happens off the
    caller's thread, so a UI handler can simply drain the queue and yield — it can never block or
    crash the Gradio/Primus process, and recovery continues even if the request disconnects.
    """
    q: "queue.Queue" = queue.Queue()

    def _worker() -> None:
        try:
            for item in recover_ollama_gpu_stream(max_attempts, model):
                q.put(item)
        except Exception as exc:  # noqa: BLE001 - isolate any failure to this thread
            log.exception("recover_ollama_gpu worker crashed")
            q.put((f"✗ Recovery crashed: {exc}", True, False))
        finally:
            q.put(None)

    threading.Thread(target=_worker, name="ollama-gpu-recover", daemon=True).start()
    return q

