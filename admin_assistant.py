#!/usr/bin/env python3
"""
Primus — a local-first personal Admin Agent for Linux

================================================================================
FIRST-TIME SETUP (copy/paste in terminal)
================================================================================

  # 1) System packages (Ubuntu/Debian)
  sudo apt update && sudo apt install -y \\
    python3 python3-venv git curl wmctrl \\
    chromium-browser || sudo apt install -y chromium \\
    network-manager upower bluetooth bluez

  # Optional but recommended
  sudo apt install -y rclone wmctrl

  # 2) Install Ollama + pull a model
  curl -fsSL https://ollama.com/install.sh | sh
  ollama serve &   # or: systemctl --user enable --now ollama
  ollama pull qwen2.5:7b          # Primus — fast orchestrator (default)
  ollama pull qwen2.5-coder:7b  # Forge — coding / deep technical agent
  # ollama pull llama3.2:3b       # lighter Primus alternative

  # 3) Python environment (from project dir, e.g. ~/ai-workshop)
  cd ~/ai-workshop
  curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv not installed
  uv venv && uv pip install gradio langchain langchain-ollama langchain-community \\
    langgraph langgraph-prebuilt langchain-text-splitters chromadb pydantic pillow pystray psutil \\
    openpyxl python-pptx

  # Knowledge base (Chroma + embeddings + optional document/search extras)
  # Office docs: python-docx (.docx), openpyxl (.xlsx), python-pptx (.pptx), pdfplumber (robust PDF)
  uv pip install pypdf pdfplumber python-docx openpyxl python-pptx duckduckgo-search sentence-transformers
  ollama pull nomic-embed-text
  # Alternative local embed model (fallback): BAAI/bge-small-en-v1.5 via sentence-transformers

  # 4) Verify setup
  uv run python admin_assistant.py --check-deps

  # 5) Install menu launcher + optional login autostart
  uv run python admin_assistant.py --install-desktop
  uv run python admin_assistant.py --install-autostart

  # 6) Launch sound (optional cyberpunk "Primus" voice on startup)
  sudo apt install -y alsa-utils pulseaudio-utils   # aplay / paplay playback
  uv pip install piper-tts pyttsx3                  # Piper (best) + pyttsx3 fallback
  mkdir -p ~/.primus/piper ~/.primus/audio
  cd ~/.primus/piper
  curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx
  curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
  # Toggle in ~/.config/primus/config.json → "play_launch_sound": true|false

================================================================================
LAUNCH MODES
================================================================================

  Simple (local UI only):
    uv run python admin_assistant.py

  Desktop companion (small app window + system tray):
    uv run python admin_assistant.py --browser --tray

  Always-on-top corner assistant (set in UI or config):
    uv run python admin_assistant.py --browser --tray --compact
    # then enable Pin in UI (needs wmctrl)

  Tray-only / background (starts hidden, click tray icon):
    # edit ~/.config/primus/config.json → "start_minimized": true
    uv run python admin_assistant.py --tray

  Login autostart (runs companion mode on boot):
    uv run python admin_assistant.py --install-autostart

  Print full setup guide:
    uv run python admin_assistant.py --setup

  UI-only launch (minimal gradio — Menu → Status guides the rest):
    uv pip install gradio && python admin_assistant.py

================================================================================
PATHS
================================================================================
  Config:  ~/.config/primus/config.json
  Data:    ~/.primus/
  Knowledge (RAG): ~/.primus/knowledge/chroma/  (multi-collection)
  Long-term memory: ~/.primus/memories.json
  Session memory: ~/.primus/session_memory.json
  Chat summary: ~/.primus/chat_summary.json
  Conversation archive: ~/.primus/conversation_archive.json  (cross-session summaries)
  Logs:    ~/.config/primus/primus.log
  Desktop: ~/.local/share/applications/primus.desktop

Dual-model flow (every message):
  Router → unified Memory+RAG retrieval → Primus or Forge graph → post-exchange learning

Retrieval cites: [KB-N] knowledge · [MEM-N] long-term memory · [CONV-N] past conversations

Background tasks: heavy/multi-step requests auto-run in the background · /tasks · /task <id> · /cancel task <id>
KB commands: /kb status · /kb search · /kb learn · /learn · /search · /index_projects
Memory: /memory · /recall · /forget · /approve learn (save web research to KB)
Learning: /learn now (distil session) · /summarize session · /reflect (self-review)
Self-improvement: /self (overview) · /self plan (roadmap) · /self improve · /self patterns · /self lessons · /self log · /self learn <lesson> · /ask code <query> · propose changes via chat · /approve edit · /reject edit
Meta-learning: /metrics (usage stats) · /good [note] · /bad [note] (feedback that tunes routing)
Voice: tap the 🎤 mic to speak — transcribed locally (faster-whisper) and auto-sent. Pick model size in Setup.

================================================================================
ARCHITECTURE (modular package — see README "Migration status")
================================================================================
  Primus is now a package; this file is the HOST module + entrypoint. The heavy subsystems were
  extracted verbatim into primus/ and imported back here so behavior is 100% identical:

    primus/core/      obedience, personality, prompts, guardrails
    primus/config.py  env-overridable paths + DEFAULT_CONFIG + load/save
    primus/utils/     gpu.py (AMD accel) + patches.py (gradio/launch helpers)
    primus/tools/     the full tool system (TerminalTool/GitTool + every @tool + build_tools)
    primus/memory/    LTM + KnowledgeBase/RAG (Chroma) + MemorySystem + metrics
    primus/agents/    routing, LangGraph builders, fast paths, invoke_primus + watchdog,
                      background + scheduled agent managers
    primus/ui/        the entire Gradio UI (theme, tabs, panels, handlers, voice, fallback server)

  This file keeps the shared host surface every module reaches via `_host.<name>` (config glue +
  CFG/model globals, PrimusSession, ChatProjects, chat history, web/shell/path/system helpers, the
  desktop window/tray/sound layer) plus the launch orchestration (parse_args / main). Nothing is
  locked — obedience, personality, prompts, and guardrails all remain freely editable in primus/core/.
  Run unchanged: `uv run python admin_assistant.py`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
import wave
import webbrowser
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, NamedTuple, Optional, TypedDict

# Keep startup logs clean: third-party deprecation/future noise (gradio, pydantic,
# langchain, websockets, etc.) is not actionable for the user and clutters output.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"gradio(\..*)?")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic(\..*)?")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning,ignore::FutureWarning")

# ---------------------------------------------------------------------------
# Lazy bootstrap — UI launches with gradio alone; AI stack loads when available
# ---------------------------------------------------------------------------

HAS_GRADIO = False
HAS_AI_STACK = False
AI_STACK_ERROR = ""
_GRADIO_ERROR = ""

try:
    import gradio as gr

    HAS_GRADIO = True
except ImportError as exc:
    gr = None  # type: ignore[assignment,misc]
    _GRADIO_ERROR = str(exc)

try:
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import BaseTool, tool
    from langchain_ollama import ChatOllama
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel, Field

    HAS_AI_STACK = True
except ImportError as exc:
    AI_STACK_ERROR = str(exc)
    Document = Any  # type: ignore[misc,assignment]
    AIMessage = BaseMessage = HumanMessage = SystemMessage = Any  # type: ignore
    ChatPromptTemplate = Any  # type: ignore
    MemorySaver = END = StateGraph = add_messages = create_agent = Any  # type: ignore

    class BaseTool:  # type: ignore[no-redef]
        name: str = ""
        description: str = ""
        args_schema: Any = None

        def _run(self, *args: Any, **kwargs: Any) -> str:
            return "AI stack unavailable — install packages from Menu → Status."

    def tool(fn: Callable) -> Callable:  # type: ignore[misc]
        return fn

    class ChatOllama:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("AI stack not installed")

        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            return type("Resp", (), {"content": ""})()

    class BaseModel:  # type: ignore[no-redef]
        pass

    class Field:  # type: ignore[no-redef]
        def __class_getitem__(cls, item: Any) -> type:
            return cls

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

# Core directives now live in the primus package (primus/core/). They are imported here
# verbatim so the module-level namespace and runtime behavior are byte-for-byte identical to
# the original single file. Edit them in primus/core/ — nothing is locked. ***CURSOR DO NOT TOUCH***
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primus.core.obedience import GLOBAL_OBEDIENCE  # noqa: E402
from primus.core.identity import (  # noqa: E402
    BUSINESS_CONTEXT_FILE,
    OPERATOR,
    SEED_KNOWLEDGE_FILE,
    load_json_override,
)

# ---------------------------------------------------------------------------
# Configuration
#
# Paths + DEFAULT_CONFIG now live in primus/config.py (env-overridable for cloud/Docker).
# They are imported here so this module's namespace is unchanged. SCRIPT_PATH and the live
# CFG + model globals deliberately stay in THIS file. ***CURSOR DO NOT TOUCH***
# ---------------------------------------------------------------------------

from primus import config  # noqa: E402
from primus.config import (  # noqa: E402
    APP_NAME, APP_ID, APP_DIR, CONFIG_DIR, CONFIG_FILE, LOG_FILE, DESKTOP_DIR,
    AUTOSTART_DIR, LEGACY_DIR, MEMORY_FILE, CHAT_HISTORY_FILE, PROJECTS_FILE,
    PROJECT_HISTORY_DIR, GENERAL_CHAT_ID, TODOS_FILE, NOTES_FILE, TASKS_FILE,
    SETTINGS_FILE, EXPORT_DIR, PLANS_DIR, KNOWLEDGE_DIR, CHROMA_DIR, KB_MANIFEST,
    KB_UPLOADS_DIR, KB_DOWNLOADS_DIR, KB_INTERACTIONS, MEMORIES_FILE, SESSION_MEMORY_FILE,
    CHAT_SUMMARY_FILE, CONVERSATION_ARCHIVE, REFLECTIONS_FILE, METRICS_FILE, WATCHDOG_FILE,
    AUDIO_DIR, PIPER_DIR, LAUNCH_SOUND_WAV, SELF_EDIT_BACKUP_DIR, SELF_IMPROVE_LOG,
    SELF_IMPROVE_MEMORY, BACKGROUND_TASKS_FILE, BACKGROUND_RESULTS_DIR, HOME, DOWNLOADS,
    DEFAULT_CONFIG, DEFAULT_INGEST_PATHS,
)

# Resolved path to THIS entrypoint file — kept in the main module on purpose so self-edit /
# desktop-install logic always targets admin_assistant.py, never primus/config.py.
SCRIPT_PATH = Path(__file__).resolve()

# DEFAULT_CONFIG now lives in primus/config.py (imported above). Edit it there.

def _ensure_ollama_host_env() -> None:
    """Point the ollama Python client at CFG ollama_url (OllamaEmbeddings has no base_url param)."""
    url = (CFG.get("ollama_url", OLLAMA_URL) or "").strip().rstrip("/")
    if url:
        os.environ["OLLAMA_HOST"] = url


# _ensure_localhost_direct moved to primus/utils/patches.py (imported below).


# ---------------------------------------------------------------------------
# Shared utilities now live in primus/utils/ — GPU acceleration (gpu.py) and gradio/launch
# patches (patches.py). Imported back here so every call site is unchanged. The GPU helpers
# read the live CFG (which stays in THIS module and is rebound by init_config) through a
# provider registered just below. ***CURSOR DO NOT TOUCH***
# ---------------------------------------------------------------------------
from primus.utils import gpu as _gpu  # noqa: E402
from primus.utils.gpu import (  # noqa: E402
    gpu_backend_value, set_gpu_backend, setup_gpu_acceleration, get_gpu_status,
    gpu_effective_mode, gpu_status_text, _detect_amd_gpu_name, _is_ryzen_ai_igpu,
    _ollama_ps_gpu, _rocm_smi_util, _RYZEN_AI_HSA_DEFAULT,
)
from primus.utils.patches import (  # noqa: E402
    _ensure_localhost_direct, _patch_gradio_localhost_check, _patch_gradio_schema_bug,
    _find_free_port,
)

# GPU helpers read the live CFG (rebound by init_config) through this provider. The lambda
# resolves CFG at call time, so it always returns the current dict — defined further below.
_gpu.bind_cfg(lambda: CFG)

# Gradio compatibility patches moved to primus/utils/patches.py (imported above).

# (gradio_client schema patch also in primus/utils/patches.py.)

# _find_free_port moved to primus/utils/patches.py (imported above).

KB_COLLECTIONS = (
    "core_business",       # company, consulting, operator context
    "technical",           # code, Linux, dev tooling, pipelines
    "projects",            # operator projects (auto-tagged via operator.json)
    "personal_preferences",  # prefs + structured LTM (kind=ltm)
    "learned",             # manual /learn, web research, interactions
)

PROJECT_INDEX_PATHS = [
    "~/ai-workshop",
    "~/Documents",
    "~/Clients",
    "~/.primus",
]

MEMORY_TAGS = ("business", "technical", "personal", "preferences", "project", "session")

TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "business": ("client", "consulting", "billing", "invoice"),
    "technical": ("python", "linux", "pipeline", "crewai", "ollama", "docker", "git", "rclone", "api"),
    "personal": ("ubuntu", "home"),
    "preferences": ("prefer", "always", "never", "default", "favorite", "like to", "don't use"),
    "project": ("primus",),
}
# Personalize auto-tagging from the operator identity (~/.config/primus/operator.json):
# project names tag as project+business; explicit extra keywords extend personal/business.
TAG_KEYWORDS["project"] += tuple(p.lower() for p in OPERATOR["projects"])
TAG_KEYWORDS["business"] += tuple(p.lower() for p in OPERATOR["projects"])
TAG_KEYWORDS["personal"] += tuple(w.lower() for w in OPERATOR["tag_keywords_personal"])
TAG_KEYWORDS["business"] += tuple(w.lower() for w in OPERATOR["tag_keywords_business"])

# Runtime config (populated by init_config)
CFG: dict[str, Any] = DEFAULT_CONFIG.copy()
CFG["script_path"] = str(SCRIPT_PATH)  # SCRIPT_PATH owns the entrypoint path (top of file)
DEFAULT_MODEL = DEFAULT_CONFIG["model"]
FORGE_MODEL = DEFAULT_CONFIG["forge_model"]
OLLAMA_URL = DEFAULT_CONFIG["ollama_url"]
SERVER_HOST = DEFAULT_CONFIG["host"]
DEFAULT_PORT = DEFAULT_CONFIG["port"]

SHELL_TIMEOUT_SEC = 180
FIND_MAX_RESULTS = 100
READ_MAX_CHARS = 24_000
MAX_PLAN_STEPS = 8
MAX_STEP_ITERATIONS = 14

# Generic operator facts seeded into long-term memory on first run (and merged by
# sync_core_memory). The repo default is deliberately identity-free; a deployment
# personalizes it WITHOUT editing the repo via ~/.config/primus/business_context.json
# (a JSON list of fact strings). See examples/operator.example.md.
_DEFAULT_BUSINESS_CONTEXT = [
    "The operator is the local user Primus serves; Primus is a fully local, offline agent.",
    "Specialties: AI automation, agents, data pipelines, Linux administration.",
    "Tooling: Ollama, uv, git, LangChain/LangGraph, Gradio.",
    "Workspace: ~/ai-workshop (Primus and agent projects).",
    "Style: concise, actionable, natural/human conversational tone — never robotic.",
]

_override_bc = load_json_override(BUSINESS_CONTEXT_FILE)
DEFAULT_BUSINESS_CONTEXT = (
    [str(f) for f in _override_bc if str(f).strip()]
    if isinstance(_override_bc, list) and _override_bc
    else _DEFAULT_BUSINESS_CONTEXT
)

# CORE_OPERATOR_PROFILE now lives in primus/core/personality.py (imported verbatim; fully editable).
from primus.core.personality import CORE_OPERATOR_PROFILE  # noqa: E402

# Seed documents ingested into Chroma (versioned — re-synced on bump)
SEED_KNOWLEDGE_VERSION = 2

# Generic example seeds — a fresh clone ingests these so retrieval has something to
# chew on. A deployment replaces them WITHOUT editing the repo via
# ~/.config/primus/seed_knowledge.json (JSON list of {"source", "text"}). The version
# is NOT bumped for this swap: existing installs keep their already-synced seeds
# (sync_seed_knowledge only re-runs on a version bump), so live knowledge bases are
# never rewritten by an upgrade.
_SEED_KNOWLEDGE_DEFAULT: list[dict[str, str]] = [
    {
        "source": "primus:seed:example-operator",
        "text": (
            "The operator is the local user Primus serves. Primus is a local-first executive agent: "
            "chat, tools, long-term memory, and a knowledge base, all running on the operator's machine."
        ),
    },
    {
        "source": "primus:seed:example-toolstack",
        "text": (
            "Default toolstack: Linux, Python 3.12 with uv, Ollama local models, LangChain/LangGraph "
            "for agents, Chroma for retrieval, Gradio for the UI. Prefer apt, systemd, and standard "
            "Linux paths."
        ),
    },
    {
        "source": "primus:seed:example-projects",
        "text": (
            "Example project names used in the bundled docs and fixtures: Acme Co. (sample proposal), "
            "the portal rebuild, the inventory tool. Replace them with real projects via "
            "~/.config/primus/operator.json."
        ),
    },
]


def _load_seed_knowledge() -> list[dict[str, str]]:
    override = load_json_override(SEED_KNOWLEDGE_FILE)
    if isinstance(override, list) and all(
        isinstance(d, dict) and d.get("source") and d.get("text") for d in override
    ):
        return [{"source": str(d["source"]), "text": str(d["text"])} for d in override]
    return _SEED_KNOWLEDGE_DEFAULT


SEED_KNOWLEDGE: list[dict[str, str]] = _load_seed_knowledge()

# DEFAULT_INGEST_PATHS now lives in primus/config.py (imported at the top).

TEXT_INGEST_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".odt", ".doc", ".rtf",
    # Office spreadsheets / presentations
    ".xlsx", ".pptx",
    # Plain text / notes
    ".txt", ".text", ".md", ".markdown", ".rst", ".log",
    # Data / config
    ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".xml", ".ini", ".cfg", ".conf",
    # Code
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".rs", ".c", ".cpp",
    ".h", ".sh", ".html", ".css", ".sql",
}

SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".cache",
    "chroma", ".chroma", "site-packages", ".gradio",
    "vault",
}

DEFAULT_PREFERENCES = {
    "tone": "calm, professional, cyberpunk-lite",
    "default_mode": "execute",
    "confirm_dangerous": True,
    "downloads_folder": str(DOWNLOADS),
    "client_folders_root": str(HOME / "Clients"),
    "backup_remote_hint": "onedrive:",
    "primary_workspace": str(HOME / "ai-workshop"),
    "coding_ide": "Cursor",
    "cloud_sync": "rclone → OneDrive (onedrive:)",
    "llm_stack": "Ollama local models",
    "agent_frameworks": "CrewAI, LangChain, LangGraph",
}

# Command safety guardrails now live in primus/core/guardrails.py (imported verbatim; fully editable).
from primus.core.guardrails import (  # noqa: E402
    DANGEROUS_PATTERNS,
    SAFE_COMMAND_LEADERS,
    SAFE_LEADING_RE,
    UNSAFE_IN_SAFE,
)

WORKFLOW_SUGGESTIONS = [
    "Organize my Downloads folder (preview first)",
    "Clean temp files and show space recovered",
    "Full system health check and tuning suggestions",
    "Check WiFi, battery, and lock screen options",
    "Sort client folders under ~/Clients",
    "Backup strategy for home directory to OneDrive via rclone",
    "Troubleshoot why the system feels slow",
    "Show git status in ~/ai-workshop",
    "Search knowledge base for project automations",
    "Index ~/ai-workshop into Primus knowledge base",
]

# ---------------------------------------------------------------------------
# Setup & dependency registry
# ---------------------------------------------------------------------------

SYSTEM_APT_PACKAGES = [
    "python3", "python3-venv", "git", "curl", "wmctrl",
    "network-manager", "upower", "bluetooth", "bluez",
]

OPTIONAL_APT_PACKAGES = [
    "rclone", "chromium-browser", "chromium", "google-chrome-stable",
    "rocm-smi", "radeontop",  # AMD GPU monitoring (Ryzen AI acceleration)
]

PYTHON_PACKAGES = [
    "gradio>=5.0",
    "langchain>=0.3",
    "langchain-core>=0.3",
    "langchain-ollama>=0.3",
    "langchain-community>=0.3",
    "langchain-chroma>=0.2",
    "langgraph>=0.2",
    "langgraph-prebuilt>=0.1.8",
    "langchain-text-splitters>=0.3",
    "chromadb",
    "pydantic>=2",
]

OPTIONAL_PYTHON_PACKAGES = [
    "pystray", "pillow", "psutil", "pypdf", "pdfplumber",
    "python-docx", "openpyxl", "python-pptx",
    "ddgs", "trafilatura", "selenium", "pyautogui",
    "newspaper3k", "beautifulsoup4", "lxml", "feedparser",
    "faster-whisper",
    "sentence-transformers", "langchain-huggingface",
    "piper-tts", "pyttsx3",
    "ruff",  # Forge code verification (verify_python lint pass)
]

RECOMMENDED_MODELS = [
    ("qwen2.5:7b", "Primus orchestrator — fast daily driver (default)"),
    ("qwen2.5-coder:7b", "Forge — coding, debugging, scripts (required for delegation)"),
    ("llama3.2:3b", "Lighter Primus — maximum chat speed"),
    ("qwen2.5:14b", "Heavier Primus — slower, smarter general tasks"),
    ("mistral:7b", "Alternative general model"),
]

LAUNCH_PROFILES = {
    "simple": "{py} {script}",
    "desktop": "{py} {script} --browser --tray",
    "compact_pin": "{py} {script} --browser --tray --compact",
    "tray_background": "{py} {script} --tray",
    "autostart": "{py} {script} --tray --browser",
}

DEPENDENCY_REGISTRY: list[dict[str, Any]] = [
    {
        "id": "ollama",
        "name": "Ollama",
        "required": True,
        "kind": "binary",
        "bin": "ollama",
        "install": "curl -fsSL https://ollama.com/install.sh | sh && ollama serve",
    },
    {
        "id": "wmctrl",
        "name": "wmctrl (pin/hide window)",
        "required": False,
        "kind": "binary",
        "bin": "wmctrl",
        "install": "sudo apt install -y wmctrl",
    },
    {
        "id": "chromium",
        "name": "Chromium/Chrome (app window)",
        "required": False,
        "kind": "any_binary",
        "bins": ["chromium-browser", "chromium", "google-chrome"],
        "install": "sudo apt install -y chromium-browser || sudo apt install -y chromium",
    },
    {
        "id": "rclone",
        "name": "rclone (cloud sync tools)",
        "required": False,
        "kind": "binary",
        "bin": "rclone",
        "install": "sudo apt install -y rclone",
    },
    {
        "id": "nmcli",
        "name": "NetworkManager (WiFi tools)",
        "required": False,
        "kind": "binary",
        "bin": "nmcli",
        "install": "sudo apt install -y network-manager",
    },
    {
        "id": "gradio",
        "name": "gradio",
        "required": True,
        "kind": "python",
        "module": "gradio",
        "install": "uv pip install gradio",
    },
    {
        "id": "langchain",
        "name": "langchain + ollama + langgraph",
        "required": True,
        "kind": "python",
        "module": "langchain",
        "install": (
            "uv pip install -U langchain langchain-core langchain-ollama langchain-community "
            "langgraph langgraph-prebuilt pydantic"
        ),
    },
    {
        "id": "pystray",
        "name": "pystray + pillow (system tray)",
        "required": False,
        "kind": "python",
        "module": "pystray",
        "install": "uv pip install pystray pillow",
    },
    {
        "id": "psutil",
        "name": "psutil (richer system stats)",
        "required": False,
        "kind": "python",
        "module": "psutil",
        "install": "uv pip install psutil",
    },
    {
        "id": "trafilatura",
        "name": "trafilatura (article extraction)",
        "required": False,
        "kind": "python",
        "module": "trafilatura",
        "install": "uv pip install trafilatura",
    },
    {
        "id": "selenium",
        "name": "selenium (Firefox browser fallback)",
        "required": False,
        "kind": "python",
        "module": "selenium",
        "install": "uv pip install selenium",
    },
    {
        "id": "pyautogui",
        "name": "pyautogui (desktop GUI control — needs X11)",
        "required": False,
        "kind": "python",
        "module": "pyautogui",
        "install": "uv pip install pyautogui",
    },
    {
        "id": "chromadb",
        "name": "chromadb + langchain-chroma (local RAG vector store)",
        "required": False,
        "kind": "python",
        "module": "langchain_chroma",
        "install": "uv pip install chromadb langchain-chroma langchain-text-splitters",
    },
    {
        "id": "pydantic",
        "name": "pydantic (tool schemas)",
        "required": True,
        "kind": "python",
        "module": "pydantic",
        "install": "uv pip install pydantic",
    },
    {
        "id": "python_docx",
        "name": "python-docx (Word .docx read/edit)",
        "required": False,
        "kind": "python",
        "module": "docx",
        "install": "uv pip install python-docx",
    },
    {
        "id": "openpyxl",
        "name": "openpyxl (Excel .xlsx read/edit)",
        "required": False,
        "kind": "python",
        "module": "openpyxl",
        "install": "uv pip install openpyxl",
    },
    {
        "id": "python_pptx",
        "name": "python-pptx (PowerPoint .pptx read/edit)",
        "required": False,
        "kind": "python",
        "module": "pptx",
        "install": "uv pip install python-pptx",
    },
    {
        "id": "paplay",
        "name": "paplay/aplay (launch sound playback)",
        "required": False,
        "kind": "any_binary",
        "bins": ["paplay", "aplay"],
        "install": "sudo apt install -y pulseaudio-utils alsa-utils",
    },
    {
        "id": "piper_tts",
        "name": "piper-tts (cyberpunk launch voice)",
        "required": False,
        "kind": "python",
        "module": "piper",
        "install": "uv pip install piper-tts && see docstring §6 for voice model download",
    },
]

PrimusSession_deps_cache: list[dict[str, Any]] = []

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("primus")


def _python_module_ok(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def check_dependencies(refresh: bool = True) -> list[dict[str, Any]]:
    """Check system + Python dependencies; cache for UI."""
    global PrimusSession_deps_cache
    results: list[dict[str, Any]] = []
    for dep in DEPENDENCY_REGISTRY:
        ok = False
        kind = dep.get("kind")
        if kind == "binary":
            ok = bool(shutil.which(dep["bin"]))
        elif kind == "any_binary":
            ok = any(shutil.which(b) for b in dep.get("bins", []))
        elif kind == "python":
            ok = _python_module_ok(dep["module"])
        results.append({**dep, "ok": ok, "status": "OK" if ok else "MISSING"})

    # Runtime Ollama + model checks (not just binary presence)
    ollama = get_ollama_details()
    embed = CFG.get("embedding_model", "nomic-embed-text") if CFG else "nomic-embed-text"
    primus = ollama.get("configured", DEFAULT_MODEL)
    forge = ollama.get("forge_configured", FORGE_MODEL)

    def _embed_ok(models: list[str], want: str) -> bool:
        base = want.split(":")[0]
        return any(m == want or m.startswith(f"{base}:") for m in models)

    runtime_checks = [
        {
            "id": "ollama_server",
            "name": "Ollama server (API reachable)",
            "required": True,
            "kind": "runtime",
            "ok": ollama["reachable"],
            "install": "ollama serve &   # or: systemctl --user enable --now ollama",
        },
        {
            "id": "primus_model",
            "name": f"Primus model ({primus})",
            "required": True,
            "kind": "runtime",
            "ok": ollama.get("model_ready", False),
            "install": f"ollama pull {primus}",
        },
        {
            "id": "forge_model",
            "name": f"Forge model ({forge})",
            "required": False,
            "kind": "runtime",
            "ok": ollama.get("forge_ready", False),
            "install": f"ollama pull {forge}",
        },
        {
            "id": "embedding_model",
            "name": f"Embedding model ({embed})",
            "required": False,
            "kind": "runtime",
            "ok": _embed_ok(ollama.get("models", []), embed) if ollama["reachable"] else False,
            "install": f"ollama pull {embed}",
        },
        {
            "id": "ai_stack",
            "name": "LangChain + LangGraph (agent core)",
            "required": True,
            "kind": "runtime",
            "ok": HAS_AI_STACK,
            "install": (
                "uv pip install -U langchain langchain-core langchain-ollama langchain-community "
                "langgraph langgraph-prebuilt pydantic"
            ),
        },
        {
            "id": "gradio_ui",
            "name": "Gradio (UI framework)",
            "required": True,
            "kind": "runtime",
            "ok": HAS_GRADIO,
            "install": "uv pip install gradio",
        },
    ]
    for item in runtime_checks:
        results.append({**item, "status": "OK" if item["ok"] else "MISSING"})

    if refresh:
        PrimusSession_deps_cache = results
    return results


def get_ollama_details() -> dict[str, Any]:
    """Return Ollama reachability, models, and Primus + Forge model status."""
    import urllib.error
    import urllib.request

    url = CFG.get("ollama_url", OLLAMA_URL).rstrip("/") if CFG else OLLAMA_URL.rstrip("/")
    primus = CFG.get("model", DEFAULT_MODEL) if CFG else DEFAULT_MODEL
    forge = CFG.get("forge_model", FORGE_MODEL) if CFG else FORGE_MODEL
    out: dict[str, Any] = {
        "url": url,
        "reachable": False,
        "models": [],
        "configured": primus,
        "forge_configured": forge,
        "model_ready": False,
        "forge_ready": False,
        "message": "",
    }

    def _model_ok(want: str, models: list[str]) -> bool:
        base = want.split(":")[0]
        return any(m == want or m.startswith(f"{base}:") for m in models)

    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode())
        models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        out["reachable"] = True
        out["models"] = models
        out["model_ready"] = _model_ok(primus, models)
        out["forge_ready"] = _model_ok(forge, models)
        if not models:
            out["message"] = "Ollama running but no models pulled."
        elif not out["model_ready"]:
            out["message"] = f"Primus model `{primus}` missing — ollama pull {primus}"
        elif not out["forge_ready"]:
            out["message"] = f"Primus OK; Forge `{forge}` missing — ollama pull {forge}"
        else:
            out["message"] = f"Ready — Primus `{primus}` · Forge `{forge}`"
    except urllib.error.URLError as exc:
        out["message"] = f"Unreachable: {exc.reason}. Start with: ollama serve"
    except Exception as exc:
        out["message"] = f"Check failed: {exc}"
    return out


def _ollama_post(path: str, payload: dict[str, Any], *, timeout: int = 15) -> Optional[dict]:
    """POST JSON to the Ollama HTTP API. Returns parsed JSON or None on failure."""
    import urllib.error
    import urllib.request

    url = (CFG.get("ollama_url", OLLAMA_URL).rstrip("/") if CFG else OLLAMA_URL.rstrip("/")) + path
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local trusted host)
            raw = resp.read().decode("utf-8", "replace").strip()
        # /api/generate may stream NDJSON; we only need the request to have completed.
        if not raw:
            return {}
        try:
            return json.loads(raw.splitlines()[-1])
        except json.JSONDecodeError:
            return {}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.debug("Ollama POST %s failed: %s", path, exc)
        return None


def ollama_loaded_models() -> list[str]:
    """Models currently resident in memory (Ollama /api/ps). Empty list if none/unreachable."""
    import urllib.error
    import urllib.request

    url = (CFG.get("ollama_url", OLLAMA_URL).rstrip("/") if CFG else OLLAMA_URL.rstrip("/")) + "/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.debug("Ollama /api/ps failed: %s", exc)
        return []


def ollama_unload_model(name: str) -> bool:
    """Evict a model from memory now (keep_alive=0). Falls back to `ollama stop`.

    Returns True if the unload request was accepted. Safe/no-op if `name` is empty.
    """
    if not name:
        return False
    # Preferred: HTTP API with keep_alive=0 → Ollama unloads the model immediately.
    res = _ollama_post("/api/generate", {"model": name, "keep_alive": 0}, timeout=15)
    if res is not None:
        log.info("Ollama: unloaded %s (keep_alive=0)", name)
        return True
    # Fallback: CLI `ollama stop <model>` (newer Ollama releases).
    if shutil.which("ollama"):
        try:
            subprocess.run(
                ["ollama", "stop", name],
                capture_output=True, text=True, timeout=15, check=False,
            )
            log.info("Ollama: stopped %s via CLI", name)
            return True
        except (subprocess.SubprocessError, OSError) as exc:
            log.debug("ollama stop %s failed: %s", name, exc)
    return False


def setup_instructions_text() -> str:
    """Full setup guide for --setup and Menu → Status."""
    apt_line = "sudo apt install -y " + " ".join(SYSTEM_APT_PACKAGES)
    opt_apt = "sudo apt install -y " + " ".join(OPTIONAL_APT_PACKAGES[:3])
    py_line = "uv pip install " + " ".join(PYTHON_PACKAGES + OPTIONAL_PYTHON_PACKAGES)
    models = "\n".join(f"  ollama pull {m}   # {d}" for m, d in RECOMMENDED_MODELS)
    py = shlex.quote(CFG.get("python_path", sys.executable) if CFG else sys.executable)
    script = shlex.quote(CFG.get("script_path", str(SCRIPT_PATH)) if CFG else str(SCRIPT_PATH))
    launches = "\n".join(
        f"  {name}: {cmd.format(py=py, script=script)}"
        for name, cmd in LAUNCH_PROFILES.items()
    )
    return f"""# Primus Setup Guide

## 1. System packages
{apt_line}

Optional:
{opt_apt}

## 2. Ollama + models
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
{models}

## 3. Python packages (in project venv)
cd {SCRIPT_PATH.parent}
uv venv
{py_line}

## 4. Install launcher & autostart
uv run python admin_assistant.py --install-desktop
uv run python admin_assistant.py --install-autostart

## 5. Launch profiles
{launches}

Config: `{CONFIG_FILE}`
Logs:   `{LOG_FILE}`
"""


def setup(print_guide: bool = True) -> int:
    """Run dependency check and print setup guide. Returns exit code (0=all required OK)."""
    check_dependencies()
    missing_required = [d for d in PrimusSession_deps_cache if d["required"] and not d["ok"]]
    if print_guide:
        print(setup_instructions_text())
        print("\n## Dependency status")
        for d in PrimusSession_deps_cache:
            mark = "✓" if d["ok"] else ("✗ REQUIRED" if d["required"] else "○ optional")
            print(f"  {mark} {d['name']}")
            if not d["ok"]:
                print(f"      → {d['install']}")
        ollama = get_ollama_details()
        print(f"\n## Ollama: {ollama['message']}")
        if ollama["models"]:
            print("  Models:", ", ".join(ollama["models"][:8]))
    return 1 if missing_required else 0


def print_startup_diagnostics() -> None:
    """Log warnings for missing deps on startup — UI always loads; chat may be limited."""
    deps = check_dependencies()
    missing_req = [d["name"] for d in deps if d["required"] and not d["ok"]]
    missing_opt = [d["name"] for d in deps if not d["required"] and not d["ok"]]
    ollama = get_ollama_details()

    if missing_req:
        log.warning("Missing required: %s — open Menu → Status in the UI", ", ".join(missing_req))
        print("\n◈ Primus UI is running — chat needs setup:")
        for d in deps:
            if d["required"] and not d["ok"]:
                print(f"  ❌ {d['name']}")
                print(f"     → {d['install']}")
        print("  Open Menu → Status in the UI for copy-paste install commands.\n")
    else:
        print("\n◈ Primus ready — all required components OK.\n")

    if not ollama["reachable"]:
        log.warning("Ollama not reachable: %s", ollama["message"])
    elif not ollama["model_ready"]:
        log.warning("Primus model not ready: %s", ollama["message"])

    if missing_opt:
        log.info("Optional missing: %s", ", ".join(missing_opt))


def _module_available(mod: str) -> bool:
    """True if an importable module is present, without importing it (never raises)."""
    try:
        import importlib.util

        return importlib.util.find_spec(mod) is not None
    except Exception:  # noqa: BLE001
        return False


def print_tool_health() -> None:
    """Print a concise capability/tool status block at startup. Never raises.

    Gives a clear, at-a-glance view of which features are live (models, search,
    extraction, desktop control, system stats, document ingestion) so issues are
    obvious before testing. Optional pieces show ○ with a one-line fix hint.
    """
    try:
        ollama = get_ollama_details()
        models = ollama.get("models", [])

        def model_present(name: str) -> bool:
            if not name:
                return False
            base = name.split(":")[0]
            return any(m == name or m.startswith(f"{base}:") for m in models)

        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        web_ok = _module_available("ddgs") or _module_available("duckduckgo_search")
        checks: list[tuple[bool, str, str]] = [
            (ollama.get("reachable", False), "Ollama server", "" if ollama.get("reachable") else "start: ollama serve"),
            (ollama.get("model_ready", False), f"Primus model · {CFG.get('model', DEFAULT_MODEL)}", f"ollama pull {CFG.get('model', DEFAULT_MODEL)}"),
            (model_present(CFG.get("primus_fast_model", "")), f"Fast chat model · {CFG.get('primus_fast_model', '—')}", f"ollama pull {CFG.get('primus_fast_model', '')}"),
            (ollama.get("forge_ready", False), f"Forge model · {CFG.get('forge_model', FORGE_MODEL)}", f"ollama pull {CFG.get('forge_model', FORGE_MODEL)}"),
            (HAS_AI_STACK, "Agent stack (LangGraph/LangChain)", "see Menu → Status"),
            (_module_available("langchain_chroma"), "Knowledge base (Chroma RAG)", "uv pip install chromadb langchain-chroma"),
            (web_ok, "Web search (ddgs + Bing/Brave fallback)", "uv pip install ddgs"),
            (_module_available("trafilatura"), "Article extraction (trafilatura)", "uv pip install trafilatura"),
            (_module_available("newspaper"), "Article extraction+ (newspaper3k)", "uv pip install newspaper3k"),
            (_module_available("bs4"), "Structured scraping (BeautifulSoup)", "uv pip install beautifulsoup4 lxml"),
            (_module_available("feedparser"), "News / RSS (feedparser)", "uv pip install feedparser"),
            (_module_available("faster_whisper"), "Voice input / STT (faster-whisper)", "uv pip install faster-whisper"),
            (_module_available("pypdf") or _module_available("pdfplumber"), "PDF ingestion (pypdf + pdfplumber)", "uv pip install pypdf pdfplumber"),
            (_module_available("docx"), "Word .docx read/edit (python-docx)", "uv pip install python-docx"),
            (_module_available("openpyxl"), "Excel .xlsx read/edit (openpyxl)", "uv pip install openpyxl"),
            (_module_available("pptx"), "PowerPoint .pptx read/edit (python-pptx)", "uv pip install python-pptx"),
            (_module_available("psutil"), "System monitor (psutil)", "uv pip install psutil"),
            (_module_available("selenium"), "Browser fallback (selenium)", "uv pip install selenium"),
            (_module_available("pyautogui") and has_display, "Desktop control (pyautogui)", "needs X11 session" if not has_display else "uv pip install pyautogui"),
            (bool(shutil.which("xdg-open")), "Browser/open (xdg-open)", "sudo apt install xdg-utils"),
        ]
        lines = ["◈ Tool & capability check"]
        for ok, name, hint in checks:
            sym = "✓" if ok else "○"
            lines.append(f"  {sym} {name}" + (f"   → {hint}" if (not ok and hint) else ""))
        live = sum(1 for ok, _, _ in checks if ok)
        lines.append(f"  ── {live}/{len(checks)} capabilities live (○ = optional, fix hint shown)")
        # GPU / acceleration summary line (AMD Ryzen AI awareness).
        try:
            g = get_gpu_status()
            accel = []
            if g.get("vulkan"):
                accel.append("Vulkan")
            if g.get("hsa"):
                accel.append(f"HSA={g['hsa']}")
            if g.get("flash"):
                accel.append("FlashAttn")
            placement = g.get("ollama_ps") or "no model loaded yet"
            lines.append(
                f"  ◇ GPU: {g.get('gpu_name', '?')} · backend={g.get('backend','auto')}"
                + (f" · {', '.join(accel)}" if accel else "")
                + f" · placement: {placement}"
            )
        except Exception:  # noqa: BLE001
            pass
        print("\n" + "\n".join(lines) + "\n", flush=True)
        # Also persist a one-line summary to the log so daily-use issues are diagnosable later.
        missing = [name for ok, name, _ in checks if not ok]
        log.info("Startup capabilities: %d/%d live%s", live, len(checks),
                 (" — missing: " + "; ".join(missing[:8])) if missing else "")
        # Warn loudly (log) about anything that blocks core chat.
        if not ollama.get("reachable", False):
            log.warning("Ollama is not reachable — chat will be unavailable until `ollama serve` is running.")
        elif not ollama.get("model_ready", False):
            log.warning("Primus model not pulled — run `ollama pull %s`.", CFG.get("model", DEFAULT_MODEL))
    except Exception as exc:  # noqa: BLE001
        log.debug("Tool health print skipped: %s", exc)


def runtime_ready() -> bool:
    """True when chat can invoke agents (AI stack + Ollama + Primus model)."""
    if not HAS_AI_STACK or not HAS_GRADIO:
        return False
    ollama = get_ollama_details()
    return bool(ollama["reachable"] and ollama["model_ready"])


def _agent_unavailable_message() -> str:
    """User-facing message when chat cannot run — points at Menu → Status."""
    deps = check_dependencies(refresh=True)
    missing = [d for d in deps if d["required"] and not d["ok"]]
    ollama = get_ollama_details()
    lines = [
        "**Primus can't chat yet** — open **Menu → Status** to fix missing components.",
        "",
    ]
    if not HAS_GRADIO:
        lines.append(f"- Gradio UI: ❌ `{_GRADIO_ERROR[:100]}`")
    if not HAS_AI_STACK:
        lines.append(f"- AI stack: ❌ `{AI_STACK_ERROR[:120]}`")
    if not ollama["reachable"]:
        lines.append(f"- Ollama: ❌ {ollama['message']}")
    elif not ollama["model_ready"]:
        lines.append(f"- Primus model: ❌ `ollama pull {ollama['configured']}`")
    if missing:
        lines.append("")
        lines.append("**Required fixes:**")
        for d in missing[:6]:
            lines.append(f"- **{d['name']}** → `{d['install']}`")
    lines.append("")
    lines.append("After installing, click **↻ Refresh status** in Setup / Status.")
    return "\n".join(lines)


def set_config_models(
    *,
    primus: Optional[str] = None,
    forge: Optional[str] = None,
    embed: Optional[str] = None,
) -> str:
    """Update config models and rebuild agent graphs when possible."""
    global _agent_graphs, DEFAULT_MODEL, FORGE_MODEL
    if primus:
        CFG["model"] = primus.strip()
        DEFAULT_MODEL = CFG["model"]
    if forge:
        CFG["forge_model"] = forge.strip()
        FORGE_MODEL = CFG["forge_model"]
    if embed:
        CFG["embedding_model"] = embed.strip()
    save_config_file()
    _agent_graphs = {}
    PrimusSession.graphs = {}
    PrimusSession.graph = None
    if HAS_AI_STACK:
        try:
            init_agent_graphs(force=True)
        except Exception as exc:
            return (
                f"Config saved — Primus `{CFG['model']}` · Forge `{CFG.get('forge_model')}` · "
                f"Embed `{CFG.get('embedding_model')}`\n\n"
                f"*(Graph rebuild deferred: {exc})*"
            )
    return (
        f"**Models updated**\n"
        f"- Primus: `{CFG['model']}`\n"
        f"- Forge: `{CFG.get('forge_model', FORGE_MODEL)}`\n"
        f"- Embeddings: `{CFG.get('embedding_model', 'nomic-embed-text')}`\n\n"
        f"Pull if needed: `ollama pull {CFG['model']}` · `ollama pull {CFG.get('forge_model')}`"
    )


def handle_model_slash(raw: str) -> str:
    """Handle /model list|set|forge|embed commands."""
    parts = raw.split()
    ollama = get_ollama_details()
    installed = set(ollama.get("models") or [])

    if len(parts) == 1 or parts[1].lower() == "list":
        lines = [
            "**Configured models**",
            f"- Primus: `{CFG.get('model', DEFAULT_MODEL)}` "
            f"{'✅' if ollama.get('model_ready') else '❌'}",
            f"- Forge: `{CFG.get('forge_model', FORGE_MODEL)}` "
            f"{'✅' if ollama.get('forge_ready') else '❌'}",
            f"- Embeddings: `{CFG.get('embedding_model', 'nomic-embed-text')}`",
            "",
            "**Recommended (installed ✅ / missing ❌):**",
        ]
        for model, desc in RECOMMENDED_MODELS:
            ok = any(m == model or m.startswith(model.split(":")[0] + ":") for m in installed)
            lines.append(f"- {'✅' if ok else '❌'} `{model}` — {desc}")
        if installed:
            lines.append("")
            lines.append("**On disk:** " + ", ".join(f"`{m}`" for m in list(installed)[:12]))
        seq = "on" if bool(CFG.get("sequential_models", True)) else "off"
        try:
            loaded = ollama_loaded_models()
        except Exception:  # noqa: BLE001
            loaded = []
        lines.append("")
        lines.append(
            f"**Sequential policy:** {seq} (one heavy model at a time) · "
            f"**Loaded now:** {', '.join(f'`{m}`' for m in loaded) or '_none_'}"
        )
        lines.append("")
        lines.append(
            "Switch: `/model set qwen2.5:7b` · `/model forge qwen2.5-coder:7b` · "
            "`/model embed nomic-embed-text` · `/model loaded` · `/model sequential on|off`"
        )
        return "\n".join(lines)

    sub = parts[1].lower()
    # Dynamic switching overrides (no model name needed).
    if sub in ("fast", "large", "full", "auto"):
        return DynamicModelManager.set_override("large" if sub == "full" else sub)
    if sub == "loaded":
        try:
            loaded = ollama_loaded_models()
        except Exception as exc:  # noqa: BLE001
            return f"Couldn't query loaded models: {exc}"
        return "**Resident in memory:** " + (", ".join(f"`{m}`" for m in loaded) or "_none_")
    if sub == "sequential" and len(parts) >= 3:
        val = parts[2].lower() in ("on", "true", "1", "yes")
        CFG["sequential_models"] = val
        try:
            save_config_file()
        except Exception:  # noqa: BLE001
            pass
        return f"Sequential model policy **{'on' if val else 'off'}**."
    if sub == "set" and len(parts) >= 3:
        return set_config_models(primus=parts[2])
    if sub == "forge" and len(parts) >= 3:
        return set_config_models(forge=parts[2])
    if sub == "embed" and len(parts) >= 3:
        return set_config_models(embed=parts[2])
    return (
        "**Model commands**\n"
        "- `/model list` — show configured + installed models\n"
        "- `/model set <name>` — set Primus orchestrator model\n"
        "- `/model forge <name>` — set Forge coding model\n"
        "- `/model embed <name>` — set Ollama embedding model\n"
        "- `/model fast` — pin the small fast model for simple turns\n"
        "- `/model large` — pin the full model (no auto-switching)\n"
        "- `/model auto` — dynamic switching (default)\n"
        "- `/model loaded` — show models resident in memory\n"
        "- `/model sequential on|off` — one heavy model at a time (default on)\n"
        f"_Current: {DynamicModelManager.status()}_"
    )


def render_setup_dashboard_html() -> str:
    """Rich HTML status grid for Menu → Status."""
    deps = check_dependencies(refresh=True)
    ollama = get_ollama_details()
    req_ok = all(d["ok"] for d in deps if d["required"])
    ready = runtime_ready()

    badge = "READY" if ready else ("PARTIAL" if req_ok else "SETUP NEEDED")
    badge_cls = "ready" if ready else ("partial" if req_ok else "needs-setup")

    chips = []
    for d in deps:
        icon = "✅" if d["ok"] else ("❌" if d["required"] else "⚪")
        cls = "ok" if d["ok"] else ("bad" if d["required"] else "opt")
        install = "" if d["ok"] else f'<code class="fix-cmd">{d["install"]}</code>'
        chips.append(
            f'<div class="setup-chip {cls}">'
            f'<span class="chip-icon">{icon}</span>'
            f'<span class="chip-name">{d["name"]}</span>{install}</div>'
        )

    forge_st = "online" if ollama.get("forge_ready") else "offline"
    mem_stats = ""
    kb_stats = ""
    try:
        ms = get_memory_system().stats()
        mem_stats = f"{ms['ltm_count']} LTM · {ms['session_count']} session · {ms.get('archive_count', 0)} archive"
    except Exception:
        mem_stats = "unavailable"
    try:
        kb_stats = str(get_kb().stats().get("total_chunks", "?")) + " chunks"
    except Exception:
        kb_stats = "unavailable"

    return f"""
<div class="setup-dashboard">
  <div class="setup-banner {badge_cls}">
    <span class="setup-badge">{badge}</span>
    <span class="setup-msg">{ollama.get("message", "Checking…")}</span>
  </div>
  <div class="setup-metrics">
    <div class="metric"><span>Primus</span><code>{CFG.get("model", DEFAULT_MODEL)}</code></div>
    <div class="metric"><span>Forge</span><code>{CFG.get("forge_model", FORGE_MODEL)}</code>
      <em class="forge-{forge_st}">{forge_st}</em></div>
    <div class="metric"><span>Memory</span>{mem_stats}</div>
    <div class="metric"><span>Knowledge</span>{kb_stats}</div>
  </div>
  <div class="setup-grid">{"".join(chips)}</div>
</div>
"""


def render_health_header_html() -> str:
    """Header strip with health dot + readiness summary."""
    ollama = get_ollama_details()
    ok = runtime_ready()
    health_class = "ok" if ok else ("warn" if ollama["reachable"] else "bad")
    if ok:
        sub = f"Primus `{CFG.get('model', DEFAULT_MODEL).split(':')[0]}` · Forge `{CFG.get('forge_model', FORGE_MODEL).split(':')[0]}` · online"
    elif not HAS_AI_STACK:
        sub = "Setup required — AI packages missing · open Menu → Status"
    elif not ollama["reachable"]:
        sub = "Ollama offline — open Menu → Status"
    else:
        sub = ollama.get("message", "Partial setup — open Menu → Status")
    return (
        f'<div class="primus-header"><h1>◈ PRIMUS</h1>'
        f'<div class="sub"><span class="health-dot {health_class}"></span>{sub}</div></div>'
    )


def render_setup_status_markdown() -> str:
    """Markdown for Menu → Status (companion to HTML dashboard)."""
    deps = check_dependencies(refresh=True)
    ollama = get_ollama_details()
    ok_n = sum(1 for d in deps if d["ok"])
    req_missing = [d for d in deps if d["required"] and not d["ok"]]

    lines = [
        "## System status",
        "",
        f"**Overall:** {'✅ Chat ready' if runtime_ready() else '❌ Setup incomplete — use commands below'}",
        "",
        build_status_bar(),
        "",
        "## Ollama",
    ]
    st = "🟢" if ollama["reachable"] and ollama["model_ready"] else ("🟡" if ollama["reachable"] else "🔴")
    lines.append(f"{st} **{ollama['message']}**")
    lines.append(f"- URL: `{ollama['url']}`")
    lines.append(f"- Configured Primus: `{ollama['configured']}`")
    lines.append(f"- Configured Forge: `{ollama.get('forge_configured', FORGE_MODEL)}`")
    if ollama["models"]:
        lines.append(f"- Installed: {', '.join(f'`{m}`' for m in ollama['models'][:10])}")
    else:
        lines.append("- Installed: *(none)*")
    lines.append("")
    lines.append("### Model switching")
    lines.append(
        "- Chat: `/model list` · `/model set qwen2.5:7b` · `/model forge qwen2.5-coder:7b` · "
        "`/model embed nomic-embed-text`"
    )
    lines.append(f"- Config file: `{CONFIG_FILE}`")
    lines.append("")

    # GPU / acceleration snapshot (AMD Ryzen iGPU friendly) — shows current GPU vs CPU mode.
    try:
        lines.append(gpu_status_text())
        lines.append("")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"*(GPU status unavailable: {exc})*")
        lines.append("")

    lines.append(f"## Dependencies ({ok_n}/{len(deps)} OK)")
    lines.append("| Status | Component | Install |")
    lines.append("|--------|-----------|---------|")
    for d in deps:
        icon = "✅" if d["ok"] else ("❌" if d["required"] else "⚪")
        fix = "—" if d["ok"] else f"`{d['install']}`"
        lines.append(f"| {icon} | {d['name']} | {fix} |")
    lines.append("")

    lines.append("## Quick install commands")
    lines.append("```bash")
    lines.append("sudo apt install -y " + " ".join(SYSTEM_APT_PACKAGES[:6]))
    lines.append("uv pip install " + " ".join(PYTHON_PACKAGES + OPTIONAL_PYTHON_PACKAGES))
    if not ollama["model_ready"] and ollama["reachable"]:
        lines.append(f"ollama pull {ollama['configured']}")
    elif not ollama["reachable"]:
        lines.append("curl -fsSL https://ollama.com/install.sh | sh && ollama serve &")
    for d in req_missing[:3]:
        lines.append(d["install"])
    lines.append("```")
    lines.append("")

    lines.append("## Launch commands")
    lines.append("```bash")
    lines.append("uv run python admin_assistant.py                    # simple")
    lines.append("uv run python admin_assistant.py --browser --tray  # desktop companion")
    lines.append("uv run python admin_assistant.py --install-desktop")
    lines.append("uv run python admin_assistant.py --install-autostart")
    lines.append("```")

    if req_missing:
        lines.append("")
        lines.append("> ⚠ Fix **required** items above before chat will work reliably.")

    lines.append("")
    lines.append("## Knowledge base (RAG)")
    try:
        lines.append(knowledge_status_markdown())
    except Exception as exc:
        lines.append(f"*(unavailable: {exc})*")
    lines.append("")
    lines.append("```bash")
    lines.append("ollama pull nomic-embed-text   # embeddings for /learn /recall")
    lines.append("uv run python admin_assistant.py --index ~/Documents  # bulk ingest")
    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration (~/.config/primus/config.json)
# ---------------------------------------------------------------------------


def setup_file_logging(level: str = "INFO") -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lvl = getattr(logging, level.upper(), logging.INFO)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    fh.setLevel(lvl)
    log.addHandler(fh)
    log.setLevel(lvl)


def load_config_file() -> dict[str, Any]:
    """Thin wrapper — config loading/merge/migration lives in primus/config.py."""
    return config.load_config_file(SCRIPT_PATH, sys.executable)


def save_config_file(cfg: Optional[dict[str, Any]] = None) -> None:
    """Persist config; defaults to the live CFG (which stays in this module)."""
    config.save_config_file(cfg if cfg is not None else CFG)


def init_config(args: Optional[argparse.Namespace] = None) -> dict[str, Any]:
    """Load config, apply CLI overrides, set module globals (CFG rebinding stays here)."""
    global CFG, DEFAULT_MODEL, FORGE_MODEL, OLLAMA_URL, SERVER_HOST, DEFAULT_PORT
    cfg = config.load_config_file(SCRIPT_PATH, sys.executable)
    cfg = config.apply_cli_and_sync(cfg, args)
    CFG = cfg
    DEFAULT_MODEL = cfg["model"]
    FORGE_MODEL = cfg.get("forge_model", "qwen2.5-coder:7b")
    OLLAMA_URL = cfg["ollama_url"]
    SERVER_HOST = cfg["host"]
    DEFAULT_PORT = int(cfg["port"])
    config.save_config_file(cfg)
    return cfg


def load_settings() -> dict[str, Any]:
    """Backward-compatible settings accessor."""
    return {
        "always_on_top": CFG.get("always_on_top", False),
        "execution_mode": CFG.get("execution_mode", "suggest"),
        "compact_mode": CFG.get("compact_mode", False),
    }


def save_settings(settings: dict[str, Any]) -> None:
    for key in ("always_on_top", "execution_mode", "compact_mode"):
        if key in settings:
            CFG[key] = settings[key]
    save_config_file()


def app_url() -> str:
    return f"http://{CFG.get('host', SERVER_HOST)}:{CFG.get('port', DEFAULT_PORT)}"


def browser_url() -> str:
    """URL to open in the browser — forces Gradio's dark theme for the corner widget."""
    return f"{app_url()}/?__theme=dark"


def check_ollama_health() -> tuple[bool, str]:
    """Verify Ollama is reachable; message includes model status."""
    details = get_ollama_details()
    return details["reachable"], details["message"]


def _desktop_exec(profile: str = "desktop") -> str:
    py = shlex.quote(CFG.get("python_path", sys.executable))
    script = shlex.quote(CFG.get("script_path", str(SCRIPT_PATH)))
    tpl = LAUNCH_PROFILES.get(profile, LAUNCH_PROFILES["desktop"])
    return tpl.format(py=py, script=script)


def install_desktop_entry(profile: str = "desktop") -> Path:
    """Write ~/.local/share/applications/primus.desktop"""
    DESKTOP_DIR.mkdir(parents=True, exist_ok=True)
    path = DESKTOP_DIR / "primus.desktop"
    exec_line = _desktop_exec(profile)
    content = f"""[Desktop Entry]
Type=Application
Name=Primus
GenericName=Personal Admin Agent
Comment=Offline AI admin companion — plan, execute, organize
Exec={exec_line}
Path={SCRIPT_PATH.parent}
Icon=utilities-terminal
Terminal=false
Categories=Utility;System;Development;
StartupNotify=true
StartupWMClass=Primus
Keywords=ai;assistant;admin;ollama;linux;
Actions=Setup;

[Desktop Action Setup]
Name=Run Setup Check
Exec={shlex.quote(CFG.get("python_path", sys.executable))} {shlex.quote(str(SCRIPT_PATH))} --check-deps
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    log.info("Desktop entry: %s", path)
    return path


def install_all_launchers() -> list[Path]:
    """Install menu launcher + autostart."""
    paths = [install_desktop_entry("desktop"), install_autostart(True)]
    return paths


def install_autostart(enable: bool = True) -> Path:
    """Write or remove ~/.config/autostart/primus.desktop"""
    AUTOSTART_DIR.mkdir(parents=True, exist_ok=True)
    path = AUTOSTART_DIR / "primus.desktop"
    if not enable:
        if path.exists():
            path.unlink()
            log.info("Autostart removed")
        CFG["autostart"] = False
        save_config_file()
        return path

    py = shlex.quote(CFG.get("python_path", sys.executable))
    script = shlex.quote(CFG.get("script_path", str(SCRIPT_PATH)))
    content = f"""[Desktop Entry]
Type=Application
Name=Primus
Comment=Start Primus admin agent on login
Exec={_desktop_exec("autostart")}
Path={SCRIPT_PATH.parent}
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=3
"""
    path.write_text(content, encoding="utf-8")
    CFG["autostart"] = True
    save_config_file()
    log.info("Autostart entry: %s", path)
    return path


# ---------------------------------------------------------------------------
# Execution modes & session
# ---------------------------------------------------------------------------


class ExecutionMode(str, Enum):
    SUGGEST = "suggest"
    EXECUTE = "execute"


class _TurnCancelled(Exception):
    """Raised inside the invocation path when a turn is halted (user Halt or watchdog)."""


class PrimusSession:
    mode: ExecutionMode = ExecutionMode.EXECUTE
    pending_shell_command: Optional[str] = None
    pending_suggestion: Optional[str] = None
    pending_queue: list[dict[str, Any]] = []
    allow_dangerous_once: bool = False
    command_history: list[str] = []
    history_index: int = -1
    thinking_steps: list[dict[str, Any]] = []
    think_callback: Optional[Callable[[], None]] = None
    window_hidden: bool = False
    tray_icon: Any = None
    last_health: str = ""
    graph: Any = None
    graphs: dict[str, Any] = {}
    pending_kb_learn: list[dict[str, Any]] = []
    pending_edit: Optional[dict[str, Any]] = None  # staged self-edit, applied only via /approve edit
    active_agent: str = "primus"
    active_model: str = ""
    delegation_reason: str = ""
    force_forge_next: bool = False
    current_topic: str = ""  # active conversation thread, for focus/continuity
    active_project_id: str = GENERAL_CHAT_ID  # active project chat (UI/session scoping only)
    last_path: str = "primus"  # which execution path answered (meta-learning latency label)
    last_intent: str = "chat"  # router intent of the last turn (for dynamic model switching)
    # --- Cancellation + per-turn action dedup ---
    cancel_event: "threading.Event" = threading.Event()
    completed_actions: set[str] = set()
    turn_active: bool = False
    turn_started_at: float = 0.0   # epoch when the current turn began (watchdog uses this)
    halt_reason: str = ""          # why the turn was halted (user | watchdog:<sec>)

    @classmethod
    def begin_turn(cls) -> None:
        """Start a fresh user turn: clear cancel flag + the dedup ledger, stamp start time."""
        cls.cancel_event.clear()
        cls.completed_actions = set()
        cls.turn_active = True
        cls.turn_started_at = time.time()
        cls.halt_reason = ""

    @classmethod
    def end_turn(cls) -> None:
        cls.turn_active = False
        cls.turn_started_at = 0.0

    @classmethod
    def request_halt(cls, reason: str = "user") -> None:
        """Signal all running agent work to stop ASAP. `reason` aids logging/recovery."""
        cls.cancel_event.set()
        cls.halt_reason = reason
        log.info("HALT requested (%s)", reason)

    @classmethod
    def is_cancelled(cls) -> bool:
        return cls.cancel_event.is_set()

    @classmethod
    def turn_runtime(cls) -> float:
        """Seconds the current turn has been running (0 if idle)."""
        return (time.time() - cls.turn_started_at) if (cls.turn_active and cls.turn_started_at) else 0.0

    @classmethod
    def action_signature(cls, kind: str, detail: str) -> str:
        norm = re.sub(r"\s+", " ", (detail or "").strip().lower())
        return f"{kind}:{norm}"

    @classmethod
    def already_did(cls, kind: str, detail: str) -> bool:
        """True if this exact action already ran this turn (prevents duplicate side effects)."""
        return cls.action_signature(kind, detail) in cls.completed_actions

    @classmethod
    def mark_done(cls, kind: str, detail: str) -> None:
        cls.completed_actions.add(cls.action_signature(kind, detail))

    @classmethod
    def queue_kb_learn(
        cls,
        content: str,
        *,
        title: str = "",
        source_url: str = "",
        collection: str = "learned",
        importance: float = 0.7,
    ) -> None:
        cls.pending_kb_learn.append(
            {
                "id": uuid.uuid4().hex[:10],
                "title": title or content[:80],
                "content": content[:4000],
                "source_url": source_url,
                "collection": collection,
                "importance": importance,
                "queued_at": _now_iso(),
            }
        )

    @classmethod
    def flush_kb_learn_queue(cls) -> list[str]:
        results: list[str] = []
        for item in cls.pending_kb_learn:
            try:
                src = item.get("source_url") or f"web:{item['id']}"
                msg = get_kb().learn_text(
                    item["content"],
                    source=src,
                    kind="web",
                    collection=item.get("collection", "learned"),
                    importance=float(item.get("importance", 0.7)),
                    tags=["web", "learned"],
                    replace_source=True,
                )
                results.append(msg)
            except Exception as exc:
                results.append(f"Failed {item.get('title', '?')}: {exc}")
        cls.pending_kb_learn = []
        return results

    @classmethod
    def clear_thinking(cls) -> None:
        cls.thinking_steps = []

    @classmethod
    def emit_think(cls, title: str, detail: str = "", status: str = "pending") -> None:
        cls.thinking_steps.append(
            {
                "title": title,
                "detail": detail,
                "status": status,
                "ts": datetime.now().strftime("%H:%M:%S"),
            }
        )
        # Internal reasoning goes to the console log only — never the chat window.
        log.debug("think[%s] %s — %s", status, title, str(detail)[:200])
        if cls.think_callback:
            try:
                cls.think_callback()
            except Exception:
                pass

    @classmethod
    def push_history(cls, text: str) -> None:
        text = text.strip()
        if text and (not cls.command_history or cls.command_history[-1] != text):
            cls.command_history.append(text)
            cls.command_history = cls.command_history[-100:]

    @classmethod
    def prev_command(cls) -> str:
        if not cls.command_history:
            return ""
        if cls.history_index < 0:
            cls.history_index = len(cls.command_history) - 1
        else:
            cls.history_index = max(0, cls.history_index - 1)
        return cls.command_history[cls.history_index]

    @classmethod
    def reset_history_nav(cls) -> None:
        cls.history_index = -1

    @classmethod
    def queue_command(cls, command: str, source: str = "tool", *, risk: str = "review", reasons: Optional[list[str]] = None) -> str:
        is_dangerous = risk == "dangerous" or bool(classify_command(command)[0])
        reasons = reasons or (classify_command(command)[1] if is_dangerous else [])
        entry = {
            "id": str(uuid.uuid4())[:8],
            "command": command,
            "reasons": reasons,
            "dangerous": is_dangerous,
            "risk": risk,
            "source": source,
        }
        cls.pending_queue.append(entry)
        cls.pending_suggestion = command
        if is_dangerous:
            cls.pending_shell_command = command
        try:
            from primus.core.audit import write_audit  # noqa: PLC0415

            write_audit("queued", ok=True, detail=command[:160])
        except Exception:  # noqa: BLE001 — audit must never break the queue
            pass
        preview = f"**Approval needed** ({risk}):\n```bash\n{command}\n```"
        if reasons:
            preview += f"\nFlags: {', '.join(reasons)}"
        preview += (
            "\n\nUse **Execute** (once), **Approve All** (safe queue), or edit the command and re-send."
        )
        return preview


# ---------------------------------------------------------------------------
# Persistence & memory
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to `path` atomically (temp file in the same dir + os.replace).

    A crash or power loss mid-write can otherwise leave a truncated/corrupt JSON file,
    which then fails to parse on the next load (silent data loss for chat history, tasks,
    projects, etc.). os.replace is atomic on the same filesystem, so readers only ever see
    the complete old or complete new file. Raises OSError on failure (callers log/handle).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def migrate_legacy_data() -> None:
    if LEGACY_DIR.exists() and not APP_DIR.exists():
        try:
            shutil.copytree(LEGACY_DIR, APP_DIR)
            log.info("Migrated %s → %s", LEGACY_DIR, APP_DIR)
        except (OSError, shutil.Error) as exc:
            # A partial/failed migration must not crash every caller of ensure_app_dirs().
            # Log and continue — ensure_app_dirs() recreates any missing files below.
            log.warning("Legacy data migration failed (%s) — continuing with fresh dirs", exc)


def ensure_app_dirs() -> None:
    migrate_legacy_data()
    APP_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    KB_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    SELF_EDIT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not KB_MANIFEST.exists():
        KB_MANIFEST.write_text("{}", encoding="utf-8")
    if not KB_INTERACTIONS.exists():
        KB_INTERACTIONS.write_text("[]", encoding="utf-8")
    if not MEMORIES_FILE.exists():
        MEMORIES_FILE.write_text(
            json.dumps({"version": 1, "memories": [], "meta": {"last_consolidation": None}}, indent=2),
            encoding="utf-8",
        )
    if not SESSION_MEMORY_FILE.exists():
        SESSION_MEMORY_FILE.write_text(
            json.dumps({"session_id": "", "started_at": None, "facts": []}, indent=2),
            encoding="utf-8",
        )
    if not CHAT_SUMMARY_FILE.exists():
        CHAT_SUMMARY_FILE.write_text(
            json.dumps({"summary": "", "updated_at": None, "message_count": 0}, indent=2),
            encoding="utf-8",
        )
    if not CONVERSATION_ARCHIVE.exists():
        CONVERSATION_ARCHIVE.write_text("[]", encoding="utf-8")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    PIPER_DIR.mkdir(parents=True, exist_ok=True)

    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(
            json.dumps(
                {
                    "facts": DEFAULT_BUSINESS_CONTEXT.copy(),
                    "preferences": DEFAULT_PREFERENCES.copy(),
                    "past_tasks": [],
                    "updated_at": _now_iso(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if not TASKS_FILE.exists():
        TASKS_FILE.write_text("[]", encoding="utf-8")
    if not TODOS_FILE.exists():
        TODOS_FILE.write_text("# Primus Todos\n\n", encoding="utf-8")
    if not NOTES_FILE.exists():
        NOTES_FILE.write_text("# Primus Notes\n\n", encoding="utf-8")
    if not SETTINGS_FILE.exists() and not CONFIG_FILE.exists():
        pass  # config created by load_config_file
    elif not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(
            json.dumps({"always_on_top": False, "execution_mode": "suggest"}, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Memory system — long-term memory, the Chroma Knowledge Base + RAG, MemorySystem, metrics,
# reflections, and recall helpers now live in primus/memory/ (system.py). Imported back here
# so every call site (tools, graphs, UI, fast paths) is unchanged. Memory reaches this
# module's runtime globals (CFG, models, web helpers) via the live module object.
# ---------------------------------------------------------------------------
from primus.memory import (  # noqa: E402
    KB_CATEGORY_CHOICES,
    KB_CATEGORY_LABELS,
    KB_CATEGORY_TO_COLLECTION,
    KB_PROJECT_CHOICES,
    # Document extractors — read_office_file (and chat attachments) call these via this module.
    _extract_docx_text,
    _extract_pdf_text,
    _extract_pptx_text,
    _extract_xlsx_text,
    build_conversation_focus,
    build_memory_context,
    detect_memory_tags,
    get_kb,
    get_memory_system,
    get_metrics,
    handle_kb_command,
    index_projects,
    ingest_uploaded_files,
    init_knowledge_base,
    knowledge_status_markdown,
    load_document_text,
    load_memory,
    maybe_store_interaction,
    prepare_agent_messages,
    render_kb_dashboard,
    save_memory,
    start_background_index,
)


def load_task_history() -> list[dict[str, Any]]:
    ensure_app_dirs()
    try:
        data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_task_history(tasks: list[dict[str, Any]]) -> None:
    try:
        _atomic_write_text(TASKS_FILE, json.dumps(tasks[-150:], indent=2))
    except (OSError, TypeError, ValueError) as exc:
        # Task history is best-effort telemetry — never let a write failure crash the
        # background learning / post-exchange path that produced it.
        log.warning("Could not save task history: %s", exc)


def log_past_task(task: str, summary: str, outcome: str = "completed") -> None:
    tasks = load_task_history()
    tasks.append(
        {
            "task": task[:500],
            "summary": summary[:2000],
            "outcome": outcome,
            "date": _now_iso(),
        }
    )
    save_task_history(tasks)


def normalize_gradio_status(raw: Any) -> str:
    """Gradio 6 Chatbot metadata.status allows only 'pending' or 'done'."""
    val = str(raw or "done").lower().strip()
    if val in ("pending", "running"):
        return "pending"
    return "done"


def sanitize_chat_message(item: Any) -> Optional[dict[str, Any]]:
    """Normalize one chat message for Gradio messages-mode Chatbot."""
    if not isinstance(item, dict):
        return None
    role = item.get("role")
    if role not in ("user", "assistant"):
        return None
    # Flatten any structured/stringified content into clean text. Naive str() on a
    # list/dict of content blocks is what produced the leaked "{'text': …}" artifacts.
    content = coerce_message_text(item.get("content", ""))
    msg: dict[str, Any] = {"role": role, "content": content}
    meta = item.get("metadata")
    if isinstance(meta, dict) and meta:
        clean_meta: dict[str, Any] = {}
        if meta.get("title") is not None:
            clean_meta["title"] = str(meta.get("title", ""))
        clean_meta["status"] = normalize_gradio_status(meta.get("status", "done"))
        if meta.get("log") is not None:
            clean_meta["log"] = str(meta.get("log", ""))
        msg["metadata"] = clean_meta
    return msg


def sanitize_chat_history(history: Optional[list]) -> list[dict[str, Any]]:
    """Defensive cleanup before load/save/display.

    Always returns a list of Gradio messages-format dicts ({"role", "content"}).
    Tolerates legacy formats: the old tuples format ([user, assistant] pairs) is
    converted to two messages, and anything unrecognized is skipped rather than raising.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(history, list):
        return out
    for item in history:
        # Legacy tuples format: [user_message, assistant_message]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_text, bot_text = item
            if user_text is not None and str(user_text).strip():
                out.append({"role": "user", "content": str(user_text)})
            if bot_text is not None and str(bot_text).strip():
                out.append({"role": "assistant", "content": str(bot_text)})
            continue
        clean = sanitize_chat_message(item)
        if clean:
            out.append(clean)
    return out


# Bounds that keep the saved history (and the initial page payload) small. A single
# huge message (e.g. a big terminal/file dump) embedded in the page config can balloon
# the Gradio startup HTML to tens of MB and make the browser hang on load.
MAX_SAVED_MESSAGES = 120        # messages kept on disk
MAX_MSG_CHARS = 8000            # per-message content cap on disk
MAX_INITIAL_MESSAGES = 40       # messages rendered into the initial page
MAX_INITIAL_MSG_CHARS = 4000    # per-message cap for the initial page render


def _cap_content(text: Any, limit: int) -> str:
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n\n…[truncated {len(s) - limit:,} chars]"


def _cap_messages(messages: list[dict[str, Any]], max_messages: int, max_chars: int) -> list[dict[str, Any]]:
    """Trim to the most recent messages and cap each message's content length."""
    out: list[dict[str, Any]] = []
    for m in messages[-max_messages:]:
        capped = dict(m)
        capped["content"] = _cap_content(m.get("content", ""), max_chars)
        out.append(capped)
    return out


def load_chat_history(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load saved chat as Gradio messages-format dicts; never raises, falls back to [].

    `path` defaults to the General Chat history file; project chats pass their own file
    (see chat_history_path) so each chat keeps a separate transcript.
    """
    path = path or CHAT_HISTORY_FILE
    try:
        ensure_app_dirs()
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            log.warning("Chat history file was not a list (%s) — ignoring", type(raw).__name__)
            return []
        cleaned = sanitize_chat_history(raw)
        capped = _cap_messages(cleaned, MAX_SAVED_MESSAGES, MAX_MSG_CHARS)
        # Persist migration/trim if legacy formats or oversized content were normalized.
        if capped != raw:
            if len(capped) != len(cleaned) or capped != cleaned:
                log.info(
                    "Normalized chat history (%d → %d messages, content capped)",
                    len(raw) if isinstance(raw, list) else 0,
                    len(capped),
                )
            try:
                _atomic_write_text(path, json.dumps(capped, indent=2))
            except OSError as exc:
                log.warning("Could not persist normalized chat history: %s", exc)
        return capped
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load chat history (%s) — starting empty", exc)
        return []


def initial_chat_history() -> list[dict[str, Any]]:
    """Small, capped slice of recent history for the initial page render.

    Keeps the Gradio startup HTML tiny so the browser never hangs loading a huge
    embedded config. The full (still-bounded) history remains on disk.
    """
    try:
        full = load_active_chat()
        return _cap_messages(full, MAX_INITIAL_MESSAGES, MAX_INITIAL_MSG_CHARS)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not prepare initial chat history (%s) — starting empty", exc)
        return []


def save_chat_history(history: list, path: Optional[Path] = None) -> None:
    path = path or CHAT_HISTORY_FILE
    try:
        cleaned = sanitize_chat_history(history)
        capped = _cap_messages(cleaned, MAX_SAVED_MESSAGES, MAX_MSG_CHARS)
        _atomic_write_text(path, json.dumps(capped, indent=2))
    except (OSError, TypeError, ValueError) as exc:
        log.warning("Could not save chat history: %s", exc)


# ---------------------------------------------------------------------------
# Project / subject chat organization (UI + lightweight session scoping)
#
# Pure session/UI layer on top of the existing memory system. A "project chat" only
# NARROWS THE DEFAULT recall scope (via a short retrieval-query hint) for efficiency and
# focus — it never isolates Primus from full memory. General Chat always has full access.
# Core memory classes, Chroma collections, and agent prompts are untouched.
# ---------------------------------------------------------------------------


def _project_tags(name: str, description: str = "") -> list[str]:
    """Derive a few lowercase keyword tags from a project's name + description.

    Used only to bias retrieval toward the project (scope hint) — not for storage logic.
    """
    blob = f"{name} {description}".lower()
    words = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", blob)
    stop = {"the", "and", "for", "with", "project", "chat", "about", "this", "that"}
    out: list[str] = []
    for w in words:
        if w not in stop and w not in out:
            out.append(w)
        if len(out) >= 6:
            break
    return out


class ChatProjects:
    """Registry of chat 'projects' (General Chat + user-created subject chats).

    Lightweight JSON-backed organization layer. Each entry: id, name, description, tags.
    General Chat is a built-in pseudo-project with full memory access.
    """

    _BUILTIN = {
        "id": GENERAL_CHAT_ID,
        "name": "General Chat",
        "description": "Full memory + knowledge access across everything.",
        "tags": [],
        "builtin": True,
    }

    @classmethod
    def _load(cls) -> dict[str, Any]:
        ensure_app_dirs()
        try:
            data = json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("projects"), list):
                data.setdefault("active", GENERAL_CHAT_ID)
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return {"version": 1, "active": GENERAL_CHAT_ID, "projects": []}

    @classmethod
    def _save(cls, data: dict[str, Any]) -> None:
        try:
            ensure_app_dirs()
            _atomic_write_text(PROJECTS_FILE, json.dumps(data, indent=2))
        except (OSError, TypeError, ValueError) as exc:
            log.warning("Could not save projects: %s", exc)

    @classmethod
    def list(cls) -> list[dict[str, Any]]:
        """General Chat first, then user projects (creation order)."""
        return [dict(cls._BUILTIN)] + list(cls._load().get("projects", []))

    @classmethod
    def get(cls, pid: str) -> dict[str, Any]:
        if not pid or pid == GENERAL_CHAT_ID:
            return dict(cls._BUILTIN)
        for p in cls._load().get("projects", []):
            if p.get("id") == pid:
                return p
        return dict(cls._BUILTIN)

    @classmethod
    def active_id(cls) -> str:
        return cls._load().get("active", GENERAL_CHAT_ID)

    @classmethod
    def set_active(cls, pid: str) -> dict[str, Any]:
        data = cls._load()
        valid = {GENERAL_CHAT_ID} | {p.get("id") for p in data.get("projects", [])}
        data["active"] = pid if pid in valid else GENERAL_CHAT_ID
        cls._save(data)
        return cls.get(data["active"])

    @classmethod
    def create(cls, name: str, description: str = "") -> dict[str, Any]:
        name = (name or "").strip()[:60]
        if not name:
            return {}
        data = cls._load()
        pid = "proj-" + uuid.uuid4().hex[:8]
        entry = {
            "id": pid,
            "name": name,
            "description": (description or "").strip()[:500],
            "tags": _project_tags(name, description),
            "builtin": False,
            "created_at": _now_iso(),
        }
        data.setdefault("projects", []).append(entry)
        data["active"] = pid
        cls._save(data)
        return entry

    @classmethod
    def rename(cls, pid: str, name: str) -> bool:
        name = (name or "").strip()[:60]
        if not name or pid == GENERAL_CHAT_ID:
            return False
        data = cls._load()
        for p in data.get("projects", []):
            if p.get("id") == pid:
                p["name"] = name
                p["tags"] = _project_tags(name, p.get("description", ""))
                cls._save(data)
                return True
        return False

    @classmethod
    def set_description(cls, pid: str, description: str) -> bool:
        if pid == GENERAL_CHAT_ID:
            return False
        data = cls._load()
        for p in data.get("projects", []):
            if p.get("id") == pid:
                p["description"] = (description or "").strip()[:500]
                p["tags"] = _project_tags(p.get("name", ""), p["description"])
                cls._save(data)
                return True
        return False

    @classmethod
    def delete(cls, pid: str) -> bool:
        if not pid or pid == GENERAL_CHAT_ID:
            return False
        data = cls._load()
        before = len(data.get("projects", []))
        data["projects"] = [p for p in data.get("projects", []) if p.get("id") != pid]
        if data.get("active") == pid:
            data["active"] = GENERAL_CHAT_ID
        cls._save(data)
        # Remove the project's separate transcript (memory/KB stay intact).
        try:
            chat_history_path(pid).unlink(missing_ok=True)
        except OSError:
            pass
        return len(data.get("projects", [])) < before


def chat_history_path(pid: str) -> Path:
    """History file for a chat: General Chat reuses the legacy file; projects get their own."""
    if not pid or pid == GENERAL_CHAT_ID:
        return CHAT_HISTORY_FILE
    PROJECT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECT_HISTORY_DIR / f"chat_{pid}.json"


def load_active_chat() -> list[dict[str, Any]]:
    return load_chat_history(chat_history_path(ChatProjects.active_id()))


def save_active_chat(history: list) -> None:
    save_chat_history(history, chat_history_path(ChatProjects.active_id()))


def current_scope_hint() -> str:
    """Short retrieval-bias hint for the active project ("" for General Chat = full scope).

    Prepended to the retrieval query only — biases recall toward the project while still
    allowing global matches. Never filters memory out, so total recall remains available.
    """
    pid = ChatProjects.active_id()
    if not pid or pid == GENERAL_CHAT_ID:
        return ""
    p = ChatProjects.get(pid)
    bits = [f"Project focus: {p.get('name', '')}"]
    desc = (p.get("description") or "").strip()
    if desc:
        bits.append(desc)
    tags = ", ".join(p.get("tags") or [])
    if tags:
        bits.append(f"Keywords: {tags}")
    return " — ".join(bits)[:240]



def memory_context_block() -> str:
    """Legacy JSON facts + preferences for agent prompt (condensed)."""
    try:
        return get_memory_system().agent_memory_block()
    except Exception:
        mem = load_memory()
        facts = mem.get("facts") or []
        prefs = mem.get("preferences") or {}
        lines = ["## Persistent memory"]
        lines.extend(f"- {f}" for f in facts[:20])
        if prefs:
            lines.append("## Preferences")
            for k, v in list(prefs.items())[:15]:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)


def save_plan(task: str, plan: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = PLANS_DIR / f"plan_{stamp}.md"
    try:
        PLANS_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, f"# Plan — {stamp}\n\n**Task:** {task}\n\n{plan}\n")
    except OSError as exc:
        log.warning("Could not save plan: %s", exc)
    return str(path)


def parse_plan_steps(plan: str) -> list[str]:
    steps = re.findall(r"^\s*(?:\d+[\.\)]|[-*])\s+(.+)$", plan, re.MULTILINE)
    steps = [s.strip() for s in steps if s.strip()]
    return steps[:MAX_PLAN_STEPS] if steps else [plan.strip() or "Complete the user request."]


# ---------------------------------------------------------------------------
# Agents — the full agent/orchestration system now lives in primus/agents/ (system.py):
# routing, LangGraph builders, fast paths, invoke_primus + watchdog, and the background /
# scheduled agent managers. It is imported back here so every call site (build_ui, slash
# commands, PrimusSession, main) is unchanged. Agents reach this module's live globals
# (CFG, models, tools, memory, prompts) via the host module object, so behavior is identical.
# `_agent_graphs` stays defined here (above) because it is rebound from both sides.
# ---------------------------------------------------------------------------
from primus.agents import (  # noqa: E402
    AGENT_OUTPUTS_DIR,
    BackgroundAgentManager,
    BackgroundTaskManager,
    DynamicModelManager,
    ModelRouter,
    ScheduledTaskManager,
    _BG_TASK_ICONS,
    _WEEKDAYS,
    _fast_model_available,
    _response_path_label,
    _schedule_label,
    acknowledgment_text,
    append_followup,
    build_thought_messages,
    classify_long_task,
    coerce_message_text,
    enforce_sequential_models,
    gradio_history_to_messages,
    handle_meta_instruction,
    init_agent_graphs,
    invoke_primus,
    is_meta_instruction,
    list_agent_outputs,
    make_chat_ollama,
    polish_response,
    render_background_agents_html,
    render_background_tasks_md,
    render_recent_outputs_md,
    render_scheduled_tasks_md,
    start_turn_watchdog,
)

def _http_get(url: str, *, timeout: int = 10) -> str:
    """Simple, polite HTTP GET returning decoded text ('' on any failure). Never raises."""
    import urllib.request

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        log.debug("HTTP GET failed for %s: %s", url[:80], exc)
        return ""


def _soup(html: str) -> Any:
    """BeautifulSoup parser if available, else None. Never raises."""
    try:
        from bs4 import BeautifulSoup  # type: ignore

        try:
            return BeautifulSoup(html, "lxml")
        except Exception:  # noqa: BLE001 — lxml missing → built-in parser
            return BeautifulSoup(html, "html.parser")
    except Exception:  # noqa: BLE001
        return None


def _html_search_fallback(query: str, max_results: int = 4) -> list[dict]:
    """Engine-agnostic HTML search fallback: DuckDuckGo HTML → Bing → Brave.

    Used when the `ddgs` package is unavailable or returns nothing. Pure HTTP + parsing,
    fully headless. Returns a list of {title, body, href}. Empty list if all engines fail.
    """
    import urllib.parse

    q = urllib.parse.quote(query)
    engines = (
        ("duckduckgo", f"https://html.duckduckgo.com/html/?q={q}"),
        ("bing", f"https://www.bing.com/search?q={q}"),
        ("brave", f"https://search.brave.com/search?q={q}"),
    )
    for name, url in engines:
        html = _http_get(url, timeout=10)
        if not html:
            continue
        hits = _parse_search_html(name, html, max_results)
        if hits:
            PrimusSession.emit_think("Web search", f"Fallback engine: {name} ({len(hits)} hits)", "running")
            return hits
    return []


def _parse_search_html(engine: str, html: str, max_results: int) -> list[dict]:
    """Parse a search-results page from a known engine into {title, body, href} dicts."""
    out: list[dict] = []
    soup = _soup(html)
    try:
        if engine == "duckduckgo":
            if soup is not None:
                for res in soup.select(".result, .web-result")[: max_results * 2]:
                    a = res.select_one("a.result__a")
                    snip = res.select_one(".result__snippet")
                    if a and a.get_text(strip=True):
                        out.append({
                            "title": a.get_text(strip=True),
                            "href": a.get("href", ""),
                            "body": snip.get_text(" ", strip=True) if snip else "",
                        })
        elif engine == "bing":
            if soup is not None:
                for li in soup.select("li.b_algo")[: max_results * 2]:
                    a = li.select_one("h2 a")
                    p = li.select_one("p")
                    if a and a.get_text(strip=True):
                        out.append({
                            "title": a.get_text(strip=True),
                            "href": a.get("href", ""),
                            "body": p.get_text(" ", strip=True) if p else "",
                        })
        elif engine == "brave":
            if soup is not None:
                for res in soup.select("div.snippet, .result")[: max_results * 2]:
                    a = res.select_one("a")
                    title_el = res.select_one(".title, .snippet-title")
                    desc = res.select_one(".snippet-description, .description, p")
                    title = (title_el.get_text(strip=True) if title_el else (a.get_text(strip=True) if a else ""))
                    if a and title:
                        out.append({
                            "title": title,
                            "href": a.get("href", ""),
                            "body": desc.get_text(" ", strip=True) if desc else "",
                        })
    except Exception as exc:  # noqa: BLE001
        log.debug("Parse %s failed: %s", engine, exc)

    # Regex safety net when BeautifulSoup is unavailable or selectors miss.
    if not out:
        for m in re.finditer(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', html, re.I | re.S):
            href, raw = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if raw and len(raw) > 12 and not any(b in href for b in ("bing.com", "duckduckgo.com", "brave.com", "microsoft", "/aclk")):
                out.append({"title": raw[:140], "href": href, "body": ""})
            if len(out) >= max_results:
                break

    # De-dup by href and trim.
    seen: set[str] = set()
    deduped: list[dict] = []
    for h in out:
        href = h.get("href", "")
        if href and href not in seen:
            seen.add(href)
            deduped.append(h)
    return deduped[:max_results]


def _run_web_search(query: str, *, max_results: int = 4, queue_kb: bool = True) -> str:
    """Headless DuckDuckGo search (+ Bing/Brave fallback) → concise, source-cited summary.

    Multi-layered & offline-safe: tries the `ddgs` package first, then falls back to HTML
    scraping of DuckDuckGo/Bing/Brave. Returns clean natural text (no JSON), trims each
    snippet, and — when ``queue_kb`` — stages findings for ``/approve learn``. Never opens a
    visible browser; it's pure HTTP, so it stays fast and lightweight.
    """
    if not CFG.get("web_search_enabled", True):
        return "Web search is turned off in config (`web_search_enabled=false`)."
    query = (query or "").strip()
    if not query:
        return "Tell me what you'd like me to search for."

    PrimusSession.emit_think("Web search", f"Searching the web for: {query}", "running")
    # Layer 1 — the `ddgs` package (preferred); legacy `duckduckgo_search` as alias.
    DDGS = None
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore  # legacy name
        except ImportError:
            DDGS = None

    hits: list[dict] = []
    last_exc: Optional[Exception] = None
    if DDGS is not None:
        # The DDG backend occasionally returns an empty page on a cold call — retry briefly.
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    hits = list(ddgs.text(query, max_results=max_results))
                if hits:
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            time.sleep(0.5 * (attempt + 1))

    # Layer 2 — HTML fallback engines (DuckDuckGo HTML → Bing → Brave) when ddgs is
    # missing or comes up empty. Keeps search working even without the ddgs package.
    if not hits:
        try:
            hits = _html_search_fallback(query, max_results)
        except Exception as exc:  # noqa: BLE001
            last_exc = last_exc or exc

    if not hits and last_exc is not None:
        PrimusSession.emit_think("Web search", "Offline / unavailable", "error")
        return (
            "I couldn't reach the web just now — looks like I'm offline or every search service is "
            f"busy ({str(last_exc)[:100]}). I can still help with anything on your system."
        )
    if not hits:
        PrimusSession.emit_think("Web search", "No results", "done")
        return f'I searched the web for "{query}" but didn\'t find anything useful.'

    lines = [f"Here's what I found for **{query}**:", ""]
    for i, hit in enumerate(hits, 1):
        title = (hit.get("title") or "Result").strip()
        body = (hit.get("body") or hit.get("snippet") or "").strip()
        url = (hit.get("href") or hit.get("link") or "").strip()
        snippet = re.sub(r"\s+", " ", body)[:220].strip()
        if snippet and snippet[-1] not in ".!?":
            snippet += "…"
        src = f"\n   ↳ {url}" if url else ""
        lines.append(f"{i}. **{title}** — {snippet}{src}")
        if queue_kb and body:
            PrimusSession.queue_kb_learn(
                f"{title}\n{body}",
                title=title,
                source_url=url,
                collection="learned",
                importance=0.6,
            )
    PrimusSession.emit_think("Web search", f"{len(hits)} result(s)", "done")
    if queue_kb and PrimusSession.pending_kb_learn:
        n = len(PrimusSession.pending_kb_learn)
        lines.append(
            f"\nWant me to save {'these' if n > 1 else 'this'} to your knowledge base? "
            "Say `/approve learn` (or `/reject learn` to skip)."
        )
    return "\n".join(lines)


def web_search(query: str, max_results: int = 5) -> str:
    """Back-compat wrapper around the headless search core."""
    return _run_web_search(query, max_results=max_results, queue_kb=True)



# ---------------------------------------------------------------------------
# Safety & shell
# ---------------------------------------------------------------------------


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (HOME / path).resolve()


def _path_in_home(path: Path) -> bool:
    try:
        path.relative_to(HOME)
        return True
    except ValueError:
        return False


# Safe read-only system locations Primus may list/read/search (never write/execute).
READONLY_SYSTEM_DIRS: list[Path] = [
    Path("/usr/bin"),
    Path("/usr/local/bin"),
    Path("/usr/share/applications"),
    Path("/usr/share/doc"),
    Path("/opt"),
    Path("/snap/bin"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path("/var/lib/snapd/desktop/applications"),
    HOME / ".local/share/applications",
    HOME / ".local/bin",
    HOME / ".var/app",  # flatpak per-user data
]

# Standard desktop-entry locations for "list installed applications".
APPLICATION_DIRS: list[Path] = [
    Path("/usr/share/applications"),
    Path("/usr/local/share/applications"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path("/var/lib/snapd/desktop/applications"),
    HOME / ".local/share/applications",
]


def _lexical_path(raw: str) -> Path:
    """Absolute, normalized path WITHOUT following symlinks (collapses '..' to block traversal)."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = HOME / p
    return Path(os.path.normpath(str(p)))


def _path_in_readonly_zone(path: Path) -> bool:
    for base in READONLY_SYSTEM_DIRS:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            continue
    return False


def _guard_path(
    raw: str,
    *,
    must_exist: bool = False,
    allow_system_read: bool = False,
) -> tuple[Optional[Path], Optional[str]]:
    """Resolve + authorize a path.

    Writes are confined to HOME. Read-only tools may opt into the system read zone
    (allow_system_read=True) to list/read common app + binary locations. Authorization uses
    the lexically-normalized path so legitimate symlinks (e.g. /usr/bin/ls → coreutils) work,
    while '..' traversal is still collapsed and blocked.
    """
    target = _resolve_path(raw)
    lexical = _lexical_path(raw)
    in_home = _path_in_home(target) or _path_in_home(lexical)
    in_ro = allow_system_read and (_path_in_readonly_zone(lexical) or _path_in_readonly_zone(target))
    if not (in_home or in_ro):
        if allow_system_read:
            return None, (
                f"Refusing path outside home and allowed read zones: {lexical}. "
                "I can read ~ plus common system app/binary folders."
            )
        return None, f"Refusing path outside home: {lexical}"
    # Home paths use the symlink-resolved path (legacy behavior); read-zone uses the path as named.
    io_target = target if in_home else lexical
    if must_exist and not io_target.exists():
        return None, f"Not found: {io_target}"
    return io_target, None


def classify_command(command: str) -> tuple[bool, list[str]]:
    reasons = [label for pattern, label in DANGEROUS_PATTERNS if pattern.search(command)]
    return bool(reasons), reasons


def split_shell_pipeline(command: str) -> list[str]:
    """Split on pipes for per-segment safety checks."""
    return [seg.strip() for seg in command.split("|") if seg.strip()]


def split_shell_chain(command: str) -> list[str]:
    """Split on && or ; for chained commands."""
    return [p.strip() for p in re.split(r"\s*&&\s*|\s*;\s*", command.strip()) if p.strip()]


def _is_safe_command_part(part: str) -> bool:
    part = part.strip()
    if not part or UNSAFE_IN_SAFE.search(part):
        return False
    if re.search(r"(?<![<\\])>>?(?!>)", part):
        return False
    for segment in split_shell_pipeline(part):
        if not SAFE_LEADING_RE.match(segment):
            return False
    return True


def assess_command_risk(command: str) -> tuple[str, list[str]]:
    """Return risk tier: safe | review | dangerous."""
    command = command.strip()
    if not command:
        return "review", []
    is_dangerous, reasons = classify_command(command)
    if is_dangerous:
        return "dangerous", reasons
    for part in split_shell_chain(command):
        if not _is_safe_command_part(part):
            return "review", [f"needs review: {part[:60]}"]
    return "safe", []


def suggest_command(command: str, *, risk: str = "review", reasons: Optional[list[str]] = None) -> str:
    return PrimusSession.queue_command(command, source="suggest", risk=risk, reasons=reasons)


def _execute_shell_core(
    command: str,
    *,
    cwd: Optional[Path] = None,
    timeout: int = SHELL_TIMEOUT_SEC,
) -> str:
    if PrimusSession.is_cancelled():
        return "⏹ Halted — command not run."
    PrimusSession.push_history(command)
    # Run in a fresh process group so a timeout can reap the whole tree. With shell=True,
    # subprocess.run's own timeout only kills the shell, leaving spawned children orphaned
    # and consuming CPU/RAM. We manage the group explicitly and SIGKILL it on timeout.
    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(cwd or HOME),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "LANG": "C.UTF-8"},
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        try:
            proc.communicate(timeout=5)  # reap so we don't leak a zombie
        except Exception:  # noqa: BLE001
            pass
        try:
            from primus.core.audit import write_audit  # noqa: PLC0415

            write_audit("executed", ok=False, detail=command[:160])
        except Exception:  # noqa: BLE001
            pass
        return f"Command timed out after {timeout}s (process tree terminated)."
    except OSError as exc:
        _kill_process_tree(proc)
        try:
            from primus.core.audit import write_audit  # noqa: PLC0415

            write_audit("executed", ok=False, detail=command[:160])
        except Exception:  # noqa: BLE001
            pass
        return f"Execution error: {exc}"

    parts = [f"exit_code={returncode}"]
    if stdout and stdout.strip():
        parts.append("STDOUT:\n" + stdout.rstrip())
    if stderr and stderr.strip():
        parts.append("STDERR:\n" + stderr.rstrip())
    output = "\n".join(parts)
    try:
        from primus.core.audit import write_audit  # noqa: PLC0415

        write_audit("executed", ok=returncode == 0, detail=command[:160])
    except Exception:  # noqa: BLE001 — audit must never break the shell
        pass
    return output[:12_000] + ("\n… (truncated)" if len(output) > 12_000 else "")


def _kill_process_tree(proc: "Optional[subprocess.Popen]") -> None:
    """Best-effort SIGKILL of a Popen and its process group (started via start_new_session)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError, AttributeError):
        try:
            proc.kill()
        except OSError:
            pass


def run_shell(
    command: str,
    *,
    cwd: Optional[Path] = None,
    timeout: int = SHELL_TIMEOUT_SEC,
    force: bool = False,
) -> str:
    command = command.strip()
    if not command:
        return "No command provided."

    risk, reasons = assess_command_risk(command)
    auto_safe = bool(CFG.get("auto_execute_safe_commands", True))
    log.info("Shell risk=%s force=%s cmd=%s", risk, force, command[:120])

    if risk == "dangerous" and not force and not PrimusSession.allow_dangerous_once:
        return suggest_command(command, risk="dangerous", reasons=reasons)

    if risk == "safe" and auto_safe and not force:
        header = f"▶ **Auto-ran** (safe read-only):\n```bash\n{command}\n```\n\n"
        PrimusSession.allow_dangerous_once = False
        PrimusSession.pending_shell_command = None
        PrimusSession.pending_suggestion = None
        PrimusSession.pending_queue = [q for q in PrimusSession.pending_queue if q.get("command") != command]
        return header + _execute_shell_core(command, cwd=cwd, timeout=timeout)

    if risk == "review" and not force:
        if PrimusSession.mode == ExecutionMode.SUGGEST:
            return suggest_command(command, risk="review", reasons=reasons)
        header = f"▶ **Running**:\n```bash\n{command}\n```\n\n"
        PrimusSession.allow_dangerous_once = False
        PrimusSession.pending_shell_command = None
        PrimusSession.pending_suggestion = None
        PrimusSession.pending_queue = [q for q in PrimusSession.pending_queue if q.get("command") != command]
        return header + _execute_shell_core(command, cwd=cwd, timeout=timeout)

    PrimusSession.allow_dangerous_once = False
    PrimusSession.pending_shell_command = None
    PrimusSession.pending_suggestion = None
    PrimusSession.pending_queue = [q for q in PrimusSession.pending_queue if q.get("command") != command]
    header = f"▶ **Executed**:\n```bash\n{command}\n```\n\n" if force else ""
    return header + _execute_shell_core(command, cwd=cwd, timeout=timeout)


def run_queued_safe_commands() -> str:
    """Execute all queued safe (non-dangerous) commands."""
    if not PrimusSession.pending_queue:
        return "No commands queued."
    results: list[str] = []
    remaining: list[dict] = []
    for item in PrimusSession.pending_queue:
        cmd = item.get("command", "")
        risk, _ = assess_command_risk(cmd)
        if item.get("dangerous") or risk != "safe":
            remaining.append(item)
            continue
        out = run_shell(cmd, force=True)
        results.append(f"### `{cmd[:80]}`\n{out}")
    PrimusSession.pending_queue = remaining
    if remaining:
        PrimusSession.pending_shell_command = remaining[0].get("command")
        PrimusSession.pending_suggestion = remaining[0].get("command")
    else:
        PrimusSession.pending_shell_command = None
        PrimusSession.pending_suggestion = None
    if not results:
        return "No safe commands in queue — use **Execute** for risky items."
    return "**Approved safe commands:**\n\n" + "\n\n".join(results)


# ---------------------------------------------------------------------------
# System status bar
# ---------------------------------------------------------------------------


def _read_mem_stats() -> tuple[int, int, float]:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        total = avail = 0
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) // 1024
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) // 1024
        used = total - avail
        return used, total, (used / total * 100) if total else 0.0
    except OSError:
        return 0, 0, 0.0


def _read_cpu_pct() -> float:
    try:
        import psutil  # type: ignore
        return psutil.cpu_percent(interval=0.15)
    except ImportError:
        try:
            load1, _, _ = os.getloadavg()
            nproc = max(1, int(subprocess.check_output(["nproc"], text=True, timeout=5).strip() or "1"))
            return min(100.0, load1 / nproc * 100)
        except (OSError, AttributeError, ValueError, subprocess.SubprocessError):
            return 0.0
    except Exception:  # noqa: BLE001 — psutil edge cases must never break the status bar
        return 0.0


def _read_battery() -> str:
    try:
        for bat in Path("/sys/class/power_supply").glob("BAT*"):
            cap = bat / "capacity"
            status = bat / "status"
            if cap.exists():
                pct = cap.read_text(encoding="utf-8").strip()
                st = status.read_text(encoding="utf-8").strip() if status.exists() else "?"
                return f"{pct}% {st}"
    except OSError:
        return "?"
    return "AC"


def _read_rclone_brief() -> str:
    if not shutil.which("rclone"):
        return "rclone:n/a"
    try:
        out = subprocess.check_output(["rclone", "listremotes"], text=True, timeout=6, stderr=subprocess.DEVNULL)
        n = len([r for r in out.splitlines() if r.strip()])
        return f"rclone:{n}"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return "rclone:err"


def build_status_bar() -> str:
    used, total, mem_pct = _read_mem_stats()
    cpu = _read_cpu_pct()
    bat = _read_battery()
    try:
        disk = shutil.disk_usage(HOME)
        disk_pct = disk.used / disk.total * 100 if disk.total else 0
    except OSError:
        disk_pct = 0
    agent = (PrimusSession.active_agent or "primus").upper()
    model = PrimusSession.active_model or CFG.get("model", DEFAULT_MODEL)
    if agent == "PRIMUS" and not PrimusSession.active_model:
        model = CFG.get("model", DEFAULT_MODEL)

    forge_ok = get_ollama_details().get("forge_ready", False)
    forge_tag = "Forge✓" if forge_ok else "Forge—"

    mem_n = kb_n = "?"
    try:
        ms = get_memory_system().stats()
        mem_n = str(ms["ltm_count"])
    except Exception:
        pass
    try:
        kb_n = str(get_kb().stats().get("total_chunks", 0))
    except Exception:
        pass

    chat_st = "●" if runtime_ready() else "○"
    gpu_tag = ""
    try:
        g = get_gpu_status()
        if g.get("util"):
            gpu_tag = f" · GPU {g['util']}"
        elif g.get("ollama_ps") and "GPU" in g["ollama_ps"]:
            gpu_tag = " · GPU✓"
    except Exception:
        pass
    return (
        f"{chat_st} {agent}·{model.split(':')[0]} · {forge_tag} · "
        f"MEM {mem_n} · KB {kb_n} · "
        f"CPU {cpu:.0f}% · RAM {mem_pct:.0f}% · Disk {disk_pct:.0f}%{gpu_tag} · {bat}"
    )


def render_thinking_html() -> str:
    steps = PrimusSession.thinking_steps
    if not steps:
        return '<div class="think-panel empty">Awaiting task…</div>'
    rows = []
    for s in steps:
        icon = {"pending": "◌", "running": "◉", "done": "✓", "error": "✗"}.get(s.get("status", ""), "·")
        detail = s.get("detail", "")
        extra = f'<div class="think-detail">{detail[:300]}</div>' if detail else ""
        rows.append(
            f'<div class="think-step {s.get("status", "")}">'
            f'<span class="think-icon">{icon}</span>'
            f'<span class="think-title">{s.get("title", "")}</span>'
            f'<span class="think-ts">{s.get("ts", "")}</span>{extra}</div>'
        )
    return f'<div class="think-panel">{"".join(rows)}</div>'


# ---------------------------------------------------------------------------
# Tools — the full tool system now lives in primus/tools/ (registry.py). It is imported
# back here so every call site (build_tools, fast paths, slash commands, graphs) is
# unchanged. Tools reach this module's runtime globals (CFG, models, helpers) via the live
# module object, so behavior is identical. Add/extend tools in primus/tools/registry.py.
# ---------------------------------------------------------------------------
from primus.tools import (  # noqa: E402
    SelfImprovementLog,
    SelfImprovementMemory,
    _SITE_HOMES,
    _infer_lesson_category,
    _verify_python_text,
    analyze_self,
    apply_pending_self_edit,
    browse_web,
    build_tools,
    connections_add_ui,
    connections_health_line,
    connections_refresh_markdown,
    connections_remove_ui,
    connections_status_markdown,
    desktop_control,
    device_management,
    disk_cleanup,
    download_pdf_from_url,
    download_webpage_as_pdf,
    get_news,
    get_weather,
    inbox_connect_ui,
    inbox_draft_reply_ui,
    inbox_list_ui,
    inbox_read_ui,
    inbox_reply_prefill_ui,
    inbox_setup_markdown,
    inbox_status_markdown,
    ingest_web_document,
    open_application,
    read_article,
    search_own_codebase,
    search_reddit,
    system_monitor,
)


# ---------------------------------------------------------------------------
# LangGraph — plan → step loop → summarize
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dual-model orchestration — Primus (fast) + Forge (coding/reasoning)
#
# Flow per message:
#   1. ModelRouter.analyze() → weighted intent + reasoning trace (no LLM call)
#   2. build_memory_context() → unified KB + LTM + [CONV-N] + anti-hallucination (both agents)
#   3. ResponseCache hit? → return instantly (Primus trivial only, no retrieval hits)
#   4. Forge selected but unavailable/slow? → fallback to Primus; rebuild retrieval for Primus
#   5. invoke LangGraph with rag_context injected into system prompt
# ---------------------------------------------------------------------------

_agent_graphs: dict[str, Any] = {}
# --- agent routing / graphs / orchestration moved to primus/agents/system.py ---

# ---------------------------------------------------------------------------
# Desktop companion — window, tray, pin
# ---------------------------------------------------------------------------

WINDOW_TITLES = ("Primus", "Gradio", APP_NAME)


def _wmctrl(*args: str) -> bool:
    if not shutil.which("wmctrl"):
        return False
    try:
        return subprocess.run(["wmctrl", *args], capture_output=True, timeout=5).returncode == 0
    except (subprocess.SubprocessError, OSError) as exc:
        log.debug("wmctrl %s failed: %s", args, exc)
        return False


def hide_app_window() -> str:
    ok = any(_wmctrl("-r", t, "-b", "add,hidden") for t in WINDOW_TITLES)
    PrimusSession.window_hidden = ok
    return "Hidden to tray." if ok else "Install wmctrl to hide window."


def show_app_window() -> str:
    url = app_url()
    # Unhide if already open
    unhidden = any(_wmctrl("-r", t, "-b", "remove,hidden") for t in WINDOW_TITLES)
    any(_wmctrl("-r", t, "-a") for t in WINDOW_TITLES)  # activate
    if unhidden:
        PrimusSession.window_hidden = False
        if CFG.get("always_on_top"):
            set_always_on_top(True)
        return "Primus restored."
    return open_app_window(url)


# GPU-disabling flags that prevent the Chromium "Aw, Snap! (SIGILL)" renderer crash —
# common on Linux with AMD GPUs. Applied to EVERY Chromium-family launch (tabs too),
# because the crash hits normal tabs, not just --app window mode.
_CHROME_GPU_SAFE_FLAGS = [
    "--disable-gpu",
    "--disable-gpu-compositing",
    "--disable-software-rasterizer",
    "--disable-features=VizDisplayCompositor",
]

_CHROMIUM_BINARIES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium-browser",
    "chromium",
    "brave-browser",
    "microsoft-edge",
)


def _find_chromium() -> Optional[str]:
    for binary in _CHROMIUM_BINARIES:
        if shutil.which(binary):
            return binary
    return None


def _chromium_command(binary: str, url: str) -> list[str]:
    cmd = [binary, *_CHROME_GPU_SAFE_FLAGS]
    if CFG.get("use_app_window", False):
        cmd += [f"--app={url}", "--window-name=Primus"]
    else:
        cmd += [url]
    return cmd


def open_app_window(url: Optional[str] = None) -> str:
    url = url or browser_url()  # dark-theme corner widget by default
    PrimusSession.window_hidden = False

    # Prefer launching a Chromium-family browser directly with GPU disabled. This is the
    # reliable fix for the AMD/Linux "Aw, Snap! (SIGILL)" crash — webbrowser.open() can't
    # pass the GPU-disabling flags, so the default browser may still crash without them.
    binary = _find_chromium()
    if binary:
        try:
            subprocess.Popen(
                _chromium_command(binary, url),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if CFG.get("always_on_top"):
                threading.Timer(2.0, lambda: set_always_on_top(True)).start()
            return "Primus window opened (GPU-safe)."
        except Exception as exc:  # noqa: BLE001
            log.warning("Chromium launch failed (%s) — falling back to default browser", exc)

    # Non-Chromium default (e.g. Firefox, which doesn't suffer this crash) or launch failed.
    try:
        webbrowser.open(url)
        return "Opened browser tab."
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not open a browser automatically: %s", exc)
        return f"Open {url} in your browser."


def set_always_on_top(enabled: bool) -> str:
    CFG["always_on_top"] = enabled
    save_config_file()
    if not shutil.which("wmctrl"):
        return "Pinned in config (install wmctrl to apply)."
    action = "add,above" if enabled else "remove,above"
    ok = any(_wmctrl("-r", t, "-b", action) for t in WINDOW_TITLES)
    return "Pinned." if enabled and ok else "Unpinned." if ok else "Open window first."


def set_compact_mode(enabled: bool) -> str:
    CFG["compact_mode"] = enabled
    save_config_file()
    return "Compact mode saved — refresh UI to apply fully."


# ---------------------------------------------------------------------------
# Launch sound — cyberpunk "Primus" voice (Piper TTS → pyttsx3 fallback)
# Plays asynchronously on startup; never blocks UI launch.
# ---------------------------------------------------------------------------

_launch_sound_played = False
_launch_sound_lock = threading.Lock()


def _find_piper_model() -> Optional[Path]:
    """Locate a Piper ONNX voice model (config path → ~/.primus/piper → PATH)."""
    configured = (CFG.get("piper_model_path") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return path
    if PIPER_DIR.is_dir():
        for candidate in sorted(PIPER_DIR.glob("*.onnx")):
            return candidate
    for env_key in ("PIPER_MODEL", "PIPER_VOICE"):
        env_path = os.environ.get(env_key, "").strip()
        if env_path and Path(env_path).expanduser().exists():
            return Path(env_path).expanduser()
    return None


def _find_audio_player() -> Optional[list[str]]:
    """Return command prefix for playing a WAV file locally."""
    for cmd in (
        ["paplay"],
        ["aplay", "-q"],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
        ["mpv", "--no-video", "--really-quiet"],
    ):
        if shutil.which(cmd[0]):
            return cmd
    return None


def _play_wav_file(path: Path) -> None:
    player = _find_audio_player()
    if not player:
        log.debug("Launch sound: no player (install paplay via alsa-utils / pulseaudio-utils)")
        return
    try:
        subprocess.run(
            [*player, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.debug("Launch sound playback failed: %s", exc)


def _launch_sound_meta_path() -> Path:
    return AUDIO_DIR / "launch_pronounce.meta.json"


def _launch_sound_cache_valid(text: str) -> bool:
    if not LAUNCH_SOUND_WAV.exists() or LAUNCH_SOUND_WAV.stat().st_size < 200:
        return False
    meta_path = _launch_sound_meta_path()
    if not meta_path.exists():
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return (
            meta.get("text") == text
            and meta.get("length_scale") == float(CFG.get("launch_sound_length_scale", 1.22))
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def _write_launch_sound_cache_meta(text: str, engine: str) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    _launch_sound_meta_path().write_text(
        json.dumps(
            {
                "text": text,
                "engine": engine,
                "length_scale": float(CFG.get("launch_sound_length_scale", 1.22)),
                "ts": _now_iso(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _synthesize_piper_python(text: str, out_path: Path, model_path: Path) -> bool:
    try:
        from piper import PiperVoice  # type: ignore

        voice = PiperVoice.load(str(model_path))
        length = float(CFG.get("launch_sound_length_scale", 1.22))
        noise = float(CFG.get("launch_sound_noise_scale", 0.55))
        with wave.open(str(out_path), "wb") as wav_file:
            voice.synthesize(text, wav_file, length_scale=length, noise_scale=noise)
        return out_path.exists() and out_path.stat().st_size > 100
    except Exception as exc:
        log.debug("Piper Python synth failed: %s", exc)
        return False


def _synthesize_piper_cli(text: str, out_path: Path, model_path: Path) -> bool:
    piper_bin = shutil.which("piper")
    if not piper_bin:
        return False
    try:
        length = float(CFG.get("launch_sound_length_scale", 1.22))
        proc = subprocess.run(
            [
                piper_bin,
                "--model",
                str(model_path),
                "--output_file",
                str(out_path),
                "--length_scale",
                str(length),
            ],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=45,
            check=False,
        )
        return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 100
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.debug("Piper CLI synth failed: %s", exc)
        return False


def _synthesize_pyttsx3(text: str, out_path: Path) -> bool:
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        try:
            rate = engine.getProperty("rate")
            if rate:
                engine.setProperty("rate", max(75, int(rate) - 70))
            voices = engine.getProperty("voices") or []
            chosen = None
            for v in voices:
                blob = f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}".lower()
                if any(k in blob for k in ("male", "english", "ryan", "alan", "david", "brian")):
                    chosen = v.id
                    break
            if chosen:
                engine.setProperty("voice", chosen)
        except Exception:
            pass
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        return out_path.exists() and out_path.stat().st_size > 100
    except Exception as exc:
        log.debug("pyttsx3 synth failed: %s", exc)
        return False


def synthesize_launch_sound(text: str) -> Optional[Path]:
    """Build or reuse cached WAV for the launch phrase."""
    engine_pref = (CFG.get("launch_sound_engine") or "auto").lower()
    if engine_pref == "off":
        return None

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if _launch_sound_cache_valid(text):
        return LAUNCH_SOUND_WAV

    out_path = LAUNCH_SOUND_WAV
    used_engine = ""

    if engine_pref in ("auto", "piper"):
        model = _find_piper_model()
        if model:
            if _synthesize_piper_python(text, out_path, model):
                used_engine = "piper-python"
            elif _synthesize_piper_cli(text, out_path, model):
                used_engine = "piper-cli"

    if not used_engine and engine_pref in ("auto", "pyttsx3"):
        if _synthesize_pyttsx3(text, out_path):
            used_engine = "pyttsx3"

    if used_engine:
        _write_launch_sound_cache_meta(text, used_engine)
        return out_path
    return None


def _launch_sound_worker() -> None:
    text = str(CFG.get("launch_sound_text") or "Primus.").strip() or "Primus."
    wav = synthesize_launch_sound(text)
    if wav and wav.exists():
        _play_wav_file(wav)
        log.info("Launch sound played (%s)", wav.name)
    else:
        log.debug(
            "Launch sound skipped — install Piper model in %s or: uv pip install piper-tts pyttsx3",
            PIPER_DIR,
        )


def play_launch_sound_async() -> None:
    """Fire-and-forget cyberpunk startup voice — does not block UI."""
    global _launch_sound_played
    if not CFG.get("play_launch_sound", True):
        return
    if (CFG.get("launch_sound_engine") or "auto").lower() == "off":
        return
    with _launch_sound_lock:
        if _launch_sound_played:
            return
        _launch_sound_played = True
    threading.Thread(
        target=_launch_sound_worker,
        name="primus-launch-sound",
        daemon=True,
    ).start()


class TrayManager:
    """System tray with show/hide/pin/quit."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._icon = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        try:
            import pystray  # type: ignore
            from PIL import Image, ImageDraw  # type: ignore
        except ImportError:
            log.warning("Tray unavailable — pip install pystray pillow")
            return False

        img = Image.new("RGB", (64, 64), (7, 4, 15))
        d = ImageDraw.Draw(img)
        d.polygon([(32, 6), (54, 54), (10, 54)], outline=(34, 211, 238), width=2)
        d.line([(22, 38), (42, 38)], fill=(217, 70, 239), width=2)
        d.text((26, 18), "P", fill=(167, 139, 250))

        menu = pystray.Menu(
            pystray.MenuItem("Show Primus", self._show, default=True),
            pystray.MenuItem("Hide to Tray", self._hide),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Always on Top", self._toggle_pin, checked=lambda _: CFG.get("always_on_top", False)),
            pystray.MenuItem("Compact Mode", self._toggle_compact, checked=lambda _: CFG.get("compact_mode", False)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Primus", self._quit),
        )
        self._icon = pystray.Icon(APP_NAME, img, "Primus — Admin Agent", menu)
        PrimusSession.tray_icon = self._icon
        # Non-daemon: keeps process alive alongside Gradio when using prevent_thread_lock
        self._thread = threading.Thread(target=self._icon.run, daemon=False, name="primus-tray")
        self._thread.start()
        log.info("System tray active")
        return True

    def _show(self, _icon=None, _item=None) -> None:
        show_app_window()

    def _hide(self, _icon=None, _item=None) -> None:
        hide_app_window()

    def _toggle_pin(self, _icon=None, _item=None) -> None:
        set_always_on_top(not CFG.get("always_on_top", False))

    def _toggle_compact(self, _icon=None, _item=None) -> None:
        set_compact_mode(not CFG.get("compact_mode", False))

    def _quit(self, _icon=None, _item=None) -> None:
        if self._icon:
            self._icon.stop()
        os._exit(0)


def start_tray_icon(url: str, on_show, on_quit) -> None:
    """Legacy wrapper — prefer TrayManager."""
    mgr = TrayManager(url)
    if not mgr.start():
        return


# --- background + scheduled agent systems moved to primus/agents/system.py ---

# ---------------------------------------------------------------------------
# Gradio UI — the entire presentation layer now lives in primus/ui/ (builder.py): theme/CSS,
# the chat-first shell (top bar + transcript + composer), the slide-over Menu drawer that holds
# chats/projects, mode, status, knowledge, scheduled agents, voice & window, connections and
# activity, plus voice input, the typing effect, every event handler, and the
# no-Gradio fallback server. It is imported back here so main() launches it unchanged. The UI
# reaches this module's live globals (CFG, models, tools, memory, agents) via the host module.
# ---------------------------------------------------------------------------
from primus.ui import (  # noqa: E402
    build_ui,
    prewarm_whisper,
    run_fallback_setup_server,
)


def _str2bool(value: str) -> bool:
    """Lenient bool parser so flags like --share=False / --share true / --share work."""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("y", "yes", "t", "true", "on", "1"):
        return True
    if v in ("n", "no", "f", "false", "off", "0", ""):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Primus — offline Admin Agent")
    p.add_argument("--port", type=int, default=None, help="Server port (default 7860)")
    p.add_argument("--host", default=None, help="Bind address (default 127.0.0.1, local-only)")
    p.add_argument(
        "--share",
        nargs="?",
        type=_str2bool,
        const=True,
        default=None,
        metavar="{true,false}",
        help="Create a public Gradio link (default: off). Accepts --share, --share=true, --share=false",
    )
    p.add_argument(
        "--no-share",
        dest="share",
        action="store_const",
        const=False,
        help="Force local-only mode (no public link) — this is the default",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Force local-only mode on 127.0.0.1 with no public link",
    )
    p.add_argument("--model", default=None, help="Primus orchestrator model")
    p.add_argument("--forge-model", default=None, help="Forge coding agent model")
    p.add_argument("--browser", action="store_true", help="Open app window on start")
    p.add_argument("--tray", action="store_true", help="Enable system tray")
    p.add_argument("--no-tray", action="store_true", help="Disable system tray")
    p.add_argument("--compact", action="store_true", help="Compact UI mode")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--debug", action="store_true", help="Verbose logs + show Gradio errors in the UI")
    p.add_argument("--app-window", action="store_true", help="Open in a dedicated Chromium app window (GPU-safe)")
    p.add_argument("--install-desktop", action="store_true", help="Install .desktop launcher")
    p.add_argument("--install-autostart", action="store_true", help="Start Primus on login")
    p.add_argument("--uninstall-autostart", action="store_true", help="Remove login autostart")
    p.add_argument("--install-all", action="store_true", help="Install desktop + autostart")
    p.add_argument("--setup", action="store_true", help="Print setup guide and check deps")
    p.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    p.add_argument("--index", nargs="?", const="", metavar="PATH", help="Index folder into knowledge base")
    p.add_argument("--export", action="store_true", help="Write a backup zip under APP_DIR/exports/")
    p.add_argument("--import", dest="import_file", metavar="FILE", default=None, help="Restore a backup zip")
    p.add_argument(
        "--include-secrets",
        action="store_true",
        help="Include/restore OAuth tokens and vault blobs (both zip and import must pass this)",
    )
    return p.parse_args()


def _global_excepthook(exc_type, exc, tb) -> None:
    log.critical("Unhandled exception:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
    sys.__excepthook__(exc_type, exc, tb)


def main() -> None:
    sys.excepthook = _global_excepthook
    args = parse_args()

    # Install / setup commands (no server)
    if args.install_desktop or args.install_autostart or args.uninstall_autostart or args.install_all:
        init_config(args)
        setup_file_logging(CFG.get("log_level", "INFO"))
        if args.install_all:
            for p in install_all_launchers():
                print(f"Installed: {p}")
        if args.install_desktop:
            p = install_desktop_entry()
            print(f"Desktop launcher installed: {p}")
        if args.install_autostart:
            p = install_autostart(True)
            print(f"Autostart enabled: {p}")
        if args.uninstall_autostart:
            install_autostart(False)
            print("Autostart disabled.")
        return

    init_config(args)
    if args.export or getattr(args, "import_file", None):
        ensure_app_dirs()
        from primus.core.backup import export_bundle, import_bundle  # noqa: PLC0415

        include_secrets = bool(getattr(args, "include_secrets", False))
        if args.export:
            path = export_bundle(include_secrets=include_secrets)
            print(path)
            return
        print(import_bundle(args.import_file, include_secrets=include_secrets))
        return
    if getattr(args, "app_window", False):
        CFG["use_app_window"] = True
    debug_mode = bool(getattr(args, "debug", False))
    if debug_mode:
        CFG["log_level"] = "DEBUG"
    setup_file_logging(CFG.get("log_level", "INFO"))
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug mode enabled — verbose logging + Gradio UI errors visible")
    ensure_app_dirs()
    _ensure_localhost_direct()
    _ensure_ollama_host_env()
    try:
        setup_gpu_acceleration()  # AMD/ROCm/Vulkan env setup before models load
    except Exception as exc:  # noqa: BLE001
        log.warning("GPU acceleration setup skipped: %s", exc)

    if not HAS_GRADIO:
        host = CFG.get("host", SERVER_HOST)
        port = int(CFG.get("port", DEFAULT_PORT))
        print(f"\n❌ Gradio not installed: {_GRADIO_ERROR}\n")
        run_fallback_setup_server(host, port)
        return

    if args.setup:
        sys.exit(setup(print_guide=True))

    if args.check_deps:
        code = setup(print_guide=False)
        deps = PrimusSession_deps_cache
        ollama = get_ollama_details()
        print(f"Ollama: {ollama['message']}")
        for d in deps:
            print(f"  {'OK' if d['ok'] else 'MISSING':7} {d['name']}")
        try:
            setup_gpu_acceleration()
            print("\n" + gpu_status_text())
        except Exception as exc:  # noqa: BLE001
            print(f"\nGPU status unavailable: {exc}")
        sys.exit(code)

    if getattr(args, "index", None) is not None:
        if not HAS_AI_STACK:
            print(f"AI stack required for indexing: {AI_STACK_ERROR}")
            sys.exit(1)
        init_knowledge_base(seed=True, auto_index=False)
        if args.index == "":
            t = start_background_index()
            t.join(timeout=600)
            print(knowledge_status_markdown())
        else:
            path = Path(args.index).expanduser().resolve()
            print(get_kb().ingest_folder(path) if path.is_dir() else get_kb().ingest_file(path))
        return

    if (args.verbose or debug_mode) and HAS_AI_STACK:
        logging.getLogger("langchain").setLevel(logging.DEBUG)

    check_dependencies()
    if HAS_AI_STACK:
        try:
            init_knowledge_base(seed=True)
        except Exception as exc:
            log.warning("Knowledge base init skipped: %s", exc)
    print_startup_diagnostics()
    print_tool_health()
    prewarm_whisper()  # background-load the STT model so first voice input is instant

    graphs: dict[str, Any] = {}
    if HAS_AI_STACK:
        try:
            graphs = init_agent_graphs()
        except Exception as exc:
            log.error("Failed to build agent graphs: %s", exc)
            print(f"Warning: agent graphs failed ({exc}). UI will load; chat limited until fixed.")

    demo, ui_theme, ui_css = build_ui(graphs, CFG["model"])

    # Background watchdog: keep the UI recoverable if any turn ever overruns the hard limit.
    if bool(CFG.get("turn_watchdog_enabled", True)):
        start_turn_watchdog()

    # Resolve the final launch target up-front: local-only by default, fixed port,
    # public link only when explicitly requested. Pick a free port so the browser and
    # tray open the correct URL even if the preferred port was taken.
    _ensure_localhost_direct()
    _patch_gradio_localhost_check()
    _patch_gradio_schema_bug()
    share = bool(getattr(args, "share", False)) and not getattr(args, "local", False)
    requested_host = "127.0.0.1" if getattr(args, "local", False) else (CFG.get("host") or SERVER_HOST or "127.0.0.1")
    requested_port = int(CFG.get("port") or DEFAULT_PORT or 7860)
    bound_port = _find_free_port(requested_host, requested_port)
    if bound_port != requested_port:
        log.info("Port %s busy — Primus will use %s instead", requested_port, bound_port)
    CFG["host"], CFG["port"] = requested_host, bound_port
    if share:
        log.warning("Public share link requested (--share) — Primus will be reachable beyond localhost.")

    url = app_url()
    ollama = get_ollama_details()
    log.info(
        "Primus starting — primus=%s forge=%s %s — %s",
        CFG["model"],
        CFG.get("forge_model", FORGE_MODEL),
        url,
        ollama.get("message", ""),
    )

    want_tray = bool(CFG.get("tray_enabled", True) and not args.no_tray and HAS_AI_STACK)
    tray_ok = False
    if want_tray or args.tray:
        tray_ok = TrayManager(url).start()
        if (want_tray or args.tray) and not tray_ok:
            log.info("Running without tray — server will block in foreground (install pystray pillow for tray mode)")

    open_browser = CFG.get("open_browser_on_start", True) or args.browser
    start_min = CFG.get("start_minimized", False)
    if open_browser and not start_min:
        threading.Timer(1.5, lambda: open_app_window(browser_url())).start()
    elif start_min and tray_ok:
        threading.Timer(2.0, hide_app_window).start()

    if CFG.get("always_on_top") and open_browser and not start_min:
        threading.Timer(3.5, lambda: set_always_on_top(True)).start()

    w = int(CFG.get("window_width", 420))
    h = int(CFG.get("window_height", 660))

    # NOTE: theme/css are applied on gr.Blocks() in build_ui (they are NOT launch kwargs).
    base_kwargs = dict(
        inbrowser=False,
        share=share,
        quiet=not debug_mode,
        show_error=debug_mode,
        height=h,
        width=w,
        # Only non-blocking when tray actually started; otherwise main exits and kills server
        prevent_thread_lock=tray_ok,
    )

    def _try_launch(server_name: str, server_port: int, share_flag: bool) -> None:
        # Keep the proxy bypass + localhost/schema patches fresh on every attempt.
        _ensure_localhost_direct()
        _patch_gradio_localhost_check()
        _patch_gradio_schema_bug()
        kwargs = {
            **base_kwargs,
            "server_name": server_name,
            "server_port": server_port,
            "share": share_flag,
        }
        try:
            demo.launch(**kwargs)
        except TypeError as exc:
            # Graceful degradation: an older/newer Gradio may not accept some cosmetic
            # kwargs (footer_links, height, width, theme, css). Drop them and retry with
            # the essential, always-supported set so the UI still comes up.
            log.warning("launch() rejected an option (%s) — retrying with minimal kwargs", exc)
            essential = {
                "server_name": server_name,
                "server_port": server_port,
                "share": share_flag,
                "inbrowser": False,
                "quiet": True,
                "prevent_thread_lock": tray_ok,
            }
            demo.launch(**essential)

    def _print_ready_banner() -> None:
        bar = "─" * 56
        share_line = "  Public link: enabled (--share)\n" if share else ""
        print(
            f"\n{bar}\n"
            f"  ◈ Primus is ready\n"
            f"  Local UI : {browser_url()}\n"
            f"  Mode     : local-only{' + tray' if tray_ok else ''} · dark · "
            f"{'compact' if CFG.get('compact_mode') else 'standard'}\n"
            f"{share_line}"
            f"  Stop     : Ctrl+C\n"
            f"  Restart  : uv run python admin_assistant.py\n"
            f"  If the browser shows \"Aw, Snap!\": the GPU-safe window failed —\n"
            f"    open {app_url()} manually, or try a different browser.\n"
            f"{bar}\n",
            flush=True,
        )

    # In foreground (no-tray) mode launch() blocks, so announce readiness on a short timer.
    if not tray_ok:
        threading.Timer(1.0, _print_ready_banner).start()

    launched = False
    last_err: Optional[BaseException] = None

    # Primary attempt on the resolved host/port, then a small port-bump fallback in case
    # the port was claimed between selection and bind, then a guaranteed local-only retry.
    attempts: list[tuple[str, int, bool]] = [(requested_host, bound_port, share)]
    for offset in range(1, 8):
        attempts.append((requested_host, bound_port + offset, share))
    if requested_host != "127.0.0.1":
        attempts.append(("127.0.0.1", bound_port, False))

    for host_try, port_try, share_try in attempts:
        try:
            _try_launch(host_try, port_try, share_try)
            launched = True
            CFG["host"], CFG["port"] = host_try, port_try
            if (host_try, port_try) != (requested_host, bound_port):
                log.info("Primus bound to http://%s:%s (fallback)", host_try, port_try)
            break
        except OSError as exc:
            last_err = exc
            log.warning("Could not bind %s:%s (%s) — trying fallback", host_try, port_try, exc)
            continue
        except ValueError as exc:
            last_err = exc
            msg = str(exc).lower()
            if "localhost" in msg or "share" in msg or "accessible" in msg:
                log.warning("Gradio accessibility check failed (%s) — forcing local-only retry", exc)
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.error("Unexpected launch error on %s:%s: %s", host_try, port_try, exc)
            continue

    if not launched:
        log.error("Primus could not start a local server: %s", last_err)
        print(
            "\nPrimus couldn't start the web server.\n"
            f"  Last error: {last_err}\n\n"
            "Things to try:\n"
            "  • Pick another port:        uv run python admin_assistant.py --port 7870\n"
            "  • Force local-only mode:    uv run python admin_assistant.py --local\n"
            "  • Verbose diagnostics:      uv run python admin_assistant.py --debug\n"
            "  • Clear proxy interference: unset http_proxy https_proxy ALL_PROXY && "
            "NO_PROXY=localhost,127.0.0.1 uv run python admin_assistant.py\n"
            "\nBrowser shows \"Aw, Snap!\" / SIGILL (common on Linux + AMD GPUs)?\n"
            "  The server is fine — it's a browser GPU crash. Try:\n"
            "  • Open the URL in Firefox (not affected by this crash)\n"
            "  • Launch Chrome/Chromium with: --disable-gpu --disable-software-rasterizer\n"
        )
        sys.exit(1)

    play_launch_sound_async()

    if tray_ok:
        # Gradio server thread is daemon — block main thread so the process stays alive
        _print_ready_banner()
        log.info("Primus running at %s (tray mode — Ctrl+C or tray Quit to stop)", url)
        try:
            demo.block_thread()
        except KeyboardInterrupt:
            log.info("Primus stopped.")
    else:
        # prevent_thread_lock=False: launch() already blocked until server stopped
        log.info("Primus stopped.")


if __name__ == "__main__":
    main()
