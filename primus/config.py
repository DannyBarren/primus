"""Primus configuration substrate — paths, DEFAULT_CONFIG, and config load/save logic.

Extracted from admin_assistant.py in Phase 2 of the modularization. Behavior is identical to the
single-file version, with two *additive* enhancements for future cloud / Docker use:

  * Base directories can be redirected with environment variables (unset => original local defaults):
        PRIMUS_DATA_DIR       -> APP_DIR        (default ~/.primus)
        PRIMUS_CONFIG_DIR     -> CONFIG_DIR     (default ~/.config/primus)
        PRIMUS_DOWNLOADS_DIR  -> DOWNLOADS      (default ~/Downloads)
  * OLLAMA_URL is accepted as an alias for the existing OLLAMA_HOST default.

Precedence is unchanged: env vars only set DEFAULTS; a saved ~/.config/primus/config.json still wins,
exactly as before. The live CFG dict, the model globals, SCRIPT_PATH, and the global rebinding all
remain in admin_assistant.py — this module is pure config data + stateless helpers. Nothing here is
locked or read-only; everything stays freely editable.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("primus")


def _env_dir(var: str, default: Path) -> Path:
    """Return a directory from env var ``var`` (expanded) when set & non-empty, else ``default``.

    Lets containerized / cloud deployments point Primus at mounted volumes without touching code or
    changing local behavior (env unset => identical to the original hard-coded path).
    """
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser() if raw else default


# ---------------------------------------------------------------------------
# Paths (env-overridable bases; everything else derives from them as before)
# ---------------------------------------------------------------------------
HOME = Path.home()

APP_NAME = "Primus"
APP_ID = "primus"
APP_DIR = _env_dir("PRIMUS_DATA_DIR", HOME / ".primus")
CONFIG_DIR = _env_dir("PRIMUS_CONFIG_DIR", HOME / ".config" / "primus")
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_FILE = CONFIG_DIR / "primus.log"
DESKTOP_DIR = HOME / ".local" / "share" / "applications"
AUTOSTART_DIR = HOME / ".config" / "autostart"
LEGACY_DIR = HOME / ".admin_assistant"
MEMORY_FILE = APP_DIR / "memory.json"
CHAT_HISTORY_FILE = APP_DIR / "chat_history.json"
# Project/subject chat organization (UI/session layer only — never isolates core memory).
PROJECTS_FILE = APP_DIR / "projects.json"
PROJECT_HISTORY_DIR = APP_DIR / "project_chats"
GENERAL_CHAT_ID = "general"
TODOS_FILE = APP_DIR / "todos.md"
NOTES_FILE = APP_DIR / "notes.md"
TASKS_FILE = APP_DIR / "task_history.json"
SETTINGS_FILE = APP_DIR / "settings.json"  # legacy; migrated into config
EXPORT_DIR = APP_DIR / "exports"
PLANS_DIR = APP_DIR / "plans"
KNOWLEDGE_DIR = APP_DIR / "knowledge"
CHROMA_DIR = KNOWLEDGE_DIR / "chroma"
KB_MANIFEST = KNOWLEDGE_DIR / "manifest.json"
KB_UPLOADS_DIR = KNOWLEDGE_DIR / "uploads"  # drag-and-drop originals, kept for re-ingest
KB_DOWNLOADS_DIR = KNOWLEDGE_DIR / "downloads"  # autonomously downloaded web docs/PDFs
KB_INTERACTIONS = KNOWLEDGE_DIR / "interactions.json"
MEMORIES_FILE = APP_DIR / "memories.json"
SESSION_MEMORY_FILE = APP_DIR / "session_memory.json"
CHAT_SUMMARY_FILE = APP_DIR / "chat_summary.json"
CONVERSATION_ARCHIVE = APP_DIR / "conversation_archive.json"
REFLECTIONS_FILE = APP_DIR / "reflections.json"  # self-reflection loop: how to serve the operator better
METRICS_FILE = APP_DIR / "metrics.json"  # meta-learning: tool/route/latency/feedback statistics
FEEDBACK_FILE = APP_DIR / "feedback.jsonl"  # append-only correction log (sibling of audit.jsonl)
AUDIT_FILE = APP_DIR / "audit.jsonl"  # shared action log (tools, queue/execute, export)
BRAIN_BUDGET_FILE = APP_DIR / "brain_budget.json"  # weekly $ / per-turn caps (governor + drawer)
VAULT_DIR = APP_DIR / "vault"  # encrypted secret blob directory
VAULT_INDEX = APP_DIR / "vault_index.json"  # ids + labels only — never plaintext
VAULT_ACCESS_LOG = APP_DIR / "vault_access.jsonl"  # vault audit (no secret values)
WATCHDOG_FILE = APP_DIR / "watchdog_incidents.json"  # log of force-recovered (stuck) turns
AUDIO_DIR = APP_DIR / "audio"
PIPER_DIR = APP_DIR / "piper"
LAUNCH_SOUND_WAV = AUDIO_DIR / "launch_pronounce.wav"
SELF_EDIT_BACKUP_DIR = APP_DIR / "self_edits"  # timestamped backups before any approved self-edit
SELF_IMPROVE_LOG = APP_DIR / "self_improvements.json"  # history of self-improvement proposals + outcomes
SELF_IMPROVE_MEMORY = APP_DIR / "self_improvement_memory.json"  # lessons learned about self-improvement
BACKGROUND_TASKS_FILE = APP_DIR / "background_tasks.json"  # persisted auto/queued background tasks
BACKGROUND_RESULTS_DIR = APP_DIR / "background_results"  # full outputs of completed background tasks

DOWNLOADS = _env_dir("PRIMUS_DOWNLOADS_DIR", HOME / "Downloads")


DEFAULT_CONFIG: dict[str, Any] = {
    "model": os.environ.get("PRIMUS_MODEL", "qwen2.5:7b"),
    "forge_model": os.environ.get("PRIMUS_FORGE_MODEL", "qwen2.5-coder:7b"),
    "auto_delegate_forge": True,
    "router_forge_threshold": 3.0,
    "router_forge_strong_threshold": 5.0,
    "response_cache_ttl_sec": 300,
    "response_cache_max": 64,
    "forge_fallback_to_primus": True,
    "forge_invoke_timeout_sec": 180,
    # --- Forge coding performance (qwen2.5-coder punches above its weight) ---
    # Fast coding path: short, self-contained one-file coding asks ("write a 15-line script
    # that…") skip the plan→step→summarize graph and answer in ONE focused coder-model call.
    "fast_coding_enabled": True,
    "fast_coding_max_chars": 320,          # only trivial/self-contained asks take the fast path
    "forge_fast_invoke_timeout_sec": 100,  # tighter cap for the single-shot fast coding call
    "forge_max_plan_steps": 4,             # keep Forge plans focused (2–4 steps)
    "coding_verify_on_generate": True,     # Forge is told to verify code with verify_python after writing
    "forge_lean_context": True,            # prune RAG/memory on self-contained coding (cleaner context, faster)
    # Automated self-correction: after Forge delivers code, compile/lint it for real and, if it has
    # actual errors, do ONE repair round-trip so the operator gets code that runs. Static-only (never executes).
    "forge_autocorrect": True,
    "forge_autocorrect_timeout_sec": 90,   # bound the single repair call so it can't hang a turn
    # Hard cap for the main Primus graph path so a stuck turn can't hang the UI forever.
    # On expiry Primus returns a graceful PARTIAL answer (0 = no cap; legacy behavior).
    "primus_invoke_timeout_sec": 240,
    # Fast meta-instruction path: preference/style directives ("be more concise",
    # "always provide complete info", "remember I prefer…") are saved + confirmed instantly,
    # bypassing the full agent loop unless the user explicitly asks for deep analysis.
    "fast_meta_path": True,
    # Background watchdog: force-recovers any turn that runs longer than the hard limit so
    # the Halt button + UI always stay responsive, even during long model/tool chains.
    "turn_watchdog_enabled": True,
    "turn_hard_limit_sec": 420,  # ≈7 min; tune lower for a snappier safety net
    "primus_temperature": 0.2,
    "forge_temperature": 0.15,
    # --- Speed / responsiveness tuning ---
    "ollama_keep_alive": "30m",   # keep model resident in RAM/VRAM between requests (avoids reload stalls)
    # --- Sequential model policy (only one HEAVY model in RAM at a time on the Ryzen laptop) ---
    "sequential_models": True,    # unload Primus before Forge runs, and vice-versa; reload after
    "sequential_warm_reload": True,  # after Forge, warm Primus back up in the background
    "ollama_num_ctx": 4096,        # context window — smaller = faster prompt processing
    "primus_num_predict": 768,     # cap Primus output tokens for snappy replies
    "forge_num_predict": 2048,     # Forge may need longer code output
    "ollama_top_k": 30,
    "ollama_top_p": 0.9,
    "skip_summary_max_chars": 900,  # below this, present step results directly (no extra LLM call)
    "warm_up_models": True,          # preload Primus model on startup for a fast first reply
    # --- Fast conversational tier: light model answers pure chat in one call (no graph/tools) ---
    "fast_chat_tier": True,
    "primus_fast_model": os.environ.get("PRIMUS_FAST_MODEL", "llama3.2:3b"),
    # Hard cap on that single chat call. A greeting or identity turn is never worth more than a
    # few seconds, and on expiry Primus answers plainly — it must never escalate to the planner.
    "fast_chat_invoke_timeout_sec": 15,
    # --- Deterministic tool fast-path: clear single-tool requests (weather, open app,
    #     battery, disk) run the tool directly in code — no planning LLM, no RAG. ---
    "fast_tool_path": True,
    # --- Clean chat: keep internal reasoning/routing out of the chat window ---
    "show_thoughts_in_chat": False,   # reasoning lives in the side panel + console log, not the chat
    "show_model_badge_in_chat": False,  # model indicator shown in status bar, not on every message
    "ollama_url": os.environ.get("OLLAMA_URL", os.environ.get("OLLAMA_HOST", "http://localhost:11434")),
    "host": os.environ.get("PRIMUS_HOST", "127.0.0.1"),
    "port": int(os.environ.get("PRIMUS_PORT", "7860")),
    "execution_mode": "execute",
    # primus_path (default): Primus decides order and tools (leftover → plan→execute graph).
    # user_path: execute the operator's listed steps in that order. No extra store file.
    "path_mode": "primus_path",
    "auto_execute_safe_commands": True,
    "always_on_top": False,
    "compact_mode": False,
    "tray_enabled": True,
    "start_minimized": False,
    "open_browser_on_start": True,
    "autostart": False,
    "window_width": 420,
    "window_height": 660,
    "python_path": sys.executable,
    "script_path": "",  # set by the entrypoint at runtime (SCRIPT_PATH lives in admin_assistant.py)
    "theme": "cyberpunk",
    "log_level": "INFO",
    "ollama_retries": 2,
    # Embeddings: switchable. "nomic-embed-text" (Ollama) by default; set to
    # "bge-small" / "BAAI/bge-small-en-v1.5" for a lighter CPU-only model (needs a
    # reindex — different vector dim). "auto" picks bge-small when sentence-transformers
    # is installed, else nomic. See Setup tab → embeddings.
    "embedding_model": "nomic-embed-text",
    "embedding_fallback": "BAAI/bge-small-en-v1.5",
    "embedding_cache_size": 256,      # in-memory LRU cache of query embeddings (perf)
    # --- RAG performance (tuned for the Ryzen laptop: lower CPU/temps, faster replies) ---
    "rag_top_k": 3,
    "kb_search_k": 4,
    "hybrid_retrieval": True,         # BM25-style keyword rerank over vector candidates
    "rag_scope_by_intent": True,      # search core_business/projects first for business/eng queries
    # --- Gmail demo mode ---
    # When no real gmail_token.json exists, serve examples/gmail/fixture_inbox.json from the
    # list/read/status tools so a fresh clone has a working Inbox UI. Auto-disabled the moment
    # a real token appears. Draft/send stay preview-only in fixture mode.
    "gmail_fixture_mode": True,
    # Host OAuth: run a loopback listener during gmail_auth so browser consent completes
    # automatically (no code pasting). Auto-skipped on headless/Docker (no DISPLAY), where
    # the manual gmail_auth_code flow remains. Desktop-app clients need no console change
    # for the ephemeral loopback port (RFC 8252).
    "gmail_oauth_local_server": True,
    # --- Agent stability (middleware stack on every react agent) ---
    "model_retry_max": 2,          # transient Ollama errors retry w/ backoff before failing a step
    "tool_retry_max": 2,           # transient tool errors retry; then the error reaches the model
    "no_progress_repeat_limit": 3, # identical tool+args calls this many times → forced stop
    "agent_recursion_limit": 30,   # hard cap on react-loop steps per agent run
    "prune_tool_traces": True,     # clear old tool outputs mid-run (no LLM call) when context grows
    "prune_trigger_tokens": 3000,  # estimated token count that triggers pruning
    "prune_keep_messages": 6,      # most-recent messages whose tool outputs stay intact
    # Optional SQLite checkpointer for agent threads (~/.primus/checkpoints.sqlite). NOT
    # load-bearing: any failure (or false here) falls back to the in-memory MemorySaver, and
    # chat correctness never depends on it. Requires the optional langgraph-checkpoint-sqlite extra.
    "use_sqlite_checkpointer": True,
    "memory_recall_k": 4,
    "memory_recall_past_convos": 3,
    "memory_max_entries": 500,
    "memory_consolidate_hours": 6,
    "conversation_archive_max": 200,
    "conversation_consolidate_batch": 12,
    "chat_summarize_threshold": 24,
    "chat_recent_keep": 12,
    # --- Continuous learning ---
    "self_reflection": True,          # reflect after each substantive turn (background)
    "learning_digest_turns": 6,       # auto learning-digest every N user turns
    "reflection_max_entries": 80,     # cap stored reflections
    "learning_digest_min_turns": 3,   # don't digest a near-empty session
    # --- Meta-learning / adaptive intelligence ---
    "meta_learning": True,            # track tool/route/latency/feedback statistics
    "adaptive_routing": True,         # let feedback nudge routing scores (bounded)
    "self_improve_semantic": True,    # semantic (embedding) recall over self-improvement lessons
    "feedback_routing_max_nudge": 1.5,  # max absolute score nudge from learned feedback
    "proactive_followups": True,      # offer a smart next-step suggestion when useful
    # --- Automatic background tasks ---
    # Heavy / multi-step requests ("research deeply", "build a full app", "analyze all files")
    # are detected and run in isolated background threads so the chat turn never blocks or times
    # out. Primus acks immediately, runs the work on Forge with all tools, and injects the result
    # back into the conversation when ready. Fully config-driven and additive.
    "background_tasks": {
        "auto_background_long_tasks": True,    # auto-detect heavy requests → background
        "max_concurrent_background_tasks": 2,  # bounded concurrency (daemon threads)
        "min_chars_for_auto": 80,              # length floor before heuristic auto-detect fires
        "keep_tasks": 100,                     # cap persisted task history
        "notify_in_chat": True,                # inject completed results into chat history
    },
    # --- Speech-to-Text (voice input, local/offline via faster-whisper) ---
    "stt_enabled": True,              # show the microphone + allow voice input
    "stt_model_size": "base",         # tiny | base | small | medium (accuracy ↔ speed)
    "stt_device": "auto",             # auto | cpu | cuda  (auto-detects GPU, falls back to cpu)
    "stt_compute_type": "auto",       # auto | int8 | float16 | float32
    "stt_vad": True,                  # voice-activity detection (trims silence/noise)
    "stt_language": "",               # "" = auto-detect; or e.g. "en"
    "stt_autosubmit": True,           # auto-send transcription (off = drop into box to edit)
    "stt_beam_size": 5,
    "typing_effect": True,            # character-by-character typewriter reveal for replies
    "typing_speed": 18,               # ms per character when typing_effect is on (1 fast … 100 slow)
    "rag_chunk_size": 600,
    "rag_chunk_overlap": 80,
    "web_search_enabled": True,
    "auto_index_on_start": False,
    "ingest_paths": [],  # filled at init with defaults
    "play_launch_sound": True,
    "launch_sound_text": "Primus.",
    "launch_sound_engine": "auto",  # auto | piper | pyttsx3 | off
    "piper_model_path": "",  # empty = auto-search ~/.primus/piper/*.onnx
    "launch_sound_length_scale": 1.22,  # Piper: slower = deeper/more menacing
    "launch_sound_noise_scale": 0.55,  # Piper: lower = cleaner robot tone
    # --- AMD GPU acceleration (Ryzen AI / Radeon iGPU) ---
    # Ollama runs as a separate server; these env vars take effect when `ollama serve`
    # inherits them (Primus sets them in-process and surfaces the recommended export
    # block in Setup / `/gpu status` so you can restart the server with them applied).
    # GPU-first: prefer the GPU whenever one is usable, fall back to CPU automatically.
    "prefer_gpu": True,                     # try GPU offload first; CPU fallback is automatic
    "gpu_layers": -1,                       # layers to offload: -1 = all that fit (best for iGPU), 0 = CPU, N = explicit
    "gpu_backend": "auto",                  # auto | rocm | vulkan | cpu  (canonical key)
    "ollama_gpu_backend": "auto",          # legacy alias of gpu_backend (kept in sync)
    "ollama_flash_attention": True,        # OLLAMA_FLASH_ATTENTION=1 (faster, less VRAM)
    "hsa_override_gfx_version": "auto",     # "" = off · "auto" = detect Ryzen AI iGPU · e.g. "11.0.2"
    "gpu_apply_env": True,                  # set OLLAMA_*/HSA_* env at startup
    # --- Smaller fast fallback model + intelligent dynamic model switching ---
    "fast_fallback_model": "llama3.2:3b",  # small/quantized model for snappy simple turns
    "dynamic_switching_enabled": True,     # auto-switch to the fast model when big model is slow
    "slow_threshold_sec": 5.0,             # a simple turn slower than this trips fast mode
    "fast_model_switch_cooldown": 300,     # seconds to stay in fast mode after tripping
}
DEFAULT_INGEST_PATHS = [
    "~/Documents",
    "~/ai-workshop",
    "~/.primus",
]


def load_config_file(script_path: Any, python_path: str) -> dict[str, Any]:
    """Load config.json merged over DEFAULT_CONFIG, migrate legacy keys, fill ingest defaults.

    ``script_path``/``python_path`` are supplied by the entrypoint (admin_assistant.py keeps the
    authoritative SCRIPT_PATH). Pure logic — does not touch any live module-level CFG.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT_CONFIG.copy()
    cfg["script_path"] = str(script_path)
    cfg["python_path"] = python_path

    if CONFIG_FILE.exists():
        try:
            disk = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(disk, dict):
                cfg.update(disk)
                # Migrate legacy GPU backend key: if the file pre-dates `gpu_backend`,
                # adopt its `ollama_gpu_backend` value so existing setups keep their choice.
                if "gpu_backend" not in disk and "ollama_gpu_backend" in disk:
                    cfg["gpu_backend"] = disk["ollama_gpu_backend"]
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Config parse error (%s) — using defaults", exc)
    else:
        save_config_file(cfg)

    # Migrate legacy ~/.primus/settings.json
    if SETTINGS_FILE.exists():
        try:
            legacy = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for key in ("always_on_top", "execution_mode", "compact_mode"):
                if key in legacy and key not in cfg:
                    cfg[key] = legacy[key]
        except (json.JSONDecodeError, OSError):
            pass

    if not cfg.get("ingest_paths"):
        cfg["ingest_paths"] = [str(Path(p).expanduser()) for p in DEFAULT_INGEST_PATHS]
    return cfg


def save_config_file(cfg: dict[str, Any]) -> None:
    """Write ``cfg`` to CONFIG_FILE. The live-CFG default is handled by the main-module wrapper."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def apply_cli_and_sync(cfg: dict[str, Any], args: Optional[argparse.Namespace] = None) -> dict[str, Any]:
    """Apply CLI overrides + keep the fast tier / small fallback model in sync. Returns ``cfg``.

    This is the override logic that used to live inline in init_config(); the global rebinding
    (CFG = ...; DEFAULT_MODEL = ...) intentionally stays in admin_assistant.py.
    """
    if args:
        if getattr(args, "port", None):
            cfg["port"] = args.port
        if getattr(args, "host", None):
            cfg["host"] = args.host
        if getattr(args, "local", False):
            cfg["host"] = "127.0.0.1"
        if getattr(args, "model", None):
            cfg["model"] = args.model
        if getattr(args, "forge_model", None):
            cfg["forge_model"] = args.forge_model
        if getattr(args, "no_tray", False):
            cfg["tray_enabled"] = False
        elif getattr(args, "tray", False):
            cfg["tray_enabled"] = True
        if getattr(args, "browser", False):
            cfg["open_browser_on_start"] = True
        if getattr(args, "compact", False):
            cfg["compact_mode"] = True

    # Keep the fast tier and the small fallback model in sync: if no explicit fast
    # chat model is set, derive it from fast_fallback_model so both point at the small model.
    if not cfg.get("primus_fast_model") and cfg.get("fast_fallback_model"):
        cfg["primus_fast_model"] = cfg["fast_fallback_model"]
    elif not cfg.get("fast_fallback_model") and cfg.get("primus_fast_model"):
        cfg["fast_fallback_model"] = cfg["primus_fast_model"]
    if cfg.get("path_mode") not in ("primus_path", "user_path"):
        cfg["path_mode"] = "primus_path"
    return cfg
