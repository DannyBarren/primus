"""Primus — a local-first executive assistant (modular package).

This package is the in-progress modular home for the Primus application that currently
lives in the single file ``admin_assistant.py``. Modules are extracted incrementally and
imported back into ``admin_assistant.py`` so behavior stays 100% identical at every step.

Layout:
  core/         directives, personality, prompts, command guardrails        [done]
  config.py     env-overridable paths + DEFAULT_CONFIG + load/save logic     [done]
  utils/        gpu.py (AMD GPU accel) + patches.py (gradio/launch); more to come  [partial]
  memory/       system.py — LTM + KnowledgeBase/RAG + MemorySystem + metrics  [done]
  tools/        registry.py — full tool system (68+ @tools + build_tools)    [done]
  agents/       system.py — routing/graphs, fast paths, invoke_primus,
                watchdog, background + scheduled agents                      [done]
  ui/           builder.py — the entire Gradio UI (theme, tabs, panels,
                handlers, voice, fallback server)                           [done]
  self_improve/ placeholder — self-improvement ships inside tools/           [folded in]

``admin_assistant.py`` is the host module + entrypoint: it imports every subsystem back and keeps
the shared host surface (config/CFG + model globals, PrimusSession, chat/projects, web/shell/system
helpers, desktop window/tray/sound) plus launch orchestration (main). Modules reach host globals via
a live ``_host`` reference, so call sites are unchanged and behavior is 100% identical.

Nothing here is locked or read-only — every directive remains freely editable. Run with
``uv run python admin_assistant.py``.
"""
__version__ = "0.1.0"
