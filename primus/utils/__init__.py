"""utils/ — shared utilities extracted from admin_assistant.py.

Migrated in Phase 3:
  gpu.py      AMD GPU detection / acceleration env / status (reads live CFG via a provider)
  patches.py  gradio localhost + schema patches, no-proxy localhost helper, free-port finder
  text.py     dependency-free text/filename helpers (safe_filename)

Other low-level helpers (logging setup, filesystem/text/formatting) still live in admin_assistant.py
and can move here incrementally. Nothing is locked or read-only.
"""
from . import gpu, patches, text  # noqa: F401
from .text import safe_filename  # noqa: F401
