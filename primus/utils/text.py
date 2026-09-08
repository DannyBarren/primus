"""Small, dependency-free text/filename helpers shared across the package.

Extracted so both the memory layer (knowledge-base uploads) and the tools layer (web downloads /
ingestion) can sanitize filenames the same way without depending on a host re-export.
"""
from __future__ import annotations

import os
import re


def safe_filename(name: str, *, default: str = "document", max_len: int = 120) -> str:
    """Sanitize an arbitrary string into a safe, basename-only filename.

    Strips any directory components, replaces characters outside ``[\\w.\\- ]`` with ``_``, trims
    surrounding whitespace, falls back to ``default`` when empty, and caps the length. Behavior
    matches the original ``_safe_filename`` from the monolithic ``admin_assistant.py``.
    """
    name = os.path.basename(name or default)
    name = re.sub(r"[^\w.\- ]+", "_", name).strip() or default
    return name[:max_len]
