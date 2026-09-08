"""self_improve/ — self-analysis, code-change proposals, and self-improvement memory.

Implemented, but not a separate module: the self-improvement layer ships inside ``primus/tools/``
(``analyze_self``, ``propose_new_tool``/``propose_code_change``, ``apply_pending_self_edit``,
``search_own_codebase``, plus ``SelfImprovementLog`` and ``SelfImprovementMemory``). The human
approval gate (``/approve edit`` / ``/reject edit``) is unchanged. This package is kept as a
documented placeholder; there is intentionally nothing to import here.
"""
