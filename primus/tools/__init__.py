"""tools/ — Primus's tool system (migrated in Phase 4).

``registry.py`` holds the full implementation; this re-exports ``build_tools()`` plus the tool/class
symbols that the rest of admin_assistant references directly, so call sites are unchanged.
"""
from .registry import (
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

__all__ = [
    'SelfImprovementLog', 'SelfImprovementMemory', '_SITE_HOMES', '_infer_lesson_category', '_verify_python_text', 'analyze_self', 'apply_pending_self_edit', 'browse_web', 'build_tools', 'connections_add_ui', 'connections_health_line', 'connections_refresh_markdown', 'connections_remove_ui', 'connections_status_markdown', 'desktop_control', 'device_management', 'disk_cleanup', 'download_pdf_from_url', 'download_webpage_as_pdf', 'get_news', 'get_weather', 'inbox_connect_ui', 'inbox_draft_reply_ui', 'inbox_list_ui', 'inbox_read_ui', 'inbox_reply_prefill_ui', 'inbox_setup_markdown', 'inbox_status_markdown', 'ingest_web_document', 'open_application', 'read_article', 'search_own_codebase', 'search_reddit', 'system_monitor',
]
