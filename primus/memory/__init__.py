"""memory/ — Primus's memory + RAG system (migrated in Phase 5).

``system.py`` holds the full implementation; this re-exports the singleton accessors, KB/RAG
helpers, and constants that the rest of admin_assistant references directly.
"""
from .system import (
    KB_CATEGORY_CHOICES,
    KB_CATEGORY_LABELS,
    KB_CATEGORY_TO_COLLECTION,
    KB_PROJECT_CHOICES,
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

__all__ = [
    'KB_CATEGORY_CHOICES', 'KB_CATEGORY_LABELS', 'KB_CATEGORY_TO_COLLECTION', 'KB_PROJECT_CHOICES', '_extract_docx_text', '_extract_pdf_text', '_extract_pptx_text', '_extract_xlsx_text', 'build_conversation_focus', 'build_memory_context', 'detect_memory_tags', 'get_kb', 'get_memory_system', 'get_metrics', 'handle_kb_command', 'index_projects', 'ingest_uploaded_files', 'init_knowledge_base', 'knowledge_status_markdown', 'load_document_text', 'load_memory', 'maybe_store_interaction', 'prepare_agent_messages', 'render_kb_dashboard', 'save_memory', 'start_background_index',
]
