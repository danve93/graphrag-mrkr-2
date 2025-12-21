# Changelog

All notable changes to the Amber codebase following the December 2024 Audit (Codebase Inspection & Remediation).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-12-21

### New Features

- **Docling Document Processing Integration**: Added optional Docling library support for enhanced document conversion and chunking. Docling provides state-of-the-art document understanding with layout analysis and structure preservation:
  - `vendor/docling_adapter.py`: Wrapper for Docling's DocumentConverter with markdown export
  - `core/docling_chunker.py`: DoclingHybridChunker adapter with token-aware chunking and heading path extraction
  - `ingestion/converters.py`: Integrated Docling as conversion provider supporting PDF, DOCX, PPTX, XLSX, HTML, and images
  - New `docling_hybrid` chunker strategy with configurable token budgets (target/min/max tokens, overlap)
  - Graceful fallback to native loaders when Docling library is unavailable
  - Configure via `DOCUMENT_CONVERSION_PROVIDER=docling` or leave as `auto` for marker/native selection

- **LLM Token Usage Metrics Panel**: Comprehensive analytics dashboard for monitoring and optimizing LLM costs across all operations:
  - SQLite-based persistence for all LLM calls (`core/llm_usage_tracker.py`, database at `data/llm_usage.db`)
  - Cost estimates with USD/EUR conversion (configurable exchange rate via `USD_TO_EUR_RATE`)
  - Multi-dimensional breakdowns: by operation type (ingestion, RAG, chat), by provider (OpenAI, Anthropic, Mistral, Ollama), by model, and by conversation
  - Time trends with daily and hourly aggregates for usage pattern analysis
  - Success rate monitoring with recent error tracking for debugging failed LLM calls
  - Efficiency metrics showing input/output token ratios and per-operation averages
  - Query records table with TruLens-style layout for detailed call inspection
  - Export functionality for full JSON reports
  - New API router at `/api/metrics/llm-usage/*` with 10+ endpoints
  - Frontend panel at Metrics → LLM Token Usage with auto-refresh every 30 seconds

- **Selective Database Clearing**: Improved safety and granularity for database cleanup operations:
  - New modal dialog (`frontend/src/components/Sidebar/ClearDatabaseDialog.tsx`) replacing single "Clear All" button
  - Separate toggles for Knowledge Base (documents, chunks, entities, communities, folders) and Conversation History (chats, messages)
  - Backend support for selective clearing via `POST /api/database/clear` with `clear_knowledge_base` and `clear_conversations` flags
  - Warning indicators and confirmation required before destructive operations
  - Prevents accidental data loss by requiring explicit selection

- **Google Gemini LLM Integration**: Added Google Gemini as a first-class LLM provider option:
  - Configuration via `GEMINI_API_KEY` (also accepts `GOOGLE_API_KEY` alias)
  - Model selection via `GEMINI_MODEL` (default: `gemini-3-flash-preview`)
  - Embedding support via `GEMINI_EMBEDDING_MODEL` (default: `models/text-embedding-004`)
  - Full integration in `core/llm.py` with provider routing
  - Available in all UI dropdowns (Chat Tuning, RAG Tuning, Metrics)
  - Set `LLM_PROVIDER=gemini` to use as default provider

- **HTML Heading Chunker**: New chunker strategy optimized for HTML documents with semantic structure:
  - `core/html_chunker.py`: Heading-aware chunker that preserves document hierarchy
  - Extracts heading paths (e.g., "Introduction > Background > Related Work") for better context
  - Chunk metadata includes `heading_path`, `section_title`, and structural information
  - Configure via `CHUNKER_STRATEGY_HTML=html_heading` (now the default for HTML files)
  - Optional heading path prefixing via `CHUNK_INCLUDE_HEADING_PATH=true`

- **Search UI Enhancement**: Improved discoverability and consistency across tuning panels:
  - Added fuzzy search functionality to Chat Tuning sidebar (matches RAG Tuning and Documentation patterns)
  - Added fuzzy search functionality to RAG Tuning sidebar
  - Consolidated search utilities in `frontend/src/lib/searchUtils.ts`
  - Search results highlight matching parameters with keyboard navigation support
  - Consistent UX across all configuration panels

### Infrastructure Improvements

- **Token Management Enhancements**: Comprehensive improvements to token counting and context management across the platform:
  - **`core/token_counter.py`**: New dedicated token counter utility with tiktoken integration
    - Provides `count()`, `encode()`, `decode()`, and `tail_text()` methods for precise token operations
    - Used by both HTML heading chunker and Docling hybrid chunker
    - Graceful fallback when tiktoken unavailable
  - **`core/token_manager.py`**: Enhanced context splitting and merging capabilities
    - Updated model context sizes for 2024-2025 models (GPT-5, Claude 4.5, Mistral, Gemini, Qwen, DeepSeek-R1)
    - Intelligent context splitting when requests exceed token limits
    - LLM-based response merging for multi-batch queries
    - Automatic truncation detection and continuation for cut-off responses
  - **`core/llm.py`**: Standardized token usage tracking across all LLM providers
    - All `generate_response()` methods now support `include_usage=True` parameter
    - Returns `{"content": str, "usage": {"input": int, "output": int}}` when enabled
    - Integration with `llm_usage_tracker` for automatic recording
    - Consistent token reporting for OpenAI, Anthropic, Mistral, Gemini, and Ollama
  - **Global OpenAI Patch**: Fixed `max_tokens` vs `max_completion_tokens` compatibility
    - Automatic parameter translation for modern vs legacy models
    - Prevents third-party library failures (e.g., Marker)
    - Retry logic with alternative parameter on 400 errors

### Configuration

- **TruLens Toggle Persistence**: TruLens evaluation state now persists across server restarts:
  - Toggle state saved to `config.yaml` when changed via UI
  - Defaults to disabled on fresh installations (opt-in for evaluation overhead)
  - Backend API at `/api/metrics/trulens/control` handles state management
  - Prevents unexpected evaluation costs on server restart

### UI Improvements

- **Bottom Panel Padding**: Applied consistent `pb-28` bottom padding across all scrollable panels to prevent content being hidden behind bottom dock:
  - `GraphView.tsx`, `NodeSidebar.tsx`: Graph panel and sidebar
  - RAG Tuning content filtering section
  - All panels with `overflow-y-auto` class
  - Ensures last items in scrollable lists are fully accessible

---

## [2.1.4] - 2024-12-19

### Bug Fixes
- **Precomputed Stats Not Persisting**: Fixed critical bug in `graph_db.py:1032` where `update_document_precomputed_summary()` was computing entity, community, and similarity counts but NOT persisting them to the database. Unreachable code (lines 1240-1265) was mistakenly placed after a `return` statement in `update_node_properties()`. Moved the persistence logic into the correct function, ensuring stats are now properly stored on Document nodes and displayed in the UI.
- **Orphaned Entities and Missing CONTAINS_ENTITY Relationships**: Cleaned up 857 orphaned entities from deleted documents and repaired 145 missing CONTAINS_ENTITY relationships linking entities to chunks. Documents now correctly display entity counts (e.g., Chap10.pdf: 75 entities, Chap1.pdf: 60 entities).
- **CONTAINS_ENTITY Relationships Not Created During Entity Extraction**: Fixed `_create_entity_node_sync()` at `graph_db.py:3024` to automatically create CONTAINS_ENTITY relationships when entities are added. Previously, entities were created with `source_chunks` property but the relationships were never created, causing entity counts to show as 0.
- **Clean Orphans Button Not Detecting Orphaned Entities**: Fixed `cleanup_orphaned_chunks()` at `graph_db.py:1945` to detect entities with non-empty `source_chunks` that reference deleted chunks. Previously only detected entities with empty/null source_chunks, missing 857 orphaned entities from deleted documents.

---

## [2.1.3] - 2024-12-19

### Bug Fixes
- **Chunk Progress Never Displayed**: Fixed chunk counts like "Extracting entities (123/456 chunks)" never showing during upload. `chunks_processed` and `total_chunks` were not being merged from progress state into document objects in DatabaseTab polling loop.
- **Chunk Progress Limited to 2 Stages**: Expanded chunk count display to show during ALL entity extraction phases (starting, llm_extraction, embedding_generation, database_operations, clustering, validation) instead of just `entity_extraction` and `llm_extraction`.
- **LLM Extraction Progress Not Updating**: Fixed progress appearing stuck at "llm_extraction (0/11)" during entity extraction. Entity extractor's `extract_from_chunks()` now receives progress callback to report chunk-by-chunk updates during LLM processing.
- **Entity/Relationship Counts Stay at 0 in DocumentView**: Fixed document detail view not updating entity and relationship counts during upload. Polling now refreshes `summaryData` (which contains these counts) in addition to `documentData`.
- **Relationships Count Includes All Relationships**: Fixed total relationships counting ALL graph relationships (HAS_CHUNK, CONTAINS_ENTITY, SIMILAR_TO, etc.) instead of only entity-to-entity RELATED_TO relationships. Changed query from `()-[r]->()` to `(e1:Entity)-[r:RELATED_TO]->(e2:Entity)`.
- **Unnamed Ghost Documents After Deletion**: Added fallback to show "Unnamed Document (abc123...)" when document filename is missing or empty, preventing blank entries in database sidebar.

---

## [2.1.2] - 2024-12-19

### Bug Fixes
- **Document Summary Stats Showing 0**: Fixed `/api/documents/{id}/summary` endpoint returning 0 for entities and communities when precomputed values were missing. Now computes missing stats individually via Neo4j queries.
- **Preview 404 Infinite Loop**: Fixed infinite HEAD requests to `/api/documents/{id}/preview` by changing useEffect dependency from entire `documentData` object to just `documentData.id`.
- **Chunk/Entity Count Display**: Fixed "0 entries" and "0 total" showing in document detail section headers when `documentData` arrays were empty, now correctly falls back to `summaryData.stats` values.
- **Entity Extraction Not Running on Upload**: Fixed document being marked "completed" immediately after starting background entity extraction, causing extraction to be orphaned. Now keeps status as "processing/entity_extraction" until worker finishes.

### Improvements
- **Real-time Knowledge Base Stats**: Added event listeners and 5-second polling to update dashboard stats (Documents, Chunks, Entities, Relationships) without requiring page reload.
- **Restart Resilience**: Smart resume on restart - documents with chunks resume entity extraction instead of failing. Direct uploads recovered from `data/staged_uploads` directory.

---
