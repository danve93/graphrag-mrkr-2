# Changelog

All notable changes to the Amber codebase following the December 2024 Audit (Codebase Inspection & Remediation).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2024-12-18

This release focuses on addressing 50 distinct issues identified during the December 2024 Codebase Audit, ranging from critical security vulnerabilities to logic flaws and stability improvements.

### Security
- **Static Admin Token Removal** (Issue #30): Removed the insecure `JOBS_ADMIN_TOKEN` fallback. Authentication now strictly enforces database-backed API keys.
- **API Key Tenant Isolation** (Issue #34): Enforced one active API key per user to prevent unlimited key generation.
- **Plaintext API Keys** (Issue #29): Implemented SHA-256 hashing for API keys in storage; keys are no longer stored in plaintext.
- **Admin Session Security** (Issue #28): Configured `Secure` and `HttpOnly` flags for admin session cookies.
- **Access Control Fix** (Issue #19): Fixed broken access control logic on `get_conversation` endpoint to prevent unauthorized chat history access.

### Logic & Correctness
- **Retrieval Return Type** (Issue #1): Fixed inconsistent return type in synchronous `retrieve_documents()` wrapper (returning list vs tuple).
- **Progress Calculation** (Issue S3): Fixed "jumping" progress bars by interpolating progress across classification, chunking, and summarization stages.
- **Orphan Node Detection** (Issue #16/36): Improved algorithm to detect and clean small disconnected components (micro-clusters).
- **Metadata Merging** (Issue #14): Changed `update_document_metadata` to use map merging instead of complete overwrite.
- **Configurable Regex** (Issue #35): Moved hardcoded technical patterns from `query_analysis.py` to `settings.py`.
- **Async Event Loop** (Issue #12): Fixed unsafe `asyncio.run()` nesting in `_create_entities_async`.
- **Community Detection** (Issue #33): Replaced Neo4j GDS dependency with `igraph` for community detection.
- **Follow-up Temperature** (Issue #23): Lowered LLM temperature for more consistent follow-up question classification.

### Stability & Reliability
- **In-Memory State** (Issue #2): Refactored processing state to persist in Neo4j, preventing data loss on restart.
- **Distributed Cleanup** (Issue #13): Added distributed-safe checks for stale job cleanup.
- **Settings Synchronization** (Issue S1, S2, S7, S8):
    - Mapped missing RAG tuning parameters.
    - Aligned LLM override key names.
    - Synced `default_llm_model` overrides.
    - Synced Static Matching thresholds.
- **Singleton Configuration** (Issue S5): Implemented dynamic reconfiguration for Singletons when settings change at runtime.
- **Error Handling** (Issue #4): Added better error tracking for entity extraction failures.
- **Race Condition** (Issue #5): Improved locking mechanisms in processing queue.
- **Cancellation Propagation** (Issue #6): Enhanced signal handling to propagate cancellation and deletion events to file system.
- **Cache Initialization** (Issue #11): Removed duplicated/unreachable cache init code.

### Code Quality & Maintenance
- **Duplicate Settings** (Issue #3): Removed duplicate field definitions in `config/settings.py`.
- **Standardized Errors** (Issue #7): Unified API error response formats (JSON schema).
- **Magic Numbers** (Issue #9): Extracted hardcoded values to named constants/settings.
- **Dead Code** (Issue #21): Removed unreachable exception blocks in `category_manager.py`.
- **Deprecated API** (Issue #22): Replaced Pydantic v1 `.dict()` with v2 `.model_dump()`.
- **Inconsistent Retry** (Issue #17): Centralized retry configuration across services.
- **Code Duplication** (Issue #25, #39, S6): Refactored duplicate logic in generation, graph restore, and RAG overrides.
- **Memory Leaks** (Issue #8, #18, #20):
    - Fixed retrieval metadata retention in singletons.
    - Added bounds to routing cache.
    - Added bounds to adaptive router feedback history.

### Pipeline Improvements
- **Chat Query Flow**:
    - Improved fallback handling to notify users when graph expansion fails.
    - Added cleanup for SSE connections.
- **Document Ingestion**:
    - Added orphan file cleanup on startup.
    - Added timeouts for long-running extraction calls.
- **Graph Operations**:
    - (Issue #24) Optimized entity linking using vector search instead of python-side cosine similarity.
    - (Issue #37) Switched to stable UUIDs for graph export instead of internal IDs.
    - (Issue #38) Fixed property shadowing during graph restore.
    - (Issue #15) Increased/configured description truncation limits for `heal_node`.
- **Documentation**:
    - (Issue #31) Restored/Updated missing User Panel source files.
    - (Issue #32) Fixed broken imports in admin user management.

### API Changes
- **Pagination** (Issue #40): Added `limit` and `offset` parameters to Admin list endpoints.
- **Input Validation** (Issue #10, #27): Added validation for hashtags and category updates.

---

## [2.1.1] - 2024-12-18

Post-audit hotfixes and feature additions.

### Bug Fixes
- **Dead Code Removal**: Removed orphaned `_load_admin_token()` call in `auth.py` that referenced a deleted function (from Issue #30), causing 500 errors on authenticated endpoints.
- **Orphaned Chunks Fix**: Fixed RAG retrieval returning 0 results due to 1889 orphaned chunks (85% of total) not connected to Document nodes via `HAS_CHUNK` relationship.
- **Document Update Progress Reporting**: Fixed progress bar stuck at 5% during document updates by:
  - Adding `processing_status: "processing"` to all update phases (conversion, diffing, embedding, entity extraction)
  - Integrating document updates with the global processing state system to trigger frontend polling
  - Adding Neo4j fallback in `get_database_stats` to read processing status for background updates
  - Resetting global processing state when document update completes
- **Redundant UI Banner**: Removed the duplicate "Processing in progress..." indicator block from DatabaseTab since per-document progress bars provide sufficient feedback.
- **HTTP Compatibility for External Chat**: Added UUID generation fallback in `ExternalChatBubble.tsx` to support non-HTTPS deployments where `crypto.randomUUID()` is unavailable.

### New Features
- **Automatic Orphan Cleanup on Startup**: Added automatic cleanup of orphaned chunks and entities on backend startup.
  - Configurable via `ENABLE_ORPHAN_CLEANUP_ON_STARTUP` (default: true)
  - Configurable grace period via `ORPHAN_CLEANUP_GRACE_PERIOD_MINUTES` (default: 5 minutes)
  - Only deletes orphans older than the grace period to avoid race conditions with in-progress ingestion
- **Manual Orphan Cleanup API**: Added `POST /api/database/cleanup-orphans` endpoint for manual cleanup (no grace period).
- **Cleanup Button in UI**: Added "Cleanup" button in Database panel toolbar to trigger manual orphan cleanup with confirmation dialog.

