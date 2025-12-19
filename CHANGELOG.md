# Changelog

All notable changes to the Amber codebase following the December 2024 Audit (Codebase Inspection & Remediation).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
