UI Reorganization and Runtime Configuration
===========================================

Table of contents
-----------------
- Scope and Objectives
- High-level Changes
  - Backend
  - Frontend
- Why these changes
- Configuration File Schemas
- API Endpoints (Backend)
- Frontend Integration Details
- Reindex Endpoint Implementation (Technical Details)
- Testing and Verification
- Quick local run commands
- Where to look for code
- Design decisions and rationale
- Limitations and Next Steps
- Contact and Maintenance Notes
- Document history

This document describes the design and implementation details of the UI reorganization and runtime configuration work completed across backend and frontend. It covers the goals, architecture decisions, file locations, API contract, frontend changes, reindex implementation, testing, and next steps.

Scope and Objectives
--------------------
- Centralize configuration (classification + chat tuning) into runtime-editable JSON files.
- Provide backend APIs to read and update configurations at runtime.
- Rework the UI to expose dedicated panels for Chat Tuning and Classification management.
- Remove advanced settings embedded in the chat input and load parameters centrally from the Chat Tuning API.
- Add a safe, explicit reindex endpoint to re-run entity extraction and graph clustering across all documents.

High-level Changes
------------------
1. Backend
   - New JSON config files created in `config/`:
     - `classification_config.json` — entity types, overrides, relationship suggestions, low-value regex patterns, leiden parameters.
     - `chat_tuning_config.json` — chat tuning parameters with metadata (label, type, min/max/step, category, tooltip).
   - New FastAPI routers:
     - `api/routers/classification.py` — GET/POST for classification config and POST `/reindex` to trigger reindex.
     - `api/routers/chat_tuning.py` — GET/POST for chat tuning config and GET `/config/values` for compact key/value pairs.
   - `core/entity_extraction.py` updated to load classification config from JSON (with backward-compatible fallback to previous hardcoded mappings).
   - `core/graph_clustering.py` updated to use `resolution` parameter (fixing deprecated API usage).

2. Frontend
   - Reorganized navigation in `frontend/src/app/page.tsx`: top-level buttons changed to `Graph`, `Chat Tuning`, `Classification`.
   - Sidebar (`frontend/src/components/Sidebar/Sidebar.tsx`) simplified: `History` renamed to `Chat`, `Graph` removed from sidebar tabs.
   - New UI components:
     - `frontend/src/components/ChatTuning/ChatTuningPanel.tsx` — full editor for chat tuning parameters grouped by category, with sliders/toggles, tooltips, save/reset states.
     - `frontend/src/components/Classification/ClassificationPanel.tsx` — editor with 4 tabs: `Entity Types`, `Type Overrides`, `Relationships`, `Leiden Config`; includes `Update & Reindex` button and confirmation modal.
   - `ChatInterface` simplified to remove advanced inline settings and instead load parameters from `GET /api/chat-tuning/config/values`.
   - `frontend/src/store/chatStore.ts` updated to include new `ActiveView` options (`chatTuning`, `classification`).

Why these changes
------------------
- Centralizing configuration in JSON enables runtime editing without code changes and facilitates UI management for non-developers.
- Moving advanced settings into a dedicated panel clarifies the chat UI for users and surfaces explanations/tooltips for each tuning parameter.
- The reindex operation is destructive and time-consuming; implementing it on the backend and exposing it behind a confirmation modal ensures safe, auditable reapplication of configuration changes.

Configuration File Schemas
--------------------------
1. `config/classification_config.json`
   - entity_types: string[]
   - entity_type_overrides: { [raw_text: string]: canonical_type }
   - relationship_suggestions: string[]
   - low_value_patterns: string[] (regexes)
   - leiden_parameters: { resolution: number, min_edge_weight: number, relationship_types: string[] }

2. `config/chat_tuning_config.json`
   - parameters: [
       {
         key: string,
         label: string,
         value: number | boolean,
         min?: number,
         max?: number,
         step?: number,
         type: 'slider' | 'toggle',
         category: string,
         tooltip: string
       }
     ]

API Endpoints (Backend)
-----------------------
Note: all new endpoints are prefixed under `/api`.

Classification Router (`/api/classification`)
- GET `/config` — returns full classification configuration as JSON (Pydantic model `ClassificationConfig`).
- POST `/config` — accepts `ClassificationConfig` JSON to persist to `config/classification_config.json`.
- POST `/reindex` — accepts `{ "confirmed": true }`; triggers the reindex workflow (detailed below). Requires careful use and UI confirmation.

Chat Tuning Router (`/api/chat-tuning`)
- GET `/config` — returns detailed parameters with UI metadata (labels, tooltips, types).
- POST `/config` — updates and persists chat tuning JSON.
- GET `/config/values` — returns compact key/value pairs (useful for `ChatInterface` to load param values quickly).

Frontend Integration Details
----------------------------
- `ChatTuningPanel` reads the full parameter definitions and shows groups by `category`. Each slider/toggle maps to the parameter's `value` field and calls POST `/api/chat-tuning/config` when the user saves.
- `ClassificationPanel` provides CRUD for:
  - `Entity Types` (add/remove canonical types),
  - `Type Overrides` (map raw strings to canonical types),
  - `Relationship Suggestions` (list management),
  - `Leiden Config` (resolution and minimum edge weight sliders, editable relationship types list).
  The panel saves via POST `/api/classification/config` and exposes `Update & Reindex` which calls POST `/api/classification/reindex` after a UI confirmation modal.
- `ChatInterface` loads compact tuning values on mount from `/api/chat-tuning/config/values` and uses those values when calling the chat API. The inline advanced settings UI was removed.

Reindex Endpoint Implementation (Technical Details)
--------------------------------------------------
The reindex endpoint performs the following steps inside a single server request (safely logged and guarded):

1. Validate confirmation payload and that `settings.enable_entity_extraction` is true.
2. Query Neo4j for all documents: `MATCH (d:Document) RETURN d.id as doc_id, d.filename as filename`.
3. For each document, call `graph_db.reset_document_entities(doc_id)`:
   - This clears `CONTAINS_ENTITY` relationships from chunks.
   - Deletes `RELATED_TO` relationships sourced from the affected chunks.
   - Cleans up `source_chunks` arrays on Entity nodes and deletes orphaned entities.
   - Resets per-document extraction metadata (if present).
4. Instantiate `DocumentProcessor` and call `extract_entities_for_all_documents()` which:
   - Initializes the `EntityExtractor`.
   - Iterates documents lacking entities and runs LLM-based extraction on chunks.
   - Generates embeddings and persists entities/relationships via `_persist_extraction_results`.
   - Returns extraction statistics (processed_documents, created_entities, created_relationships, etc.).
5. Call `graph_db.create_all_entity_similarities()` to create SIMILAR_TO relationships based on embeddings.
6. Attempt to run Leiden clustering (`core.graph_clustering.run_leiden_clustering()`); failures are logged and do not abort the operation.
7. Return a summary JSON with documents processed, entities cleared, and extraction result details.

Important safety notes:
- The endpoint requires `{ "confirmed": true }` to proceed.
- It is destructive relative to entity/relationship data; document nodes and chunk text remain intact.
- The endpoint currently runs synchronously inside the API call. For large installations this should be turned into an asynchronous background job with progress reporting (SSE/WebSocket or a job queue).

Testing and Verification
------------------------
- Backend tests (pytest) currently pass locally except for tests that require a running Neo4j instance. Two ingestion tests failed where Neo4j was not reachable; this is expected if Neo4j is not running.
- Frontend builds successfully (`npm run build`) and compiles with no TypeScript errors.

Quick local run commands
------------------------
Backend (dev):

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

Frontend (dev):

```bash
cd frontend
npm install
npm run dev
```

Test endpoints (example):

```bash
# Read classification config
curl -s http://localhost:8000/api/classification/config | jq

# Read chat tuning values
curl -s http://localhost:8000/api/chat-tuning/config/values | jq

# Trigger a reindex (BE CAREFUL)
curl -X POST -H 'Content-Type: application/json' \
  -d '{"confirmed": true}' \
  http://localhost:8000/api/classification/reindex
```

Where to look for code
----------------------
- Backend routers:
  - `api/routers/classification.py`
  - `api/routers/chat_tuning.py`
- Config files:
  - `config/classification_config.json`
  - `config/chat_tuning_config.json`
- Entity extraction + ingestion:
  - `core/entity_extraction.py`
  - `ingestion/document_processor.py`
- DB layer:
  - `core/graph_db.py`
- Frontend UI:
  - `frontend/src/components/ChatTuning/ChatTuningPanel.tsx`
  - `frontend/src/components/Classification/ClassificationPanel.tsx`
  - `frontend/src/components/Chat/ChatInterface.tsx`
  - `frontend/src/components/Sidebar/Sidebar.tsx`
  - `frontend/src/app/page.tsx`

Design decisions and rationale
-----------------------------
- Use JSON for configuration to keep editing simple and human-readable; UI performs validation before persisting.
- Expose both detailed config (with labels/tooltips) and compact key/value endpoints to avoid sending metadata repeatedly to performance-sensitive components.
- Reindex operation kept on the backend because it requires direct database access and heavy CPU/IO; the frontend only triggers it after explicit user confirmation.
- The system retains backward compatibility: `core/entity_extraction.py` falls back to previous hardcoded defaults if config files are missing.

Limitations and Next Steps
--------------------------
- Reindex currently runs synchronously in the request. For production-scale datasets, convert to a background job with progress reporting and optional cancellation.
- Add RBAC or authenticated checks around destructive operations (reindex/clear) to avoid accidental misuse.
- Add optimistic locking/versioning for configuration updates to prevent races between multiple UI users.
- Expand tests to exercise the new routers and reindex path using a test Neo4j fixture or Docker-based test environment.

Contact and Maintenance Notes
----------------------------
- When updating `classification_config.json`, be mindful that changes only affect new ingestion or after a reindex. The `Update & Reindex` button is the recommended workflow to apply changes globally.
- Keep `chat_tuning_config.json` concise; each parameter must include `tooltip` to help non-technical users understand effects.

Document history
----------------
- Created: 25 November 2025
- Author: automated changes by development workflow (see commit history for individual file diffs)

