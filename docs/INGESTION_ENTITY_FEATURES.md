# Document Ingestion & Entity Extraction — Features & Implementation

This document describes the recent additions and fixes to the GraphRAG ingestion
pipeline and entity-extraction subsystem. It is intended as a concise reference
for developers and operators.

## Summary of Changes

- Fix: OpenAI base URL handling and client configuration (ensure trailing slash)
  - File: `core/llm.py`
  - Symptom: Malformed request paths like `/v1chat/completions`.
  - Resolution: Set explicit default base URL `https://api.openai.com/v1/` when
    none provided and validated in `.env`.

- Feature: Format-specific converters (Marker-style PDF OCR → Markdown)
  - File: `ingestion/converters.py` (DocumentConverter)
  - Behavior: PDF loader uses a marker-style OCR pipeline to produce
    Markdown with `## Page N` headers and rich conversion metadata
    (e.g. `conversion_pipeline: marker_ocr_markdown`).

- Feature: Processing stages & document-linked metadata
  - File: `ingestion/document_processor.py`
  - Behavior: Pipeline updates document nodes as it progresses:
    `conversion` → `chunking` (25%) → `summarization` (60%) → `embedding` (75%) → `completed`.
  - Stores `processing_status`, `processing_stage`, `processing_progress` and timestamps.

- Feature: Entity extraction helpers, persistence, metrics, and retry/backoff
  - File: `ingestion/document_processor.py` (helpers: `_prepare_chunks_for_extraction`,
    `_persist_extraction_results`, `_build_extraction_metrics`)
  - File: `core/entity_extraction.py` (LLM extraction + `retry_with_exponential_backoff`)
  - Behavior:
    - Prepare chunk payloads for extraction, preferring freshly-processed
      chunks but falling back to database-chunks when needed.
    - Consolidate entity/relationship persistence into `_persist_extraction_results`
      which runs async creation of entity nodes and relationships, then computes
      lightweight extraction metrics and persists them to the document node
      under `entity_extraction_metrics`.
    - LLM calls use an exponential-backoff retry wrapper (defaults: 5 tries,
      base 3s, max 180s) and are offloaded to a thread executor to avoid
      blocking the event loop.
    - Validation steps (`validate_entity_embeddings`, `fix_invalid_embeddings`) run
      after persistence to ensure embedding quality.

- Feature: Unified extraction flows
  - The following flows reuse the same helpers and persistence pipeline:
    - `extract_entities_for_document` (synchronous, single document)
    - `process_batch_entities` (batch worker)
    - `extract_entities_for_all_documents` (global worker)

## Where to look in code

- Converters: `ingestion/converters.py`
- Document processing: `ingestion/document_processor.py`
  - `_prepare_chunks_for_extraction`
  - `_persist_extraction_results`
  - `_build_extraction_metrics`
  - `process_file`, `process_file_async`, `extract_entities_for_document`
- Entity extraction core: `core/entity_extraction.py`
  - `extract_from_chunk`, `extract_from_chunks`
  - `retry_with_exponential_backoff`
- Graph operations: `core/graph_db.py`

## API changes and behavior

- Upload endpoint: `POST /api/database/upload`
  - Returns `DocumentUploadResponse` with fields:
    - `processing_status` (e.g. `processing` / `completed` / `error`)
    - `processing_stage` (e.g. `chunking`, `summarization`, `embedding`, `completed`)
    - `processing_progress` (local progress is also available via `/api/database/stats`)

- Stage endpoint: `POST /api/database/stage` returns stage info and enqueues
  background processing.

## Metrics persisted to document node

When entity persistence completes (or is attempted), a compact metrics payload is
created and stored on the document node with key `entity_extraction_metrics`. Key
fields include:

- `chunks_processed` — how many chunks were analyzed
- `entities_created` — number of entity nodes created for the document
- `relationships_created` — number of relationships created
- `relationships_requested` — number of relationship requests returned by the LLM
- `chunk_coverage` — fraction/percentage of chunks that referenced at least one entity
- `entities_per_chunk` — average entities per chunk
- `unique_chunks_with_entities` — count of unique chunks with ≥1 entity

These metrics are used for monitoring extraction quality and provenance.

## How to test locally (quick commands)

1. Ensure environment and services running (Neo4j, backend):

```bash
# Start services (from repo root)
docker compose up -d --build

# Check API health
curl -s http://localhost:8000/api/health | jq .
```

2. Upload a sample document and inspect response:

```bash
curl -s -X POST http://localhost:8000/api/database/upload -F "file=@/path/to/sample.pdf" | jq .
```

Response should include `processing_status` and `processing_stage` fields.

3. Inspect document metrics from DB or API stats:

```bash
curl -s http://localhost:8000/api/database/stats | jq '.documents[] | select(.document_id=="<DOCUMENT_ID>")'
```

4. Trigger global entity extraction (optional):

```bash
# Runs the global extraction worker
python -c "from ingestion.document_processor import document_processor; print(document_processor.extract_entities_for_all_documents())"
```

## Testing notes & observations

- The OpenAI base URL fix was required to get embeddings and entity LLM calls
  working. Ensure `.env` has `OPENAI_BASE_URL=https://api.openai.com/v1/` (trailing slash).
- Entity extraction runs in background threads in several flows; when testing
  look at backend logs to follow background operations.
- If entity extraction seems to be missing, check that `ENABLE_ENTITY_EXTRACTION`
  (or equivalent) is enabled in `config/settings.py` or environment variables.

## Recommendations

- Monitor `entity_extraction_metrics` for documents to identify low-coverage
  documents that may benefit from OCR adjustments or chunking tuning.
- Introduce a periodic job to re-run extraction on documents with low coverage
  after OCR or chunker improvements.
- Add a small UI panel that surfaces `chunk_coverage` and top problematic chunks
  to help triage noisy or scanned documents.

## Changelog (recent)

- 2025-11-24: Added converters, processing-stage propagation, extraction helpers,
  unified flows, backoff/retry, metrics persistence. Fixed OpenAI base URL bug.

---

If you want this doc copied into other formats (e.g. a release note or a
developer-facing checklist) I can generate those as well.
