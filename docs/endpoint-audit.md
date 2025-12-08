# Backend Endpoint Audit

This document summarizes the HTTP endpoints exposed by the FastAPI backend as of the current audit. Paths are grouped by router/prefix and include purpose notes based on the in-code docstrings and behavior.

## Core application routes

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Root helper returning service name and links. |
| GET | `/api/health` | Health check with version, feature flags, and FlashRank prewarm status. |
| GET | `/api/admin/prewarm-status` | Operational status for FlashRank prewarming (started/completed/error). |

## Chat (`/api/chat`)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/chat/query` | Runs the RAG pipeline for a chat query; can stream or return full response with sources, metadata, and follow-up suggestions. |
| POST | `/api/chat/stream` | Always-streaming SSE endpoint for chat responses using current tuning defaults. |
| POST | `/api/chat/follow-ups` | Generates follow-up questions from the provided query/response context. |

## Chat tuning (`/api/chat-tuning`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/chat-tuning/config` | Retrieves saved chat tuning parameter values. |
| POST | `/api/chat-tuning/config` | Persists chat tuning configuration. |
| GET | `/api/chat-tuning/config/values` | Returns effective tuning defaults merged with environment settings. |

## RAG tuning (`/api/rag-tuning`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/rag-tuning/config` | Fetches stored retrieval/generation tuning config. |
| POST | `/api/rag-tuning/config` | Updates tuning config (weights, toggles, limits). |
| GET | `/api/rag-tuning/config/values` | Returns resolved tuning values with defaults applied. |

## Classification (`/api/classification`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/classification/config` | Reads the entity/relationship classification configuration. |
| POST | `/api/classification/config` | Updates classification configuration on disk. |
| POST | `/api/classification/reindex` | Triggers full reindex (clear + re-extract + recluster entities); queued as background job. |
| GET | `/api/classification/reindex/{job_id}` | Retrieves status for a classification reindex job. |
| POST | `/api/classification/categories/generate` | Generates category suggestions from documents. |
| POST | `/api/classification/categories` | Creates new categories. |
| GET | `/api/classification/categories` | Lists categories and usage metadata. |
| PATCH | `/api/classification/categories/{category_id}` | Updates a category. |
| GET | `/api/classification/categories/{category_id}` | Fetches details for a category. |
| POST | `/api/classification/categories/{category_id}/approve` | Approves pending/auto-generated categories. |
| DELETE | `/api/classification/categories/{category_id}` | Deletes a category. |
| POST | `/api/classification/categories/classify` | Classifies documents into categories (batch). |
| POST | `/api/classification/categories/auto-assign` | Auto-assigns categories to documents. |
| POST | `/api/classification/documents/{document_id}/categories/{category_id}` | Manually assigns a category to a document. |

## Database & ingestion (`/api/database`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/database/cache-stats` | Reports cache performance metrics. |
| GET | `/api/database/routing-metrics` | Returns query-routing metrics. |
| GET | `/api/database/stats` | Database stats with document summaries and processing state. |
| POST | `/api/database/upload` | Uploads a document for processing. |
| DELETE | `/api/database/documents/{document_id}` | Deletes a document (if enabled). |
| POST | `/api/database/clear` | Clears all documents and related data. |
| GET | `/api/database/documents` | Lists documents with status and metadata. |
| GET | `/api/database/hashtags` | Returns hashtag summaries across documents. |
| POST | `/api/database/stage` | Stages an existing file for processing (without upload). |
| GET | `/api/database/staged` | Lists staged files awaiting processing. |
| DELETE | `/api/database/staged/{file_id}` | Removes a staged file. |
| POST | `/api/database/process` | Starts processing for staged files. |
| GET | `/api/database/progress/{file_id}` | Progress for a specific staged upload. |
| GET | `/api/database/progress` | Progress snapshot for all uploads plus global state. |
| POST | `/api/database/documents/{document_id}/process/chunks` | Reprocesses chunking for an existing document. |
| POST | `/api/database/documents/{document_id}/process/entities` | Re-runs entity extraction for an existing document. |

## Document details (`/api/documents`)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/documents/{document_id}/generate-summary` | Generates a document summary. |
| GET | `/api/documents/{document_id}/summary` | Retrieves stored summary for a document. |
| GET | `/api/documents/{document_id}/entities` | Lists entities linked to a document. |
| GET | `/api/documents/{document_id}/similarities` | Returns similarity metrics for document chunks/entities. |
| POST | `/api/documents/{document_id}/hashtags` | Generates hashtags for a document. |
| GET | `/api/documents/{document_id}` | Metadata for a document. |
| GET | `/api/documents/chunks/{chunk_id}` | Retrieves a specific chunk (content and metadata). |
| GET/HEAD | `/api/documents/{document_id}/preview` | Streams or checks availability of the document preview. |
| GET | `/api/documents/{document_id}/chunks` | Lists chunks for a document. |
| GET | `/api/documents/{document_id}/entity-summary` | Returns a summary focused on document entities. |

## Graph visualization (`/api/graph`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/graph/clustered` | Clustered graph JSON with optional filters (community, type, document, limit). |
| GET | `/api/graph/communities` | Community listings for a given clustering level. |

## History (`/api/history`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/history/sessions` | Lists conversation sessions. |
| POST | `/api/history/sessions` | Creates a new conversation session. |
| GET | `/api/history/{session_id}` | Full message history for a session. |
| POST | `/api/history/{session_id}/messages` | Appends a message to session history. |
| DELETE | `/api/history/{session_id}` | Deletes a session. |
| PATCH | `/api/history/{session_id}/restore` | Restores a previously deleted session. |
| POST | `/api/history/clear` | Clears all conversations. |
| DELETE | `/api/history/{session_id}/messages/{message_id}` | Deletes a specific message. |
| GET | `/api/history/search` | Searches messages across sessions. |

## Jobs (`/api/jobs`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/jobs` | Lists background jobs. |
| GET | `/api/jobs/token` | Returns persistent user token (admin-only). |
| GET | `/api/jobs/{job_id}` | Job detail view. |
| POST | `/api/jobs/pause` | Pauses job processing. |
| POST | `/api/jobs/resume` | Resumes job processing. |
| POST | `/api/jobs/requeue` | Requeues stuck jobs. |
| POST | `/api/jobs/{job_id}/cancel` | Cancels a job. |
| POST | `/api/jobs/cancel-all` | Cancels all jobs. |
| POST | `/api/jobs/purge` | Purges all jobs and queue data. |
| POST | `/api/jobs/{job_id}/retry` | Retries a job. |

## Prompts (`/api/prompts`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/prompts/` | Lists all prompt templates. |
| GET | `/api/prompts/{category}` | Retrieves templates for a category. |
| POST | `/api/prompts/` | Creates a new prompt template. |
| PUT | `/api/prompts/{category}` | Replaces templates for a category. |
| DELETE | `/api/prompts/{category}` | Removes templates for a category. |
| POST | `/api/prompts/reload` | Reloads prompt templates from disk. |

## Structured KG (`/api/structured-kg`)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/structured-kg/execute` | Executes a structured knowledge-graph query. |
| GET | `/api/structured-kg/config` | Returns structured KG configuration/thresholds. |
| GET | `/api/structured-kg/schema` | Returns available schema/types for structured queries. |
| POST | `/api/structured-kg/validate` | Validates a structured KG query definition. |

## Feedback (`/api` prefix)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/feedback` | Submits user feedback on chat responses. |
| GET | `/api/feedback/metrics` | Aggregated feedback metrics. |
| GET | `/api/feedback/weights` | Current weighting configuration for feedback signals. |
| POST | `/api/feedback/reset` | Resets feedback metrics/weights. |
| GET | `/api/feedback/recent` | Recent feedback items. |

## Documentation (`/api/documentation`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/documentation/` | Lists available documentation files. |
| GET | `/api/documentation/{file_path}` | Serves a documentation file from the `documentation/` directory. |

## Graph documentation routers

The audit reflects routers registered in `api/main.py` at the time of review. Any future router additions should be appended here for completeness.
