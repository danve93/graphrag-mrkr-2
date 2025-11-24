## Project Overview
GraphRAG v2.0 is a state-of-the-art document intelligence system powered by graph-based RAG (Retrieval-Augmented Generation) with LangGraph, FastAPI, Next.js, and Neo4j.

## Architecture

### Core Pipeline (LangGraph State Machine)
The RAG pipeline in `rag/graph_rag.py` uses LangGraph's StateGraph with 4 sequential nodes:
1. **Query Analysis** (`query_analysis`) - Analyzes user intent and extracts structured query parameters
2. **Retrieval** (`retrieval`) - Fetches chunks via hybrid search (vector similarity + graph expansion)
3. **Graph Reasoning** (`graph_reasoning`) - Traverses relationships to enrich context and handle multi-hop queries
4. **Generation** (`generation`) - Streams LLM response with quality scoring

**State Management**: Uses plain dict-based state (not custom classes) that flows through all nodes. Each node appends to `state["stages"]` for progress UI updates.

### Ingestion Pipeline
`ingestion/document_processor.py` handles multi-format documents with:
- **Format Loaders** (PDF, DOCX, TXT, PPTX, XLSX, CSV, Images) in `ingestion/loaders/`
- **OCR** - Applied intelligently (enabled by default, quality-filtered)
- **Chunking** (`core/chunking.py`) - Configurable size/overlap with metadata preservation
- **Entity Extraction** (`core/entity_extraction.py`) - Optional LLM-based entity identification
- **Quality Scoring** (`core/quality_scorer.py`) - Filters low-quality chunks
- **Neo4j Storage** - Creates Document nodes, Chunk nodes, Entity nodes, and relationships

### API Layer (FastAPI)
- **Routers**: `api/routers/` - chat, documents, database, history endpoints
- **Models**: `api/models.py` - Pydantic schemas for requests/responses (ChatRequest, ChatResponse, etc.)
- **Services**: `api/services/` - Business logic (chat_history_service, follow_up_service)
- **Streaming**: Chat responses use SSE (Server-Sent Events) with dict payloads: `{"type": "stage"|"token"|"quality", "content": ...}`

### Frontend (Next.js + React)
- **State Management**: Zustand stores (`store/chatStore.ts`, `themeStore.ts`)
- **Components**: Organized by feature - Chat, Document, Sidebar, Theme, Toast
- **SSE Consumption**: `ChatInterface` parses streaming responses and updates UI progressively
- **Type Safety**: TypeScript types mirror API models in `src/types/index.ts`

## Critical Data Flows

### Chat Query Flow
1. Frontend sends `ChatRequest` with message, session_id, retrieval_mode, top_k, temperature
2. Backend initializes `RAGState` dict with query + parameters
3. LangGraph invokes workflow: query → retrieve → reason → generate
4. Streaming response emits: stages → tokens → quality_score
5. Frontend reconstructs Message with sources, quality_score, follow_up_questions

### Document Ingestion Flow
1. Upload triggers `ingestion/document_processor.py::process_document_async()`
2. Loader extracts text, chunker segments with overlap
3. Entity extraction runs (optional, async)
4. All chunks embedded via `core/embeddings.py::embedding_manager`
5. Neo4j stores: Document → Chunk nodes + Entity nodes + relationships
6. UI polls `/api/documents/processing-status` to track progress

### Hybrid Retrieval (Key Algorithm)
Combine chunk similarity + graph expansion:
- Vector search returns top_k chunks by embedding similarity
- For each chunk: traverse SIMILAR_TO edges, follow entity relationships
- Controlled by `expansion_similarity_threshold`, `max_expansion_depth`, `max_expanded_chunks`
- Multi-hop reasoning (if enabled) scores paths through entities: stronger relationships + higher entity importance = higher scores

## Key Configuration Patterns
All settings in `config/settings.py` are environment-based:
- **LLM Selection**: `llm_provider` (openai|ollama) determines which backend to use
- **Concurrency**: `embedding_concurrency` (default 3), `llm_concurrency` (default 2) prevent rate limiting
- **Rate Limiting**: `embedding_delay_min/max`, `llm_delay_min/max` add random delays between requests
- **Retrieval Tuning**: `hybrid_chunk_weight` (default 0.6), `max_expanded_chunks` (default 500), `max_expansion_depth` (default 2)
- **Quality Filtering**: `enable_quality_filtering`, `ocr_quality_threshold` (default 0.6)

## Development Workflow
- **Python Environment**: Always `source .venv/bin/activate` before running backend commands
- **Configuration**: Copy `.env.example` to `.env` and populate required keys (OPENAI_API_KEY, NEO4J_* credentials)
- **Database**: Neo4j must be running (Docker or local) before ingestion/queries
- **Frontend**: Runs on http://localhost:3000, proxies API calls to http://localhost:8000
- **Testing**: `pytest api/tests/` for Python, `npm run test` for frontend
- **Debugging**: Enable `LOG_LEVEL=DEBUG` in `.env` for verbose logs; check `/api/health` for service status

## Code Patterns & Conventions
- **Error Handling**: Async functions wrapped in try/except; errors returned in state/response dicts (never thrown to client)
- **Streaming**: All LLM responses use SSE; chunk data must be JSON serializable
- **Pydantic Models**: All API request/response types defined in `api/models.py` (e.g., ChatRequest has `message`, `session_id`, `context_documents`)
- **Neo4j Queries**: Use parameterized queries with `$param` syntax; avoid string concatenation
- **Service Isolation**: Business logic in `api/services/` and `core/`; routers only handle HTTP concerns (validation, response formatting)

## MVP Scope

### Purpose
Deliver a production-ready minimum viable product that showcases graph-based retrieval-augmented generation across a modern web interface. The MVP must allow users to ingest multi-format documents, explore them via graph-aware retrieval, and chat with high-quality, contextual responses.

### Core User Experience
- **Modern responsive UI** with dark-mode toggle and accessibility-friendly styling for desktop, tablet, and mobile
- **Persistent conversation history** to resume, search, and delete sessions while preserving context for follow-up turns
- **Intelligent chat** that streams answers, detects follow-up questions, re-contextualizes with chat history, and surfaces quality scores
- **Document management workspace** to upload PDFs, DOCX, TXT, MD, PPT, or XLS; monitor ingestion; view metadata/chunks/entities; manage database
- **Advanced retrieval controls** supporting context restriction, hybrid search, and multi-hop graph reasoning

### Data Ingestion & Processing
- **Multi-format ingestion** accepts PDF, DOCX, TXT, MD, PPT, XLS; PDFs use Marker OCR/LLM pipeline for high-fidelity Markdown
- **Graph enrichment** extracts entities and relationships with provenance for multi-hop traversal
- **Retrieval pipeline** combines vector similarity, entity search, and multi-hop paths with configurable weights and depth limits

### Current Entity Model (from `core/entity_extraction.py`)
**Canonical Types:** Component, Service, Node, Domain, Class of Service, Account, Account Type, Role, Resource, Quota Object, Backup Object, Item, Storage Object, Migration Procedure, Certificate, Config Option, Security Feature, CLI Command, API Object, Task, Procedure, Concept, Document, Person, Organization, Location, Event, Technology, Product, Date, Money

**Relationship Patterns:** Component↔Node/Component/Feature, provisioning links, backup/storage coverage, config/security associations, migration/CLI/task/procedure edges, generic RELATED_TO fallback

### Operations & Setup
- **Prerequisites:** Python 3.10+, Node 18+, Neo4j 5+, OpenAI-compatible API key
- **Runtime:** Backend via `python api/main.py`, frontend via `npm run dev`, Neo4j local or Docker, SSE streams responses
- **Quality scoring:** Feature toggles in settings/env vars; scoring is non-blocking and cached
- **Docker deployment:** Unified entrypoint (`docker-compose up -d --build`) for chat UI, backend API, and database upload

### MVP Success Criteria
- Users upload documents → appear in database view with parsed metadata → can be removed
- Chat streams responses with sources, respects conversation context, displays quality scores
- Retrieval leverages hybrid + multi-hop reasoning to connect entities across documents with adjustable parameters
- Full setup via provided scripts without additional code changes

## Do NOT
- Write MD documentation files unless explicitly requested
- Write tests unless explicitly requested
- Skip virtual environment activation for Python commands
- Use hardcoded API keys or credentials (always use .env)
- Modify CORS origins without updating `api/main.py`