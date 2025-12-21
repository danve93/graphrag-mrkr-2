# Sentence-Window Retrieval Implementation Plan

## Goal
Implement "Sentence-Window Retrieval": embed single sentences for fine-grained search, but retrieve ±5 surrounding sentences to preserve context.

## User Review Required
> [!WARNING]
> This is a significant architectural change that affects:
> - **Storage**: A new `Sentence` node type and vector index will increase storage requirements proportionally to the number of sentences.
> - **Ingestion Time**: Each chunk will be split into sentences, each requiring an embedding API call (can be batched).
> - **Backward Compatibility**: Existing chunks remain functional. Sentence retrieval is additive and can be toggled.

**Design Decision**: Should sentence vectors be a complete replacement for chunk vectors, or a supplementary retrieval path?
- **Option A**: Supplementary (default) – Both chunk and sentence retrieval exist; a new flag enables sentence mode.
- **Option B**: Replacement – All retrieval uses sentences; chunks are only for context assembly.

**Recommendation**: Start with Option A (supplementary) to minimize risk.

---

## Proposed Changes

### 1. Core Schema Changes
#### [NEW] `core/sentence_chunker.py`
- New utility to split chunk text into sentences using NLTK or regex.
- Function: `split_into_sentences(text: str) -> List[str]`

#### [MODIFY] [graph_db.py](file:///home/daniele/amber/core/graph_db.py)
- Add `create_sentence_node(sentence_id, chunk_id, content, embedding, index_in_chunk)`.
- Add `sentence_embeddings` vector index.
- Add `get_sentence_context(sentence_id, window_size=5)` to fetch ±N sentences from the same chunk.

---

### 2. Ingestion Pipeline Changes
#### [MODIFY] [document_processor.py](file:///home/daniele/amber/ingestion/document_processor.py)
- In `_embed_and_store`, after storing the chunk, split its content into sentences.
- For each sentence, generate an embedding and call `create_sentence_node`.
- Link sentences to their parent chunk with `(:Chunk)-[:HAS_SENTENCE]->(:Sentence)`.

---

### 3. Retrieval Changes
#### [MODIFY] [retriever.py](file:///home/daniele/amber/rag/retriever.py)
- Add `sentence_based_retrieval()` method:
  1. Perform vector search on `sentence_embeddings` index.
  2. For each matched sentence, call `get_sentence_context(sentence_id, window_size=5)`.
  3. Assemble the expanded context (sentence + surrounding sentences).
  4. Return deduplicated, ranked results.
- Integrate into `hybrid_retrieval` as an optional path controlled by `settings.enable_sentence_window_retrieval`.

---

### 4. Configuration Changes
#### [MODIFY] [settings.py](file:///home/daniele/amber/config/settings.py)
- Add `enable_sentence_window_retrieval: bool = False`.
- Add `sentence_window_size: int = 5`.

---

## Impact Analysis

### Affected Flows

| Flow | Component | Impact |
|------|-----------|--------|
| **Ingestion** | `ingestion/document_processor.py` | Modified – after storing chunks, also stores sentences |
| **Ingestion** | `core/graph_db.py` | Modified – new `create_sentence_node()`, new vector index |
| **Retrieval** | `rag/retriever.py` | Modified – new `sentence_based_retrieval()` method |
| **Retrieval** | `rag/nodes/retrieval.py` | Modified – calls new sentence retrieval if enabled |
| **Chat Pipeline** | `rag/graph_rag.py` | Indirect – consumes retrieval results; no code changes needed |
| **Follow-ups** | `api/services/follow_up_service.py` | Indirect – uses `chunk_id` from retrieval; sentence results will include parent `chunk_id` |
| **Quality Monitor** | `rag/quality_monitor.py` | Indirect – tracks `num_chunks_retrieved`; may need to distinguish sentences vs chunks |
| **Adaptive Router** | `rag/nodes/adaptive_router.py` | Indirect – adjusts weights; no direct changes needed |
| **TruLens** | `evals/trulens/trulens_wrapper.py` | Indirect – instruments `GraphRAG.query()`; no changes needed |
| **API** | `api/routers/documents.py` | Potentially – if exposing sentence-level endpoints (optional) |

### NOT Affected

| Component | Reason |
|-----------|--------|
| Entity extraction | Operates on chunks, not sentences |
| Graph clustering | Operates on entities |
| LLM generation | Receives context chunks/sentences transparently |
| Frontend | Unless we add a toggle (optional) |
| Existing chunk retrieval | Remains fully functional |

### Key Design Considerations

1. **Sentence ↔ Chunk Association**: Each `Sentence` node stores its parent `chunk_id`. Retrieval returns `chunk_id` in metadata so downstream consumers (follow-ups, sources) work unchanged.

2. **Quality Monitor**: Currently tracks `num_chunks_retrieved`. For sentences, we should either:
   - Count expanded context as chunks (recommended for minimal change)
   - Add a new `num_sentences_retrieved` metric (more accurate but more changes)

3. **TruLens**: Monitors query → response. No changes needed as it instruments at the `GraphRAG.query()` level, which is above retrieval.

4. **Tests**: Multiple test files reference `hybrid_retrieval`:
   - `test_follow_up_retrieval.py`
   - `test_retrieval_caching.py`
   - `test_caching_integration.py`
   - `test_query_routing.py`
   - `test_reranking.py`
   - `test_keyword_search.py`
   - `integration/test_chat_pipeline.py`
   - `integration/test_reranker.py`
   These need updates to cover the new sentence retrieval path.

---

## Verification Plan

### Automated Tests
1. **Unit Test**: `test_sentence_chunker.py` – Verify sentence splitting.
2. **Unit Test**: `test_sentence_retrieval.py` – Mock sentence retrieval and context expansion.

### Manual Verification
1. Ingest a test document with paragraph-rich content.
2. Query for a specific detail embedded in a sentence.
3. Verify the returned context includes surrounding sentences.
4. Compare retrieval quality vs. chunk-only mode.

---

## Migration Strategy
- **New documents**: Automatically generate sentence nodes if `enable_sentence_window_retrieval` is True.
- **Existing documents**: Provide a CLI command `amber-cli reindex-sentences` to backfill sentence nodes.
