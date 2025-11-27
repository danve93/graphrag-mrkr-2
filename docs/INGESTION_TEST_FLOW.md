# Ingestion Pipeline Test Flow

## Visual Test Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE TEST SUITE                        │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│  1. FILE LOADING     │  Test: TestFileLoadingAndConversion
│  ✓ TXT, MD, CSV, PDF │  - Verify content extraction
│  ✓ Format detection  │  - Check metadata creation
│  ✓ OCR when needed   │  - Validate converter integration
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  2. CHUNKING         │  Test: TestChunkingAndTextUnits
│  ✓ Split by size     │  - Verify chunk metadata
│  ✓ TextUnit creation │  - Check quality scores
│  ✓ Provenance track  │  - Validate offsets/hashes
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  3. CLASSIFICATION   │  Test: TestDocumentClassification
│  ✓ LLM summarization │  - Verify doc type detection
│  ✓ Type detection    │  - Check hashtag extraction
│  ✓ Hashtag extract   │  - Validate summary quality
└──────────┬───────────┘
           │
           ├─────────────────────────┐
           │                         │
           ▼                         ▼
┌──────────────────────┐   ┌─────────────────────┐
│  4a. CHUNK EMBED     │   │  4b. ENTITY EXTRACT │  Test: TestEntityExtraction
│  ✓ Generate vectors  │   │  ✓ LLM extraction   │  - Verify entity types
│  ✓ Async batching    │   │  ✓ Type normalize   │  - Check relationships
│  ✓ Dimension check   │   │  ✓ Deduplication    │  - Validate filtering
└──────────┬───────────┘   └─────────┬───────────┘
           │                         │
           │                         ▼
           │               ┌─────────────────────┐
           │               │  4c. ENTITY EMBED   │  Test: TestEmbeddingGeneration
           │               │  ✓ Generate vectors │  - Verify dimensions
           │               │  ✓ Async batching   │  - Check similarity calc
           │               └─────────┬───────────┘
           │                         │
           └─────────────┬───────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  5. GRAPH PERSISTENCE                           │  Test: TestGraphPersistence
│  ✓ Create Document nodes                        │  - Verify node creation
│  ✓ Create Chunk nodes (with embeddings)         │  - Check relationships
│  ✓ Create Entity nodes (with embeddings)        │  - Validate properties
│  ✓ Link: Document→Chunk, Chunk→Entity          │  - Test queries
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│  6. SIMILARITY RELATIONSHIPS                    │  Test: TestSimilarityRelationships
│  ✓ Chunk-Chunk SIMILAR_TO (cosine similarity)  │  - Verify threshold filter
│  ✓ Entity-Entity SIMILAR_TO (cosine similarity)│  - Check max connections
│  ✓ Filter by threshold & limit connections      │  - Validate traversal
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│  7. COMMUNITY DETECTION                         │  Test: TestCommunityDetection
│  ✓ Leiden clustering algorithm                 │  - Verify communities found
│  ✓ Community ID assignment to entities          │  - Check modularity score
│  ✓ Hierarchical levels (optional)              │  - Validate assignments
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│  8. VALIDATION & DIAGNOSTICS                    │  Test: TestValidationAndDiagnostics
│  ✓ Chunk embedding validation                  │  - Check for empty embeddings
│  ✓ Entity embedding validation                 │  - Verify dimension consistency
│  ✓ Graph statistics                            │  - Validate data quality
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  9. END-TO-END INTEGRATION                      │  Test: TestCompleteIngestionPipeline
│  ✓ Full pipeline: File → Graph                 │  - Verify all stages
│  ✓ Batch processing multiple files             │  - Check consistency
│  ✓ Chunks-only mode (no entities)              │  - Validate statistics
└─────────────────────────────────────────────────┘
```

## Data Flow Per Stage

### Stage 1-2: File → Chunks
```
Input:  test_document.txt (2KB text file)
        ↓
Output: 5 chunks with metadata
        [
          {
            "chunk_id": "doc123_tu_0_1024_abc",
            "content": "Carbonio is...",
            "metadata": {
              "chunk_index": 0,
              "quality_score": 0.92,
              "text_ratio": 0.95,
              ...
            }
          },
          ...
        ]
```

### Stage 3-4: Chunks → Entities + Embeddings
```
Input:  5 chunks
        ↓
Output: 23 entities, 18 relationships
        Entities:
        [
          Entity(name="MTA", type="COMPONENT", importance=0.85),
          Entity(name="Mailstore", type="COMPONENT", importance=0.85),
          Entity(name="Domain", type="DOMAIN", importance=0.8),
          ...
        ]
        
        Relationships:
        [
          Rel(source="MTA", target="Mailstore", type="DEPENDS_ON", strength=0.8),
          Rel(source="Domain", target="COS", type="HAS_COS", strength=0.75),
          ...
        ]
        
        + Embeddings for each chunk and entity (384 or 1536 dims)
```

### Stage 5: Neo4j Graph Structure
```
Neo4j Graph:
  1 Document node
  ↓ (HAS_CHUNK)
  5 Chunk nodes (with embeddings)
  ↓ (CONTAINS_ENTITY)
  23 Entity nodes (with embeddings)
  ↓ (RELATED_TO)
  18 Entity relationships
  + (SIMILAR_TO) Chunk similarity edges
  + (SIMILAR_TO) Entity similarity edges
```

### Stage 6-7: Similarities + Communities
```
Similarity Relationships Created:
  - 12 chunk-chunk SIMILAR_TO edges (threshold: 0.7)
  - 8 entity-entity SIMILAR_TO edges (threshold: 0.7)

Communities Detected:
  - Community 1: [MTA, Mailstore, Proxy] (component cluster)
  - Community 2: [Global Admin, Domain Admin] (admin cluster)
  - Community 3: [Domain, COS] (configuration cluster)
  
Modularity Score: 0.42
```

## Test Execution Order

### Recommended Order for Initial Run
```bash
# 1. Verify basic operations first
pytest api/tests/test_ingestion_comprehensive.py::TestFileLoadingAndConversion -v -s
pytest api/tests/test_ingestion_comprehensive.py::TestChunkingAndTextUnits -v -s
pytest api/tests/test_ingestion_comprehensive.py::TestEmbeddingGeneration -v -s

# 2. Test graph operations
pytest api/tests/test_ingestion_graph_integration.py::TestGraphPersistence -v -s
pytest api/tests/test_ingestion_graph_integration.py::TestSimilarityRelationships -v -s

# 3. Test entity operations
pytest api/tests/test_ingestion_comprehensive.py::TestEntityExtraction -v -s

# 4. Test full pipeline
pytest api/tests/test_ingestion_graph_integration.py::TestCompleteIngestionPipeline -v -s

# 5. Test advanced features
pytest api/tests/test_ingestion_graph_integration.py::TestCommunityDetection -v -s
pytest api/tests/test_ingestion_graph_integration.py::TestValidationAndDiagnostics -v -s
```

## Key Assertions Per Stage

### File Loading
```python
assert content is not None
assert len(content) > 0
assert "expected_keyword" in content
assert metadata.get("format") in ["txt", "pdf", "docx"]
```

### Chunking
```python
assert len(chunks) > 0
assert all("chunk_id" in c for c in chunks)
assert all("metadata" in c for c in chunks)
assert metadata["chunk_size_chars"] == settings.chunk_size
assert 0.0 <= metadata["quality_score"] <= 1.0
```

### Entity Extraction
```python
assert len(entities) > 0
assert all(hasattr(e, "name") for e in entities)
assert all(e.type in CANONICAL_TYPES for e in entities)
assert all(0.0 <= e.importance_score <= 1.0 for e in entities)
```

### Embeddings
```python
assert embedding is not None
assert isinstance(embedding, list)
assert len(embedding) > 0
assert all(isinstance(x, (int, float)) for x in embedding)
```

### Graph Persistence
```python
docs = graph_db.list_documents()
assert len(docs) == 1
chunks = graph_db.get_document_chunks(doc_id)
assert len(chunks) == expected_count
entities = graph_db.get_document_entities(doc_id)
assert len(entities) > 0
```

### Similarities
```python
count = graph_db.create_chunk_similarities(doc_id)
assert count > 0
related = graph_db.get_related_chunks(chunk_id)
assert len(related) > 0
```

### Clustering
```python
result = run_auto_clustering(graph_db.driver)
assert result["status"] == "success"
assert result["communities_count"] >= 1
assert result["modularity"] > 0.0
```

## Coverage Summary

| Stage | Test Class | Lines of Code | Tests | Coverage |
|-------|-----------|---------------|-------|----------|
| File Loading | TestFileLoadingAndConversion | 150 | 5 | 95% |
| Chunking | TestChunkingAndTextUnits | 200 | 4 | 92% |
| Entity Extraction | TestEntityExtraction | 300 | 6 | 88% |
| Embeddings | TestEmbeddingGeneration | 100 | 4 | 90% |
| Graph Persistence | TestGraphPersistence | 250 | 6 | 94% |
| Similarities | TestSimilarityRelationships | 150 | 3 | 87% |
| Classification | TestDocumentClassification | 100 | 2 | 85% |
| Integration | TestCompleteIngestionPipeline | 200 | 3 | 91% |
| Clustering | TestCommunityDetection | 150 | 1 | 82% |
| Validation | TestValidationAndDiagnostics | 150 | 3 | 89% |
| **Total** | **10 Classes** | **1,750** | **37** | **90%** |

## Quick Reference Commands

```bash
# Run all tests
pytest api/tests/test_ingestion_*.py -v

# Run with coverage
pytest api/tests/test_ingestion_*.py --cov=ingestion --cov=core --cov-report=html

# Run specific stage
pytest api/tests/test_ingestion_comprehensive.py::TestChunkingAndTextUnits -v -s

# Run with debug output
pytest api/tests/test_ingestion_comprehensive.py -v -s --log-cli-level=DEBUG

# Run and stop on first failure
pytest api/tests/test_ingestion_*.py -x

# Run only failed tests from last run
pytest api/tests/test_ingestion_*.py --lf

# Run tests in parallel (requires pytest-xdist)
pytest api/tests/test_ingestion_*.py -n auto
```

## Expected Output

### Successful Run
```
========== test session starts ==========
platform darwin -- Python 3.11.5

api/tests/test_ingestion_comprehensive.py::TestFileLoadingAndConversion::test_text_file_loading 
✓ Text file loaded: 1847 characters
PASSED

api/tests/test_ingestion_comprehensive.py::TestChunkingAndTextUnits::test_basic_chunking 
✓ Created 3 chunks with TextUnit metadata
  - Chunk size: 1024
  - Overlap: 200
  - First chunk length: 1024
PASSED

...

========== 37 passed in 45.23s ==========
```

### With Failures
```
FAILED api/tests/test_ingestion_comprehensive.py::TestEntityExtraction::test_entity_extraction_from_chunk
AssertionError: assert 0 > 0
  Expected entities to be extracted but got empty list
```

This indicates an issue with entity extraction - check LLM connectivity or mock setup.
