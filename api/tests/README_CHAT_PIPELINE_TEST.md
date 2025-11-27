# Chat Pipeline Test

This test suite provides comprehensive end-to-end testing of the Amber chat pipeline for the query "What is Carbonio?".

## What It Tests

The test suite validates the complete RAG pipeline including:

1. **Document Ingestion**
   - Document processing and chunking
   - Embedding generation
   - Entity extraction (if enabled)
   - Graph storage in Neo4j

2. **Vector Retrieval**
   - Chunk-based similarity search
   - Embedding quality and relevance
   - Content verification

3. **Entity-Based Retrieval** (if entity extraction enabled)
   - Entity detection and matching
   - Entity-to-chunk relationships
   - Carbonio-specific entity recognition

4. **Hybrid Retrieval**
   - Combination of vector and entity approaches
   - Weight balancing
   - Source diversity

5. **Graph Expansion**
   - Multi-hop entity traversal
   - Chunk similarity expansion
   - Context enrichment

6. **Reranking** (if FlashRank enabled)
   - Score adjustment
   - Ranking quality improvement

7. **Complete RAG Pipeline**
   - Query analysis
   - Retrieval execution
   - Graph reasoning
   - Response generation
   - Quality scoring

8. **Multi-Turn Conversations**
   - Context preservation across turns
   - Follow-up question handling

## Prerequisites

### Required Services

- **Neo4j Database**: The test requires a running Neo4j instance
- **LLM Provider**: OpenAI API key or Ollama running locally
- **Python Dependencies**: All requirements from `requirements.txt`

## Running the Tests

### Option 1: Using Docker Compose (Recommended)

Start all required services:

```bash
# Start Neo4j and other services
docker-compose up -d

# Run the test
python -m pytest api/tests/test_chat_pipeline.py -v -s

# Stop services when done
docker-compose down
```

### Option 2: Local Neo4j

If you have Neo4j running locally:

```bash
# Ensure Neo4j is running on bolt://localhost:7687

# Set environment variables
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your_password"
export OPENAI_API_KEY="your_openai_key"  # or configure Ollama

# Run the test
python -m pytest api/tests/test_chat_pipeline.py -v -s
```

### Option 3: Skip Database Tests

Run only tests that don't require Neo4j:

```bash
python -m pytest api/tests/test_chat_pipeline.py -v -s -k "not (connection or ingestion)"
```

## Test Output

The test provides detailed logging for each stage:

```
================================================================================
TEST: Vector Retrieval
================================================================================
✓ Retrieved 5 chunks via vector search
  1. Similarity: 0.8542
     Content: Carbonio is a comprehensive email and collaboration platform...
  2. Similarity: 0.7821
     Content: Key Features...
✓ Retrieved chunks contain relevant content about Carbonio
================================================================================
```

## Configuration

### Enable/Disable Features

Tests automatically adapt based on settings:

```python
# In config/settings.py or .env
ENABLE_ENTITY_EXTRACTION=true    # Enable entity-based tests
ENABLE_QUALITY_FILTERING=true    # Enable quality scoring tests
FLASHRANK_ENABLED=true           # Enable reranking tests
```

### Customize Test Document

The test uses a built-in Carbonio documentation sample. To use your own document:

```python
# Modify TEST_DOCUMENT_CONTENT in test_chat_pipeline.py
TEST_DOCUMENT_CONTENT = """
Your custom content here...
"""
```

## Expected Results

A successful test run shows:

```
================================================================================
PIPELINE TEST SUMMARY
================================================================================
Test Document: abc123...
Chunks: 15
Entities: 23

✓ All pipeline components verified:
  ✓ Document ingestion
  ✓ Vector retrieval
  ✓ Entity retrieval
  ✓ Hybrid retrieval
  ✓ Graph expansion
  ✓ Reranking
  ✓ Response generation
  ✓ Quality scoring
  ✓ Multi-turn conversation

Pipeline Status: OPERATIONAL ✓
================================================================================
```

## Troubleshooting

### Neo4j Connection Failed

```
SKIPPED (Neo4j not available: Failed to DNS resolve address neo4j:7687)
```

**Solution**: Ensure Neo4j is running and accessible. Check `NEO4J_URI` environment variable.

### No Entities Found

```
⚠ No entities found - entity extraction may have failed
```

**Solutions**:
- Ensure `ENABLE_ENTITY_EXTRACTION=true`
- Check LLM provider configuration
- Verify API key or Ollama is running

### FlashRank Not Available

```
⚠ FlashRank reranking disabled - skipping test
```

**Solution**: Set `FLASHRANK_ENABLED=true` in settings or `.env`

### Response Generation Failed

**Solutions**:
- Verify OpenAI API key: `echo $OPENAI_API_KEY`
- Or ensure Ollama is running: `ollama serve`
- Check `LLM_PROVIDER` setting (openai or ollama)

## Test Structure

```
test_chat_pipeline.py
├── Fixtures
│   ├── test_document_path      # Creates temp document
│   ├── document_processor      # Document processor instance
│   ├── test_document_id        # Ingests document, returns ID
│   └── cleanup_test_data       # Cleanup after tests
│
├── Connection Tests
│   └── test_neo4j_connection   # Verify database connectivity
│
├── Ingestion Tests
│   └── test_document_ingestion # Verify document processing
│
├── Retrieval Tests
│   ├── test_vector_retrieval   # Pure vector search
│   ├── test_entity_retrieval   # Entity-based search
│   ├── test_hybrid_retrieval   # Combined approach
│   └── test_graph_expansion    # Graph-based expansion
│
├── Enhancement Tests
│   └── test_reranking          # FlashRank reranking
│
├── Pipeline Tests
│   ├── test_complete_rag_pipeline      # Full pipeline
│   ├── test_chat_router_integration    # API integration
│   └── test_multi_turn_conversation    # Conversation flow
│
└── Summary
    └── test_full_pipeline_summary      # Overall results
```

## Continuous Integration

To run in CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Start Neo4j
  run: docker-compose up -d neo4j

- name: Wait for Neo4j
  run: |
    until docker exec neo4j cypher-shell "RETURN 1"; do
      sleep 2
    done

- name: Run Chat Pipeline Tests
  run: python -m pytest api/tests/test_chat_pipeline.py -v
  env:
    NEO4J_URI: bolt://localhost:7687
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Performance Benchmarks

Typical execution times (with all features enabled):

- Document Ingestion: 2-5 seconds
- Vector Retrieval: 0.1-0.3 seconds
- Entity Retrieval: 0.3-0.8 seconds
- Hybrid Retrieval: 0.4-1.2 seconds
- Graph Expansion: 0.5-2.0 seconds
- Complete Pipeline: 3-8 seconds
- **Total Test Suite**: 30-60 seconds

## Contributing

When adding new features to the pipeline:

1. Add corresponding test cases to this suite
2. Update the test document content if needed
3. Document expected behavior in docstrings
4. Run the full suite before committing

## See Also

- [AGENTS.md](../../../AGENTS.md) - Project architecture overview
- [README.md](../../../README.md) - Setup instructions
- [api/routers/chat.py](../../routers/chat.py) - Chat endpoint implementation
- [rag/graph_rag.py](../../../rag/graph_rag.py) - RAG pipeline implementation
