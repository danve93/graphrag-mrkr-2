# Chat Pipeline Test - Quick Reference

## 🚀 Quick Start

```bash
# Start services
docker-compose up -d neo4j

# Run test
./scripts/test_chat_pipeline.sh
```

## 📋 What It Tests

| Component | What's Validated |
|-----------|------------------|
| **Ingestion** | Document → Chunks → Embeddings → Entities → Neo4j |
| **Vector Retrieval** | Query embedding → Similarity search → Top-K chunks |
| **Entity Retrieval** | Entity matching → Carbonio entities → Related chunks |
| **Hybrid Retrieval** | Vector + Entity → Merged & scored → Deduplicated |
| **Graph Expansion** | Initial chunks → Graph traversal → Enriched context |
| **Reranking** | FlashRank scoring → Reordered results (if enabled) |
| **Generation** | LLM response → Carbonio answer → Source citations |
| **Quality Score** | Relevance + Completeness + Grounding → 0-1 score |
| **Multi-turn** | Context preservation → Follow-up handling |

## ✅ Success Indicators

```bash
# Expected output
================================================================================
PIPELINE TEST SUMMARY
================================================================================
Test Document: f8a7b2c1d4e5...
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
11 passed in 45.23s
```

## 🔧 Common Commands

### Run Specific Test
```bash
# Just vector retrieval
pytest api/tests/test_chat_pipeline.py::test_vector_retrieval -v -s

# Just complete pipeline
pytest api/tests/test_chat_pipeline.py::test_complete_rag_pipeline -v -s
```

### Run with Markers
```bash
# Skip slow tests
pytest api/tests/test_chat_pipeline.py -m "not slow" -v

# Run only integration tests
pytest api/tests/test_chat_pipeline.py -m "integration" -v
```

### Debug Mode
```bash
# With extra logging
pytest api/tests/test_chat_pipeline.py -v -s --log-cli-level=DEBUG

# Stop on first failure
pytest api/tests/test_chat_pipeline.py -x -v -s
```

## 🐛 Troubleshooting

### Neo4j Connection Failed
```bash
# Check Neo4j status
docker ps | grep neo4j

# Check Neo4j logs
docker logs neo4j

# Restart Neo4j
docker-compose restart neo4j

# Wait for readiness
docker exec neo4j cypher-shell -u neo4j -p password "RETURN 1"
```

### No Entities Extracted
```bash
# Check settings
grep ENABLE_ENTITY_EXTRACTION .env

# Set if missing
echo "ENABLE_ENTITY_EXTRACTION=true" >> .env

# Verify LLM provider
echo $OPENAI_API_KEY  # or check Ollama
```

### Tests Too Slow
```bash
# Disable expensive features temporarily
export ENABLE_ENTITY_EXTRACTION=false
export FLASHRANK_ENABLED=false

# Run test
pytest api/tests/test_chat_pipeline.py -v -s
```

## 📊 Performance Targets

| Operation | Target Time | Timeout |
|-----------|-------------|---------|
| Document Ingestion | < 5s | 30s |
| Vector Retrieval | < 0.5s | 5s |
| Entity Retrieval | < 1s | 10s |
| Hybrid Retrieval | < 1.5s | 15s |
| Graph Expansion | < 2s | 20s |
| Complete Pipeline | < 8s | 60s |
| **Total Test Suite** | **< 60s** | **5min** |

## 🔑 Environment Variables

```bash
# Required
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=sk-...

# Optional
ENABLE_ENTITY_EXTRACTION=true
ENABLE_QUALITY_FILTERING=true
FLASHRANK_ENABLED=true
LLM_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `api/tests/test_chat_pipeline.py` | Main test suite (650+ lines) |
| `api/tests/README_CHAT_PIPELINE_TEST.md` | Detailed documentation |
| `scripts/test_chat_pipeline.sh` | Test runner script |
| `pytest.ini` | Pytest configuration |
| `docs/CHAT_PIPELINE_TEST_FLOW.md` | Visual test flow diagram |

## 🧪 Test Coverage

```
test_chat_pipeline.py
├── test_neo4j_connection              ✓ Database connectivity
├── test_document_ingestion            ✓ Document processing
├── test_vector_retrieval              ✓ Similarity search
├── test_entity_retrieval              ✓ Entity-based search
├── test_hybrid_retrieval              ✓ Combined approach
├── test_graph_expansion               ✓ Graph traversal
├── test_reranking                     ✓ FlashRank scoring
├── test_complete_rag_pipeline         ✓ Full pipeline
├── test_chat_router_integration       ✓ API endpoint
├── test_multi_turn_conversation       ✓ Context preservation
└── test_full_pipeline_summary         ✓ Results summary
```

## 🎯 Query Being Tested

**"What is Carbonio?"**

Expected response should include:
- Definition: "comprehensive email and collaboration platform"
- Features: email, calendar, contacts, collaboration
- Architecture: nodes, web interface, storage
- Target users: businesses, enterprises

## 📈 CI/CD Integration

```yaml
# GitHub Actions example
- name: Run Chat Pipeline Test
  run: |
    docker-compose up -d neo4j
    sleep 10  # Wait for Neo4j
    pytest api/tests/test_chat_pipeline.py -v
  env:
    NEO4J_URI: bolt://localhost:7687
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## 🔗 Related Documentation

- [AGENTS.md](../AGENTS.md) - Architecture overview
- [README.md](../README.md) - Setup instructions
- [api/routers/chat.py](../api/routers/chat.py) - Chat implementation
- [rag/graph_rag.py](../rag/graph_rag.py) - Pipeline orchestration

## 💡 Tips

1. **First time?** Use the test runner script: `./scripts/test_chat_pipeline.sh`
2. **Debugging?** Add `-s` flag to see print statements
3. **Time-constrained?** Disable entity extraction temporarily
4. **CI/CD?** Ensure Neo4j health check passes before running test
5. **Failed test?** Check logs for specific error messages

## 📝 Example Test Output

```bash
$ ./scripts/test_chat_pipeline.sh

==================================================
Chat Pipeline Test Runner
==================================================

✓ Docker is running

Starting Neo4j...
Waiting for Neo4j to be ready...
  Waiting... (1/30)
  Waiting... (2/30)
✓ Neo4j is ready

==================================================
Running Chat Pipeline Tests
==================================================

api/tests/test_chat_pipeline.py::test_neo4j_connection PASSED
api/tests/test_chat_pipeline.py::test_document_ingestion PASSED
api/tests/test_chat_pipeline.py::test_vector_retrieval PASSED
api/tests/test_chat_pipeline.py::test_entity_retrieval PASSED
api/tests/test_chat_pipeline.py::test_hybrid_retrieval PASSED
api/tests/test_chat_pipeline.py::test_graph_expansion PASSED
api/tests/test_chat_pipeline.py::test_reranking PASSED
api/tests/test_chat_pipeline.py::test_complete_rag_pipeline PASSED
api/tests/test_chat_pipeline.py::test_chat_router_integration PASSED
api/tests/test_chat_pipeline.py::test_multi_turn_conversation PASSED
api/tests/test_chat_pipeline.py::test_full_pipeline_summary PASSED

==================================================
Test Results
==================================================
✓ All tests passed!

To stop Neo4j, run: docker-compose down
```

## 🆘 Need Help?

1. Check test logs: Look for specific error messages
2. Verify services: `docker ps` - ensure Neo4j is running
3. Check environment: `env | grep NEO4J` - verify variables set
4. Review docs: See `api/tests/README_CHAT_PIPELINE_TEST.md`
5. Check Neo4j: Visit http://localhost:7474 in browser

## 🎓 Learning Resources

- **Test code**: Read `api/tests/test_chat_pipeline.py` for examples
- **Pipeline flow**: See `docs/CHAT_PIPELINE_TEST_FLOW.md` for visuals
- **Architecture**: Review `AGENTS.md` for system design
- **API docs**: Check FastAPI docs at http://localhost:8000/docs
