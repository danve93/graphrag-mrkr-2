# Chat Pipeline Test Flow

This document visualizes what the test suite validates at each stage.

## Test Query: "What is Carbonio?"

```
┌─────────────────────────────────────────────────────────────────────┐
│                     USER QUERY                                      │
│                  "What is Carbonio?"                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TEST 1: Document Ingestion                                          │
│ ✓ Creates test document with Carbonio content                      │
│ ✓ Chunks document (expected: 15 chunks)                            │
│ ✓ Generates embeddings for each chunk                              │
│ ✓ Extracts entities (expected: ~23 entities including "Carbonio")  │
│ ✓ Stores in Neo4j graph database                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TEST 2: Vector Retrieval                                            │
│ ✓ Embeds query: "What is Carbonio?"                                │
│ ✓ Performs similarity search against chunk embeddings              │
│ ✓ Returns top-K most similar chunks (K=5)                          │
│ ✓ Verifies similarity scores > 0                                   │
│ ✓ Confirms chunks contain "carbonio" text                          │
│                                                                     │
│ Example Output:                                                     │
│   Chunk 1: similarity=0.8542 "Carbonio is a comprehensive..."      │
│   Chunk 2: similarity=0.7821 "Key Features..."                     │
│   Chunk 3: similarity=0.7234 "Architecture..."                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TEST 3: Entity-Based Retrieval                                      │
│ ✓ Searches for relevant entities: finds "Carbonio Node", etc.      │
│ ✓ Retrieves chunks containing those entities                       │
│ ✓ Calculates entity-aware similarity scores                        │
│ ✓ Verifies Carbonio-related entities are found                     │
│                                                                     │
│ Example Output:                                                     │
│   Entities: ["Carbonio", "Carbonio Node", "Carbonio CLI Command"]  │
│   Chunks: 5 chunks containing these entities                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TEST 4: Hybrid Retrieval                                            │
│ ✓ Combines vector search (50%) + entity search (50%)               │
│ ✓ Deduplicates and scores chunks from both methods                 │
│ ✓ Creates hybrid scores                                            │
│ ✓ Verifies retrieval source diversity                              │
│                                                                     │
│ Example Output:                                                     │
│   Chunk 1: hybrid_score=0.91, source=hybrid (both methods)         │
│   Chunk 2: hybrid_score=0.84, source=chunk_based                   │
│   Chunk 3: hybrid_score=0.78, source=entity_based                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TEST 5: Graph Expansion                                             │
│ ✓ Takes top chunks and expands via graph relationships             │
│ ✓ Follows entity relationships (SIMILAR_TO, RELATED_TO)            │
│ ✓ Follows chunk similarity edges                                   │
│ ✓ Enriches context with expanded chunks                            │
│                                                                     │
│ Example Output:                                                     │
│   Original: 3 chunks                                                │
│   Expanded: 8 chunks (3 original + 5 graph-expanded)               │
│   Expansion types: 3 entity-based, 2 similarity-based              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TEST 6: Reranking (Optional - if FlashRank enabled)                │
│ ✓ Applies FlashRank model to reorder chunks                        │
│ ✓ Adjusts scores based on query-chunk relevance                    │
│ ✓ Compares before/after rankings                                   │
│                                                                     │
│ Example Output:                                                     │
│   Before: [chunk_2, chunk_1, chunk_3]                              │
│   After:  [chunk_1, chunk_2, chunk_3] (reordered by rerank score)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TEST 7: Complete RAG Pipeline (LangGraph)                           │
│ ✓ Stage 1: Query Analysis - analyzes query type, intent            │
│ ✓ Stage 2: Retrieval - executes hybrid retrieval                   │
│ ✓ Stage 3: Graph Reasoning - expands context via graph             │
│ ✓ Stage 4: Generation - generates response using LLM               │
│ ✓ Verifies all 4 stages executed                                   │
│ ✓ Verifies response mentions "carbonio"                            │
│ ✓ Verifies sources are cited                                       │
│ ✓ Verifies metadata is populated                                   │
│                                                                     │
│ Example Output:                                                     │
│   Stages: [query_analysis, retrieval, graph_reasoning, generation] │
│   Response: "Carbonio is a comprehensive email and collaboration..." │
│   Sources: 5 chunks cited with document references                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TEST 8: Quality Scoring                                             │
│ ✓ Calculates quality score components:                             │
│   - Relevance: how well response answers query                     │
│   - Completeness: coverage of key points                           │
│   - Grounding: support from source chunks                          │
│   - Clarity: response coherence and structure                      │
│ ✓ Verifies overall score is calculated                             │
│                                                                     │
│ Example Output:                                                     │
│   Overall Score: 0.85                                               │
│   Relevance: 0.90                                                   │
│   Completeness: 0.82                                                │
│   Grounding: 0.88                                                   │
│   Clarity: 0.80                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TEST 9: Chat Router Integration                                     │
│ ✓ Tests FastAPI endpoint with ChatRequest model                    │
│ ✓ Verifies session ID creation                                     │
│ ✓ Verifies response structure (message, sources, metadata)         │
│ ✓ Verifies context documents are preserved                         │
│ ✓ Verifies quality score in metadata                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TEST 10: Multi-Turn Conversation                                    │
│ ✓ Turn 1: "What is Carbonio?" → response generated                 │
│ ✓ Turn 2: "What are its key features?" with history                │
│ ✓ Verifies context is maintained across turns                      │
│ ✓ Verifies follow-up response addresses features                   │
│                                                                     │
│ Example Output:                                                     │
│   Turn 1: "Carbonio is a comprehensive email platform..."          │
│   Turn 2: "The key features include email management, calendar..." │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FINAL RESULT                                    │
│                                                                     │
│   ✓ All pipeline components verified                               │
│   ✓ Retrieval working correctly                                    │
│   ✓ Graph reasoning operational                                    │
│   ✓ Response generation successful                                 │
│   ✓ Quality scoring functional                                     │
│   ✓ Multi-turn context preserved                                   │
│                                                                     │
│   Pipeline Status: OPERATIONAL ✓                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## What Gets Validated

### Data Flow Verification

```
Document → Chunks → Embeddings → Neo4j ✓
                ↓
Query → Embedding → Vector Search → Ranked Chunks ✓
                ↓
Query → Entity Search → Entity Matches → Related Chunks ✓
                ↓
Vector + Entity Results → Hybrid Scoring → Merged Results ✓
                ↓
Top Results → Graph Expansion → Enriched Context ✓
                ↓
Enriched Context → Reranking (optional) → Final Chunks ✓
                ↓
Final Chunks → LLM Generation → Response ✓
                ↓
Response + Chunks → Quality Scoring → Metrics ✓
```

### Database Queries Verified

1. **Document Storage**
   ```cypher
   MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
   WHERE d.id = $doc_id
   RETURN count(c)  // Should return 15
   ```

2. **Entity Extraction**
   ```cypher
   MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
         -[:CONTAINS_ENTITY]->(e:Entity)
   WHERE e.name CONTAINS "Carbonio"
   RETURN count(DISTINCT e)  // Should find Carbonio entities
   ```

3. **Vector Similarity**
   ```cypher
   CALL db.index.vector.queryNodes(
     'chunk-embeddings', 
     $k, 
     $query_embedding
   )
   YIELD node, score
   RETURN node.id, score  // Should return similar chunks
   ```

4. **Graph Expansion**
   ```cypher
   MATCH (c:Chunk {id: $chunk_id})-[:SIMILAR_TO*1..2]->(related:Chunk)
   RETURN related  // Should find related chunks
   ```

## Performance Monitoring

Each test logs execution time:

```
TEST: Vector Retrieval
✓ Retrieved 5 chunks via vector search (0.23s)

TEST: Entity Retrieval  
✓ Entity retrieval returned 5 chunks (0.64s)

TEST: Hybrid Retrieval
✓ Hybrid retrieval returned 5 chunks (0.89s)

TEST: Graph Expansion
✓ Graph expansion returned 8 chunks (1.34s)

TEST: Complete RAG Pipeline
✓ Pipeline executed 4 stages (5.67s)
```

## Test Data

### Test Document Structure

```markdown
# Carbonio Documentation
## What is Carbonio?
[Description: 150 words about Carbonio platform]

### Key Features
[List: 5 main features with descriptions]

### Architecture
[Components: Carbonio Node, web interface, storage]

### Administration
[Tools: CLI commands, admin panel, API access]

### Security Features
[Features: encryption, authentication, logging]

### Use Cases
[Scenarios: SMB, enterprises, migrations]
```

### Expected Entities

The test verifies extraction of:
- Product: "Carbonio"
- Component: "Carbonio Node"
- Technology: "Email", "Calendar", "Collaboration"
- Concept: "Email Management", "Web Interface"
- CLI Command: "Carbonio CLI Command"
- Security Feature: Various security components

### Expected Quality Scores

For "What is Carbonio?" query:
- Relevance: 0.85-0.95 (direct answer about Carbonio)
- Completeness: 0.75-0.90 (covers main points)
- Grounding: 0.85-0.95 (well-supported by chunks)
- Overall: 0.80-0.90

## Success Criteria

✅ **All 11 tests pass**  
✅ **No database errors**  
✅ **Response contains "carbonio"**  
✅ **Entities extracted successfully**  
✅ **Sources cited correctly**  
✅ **Quality score > 0.7**  
✅ **Multi-turn context preserved**  
✅ **Execution time < 60 seconds**  

## Failure Scenarios

### If Neo4j Not Available
```
SKIPPED: Neo4j not available
Solution: Start Neo4j with docker-compose
```

### If LLM Not Available
```
ERROR: OpenAI API key not set
Solution: Set OPENAI_API_KEY or configure Ollama
```

### If No Entities Found
```
WARNING: No entities extracted
Solution: Enable ENABLE_ENTITY_EXTRACTION=true
```

### If Poor Quality Score
```
WARNING: Quality score below threshold
Solution: Check retrieval relevance and chunk content
```
