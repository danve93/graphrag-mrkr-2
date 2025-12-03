# Graph-Enhanced RAG Pipeline

Amber's retrieval-augmented generation system combining vector search with graph reasoning.

## Overview

Traditional RAG systems rely solely on embedding similarity to retrieve context. Amber enhances this with graph-based reasoning that leverages:

- **Chunk similarity relationships** - Semantic connections between text segments
- **Entity co-occurrence** - Shared entities across documents
- **Multi-hop traversal** - Indirect relationships through graph paths
- **Community detection** - Clustered semantic neighborhoods

This graph-enhanced approach surfaces contextually relevant information that pure vector search might miss.

## Pipeline Architecture

The RAG pipeline is implemented as a LangGraph StateGraph with the following stages:

```
┌─────────────────┐
│  Query Analysis │  Parse query, extract filters, normalize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │  Hybrid vector + entity search → initial candidates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Graph Reasoning │  Multi-hop expansion via chunk/entity relationships
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reranking     │  Optional FlashRank reordering (if enabled)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generation    │  LLM generates response with sources
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quality Scoring │  Evaluate response quality (background)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Follow-ups    │  Generate suggested questions (background)
└─────────────────┘
```

## Implementation

**File**: `rag/graph_rag.py`

```python
from langgraph.graph import StateGraph

# Define state schema
class RAGState(TypedDict):
    query: str
    context_documents: List[str]
    retrieved_chunks: List[Dict]
    expanded_chunks: List[Dict]
    reranked_chunks: List[Dict]
    answer: str
    sources: List[str]
    quality_score: Optional[float]
    follow_up_questions: List[str]
    stages: List[str]

# Build graph
graph = StateGraph(RAGState)

# Add nodes
graph.add_node("query_analysis", query_analysis_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("graph_reasoning", graph_reasoning_node)
graph.add_node("reranking", reranking_node)
graph.add_node("generation", generation_node)

# Define edges
graph.add_edge("query_analysis", "retrieval")
graph.add_edge("retrieval", "graph_reasoning")
graph.add_conditional_edges(
    "graph_reasoning",
    should_rerank,
    {True: "reranking", False: "generation"}
)
graph.add_edge("reranking", "generation")

# Compile
rag_chain = graph.compile()
```

## Pipeline Stages

### 1. Query Analysis

**Purpose**: Normalize and prepare query for retrieval

**Node**: `rag/nodes/query_analysis.py`

**Operations**:
- Extract context document filters (e.g., "in document X")
- Parse hashtag filters (e.g., "#topic")
- Normalize whitespace and punctuation
- Identify query intent (factual, exploratory, comparative)

**Input**:
```python
{
  "query": "What is the backup procedure in document VMware Guide?",
  "context_documents": []
}
```

**Output**:
```python
{
  "query": "What is the backup procedure?",
  "context_documents": ["VMware Guide"],
  "stages": ["query_analysis"]
}
```

### 2. Retrieval

**Purpose**: Retrieve initial candidate chunks via hybrid search

**Node**: `rag/nodes/retrieval.py`

**Hybrid Search Components**:

**Vector Search** (70% weight):
```cypher
CALL db.index.vector.queryNodes(
  'chunk_embeddings',
  $top_k,
  $query_embedding
) YIELD node, score
```

**Entity Search** (30% weight):
```cypher
MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
WHERE e.name CONTAINS $query_term
RETURN c, count(DISTINCT e) as entity_matches
```

**Combined Score**:
```python
hybrid_score = (
    chunk_weight * vector_score +
    entity_weight * entity_match_score
)
```

**Configuration**:
- `HYBRID_CHUNK_WEIGHT` - Vector similarity weight (default: 0.7)
- `HYBRID_ENTITY_WEIGHT` - Entity match weight (default: 0.3)
- `TOP_K` - Initial retrieval count (default: 10)

**Output**:
```python
{
  "retrieved_chunks": [
    {
      "chunk_id": "abc123",
      "text": "Backup procedure involves...",
      "score": 0.92,
      "document_id": "doc456"
    },
    ...
  ],
  "stages": ["query_analysis", "retrieval"]
}
```

### 3. Graph Reasoning

**Purpose**: Expand context via graph traversal

**Node**: `rag/nodes/graph_reasoning.py`

**Expansion Strategies**:

**Chunk Similarity Traversal**:
```cypher
MATCH (seed:Chunk)-[r:SIMILAR_TO*1..{max_depth}]-(related:Chunk)
WHERE r.strength >= $threshold
RETURN DISTINCT related, min(r.strength) as min_strength
```

**Entity Relationship Traversal**:
```cypher
MATCH (seed:Chunk)-[:CONTAINS_ENTITY]->(e1:Entity)
MATCH (e1)-[r:RELATED_TO*1..{max_depth}]-(e2:Entity)
MATCH (e2)<-[:CONTAINS_ENTITY]-(related:Chunk)
WHERE r.strength >= $threshold
RETURN DISTINCT related
```

**Path Strength Calculation**:
```python
path_strength = product([edge.strength for edge in path])
final_score = path_strength * entity_importance
```

**Configuration**:
- `MAX_EXPANSION_DEPTH` - Maximum hops (default: 2, range: 1-3)
- `EXPANSION_SIMILARITY_THRESHOLD` - Min edge strength (default: 0.7)
- `MAX_EXPANDED_CHUNKS` - Maximum total chunks (default: 30)

**Algorithm**:
1. Start with seed chunks from retrieval
2. Traverse similarity edges up to max_depth
3. Traverse entity relationships up to max_depth
4. Score each path by edge strength product
5. Deduplicate and rank by score
6. Return top N chunks

**Output**:
```python
{
  "expanded_chunks": [
    {
      "chunk_id": "xyz789",
      "text": "Related backup information...",
      "expansion_score": 0.85,
      "hops": 2,
      "path": ["abc123", "def456", "xyz789"]
    },
    ...
  ],
  "stages": ["query_analysis", "retrieval", "graph_reasoning"]
}
```

**Performance**:
- 1-hop expansion: 50-100ms
- 2-hop expansion: 200-500ms
- 3-hop expansion: 1-2 seconds

### 4. Reranking (Optional)

**Purpose**: Reorder candidates using cross-encoder model

**Node**: `rag/nodes/reranking.py`

**When Enabled**: `FLASHRANK_ENABLED=true`

**Model**: FlashRank ms-marco-MiniLM-L-12-v2

**Process**:
1. Take top N candidates from expansion
2. Score each chunk against query using cross-encoder
3. Blend rerank scores with hybrid scores
4. Return reordered list

**Score Blending**:
```python
final_score = (
    flashrank_blend_weight * rerank_score +
    (1 - flashrank_blend_weight) * hybrid_score
)
```

**Configuration**:
- `FLASHRANK_ENABLED` - Enable reranking (default: false)
- `FLASHRANK_MAX_CANDIDATES` - Chunks to rerank (default: 50)
- `FLASHRANK_BLEND_WEIGHT` - Blend ratio (default: 0.5)
- `FLASHRANK_BATCH_SIZE` - Inference batch size (default: 32)

**Performance Impact**:
- Latency: +200-500ms
- Quality: +10-15% relevance improvement
- Cost: CPU/GPU inference time

**Output**:
```python
{
  "reranked_chunks": [
    {
      "chunk_id": "abc123",
      "text": "Backup procedure...",
      "rerank_score": 0.94,
      "hybrid_score": 0.87,
      "final_score": 0.905
    },
    ...
  ],
  "stages": [..., "reranking"]
}
```

### 5. Generation

**Purpose**: Generate natural language response with sources

**Node**: `rag/nodes/generation.py`

**Process**:
1. Construct context from top chunks
2. Build prompt with query + context
3. Stream LLM response token-by-token
4. Track source chunk references

**Prompt Template**:
```
You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context_chunks}

Question: {query}

Instructions:
- Provide a clear, accurate answer
- Cite sources using [1], [2], etc.
- If information is not in context, say so
- Be concise but complete

Answer:
```

**Streaming**:
```python
async for token in llm.astream(prompt):
    yield {
        "type": "token",
        "content": token,
        "metadata": {"chunk_refs": extract_refs(token)}
    }
```

**Configuration**:
- `LLM_MODEL` - Model selection (gpt-4o-mini, gpt-4o, llama3.1)
- `TEMPERATURE` - Generation randomness (default: 0.1)
- `MAX_TOKENS` - Response length limit (default: 1000)

**Output**:
```python
{
  "answer": "The backup procedure involves three steps: [1]...",
  "sources": [
    {
      "chunk_id": "abc123",
      "text": "Backup procedure involves...",
      "document_name": "VMware Guide",
      "page_number": 42
    }
  ],
  "stages": [..., "generation"]
}
```

### 6. Quality Scoring (Background)

**Purpose**: Evaluate response quality metrics

**Service**: `core/quality_scorer.py`

**Metrics**:
- **Relevance**: Answer addresses query
- **Completeness**: Answer covers key aspects
- **Groundedness**: Answer supported by sources
- **Clarity**: Answer is clear and coherent

**Scoring**:
```python
quality_score = (
    0.3 * relevance +
    0.25 * completeness +
    0.25 * groundedness +
    0.2 * clarity
)
```

**Implementation**: Runs asynchronously, does not block response streaming

**Output**:
```python
{
  "quality_score": 0.87,
  "metrics": {
    "relevance": 0.92,
    "completeness": 0.85,
    "groundedness": 0.88,
    "clarity": 0.82
  }
}
```

### 7. Follow-up Generation (Background)

**Purpose**: Suggest related questions

**Service**: `api/services/follow_up_generator.py`

**Process**:
1. Analyze query intent
2. Identify knowledge gaps
3. Generate 3-5 natural follow-up questions

**Example**:
```python
{
  "follow_up_questions": [
    "How often should backups be performed?",
    "What is the backup retention policy?",
    "How do you restore from a backup?"
  ]
}
```

## State Management

**State Type**: Plain Python dict (TypedDict)

**State Flow**:
```python
state = {
    "query": "user question",
    "stages": [],
    "retrieved_chunks": [],
    "expanded_chunks": [],
    "answer": "",
    "sources": []
}

# Each node modifies state
state = query_analysis_node(state)
state = retrieval_node(state)
state = graph_reasoning_node(state)
# ...
```

**Benefits**:
- Simple debugging (inspect state at any stage)
- Easy testing (mock state transitions)
- Observability (track stage timing)

## Retrieval Modes

### Vector Only

Pure embedding similarity, no graph expansion:
```python
{
  "retrieval_mode": "vector",
  "max_expansion_depth": 0
}
```

**Use case**: Fast queries, simple questions

### Hybrid

Vector + entity search, no expansion:
```python
{
  "retrieval_mode": "hybrid",
  "max_expansion_depth": 0
}
```

**Use case**: Entity-focused queries

### Graph-Enhanced (Default)

Hybrid + multi-hop expansion:
```python
{
  "retrieval_mode": "hybrid",
  "max_expansion_depth": 2
}
```

**Use case**: Complex queries requiring context

## Performance Characteristics

### Latency Breakdown

**Typical Query** (2-hop expansion, no reranking):
```
Query Analysis:     10-20ms
Retrieval:          100-200ms
Graph Reasoning:    200-500ms
Generation Start:   100-200ms (first token)
Generation Full:    2-3 seconds (streaming)
Quality Scoring:    500-1000ms (background)
Follow-ups:         1-2 seconds (background)
---
Total to first token: 400-900ms
Total to complete:    3-4 seconds
```

**With Reranking**:
```
Add Reranking:      +200-500ms
Total to first token: 600-1400ms
```

### Throughput

- Concurrent queries: 10-20 (limited by LLM API rate limits)
- Queries per minute: 60-120
- Cache hit optimization: 2-3x speedup for repeated queries

### Scalability

**Bottlenecks**:
1. LLM API rate limits (primary)
2. Neo4j graph traversal (secondary)
3. Embedding API calls (mitigated by caching)

**Optimization Strategies**:
- Enable multi-layer caching
- Reduce expansion depth for simple queries
- Use faster LLM models (gpt-3.5-turbo)
- Batch embedding requests

## Configuration Reference

### Essential Parameters

```bash
# Retrieval
TOP_K=10                              # Initial candidates
HYBRID_CHUNK_WEIGHT=0.7               # Vector weight
HYBRID_ENTITY_WEIGHT=0.3              # Entity weight

# Expansion
MAX_EXPANSION_DEPTH=2                 # Graph hops
MAX_EXPANDED_CHUNKS=30                # Total chunks
EXPANSION_SIMILARITY_THRESHOLD=0.7    # Edge threshold

# Generation
LLM_MODEL=gpt-4o-mini                 # Model selection
TEMPERATURE=0.1                       # Randomness
```

### Advanced Parameters

```bash
# Reranking
FLASHRANK_ENABLED=false               # Enable reranking
FLASHRANK_MAX_CANDIDATES=50           # Rerank count
FLASHRANK_BLEND_WEIGHT=0.5            # Score blending

# Caching
ENABLE_CACHING=true                   # Master switch
RETRIEVAL_CACHE_TTL=60                # Cache lifetime

# Concurrency
LLM_CONCURRENCY=5                     # Parallel requests
EMBEDDING_CONCURRENCY=5               # Parallel embeddings
```

## Testing

### Unit Tests

```bash
pytest tests/unit/test_graph_rag.py
```

### Integration Tests

```bash
pytest tests/integration/test_rag_pipeline.py
```

### Example Test

```python
def test_graph_reasoning_expansion():
    state = {
        "query": "backup procedure",
        "retrieved_chunks": [seed_chunk],
        "context_documents": []
    }
    
    result = graph_reasoning_node(state)
    
    assert len(result["expanded_chunks"]) > 0
    assert all(c["expansion_score"] > 0 for c in result["expanded_chunks"])
```

## Related Documentation

- [Retrieval Strategies](02-core-concepts/retrieval-strategies.md)
- [Caching System](02-core-concepts/caching-system.md)
- [Retriever Implementation](03-components/backend/retriever.md)
- [Graph Database](03-components/backend/graph-database.md)
