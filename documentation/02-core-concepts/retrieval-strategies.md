# Retrieval Strategies

Comprehensive guide to Amber's hybrid retrieval and graph expansion techniques.

## Overview

Amber implements a multi-stage retrieval strategy that combines:
1. **Vector search** - Embedding similarity for semantic matching
2. **Entity search** - Entity name matching for precise recall
3. **Graph expansion** - Multi-hop traversal for context enrichment
4. **Reranking** - Cross-encoder scoring for improved relevance

This hybrid approach balances precision, recall, and contextual richness.

## Retrieval Modes

### Vector Only

**Description**: Pure embedding similarity search

**Use case**: Fast, straightforward queries

**Configuration**:
```python
{
  "retrieval_mode": "vector",
  "top_k": 10,
  "max_expansion_depth": 0
}
```

**Query**:
```cypher
CALL db.index.vector.queryNodes(
  'chunk_embeddings',
  $top_k,
  $query_embedding
) YIELD node, score
RETURN node, score
ORDER BY score DESC;
```

**Performance**: 100-200ms

**Advantages**:
- Fast retrieval
- Good semantic matching
- Simple implementation

**Limitations**:
- Misses exact entity matches
- No contextual expansion
- Limited to top-K results

### Hybrid (Vector + Entity)

**Description**: Combines vector similarity with entity matching

**Use case**: Queries mentioning specific entities

**Configuration**:
```python
{
  "retrieval_mode": "hybrid",
  "top_k": 10,
  "hybrid_chunk_weight": 0.7,
  "hybrid_entity_weight": 0.3,
  "max_expansion_depth": 0
}
```

**Implementation**: `rag/retriever.py::hybrid_retrieval()`

```python
async def hybrid_retrieval(query: str, top_k: int) -> List[Chunk]:
    # Vector search
    vector_results = await vector_search(query, top_k)
    
    # Entity search
    entity_results = await entity_search(query, top_k)
    
    # Combine scores
    combined = {}
    for chunk in vector_results:
        combined[chunk.id] = chunk_weight * chunk.score
    
    for chunk in entity_results:
        combined[chunk.id] = combined.get(chunk.id, 0) + entity_weight * chunk.score
    
    # Sort and return
    sorted_chunks = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks[:top_k]
```

**Performance**: 150-250ms

**Advantages**:
- Better entity recall
- Balanced semantic + exact matching
- Adjustable weight blending

**Limitations**:
- Slightly slower than vector-only
- Still limited to top-K
- No multi-hop reasoning

### Graph-Enhanced (Hybrid + Expansion)

**Description**: Hybrid retrieval with multi-hop graph expansion

**Use case**: Complex queries requiring contextual information (default)

**Configuration**:
```python
{
  "retrieval_mode": "hybrid",
  "top_k": 10,
  "hybrid_chunk_weight": 0.7,
  "hybrid_entity_weight": 0.3,
  "max_expansion_depth": 2,
  "max_expanded_chunks": 30,
  "expansion_similarity_threshold": 0.7
}
```

**Process**:
1. Hybrid retrieval → seed chunks
2. Graph expansion → related chunks
3. Deduplication and ranking
4. Return top N chunks

**Performance**: 300-600ms (depth 2)

**Advantages**:
- Rich contextual information
- Surfaces indirect relationships
- Handles complex queries

**Limitations**:
- Higher latency
- More complex tuning
- Potential noise

## Vector Search

### Embedding Generation

**Query Embedding**:
```python
from core.embeddings import embedding_manager

query_embedding = await embedding_manager.get_embedding(query_text)
# Returns: List[float] with 1536 or 3072 dimensions
```

**Models**:
- `text-embedding-3-small` (1536 dims, fast, default)
- `text-embedding-3-large` (3072 dims, higher quality)
- `text-embedding-ada-002` (1536 dims, legacy)

### Vector Index

**Neo4j HNSW Index**:
```cypher
CALL db.index.vector.createNodeIndex(
  'chunk_embeddings',
  'Chunk',
  'embedding',
  1536,
  'cosine'
);
```

**Parameters**:
- `name`: Index name
- `label`: Node label
- `property`: Embedding property
- `dimensions`: Vector dimensions
- `similarity`: Distance metric (cosine, euclidean, dot-product)

### Vector Query

**Basic Query**:
```cypher
CALL db.index.vector.queryNodes(
  'chunk_embeddings',
  $top_k,
  $query_embedding
) YIELD node, score
RETURN node, score
ORDER BY score DESC;
```

**With Filters**:
```cypher
CALL db.index.vector.queryNodes(
  'chunk_embeddings',
  $top_k,
  $query_embedding
) YIELD node, score
MATCH (d:Document)-[:HAS_CHUNK]->(node)
WHERE d.id IN $document_ids
RETURN node, score
ORDER BY score DESC;
```

**Performance**:
- Cold query: 100-150ms
- Warm query (cached): 50-100ms
- HNSW provides O(log N) approximate search

## Entity Search

### Entity Name Matching

**Query**:
```cypher
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($query_term)
MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
WITH c, count(DISTINCT e) as entity_count
RETURN c, entity_count
ORDER BY entity_count DESC
LIMIT $top_k;
```

**Fuzzy Matching** (optional):
```cypher
CALL db.index.fulltext.queryNodes('entity_name_fulltext', $query_term)
YIELD node as e, score
MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
RETURN c, e, score;
```

### Entity Type Filtering

```cypher
MATCH (e:Entity {type: $entity_type})
WHERE toLower(e.name) CONTAINS toLower($query_term)
MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
RETURN DISTINCT c, count(e) as match_count
ORDER BY match_count DESC;
```

## Hybrid Score Calculation

### Weighted Combination

**Formula**:
```python
hybrid_score = (
    chunk_weight * vector_score +
    entity_weight * entity_match_score
)
```

**Default Weights**:
- `HYBRID_CHUNK_WEIGHT = 0.7` (vector similarity)
- `HYBRID_ENTITY_WEIGHT = 0.3` (entity matching)

**Normalization**:
```python
# Vector scores are already 0-1 (cosine similarity)
vector_score_normalized = vector_score

# Entity match count normalized by max matches
entity_score_normalized = entity_match_count / max_entity_matches
```

### Weight Tuning

**Entity-heavy queries** (e.g., "vCenter configuration"):
```python
HYBRID_CHUNK_WEIGHT = 0.5
HYBRID_ENTITY_WEIGHT = 0.5
```

**Semantic queries** (e.g., "explain high availability"):
```python
HYBRID_CHUNK_WEIGHT = 0.9
HYBRID_ENTITY_WEIGHT = 0.1
```

## Graph Expansion

### Expansion Strategies

**1. Chunk Similarity Expansion**

Traverse `SIMILAR_TO` relationships:
```cypher
MATCH (seed:Chunk)
WHERE seed.id IN $seed_ids
MATCH (seed)-[r:SIMILAR_TO*1..{max_depth}]-(related:Chunk)
WHERE all(rel IN r WHERE rel.strength >= $threshold)
WITH related, relationships(r) as path_edges
WITH related, reduce(s = 1.0, edge IN path_edges | s * edge.strength) as path_strength
RETURN DISTINCT related, path_strength
ORDER BY path_strength DESC
LIMIT $max_expanded_chunks;
```

**2. Entity Relationship Expansion**

Traverse entity relationships:
```cypher
MATCH (seed:Chunk)
WHERE seed.id IN $seed_ids
MATCH (seed)-[:CONTAINS_ENTITY]->(e1:Entity)
MATCH (e1)-[r:RELATED_TO*1..{max_depth}]-(e2:Entity)
MATCH (e2)<-[:CONTAINS_ENTITY]-(related:Chunk)
WHERE all(rel IN r WHERE rel.strength >= $threshold)
  AND seed.id <> related.id
WITH related, e1, e2, r
WITH related, count(DISTINCT e2) as entity_count,
     reduce(s = 1.0, rel IN r | s * rel.strength) as path_strength
RETURN DISTINCT related, entity_count * path_strength as expansion_score
ORDER BY expansion_score DESC
LIMIT $max_expanded_chunks;
```

**3. Combined Expansion**

Union of both strategies:
```python
async def graph_expansion(seed_chunks, max_depth, threshold, limit):
    # Chunk similarity expansion
    similarity_results = await chunk_similarity_expansion(
        seed_chunks, max_depth, threshold, limit
    )
    
    # Entity relationship expansion
    entity_results = await entity_relationship_expansion(
        seed_chunks, max_depth, threshold, limit
    )
    
    # Merge and deduplicate
    all_results = merge_results(similarity_results, entity_results)
    
    # Rank by combined score
    ranked = rank_by_score(all_results, limit)
    
    return ranked[:limit]
```

### Path Strength Calculation

**Simple Product**:
```python
path_strength = product([edge.strength for edge in path])
```

**Example**:
- Path: Chunk A → (0.8) → Chunk B → (0.9) → Chunk C
- Strength: 0.8 × 0.9 = 0.72

**Entity-Weighted**:
```python
path_strength = product([edge.strength for edge in path]) * entity_importance
```

**Multi-Path Aggregation**:
```python
# If multiple paths exist between chunks, take maximum
final_strength = max([path_strength(p) for p in paths])
```

### Expansion Depth Trade-offs

**Depth 1** (single hop):
- Latency: 50-100ms
- Context: Directly similar chunks only
- Precision: High
- Recall: Medium

**Depth 2** (two hops):
- Latency: 200-500ms
- Context: Extended neighborhood
- Precision: Medium-High
- Recall: High
- **Recommended default**

**Depth 3** (three hops):
- Latency: 1-2 seconds
- Context: Broad graph coverage
- Precision: Medium
- Recall: Very High
- Risk: Noise accumulation

### Threshold Tuning

**EXPANSION_SIMILARITY_THRESHOLD**:

**0.5** (permissive):
- More expansion
- More noise
- Better for broad exploration

**0.7** (balanced, default):
- Moderate expansion
- Good precision/recall balance

**0.9** (strict):
- Limited expansion
- High precision
- Risk of missing connections

## Reranking

### FlashRank Integration

**When to Enable**:
- Complex queries requiring fine-grained relevance
- Large candidate sets (50+)
- Quality-critical applications

**Model**: `ms-marco-MiniLM-L-12-v2`

**Process**:
```python
from rag.rerankers.flashrank_reranker import FlashRankReranker

reranker = FlashRankReranker()

# Rerank candidates
reranked = reranker.rerank(
    query=query,
    chunks=expanded_chunks,
    top_k=50
)
```

**Score Blending**:
```python
final_score = (
    flashrank_blend_weight * rerank_score +
    (1 - flashrank_blend_weight) * hybrid_score
)
```

**Configuration**:
```bash
FLASHRANK_ENABLED=true
FLASHRANK_MAX_CANDIDATES=50
FLASHRANK_BLEND_WEIGHT=0.5
FLASHRANK_BATCH_SIZE=32
```

**Performance Impact**:
- Latency: +200-500ms
- Quality improvement: +10-15%
- CPU/GPU intensive

## Context Document Filtering

### Filter by Document

Restrict retrieval to specific documents:

**API Request**:
```python
{
  "query": "backup procedure",
  "context_documents": ["VMware Guide", "Admin Manual"]
}
```

**Neo4j Query**:
```cypher
CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
YIELD node, score
MATCH (d:Document)-[:HAS_CHUNK]->(node)
WHERE d.filename IN $context_documents
RETURN node, score;
```

### Filter by Hashtag

Extract and apply hashtag filters:

**Query**: `"#vmware #backup procedure"`

**Parsing**:
```python
import re

def extract_hashtags(query: str) -> Tuple[str, List[str]]:
    hashtags = re.findall(r'#(\w+)', query)
    clean_query = re.sub(r'#\w+', '', query).strip()
    return clean_query, hashtags
```

**Filtering**:
```cypher
MATCH (c:Chunk)
WHERE any(tag IN $hashtags WHERE c.text CONTAINS tag)
// Continue with retrieval
```

## Caching Strategy

### Retrieval Cache

**Configuration**:
```bash
ENABLE_CACHING=true
RETRIEVAL_CACHE_SIZE=1000
RETRIEVAL_CACHE_TTL=60
```

**Cache Key**:
```python
cache_key = hash((
    query,
    top_k,
    context_documents,
    retrieval_mode,
    max_expansion_depth
))
```

**Cache Lookup**:
```python
if cache_key in retrieval_cache:
    return retrieval_cache[cache_key]

# Otherwise, perform retrieval
results = await retrieve(query, top_k, ...)
retrieval_cache[cache_key] = results
return results
```

**Hit Rate**: 20-30% typical

## Performance Optimization

### Query Optimization

**Use LIMIT early**:
```cypher
// Good
MATCH (c:Chunk) WHERE c.quality_score > 0.8
WITH c LIMIT 100
MATCH (c)-[:CONTAINS_ENTITY]->(e)
RETURN c, e;

// Bad
MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
WHERE c.quality_score > 0.8
RETURN c, e LIMIT 100;
```

**Use indexes**:
```cypher
// Ensure indexes exist
CREATE INDEX chunk_id FOR (c:Chunk) ON (c.id);
CREATE INDEX entity_name FOR (e:Entity) ON (e.name);
```

**Avoid Cartesian products**:
```cypher
// Good
MATCH (c:Chunk {id: $id})-[:CONTAINS_ENTITY]->(e)

// Bad
MATCH (c:Chunk), (e:Entity)
WHERE c.id = $id AND (c)-[:CONTAINS_ENTITY]->(e)
```

### Concurrent Retrieval

```python
import asyncio

async def retrieve_all(queries):
    tasks = [hybrid_retrieval(q, top_k) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

### Batch Embedding

```python
# Good: Batch processing
embeddings = await embedding_manager.embed_batch(texts)

# Bad: Sequential processing
embeddings = [await embedding_manager.get_embedding(t) for t in texts]
```

## Evaluation Metrics

### Precision @ K

```python
def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    relevant_retrieved = set(retrieved_k) & set(relevant)
    return len(relevant_retrieved) / k
```

### Recall @ K

```python
def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    relevant_retrieved = set(retrieved_k) & set(relevant)
    return len(relevant_retrieved) / len(relevant)
```

### Mean Reciprocal Rank

```python
def mrr(retrieved: List[str], relevant: List[str]) -> float:
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0
```

## Configuration Presets

### Fast (Latency-Optimized)

```python
{
  "retrieval_mode": "hybrid",
  "top_k": 5,
  "max_expansion_depth": 1,
  "max_expanded_chunks": 15,
  "expansion_similarity_threshold": 0.8,
  "flashrank_enabled": false
}
```

**Latency**: 200-400ms

### Balanced (Default)

```python
{
  "retrieval_mode": "hybrid",
  "top_k": 10,
  "max_expansion_depth": 2,
  "max_expanded_chunks": 30,
  "expansion_similarity_threshold": 0.7,
  "flashrank_enabled": false
}
```

**Latency**: 400-900ms

### Quality (Precision-Optimized)

```python
{
  "retrieval_mode": "hybrid",
  "top_k": 20,
  "max_expansion_depth": 3,
  "max_expanded_chunks": 50,
  "expansion_similarity_threshold": 0.6,
  "flashrank_enabled": true,
  "flashrank_max_candidates": 50
}
```

**Latency**: 1.5-3 seconds

## Testing

### Unit Tests

```bash
pytest tests/unit/test_retriever.py
```

### Integration Tests

```bash
pytest tests/integration/test_hybrid_retrieval.py
```

### Evaluation Script

```python
# tests/evaluation/test_retrieval_quality.py
def test_retrieval_precision():
    queries = load_test_queries()
    for query, expected_docs in queries:
        results = hybrid_retrieval(query, top_k=10)
        precision = precision_at_k(results, expected_docs, 10)
        assert precision >= 0.7, f"Low precision for '{query}': {precision}"
```

## Related Documentation

- [Graph RAG Pipeline](02-core-concepts/graph-rag-pipeline.md)
- [Caching System](02-core-concepts/caching-system.md)
- [Retriever Component](03-components/backend/retriever.md)
- [Reranking Feature](04-features/reranking.md)
