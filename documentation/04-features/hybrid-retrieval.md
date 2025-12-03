# Hybrid Retrieval Feature

Multi-signal retrieval combining vector search, graph expansion, and entity relationships.

## Overview

Hybrid Retrieval is Amber's core retrieval strategy that combines multiple signals to find the most relevant context for a query. It orchestrates vector similarity search, graph-based expansion through chunk and entity relationships, and optional reranking to deliver comprehensive, contextually rich results.

**Key Capabilities**:
- Vector similarity search via embeddings
- Graph expansion through SIMILAR_TO and entity relationships
- Multi-hop reasoning across documents
- Configurable signal weights and thresholds
- Optional FlashRank reranking

## Architecture

```
┌────────────────────────────────────────────────────────┐
│          Hybrid Retrieval Architecture                  │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Query: "How does authentication work?"                 │
│    ↓                                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Stage 1: Vector Search                         │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Embed query with same model as chunks     │  │   │
│  │  │ Neo4j vector index search                 │  │   │
│  │  │ Returns: Top-K chunks by cosine similarity│  │   │
│  │  │                                            │  │   │
│  │  │ Result: [c1: 0.89, c2: 0.85, c3: 0.82]    │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│    ↓                                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Stage 2: Graph Expansion                       │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ For each seed chunk:                      │  │   │
│  │  │   ├─ Follow SIMILAR_TO edges              │  │   │
│  │  │   │   (similarity >= threshold)           │  │   │
│  │  │   ├─ Follow entity MENTIONS               │  │   │
│  │  │   └─ Follow entity RELATED_TO             │  │   │
│  │  │                                            │  │   │
│  │  │ Multi-hop: Up to max_depth hops           │  │   │
│  │  │ Limit: max_expanded_chunks total          │  │   │
│  │  │                                            │  │   │
│  │  │ Result: [c1, c4, c5, c7, e1, e3]          │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│    ↓                                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Stage 3: Scoring & Combination                 │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Hybrid score calculation:                 │  │   │
│  │  │                                            │  │   │
│  │  │ chunk_score = (                           │  │   │
│  │  │   vector_score * chunk_weight +           │  │   │
│  │  │   graph_score * (1 - chunk_weight)        │  │   │
│  │  │ )                                          │  │   │
│  │  │                                            │  │   │
│  │  │ entity_score = importance * relevance     │  │   │
│  │  │                                            │  │   │
│  │  │ Result: Ranked candidates with scores     │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│    ↓                                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Stage 4: Optional Reranking                    │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ FlashRank reranker (if enabled)           │  │   │
│  │  │ Re-scores top candidates                  │  │   │
│  │  │ Blends rerank_score with hybrid_score     │  │   │
│  │  │                                            │  │   │
│  │  │ final_score = (                           │  │   │
│  │  │   rerank * blend_weight +                 │  │   │
│  │  │   hybrid * (1 - blend_weight)             │  │   │
│  │  │ )                                          │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│    ↓                                                    │
│  Final Context: Top-N chunks for generation            │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Core Implementation

### Hybrid Retrieval Function

```python
# rag/retriever.py
from typing import List, Dict, Optional
from core.embeddings import get_embedding
from core.graph_db import GraphDatabase
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

async def hybrid_retrieval(
    query: str,
    top_k: int = 10,
    expansion_depth: int = 1,
    expansion_threshold: float = 0.7,
    context_documents: Optional[List[str]] = None,
    enable_reranking: bool = None,
) -> List[Dict]:
    """
    Hybrid retrieval combining vector search and graph expansion.
    
    Args:
        query: Search query
        top_k: Number of final results
        expansion_depth: Max hops for graph expansion
        expansion_threshold: Min similarity for SIMILAR_TO edges
        context_documents: Optional document ID filter
        enable_reranking: Override reranking setting
    
    Returns:
        List of scored chunks with metadata
    """
    # Check cache
    if settings.enable_caching:
        cache_key = f"retrieval:{query}:{top_k}:{expansion_depth}"
        cached = retrieval_cache.get(cache_key)
        if cached:
            logger.info("Cache hit for retrieval")
            return cached
    
    # Step 1: Vector search
    logger.info(f"Vector search for: {query[:50]}...")
    query_embedding = await get_embedding(query)
    
    vector_candidates = await vector_search(
        query_embedding,
        top_k=settings.max_expanded_chunks or 20,
        context_documents=context_documents,
    )
    
    logger.info(f"Vector search returned {len(vector_candidates)} candidates")
    
    # Step 2: Graph expansion
    if expansion_depth > 0:
        logger.info(f"Expanding with depth={expansion_depth}")
        expanded = await expand_via_graph(
            seed_chunks=vector_candidates,
            max_depth=expansion_depth,
            similarity_threshold=expansion_threshold,
            max_total=settings.max_expanded_chunks or 50,
        )
        logger.info(f"Expanded to {len(expanded)} total candidates")
    else:
        expanded = vector_candidates
    
    # Step 3: Hybrid scoring
    scored = calculate_hybrid_scores(
        candidates=expanded,
        query_embedding=query_embedding,
        chunk_weight=settings.hybrid_chunk_weight,
        entity_weight=settings.hybrid_entity_weight,
    )
    
    # Step 4: Optional reranking
    if enable_reranking is None:
        enable_reranking = settings.flashrank_enabled
    
    if enable_reranking and len(scored) > 0:
        logger.info("Applying FlashRank reranking")
        scored = await apply_reranking(
            query=query,
            candidates=scored[:settings.flashrank_max_candidates],
            blend_weight=settings.flashrank_blend_weight,
        )
    
    # Take top-K
    final = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
    
    # Cache result
    if settings.enable_caching:
        retrieval_cache.set(cache_key, final)
    
    return final
```

### Vector Search

```python
async def vector_search(
    query_embedding: List[float],
    top_k: int = 20,
    context_documents: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Vector similarity search using Neo4j vector index.
    
    Args:
        query_embedding: Query embedding vector
        top_k: Number of results
        context_documents: Optional document ID filter
    
    Returns:
        List of chunks with vector scores
    """
    db = GraphDatabase.get_instance()
    
    # Build Cypher query
    if context_documents:
        doc_filter = "AND c.document_id IN $doc_ids"
    else:
        doc_filter = ""
    
    query = f"""
        CALL db.index.vector.queryNodes(
            'chunk_embeddings',
            $top_k,
            $embedding
        )
        YIELD node AS c, score
        WHERE 1=1 {doc_filter}
        MATCH (d:Document)-[:HAS_CHUNK]->(c)
        RETURN 
            c.id AS chunk_id,
            c.text AS text,
            c.chunk_index AS chunk_index,
            c.quality_score AS quality_score,
            d.id AS document_id,
            d.name AS document_name,
            score AS vector_score
        ORDER BY score DESC
        LIMIT $top_k
    """
    
    params = {
        "embedding": query_embedding,
        "top_k": top_k,
    }
    
    if context_documents:
        params["doc_ids"] = context_documents
    
    results = db.execute_read(query, params)
    
    return [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "chunk_index": r["chunk_index"],
            "document_id": r["document_id"],
            "document_name": r["document_name"],
            "vector_score": r["vector_score"],
            "quality_score": r.get("quality_score"),
            "graph_score": 0.0,
            "entity_score": 0.0,
        }
        for r in results
    ]
```

### Graph Expansion

```python
async def expand_via_graph(
    seed_chunks: List[Dict],
    max_depth: int = 1,
    similarity_threshold: float = 0.7,
    max_total: int = 50,
) -> List[Dict]:
    """
    Expand seed chunks via graph relationships.
    
    Traverses:
        - SIMILAR_TO edges (chunk similarity)
        - MENTIONS edges (chunk → entity)
        - RELATED_TO edges (entity relationships)
    
    Args:
        seed_chunks: Initial vector search results
        max_depth: Maximum traversal depth
        similarity_threshold: Min similarity for SIMILAR_TO edges
        max_total: Maximum expanded candidates
    
    Returns:
        Combined seed + expanded chunks
    """
    db = GraphDatabase.get_instance()
    
    seed_ids = [c["chunk_id"] for c in seed_chunks]
    expanded = {c["chunk_id"]: c for c in seed_chunks}
    
    for depth in range(1, max_depth + 1):
        if len(expanded) >= max_total:
            break
        
        # Expand current frontier
        query = """
            MATCH (seed:Chunk)
            WHERE seed.id IN $seed_ids
            
            // Follow SIMILAR_TO edges
            OPTIONAL MATCH (seed)-[sim:SIMILAR_TO]->(similar:Chunk)
            WHERE sim.similarity >= $threshold
            
            // Follow entity paths
            OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)
            OPTIONAL MATCH (e)-[:RELATED_TO]-(e2:Entity)
            OPTIONAL MATCH (c2:Chunk)-[:MENTIONS]->(e2)
            
            WITH DISTINCT 
                COALESCE(similar, c2) AS expanded_chunk,
                COALESCE(sim.similarity, 0.5) AS graph_score
            WHERE expanded_chunk IS NOT NULL
            
            MATCH (d:Document)-[:HAS_CHUNK]->(expanded_chunk)
            
            RETURN 
                expanded_chunk.id AS chunk_id,
                expanded_chunk.text AS text,
                expanded_chunk.chunk_index AS chunk_index,
                d.id AS document_id,
                d.name AS document_name,
                graph_score,
                expanded_chunk.quality_score AS quality_score
            LIMIT $limit
        """
        
        results = db.execute_read(query, {
            "seed_ids": seed_ids,
            "threshold": similarity_threshold,
            "limit": max_total - len(expanded),
        })
        
        # Add new chunks
        for r in results:
            if r["chunk_id"] not in expanded:
                expanded[r["chunk_id"]] = {
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "chunk_index": r["chunk_index"],
                    "document_id": r["document_id"],
                    "document_name": r["document_name"],
                    "vector_score": 0.0,
                    "graph_score": r["graph_score"],
                    "quality_score": r.get("quality_score"),
                    "entity_score": 0.0,
                    "depth": depth,
                }
        
        logger.info(f"Depth {depth}: {len(expanded)} total candidates")
    
    return list(expanded.values())
```

### Hybrid Scoring

```python
def calculate_hybrid_scores(
    candidates: List[Dict],
    query_embedding: List[float],
    chunk_weight: float = 0.7,
    entity_weight: float = 0.3,
) -> List[Dict]:
    """
    Calculate hybrid scores combining multiple signals.
    
    Formula:
        chunk_component = vector_score * chunk_weight + graph_score * (1 - chunk_weight)
        entity_component = entity_score * entity_weight
        final_score = chunk_component + entity_component
    
    Args:
        candidates: List of chunks with individual scores
        query_embedding: Query embedding for entity scoring
        chunk_weight: Weight for chunk signals (vs graph signals)
        entity_weight: Weight for entity signals
    
    Returns:
        Candidates with hybrid scores
    """
    for candidate in candidates:
        # Normalize scores to [0, 1]
        vector_score = candidate.get("vector_score", 0.0)
        graph_score = candidate.get("graph_score", 0.0)
        entity_score = candidate.get("entity_score", 0.0)
        
        # Chunk component (vector + graph)
        chunk_component = (
            vector_score * chunk_weight +
            graph_score * (1 - chunk_weight)
        )
        
        # Entity component
        entity_component = entity_score * entity_weight
        
        # Combined hybrid score
        hybrid_score = chunk_component + entity_component
        
        # Apply quality score boost if available
        if candidate.get("quality_score"):
            quality_boost = (candidate["quality_score"] - 0.5) * 0.1
            hybrid_score += quality_boost
        
        candidate["score"] = hybrid_score
    
    return candidates
```

## Reranking Integration

### FlashRank Reranking

```python
# rag/rerankers/flashrank.py
from flashrank import Ranker, RerankRequest
from typing import List, Dict

class FlashRankReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.ranker = Ranker(model_name=model_name, cache_dir="data/flashrank_cache")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Rerank candidates using FlashRank.
        
        Args:
            query: Search query
            candidates: Chunks to rerank
            top_k: Number to return
        
        Returns:
            Reranked candidates with rerank_score
        """
        # Prepare passages
        passages = [
            {"id": i, "text": c["text"]}
            for i, c in enumerate(candidates)
        ]
        
        # Rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        # Map scores back
        score_map = {r["id"]: r["score"] for r in results[:top_k]}
        
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = score_map.get(i, 0.0)
        
        return candidates

async def apply_reranking(
    query: str,
    candidates: List[Dict],
    blend_weight: float = 0.5,
) -> List[Dict]:
    """
    Apply reranking and blend with hybrid scores.
    
    Args:
        query: Search query
        candidates: Chunks with hybrid scores
        blend_weight: Weight for rerank score (vs hybrid score)
    
    Returns:
        Candidates with blended final scores
    """
    reranker = FlashRankReranker()
    reranked = reranker.rerank(query, candidates)
    
    # Blend scores
    for candidate in reranked:
        hybrid_score = candidate.get("score", 0.0)
        rerank_score = candidate.get("rerank_score", 0.0)
        
        candidate["score"] = (
            rerank_score * blend_weight +
            hybrid_score * (1 - blend_weight)
        )
    
    return reranked
```

## Configuration

### Retrieval Parameters

```python
# config/settings.py
class Settings(BaseSettings):
    # Vector search
    retrieval_top_k: int = 10
    
    # Graph expansion
    max_expansion_depth: int = 1
    expansion_similarity_threshold: float = 0.7
    max_expanded_chunks: int = 50
    
    # Hybrid scoring
    hybrid_chunk_weight: float = 0.7  # Vector + graph
    hybrid_entity_weight: float = 0.3
    
    # Reranking
    flashrank_enabled: bool = False
    flashrank_model_name: str = "ms-marco-MiniLM-L-12-v2"
    flashrank_max_candidates: int = 20
    flashrank_blend_weight: float = 0.5
    flashrank_batch_size: int = 32
```

### Runtime Tuning

```python
# api/models.py
class ChatRequest(BaseModel):
    message: str
    session_id: str
    
    # Retrieval overrides
    retrieval_mode: str = "hybrid"  # "vector", "graph", "hybrid"
    top_k: Optional[int] = None
    expansion_depth: Optional[int] = None
    expansion_threshold: Optional[float] = None
    enable_reranking: Optional[bool] = None
```

## Usage Examples

### Basic Hybrid Retrieval

```python
from rag.retriever import hybrid_retrieval

# Simple query
results = await hybrid_retrieval(
    query="How does authentication work?",
    top_k=5,
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:100]}...")
    print(f"Source: {result['document_name']}")
    print()
```

### Document-Scoped Search

```python
# Search within specific documents
results = await hybrid_retrieval(
    query="API rate limits",
    top_k=5,
    context_documents=["doc_id_1", "doc_id_2"],
)
```

### Deep Expansion

```python
# Multi-hop reasoning
results = await hybrid_retrieval(
    query="Database migration procedure",
    top_k=10,
    expansion_depth=2,  # 2-hop traversal
    expansion_threshold=0.6,  # Lower threshold for more expansion
)
```

### With Reranking

```python
# Enable reranking for better quality
results = await hybrid_retrieval(
    query="Security best practices",
    top_k=5,
    enable_reranking=True,
)
```

## Performance Optimization

### Caching Strategy

```python
# Cache at multiple levels
from core.cache import retrieval_cache

# Cache key includes query + parameters
cache_key = f"retrieval:{query_hash}:{top_k}:{depth}"

# Short TTL (60s) for fresh results
cached = retrieval_cache.get(cache_key)
if cached:
    return cached

# Store result
retrieval_cache.set(cache_key, results, ttl=60)
```

### Batch Processing

```python
# Process multiple queries efficiently
async def batch_retrieve(queries: List[str], top_k: int = 5):
    # Batch embed queries
    embeddings = await batch_embed(queries)
    
    # Parallel retrieval
    tasks = [
        hybrid_retrieval(q, top_k=top_k)
        for q in queries
    ]
    
    return await asyncio.gather(*tasks)
```

## Testing

### Unit Tests

```python
# tests/test_retrieval.py
import pytest
from rag.retriever import hybrid_retrieval, vector_search, expand_via_graph

@pytest.mark.asyncio
async def test_vector_search():
    embedding = [0.1] * 1536
    results = await vector_search(embedding, top_k=5)
    
    assert len(results) <= 5
    assert all("vector_score" in r for r in results)
    assert all(0 <= r["vector_score"] <= 1 for r in results)

@pytest.mark.asyncio
async def test_hybrid_retrieval():
    results = await hybrid_retrieval("test query", top_k=3)
    
    assert len(results) <= 3
    assert all("score" in r for r in results)
    assert results == sorted(results, key=lambda x: x["score"], reverse=True)

@pytest.mark.asyncio
async def test_graph_expansion():
    seeds = [{"chunk_id": "c1", "vector_score": 0.9}]
    expanded = await expand_via_graph(seeds, max_depth=1)
    
    assert len(expanded) >= len(seeds)
```

## Troubleshooting

### Common Issues

**Issue**: Low recall (missing relevant chunks)
```python
# Solution 1: Increase expansion
expansion_depth = 2
max_expanded_chunks = 100

# Solution 2: Lower similarity threshold
expansion_threshold = 0.5

# Solution 3: Adjust hybrid weights
hybrid_chunk_weight = 0.5  # Balance vector and graph more
```

**Issue**: Slow retrieval performance
```python
# Solution 1: Enable caching
ENABLE_CACHING=true

# Solution 2: Reduce expansion
max_expansion_depth = 1
max_expanded_chunks = 30

# Solution 3: Disable reranking
flashrank_enabled = false
```

**Issue**: Poor reranking results
```python
# Solution: Adjust blend weight
flashrank_blend_weight = 0.7  # Trust reranker more

# Or disable if not helping
enable_reranking = False
```

## Related Documentation

- [Retrieval Strategies](02-core-concepts/retrieval-strategies.md)
- [Graph Database](03-components/backend/graph-database.md)
- [Embeddings](03-components/backend/embeddings.md)
- [RAG Pipeline](03-components/backend/graph-rag.md)
