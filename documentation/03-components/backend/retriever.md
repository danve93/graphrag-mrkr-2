# Retriever Component

Hybrid retrieval system combining vector search, entity matching, and graph expansion.

## Overview

The retriever implements Amber's core retrieval strategy, combining multiple search techniques to provide comprehensive, contextually relevant results. It serves as the bridge between user queries and the knowledge graph.

**Location**: `rag/retriever.py`
**Key Functions**: `hybrid_retrieval()`, `expand_via_graph()`

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Hybrid Retrieval                     │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────┐              ┌──────────────┐      │
│  │   Vector    │              │   Entity     │      │
│  │   Search    │              │   Search     │      │
│  │  (70% wt)   │              │  (30% wt)    │      │
│  └──────┬──────┘              └──────┬───────┘      │
│         │                            │              │
│         └────────────┬───────────────┘              │
│                      ↓                               │
│              ┌───────────────┐                       │
│              │ Score Blending│                       │
│              └───────┬───────┘                       │
│                      ↓                               │
│              ┌───────────────┐                       │
│              │  Top-K Seeds  │                       │
│              └───────┬───────┘                       │
│                      ↓                               │
│              ┌───────────────┐                       │
│              │    Graph      │                       │
│              │  Expansion    │                       │
│              └───────┬───────┘                       │
│                      ↓                               │
│              ┌───────────────┐                       │
│              │ Final Results │                       │
│              └───────────────┘                       │
│                                                       │
└──────────────────────────────────────────────────────┘
```

## Core Functions

### Hybrid Retrieval

**Function**: `hybrid_retrieval()`

```python
from typing import List, Dict, Optional
from core.embeddings import embedding_manager
from core.graph_db import get_db

async def hybrid_retrieval(
    query: str,
    query_embedding: Optional[List[float]] = None,
    top_k: int = 10,
    context_documents: List[str] = None,
    chunk_weight: float = 0.7,
    entity_weight: float = 0.3,
    retrieval_mode: str = "hybrid"
) -> List[Dict]:
    """
    Perform hybrid retrieval combining vector and entity search.
    
    Args:
        query: User query text
        query_embedding: Precomputed query embedding (optional)
        top_k: Number of results to return
        context_documents: Filter by document names
        chunk_weight: Weight for vector similarity (0-1)
        entity_weight: Weight for entity matching (0-1)
        retrieval_mode: "vector" or "hybrid"
    
    Returns:
        List of chunk dictionaries with scores
    """
    db = get_db()
    context_documents = context_documents or []
    
    # Generate embedding if not provided
    if query_embedding is None:
        query_embedding = await embedding_manager.get_embedding(query)
    
    # Vector search
    vector_results = await vector_search(
        query_embedding=query_embedding,
        top_k=top_k * 2,  # Retrieve more for blending
        context_documents=context_documents
    )
    
    if retrieval_mode == "vector":
        return vector_results[:top_k]
    
    # Entity search
    entity_results = await entity_search(
        query=query,
        top_k=top_k * 2,
        context_documents=context_documents
    )
    
    # Blend scores
    blended = blend_retrieval_scores(
        vector_results=vector_results,
        entity_results=entity_results,
        chunk_weight=chunk_weight,
        entity_weight=entity_weight
    )
    
    # Return top-K
    return blended[:top_k]
```

### Vector Search

```python
async def vector_search(
    query_embedding: List[float],
    top_k: int,
    context_documents: List[str] = None
) -> List[Dict]:
    """
    Perform vector similarity search using Neo4j HNSW index.
    
    Args:
        query_embedding: Query vector
        top_k: Number of results
        context_documents: Optional document filter
    
    Returns:
        List of chunks with similarity scores
    """
    db = get_db()
    
    if context_documents:
        # With document filter
        query = """
        CALL db.index.vector.queryNodes(
            'chunk_embeddings',
            $top_k * 2,
            $query_embedding
        ) YIELD node, score
        MATCH (d:Document)-[:HAS_CHUNK]->(node)
        WHERE d.filename IN $context_documents
        RETURN node.id as chunk_id,
               node.text as text,
               node.page_number as page_number,
               d.id as document_id,
               d.filename as document_name,
               score
        ORDER BY score DESC
        LIMIT $top_k
        """
        params = {
            "query_embedding": query_embedding,
            "top_k": top_k,
            "context_documents": context_documents
        }
    else:
        # No filter
        query = """
        CALL db.index.vector.queryNodes(
            'chunk_embeddings',
            $top_k,
            $query_embedding
        ) YIELD node, score
        MATCH (d:Document)-[:HAS_CHUNK]->(node)
        RETURN node.id as chunk_id,
               node.text as text,
               node.page_number as page_number,
               d.id as document_id,
               d.filename as document_name,
               score
        ORDER BY score DESC
        """
        params = {
            "query_embedding": query_embedding,
            "top_k": top_k
        }
    
    results = await db.execute_read(query, params)
    
    return [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "page_number": r["page_number"],
            "document_id": r["document_id"],
            "document_name": r["document_name"],
            "score": float(r["score"]),
            "source": "vector"
        }
        for r in results
    ]
```

### Entity Search

```python
async def entity_search(
    query: str,
    top_k: int,
    context_documents: List[str] = None
) -> List[Dict]:
    """
    Search for chunks containing entities matching query terms.
    
    Args:
        query: Query text
        top_k: Number of results
        context_documents: Optional document filter
    
    Returns:
        List of chunks with entity match scores
    """
    db = get_db()
    
    # Extract query terms (simple tokenization)
    query_terms = [term.lower() for term in query.split() if len(term) > 2]
    
    if not query_terms:
        return []
    
    # Build query
    if context_documents:
        cypher_query = """
        MATCH (e:Entity)
        WHERE any(term IN $query_terms WHERE toLower(e.name) CONTAINS term)
        MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
        MATCH (d:Document)-[:HAS_CHUNK]->(c)
        WHERE d.filename IN $context_documents
        WITH c, d, count(DISTINCT e) as entity_count, collect(DISTINCT e.name) as matched_entities
        RETURN c.id as chunk_id,
               c.text as text,
               c.page_number as page_number,
               d.id as document_id,
               d.filename as document_name,
               entity_count,
               matched_entities
        ORDER BY entity_count DESC
        LIMIT $top_k
        """
        params = {
            "query_terms": query_terms,
            "top_k": top_k,
            "context_documents": context_documents
        }
    else:
        cypher_query = """
        MATCH (e:Entity)
        WHERE any(term IN $query_terms WHERE toLower(e.name) CONTAINS term)
        MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
        MATCH (d:Document)-[:HAS_CHUNK]->(c)
        WITH c, d, count(DISTINCT e) as entity_count, collect(DISTINCT e.name) as matched_entities
        RETURN c.id as chunk_id,
               c.text as text,
               c.page_number as page_number,
               d.id as document_id,
               d.filename as document_name,
               entity_count,
               matched_entities
        ORDER BY entity_count DESC
        LIMIT $top_k
        """
        params = {
            "query_terms": query_terms,
            "top_k": top_k
        }
    
    results = await db.execute_read(cypher_query, params)
    
    # Normalize entity count to 0-1 score
    max_count = max([r["entity_count"] for r in results], default=1)
    
    return [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "page_number": r["page_number"],
            "document_id": r["document_id"],
            "document_name": r["document_name"],
            "score": r["entity_count"] / max_count,
            "entity_count": r["entity_count"],
            "matched_entities": r["matched_entities"],
            "source": "entity"
        }
        for r in results
    ]
```

### Score Blending

```python
def blend_retrieval_scores(
    vector_results: List[Dict],
    entity_results: List[Dict],
    chunk_weight: float = 0.7,
    entity_weight: float = 0.3
) -> List[Dict]:
    """
    Blend vector and entity search scores.
    
    Args:
        vector_results: Results from vector search
        entity_results: Results from entity search
        chunk_weight: Weight for vector scores
        entity_weight: Weight for entity scores
    
    Returns:
        Combined and sorted results
    """
    # Index results by chunk_id
    chunks_by_id = {}
    
    # Add vector results
    for chunk in vector_results:
        chunk_id = chunk["chunk_id"]
        chunks_by_id[chunk_id] = {
            **chunk,
            "vector_score": chunk["score"],
            "entity_score": 0.0,
            "blended_score": chunk_weight * chunk["score"]
        }
    
    # Add/merge entity results
    for chunk in entity_results:
        chunk_id = chunk["chunk_id"]
        if chunk_id in chunks_by_id:
            # Merge scores
            existing = chunks_by_id[chunk_id]
            existing["entity_score"] = chunk["score"]
            existing["blended_score"] += entity_weight * chunk["score"]
            existing["entity_count"] = chunk.get("entity_count", 0)
            existing["matched_entities"] = chunk.get("matched_entities", [])
        else:
            # New chunk from entity search
            chunks_by_id[chunk_id] = {
                **chunk,
                "vector_score": 0.0,
                "entity_score": chunk["score"],
                "blended_score": entity_weight * chunk["score"]
            }
    
    # Sort by blended score
    results = sorted(
        chunks_by_id.values(),
        key=lambda c: c["blended_score"],
        reverse=True
    )
    
    # Set final score
    for chunk in results:
        chunk["score"] = chunk["blended_score"]
    
    return results
```

## Graph Expansion

### Chunk Similarity Expansion

```python
async def expand_via_chunk_similarity(
    seed_chunk_ids: List[str],
    max_depth: int = 2,
    threshold: float = 0.7,
    limit: int = 30
) -> List[Dict]:
    """
    Expand results via SIMILAR_TO relationships.
    
    Args:
        seed_chunk_ids: Starting chunk IDs
        max_depth: Maximum traversal hops
        threshold: Minimum edge strength
        limit: Maximum expanded chunks
    
    Returns:
        List of related chunks with path scores
    """
    db = get_db()
    
    query = """
    MATCH (seed:Chunk)
    WHERE seed.id IN $seed_ids
    MATCH path = (seed)-[r:SIMILAR_TO*1..$max_depth]-(related:Chunk)
    WHERE all(rel IN relationships(path) WHERE rel.strength >= $threshold)
      AND seed.id <> related.id
    WITH related,
         relationships(path) as rels,
         reduce(s = 1.0, rel IN relationships(path) | s * rel.strength) as path_strength
    MATCH (d:Document)-[:HAS_CHUNK]->(related)
    RETURN DISTINCT related.id as chunk_id,
           related.text as text,
           related.page_number as page_number,
           d.id as document_id,
           d.filename as document_name,
           max(path_strength) as score,
           length(rels) as hops
    ORDER BY score DESC
    LIMIT $limit
    """
    
    params = {
        "seed_ids": seed_chunk_ids,
        "max_depth": max_depth,
        "threshold": threshold,
        "limit": limit
    }
    
    results = await db.execute_read(query, params)
    
    return [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "page_number": r["page_number"],
            "document_id": r["document_id"],
            "document_name": r["document_name"],
            "score": float(r["score"]),
            "hops": r["hops"],
            "source": "similarity_expansion"
        }
        for r in results
    ]
```

### Entity Relationship Expansion

```python
async def expand_via_entity_relationships(
    seed_chunk_ids: List[str],
    max_depth: int = 2,
    threshold: float = 0.7,
    limit: int = 30
) -> List[Dict]:
    """
    Expand results via entity RELATED_TO relationships.
    
    Args:
        seed_chunk_ids: Starting chunk IDs
        max_depth: Maximum traversal hops
        threshold: Minimum edge strength
        limit: Maximum expanded chunks
    
    Returns:
        List of related chunks via entities
    """
    db = get_db()
    
    query = """
    MATCH (seed:Chunk)
    WHERE seed.id IN $seed_ids
    MATCH (seed)-[:CONTAINS_ENTITY]->(e1:Entity)
    MATCH path = (e1)-[r:RELATED_TO*1..$max_depth]-(e2:Entity)
    WHERE all(rel IN relationships(path) WHERE rel.strength >= $threshold)
    MATCH (related:Chunk)-[:CONTAINS_ENTITY]->(e2)
    WHERE seed.id <> related.id
    WITH related, e2,
         relationships(path) as rels,
         reduce(s = 1.0, rel IN relationships(path) | s * rel.strength) as path_strength
    MATCH (d:Document)-[:HAS_CHUNK]->(related)
    WITH related, d,
         count(DISTINCT e2) as entity_count,
         max(path_strength) as max_path_strength
    RETURN DISTINCT related.id as chunk_id,
           related.text as text,
           related.page_number as page_number,
           d.id as document_id,
           d.filename as document_name,
           entity_count * max_path_strength as score,
           entity_count
    ORDER BY score DESC
    LIMIT $limit
    """
    
    params = {
        "seed_ids": seed_chunk_ids,
        "max_depth": max_depth,
        "threshold": threshold,
        "limit": limit
    }
    
    results = await db.execute_read(query, params)
    
    return [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "page_number": r["page_number"],
            "document_id": r["document_id"],
            "document_name": r["document_name"],
            "score": float(r["score"]),
            "entity_count": r["entity_count"],
            "source": "entity_expansion"
        }
        for r in results
    ]
```

### Combined Expansion

```python
async def expand_via_graph(
    seed_chunks: List[Dict],
    max_depth: int = 2,
    threshold: float = 0.7,
    limit: int = 30
) -> List[Dict]:
    """
    Expand results using both similarity and entity relationships.
    
    Args:
        seed_chunks: Initial retrieval results
        max_depth: Maximum traversal hops
        threshold: Minimum edge strength
        limit: Maximum total chunks
    
    Returns:
        Merged and deduplicated expansion results
    """
    seed_ids = [c["chunk_id"] for c in seed_chunks]
    
    # Parallel expansion
    similarity_task = expand_via_chunk_similarity(
        seed_ids, max_depth, threshold, limit
    )
    entity_task = expand_via_entity_relationships(
        seed_ids, max_depth, threshold, limit
    )
    
    similarity_results, entity_results = await asyncio.gather(
        similarity_task, entity_task
    )
    
    # Merge results
    chunks_by_id = {}
    
    # Add seed chunks
    for chunk in seed_chunks:
        chunks_by_id[chunk["chunk_id"]] = chunk
    
    # Add similarity expansion
    for chunk in similarity_results:
        chunk_id = chunk["chunk_id"]
        if chunk_id not in chunks_by_id:
            chunks_by_id[chunk_id] = chunk
        else:
            # Take maximum score
            existing = chunks_by_id[chunk_id]
            existing["score"] = max(existing["score"], chunk["score"])
    
    # Add entity expansion
    for chunk in entity_results:
        chunk_id = chunk["chunk_id"]
        if chunk_id not in chunks_by_id:
            chunks_by_id[chunk_id] = chunk
        else:
            existing = chunks_by_id[chunk_id]
            existing["score"] = max(existing["score"], chunk["score"])
    
    # Sort and limit
    results = sorted(
        chunks_by_id.values(),
        key=lambda c: c["score"],
        reverse=True
    )
    
    return results[:limit]
```

## Caching

### Retrieval Cache

```python
from core.singletons import cache_manager
import hashlib
import json

async def hybrid_retrieval_cached(
    query: str,
    query_embedding: Optional[List[float]] = None,
    **kwargs
) -> List[Dict]:
    """Cached version of hybrid_retrieval."""
    from config.settings import settings
    
    if not settings.enable_caching:
        return await hybrid_retrieval(query, query_embedding, **kwargs)
    
    # Generate cache key
    cache_key = hashlib.sha256(
        json.dumps({
            "query": query,
            "top_k": kwargs.get("top_k", 10),
            "context_documents": kwargs.get("context_documents", []),
            "chunk_weight": kwargs.get("chunk_weight", 0.7),
            "entity_weight": kwargs.get("entity_weight", 0.3),
            "retrieval_mode": kwargs.get("retrieval_mode", "hybrid")
        }, sort_keys=True).encode()
    ).hexdigest()
    
    # Check cache
    if cache_key in cache_manager.retrieval_cache:
        return cache_manager.retrieval_cache[cache_key]
    
    # Execute retrieval
    results = await hybrid_retrieval(query, query_embedding, **kwargs)
    
    # Cache results
    cache_manager.retrieval_cache[cache_key] = results
    
    return results
```

## Performance Optimization

### Batch Vector Search

```python
async def batch_vector_search(
    query_embeddings: List[List[float]],
    top_k: int = 10
) -> List[List[Dict]]:
    """
    Perform batch vector searches in parallel.
    
    Args:
        query_embeddings: List of query vectors
        top_k: Results per query
    
    Returns:
        List of result lists
    """
    tasks = [
        vector_search(embedding, top_k)
        for embedding in query_embeddings
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

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
CREATE INDEX chunk_id FOR (c:Chunk) ON (c.id);
CREATE INDEX entity_name FOR (e:Entity) ON (e.name);
```

## Testing

### Unit Tests

```python
import pytest
from rag.retriever import hybrid_retrieval, blend_retrieval_scores

@pytest.mark.asyncio
async def test_hybrid_retrieval():
    results = await hybrid_retrieval(
        query="backup procedure",
        top_k=5
    )
    
    assert len(results) <= 5
    assert all("chunk_id" in r for r in results)
    assert all("score" in r for r in results)
    assert all(0 <= r["score"] <= 1 for r in results)

def test_blend_retrieval_scores():
    vector_results = [
        {"chunk_id": "1", "score": 0.9, "text": "text1"},
        {"chunk_id": "2", "score": 0.8, "text": "text2"}
    ]
    entity_results = [
        {"chunk_id": "2", "score": 0.7, "text": "text2"},
        {"chunk_id": "3", "score": 0.6, "text": "text3"}
    ]
    
    blended = blend_retrieval_scores(
        vector_results, entity_results,
        chunk_weight=0.7, entity_weight=0.3
    )
    
    assert len(blended) == 3
    assert blended[0]["chunk_id"] == "2"  # Highest combined score
    assert "blended_score" in blended[0]
```

### Integration Tests

```bash
pytest tests/integration/test_retriever.py -v
```

## Related Documentation

- [Retrieval Strategies](02-core-concepts/retrieval-strategies.md)
- [Graph Database](03-components/backend/graph-database.md)
- [Embeddings](03-components/backend/embeddings.md)
- [RAG Pipeline](03-components/backend/rag-pipeline.md)
