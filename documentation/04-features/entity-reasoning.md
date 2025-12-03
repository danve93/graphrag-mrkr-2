# Entity Reasoning Feature

Multi-hop entity relationship traversal for contextual knowledge discovery.

## Overview

Entity Reasoning enables Amber to discover relationships between concepts across documents by traversing entity graphs. It identifies entities in queries, follows relationship paths through the knowledge graph, and uses entity importance and connection strength to surface relevant multi-hop evidence for generation.

**Key Capabilities**:
- Query entity extraction and matching
- Multi-hop relationship traversal
- Path strength evaluation
- Entity importance scoring
- Cross-document reasoning

## Architecture

```
┌────────────────────────────────────────────────────────┐
│          Entity Reasoning Architecture                  │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Query: "How does User authentication relate to API?"   │
│    ↓                                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 1: Entity Extraction from Query          │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ LLM extracts entities:                    │  │   │
│  │  │   • User (Entity)                         │  │   │
│  │  │   • authentication (Concept)              │  │   │
│  │  │   • API (Service)                         │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│    ↓                                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 2: Entity Matching in Graph              │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Fuzzy match query entities to graph:      │  │   │
│  │  │   • "User" → User (exact)                 │  │   │
│  │  │   • "authentication" → Authentication     │  │   │
│  │  │   • "API" → REST API                      │  │   │
│  │  │                                            │  │   │
│  │  │ Matched entities: [e1, e2, e3]            │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│    ↓                                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 3: Multi-Hop Path Discovery              │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Traverse RELATED_TO relationships:        │  │   │
│  │  │                                            │  │   │
│  │  │ Path 1: User → Authentication → API       │  │   │
│  │  │   Strength: 0.9 * 0.85 = 0.765            │  │   │
│  │  │                                            │  │   │
│  │  │ Path 2: User → Token → API                │  │   │
│  │  │   Strength: 0.8 * 0.9 = 0.72              │  │   │
│  │  │                                            │  │   │
│  │  │ Path 3: Authentication → OAuth → API      │  │   │
│  │  │   Strength: 0.7 * 0.85 = 0.595            │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│    ↓                                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 4: Retrieve Related Chunks                │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ For each entity in high-strength paths:  │  │   │
│  │  │   • Find chunks mentioning entity         │  │   │
│  │  │   • Score by path strength + importance   │  │   │
│  │  │   • Deduplicate across paths              │  │   │
│  │  │                                            │  │   │
│  │  │ Result: Chunks with entity context        │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│    ↓                                                    │
│  Final Context: Entity-enriched chunks for generation  │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Core Implementation

### Entity Reasoning Pipeline

```python
# rag/nodes/graph_reasoning.py
from typing import List, Dict, Set
from core.entity_extraction import extract_entities_from_query
from core.graph_db import GraphDatabase
import logging

logger = logging.getLogger(__name__)

async def entity_reasoning(
    query: str,
    max_hops: int = 2,
    min_relationship_strength: float = 0.5,
    max_entities: int = 20,
) -> List[Dict]:
    """
    Perform entity-based reasoning over knowledge graph.
    
    Args:
        query: User query
        max_hops: Maximum relationship traversal depth
        min_relationship_strength: Minimum edge weight threshold
        max_entities: Maximum entities to consider
    
    Returns:
        List of chunks with entity reasoning scores
    """
    # Step 1: Extract entities from query
    logger.info("Extracting entities from query")
    query_entities = await extract_entities_from_query(query)
    
    if not query_entities:
        logger.info("No entities found in query")
        return []
    
    logger.info(f"Found {len(query_entities)} query entities: {query_entities}")
    
    # Step 2: Match entities in graph
    matched_entities = await match_entities_in_graph(query_entities)
    
    if not matched_entities:
        logger.info("No matching entities in graph")
        return []
    
    logger.info(f"Matched {len(matched_entities)} entities in graph")
    
    # Step 3: Find multi-hop paths
    paths = await discover_entity_paths(
        seed_entities=matched_entities,
        max_hops=max_hops,
        min_strength=min_relationship_strength,
    )
    
    logger.info(f"Discovered {len(paths)} entity paths")
    
    # Step 4: Retrieve chunks mentioning path entities
    chunks = await retrieve_entity_chunks(
        paths=paths,
        max_chunks=max_entities,
    )
    
    logger.info(f"Retrieved {len(chunks)} entity-related chunks")
    
    return chunks
```

### Query Entity Extraction

```python
# core/entity_extraction.py (addition)
async def extract_entities_from_query(query: str) -> List[str]:
    """
    Extract entities from user query using LLM.
    
    Args:
        query: User query text
    
    Returns:
        List of entity names
    """
    from core.llm import get_llm_client
    
    prompt = f"""Extract key entities and concepts from this query.
Return only entity names, one per line.

Query: {query}

Entities:"""

    llm = get_llm_client()
    response = await llm.generate(
        prompt=prompt,
        temperature=0.0,
        max_tokens=200,
    )
    
    # Parse entity names
    entities = [
        line.strip().strip('-•*')
        for line in response.split('\n')
        if line.strip()
    ]
    
    return entities[:10]  # Limit to top 10
```

### Entity Matching

```python
async def match_entities_in_graph(
    query_entities: List[str],
    similarity_threshold: float = 0.8,
) -> List[Dict]:
    """
    Match query entities to entities in knowledge graph.
    
    Uses fuzzy matching and embeddings for flexible matching.
    
    Args:
        query_entities: Entity names from query
        similarity_threshold: Min similarity for fuzzy match
    
    Returns:
        Matched graph entities with metadata
    """
    db = GraphDatabase.get_instance()
    
    matched = []
    
    for entity_name in query_entities:
        # Try exact match first
        query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower($name)
            RETURN 
                e.id AS id,
                e.name AS name,
                e.type AS type,
                e.importance AS importance,
                size((e)-[:RELATED_TO]-()) AS degree
            LIMIT 1
        """
        
        results = db.execute_read(query, {"name": entity_name})
        
        if results:
            matched.append(results[0])
            continue
        
        # Fuzzy match by similarity
        query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $partial
            RETURN 
                e.id AS id,
                e.name AS name,
                e.type AS type,
                e.importance AS importance,
                size((e)-[:RELATED_TO]-()) AS degree
            ORDER BY e.importance DESC
            LIMIT 1
        """
        
        results = db.execute_read(query, {"partial": entity_name.lower()})
        
        if results:
            matched.append(results[0])
    
    return matched
```

### Path Discovery

```python
async def discover_entity_paths(
    seed_entities: List[Dict],
    max_hops: int = 2,
    min_strength: float = 0.5,
) -> List[Dict]:
    """
    Discover paths between seed entities through relationships.
    
    Args:
        seed_entities: Starting entities
        max_hops: Maximum path length
        min_strength: Minimum relationship strength
    
    Returns:
        Paths with strength scores
    """
    db = GraphDatabase.get_instance()
    
    seed_ids = [e["id"] for e in seed_entities]
    
    # Find paths between seed entities
    query = """
        MATCH path = (e1:Entity)-[r:RELATED_TO*1..$max_hops]-(e2:Entity)
        WHERE e1.id IN $seed_ids AND e2.id IN $seed_ids
        AND e1.id <> e2.id
        AND all(rel in relationships(path) WHERE rel.strength >= $min_strength)
        
        WITH path, relationships(path) AS rels, nodes(path) AS nodes
        
        // Calculate path strength (product of edge strengths)
        WITH path, 
             reduce(s = 1.0, rel in rels | s * rel.strength) AS path_strength,
             [n in nodes | {
                 id: n.id, 
                 name: n.name, 
                 type: n.type,
                 importance: n.importance
             }] AS entities
        
        RETURN 
            entities,
            path_strength,
            length(path) AS path_length
        ORDER BY path_strength DESC
        LIMIT 50
    """
    
    results = db.execute_read(query, {
        "seed_ids": seed_ids,
        "max_hops": max_hops,
        "min_strength": min_strength,
    })
    
    paths = []
    for r in results:
        paths.append({
            "entities": r["entities"],
            "strength": r["path_strength"],
            "length": r["path_length"],
        })
    
    return paths
```

### Chunk Retrieval

```python
async def retrieve_entity_chunks(
    paths: List[Dict],
    max_chunks: int = 20,
) -> List[Dict]:
    """
    Retrieve chunks mentioning entities in discovered paths.
    
    Args:
        paths: Entity paths with strength scores
        max_chunks: Maximum chunks to return
    
    Returns:
        Chunks scored by entity relevance
    """
    db = GraphDatabase.get_instance()
    
    # Collect all entities from paths with scores
    entity_scores = {}
    for path in paths:
        for entity in path["entities"]:
            entity_id = entity["id"]
            importance = entity.get("importance", 0.5)
            path_strength = path["strength"]
            
            # Score = importance * path_strength
            score = importance * path_strength
            
            if entity_id not in entity_scores:
                entity_scores[entity_id] = score
            else:
                entity_scores[entity_id] = max(entity_scores[entity_id], score)
    
    entity_ids = list(entity_scores.keys())
    
    # Retrieve chunks mentioning these entities
    query = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE e.id IN $entity_ids
        MATCH (d:Document)-[:HAS_CHUNK]->(c)
        
        WITH c, d, e, $entity_scores[e.id] AS entity_score
        
        RETURN DISTINCT
            c.id AS chunk_id,
            c.text AS text,
            c.chunk_index AS chunk_index,
            d.id AS document_id,
            d.name AS document_name,
            entity_score,
            collect(DISTINCT e.name) AS mentioned_entities
        ORDER BY entity_score DESC
        LIMIT $limit
    """
    
    results = db.execute_read(query, {
        "entity_ids": entity_ids,
        "entity_scores": entity_scores,
        "limit": max_chunks,
    })
    
    chunks = []
    for r in results:
        chunks.append({
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "chunk_index": r["chunk_index"],
            "document_id": r["document_id"],
            "document_name": r["document_name"],
            "entity_score": r["entity_score"],
            "mentioned_entities": r["mentioned_entities"],
            "reasoning_type": "entity_path",
        })
    
    return chunks
```

## Integration with Hybrid Retrieval

### Combined Scoring

```python
# rag/retriever.py (extension)
async def hybrid_retrieval_with_entities(
    query: str,
    top_k: int = 10,
) -> List[Dict]:
    """
    Combine vector search, graph expansion, and entity reasoning.
    """
    # Standard hybrid retrieval
    hybrid_results = await hybrid_retrieval(query, top_k=top_k * 2)
    
    # Entity reasoning
    entity_results = await entity_reasoning(query, max_entities=top_k)
    
    # Merge and deduplicate
    all_results = {r["chunk_id"]: r for r in hybrid_results}
    
    for entity_chunk in entity_results:
        chunk_id = entity_chunk["chunk_id"]
        
        if chunk_id in all_results:
            # Boost existing chunk with entity score
            all_results[chunk_id]["entity_score"] = entity_chunk["entity_score"]
            all_results[chunk_id]["mentioned_entities"] = entity_chunk["mentioned_entities"]
        else:
            # Add new entity-discovered chunk
            all_results[chunk_id] = entity_chunk
    
    # Recalculate final scores
    for chunk in all_results.values():
        base_score = chunk.get("score", 0.0)
        entity_score = chunk.get("entity_score", 0.0)
        
        # Boost with entity signal
        chunk["score"] = base_score + (entity_score * 0.3)
    
    # Sort and return top-K
    final = sorted(
        all_results.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]
    
    return final
```

## Configuration

### Entity Reasoning Settings

```python
# config/settings.py
class Settings(BaseSettings):
    # Entity reasoning
    enable_entity_reasoning: bool = True
    max_entity_hops: int = 2
    min_relationship_strength: float = 0.5
    max_reasoning_entities: int = 20
    entity_reasoning_weight: float = 0.3
    
    # Entity extraction
    query_entity_extraction: bool = True
    max_query_entities: int = 10
```

## Usage Examples

### Basic Entity Reasoning

```python
from rag.nodes.graph_reasoning import entity_reasoning

# Find entity-related context
chunks = await entity_reasoning(
    query="How does User authentication work with JWT?",
    max_hops=2,
)

for chunk in chunks:
    print(f"Score: {chunk['entity_score']:.3f}")
    print(f"Entities: {', '.join(chunk['mentioned_entities'])}")
    print(f"Text: {chunk['text'][:100]}...")
    print()
```

### Deep Reasoning

```python
# Multi-hop exploration
chunks = await entity_reasoning(
    query="Database migration impact on API performance",
    max_hops=3,  # Longer paths
    min_relationship_strength=0.3,  # More permissive
)
```

### Combined with Retrieval

```python
# Full pipeline
results = await hybrid_retrieval_with_entities(
    query="Security considerations for OAuth implementation",
    top_k=10,
)

for result in results:
    if result.get("mentioned_entities"):
        print(f"Entity-enhanced: {result['mentioned_entities']}")
```

## Visualization

### Path Explanation

```python
def explain_entity_path(path: Dict) -> str:
    """Generate human-readable path explanation."""
    entities = path["entities"]
    strength = path["strength"]
    
    entity_names = [e["name"] for e in entities]
    path_str = " → ".join(entity_names)
    
    return f"Path: {path_str} (strength: {strength:.2f})"

# Example output:
# Path: User → Authentication → JWT Token → API (strength: 0.68)
```

## Testing

### Unit Tests

```python
# tests/test_entity_reasoning.py
import pytest
from rag.nodes.graph_reasoning import (
    entity_reasoning,
    match_entities_in_graph,
    discover_entity_paths,
)

@pytest.mark.asyncio
async def test_entity_extraction_from_query():
    entities = await extract_entities_from_query(
        "How does User authentication work?"
    )
    
    assert len(entities) > 0
    assert any("user" in e.lower() for e in entities)

@pytest.mark.asyncio
async def test_entity_matching():
    query_entities = ["User", "API"]
    matched = await match_entities_in_graph(query_entities)
    
    assert len(matched) > 0
    assert all("id" in e for e in matched)

@pytest.mark.asyncio
async def test_path_discovery():
    seeds = [
        {"id": "e1", "name": "User"},
        {"id": "e2", "name": "API"},
    ]
    
    paths = await discover_entity_paths(seeds, max_hops=2)
    
    assert all("strength" in p for p in paths)
    assert all(p["strength"] > 0 for p in paths)

@pytest.mark.asyncio
async def test_entity_reasoning_pipeline():
    chunks = await entity_reasoning("User authentication", max_hops=2)
    
    assert all("entity_score" in c for c in chunks)
    assert all("mentioned_entities" in c for c in chunks)
```

## Performance Optimization

### Entity Caching

```python
# Cache entity lookups
from core.cache import entity_label_cache

async def match_entities_cached(query_entities: List[str]):
    matched = []
    
    for entity_name in query_entities:
        # Check cache
        cached = entity_label_cache.get(entity_name.lower())
        if cached:
            matched.append(cached)
            continue
        
        # Lookup and cache
        result = await match_entity_in_graph(entity_name)
        if result:
            entity_label_cache.set(entity_name.lower(), result)
            matched.append(result)
    
    return matched
```

### Path Pruning

```python
# Limit path exploration
def prune_paths(paths: List[Dict], max_paths: int = 20) -> List[Dict]:
    """Keep only highest-strength paths."""
    return sorted(
        paths,
        key=lambda p: p["strength"],
        reverse=True
    )[:max_paths]
```

## Troubleshooting

### Common Issues

**Issue**: No entities found in query
```python
# Solution: Improve entity extraction prompt
prompt = f"""Extract ALL entities, concepts, and key terms from this query.
Include proper nouns, technical terms, and concepts.

Query: {query}

Entities (one per line):"""
```

**Issue**: Weak entity paths
```python
# Solution: Lower strength threshold
min_relationship_strength = 0.3

# Or increase max hops
max_hops = 3
```

**Issue**: Too many irrelevant entities
```python
# Solution: Filter by entity importance
min_entity_importance = 0.5

query += """
WHERE all(n in nodes(path) WHERE n.importance >= $min_importance)
"""
```

## Related Documentation

- [Entity Extraction](03-components/backend/entity-extraction.md)
- [Graph Database](03-components/backend/graph-database.md)
- [Hybrid Retrieval](04-features/hybrid-retrieval.md)
- [Entity Types](02-core-concepts/entity-types.md)
