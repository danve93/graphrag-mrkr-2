# Graph Expansion Flow

Multi-hop graph traversal during hybrid retrieval.

## Overview

This document traces how the system expands from initial vector search candidates through graph edges (SIMILAR_TO, MENTIONS, RELATED_TO) to discover contextually relevant chunks during retrieval. It shows the multi-hop traversal algorithm, similarity threshold filtering, and deduplication.

## Flow Diagram

```
Query: "What are the backup procedures for VxRail?"
Initial Vector Search: 10 candidate chunks
│
├─> 1. Vector Search Results
│   ├─ Chunk A (score: 0.92) - "VxRail backup procedures..."
│   ├─ Chunk B (score: 0.88) - "To configure backup policies..."
│   ├─ Chunk C (score: 0.85) - "Data protection for VxRail..."
│   └─ ... 7 more chunks
│
├─> 2. Graph Expansion (Depth 1)
│   │
│   ├─> For Chunk A:
│   │   │
│   │   ├─> Traverse SIMILAR_TO Edges
│   │   │   ├─ Query: MATCH (c)-[r:SIMILAR_TO]->(c2) WHERE r.similarity >= 0.7
│   │   │   ├─ Found: Chunk D (similarity: 0.82)
│   │   │   └─ Found: Chunk E (similarity: 0.75)
│   │   │
│   │   ├─> Traverse MENTIONS Edges (Entity Relationships)
│   │   │   ├─ Chunk A mentions: ["VxRail", "Backup", "Data Protection"]
│   │   │   ├─ Query entities: MATCH (c)-[:MENTIONS]->(e:Entity)
│   │   │   ├─ Found entities: VxRail, Backup, Data Protection
│   │   │   │
│   │   │   ├─> Follow Entity RELATED_TO Edges
│   │   │   │   ├─ VxRail -[RELATED_TO:0.9]-> RecoverPoint
│   │   │   │   ├─ Backup -[RELATED_TO:0.85]-> Data Protection Suite
│   │   │   │   └─ Data Protection -[RELATED_TO:0.8]-> Replication
│   │   │   │
│   │   │   └─> Find Chunks Mentioning Related Entities
│   │   │       ├─ Query: MATCH (e2)<-[:MENTIONS]-(c2:Chunk)
│   │   │       ├─ Found: Chunk F mentions RecoverPoint
│   │   │       ├─ Found: Chunk G mentions Data Protection Suite
│   │   │       └─ Found: Chunk H mentions Replication
│   │   │
│   │   └─> Collect Expanded Chunks
│   │       └─ From Chunk A: [D, E, F, G, H]
│   │
│   ├─> Repeat for Chunk B, C, ... (all 10 initial candidates)
│   │   └─ Expanded: 35 new chunks discovered
│   │
│   └─> Deduplicate
│       ├─ Remove chunks already in initial set
│       ├─ Remove duplicate chunk IDs
│       └─ Result: 30 unique expanded chunks
│
├─> 3. Graph Expansion (Depth 2, Optional)
│   ├─ If max_expansion_depth >= 2:
│   ├─> For each depth-1 expanded chunk:
│   │   ├─ Traverse SIMILAR_TO (threshold: 0.7)
│   │   ├─ Traverse MENTIONS → Entity → RELATED_TO → MENTIONS
│   │   └─ Collect 2-hop expanded chunks
│   ├─ Found: 15 additional chunks
│   └─ Deduplicate: 12 unique new chunks
│
├─> 4. Score Calculation
│   │
│   ├─> For Each Expanded Chunk:
│   │   │
│   │   ├─> Graph Score Components:
│   │   │   │
│   │   │   ├─ Similarity Score (if SIMILAR_TO edge exists)
│   │   │   │   └─ Use edge.similarity value (0.7-1.0)
│   │   │   │
│   │   │   ├─ Entity Path Score
│   │   │   │   ├─ Path: Query entities → RELATED_TO → Chunk entities
│   │   │   │   ├─ Strength: Product of edge strengths along path
│   │   │   │   └─ Example: 0.9 * 0.85 = 0.765
│   │   │   │
│   │   │   └─ Hop Distance Penalty
│   │   │       ├─ Depth 1: multiplier = 1.0
│   │   │       ├─ Depth 2: multiplier = 0.8
│   │   │       └─ Depth 3: multiplier = 0.6
│   │   │
│   │   └─> Combined Graph Score:
│   │       └─ max(similarity_score, entity_path_score) * distance_penalty
│   │
│   └─> Merge with Vector Scores:
│       ├─ Initial chunks: vector_score available
│       ├─ Expanded chunks: vector_score = 0.0 (not from vector search)
│       └─ Final: hybrid_score = vector_score * 0.7 + graph_score * 0.3
│
├─> 5. Limit and Filter
│   ├─ Sort by hybrid_score descending
│   ├─ Limit to max_expanded_chunks (50)
│   ├─ Filter: score >= min_score_threshold (0.3)
│   └─ Result: 42 chunks (10 initial + 32 expanded)
│
└─> 6. Return to Hybrid Retrieval
    └─ Candidates ready for reranking
```

## Step-by-Step Trace

### Step 1: Initial Vector Search

**Location**: `rag/retriever.py`

```python
async def vector_search(
    query: str,
    top_k: int = 30,
    embedding_model: str = "text-embedding-3-small",
) -> List[Chunk]:
    """
    Vector similarity search in Neo4j.
    
    Args:
        query: Search query
        top_k: Number of results
        embedding_model: Embedding model name
    
    Returns:
        List of chunks with vector_score
    """
    # Generate query embedding
    embedding_manager = get_embedding_manager()
    query_embedding = await embedding_manager.get_embedding(
        query,
        model=embedding_model,
    )
    
    # Neo4j vector search
    cypher = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding)
        YIELD node, score
        MATCH (d:Document)-[:CONTAINS]->(node)
        RETURN
            node.id AS chunk_id,
            node.content AS content,
            node.page_number AS page_number,
            d.id AS document_id,
            d.filename AS document_name,
            score AS vector_score
        ORDER BY score DESC
    """
    
    results = graph_db.execute_read_query(
        cypher,
        top_k=top_k,
        embedding=query_embedding,
    )
    
    chunks = [
        Chunk(
            chunk_id=r["chunk_id"],
            content=r["content"],
            page_number=r["page_number"],
            document_id=r["document_id"],
            document_name=r["document_name"],
            vector_score=r["vector_score"],
        )
        for r in results
    ]
    
    return chunks
```

**Initial Results**:
```python
[
    Chunk(id="chunk-047", content="VxRail backup procedures...", vector_score=0.92),
    Chunk(id="chunk-048", content="To configure backup...", vector_score=0.88),
    Chunk(id="chunk-052", content="Data protection for VxRail...", vector_score=0.85),
    # ... 7 more chunks
]
```

### Step 2: Graph Expansion

**Location**: `rag/retriever.py`

```python
async def expand_via_graph(
    initial_chunks: List[Chunk],
    max_depth: int = 1,
    similarity_threshold: float = 0.7,
    max_expanded: int = 50,
) -> List[Chunk]:
    """
    Expand chunks via graph edges.
    
    Args:
        initial_chunks: Seed chunks from vector search
        max_depth: Maximum traversal depth
        similarity_threshold: Minimum SIMILAR_TO edge similarity
        max_expanded: Maximum expanded chunks to return
    
    Returns:
        Initial chunks + expanded chunks
    """
    expanded_chunks = []
    visited_chunk_ids = {c.chunk_id for c in initial_chunks}
    
    current_depth_chunks = initial_chunks
    
    for depth in range(1, max_depth + 1):
        # Get chunk IDs for current depth
        chunk_ids = [c.chunk_id for c in current_depth_chunks]
        
        # Expand via SIMILAR_TO edges
        similar_chunks = await _expand_via_similarity(
            chunk_ids,
            similarity_threshold,
            visited_chunk_ids,
        )
        
        # Expand via entity relationships
        entity_chunks = await _expand_via_entities(
            chunk_ids,
            visited_chunk_ids,
        )
        
        # Merge and deduplicate
        depth_expanded = similar_chunks + entity_chunks
        depth_expanded = _deduplicate_chunks(depth_expanded)
        
        # Track visited
        for chunk in depth_expanded:
            visited_chunk_ids.add(chunk.chunk_id)
            chunk.expansion_depth = depth
        
        expanded_chunks.extend(depth_expanded)
        
        # Prepare for next depth
        current_depth_chunks = depth_expanded
        
        # Stop if we've expanded enough
        if len(expanded_chunks) >= max_expanded:
            break
    
    # Combine initial + expanded
    all_chunks = initial_chunks + expanded_chunks[:max_expanded]
    
    return all_chunks
```

### Step 2a: Similarity Expansion

**Location**: `rag/retriever.py`

```python
async def _expand_via_similarity(
    chunk_ids: List[str],
    threshold: float,
    visited: Set[str],
) -> List[Chunk]:
    """
    Expand via SIMILAR_TO edges.
    
    Args:
        chunk_ids: Source chunk IDs
        threshold: Minimum similarity score
        visited: Already visited chunk IDs
    
    Returns:
        Similar chunks
    """
    cypher = """
        UNWIND $chunk_ids AS chunk_id
        MATCH (c:Chunk {id: chunk_id})
        MATCH (c)-[r:SIMILAR_TO]-(c2:Chunk)
        WHERE r.similarity >= $threshold
          AND NOT c2.id IN $visited
        MATCH (d:Document)-[:CONTAINS]->(c2)
        RETURN DISTINCT
            c2.id AS chunk_id,
            c2.content AS content,
            c2.page_number AS page_number,
            d.id AS document_id,
            d.filename AS document_name,
            r.similarity AS similarity_score
    """
    
    results = graph_db.execute_read_query(
        cypher,
        chunk_ids=chunk_ids,
        threshold=threshold,
        visited=list(visited),
    )
    
    chunks = [
        Chunk(
            chunk_id=r["chunk_id"],
            content=r["content"],
            page_number=r["page_number"],
            document_id=r["document_id"],
            document_name=r["document_name"],
            similarity_score=r["similarity_score"],
        )
        for r in results
    ]
    
    return chunks
```

### Step 2b: Entity Expansion

**Location**: `rag/retriever.py`

```python
async def _expand_via_entities(
    chunk_ids: List[str],
    visited: Set[str],
) -> List[Chunk]:
    """
    Expand via entity relationships.
    
    Flow:
    1. Get entities mentioned in source chunks
    2. Find related entities via RELATED_TO edges
    3. Find chunks mentioning related entities
    
    Args:
        chunk_ids: Source chunk IDs
        visited: Already visited chunk IDs
    
    Returns:
        Entity-related chunks
    """
    cypher = """
        UNWIND $chunk_ids AS chunk_id
        
        // Get source chunk entities
        MATCH (c:Chunk {id: chunk_id})-[:MENTIONS]->(e:Entity)
        
        // Find related entities
        MATCH (e)-[r:RELATED_TO]-(e2:Entity)
        WHERE r.strength >= 0.5
        
        // Find chunks mentioning related entities
        MATCH (e2)<-[:MENTIONS]-(c2:Chunk)
        WHERE NOT c2.id IN $visited
        
        MATCH (d:Document)-[:CONTAINS]->(c2)
        
        RETURN DISTINCT
            c2.id AS chunk_id,
            c2.content AS content,
            c2.page_number AS page_number,
            d.id AS document_id,
            d.filename AS document_name,
            e2.name AS entity_name,
            r.strength AS relationship_strength
    """
    
    results = graph_db.execute_read_query(
        cypher,
        chunk_ids=chunk_ids,
        visited=list(visited),
    )
    
    # Group by chunk (may have multiple entity paths)
    chunk_dict = {}
    for r in results:
        chunk_id = r["chunk_id"]
        
        if chunk_id not in chunk_dict:
            chunk_dict[chunk_id] = Chunk(
                chunk_id=chunk_id,
                content=r["content"],
                page_number=r["page_number"],
                document_id=r["document_id"],
                document_name=r["document_name"],
                entity_paths=[],
            )
        
        # Track entity path
        chunk_dict[chunk_id].entity_paths.append({
            "entity": r["entity_name"],
            "strength": r["relationship_strength"],
        })
    
    return list(chunk_dict.values())
```

### Step 4: Graph Score Calculation

**Location**: `rag/retriever.py`

```python
def calculate_graph_scores(
    chunks: List[Chunk],
    max_depth: int,
) -> List[Chunk]:
    """
    Calculate graph-based relevance scores.
    
    Args:
        chunks: Chunks with expansion metadata
        max_depth: Maximum expansion depth
    
    Returns:
        Chunks with graph_score
    """
    for chunk in chunks:
        # Distance penalty based on depth
        depth = getattr(chunk, 'expansion_depth', 0)
        distance_penalty = 1.0 / (1.0 + 0.2 * depth)
        
        # Similarity score (if SIMILAR_TO edge)
        similarity_score = getattr(chunk, 'similarity_score', 0.0)
        
        # Entity path score (if via entity relationships)
        entity_path_score = 0.0
        if hasattr(chunk, 'entity_paths') and chunk.entity_paths:
            # Use strongest path
            entity_path_score = max(
                path['strength'] for path in chunk.entity_paths
            )
        
        # Combined graph score
        graph_score = max(similarity_score, entity_path_score) * distance_penalty
        
        chunk.graph_score = graph_score
    
    return chunks
```

**Score Examples**:
```python
# Initial chunk (depth 0)
Chunk(
    id="chunk-047",
    vector_score=0.92,
    graph_score=0.0,  # Not expanded
    expansion_depth=0
)

# Expanded via SIMILAR_TO (depth 1)
Chunk(
    id="chunk-123",
    vector_score=0.0,  # Not from vector search
    similarity_score=0.82,
    graph_score=0.82 * 1.0 = 0.82,
    expansion_depth=1
)

# Expanded via entity (depth 1)
Chunk(
    id="chunk-156",
    vector_score=0.0,
    entity_paths=[{"entity": "RecoverPoint", "strength": 0.9}],
    graph_score=0.9 * 1.0 = 0.9,
    expansion_depth=1
)

# Expanded at depth 2
Chunk(
    id="chunk-201",
    similarity_score=0.75,
    graph_score=0.75 * 0.8 = 0.6,  # 0.8 penalty for depth 2
    expansion_depth=2
)
```

### Step 5: Hybrid Scoring

**Location**: `rag/retriever.py`

```python
def calculate_hybrid_scores(
    chunks: List[Chunk],
    chunk_weight: float = 0.7,
) -> List[Chunk]:
    """
    Combine vector and graph scores.
    
    Args:
        chunks: Chunks with vector_score and graph_score
        chunk_weight: Weight for vector component
    
    Returns:
        Chunks with hybrid_score
    """
    for chunk in chunks:
        vector_score = getattr(chunk, 'vector_score', 0.0)
        graph_score = getattr(chunk, 'graph_score', 0.0)
        
        # Weighted combination
        hybrid_score = (
            vector_score * chunk_weight +
            graph_score * (1.0 - chunk_weight)
        )
        
        chunk.hybrid_score = hybrid_score
    
    # Sort by hybrid score
    chunks.sort(key=lambda c: c.hybrid_score, reverse=True)
    
    return chunks
```

**Final Scores**:
```python
[
    Chunk(id="chunk-047", vector=0.92, graph=0.0, hybrid=0.644),   # 0.92*0.7 + 0*0.3
    Chunk(id="chunk-156", vector=0.0, graph=0.9, hybrid=0.27),     # 0*0.7 + 0.9*0.3
    Chunk(id="chunk-048", vector=0.88, graph=0.0, hybrid=0.616),
    Chunk(id="chunk-123", vector=0.0, graph=0.82, hybrid=0.246),
    # ... sorted by hybrid_score
]
```

## Performance Notes

### Query Complexity

**Single-Hop Expansion** (depth=1):
```cypher
// Similarity: O(N * K) where N=initial chunks, K=avg similar chunks
MATCH (c)-[r:SIMILAR_TO]-(c2) WHERE r.similarity >= 0.7

// Entity: O(N * E * R * C) where E=entities/chunk, R=related entities, C=chunks/entity
MATCH (c)-[:MENTIONS]->(e)-[:RELATED_TO]-(e2)<-[:MENTIONS]-(c2)
```

**Optimization**: Single query with UNWIND for batch processing

### Caching

- **Entity Label Cache**: Reduces entity lookups (70-80% hit rate)
- **Graph Structure Cache**: No built-in cache (consider for frequently accessed subgraphs)

### Limits

- `max_expansion_depth`: 1-2 (higher = exponential growth)
- `max_expanded_chunks`: 50 (limits result set size)
- `similarity_threshold`: 0.7 (filters weak connections)

## Related Documentation

- [Hybrid Retrieval](04-features/hybrid-retrieval.md)
- [Entity Reasoning](04-features/entity-reasoning.md)
- [Graph Database](03-components/backend/graph-database.md)
- [Chat Query Flow](05-data-flows/chat-query-flow.md)
