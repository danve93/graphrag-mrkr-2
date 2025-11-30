# Similarity Query Optimization Plan

## Problem Statement

The similarities endpoint takes ~13.5 seconds due to `ORDER BY score DESC` on 8,709 relationships. Neo4j must evaluate and sort all relationships before applying SKIP/LIMIT pagination.

**Current Query:**
```cypher
MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
WHERE c1.id IN $chunk_ids 
  AND c2.id IN $chunk_ids 
  AND c1.id < c2.id
WITH c1.id AS chunk1_id,
     c2.id AS chunk2_id,
     coalesce(sim.score, 0) AS score
ORDER BY score DESC  # ← 13s bottleneck
SKIP $offset
LIMIT $limit
RETURN chunk1_id, chunk2_id, score
```

## Solution Options

### Option 1: Remove Sorting (Quick Win, UX Trade-off)

**Approach:** Return similarities in arbitrary order for instant response.

**Implementation:**
```python
# api/routers/documents.py
@router.get("/{document_id}/similarities")
async def get_document_chunk_similarities(
    ...
    sorted: bool = Query(default=False, description="Sort by score (slower)")
):
    """
    Get chunk similarities. By default returns unsorted for speed.
    Set sorted=true for score-ordered results (adds ~13s).
    """
    query = f"""
        MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
        WHERE c1.id IN $chunk_ids 
          AND c2.id IN $chunk_ids 
          AND c1.id < c2.id
        WITH c1.id AS chunk1_id,
             c2.id AS chunk2_id,
             coalesce(sim.score, 0) AS score
        {"ORDER BY score DESC" if sorted else ""}
        SKIP $offset
        LIMIT $limit
        RETURN chunk1_id, chunk2_id, score
    """
```

**Pros:**
- Instant (<0.5s) response
- Minimal code changes
- Opt-in sorting for power users

**Cons:**
- Random order makes it hard to find high-quality similarities
- Users can't easily identify most relevant chunk pairs
- Poor default UX

**Verdict:** ❌ Not recommended as default. Sorting by relevance is core to the feature.

---

### Option 2: Pre-rank During Ingestion ⭐ RECOMMENDED

**Approach:** Store a `rank` property on each SIMILAR_TO relationship during creation, enabling indexed retrieval.

**Architecture:**

1. **During Ingestion** - Rank relationships by score per document:
```python
# ingestion/document_processor.py or scripts/create_similarities.py

def create_ranked_similarities(doc_id: str):
    """Create similarities with rank property for efficient pagination."""
    
    # Step 1: Create similarities as usual
    create_similarities(doc_id)
    
    # Step 2: Add rank property based on score ordering
    query = """
    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
    WITH d, collect(c.id) as chunk_ids
    MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
    WHERE c1.id IN chunk_ids 
      AND c2.id IN chunk_ids 
      AND c1.id < c2.id
    WITH sim, coalesce(sim.score, 0) as score
    ORDER BY score DESC
    WITH collect(sim) as sims
    UNWIND range(0, size(sims)-1) as idx
    WITH sims[idx] as sim, idx
    SET sim.rank = idx
    RETURN count(*) as ranked_count
    """
    
    with graph_db.driver.session() as session:
        result = session.run(query, doc_id=doc_id).single()
        logger.info(f"Ranked {result['ranked_count']} similarities for {doc_id}")
```

2. **API Query** - Use rank for efficient pagination:
```python
# api/routers/documents.py

query = """
    MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
    WHERE c1.id IN $chunk_ids 
      AND c2.id IN $chunk_ids 
      AND c1.id < c2.id
      AND sim.rank IS NOT NULL
    WITH c1.id AS chunk1_id,
         c2.id AS chunk2_id,
         coalesce(sim.score, 0) AS score,
         sim.rank as rank
    ORDER BY rank ASC  # Fast: rank is indexed
    SKIP $offset
    LIMIT $limit
    RETURN chunk1_id, chunk2_id, score
"""
```

**Index Creation:**
```cypher
// Run once during deployment
CREATE INDEX similarity_rank IF NOT EXISTS 
FOR ()-[r:SIMILAR_TO]-() 
ON (r.rank)
```

**Migration Script:**
```python
# scripts/add_similarity_ranks.py

def migrate_all_documents():
    """Add rank property to all existing similarity relationships."""
    
    # Get all documents with chunks
    query_docs = """
    MATCH (d:Document)-[:HAS_CHUNK]->(:Chunk)
    RETURN DISTINCT d.id as doc_id
    """
    
    with graph_db.driver.session() as session:
        docs = session.run(query_docs).data()
        
    logger.info(f"Migrating {len(docs)} documents...")
    
    for idx, doc in enumerate(docs):
        doc_id = doc['doc_id']
        try:
            create_ranked_similarities(doc_id)
            logger.info(f"[{idx+1}/{len(docs)}] Ranked similarities for {doc_id}")
        except Exception as e:
            logger.error(f"Failed to rank {doc_id}: {e}")
            continue
    
    logger.info("Migration complete")

if __name__ == "__main__":
    migrate_all_documents()
```

**Pros:**
- ✅ Sub-second query time (<0.5s)
- ✅ Maintains score-based ordering
- ✅ Scales to millions of relationships
- ✅ One-time migration cost
- ✅ No UX compromise

**Cons:**
- Requires ingestion pipeline update
- One-time migration for existing documents (~10 min for large dataset)
- Rank becomes stale if scores are recomputed (rare)

**Implementation Timeline:**
- Day 1: Add rank logic to similarity creation
- Day 2: Create migration script and index
- Day 3: Update API query to use rank
- Day 4: Run migration on production data
- Day 5: Validate and measure performance

**Verdict:** ⭐ **RECOMMENDED** - Best balance of performance, UX, and maintainability.

---

### Option 3: Score-Threshold Pagination (Keyset Pagination)

**Approach:** Use score thresholds instead of SKIP/LIMIT for cursor-based pagination.

**Implementation:**

```python
# api/routers/documents.py

@router.get("/{document_id}/similarities")
async def get_document_chunk_similarities(
    document_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    max_score: Optional[float] = Query(default=None, description="Score threshold for next page"),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0)
):
    """
    Get chunk similarities using cursor-based pagination.
    
    First request: /similarities?limit=50
    Next page: /similarities?limit=50&max_score=0.85
    (where 0.85 is the lowest score from previous page)
    """
    
    # Build query with score cursor
    score_filter = f"AND coalesce(sim.score, 0) < $max_score" if max_score else ""
    
    query = f"""
        MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
        WHERE c1.id IN $chunk_ids 
          AND c2.id IN $chunk_ids 
          AND c1.id < c2.id
          AND coalesce(sim.score, 0) >= $min_score
          {score_filter}
        WITH c1.id AS chunk1_id,
             c2.id AS chunk2_id,
             coalesce(sim.score, 0) AS score
        ORDER BY score DESC
        LIMIT $limit
        RETURN chunk1_id, chunk2_id, score
    """
    
    results = session.run(query, 
        chunk_ids=chunk_ids,
        limit=limit,
        min_score=min_score,
        max_score=max_score
    ).data()
    
    # Return next cursor
    next_max_score = results[-1]['score'] if results else None
    
    return {
        "document_id": document_id,
        "similarities": results,
        "next_cursor": next_max_score,
        "has_more": len(results) == limit
    }
```

**Frontend Integration:**
```typescript
// frontend/src/components/Document/ChunkSimilaritiesSection.tsx

const [cursor, setCursor] = useState<number | null>(null)

const loadNextPage = async () => {
  const response = await api.getDocumentSimilaritiesPaginated(documentId, {
    limit: 50,
    maxScore: cursor  // Pass previous page's lowest score
  })
  
  setSimilarities(prev => [...prev, ...response.similarities])
  setCursor(response.next_cursor)
}
```

**Performance Analysis:**

With max_score filter, Neo4j can stop traversal early:
```cypher
// First page: ORDER BY + LIMIT 50 → Still evaluates all (slow)
// Second page: score < 0.85 + ORDER BY + LIMIT 50 → Still slow
```

**Problem:** ORDER BY still requires full scan even with score filter. This approach doesn't solve the sorting bottleneck.

**Pros:**
- Stateless pagination (no offset tracking)
- Prevents "page drift" issues

**Cons:**
- ❌ Doesn't improve query performance (ORDER BY still slow)
- Can't jump to arbitrary pages
- Frontend must maintain cursor state
- More complex UX (no page numbers)

**Verdict:** ❌ Not recommended - doesn't solve the core performance issue.

---

### Option 4: Materialized Top-N Cache

**Approach:** Pre-compute and cache top N similarities per document in a fast-access structure.

**Architecture:**

1. **Cache Structure** - Add property to Document node:
```cypher
MATCH (d:Document {id: $doc_id})
SET d.top_similarities = [
  {chunk1_id: "abc", chunk2_id: "def", score: 0.98},
  {chunk1_id: "ghi", chunk2_id: "jkl", score: 0.97},
  // ... top 500 similarities
]
```

2. **Populate Cache** - During ingestion or on-demand:
```python
# core/graph_db.py

def cache_top_similarities(doc_id: str, top_n: int = 500):
    """Cache top N similarities for fast retrieval."""
    
    query = """
    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
    WITH d, collect(c.id) as chunk_ids
    MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
    WHERE c1.id IN chunk_ids 
      AND c2.id IN chunk_ids 
      AND c1.id < c2.id
    WITH d, c1.id as chunk1_id, c2.id as chunk2_id, 
         coalesce(sim.score, 0) as score
    ORDER BY score DESC
    LIMIT $top_n
    WITH d, collect({
      chunk1_id: chunk1_id,
      chunk2_id: chunk2_id,
      score: score
    }) as top_sims
    SET d.top_similarities = [sim IN top_sims | sim]
    RETURN size(d.top_similarities) as cached_count
    """
    
    with driver.session() as session:
        result = session.run(query, doc_id=doc_id, top_n=top_n).single()
        return result['cached_count']
```

3. **API Retrieval** - Fast property access:
```python
# api/routers/documents.py

@router.get("/{document_id}/similarities")
async def get_document_chunk_similarities(
    document_id: str,
    limit: int = Query(default=50),
    offset: int = Query(default=0),
    use_cache: bool = Query(default=True)
):
    """
    Get similarities. Uses cached top-500 by default for speed.
    Set use_cache=false to query all similarities (slow).
    """
    
    if use_cache:
        # Fast path: retrieve from cached property
        query = """
        MATCH (d:Document {id: $doc_id})
        WHERE d.top_similarities IS NOT NULL
        RETURN d.top_similarities as all_sims
        """
        result = session.run(query, doc_id=document_id).single()
        
        if result and result['all_sims']:
            all_sims = result['all_sims']
            paginated = all_sims[offset:offset+limit]
            
            return {
                "document_id": document_id,
                "total": len(all_sims),
                "cached": True,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < len(all_sims),
                "similarities": paginated
            }
    
    # Fallback: slow query for all similarities
    # ... existing query logic ...
```

**Cache Invalidation:**
```python
# On similarity recomputation:
def recompute_similarities(doc_id: str):
    create_similarities(doc_id)
    cache_top_similarities(doc_id)  # Refresh cache
```

**Pros:**
- ✅ Instant retrieval (<50ms) for top similarities
- ✅ No schema changes to relationships
- ✅ Works with existing queries as fallback
- ✅ Predictable memory usage (500 items per document)

**Cons:**
- Limited to top N (e.g., 500 similarities)
- Can't access lower-scored similarities without slow query
- Cache staleness if similarities change
- Property storage limitations (Neo4j property size limits)

**Verdict:** ✅ Good **complement** to Option 2. Use for "hot path" (first few pages), fall back to ranked query for deep pagination.

---

## Recommended Implementation Strategy

### Phase 1: Pre-ranking (Weeks 1-2) ⭐ PRIMARY SOLUTION

**Goal:** Eliminate ORDER BY bottleneck via rank indexing.

**Tasks:**
1. ✅ Add `rank` property to SIMILAR_TO relationships during ingestion
2. ✅ Create Neo4j index on `sim.rank`
3. ✅ Update API query to use `ORDER BY rank ASC`
4. ✅ Create migration script for existing documents
5. ✅ Run migration on production data
6. ✅ Validate <0.5s query time

**Expected Outcome:** 13.5s → 0.4s (30x improvement)

### Phase 2: Top-N Caching (Week 3) 🎯 OPTIMIZATION

**Goal:** Instant retrieval for most common use case (browsing top similarities).

**Tasks:**
1. ✅ Add `top_similarities` property to Document nodes
2. ✅ Populate cache during ingestion (top 500)
3. ✅ Update API to check cache first
4. ✅ Add cache invalidation on recomputation
5. ✅ Monitor cache hit rate

**Expected Outcome:** 0.4s → 0.05s for cached results (8x additional improvement)

### Phase 3: Frontend Progressive Enhancement (Week 4)

**Tasks:**
1. ✅ Update UI to show "cached" indicator when using fast path
2. ✅ Add "Load more" button for similarities beyond cache
3. ✅ Implement virtual scrolling for large lists
4. ✅ Add performance metrics display

---

## Implementation Details

### 1. Add Rank to Similarity Creation

```python
# ingestion/document_processor.py

async def _create_chunk_similarities_with_rank(self, doc_id: str):
    """Create similarity relationships with rank property."""
    
    # Step 1: Create relationships with scores
    await self._create_chunk_similarities(doc_id)  # Existing logic
    
    # Step 2: Add rank based on score ordering
    rank_query = """
    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
    WITH d, collect(c.id) as chunk_ids
    MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
    WHERE c1.id IN chunk_ids 
      AND c2.id IN chunk_ids 
      AND c1.id < c2.id
    WITH sim, coalesce(sim.score, 0) as score
    ORDER BY score DESC
    WITH collect(sim) as sims
    UNWIND range(0, size(sims)-1) as idx
    WITH sims[idx] as sim, idx
    SET sim.rank = idx
    RETURN count(*) as ranked_count
    """
    
    try:
        with self.graph_db.driver.session() as session:
            result = session.run(rank_query, doc_id=doc_id).single()
            count = result['ranked_count'] if result else 0
            logger.info(f"Added rank to {count} similarities for document {doc_id}")
    except Exception as e:
        logger.error(f"Failed to add rank to similarities for {doc_id}: {e}")
        # Non-fatal: similarities still work without rank, just slower
```

### 2. Create Neo4j Index

```python
# scripts/setup_neo4j.py

def create_similarity_rank_index():
    """Create index on SIMILAR_TO.rank for fast pagination."""
    
    index_query = """
    CREATE INDEX similarity_rank IF NOT EXISTS 
    FOR ()-[r:SIMILAR_TO]-() 
    ON (r.rank)
    """
    
    with driver.session() as session:
        session.run(index_query)
        logger.info("Created similarity rank index")

# Add to initialization
if __name__ == "__main__":
    setup_neo4j()
    create_similarity_rank_index()
```

### 3. Update API Query

```python
# api/routers/documents.py

@router.get("/{document_id}/similarities")
async def get_document_chunk_similarities(
    document_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    use_rank: bool = Query(default=True, description="Use pre-computed rank for speed")
):
    """
    Get chunk similarities with efficient pagination.
    
    Performance: <0.5s with rank index (vs 13.5s with ORDER BY score)
    """
    
    if use_rank:
        # Fast path: use pre-computed rank
        query = """
            MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
            WHERE c1.id IN $chunk_ids 
              AND c2.id IN $chunk_ids 
              AND c1.id < c2.id
              AND sim.rank IS NOT NULL
              AND coalesce(sim.score, 0) >= $min_score
            WITH c1.id AS chunk1_id,
                 c2.id AS chunk2_id,
                 coalesce(sim.score, 0) AS score,
                 sim.rank as rank
            ORDER BY rank ASC
            SKIP $offset
            LIMIT $limit
            RETURN chunk1_id, chunk2_id, score
        """
    else:
        # Slow path: compute ordering on demand (backwards compatibility)
        query = """
            MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
            WHERE c1.id IN $chunk_ids 
              AND c2.id IN $chunk_ids 
              AND c1.id < c2.id
              AND coalesce(sim.score, 0) >= $min_score
            WITH c1.id AS chunk1_id,
                 c2.id AS chunk2_id,
                 coalesce(sim.score, 0) AS score
            ORDER BY score DESC
            SKIP $offset
            LIMIT $limit
            RETURN chunk1_id, chunk2_id, score
        """
    
    # ... rest of logic ...
```

### 4. Migration Script

```python
# scripts/add_similarity_ranks.py

import logging
from core.graph_db import get_graph_db
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_rank_to_document(doc_id: str, graph_db):
    """Add rank property to all similarities for a document."""
    
    rank_query = """
    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
    WITH d, collect(c.id) as chunk_ids
    MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
    WHERE c1.id IN chunk_ids 
      AND c2.id IN chunk_ids 
      AND c1.id < c2.id
    WITH sim, coalesce(sim.score, 0) as score
    ORDER BY score DESC
    WITH collect(sim) as sims
    UNWIND range(0, size(sims)-1) as idx
    WITH sims[idx] as sim, idx
    SET sim.rank = idx
    RETURN count(*) as ranked_count
    """
    
    with graph_db.driver.session() as session:
        result = session.run(rank_query, doc_id=doc_id).single()
        return result['ranked_count'] if result else 0

def migrate_all_documents():
    """Add rank to all existing similarity relationships."""
    
    graph_db = get_graph_db()
    
    # Get all documents with chunks
    query_docs = """
    MATCH (d:Document)-[:HAS_CHUNK]->(:Chunk)
    RETURN DISTINCT d.id as doc_id, d.filename as filename
    """
    
    with graph_db.driver.session() as session:
        docs = session.run(query_docs).data()
    
    logger.info(f"Found {len(docs)} documents to migrate")
    
    total_ranked = 0
    failed = []
    
    for doc in tqdm(docs, desc="Ranking similarities"):
        doc_id = doc['doc_id']
        filename = doc['filename']
        
        try:
            count = add_rank_to_document(doc_id, graph_db)
            total_ranked += count
            logger.info(f"✓ {filename}: {count} similarities ranked")
        except Exception as e:
            logger.error(f"✗ {filename}: {e}")
            failed.append((doc_id, str(e)))
    
    logger.info(f"Migration complete: {total_ranked} similarities ranked")
    
    if failed:
        logger.warning(f"{len(failed)} documents failed:")
        for doc_id, error in failed:
            logger.warning(f"  - {doc_id}: {error}")

if __name__ == "__main__":
    migrate_all_documents()
```

---

## Testing & Validation

### Performance Tests

```bash
# Before optimization (baseline)
time curl "http://localhost:8000/api/documents/$DOC_ID/similarities?limit=50"
# Expected: 13.5s

# After ranking (Phase 1)
time curl "http://localhost:8000/api/documents/$DOC_ID/similarities?limit=50&use_rank=true"
# Target: <0.5s (30x improvement)

# After caching (Phase 2)
time curl "http://localhost:8000/api/documents/$DOC_ID/similarities?limit=50&use_cache=true"
# Target: <0.05s (270x improvement)
```

### Validation Queries

```cypher
// Check rank coverage
MATCH ()-[sim:SIMILAR_TO]-()
WHERE sim.rank IS NOT NULL
RETURN count(*) as ranked, 
       count(DISTINCT sim) as total,
       100.0 * count(*) / count(DISTINCT sim) as coverage_pct

// Verify rank ordering matches score ordering
MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
WHERE sim.rank IS NOT NULL AND c1.id < c2.id
WITH sim.rank as rank, sim.score as score
ORDER BY rank ASC
LIMIT 100
RETURN rank, score
// Should show descending scores

// Check index usage
PROFILE
MATCH ()-[sim:SIMILAR_TO]-()
WHERE sim.rank >= 0 AND sim.rank < 50
RETURN sim.rank, sim.score
ORDER BY sim.rank ASC
// Should show index scan, not full scan
```

---

## Success Criteria

### Phase 1 Success (Pre-ranking)
- ✅ 95%+ of similarities have rank property
- ✅ Query time <0.5s for 50-item page
- ✅ Results ordered correctly (highest scores first)
- ✅ No regression in result quality

### Phase 2 Success (Caching)
- ✅ 80%+ of requests served from cache
- ✅ Cache hit query time <0.1s
- ✅ Cache miss falls back to ranked query (<0.5s)
- ✅ No stale data issues

### Overall Success
- ✅ **30x improvement** in query time (13.5s → 0.4s)
- ✅ Zero UX compromise (still sorted by relevance)
- ✅ Scales to 100K+ similarities per document
- ✅ Graceful fallback for edge cases

---

## Rollout Plan

### Week 1: Development & Testing
- Mon-Tue: Implement ranking logic
- Wed-Thu: Create migration script, test on dev data
- Fri: Code review & unit tests

### Week 2: Staging & Migration
- Mon: Deploy to staging environment
- Tue: Run migration on staging data (10K documents)
- Wed-Thu: Performance testing & validation
- Fri: Rollback testing

### Week 3: Production Rollout
- Mon: Deploy code with feature flag (use_rank=false default)
- Tue: Run migration script in maintenance window
- Wed: Enable use_rank=true for 10% of traffic (canary)
- Thu: Ramp to 100% if metrics look good
- Fri: Monitor & optimize

### Week 4: Caching & Polish
- Implement top-N caching
- Frontend enhancements
- Documentation & metrics dashboard

---

## Risk Mitigation

### Risk 1: Migration Timeout
**Mitigation:** Batch migration (100 docs at a time), resume capability

### Risk 2: Rank Staleness
**Mitigation:** Re-rank on similarity recomputation, periodic validation

### Risk 3: Index Performance
**Mitigation:** Monitor index size, add LIMIT to subqueries

### Risk 4: Backward Compatibility
**Mitigation:** Keep use_rank parameter, fall back to slow query if rank missing

---

## Next Steps

1. **Immediate:** Create feature branch `feat/similarity-ranking`
2. **This Week:** Implement ranking logic + migration script
3. **Next Week:** Test on dev environment with 944b7e4c document
4. **Following Week:** Production rollout with monitoring

**Estimated Total Effort:** 3-4 weeks (1 engineer)  
**Expected ROI:** 30x performance improvement, no UX trade-offs
