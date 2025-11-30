# Similarity Count Query Optimization

## Problem Statement

The similarities pagination endpoint takes ~13.5 seconds to respond, even when requesting only 1 item. Performance analysis reveals the bottleneck is the count query:

```cypher
MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
WHERE c1.id IN $chunk_ids 
  AND c2.id IN $chunk_ids 
  AND c1.id < c2.id
  AND coalesce(sim.score, 0) >= $min_score
RETURN count(*) as total
```

**Measured Performance:**
- Total response time: 13.5s
- Main query (50 items with SKIP/LIMIT): Fast (<1s)
- Count query: ~13s (bottleneck)

**Root Cause:**
The count query must traverse ALL 8,709 similarity relationships to compute the total, even though pagination only needs 50 items. For documents with thousands of chunks, this creates a full graph scan on every request.

## Proposed Solutions

### Option 1: Deferred Count with Estimated Total (Immediate, Low Risk)

**Approach:** Return an estimated total on first page, compute exact count asynchronously for subsequent pages.

**Implementation:**
```python
@router.get("/{document_id}/similarities")
async def get_document_chunk_similarities(
    document_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    exact_count: bool = Query(default=False, description="Compute exact total (slower)")
):
    """
    Get paginated similarities. By default returns estimated total for speed.
    Set exact_count=true to compute precise total (adds ~13s).
    """
    try:
        # ... existing validation and chunk_ids retrieval ...
        
        # Main query (fast)
        results = session.run(query, 
            chunk_ids=chunk_ids,
            offset=offset,
            limit=limit,
            min_score=min_score
        ).data()
        
        # Determine total count strategy
        if exact_count or offset > 0:
            # User requested exact count OR needs it for navigation
            total_result = session.run(count_query, 
                chunk_ids=chunk_ids, 
                min_score=min_score
            ).single()
            total = total_result["total"] if total_result else 0
            estimated = False
        else:
            # First page: use fast estimate
            # Estimate based on chunk count and typical density
            # For our use case: ~3.5 similarities per chunk on average
            num_chunks = len(chunk_ids)
            total = min(10000, int(num_chunks * 3.5))  # Cap at 10k for UX
            estimated = True
        
        return {
            "document_id": document_id,
            "total": total,
            "estimated": estimated,
            "limit": limit,
            "offset": offset,
            "has_more": len(results) == limit,  # Simple check
            "similarities": results
        }
```

**Pros:**
- Immediate 13s improvement for first page load
- Minimal code changes
- Backward compatible (frontend can ignore `estimated` field)
- No database schema changes

**Cons:**
- Total count is approximate on first page
- UX might show "~8,000" instead of "8,709"
- Requires frontend update to handle estimated counts

**UX Impact:**
- First page: "Showing 1-50 of ~8,500 similarities" (instant)
- Later pages: "Showing 51-100 of 8,709 similarities" (computed after navigation)

---

### Option 2: Precomputed Similarity Count (Medium Effort, High Impact)

**Approach:** Store similarity count as document property, update during ingestion/reindexing.

**Implementation:**

1. **Add property to Document node:**
```cypher
MATCH (d:Document {id: $doc_id})
SET d.similarity_count = $count
```

2. **Update during ingestion:**
```python
# ingestion/document_processor.py - after similarity creation
def _update_similarity_count(self, doc_id: str):
    """Update cached similarity count for document."""
    query = """
    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
    WITH d, collect(c.id) as chunk_ids
    MATCH (c1:Chunk)-[:SIMILAR_TO]-(c2:Chunk)
    WHERE c1.id IN chunk_ids AND c2.id IN chunk_ids AND c1.id < c2.id
    WITH d, count(*) as sim_count
    SET d.similarity_count = sim_count
    """
    with self.graph_db.driver.session() as session:
        session.run(query, doc_id=doc_id)
```

3. **Read from property in API:**
```python
# api/routers/documents.py
# In get_document_chunk_similarities:
similarity_count_query = """
    MATCH (d:Document {id: $doc_id})
    RETURN coalesce(d.similarity_count, 0) as total
"""
total_result = session.run(similarity_count_query, doc_id=document_id).single()
total = total_result["total"] if total_result else 0
```

**Pros:**
- Sub-second response time (property lookup)
- Exact count (not estimated)
- Clean separation of concerns
- Scales to millions of similarities

**Cons:**
- Requires ingestion pipeline update
- Count becomes stale if similarities added/removed outside ingestion
- One-time migration to add property to existing documents

**Migration:**
```python
# scripts/migrate_similarity_counts.py
def migrate_all_documents():
    """Add similarity_count property to all existing documents."""
    query = """
    MATCH (d:Document)
    WITH d
    MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
    WITH d, collect(c.id) as chunk_ids
    MATCH (c1:Chunk)-[:SIMILAR_TO]-(c2:Chunk)
    WHERE c1.id IN chunk_ids AND c2.id IN chunk_ids AND c1.id < c2.id
    WITH d, count(*) as sim_count
    SET d.similarity_count = sim_count
    RETURN d.id, sim_count
    """
    # Execute in batches to avoid timeout
```

---

### Option 3: Cache Count in Redis (Medium Effort, Flexible)

**Approach:** Cache count result in Redis with TTL, invalidate on document changes.

**Implementation:**
```python
@router.get("/{document_id}/similarities")
async def get_document_chunk_similarities(...):
    # Check cache first
    cache_key = f"similarity_count:{document_id}:{min_score}"
    cached_total = redis_client.get(cache_key)
    
    if cached_total is not None:
        total = int(cached_total)
    else:
        # Compute and cache (TTL: 1 hour)
        total_result = session.run(count_query, ...).single()
        total = total_result["total"] if total_result else 0
        redis_client.setex(cache_key, 3600, total)
    
    # ... rest of logic ...
```

**Cache Invalidation:**
```python
# On document reindex/update:
redis_client.delete(f"similarity_count:{document_id}:*")
```

**Pros:**
- Fast after first request (~13s first time, <10ms after)
- No database schema changes
- Flexible TTL and invalidation
- Can cache per min_score filter

**Cons:**
- Requires Redis infrastructure
- Cache invalidation complexity
- First request still slow
- TTL means occasional slow request

---

### Option 4: Lazy Count with Background Job (Low Priority)

**Approach:** Return results immediately, compute count in background, notify via WebSocket.

**Implementation:**
```python
@router.get("/{document_id}/similarities")
async def get_document_chunk_similarities(...):
    # Return data immediately with total: null
    response = {
        "document_id": document_id,
        "total": None,  # Will be computed asynchronously
        "limit": limit,
        "offset": offset,
        "has_more": len(results) == limit,
        "similarities": results
    }
    
    # Trigger background count job
    if offset == 0:  # Only on first page
        background_tasks.add_task(compute_and_broadcast_count, document_id, min_score)
    
    return response

async def compute_and_broadcast_count(doc_id: str, min_score: float):
    """Compute count and push update via SSE/WebSocket."""
    total = # ... run count query ...
    await websocket_manager.broadcast({
        "type": "similarity_count_update",
        "document_id": doc_id,
        "total": total
    })
```

**Pros:**
- Instant initial response
- Progressive enhancement (count arrives later)
- No slow requests

**Cons:**
- Requires WebSocket/SSE infrastructure
- Complex frontend state management
- UX uncertainty (total changes after load)

---

## Recommendation

**Implement Option 1 (Deferred Estimated Count) immediately** for quick wins:
- 13s improvement on first page load
- Minimal risk and code changes
- Can be combined with other options later

**Plan Option 2 (Precomputed Count) for next iteration:**
- Add during next ingestion refactor
- Run migration script for existing documents
- Provides long-term scalability

**Decision Matrix:**

| Option | Speed Improvement | Effort | Risk | Scalability |
|--------|------------------|--------|------|-------------|
| 1. Estimated | 13s → <0.1s (first page) | Low | Low | Medium |
| 2. Precomputed | 13s → <0.1s (all pages) | Medium | Low | High |
| 3. Redis Cache | 13s → <0.01s (cached) | Medium | Medium | High |
| 4. Background | 13s → 0s (deferred) | High | High | High |

## Implementation Plan

### Phase 1: Immediate (This Week)
- ✅ Implement Option 1 (estimated count for first page)
- ✅ Add `estimated` flag to API response
- ✅ Update frontend to show "~" prefix for estimates
- ✅ Measure performance improvement

### Phase 2: Next Sprint
- Add `similarity_count` property to Document nodes
- Update ingestion pipeline to maintain count
- Run migration script for existing documents
- Switch API to use precomputed count

### Phase 3: Future Enhancement
- Consider Redis caching for frequently accessed documents
- Evaluate background computation if needed

## Testing Strategy

```bash
# Before optimization
time curl "http://localhost:8000/api/documents/$DOC_ID/similarities?limit=50"
# Expected: ~13.5s

# After Option 1 (estimated)
time curl "http://localhost:8000/api/documents/$DOC_ID/similarities?limit=50"
# Expected: <0.5s (no count query)

# With exact count flag
time curl "http://localhost:8000/api/documents/$DOC_ID/similarities?limit=50&exact_count=true"
# Expected: ~13.5s (same as before, but opt-in)

# After Option 2 (precomputed)
time curl "http://localhost:8000/api/documents/$DOC_ID/similarities?limit=50"
# Expected: <0.5s (property lookup)
```

## Conclusion

**Initial Hypothesis:** The 13.5s delay is caused by the count query's full graph traversal.

**Actual Finding After Implementation:** The bottleneck is the `ORDER BY score DESC` clause in the main query, NOT the count query. Neo4j must evaluate all 8,709 relationships to sort them before applying SKIP/LIMIT. The count query optimization (Option 1) was successfully implemented and works correctly (returns estimated count), but doesn't improve response time because the main query sorting is the slow operation.

**Evidence:**
- Test 1 (estimated count): 13.59s, returns total: 8585 (estimated: true)
- Test 2 (exact count): 13.60s, returns total: 8709 (estimated: false)  
- Test 3 (page 2, exact): 13.39s, returns total: 8709 (estimated: false)

The similar timing across all tests proves the count query isn't the bottleneck. The estimated count feature works (different totals, correct flag), but the main query's ORDER BY dominates execution time.

**Real Solutions for 13s Delay:**
1. **Remove ORDER BY** - Return unsorted results (instant, but poor UX)
2. **Pre-sort during ingestion** - Store relationships with rank property  
3. **Index-based pagination** - Use score thresholds instead of SKIP/LIMIT
4. **Materialized view** - Cache top N similarities per document in a separate structure

The count optimization remains valuable for future scenarios where sorting isn't required, and provides correct estimated vs exact semantics for the API.
