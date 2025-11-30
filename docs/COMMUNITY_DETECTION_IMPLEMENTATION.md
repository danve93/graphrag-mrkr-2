# Community Detection Implementation

**Date:** November 28, 2025  
**Issue:** Entities were displaying in the sidebar but communities were not showing in the document detail page.

## Problem Summary

After successfully fixing the entity relationship bug (where `CONTAINS_ENTITY` relationships were missing), a new issue emerged: communities were not appearing in the document detail page despite entities being visible. Investigation revealed that:

1. No community detection had been run on the entity graph
2. All Entity nodes had `community_id=NULL` and `level=NULL`
3. The Leiden clustering algorithm had never been executed

## Root Cause

The system supports community detection through the Leiden clustering algorithm (implemented in `core/graph_clustering.py`), but:

- Clustering had never been executed on the existing entity graph
- The backend API was not returning `community_id` and `level` fields for entities in the document detail endpoint

## Solution Implemented

### 1. Run Leiden Clustering Algorithm

Executed the clustering script to detect communities in the entity graph:

```bash
docker exec graphrag-backend python scripts/run_clustering.py
```

**Results:**
- **67 entities** processed
- **27 communities** created using Leiden algorithm
- **Modularity score:** 0.4854 (indicating good clustering quality)
- **45 edges** analyzed from entity relationships

**Community Distribution:**
- Largest community (ID 2): 25 entities
- Medium-sized communities (IDs 4, 0): 4-5 entities each
- Small communities: 1-3 entities each

### 2. Updated Backend API to Return Community Data

Modified the `get_document_details()` function in `core/graph_db.py` to include community information:

**File:** `core/graph_db.py` (lines 2684-2702)

**Changes:**
```python
# Added community_id and level to the Cypher query
entity_records = session.run(
    """
    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
    RETURN e.type as type,
           e.name as text,
           e.community_id as community_id,  # ADDED
           e.level as level,                 # ADDED
           count(*) as count,
           collect(DISTINCT c.chunk_index) as positions
    ORDER BY type ASC, text ASC
    """,
    doc_id=doc_id,
)

# Added community_id and level to the entity dictionary
entities = [
    {
        "type": record["type"],
        "text": record["text"],
        "community_id": record["community_id"],  # ADDED
        "level": record.get("level"),            # ADDED
        "count": record["count"],
        "positions": [pos for pos in (record["positions"] or []) if pos is not None],
    }
    for record in entity_records
]
```

### 3. Updated API Models

Modified the `DocumentEntity` Pydantic model in `api/models.py` to include community fields:

**File:** `api/models.py` (lines 216-223)

**Changes:**
```python
class DocumentEntity(BaseModel):
    """Entity extracted from a document."""

    type: str
    text: str
    community_id: int | None = None  # ADDED
    level: int | None = None         # ADDED
    count: int | None = None
    positions: List[int] | None = None
```

## Deployment Process

Since the Docker containers don't mount source code as volumes (only data directory), the following deployment steps were required:

1. **Copy updated files to remote host:**
   ```bash
   scp core/graph_db.py root@cph-01.demo.zextras.io:~/amber/core/
   scp api/models.py root@cph-01.demo.zextras.io:~/amber/api/
   ```

2. **Copy files into running container:**
   ```bash
   docker cp amber/core/graph_db.py graphrag-backend:/app/core/graph_db.py
   docker cp amber/api/models.py graphrag-backend:/app/api/models.py
   ```

3. **Restart backend container:**
   ```bash
   docker restart graphrag-backend
   ```

## Verification

### Neo4j Database
Verified communities exist in the database:
```cypher
MATCH (e:Entity) 
WHERE e.community_id IS NOT NULL 
RETURN count(DISTINCT e.community_id) AS community_count, 
       count(e) AS entities_with_communities

// Result: 27 communities, 67 entities
```

### API Response
Tested document API endpoint:
```bash
curl http://localhost:8000/api/documents/782f142dc2dd7d32f84755361d4d16bb
```

**Sample Response:**
```json
{
  "entities": [
    {
      "type": "ACCOUNT",
      "text": "KYRGYZ TEAM",
      "community_id": 2,
      "level": 0,
      "count": 1,
      "positions": [4]
    },
    {
      "type": "PRODUCT",
      "text": "CARBONIO",
      "community_id": 2,
      "level": 0,
      "count": 5,
      "positions": [5, 4, 3, 2, 1]
    }
  ]
}
```

## Current Status

✅ **Complete** - All 67 entities now have community assignments  
✅ **Complete** - API returns `community_id` and `level` fields  
✅ **Complete** - 27 communities successfully detected and persisted  
✅ **Complete** - Document detail page can now display community information

## Frontend Integration

The frontend can now:

1. **Display community information** for each entity in the document detail view
2. **Filter entities by community** to group related concepts
3. **Visualize community clusters** in the graph view (if graph visualization supports community coloring)
4. **Show community statistics** (e.g., "27 communities with 67 entities")

## Configuration

Community detection is controlled by these environment variables in `.env`:

```bash
ENABLE_CLUSTERING=true
ENABLE_GRAPH_CLUSTERING=true
```

### Clustering Parameters

In `config/settings.py`:
- `clustering_resolution` (default: 1.0) - Higher values create more communities
- `clustering_min_edge_weight` (default: 0.01) - Minimum relationship strength to consider
- `clustering_relationship_types` (default: ['SIMILAR_TO', 'RELATED_TO']) - Relationship types used for clustering
- `clustering_level` (default: 0) - Hierarchical clustering level

## Related Scripts

- **Manual clustering:** `python scripts/run_clustering.py`
- **Build projection:** `python scripts/build_leiden_projection.py`
- **Reindex with clustering:** Set `ENABLE_CLUSTERING=true` and use reindex API

## Notes

- Community detection uses the **Leiden algorithm** (python-igraph library)
- Clustering is performed on the entity relationship graph (not document-chunk graph)
- Communities represent semantically related entities discovered through graph topology
- The modularity score of 0.4854 indicates well-defined community structure
- Community IDs are 0-indexed integers assigned by the clustering algorithm
- The `level` field supports hierarchical clustering (currently all at level 0)

## Future Enhancements

Potential improvements for community detection:

1. **Community summarization:** Generate LLM-powered descriptions of each community's theme (see `core/community_summarizer.py`)
2. **Dynamic re-clustering:** Automatically update communities when new entities are added
3. **Hierarchical clustering:** Enable multi-level community detection for large graphs
4. **Community filtering UI:** Add UI controls to filter/highlight entities by community
5. **Community visualization:** Add distinct colors for each community in graph views
