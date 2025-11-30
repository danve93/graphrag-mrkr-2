# Graph Visualization Memory Fix

## Problem
The graph visualization endpoint (`/api/graph/clustered`) was returning 503 Service Unavailable errors when trying to load large documents (6,679+ entities) due to Neo4j memory exhaustion:

```
Neo.TransientError.General.MemoryPoolOutOfMemoryError: 
The allocation of an extra 2.1 MiB would use more than the limit 2.7 GiB
```

## Root Cause
- Neo4j transaction memory limit: 2.7 GB
- Large documents with thousands of entities would exceed this limit
- The graph query loads all entities plus their neighbors, causing exponential memory growth
- No safeguards existed to prevent oversized queries

## Solution Implemented

### 1. Entity Count Pre-Check (`core/graph_db.py`)
Added `count_document_entities()` method with optional community filtering:

```python
def count_document_entities(self, document_id: str, community_id: Optional[int] = None) -> int:
    """Count entities before loading graph to prevent memory issues."""
    # Returns count of entities matching the filters
    # Used to make intelligent decisions about graph loading
```

### 2. Smart Limits (`api/routers/graph.py`)
Implemented tiered limits based on entity count:

- **>1000 entities**: Block the request with helpful error message
- **>500 entities**: Force limit to 50 nodes maximum
- **>100 entities**: Cap limit at 100 nodes
- **Community-filtered**: Allow visualization if filtered community is under limits

### 3. Error Messages
Provide actionable guidance to users:
```
"This document has 6679 entities. Please filter by a specific community_id 
to visualize a subset. Full graph visualization is not available for 
collections with >1000 entities."
```

## Test Results

### Before Fix
- ❌ Full document graph: 503 error, Neo4j memory exhaustion
- ❌ No feedback to user about why it failed
- ❌ Frontend shows generic error message

### After Fix
- ✅ Full document graph (6,679 entities): Returns 400 with helpful error
- ✅ Community-filtered graph (295 nodes): Successfully loads and renders
- ✅ Global graph (1,068 nodes): Successfully loads and renders
- ✅ No Neo4j memory errors in logs

### Example Successful Requests
```bash
# Community-filtered graph (works)
GET /api/graph/clustered?document_id=944b...&community_id=0&limit=50
Response: 295 nodes, 370 edges

# Global graph without document filter (works)
GET /api/graph/clustered?limit=50
Response: 1,068 nodes, 1,780 edges

# Unfiltered large document (blocked with helpful error)
GET /api/graph/clustered?document_id=944b...
Response: 400 - "Please filter by specific community_id..."
```

## Benefits
1. **Prevents crashes**: No more 503 errors or Neo4j memory exhaustion
2. **Maintains functionality**: Graph visualization still works for reasonable sizes
3. **User guidance**: Clear error messages explain how to proceed
4. **Graceful degradation**: System stays responsive even with large documents
5. **Community-driven**: Encourages users to explore via community filtering

## Deployment
Files modified:
- `api/routers/graph.py` - Added entity count checks and smart limits
- `core/graph_db.py` - Added `count_document_entities()` with community filter support

Deploy via:
```bash
scp api/routers/graph.py core/graph_db.py root@server:~/amber/
docker cp amber/graph.py graphrag-backend:/app/api/routers/graph.py
docker cp amber/graph_db.py graphrag-backend:/app/core/graph_db.py
docker restart graphrag-backend
```

## Future Enhancements
1. **UI improvements**: Show entity/community counts before loading graph
2. **Community selector**: Dropdown to choose specific community to visualize
3. **Pagination**: Load graph in chunks for very large communities
4. **Sampling**: Random sample of nodes for overview visualization
5. **Neo4j config**: Consider increasing transaction memory limit if needed

## Related Issues
- Fixed alongside entity relationship bug (CONTAINS_ENTITY)
- Part of community detection feature rollout
- Tested with CarbonioAdminGuide.pdf (6,679 entities, 1,598 communities)
