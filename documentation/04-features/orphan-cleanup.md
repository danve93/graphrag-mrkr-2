# Orphan Cleanup

## Overview

The Orphan Cleanup feature automatically detects and removes disconnected chunks and entities that are no longer linked to any documents in the knowledge graph. This prevents database bloat and ensures retrieval accuracy by eliminating orphaned data that could contaminate search results.

> [!NOTE]
> This feature was introduced in **v2.1.1** to address an issue where orphaned chunks were causing RAG retrieval to return 0 results (Issue: 1889 orphaned chunks representing 85% of total chunks).

## Problem Statement

Orphaned nodes occur when:
- Documents are deleted but their chunks/entities remain in the graph
- Processing failures leave incomplete data structures
- Relationship deletion logic fails or is interrupted
- Manual graph modifications break expected connections

**Impact of orphans:**
- **Retrieval Contamination**: Vector search returns chunks not connected to any document
- **Database Bloat**: Wasted storage and memory for unused nodes
- **Query Performance**: Slower graph traversals due to disconnected components
- **Inaccurate Statistics**: Database stats include nodes that shouldn't exist

## Features

### Automatic Cleanup on Startup

The backend automatically scans for and removes orphaned chunks and entities when it starts up.

**Default Behavior:**
- **Enabled by default** (`ENABLE_ORPHAN_CLEANUP_ON_STARTUP=true`)
- **Grace period**: 5 minutes (configurable via `ORPHAN_CLEANUP_GRACE_PERIOD_MINUTES`)
- **Safe for concurrent operations**: Only deletes orphans older than the grace period

**Configuration:**
```bash
# .env file
ENABLE_ORPHAN_CLEANUP_ON_STARTUP=true          # Enable/disable automatic cleanup
ORPHAN_CLEANUP_GRACE_PERIOD_MINUTES=5          # Grace period in minutes (default: 5)
```

**Rationale for Grace Period:**
The grace period prevents race conditions during document ingestion. If a document is being uploaded concurrently with the cleanup, newly created chunks might temporarily appear orphaned before the `HAS_CHUNK` relationship is created. The grace period ensures only truly abandoned orphans are deleted.

### Manual Cleanup

Administrators can trigger orphan cleanup on demand via:
1. **UI Button**: Database panel → "Cleanup" button in toolbar
2. **API Endpoint**: `POST /api/database/cleanup-orphans`

**Manual cleanup has NO grace period** - it immediately deletes all detected orphans.

## Detection Algorithm

### Orphaned Chunks

A chunk is considered orphaned if:
```cypher
MATCH (c:Chunk)
WHERE NOT EXISTS {
    MATCH (d:Document)-[:HAS_CHUNK]->(c)
}
AND datetime(c.created_at) < datetime() - duration({minutes: $grace_period})
RETURN c
```

**Criteria:**
- No incoming `HAS_CHUNK` relationship from any `Document` node
- Created more than `grace_period` minutes ago

### Orphaned Entities

An entity is considered orphaned if:
```cypher
MATCH (e:Entity)
WHERE NOT EXISTS {
    MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
}
AND datetime(e.created_at) < datetime() - duration({minutes: $grace_period})
RETURN e
```

**Criteria:**
- No incoming `CONTAINS_ENTITY` relationship from any `Chunk` node
- Created more than `grace_period` minutes ago

## Implementation

### Backend Service

Location: `api/services/database_service.py`

```python
async def cleanup_orphaned_nodes(
    grace_period_minutes: int = 0
) -> Dict[str, int]:
    """
    Clean up orphaned chunks and entities.
    
    Args:
        grace_period_minutes: Only delete orphans older than this (0 = no grace period)
    
    Returns:
        Dict with counts: {"orphaned_chunks": N, "orphaned_entities": M}
    """
    stats = {"orphaned_chunks": 0, "orphaned_entities": 0}
    
    # Delete orphaned chunks
    chunks_query = """
    MATCH (c:Chunk)
    WHERE NOT EXISTS {
        MATCH (d:Document)-[:HAS_CHUNK]->(c)
    }
    AND datetime(c.created_at) < datetime() - duration({minutes: $grace_period})
    WITH c
    DETACH DELETE c
    RETURN count(c) as deleted
    """
    
    result = await db.run(chunks_query, grace_period=grace_period_minutes)
    stats["orphaned_chunks"] = result.single()["deleted"]
    
    # Delete orphaned entities
    entities_query = """
    MATCH (e:Entity)
    WHERE NOT EXISTS {
        MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
    }
    AND datetime(e.created_at) < datetime() - duration({minutes: $grace_period})
    WITH e
    DETACH DELETE e
    RETURN count(e) as deleted
    """
    
    result = await db.run(entities_query, grace_period=grace_period_minutes)
    stats["orphaned_entities"] = result.single()["deleted"]
    
    return stats
```

### Startup Integration

Location: `api/main.py`

```python
@app.on_event("startup")
async def startup_event():
    # ... other startup tasks ...
    
    if settings.ENABLE_ORPHAN_CLEANUP_ON_STARTUP:
        logger.info(
            f"Running orphan cleanup with {settings.ORPHAN_CLEANUP_GRACE_PERIOD_MINUTES}min grace period"
        )
        stats = await cleanup_orphaned_nodes(
            grace_period_minutes=settings.ORPHAN_CLEANUP_GRACE_PERIOD_MINUTES
        )
        logger.info(
            f"Orphan cleanup complete: {stats['orphaned_chunks']} chunks, "
            f"{stats['orphaned_entities']} entities removed"
        )
```

### API Endpoint

Location: `api/routers/database.py`

```python
@router.post("/cleanup-orphans")
async def cleanup_orphans(
    current_user: dict = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Manually trigger orphan cleanup (no grace period).
    
    Admin only. Immediately deletes all orphaned chunks and entities.
    """
    stats = await cleanup_orphaned_nodes(grace_period_minutes=0)
    
    return {
        "success": True,
        "stats": stats,
        "message": f"Removed {stats['orphaned_chunks']} orphaned chunks "
                   f"and {stats['orphaned_entities']} orphaned entities"
    }
```

## Frontend Integration

### Cleanup Button

Location: `frontend/src/components/Database/DatabaseTab.tsx`

**UI Elements:**
- **Button**: "Cleanup" in toolbar, next to refresh/reset buttons
- **Icon**: Trash or broom icon
- **Confirmation Dialog**: Prevents accidental cleanup
- **Loading State**: Shows spinner during cleanup operation
- **Result Toast**: Displays cleanup statistics

**Implementation:**
```typescript
const handleCleanup = async () => {
  if (!confirm('Remove all orphaned chunks and entities? This cannot be undone.')) {
    return;
  }
  
  setIsCleaningUp(true);
  try {
    const response = await api.post('/api/database/cleanup-orphans');
    const { stats } = response.data;
    
    toast.success(
      `Cleanup complete: ${stats.orphaned_chunks} chunks, ` +
      `${stats.orphaned_entities} entities removed`
    );
    
    // Refresh database stats
    await loadDatabaseStats();
  } catch (error) {
    toast.error('Cleanup failed: ' + error.message);
  } finally {
    setIsCleaningUp(false);
  }
};
```

### Database Stats Integration

The database stats endpoint now includes orphan counts:

```typescript
interface DatabaseStats {
  // ... existing fields ...
  orphan_counts?: {
    orphaned_chunks: number;
    orphaned_entities: number;
  };
}
```

**Display:**
- Show orphan counts in database panel
- Visual warning if orphan count exceeds threshold (e.g., > 100)
- Recommend cleanup if significant orphans detected

## Configuration Reference

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_ORPHAN_CLEANUP_ON_STARTUP` | boolean | `true` | Enable automatic cleanup on backend startup |
| `ORPHAN_CLEANUP_GRACE_PERIOD_MINUTES` | integer | `5` | Only delete orphans older than this (minutes) |

### Runtime Configuration

Manual cleanup can be triggered at any time via:
- **UI**: Database panel → Cleanup button
- **API**: `POST /api/database/cleanup-orphans` (admin required)
- **Script**: `python scripts/cleanup_orphans.py` (if available)

## Best Practices

### When to Use Manual Cleanup

**Recommended scenarios:**
- After bulk document deletions
- After failed ingestion jobs
- Before database backups
- When retrieval quality degrades unexpectedly
- When database stats show high orphan counts

**Not recommended:**
- During active document ingestion (use startup cleanup with grace period instead)
- In production without testing in staging first
- Without database backup if unsure about impact

### Monitoring Orphan Growth

**Track orphan counts over time:**
```cypher
// Check current orphan counts
MATCH (c:Chunk)
WHERE NOT EXISTS { MATCH (d:Document)-[:HAS_CHUNK]->(c) }
RETURN count(c) as orphaned_chunks;

MATCH (e:Entity)
WHERE NOT EXISTS { MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e) }
RETURN count(e) as orphaned_entities;
```

**Set up alerts:**
- Alert if orphaned_chunks > 100
- Alert if orphaned_chunks > 10% of total chunks
- Alert if orphan growth rate is abnormal

### Database Maintenance Schedule

**Suggested schedule:**
1. **Startup cleanup**: Automatic (enabled by default)
2. **Weekly manual cleanup**: Every Sunday during low-traffic hours
3. **Post-bulk-operation cleanup**: After deleting >10 documents
4. **Pre-backup cleanup**: Before scheduled database backups

## Troubleshooting

### High Orphan Counts After Startup

**Symptom**: Startup cleanup reports thousands of orphaned nodes

**Possible causes:**
1. Previous document deletion logic was broken
2. Manual Cypher queries deleted documents without cleanup
3. Processing failures left incomplete data structures

**Solution:**
- Run manual cleanup to remove orphans
- Investigate root cause to prevent recurrence
- Review document deletion logic

### Cleanup Deletes Too Many Nodes

**Symptom**: Cleanup removes nodes that should exist

**Possible causes:**
1. Grace period is too short (concurrent ingestion)
2. Processing job hasn't created relationships yet
3. Custom graph modifications broke expected schema

**Solution:**
- Increase grace period: `ORPHAN_CLEANUP_GRACE_PERIOD_MINUTES=10`
- Disable automatic cleanup during bulk ingestion
- Verify graph schema matches expected structure

### Manual Cleanup Fails with Timeout

**Symptom**: API request times out during cleanup

**Possible causes:**
1. Extremely high orphan count (>10,000 nodes)
2. Database under heavy load
3. Insufficient Neo4j memory

**Solution:**
- Run cleanup during low-traffic hours
- Increase API timeout: `CLEANUP_TIMEOUT_SECONDS=300`
- Use Cypher query directly with batching:
```cypher
CALL apoc.periodic.iterate(
  'MATCH (c:Chunk) WHERE NOT EXISTS { MATCH (d:Document)-[:HAS_CHUNK]->(c) } RETURN c',
  'DETACH DELETE c',
  {batchSize: 100}
);
```

## Related Issues (v2.1.1)

### Issue: Orphaned Chunks

**Root cause**: Document deletion logic in prior versions didn't properly cascade delete chunks and entities.

**Impact**: 1889 orphaned chunks (85% of total) caused RAG retrieval to return 0 results.

**Fix**: 
1. Implemented proper cascade deletion in document delete endpoint
2. Added orphan cleanup feature for existing orphans
3. Added startup cleanup to prevent accumulation

### Issue: Document Update Progress

**Related issue**: Document updates didn't trigger orphan cleanup for replaced chunks.

**Fix**: Update logic now explicitly deletes replaced chunks before creating new ones, preventing orphan accumulation during incremental updates.

## API Reference

### POST /api/database/cleanup-orphans

**Authentication**: Admin required

**Request:**
```http
POST /api/database/cleanup-orphans
Authorization: Bearer <admin_token>
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "orphaned_chunks": 1889,
    "orphaned_entities": 342
  },
  "message": "Removed 1889 orphaned chunks and 342 orphaned entities"
}
```

**Error Responses:**
- `401 Unauthorized`: Missing or invalid admin token
- `500 Internal Server Error`: Database error during cleanup

## Related Documentation

- [Database Management](../08-operations/database-management.md)
- [Document Deletion](./document-upload.md#deletion)
- [Graph Database](../03-components/backend/graph-database.md)
- [Job Management](../03-components/backend/job-management.md)
