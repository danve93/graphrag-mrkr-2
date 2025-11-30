# Entity Relationship Bug Fix

## Issue Summary

Entities were being created during document ingestion but the `CONTAINS_ENTITY` relationships between Chunk and Entity nodes were not being created, causing the document detail page to show 0 entities despite the sidebar showing the correct count.

## Root Cause

In `ingestion/document_processor.py`, the `_create_entities_async()` function (async code path, used when `SYNC_ENTITY_EMBEDDINGS=False`) was catching ALL exceptions during relationship creation and only logging them at DEBUG level (line 681-683):

```python
except Exception as e:
    logger.debug(
        f"Failed to create chunk-entity rel {chunk_id}->{entity_id}: {e}"
    )
```

This caused relationship creation failures to be silently suppressed, making it impossible to diagnose the issue from logs.

## Database Impact

- Entity nodes were created successfully with `source_chunks` metadata
- Chunk nodes existed and were linked to documents via `HAS_CHUNK` relationships
- **BUT**: Zero `CONTAINS_ENTITY` relationships existed between Chunk and Entity nodes
- This caused `get_document_details()` query to return empty entities array because the query requires the relationship path: `(d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)`
- The sidebar showed correct entity count because it uses a global count query: `MATCH (e:Entity) RETURN count(e)`

## Fix Applied

Changed the exception logging level from `logger.debug()` to `logger.error()` in `ingestion/document_processor.py` line 681:

```python
except Exception as e:
    logger.error(
        f"Failed to create chunk-entity rel {chunk_id}->{entity_id}: {e}"
    )
```

This ensures relationship creation failures are visible in logs at ERROR level, making debugging much easier.

## Data Repair

Created `scripts/repair_entity_relationships.py` to retroactively create missing relationships using the `source_chunks` metadata already present on Entity nodes.

### Repair Process

1. Query all Entity nodes with `source_chunks` metadata
2. For each entity and each chunk ID in `source_chunks`:
   - Call `graph_db.create_chunk_entity_relationship(chunk_id, entity_id)`
   - Create the `CONTAINS_ENTITY` relationship via MERGE

### Repair Results

- **Entities processed**: 67
- **Relationships created**: 72 (some entities referenced multiple chunks)
- **Relationships failed**: 0

### Verification

After repair:
```
MATCH (d:Document {id: "782f142dc2dd7d32f84755361d4d16bb"})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
RETURN count(DISTINCT e) as entity_count
```
Result: **67 entities** (previously 0)

API response: `/api/documents/782f142dc2dd7d32f84755361d4d16bb` now returns 67 entities in the entities array.

## Future Prevention

1. **Code fix deployed**: Exception logging upgraded to ERROR level
2. **Repair script available**: `scripts/repair_entity_relationships.py` can be run anytime to fix missing relationships
3. **Monitoring**: ERROR-level logs will now surface relationship creation failures immediately

## Testing Recommendations

1. Upload a new document and verify entities appear in document detail page
2. Check backend logs for any ERROR messages during entity extraction
3. Query Neo4j to confirm `CONTAINS_ENTITY` relationships are being created:
   ```
   MATCH ()-[r:CONTAINS_ENTITY]->() RETURN count(r)
   ```

## Related Files

- `ingestion/document_processor.py` - Entity extraction and persistence
- `core/graph_db.py` - Database operations including `create_chunk_entity_relationship()`
- `scripts/repair_entity_relationships.py` - Repair script for missing relationships
- `api/routers/documents.py` - Document API endpoints using `get_document_details()`
