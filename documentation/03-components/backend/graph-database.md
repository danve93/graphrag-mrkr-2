# Graph Database Component

Neo4j operations and connection management for the knowledge graph.

## Overview

The graph database component provides a unified interface for all Neo4j operations, including connection management, query execution, and data persistence. It abstracts the Neo4j driver and provides domain-specific methods for document, chunk, entity, and relationship operations.

**Location**: `core/graph_db.py`
**Database**: Neo4j 5.x
**Protocol**: Bolt (binary)

## Architecture

```
┌──────────────────────────────────────────────────┐
│              GraphDB Class                        │
├──────────────────────────────────────────────────┤
│                                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ Connection │  │   Query    │  │   Batch    │ │
│  │   Pool     │  │ Execution  │  │ Operations │ │
│  └────────────┘  └────────────┘  └────────────┘ │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │         Domain-Specific Methods            │  │
│  ├────────────────────────────────────────────┤  │
│  │  • Documents  • Chunks     • Entities      │  │
│  │  • Relationships  • Communities  • Stats   │  │
│  └────────────────────────────────────────────┘  │
│                                                   │
└───────────────────┬──────────────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │  Neo4j 5.x    │
            │   Database    │
            └───────────────┘
```

## Connection Management

### Database Initialization

```python
from neo4j import GraphDatabase, AsyncGraphDatabase
from config.settings import settings

class GraphDB:
    def __init__(self):
        self.driver = None
        self._async_driver = None
    
    def connect(self):
        """Initialize synchronous driver."""
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_pool_size=settings.neo4j_max_connection_pool_size
        )
        
        # Verify connectivity
        self.driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {settings.neo4j_uri}")
    
    async def connect_async(self):
        """Initialize asynchronous driver."""
        self._async_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_pool_size=settings.neo4j_max_connection_pool_size
        )
        
        await self._async_driver.verify_connectivity()
    
    def close(self):
        """Close database connections."""
        if self.driver:
            self.driver.close()
        if self._async_driver:
            self._async_driver.close()
```

### Singleton Pattern

```python
from core.singletons import Singleton

class GraphDB(Singleton):
    """Singleton GraphDB instance."""
    pass

# Usage
def get_db() -> GraphDB:
    """Get singleton database instance."""
    db = GraphDB.get_instance()
    if not db.driver:
        db.connect()
    return db
```

### Connection Configuration

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_MAX_CONNECTION_POOL_SIZE=50
```

## Query Execution

### Synchronous Execution

```python
def execute_read(self, query: str, params: Dict = None) -> List[Dict]:
    """Execute read query and return results."""
    params = params or {}
    
    with self.driver.session() as session:
        result = session.run(query, params)
        return [dict(record) for record in result]

def execute_write(self, query: str, params: Dict = None) -> Dict:
    """Execute write query and return summary."""
    params = params or {}
    
    with self.driver.session() as session:
        result = session.run(query, params)
        summary = result.consume()
        return {
            "nodes_created": summary.counters.nodes_created,
            "relationships_created": summary.counters.relationships_created,
            "properties_set": summary.counters.properties_set
        }
```

### Asynchronous Execution

```python
async def execute_read_async(self, query: str, params: Dict = None) -> List[Dict]:
    """Execute read query asynchronously."""
    params = params or {}
    
    async with self._async_driver.session() as session:
        result = await session.run(query, params)
        records = await result.data()
        return records

async def execute_write_async(self, query: str, params: Dict = None) -> Dict:
    """Execute write query asynchronously."""
    params = params or {}
    
    async with self._async_driver.session() as session:
        result = await session.run(query, params)
        summary = await result.consume()
        return {
            "nodes_created": summary.counters.nodes_created,
            "relationships_created": summary.counters.relationships_created,
            "properties_set": summary.counters.properties_set
        }
```

### Transaction Management

```python
def execute_transaction(self, func, *args, **kwargs):
    """Execute function within a transaction."""
    with self.driver.session() as session:
        return session.execute_write(func, *args, **kwargs)

# Usage
def create_document_with_chunks(tx, document_data, chunks_data):
    # Create document
    tx.run(
        "CREATE (d:Document {id: $id, filename: $filename})",
        document_data
    )
    
    # Create chunks
    for chunk in chunks_data:
        tx.run(
            "MATCH (d:Document {id: $doc_id}) "
            "CREATE (c:Chunk {id: $chunk_id, text: $text}) "
            "CREATE (d)-[:HAS_CHUNK]->(c)",
            {"doc_id": document_data["id"], **chunk}
        )

db.execute_transaction(create_document_with_chunks, doc_data, chunks)
```

## Document Operations

### Create Document

```python
def create_document(self, document_data: Dict) -> str:
    """Create document node."""
    query = """
    CREATE (d:Document {
        id: $id,
        filename: $filename,
        file_path: $file_path,
        file_type: $file_type,
        file_size: $file_size,
        title: $title,
        created_at: datetime(),
        page_count: $page_count,
        word_count: $word_count,
        precomputed_chunk_count: 0,
        precomputed_entity_count: 0,
        precomputed_community_count: 0,
        precomputed_similarity_count: 0
    })
    RETURN d.id as document_id
    """
    
    result = self.execute_write(query, document_data)
    return document_data["id"]
```

### Get Document

```python
def get_document(self, document_id: str) -> Optional[Dict]:
    """Get document by ID."""
    query = """
    MATCH (d:Document {id: $document_id})
    RETURN d {.*} as document
    """
    
    results = self.execute_read(query, {"document_id": document_id})
    return results[0]["document"] if results else None
```

### Get Document Details

```python
def get_document_details(self, document_id: str) -> Optional[Dict]:
    """Get document with precomputed statistics."""
    query = """
    MATCH (d:Document {id: $document_id})
    RETURN d.id as id,
           d.filename as filename,
           d.file_path as file_path,
           d.file_type as file_type,
           d.file_size as file_size,
           d.title as title,
           d.created_at as created_at,
           d.page_count as page_count,
           d.word_count as word_count,
           d.precomputed_chunk_count as chunk_count,
           d.precomputed_entity_count as entity_count,
           d.precomputed_community_count as community_count,
           d.precomputed_similarity_count as similarity_count
    """
    
    results = self.execute_read(query, {"document_id": document_id})
    return results[0] if results else None
```

### List All Documents

```python
def get_all_documents(self) -> List[Dict]:
    """Get all documents with metadata."""
    query = """
    MATCH (d:Document)
    RETURN d.id as id,
           d.filename as filename,
           d.title as title,
           d.created_at as created_at,
           d.precomputed_chunk_count as chunk_count,
           d.precomputed_entity_count as entity_count
    ORDER BY d.created_at DESC
    """
    
    return self.execute_read(query)
```

### Delete Document

```python
def delete_document(self, document_id: str) -> bool:
    """Delete document and cascade delete chunks/entities."""
    query = """
    MATCH (d:Document {id: $document_id})
    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
    OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
    WHERE NOT (e)<-[:CONTAINS_ENTITY]-(:Chunk)
      OR all(chunk IN [(e)<-[:CONTAINS_ENTITY]-(ch:Chunk) | ch] 
          WHERE chunk.id IN [c.id])
    DETACH DELETE d, c, e
    RETURN count(d) as deleted
    """
    
    result = self.execute_write(query, {"document_id": document_id})
    return result.get("nodes_deleted", 0) > 0
```

## Chunk Operations

### Create Chunks (Batch)

```python
def create_chunks_batch(self, document_id: str, chunks: List[Dict]) -> int:
    """Create multiple chunks using UNWIND for efficiency."""
    query = """
    MATCH (d:Document {id: $document_id})
    UNWIND $chunks AS chunk
    CREATE (c:Chunk {
        id: chunk.id,
        text: chunk.text,
        start_char: chunk.start_char,
        end_char: chunk.end_char,
        page_number: chunk.page_number,
        chunk_index: chunk.chunk_index,
        embedding: chunk.embedding,
        word_count: chunk.word_count,
        quality_score: chunk.quality_score,
        created_at: datetime()
    })
    CREATE (d)-[:HAS_CHUNK {chunk_index: chunk.chunk_index}]->(c)
    RETURN count(c) as created_count
    """
    
    # Process in batches
    batch_size = settings.neo4j_unwind_batch_size
    total_created = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        result = self.execute_write(query, {
            "document_id": document_id,
            "chunks": batch
        })
        total_created += result.get("nodes_created", 0)
    
    return total_created
```

### Get Chunks Paginated

```python
def get_chunks_paginated(
    self,
    document_id: str,
    limit: int = 50,
    offset: int = 0
) -> List[Dict]:
    """Get document chunks with pagination."""
    query = """
    MATCH (d:Document {id: $document_id})-[r:HAS_CHUNK]->(c:Chunk)
    RETURN c.id as chunk_id,
           c.text as text,
           c.page_number as page_number,
           c.chunk_index as chunk_index,
           c.word_count as word_count,
           c.quality_score as quality_score,
           r.chunk_index as index
    ORDER BY r.chunk_index
    SKIP $offset
    LIMIT $limit
    """
    
    return self.execute_read(query, {
        "document_id": document_id,
        "limit": limit,
        "offset": offset
    })
```

### Count Document Chunks

```python
def count_document_chunks(self, document_id: str) -> int:
    """Count chunks for a document."""
    query = """
    MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
    RETURN count(c) as total
    """
    
    results = self.execute_read(query, {"document_id": document_id})
    return results[0]["total"] if results else 0
```

## Entity Operations

### Create Entities (Batch with Deduplication)

```python
def create_entities_batch(self, entities: List[Dict]) -> int:
    """Create or merge entities with deduplication."""
    query = """
    UNWIND $entities AS entity
    MERGE (e:Entity {name: entity.name, type: entity.type})
    ON CREATE SET
        e.id = entity.id,
        e.description = entity.description,
        e.embedding = entity.embedding,
        e.importance = entity.importance,
        e.provenance = [entity.chunk_id],
        e.created_at = datetime()
    ON MATCH SET
        e.description = e.description + ' ' + entity.description,
        e.importance = (e.importance + entity.importance) / 2,
        e.provenance = e.provenance + entity.chunk_id
    RETURN count(e) as processed
    """
    
    batch_size = settings.neo4j_unwind_batch_size
    total_processed = 0
    
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        result = self.execute_write(query, {"entities": batch})
        total_processed += result.get("nodes_created", 0) + result.get("properties_set", 0)
    
    return total_processed
```

### Get Entity Summary

```python
def get_entity_summary(self, document_id: str) -> Dict:
    """Get aggregated entity type counts."""
    query = """
    MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
    MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
    WITH e.type as entity_type, count(DISTINCT e) as count
    RETURN entity_type, count
    ORDER BY count DESC
    """
    
    results = self.execute_read(query, {"document_id": document_id})
    
    total = sum(r["count"] for r in results)
    groups = [
        {"type": r["entity_type"], "count": r["count"]}
        for r in results
    ]
    
    return {
        "total": total,
        "groups": groups
    }
```

### Get Entities by Type

```python
def get_entities_by_type(
    self,
    document_id: str,
    entity_type: str,
    limit: int = 50,
    offset: int = 0
) -> List[Dict]:
    """Get entities filtered by type with pagination."""
    query = """
    MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
    MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity {type: $entity_type})
    RETURN DISTINCT e.id as entity_id,
           e.name as name,
           e.type as type,
           e.description as description,
           e.importance as importance,
           e.community_id as community_id
    ORDER BY e.importance DESC
    SKIP $offset
    LIMIT $limit
    """
    
    return self.execute_read(query, {
        "document_id": document_id,
        "entity_type": entity_type,
        "limit": limit,
        "offset": offset
    })
```

### Get Entity Label (Cached)

```python
from core.singletons import cache_manager

async def get_entity_label_cached(entity_name: str) -> Optional[str]:
    """Get entity label with caching."""
    from config.settings import settings
    
    if not settings.enable_caching:
        return await _get_entity_label(entity_name)
    
    # Check cache
    if entity_name in cache_manager.entity_label_cache:
        return cache_manager.entity_label_cache[entity_name]
    
    # Query database
    label = await _get_entity_label(entity_name)
    
    # Cache result
    if label:
        cache_manager.entity_label_cache[entity_name] = label
    
    return label

async def _get_entity_label(entity_name: str) -> Optional[str]:
    """Query entity label from database."""
    db = get_db()
    query = """
    MATCH (e:Entity {name: $name})
    RETURN e.type as label
    LIMIT 1
    """
    
    results = await db.execute_read_async(query, {"name": entity_name})
    return results[0]["label"] if results else None
```

## Relationship Operations

### Create Chunk Similarities

```python
def create_similarities_batch(self, similarities: List[Dict]) -> int:
    """Create SIMILAR_TO relationships between chunks."""
    query = """
    UNWIND $similarities AS sim
    MATCH (c1:Chunk {id: sim.chunk1_id})
    MATCH (c2:Chunk {id: sim.chunk2_id})
    MERGE (c1)-[r:SIMILAR_TO]-(c2)
    SET r.strength = sim.strength,
        r.created_at = datetime()
    RETURN count(r) as created
    """
    
    batch_size = settings.neo4j_unwind_batch_size
    total_created = 0
    
    for i in range(0, len(similarities), batch_size):
        batch = similarities[i:i + batch_size]
        result = self.execute_write(query, {"similarities": batch})
        total_created += result.get("relationships_created", 0)
    
    return total_created
```

### Create Entity Relationships

```python
def create_entity_relationships_batch(self, relationships: List[Dict]) -> int:
    """Create RELATED_TO relationships between entities."""
    query = """
    UNWIND $relationships AS rel
    MATCH (e1:Entity {id: rel.source_id})
    MATCH (e2:Entity {id: rel.target_id})
    MERGE (e1)-[r:RELATED_TO]-(e2)
    ON CREATE SET
        r.strength = rel.strength,
        r.co_occurrence = 1,
        r.created_at = datetime()
    ON MATCH SET
        r.strength = (r.strength + rel.strength) / 2,
        r.co_occurrence = r.co_occurrence + 1
    RETURN count(r) as processed
    """
    
    result = self.execute_write(query, {"relationships": relationships})
    return result.get("relationships_created", 0)
```

## Statistics and Metadata

### Get Database Stats

```python
def get_database_stats(self) -> Dict:
    """Get comprehensive database statistics."""
    query = """
    MATCH (d:Document)
    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
    OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
    OPTIONAL MATCH (c)-[s:SIMILAR_TO]->()
    OPTIONAL MATCH (e)-[r:RELATED_TO]->()
    RETURN count(DISTINCT d) as document_count,
           count(DISTINCT c) as chunk_count,
           count(DISTINCT e) as entity_count,
           count(DISTINCT s) as similarity_count,
           count(DISTINCT r) as relationship_count
    """
    
    results = self.execute_read(query)
    return results[0] if results else {}
```

### Update Precomputed Stats

```python
def update_document_stats(self, document_id: str):
    """Update precomputed statistics for a document."""
    query = """
    MATCH (d:Document {id: $document_id})
    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
    OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
    OPTIONAL MATCH (c)-[s:SIMILAR_TO]->()
    WITH d,
         count(DISTINCT c) as chunk_count,
         count(DISTINCT e) as entity_count,
         count(DISTINCT e.community_id) as community_count,
         count(DISTINCT s) as similarity_count
    SET d.precomputed_chunk_count = chunk_count,
        d.precomputed_entity_count = entity_count,
        d.precomputed_community_count = community_count,
        d.precomputed_similarity_count = similarity_count
    RETURN d.id as document_id
    """
    
    self.execute_write(query, {"document_id": document_id})
```

## Performance Optimization

### Query Profiling

```python
def profile_query(self, query: str, params: Dict = None) -> Dict:
    """Profile query execution."""
    profile_query = f"PROFILE {query}"
    
    with self.driver.session() as session:
        result = session.run(profile_query, params or {})
        profile = result.consume().profile
        
        return {
            "db_hits": profile.db_hits,
            "rows": profile.rows,
            "time_ms": profile.time
        }
```

### Connection Pool Monitoring

```python
def get_connection_pool_stats(self) -> Dict:
    """Get connection pool statistics."""
    return {
        "max_pool_size": settings.neo4j_max_connection_pool_size,
        "active_connections": len(self.driver._pool._connections),
        "idle_connections": len(self.driver._pool._idle)
    }
```

## Testing

### Unit Tests

```python
import pytest
from core.graph_db import get_db

@pytest.fixture
def db():
    db = get_db()
    yield db
    # Cleanup
    db.execute_write("MATCH (n) DETACH DELETE n")

def test_create_document(db):
    doc_data = {
        "id": "test123",
        "filename": "test.pdf",
        "file_path": "/test/test.pdf",
        "file_type": "application/pdf",
        "file_size": 1024,
        "title": "Test Document",
        "page_count": 10,
        "word_count": 500
    }
    
    doc_id = db.create_document(doc_data)
    assert doc_id == "test123"
    
    doc = db.get_document("test123")
    assert doc["filename"] == "test.pdf"

def test_get_database_stats(db):
    stats = db.get_database_stats()
    assert "document_count" in stats
    assert "chunk_count" in stats
    assert "entity_count" in stats
```

## Related Documentation

- [Data Model](02-core-concepts/data-model.md)
- [Entity Extraction](03-components/backend/entity-extraction.md)
- [Retriever](03-components/backend/retriever.md)
- [Caching System](02-core-concepts/caching-system.md)
