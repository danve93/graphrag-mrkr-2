# Data Model

Neo4j graph schema for documents, chunks, entities, and relationships.

## Overview

Amber uses Neo4j as a graph database to represent document structure, semantic relationships, and entity connections. The schema supports:

- **Hierarchical document structure** (Document → Chunks)
- **Entity extraction and linking** (Chunks ↔ Entities)
- **Semantic similarity** (Chunk-Chunk relationships)
- **Entity relationships** (Entity-Entity connections)
- **Community clustering** (Entity communities)

## Node Types

### Document

Represents an ingested document file.

**Label**: `Document`

**Properties**:
```python
{
  "id": str,                          # Unique document ID (MD5 hash)
  "filename": str,                    # Original filename
  "file_path": str,                   # Storage path
  "file_type": str,                   # MIME type (e.g., "application/pdf")
  "file_size": int,                   # Size in bytes
  "title": str,                       # Extracted or inferred title
  "folder_id": str | null,            # Optional folder id
  "folder_name": str | null,          # Optional folder name (denormalized)
  "folder_order": int | null,         # Manual ordering index within folder/root
  "created_at": datetime,             # Ingestion timestamp
  "page_count": int,                  # Number of pages (if applicable)
  "word_count": int,                  # Total word count
  "precomputed_chunk_count": int,     # Number of chunks (cached)
  "precomputed_entity_count": int,    # Number of entities (cached)
  "precomputed_community_count": int, # Number of communities (cached)
  "precomputed_similarity_count": int # Number of similarities (cached)
}
```

**Indexes**:
```cypher
CREATE INDEX document_id FOR (d:Document) ON (d.id);
CREATE INDEX document_filename FOR (d:Document) ON (d.filename);
```

**Example Query**:
```cypher
MATCH (d:Document {id: "abc123"})
RETURN d.filename, d.precomputed_chunk_count, d.precomputed_entity_count;
```

### Folder

Represents a user-facing folder for grouping documents.

**Label**: `Folder`

**Properties**:
```python
{
  "id": str,                  # Unique folder ID (UUID)
  "name": str,                # Unique folder name
  "created_at": datetime      # Creation timestamp
}
```

**Indexes**:
```cypher
CREATE INDEX folder_id FOR (f:Folder) ON (f.id);
CREATE INDEX folder_name FOR (f:Folder) ON (f.name);
```

**Example Query**:
```cypher
MATCH (f:Folder)
RETURN f.name
ORDER BY f.name ASC;
```

### Chunk

Represents a text segment from a document.

**Label**: `Chunk`

**Properties**:
```python
{
  "id": str,                    # Unique chunk ID (UUID)
  "text": str,                  # Chunk text content
  "start_char": int,            # Start position in document
  "end_char": int,              # End position in document
  "page_number": int,           # Page number (if applicable)
  "chunk_index": int,           # Sequential index in document
  "embedding": List[float],     # Vector embedding (1536 or 3072 dims)
  "word_count": int,            # Words in chunk
  "quality_score": float,       # Quality score (0.0-1.0, optional)
  "created_at": datetime        # Creation timestamp
}
```

**Indexes**:
```cypher
CREATE INDEX chunk_id FOR (c:Chunk) ON (c.id);
CREATE INDEX chunk_document FOR (c:Chunk) ON (c.document_id);

# Vector index for similarity search
CALL db.index.vector.createNodeIndex(
  'chunk_embeddings',
  'Chunk',
  'embedding',
  1536,
  'cosine'
);
```

**Example Query**:
```cypher
MATCH (c:Chunk {id: "xyz789"})
RETURN c.text, c.page_number, c.quality_score;
```

### Entity

Represents an extracted named entity.

**Label**: `Entity`

**Properties**:
```python
{
  "id": str,                    # Unique entity ID (UUID or canonical)
  "name": str,                  # Entity name
  "type": str,                  # Entity type (see Entity Taxonomy)
  "description": str,           # Accumulated descriptions
  "embedding": List[float],     # Vector embedding
  "importance": float,          # Importance score (0.0-1.0)
  "provenance": List[str],      # Source chunk IDs
  "community_id": int,          # Leiden community ID (optional)
  "created_at": datetime        # Creation timestamp
}
```

**Indexes**:
```cypher
CREATE INDEX entity_id FOR (e:Entity) ON (e.id);
CREATE INDEX entity_name FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type FOR (e:Entity) ON (e.type);
CREATE INDEX entity_community FOR (e:Entity) ON (e.community_id);

# Vector index for entity search
CALL db.index.vector.createNodeIndex(
  'entity_embeddings',
  'Entity',
  'embedding',
  1536,
  'cosine'
);
```

**Example Query**:
```cypher
MATCH (e:Entity {type: "COMPONENT", community_id: 5})
RETURN e.name, e.description, e.importance
ORDER BY e.importance DESC
LIMIT 10;
```

### Community

Represents a semantic cluster of related entities.

**Label**: `Community`

**Properties**:
```python
{
  "id": int,                    # Community ID (from Leiden)
  "size": int,                  # Number of entities
  "summary": str,               # LLM-generated summary (optional)
  "representative_entities": List[str], # Key entity names
  "color": str,                 # Hex color for visualization
  "created_at": datetime        # Creation timestamp
}
```

**Indexes**:
```cypher
CREATE INDEX community_id FOR (c:Community) ON (c.id);
```

**Example Query**:
```cypher
MATCH (comm:Community {id: 5})
MATCH (e:Entity {community_id: 5})
RETURN comm.summary, count(e) as entity_count;
```

## Relationship Types

### IN_FOLDER

Links Document to its folder.

**Type**: `IN_FOLDER`

**Direction**: `(Document)-[:IN_FOLDER]->(Folder)`

**Example Query**:
```cypher
MATCH (d:Document {id: "abc123"})-[:IN_FOLDER]->(f:Folder)
RETURN f.name;
```

### HAS_CHUNK

Links Document to its Chunks.

**Type**: `HAS_CHUNK`

**Direction**: `(Document)-[:HAS_CHUNK]->(Chunk)`

**Properties**:
```python
{
  "chunk_index": int  # Sequential position
}
```

**Example Query**:
```cypher
MATCH (d:Document {id: "abc123"})-[r:HAS_CHUNK]->(c:Chunk)
RETURN c.text, r.chunk_index
ORDER BY r.chunk_index;
```

### CONTAINS_ENTITY

Links Chunk to extracted Entities.

**Type**: `CONTAINS_ENTITY`

**Direction**: `(Chunk)-[:CONTAINS_ENTITY]->(Entity)`

**Properties**:
```python
{
  "mention_count": int  # Occurrences in chunk
}
```

**Example Query**:
```cypher
MATCH (c:Chunk)-[r:CONTAINS_ENTITY]->(e:Entity {type: "COMPONENT"})
WHERE c.document_id = "abc123"
RETURN e.name, count(c) as chunk_count
ORDER BY chunk_count DESC;
```

### SIMILAR_TO

Semantic similarity between Chunks.

**Type**: `SIMILAR_TO`

**Direction**: `(Chunk)-[:SIMILAR_TO]-(Chunk)` (undirected)

**Properties**:
```python
{
  "strength": float,     # Cosine similarity (0.0-1.0)
  "created_at": datetime
}
```

**Creation**: Generated by `scripts/create_similarities.py` or during ingestion.

**Example Query**:
```cypher
MATCH (c1:Chunk {id: "xyz789"})-[r:SIMILAR_TO]-(c2:Chunk)
WHERE r.strength >= 0.7
RETURN c2.text, r.strength
ORDER BY r.strength DESC
LIMIT 5;
```

### RELATED_TO

Semantic relationship between Entities.

**Type**: `RELATED_TO`

**Direction**: `(Entity)-[:RELATED_TO]-(Entity)` (undirected)

**Properties**:
```python
{
  "strength": float,     # Relationship strength (0.0-1.0)
  "co_occurrence": int,  # Times entities appear together
  "created_at": datetime
}
```

**Creation**: Generated during entity extraction based on co-occurrence and context.

**Example Query**:
```cypher
MATCH (e1:Entity {name: "VMware"})-[r:RELATED_TO]-(e2:Entity)
WHERE r.strength >= 0.5
RETURN e2.name, e2.type, r.strength
ORDER BY r.strength DESC;
```

### IN_COMMUNITY

Assigns Entity to a Community cluster.

**Type**: `IN_COMMUNITY`

**Direction**: `(Entity)-[:IN_COMMUNITY]->(Community)`

**Properties**: None

**Creation**: Generated by Leiden clustering algorithm.

**Example Query**:
```cypher
MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community {id: 5})
RETURN e.name, e.type
ORDER BY e.importance DESC;
```

## Graph Schema Diagram

```
┌──────────────┐
│   Document   │
│  - id        │
│  - filename  │
│  - title     │
└──────┬───────┘
       │ HAS_CHUNK
       │
       ▼
┌──────────────┐           ┌──────────────┐
│    Chunk     │───────────│    Chunk     │
│  - id        │ SIMILAR_TO│  - id        │
│  - text      │◄──────────│  - text      │
│  - embedding │           │  - embedding │
└──────┬───────┘           └──────┬───────┘
       │ CONTAINS_ENTITY          │ CONTAINS_ENTITY
       │                          │
       ▼                          ▼
┌──────────────┐           ┌──────────────┐
│    Entity    │───────────│    Entity    │
│  - name      │ RELATED_TO│  - name      │
│  - type      │◄──────────│  - type      │
│  - community │           │  - community │
└──────┬───────┘           └──────┬───────┘
       │ IN_COMMUNITY             │ IN_COMMUNITY
       │                          │
       ▼                          ▼
┌──────────────────────────────────┐
│          Community               │
│  - id                            │
│  - summary                       │
│  - representative_entities       │
└──────────────────────────────────┘
```

## Data Creation Flow

### Document Ingestion

1. **Document Node Creation**:
```cypher
CREATE (d:Document {
  id: $id,
  filename: $filename,
  file_path: $file_path,
  file_type: $file_type,
  file_size: $file_size,
  title: $title,
  created_at: datetime(),
  precomputed_chunk_count: 0,
  precomputed_entity_count: 0
})
```

2. **Chunk Creation (Batch)**:
```cypher
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
  created_at: datetime()
})
CREATE (d)-[:HAS_CHUNK {chunk_index: chunk.chunk_index}]->(c)
```

3. **Update Precomputed Stats**:
```cypher
MATCH (d:Document {id: $document_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
WITH d, count(c) as chunk_count
SET d.precomputed_chunk_count = chunk_count
```

### Entity Extraction

1. **Entity Creation with Deduplication**:
```cypher
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
```

2. **Link Chunks to Entities**:
```cypher
MATCH (c:Chunk {id: $chunk_id})
MATCH (e:Entity {id: $entity_id})
MERGE (c)-[r:CONTAINS_ENTITY]->(e)
ON CREATE SET r.mention_count = 1
ON MATCH SET r.mention_count = r.mention_count + 1
```

3. **Create Entity Relationships**:
```cypher
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
```

### Similarity Calculation

**Chunk Similarities**:
```cypher
MATCH (c1:Chunk), (c2:Chunk)
WHERE c1.id < c2.id
  AND gds.similarity.cosine(c1.embedding, c2.embedding) >= $threshold
WITH c1, c2, gds.similarity.cosine(c1.embedding, c2.embedding) AS similarity
MERGE (c1)-[r:SIMILAR_TO]-(c2)
SET r.strength = similarity,
    r.created_at = datetime()
```

### Community Detection

**Leiden Clustering**:
```python
# See scripts/run_clustering.py
# Creates Community nodes and IN_COMMUNITY relationships
```

## Query Patterns

### Retrieve Document with Chunks

```cypher
MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
RETURN d, collect(c) as chunks
ORDER BY c.chunk_index;
```

### Find Similar Chunks

```cypher
MATCH (seed:Chunk {id: $chunk_id})
MATCH (seed)-[r:SIMILAR_TO]-(similar:Chunk)
WHERE r.strength >= $threshold
RETURN similar, r.strength
ORDER BY r.strength DESC
LIMIT $limit;
```

### Multi-Hop Entity Traversal

```cypher
MATCH (seed:Chunk {id: $chunk_id})
MATCH (seed)-[:CONTAINS_ENTITY]->(e1:Entity)
MATCH (e1)-[:RELATED_TO*1..2]-(e2:Entity)
MATCH (e2)<-[:CONTAINS_ENTITY]-(related:Chunk)
WHERE seed.id <> related.id
RETURN DISTINCT related, count(DISTINCT e2) as entity_count
ORDER BY entity_count DESC
LIMIT $limit;
```

### Community Members

```cypher
MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community {id: $community_id})
RETURN e.name, e.type, e.importance
ORDER BY e.importance DESC;
```

### Entity Co-occurrence

```cypher
MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e1:Entity {name: $entity_name})
MATCH (c)-[:CONTAINS_ENTITY]->(e2:Entity)
WHERE e1 <> e2
RETURN e2.name, e2.type, count(c) as co_occurrence
ORDER BY co_occurrence DESC
LIMIT 20;
```

## Precomputed Statistics

To optimize UI rendering, document nodes cache aggregate counts:

**Update Script** (run after ingestion):
```cypher
MATCH (d:Document)
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
OPTIONAL MATCH (c)-[s:SIMILAR_TO]-()
WITH d,
     count(DISTINCT c) as chunk_count,
     count(DISTINCT e) as entity_count,
     count(DISTINCT s) as similarity_count
SET d.precomputed_chunk_count = chunk_count,
    d.precomputed_entity_count = entity_count,
    d.precomputed_similarity_count = similarity_count
```

**Community Count**:
```cypher
MATCH (d:Document)-[:HAS_CHUNK]->(:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
WHERE e.community_id IS NOT NULL
WITH d, count(DISTINCT e.community_id) as community_count
SET d.precomputed_community_count = community_count
```

## Index Management

### Create Required Indexes

```cypher
# Document indexes
CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id);
CREATE INDEX document_filename IF NOT EXISTS FOR (d:Document) ON (d.filename);

# Chunk indexes
CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id);
CREATE TEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON (c.text);

# Entity indexes
CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_community IF NOT EXISTS FOR (e:Entity) ON (e.community_id);

# Vector indexes
CALL db.index.vector.createNodeIndex(
  'chunk_embeddings',
  'Chunk',
  'embedding',
  1536,
  'cosine'
);

CALL db.index.vector.createNodeIndex(
  'entity_embeddings',
  'Entity',
  'embedding',
  1536,
  'cosine'
);
```

### Verify Indexes

```cypher
SHOW INDEXES;
```

### Drop Index

```cypher
DROP INDEX index_name;
```

## Database Maintenance

### Count All Nodes

```cypher
MATCH (n)
RETURN labels(n) as label, count(n) as count
ORDER BY count DESC;
```

### Count All Relationships

```cypher
MATCH ()-[r]->()
RETURN type(r) as relationship, count(r) as count
ORDER BY count DESC;
```

### Delete Document and Cascade

```cypher
MATCH (d:Document {id: $document_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
WHERE NOT (e)<-[:CONTAINS_ENTITY]-(:Chunk)
  OR all(chunk IN [(e)<-[:CONTAINS_ENTITY]-(ch) | ch] WHERE chunk.id IN [c.id])
DETACH DELETE d, c, e
```

### Clear All Data

```cypher
MATCH (n) DETACH DELETE n;
```

## Performance Optimization

### Query Optimization

**Use parameters**:
```cypher
// Good
MATCH (d:Document {id: $id}) RETURN d;

// Bad (string concatenation)
MATCH (d:Document {id: "abc123"}) RETURN d;
```

**Limit early**:
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

### Index Usage

Check query plan:
```cypher
EXPLAIN MATCH (c:Chunk {id: $chunk_id}) RETURN c;
PROFILE MATCH (c:Chunk {id: $chunk_id}) RETURN c;
```

## Related Documentation

- [Entity Types Taxonomy](02-core-concepts/entity-types.md)
- [Graph Database Component](03-components/backend/graph-database.md)
- [Entity Extraction](03-components/backend/entity-extraction.md)
- [Clustering](04-features/entity-clustering.md)
