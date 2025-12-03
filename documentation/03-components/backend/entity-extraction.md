# Entity Extraction Component

LLM-based entity and relationship extraction from document chunks.

## Overview

The entity extraction component uses LLMs to identify entities (people, organizations, concepts, etc.) and their relationships within document text. It supports both synchronous and asynchronous extraction, implements multi-phase accumulation with deduplication, and creates a knowledge graph structure for enhanced retrieval.

**Location**: `core/entity_extraction.py`, `core/entity_graph.py`
**Extraction**: LLM-based (OpenAI GPT-4, Ollama Llama3)
**Deduplication**: In-memory NetworkX graph with Phase 2 accumulation

## Architecture

```
┌──────────────────────────────────────────────────┐
│         Entity Extraction Pipeline                │
├──────────────────────────────────────────────────┤
│                                                   │
│  Phase 1: Per-Chunk Extraction                   │
│  ┌─────────────────────────────────────────────┐ │
│  │  Chunk Text → LLM Prompt → Parse Response  │ │
│  │  → Extract (Entity, Relationship) Tuples   │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Phase 2: Accumulation & Deduplication          │
│  ┌─────────────────────────────────────────────┐ │
│  │  NetworkX Graph (In-Memory)                 │ │
│  │  • Merge duplicate entities by name+type    │ │
│  │  • Accumulate descriptions                  │ │
│  │  • Sum relationship strengths               │ │
│  │  • Track provenance (source chunks)         │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Phase 3: Batch Persistence                      │
│  ┌─────────────────────────────────────────────┐ │
│  │  Export Cypher UNWIND statements            │ │
│  │  → Neo4j batch write                        │ │
│  │  → Create Entity nodes + RELATED_TO edges   │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
└──────────────────────────────────────────────────┘
```

## Entity Model

### Canonical Entity Types

**26 entity types** defined in `core/entity_models.py`:

```python
ENTITY_TYPES = [
    # Infrastructure
    "Component", "Service", "Node", "Domain",
    
    # Business
    "Class of Service", "Account", "Account Type", "Role",
    
    # Storage & Data
    "Resource", "Quota Object", "Backup Object", "Item",
    "Storage Object",
    
    # Operations
    "Migration Procedure", "Certificate", "Config Option",
    "Security Feature", "CLI Command", "API Object",
    "Task", "Procedure",
    
    # Generic
    "Concept", "Document", "Person", "Organization",
    "Location", "Event", "Technology", "Product",
    "Date", "Money"
]
```

### Entity Schema

```python
@dataclass
class Entity:
    name: str              # Entity identifier
    type: str              # One of ENTITY_TYPES
    description: str       # Accumulated descriptions
    importance: float      # 0.0-1.0 importance score
    chunk_ids: List[str]   # Source chunk provenance
    embedding: List[float] # Semantic embedding
    community_id: Optional[int] = None  # Leiden cluster
```

### Relationship Schema

```python
@dataclass
class Relationship:
    source: str           # Source entity name
    target: str           # Target entity name
    type: str             # Relationship type (e.g., "RELATED_TO")
    strength: float       # 0.0-1.0 relationship strength
    description: str      # Relationship description
    chunk_ids: List[str]  # Source chunk provenance
```

## Entity Extraction

### Core Extraction Function

```python
from core.llm import LLMManager
from core.entity_models import ENTITY_TYPES
import json

async def extract_entities_from_chunk(
    chunk_text: str,
    chunk_id: str,
    llm_manager: LLMManager
) -> tuple[List[Dict], List[Dict]]:
    """
    Extract entities and relationships from a single chunk.
    
    Args:
        chunk_text: Text content to analyze
        chunk_id: Chunk identifier for provenance
        llm_manager: LLM manager instance
    
    Returns:
        Tuple of (entities_list, relationships_list)
    """
    prompt = _build_extraction_prompt(chunk_text)
    
    response = await llm_manager.generate(
        prompt=prompt,
        temperature=0.0,  # Deterministic extraction
        max_tokens=2000
    )
    
    # Parse JSON response
    try:
        data = json.loads(response)
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        
        # Add provenance
        for entity in entities:
            entity["chunk_ids"] = [chunk_id]
        
        for rel in relationships:
            rel["chunk_ids"] = [chunk_id]
        
        return entities, relationships
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse entity extraction response: {e}")
        return [], []
```

### LLM Prompt Template

```python
def _build_extraction_prompt(text: str) -> str:
    """Build entity extraction prompt."""
    entity_types_str = ", ".join(ENTITY_TYPES)
    
    return f"""Extract entities and their relationships from the following text.

ENTITY TYPES (use these exact labels):
{entity_types_str}

TEXT:
{text}

INSTRUCTIONS:
1. Identify all significant entities in the text
2. Assign each entity to the most appropriate type from the list above
3. Extract relationships between entities
4. Provide brief descriptions for each entity and relationship
5. Assign importance scores (0.0-1.0) based on relevance

OUTPUT FORMAT (JSON):
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "Entity Type",
      "description": "Brief description",
      "importance": 0.8
    }}
  ],
  "relationships": [
    {{
      "source": "Source Entity Name",
      "target": "Target Entity Name",
      "type": "RELATED_TO",
      "strength": 0.7,
      "description": "How they are related"
    }}
  ]
}}

Respond ONLY with valid JSON. Do not include any other text."""
```

## Phase 2: Entity Accumulation

### NetworkX Graph Accumulation

```python
import networkx as nx
from core.entity_graph import EntityGraph

class EntityGraph:
    """In-memory graph for entity deduplication and accumulation."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_data = {}  # {(name, type): entity_dict}
    
    def add_entity(self, entity: Dict):
        """Add or merge entity."""
        key = (entity["name"], entity["type"])
        
        if key in self.entity_data:
            # Merge with existing
            existing = self.entity_data[key]
            
            # Accumulate descriptions
            existing["description"] = (
                f"{existing['description']} {entity['description']}"
            ).strip()
            
            # Average importance
            existing["importance"] = (
                existing["importance"] + entity["importance"]
            ) / 2
            
            # Merge provenance
            existing["chunk_ids"].extend(entity["chunk_ids"])
            existing["chunk_ids"] = list(set(existing["chunk_ids"]))
        
        else:
            # New entity
            self.entity_data[key] = entity
            self.graph.add_node(key, **entity)
    
    def add_relationship(self, relationship: Dict):
        """Add or merge relationship."""
        source_key = (relationship["source"], relationship.get("source_type", "Concept"))
        target_key = (relationship["target"], relationship.get("target_type", "Concept"))
        
        if not self.graph.has_node(source_key) or not self.graph.has_node(target_key):
            logger.warning(f"Relationship references unknown entity: {source_key} -> {target_key}")
            return
        
        if self.graph.has_edge(source_key, target_key):
            # Merge with existing
            edge_data = self.graph[source_key][target_key]
            edge_data["strength"] = (
                edge_data["strength"] + relationship["strength"]
            ) / 2
            edge_data["chunk_ids"].extend(relationship["chunk_ids"])
            edge_data["chunk_ids"] = list(set(edge_data["chunk_ids"]))
        
        else:
            # New relationship
            self.graph.add_edge(
                source_key,
                target_key,
                **relationship
            )
    
    def get_entities(self) -> List[Dict]:
        """Export accumulated entities."""
        return list(self.entity_data.values())
    
    def get_relationships(self) -> List[Dict]:
        """Export accumulated relationships."""
        relationships = []
        for source, target, data in self.graph.edges(data=True):
            relationships.append({
                "source": source[0],  # name
                "source_type": source[1],  # type
                "target": target[0],
                "target_type": target[1],
                **data
            })
        return relationships
```

### Batch Extraction with Accumulation

```python
async def extract_entities_batch(
    chunks: List[Dict],
    show_progress: bool = False
) -> tuple[List[Dict], List[Dict]]:
    """
    Extract entities from multiple chunks with accumulation.
    
    Args:
        chunks: List of chunk dicts with 'id' and 'text'
        show_progress: Show progress bar
    
    Returns:
        Tuple of (deduplicated_entities, deduplicated_relationships)
    """
    llm_manager = LLMManager()
    entity_graph = EntityGraph()
    
    # Phase 1: Extract from each chunk
    tasks = [
        extract_entities_from_chunk(
            chunk_text=chunk["text"],
            chunk_id=chunk["id"],
            llm_manager=llm_manager
        )
        for chunk in chunks
    ]
    
    if show_progress:
        from tqdm.asyncio import tqdm
        results = await tqdm.gather(*tasks, desc="Extracting entities")
    else:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Phase 2: Accumulate in graph
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Entity extraction failed: {result}")
            continue
        
        entities, relationships = result
        
        for entity in entities:
            entity_graph.add_entity(entity)
        
        for rel in relationships:
            entity_graph.add_relationship(rel)
    
    # Phase 3: Export
    deduplicated_entities = entity_graph.get_entities()
    deduplicated_relationships = entity_graph.get_relationships()
    
    logger.info(
        f"Extracted {len(deduplicated_entities)} unique entities "
        f"and {len(deduplicated_relationships)} relationships "
        f"from {len(chunks)} chunks"
    )
    
    return deduplicated_entities, deduplicated_relationships
```

## Entity Embeddings

### Generate Entity Embeddings

```python
from core.embeddings import EmbeddingManager

async def add_entity_embeddings(entities: List[Dict]) -> List[Dict]:
    """Add embeddings to entities based on name + description."""
    manager = EmbeddingManager()
    
    try:
        # Build entity texts
        texts = [
            f"{entity['name']}: {entity['description']}"
            for entity in entities
        ]
        
        # Generate embeddings
        embeddings = await manager.get_embeddings_batch(
            texts=texts,
            show_progress=True
        )
        
        # Attach to entities
        for entity, embedding in zip(entities, embeddings):
            entity["embedding"] = embedding
        
        return entities
    
    finally:
        await manager.close()
```

### Synchronous vs Asynchronous Embedding

```python
# Synchronous mode (for tests/small batches)
SYNC_ENTITY_EMBEDDINGS=true

# Asynchronous mode (for production)
SYNC_ENTITY_EMBEDDINGS=false
```

```python
if settings.sync_entity_embeddings:
    # Synchronous - blocks until complete
    entities = await add_entity_embeddings(entities)
    persist_entities(entities)
else:
    # Asynchronous - return immediately, embed in background
    asyncio.create_task(
        async_embed_and_persist(entities)
    )
```

## Persistence to Neo4j

### Batch Entity Creation

```python
from core.graph_db import get_db

def persist_entities_batch(entities: List[Dict], chunk_id: str):
    """Persist entities to Neo4j with UNWIND."""
    db = get_db()
    
    query = """
    MATCH (c:Chunk {id: $chunk_id})
    UNWIND $entities AS entity
    MERGE (e:Entity {name: entity.name, type: entity.type})
    ON CREATE SET
        e.id = randomUUID(),
        e.description = entity.description,
        e.importance = entity.importance,
        e.embedding = entity.embedding,
        e.provenance = entity.chunk_ids,
        e.created_at = datetime()
    ON MATCH SET
        e.description = e.description + ' ' + entity.description,
        e.importance = (e.importance + entity.importance) / 2,
        e.provenance = e.provenance + entity.chunk_ids
    CREATE (c)-[:CONTAINS_ENTITY {
        importance: entity.importance,
        created_at: datetime()
    }]->(e)
    """
    
    # Process in batches
    batch_size = 100
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        db.execute_write(query, {
            "chunk_id": chunk_id,
            "entities": batch
        })
```

### Batch Relationship Creation

```python
def persist_relationships_batch(relationships: List[Dict]):
    """Persist relationships to Neo4j."""
    db = get_db()
    
    query = """
    UNWIND $relationships AS rel
    MATCH (e1:Entity {name: rel.source, type: rel.source_type})
    MATCH (e2:Entity {name: rel.target, type: rel.target_type})
    MERGE (e1)-[r:RELATED_TO]-(e2)
    ON CREATE SET
        r.strength = rel.strength,
        r.description = rel.description,
        r.provenance = rel.chunk_ids,
        r.co_occurrence = 1,
        r.created_at = datetime()
    ON MATCH SET
        r.strength = (r.strength + rel.strength) / 2,
        r.co_occurrence = r.co_occurrence + 1,
        r.provenance = r.provenance + rel.chunk_ids
    """
    
    batch_size = 100
    for i in range(0, len(relationships), batch_size):
        batch = relationships[i:i + batch_size]
        db.execute_write(query, {"relationships": batch})
```

## Background Extraction

### Async Task Management

```python
import asyncio
from typing import Optional

class EntityExtractionTask:
    """Background entity extraction task tracker."""
    
    def __init__(self, document_id: str, chunks: List[Dict]):
        self.document_id = document_id
        self.chunks = chunks
        self.status = "pending"
        self.progress = 0.0
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
    
    async def run(self):
        """Execute extraction."""
        try:
            self.status = "running"
            
            # Extract entities
            entities, relationships = await extract_entities_batch(
                chunks=self.chunks,
                show_progress=False
            )
            
            self.progress = 0.5
            
            # Add embeddings
            entities = await add_entity_embeddings(entities)
            
            self.progress = 0.8
            
            # Persist to Neo4j
            for chunk in self.chunks:
                chunk_entities = [
                    e for e in entities
                    if chunk["id"] in e["chunk_ids"]
                ]
                persist_entities_batch(chunk_entities, chunk["id"])
            
            persist_relationships_batch(relationships)
            
            self.progress = 1.0
            self.status = "completed"
            
            logger.info(
                f"Entity extraction completed for {self.document_id}: "
                f"{len(entities)} entities, {len(relationships)} relationships"
            )
        
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            logger.error(f"Entity extraction failed for {self.document_id}: {e}")
    
    def start(self):
        """Start background task."""
        self.task = asyncio.create_task(self.run())
        return self.task
```

### Task Registry

```python
class EntityExtractionRegistry:
    """Global registry for entity extraction tasks."""
    
    def __init__(self):
        self.tasks: Dict[str, EntityExtractionTask] = {}
    
    def create_task(self, document_id: str, chunks: List[Dict]) -> str:
        """Create and start extraction task."""
        task = EntityExtractionTask(document_id, chunks)
        task.start()
        
        self.tasks[document_id] = task
        return document_id
    
    def get_status(self, document_id: str) -> Optional[Dict]:
        """Get task status."""
        task = self.tasks.get(document_id)
        if not task:
            return None
        
        return {
            "status": task.status,
            "progress": task.progress,
            "error": task.error
        }
    
    def cleanup_completed(self):
        """Remove completed tasks."""
        self.tasks = {
            doc_id: task
            for doc_id, task in self.tasks.items()
            if task.status not in ["completed", "failed"]
        }

# Global registry
extraction_registry = EntityExtractionRegistry()
```

## Configuration

### Environment Variables

```bash
# Entity extraction toggle
ENABLE_ENTITY_EXTRACTION=true

# Extraction mode
SYNC_ENTITY_EMBEDDINGS=false  # true for tests, false for production

# LLM settings for extraction
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0  # Deterministic extraction

# Embedding settings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

# Batch sizes
NEO4J_UNWIND_BATCH_SIZE=100
EXTRACTION_BATCH_SIZE=50
```

### Extraction Parameters

```python
from config.settings import settings

# Enable/disable extraction
if settings.enable_entity_extraction:
    entities, relationships = await extract_entities_batch(chunks)

# Synchronous vs asynchronous
if settings.sync_entity_embeddings:
    # Block until embeddings complete
    entities = await add_entity_embeddings(entities)
else:
    # Background task
    extraction_registry.create_task(document_id, chunks)
```

## Usage Examples

### Full Extraction Pipeline

```python
async def extract_and_persist_entities(document_id: str, chunks: List[Dict]):
    """Complete entity extraction and persistence."""
    if not settings.enable_entity_extraction:
        logger.info("Entity extraction disabled")
        return
    
    # Extract entities and relationships
    entities, relationships = await extract_entities_batch(
        chunks=chunks,
        show_progress=True
    )
    
    # Add embeddings
    entities = await add_entity_embeddings(entities)
    
    # Persist to Neo4j
    for chunk in chunks:
        chunk_entities = [
            e for e in entities
            if chunk["id"] in e["chunk_ids"]
        ]
        persist_entities_batch(chunk_entities, chunk["id"])
    
    persist_relationships_batch(relationships)
    
    # Update document stats
    from core.graph_db import get_db
    db = get_db()
    db.update_document_stats(document_id)
    
    logger.info(
        f"Extracted {len(entities)} entities and "
        f"{len(relationships)} relationships for {document_id}"
    )
```

### Query Entities

```python
def get_document_entities(document_id: str, entity_type: Optional[str] = None):
    """Get entities for a document."""
    from core.graph_db import get_db
    db = get_db()
    
    if entity_type:
        return db.get_entities_by_type(
            document_id=document_id,
            entity_type=entity_type,
            limit=100
        )
    else:
        return db.get_entity_summary(document_id)
```

## Testing

### Unit Tests

```python
import pytest
from core.entity_extraction import extract_entities_from_chunk
from core.llm import LLMManager

@pytest.mark.asyncio
async def test_entity_extraction():
    text = """
    OpenAI released GPT-4 in March 2023. The model was developed by
    the OpenAI research team in San Francisco.
    """
    
    llm_manager = LLMManager()
    entities, relationships = await extract_entities_from_chunk(
        chunk_text=text,
        chunk_id="test_chunk_1",
        llm_manager=llm_manager
    )
    
    # Verify entities
    entity_names = [e["name"] for e in entities]
    assert "OpenAI" in entity_names
    assert "GPT-4" in entity_names
    
    # Verify relationships
    assert len(relationships) > 0
    assert any(
        r["source"] == "OpenAI" and r["target"] == "GPT-4"
        for r in relationships
    )

@pytest.mark.asyncio
async def test_entity_deduplication():
    from core.entity_graph import EntityGraph
    
    graph = EntityGraph()
    
    # Add duplicate entities
    graph.add_entity({
        "name": "OpenAI",
        "type": "Organization",
        "description": "AI research company",
        "importance": 0.9,
        "chunk_ids": ["chunk1"]
    })
    
    graph.add_entity({
        "name": "OpenAI",
        "type": "Organization",
        "description": "Creator of GPT models",
        "importance": 0.8,
        "chunk_ids": ["chunk2"]
    })
    
    # Verify deduplication
    entities = graph.get_entities()
    assert len(entities) == 1
    assert "chunk1" in entities[0]["chunk_ids"]
    assert "chunk2" in entities[0]["chunk_ids"]
    assert "AI research" in entities[0]["description"]
    assert "GPT models" in entities[0]["description"]
```

## Troubleshooting

### Common Issues

**Issue**: Extraction returns empty results
```python
# Solution: Check LLM response format
logger.debug(f"LLM response: {response}")

# Verify JSON parsing
try:
    data = json.loads(response)
except json.JSONDecodeError as e:
    logger.error(f"JSON parse error at position {e.pos}: {e.msg}")
```

**Issue**: Duplicate entities not merging
```python
# Solution: Verify entity key generation
key = (entity["name"].strip().lower(), entity["type"])

# Normalize entity names
entity["name"] = entity["name"].strip()
```

**Issue**: Relationship references unknown entities
```python
# Solution: Create entities before relationships
for entity in entities:
    entity_graph.add_entity(entity)

# Then add relationships
for rel in relationships:
    entity_graph.add_relationship(rel)
```

**Issue**: High memory usage with large documents
```python
# Solution: Process chunks in smaller batches
batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    entities, rels = await extract_entities_batch(batch)
    persist_entities_batch(entities, document_id)
    persist_relationships_batch(rels)
```

## Related Documentation

- [Entity Types Reference](02-core-concepts/entity-types.md)
- [LLM Component](03-components/backend/llm.md)
- [Graph Database](03-components/backend/graph-database.md)
- [Embeddings](03-components/backend/embeddings.md)
- [Graph Clustering](03-components/backend/clustering.md)
