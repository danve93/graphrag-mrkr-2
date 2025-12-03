# Entity Extraction Flow

Detailed trace of LLM-based entity extraction and graph construction.

## Overview

This document traces the entity extraction process from chunk sampling through LLM prompting, response parsing, entity deduplication, relationship accumulation, and Neo4j persistence. It highlights the EntityGraph accumulator pattern and provenance tracking.

## Flow Diagram

```
Input: 147 chunks from "VxRail_Admin_Guide.pdf"
│
├─> 1. Chunk Sampling
│   ├─ Select representative chunks (50 of 147)
│   ├─ Strategy: Uniform distribution across document
│   ├─ Filter: Skip very short chunks (< 100 chars)
│   └─ Result: 50 candidate chunks for extraction
│
├─> 2. LLM Extraction (Per Chunk)
│   │
│   ├─> 2a. Build Extraction Prompt
│   │   ├─ Include chunk content
│   │   ├─ Specify canonical entity types
│   │   ├─ Request JSON format output
│   │   └─ Examples: (entity, type, description) tuples
│   │
│   ├─> 2b. LLM API Call
│   │   ├─ Model: gpt-4 (or configured LLM)
│   │   ├─ Temperature: 0.3 (low for consistency)
│   │   ├─ Max tokens: 1000
│   │   └─ Concurrency: 3 parallel requests
│   │
│   ├─> 2c. Response Parsing
│   │   ├─ Extract JSON from response
│   │   ├─ Validate entity structure
│   │   ├─ Validate relationship structure
│   │   └─ Handle parse errors gracefully
│   │
│   └─> Repeat for all 50 chunks (~30-40 seconds total)
│
├─> 3. EntityGraph Accumulation
│   │
│   ├─> 3a. Entity Addition
│   │   ├─ For each extracted entity:
│   │   │   ├─ Check if name exists in graph
│   │   │   ├─ If NEW: Add to entities dict
│   │   │   ├─ If EXISTS: Merge descriptions
│   │   │   ├─ Track source chunk (provenance)
│   │   │   └─ Increment mention count
│   │   │
│   │   ├─ Normalize entity names (case, whitespace)
│   │   └─ Calculate importance scores
│   │
│   ├─> 3b. Relationship Addition
│   │   ├─ For each extracted relationship:
│   │   │   ├─ Validate both entities exist
│   │   │   ├─ Check if relationship exists
│   │   │   ├─ If NEW: Add to relationships list
│   │   │   ├─ If EXISTS: Sum strengths
│   │   │   └─ Track source chunk
│   │   │
│   │   ├─ Normalize relationship types
│   │   └─ Bidirectional edges for symmetric relations
│   │
│   └─> 3c. Deduplication
│       ├─ Fuzzy matching on entity names
│       ├─ Merge similar descriptions
│       ├─ Combine provenance lists
│       └─ Update relationship entity references
│
├─> 4. Entity Embedding Generation
│   ├─ Build entity text: "{name}: {description}"
│   ├─ Batch embed all 83 entities
│   ├─ Model: text-embedding-3-small
│   ├─ Check embedding cache (SYNC_ENTITY_EMBEDDINGS)
│   └─ Store embeddings in entity objects
│
├─> 5. Importance Scoring
│   ├─ Mention frequency: count(provenance)
│   ├─ Relationship degree: count(edges)
│   ├─ Description richness: len(description)
│   ├─ Normalize scores to [0, 1]
│   └─ Update entity.importance
│
├─> 6. Neo4j Persistence
│   │
│   ├─> 6a. Entity Nodes (MERGE + UNWIND)
│   │   ├─ Batch: 83 entities
│   │   ├─ MERGE on name (handles duplicates across docs)
│   │   ├─ ON CREATE: Set all properties
│   │   ├─ ON MATCH: Merge descriptions, average importance
│   │   └─ Query time: ~200ms
│   │
│   ├─> 6b. MENTIONS Relationships (CREATE + UNWIND)
│   │   ├─ Batch: 215 chunk-entity connections
│   │   ├─ CREATE (c:Chunk)-[:MENTIONS]->(e:Entity)
│   │   ├─ Properties: None (existence only)
│   │   └─ Query time: ~150ms
│   │
│   └─> 6c. RELATED_TO Relationships (MERGE + UNWIND)
│       ├─ Batch: 142 entity-entity connections
│       ├─ MERGE undirected edges
│       ├─ ON CREATE: Set strength
│       ├─ ON MATCH: Sum strengths
│       ├─ Properties: strength (0.0-1.0)
│       └─ Query time: ~180ms
│
└─> 7. Result
    ├─ 83 unique entities created/updated
    ├─ 215 chunk-entity MENTIONS edges
    ├─ 142 entity-entity RELATED_TO edges
    └─ Total time: ~35 seconds (LLM-bound)
```

## Step-by-Step Trace

### Step 1: Chunk Sampling

**Location**: `core/entity_extraction.py`

```python
def _sample_chunks(
    self,
    chunks: List[Chunk],
    sample_size: int = 50,
) -> List[Chunk]:
    """
    Sample representative chunks for entity extraction.
    
    Args:
        chunks: All document chunks
        sample_size: Target sample size
    
    Returns:
        Sampled chunk list
    """
    # Filter out very short chunks
    candidate_chunks = [
        c for c in chunks
        if len(c.content) >= 100
    ]
    
    if len(candidate_chunks) <= sample_size:
        return candidate_chunks
    
    # Uniform sampling across document
    step = len(candidate_chunks) / sample_size
    sampled_indices = [int(i * step) for i in range(sample_size)]
    
    sampled = [candidate_chunks[i] for i in sampled_indices]
    
    logger.info(f"Sampled {len(sampled)} of {len(chunks)} chunks")
    return sampled
```

**Sample Selection**:
```python
# Original chunks: 147
# Filtered (>= 100 chars): 142
# Sample step: 142 / 50 = 2.84
# Selected indices: [0, 2, 5, 8, 11, ..., 139]
# Result: 50 evenly distributed chunks
```

### Step 2a: Extraction Prompt

**Location**: `core/entity_extraction.py`

```python
def _build_extraction_prompt(self, chunk_content: str) -> str:
    """Build entity extraction prompt."""
    
    entity_types_str = ", ".join(CANONICAL_ENTITY_TYPES)
    
    prompt = f"""Extract entities and relationships from the following text.

Text:
{chunk_content}

Identify entities and classify them using ONLY these types:
{entity_types_str}

Return a JSON object with this exact structure:
{{
  "entities": [
    {{
      "name": "VxRail",
      "type": "Component",
      "description": "Hyper-converged infrastructure appliance"
    }},
    ...
  ],
  "relationships": [
    {{
      "entity1": "VxRail",
      "relation": "RELATED_TO",
      "entity2": "Backup",
      "strength": 0.8
    }},
    ...
  ]
}}

Guidelines:
- Extract 3-10 entities per text
- Use canonical types only
- Relationship strength: 0.0 (weak) to 1.0 (strong)
- Relation types: RELATED_TO, PART_OF, USES, CONFIGURES
"""
    
    return prompt
```

### Step 2b: LLM Extraction

**Location**: `core/entity_extraction.py`

```python
async def _extract_from_chunk(
    self,
    chunk: Chunk,
) -> Tuple[List[Entity], List[Relationship]]:
    """
    Extract entities and relationships from single chunk.
    
    Args:
        chunk: Source chunk
    
    Returns:
        (entities, relationships) tuple
    """
    prompt = self._build_extraction_prompt(chunk.content)
    
    try:
        # LLM call
        response = await self.llm_manager.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000,
        )
        
        # Parse JSON
        data = json.loads(response)
        
        # Build entity objects
        entities = []
        for e_data in data.get("entities", []):
            entity = Entity(
                name=e_data["name"],
                entity_type=e_data["type"],
                description=e_data.get("description", ""),
                source_chunk_id=chunk.chunk_id,
            )
            entities.append(entity)
        
        # Build relationship objects
        relationships = []
        for r_data in data.get("relationships", []):
            rel = Relationship(
                entity1=r_data["entity1"],
                entity2=r_data["entity2"],
                relation_type=r_data.get("relation", "RELATED_TO"),
                strength=r_data.get("strength", 0.5),
                source_chunk_id=chunk.chunk_id,
            )
            relationships.append(rel)
        
        return entities, relationships
    
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse extraction: {e}")
        return [], []
    
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return [], []
```

**LLM Response Example**:
```json
{
  "entities": [
    {
      "name": "VxRail",
      "type": "Component",
      "description": "Hyper-converged infrastructure appliance"
    },
    {
      "name": "Backup",
      "type": "Procedure",
      "description": "Data protection and recovery procedures"
    },
    {
      "name": "Data Protection Suite",
      "type": "Service",
      "description": "Integrated backup and recovery solution"
    },
    {
      "name": "RecoverPoint",
      "type": "Product",
      "description": "Replication and disaster recovery product"
    }
  ],
  "relationships": [
    {
      "entity1": "VxRail",
      "relation": "RELATED_TO",
      "entity2": "Backup",
      "strength": 0.9
    },
    {
      "entity1": "Backup",
      "relation": "USES",
      "entity2": "Data Protection Suite",
      "strength": 0.8
    },
    {
      "entity1": "Data Protection Suite",
      "relation": "PART_OF",
      "entity2": "RecoverPoint",
      "strength": 0.7
    }
  ]
}
```

### Step 3a: Entity Accumulation

**Location**: `core/entity_graph.py`

```python
class EntityGraph:
    """
    Accumulator for entities and relationships with deduplication.
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}  # name → Entity
        self.relationships: List[Relationship] = []
    
    def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        source_chunk_id: str,
    ):
        """
        Add or merge entity.
        
        Args:
            name: Entity name (used as key)
            entity_type: Canonical type
            description: Entity description
            source_chunk_id: Source chunk for provenance
        """
        # Normalize name
        normalized_name = self._normalize_name(name)
        
        if normalized_name in self.entities:
            # Merge with existing
            entity = self.entities[normalized_name]
            
            # Append description
            if description and description not in entity.description:
                entity.description += f"; {description}"
            
            # Track provenance
            if source_chunk_id not in entity.provenance:
                entity.provenance.append(source_chunk_id)
        
        else:
            # Create new
            entity = Entity(
                name=normalized_name,
                entity_type=entity_type,
                description=description,
                provenance=[source_chunk_id],
            )
            self.entities[normalized_name] = entity
        
        logger.debug(f"Entity: {normalized_name} (type: {entity_type}, sources: {len(entity.provenance)})")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for deduplication."""
        # Strip whitespace
        name = name.strip()
        
        # Title case
        name = name.title()
        
        # Remove trailing punctuation
        name = name.rstrip(".,;:")
        
        return name
```

**Entity Accumulation Example**:
```python
# Chunk 1 extracts: "VxRail", "Backup"
entities = {
    "Vxrail": Entity(
        name="Vxrail",
        type="Component",
        description="Hyper-converged infrastructure appliance",
        provenance=["chunk-001"]
    ),
    "Backup": Entity(
        name="Backup",
        type="Procedure",
        description="Data protection procedures",
        provenance=["chunk-001"]
    )
}

# Chunk 15 extracts: "VxRail", "Data Protection"
# After add_entity("VxRail", ...):
entities = {
    "Vxrail": Entity(
        name="Vxrail",
        type="Component",
        description="Hyper-converged infrastructure appliance; integrated compute and storage",
        provenance=["chunk-001", "chunk-015"]  # Merged!
    ),
    "Backup": Entity(...),
    "Data Protection": Entity(
        name="Data Protection",
        type="Concept",
        description="Data backup and recovery strategies",
        provenance=["chunk-015"]
    )
}
```

### Step 3b: Relationship Accumulation

**Location**: `core/entity_graph.py`

```python
def add_relationship(
    self,
    entity1: str,
    entity2: str,
    relation_type: str,
    strength: float,
    source_chunk_id: str,
):
    """
    Add or merge relationship.
    
    Args:
        entity1: Source entity name
        entity2: Target entity name
        relation_type: Relationship type
        strength: Relationship strength (0.0-1.0)
        source_chunk_id: Source chunk
    """
    # Normalize names
    name1 = self._normalize_name(entity1)
    name2 = self._normalize_name(entity2)
    
    # Ensure both entities exist
    if name1 not in self.entities or name2 not in self.entities:
        logger.warning(f"Relationship refers to unknown entity: {name1} or {name2}")
        return
    
    # Check for existing relationship
    for rel in self.relationships:
        if self._match_relationship(rel, name1, name2, relation_type):
            # Merge: sum strengths, add provenance
            rel.strength += strength
            if source_chunk_id not in rel.provenance:
                rel.provenance.append(source_chunk_id)
            return
    
    # Create new relationship
    rel = Relationship(
        entity1=name1,
        entity2=name2,
        relation_type=relation_type,
        strength=strength,
        provenance=[source_chunk_id],
    )
    self.relationships.append(rel)
    
    logger.debug(f"Relationship: {name1} -{relation_type}-> {name2} ({strength})")

def _match_relationship(
    self,
    rel: Relationship,
    entity1: str,
    entity2: str,
    relation_type: str,
) -> bool:
    """Check if relationship matches (bidirectional for symmetric types)."""
    if relation_type in ["RELATED_TO", "SIMILAR_TO"]:
        # Symmetric: match either direction
        return (
            rel.relation_type == relation_type and (
                (rel.entity1 == entity1 and rel.entity2 == entity2) or
                (rel.entity1 == entity2 and rel.entity2 == entity1)
            )
        )
    else:
        # Directional: exact match
        return (
            rel.entity1 == entity1 and
            rel.entity2 == entity2 and
            rel.relation_type == relation_type
        )
```

**Relationship Accumulation Example**:
```python
# Chunk 1: VxRail -[RELATED_TO:0.9]-> Backup
relationships = [
    Relationship(
        entity1="Vxrail",
        entity2="Backup",
        relation_type="RELATED_TO",
        strength=0.9,
        provenance=["chunk-001"]
    )
]

# Chunk 15: Backup -[RELATED_TO:0.8]-> VxRail
# After add_relationship (symmetric match):
relationships = [
    Relationship(
        entity1="Vxrail",
        entity2="Backup",
        relation_type="RELATED_TO",
        strength=1.7,  # 0.9 + 0.8 summed!
        provenance=["chunk-001", "chunk-015"]
    )
]
```

### Step 5: Importance Scoring

**Location**: `core/entity_graph.py`

```python
def calculate_importance(self):
    """
    Calculate importance scores for all entities.
    
    Factors:
    - Mention frequency (provenance count)
    - Relationship degree (edge count)
    - Description richness (character count)
    """
    for entity in self.entities.values():
        # Mention frequency (normalized to [0, 1])
        max_mentions = max(len(e.provenance) for e in self.entities.values())
        mention_score = len(entity.provenance) / max_mentions
        
        # Relationship degree
        degree = sum(
            1 for r in self.relationships
            if r.entity1 == entity.name or r.entity2 == entity.name
        )
        max_degree = max(
            sum(1 for r in self.relationships if r.entity1 == e.name or r.entity2 == e.name)
            for e in self.entities.values()
        )
        degree_score = degree / max_degree if max_degree > 0 else 0
        
        # Description richness
        max_desc_len = max(len(e.description) for e in self.entities.values())
        desc_score = len(entity.description) / max_desc_len
        
        # Weighted combination
        entity.importance = (
            0.5 * mention_score +
            0.3 * degree_score +
            0.2 * desc_score
        )
        
        logger.debug(f"{entity.name}: importance={entity.importance:.2f}")
```

**Importance Scores**:
```python
{
    "Vxrail": 0.95,      # 15 mentions, 12 edges, rich description
    "Backup": 0.82,      # 8 mentions, 10 edges
    "Data Protection": 0.71,  # 5 mentions, 6 edges
    "Recoverpoint": 0.65,     # 3 mentions, 4 edges
    # ... lower scores for less prominent entities
}
```

### Step 6: Neo4j Persistence

**Location**: `core/graph_db.py`

```python
def persist_entity_graph(self, entity_graph: EntityGraph):
    """
    Persist entity graph to Neo4j.
    
    Args:
        entity_graph: EntityGraph with entities and relationships
    """
    with self.driver.session() as session:
        # Batch insert entities
        entity_data = [
            {
                "name": e.name,
                "type": e.entity_type,
                "description": e.description,
                "importance": e.importance,
                "embedding": e.embedding,
            }
            for e in entity_graph.entities.values()
        ]
        
        session.run("""
            UNWIND $entities AS entity
            MERGE (e:Entity {name: entity.name})
            ON CREATE SET
                e.entity_type = entity.type,
                e.description = entity.description,
                e.importance = entity.importance,
                e.embedding = entity.embedding,
                e.created_at = datetime()
            ON MATCH SET
                e.description = CASE
                    WHEN entity.description NOT IN [e.description, '']
                    THEN e.description + '; ' + entity.description
                    ELSE e.description
                END,
                e.importance = (e.importance + entity.importance) / 2,
                e.updated_at = datetime()
        """, entities=entity_data)
        
        logger.info(f"Persisted {len(entity_data)} entities")
        
        # Batch create MENTIONS relationships
        mentions_data = []
        for entity in entity_graph.entities.values():
            for chunk_id in entity.provenance:
                mentions_data.append({
                    "chunk_id": chunk_id,
                    "entity_name": entity.name,
                })
        
        session.run("""
            UNWIND $mentions AS mention
            MATCH (c:Chunk {id: mention.chunk_id})
            MATCH (e:Entity {name: mention.entity_name})
            CREATE (c)-[:MENTIONS]->(e)
        """, mentions=mentions_data)
        
        logger.info(f"Created {len(mentions_data)} MENTIONS relationships")
        
        # Batch create RELATED_TO relationships
        rel_data = [
            {
                "entity1": r.entity1,
                "entity2": r.entity2,
                "relation_type": r.relation_type,
                "strength": r.strength,
            }
            for r in entity_graph.relationships
        ]
        
        session.run("""
            UNWIND $relationships AS rel
            MATCH (e1:Entity {name: rel.entity1})
            MATCH (e2:Entity {name: rel.entity2})
            MERGE (e1)-[r:RELATED_TO]-(e2)
            ON CREATE SET
                r.strength = rel.strength,
                r.relation_type = rel.relation_type
            ON MATCH SET
                r.strength = r.strength + rel.strength
        """, relationships=rel_data)
        
        logger.info(f"Created/updated {len(rel_data)} RELATED_TO relationships")
```

## Performance Notes

### Bottlenecks

1. **LLM API Calls**: 50 chunks × 1-2 seconds/call = ~60-100 seconds total
2. **JSON Parsing**: Negligible (~10ms total)
3. **Neo4j Persistence**: ~500ms for batch operations

### Optimization Strategies

- **Concurrent Extraction**: 3-5 parallel LLM requests
- **Chunk Sampling**: Extract from 50 instead of all 147 chunks
- **UNWIND Batching**: Single query for 83 entities, 142 relationships
- **Embedding Cache**: SYNC mode for deterministic testing

### Total Time

- **Extraction**: 30-40 seconds (LLM-bound)
- **Accumulation**: <1 second (in-memory)
- **Persistence**: <1 second (batch operations)
- **Total**: ~35-45 seconds

## Related Documentation

- [Entity Extraction](03-components/ingestion/entity-extraction.md)
- [Entity Models](02-core-concepts/entity-models.md)
- [Graph Database](03-components/backend/graph-database.md)
- [Document Ingestion Flow](05-data-flows/document-ingestion-flow.md)
