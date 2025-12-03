# Clustering Settings

Configuration for Leiden community detection algorithm.

## Overview

Amber uses the Leiden algorithm to detect communities in the entity relationship graph. Communities group semantically related entities that are strongly connected via `RELATED_TO` and `SIMILAR_TO` relationships.

**Benefits**:
- Visual organization in GraphView (distinct colors per community)
- Semantic grouping of related concepts
- Optional LLM-generated community summaries
- Filtering and navigation by community

---

## Configuration

### Environment Variables

```bash
# Master toggles
ENABLE_CLUSTERING=true
ENABLE_GRAPH_CLUSTERING=true

# Algorithm parameters
CLUSTERING_RESOLUTION=1.0
CLUSTERING_MIN_EDGE_WEIGHT=0.0
CLUSTERING_LEVEL=0
DEFAULT_EDGE_WEIGHT=1.0

# Relationship types (JSON array)
CLUSTERING_RELATIONSHIP_TYPES='["SIMILAR_TO","RELATED_TO"]'
```

### Settings Class

```python
from config.settings import Settings

settings = Settings()

print(f"Clustering enabled: {settings.enable_clustering}")
print(f"Resolution: {settings.clustering_resolution}")
print(f"Min edge weight: {settings.clustering_min_edge_weight}")
print(f"Relationship types: {settings.clustering_relationship_types}")
```

---

## Parameters

### enable_clustering

**Type**: `bool`  
**Default**: `true`

Master toggle for clustering feature. When disabled, clustering jobs won't run and community visualizations are hidden.

```bash
ENABLE_CLUSTERING=true
```

### enable_graph_clustering

**Type**: `bool`  
**Default**: `true`

Controls whether Leiden clustering jobs are executed. Keep separate from `enable_clustering` to allow toggling job execution without hiding UI features.

```bash
ENABLE_GRAPH_CLUSTERING=true
```

### clustering_resolution

**Type**: `float`  
**Default**: `1.0`  
**Range**: `0.1 - 10.0`

Controls community granularity. Higher values create more, smaller communities.

**Guidelines**:
- `0.5` - Fewer, larger communities (broad themes)
- `1.0` - Balanced (default, ~50-100 communities for 1000 entities)
- `2.0` - More, smaller communities (fine-grained topics)
- `5.0+` - Many small communities (very specific groupings)

**Example**:
```bash
# Broad communities
CLUSTERING_RESOLUTION=0.5

# Fine-grained communities
CLUSTERING_RESOLUTION=2.0
```

### clustering_min_edge_weight

**Type**: `float`  
**Default**: `0.0`  
**Range**: `0.0 - 1.0`

Minimum relationship strength to include in clustering projection. Filters out weak connections.

**Guidelines**:
- `0.0` - Include all edges (default)
- `0.3` - Filter weak relationships
- `0.5` - Only strong relationships
- `0.7+` - Very strong relationships only

**Example**:
```bash
# Filter weak edges
CLUSTERING_MIN_EDGE_WEIGHT=0.3
```

**Impact**:
- Higher values → fewer edges → smaller communities
- May result in isolated entities (community_id = null)

### clustering_level

**Type**: `int`  
**Default**: `0`  
**Range**: `0 - 10`

Hierarchy level for multi-level clustering. Currently single-level (0) is used.

```bash
CLUSTERING_LEVEL=0
```

### default_edge_weight

**Type**: `float`  
**Default**: `1.0`

Fallback weight when relationship has no `strength` property.

```bash
DEFAULT_EDGE_WEIGHT=1.0
```

### clustering_relationship_types

**Type**: `List[str]`  
**Default**: `["SIMILAR_TO", "RELATED_TO"]`

Relationship labels to include in clustering projection.

**Available Types**:
- `SIMILAR_TO` - Chunk similarity edges
- `RELATED_TO` - Entity semantic relationships
- `MENTIONS` - Chunk-to-entity edges (not recommended for clustering)

**Example**:
```bash
# Only entity relationships
CLUSTERING_RELATIONSHIP_TYPES='["RELATED_TO"]'

# Include both
CLUSTERING_RELATIONSHIP_TYPES='["SIMILAR_TO","RELATED_TO"]'
```

---

## Running Clustering

### Manual Execution

```bash
# Run clustering script
python scripts/run_clustering.py
```

Script output:
```
Building graph projection...
Projection created: 1247 nodes, 2847 edges
Running Leiden algorithm (resolution=1.0)...
Detected 87 communities
Writing community assignments to Neo4j...
Community detection complete!

Community size distribution:
  Largest: 42 entities
  Smallest: 3 entities
  Average: 14.3 entities
  Median: 11 entities
```

### Via Reindex API

```bash
curl -X POST http://localhost:8000/api/database/reindex \
  -H "Content-Type: application/json" \
  -d '{
    "rebuild_embeddings": false,
    "rebuild_similarities": true,
    "run_clustering": true
  }'
```

### Automatic (Post-Ingestion)

Clustering runs automatically after document ingestion when `ENABLE_CLUSTERING=true`.

---

## Algorithm Details

### Leiden Algorithm

Neo4j implementation of the Leiden community detection algorithm:

1. **Build Projection**: Create in-memory graph with filtered edges
2. **Run Leiden**: Execute algorithm with resolution parameter
3. **Write Communities**: Assign `community_id` property to Entity nodes
4. **Generate Summaries** (optional): LLM descriptions of each community

**Cypher Query** (simplified):
```cypher
// Create projection
CALL gds.graph.project(
  'entity-clustering',
  'Entity',
  {
    SIMILAR_TO: {orientation: 'UNDIRECTED'},
    RELATED_TO: {orientation: 'UNDIRECTED'}
  },
  {relationshipProperties: 'strength'}
)

// Run Leiden
CALL gds.leiden.write(
  'entity-clustering',
  {
    writeProperty: 'community_id',
    relationshipWeightProperty: 'strength',
    includeIntermediateCommunities: false,
    resolution: 1.0
  }
)
```

### Performance

**Runtime**:
- Small (< 500 entities): 5-15 seconds
- Medium (500-2000 entities): 15-60 seconds
- Large (2000+ entities): 1-5 minutes

**Memory**: Runs in Neo4j Graph Data Science (GDS) memory, doesn't impact Python process.

---

## Community Summaries

### Enable Summarization

```bash
ENABLE_CLUSTERING=true
```

### Generate Summaries

```bash
python core/community_summarizer.py
```

**Process**:
1. Load community members (entities + descriptions)
2. Batch communities (groups of 5-10)
3. LLM generates thematic summary for each community
4. Store summaries on Community nodes or as Entity metadata

**Example Summary**:
```
Community 23 (18 members):
"VxRail infrastructure components and backup procedures, including storage
configuration, RAID levels, and disaster recovery workflows."
```

---

## Visualization

Communities are visualized in GraphView with distinct colors:

```typescript
// Color assignment
const communityColors = {
  0: '#FF6B6B',  // Red
  1: '#4ECDC4',  // Teal
  2: '#45B7D1',  // Blue
  3: '#FFA07A',  // Orange
  // ... up to 20 predefined colors
};

function getNodeColor(node: Node): string {
  if (node.type === 'Entity' && node.community_id !== null) {
    return communityColors[node.community_id % 20];
  }
  return defaultColor;
}
```

---

## Tuning Guidelines

### Knowledge Base Size

**Small (< 500 entities)**:
```bash
CLUSTERING_RESOLUTION=0.5
CLUSTERING_MIN_EDGE_WEIGHT=0.0
```

**Medium (500-2000 entities)**:
```bash
CLUSTERING_RESOLUTION=1.0
CLUSTERING_MIN_EDGE_WEIGHT=0.2
```

**Large (2000+ entities)**:
```bash
CLUSTERING_RESOLUTION=1.5
CLUSTERING_MIN_EDGE_WEIGHT=0.3
```

### Domain Type

**Technical Documentation** (precise concepts):
```bash
CLUSTERING_RESOLUTION=1.5
CLUSTERING_MIN_EDGE_WEIGHT=0.4
CLUSTERING_RELATIONSHIP_TYPES='["RELATED_TO"]'
```

**General Knowledge** (broad topics):
```bash
CLUSTERING_RESOLUTION=0.8
CLUSTERING_MIN_EDGE_WEIGHT=0.2
CLUSTERING_RELATIONSHIP_TYPES='["SIMILAR_TO","RELATED_TO"]'
```

**Mixed Content**:
```bash
CLUSTERING_RESOLUTION=1.0
CLUSTERING_MIN_EDGE_WEIGHT=0.3
CLUSTERING_RELATIONSHIP_TYPES='["RELATED_TO"]'
```

---

## Monitoring

### Check Community Stats

```bash
curl http://localhost:8000/api/database/stats | jq '.communities'
```

Response:
```json
{
  "total": 87,
  "largest_size": 42,
  "avg_size": 14.3
}
```

### Query Communities

```cypher
// Community size distribution
MATCH (e:Entity)
WHERE e.community_id IS NOT NULL
RETURN e.community_id, count(*) as size
ORDER BY size DESC
LIMIT 10
```

### Entities Without Communities

```cypher
// Isolated entities
MATCH (e:Entity)
WHERE e.community_id IS NULL
RETURN e.name, e.type
```

**Common reasons**:
- Entity has no relationships
- All relationships below `min_edge_weight`
- Entity created after clustering run

---

## Troubleshooting

### Too Many Communities

**Symptoms**: 200+ communities for 1000 entities, many tiny communities

**Solutions**:
```bash
# Reduce resolution
CLUSTERING_RESOLUTION=0.5

# Increase min edge weight
CLUSTERING_MIN_EDGE_WEIGHT=0.4

# Use only RELATED_TO edges
CLUSTERING_RELATIONSHIP_TYPES='["RELATED_TO"]'
```

### Too Few Communities

**Symptoms**: < 10 communities for 1000 entities, communities too broad

**Solutions**:
```bash
# Increase resolution
CLUSTERING_RESOLUTION=2.0

# Decrease min edge weight
CLUSTERING_MIN_EDGE_WEIGHT=0.1

# Include more edge types
CLUSTERING_RELATIONSHIP_TYPES='["SIMILAR_TO","RELATED_TO"]'
```

### Many Isolated Entities

**Symptoms**: High count of entities with `community_id = null`

**Solutions**:
```bash
# Lower min edge weight
CLUSTERING_MIN_EDGE_WEIGHT=0.0

# Lower resolution
CLUSTERING_RESOLUTION=0.8

# Improve entity extraction to create more relationships
ENABLE_GLEANING=true
MAX_GLEANINGS=2
```

### Clustering Fails

**Error**: "Projection already exists"

**Solution**:
```cypher
// Drop existing projection
CALL gds.graph.drop('entity-clustering', false)
```

**Error**: "Not enough memory"

**Solution**:
- Increase Neo4j heap size (`NEO4J_dbms_memory_heap_max_size`)
- Reduce entity count (increase `IMPORTANCE_SCORE_THRESHOLD`)
- Increase `CLUSTERING_MIN_EDGE_WEIGHT` to reduce edges

---

## Recomputing Communities

### When to Recompute

- After ingesting significant new documents (20%+ of corpus)
- After changing clustering parameters
- After modifying entity relationships manually

### Full Recompute

```bash
# 1. Drop existing projection (if any)
curl -X POST http://localhost:8000/api/database/reindex \
  -H "Content-Type: application/json" \
  -d '{
    "rebuild_similarities": true,
    "run_clustering": true
  }'

# 2. Or via script
python scripts/run_clustering.py --force
```

---

## Related Documentation

- [Environment Variables](07-configuration/environment-variables.md)
- [Community Detection](04-features/community-detection.md)
- [Graph Clustering Module](03-components/backend/graph-clustering.md)
- [Graph Visualization](03-components/frontend/graph-visualization.md)
- [Leiden Algorithm (Neo4j)](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/)
