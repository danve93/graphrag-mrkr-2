# Graph Clustering Component

Leiden community detection for entity clustering and semantic grouping.

## Overview

The clustering component uses the Leiden algorithm to detect communities of related entities in the knowledge graph. Entities connected by strong relationships are grouped into semantic clusters (communities) with distinct identifiers and optional LLM-generated summaries. Clusters are visualized with unique colors in the frontend and can be filtered independently.

**Location**: `core/graph_clustering.py`, `core/community_summarizer.py`
**Algorithm**: Leiden community detection (via NetworkX or igraph)
**Persistence**: `community_id` property on Entity nodes
**Visualization**: Color-coded communities in GraphView

## Architecture

```
┌──────────────────────────────────────────────────┐
│        Clustering Pipeline                        │
├──────────────────────────────────────────────────┤
│                                                   │
│  Step 1: Graph Projection                        │
│  ┌─────────────────────────────────────────────┐ │
│  │  Neo4j → NetworkX/igraph                    │ │
│  │  • Extract Entity nodes                     │ │
│  │  • Extract RELATED_TO edges                 │ │
│  │  • Filter by min_edge_weight threshold      │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Step 2: Community Detection                     │
│  ┌─────────────────────────────────────────────┐ │
│  │  Leiden Algorithm                            │ │
│  │  • Resolution parameter (granularity)       │ │
│  │  • Modularity optimization                  │ │
│  │  • Hierarchical partitioning                │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Step 3: Assignment & Persistence               │
│  ┌─────────────────────────────────────────────┐ │
│  │  Assign community_id to entities            │ │
│  │  → Update Neo4j Entity.community_id         │ │
│  │  → Create Community metadata nodes          │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Step 4: Summarization (Optional)               │
│  ┌─────────────────────────────────────────────┐ │
│  │  LLM generates community descriptions       │ │
│  │  → Update Community.summary                 │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
└──────────────────────────────────────────────────┘
```

## Leiden Algorithm

### Overview

The Leiden algorithm is a community detection method that:
- Optimizes modularity (density of connections within communities vs between)
- Guarantees well-connected communities
- Scales to large graphs (millions of nodes)
- Provides hierarchical clustering at multiple resolutions

### Resolution Parameter

```python
# Low resolution (0.5-1.0): Fewer, larger communities
CLUSTERING_RESOLUTION=0.8

# Medium resolution (1.0-2.0): Balanced clustering
CLUSTERING_RESOLUTION=1.5

# High resolution (2.0-5.0): Many small communities
CLUSTERING_RESOLUTION=3.0
```

**Effect**:
- Higher resolution → more granular communities (specialized topics)
- Lower resolution → broader communities (general themes)

## Configuration

### Environment Variables

```bash
# Enable clustering
ENABLE_CLUSTERING=true
ENABLE_GRAPH_CLUSTERING=true

# Algorithm parameters
CLUSTERING_RESOLUTION=1.5           # Granularity (0.5-5.0)
CLUSTERING_MIN_EDGE_WEIGHT=0.3      # Filter weak edges
CLUSTERING_RELATIONSHIP_TYPES=RELATED_TO,SIMILAR_TO
CLUSTERING_LEVEL=0                  # Hierarchy level (0=finest)

# Community summarization
ENABLE_COMMUNITY_SUMMARIES=true
COMMUNITY_SUMMARY_MAX_ENTITIES=20   # Entities per summary
```

### Settings

```python
from config.settings import settings

# Check if clustering is enabled
if settings.enable_clustering:
    run_clustering(document_id)

# Get clustering parameters
resolution = settings.clustering_resolution
min_weight = settings.clustering_min_edge_weight
rel_types = settings.clustering_relationship_types  # ["RELATED_TO", "SIMILAR_TO"]
```

## Graph Projection

### Extract Graph from Neo4j

```python
import networkx as nx
from core.graph_db import get_db

def build_entity_graph(
    document_id: Optional[str] = None,
    min_edge_weight: float = 0.3
) -> nx.Graph:
    """
    Build NetworkX graph from Neo4j entities and relationships.
    
    Args:
        document_id: Filter by document (None = all entities)
        min_edge_weight: Minimum relationship strength threshold
    
    Returns:
        NetworkX Graph with entity nodes and relationship edges
    """
    db = get_db()
    
    # Query entities
    if document_id:
        entity_query = """
        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
        RETURN DISTINCT e.id as id,
               e.name as name,
               e.type as type,
               e.importance as importance
        """
        params = {"document_id": document_id}
    else:
        entity_query = """
        MATCH (e:Entity)
        RETURN e.id as id,
               e.name as name,
               e.type as type,
               e.importance as importance
        """
        params = {}
    
    entities = db.execute_read(entity_query, params)
    
    # Query relationships
    if document_id:
        rel_query = """
        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:CONTAINS_ENTITY]->(e1:Entity)
        MATCH (e1)-[r:RELATED_TO]-(e2:Entity)
        WHERE r.strength >= $min_weight
        RETURN DISTINCT e1.id as source,
               e2.id as target,
               r.strength as weight
        """
        params = {"document_id": document_id, "min_weight": min_edge_weight}
    else:
        rel_query = """
        MATCH (e1:Entity)-[r:RELATED_TO]-(e2:Entity)
        WHERE r.strength >= $min_weight
        RETURN e1.id as source,
               e2.id as target,
               r.strength as weight
        """
        params = {"min_weight": min_edge_weight}
    
    relationships = db.execute_read(rel_query, params)
    
    # Build NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for entity in entities:
        G.add_node(
            entity["id"],
            name=entity["name"],
            type=entity["type"],
            importance=entity["importance"]
        )
    
    # Add edges
    for rel in relationships:
        G.add_edge(
            rel["source"],
            rel["target"],
            weight=rel["weight"]
        )
    
    logger.info(
        f"Built graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    
    return G
```

### Filter by Relationship Types

```python
def build_entity_graph_multi_rel(
    relationship_types: List[str] = ["RELATED_TO", "SIMILAR_TO"],
    min_edge_weight: float = 0.3
) -> nx.Graph:
    """Build graph with multiple relationship types."""
    db = get_db()
    
    # Build relationship type filter
    rel_types_str = ", ".join([f"'{rt}'" for rt in relationship_types])
    
    query = f"""
    MATCH (e1:Entity)-[r]-(e2:Entity)
    WHERE type(r) IN [{rel_types_str}]
      AND r.strength >= $min_weight
    RETURN e1.id as source,
           e2.id as target,
           r.strength as weight,
           type(r) as rel_type
    """
    
    relationships = db.execute_read(query, {"min_weight": min_edge_weight})
    
    # Build graph
    G = nx.Graph()
    for rel in relationships:
        G.add_edge(
            rel["source"],
            rel["target"],
            weight=rel["weight"],
            rel_type=rel["rel_type"]
        )
    
    return G
```

## Leiden Clustering

### Core Algorithm

```python
import networkx as nx
from networkx.algorithms import community as nx_community

def detect_communities_leiden(
    graph: nx.Graph,
    resolution: float = 1.5,
    level: int = 0
) -> Dict[str, int]:
    """
    Detect communities using Leiden algorithm.
    
    Args:
        graph: NetworkX graph
        resolution: Resolution parameter (higher = more communities)
        level: Hierarchy level (0 = finest granularity)
    
    Returns:
        Dict mapping node_id -> community_id
    """
    if graph.number_of_nodes() == 0:
        return {}
    
    # Run Leiden algorithm (via louvain as approximation)
    # Note: For true Leiden, use python-igraph with leidenalg
    communities = nx_community.louvain_communities(
        graph,
        resolution=resolution,
        weight='weight',
        seed=42  # Reproducibility
    )
    
    # Convert to dict
    node_to_community = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            node_to_community[node] = community_id
    
    logger.info(f"Detected {len(communities)} communities")
    
    return node_to_community
```

### Using igraph (True Leiden)

```python
import igraph as ig
import leidenalg

def detect_communities_leiden_igraph(
    graph: nx.Graph,
    resolution: float = 1.5
) -> Dict[str, int]:
    """
    Detect communities using true Leiden algorithm via igraph.
    
    Requires: pip install python-igraph leidenalg
    """
    # Convert NetworkX to igraph
    edges = [(u, v) for u, v in graph.edges()]
    weights = [graph[u][v].get('weight', 1.0) for u, v in edges]
    
    ig_graph = ig.Graph(edges=edges)
    ig_graph.es['weight'] = weights
    
    # Run Leiden
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights='weight'
    )
    
    # Convert to dict
    node_to_community = {}
    node_ids = list(graph.nodes())
    
    for node_idx, community_id in enumerate(partition.membership):
        node_id = node_ids[node_idx]
        node_to_community[node_id] = community_id
    
    logger.info(
        f"Leiden detected {len(partition)} communities "
        f"(modularity: {partition.modularity:.3f})"
    )
    
    return node_to_community
```

## Persistence to Neo4j

### Update Entity Community IDs

```python
def assign_communities_to_entities(node_to_community: Dict[str, int]):
    """Assign community_id to Entity nodes."""
    db = get_db()
    
    # Prepare batch update
    updates = [
        {"entity_id": entity_id, "community_id": community_id}
        for entity_id, community_id in node_to_community.items()
    ]
    
    query = """
    UNWIND $updates AS update
    MATCH (e:Entity {id: update.entity_id})
    SET e.community_id = update.community_id
    """
    
    # Process in batches
    batch_size = 1000
    for i in range(0, len(updates), batch_size):
        batch = updates[i:i + batch_size]
        db.execute_write(query, {"updates": batch})
    
    logger.info(f"Assigned community IDs to {len(updates)} entities")
```

### Create Community Metadata Nodes

```python
def create_community_nodes(
    graph: nx.Graph,
    node_to_community: Dict[str, int]
):
    """Create Community metadata nodes in Neo4j."""
    db = get_db()
    
    # Calculate community stats
    community_stats = {}
    for node_id, community_id in node_to_community.items():
        if community_id not in community_stats:
            community_stats[community_id] = {
                "id": community_id,
                "entity_count": 0,
                "total_importance": 0.0,
                "entity_types": set()
            }
        
        stats = community_stats[community_id]
        stats["entity_count"] += 1
        stats["total_importance"] += graph.nodes[node_id].get("importance", 0.5)
        stats["entity_types"].add(graph.nodes[node_id].get("type", "Unknown"))
    
    # Convert to list
    communities = [
        {
            "id": cid,
            "entity_count": stats["entity_count"],
            "avg_importance": stats["total_importance"] / stats["entity_count"],
            "entity_types": list(stats["entity_types"])
        }
        for cid, stats in community_stats.items()
    ]
    
    # Create nodes
    query = """
    UNWIND $communities AS comm
    MERGE (c:Community {id: comm.id})
    SET c.entity_count = comm.entity_count,
        c.avg_importance = comm.avg_importance,
        c.entity_types = comm.entity_types,
        c.created_at = datetime()
    """
    
    db.execute_write(query, {"communities": communities})
    
    logger.info(f"Created {len(communities)} Community nodes")
```

## Community Summarization

### LLM-Based Summarization

```python
from core.llm import LLMManager

async def generate_community_summary(
    community_id: int,
    max_entities: int = 20
) -> str:
    """
    Generate LLM summary for a community.
    
    Args:
        community_id: Community identifier
        max_entities: Max entities to include in prompt
    
    Returns:
        Natural language summary of community theme
    """
    db = get_db()
    
    # Get community entities
    query = """
    MATCH (e:Entity {community_id: $community_id})
    RETURN e.name as name,
           e.type as type,
           e.description as description,
           e.importance as importance
    ORDER BY e.importance DESC
    LIMIT $max_entities
    """
    
    entities = db.execute_read(query, {
        "community_id": community_id,
        "max_entities": max_entities
    })
    
    if not entities:
        return "Empty community"
    
    # Build prompt
    entity_list = "\n".join([
        f"- {e['name']} ({e['type']}): {e['description']}"
        for e in entities
    ])
    
    prompt = f"""Analyze this group of related entities and provide a concise summary of their common theme or domain.

ENTITIES:
{entity_list}

Provide a 1-2 sentence summary describing what these entities have in common and their overall theme.
Focus on the primary domain, topic, or relationship that unifies them."""
    
    # Generate summary
    llm_manager = LLMManager()
    summary = await llm_manager.generate(
        prompt=prompt,
        temperature=0.3,
        max_tokens=150
    )
    
    return summary.strip()
```

### Batch Summarization

```python
async def summarize_all_communities(
    max_entities_per_community: int = 20
):
    """Generate summaries for all communities."""
    db = get_db()
    
    # Get all community IDs
    query = "MATCH (c:Community) RETURN c.id as id ORDER BY c.id"
    communities = db.execute_read(query)
    
    # Generate summaries
    for comm in communities:
        community_id = comm["id"]
        
        try:
            summary = await generate_community_summary(
                community_id=community_id,
                max_entities=max_entities_per_community
            )
            
            # Update Community node
            update_query = """
            MATCH (c:Community {id: $community_id})
            SET c.summary = $summary,
                c.summarized_at = datetime()
            """
            
            db.execute_write(update_query, {
                "community_id": community_id,
                "summary": summary
            })
            
            logger.info(f"Generated summary for community {community_id}")
        
        except Exception as e:
            logger.error(f"Failed to summarize community {community_id}: {e}")
```

## Full Clustering Pipeline

### Complete Workflow

```python
async def run_clustering_pipeline(
    document_id: Optional[str] = None,
    resolution: float = 1.5,
    min_edge_weight: float = 0.3,
    generate_summaries: bool = True
):
    """
    Execute complete clustering pipeline.
    
    Args:
        document_id: Filter by document (None = global clustering)
        resolution: Leiden resolution parameter
        min_edge_weight: Minimum edge weight threshold
        generate_summaries: Generate LLM summaries for communities
    """
    logger.info("Starting clustering pipeline")
    
    # Step 1: Build graph
    graph = build_entity_graph(
        document_id=document_id,
        min_edge_weight=min_edge_weight
    )
    
    if graph.number_of_nodes() == 0:
        logger.warning("No entities found for clustering")
        return
    
    # Step 2: Detect communities
    node_to_community = detect_communities_leiden(
        graph=graph,
        resolution=resolution
    )
    
    # Step 3: Persist assignments
    assign_communities_to_entities(node_to_community)
    create_community_nodes(graph, node_to_community)
    
    # Step 4: Generate summaries (optional)
    if generate_summaries and settings.enable_community_summaries:
        await summarize_all_communities()
    
    # Step 5: Update document stats
    if document_id:
        db = get_db()
        db.update_document_stats(document_id)
    
    logger.info("Clustering pipeline complete")
```

### CLI Script

```python
# scripts/run_clustering.py
import asyncio
from core.graph_clustering import run_clustering_pipeline
from config.settings import settings

async def main():
    """Run clustering on all entities."""
    await run_clustering_pipeline(
        document_id=None,  # Global clustering
        resolution=settings.clustering_resolution,
        min_edge_weight=settings.clustering_min_edge_weight,
        generate_summaries=True
    )

if __name__ == "__main__":
    asyncio.run(main())
```

**Usage**:
```bash
python scripts/run_clustering.py
```

## Query Communities

### Get Community Details

```python
def get_community_details(community_id: int) -> Dict:
    """Get community metadata and member entities."""
    db = get_db()
    
    query = """
    MATCH (c:Community {id: $community_id})
    OPTIONAL MATCH (e:Entity {community_id: $community_id})
    RETURN c.id as id,
           c.entity_count as entity_count,
           c.avg_importance as avg_importance,
           c.entity_types as entity_types,
           c.summary as summary,
           collect(e.name) as entity_names
    """
    
    results = db.execute_read(query, {"community_id": community_id})
    return results[0] if results else None
```

### Get Document Communities

```python
def get_document_communities(document_id: str) -> List[Dict]:
    """Get all communities in a document."""
    db = get_db()
    
    query = """
    MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
    MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
    WHERE e.community_id IS NOT NULL
    WITH DISTINCT e.community_id as community_id
    MATCH (comm:Community {id: community_id})
    RETURN comm.id as id,
           comm.entity_count as entity_count,
           comm.summary as summary
    ORDER BY comm.entity_count DESC
    """
    
    return db.execute_read(query, {"document_id": document_id})
```

## Visualization Integration

### Community Colors

```python
# Frontend: Generate color palette for communities
def generate_community_colors(num_communities: int) -> List[str]:
    """Generate distinct colors for communities."""
    import colorsys
    
    colors = []
    for i in range(num_communities):
        hue = i / num_communities
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors
```

### Community Filtering

```python
# API endpoint for filtering by community
@router.get("/documents/{document_id}/communities/{community_id}/entities")
def get_community_entities(
    document_id: str,
    community_id: int,
    limit: int = 50,
    offset: int = 0
):
    """Get entities in a specific community."""
    db = get_db()
    
    query = """
    MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
    MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity {community_id: $community_id})
    RETURN DISTINCT e.id as entity_id,
           e.name as name,
           e.type as type,
           e.importance as importance
    ORDER BY e.importance DESC
    SKIP $offset
    LIMIT $limit
    """
    
    return db.execute_read(query, {
        "document_id": document_id,
        "community_id": community_id,
        "limit": limit,
        "offset": offset
    })
```

## Performance Optimization

### Incremental Clustering

```python
def incremental_clustering(new_entity_ids: List[str]):
    """Add new entities to existing clusters."""
    # Option 1: Rerun full clustering
    # Option 2: Assign to nearest community based on connections
    
    db = get_db()
    
    for entity_id in new_entity_ids:
        # Find connected entities with community_id
        query = """
        MATCH (e:Entity {id: $entity_id})-[r:RELATED_TO]-(neighbor:Entity)
        WHERE neighbor.community_id IS NOT NULL
        WITH neighbor.community_id as comm_id, sum(r.strength) as total_strength
        ORDER BY total_strength DESC
        LIMIT 1
        RETURN comm_id
        """
        
        results = db.execute_read(query, {"entity_id": entity_id})
        
        if results:
            community_id = results[0]["comm_id"]
            
            # Assign to community
            update_query = """
            MATCH (e:Entity {id: $entity_id})
            SET e.community_id = $community_id
            """
            
            db.execute_write(update_query, {
                "entity_id": entity_id,
                "community_id": community_id
            })
```

## Testing

### Unit Tests

```python
import pytest
import networkx as nx
from core.graph_clustering import detect_communities_leiden

def test_leiden_clustering():
    # Create test graph
    G = nx.Graph()
    
    # Community 1
    G.add_edges_from([(1, 2), (2, 3), (3, 1)], weight=0.9)
    
    # Community 2
    G.add_edges_from([(4, 5), (5, 6), (6, 4)], weight=0.9)
    
    # Weak inter-community edge
    G.add_edge(3, 4, weight=0.2)
    
    # Run clustering
    node_to_community = detect_communities_leiden(G, resolution=1.0)
    
    # Verify two communities
    communities = set(node_to_community.values())
    assert len(communities) == 2
    
    # Verify community coherence
    comm1 = node_to_community[1]
    assert node_to_community[2] == comm1
    assert node_to_community[3] == comm1
    
    comm2 = node_to_community[4]
    assert node_to_community[5] == comm2
    assert node_to_community[6] == comm2
```

## Troubleshooting

### Common Issues

**Issue**: All entities in one community
```python
# Solution: Increase resolution
CLUSTERING_RESOLUTION=3.0

# Or increase min_edge_weight to filter weak edges
CLUSTERING_MIN_EDGE_WEIGHT=0.5
```

**Issue**: Too many small communities
```python
# Solution: Decrease resolution
CLUSTERING_RESOLUTION=0.8

# Or decrease min_edge_weight to include more edges
CLUSTERING_MIN_EDGE_WEIGHT=0.2
```

**Issue**: No communities detected
```python
# Solution: Check if entities have relationships
query = """
MATCH (e:Entity)-[r:RELATED_TO]-()
RETURN count(DISTINCT e) as connected_entities
"""

# Ensure min_edge_weight doesn't filter all edges
```

## Related Documentation

- [Entity Extraction](03-components/backend/entity-extraction.md)
- [Graph Database](03-components/backend/graph-database.md)
- [Graph Visualization](03-components/frontend/graph-visualization.md)
- [Community Detection Implementation](../../../COMMUNITY_DETECTION_IMPLEMENTATION.md)
