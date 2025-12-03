# Community Detection Feature

Graph clustering using Leiden algorithm for semantic entity grouping.

## Overview

Community Detection groups related entities into semantic clusters using the Leiden algorithm on the Neo4j knowledge graph. Entities connected by strong relationships (RELATED_TO, SIMILAR_TO) are assigned community IDs, enabling cluster-based filtering, visualization with distinct colors, and optional LLM-generated community summaries.

**Key Capabilities**:
- Leiden clustering with configurable resolution
- Community color palette for visualization
- LLM-generated cluster summaries
- GraphView filtering by community
- Hierarchical clustering levels

## Architecture

```
┌────────────────────────────────────────────────────────┐
│       Community Detection Architecture                  │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Graph Clustering Pipeline              │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ 1. Build Leiden Projection                │  │   │
│  │  │    ├─ Query entity relationships          │  │   │
│  │  │    ├─ Filter by strength threshold        │  │   │
│  │  │    └─ Create Neo4j graph projection       │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │    ↓                                              │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ 2. Run Leiden Algorithm                   │  │   │
│  │  │    ├─ gds.leiden.stream()                 │  │   │
│  │  │    ├─ Apply resolution parameter          │  │   │
│  │  │    └─ Generate community IDs              │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │    ↓                                              │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ 3. Persist Community IDs                  │  │   │
│  │  │    ├─ SET entity.community_id             │  │   │
│  │  │    ├─ Track community sizes               │  │   │
│  │  │    └─ Log statistics                      │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │    ↓                                              │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ 4. Generate Summaries (Optional)          │  │   │
│  │  │    ├─ Query entities per community        │  │   │
│  │  │    ├─ LLM summarization prompt            │  │   │
│  │  │    └─ Store community descriptions        │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          GraphView Integration                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Node Coloring:                            │  │   │
│  │  │   getCommunityColor(community_id)         │  │   │
│  │  │   → Consistent color per cluster          │  │   │
│  │  │                                            │  │   │
│  │  │ Filtering:                                 │  │   │
│  │  │   Show only selected communities           │  │   │
│  │  │   → Interactive cluster exploration        │  │   │
│  │  │                                            │  │   │
│  │  │ Node Detail:                               │  │   │
│  │  │   Display community_id + summary          │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Backend Implementation

### Leiden Clustering

```python
# core/graph_clustering.py
from neo4j import GraphDatabase
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class GraphClusterer:
    """
    Leiden community detection on entity graph.
    """
    
    def __init__(
        self,
        driver: GraphDatabase.driver,
        resolution: float = 1.0,
        min_edge_weight: float = 0.5,
        relationship_types: List[str] = None,
    ):
        """
        Initialize clusterer.
        
        Args:
            driver: Neo4j driver
            resolution: Leiden resolution (higher = more clusters)
            min_edge_weight: Minimum relationship strength
            relationship_types: Edge types to include
        """
        self.driver = driver
        self.resolution = resolution
        self.min_edge_weight = min_edge_weight
        self.relationship_types = relationship_types or [
            "RELATED_TO",
            "SIMILAR_TO",
        ]
    
    def run_clustering(self) -> Dict[str, int]:
        """
        Run Leiden clustering on entity graph.
        
        Returns:
            Dict mapping entity names to community IDs
        """
        with self.driver.session() as session:
            # Step 1: Build projection
            projection_name = self._build_projection(session)
            
            try:
                # Step 2: Run Leiden
                communities = self._run_leiden(session, projection_name)
                
                # Step 3: Persist community IDs
                self._persist_communities(session, communities)
                
                # Log statistics
                self._log_statistics(communities)
                
                return communities
            
            finally:
                # Cleanup projection
                self._drop_projection(session, projection_name)
    
    def _build_projection(self, session, projection_name: str = "leiden_graph") -> str:
        """
        Create Neo4j graph projection for Leiden.
        
        Args:
            session: Neo4j session
            projection_name: Projection identifier
        
        Returns:
            Projection name
        """
        # Drop existing projection
        self._drop_projection(session, projection_name)
        
        # Build relationship filter
        rel_filter = " | ".join(self.relationship_types)
        
        # Create projection
        query = f"""
        CALL gds.graph.project(
            $projection_name,
            'Entity',
            {{
                {rel_filter}: {{
                    orientation: 'UNDIRECTED',
                    properties: 'strength'
                }}
            }}
        )
        """
        
        session.run(query, projection_name=projection_name)
        
        logger.info(f"Built graph projection: {projection_name}")
        return projection_name
    
    def _run_leiden(
        self,
        session,
        projection_name: str,
    ) -> Dict[str, int]:
        """
        Execute Leiden algorithm.
        
        Args:
            session: Neo4j session
            projection_name: Graph projection name
        
        Returns:
            Entity name → community ID mapping
        """
        query = """
        CALL gds.leiden.stream($projection_name, {
            relationshipWeightProperty: 'strength',
            includeIntermediateCommunities: false,
            gamma: $resolution
        })
        YIELD nodeId, communityId
        WITH gds.util.asNode(nodeId) AS entity, communityId
        WHERE entity.strength >= $min_edge_weight
        RETURN entity.name AS name, communityId
        """
        
        result = session.run(
            query,
            projection_name=projection_name,
            resolution=self.resolution,
            min_edge_weight=self.min_edge_weight,
        )
        
        communities = {
            record["name"]: record["communityId"]
            for record in result
        }
        
        logger.info(f"Detected {len(set(communities.values()))} communities")
        return communities
    
    def _persist_communities(
        self,
        session,
        communities: Dict[str, int],
    ):
        """
        Store community IDs on entity nodes.
        
        Args:
            session: Neo4j session
            communities: Entity → community mapping
        """
        query = """
        UNWIND $communities AS item
        MATCH (e:Entity {name: item.name})
        SET e.community_id = item.community_id
        """
        
        community_list = [
            {"name": name, "community_id": cid}
            for name, cid in communities.items()
        ]
        
        session.run(query, communities=community_list)
        logger.info(f"Persisted {len(communities)} community assignments")
    
    def _drop_projection(self, session, projection_name: str):
        """Drop graph projection."""
        try:
            session.run(
                "CALL gds.graph.drop($name)",
                name=projection_name,
            )
        except Exception:
            pass  # Projection may not exist
    
    def _log_statistics(self, communities: Dict[str, int]):
        """Log clustering statistics."""
        from collections import Counter
        
        community_sizes = Counter(communities.values())
        
        logger.info("Community Detection Statistics:")
        logger.info(f"  Total entities: {len(communities)}")
        logger.info(f"  Total communities: {len(community_sizes)}")
        logger.info(f"  Largest community: {max(community_sizes.values())}")
        logger.info(f"  Smallest community: {min(community_sizes.values())}")
        logger.info(f"  Average size: {sum(community_sizes.values()) / len(community_sizes):.1f}")
```

### Community Summarization

```python
# core/community_summarizer.py
from core.llm import LLMManager
from core.graph_db import GraphDBManager
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class CommunitySummarizer:
    """
    Generate LLM summaries for entity communities.
    """
    
    def __init__(
        self,
        graph_db: GraphDBManager,
        llm_manager: LLMManager,
    ):
        self.graph_db = graph_db
        self.llm_manager = llm_manager
    
    async def summarize_communities(self) -> Dict[int, str]:
        """
        Generate summaries for all communities.
        
        Returns:
            Community ID → summary text mapping
        """
        # Get communities
        communities = self._get_communities()
        
        summaries = {}
        for community_id, entities in communities.items():
            summary = await self._summarize_community(
                community_id,
                entities,
            )
            summaries[community_id] = summary
            
            # Store in graph
            self._store_summary(community_id, summary)
        
        logger.info(f"Generated {len(summaries)} community summaries")
        return summaries
    
    def _get_communities(self) -> Dict[int, List[Dict]]:
        """
        Query entities grouped by community.
        
        Returns:
            Community ID → entity list mapping
        """
        query = """
        MATCH (e:Entity)
        WHERE e.community_id IS NOT NULL
        RETURN
            e.community_id AS community_id,
            e.name AS name,
            e.entity_type AS type,
            e.description AS description
        ORDER BY e.community_id, e.importance DESC
        """
        
        records = self.graph_db.execute_read_query(query)
        
        from collections import defaultdict
        communities = defaultdict(list)
        
        for record in records:
            communities[record["community_id"]].append({
                "name": record["name"],
                "type": record["type"],
                "description": record["description"],
            })
        
        return dict(communities)
    
    async def _summarize_community(
        self,
        community_id: int,
        entities: List[Dict],
    ) -> str:
        """
        Generate LLM summary for community.
        
        Args:
            community_id: Community identifier
            entities: List of entity dicts
        
        Returns:
            Summary text
        """
        # Build entity descriptions
        entity_lines = []
        for entity in entities[:20]:  # Limit to top 20
            line = f"- {entity['name']} ({entity['type']})"
            if entity.get("description"):
                line += f": {entity['description']}"
            entity_lines.append(line)
        
        entity_text = "\n".join(entity_lines)
        
        # Build prompt
        prompt = f"""Analyze this group of related entities and provide a concise theme description (1-2 sentences).

Entities:
{entity_text}

Theme:"""
        
        # Generate summary
        response = await self.llm_manager.generate_text(
            prompt=prompt,
            max_tokens=100,
            temperature=0.3,
        )
        
        summary = response.strip()
        logger.debug(f"Community {community_id}: {summary}")
        
        return summary
    
    def _store_summary(self, community_id: int, summary: str):
        """
        Store summary in graph.
        
        Args:
            community_id: Community identifier
            summary: Summary text
        """
        query = """
        MATCH (e:Entity {community_id: $community_id})
        SET e.community_summary = $summary
        """
        
        self.graph_db.execute_write_query(
            query,
            community_id=community_id,
            summary=summary,
        )
```

### Clustering Script

```python
# scripts/run_clustering.py
#!/usr/bin/env python3
"""
Run community detection on entity graph.

Usage:
    python scripts/run_clustering.py [--summarize]
"""
import asyncio
import argparse
from core.singletons import get_neo4j_driver, get_llm_manager
from core.graph_clustering import GraphClusterer
from core.community_summarizer import CommunitySummarizer
from core.graph_db import GraphDBManager
from config.settings import settings

async def main(summarize: bool = False):
    """Run clustering pipeline."""
    driver = get_neo4j_driver()
    
    # Run Leiden clustering
    print("Running Leiden community detection...")
    clusterer = GraphClusterer(
        driver=driver,
        resolution=settings.clustering_resolution,
        min_edge_weight=settings.clustering_min_edge_weight,
        relationship_types=settings.clustering_relationship_types,
    )
    
    communities = clusterer.run_clustering()
    
    print(f"Detected {len(set(communities.values()))} communities")
    print(f"Assigned {len(communities)} entities")
    
    # Generate summaries
    if summarize:
        print("\nGenerating community summaries...")
        
        graph_db = GraphDBManager(driver)
        llm_manager = get_llm_manager()
        
        summarizer = CommunitySummarizer(graph_db, llm_manager)
        summaries = await summarizer.summarize_communities()
        
        print(f"Generated {len(summaries)} summaries")
        
        # Display sample summaries
        for cid, summary in list(summaries.items())[:5]:
            print(f"\nCommunity {cid}: {summary}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generate LLM summaries for communities",
    )
    args = parser.parse_args()
    
    asyncio.run(main(summarize=args.summarize))
```

## Frontend Integration

### Community Colors

```typescript
// frontend/src/lib/community-colors.ts
const COMMUNITY_PALETTE = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
  '#DDA15E', '#BC6C25', '#9C89B8', '#F0A6CA', '#B8F2E6',
  '#FFA07A', '#87CEEB', '#DDA0DD', '#F0E68C', '#98D8C8',
  '#FFB6C1', '#20B2AA', '#FF69B4', '#BA55D3', '#ADFF2F',
];

export function getCommunityColor(communityId: number | null): string {
  if (communityId === null) {
    return '#9CA3AF'; // neutral-400
  }
  
  return COMMUNITY_PALETTE[communityId % COMMUNITY_PALETTE.length];
}
```

### GraphView Node Coloring

```typescript
// frontend/src/components/graph/GraphView.tsx (excerpt)
function GraphView({ data }: GraphViewProps) {
  // ...existing code...
  
  return (
    <ForceGraph3D
      graphData={graphData}
      nodeColor={(node) => getCommunityColor(node.community_id)}
      nodeLabel={(node) => {
        const parts = [
          `Name: ${node.name}`,
          `Type: ${node.type}`,
        ];
        
        if (node.community_id !== null) {
          parts.push(`Community: ${node.community_id}`);
        }
        
        if (node.community_summary) {
          parts.push(`Theme: ${node.community_summary}`);
        }
        
        return parts.join('\n');
      }}
      // ...other props...
    />
  );
}
```

### Community Filter Controls

```typescript
// frontend/src/components/graph/CommunityFilter.tsx
'use client';

import { useState, useMemo } from 'react';
import { Check } from 'lucide-react';
import { getCommunityColor } from '@/lib/community-colors';

interface CommunityFilterProps {
  communities: Array<{ id: number; size: number; summary?: string }>;
  selectedCommunities: Set<number>;
  onSelectionChange: (selected: Set<number>) => void;
}

export function CommunityFilter({
  communities,
  selectedCommunities,
  onSelectionChange,
}: CommunityFilterProps) {
  const toggleCommunity = (id: number) => {
    const newSelection = new Set(selectedCommunities);
    
    if (newSelection.has(id)) {
      newSelection.delete(id);
    } else {
      newSelection.add(id);
    }
    
    onSelectionChange(newSelection);
  };
  
  const selectAll = () => {
    onSelectionChange(new Set(communities.map((c) => c.id)));
  };
  
  const deselectAll = () => {
    onSelectionChange(new Set());
  };
  
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Communities</h3>
        
        <div className="flex gap-2 text-xs">
          <button
            onClick={selectAll}
            className="text-primary-600 hover:text-primary-700"
          >
            All
          </button>
          <button
            onClick={deselectAll}
            className="text-neutral-600 hover:text-neutral-700"
          >
            None
          </button>
        </div>
      </div>
      
      <div className="space-y-1">
        {communities.map((community) => {
          const isSelected = selectedCommunities.has(community.id);
          const color = getCommunityColor(community.id);
          
          return (
            <button
              key={community.id}
              onClick={() => toggleCommunity(community.id)}
              className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left hover:bg-neutral-100 dark:hover:bg-neutral-800"
            >
              <div
                className="h-4 w-4 flex-shrink-0 rounded border-2"
                style={{
                  backgroundColor: isSelected ? color : 'transparent',
                  borderColor: color,
                }}
              >
                {isSelected && (
                  <Check className="h-3 w-3 text-white" strokeWidth={3} />
                )}
              </div>
              
              <div className="flex-1 text-sm">
                <div className="font-medium">Community {community.id}</div>
                {community.summary && (
                  <div className="text-xs text-neutral-500 line-clamp-1">
                    {community.summary}
                  </div>
                )}
              </div>
              
              <div className="text-xs text-neutral-500">
                {community.size} entities
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
```

## Configuration

### Clustering Settings

```python
# config/settings.py
class Settings(BaseSettings):
    # Community detection
    enable_clustering: bool = True
    enable_graph_clustering: bool = True
    
    # Leiden parameters
    clustering_resolution: float = 1.0  # Higher = more clusters
    clustering_min_edge_weight: float = 0.5
    clustering_relationship_types: List[str] = [
        "RELATED_TO",
        "SIMILAR_TO",
    ]
    clustering_level: int = 0  # 0 = final, 1+ = intermediate
    
    # Summarization
    enable_community_summaries: bool = True
    community_summary_max_entities: int = 20
```

## Testing

### Clustering Tests

```python
# tests/test_clustering.py
import pytest
from core.graph_clustering import GraphClusterer
from core.singletons import get_neo4j_driver

@pytest.fixture
def sample_graph(neo4j_driver):
    """Create sample entity graph."""
    with neo4j_driver.session() as session:
        # Create entities
        session.run("""
            CREATE (e1:Entity {name: 'ComponentA', entity_type: 'Component'})
            CREATE (e2:Entity {name: 'ComponentB', entity_type: 'Component'})
            CREATE (e3:Entity {name: 'ServiceX', entity_type: 'Service'})
            CREATE (e4:Entity {name: 'ServiceY', entity_type: 'Service'})
            
            CREATE (e1)-[:RELATED_TO {strength: 0.8}]->(e2)
            CREATE (e3)-[:RELATED_TO {strength: 0.9}]->(e4)
        """)
    
    yield
    
    # Cleanup
    with neo4j_driver.session() as session:
        session.run("MATCH (e:Entity) DETACH DELETE e")

def test_leiden_clustering(neo4j_driver, sample_graph):
    """Test Leiden algorithm execution."""
    clusterer = GraphClusterer(
        driver=neo4j_driver,
        resolution=1.0,
        min_edge_weight=0.5,
    )
    
    communities = clusterer.run_clustering()
    
    # Should detect 2 communities
    unique_communities = set(communities.values())
    assert len(unique_communities) == 2
    
    # ComponentA and ComponentB should be in same community
    assert communities['ComponentA'] == communities['ComponentB']
    
    # ServiceX and ServiceY should be in same community
    assert communities['ServiceX'] == communities['ServiceY']

def test_community_persistence(neo4j_driver, sample_graph):
    """Test community ID persistence."""
    clusterer = GraphClusterer(driver=neo4j_driver)
    clusterer.run_clustering()
    
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.community_id IS NOT NULL
            RETURN count(e) AS count
        """)
        
        count = result.single()["count"]
        assert count == 4  # All entities have community_id
```

## Troubleshooting

### Common Issues

**Issue**: GDS library not available
```cypher
-- Solution: Install Graph Data Science plugin
-- In Neo4j Desktop: Add Graph Data Science plugin
-- In Neo4j AuraDS: Pre-installed

-- Verify installation
RETURN gds.version() AS version;
```

**Issue**: Memory issues with large graphs
```python
# Solution: Batch clustering by subgraph
def cluster_by_subgraph(clusterer, batch_size=1000):
    with clusterer.driver.session() as session:
        # Get entity batches
        result = session.run("""
            MATCH (e:Entity)
            RETURN e.name AS name
            SKIP $offset LIMIT $limit
        """, offset=0, limit=batch_size)
        
        # Cluster each batch...
```

**Issue**: Communities too large/small
```python
# Solution: Adjust resolution parameter
# Higher resolution → more, smaller communities
clusterer = GraphClusterer(resolution=2.0)  # More granular

# Lower resolution → fewer, larger communities
clusterer = GraphClusterer(resolution=0.5)  # More coarse
```

**Issue**: Summarization takes too long
```python
# Solution: Limit entities per community
summarizer = CommunitySummarizer(
    max_entities=10,  # Reduce from 20
)

# Or use async batch processing
summaries = await asyncio.gather(*[
    summarizer._summarize_community(cid, entities)
    for cid, entities in communities.items()
])
```

## Related Documentation

- [Graph Database](03-components/backend/graph-database.md)
- [Entity Extraction](03-components/ingestion/entity-extraction.md)
- [Graph Visualization](03-components/frontend/graph-visualization.md)
- [Database Management](06-api-reference/database.md)
