# Graph Visualization Component

Interactive 3D knowledge graph visualization using Force-Graph.

## Overview

The Graph Visualization component provides an interactive 3D force-directed graph for exploring the knowledge graph structure. It visualizes documents, chunks, entities, and their relationships with community-based coloring, interactive navigation, and detail panels.

**Location**: `frontend/src/components/graph/`
**Library**: react-force-graph-3d (based on three.js)
**Features**: 3D navigation, node clustering, community colors, interactive tooltips

## Architecture

```
┌────────────────────────────────────────────────────────┐
│        Graph Visualization Architecture                 │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          GraphView Component                    │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │  Control Panel                            │  │   │
│  │  │  ┌─────────────────────────────────────┐  │  │   │
│  │  │  │ Node Type Filters                   │  │  │   │
│  │  │  │   ├─ Show Documents                 │  │  │   │
│  │  │  │   ├─ Show Chunks                    │  │  │   │
│  │  │  │   └─ Show Entities                  │  │  │   │
│  │  │  │                                       │  │  │   │
│  │  │  │ Layout Controls                      │  │  │   │
│  │  │  │   ├─ Force strength slider          │  │  │   │
│  │  │  │   ├─ Distance slider                │  │  │   │
│  │  │  │   └─ Reset view button              │  │  │   │
│  │  │  │                                       │  │  │   │
│  │  │  │ Community Filter                     │  │  │   │
│  │  │  │   └─ Community dropdown             │  │  │   │
│  │  │  └─────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │  ForceGraph3D Canvas                      │  │   │
│  │  │  ┌─────────────────────────────────────┐  │  │   │
│  │  │  │ Nodes (Documents/Chunks/Entities)   │  │  │   │
│  │  │  │   ├─ Color by community_id          │  │  │   │
│  │  │  │   ├─ Size by degree/importance      │  │  │   │
│  │  │  │   └─ Labels on hover                │  │  │   │
│  │  │  │                                       │  │  │   │
│  │  │  │ Links (Relationships)                │  │  │   │
│  │  │  │   ├─ SIMILAR_TO (chunk similarity)  │  │  │   │
│  │  │  │   ├─ RELATED_TO (entity relations)  │  │  │   │
│  │  │  │   ├─ HAS_CHUNK (doc→chunk)          │  │  │   │
│  │  │  │   └─ MENTIONS (chunk→entity)        │  │  │   │
│  │  │  └─────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │  NodeDetail Panel (on click)              │  │   │
│  │  │  ┌─────────────────────────────────────┐  │  │   │
│  │  │  │ Node properties                     │  │  │   │
│  │  │  │ Connected nodes list                │  │  │   │
│  │  │  │ Expand/collapse button              │  │  │   │
│  │  │  └─────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Data Layer                             │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Fetch graph data from /api/database/graph │  │   │
│  │  │ Transform to ForceGraph format            │  │   │
│  │  │ Apply community colors                    │  │   │
│  │  │ Calculate node sizes                      │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Core Components

### GraphView (Main Container)

```typescript
// frontend/src/components/graph/GraphView.tsx
'use client';

import { useRef, useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { useQuery } from '@tanstack/react-query';
import { Loader2 } from 'lucide-react';
import { GraphControls } from './GraphControls';
import { NodeDetail } from './NodeDetail';
import { getGraphData } from '@/lib/api-client';
import { transformGraphData, getCommunityColor } from '@/lib/graph-utils';
import type { GraphData, GraphNode, GraphLink } from '@/types';

// Dynamic import for client-side only rendering
const ForceGraph3D = dynamic(
  () => import('react-force-graph-3d').then((mod) => mod.default),
  { ssr: false }
);

export function GraphView() {
  const graphRef = useRef<any>();
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [filters, setFilters] = useState({
    showDocuments: true,
    showChunks: true,
    showEntities: true,
    communityId: null as string | null,
  });

  // Fetch graph data
  const { data: rawData, isLoading, error } = useQuery({
    queryKey: ['graph-data'],
    queryFn: getGraphData,
  });

  // Transform and filter data
  const graphData = useCallback(() => {
    if (!rawData) return { nodes: [], links: [] };
    
    const transformed = transformGraphData(rawData);
    
    // Apply filters
    const filteredNodes = transformed.nodes.filter((node) => {
      if (!filters.showDocuments && node.type === 'Document') return false;
      if (!filters.showChunks && node.type === 'Chunk') return false;
      if (!filters.showEntities && node.type === 'Entity') return false;
      if (filters.communityId && node.community_id !== filters.communityId) {
        return false;
      }
      return true;
    });
    
    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredLinks = transformed.links.filter(
      (link) => nodeIds.has(link.source) && nodeIds.has(link.target)
    );
    
    return { nodes: filteredNodes, links: filteredLinks };
  }, [rawData, filters]);

  // Handle node click
  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
    
    // Center camera on node
    if (graphRef.current) {
      const distance = 300;
      graphRef.current.cameraPosition(
        { x: node.x, y: node.y, z: node.z + distance },
        node,
        1000
      );
    }
  }, []);

  // Reset camera
  const handleResetView = useCallback(() => {
    if (graphRef.current) {
      graphRef.current.zoomToFit(1000);
    }
  }, []);

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center text-error-500">
        Failed to load graph data
      </div>
    );
  }

  const data = graphData();

  return (
    <div className="relative h-full w-full">
      {/* Controls */}
      <GraphControls
        filters={filters}
        onFiltersChange={setFilters}
        onResetView={handleResetView}
        nodeCount={data.nodes.length}
        linkCount={data.links.length}
      />

      {/* Graph canvas */}
      <ForceGraph3D
        ref={graphRef}
        graphData={data}
        nodeLabel={(node: any) => node.name || node.id}
        nodeColor={(node: any) => getCommunityColor(node.community_id)}
        nodeVal={(node: any) => {
          // Size based on type and degree
          const baseSize = node.type === 'Entity' ? 5 : 3;
          const degree = node.degree || 1;
          return baseSize + Math.log(degree);
        }}
        nodeOpacity={0.9}
        linkColor={() => 'rgba(255, 255, 255, 0.2)'}
        linkWidth={(link: any) => {
          // Width based on relationship strength
          return link.strength ? link.strength * 2 : 1;
        }}
        linkOpacity={0.5}
        linkDirectionalParticles={2}
        linkDirectionalParticleSpeed={0.005}
        onNodeClick={handleNodeClick}
        onNodeHover={(node) => {
          document.body.style.cursor = node ? 'pointer' : 'default';
        }}
        enableNodeDrag={true}
        enableNavigationControls={true}
        showNavInfo={false}
        backgroundColor="rgba(0, 0, 0, 0.9)"
      />

      {/* Node detail panel */}
      {selectedNode && (
        <NodeDetail
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  );
}
```

### GraphControls Component

```typescript
// frontend/src/components/graph/GraphControls.tsx
import { Settings, RotateCcw, Filter } from 'lucide-react';

interface GraphControlsProps {
  filters: {
    showDocuments: boolean;
    showChunks: boolean;
    showEntities: boolean;
    communityId: string | null;
  };
  onFiltersChange: (filters: any) => void;
  onResetView: () => void;
  nodeCount: number;
  linkCount: number;
}

export function GraphControls({
  filters,
  onFiltersChange,
  onResetView,
  nodeCount,
  linkCount,
}: GraphControlsProps) {
  return (
    <div className="absolute left-4 top-4 z-10 space-y-2">
      {/* Stats card */}
      <div className="rounded-lg border border-neutral-700 bg-neutral-900/90 p-3 text-sm backdrop-blur">
        <div className="mb-1 flex items-center gap-2 text-neutral-400">
          <Settings className="h-4 w-4" />
          <span className="font-semibold">Graph Stats</span>
        </div>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between gap-4">
            <span className="text-neutral-500">Nodes:</span>
            <span className="font-mono text-white">{nodeCount}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-neutral-500">Links:</span>
            <span className="font-mono text-white">{linkCount}</span>
          </div>
        </div>
      </div>

      {/* Filter card */}
      <div className="rounded-lg border border-neutral-700 bg-neutral-900/90 p-3 backdrop-blur">
        <div className="mb-2 flex items-center gap-2 text-sm text-neutral-400">
          <Filter className="h-4 w-4" />
          <span className="font-semibold">Filters</span>
        </div>
        
        <div className="space-y-2">
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={filters.showDocuments}
              onChange={(e) =>
                onFiltersChange({ ...filters, showDocuments: e.target.checked })
              }
              className="rounded border-neutral-600 bg-neutral-800"
            />
            <span className="text-xs text-neutral-300">Documents</span>
          </label>
          
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={filters.showChunks}
              onChange={(e) =>
                onFiltersChange({ ...filters, showChunks: e.target.checked })
              }
              className="rounded border-neutral-600 bg-neutral-800"
            />
            <span className="text-xs text-neutral-300">Chunks</span>
          </label>
          
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={filters.showEntities}
              onChange={(e) =>
                onFiltersChange({ ...filters, showEntities: e.target.checked })
              }
              className="rounded border-neutral-600 bg-neutral-800"
            />
            <span className="text-xs text-neutral-300">Entities</span>
          </label>
        </div>
      </div>

      {/* Reset button */}
      <button
        onClick={onResetView}
        className="flex w-full items-center justify-center gap-2 rounded-lg border border-neutral-700 bg-neutral-900/90 px-3 py-2 text-sm text-neutral-300 backdrop-blur hover:bg-neutral-800"
      >
        <RotateCcw className="h-4 w-4" />
        Reset View
      </button>
    </div>
  );
}
```

### NodeDetail Component

```typescript
// frontend/src/components/graph/NodeDetail.tsx
import { X, FileText, Layers, Tag } from 'lucide-react';
import type { GraphNode } from '@/types';

interface NodeDetailProps {
  node: GraphNode;
  onClose: () => void;
}

export function NodeDetail({ node, onClose }: NodeDetailProps) {
  const getIcon = () => {
    switch (node.type) {
      case 'Document':
        return <FileText className="h-5 w-5" />;
      case 'Chunk':
        return <Layers className="h-5 w-5" />;
      case 'Entity':
        return <Tag className="h-5 w-5" />;
      default:
        return null;
    }
  };

  return (
    <div className="absolute right-4 top-4 z-10 w-80 rounded-lg border border-neutral-700 bg-neutral-900/95 p-4 backdrop-blur">
      {/* Header */}
      <div className="mb-3 flex items-start justify-between">
        <div className="flex items-center gap-2">
          <div className="rounded bg-neutral-800 p-2 text-primary-400">
            {getIcon()}
          </div>
          <div>
            <h3 className="font-semibold text-white">{node.name || node.id}</h3>
            <p className="text-xs text-neutral-500">{node.type}</p>
          </div>
        </div>
        
        <button
          onClick={onClose}
          className="rounded p-1 hover:bg-neutral-800"
        >
          <X className="h-4 w-4 text-neutral-400" />
        </button>
      </div>

      {/* Properties */}
      <div className="space-y-3 text-sm">
        {node.community_id && (
          <div>
            <span className="text-neutral-500">Community:</span>
            <div
              className="mt-1 inline-block rounded-full px-2 py-1 text-xs"
              style={{
                backgroundColor: `${node.color}20`,
                color: node.color,
              }}
            >
              {node.community_id}
            </div>
          </div>
        )}
        
        {node.degree && (
          <div>
            <span className="text-neutral-500">Connections:</span>
            <span className="ml-2 font-mono text-white">{node.degree}</span>
          </div>
        )}
        
        {node.description && (
          <div>
            <span className="text-neutral-500">Description:</span>
            <p className="mt-1 text-xs text-neutral-300">{node.description}</p>
          </div>
        )}
        
        {node.metadata && Object.keys(node.metadata).length > 0 && (
          <div>
            <span className="text-neutral-500">Metadata:</span>
            <div className="mt-1 space-y-1">
              {Object.entries(node.metadata).map(([key, value]) => (
                <div key={key} className="flex justify-between text-xs">
                  <span className="text-neutral-400">{key}:</span>
                  <span className="text-neutral-300">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
```

## Data Transformation

### Graph Data Utils

```typescript
// frontend/src/lib/graph-utils.ts
import type { GraphData, GraphNode, GraphLink } from '@/types';

export function transformGraphData(rawData: any): GraphData {
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  const nodeMap = new Map<string, GraphNode>();

  // Process nodes
  for (const node of rawData.nodes || []) {
    const graphNode: GraphNode = {
      id: node.id,
      name: node.name || node.id,
      type: node.labels?.[0] || 'Unknown',
      community_id: node.community_id,
      degree: node.degree || 0,
      color: getCommunityColor(node.community_id),
      description: node.description,
      metadata: node.properties || {},
    };
    
    nodes.push(graphNode);
    nodeMap.set(node.id, graphNode);
  }

  // Process links
  for (const link of rawData.links || []) {
    const graphLink: GraphLink = {
      source: link.source,
      target: link.target,
      type: link.type,
      strength: link.strength || 1,
    };
    
    links.push(graphLink);
  }

  return { nodes, links };
}

// Color palette for communities
const COMMUNITY_COLORS = [
  '#3B82F6', // blue
  '#10B981', // green
  '#F59E0B', // amber
  '#EF4444', // red
  '#8B5CF6', // purple
  '#EC4899', // pink
  '#14B8A6', // teal
  '#F97316', // orange
  '#6366F1', // indigo
  '#84CC16', // lime
];

export function getCommunityColor(communityId?: string | null): string {
  if (!communityId) return '#6B7280'; // neutral gray
  
  // Hash community ID to color index
  let hash = 0;
  for (let i = 0; i < communityId.length; i++) {
    hash = communityId.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  const index = Math.abs(hash) % COMMUNITY_COLORS.length;
  return COMMUNITY_COLORS[index];
}

export function calculateNodeSize(node: GraphNode): number {
  const baseSize = node.type === 'Entity' ? 5 : 3;
  const degree = node.degree || 1;
  return baseSize + Math.log(degree);
}
```

## API Integration

### Fetch Graph Data

```typescript
// frontend/src/lib/api-client.ts
export interface RawGraphData {
  nodes: Array<{
    id: string;
    name?: string;
    labels: string[];
    community_id?: string;
    degree?: number;
    description?: string;
    properties?: Record<string, any>;
  }>;
  links: Array<{
    source: string;
    target: string;
    type: string;
    strength?: number;
  }>;
}

export async function getGraphData(): Promise<RawGraphData> {
  const response = await fetch('/api/database/graph');
  if (!response.ok) throw new Error('Failed to fetch graph data');
  return response.json();
}

export async function getNodeNeighbors(nodeId: string): Promise<RawGraphData> {
  const response = await fetch(`/api/database/graph/neighbors/${nodeId}`);
  if (!response.ok) throw new Error('Failed to fetch neighbors');
  return response.json();
}
```

## Type Definitions

```typescript
// frontend/src/types/index.ts
export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface GraphNode {
  id: string;
  name: string;
  type: 'Document' | 'Chunk' | 'Entity';
  community_id?: string;
  degree?: number;
  color?: string;
  description?: string;
  metadata?: Record<string, any>;
  x?: number;
  y?: number;
  z?: number;
}

export interface GraphLink {
  source: string;
  target: string;
  type: string;
  strength?: number;
}
```

## Styling & Layout

### Force Simulation Parameters

```typescript
// Adjust physics parameters
<ForceGraph3D
  d3AlphaDecay={0.02}        // Simulation cooldown rate
  d3VelocityDecay={0.3}       // Friction
  d3Force={{
    charge: { strength: -120 }, // Repulsion between nodes
    link: { distance: 50 },     // Link length
    center: { strength: 0.1 },  // Centering force
  }}
/>
```

## Performance Optimization

### Large Graph Handling

```typescript
// Limit visible nodes for performance
const MAX_NODES = 1000;

const limitedData = useMemo(() => {
  if (graphData.nodes.length <= MAX_NODES) {
    return graphData;
  }
  
  // Sort by degree and take top nodes
  const topNodes = [...graphData.nodes]
    .sort((a, b) => (b.degree || 0) - (a.degree || 0))
    .slice(0, MAX_NODES);
  
  const nodeIds = new Set(topNodes.map((n) => n.id));
  const filteredLinks = graphData.links.filter(
    (l) => nodeIds.has(l.source) && nodeIds.has(l.target)
  );
  
  return { nodes: topNodes, links: filteredLinks };
}, [graphData]);
```

### Debounced Updates

```typescript
import { useMemo } from 'react';
import debounce from 'lodash/debounce';

const debouncedUpdate = useMemo(
  () =>
    debounce((data: GraphData) => {
      // Update graph
    }, 300),
  []
);
```

## Testing

### Component Tests

```typescript
// frontend/src/components/graph/__tests__/GraphView.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { GraphView } from '../GraphView';
import * as apiClient from '@/lib/api-client';

jest.mock('@/lib/api-client');
jest.mock('react-force-graph-3d', () => ({
  __esModule: true,
  default: ({ graphData }: any) => (
    <div data-testid="force-graph">
      Nodes: {graphData.nodes.length}, Links: {graphData.links.length}
    </div>
  ),
}));

describe('GraphView', () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  const mockGraphData = {
    nodes: [
      { id: '1', name: 'Entity1', labels: ['Entity'], degree: 5 },
      { id: '2', name: 'Chunk1', labels: ['Chunk'], degree: 3 },
    ],
    links: [
      { source: '1', target: '2', type: 'MENTIONS', strength: 1 },
    ],
  };

  beforeEach(() => {
    jest.mocked(apiClient.getGraphData).mockResolvedValue(mockGraphData);
  });

  it('renders graph with nodes and links', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <GraphView />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('force-graph')).toHaveTextContent('Nodes: 2');
      expect(screen.getByTestId('force-graph')).toHaveTextContent('Links: 1');
    });
  });
});
```

## Troubleshooting

### Common Issues

**Issue**: Graph not rendering
```typescript
// Check if ForceGraph is loaded
console.log('ForceGraph component:', ForceGraph3D);

// Verify data format
console.log('Graph data:', graphData);
console.log('Nodes:', graphData.nodes.length);
console.log('Links:', graphData.links.length);
```

**Issue**: Poor performance with large graphs
```typescript
// Reduce node count
const MAX_NODES = 500;

// Disable particle animations
<ForceGraph3D
  linkDirectionalParticles={0}
  enablePointerInteraction={false} // Disable hover
/>
```

**Issue**: Colors not showing
```typescript
// Verify community_id exists
console.log('Node community:', node.community_id);

// Check color function
console.log('Color:', getCommunityColor(node.community_id));
```

## Related Documentation

- [Graph Database Component](03-components/backend/graph-database.md)
- [Community Detection](03-components/backend/clustering.md)
- [Database API](06-api-reference/database.md)
- [Entity Extraction](03-components/backend/entity-extraction.md)
