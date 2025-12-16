# Graph Visualization Component

Interactive 2D knowledge graph visualization using Cytoscape.js.

## Overview

The Graph Visualization component (`CytoscapeGraph`) provides a high-performance, interactive 2D graph for exploring and curating the Knowledge Graph. It supports advanced layouts, community-based coloring, and direct manipulation (creating, deleting, merging nodes/edges).

**Location**: `frontend/src/components/Graph/CytoscapeGraph.tsx`
**Library**: cytoscape.js
**Layout**: fcose (Fast Compound Spring Embedder)
**Features**:
- Force-directed layout with compound node support
- Drag-and-drop edge creation (EdgeHandles)
- Multi-selection (Shift+Click)
- Context-sensitive modes (Select, Connect, Prune, Heal)
- AI-assisted suggestions ("Ghost Edges")

## Architecture

```
┌────────────────────────────────────────────────────────┐
│        Graph Visualization Architecture                 │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          GraphView (Container)                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │  GraphToolbar                             │  │   │
│  │  │  ┌─────────────────────────────────────┐  │  │   │
│  │  │  │ Modes: Select, Connect, Prune, Heal │  │  │   │
│  │  │  └─────────────────────────────────────┘  │  │   │
│  │  │                                           │  │   │
│  │  │  Filters & Controls                       │  │   │
│  │  │  ┌─────────────────────────────────────┐  │  │   │
│  │  │  │ Community Filter                    │  │  │   │
│  │  │  │ Node Type Filter                    │  │  │   │
│  │  │  └─────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │  CytoscapeGraph (Canvas)                  │  │   │
│  │  │  ┌─────────────────────────────────────┐  │  │   │
│  │  │  │ Nodes (Documents/Chunks/Entities)   │  │  │   │
│  │  │  │ Edges (Relationships)               │  │  │   │
│  │  │  │ Ghost Edges (AI Suggestions)        │  │  │   │
│  │  │  └─────────────────────────────────────┘  │  │   │
│  │  │                                           │  │   │
│  │  │  Interactions                             │  │   │
│  │  │  ├─ Tap/Click (Select/Heal)            │  │   │
│  │  │  ├─ Drag (Pan/Move Node)               │  │   │
│  │  │  ├─ EdgeHandle Drag (Connect)          │  │   │
│  │  │  └─ Shift+Click (Multi-select)         │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │  Modals / Overlays                        │  │   │
│  │  │  ├─ ConfirmActionModal                    │  │   │
│  │  │  ├─ MergeNodesModal                       │  │   │
│  │  │  └─ RestoreGraphModal                     │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

## Core Components

### CytoscapeGraph (The Canvas)

The core component that initializes the Cytoscape instance and manages event listeners.

```typescript
// frontend/src/components/Graph/CytoscapeGraph.tsx

export default function CytoscapeGraph({ nodes, edges, mode, ...props }) {
    // ...
    
    // Initialize Cytoscape
    useEffect(() => {
        const cy = cytoscape({
            container: containerRef.current,
            elements: [...nodes, ...edges],
            style: stylesheet,
            layout: { name: 'fcose', ... }
        });
        
        // Register EdgeHandles extension
        cy.edgehandles({...});
        
        // Event Listeners
        cy.on('tap', 'node', handleNodeTap);
        cy.on('tap', 'edge', handleEdgeTap);
        
        return () => cy.destroy();
    }, []);
    
    // ...
}
```

### GraphToolbar (The Controls)

Manages the valid interaction modes and exposes actions like "Undo" or "Backup".

```typescript
// frontend/src/components/Graph/GraphToolbar.tsx

export const GraphToolbar = () => {
    const { mode, setMode } = useGraphEditorStore();
    
    return (
        <div className="toolbar">
            <button onClick={() => setMode('select')} active={mode === 'select'}>
                <MousePointer2 />
            </button>
            <button onClick={() => setMode('connect')} active={mode === 'connect'}>
                <Link />
            </button>
            <button onClick={() => setMode('heal')} active={mode === 'heal'}>
                <Wand2 />
            </button>
            <button onClick={() => setMode('prune')} active={mode === 'prune'}>
                <Scissors />
            </button>
        </div>
    );
}
```

## Interaction Modes

1.  **Select (`select`)**: Default mode. Click nodes to view details in the **NodeSidebar**. Shift+Click to select multiple nodes for Batch Actions (like Merge).
2.  **Connect (`connect`)**: Drag from the "handle" of a source node to a target node to create a relationship.
3.  **Heal (`heal`)**: Click a node to trigger AI Graph Healing. Suggestions appear as dashed "Ghost Edges". Click a ghost edge to accept it.
4.  **Prune (`prune`)**: Click any node or edge to delete it.
5.  **Orphan (`orphan`)**: Toggle to highlight disconnected nodes with a cyan border. Useful for finding entities that need connections.

## NodeSidebar (Inspector Panel)

When a node is selected, the `NodeSidebar` slides in from the right:

| Section | Description |
|---------|-------------|
| Header | Node label, type badge, and ID |
| Actions | Chat, Edit, Delete buttons |
| Description | Inline editable text field |
| Stats | Community ID and Degree |
| Source Documents | Provenance - which docs mention this entity |
| Metadata | Additional properties |

### FocusedChatPanel

Click "Chat" in NodeSidebar to open a modal pre-seeded with the entity context. Uses the existing `/api/chat/query` endpoint with node information.

## Styling

We use a Cytoscape stylesheet to define the visual appearance.

```typescript
const stylesheet = [
    {
        selector: 'node',
        style: {
            'background-color': '#6366f1',
            'label': 'data(label)',
            'color': '#fff',
            // ...
        }
    },
    {
        selector: '.ghost-edge',
        style: {
            'line-style': 'dashed',
            'line-color': '#fbbf24', // Amber
            'target-arrow-shape': 'triangle',
            // ...
        }
    }
]
```

## Performance

- **Layout**: `fcose` is used for its speed and quality in handling compound graphs.
- **Memoization**: `nodes` and `edges` are memoized to prevent re-running the layout unnecessarily.
- **Ghost Edges**: Only generated for visible nodes to keep the graph size manageable.

## Related Documentation

- [Graph Editor API](../api-reference/graph-editor.md)
- [Graph Curation Workbench](../../04-features/graph-curation-workbench.md)
