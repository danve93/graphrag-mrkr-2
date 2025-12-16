# Graph Curation Workbench

The **Graph Curation Workbench** is a suite of tools designed to help administrators explore, repair, and enhance the Knowledge Graph. It combines manual curation interactions with AI-assisted "Healing" to ensure high data quality.

## Features Overview

### 1. Interactive Graph Visualization
We use **Cytoscape.js** to render a high-performance 2D interactive graph.
- **Force-Directed Layout**: Uses `fcose` (Fast Compound Spring Embedder) for clear clustering and structure.
- **Community Coloring**: Nodes are colored based on their detected community (Leiden algorithm).
- **Navigation**: Zoom, Pan, and Fit to view.

### 2. Manual Curation Tools
The toolbar provides several modes for manual intervention:

| Mode | Icon | Description |
| :--- | :--- | :--- |
| **Select** | `Cursor` | View node details. Supports **Shift+Click** for multi-selection. |
| **Connect** | `Link` | Drag from one node to another to create a relationship (`RELATED_TO`). |
| **Prune** | `Scissors` | Click a node or edge to permanently delete it (requires confirmation). |
| **Heal** | `Wand` | Find semantically similar nodes using vector search. |
| **Orphans** | `Ghost` | Highlight disconnected nodes that lack document connections. |

### 3. AI Graph Healing ("Magic Wand")
The "Heal" mode uses vector Similarity Search to find missing connections between existing entities.

#### Ghost Edges
Instead of a crowded list, AI suggestions are visualized directly on the canvas as **Ghost Edges**.
1. Activate **Heal Mode** (Magic Wand icon).
2. Click on a node of interest.
3. The system finds semantically similar nodes (using `entity_embeddings` vector index).
4. **Dashed Amber Lines** appear connecting the node to suggested neighbors.
5. **Click a Ghost Edge** to confirm and create the relationship.

### 4. Advanced Curation: Node Merging
Fix duplicate or fragmented entities (e.g., merging "Elon Musk" and "Elon R. Musk").

1. Switch to **Select Mode**.
2. **Shift+Click** multiple nodes you wish to merge.
3. A **"Merge X Nodes"** button appears at the bottom of the canvas.
4. Click to open the **Merge Modal**.
5. Select the **Primary Node** (the one to keep).
6. Confirm Merge.
    - All edges from source nodes are re-routed to the primary node.
    - Descriptions are concatenated.
    - Source nodes are deleted.

### 5. Inspector Panel (Node Sidebar)
Click any node to open a detailed **Inspector Panel** on the right side:
- **Node Details**: Label, Type, Community ID, Degree
- **Description**: View and **inline edit** the node description
- **Source Documents**: See which documents mention this entity (provenance)
- **Metadata**: View additional properties
- **Actions**: Chat, Edit, Delete buttons

### 6. Focused Chat
Chat with the LLM in the context of a specific entity:
1. Click a node to open the Inspector Panel.
2. Click the **"Chat"** button.
3. A modal opens pre-seeded with the entity context.
4. Ask questions specifically about that entity.

### 7. Orphanage Mode
Find disconnected or poorly-connected entities:
1. Click the **"Orphans"** button in the toolbar (Ghost icon).
2. The system highlights nodes that:
   - Have no relationships, OR
   - Are not connected to any Document within 3 hops
3. Orphan nodes are styled with a **cyan double border**.
4. Click Orphans again to toggle off the highlight.

### 8. Disaster Recovery
Because curation involves destructive actions, safety tools are built-in:

- **Backup**: Download the current graph state (Nodes + Edges) as a JSON snapshot.
- **Restore**: Upload a JSON snapshot to **Wipe & Replace** the current graph. *Warning: This is a destructive action.*

## Technical Architecture

### Backend (`core/graph_db.py`)
- **Vector Index**: Uses `entity_embeddings` (1536d) for finding semantic neighbors.
- **`heal_node()`**: Performs vector search excluding existing neighbors.
- **`merge_nodes()`**: Transactional logic to move edges, merge properties, and delete nodes.
- **`find_orphan_nodes()`**: Detects entities not connected to the main graph component.

### Frontend (`components/Graph/`)
| Component | Purpose |
|-----------|---------|
| `CytoscapeGraph.tsx` | Main graph canvas with all interaction modes |
| `GraphToolbar.tsx` | Toolbar with mode buttons and safety tools |
| `NodeSidebar.tsx` | Inspector panel with details, edit, and actions |
| `FocusedChatPanel.tsx` | Entity-context chat modal |
| `MergeNodesModal.tsx` | Node merge confirmation dialog |
| `RestoreGraphModal.tsx` | Graph restore with confirmation |

## API Endpoints (`api/routers/graph_editor.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/graph/editor/heal` | Get healing suggestions |
| POST | `/api/graph/editor/edge` | Create a new relationship |
| DELETE | `/api/graph/editor/edge` | Remove a relationship |
| PATCH | `/api/graph/editor/node` | Update node properties |
| POST | `/api/graph/editor/nodes/merge` | Merge multiple nodes |
| GET | `/api/graph/editor/snapshot` | Download graph backup |
| POST | `/api/graph/editor/restore` | Restore graph from backup |
| GET | `/api/graph/editor/orphans` | Detect orphan nodes |

## Testing

Unit and integration tests are available:
- `tests/unit/test_graph_curation.py` - 9 tests for graph_db methods
- `tests/integration/test_graph_editor_api.py` - 19 tests for API endpoints
