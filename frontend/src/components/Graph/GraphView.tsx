'use client'

import dynamic from 'next/dynamic'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Info, Network, Box, Grid2X2 } from 'lucide-react'
import { Button } from '@mui/material'
import ExpandablePanel from '@/components/Utils/ExpandablePanel'
import Loader from '@/components/Utils/Loader'

import { api } from '@/lib/api'
import type { GraphCommunity, GraphEdge, GraphNode } from '@/types/graph'

import { GraphToolbar } from './GraphToolbar'
import NodeSidebar from './NodeSidebar'
import FocusedChatPanel from './FocusedChatPanel'


// Dynamic import for the new Cytoscape component to avoid SSR issues
const CytoscapeGraph = dynamic(() => import('./CytoscapeGraph'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center">
      <Loader size={28} label="Loading graph..." />
    </div>
  ),
})

// Dynamic import for 3D graph (heavier, only load when needed)
const ThreeGraph = dynamic(() => import('./ThreeGraph'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center bg-[var(--bg-primary)]">
      <Loader size={28} label="Loading 3D visualization..." />
    </div>
  ),
})

export default function GraphView() {
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] }>({
    nodes: [],
    edges: [],
  })
  const [communities, setCommunities] = useState<GraphCommunity[]>([])
  const [nodeTypes, setNodeTypes] = useState<string[]>([])
  const [selectedCommunity, setSelectedCommunity] = useState<string>('all')
  const [selectedNodeType, setSelectedNodeType] = useState<string>('all')
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [chatNode, setChatNode] = useState<GraphNode | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Ref for container sizing
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [dimensions, setDimensions] = useState({ width: 600, height: 600 })
  const [expanded, setExpanded] = useState(false)
  const [show3DView, setShow3DView] = useState(false)

  const [expandedPanels, setExpandedPanels] = useState<Set<string>>(new Set(['filters']))

  const togglePanel = (panel: string) => {
    setExpandedPanels((prev) => {
      const next = new Set(prev);
      if (next.has(panel)) {
        next.delete(panel);
      } else {
        next.add(panel);
      }
      return next;
    });
  };

  const fetchGraph = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await api.getGraph({
        community_id: selectedCommunity !== 'all' ? Number(selectedCommunity) : undefined,
        node_type: selectedNodeType !== 'all' ? selectedNodeType : undefined,
        limit: 100,
      })
      setGraphData({ nodes: response.nodes, edges: response.edges })
      setCommunities(response.communities)
      setNodeTypes(response.node_types)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load graph data'
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }, [selectedCommunity, selectedNodeType])

  useEffect(() => {
    void fetchGraph()
  }, [fetchGraph])

  // Resize observer to keep the canvas within parent bounds
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect()
      // Subtract header/padding roughly
      const availableHeight = typeof window !== 'undefined' ? Math.floor(window.innerHeight - rect.top - 32) : rect.height
      const height = Math.max(200, Math.min(Math.floor(rect.height), availableHeight))
      setDimensions({ width: Math.max(200, Math.floor(rect.width)), height })
    })
    ro.observe(el)

    // Initial size
    const r = el.getBoundingClientRect()
    const available = typeof window !== 'undefined' ? Math.floor(window.innerHeight - r.top - 32) : r.height
    setDimensions({ width: Math.max(200, Math.floor(r.width)), height: Math.max(200, Math.min(Math.floor(r.height), available)) })

    return () => ro.disconnect()
  }, [containerRef])

  // Resolve CSS variable values
  const backgroundColorResolved = useMemo(() => {
    if (typeof window === 'undefined') return '#0f172a' // fallback to dark
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue('--bg-primary')
      if (!v) return '#0f172a'
      return v.trim()
    } catch (e) {
      return '#0f172a'
    }
  }, [])

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)]">
      {/* Header */}
      <div className="border-b border-[var(--border)] p-[var(--space-6)]">
        <div className="flex items-center gap-3 mb-[var(--space-2)]">
          <div className="w-10 h-10 rounded-lg bg-accent-primary/10 border border-accent-primary flex items-center justify-center">
            <Network size={24} className="text-accent-primary" />
          </div>
          <div className="flex-1">
            <h1 className="font-display text-2xl font-bold text-[var(--heading-text)]">
              Graph Explorer
            </h1>
            <p className="text-sm text-[var(--text-secondary)]">
              Interactive 2D layout with filtering and analysis tools
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)] px-3 py-1.5 rounded-md bg-[var(--bg-secondary)]">
            <Info className="h-4 w-4" />
            <span>Scroll to zoom · Drag to pan</span>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto p-[var(--space-6)] flex flex-col gap-3 relative">

        {/* Filters Panel - Refactored for density */}
        <ExpandablePanel
          title="Filters & Controls"
          expanded={expandedPanels.has('filters')}
          onToggle={() => togglePanel('filters')}
        >
          <div className="grid grid-cols-1 md:grid-cols-[1fr_1fr_auto] gap-4 items-end p-2">
            <div className="flex flex-col gap-1.5 min-w-[150px]">
              <label className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">Community</label>
              <select
                value={selectedCommunity}
                onChange={(event) => setSelectedCommunity(event.target.value)}
                className="input-field py-1.5 text-sm"
              >
                <option value="all">All communities</option>
                {communities.map((community) => (
                  <option key={`${community.community_id}-${community.level}`} value={community.community_id}>
                    Community {community.community_id}
                    {community.level !== undefined ? ` (level ${community.level})` : ''}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1.5 min-w-[150px]">
              <label className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">Node type</label>
              <select
                value={selectedNodeType}
                onChange={(event) => setSelectedNodeType(event.target.value)}
                className="input-field py-1.5 text-sm"
              >
                <option value="all">All types</option>
                {nodeTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex gap-2 text-xs font-mono text-[var(--text-secondary)]">
                <span className="bg-[var(--bg-tertiary)] px-2 py-1 rounded">
                  {graphData.nodes.length} nodes
                </span>
                <span className="bg-[var(--bg-tertiary)] px-2 py-1 rounded">
                  {graphData.edges.length} edges
                </span>
              </div>
              <button
                onClick={() => void fetchGraph()}
                className="button-primary text-xs py-1.5 px-3 h-[34px]"
              >
                Refresh
              </button>
            </div>
          </div>
        </ExpandablePanel>

        {/* Graph Visualization */}
        <div
          ref={containerRef}
          className="flex-1 relative overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]"
          style={{ minHeight: '400px' }}
        >
          {isLoading && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-[var(--bg-secondary)]/80 backdrop-blur-sm">
              <Loader size={28} label="Loading graph…" />
            </div>
          )}
          {error && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/50">
              <div className="rounded-lg bg-red-950/90 border border-red-900 p-4 text-sm text-red-200 shadow-xl max-w-md text-center">
                {error}
              </div>
            </div>
          )}

          <CytoscapeGraph
            nodes={graphData.nodes}
            edges={graphData.edges}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor={backgroundColorResolved}
            onGraphUpdate={fetchGraph}
            onNodeClick={setSelectedNode}
          />

          <GraphToolbar onFit={() => { /* Future: Fit */ }} />

          {/* Expand button */}
          <button
            aria-label="Open graph fullscreen"
            onClick={() => setExpanded(true)}
            className="absolute top-4 right-4 z-10 small-button bg-[var(--bg-secondary)] border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] shadow-sm"
          >
            Expand
          </button>

          {/* Node Sidebar */}
          {selectedNode && (
            <NodeSidebar
              node={selectedNode}
              onClose={() => setSelectedNode(null)}
              onEdit={(node) => console.log('Edit node', node)} // TODO: Implement edit
              onDelete={(nodeId) => console.log('Delete node', nodeId)} // TODO: Implement delete
              onChat={(node) => setChatNode(node)}
            />
          )}
        </div>
      </div>

      {/* Fullscreen modal */}
      {expanded && (
        <div className="fixed inset-0 z-[100] bg-[var(--bg-primary)] flex flex-col">
          {/* Header for Fullscreen */}
          <div className="h-14 border-b border-[var(--border)] flex items-center justify-between px-6 bg-[var(--bg-secondary)]">
            <div className="flex items-center gap-2">
              <Network size={20} className="text-accent-primary" />
              <h2 className="font-display font-bold text-[var(--heading-text)]">Graph Explorer</h2>
              <span className="text-xs px-2 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
                {show3DView ? '3D View' : 'Fullscreen'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {/* 3D/2D Toggle Button */}
              <button
                onClick={() => setShow3DView(!show3DView)}
                className={`small-button flex items-center gap-2 ${show3DView
                    ? 'bg-[var(--accent-primary)] text-white border-[var(--accent-primary)]'
                    : 'bg-[var(--bg-tertiary)] hover:bg-[var(--bg-primary)] border-[var(--border)]'
                  }`}
              >
                {show3DView ? (
                  <>
                    <Grid2X2 size={16} />
                    <span>2D View</span>
                  </>
                ) : (
                  <>
                    <Box size={16} />
                    <span>3D View</span>
                  </>
                )}
              </button>
              <button
                onClick={() => {
                  setExpanded(false)
                  setShow3DView(false)
                }}
                className="small-button bg-[var(--bg-tertiary)] hover:bg-[var(--bg-primary)] border-[var(--border)]"
              >
                Exit Fullscreen
              </button>
            </div>
          </div>

          <div className="flex-1 relative w-full h-full overflow-hidden">
            {show3DView ? (
              <ThreeGraph
                nodes={graphData.nodes}
                edges={graphData.edges}
                onNodeClick={setSelectedNode}
                gestureEnabled={true}
              />
            ) : (
              <CytoscapeGraph
                nodes={graphData.nodes}
                edges={graphData.edges}
                backgroundColor={backgroundColorResolved}
                onGraphUpdate={fetchGraph}
                onNodeClick={setSelectedNode}
              />
            )}

            {/* Toolbar in Fullscreen (2D only) */}
            {!show3DView && <GraphToolbar onFit={() => { }} />}

            {/* Node Sidebar in Fullscreen */}
            {selectedNode && (
              <NodeSidebar
                node={selectedNode}
                onClose={() => setSelectedNode(null)}
                onEdit={(node) => console.log('Edit node', node)}
                onDelete={(nodeId) => console.log('Delete node', nodeId)}
                onChat={(node) => setChatNode(node)}
              />
            )}

            {/* Focused Chat Panel needs to be here too or at root with high Z-index */}
          </div>
        </div>
      )}

      {/* Focused Chat Panel - Rendered at root level with high z-index to overlay fullscreen or normal mode */}
      {chatNode && (
        <FocusedChatPanel
          node={chatNode}
          onClose={() => setChatNode(null)}
        />
      )}
    </div>
  )
}
