'use client'

import dynamic from 'next/dynamic'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Info, Network, Search } from 'lucide-react'
import ExpandablePanel from '@/components/Utils/ExpandablePanel'
import Loader from '@/components/Utils/Loader'

import { api } from '@/lib/api'
import type { GraphCommunity, GraphEdge, GraphNode } from '@/types/graph'

import { GraphToolbar } from './GraphToolbar'
import NodeSidebar from './NodeSidebar'
import FocusedChatPanel from './FocusedChatPanel'
import { HealingSuggestionsModal } from './HealingSuggestionsModal'
import { useGraphEditorStore } from './useGraphEditorStore'


// Dynamic import for the new Cytoscape component to avoid SSR issues
const CytoscapeGraph = dynamic(() => import('./CytoscapeGraph'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center">
      <Loader size={28} label="Loading graph..." />
    </div>
  ),
})

export default function GraphView() {
  const { mode } = useGraphEditorStore()
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
  const [expandedPanels, setExpandedPanels] = useState<Set<string>>(new Set())
  const [searchTerm, setSearchTerm] = useState('')
  const handleNodeEdit = useCallback((updatedNode: GraphNode) => {
    setGraphData((prev) => ({
      ...prev,
      nodes: prev.nodes.map((node) => (node.id === updatedNode.id ? updatedNode : node)),
    }))
    setSelectedNode(updatedNode)
  }, [])

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
        limit: 500,
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

  const normalizedSearch = useMemo(() => searchTerm.trim().toLowerCase(), [searchTerm])

  const filteredGraphData = useMemo(() => {
    if (!normalizedSearch) {
      return graphData
    }

    const matchesSearch = (node: GraphNode) => {
      const label = node.label?.toLowerCase() ?? ''
      const id = node.id?.toLowerCase() ?? ''
      const type = node.type?.toLowerCase() ?? ''
      const description = node.description?.toLowerCase() ?? ''
      return (
        label.includes(normalizedSearch) ||
        id.includes(normalizedSearch) ||
        type.includes(normalizedSearch) ||
        description.includes(normalizedSearch)
      )
    }

    const nodes = graphData.nodes.filter(matchesSearch)
    const allowedIds = new Set(nodes.map((node) => node.id))
    const edges = graphData.edges.filter((edge) => allowedIds.has(edge.source) && allowedIds.has(edge.target))
    return { nodes, edges }
  }, [graphData, normalizedSearch])

  useEffect(() => {
    if (!selectedNode) return
    if (!filteredGraphData.nodes.some((node) => node.id === selectedNode.id)) {
      setSelectedNode(null)
    }
  }, [filteredGraphData.nodes, selectedNode])

  const isSearchActive = normalizedSearch.length > 0
  const countLabel = isSearchActive
    ? `${filteredGraphData.nodes.length} of ${graphData.nodes.length} nodes · ${filteredGraphData.edges.length} edges`
    : `${graphData.nodes.length} nodes · ${graphData.edges.length} edges`

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
              Graph Editor
            </h1>
            <p className="text-sm text-[var(--text-secondary)]">
              Curate entities and relationships that power retrieval and reasoning.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)] px-3 py-1.5 rounded-md bg-[var(--bg-secondary)]">
            <Info className="h-4 w-4" />
            <span>Drag to pan · Scroll to zoom · Click a node to inspect</span>
          </div>
        </div>
        <div className="mt-3 flex items-start gap-2 rounded-md border border-[var(--systemOrange)]/40 bg-[var(--systemOrange)]/10 px-3 py-2 text-xs">
          <AlertTriangle className="mt-0.5 h-4 w-4 text-[var(--systemOrange)]" />
          <div className="text-[var(--text-primary)]">
            <p className="font-semibold">Curation mode: edits are applied directly to the graph.</p>
            <p className="text-[var(--text-secondary)]">Use Backup before large changes. Prune is destructive and cannot be undone.</p>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto p-[var(--space-6)] pb-28 flex flex-col gap-3 relative">

        <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] p-3">
          <label className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
            Search entities
          </label>
          <div className="mt-2 flex flex-col gap-2 md:flex-row md:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--text-secondary)]" />
              <input
                value={searchTerm}
                onChange={(event) => setSearchTerm(event.target.value)}
                placeholder="Search by name, id, type, or description"
                className="input-field w-full py-2 pl-9 text-sm"
              />
            </div>
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="button-secondary text-xs py-2 px-3"
              >
                Clear
              </button>
            )}
          </div>
          <div className="mt-2 text-xs text-[var(--text-secondary)]">
            {countLabel}
          </div>
        </div>

        {/* Filters Panel - Refactored for density */}
        <ExpandablePanel
          title="Scope & Filters"
          expanded={expandedPanels.has('filters')}
          onToggle={() => togglePanel('filters')}
        >
          <p className="px-2 pb-2 text-xs text-[var(--text-secondary)]">
            Narrow the view before editing to reduce the blast radius of changes.
          </p>
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
                  {filteredGraphData.nodes.length} nodes
                </span>
                <span className="bg-[var(--bg-tertiary)] px-2 py-1 rounded">
                  {filteredGraphData.edges.length} edges
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
            nodes={filteredGraphData.nodes}
            edges={filteredGraphData.edges}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor={backgroundColorResolved}
            onGraphUpdate={fetchGraph}
            onNodeClick={(node) => setSelectedNode(node)}
            editable
          />

          {/* Graph Toolbar - visible in both modes */}
          <GraphToolbar />

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
              onEdit={handleNodeEdit}
              onChat={(node) => setChatNode(node)}
              allNodes={filteredGraphData.nodes}
              allEdges={filteredGraphData.edges}
              onNodeSelect={setSelectedNode}
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
              <h2 className="font-display font-bold text-[var(--heading-text)]">Graph Editor</h2>
              <span className="text-xs px-2 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
                Fullscreen
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  setExpanded(false)
                }}
                className="small-button bg-[var(--bg-tertiary)] hover:bg-[var(--bg-primary)] border-[var(--border)]"
              >
                Exit Fullscreen
              </button>
            </div>
          </div>

          <div className="flex-1 relative w-full h-full overflow-hidden">
            <CytoscapeGraph
              nodes={filteredGraphData.nodes}
              edges={filteredGraphData.edges}
              backgroundColor={backgroundColorResolved}
              onGraphUpdate={fetchGraph}
              onNodeClick={(node) => setSelectedNode(node)}
              editable
            />

            {/* Toolbar in Fullscreen */}
            <GraphToolbar />

            {/* Node Sidebar in Fullscreen */}
            {selectedNode && (
              <NodeSidebar
                node={selectedNode}
                onClose={() => setSelectedNode(null)}
                onEdit={handleNodeEdit}
                onChat={(node) => setChatNode(node)}
                allNodes={filteredGraphData.nodes}
                allEdges={filteredGraphData.edges}
                onNodeSelect={setSelectedNode}
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
