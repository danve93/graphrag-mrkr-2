'use client'

import dynamic from 'next/dynamic'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Info, Network } from 'lucide-react'
import { Button } from '@mui/material'
import ExpandablePanel from '@/components/Utils/ExpandablePanel'
import Loader from '@/components/Utils/Loader'

import { api } from '@/lib/api'
import type { GraphCommunity, GraphEdge, GraphNode } from '@/types/graph'

const ForceGraph3D = dynamic(
  () => import('react-force-graph-3d').then((mod) => mod.default),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full w-full items-center justify-center">
        <Loader size={28} label="Loading 3D graph..." />
      </div>
    ),
  }
)

const COMMUNITY_COLORS = [
  '#22d3ee',
  '#a855f7',
  '#f97316',
  '#10b981',
  '#3b82f6',
  '#f59e0b',
  '#e11d48',
  '#0ea5e9',
  '#8b5cf6',
  '#14b8a6',
]

function getCommunityColor(communityId?: number | null) {
  // Defensive: coerce to number and ensure finiteness. If invalid, return default grey.
  if (communityId === undefined || communityId === null) return '#9ca3af'
  const idNum = Number(communityId)
  if (!Number.isFinite(idNum)) return '#9ca3af'
  const idx = Math.abs(Math.trunc(idNum)) % COMMUNITY_COLORS.length
  return COMMUNITY_COLORS[idx]
}

export default function GraphView() {
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] }>({
    nodes: [],
    edges: [],
  })
  const [communities, setCommunities] = useState<GraphCommunity[]>([])
  const [nodeTypes, setNodeTypes] = useState<string[]>([])
  const [selectedCommunity, setSelectedCommunity] = useState<string>('all')
  const [selectedNodeType, setSelectedNodeType] = useState<string>('all')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hoverNode, setHoverNode] = useState<GraphNode | null>(null)
  const [hoverEdge, setHoverEdge] = useState<GraphEdge | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [dimensions, setDimensions] = useState({ width: 400, height: 400 })
  const [expanded, setExpanded] = useState(false)
  const [modalLoading, setModalLoading] = useState(false)
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

  // Resize observer to keep the 3D canvas within parent bounds and avoid horizontal scroll
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect()
      // Prevent the graph from expanding past the bottom of the viewport.
      const availableHeight = typeof window !== 'undefined' ? Math.floor(window.innerHeight - rect.top - 32) : rect.height
      const height = Math.max(200, Math.min(Math.floor(rect.height), availableHeight))
      setDimensions({ width: Math.max(200, Math.floor(rect.width)), height })
    })
    ro.observe(el)
    // set initial
    const r = el.getBoundingClientRect()
    const available = typeof window !== 'undefined' ? Math.floor(window.innerHeight - r.top - 32) : r.height
    setDimensions({ width: Math.max(200, Math.floor(r.width)), height: Math.max(200, Math.min(Math.floor(r.height), available)) })
    return () => ro.disconnect()
  }, [containerRef])

  // Close modal on Escape and show a brief modal loading spinner
  useEffect(() => {
    if (!expanded) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpanded(false)
    }
    window.addEventListener('keydown', onKey)
    setModalLoading(true)
    const tm = window.setTimeout(() => setModalLoading(false), 350)
    return () => {
      window.removeEventListener('keydown', onKey)
      clearTimeout(tm)
      setModalLoading(false)
    }
  }, [expanded])

  const graphPayload = useMemo(
    () => ({
      nodes: graphData.nodes.map((node) => ({
        ...node,
        color: getCommunityColor(node.community_id),
        val: Math.max(node.degree ?? 1, 1),
      })),
      links: graphData.edges.map((edge) => ({
        ...edge,
        value: Math.max(edge.weight ?? 0.5, 0.2),
      })),
    }),
    [graphData.edges, graphData.nodes]
  )

  // Resolve CSS variable values to concrete color strings so downstream
  // libraries (like polished) don't receive `var(--...)` tokens which they
  // can't parse. We do this lazily in the browser environment.
  const backgroundColorResolved = useMemo(() => {
    if (typeof window === 'undefined') return '#ffffff'
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue('--bg-primary')
      if (!v) return '#ffffff'
      return v.trim()
    } catch (e) {
      return '#ffffff'
    }
  }, [])

  const nodeLabel = useCallback((node: GraphNode) => {
    const docs = node.documents?.map((doc) => doc.document_name || doc.document_id).filter(Boolean) || []
    return `
      <div>
        <div><strong>${node.label}</strong></div>
        <div>Type: ${node.type ?? 'Unknown'}</div>
        <div>Community: ${node.community_id ?? 'Unassigned'}</div>
        <div>Degree: ${node.degree ?? 0}</div>
        <div>Documents: ${docs.slice(0, 3).join(', ') || 'None'}</div>
      </div>
    `
  }, [])

  const linkLabel = useCallback((link: GraphEdge) => {
    const docSummaries = (link.text_units || []).map((unit) => unit.document_name || unit.document_id)
    return `
      <div>
        <div><strong>${link.type ?? 'RELATED_TO'}</strong> (${link.weight ?? 0.5})</div>
        <div>${link.description ?? 'No description'}</div>
        <div>TextUnits: ${docSummaries.slice(0, 3).join(', ') || 'Unspecified'}</div>
      </div>
    `
  }, [])

  const hoveredPanel = useMemo(() => {
    if (hoverNode) {
      return (
        <div className="card" style={{ boxShadow: 'var(--shadow-xl)' }}>
          <div className="flex items-center" style={{ gap: 'var(--space-2)', color: 'var(--text-primary)' }}>
            <div className="h-3 w-3 rounded-full" style={{ backgroundColor: getCommunityColor(hoverNode.community_id || undefined) }} />
            <div>
              <p style={{ fontSize: 'var(--text-sm)', fontWeight: 600 }}>{hoverNode.label}</p>
              <p style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>Type: {hoverNode.type || 'Unknown'}</p>
            </div>
          </div>
          <dl className="mt-3 space-y-2" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
            <div className="flex justify-between">
              <dt>Community</dt>
              <dd>{hoverNode.community_id ?? 'Unassigned'}</dd>
            </div>
            <div className="flex justify-between">
              <dt>Degree</dt>
              <dd>{hoverNode.degree ?? 0}</dd>
            </div>
          </dl>
          {hoverNode.documents && hoverNode.documents.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-semibold text-secondary-800 dark:text-secondary-200">Documents</p>
              <ul className="mt-1 space-y-1 text-xs">
                {hoverNode.documents.slice(0, 5).map((doc) => (
                  <li key={`${doc.document_id}-${doc.document_name}`} className="flex items-center gap-2">
                    <span className="block truncate" title={doc.document_name || doc.document_id}>
                      {doc.document_name || doc.document_id}
                    </span>
                    {doc.document_id && (
                      <a
                        className="hover:underline"
                        href={`/?doc=${doc.document_id}`}
                        style={{ color: 'var(--primary-500)' }}
                      >
                        Open
                      </a>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )
    }

    if (hoverEdge) {
      return (
        <div className="card" style={{ boxShadow: 'var(--shadow-xl)' }}>
          <p style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)' }}>{hoverEdge.type || 'RELATED_TO'}</p>
          <p style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>Weight: {hoverEdge.weight ?? 0.5}</p>
          {hoverEdge.description && (
            <p className="mt-2" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>{hoverEdge.description}</p>
          )}
          {hoverEdge.text_units && hoverEdge.text_units.length > 0 && (
            <div className="mt-3">
              <p style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-primary)' }}>Supporting TextUnits</p>
              <ul className="mt-1 space-y-1 text-xs">
                {hoverEdge.text_units.slice(0, 6).map((unit) => (
                  <li key={`${unit.id}-${unit.document_id}`} className="flex items-center gap-2">
                    <span className="truncate" title={unit.id}>
                      {unit.id}
                    </span>
                    {unit.document_id && (
                      <a
                        className="hover:underline"
                        href={`/?doc=${unit.document_id}`}
                        style={{ color: 'var(--primary-500)' }}
                      >
                        {unit.document_name || 'Open document'}
                      </a>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )
    }

    return null
  }, [hoverEdge, hoverNode])

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      {/* Header */}
      <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: 'var(--space-2)' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '8px',
            backgroundColor: '#f27a0320',
            border: '1px solid #f27a03',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Network size={24} color="#f27a03" />
          </div>
          <div style={{ flex: 1 }}>
            <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)' }}>
              Graph Explorer
            </h1>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              3D force layout with communities and provenance-rich tooltips
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.75rem', color: 'var(--text-secondary)', padding: '6px 12px', borderRadius: '6px', backgroundColor: 'var(--bg-secondary)' }}>
            <Info className="h-4 w-4" />
            <span>Hover nodes/edges for details</span>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto" style={{ padding: 'var(--space-6)', display: 'flex', flexDirection: 'column', gap: '12px' }}>

        {/* Filters Panel */}
        <ExpandablePanel
          title="Filters & Controls"
          expanded={expandedPanels.has('filters')}
          onToggle={() => togglePanel('filters')}
        >
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'flex-end' }}>
            <div style={{ flex: '1 1 200px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Community</label>
              <select
                value={selectedCommunity}
                onChange={(event) => setSelectedCommunity(event.target.value)}
                className="input-field"
                style={{ width: '100%' }}
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

            <div style={{ flex: '1 1 200px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Node type</label>
              <select
                value={selectedNodeType}
                onChange={(event) => setSelectedNodeType(event.target.value)}
                className="input-field"
                style={{ width: '100%' }}
              >
                <option value="all">All types</option>
                {nodeTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: 'auto' }}>
              <span style={{ fontSize: '0.75rem', padding: '4px 8px', borderRadius: '4px', backgroundColor: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}>
                {graphData.nodes.length} nodes
              </span>
              <span style={{ fontSize: '0.75rem', padding: '4px 8px', borderRadius: '4px', backgroundColor: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}>
                {graphData.edges.length} edges
              </span>
              <Button
                size="small"
                variant="contained"
                onClick={() => void fetchGraph()}
                style={{
                  textTransform: 'none',
                  backgroundColor: 'var(--accent-primary)',
                  color: 'white',
                  fontSize: '0.75rem'
                }}
              >
                Refresh
              </Button>
            </div>
          </div>
        </ExpandablePanel>

        {/* Graph Visualization */}
        <div
          ref={containerRef}
          className="flex-1 relative overflow-hidden"
          style={{ minHeight: '400px', borderRadius: '8px', border: '1px solid var(--border)', background: backgroundColorResolved }}
        >
          {isLoading && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-secondary-900/70 text-secondary-50">
              <Loader size={28} label="Loading graph…" />
            </div>
          )}
          {error && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-secondary-900/70">
              <div className="rounded bg-white/90 p-4 text-sm text-red-700 shadow dark:bg-secondary-800/90 dark:text-red-200">
                {error}
              </div>
            </div>
          )}
          <ForceGraph3D
            width={dimensions.width}
            height={dimensions.height}
            graphData={graphPayload as any}
            // Use property accessors where possible to avoid TS generic mismatch
            nodeColor={(node: any) => getCommunityColor(node.community_id || undefined)}
            linkColor={() => '#94a3b8'}
            nodeVal={(node: any) => node.val ?? Math.max(node.degree ?? 1, 1)}
            linkWidth={(link: any) => (link.value ?? Math.max(link.weight ?? 0.5, 0.2)) * 2}
            nodeLabel={(node: any) => nodeLabel(node as GraphNode)}
            linkLabel={(link: any) => linkLabel(link as GraphEdge)}
            backgroundColor={backgroundColorResolved}
            onNodeHover={(node: any) => {
              setHoverNode((node as GraphNode) || null)
              setHoverEdge(null)
            }}
            onLinkHover={(link: any) => {
              setHoverEdge((link as GraphEdge) || null)
              setHoverNode(null)
            }}
          />

          {hoveredPanel && (
            <div className="pointer-events-none absolute right-4 top-12 z-20 w-80">{hoveredPanel}</div>
          )}

          {/* Expand button */}
          <button
            aria-label="Open graph fullscreen"
            onClick={() => setExpanded(true)}
            className="absolute top-4 right-4 z-30 rounded bg-white/90 p-2 text-xs font-semibold shadow hover:bg-white dark:bg-secondary-800 dark:text-secondary-100"
          >
            Expand
          </button>
        </div>
      </div>

      {/* Fullscreen modal */}
      {expanded && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
          {/* Close button placed at overlay level so it is always above the ForceGraph canvas */}
          <button
            onClick={() => setExpanded(false)}
            aria-label="Close fullscreen graph"
            className="absolute top-4 right-4 z-60 pointer-events-auto rounded bg-white/90 px-3 py-1 text-sm font-medium shadow dark:bg-secondary-800"
          >
            Close
          </button>
          <div className="relative w-[95%] h-[95%] rounded-lg bg-black/90 p-4">
            <div className="w-full h-full relative">
              {modalLoading && (
                <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/40">
                  <Loader size={40} />
                </div>
              )}
              <div className="absolute inset-0 z-10">
                <ForceGraph3D
                  width={Math.floor(window.innerWidth * 0.9)}
                  height={Math.floor(window.innerHeight * 0.9)}
                  graphData={graphPayload as any}
                  nodeColor={(node: any) => getCommunityColor(node.community_id || undefined)}
                  linkColor={() => '#94a3b8'}
                  nodeVal={(node: any) => node.val ?? Math.max(node.degree ?? 1, 1)}
                  linkWidth={(link: any) => (link.value ?? Math.max(link.weight ?? 0.5, 0.2)) * 2}
                  nodeLabel={(node: any) => nodeLabel(node as GraphNode)}
                  linkLabel={(link: any) => linkLabel(link as GraphEdge)}
                  backgroundColor={backgroundColorResolved}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
