'use client'

import dynamic from 'next/dynamic'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { InformationCircleIcon } from '@heroicons/react/24/outline'

import { api } from '@/lib/api'
import type { GraphCommunity, GraphEdge, GraphNode } from '@/types/graph'

const ForceGraph3D = dynamic(() => import('react-force-graph-3d'), {
  ssr: false,
})

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
  if (communityId === undefined || communityId === null) {
    return '#9ca3af'
  }
  const idx = Math.abs(communityId) % COMMUNITY_COLORS.length
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

  const fetchGraph = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await api.getGraph({
        community_id: selectedCommunity !== 'all' ? Number(selectedCommunity) : undefined,
        node_type: selectedNodeType !== 'all' ? selectedNodeType : undefined,
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
        <div className="rounded-lg border border-secondary-200 bg-white p-4 shadow-xl dark:border-secondary-700 dark:bg-secondary-800">
          <div className="flex items-center gap-2 text-secondary-900 dark:text-secondary-50">
            <div className="h-3 w-3 rounded-full" style={{ backgroundColor: getCommunityColor(hoverNode.community_id || undefined) }} />
            <div>
              <p className="text-sm font-semibold">{hoverNode.label}</p>
              <p className="text-xs text-secondary-500 dark:text-secondary-400">Type: {hoverNode.type || 'Unknown'}</p>
            </div>
          </div>
          <dl className="mt-3 space-y-2 text-xs text-secondary-700 dark:text-secondary-300">
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
                        className="text-primary-600 hover:underline dark:text-primary-400"
                        href={`/?doc=${doc.document_id}`}
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
        <div className="rounded-lg border border-secondary-200 bg-white p-4 shadow-xl dark:border-secondary-700 dark:bg-secondary-800">
          <p className="text-sm font-semibold text-secondary-900 dark:text-secondary-50">{hoverEdge.type || 'RELATED_TO'}</p>
          <p className="text-xs text-secondary-600 dark:text-secondary-300">Weight: {hoverEdge.weight ?? 0.5}</p>
          {hoverEdge.description && (
            <p className="mt-2 text-xs text-secondary-700 dark:text-secondary-200">{hoverEdge.description}</p>
          )}
          {hoverEdge.text_units && hoverEdge.text_units.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-semibold text-secondary-800 dark:text-secondary-200">Supporting TextUnits</p>
              <ul className="mt-1 space-y-1 text-xs">
                {hoverEdge.text_units.slice(0, 6).map((unit) => (
                  <li key={`${unit.id}-${unit.document_id}`} className="flex items-center gap-2">
                    <span className="truncate" title={unit.id}>
                      {unit.id}
                    </span>
                    {unit.document_id && (
                      <a
                        className="text-primary-600 hover:underline dark:text-primary-400"
                        href={`/?doc=${unit.document_id}`}
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
    <div className="flex h-full flex-col gap-4 p-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-secondary-900 dark:text-secondary-50">Graph Explorer</h2>
          <p className="text-sm text-secondary-600 dark:text-secondary-400">
            3D force layout with communities and provenance-rich tooltips.
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-lg bg-secondary-100 px-3 py-2 text-xs text-secondary-700 dark:bg-secondary-800 dark:text-secondary-300">
          <InformationCircleIcon className="h-4 w-4" />
          <span>Hover nodes/edges for TextUnit and document provenance.</span>
        </div>
      </div>

      <div className="flex flex-wrap gap-3 rounded-lg border border-secondary-200 bg-white p-4 shadow-sm dark:border-secondary-700 dark:bg-secondary-800">
        <div className="flex min-w-[200px] flex-col gap-1">
          <label className="text-xs font-semibold text-secondary-700 dark:text-secondary-300">Community</label>
          <select
            value={selectedCommunity}
            onChange={(event) => setSelectedCommunity(event.target.value)}
            className="rounded border border-secondary-300 px-3 py-2 text-sm text-secondary-900 shadow-sm focus:border-primary-500 focus:outline-none dark:border-secondary-700 dark:bg-secondary-900 dark:text-secondary-100"
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

        <div className="flex min-w-[200px] flex-col gap-1">
          <label className="text-xs font-semibold text-secondary-700 dark:text-secondary-300">Node type</label>
          <select
            value={selectedNodeType}
            onChange={(event) => setSelectedNodeType(event.target.value)}
            className="rounded border border-secondary-300 px-3 py-2 text-sm text-secondary-900 shadow-sm focus:border-primary-500 focus:outline-none dark:border-secondary-700 dark:bg-secondary-900 dark:text-secondary-100"
          >
            <option value="all">All types</option>
            {nodeTypes.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>

        <div className="ml-auto flex items-center gap-2 text-xs text-secondary-600 dark:text-secondary-300">
          <span className="rounded bg-secondary-100 px-2 py-1 dark:bg-secondary-700">{graphData.nodes.length} nodes</span>
          <span className="rounded bg-secondary-100 px-2 py-1 dark:bg-secondary-700">{graphData.edges.length} edges</span>
          <button
            type="button"
            onClick={() => void fetchGraph()}
            className="rounded bg-primary-600 px-3 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-primary-500"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="relative min-h-[640px] overflow-hidden rounded-xl border border-secondary-200 bg-secondary-900 shadow-inner dark:border-secondary-700">
        {isLoading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-secondary-900/70 text-secondary-50">
            Loading graph…
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
          graphData={graphPayload}
          nodeColor={(node: GraphNode) => getCommunityColor(node.community_id || undefined)}
          linkColor={() => '#94a3b8'}
          nodeVal={(node: GraphNode & { val?: number }) => node.val ?? Math.max(node.degree ?? 1, 1)}
          linkWidth={(link: GraphEdge & { value?: number }) => (link.value ?? Math.max(link.weight ?? 0.5, 0.2)) * 2}
          nodeLabel={(node: GraphNode) => nodeLabel(node)}
          linkLabel={(link: GraphEdge) => linkLabel(link)}
          backgroundColor="#0b1120"
          onNodeHover={(node) => {
            setHoverNode(node || null)
            setHoverEdge(null)
          }}
          onLinkHover={(link) => {
            setHoverEdge(link || null)
            setHoverNode(null)
          }}
        />

        {hoveredPanel && (
          <div className="pointer-events-none absolute right-4 top-4 z-20 w-80">{hoveredPanel}</div>
        )}
      </div>
    </div>
  )
}
