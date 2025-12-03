 'use client'

import { useEffect, useMemo, useState } from 'react'
import dynamic from 'next/dynamic'
import type { GraphResponse } from '@/types/graph'
import { api } from '@/lib/api'

const ForceGraph3D = dynamic(() => import('react-force-graph-3d'), { ssr: false })

// Community colors for visual distinction
const COMMUNITY_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#A9CCE3',
]

interface DocumentGraphProps {
  documentId: string
  height?: number
}

export default function DocumentGraph({
  documentId,
  height = 500,
}: DocumentGraphProps) {
  const [graphData, setGraphData] = useState<GraphResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // filtering state: node types & communities
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set<string>())
  const [selectedCommunities, setSelectedCommunities] = useState<Set<number | null>>(new Set<number | null>())
  const [selectedNode, setSelectedNode] = useState<any | null>(null)
  const [width, setWidth] = useState(800)
  const [computedHeight, setComputedHeight] = useState(height)
  const [expanded, setExpanded] = useState(false)
  const [modalLoading, setModalLoading] = useState(false)

  useEffect(() => {
    // derive canvas background color from CSS variables so canvas matches panel
    const setCanvasBgFromCss = () => {
      try {
        const root = document.documentElement
        const isDark = root.classList.contains('dark') || root.getAttribute('data-theme') === 'dark'
        const light = getComputedStyle(root).getPropertyValue('--gray-50') || '#f8fafc'
        const dark = getComputedStyle(root).getPropertyValue('--bg-primary') || '#0f172a'
        setCanvasBg((isDark ? dark : light).trim())
      } catch (e) {
        // ignore and keep default
      }
    }

    setCanvasBgFromCss()
    // react to theme changes (class toggles)
    const mo = new MutationObserver(() => setCanvasBgFromCss())
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] })
    return () => mo.disconnect()
  }, [documentId])

  const [canvasBg, setCanvasBg] = useState('#f8fafc')

    const loadGraph = async () => {
      try {
        setLoading(true)
        setError(null)
        const data = await api.getGraph({ document_id: documentId, limit: 50 })
        setGraphData(data)
      } catch (err) {
        console.error('Failed to load document graph:', err)
        const errorMessage = err instanceof Error ? err.message : 'Failed to load graph data'
        // Check if it's the "too many entities" error
        if (errorMessage.includes('entities') || errorMessage.includes('community_id')) {
          setError('This document has too many entities for graph visualization. Please use the Communities tab to explore specific communities.')
        } else {
          setError(errorMessage)
        }
      } finally {
        setLoading(false)
      }
    }

  useEffect(() => {
    if (documentId) {
      loadGraph()
    }
  }, [documentId])

  useEffect(() => {
    const handleResize = () => {
      const container = document.getElementById(`graph-${documentId}`)
      if (container) {
        setWidth(Math.max(300, container.offsetWidth - 2))
      }
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [documentId])

  useEffect(() => {
    // clamp embedded graph height to available viewport space so it doesn't push the page
    const clampToViewport = () => {
      const container = document.getElementById(`graph-${documentId}`)
      const top = container ? container.getBoundingClientRect().top : 0
      const viewportAvailable = typeof window !== 'undefined' ? Math.floor(window.innerHeight - top - 32) : height
      const maxByRatio = typeof window !== 'undefined' ? Math.floor(window.innerHeight * 0.55) : height
      const max = Math.min(maxByRatio, viewportAvailable)
      setComputedHeight(Math.min(height, Math.max(200, Math.floor(max))))
    }

    clampToViewport()
    window.addEventListener('resize', clampToViewport)
    return () => window.removeEventListener('resize', clampToViewport)
  }, [height, documentId])

  // ESC handling and modal loading spinner for fullscreen viewer
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

  const getCommunityColor = (communityId?: number | null) => {
    if (communityId === undefined || communityId === null) return '#9ca3af'
    return COMMUNITY_COLORS[Math.abs(communityId) % COMMUNITY_COLORS.length]
  }
  // Derive lists of available node types & communities from the graph data
  const availableTypes = useMemo(() => {
    if (!graphData) return [] as string[]
    const types = graphData.nodes
      .map((n) => n.type)
      .filter((t): t is string => typeof t === 'string' && t.length > 0)
    return Array.from(new Set(types)).sort()
  }, [graphData])

  const availableCommunities = useMemo(() => {
    if (!graphData) return [] as Array<number | null>
    const s = new Set<number | null>()
    graphData.nodes.forEach((n) => s.add(n.community_id === undefined ? null : n.community_id))
    return Array.from(s).sort((a, b) => {
      if (a === null) return 1
      if (b === null) return -1
      return (a as number) - (b as number)
    })
  }, [graphData])

  // initialize selection to include all available items when graphData changes
  useEffect(() => {
    if (!graphData) return
    setSelectedTypes(new Set<string>(availableTypes))
    setSelectedCommunities(new Set<number | null>(availableCommunities))
  }, [graphData, availableTypes, availableCommunities])

  // Build filtered payload for ForceGraph; preserve metadata (level, documents, text_units)
  const graphPayload = useMemo(() => {
    if (!graphData) return { nodes: [], links: [] }

    // first, compute allowed node ids based on selected filters
    const allowedNodeIds = new Set<string>()
    graphData.nodes.forEach((node) => {
      const nodeType = node.type
      const typeOk = selectedTypes.size === 0 || (typeof nodeType === 'string' && selectedTypes.has(nodeType))
      const comm = node.community_id === undefined ? null : node.community_id
      const commOk = selectedCommunities.size === 0 || selectedCommunities.has(comm)
      if (typeOk && commOk) allowedNodeIds.add(node.id)
    })

    const nodes = graphData.nodes
      .filter((n) => allowedNodeIds.has(n.id))
      .map((node) => ({
        id: node.id,
        name: node.label,
        val: Math.max(Math.sqrt(node.degree || 1) * 2, 2),
        color: getCommunityColor(node.community_id),
        type: node.type,
        communityId: node.community_id,
        degree: node.degree,
        level: node.level, // preserve level
        documents: node.documents, // preserve document refs
      }))

    const links = graphData.edges
      .filter((edge) => allowedNodeIds.has(edge.source) && allowedNodeIds.has(edge.target))
      .map((edge) => ({
        source: edge.source,
        target: edge.target,
        value: edge.weight || 1,
        type: edge.type,
        description: edge.description,
        text_units: edge.text_units,
      }))

    return { nodes, links }
  }, [graphData, selectedTypes, selectedCommunities])

  if (loading) {
    return (
      <div
        id={`graph-${documentId}`}
        style={{ height: `${height}px` }}
        className="flex items-center justify-center bg-secondary-50 dark:bg-secondary-900 rounded-lg border border-slate-200 dark:border-slate-700"
      >
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-2" />
          <p className="text-sm text-slate-600 dark:text-slate-400">Loading graph...</p>
        </div>
      </div>
    )
  }

  if (error || !graphData) {
    return (
      <div
        id={`graph-${documentId}`}
        style={{ height: `${height}px` }}
        className="flex items-center justify-center bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-700"
      >
        <p className="text-sm text-red-600 dark:text-red-400">
          {error || 'No graph data available'}
        </p>
      </div>
    )
  }

  // If the graph returned no nodes for this document, show a clear CTA to process entities
  if (graphData && Array.isArray(graphData.nodes) && graphData.nodes.length === 0) {
    return (
      <div id={`graph-${documentId}`} style={{ height: `${height}px` }} className="flex items-center justify-center bg-secondary-50 dark:bg-secondary-900 rounded-lg border border-slate-200 dark:border-slate-700">
        <div className="text-center p-6">
          <p className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-2">No entities extracted for this document</p>
          <p className="text-xs text-slate-600 dark:text-slate-400 mb-4">Run entity extraction to populate the graph and community data.</p>
          <button
            type="button"
            onClick={async () => {
              try {
                setLoading(true)
                setError(null)
                await api.reprocessDocumentEntities(documentId)
                // notify processing state listeners
                if (typeof window !== 'undefined') {
                  window.dispatchEvent(new CustomEvent('documents:processing-updated'))
                }
              } catch (e) {
                console.error('Failed to queue entity extraction', e)
                setError('Failed to queue entity extraction')
              } finally {
                setLoading(false)
              }
            }}
            className="button-primary px-4 py-2 text-sm"
          >
            Process entities
          </button>
        </div>
      </div>
    )
  }

  const toggleType = (t: string) => {
    setSelectedTypes((prev) => {
      const next = new Set(prev)
      if (next.has(t)) next.delete(t)
      else next.add(t)
      return next
    })
  }

  const toggleCommunity = (c: number | null) => {
    setSelectedCommunities((prev) => {
      const next = new Set(prev)
      if (next.has(c)) next.delete(c)
      else next.add(c)
      return next
    })
  }

  const clearFilters = () => {
    setSelectedTypes(new Set<string>(availableTypes))
    setSelectedCommunities(new Set<number | null>(availableCommunities))
  }

  return (
    <>
      <div id={`graph-${documentId}`} className="w-full grid grid-cols-4 gap-4 min-h-[250px]" style={{ height: `${computedHeight}px` }}>
        {/* Left sidebar: filters (1 column) */}
        <div className="col-span-1 bg-white dark:bg-secondary-800 border border-slate-200 dark:border-slate-700 rounded-lg p-4 overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold">Filters</h3>
            <button className="text-xs text-blue-600 dark:text-blue-400 underline" onClick={clearFilters}>Reset</button>
          </div>

          {/* Node Types Dropdown */}
          <div className="mb-4">
            <label className="block text-xs font-medium mb-2">Node Types</label>
            <div className="space-y-2 max-h-48 overflow-y-auto border border-slate-200 dark:border-slate-600 rounded p-2">
              {availableTypes.map((t) => (
                <label key={t} className="flex items-center text-xs cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-700 p-1 rounded">
                  <input
                    type="checkbox"
                    checked={selectedTypes.has(t)}
                    onChange={() => toggleType(t)}
                    className="mr-2"
                  />
                  <span className="truncate">{t}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Communities Dropdown */}
          {availableCommunities.length > 0 && (
            <div>
              <label className="block text-xs font-medium mb-2">Communities</label>
              <div className="space-y-2 max-h-48 overflow-y-auto border border-slate-200 dark:border-slate-600 rounded p-2">
                {availableCommunities.map((c) => (
                  <label key={`${String(c)}`} className="flex items-center text-xs cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-700 p-1 rounded">
                    <input
                      type="checkbox"
                      checked={selectedCommunities.has(c)}
                      onChange={() => toggleCommunity(c)}
                      className="mr-2"
                    />
                    <span className="inline-flex items-center">
                      <span style={{ background: getCommunityColor(c as any) }} className="inline-block w-3 h-3 rounded mr-2" />
                      <span>{c === null ? 'None' : String(c)}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: graph view (3 columns) */}
        <div className="col-span-3 relative bg-secondary-50 dark:bg-secondary-900 border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
          <ForceGraph3D
            graphData={graphPayload}
            nodeLabel={(node: any) => `${node.name} (${node.type})`}
            nodeColor={(node: any) => node.color}
            nodeVal={(node: any) => node.val}
            onNodeClick={(node: any) => setSelectedNode(node)}
            linkWidth={(link: any) => Math.sqrt(link.value || 1) * 0.5}
            linkColor={() => '#cbd5e1'}
            linkOpacity={0.5}
            backgroundColor={canvasBg}
            width={width}
            height={computedHeight}
            {...({} as any)}
          />
          
          {/* Node details panel with slide-in animation */}
          <div 
            className={`absolute bottom-16 left-3 z-30 w-64 p-3 bg-white/95 dark:bg-secondary-800 rounded-md shadow-md text-xs transition-all duration-200 ease-out ${
              selectedNode ? 'translate-x-0 opacity-100' : '-translate-x-full opacity-0 pointer-events-none'
            }`}
          >
            {selectedNode && (
              <>
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="font-semibold text-sm">{selectedNode.name}</div>
                    <div className="text-[11px] text-slate-600 dark:text-slate-300">{selectedNode.type}</div>
                  </div>
                  <button onClick={() => setSelectedNode(null)} className="text-xs ml-2">Close</button>
                </div>
                <div className="text-[11px] text-slate-700 dark:text-slate-300">
                  <div><strong>Degree:</strong> {selectedNode.degree ?? '-'}</div>
                  <div><strong>Level:</strong> {selectedNode.level ?? '-'}</div>
                  {selectedNode.documents && selectedNode.documents.length > 0 && (
                    <div className="mt-2">
                      <div className="font-medium">Documents</div>
                      <ul className="list-disc ml-4 max-h-28 overflow-auto mt-1">
                        {selectedNode.documents.slice(0, 6).map((d: any) => {
                          const docId = (d && (d.id || d.document_id)) || (typeof d === 'string' ? d : null)
                          const title = (d && (d.title || d.name)) || docId || String(d)
                          return (
                            <li key={docId || title} className="flex items-center justify-between">
                              <span className="truncate pr-2">{title}</span>
                              {docId ? (
                                <a
                                  href={`/documents/${docId}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-xs text-blue-600 dark:text-blue-400 ml-2"
                                >
                                  Open
                                </a>
                              ) : null}
                            </li>
                          )
                        })}
                        {selectedNode.documents.length > 6 && <li className="text-[11px]">and more...</li>}
                      </ul>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
          
          {/* Expand button */}
          <div className="absolute top-2 right-2 z-20">
            <button
              onClick={() => setExpanded(true)}
              className="rounded bg-white/90 px-2 py-1 text-xs font-medium shadow dark:bg-secondary-800"
            >
              Expand
            </button>
          </div>
          
          {/* Stats bar pinned to bottom overlay */}
          <div className="absolute bottom-0 left-0 right-0 z-10 text-xs text-slate-500 dark:text-slate-400 p-2 border-t border-slate-200 dark:border-slate-700 bg-white/95 dark:bg-secondary-800/90 backdrop-blur-sm">
            <span>{graphData.nodes.length} entities</span>
            {' · '}
            <span>{graphData.edges.length} relationships</span>
            {graphData.communities && graphData.communities.length > 0 && (
              <>
                {' · '}
                <span>{graphData.communities.length} communities</span>
              </>
            )}
          </div>
        </div>
      </div>

      {expanded && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
          {/* overlay-level close button so it stays above the canvas */}
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
                  <div className="inline-block animate-spin rounded-full h-10 w-10 border-b-2 border-white" />
                </div>
              )}
              <div className="absolute inset-0 z-10">
                <ForceGraph3D
                  width={Math.floor(window.innerWidth * 0.9)}
                  height={Math.floor(window.innerHeight * 0.9)}
                  graphData={graphPayload}
                  nodeLabel={(node: any) => `${node.name} (${node.type})`}
                  nodeColor={(node: any) => node.color}
                  nodeVal={(node: any) => node.val}
                  onNodeClick={(node: any) => setSelectedNode(node)}
                  linkWidth={(link: any) => Math.sqrt(link.value || 1) * 0.5}
                  linkColor={() => '#cbd5e1'}
                  linkOpacity={0.5}
                  backgroundColor="#f8fafc"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}