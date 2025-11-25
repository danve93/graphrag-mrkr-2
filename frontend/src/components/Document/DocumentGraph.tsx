/* Replaced duplicate/corrupted content with a single clean implementation below */

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
  const [width, setWidth] = useState(800)
  const [computedHeight, setComputedHeight] = useState(height)
  const [expanded, setExpanded] = useState(false)
  const [modalLoading, setModalLoading] = useState(false)

  useEffect(() => {
    const loadGraph = async () => {
      try {
        setLoading(true)
        setError(null)
        const data = await api.getGraph({ document_id: documentId })
        setGraphData(data)
      } catch (err) {
        console.error('Failed to load document graph:', err)
        setError('Failed to load graph data')
      } finally {
        setLoading(false)
      }
    }

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

  const graphPayload = useMemo(() => {
    if (!graphData) return { nodes: [], links: [] }

    return {
      nodes: graphData.nodes.map((node) => ({
        id: node.id,
        name: node.label,
        val: Math.max(Math.sqrt(node.degree || 1) * 2, 2),
        color: getCommunityColor(node.community_id),
        type: node.type,
        communityId: node.community_id,
        degree: node.degree,
      })),
      links: graphData.edges.map((edge) => ({
        source: edge.source,
        target: edge.target,
        value: edge.weight || 1,
        type: edge.type,
        description: edge.description,
      })),
    }
  }, [graphData])

  if (loading) {
    return (
      <div
        id={`graph-${documentId}`}
        style={{ height: `${height}px` }}
        className="flex items-center justify-center bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700"
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
      <div id={`graph-${documentId}`} style={{ height: `${height}px` }} className="flex items-center justify-center bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700">
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

  return (
    <>
      <div id={`graph-${documentId}`} className="relative w-full border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
        <div style={{ height: `${computedHeight}px`, width: '100%' }}>
          <ForceGraph3D
            graphData={graphPayload}
            nodeLabel={(node: any) => `${node.name} (${node.type})`}
            nodeColor={(node: any) => node.color}
            nodeVal={(node: any) => node.val}
            linkWidth={(link: any) => Math.sqrt(link.value || 1) * 0.5}
            linkColor={() => '#cbd5e1'}
            linkOpacity={0.5}
            backgroundColor="#f8fafc"
            height={computedHeight}
            width={width}
            {...({} as any)}
          />
        </div>
        <div className="absolute top-2 right-2">
          <button
            onClick={() => setExpanded(true)}
            className="rounded bg-white/90 px-2 py-1 text-xs font-medium shadow dark:bg-secondary-800"
          >
            Expand
          </button>
        </div>
        <div className="text-xs text-slate-500 dark:text-slate-400 p-2 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900">
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