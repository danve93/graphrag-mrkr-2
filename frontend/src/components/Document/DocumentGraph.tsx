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
        setWidth(container.offsetWidth - 2)
      }
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [documentId])

  const getCommunityColor = (communityId?: number | null) => {
    if (communityId === undefined || communityId === null) return '#9ca3af'
    return COMMUNITY_COLORS[communityId % COMMUNITY_COLORS.length]
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

  return (
    <div id={`graph-${documentId}`} className="w-full border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
      <div style={{ height: `${height}px`, width: `${width}px` }}>
        <ForceGraph3D
          graphData={graphPayload}
          nodeLabel={(node: any) => `${node.name} (${node.type})`}
          nodeColor={(node: any) => node.color}
          nodeVal={(node: any) => node.val}
          linkWidth={(link: any) => Math.sqrt(link.value || 1) * 0.5}
          linkColor={() => '#cbd5e1'}
          linkOpacity={0.5}
          backgroundColor="#f8fafc"
          height={height}
          width={width}
          {...({} as any)}
        />
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
  )
}
