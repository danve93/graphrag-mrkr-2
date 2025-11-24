'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { ChevronDownIcon } from '@heroicons/react/24/outline'
import { api } from '@/lib/api'

interface CommunitiesSectionProps {
  documentId: string
  communities?: Array<{
    community_id: number
    level?: number | null
    count?: number
  }>
}

export default function CommunitiesSection({
  documentId,
  communities: initialCommunities,
}: CommunitiesSectionProps) {
  const [communities, setCommunities] = useState(initialCommunities || [])
  const [expanded, setExpanded] = useState(true)
  const [loading, setLoading] = useState(!initialCommunities)
  const [processing, setProcessing] = useState(false)

  useEffect(() => {
    const loadCommunities = async () => {
      if (initialCommunities) return

      try {
        setLoading(true)
        const graphData = await api.getGraph({ document_id: documentId, limit: 1000 })
        
        if (graphData.nodes.length === 0) {
          setCommunities([])
          return
        }

        // Extract unique communities from nodes
        const communityMap = new Map<
          number,
          { community_id: number; level?: number | null; count: number }
        >()

        graphData.nodes.forEach((node) => {
          if (node.community_id !== null && node.community_id !== undefined) {
            const key = node.community_id
            const existing = communityMap.get(key) || {
              community_id: node.community_id,
              level: node.level,
              count: 0,
            }
            existing.count += 1
            communityMap.set(key, existing)
          }
        })

        setCommunities(Array.from(communityMap.values()).sort((a, b) => a.community_id - b.community_id))
      } catch (err) {
        console.error('Failed to load communities:', err)
        setCommunities([])
      } finally {
        setLoading(false)
      }
    }

    loadCommunities()
  }, [documentId, initialCommunities])

  if (loading) {
    return (
      <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 overflow-hidden">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center gap-2 px-4 py-3 font-semibold text-slate-900 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition"
        >
          <ChevronDownIcon className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
          <span>Communities</span>
          <span className="ml-auto text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-slate-600 dark:text-slate-400">
            Loading...
          </span>
        </button>
      </div>
    )
  }

  const handleProcess = async () => {
    try {
      setProcessing(true)
      await api.reprocessDocumentEntities(documentId)
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:processing-updated'))
      }
    } catch (e) {
      console.error('Failed to queue entity extraction', e)
    } finally {
      setProcessing(false)
    }
  }

  if (communities.length === 0) {
    return (
      <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 overflow-hidden">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center gap-2 px-4 py-3 font-semibold text-slate-900 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition"
        >
          <ChevronDownIcon className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
          <span>Communities</span>
          <span className="ml-auto text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-slate-600 dark:text-slate-400">
            None
          </span>
        </button>
        <div className="p-4 border-t border-slate-200 dark:border-slate-700">
          <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">No communities found for this document.</p>
          <div className="flex items-center gap-2">
            <button
              onClick={handleProcess}
              disabled={processing}
              className="button-primary text-sm px-4 py-2"
            >
              {processing ? 'Queuing...' : 'Process entities'}
            </button>
            <button
              onClick={() => setExpanded(true)}
              className="button-secondary text-sm px-3 py-2"
            >
              Refresh
            </button>
          </div>
        </div>
      </div>
    )
  }


  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-4 py-3 font-semibold text-slate-900 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition"
      >
        <ChevronDownIcon className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
        <span>Communities</span>
        <span className="ml-auto text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-slate-600 dark:text-slate-400">
          {communities.length}
        </span>
      </button>

      {expanded && (
        <div className="border-t border-slate-200 dark:border-slate-700 p-4">
          <div className="space-y-2">
            {communities.map((community) => (
              <div
                key={`${community.community_id}-${community.level}`}
                className="flex items-center justify-between p-3 rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/50 hover:bg-slate-100 dark:hover:bg-slate-900 transition"
              >
                <div className="flex-1">
                  <p className="font-medium text-slate-900 dark:text-slate-100">
                    Community {community.community_id}
                  </p>
                  <p className="text-xs text-slate-600 dark:text-slate-400">
                    {community.count} entities
                    {community.level !== undefined && ` · Level ${community.level}`}
                  </p>
                </div>
                <Link
                  href={`/graph?community=${community.community_id}`}
                  className="text-xs px-3 py-1 rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-800 transition"
                >
                  View
                </Link>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
