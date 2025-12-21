'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { ChevronDown } from 'lucide-react'
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
        performance.mark('communities-load-start')
        const communityMap = new Map<number, { community_id: number; level?: number | null; count: number }>()
        let offset = 0
        const limit = 500 // backend max
        let hasMore = true
        let pageCount = 0
        while (hasMore) {
          const page = await api.getDocumentEntitiesPaginated(documentId, { limit, offset })
          pageCount++
          page.entities.forEach(entity => {
            if (entity.community_id !== null && entity.community_id !== undefined) {
              const existing = communityMap.get(entity.community_id) || { community_id: entity.community_id, level: entity.level, count: 0 }
              existing.count += 1
              communityMap.set(entity.community_id, existing)
            }
          })
          hasMore = page.has_more
          offset += limit
        }
        const communitiesArray = Array.from(communityMap.values()).sort((a, b) => a.community_id - b.community_id)
        setCommunities(communitiesArray)
        performance.mark('communities-load-end')
        performance.measure('communities-load', 'communities-load-start', 'communities-load-end')
      } catch (err) {
        console.error('Failed to load communities:', err)
        setCommunities([])
      } finally {
        setLoading(false)
      }
    }
    void loadCommunities()
  }, [documentId, initialCommunities])

  if (loading) {
    return (
      <div
        className="rounded-lg overflow-hidden"
        style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border)' }}
      >
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center gap-2 px-4 py-3 font-semibold transition"
          style={{ color: 'var(--text-primary)' }}
        >
          <ChevronDown className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
          <span>Communities</span>
          <span
            className="ml-auto text-xs px-2 py-1 rounded"
            style={{ backgroundColor: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}
          >
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
      <div
        className="rounded-lg overflow-hidden"
        style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border)' }}
      >
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center gap-2 px-4 py-3 font-semibold transition"
          style={{ color: 'var(--text-primary)' }}
        >
          <ChevronDown className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
          <span>Communities</span>
          <span
            className="ml-auto text-xs px-2 py-1 rounded"
            style={{ backgroundColor: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}
          >
            None
          </span>
        </button>
        <div className="p-4" style={{ borderTop: '1px solid var(--border)' }}>
          <p className="text-sm mb-3" style={{ color: 'var(--text-secondary)' }}>No communities found for this document.</p>
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
    <div
      className="rounded-lg overflow-hidden"
      style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border)' }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-4 py-3 font-semibold transition"
        style={{ color: 'var(--text-primary)' }}
      >
        <ChevronDown className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
        <span>Communities</span>
        <span
          className="ml-auto text-xs px-2 py-1 rounded"
          style={{ backgroundColor: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}
        >
          {communities.length}
        </span>
      </button>

      {expanded && (
        <div className="p-4" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="space-y-2">
            {communities.map((community) => (
              <div
                key={`${community.community_id}-${community.level}`}
                className="flex items-center justify-between p-3 rounded transition"
                style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
              >
                <div className="flex-1">
                  <p className="font-medium" style={{ color: 'var(--text-primary)' }}>
                    Community {community.community_id}
                  </p>
                  <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    {community.count} entities
                    {community.level !== undefined && ` · Level ${community.level}`}
                  </p>
                </div>
                <Link
                  href={`/graph?community=${community.community_id}`}
                  className="text-xs px-3 py-1 rounded transition"
                  style={{ backgroundColor: 'var(--accent-subtle)', color: 'var(--accent-primary)' }}
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
