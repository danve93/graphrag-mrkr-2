"use client"

import { useEffect, useState } from 'react'
import { ChevronDownIcon } from '@heroicons/react/24/outline'
import { api } from '@/lib/api'

interface EntitiesSectionProps {
  documentId: string
}

interface EntityRecord {
  type: string
  text: string
  community_id?: number | null
  level?: number | null
  count: number
  positions: number[]
}

export default function EntitiesSection({ documentId }: EntitiesSectionProps) {
  const [page, setPage] = useState(0)
  const [limit] = useState(100)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<null | {
    total: number
    entities: EntityRecord[]
    has_more: boolean
  }>(null)
  const [expanded, setExpanded] = useState(true)
  const [filterType, setFilterType] = useState<string>('')
  const [filterCommunity, setFilterCommunity] = useState<string>('')

  useEffect(() => {
    const loadEntities = async () => {
      try {
        setLoading(true)
        setError(null)
        const response = await api.getDocumentEntitiesPaginated(documentId, {
          limit,
          offset: page * limit,
          entityType: filterType || undefined,
          communityId: filterCommunity ? Number(filterCommunity) : undefined,
        })
        setData({
          total: response.total,
            entities: response.entities,
            has_more: response.has_more,
        })
      } catch (e) {
        console.error('Failed to load entities', e)
        setError('Failed to load entities')
      } finally {
        setLoading(false)
      }
    }
    loadEntities()
  }, [documentId, page, limit, filterType, filterCommunity])

  if (loading) {
    return (
      <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 overflow-hidden">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center gap-2 px-4 py-3 font-semibold text-slate-900 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition"
        >
          <ChevronDownIcon className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
          <span>Entities</span>
          <span className="ml-auto text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-slate-600 dark:text-slate-400">Loading...</span>
        </button>
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-700 bg-red-50 dark:bg-red-900/30 overflow-hidden">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center gap-2 px-4 py-3 font-semibold text-red-900 dark:text-red-100 hover:bg-red-100 dark:hover:bg-red-900/40 transition"
        >
          <ChevronDownIcon className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
          <span>Entities</span>
          <span className="ml-auto text-xs text-red-600 dark:text-red-400">{error}</span>
        </button>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-4 py-3 font-semibold text-slate-900 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition"
      >
        <ChevronDownIcon className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
        <span>Entities</span>
        <span className="ml-auto text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-slate-600 dark:text-slate-400">
          {data.total}
        </span>
      </button>

      {expanded && (
        <div className="border-t border-slate-200 dark:border-slate-700 p-4">
          <div className="flex flex-wrap gap-2 mb-4 items-center">
            <input
              placeholder="Filter type"
              value={filterType}
              onChange={(e) => { setPage(0); setFilterType(e.target.value) }}
              className="text-xs px-2 py-1 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900"
            />
            <input
              placeholder="Community ID"
              value={filterCommunity}
              onChange={(e) => { setPage(0); setFilterCommunity(e.target.value) }}
              className="text-xs px-2 py-1 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900"
            />
            <div className="ml-auto text-xs text-slate-500 dark:text-slate-400">
              Page {page + 1} · Showing {data.entities.length} of {data.total}
            </div>
          </div>
          <div className="space-y-2">
            {data.entities.map((entity, idx) => (
              <div
                key={`${entity.type}-${entity.text}-${idx}`}
                className="p-3 rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/50"
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold px-2 py-1 rounded bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300">
                      {entity.type}
                    </span>
                    {entity.community_id !== null && entity.community_id !== undefined && (
                      <span className="text-xs px-2 py-1 rounded bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300">
                        C{entity.community_id}
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-slate-600 dark:text-slate-400">{entity.count} occurrences</span>
                </div>
                <p className="text-xs text-slate-700 dark:text-slate-300 break-words">{entity.text}</p>
              </div>
            ))}
          </div>
          <div className="mt-4 flex items-center justify-between">
            <button
              disabled={page === 0}
              onClick={() => setPage(p => Math.max(0, p - 1))}
              className="text-xs px-3 py-1 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 disabled:opacity-50"
            >Prev</button>
            <button
              disabled={!data.has_more}
              onClick={() => setPage(p => p + 1)}
              className="text-xs px-3 py-1 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 disabled:opacity-50"
            >Next</button>
          </div>
        </div>
      )}
    </div>
  )
}
