'use client'

import { useEffect, useState } from 'react'
import { ChevronDownIcon } from '@heroicons/react/24/outline'
import { api } from '@/lib/api'

interface ChunkSimilarity {
  chunk1_id: string
  chunk1_text?: string
  chunk2_id: string
  chunk2_text?: string
  score: number
}

interface ChunkSimilaritiesSectionProps {
  documentId: string
}

export default function ChunkSimilaritiesSection({ documentId }: ChunkSimilaritiesSectionProps) {
  const [similarities, setSimilarities] = useState<ChunkSimilarity[]>([])
  const [expanded, setExpanded] = useState(true)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadSimilarities = async () => {
      try {
        setLoading(true)
        setError(null)
        const data = await api.getDocumentChunkSimilarities(documentId)
        setSimilarities(data.similarities || [])
      } catch (err) {
        console.error('Failed to load chunk similarities:', err)
        setError('Failed to load similarities')
      } finally {
        setLoading(false)
      }
    }

    loadSimilarities()
  }, [documentId])

  const getPreview = (text?: string, maxLength = 100) => {
    if (!text) return '(no content)'
    if (text.length > maxLength) return text.slice(0, maxLength) + '...'
    return text
  }

  if (loading) {
    return (
      <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 overflow-hidden">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center gap-2 px-4 py-3 font-semibold text-slate-900 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition"
        >
          <ChevronDownIcon className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
          <span>Chunk Similarities</span>
          <span className="ml-auto text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-slate-600 dark:text-slate-400">
            Loading...
          </span>
        </button>
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-700 bg-red-50 dark:bg-red-900/20 overflow-hidden">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center gap-2 px-4 py-3 font-semibold text-red-900 dark:text-red-100 hover:bg-red-100 dark:hover:bg-red-900/30 transition"
        >
          <ChevronDownIcon className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
          <span>Chunk Similarities</span>
          <span className="ml-auto text-xs text-red-600 dark:text-red-400">{error}</span>
        </button>
      </div>
    )
  }

  if (similarities.length === 0) {
    return (
      <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 overflow-hidden">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center gap-2 px-4 py-3 font-semibold text-slate-900 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition"
        >
          <ChevronDownIcon className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
          <span>Chunk Similarities</span>
          <span className="ml-auto text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-slate-600 dark:text-slate-400">
            None
          </span>
        </button>
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
        <span>Chunk Similarities</span>
        <span className="ml-auto text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-slate-600 dark:text-slate-400">
          {similarities.length}
        </span>
      </button>

      {expanded && (
        <div className="border-t border-slate-200 dark:border-slate-700 p-4">
          <div className="space-y-3">
            {similarities.map((sim, idx) => (
              <div
                key={`${sim.chunk1_id}-${sim.chunk2_id}-${idx}`}
                className="p-3 rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/50 hover:bg-slate-100 dark:hover:bg-slate-900 transition"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold px-2 py-1 rounded bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300">
                      Similarity
                    </span>
                    <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      {(sim.score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <code className="text-xs text-slate-500 dark:text-slate-400">
                    {sim.chunk1_id.slice(0, 8)}...
                  </code>
                </div>

                <div className="space-y-2 mb-2">
                  <div>
                    <p className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                      Chunk 1
                    </p>
                    <p className="text-xs text-slate-600 dark:text-slate-400 bg-white dark:bg-slate-800 p-2 rounded border border-slate-200 dark:border-slate-700">
                      {getPreview(sim.chunk1_text)}
                    </p>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                      Chunk 2
                    </p>
                    <p className="text-xs text-slate-600 dark:text-slate-400 bg-white dark:bg-slate-800 p-2 rounded border border-slate-200 dark:border-slate-700">
                      {getPreview(sim.chunk2_text)}
                    </p>
                  </div>
                </div>

                <p className="text-xs text-slate-500 dark:text-slate-400">
                  <code className="text-xs text-slate-600 dark:text-slate-300">
                    {sim.chunk1_id.slice(0, 12)}... ↔ {sim.chunk2_id.slice(0, 12)}...
                  </code>
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
