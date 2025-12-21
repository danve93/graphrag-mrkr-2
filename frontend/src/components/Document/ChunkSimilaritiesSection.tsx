'use client'

import { useEffect, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { api } from '@/lib/api'

interface SimilarityPair {
  chunk1_id: string
  chunk2_id: string
  score: number
}

interface ChunkDetails {
  id: string
  content: string
  index: number
  offset: number
  document_id: string
  document_name?: string | null
}

interface ChunkSimilaritiesSectionProps {
  documentId: string
}

export default function ChunkSimilaritiesSection({ documentId }: ChunkSimilaritiesSectionProps) {
  const [page, setPage] = useState(0)
  const [limit] = useState(50)
  const [data, setData] = useState<null | {
    total: number
    estimated: boolean
    similarities: SimilarityPair[]
    has_more: boolean
  }>(null)
  const [expanded, setExpanded] = useState(true)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [chunkCache, setChunkCache] = useState<Record<string, ChunkDetails>>({})

  useEffect(() => {
    const loadPage = async () => {
      try {
        setLoading(true)
        setError(null)
        performance.mark('similarities-page-fetch-start')
        const response = await api.getDocumentSimilaritiesPaginated(documentId, {
          limit,
          offset: page * limit,
          minScore: 0.0,
          exactCount: page > 0,  // Request exact count when navigating past first page
        })
        performance.mark('similarities-page-fetch-end')
        performance.measure('similarities-page-fetch', 'similarities-page-fetch-start', 'similarities-page-fetch-end')
        setData({
          total: response.total,
          estimated: response.estimated || false,
          similarities: response.similarities,
          has_more: response.has_more,
        })
      } catch (err) {
        console.error('Failed to load chunk similarities:', err)
        setError('Failed to load similarities')
      } finally {
        setLoading(false)
      }
    }

    loadPage()
  }, [documentId, page, limit])

  const loadChunk = async (chunkId: string) => {
    if (chunkCache[chunkId]) return
    try {
      const details = await api.getChunkDetails(chunkId)
      setChunkCache(prev => ({ ...prev, [chunkId]: details }))
    } catch (e) {
      console.error('Failed to load chunk details', e)
    }
  }

  const getPreview = (text?: string, maxLength = 100) => {
    if (!text) return '(no content)'
    if (text.length > maxLength) return text.slice(0, maxLength) + '...'
    return text
  }

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
          <span>Chunk Similarities</span>
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

  if (error) {
    return (
      <div
        className="rounded-lg overflow-hidden"
        style={{ backgroundColor: 'rgba(255, 69, 58, 0.1)', border: '1px solid rgba(255, 69, 58, 0.3)' }}
      >
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center gap-2 px-4 py-3 font-semibold transition"
          style={{ color: '#FF453A' }}
        >
          <ChevronDown className={`h-5 w-5 transition-transform ${expanded ? '' : '-rotate-90'}`} />
          <span>Chunk Similarities</span>
          <span className="ml-auto text-xs" style={{ color: '#FF453A' }}>{error}</span>
        </button>
      </div>
    )
  }

  if (data && data.similarities.length === 0) {
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
          <span>Chunk Similarities</span>
          <span
            className="ml-auto text-xs px-2 py-1 rounded"
            style={{ backgroundColor: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}
          >
            None
          </span>
        </button>
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
        <span>Chunk Similarities</span>
        <span
          className="ml-auto text-xs px-2 py-1 rounded"
          style={{ backgroundColor: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}
        >
          {data ? (data.estimated ? `~${data.total}` : data.total) : 0}
        </span>
      </button>

      {expanded && (
        <div className="p-4" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="space-y-3">
            {data?.similarities.map((sim, idx) => (
              <div
                key={`${sim.chunk1_id}-${sim.chunk2_id}-${idx}`}
                className="p-3 rounded transition"
                style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span
                      className="text-xs font-semibold px-2 py-1 rounded"
                      style={{ backgroundColor: 'var(--accent-subtle)', color: 'var(--accent-primary)' }}
                    >
                      Similarity
                    </span>
                    <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                      {(sim.score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <code className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                    {sim.chunk1_id.slice(0, 8)}...
                  </code>
                </div>
                <div className="space-y-2 mb-2">
                  <div>
                    <p className="text-xs font-semibold mb-1" style={{ color: 'var(--text-secondary)' }}>Chunk 1</p>
                    {chunkCache[sim.chunk1_id] ? (
                      <p
                        className="text-xs p-2 rounded"
                        style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
                      >
                        {getPreview(chunkCache[sim.chunk1_id].content)}
                      </p>
                    ) : (
                      <button
                        onClick={() => loadChunk(sim.chunk1_id)}
                        className="text-xs px-2 py-1 rounded"
                        style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                      >
                        Load chunk
                      </button>
                    )}
                  </div>
                  <div>
                    <p className="text-xs font-semibold mb-1" style={{ color: 'var(--text-secondary)' }}>Chunk 2</p>
                    {chunkCache[sim.chunk2_id] ? (
                      <p
                        className="text-xs p-2 rounded"
                        style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
                      >
                        {getPreview(chunkCache[sim.chunk2_id].content)}
                      </p>
                    ) : (
                      <button
                        onClick={() => loadChunk(sim.chunk2_id)}
                        className="text-xs px-2 py-1 rounded"
                        style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                      >
                        Load chunk
                      </button>
                    )}
                  </div>
                </div>

                <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  <code className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    {sim.chunk1_id.slice(0, 12)}... ↔ {sim.chunk2_id.slice(0, 12)}...
                  </code>
                </p>
              </div>
            ))}
          </div>
          {data && (
            <div className="mt-4 flex items-center justify-between">
              <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                Page {page + 1} · Showing {data.similarities.length} of {data.estimated ? `~${data.total}` : data.total}
              </div>
              <div className="flex gap-2">
                <button
                  disabled={page === 0}
                  onClick={() => setPage(p => Math.max(0, p - 1))}
                  className="text-xs px-3 py-1 rounded disabled:opacity-50"
                  style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                >
                  Prev
                </button>
                <button
                  disabled={!data.has_more}
                  onClick={() => setPage(p => p + 1)}
                  className="text-xs px-3 py-1 rounded disabled:opacity-50"
                  style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
