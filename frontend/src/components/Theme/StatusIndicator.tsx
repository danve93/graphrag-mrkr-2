'use client'

import React, { useEffect, useState } from 'react'
import { useChatStore } from '@/store/chatStore'
import Loader from '@/components/Utils/Loader'

export default function StatusIndicator() {
  const isConnected = useChatStore((s) => s.isConnected)
  const [flashrankInProgress, setFlashrankInProgress] = useState(false)
  const [flashrankError, setFlashrankError] = useState<string | null>(null)
  const [flashrankStartedAt, setFlashrankStartedAt] = useState<number | null>(null)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    let mounted = true
    let timer: number | undefined

    async function fetchHealth() {
      try {
        const res = await fetch('/api/health')
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
        if (!mounted) return
        const fr = data?.flashrank
        setFlashrankInProgress(Boolean(fr?.in_progress))
        setFlashrankError(fr?.error || null)
        // Reset dismissal if a new prewarm run started
        if (fr?.started_at && fr.started_at !== flashrankStartedAt) {
          setDismissed(false)
          setFlashrankStartedAt(fr.started_at)
        }
      } catch (err: any) {
        if (!mounted) return
        // network or parse error — treat as not in progress but store error
        setFlashrankInProgress(false)
        setFlashrankError(err?.message || String(err))
      } finally {
        if (!mounted) return
        timer = window.setTimeout(fetchHealth, 5000)
      }
    }

    fetchHealth()
    return () => {
      mounted = false
      if (timer) clearTimeout(timer)
    }
  }, [])

  return (
    <div className="fixed top-4 right-4 z-50 flex items-center gap-3">
      {isConnected ? (
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#24c6e6' }} />
          <span className="text-sm text-secondary-200">Connected</span>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-secondary-700" />
          <span className="text-sm text-secondary-500">Disconnected</span>
        </div>
      )}
      {/* FlashRank prewarm indicator */}
      {flashrankInProgress && !dismissed ? (
        <div className="flex items-center gap-2">
          <Loader size={14} label={flashrankError ? 'Prewarm failed' : 'Pre-warming ranker...'} />
          <button
            aria-label="Dismiss prewarm notice"
            title="Dismiss"
            className="ml-2 text-xs text-secondary-400 hover:text-secondary-200"
            onClick={() => setDismissed(true)}
          >
            ×
          </button>
        </div>
      ) : null}
    </div>
  )
}
