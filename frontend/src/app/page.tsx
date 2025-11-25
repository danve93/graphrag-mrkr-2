'use client'

import ChatInterface from '@/components/Chat/ChatInterface'
import DocumentView from '@/components/Document/DocumentView'
import GraphView from '@/components/Graph/GraphView'
import ChatTuningPanel from '@/components/ChatTuning/ChatTuningPanel'
import ClassificationPanel from '@/components/Classification/ClassificationPanel'
import Sidebar from '@/components/Sidebar/Sidebar'
import { ThemeToggle } from '@/components/Theme/ThemeToggle'
import { useEffect, useState } from 'react'
import { useChatStore } from '@/store/chatStore'
import { api } from '@/lib/api'

export default function Home() {
  const activeView = useChatStore((state) => state.activeView)
  const setActiveView = useChatStore((state) => state.setActiveView)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const isConnected = useChatStore((s) => s.isConnected)
  const setIsConnected = useChatStore((s) => s.setIsConnected)
  const DEFAULT_WIDTH = 320
  const MIN_WIDTH = 260
  const MAX_WIDTH = 480
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_WIDTH)
  const [userResized, setUserResized] = useState(false)
  const navigation = [
    { id: 'graph', label: 'Graph' },
    { id: 'chatTuning', label: 'Chat Tuning' },
    { id: 'classification', label: 'Classification' },
  ] as const

  useEffect(() => {
    const storedWidth = typeof window !== 'undefined' ? window.localStorage.getItem('sidebar-width') : null
    if (storedWidth) {
      const parsed = parseInt(storedWidth, 10)
      if (!Number.isNaN(parsed)) {
        setSidebarWidth(Math.min(Math.max(parsed, MIN_WIDTH), MAX_WIDTH))
        setUserResized(true)
        return
      }
    }

    // If no stored width, initialize to 25% of viewport (clamped)
    if (typeof window !== 'undefined') {
      const vw = Math.floor(window.innerWidth * 0.25)
      setSidebarWidth(Math.min(Math.max(vw, MIN_WIDTH), MAX_WIDTH))
    }
  }, [])

  // Update sidebar width responsively when user hasn't manually resized
  useEffect(() => {
    if (userResized) return
    const onResize = () => {
      const vw = Math.floor(window.innerWidth * 0.25)
      setSidebarWidth(Math.min(Math.max(vw, MIN_WIDTH), MAX_WIDTH))
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [userResized])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem('sidebar-width', String(sidebarWidth))
    }
  }, [sidebarWidth])

  // Poll backend health and set global connected state
  useEffect(() => {
    let mounted = true
    async function check() {
      try {
        const ok = await api.checkHealth()
        if (mounted) setIsConnected(ok)
      } catch (err) {
        if (mounted) setIsConnected(false)
      }
    }

    // initial check
    void check()
    const id = setInterval(() => {
      void check()
    }, 4000)
    return () => {
      mounted = false
      clearInterval(id)
    }
  }, [setIsConnected])

  const clampWidth = (next: number) => Math.min(Math.max(next, MIN_WIDTH), MAX_WIDTH)

  return (
    <div
      className="grid h-screen bg-secondary-50 dark:bg-secondary-900"
      style={{
        // Left column: sidebar (fixed pixel or collapsed width), right column: main (1fr)
        gridTemplateColumns: `${sidebarOpen ? (sidebarCollapsed ? '72px' : `${sidebarWidth}px`) : '0px'} 1fr`,
      }}
    >
      {/* Top-level connection banner/loader */}
      {!isConnected && (
        <div className="fixed left-1/2 top-4 z-50 -translate-x-1/2 rounded bg-yellow-100 px-4 py-2 text-sm font-medium text-yellow-800 shadow dark:bg-yellow-900/80 dark:text-yellow-200">
          Reconnecting to backend... some features may be unavailable
        </div>
      )}
      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        collapsed={sidebarCollapsed}
        onCollapseToggle={() => setSidebarCollapsed((c) => !c)}
        width={sidebarWidth}
        onWidthChange={(value) => {
          setSidebarWidth(clampWidth(value))
          setUserResized(true)
        }}
        minWidth={MIN_WIDTH}
        maxWidth={MAX_WIDTH}
      />

      {/* Main Content (grid column 2) */}
      <main className={`min-w-0 flex flex-col transition-all duration-300`}>
        <div className="flex items-center gap-3 border-b border-secondary-200 bg-white px-6 py-4 dark:border-secondary-800 dark:bg-secondary-900">
          {navigation.map((view) => (
            <button
              key={view.id}
              onClick={() => setActiveView(view.id)}
              className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                activeView === view.id
                  ? 'bg-primary-600 text-white shadow'
                  : 'bg-secondary-100 text-secondary-700 hover:bg-secondary-200 dark:bg-secondary-800 dark:text-secondary-300 dark:hover:bg-secondary-700'
              }`}
            >
              {view.label}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-auto bg-secondary-50 dark:bg-secondary-900">
          {activeView === 'graph' && <GraphView />}
          {activeView === 'chatTuning' && <ChatTuningPanel />}
          {activeView === 'classification' && <ClassificationPanel />}
          {activeView === 'chat' && <ChatInterface />}
          {activeView === 'document' && <DocumentView />}
        </div>
      </main>

      {/* Theme Toggle */}
      <ThemeToggle />
    </div>
  )
}
