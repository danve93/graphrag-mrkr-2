'use client'

import ChatInterface from '@/components/Chat/ChatInterface'
import DocumentView from '@/components/Document/DocumentView'
import GraphView from '@/components/Graph/GraphView'
import ComblocksPanel from '@/components/Comblocks/ComblocksPanel'
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

  // Health check is now managed by ChatInterface to avoid race conditions
  // Initial connection state is optimistic (true) until proven otherwise

  const clampWidth = (next: number) => Math.min(Math.max(next, MIN_WIDTH), MAX_WIDTH)

  const mainMarginLeft = sidebarOpen ? (sidebarCollapsed ? 72 : sidebarWidth) : 0

  return (
    <div className="h-screen overflow-hidden bg-secondary-50 dark:bg-secondary-900">
      {/* Connection status is shown in the chat UI via `ConnectionStatus` component */}
      {/* Sidebar (fixed to left edge) */}
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
      {/* Main Content (offset by sidebar) */}
      <main
        className="h-full overflow-hidden transition-all duration-300 bg-secondary-50 dark:bg-secondary-900"
        style={{ marginLeft: `${mainMarginLeft}px` }}
      >
        {activeView === 'graph' && <GraphView />}
        {activeView === 'chatTuning' && <ChatTuningPanel />}
        {activeView === 'classification' && <ClassificationPanel />}
        {activeView === 'comblocks' && <ComblocksPanel />}
        {activeView === 'chat' && <ChatInterface />}
        {activeView === 'document' && <DocumentView />}
      </main>

      {/* Theme Toggle */}
      <ThemeToggle />
    </div>
  )
}
