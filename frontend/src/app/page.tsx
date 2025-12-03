'use client'

import ChatInterface from '@/components/Chat/ChatInterface'
import DocumentView from '@/components/Document/DocumentView'
import GraphView from '@/components/Graph/GraphView'
import ChatTuningPanel from '@/components/ChatTuning/ChatTuningPanel'
import RAGTuningPanel from '@/components/RAGTuning/RAGTuningPanel'
import CategoriesView from '@/components/Categories/CategoriesView'
import RoutingView from '@/components/Routing/RoutingView'
import StructuredKgView from '@/components/StructuredKg/StructuredKgView'
import DocumentationView from '@/components/Documentation/DocumentationView'
import Sidebar from '@/components/Sidebar/Sidebar'
import ViewTransition from '@/components/Utils/ViewTransition'
import { useEffect, useState } from 'react'
import { useChatStore } from '@/store/chatStore'
import { api } from '@/lib/api'
import { useSwipeable } from 'react-swipeable'

export default function Home() {
  const activeView = useChatStore((state) => state.activeView)
  const setActiveView = useChatStore((state) => state.setActiveView)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const isConnected = useChatStore((s) => s.isConnected)
  const setIsConnected = useChatStore((s) => s.setIsConnected)
  const DEFAULT_WIDTH = 280
  const MIN_WIDTH = 240
  const MAX_WIDTH = 400
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

  // Swipe gestures for mobile
  const swipeHandlers = useSwipeable({
    onSwipedRight: () => {
      if (window.innerWidth < 1024 && !sidebarOpen) {
        setSidebarOpen(true);
      }
    },
    onSwipedLeft: () => {
      if (window.innerWidth < 1024 && sidebarOpen) {
        setSidebarOpen(false);
      }
    },
    trackMouse: false,
    trackTouch: true,
    delta: 50, // Minimum swipe distance in pixels
  });

  return (
    <div {...swipeHandlers} className="h-screen overflow-hidden bg-secondary-50 dark:bg-secondary-900">
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
        className="h-screen overflow-hidden transition-all duration-300 bg-secondary-50 dark:bg-secondary-900 lg:ml-0"
        style={{ marginLeft: mainMarginLeft > 0 ? `${mainMarginLeft}px` : '0' }}
      >
        <ViewTransition viewKey={activeView}>
          {activeView === 'chat' && <ChatInterface />}
          {activeView === 'document' && <DocumentView />}
          {activeView === 'graph' && <GraphView />}
          {activeView === 'categories' && <CategoriesView />}
          {activeView === 'routing' && <RoutingView />}
          {activeView === 'structuredKg' && <StructuredKgView />}
          {activeView === 'ragTuning' && <RAGTuningPanel />}
          {activeView === 'chatTuning' && <ChatTuningPanel />}
          {activeView === 'documentation' && <DocumentationView />}
        </ViewTransition>
      </main>
    </div>
  )
}
