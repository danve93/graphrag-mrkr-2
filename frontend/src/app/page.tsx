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
import MetricsPanel from '@/components/Metrics/MetricsPanel'
import AdminApiKeys from '@/components/Admin/AdminApiKeys'
import AdminSharedChats from '@/components/Admin/AdminSharedChats'
import Sidebar from '@/components/Sidebar/Sidebar'
import ViewTransition from '@/components/Utils/ViewTransition'
import ExternalChatBubble from '@/components/Chat/ExternalChatBubble'
import { useEffect, useState } from 'react'
import { useChatStore } from '@/store/chatStore'
import { api } from '@/lib/api'
import { useSwipeable } from 'react-swipeable'

export default function Home() {
  const activeView = useChatStore((state) => state.activeView)
  const setActiveView = useChatStore((state) => state.setActiveView)
  const user = useChatStore((state) => state.user)
  const identifyUser = useChatStore((state) => state.identifyUser)

  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(280)
  const [userResized, setUserResized] = useState(false)
  const [isAuthChecking, setIsAuthChecking] = useState(true)
  const [forceExternalView, setForceExternalView] = useState(false)

  const MIN_WIDTH = 240
  const MAX_WIDTH = 400

  // Check for force external View (for testing/demo)
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search)
      if (params.get('view') === 'external') {
        setForceExternalView(true)
        setIsAuthChecking(false)
      }
    }
  }, [])

  const clampWidth = (next: number) => Math.min(Math.max(next, MIN_WIDTH), MAX_WIDTH)
  const mainMarginLeft = sidebarOpen ? (sidebarCollapsed ? 72 : sidebarWidth) : 0

  const swipeHandlers = useSwipeable({
    onSwipedRight: () => {
      if (window.innerWidth < 1024 && !sidebarOpen) setSidebarOpen(true);
    },
    onSwipedLeft: () => {
      if (window.innerWidth < 1024 && sidebarOpen) setSidebarOpen(false);
    },
    trackMouse: false,
    trackTouch: true,
    delta: 50,
  });

  // Handle Authentication & Routing
  useEffect(() => {
    // If forcing external view, skip auth check
    if (forceExternalView) {
      setIsAuthChecking(false)
      return
    }

    const checkAuth = async () => {
      // If we already have a user in store, we are good to go
      if (user) {
        setIsAuthChecking(false)
        return
      }

      // Try to load from local storage via store initialization (handled in store)
      // If still no user, we check if there's a token in API client
      // But actually, chatStore handles hydration.

      // If no user is found after a short delay (to allow hydration), redirect or show login
      // However, for now, let's assume if 'chatUser' is in localStorage, store picks it up.

      // If we strictly want to force login for everyone:
      const storedUser = localStorage.getItem('chatUser')
      if (storedUser) {
        // It will be hydrated by store. 
        setIsAuthChecking(false)
      } else {
        // No user found. Redirect to admin login.
        // But wait, external users might arrive via a share link with a token/key?
        // For now, implementing the requirement: "Unauthenticated => /admin"
        // We'll give it a moment to hydrate.

        // Actually, we can just check if user is null *after* hydration.
        // But Zustand hydration is synchronous usually if from localStorage, 
        // or we can just wait for 'user' to be populated.

        setIsAuthChecking(false)
      }
    }

    checkAuth()
  }, [user, forceExternalView])

  // Effect to redirect if not authenticated
  useEffect(() => {
    if (!isAuthChecking && !user && !forceExternalView) {
      window.location.href = '/admin'
    }
  }, [isAuthChecking, user, forceExternalView])


  if (isAuthChecking && !forceExternalView) {
    return <div className="flex items-center justify-center h-screen bg-gray-50 dark:bg-neutral-900 text-gray-400">Loading Amber...</div>
  }

  // Redirecting...
  if (!user && !forceExternalView) {
    return null
  }

  // External User View
  if (user?.role === 'external') {
    return <ExternalChatBubble />
  }

  // Admin View (Full Dashboard)
  // Ensure we show dashboard only if admin
  if (user?.role !== 'admin') {
    return <ExternalChatBubble />
  }

  return (
    <div {...swipeHandlers} className="h-screen overflow-hidden bg-secondary-50 dark:bg-secondary-900">
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
          {activeView === 'metrics' && <MetricsPanel />}
          {activeView === 'adminApiKeys' && <AdminApiKeys />}
          {activeView === 'adminSharedChats' && <AdminSharedChats />}
        </ViewTransition>
      </main>
    </div>
  )
}
