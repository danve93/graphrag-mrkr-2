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
import BottomDock from '@/components/Navigation/BottomDock'
import { useEffect, useState } from 'react'
import { API_URL } from '@/lib/api'
import { useChatStore } from '@/store/chatStore'
import { useSwipeable } from 'react-swipeable'

export default function AdminPage() {
  const activeView = useChatStore((state) => state.activeView)
  const setActiveView = useChatStore((state) => state.setActiveView)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const isConnected = useChatStore((s) => s.isConnected)
  const setIsConnected = useChatStore((s) => s.setIsConnected)

  const [authenticated, setAuthenticated] = useState(false)
  const [loading, setLoading] = useState(true)
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)

  const DEFAULT_WIDTH = 280
  const MIN_WIDTH = 240
  const MAX_WIDTH = 400

  // Lazy initializer to avoid hydration mismatch
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    if (typeof window === 'undefined') return DEFAULT_WIDTH

    const storedWidth = window.localStorage.getItem('sidebar-width')
    if (storedWidth) {
      const parsed = parseInt(storedWidth, 10)
      if (!Number.isNaN(parsed)) {
        return Math.min(Math.max(parsed, MIN_WIDTH), MAX_WIDTH)
      }
    }

    const vw = Math.floor(window.innerWidth * 0.25)
    return Math.min(Math.max(vw, MIN_WIDTH), MAX_WIDTH)
  })

  const [userResized, setUserResized] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem('sidebar-width') !== null
  })

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

  // Check admin authentication on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const url = `${API_URL || ''}/api/admin/user-management/ping`
        const res = await fetch(url, { credentials: 'include' })
        if (!res.ok) {
          setAuthenticated(false)
          return
        }
        // Parse ping response to determine whether the request is actually
        // authenticated (valid session cookie or valid admin header).
        const body = await res.json().catch(() => ({}))
        const sessionValid = Boolean(body?.session_valid)
        const headerValid = Boolean(body?.header_valid)

        if (sessionValid || headerValid) {
          setAuthenticated(true)
        } else {
          // Not authenticated yet even though server ping succeeded.
          setAuthenticated(false)
        }
      } catch (e) {
        setAuthenticated(false)
      } finally {
        setLoading(false)
      }
    }
    checkAuth()
  }, [])

  const clampWidth = (next: number) => Math.min(Math.max(next, MIN_WIDTH), MAX_WIDTH)
  const mainMarginLeft = sidebarOpen ? (sidebarCollapsed ? 72 : sidebarWidth) : 0

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
    delta: 50,
  });

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    try {
      const url = `${API_URL || ''}/api/admin/user-management/login`
      const res = await fetch(url, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError(body?.detail || 'Login failed')
        return
      }
      setAuthenticated(true)
      setPassword('')
    } catch (err) {
      setError(String(err))
    }
  }

  const handleLogout = async () => {
    try {
      const url = `${API_URL || ''}/api/admin/user-management/logout`
      await fetch(url, { method: 'POST', credentials: 'include' })
    } catch (e) {
      // ignore
    }
    setAuthenticated(false)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-secondary-50 dark:bg-secondary-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Checking admin session…</p>
        </div>
      </div>
    )
  }

  if (!authenticated) {
    return (
      <div className="flex items-center justify-center h-screen bg-secondary-50 dark:bg-secondary-900">
        <div className="w-full max-w-md px-6 py-8 bg-white dark:bg-secondary-800 rounded-lg shadow-md">
          <h1 className="text-2xl font-bold text-center text-gray-900 dark:text-white mb-6">
            Admin Login
          </h1>
          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Admin Token
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter admin token"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
                           bg-white dark:bg-secondary-700 text-gray-900 dark:text-white
                           focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
            {error && (
              <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 
                            rounded text-sm text-red-700 dark:text-red-400">
                {error}
              </div>
            )}
            <button
              type="submit"
              className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white 
                         font-medium rounded-md transition-colors"
            >
              Log In
            </button>
          </form>
          <p className="mt-4 text-center text-xs text-gray-500 dark:text-gray-400">
            The admin token is sent securely and stored as an HttpOnly cookie.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div {...swipeHandlers} className="h-screen overflow-hidden bg-secondary-50 dark:bg-secondary-900">
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
        className="h-screen overflow-hidden transition-all duration-300 
                   bg-secondary-50 dark:bg-secondary-900 lg:ml-0"
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

      {/* Bottom Dock Navigation */}
      <BottomDock />
    </div>
  )
}
