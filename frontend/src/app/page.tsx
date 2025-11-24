'use client'

import ChatInterface from '@/components/Chat/ChatInterface'
import DocumentView from '@/components/Document/DocumentView'
import GraphView from '@/components/Graph/GraphView'
import Sidebar from '@/components/Sidebar/Sidebar'
import { ThemeToggle } from '@/components/Theme/ThemeToggle'
import { useEffect, useState } from 'react'
import { useChatStore } from '@/store/chatStore'

export default function Home() {
  const activeView = useChatStore((state) => state.activeView)
  const setActiveView = useChatStore((state) => state.setActiveView)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const DEFAULT_WIDTH = 320
  const MIN_WIDTH = 260
  const MAX_WIDTH = 480
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_WIDTH)
  const navigation = [
    { id: 'chat', label: 'Chat' },
    { id: 'document', label: 'Documents' },
    { id: 'graph', label: 'Graph' },
  ] as const

  useEffect(() => {
    const storedWidth = typeof window !== 'undefined' ? window.localStorage.getItem('sidebar-width') : null
    if (storedWidth) {
      const parsed = parseInt(storedWidth, 10)
      if (!Number.isNaN(parsed)) {
        setSidebarWidth(Math.min(Math.max(parsed, MIN_WIDTH), MAX_WIDTH))
      }
    }
  }, [])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem('sidebar-width', String(sidebarWidth))
    }
  }, [sidebarWidth])

  const clampWidth = (next: number) => Math.min(Math.max(next, MIN_WIDTH), MAX_WIDTH)

  return (
    <div className="flex h-screen bg-secondary-50 dark:bg-secondary-900">
      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        width={sidebarWidth}
  onWidthChange={(value) => setSidebarWidth(clampWidth(value))}
        minWidth={MIN_WIDTH}
        maxWidth={MAX_WIDTH}
      />

      {/* Main Content */}
      <main className={`flex-1 flex flex-col transition-all duration-300 ${sidebarOpen ? 'ml-0' : 'ml-0'}`}>
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
          {activeView === 'document' ? <DocumentView /> : activeView === 'graph' ? <GraphView /> : <ChatInterface />}
        </div>
      </main>

      {/* Theme Toggle */}
      <ThemeToggle />
    </div>
  )
}
