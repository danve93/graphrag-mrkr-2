'use client'

import { useCallback, useState } from 'react'
import type { MouseEvent as ReactMouseEvent, TouchEvent as ReactTouchEvent } from 'react'
import {
  Bars3Icon,
  XMarkIcon,
  ChatBubbleLeftIcon,
  CircleStackIcon,
  PlusCircleIcon,
} from '@heroicons/react/24/outline'
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline'
import HistoryTab from './HistoryTab'
import DatabaseTab from './DatabaseTab'
import { useChatStore } from '@/store/chatStore'
import { useBranding } from '@/components/Branding/BrandingProvider'
import Tooltip from '@/components/Utils/Tooltip'

interface SidebarProps {
  open: boolean
  onToggle: () => void
  width: number
  onWidthChange: (width: number) => void
  minWidth?: number
  maxWidth?: number
  collapsed?: boolean
  onCollapseToggle?: () => void
}

export default function Sidebar({
  open,
  onToggle,
  width,
  onWidthChange,
  minWidth = 260,
  maxWidth = 480,
  collapsed = false,
  onCollapseToggle = () => {},
}: SidebarProps) {
  const [activeTab, setActiveTab] = useState<'chat' | 'database'>('chat')
  const [isResizing, setIsResizing] = useState(false)
  const branding = useBranding()
  
  const isConnected = useChatStore((state) => state.isConnected)

  const tabs = [
    { id: 'chat' as const, label: 'Chat', icon: ChatBubbleLeftIcon },
    { id: 'database' as const, label: 'Database', icon: CircleStackIcon },
  ]

  const clearChat = useChatStore((state) => state.clearChat)
  const setActiveView = useChatStore((state) => state.setActiveView)

  const resizeWithinBounds = useCallback(
    (nextWidth: number) => {
      const clamped = Math.min(Math.max(nextWidth, minWidth), maxWidth)
      onWidthChange(clamped)
    },
    [maxWidth, minWidth, onWidthChange]
  )

  const handleResizeStart = useCallback(
    (startPosition: number) => {
      const startWidth = width

      const handlePointerMove = (eventPosition: number) => {
        const delta = eventPosition - startPosition
        resizeWithinBounds(startWidth + delta)
      }

      const handleMouseMove = (event: MouseEvent) => {
        handlePointerMove(event.clientX)
      }

      const handleTouchMove = (event: TouchEvent) => {
        if (event.touches.length > 0) {
          event.preventDefault()
          handlePointerMove(event.touches[0].clientX)
        }
      }

      const stopResizing = () => {
        setIsResizing(false)
        document.body.style.removeProperty('cursor')
        document.body.style.removeProperty('user-select')
        window.removeEventListener('mousemove', handleMouseMove)
        window.removeEventListener('touchmove', handleTouchMove)
        window.removeEventListener('mouseup', stopResizing)
        window.removeEventListener('touchend', stopResizing)
        window.removeEventListener('touchcancel', stopResizing)
      }

      setIsResizing(true)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('touchmove', handleTouchMove, { passive: false })
      window.addEventListener('mouseup', stopResizing)
      window.addEventListener('touchend', stopResizing)
      window.addEventListener('touchcancel', stopResizing)
    },
    [resizeWithinBounds, width]
  )

  const onMouseDown = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      event.preventDefault()
      handleResizeStart(event.clientX)
    },
    [handleResizeStart]
  )

  const onTouchStart = useCallback(
    (event: ReactTouchEvent<HTMLDivElement>) => {
      event.preventDefault()
      if (event.touches.length > 0) {
        handleResizeStart(event.touches[0].clientX)
      }
    },
    [handleResizeStart]
  )

  return (
    <>
      {/* Mobile toggle button */}
      <Tooltip content={open ? 'Close sidebar' : 'Open sidebar'}>
        <button
          onClick={onToggle}
          className="fixed top-4 left-4 z-50 lg:hidden button-primary p-2"
          aria-label={open ? 'Close sidebar' : 'Open sidebar'}
        >
          {open ? <XMarkIcon className="w-6 h-6" /> : <Bars3Icon className="w-6 h-6" />}
        </button>
      </Tooltip>

      {/* Sidebar (fixed to left edge and separate from content grid) */}
      <aside
        className={`fixed left-0 top-0 z-40 h-screen border-r ${
          isResizing ? 'no-transition' : ''
        } overflow-hidden`}
        style={{ 
          width: open ? `${collapsed ? 72 : width}px` : '0px',
          background: 'var(--bg-secondary)',
          borderColor: 'var(--border)'
        }}
      >
        <div className="flex flex-col h-full min-h-0">
          {/* Collapse button (desktop) */}
          <Tooltip content={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
            <button
              onClick={() => onCollapseToggle()}
              aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              className="hidden lg:flex absolute top-4 right-4 z-50 items-center justify-center p-2 rounded border"
              style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border)' }}
            >
              {collapsed ? (
                <ChevronRightIcon className="w-5 h-5" />
              ) : (
                <ChevronLeftIcon className="w-5 h-5" />
              )}
            </button>
          </Tooltip>

          {/* When collapsed we hide the rest of the content entirely */}
          {!collapsed && (
            <>
              {/* Logo/Brand */}
              <div className="p-6 border-b border-secondary-200 dark:border-secondary-700">
                <h1 className="text-lg branding-heading flex items-center">
                  {branding?.use_image && branding.image_path ? (
                    <img src={branding.image_path} alt={branding.short_name || branding.heading} className="w-6 h-6 mr-2" />
                  ) : null}
                  <span>{branding?.use_image ? (branding.short_name || branding.heading) : branding?.heading}</span>
                </h1>
                <p className="text-sm text-secondary-500 mt-1">{branding?.tagline}</p>
              </div>

              {/* Tabs */}
              <div className={`flex border-b border-secondary-200 dark:border-secondary-700 transition-all duration-300 ${
                !isConnected ? 'blur-sm pointer-events-none' : ''
              }`}>
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex-1 flex items-center justify-center py-3 text-sm font-medium transition-colors ${
                      activeTab === tab.id ? '' : 'text-secondary-500 hover:text-secondary-200'
                    }`}
                    style={
                      activeTab === tab.id
                        ? { color: 'var(--accent-primary)', borderBottom: '2px solid var(--accent-primary)' }
                        : undefined
                    }
                  >
                    <tab.icon className="w-5 h-5 mr-1 text-current" />
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* New Chat button in sidebar, shown under tabs when Chat tab is active */}
              {activeTab === 'chat' && (
                <div className="px-6 py-3 border-b border-secondary-200 dark:border-secondary-700">
                  <button
                    type="button"
                    onClick={() => {
                      clearChat()
                      setActiveView('chat')
                      setActiveTab('chat')
                    }}
                    className="w-full button-primary"
                    aria-label="Start a new chat"
                  >
                    <PlusCircleIcon className="w-5 h-5" />
                    New Chat
                  </button>
                </div>
              )}

              {/* Tab Content */}
              <div className={`flex-1 overflow-y-auto overscroll-contain p-6 transition-all duration-300 ${
                !isConnected ? 'blur-sm pointer-events-none' : ''
              }`}>
                <div key={activeTab} className="tab-content">
                  {activeTab === 'chat' && <HistoryTab />}
                  {activeTab === 'database' && <DatabaseTab />}
                </div>
              </div>

              {/* Additional Panels Navigation */}
              <div className="border-t border-secondary-200 dark:border-secondary-700 p-4">
                <p className="text-xs font-semibold text-secondary-500 dark:text-secondary-400 mb-2 px-2">TOOLS</p>
                <div className="space-y-1">
                  <button
                    onClick={() => setActiveView('graph')}
                    className="w-full text-left px-3 py-2 text-sm rounded"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    Graph Explorer
                  </button>
                  <button
                    onClick={() => setActiveView('chatTuning')}
                    className="w-full text-left px-3 py-2 text-sm rounded"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    Chat Tuning
                  </button>
                  <button
                    onClick={() => setActiveView('ragTuning')}
                    className="w-full text-left px-3 py-2 text-sm rounded"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    RAG Tuning
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Resize handle (hidden when collapsed or when sidebar closed) */}
        {open && !collapsed && (
          <div
                className="hidden lg:flex absolute top-0 -right-2 h-full w-4 items-center justify-center cursor-col-resize"
                style={{ zIndex: 60, backgroundColor: isResizing ? 'var(--accent-subtle)' : 'transparent' }}
              onMouseDown={onMouseDown}
              onTouchStart={onTouchStart}
            >
            <div style={{ height: '64px', width: '4px', borderRadius: 'var(--radius-full)', background: 'var(--gray-400)' }} />
          </div>
        )}
      </aside>

      {/* Overlay for mobile */}
      {open && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={onToggle}
        />
      )}
    </>
  )
}
