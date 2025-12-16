'use client'

import { useCallback, useState } from 'react'
import type { MouseEvent as ReactMouseEvent, TouchEvent as ReactTouchEvent } from 'react'
import {
  Bars3Icon,
  XMarkIcon,
} from '@heroicons/react/24/outline'
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline'
import SidebarUser from './SidebarUser'
import SidebarHeader from './SidebarHeader'
import ChatSidebarContent from './ChatSidebarContent'
import DatabaseSidebarContent from './DatabaseSidebarContent'
import GenericSidebarContent from './GenericSidebarContent'
import DocumentationSidebarContent from './DocumentationSidebarContent'
import ChatTuningSidebarContent from './ChatTuningSidebarContent'
import RAGTuningSidebarContent from './RAGTuningSidebarContent'
import MetricsSidebarContent from './MetricsSidebarContent'
import CategoriesSidebarContent from './CategoriesSidebarContent'
import StatusIndicator from '@/components/Theme/StatusIndicator'
import { useChatStore } from '@/store/chatStore'
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
  minWidth = 240,
  maxWidth = 400,
  collapsed = false,
  onCollapseToggle = () => { },
}: SidebarProps) {
  const [isResizing, setIsResizing] = useState(false)
  const activeView = useChatStore((state) => state.activeView)



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
          className="fixed top-4 left-4 z-50 lg:hidden button-primary"
          style={{ minWidth: '44px', minHeight: '44px', padding: '10px' }}
          aria-label={open ? 'Close sidebar' : 'Open sidebar'}
        >
          {open ? <XMarkIcon className="w-6 h-6" /> : <Bars3Icon className="w-6 h-6" />}
        </button>
      </Tooltip>

      {/* Sidebar (fixed to left edge and separate from content grid) */}
      <aside
        className={`fixed left-0 top-0 z-40 h-screen border-r transition-transform duration-300 ${isResizing ? 'no-transition' : ''
          } ${open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'} overflow-hidden`}
        style={{
          width: `${collapsed ? 72 : width}px`,
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
              {/* Header (Logo/Brand) - Always visible */}
              <SidebarHeader />

              {/* Dynamic Content based on active view */}
              <div
                key={activeView}
                className="flex-1 min-h-0 flex flex-col animate-in fade-in duration-200"
              >
                {activeView === 'chat' && <ChatSidebarContent />}
                {activeView === 'document' && <DatabaseSidebarContent />}
                {activeView === 'graph' && (
                  <GenericSidebarContent
                    title="Graph Explorer"
                    description="Visualize entities and relationships in your knowledge graph"
                  />
                )}
                {activeView === 'categories' && (
                  <CategoriesSidebarContent
                    onSectionClick={(sectionId) => {
                      window.dispatchEvent(new CustomEvent('categories-section-select', { detail: sectionId }));
                    }}
                  />
                )}
                {activeView === 'structuredKg' && (
                  <GenericSidebarContent
                    title="Structured KG"
                    description="Execute Text-to-Cypher/SPARQL queries with entity linking"
                  />
                )}
                {activeView === 'ragTuning' && (
                  <RAGTuningSidebarContent
                    onSectionClick={(sectionId) => {
                      window.dispatchEvent(new CustomEvent('rag-tuning-section-select', { detail: sectionId }));
                    }}
                  />
                )}
                {activeView === 'chatTuning' && (
                  <ChatTuningSidebarContent
                    onSectionClick={(sectionId) => {
                      window.dispatchEvent(new CustomEvent('chat-tuning-section-select', { detail: sectionId }));
                    }}
                  />
                )}
                {activeView === 'documentation' && (
                  <DocumentationSidebarContent
                    onFileSelect={(path) => {
                      // Store selected file in a way the DocumentationView can access it
                      window.dispatchEvent(new CustomEvent('documentation-file-select', { detail: path }));
                    }}
                    selectedFile={null}
                  />
                )}
                {activeView === 'metrics' && (
                  <MetricsSidebarContent
                    onSectionClick={(sectionId) => {
                      window.dispatchEvent(new CustomEvent('metrics-section-select', { detail: sectionId }));
                    }}
                  />
                )}
              </div>

              {/* User Identity Section */}
              <SidebarUser />

              {/* Status Indicator at bottom */}
              <div
                className="border-t p-3"
                style={{
                  borderColor: 'var(--border)',
                  background: 'var(--bg-secondary)',
                  boxShadow: '0 -4px 12px rgba(0, 0, 0, 0.15)',
                  position: 'relative',
                  zIndex: 10
                }}
              >
                <StatusIndicator />
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
