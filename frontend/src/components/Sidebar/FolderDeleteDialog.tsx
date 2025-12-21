'use client'

import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { X, Folder, AlertTriangle } from 'lucide-react'
import type { FolderSummary } from '@/types'

interface FolderDeleteDialogProps {
  folder: FolderSummary
  onClose: () => void
  onMoveToRoot: () => void
  onDeleteDocuments: () => void
  disableDeleteDocuments?: boolean
}

export default function FolderDeleteDialog({
  folder,
  onClose,
  onMoveToRoot,
  onDeleteDocuments,
  disableDeleteDocuments = false,
}: FolderDeleteDialogProps) {
  const [mounted, setMounted] = useState(false)
  const documentCount = folder.document_count ?? 0

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center font-sans"
      style={{ backgroundColor: 'rgba(0, 0, 0, 0.6)' }}
      onClick={onClose}
    >
      <div
        className="w-full max-w-md mx-4 rounded-lg shadow-xl overflow-hidden"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border)',
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <div
          className="flex items-center justify-between px-5 py-4"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <div className="flex items-center gap-2">
            <Folder className="w-5 h-5" style={{ color: 'var(--accent-primary)' }} />
            <h2 className="font-display text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
              Delete folder
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-secondary-700"
            style={{ color: 'var(--text-secondary)' }}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4">
          <div
            className="flex items-center gap-3 p-3 rounded-lg"
            style={{ backgroundColor: 'var(--bg-tertiary)' }}
          >
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                {folder.name}
              </p>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                {documentCount} document{documentCount === 1 ? '' : 's'} inside
              </p>
            </div>
          </div>

          <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Choose what should happen to the documents in this folder.
          </div>

          <div className="flex flex-col gap-2">
            <button
              type="button"
              onClick={onMoveToRoot}
              className="button-primary text-sm"
            >
              Move documents to root
            </button>
            <button
              type="button"
              onClick={onDeleteDocuments}
              disabled={disableDeleteDocuments}
              className="button-secondary text-sm text-red-600 border-red-200 hover:text-red-700 disabled:opacity-50"
            >
              Delete documents
            </button>
            <button
              type="button"
              onClick={onClose}
              className="button-ghost text-sm"
            >
              Cancel
            </button>
          </div>

          {disableDeleteDocuments && (
            <p className="text-xs text-secondary-500">
              Document deletion is disabled in this environment.
            </p>
          )}
        </div>
      </div>
    </div>,
    document.body
  )
}
