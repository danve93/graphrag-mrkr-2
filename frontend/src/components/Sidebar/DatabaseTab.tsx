'use client'

import { useEffect, useState, useRef } from 'react'
import { api } from '@/lib/api'
import { DatabaseStats, ProcessingSummary, FolderSummary, DocumentSummary } from '@/types'
import {
  Trash2,
  FileUp,
  Search,
  X,
  Folder,
  Plus,
  ChevronUp,
  ChevronDown,
} from 'lucide-react'
import Loader from '@/components/Utils/Loader'
import { useChatStore } from '@/store/chatStore'
import { showToast } from '@/components/Toast/ToastContainer'
import UploadUpdateDialog from '@/components/Document/UploadUpdateDialog'
import FolderDeleteDialog from './FolderDeleteDialog'
import ClearDatabaseDialog from './ClearDatabaseDialog'

export default function DatabaseTab() {
  const [stats, setStats] = useState<DatabaseStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [processingState, setProcessingState] = useState<ProcessingSummary | null>(null)
  const [isStuck, setIsStuck] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [draggingDocumentId, setDraggingDocumentId] = useState<string | null>(null)
  const [draggingOverFolderId, setDraggingOverFolderId] = useState<string | null>(null)
  const [uploadingFile, setUploadingFile] = useState(false)
  const [searchMode, setSearchMode] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [enableDeleteOps, setEnableDeleteOps] = useState(true)
  const [folders, setFolders] = useState<FolderSummary[]>([])
  const [activeFolderId, setActiveFolderId] = useState<string>('all')
  const [sortMode, setSortMode] = useState<'newest' | 'oldest' | 'name' | 'manual'>('newest')
  const [isCreatingFolder, setIsCreatingFolder] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [folderDeleteTarget, setFolderDeleteTarget] = useState<FolderSummary | null>(null)
  const [isReordering, setIsReordering] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const wasProcessingRef = useRef(false)
  const lastUpdateTimestampRef = useRef<number>(Date.now())
  const selectDocument = useChatStore((state) => state.selectDocument)
  const clearSelectedDocument = useChatStore((state) => state.clearSelectedDocument)
  const selectedDocumentId = useChatStore((state) => state.selectedDocumentId)

  // Upload dialog state
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [pendingFolderId, setPendingFolderId] = useState<string | null>(null)
  const [showUploadDialog, setShowUploadDialog] = useState(false)
  const [showClearDialog, setShowClearDialog] = useState(false)

  const normalizeFolderId = (folderId?: string | null) => {
    if (!folderId || folderId === 'root' || folderId === 'all') return null
    return folderId
  }

  const handleFiles = async (files: FileList | File[], targetFolderId?: string | null) => {
    const fileArray = Array.from(files)
    const normalizedFolderId = normalizeFolderId(targetFolderId)

    // For single file uploads, show the dialog
    if (fileArray.length === 1) {
      setPendingFile(fileArray[0])
      setPendingFolderId(normalizedFolderId)
      setShowUploadDialog(true)
      return
    }

    // For multiple files, upload directly as new
    setUploadingFile(true)
    try {
      for (const file of fileArray) {
        const staged = await api.stageFile(file)
        if (normalizedFolderId && staged?.document_id) {
          await api.moveDocumentToFolder(staged.document_id, normalizedFolderId)
        }
        showToast('success', `${file.name} uploaded`, 'Document queued for processing')
      }

      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:uploaded'))
      }
    } catch (error) {
      console.error('Failed to stage file:', error)
      showToast('error', 'Upload failed', error instanceof Error ? error.message : 'Failed to upload file')
    } finally {
      setUploadingFile(false)
    }
  }

  // Handle dialog: upload as new
  const handleUploadNew = async () => {
    if (!pendingFile) return
    setShowUploadDialog(false)
    setUploadingFile(true)
    try {
      const staged = await api.stageFile(pendingFile)
      if (pendingFolderId && staged?.document_id) {
        await api.moveDocumentToFolder(staged.document_id, pendingFolderId)
      }
      showToast('success', `${pendingFile.name} uploaded`, 'Document queued for processing')
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:uploaded'))
      }
    } catch (error) {
      showToast('error', 'Upload failed', error instanceof Error ? error.message : 'Failed to upload file')
    } finally {
      setUploadingFile(false)
      setPendingFile(null)
      setPendingFolderId(null)
    }
  }

  // Handle dialog: update existing document
  const handleUpdateExisting = async (documentId: string) => {
    if (!pendingFile) return
    setShowUploadDialog(false)
    setUploadingFile(true)
    try {
      const result = await api.updateDocument(documentId, pendingFile)
      if (result.status === 'success' && result.changes) {
        const { unchanged_chunks, added_chunks, removed_chunks } = result.changes
        showToast(
          'success',
          `${pendingFile.name} updated`,
          `${unchanged_chunks} chunks unchanged, ${added_chunks} added, ${removed_chunks} removed`
        )
      } else if (result.error) {
        showToast('error', 'Update failed', result.details || result.error)
      }
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:uploaded'))
      }
    } catch (error) {
      showToast('error', 'Update failed', error instanceof Error ? error.message : 'Failed to update document')
    } finally {
      setUploadingFile(false)
      setPendingFile(null)
      setPendingFolderId(null)
    }
  }

  // Handle dialog close
  const handleDialogClose = () => {
    setShowUploadDialog(false)
    setPendingFile(null)
    setPendingFolderId(null)
  }

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    await handleFiles(files, activeFolderId)
    e.target.value = ''
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    const hasFiles = Array.from(e.dataTransfer.types || []).includes('Files')
    if (hasFiles) {
      setIsDragging(true)
    }
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    setDraggingDocumentId(null)
    setDraggingOverFolderId(null)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      await handleFiles(files, activeFolderId)
    }
  }

  useEffect(() => {
    loadStats()
    loadFolders()

    // Listen for processing completion events to refresh stats automatically
    const handler = () => {
      loadStats()
      loadFolders()
    }

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
      window.addEventListener('documents:processed', handler)
      window.addEventListener('documents:processing-updated', handler)
      window.addEventListener('documents:uploaded', handler)
      window.addEventListener('documents:metadata-updated', handler)
      window.addEventListener('server:reconnected', handler)
    }

    return () => {
      if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
        window.removeEventListener('documents:processed', handler)
        window.removeEventListener('documents:processing-updated', handler)
        window.removeEventListener('documents:uploaded', handler)
        window.removeEventListener('documents:metadata-updated', handler)
        window.removeEventListener('server:reconnected', handler)
      }
    }
  }, [])

  // Focus search input when entering search mode
  useEffect(() => {
    if (searchMode && searchInputRef.current) {
      searchInputRef.current.focus()
    }
  }, [searchMode])

  useEffect(() => {
    if (activeFolderId === 'all' && sortMode === 'manual') {
      setSortMode('newest')
    }
  }, [activeFolderId, sortMode])

  // Poll for processing updates when active (without refreshing entire stats)
  useEffect(() => {
    const isProcessing = processingState?.is_processing || stats?.processing?.is_processing
    if (!isProcessing) return

    const interval = setInterval(async () => {
      try {
        const response = await api.getProcessingProgress()

        // Check if ALL processing finished (no pending documents and is_processing is false)
        const hasPendingDocs = response.global.pending_documents && response.global.pending_documents.length > 0
        const isStillProcessing = response.global.is_processing || hasPendingDocs

        if (!isStillProcessing) {
          // Clear processing state immediately
          setProcessingState(null)
          setIsStuck(false)
          wasProcessingRef.current = false
          // Force a final stats refresh
          await loadStats()
          return
        }

        setProcessingState(response.global)

        // Update timestamp on successful response with active progress
        lastUpdateTimestampRef.current = Date.now()
        setIsStuck(false)

        // Also fetch updated stats to get real-time counts AND document_type updates
        const updatedStats = await api.getStats()

        // Update document progress in stats if we have them
        if (stats && response.global.pending_documents) {
          const updatedDocs = stats.documents.map(doc => {
            const progressMatch = response.global.pending_documents.find(
              p => p.document_id === doc.document_id
            )
            // Always get fresh data from updatedStats to ensure document_type and other fields are current
            const freshDoc = updatedStats.documents.find((d: any) => d.document_id === doc.document_id)
            if (progressMatch) {
              return {
                ...freshDoc,
                processing_status: progressMatch.status,
                processing_stage: progressMatch.stage || doc.processing_stage,
                processing_progress: progressMatch.progress_percentage,
                queue_position: progressMatch.queue_position,
                chunk_progress: progressMatch.chunk_progress,
                entity_progress: progressMatch.entity_progress,
                chunks_processed: progressMatch.chunks_processed,
                total_chunks: progressMatch.total_chunks
              }
            }
            return freshDoc || doc
          })

          // Merge updated documents with their progress AND update global stats
          setStats({
            ...updatedStats,
            documents: updatedDocs,
            processing: response.global
          })
        } else {
          // Just update with fresh stats if no processing documents
          setStats({ ...updatedStats, processing: response.global })
        }
      } catch (error) {
        console.error('Failed to poll processing state:', error)
      }
    }, 1500) // Poll every 1.5s during processing

    return () => clearInterval(interval)
  }, [processingState?.is_processing, stats?.processing?.is_processing, stats])

  // Detect stuck/stale processing (server crash)
  useEffect(() => {
    const isProcessing = processingState?.is_processing || stats?.processing?.is_processing
    if (!isProcessing) return

    const checkStuck = setInterval(() => {
      const timeSinceUpdate = Date.now() - lastUpdateTimestampRef.current
      const STUCK_THRESHOLD = 30000 // 30 seconds

      if (timeSinceUpdate > STUCK_THRESHOLD) {
        console.warn('Processing appears stuck - no updates for 30s')
        setIsStuck(true)
      }
    }, 5000) // Check every 5 seconds

    return () => clearInterval(checkStuck)
  }, [processingState?.is_processing, stats?.processing?.is_processing])

  // Detect when processing completes and do a final refresh
  useEffect(() => {
    const isProcessing = processingState?.is_processing || stats?.processing?.is_processing || false

    // If we were processing before but not anymore, refresh everything and clear state
    if (wasProcessingRef.current && !isProcessing) {
      loadStats()
      // Clear processing state to hide progress indicators
      setTimeout(() => {
        setProcessingState(null)
        setIsStuck(false)
      }, 1000) // Small delay to ensure final stats are loaded
    }

    // Update the ref for next check
    wasProcessingRef.current = isProcessing
  }, [processingState?.is_processing, stats?.processing?.is_processing])

  const loadStats = async () => {
    try {
      setLoading(true)
      const data = await api.getStats()
      setStats(data)
      setProcessingState(data.processing || null)
      setEnableDeleteOps(data.enable_delete_operations ?? true)
      return data
    } catch (error) {
      console.error('Failed to load stats:', error)
      return null
    } finally {
      setLoading(false)
    }
  }

  const loadFolders = async () => {
    try {
      const data = await api.getFolders()
      const nextFolders = data.folders || []
      setFolders(nextFolders)
      if (
        activeFolderId !== 'all' &&
        activeFolderId !== 'root' &&
        !nextFolders.some((folder: FolderSummary) => folder.id === activeFolderId)
      ) {
        setActiveFolderId('all')
      }
      return data
    } catch (error) {
      console.error('Failed to load folders:', error)
      return null
    }
  }

  const handleDeleteDocument = async (documentId: string) => {
    if (!confirm('Delete this document and all its chunks?')) return

    try {
      await api.deleteDocument(documentId)
      const newStats = await loadStats()
      await loadFolders()

      // If the deleted document was selected, switch selection to next available or fallback to chat
      if (selectedDocumentId === documentId) {
        const visibleDocs = newStats?.documents ? filterDocuments(newStats.documents) : []
        if (visibleDocs.length > 0) {
          selectDocument(visibleDocs[0].document_id)
        } else {
          clearSelectedDocument()
        }
      }
    } catch (error) {
      console.error('Failed to delete document:', error)
    }
  }

  const handleSelectDocument = (documentId: string) => {
    selectDocument(documentId)
  }

  const handleClearDatabase = () => {
    setShowClearDialog(true)
  }

  const performClearDatabase = async (options: { clearKnowledgeBase: boolean; clearConversations: boolean }) => {
    try {
      setShowClearDialog(false)
      await api.clearDatabase(options)
      await loadStats()
      await loadFolders()
      setActiveFolderId('all')
      // After clearing the database, ensure the UI returns to chat view
      clearSelectedDocument()
      showToast('success', 'Database cleared successfully')
    } catch (error) {
      console.error('Failed to clear database:', error)
      showToast('error', 'Failed to clear database')
    }
  }

  const handleClearStuckState = async () => {
    setIsStuck(false)
    setProcessingState(null)
    lastUpdateTimestampRef.current = Date.now()
    wasProcessingRef.current = false
    await loadStats()
  }

  const handleCleanupOrphans = async () => {
    if (!confirm('Clean up orphaned chunks and entities? This will delete data not connected to any document.')) return

    try {
      const response = await api.cleanupOrphans()
      const chunksDeleted = response.chunks_deleted || 0
      const entitiesDeleted = response.entities_deleted || 0

      if (chunksDeleted > 0 || entitiesDeleted > 0) {
        showToast('success', `Cleaned up ${chunksDeleted} orphaned chunks and ${entitiesDeleted} orphaned entities`)
      } else {
        showToast('success', 'No orphaned data found')
      }

      await loadStats()
    } catch (error) {
      console.error('Failed to cleanup orphans:', error)
      showToast('error', 'Failed to cleanup orphans')
    }
  }

  const handleCreateFolder = async () => {
    const trimmed = newFolderName.trim()
    if (!trimmed) {
      showToast('error', 'Folder name required', 'Enter a unique folder name')
      return
    }

    try {
      const folder = await api.createFolder(trimmed)
      await loadFolders()
      setActiveFolderId(folder.id)
      setIsCreatingFolder(false)
      setNewFolderName('')
      showToast('success', `Folder "${folder.name}" created`)
    } catch (error) {
      showToast('error', 'Failed to create folder', error instanceof Error ? error.message : 'Unknown error')
    }
  }

  const handleDeleteFolder = async (mode: 'move_to_root' | 'delete_documents') => {
    if (!folderDeleteTarget) return

    try {
      const result = await api.deleteFolder(folderDeleteTarget.id, mode)
      const updatedStats = await loadStats()
      await loadFolders()
      if (activeFolderId === folderDeleteTarget.id) {
        setActiveFolderId('all')
      }
      if (selectedDocumentId) {
        const stillExists = updatedStats?.documents?.some(
          (doc: DocumentSummary) => doc.document_id === selectedDocumentId
        )
        if (!stillExists) {
          clearSelectedDocument()
        }
      }
      const deletedCount = result?.documents_deleted || 0
      const movedCount = result?.documents_moved || 0
      if (mode === 'delete_documents') {
        showToast('success', `Folder deleted`, `${deletedCount} documents removed`)
      } else {
        showToast('success', `Folder deleted`, `${movedCount} documents moved to root`)
      }
      setFolderDeleteTarget(null)
    } catch (error) {
      showToast('error', 'Failed to delete folder', error instanceof Error ? error.message : 'Unknown error')
    }
  }

  const handleMoveDocumentToFolder = async (documentId: string, targetFolderId: string | null) => {
    try {
      const normalizedFolderId = normalizeFolderId(targetFolderId)
      const currentFolderId = stats?.documents?.find((doc) => doc.document_id === documentId)?.folder_id || null
      if (normalizedFolderId === currentFolderId) {
        return
      }
      await api.moveDocumentToFolder(documentId, normalizedFolderId)
      await loadStats()
      await loadFolders()
      const folderName =
        normalizedFolderId ? folders.find((folder) => folder.id === normalizedFolderId)?.name : null
      showToast(
        'success',
        'Document moved',
        normalizedFolderId ? `Moved to ${folderName || 'folder'}` : 'Moved to root'
      )
    } catch (error) {
      showToast('error', 'Failed to move document', error instanceof Error ? error.message : 'Unknown error')
    }
  }

  const handleFolderDragOver = (event: React.DragEvent, folderId: string) => {
    event.preventDefault()
    event.stopPropagation()
    setDraggingOverFolderId(folderId)
  }

  const handleFolderDragLeave = (event: React.DragEvent) => {
    event.preventDefault()
    event.stopPropagation()
    setDraggingOverFolderId(null)
  }

  const handleFolderDrop = async (event: React.DragEvent, folderId: string) => {
    event.preventDefault()
    event.stopPropagation()
    setDraggingOverFolderId(null)

    const files = event.dataTransfer.files
    if (files && files.length > 0) {
      await handleFiles(files, folderId)
      return
    }

    if (draggingDocumentId) {
      await handleMoveDocumentToFolder(draggingDocumentId, folderId)
      setDraggingDocumentId(null)
    }
  }

  const handleReorderDocument = async (documentId: string, direction: 'up' | 'down') => {
    if (sortMode !== 'manual' || activeFolderId === 'all' || searchQuery.trim()) return

    const documents = getFilteredDocuments()
    const index = documents.findIndex((doc) => doc.document_id === documentId)
    if (index === -1) return

    const targetIndex = direction === 'up' ? index - 1 : index + 1
    if (targetIndex < 0 || targetIndex >= documents.length) return

    const reordered = [...documents]
    const [moved] = reordered.splice(index, 1)
    reordered.splice(targetIndex, 0, moved)

    const orderedIds = reordered.map((doc) => doc.document_id)
    const orderMap = new Map(orderedIds.map((id, idx) => [id, idx]))

    try {
      setIsReordering(true)
      await api.reorderDocuments(normalizeFolderId(activeFolderId), orderedIds)
      setStats((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          documents: prev.documents.map((doc) => ({
            ...doc,
            folder_order: orderMap.has(doc.document_id) ? orderMap.get(doc.document_id) : doc.folder_order,
          })),
        }
      })
    } catch (error) {
      showToast('error', 'Failed to reorder', error instanceof Error ? error.message : 'Unknown error')
    } finally {
      setIsReordering(false)
    }
  }

  const getDocumentLabel = (doc: any) =>
    doc.title || doc.original_filename || doc.filename || `Unnamed Document (${doc.document_id?.substring(0, 8)}...)`

  const getDocumentCreatedAt = (doc: any) => {
    if (!doc.created_at) return 0
    if (typeof doc.created_at === 'number') {
      return doc.created_at > 1_000_000_000_000 ? doc.created_at : doc.created_at * 1000
    }
    const parsed = Date.parse(doc.created_at)
    return Number.isNaN(parsed) ? 0 : parsed
  }

  const sortDocuments = (documents: any[]) => {
    const sorted = [...documents]
    if (sortMode === 'name') {
      sorted.sort((a, b) => getDocumentLabel(a).localeCompare(getDocumentLabel(b)))
      return sorted
    }

    if (sortMode === 'oldest') {
      sorted.sort((a, b) => getDocumentCreatedAt(a) - getDocumentCreatedAt(b))
      return sorted
    }

    if (sortMode === 'manual') {
      sorted.sort((a, b) => {
        const orderA = typeof a.folder_order === 'number' ? a.folder_order : Number.POSITIVE_INFINITY
        const orderB = typeof b.folder_order === 'number' ? b.folder_order : Number.POSITIVE_INFINITY
        if (orderA !== orderB) return orderA - orderB
        return getDocumentCreatedAt(b) - getDocumentCreatedAt(a)
      })
      return sorted
    }

    sorted.sort((a, b) => getDocumentCreatedAt(b) - getDocumentCreatedAt(a))
    return sorted
  }

  const filterDocuments = (documents: any[]) => {
    let filtered = documents

    if (activeFolderId !== 'all') {
      if (activeFolderId === 'root') {
        filtered = filtered.filter((doc) => !doc.folder_id)
      } else {
        filtered = filtered.filter((doc) => doc.folder_id === activeFolderId)
      }
    }

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(
        (doc) =>
          getDocumentLabel(doc).toLowerCase().includes(query) ||
          (doc.document_type || '').toLowerCase().includes(query)
      )
    }

    return sortDocuments(filtered)
  }

  const getFilteredDocuments = () => {
    if (!stats?.documents) return []
    return filterDocuments(stats.documents)
  }

  if (loading) {
    return <div className="text-center text-secondary-600 dark:text-secondary-400">Loading...</div>
  }

  // Use processingState which gets updated via polling, or fallback to stats.processing
  const processingSummary = processingState || stats?.processing

  const formatStatus = (doc: any) => {
    const status = doc.processing_status
    const stage = doc.processing_stage
    const progress = typeof doc.processing_progress === 'number' ? Math.round(doc.processing_progress) : null

    if (status === 'processing') {
      return null
    }
    if (status === 'queued') {
      return 'Processing queued'
    }
    if (status === 'staged') {
      return 'Ready to process'
    }
    if (status === 'error') return 'Needs attention'
    return null
  }

  const filteredDocuments = getFilteredDocuments()
  const isManualReorder = sortMode === 'manual' && activeFolderId !== 'all'
  const reorderDisabled = isReordering || !!searchQuery.trim()

  return (
    <div
      className={`relative transition-all ${isDragging ? 'is-dragging-ring rounded-lg' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDragging && (
        <div
          className="absolute inset-0 accent-selected border-2 border-dashed rounded-lg flex items-center justify-center z-10 pointer-events-none"
          style={{ borderColor: 'var(--primary-500)' }}
        >
          <div className="text-center">
            <FileUp className="w-8 h-8 mx-auto mb-2" style={{ color: 'var(--primary-600)' }} />
            <p className="text-sm font-medium" style={{ color: 'var(--primary-700)' }}>
              Drop files to upload
            </p>
          </div>
        </div>
      )}

      <div className="space-y-4">
        {/* Dashboard & Upload Buttons */}
        <div className="flex gap-2">
          {selectedDocumentId && (
            <button
              onClick={() => {
                // Clear document selection but stay on document view (dashboard)
                useChatStore.setState({ selectedDocumentId: null, selectedChunkId: null });
              }}
              className="button-secondary py-2 px-3 text-center text-sm flex items-center justify-center gap-2"
            >
              Dashboard
            </button>
          )}
          <label className="flex-1 cursor-pointer">
            <div className={`button-primary py-2 px-3 text-center text-sm flex items-center justify-center gap-2 ${uploadingFile ? 'opacity-50 pointer-events-none' : ''
              }`}>
              {uploadingFile ? (
                <Loader size={14} label="Uploading..." />
              ) : (
                <>
                  <FileUp className="w-4 h-4" />
                  <span>Upload Files</span>
                </>
              )}
            </div>
            <input
              type="file"
              className="hidden"
              onChange={handleFileInput}
              disabled={uploadingFile}
              accept=".pdf,.txt,.md,.doc,.docx,.ppt,.pptx,.xls,.xlsx"
              multiple
            />
          </label>
        </div>

        <div className="rounded-lg border border-dashed border-secondary-200 dark:border-secondary-700 p-3 text-xs text-secondary-600 dark:text-secondary-400">
          Drop files anywhere in this panel to upload. Drag documents onto a folder to move them.
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-secondary-900 dark:text-secondary-50">Folders</h3>
            {!isCreatingFolder && (
              <button
                type="button"
                onClick={() => setIsCreatingFolder(true)}
                className="text-xs text-secondary-600 dark:text-secondary-400 hover:text-secondary-900 dark:hover:text-secondary-200 flex items-center gap-1"
              >
                <Plus className="w-4 h-4" />
                New
              </button>
            )}
          </div>

          {isCreatingFolder && (
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    void handleCreateFolder()
                  }
                  if (e.key === 'Escape') {
                    setIsCreatingFolder(false)
                    setNewFolderName('')
                  }
                }}
                placeholder="Folder name"
                className="flex-1 px-3 py-2 text-sm border border-secondary-300 rounded-md focus:outline-none focus-primary"
              />
              <button
                type="button"
                onClick={handleCreateFolder}
                className="button-primary text-xs"
              >
                Create
              </button>
              <button
                type="button"
                onClick={() => {
                  setIsCreatingFolder(false)
                  setNewFolderName('')
                }}
                className="button-ghost text-xs"
              >
                Cancel
              </button>
            </div>
          )}

          <div className="space-y-2">
            <button
              type="button"
              onClick={() => setActiveFolderId('all')}
              className={`w-full flex items-center justify-between gap-2 rounded-lg border px-3 py-2 text-sm transition ${activeFolderId === 'all'
                ? 'border-primary-500 bg-secondary-50 dark:bg-secondary-900'
                : 'border-secondary-200 dark:border-secondary-700 hover:bg-secondary-50 dark:hover:bg-secondary-900'
                }`}
            >
              <div className="flex items-center gap-2 min-w-0">
                <Folder className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                <span className="truncate">All documents</span>
              </div>
              <span className="text-xs text-secondary-500 dark:text-secondary-400">
                {stats?.documents?.length || 0}
              </span>
            </button>

            <button
              type="button"
              onClick={() => setActiveFolderId('root')}
              onDragOver={(event) => handleFolderDragOver(event, 'root')}
              onDragLeave={handleFolderDragLeave}
              onDrop={(event) => handleFolderDrop(event, 'root')}
              className={`w-full flex items-center justify-between gap-2 rounded-lg border px-3 py-2 text-sm transition ${activeFolderId === 'root'
                ? 'border-primary-500 bg-secondary-50 dark:bg-secondary-900'
                : 'border-secondary-200 dark:border-secondary-700 hover:bg-secondary-50 dark:hover:bg-secondary-900'
                } ${draggingOverFolderId === 'root' ? 'border-primary-400 bg-primary-50 dark:bg-secondary-800' : ''}`}
            >
              <div className="flex items-center gap-2 min-w-0">
                <Folder className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                <span className="truncate">Unfiled</span>
              </div>
              <span className="text-xs text-secondary-500 dark:text-secondary-400">
                {stats?.documents?.filter((doc) => !doc.folder_id).length || 0}
              </span>
            </button>

            {folders.map((folder) => (
              <button
                key={folder.id}
                type="button"
                onClick={() => setActiveFolderId(folder.id)}
                onDragOver={(event) => handleFolderDragOver(event, folder.id)}
                onDragLeave={handleFolderDragLeave}
                onDrop={(event) => handleFolderDrop(event, folder.id)}
                className={`w-full flex items-center justify-between gap-2 rounded-lg border px-3 py-2 text-sm transition ${activeFolderId === folder.id
                  ? 'border-primary-500 bg-secondary-50 dark:bg-secondary-900'
                  : 'border-secondary-200 dark:border-secondary-700 hover:bg-secondary-50 dark:hover:bg-secondary-900'
                  } ${draggingOverFolderId === folder.id ? 'border-primary-400 bg-primary-50 dark:bg-secondary-800' : ''}`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <Folder className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                  <span className="truncate">{folder.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-secondary-500 dark:text-secondary-400">
                    {folder.document_count}
                  </span>
                  {enableDeleteOps && (
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation()
                        setFolderDeleteTarget(folder)
                      }}
                      className="text-red-600 hover:text-red-700 p-1"
                      title={`Delete ${folder.name}`}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Documents List */}
        {stats && stats.documents.length > 0 && (
          <>
            {!searchMode ? (
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-medium text-secondary-900 dark:text-secondary-50">Documents</h3>
                  <button
                    onClick={() => setSearchMode(true)}
                    className="text-secondary-600 dark:text-secondary-400 hover:text-secondary-900 dark:hover:text-secondary-200 p-1"
                    title="Search documents"
                  >
                    <Search className="w-4 h-4" />
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={sortMode}
                    onChange={(e) => setSortMode(e.target.value as typeof sortMode)}
                    className="text-xs border border-secondary-300 rounded-md px-2 py-1 bg-white dark:bg-secondary-900 text-secondary-700 dark:text-secondary-200"
                  >
                    <option value="newest">Newest</option>
                    <option value="oldest">Oldest</option>
                    <option value="name">Name</option>
                    <option value="manual" disabled={activeFolderId === 'all'}>
                      Manual
                    </option>
                  </select>
                  <button
                    onClick={handleCleanupOrphans}
                    className="button-ghost py-1 px-2 text-xs text-amber-600 hover:text-amber-700 dark:text-amber-400 dark:hover:text-amber-300 h-auto"
                    title="Remove orphaned chunks and entities not connected to any document"
                  >
                    Cleanup
                  </button>
                  {enableDeleteOps && (
                    <button
                      onClick={handleClearDatabase}
                      className="button-ghost py-1 px-2 text-xs text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20 h-auto"
                    >
                      Clear All
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <div className="flex-1 relative">
                  <input
                    ref={searchInputRef}
                    type="text"
                    placeholder="Search documents..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Escape') {
                        setSearchMode(false)
                        setSearchQuery('')
                      }
                    }}
                    className="w-full px-3 py-2 text-sm border border-secondary-300 rounded-md focus:outline-none focus-primary"
                  />
                </div>
                <select
                  value={sortMode}
                  onChange={(e) => setSortMode(e.target.value as typeof sortMode)}
                  className="text-xs border border-secondary-300 rounded-md px-2 py-1 bg-white dark:bg-secondary-900 text-secondary-700 dark:text-secondary-200"
                >
                  <option value="newest">Newest</option>
                  <option value="oldest">Oldest</option>
                  <option value="name">Name</option>
                  <option value="manual" disabled={activeFolderId === 'all'}>
                    Manual
                  </option>
                </select>
                <button
                  onClick={() => {
                    setSearchMode(false)
                    setSearchQuery('')
                  }}
                  className="text-secondary-600 hover:text-secondary-900 dark:text-secondary-400 dark:hover:text-secondary-200 p-1"
                  title="Close search"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}


            {isManualReorder && searchQuery.trim() && (
              <div className="text-xs text-secondary-500 dark:text-secondary-400">
                Manual ordering is disabled while searching.
              </div>
            )}

            <div className="space-y-2">
              {filteredDocuments.map((doc, index) => {
                const isActive = doc.document_id === selectedDocumentId
                const statusLabel = formatStatus(doc)
                const status = doc.processing_status
                const progress =
                  typeof doc.processing_progress === 'number'
                    ? Math.max(0, Math.min(100, doc.processing_progress))
                    : null

                return (
                  <div
                    key={index}
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.effectAllowed = 'move'
                      e.dataTransfer.setData('application/json', JSON.stringify({
                        type: 'document',
                        document_id: doc.document_id,
                        filename: doc.title || doc.original_filename || doc.filename,
                      }))
                      setDraggingDocumentId(doc.document_id)
                    }}
                    onDragEnd={() => {
                      setDraggingDocumentId(null)
                      setDraggingOverFolderId(null)
                    }}
                    onClick={() => handleSelectDocument(doc.document_id)}
                    className={`card p-3 flex flex-col gap-2 transition-all cursor-move group ${isActive
                      ? ''
                      : 'hover:shadow-md hover:cursor-grab active:cursor-grabbing'
                      }`}
                    style={isActive ? { borderColor: 'rgba(36,198,230,0.18)', boxShadow: '0 6px 18px rgba(36,198,230,0.12)' } : undefined}
                  >
                    <div className="flex w-full items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="relative flex items-center gap-2">
                          {status === 'processing' && (
                            <Loader size={14} />
                          )}
                          <p className="text-sm font-medium text-secondary-900 dark:text-secondary-50 truncate">
                            {getDocumentLabel(doc)}
                          </p>
                          {activeFolderId === 'all' && doc.folder_name && (
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-secondary-100 text-secondary-700 dark:bg-secondary-900 dark:text-secondary-300">
                              {doc.folder_name}
                            </span>
                          )}
                        </div>
                        <p className={`text-xs mt-1 ${isStuck && (status === 'queued' || status === 'staged') ? 'text-red-600 dark:text-red-400' : 'text-secondary-600 dark:text-secondary-400'}`}>
                          {status === 'queued' || status === 'staged'
                            ? (isStuck ? 'Queue stuck - processing may have crashed' : 'Processing queued')
                            : doc.document_type
                              ? doc.document_type.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())
                              : 'Reading document...'}
                        </p>
                        {statusLabel && status !== 'queued' && status !== 'staged' && (
                          <p className={`text-[11px] mt-1 ${status === 'error' ? 'text-red-600' : 'text-secondary-500 dark:text-secondary-400'}`} style={status === 'processing' ? { color: 'var(--primary-600)' } : undefined}>
                            {statusLabel}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-1 flex-shrink-0">
                        {isManualReorder && (
                          <>
                            <button
                              type="button"
                              onClick={(event) => {
                                event.stopPropagation()
                                void handleReorderDocument(doc.document_id, 'up')
                              }}
                              disabled={reorderDisabled || index === 0}
                              className="text-secondary-600 hover:text-secondary-900 p-1 disabled:opacity-40"
                              title="Move up"
                            >
                              <ChevronUp className="w-4 h-4" />
                            </button>
                            <button
                              type="button"
                              onClick={(event) => {
                                event.stopPropagation()
                                void handleReorderDocument(doc.document_id, 'down')
                              }}
                              disabled={reorderDisabled || index >= filteredDocuments.length - 1}
                              className="text-secondary-600 hover:text-secondary-900 p-1 disabled:opacity-40"
                              title="Move down"
                            >
                              <ChevronDown className="w-4 h-4" />
                            </button>
                          </>
                        )}
                        {enableDeleteOps && (
                          <button
                            onClick={(event) => {
                              event.stopPropagation()
                              handleDeleteDocument(doc.document_id)
                            }}
                            className="text-red-600 hover:text-red-700 p-1"
                            title={`Delete ${getDocumentLabel(doc)}`}
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </div>

                    {status === 'processing' && progress !== null && (
                      <div className="w-full">
                        <div className="flex justify-between text-[10px] mb-1">
                          <span className={isStuck ? 'text-red-600' : 'text-secondary-500 dark:text-secondary-400'}>
                            {isStuck ? 'Stuck - may need manual refresh' : (
                              <>
                                {doc.processing_stage || 'Processing'}
                                {(doc.processing_stage === 'embedding' && (doc as any).chunk_progress > 0) ?
                                  ` ${Math.round((doc as any).chunk_progress * 100)}%` : ''
                                }
                                {/* Show chunk count during entity extraction for granular progress */}
                                {(['entity_extraction', 'llm_extraction', 'embedding_generation', 'database_operations', 'clustering', 'validation', 'starting'].includes((doc.processing_stage || '').toLowerCase()) && (doc as any).chunks_processed > 0) ?
                                  ` (${(doc as any).chunks_processed}/${(doc as any).total_chunks || '?'})` : ''
                                }
                                {(!['classification', 'chunking', 'summarization', 'embedding', 'processing', 'queued', 'entity_extraction', 'llm_extraction'].includes((doc.processing_stage || '').toLowerCase()) && (doc as any).entity_progress > 0) ?
                                  ` ${Math.round((doc as any).entity_progress * 100)}%` : ''
                                }
                              </>
                            )}
                          </span>
                          <span className={isStuck ? 'text-red-600' : 'text-secondary-500 dark:text-secondary-400'}>
                            {Math.round(progress)}%
                          </span>
                        </div>
                        <div className="h-1.5 w-full rounded-full bg-secondary-200">
                          <div
                            className={`h-1.5 rounded-full transition-all ${isStuck ? 'bg-red-500' : ''}`}
                            style={{ width: `${progress}%`, backgroundColor: isStuck ? undefined : 'var(--accent-primary)' }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
              {searchQuery.trim() && filteredDocuments.length === 0 && (
                <div className="text-center text-secondary-600 dark:text-secondary-400 py-6">
                  <p className="text-sm">No documents match &quot;{searchQuery}&quot;</p>
                </div>
              )}
              {!searchQuery.trim() && filteredDocuments.length === 0 && (
                <div className="text-center text-secondary-600 dark:text-secondary-400 py-6">
                  <p className="text-sm">
                    {activeFolderId === 'root'
                      ? 'No unfiled documents'
                      : activeFolderId === 'all'
                        ? 'No documents found'
                        : 'No documents in this folder'}
                  </p>
                </div>
              )}
            </div>
          </>
        )}

        {stats && stats.documents.length === 0 && (
          <div className="text-center text-secondary-600 dark:text-secondary-400 py-8">
            <p>No documents in database</p>
            <p className="text-xs mt-1">Upload documents to get started</p>
          </div>
        )}
      </div>

      {/* Upload Update Dialog */}
      {showUploadDialog && pendingFile && (
        <UploadUpdateDialog
          file={pendingFile}
          onClose={handleDialogClose}
          onUploadNew={handleUploadNew}
          onUpdateExisting={handleUpdateExisting}
        />
      )}

      {folderDeleteTarget && (
        <FolderDeleteDialog
          folder={folderDeleteTarget}
          onClose={() => setFolderDeleteTarget(null)}
          onMoveToRoot={() => handleDeleteFolder('move_to_root')}
          onDeleteDocuments={() => handleDeleteFolder('delete_documents')}
          disableDeleteDocuments={!enableDeleteOps}
        />
      )}

      {showClearDialog && (
        <ClearDatabaseDialog
          onClose={() => setShowClearDialog(false)}
          onConfirm={performClearDatabase}
        />
      )}
    </div>
  )
}
