"use client"

import type React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowLeftIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  ClipboardIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline'
import Loader from '@/components/Utils/Loader'
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/solid'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '@/lib/api'
import type {
  DocumentChunk,
  DocumentDetails,
  DocumentEntity,
  RelatedDocument,
} from '@/types'
import type { ProcessingGlobalSummary } from '@/types/upload'
import { useChatStore } from '@/store/chatStore'
import DocumentPreview from './DocumentPreview'
import DocumentGraph from './DocumentGraph'
import CommunitiesSection from './CommunitiesSection'
import ChunkSimilaritiesSection from './ChunkSimilaritiesSection'

interface PreviewState {
  url: string | null
  mimeType?: string
  isLoading: boolean
  error?: string | null
  objectUrl?: string | null
  content?: string | null
}

const initialPreviewState: PreviewState = {
  url: null,
  isLoading: false,
  mimeType: undefined,
  error: null,
  objectUrl: null,
  content: null,
}

export default function DocumentView() {
  const selectedDocumentId = useChatStore((state) => state.selectedDocumentId)
  const selectedChunkId = useChatStore((state) => state.selectedChunkId)
  const clearSelectedDocument = useChatStore((state) => state.clearSelectedDocument)
  const clearSelectedChunk = useChatStore((state) => state.clearSelectedChunk)
  const selectDocument = useChatStore((state) => state.selectDocument)

  const [documentData, setDocumentData] = useState<DocumentDetails | null>(null)
  // Summary-first loading: lightweight stats before full details
  const [summaryData, setSummaryData] = useState<null | {
    id: string
    filename: string
    stats: { chunks: number; entities: number; communities: number; similarities: number }
  }>(null)
  const [hasPreview, setHasPreview] = useState<boolean | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedChunks, setExpandedChunks] = useState<Record<string | number, boolean>>({})
  const [previewState, setPreviewState] = useState<PreviewState>(initialPreviewState)
  const [showAllChunks, setShowAllChunks] = useState(false)
  const [showAllEntities, setShowAllEntities] = useState<Record<string, boolean>>({})
  const [processingState, setProcessingState] = useState<ProcessingGlobalSummary | null>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [isActionPending, setIsActionPending] = useState(false)
  const prevProcessingRef = useRef(false)
  const [isEditingHashtags, setIsEditingHashtags] = useState(false)
  const [newHashtagInput, setNewHashtagInput] = useState('')
  const [settings, setSettings] = useState<{ enable_entity_extraction?: boolean } | null>(null)
  const [isChunksExpanded, setIsChunksExpanded] = useState(false)
  const [isEntitiesExpanded, setIsEntitiesExpanded] = useState(false)
  const [isMetadataExpanded, setIsMetadataExpanded] = useState(false)

  const refreshProcessingState = useCallback(async () => {
    try {
      const response = await api.getProcessingProgress()
      setProcessingState(response.global)
    } catch (stateError) {
      console.error('Failed to load processing state', stateError)
    }
  }, [])

  const refreshSettings = useCallback(async () => {
    try {
      const response = await api.getSettings()
      setSettings(response)
    } catch (settingsError) {
      console.error('Failed to load settings', settingsError)
    }
  }, [])

  const CHUNKS_LIMIT = 10
  const ENTITIES_PER_TYPE_LIMIT = 5

  // On mount, support deep-linking via URL query params: ?doc=<id>&chunk=<index|id>
  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      const params = new URLSearchParams(window.location.search)
      const docParam = params.get('doc')
      const chunkParam = params.get('chunk')
      const chunkIdParam = params.get('chunk_id')
      if (docParam) {
        // Select document and optional chunk from URL
        if (!selectedDocumentId || selectedDocumentId !== docParam) {
          selectDocument(docParam)
        }
        if (chunkParam) {
          const parsed = Number(chunkParam)
          if (!Number.isNaN(parsed)) {
            // chunk interpreted as index
            // Use store action directly
            const selectChunk = (useChatStore.getState && (useChatStore.getState() as any).selectDocumentChunk) || null
            if (selectChunk) selectChunk(docParam, parsed)
          }
        } else if (chunkIdParam) {
          const selectChunk = (useChatStore.getState && (useChatStore.getState() as any).selectDocumentChunk) || null
          if (selectChunk) selectChunk(docParam, chunkIdParam)
        }
      }
    } catch (e) {
      // ignore
    }
  }, [selectDocument, selectedDocumentId])

  // Summary-first loading effect
  useEffect(() => {
    let isSubscribed = true

    const fetchSummary = async (documentId: string) => {
      setIsLoading(true)
      setError(null)
      setSummaryData(null)
      setDocumentData(null)
      setPreviewState(initialPreviewState)
      setShowAllChunks(false)
      setShowAllEntities({})
      try {
        performance.mark('document-summary-fetch-start')
        const summary = await api.getDocumentSummary(documentId)
        performance.mark('document-summary-fetch-end')
        performance.measure('document-summary-fetch', 'document-summary-fetch-start', 'document-summary-fetch-end')
        const measure = performance.getEntriesByName('document-summary-fetch')[0]
        if (measure) {
          console.log(`[Performance] Document summary loaded in ${measure.duration.toFixed(2)}ms`)
        }
        if (isSubscribed) {
          setSummaryData({ id: summary.id, filename: summary.filename, stats: summary.stats })
          setHasPreview(null)
        }
      } catch (e) {
        if (isSubscribed) setError(e instanceof Error ? e.message : 'Failed to load summary')
      } finally {
        if (isSubscribed) setIsLoading(false)
      }
    }

    if (selectedDocumentId) {
      void fetchSummary(selectedDocumentId)
      void refreshProcessingState()
      void refreshSettings()
    } else {
      setSummaryData(null)
      setDocumentData(null)
      setPreviewState(initialPreviewState)
      setShowAllChunks(false)
      setShowAllEntities({})
      setHasPreview(null)
      setProcessingState(null)
      setSettings(null)
    }

    return () => {
      isSubscribed = false
    }
  }, [refreshProcessingState, refreshSettings, selectedDocumentId])

  // Lazy full-document load when user expands sections
  useEffect(() => {
    const needFull = selectedDocumentId && !documentData && (isChunksExpanded || isEntitiesExpanded || isMetadataExpanded)
    if (needFull && selectedDocumentId) {
      ;(async () => {
        try {
          setIsLoading(true)
          const full = await api.getDocument(selectedDocumentId)
          setDocumentData(full)
        } catch (e) {
          setError(e instanceof Error ? e.message : 'Failed to load document')
        } finally {
          setIsLoading(false)
        }
      })()
    }
  }, [isChunksExpanded, isEntitiesExpanded, isMetadataExpanded, documentData, selectedDocumentId])

  useEffect(() => {
    let isSubscribed = true
    const checkPreview = async () => {
      if (!documentData) return
      try {
        const available = await api.hasDocumentPreview(documentData.id)
        if (isSubscribed) setHasPreview(available)
      } catch (e) {
        if (isSubscribed) setHasPreview(false)
      }
    }

    checkPreview()

    return () => {
      isSubscribed = false
    }
  }, [documentData])

  useEffect(() => {
    if (selectedChunkId !== null && documentData) {
      const chunk = documentData.chunks.find(c => c.index === selectedChunkId || c.id === selectedChunkId)
      if (chunk) {
        setExpandedChunks(prev => ({ ...prev, [chunk.id]: true }))
      }
      clearSelectedChunk()
    }
  }, [selectedChunkId, documentData, clearSelectedChunk])

  useEffect(() => {
    return () => {
      if (previewState.objectUrl) {
        URL.revokeObjectURL(previewState.objectUrl)
      }
    }
  }, [previewState.objectUrl])

  const groupedEntities = useMemo(() => {
    if (!documentData?.entities?.length) {
      return {}
    }

    return documentData.entities.reduce<Record<string, DocumentEntity[]>>((acc, entity) => {
      const typeKey = entity.type || 'Unknown'
      if (!acc[typeKey]) {
        acc[typeKey] = []
      }
      acc[typeKey].push(entity)
      return acc
    }, {})
  }, [documentData?.entities])

  const isMarkdownDocument = useMemo(() => {
    if (!documentData) return false

    const mime = documentData.mime_type?.toLowerCase() || ''
    const markdownMimeTypes = new Set([
      'text/markdown',
      'text/x-markdown',
      'text/md',
      'text/x-md',
      'application/markdown',
    ])

    if (markdownMimeTypes.has(mime)) {
      return true
    }

    const fileName = documentData.file_name?.toLowerCase() || ''
    return fileName.endsWith('.md') || fileName.endsWith('.markdown')
  }, [documentData])

  const toggleChunk = useCallback((chunk: DocumentChunk) => {
    setExpandedChunks((state) => ({
      ...state,
      [chunk.id]: !state[chunk.id],
    }))
  }, [])

  const handleCopyChunk = useCallback(async (chunk: DocumentChunk) => {
    try {
      await navigator.clipboard.writeText(chunk.text)
    } catch (copyError) {
      console.error('Failed to copy chunk', copyError)
    }
  }, [])

  const handleOpenPreview = useCallback(async () => {
    if (!selectedDocumentId) return
    if (previewState.objectUrl) {
      URL.revokeObjectURL(previewState.objectUrl)
    }

    if (documentData?.preview_url) {
      setPreviewState({ url: documentData.preview_url, mimeType: documentData.mime_type, isLoading: false, error: null })
      return
    }

    setPreviewState({ url: null, mimeType: undefined, isLoading: true, error: null, objectUrl: null, content: null })

    try {
      const response = await api.getDocumentPreview(selectedDocumentId)

      // Handle various response formats from getDocumentPreview
      if ('preview_url' in response) {
        // External preview URL (redirect)
        setPreviewState({ url: response.preview_url, mimeType: documentData?.mime_type, isLoading: false, error: null })
        return
      }

      if ('content' in response) {
        // JSON response with inline content (markdown/text)
        const mimeType = (response as any).mime_type || documentData?.mime_type
        const content = (response as any).content

        // For markdown, we'll render it on the frontend and pass to DocumentPreview
        setPreviewState({
          url: null,
          mimeType,
          isLoading: false,
          error: null,
          objectUrl: null,
          content,
        })
        return
      }

      if (response instanceof Response) {
        const mimeType = response.headers?.get?.('Content-Type') || documentData?.mime_type

        if (mimeType?.toLowerCase().includes('markdown') || mimeType?.toLowerCase().includes('text/')) {
          const content = await response.text()
          setPreviewState({
            url: null,
            mimeType,
            isLoading: false,
            error: null,
            objectUrl: null,
            content,
          })
          return
        }

        const blob = await response.blob()
        const objectUrl = URL.createObjectURL(blob)
        setPreviewState({
          url: objectUrl,
          mimeType,
          isLoading: false,
          error: null,
          objectUrl,
        })
        return
      }

      if ('object_url' in response) {
        // Blob converted to object URL (non-JSON files)
        setPreviewState({
          url: (response as any).object_url,
          mimeType: (response as any).mime_type,
          isLoading: false,
          error: null,
          objectUrl: (response as any).object_url,
        })
        return
      }

      throw new Error('Unexpected response format from preview endpoint')
    } catch (previewError) {
      const errorMessage = previewError instanceof Error ? previewError.message : 'Unable to load preview'
      setPreviewState({ url: null, mimeType: undefined, isLoading: false, error: errorMessage, objectUrl: null })
      console.error('Failed to load preview', previewError)
    }
  }, [documentData?.mime_type, documentData?.preview_url, previewState.objectUrl, selectedDocumentId])

  const handleClosePreview = useCallback(() => {
    if (previewState.objectUrl) {
      URL.revokeObjectURL(previewState.objectUrl)
    }
    setPreviewState(initialPreviewState)
  }, [previewState.objectUrl])

  const handleReprocessChunks = useCallback(async () => {
    if (!documentData?.id) return
    setIsActionPending(true)
    setActionError(null)
    setActionMessage(null)
    try {
      await api.reprocessDocumentChunks(documentData.id)
      setActionMessage('Chunk processing queued')
      await refreshProcessingState()
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:processing-updated'))
      }
    } catch (reprocessError) {
      setActionError(
        reprocessError instanceof Error
          ? reprocessError.message
          : 'Failed to queue chunk processing'
      )
    } finally {
      setIsActionPending(false)
    }
  }, [documentData?.id, refreshProcessingState])

  const handleReprocessEntities = useCallback(async () => {
    if (!documentData?.id) return
    setIsActionPending(true)
    setActionError(null)
    setActionMessage(null)
    try {
      await api.reprocessDocumentEntities(documentData.id)
      setActionMessage('Entity extraction queued')
      await refreshProcessingState()
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:processing-updated'))
      }
    } catch (reprocessError) {
      setActionError(
        reprocessError instanceof Error
          ? reprocessError.message
          : 'Failed to queue entity extraction'
      )
    } finally {
      setIsActionPending(false)
    }
  }, [documentData?.id, refreshProcessingState])

  const handleGenerateSummary = useCallback(async () => {
    if (!documentData?.id) return
    setIsActionPending(true)
    setActionError(null)
    setActionMessage(null)
    try {
      await api.generateDocumentSummary(documentData.id)
      setActionMessage('Summary generated successfully')
      // Refresh document data
      const data = await api.getDocument(documentData.id)
      setDocumentData(data)
    } catch (summaryError) {
      setActionError(
        summaryError instanceof Error
          ? summaryError.message
          : 'Failed to generate summary'
      )
    } finally {
      setIsActionPending(false)
    }
  }, [documentData?.id])

  const handleStartEditHashtags = useCallback(() => {
    setNewHashtagInput('')
    setIsEditingHashtags(true)
  }, [])

  const saveHashtags = useCallback(async (newHashtags: string[]) => {
    if (!documentData?.id) return
    setActionError(null)
    try {
      await api.updateDocumentHashtags(documentData.id, newHashtags)
      // Refresh document data
      const data = await api.getDocument(documentData.id)
      setDocumentData(data)
    } catch (updateError) {
      setActionError(
        updateError instanceof Error
          ? updateError.message
          : 'Failed to update hashtags'
      )
    }
  }, [documentData?.id])

  const handleRemoveHashtag = useCallback(async (tagToRemove: string) => {
    if (!documentData?.hashtags) return
    const newHashtags = documentData.hashtags.filter(tag => tag !== tagToRemove)
    await saveHashtags(newHashtags)
  }, [documentData?.hashtags, saveHashtags])

  const handleAddHashtag = useCallback(async () => {
    let trimmed = newHashtagInput.trim()
    if (!trimmed) return
    
    // Automatically add # prefix if not present
    if (!trimmed.startsWith('#')) {
      trimmed = '#' + trimmed
    }
    
    const currentHashtags = documentData?.hashtags || []
    if (currentHashtags.includes(trimmed)) {
      setNewHashtagInput('')
      return
    }
    
    const newHashtags = [...currentHashtags, trimmed]
    await saveHashtags(newHashtags)
    setNewHashtagInput('')
  }, [newHashtagInput, documentData?.hashtags, saveHashtags])

  const handleHashtagInputKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAddHashtag()
    }
  }, [handleAddHashtag])

  useEffect(() => {
    if (!actionMessage) return
    const timer = window.setTimeout(() => setActionMessage(null), 4000)
    return () => window.clearTimeout(timer)
  }, [actionMessage])

  useEffect(() => {
    if (!actionError) return
    const timer = window.setTimeout(() => setActionError(null), 5000)
    return () => window.clearTimeout(timer)
  }, [actionError])

  useEffect(() => {
    const isProcessingNow = Boolean(processingState?.is_processing)
    const wasProcessing = prevProcessingRef.current
    prevProcessingRef.current = isProcessingNow

    if (isProcessingNow && selectedDocumentId) {
      const intervalId = window.setInterval(() => {
        // Poll both processing state and document data to refresh summary
        void refreshProcessingState()
        void (async () => {
          try {
            const updatedDoc = await api.getDocument(selectedDocumentId)
            setDocumentData(updatedDoc)
          } catch (error) {
            console.error('Failed to refresh document during processing:', error)
          }
        })()
      }, 2000)
      return () => window.clearInterval(intervalId)
    }

    if (wasProcessing && !isProcessingNow) {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:processed'))
      }
    }
  }, [processingState?.is_processing, refreshProcessingState, selectedDocumentId])

  const disableManualActions = isActionPending || Boolean(processingState?.is_processing)
  const manualActionTitle = processingState?.is_processing
    ? 'Processing is already running for another document'
    : undefined

  const isEntityExtractionDisabled = settings?.enable_entity_extraction === false
  const entityButtonDisabled = disableManualActions || isEntityExtractionDisabled
  const entityButtonTitle = isEntityExtractionDisabled
    ? 'Entity extraction is deactivated in settings'
    : manualActionTitle

  const handleRelatedDocumentClick = useCallback(
    (doc: RelatedDocument) => {
      if (!doc.id) return
      selectDocument(doc.id)
    },
    [selectDocument]
  )

  const toggleShowAllEntities = useCallback((type: string) => {
    setShowAllEntities((state) => ({
      ...state,
      [type]: !state[type],
    }))
  }, [])

  const toggleChunksExpanded = useCallback(() => {
    setIsChunksExpanded((prev) => !prev)
  }, [])

  const toggleEntitiesExpanded = useCallback(() => {
    setIsEntitiesExpanded((prev) => !prev)
  }, [])

  const toggleMetadataExpanded = useCallback(() => {
    setIsMetadataExpanded((prev) => !prev)
  }, [])

  if (!selectedDocumentId) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-secondary-600 dark:text-secondary-400">
        <DocumentTextIcon className="w-16 h-16 text-secondary-300 mb-4" />
        <p className="text-base font-medium">Select a document to view its details.</p>
      </div>
    )
  }

  // Previously we returned early to show only the summary. Instead, render the
  // summary as a top panel inside the full document view so the section headers
  // (Chunks, Entities, Graph, etc.) are visible and can be expanded to lazy-load
  // full document details.

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="border-b border-secondary-200 dark:border-secondary-700 bg-white dark:bg-secondary-800 px-6 py-4 flex items-center gap-4">
        <button
          type="button"
          onClick={clearSelectedDocument}
          className="button-secondary flex items-center gap-2 text-sm"
        >
          <ArrowLeftIcon className="w-4 h-4" />
          Back to chat
        </button>
        <div className="flex items-center gap-3">
          <DocumentTextIcon className="w-6 h-6" style={{ color: 'var(--primary-500)' }} />
          <div>
            <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-50">
              {documentData?.title || documentData?.original_filename || documentData?.file_name || 'Unnamed document'}
            </h2>
            <p className="text-xs text-secondary-500 dark:text-secondary-400">
              {documentData?.document_type ? documentData.document_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Unknown type'}
            </p>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {documentData?.uploaded_at && (
            <span className="text-xs text-secondary-500 dark:text-secondary-400">
              Uploaded {new Date(documentData.uploaded_at).toLocaleString()}
            </span>
          )}
          {(hasPreview === null ? documentData?.preview_url : hasPreview) && (
            <button
              type="button"
              onClick={handleOpenPreview}
              className="button-primary text-sm flex items-center gap-2"
              disabled={previewState.isLoading}
            >
              <MagnifyingGlassIcon className="w-4 h-4" />
              {previewState.isLoading ? 'Loading preview…' : 'Open preview'}
            </button>
          )}
          {previewState.error && (
            <span className="text-xs text-red-600 dark:text-red-400">{previewState.error}</span>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-6 pt-0 pb-6 bg-secondary-50 dark:bg-secondary-900">
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-20 text-secondary-500 dark:text-secondary-400">
            <Loader size={40} label="Loading document metadata…" />
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 flex items-start gap-2">
            <ExclamationCircleIcon className="w-5 h-5 flex-shrink-0" />
            <div>
              <p className="font-medium">Failed to load document</p>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* When we have a summary but not the full document yet, show a lightweight
            summary panel so the rest of the document view (sections headers)
            remains visible and users can expand sections. */}
        {!isLoading && !error && !documentData && summaryData && (
          <div className="flex flex-col h-full overflow-hidden p-6">
            <div className="border border-secondary-200 dark:border-secondary-700 rounded bg-white dark:bg-secondary-800 p-4 mb-4">
              <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-50 mb-1">{summaryData.filename}</h2>
              <p className="text-xs text-secondary-500 dark:text-secondary-400 mb-4">ID: {summaryData.id}</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="p-2 rounded bg-secondary-50 dark:bg-secondary-900/40 border border-secondary-200 dark:border-secondary-700">
                  <p className="font-semibold">Chunks</p>
                  <p>{summaryData.stats.chunks}</p>
                </div>
                <div className="p-2 rounded bg-secondary-50 dark:bg-secondary-900/40 border border-secondary-200 dark:border-secondary-700">
                  <p className="font-semibold">Entities</p>
                  <p>{summaryData.stats.entities}</p>
                </div>
                <div className="p-2 rounded bg-secondary-50 dark:bg-secondary-900/40 border border-secondary-200 dark:border-secondary-700">
                  <p className="font-semibold">Communities</p>
                  <p>{summaryData.stats.communities}</p>
                </div>
                <div className="p-2 rounded bg-secondary-50 dark:bg-secondary-900/40 border border-secondary-200 dark:border-secondary-700">
                  <p className="font-semibold">Similarities</p>
                  <p>{summaryData.stats.similarities}</p>
                </div>
              </div>
              {(
                // @ts-ignore - summaryData may include previews from backend
                (summaryData as any)?.previews?.top_communities || (summaryData as any)?.previews?.top_similarities
              ) && (
                <div className="mt-4">
                  <h4 className="text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">Preview</h4>
                  <div className="flex flex-col gap-2 text-xs">
                    {( (summaryData as any)?.previews?.top_communities ) && (
                      <div>
                        <div className="text-secondary-500 dark:text-secondary-400 mb-1">Top communities</div>
                        <div className="flex gap-2 flex-wrap">
                          {((summaryData as any).previews.top_communities || []).map((c: any, idx: number) => (
                            <span key={idx} className="inline-block px-2 py-0.5 bg-secondary-100 dark:bg-secondary-800 rounded text-secondary-900 dark:text-secondary-200">
                              {`#${c?.community_id ?? 'n/a'} (${c?.count ?? 0})`}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {((summaryData as any)?.previews?.top_similarities) && (
                      <div>
                        <div className="text-secondary-500 dark:text-secondary-400 mb-1">Top similarities</div>
                        <div className="flex gap-2 flex-wrap">
                          {((summaryData as any).previews.top_similarities || []).slice(0,5).map((s: any, idx: number) => (
                            <span key={idx} className="inline-block px-2 py-0.5 bg-secondary-100 dark:bg-secondary-800 rounded text-secondary-900 dark:text-secondary-200">
                              {`${s?.chunk1_id?.slice?.(0,6) ?? 'c1'} ↔ ${s?.chunk2_id?.slice?.(0,6) ?? 'c2'} ${typeof s?.score === 'number' ? `(${s.score.toFixed(2)})` : ''}`}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
              <p className="mt-4 text-xs text-secondary-500 dark:text-secondary-400">Expand a section (Chunks, Entities, Metadata) to load full details.</p>
            </div>
          </div>
        )}

        {!isLoading && !error && documentData && (
          <div className="space-y-6">
            {actionMessage && (
              <div className="bg-green-50 border border-green-200 text-green-700 rounded-lg px-4 py-3 text-sm">
                {actionMessage}
              </div>
            )}
            {actionError && (
              <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 text-sm">
                {actionError}
              </div>
            )}
            {documentData.metadata?.processing_status === 'staged' && (
              <div className="bg-blue-50 border border-blue-200 text-blue-700 rounded-lg px-4 py-3 text-sm">
                <p className="font-medium mb-1">Document ready to process</p>
                <p>This document has been uploaded but not yet processed. Go to the <strong>Upload</strong> tab in the sidebar and click <strong>Process All</strong> to begin processing.</p>
              </div>
            )}
            <section className="bg-white dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700 p-5">
              <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50 mb-4">Overview</h3>
              <dl className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                <div>
                  <dt className="text-secondary-500 dark:text-secondary-400">File name</dt>
                  <dd className="text-secondary-900 dark:text-secondary-50">{documentData.original_filename || documentData.file_name || 'Unknown'}</dd>
                </div>
                <div>
                  <dt className="text-secondary-500 dark:text-secondary-400">Mime type</dt>
                  <dd className="text-secondary-900 dark:text-secondary-50">{documentData.mime_type || 'Unknown'}</dd>
                </div>
                <div>
                  <dt className="text-secondary-500 dark:text-secondary-400">Uploader</dt>
                  <dd className="text-secondary-900 dark:text-secondary-50">
                    {documentData.uploader?.name || documentData.uploader?.id || 'Unknown'}
                  </dd>
                </div>
                <div>
                  <dt className="text-secondary-500 dark:text-secondary-400">Uploaded at</dt>
                  <dd className="text-secondary-900 dark:text-secondary-50">
                    {documentData.uploaded_at
                      ? new Date(documentData.uploaded_at).toLocaleString()
                      : 'Unknown'}
                  </dd>
                </div>
                <div>
                  <dt className="text-secondary-500 dark:text-secondary-400">Chunk count</dt>
                  <dd className="text-secondary-900 dark:text-secondary-50">{documentData.chunks.length}</dd>
                </div>
                <div>
                  <dt className="text-secondary-500 dark:text-secondary-400">Preview available</dt>
                  <dd className="text-secondary-900 dark:text-secondary-50">
                    {hasPreview === null
                      ? (documentData.preview_url ? 'Yes' : 'No')
                      : hasPreview
                      ? 'Yes'
                      : 'No'}
                  </dd>
                </div>
              </dl>
              <div className="mt-4 pt-4 border-t border-secondary-200 dark:border-secondary-700">
                <dt className="text-secondary-500 dark:text-secondary-400 text-sm mb-2">Tags</dt>
                <div 
                  className="flex flex-wrap gap-2 cursor-pointer p-2 -m-2 rounded hover:bg-secondary-50 dark:hover:bg-secondary-900 transition-colors"
                  onClick={handleStartEditHashtags}
                >
                  {!isEditingHashtags ? (
                    <>
                      {(documentData.hashtags || []).map((tag, index) => (
                        <span
                          key={index}
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                          style={{ backgroundColor: 'var(--neon-glow)', color: 'var(--primary-500)' }}
                        >
                          {tag}
                        </span>
                      ))}
                      {(!documentData.hashtags || documentData.hashtags.length === 0) && (
                        <span className="text-sm text-secondary-400">Click to add tags</span>
                      )}
                    </>
                  ) : (
                    <div 
                      className="flex flex-wrap gap-2 w-full"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {(documentData.hashtags || []).map((tag, index) => (
                        <span
                          key={index}
                          className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium"
                          style={{ backgroundColor: 'var(--neon-glow)', color: 'var(--primary-500)' }}
                        >
                          {tag}
                          <button
                            type="button"
                            onMouseDown={(e) => {
                              e.preventDefault()
                              e.stopPropagation()
                              handleRemoveHashtag(tag)
                            }}
                            className="hover:opacity-80"
                          >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </span>
                      ))}
                      <div className="inline-flex items-center gap-1">
                        <input
                          type="text"
                          value={newHashtagInput}
                          onChange={(e) => setNewHashtagInput(e.target.value)}
                          onKeyDown={handleHashtagInputKeyDown}
                          onBlur={() => setIsEditingHashtags(false)}
                          placeholder="Add tag..."
                          className="px-2.5 py-0.5 text-xs border border-secondary-300 rounded-full focus:outline-none focus-primary"
                          autoFocus
                        />
                        <button
                          type="button"
                          onMouseDown={(e) => {
                            e.preventDefault()
                            handleAddHashtag()
                          }}
                          className=""
                          style={{ color: 'var(--primary-500)' }}
                          disabled={!newHashtagInput.trim()}
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>

            {(documentData.summary || documentData.chunks.length > 0 || documentData.metadata?.processing_status !== 'staged') && (
              <>
                <section className="bg-white dark:bg-secondary-800 dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700">
                <header className="flex items-center justify-between px-5 py-4 border-b border-secondary-200 dark:border-secondary-700">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50">Summary</h3>
                    {documentData.document_type && (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-secondary-100 dark:bg-secondary-700 text-secondary-700 dark:text-secondary-200">
                        {documentData.document_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                      </span>
                    )}
                      {!documentData.summary && documentData.chunks.length > 0 && documentData.metadata?.processing_status !== 'staged' && (
                      <button
                        type="button"
                        onClick={handleGenerateSummary}
                        className="button-primary text-xs"
                        disabled={disableManualActions}
                        title={manualActionTitle}
                      >
                        {isActionPending ? <Loader size={14} label="Processing..." /> : 'Generate summary'}
                      </button>
                    )}
                  </div>
                </header>
                <div className="px-5 py-4">
                  {documentData.summary ? (
                    <div className="prose prose-sm prose-slate dark:prose-invert dark:text-secondary-200 max-w-none">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkBreaks]}
                      >
                        {documentData.summary}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-sm text-secondary-500 dark:text-secondary-400">No summary generated yet.</p>
                  )}
                </div>
              </section>
            </>
          )}

          <section className="bg-white dark:bg-secondary-800 dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700">
              <header className={`flex items-center justify-between px-5 py-4 ${isChunksExpanded ? 'border-b border-secondary-200 dark:border-secondary-700' : ''}`}>
                <button
                  type="button"
                  onClick={toggleChunksExpanded}
                  className="flex items-center gap-2 hover:bg-secondary-50 dark:hover:bg-secondary-700 rounded px-2 py-1 -mx-2 transition-colors"
                >
                  {isChunksExpanded ? (
                    <ChevronUpIcon className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                  ) : (
                    <ChevronDownIcon className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                  )}
                  <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50">Chunks</h3>
                </button>
                <div className="flex items-center gap-2">
                  {documentData.chunks.length === 0 && documentData.metadata?.processing_status !== 'staged' && !processingState?.is_processing && (
                    <button
                      type="button"
                      onClick={handleReprocessChunks}
                      className="button-primary text-xs"
                      disabled={disableManualActions}
                      title={manualActionTitle}
                    >
                      {isActionPending ? <Loader size={14} label="Processing..." /> : 'Process chunks'}
                    </button>
                  )}
                  <span className="text-xs text-secondary-500 dark:text-secondary-400">{documentData.chunks.length} entries</span>
                </div>
              </header>
              <AnimatePresence>
                {isChunksExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3, ease: 'easeInOut' }}
                    className="divide-y divide-secondary-200 dark:divide-secondary-700 overflow-hidden"
                  >
                    {documentData.chunks.length === 0 ? (
                      <p className="px-5 py-4 text-sm text-secondary-500 dark:text-secondary-400">No chunks processed yet.</p>
                    ) : (
                      <>
                        {(showAllChunks ? documentData.chunks : documentData.chunks.slice(0, CHUNKS_LIMIT)).map((chunk: DocumentChunk) => {
                          const expanded = expandedChunks[chunk.id]
                          const firstLine = (chunk.text || '').split(/\r?\n/)[0] || ''
                          const previewLine = firstLine.trim() || 'No preview available'
                          return (
                            <article key={chunk.id} className="px-5 py-4">
                              <div className="flex flex-wrap items-center gap-3">
                                <div className="flex-1 min-w-0">
                                  <p className="text-xs text-secondary-500 dark:text-secondary-400">
                                    Chunk {typeof chunk.index === 'number' ? chunk.index + 1 : chunk.id}
                                  </p>
                                  <div className="relative overflow-hidden pr-8">
                                    {/* Allow preview line to wrap and break long words instead of truncating */}
                                    <p className="text-sm font-medium text-secondary-900 dark:text-secondary-50 break-words break-all">
                                      {previewLine}
                                    </p>
                                  </div>
                                </div>
                                <div className="flex items-center gap-2 flex-shrink-0">
                                  <button
                                    type="button"
                                    onClick={() => handleCopyChunk(chunk)}
                                    className="button-ghost text-xs flex items-center gap-1"
                                  >
                                    <ClipboardIcon className="w-4 h-4" />
                                    Copy
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => toggleChunk(chunk)}
                                    className="button-secondary text-xs flex items-center gap-1"
                                  >
                                    {expanded ? (
                                      <ChevronUpIcon className="w-4 h-4" />
                                    ) : (
                                      <ChevronDownIcon className="w-4 h-4" />
                                    )}
                                  </button>
                                </div>
                              </div>
                              <AnimatePresence>
                                {expanded && (
                                  <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.3, ease: 'easeInOut' }}
                                    className="mt-3 text-sm text-secondary-800 dark:text-secondary-200 leading-relaxed overflow-hidden border-t border-secondary-200 dark:border-secondary-700 pt-3"
                                  >
                                    {isMarkdownDocument ? (
                                      <ReactMarkdown
                                        className="prose prose-sm prose-slate dark:prose-invert dark:text-secondary-100 max-w-none break-words"
                                        remarkPlugins={[remarkGfm, remarkBreaks]}
                                      >
                                        {chunk.text || ''}
                                      </ReactMarkdown>
                                    ) : (
                                      // Use whitespace-pre-wrap to preserve newlines but allow breaking long words
                                      <p className="whitespace-pre-wrap break-words">{chunk.text ?? ''}</p>
                                    )}
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </article>
                          )
                        })}
                        {documentData.chunks.length > CHUNKS_LIMIT && (
                          <div className="px-5 py-4 border-t border-secondary-200">
                            <button
                              type="button"
                              onClick={() => setShowAllChunks(!showAllChunks)}
                              className="button-secondary text-sm flex items-center gap-2"
                            >
                              {showAllChunks ? 'Show Less' : `Show ${documentData.chunks.length - CHUNKS_LIMIT} more Chunks`}
                            </button>
                          </div>
                        )}
                      </>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </section>

            <section className="bg-white dark:bg-secondary-800 dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700">
              <header className={`flex items-center justify-between px-5 py-4 ${isEntitiesExpanded ? 'border-b border-secondary-200 dark:border-secondary-700' : ''}`}>
                <button
                  type="button"
                  onClick={toggleEntitiesExpanded}
                  className="flex items-center gap-2 hover:bg-secondary-50 dark:hover:bg-secondary-700 rounded px-2 py-1 -mx-2 transition-colors"
                >
                  {isEntitiesExpanded ? (
                    <ChevronUpIcon className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                  ) : (
                    <ChevronDownIcon className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                  )}
                  <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50">Entities</h3>
                </button>
                <div className="flex items-center gap-2">
                  {documentData.entities.length === 0 && documentData.chunks.length > 0 && documentData.metadata?.processing_status !== 'staged' && !processingState?.is_processing && (
                    <button
                      type="button"
                      onClick={handleReprocessEntities}
                      className={`button-primary text-xs ${isEntityExtractionDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                      disabled={entityButtonDisabled}
                      title={entityButtonTitle}
                    >
                      {isActionPending ? <Loader size={14} label="Processing..." /> : 'Process entities'}
                    </button>
                  )}
                  <span className="text-xs text-secondary-500 dark:text-secondary-400">
                    {documentData.entities.length > 0
                      ? `${documentData.entities.length} total`
                      : 'No entities'}
                  </span>
                </div>
              </header>
              <AnimatePresence>
                {isEntitiesExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3, ease: 'easeInOut' }}
                    className="overflow-hidden"
                  >
                    {documentData.entities.length === 0 ? (
                      <p className="px-5 py-4 text-sm text-secondary-500 dark:text-secondary-400">No entities extracted yet.</p>
                    ) : (
                      <div className="divide-y divide-secondary-100 dark:divide-secondary-700">
                        {Object.entries(groupedEntities).map(([type, entities]) => {
                          const showAll = showAllEntities[type] || false
                          const displayedEntities = showAll ? entities : entities.slice(0, ENTITIES_PER_TYPE_LIMIT)
                          const hasMore = entities.length > ENTITIES_PER_TYPE_LIMIT

                          return (
                            <div key={type} className="px-5 py-4 space-y-2">
                              <h4 className="text-xs font-semibold text-secondary-500 dark:text-secondary-400 uppercase tracking-wide">
                                {type}
                              </h4>
                              <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                {displayedEntities.map((entity) => (
                                  <li key={`${type}-${entity.text}`} className="border border-secondary-200 dark:border-secondary-700 rounded-lg px-3 py-2">
                                    <p className="text-sm font-medium text-secondary-900 dark:text-secondary-50">{entity.text}</p>
                                    <p className="text-xs text-secondary-500 dark:text-secondary-400">
                                      Count: {entity.count ?? '—'} · Positions: {entity.positions?.join(', ') || '—'}
                                    </p>
                                  </li>
                                ))}
                              </ul>
                              {hasMore && (
                                <button
                                  type="button"
                                  onClick={() => toggleShowAllEntities(type)}
                                  className="button-secondary text-xs mt-2"
                                >
                                  {showAll ? 'Show Less' : `Show ${entities.length - ENTITIES_PER_TYPE_LIMIT} more ${type.toLowerCase()}s`}
                                </button>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </section>

            {documentData.quality_scores && (
              <section className="bg-white dark:bg-secondary-800 dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700 p-5">
                <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50 mb-3">Quality scores</h3>
                <pre className="bg-secondary-900 text-secondary-50 text-xs rounded-lg p-4 overflow-auto">
                  {JSON.stringify(documentData.quality_scores, null, 2)}
                </pre>
              </section>
            )}

            {documentData.related_documents && documentData.related_documents.length > 0 && (
              <section className="bg-white dark:bg-secondary-800 dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700 p-5">
                <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50 mb-3">Related documents</h3>
                <ul className="space-y-2">
                  {documentData.related_documents.map((doc) => (
                    <li key={doc.id} className="flex items-center justify-between gap-3 p-3 border border-secondary-200 rounded-lg hover:bg-secondary-50 dark:hover:bg-secondary-900 transition-colors">
                      <div className="flex-1">
                        <p className="text-sm font-medium text-secondary-900 dark:text-secondary-50">
                          {doc.title || doc.link || doc.id}
                        </p>
                        {doc.link && (
                          <a
                              href={doc.link}
                              target="_blank"
                              rel="noreferrer"
                              className="text-xs hover:underline inline-flex items-center gap-1 mt-1"
                              style={{ color: 'var(--primary-500)' }}
                            >
                              <span>Open external link</span>
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                              </svg>
                            </a>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={() => handleRelatedDocumentClick(doc)}
                        className="button-secondary text-xs"
                      >
                        View
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            )}

              {documentData.metadata && (
                <section className="bg-white dark:bg-secondary-800 dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700">
                  <header className={`flex items-center justify-between px-5 py-4 ${isMetadataExpanded ? 'border-b border-secondary-200 dark:border-secondary-700' : ''}`}>
                    <button
                      type="button"
                      onClick={toggleMetadataExpanded}
                      className="flex items-center gap-2 hover:bg-secondary-50 dark:hover:bg-secondary-700 rounded px-2 py-1 -mx-2 transition-colors"
                    >
                      {isMetadataExpanded ? (
                        <ChevronUpIcon className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                      ) : (
                        <ChevronDownIcon className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                      )}
                      <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50">Metadata</h3>
                    </button>
                  </header>
                  <AnimatePresence>
                    {isMetadataExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3, ease: 'easeInOut' }}
                        className="overflow-hidden"
                      >
                        <pre className="bg-secondary-900 text-secondary-50 text-xs rounded-b-lg p-4 overflow-auto">
                          {JSON.stringify(documentData.metadata, null, 2)}
                        </pre>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </section>
              )}

              {/* Document-level graph and community info */}
              <section className="bg-white dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700">
                <header className="flex items-center justify-between px-5 py-4 border-b border-secondary-200 dark:border-secondary-700">
                  <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50">Graph (document)</h3>
                  <p className="text-xs text-secondary-500 dark:text-secondary-400">Interactive 3D view of entities in this document</p>
                </header>
                <div className="p-5">
                  <DocumentGraph documentId={documentData.id} height={480} />
                </div>
              </section>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <CommunitiesSection documentId={documentData.id} />
                </div>
                <div>
                  <ChunkSimilaritiesSection documentId={documentData.id} />
                </div>
              </div>
            </div>
          )}
      </div>

      {(previewState.url || previewState.content) && (
        <DocumentPreview
          previewUrl={previewState.url}
          mimeType={previewState.mimeType}
          content={previewState.content}
          onClose={handleClosePreview}
        />
      )}
    </div>
  )
}
