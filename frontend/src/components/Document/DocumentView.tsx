"use client"

import type React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowLeftIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  ClipboardIcon,
  ExclamationCircleIcon,
  ArrowUpTrayIcon,
} from '@heroicons/react/24/outline'
import { Button } from '@mui/material'
import { Description as DocumentIconMui, Storage as DatabaseIconMui } from '@mui/icons-material'
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
  // pagination state for chunks
  const [chunksPageSize] = useState(200)
  const [chunksOffset, setChunksOffset] = useState(0)
  const [docChunksTotal, setDocChunksTotal] = useState<number | null>(null)
  const [loadingMoreChunks, setLoadingMoreChunks] = useState(false)
  const [showAllEntities, setShowAllEntities] = useState<Record<string, boolean>>({})
  const [processingState, setProcessingState] = useState<ProcessingGlobalSummary | null>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [isActionPending, setIsActionPending] = useState(false)
  const prevProcessingRef = useRef(false)
  const [isEditingHashtags, setIsEditingHashtags] = useState(false)
  const [newHashtagInput, setNewHashtagInput] = useState('')
  const [isEditingMetadata, setIsEditingMetadata] = useState(false)
  const [editedMetadata, setEditedMetadata] = useState('')
  const [settings, setSettings] = useState<{ enable_entity_extraction?: boolean } | null>(null)
  const [isChunksExpanded, setIsChunksExpanded] = useState(false)
  const [isEntitiesExpanded, setIsEntitiesExpanded] = useState(false)
  const [isMetadataExpanded, setIsMetadataExpanded] = useState(false)
  const [loadingChunksData, setLoadingChunksData] = useState(false)
  const [loadingEntitiesData, setLoadingEntitiesData] = useState(false)
  const [entitySummary, setEntitySummary] = useState<null | { document_id: string; total: number; groups: Array<{ type: string; count: number }> }>(null)
  const [isUpdating, setIsUpdating] = useState(false)
  const updateFileInputRef = useRef<HTMLInputElement>(null)

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

      // Only set initial selection if we have a param
      if (docParam) {
        // Use store action directly
        selectDocument(docParam)

        if (chunkParam) {
          const parsed = Number(chunkParam)
          if (!Number.isNaN(parsed)) {
            // chunk interpreted as index
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Run ONLY on mount

  // Sync selection to URL
  useEffect(() => {
    if (typeof window === 'undefined') return

    const params = new URLSearchParams(window.location.search)
    if (selectedDocumentId) {
      if (params.get('doc') !== selectedDocumentId) {
        params.set('doc', selectedDocumentId)
        // Clear chunk params when changing doc unless specified (we could keep them but usually it doesn't make sense)
        const currentDocParam = new URLSearchParams(window.location.search).get('doc')
        if (currentDocParam !== selectedDocumentId) {
          params.delete('chunk')
          params.delete('chunk_id')
        }

        const newUrl = `${window.location.pathname}?${params.toString()}`
        window.history.replaceState({}, '', newUrl)
      }
    } else {
      // If no document selected, remove the param if it exists
      if (params.has('doc')) {
        params.delete('doc')
        params.delete('chunk')
        params.delete('chunk_id')
        const newUrl = window.location.pathname + (params.toString() ? `?${params.toString()}` : '')
        window.history.replaceState({}, '', newUrl)
      }
    }
  }, [selectedDocumentId])

  // Summary-first loading effect
  useEffect(() => {
    let isSubscribed = true

    const fetchSummary = async (documentId: string) => {
      setIsLoading(true)
      setError(null)
      setSummaryData(null)
      setDocumentData(null)
      setPreviewState(initialPreviewState)
      // reset chunk pagination state
      setChunksOffset(0)
      setDocChunksTotal(null)
      setShowAllEntities({})
      try {
        performance.mark('document-summary-fetch-start')
        const summary = await api.getDocumentSummary(documentId)
        performance.mark('document-summary-fetch-end')
        performance.measure('document-summary-fetch', 'document-summary-fetch-start', 'document-summary-fetch-end')
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
      // reset chunk pagination state
      setChunksOffset(0)
      setDocChunksTotal(null)
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
    const needFull = (selectedDocumentId || summaryData?.id) && !documentData && (isChunksExpanded || isEntitiesExpanded || isMetadataExpanded)
    if (needFull) {
      // Immediately show a minimal document object derived from the summary so
      // sections can render without triggering an immediate metadata fetch
      if (summaryData && !documentData) {
        const minimal: any = {
          id: summaryData.id,
          title: summaryData.filename,
          file_name: summaryData.filename,
          original_filename: summaryData.filename,
          mime_type: undefined,
          preview_url: undefined,
          uploaded_at: undefined,
          uploader: undefined,
          summary: undefined,
          document_type: undefined,
          hashtags: [],
          chunks: [],
          entities: [],
          quality_scores: undefined,
          related_documents: [],
          metadata: {},
        }
        setDocumentData(minimal)
      }

      // Do not fetch full document metadata here — sections will request their
      // own data via paginated endpoints to avoid fetching large payloads.
    }
  }, [isChunksExpanded, isEntitiesExpanded, isMetadataExpanded, documentData, selectedDocumentId, summaryData?.id])

  // Load document metadata when Metadata section is expanded
  useEffect(() => {
    if (!isMetadataExpanded || !selectedDocumentId) return
    let isActive = true
    const loadMetadata = async () => {
      // If we already have metadata populated (more than just empty object), maybe skip?
      // But user might want fresh data. Let's fetch the full doc details if we don't have them or to ensure freshness.
      // However, api.getDocument is the main way to get metadata.
      try {
        const doc = await api.getDocument(selectedDocumentId)
        if (!isActive) return
        setDocumentData(prev => doc)
      } catch (err) {
        console.error('Failed to reload document metadata', err)
      }
    }
    void loadMetadata()
    return () => { isActive = false }
  }, [isMetadataExpanded, selectedDocumentId])

  // Load chunks when the Chunks section is expanded. Use dedicated endpoint so
  // we don't rely on `documentData.chunks` which may be empty for lightweight metadata responses.
  useEffect(() => {
    if (!isChunksExpanded) return
    let isActive = true
    const loadChunks = async () => {
      const id = selectedDocumentId || summaryData?.id
      if (!id) return
      // If we already loaded some chunks, skip initial load
      if ((documentData?.chunks?.length ?? 0) > 0) return
      try {
        setLoadingChunksData(true)
        const page = await api.getDocumentChunksPaginated(id, { limit: chunksPageSize, offset: 0 })
        if (!isActive) return
        setDocChunksTotal(page.total)
        setChunksOffset((page.chunks || []).length)
        const chunks = page.chunks || []
        setDocumentData((prev) => {
          if (prev) return { ...prev, chunks }
          return {
            id: id,
            title: summaryData?.filename ?? id,
            file_name: summaryData?.filename ?? id,
            original_filename: summaryData?.filename ?? id,
            mime_type: undefined,
            preview_url: undefined,
            uploaded_at: undefined,
            uploader: undefined,
            summary: undefined,
            document_type: undefined,
            hashtags: [],
            chunks,
            entities: [],
            quality_scores: undefined,
            related_documents: [],
            metadata: {},
          }
        })
      } catch (err) {
        console.error('Failed to load document chunks', err)
      } finally {
        if (isActive) setLoadingChunksData(false)
      }
    }
    void loadChunks()
    return () => { isActive = false }
  }, [isChunksExpanded, selectedDocumentId, summaryData?.id, documentData?.chunks])

  // Load entities when Entities section is expanded (first page only).
  useEffect(() => {
    if (!isEntitiesExpanded) return
    let isActive = true
    const loadEntities = async () => {
      const id = selectedDocumentId || summaryData?.id
      if (!id) return
      // If we already have loaded entities (by type), don't re-fetch the summary
      if (entitySummary && entitySummary.groups && entitySummary.groups.length > 0) return
      try {
        setLoadingEntitiesData(true)
        const summary = await api.getDocumentEntitySummary(id)
        if (!isActive) return
        setEntitySummary(summary)
      } catch (err) {
        console.error('Failed to load document entities', err)
      } finally {
        if (isActive) setLoadingEntitiesData(false)
      }
    }
    void loadEntities()
    return () => { isActive = false }
  }, [isEntitiesExpanded, selectedDocumentId, summaryData?.id, documentData?.entities])

  // Check preview availability - depends on document ID only to avoid infinite loops
  const documentId = documentData?.id
  useEffect(() => {
    let isSubscribed = true
    const checkPreview = async () => {
      if (!documentId) return
      try {
        const available = await api.hasDocumentPreview(documentId)
        if (isSubscribed) setHasPreview(available)
      } catch (e) {
        if (isSubscribed) setHasPreview(false)
      }
    }

    checkPreview()

    return () => {
      isSubscribed = false
    }
  }, [documentId])

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

  // Use summaryData.stats.chunks as fallback when documentData.chunks is empty (not just null/undefined)
  const docChunkCount = docChunksTotal ?? (documentData?.chunks?.length || summaryData?.stats.chunks) ?? 0
  const docEntitiesCount = (documentData?.entities?.length || summaryData?.stats.entities) ?? 0

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
    const newHashtags = documentData?.hashtags.filter(tag => tag !== tagToRemove)
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

  const handleStartEditMetadata = useCallback(() => {
    if (documentData?.metadata) {
      setEditedMetadata(JSON.stringify(documentData.metadata, null, 2))
    } else {
      setEditedMetadata('{}')
    }
    setIsEditingMetadata(true)
  }, [documentData?.metadata])

  const handleCancelEditMetadata = useCallback(() => {
    setIsEditingMetadata(false)
    setEditedMetadata('')
    setActionError(null)
  }, [])

  const handleSaveMetadata = useCallback(async () => {
    if (!documentData?.id) return
    setActionError(null)

    try {
      let parsedMetadata = {}
      try {
        parsedMetadata = JSON.parse(editedMetadata)
      } catch (e) {
        setActionError('Invalid JSON format')
        return
      }

      await api.updateDocumentMetadata(documentData.id, parsedMetadata)

      // Refresh document data
      const data = await api.getDocument(documentData.id)
      setDocumentData(data)
      setIsEditingMetadata(false)
      setActionMessage('Metadata updated successfully')
    } catch (saveError) {
      setActionError(
        saveError instanceof Error
          ? saveError.message
          : 'Failed to update metadata'
      )
    }
  }, [documentData?.id, editedMetadata])

  // Handler for updating document with new file (incremental update)
  const handleUpdateDocument = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file || !selectedDocumentId) return

    setIsUpdating(true)
    setActionError(null)
    setActionMessage(null)

    try {
      const result = await api.updateDocument(selectedDocumentId, file)

      if (result.status === 'success' && result.changes) {
        const { unchanged_chunks, added_chunks, removed_chunks } = result.changes
        setActionMessage(
          `Document updated: ${unchanged_chunks} chunks unchanged, ${added_chunks} added, ${removed_chunks} removed (${result.processing_time?.toFixed(2) || '0'}s)`
        )
        // Refresh document data
        const summary = await api.getDocumentSummary(selectedDocumentId)
        setSummaryData({ id: summary.id, filename: summary.filename, stats: summary.stats })
        if (documentData) {
          const doc = await api.getDocument(selectedDocumentId)
          setDocumentData(doc)
        }
      } else {
        setActionError(result.error || 'Update failed')
      }
    } catch (updateError) {
      setActionError(
        updateError instanceof Error
          ? updateError.message
          : 'Failed to update document'
      )
    } finally {
      setIsUpdating(false)
      // Reset file input
      if (updateFileInputRef.current) {
        updateFileInputRef.current.value = ''
      }
    }
  }, [selectedDocumentId, documentData])

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
            // Refresh summary stats (entities, relationships, communities, similarities)
            const updatedSummary = await api.getDocumentSummary(selectedDocumentId)
            setSummaryData({ id: updatedSummary.id, filename: updatedSummary.filename, stats: updatedSummary.stats })

            // Also refresh full document data if it's loaded
            if (documentData) {
              const updatedDoc = await api.getDocument(selectedDocumentId)
              setDocumentData(updatedDoc)
            }
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

  // Fetch global stats for empty state
  const [globalStats, setGlobalStats] = useState<{
    total_documents: number;
    total_chunks: number;
    total_entities: number;
    total_relationships: number;
  } | null>(null);

  useEffect(() => {
    if (!selectedDocumentId) {
      // Function to load stats
      const loadGlobalStats = () => {
        api.getStats().then(data => {
          setGlobalStats({
            total_documents: data.total_documents || 0,
            total_chunks: data.total_chunks || 0,
            total_entities: data.total_entities || 0,
            total_relationships: data.total_relationships || 0,
          });
        }).catch(() => {
          // Silently fail
        });
      };

      // Load stats on mount
      loadGlobalStats();

      // Listen for document processing events to refresh stats
      const handleStatsRefresh = () => {
        loadGlobalStats();
      };

      if (typeof window !== 'undefined') {
        window.addEventListener('documents:processed', handleStatsRefresh);
        window.addEventListener('documents:processing-updated', handleStatsRefresh);
        window.addEventListener('documents:uploaded', handleStatsRefresh);
        window.addEventListener('server:reconnected', handleStatsRefresh);
      }

      // Poll every 5 seconds for real-time updates while on dashboard
      const pollInterval = window.setInterval(loadGlobalStats, 5000);

      return () => {
        if (typeof window !== 'undefined') {
          window.removeEventListener('documents:processed', handleStatsRefresh);
          window.removeEventListener('documents:processing-updated', handleStatsRefresh);
          window.removeEventListener('documents:uploaded', handleStatsRefresh);
          window.removeEventListener('server:reconnected', handleStatsRefresh);
        }
        window.clearInterval(pollInterval);
      };
    }
  }, [selectedDocumentId]);

  if (!selectedDocumentId) {
    return (
      <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
        {/* Header */}
        <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '8px',
              backgroundColor: '#f27a0320',
              border: '1px solid #f27a03',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <DatabaseIconMui style={{ fontSize: '24px', color: '#f27a03' }} />
            </div>
            <div style={{ flex: 1 }}>
              <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)' }}>
                Database
              </h1>
              <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
                Document repository with chunks, entities, and relationships
              </p>
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="flex-1 p-6">
          <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Knowledge Base Overview
          </h2>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div
              className="rounded-lg p-5 border"
              style={{
                backgroundColor: 'var(--bg-secondary)',
                borderColor: 'var(--border)'
              }}
            >
              <div className="text-3xl font-bold mb-1" style={{ color: 'var(--accent-primary)' }}>
                {globalStats?.total_documents || 0}
              </div>
              <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>Documents</div>
            </div>
            <div
              className="rounded-lg p-5 border"
              style={{
                backgroundColor: 'var(--bg-secondary)',
                borderColor: 'var(--border)'
              }}
            >
              <div className="text-3xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
                {globalStats?.total_chunks || 0}
              </div>
              <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>Chunks</div>
            </div>
            <div
              className="rounded-lg p-5 border"
              style={{
                backgroundColor: 'var(--bg-secondary)',
                borderColor: 'var(--border)'
              }}
            >
              <div className="text-3xl font-bold mb-1" style={{ color: '#32D74B' }}>
                {globalStats?.total_entities || 0}
              </div>
              <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>Entities</div>
            </div>
            <div
              className="rounded-lg p-5 border"
              style={{
                backgroundColor: 'var(--bg-secondary)',
                borderColor: 'var(--border)'
              }}
            >
              <div className="text-3xl font-bold mb-1" style={{ color: '#BF5AF2' }}>
                {globalStats?.total_relationships || 0}
              </div>
              <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>Relationships</div>
            </div>
          </div>

          {/* Empty State */}
          <div className="flex flex-col items-center justify-center py-16 mt-8 rounded-lg border" style={{ borderColor: 'var(--border)', borderStyle: 'dashed' }}>
            <DocumentTextIcon className="w-12 h-12 mb-3" style={{ color: 'var(--text-tertiary)' }} />
            <p className="text-base font-medium" style={{ color: 'var(--text-secondary)' }}>
              Select a document from the sidebar to view its details
            </p>
            <p className="text-sm mt-1" style={{ color: 'var(--text-tertiary)' }}>
              Or upload a new document to get started
            </p>
          </div>
        </div>
      </div>
    )
  }

  // Previously we returned early to show only the summary. Instead, render the
  // summary as a top panel inside the full document view so the section headers
  // (Chunks, Entities, Graph, etc.) are visible and can be expanded to lazy-load
  // full document details.

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      {/* Header */}
      <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: 'var(--space-2)' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '8px',
            backgroundColor: '#f27a0320',
            border: '1px solid #f27a03',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <DocumentIconMui style={{ fontSize: '24px', color: '#f27a03' }} />
          </div>
          <div style={{ flex: 1 }}>
            <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)' }}>
              {documentData?.title || documentData?.original_filename || documentData?.file_name || summaryData?.filename || 'Document Details'}
            </h1>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              {documentData?.document_type ? documentData?.document_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : (summaryData?.filename ? (summaryData.filename.split('.').pop()?.toUpperCase() || 'Unknown type') : 'Unknown type')}
              {documentData?.uploaded_at && ` • Uploaded ${new Date(documentData?.uploaded_at).toLocaleString()}`}
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {(hasPreview === null ? documentData?.preview_url : hasPreview) && (
              <Button
                size="small"
                variant="contained"
                onClick={handleOpenPreview}
                disabled={previewState.isLoading}
                startIcon={<MagnifyingGlassIcon className="w-4 h-4" />}
                style={{
                  textTransform: 'none',
                  backgroundColor: 'var(--accent-primary)',
                  color: 'white',
                  fontSize: '0.75rem'
                }}
              >
                {previewState.isLoading ? 'Loading...' : 'Preview'}
              </Button>
            )}
            {/* Update Document Button */}
            <input
              ref={updateFileInputRef}
              type="file"
              id="document-update-file"
              className="hidden"
              onChange={handleUpdateDocument}
              accept=".pdf,.doc,.docx,.md,.txt,.pptx,.xlsx"
            />
            <Button
              size="small"
              variant="outlined"
              onClick={() => updateFileInputRef.current?.click()}
              disabled={isUpdating || disableManualActions}
              startIcon={isUpdating ? undefined : <ArrowUpTrayIcon className="w-4 h-4" />}
              title="Upload a new version of this document (only changed chunks will be reprocessed)"
              style={{
                textTransform: 'none',
                borderColor: 'var(--accent-primary)',
                color: 'var(--accent-primary)',
                fontSize: '0.75rem'
              }}
            >
              {isUpdating ? 'Updating...' : 'Update'}
            </Button>
          </div>
        </div>
        {previewState.error && (
          <div style={{ marginTop: '8px', fontSize: '0.75rem', color: 'var(--error)' }}>
            {previewState.error}
          </div>
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-6 pt-0 pb-6" style={{ backgroundColor: 'var(--bg-primary)' }}>
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

        {/* Summary will be rendered inside the main wrapper below so there's a single consistent container */}

        {!isLoading && !error && (documentData || summaryData) && (
          <div className="mt-3 space-y-4">
            {/* Inline summary area placed inside the main wrapper so layout is unified */}
            {summaryData && (
              <div>
                <h2 className="text-lg font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>{summaryData.filename}</h2>
                <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>ID: {summaryData.id}</p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs mb-2">
                  <div
                    className="p-3 rounded"
                    style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
                  >
                    <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>Chunks</p>
                    <p className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{summaryData.stats.chunks}</p>
                  </div>
                  <div
                    className="p-3 rounded"
                    style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
                  >
                    <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>Entities</p>
                    <p className="text-lg font-bold" style={{ color: '#32D74B' }}>{summaryData.stats.entities}</p>
                  </div>
                  <div
                    className="p-3 rounded"
                    style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
                  >
                    <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>Communities</p>
                    <p className="text-lg font-bold" style={{ color: '#BF5AF2' }}>{summaryData.stats.communities}</p>
                  </div>
                  <div
                    className="p-3 rounded"
                    style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
                  >
                    <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>Similarities</p>
                    <p className="text-lg font-bold" style={{ color: 'var(--accent-primary)' }}>{summaryData.stats.similarities}</p>
                  </div>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Expand a section (Chunks, Entities, Metadata) to load full details.</p>
              </div>
            )}
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
            {documentData?.metadata?.processing_status === 'staged' && (
              <div className="bg-blue-50 border border-blue-200 text-blue-700 rounded-lg px-4 py-3 text-sm">
                <p className="font-medium mb-1">Document ready to process</p>
                <p>This document has been uploaded but not yet processed. Go to the <strong>Upload</strong> tab in the sidebar and click <strong>Process All</strong> to begin processing.</p>
              </div>
            )}
            {/* Overview section removed to reduce vertical space; stats shown above. */}

            {/* Summary section removed: stats panel above provides the necessary overview. */}

            <section
              className="rounded-lg"
              style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
            >
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
                  {(documentData && ((documentData.chunks?.length ?? 0) === 0) && documentData.metadata?.processing_status !== 'staged' && !processingState?.is_processing) && (
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
                  <span className="text-xs text-secondary-500 dark:text-secondary-400">{docChunkCount} entries</span>
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
                    {loadingChunksData ? (
                      <div className="px-5 py-4 text-sm text-secondary-500 dark:text-secondary-400">
                        <Loader size={24} label="Loading document chunks…" />
                      </div>
                    ) : (!documentData || (documentData.chunks.length === 0)) ? (
                      <p className="px-5 py-4 text-sm text-secondary-500 dark:text-secondary-400">No chunks processed yet.</p>
                    ) : (
                      <>
                        {documentData!.chunks.map((chunk: DocumentChunk) => {
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
                        {docChunksTotal !== null && (docChunksTotal > (documentData?.chunks?.length ?? 0)) && (
                          <div className="px-5 py-4 border-t border-secondary-200">
                            <button
                              type="button"
                              onClick={async () => {
                                try {
                                  setLoadingMoreChunks(true)
                                  const id = selectedDocumentId || summaryData?.id
                                  if (!id) return
                                  const page = await api.getDocumentChunksPaginated(id, { limit: chunksPageSize, offset: chunksOffset })
                                  const more = page.chunks || []
                                  setChunksOffset((prev) => prev + more.length)
                                  setDocumentData((prev) => {
                                    if (!prev) return prev
                                    return { ...prev, chunks: (prev.chunks || []).concat(more) }
                                  })
                                } catch (err) {
                                  console.error('Failed to load more chunks', err)
                                } finally {
                                  setLoadingMoreChunks(false)
                                }
                              }}
                              className="button-secondary text-sm flex items-center gap-2"
                            >
                              {loadingMoreChunks ? 'Loading…' : `Load more (${Math.max(0, docChunksTotal - (documentData?.chunks?.length ?? 0))} remaining)`}
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
                  {(documentData && ((documentData.entities?.length ?? 0) === 0) && ((documentData.chunks?.length ?? 0) > 0) && documentData.metadata?.processing_status !== 'staged' && !processingState?.is_processing) && (
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
                    {docEntitiesCount > 0 ? `${docEntitiesCount} total` : `${summaryData?.stats.entities ?? 0} total`}
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
                    {loadingEntitiesData ? (
                      <div className="px-5 py-4 text-sm text-secondary-500 dark:text-secondary-400">
                        <Loader size={24} label="Loading entities…" />
                      </div>
                    ) : (!documentData || (documentData.entities?.length ?? 0) === 0) ? (
                      (entitySummary && entitySummary.groups && entitySummary.groups.length > 0) ? (
                        <div className="px-5 py-4">
                          <h4 className="text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">Entity counts by type</h4>
                          <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
                            {entitySummary.groups.map((g) => (
                              <li key={g.type} className="flex items-center justify-between border border-secondary-200 dark:border-secondary-700 rounded-lg px-3 py-2">
                                <div>
                                  <p className="text-sm font-medium text-secondary-900 dark:text-secondary-50">{g.type}</p>
                                  <p className="text-xs text-secondary-500 dark:text-secondary-400">{g.count} entities</p>
                                </div>
                                <div>
                                  <button
                                    type="button"
                                    onClick={async () => {
                                      try {
                                        setLoadingEntitiesData(true)
                                        const id = selectedDocumentId || summaryData?.id
                                        if (!id) return
                                        const page = await api.getDocumentEntitiesPaginated(id, { entityType: g.type, limit: 200, offset: 0 })
                                        const entities = page.entities || []
                                        setDocumentData((prev) => {
                                          if (!prev) return prev
                                          return { ...prev, entities }
                                        })
                                      } catch (err) {
                                        console.error('Failed to load entities for type', g.type, err)
                                      } finally {
                                        setLoadingEntitiesData(false)
                                      }
                                    }}
                                    className="button-secondary text-xs"
                                  >
                                    View
                                  </button>
                                </div>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ) : (
                        <p className="px-5 py-4 text-sm text-secondary-500 dark:text-secondary-400">No entities extracted yet.</p>
                      )
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

            {documentData?.quality_scores && (
              <section className="bg-white dark:bg-secondary-800 dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700 p-5">
                <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50 mb-3">Quality scores</h3>
                <pre className="bg-secondary-900 text-secondary-50 text-xs rounded-lg p-4 overflow-auto">
                  {JSON.stringify(documentData?.quality_scores, null, 2)}
                </pre>
              </section>
            )}

            {documentData?.related_documents && (documentData?.related_documents.length ?? 0) > 0 && (
              <section className="bg-white dark:bg-secondary-800 dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700 p-5">
                <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50 mb-3">Related documents</h3>
                <ul className="space-y-2">
                  {documentData?.related_documents?.map((doc) => (
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

            {/* Always render metadata section to allow editing, even if empty initially */}
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
                {isMetadataExpanded && !isEditingMetadata && (
                  <button
                    type="button"
                    onClick={handleStartEditMetadata}
                    className="button-secondary text-xs"
                  >
                    Edit
                  </button>
                )}
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
                    {isEditingMetadata ? (
                      <div className="p-4 bg-secondary-50 dark:bg-secondary-900">
                        <div className="mb-2">
                          <label htmlFor="metadata-editor" className="block text-xs font-medium text-secondary-700 dark:text-secondary-300 mb-1">
                            Edit Metadata (JSON)
                          </label>
                          <textarea
                            id="metadata-editor"
                            value={editedMetadata}
                            onChange={(e) => setEditedMetadata(e.target.value)}
                            className="w-full text-xs font-mono p-3 rounded border border-secondary-300 dark:border-secondary-600 bg-white dark:bg-secondary-800 focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                            rows={10}
                          />
                        </div>
                        <div className="flex justify-end gap-2">
                          <button
                            type="button"
                            onClick={handleCancelEditMetadata}
                            className="button-ghost text-xs"
                          >
                            Cancel
                          </button>
                          <button
                            type="button"
                            onClick={handleSaveMetadata}
                            className="button-primary text-xs"
                          >
                            Save Changes
                          </button>
                        </div>
                      </div>
                    ) : (
                      <pre className="bg-secondary-900 text-secondary-50 text-xs rounded-b-lg p-4 overflow-auto">
                        {JSON.stringify(documentData?.metadata ?? {}, null, 2)}
                      </pre>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </section>

            {/* Document-level graph and community info */}
            <section className="bg-white dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700">
              <header className="flex items-center justify-between px-5 py-4 border-b border-secondary-200 dark:border-secondary-700">
                <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50">Graph (document)</h3>
                <p className="text-xs text-secondary-500 dark:text-secondary-400">Interactive 3D view of entities in this document</p>
              </header>
              <div className="p-5">
                <DocumentGraph documentId={documentData?.id ?? selectedDocumentId} height={480} />
              </div>
            </section>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <CommunitiesSection documentId={documentData?.id ?? selectedDocumentId} />
              </div>
              <div>
                <ChunkSimilaritiesSection documentId={documentData?.id ?? selectedDocumentId} />
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
