"use client"

import { useCallback, useEffect, useState, useRef, useMemo } from 'react'
import { X, Pencil, Trash2, Merge, Check, AlertCircle, Loader2, Save, XCircle, Lightbulb, RotateCcw, RotateCw } from 'lucide-react'
import { api } from '@/lib/api'
import { showToast } from '@/components/Toast/ToastContainer'

interface ChunkData {
    id: string
    text: string
    index: number
    originalText: string
}

interface ChunkSuggestion {
    chunk_id: string
    chunk_index: number
    action: 'delete' | 'merge' | 'edit' | 'split'
    confidence: number
    reasoning: string
    pattern_name: string
    related_chunk_ids?: string[]
}

type GlobalUndoAction =
    | { type: 'delete'; chunkId: string; content: string; index: number }
    | { type: 'merge'; mergedChunkId: string; originalChunks: { id: string; content: string; index: number }[] }

interface ChunkViewerModalProps {
    documentId: string
    documentName?: string
    isOpen: boolean
    onClose: () => void
    onChunksChanged?: () => void
}

// Color palette for chunk highlighting (cycling through)
const CHUNK_COLORS = [
    { bg: 'rgba(251, 191, 36, 0.15)', border: 'rgba(251, 191, 36, 0.6)', hover: 'rgba(251, 191, 36, 0.25)', selected: 'rgba(251, 191, 36, 0.35)' },
    { bg: 'rgba(96, 165, 250, 0.15)', border: 'rgba(96, 165, 250, 0.6)', hover: 'rgba(96, 165, 250, 0.25)', selected: 'rgba(96, 165, 250, 0.35)' },
    { bg: 'rgba(74, 222, 128, 0.15)', border: 'rgba(74, 222, 128, 0.6)', hover: 'rgba(74, 222, 128, 0.25)', selected: 'rgba(74, 222, 128, 0.35)' },
    { bg: 'rgba(251, 146, 60, 0.15)', border: 'rgba(251, 146, 60, 0.6)', hover: 'rgba(251, 146, 60, 0.25)', selected: 'rgba(251, 146, 60, 0.35)' },
    { bg: 'rgba(167, 139, 250, 0.15)', border: 'rgba(167, 139, 250, 0.6)', hover: 'rgba(167, 139, 250, 0.25)', selected: 'rgba(167, 139, 250, 0.35)' },
    { bg: 'rgba(244, 114, 182, 0.15)', border: 'rgba(244, 114, 182, 0.6)', hover: 'rgba(244, 114, 182, 0.25)', selected: 'rgba(244, 114, 182, 0.35)' },
]

export default function ChunkViewerModal({
    documentId,
    documentName,
    isOpen,
    onClose,
    onChunksChanged,
}: ChunkViewerModalProps) {
    const [chunks, setChunks] = useState<ChunkData[]>([])
    const [isLoading, setIsLoading] = useState(true)
    const [isSaving, setIsSaving] = useState(false)
    const [hoveredChunkId, setHoveredChunkId] = useState<string | null>(null)
    const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null)
    const [editingChunkId, setEditingChunkId] = useState<string | null>(null)
    const [editedContent, setEditedContent] = useState('')
    const [selectedForMerge, setSelectedForMerge] = useState<Set<string>>(new Set())
    const [showMergePreview, setShowMergePreview] = useState(false)
    const [mergedContent, setMergedContent] = useState('')
    const [suggestions, setSuggestions] = useState<ChunkSuggestion[]>([])
    const [showSuggestions, setShowSuggestions] = useState(true)
    const [undoStack, setUndoStack] = useState<string[]>([])
    const [redoStack, setRedoStack] = useState<string[]>([])
    // Global undo stack for structural changes (delete, merge)
    const [globalUndoStack, setGlobalUndoStack] = useState<GlobalUndoAction[]>([])

    const lastCaptureRef = useRef<number>(0)
    const contentRef = useRef<HTMLDivElement>(null)
    const editTextareaRef = useRef<HTMLTextAreaElement>(null)

    // Map suggestions by chunk_id for quick lookup
    const suggestionsByChunk = useMemo(() => {
        const map: Record<string, ChunkSuggestion> = {}
        for (const s of suggestions) {
            map[s.chunk_id] = s
        }
        return map
    }, [suggestions])

    // Large document performance: limit displayed chunks initially
    const INITIAL_CHUNK_LIMIT = 50
    const [displayLimit, setDisplayLimit] = useState(INITIAL_CHUNK_LIMIT)
    const visibleChunks = useMemo(() => {
        const sorted = [...chunks].sort((a, b) => a.index - b.index)
        return sorted.slice(0, displayLimit)
    }, [chunks, displayLimit])
    const hasMoreChunks = chunks.length > displayLimit

    // Load chunks and suggestions when modal opens
    useEffect(() => {
        if (!isOpen || !documentId) return

        const loadData = async () => {
            setIsLoading(true)
            try {
                // Load chunks
                const result = await api.getDocumentChunksPaginated(documentId, { limit: 2000, offset: 0 })
                const chunkData: ChunkData[] = result.chunks.map((c) => ({
                    id: String(c.id),
                    text: c.text,
                    index: c.index ?? 0,
                    originalText: c.text,
                }))
                chunkData.sort((a, b) => a.index - b.index)
                setChunks(chunkData)

                // Load suggestions (non-blocking)
                try {
                    const suggestionsResult = await api.getChunkSuggestions(documentId, { maxSuggestions: 20 })
                    setSuggestions(suggestionsResult.suggestions)
                } catch (sugErr) {
                    console.warn('Failed to load suggestions:', sugErr)
                }
            } catch (error) {
                console.error('Failed to load chunks:', error)
                showToast('error', 'Failed to load chunks')
            } finally {
                setIsLoading(false)
            }
        }

        loadData()
        return () => {
            setChunks([])
            setSuggestions([])
            setSelectedChunkId(null)
            setEditingChunkId(null)
            setSelectedForMerge(new Set())
            setShowMergePreview(false)
            setDisplayLimit(INITIAL_CHUNK_LIMIT)
            setGlobalUndoStack([]) // Reset global undo stack on close
        }
    }, [isOpen, documentId])

    const refreshChunks = useCallback(async () => {
        try {
            const result = await api.getDocumentChunksPaginated(documentId, { limit: 2000, offset: 0 })
            const chunkData: ChunkData[] = result.chunks.map((c) => ({
                id: String(c.id),
                text: c.text,
                index: c.index ?? 0,
                originalText: c.text,
            }))
            chunkData.sort((a, b) => a.index - b.index)
            setChunks(chunkData)
            onChunksChanged?.()
        } catch (error) {
            console.error('Failed to refresh chunks:', error)
        }
    }, [documentId, onChunksChanged])

    // Global Undo Handler
    const handleGlobalUndo = async () => {
        if (globalUndoStack.length === 0) return

        const action = globalUndoStack[globalUndoStack.length - 1]
        setIsSaving(true)

        try {
            if (action.type === 'delete') {
                // Restore deleted chunk
                await api.restoreChunk({
                    chunk_id: action.chunkId,
                    document_id: documentId,
                    content: action.content,
                    chunk_index: action.index
                })
                showToast('success', 'Restored deleted chunk')
            } else if (action.type === 'merge') {
                // Undo merge: delete merged chunk, then restore originals
                // 1. Unmerge (deletes the composite chunk)
                await api.unmergeChunks(action.mergedChunkId, documentId)

                // 2. Restore original chunks
                for (const chunk of action.originalChunks) {
                    await api.restoreChunk({
                        chunk_id: chunk.id,
                        document_id: documentId,
                        content: chunk.content,
                        chunk_index: chunk.index
                    })
                }
                showToast('success', 'Undid merge operation')
            }

            // Pop action from stack and refresh
            setGlobalUndoStack(prev => prev.slice(0, -1))
            await refreshChunks()

        } catch (error) {
            console.error('Undo failed:', error)
            showToast('error', 'Failed to undo action')
        } finally {
            setIsSaving(false)
        }
    }

    // Get color for a chunk by index
    const getChunkColor = useCallback((index: number) => {
        return CHUNK_COLORS[index % CHUNK_COLORS.length]
    }, [])

    // Token count estimate
    const estimateTokens = useCallback((text: string) => {
        return Math.ceil(text.length / 4)
    }, [])



    // Cancel editing
    const cancelEditing = useCallback(() => {
        setEditingChunkId(null)
        setEditedContent('')
        setUndoStack([])
        setRedoStack([])
    }, [])

    const toggleMergeSelection = useCallback((chunkId: string, e: React.MouseEvent) => {
        e.stopPropagation()
        const newSet = new Set(selectedForMerge)
        if (newSet.has(chunkId)) {
            newSet.delete(chunkId)
        } else {
            newSet.add(chunkId)
        }
        setSelectedForMerge(newSet)
    }, [selectedForMerge])

    // Suggestion logic
    const handleApplySuggestion = async (chunkId: string, action: string) => {
        if (action === 'delete') {
            await deleteChunk(chunkId, {} as React.MouseEvent)
        } else {
            // Placeholder for other actions or just open edit
            setEditingChunkId(chunkId)
            const chunk = chunks.find(c => c.id === chunkId)
            if (chunk) setEditedContent(chunk.text)
        }
    }

    // Handle single click - select chunk
    const handleChunkClick = useCallback((chunkId: string) => {
        if (editingChunkId) return
        setSelectedChunkId((prev) => (prev === chunkId ? null : chunkId))
    }, [editingChunkId])

    // Handle double click - start editing
    const handleChunkDoubleClick = useCallback((chunk: ChunkData) => {
        setEditingChunkId(chunk.id)
        setEditedContent(chunk.text)
        setUndoStack([])
        setRedoStack([])
        lastCaptureRef.current = 0
        setSelectedChunkId(chunk.id)
        setTimeout(() => editTextareaRef.current?.focus(), 100)
    }, [])

    // Undo/Redo Logic for EDITING
    const handleTextChange = useCallback((newText: string) => {
        const now = Date.now()
        // Capture snapshot if enough time passed or first edit
        if (now - lastCaptureRef.current > 1000) {
            setUndoStack(prev => [...prev, editedContent])
            lastCaptureRef.current = now
        }
        setEditedContent(newText)
        setRedoStack([])
    }, [editedContent])

    const handleUndo = useCallback(() => {
        if (undoStack.length === 0) return

        const previousState = undoStack[undoStack.length - 1]
        const newUndoStack = undoStack.slice(0, -1)

        setRedoStack(prev => [...prev, editedContent])
        setEditedContent(previousState)
        setUndoStack(newUndoStack)
    }, [undoStack, editedContent])

    const handleRedo = useCallback(() => {
        if (redoStack.length === 0) return

        const nextState = redoStack[redoStack.length - 1]
        const newRedoStack = redoStack.slice(0, -1)

        setUndoStack(prev => [...prev, editedContent])
        setEditedContent(nextState)
        setRedoStack(newRedoStack)
    }, [redoStack, editedContent])
    const hasChanges = useMemo(() => {
        if (!editingChunkId) return false
        const chunk = chunks.find((c) => c.id === editingChunkId)
        return chunk ? chunk.originalText !== editedContent : false
    }, [editingChunkId, editedContent, chunks])

    // Save chunk with conflict detection
    const saveChunk = useCallback(async () => {
        if (!editingChunkId || !editedContent.trim() || !hasChanges) return

        const chunk = chunks.find((c) => c.id === editingChunkId)
        if (!chunk) return

        setIsSaving(true)
        try {
            // Conflict detection: fetch current server content
            const serverData = await api.getChunkDetails(editingChunkId)
            const serverContent = serverData.content

            // Check if server content differs from what we loaded
            if (serverContent !== chunk.originalText) {
                const confirmOverwrite = confirm(
                    'This chunk has been modified since you opened the editor.\n\n' +
                    'Your version may overwrite changes made by another user or process.\n\n' +
                    'Do you want to overwrite with your changes?'
                )
                if (!confirmOverwrite) {
                    // Reload chunk with server content
                    setChunks((prev) =>
                        prev.map((c) =>
                            c.id === editingChunkId
                                ? { ...c, text: serverContent, originalText: serverContent }
                                : c
                        )
                    )
                    setEditedContent(serverContent)
                    showToast('success', 'Chunk refreshed', 'Loaded latest version from server')
                    setIsSaving(false)
                    return
                }
            }

            await api.updateChunkContent(editingChunkId, editedContent)
            setChunks((prev) =>
                prev.map((c) =>
                    c.id === editingChunkId
                        ? { ...c, text: editedContent, originalText: editedContent }
                        : c
                )
            )
            showToast('success', 'Chunk updated', 'Embedding regenerated')
            onChunksChanged?.()
            cancelEditing()
        } catch (error) {
            console.error('Failed to save chunk:', error)
            showToast('error', 'Failed to save chunk')
        } finally {
            setIsSaving(false)
        }
    }, [editingChunkId, editedContent, hasChanges, chunks, cancelEditing, onChunksChanged])

    // Delete chunk
    const deleteChunk = useCallback(async (chunkId: string, e: React.MouseEvent) => {
        e?.stopPropagation()
        if (!confirm('Delete this chunk? This action cannot be undone.')) return

        const chunkToDelete = chunks.find(c => c.id === chunkId)

        setIsSaving(true)
        try {
            await api.deleteChunk(chunkId)

            // Push to global undo stack
            if (chunkToDelete) {
                setGlobalUndoStack(prev => [...prev, {
                    type: 'delete',
                    chunkId: chunkToDelete.id,
                    content: chunkToDelete.text,
                    index: chunkToDelete.index
                }])
            }

            setChunks((prev) => prev.filter((c) => c.id !== chunkId))
            setSelectedChunkId(null)
            setSelectedForMerge((prev) => {
                const next = new Set(prev)
                next.delete(chunkId)
                return next
            })
            showToast('success', 'Chunk deleted')
            onChunksChanged?.()
        } catch (error) {
            console.error('Failed to delete chunk:', error)
            showToast('error', 'Failed to delete chunk')
        } finally {
            setIsSaving(false)
        }
    }, [chunks, onChunksChanged])

    // Initiate merge
    const initiateMerge = useCallback(() => {
        if (selectedForMerge.size < 2) {
            showToast('warning', 'Select at least 2 chunks to merge')
            return
        }

        const selectedChunks = chunks
            .filter((c) => selectedForMerge.has(c.id))
            .sort((a, b) => a.index - b.index)

        const combined = selectedChunks.map((c) => c.text).join('\n\n')
        setMergedContent(combined)
        setShowMergePreview(true)
    }, [selectedForMerge, chunks])

    // Execute merge
    const executeMerge = useCallback(async () => {
        if (!mergedContent.trim() || selectedForMerge.size < 2) return

        const chunkIds = Array.from(selectedForMerge)

        setIsSaving(true)
        try {
            const result = await api.mergeChunks(documentId, chunkIds, mergedContent)
            const mergedChunkId = result.merged_chunk.id

            // Push to global undo stack
            const originalChunks = chunks
                .filter(c => selectedForMerge.has(c.id))
                .map(c => ({ id: c.id, content: c.text, index: c.index }))

            setGlobalUndoStack(prev => [...prev, {
                type: 'merge',
                mergedChunkId: result.merged_chunk.id,
                originalChunks: originalChunks
            }])

            setChunks((prev) => {
                const filtered = prev.filter(
                    (c) => !chunkIds.includes(c.id) || c.id === mergedChunkId
                )
                return filtered.map((c) =>
                    c.id === mergedChunkId
                        ? { ...c, text: mergedContent, originalText: mergedContent }
                        : c
                )
            })

            setSelectedForMerge(new Set())
            setShowMergePreview(false)
            setMergedContent('')
            showToast('success', `Merged ${chunkIds.length} chunks`)
            onChunksChanged?.()
        } catch (error) {
            console.error('Failed to merge chunks:', error)
            showToast('error', 'Failed to merge chunks')
        } finally {
            setIsSaving(false)
        }
    }, [documentId, mergedContent, selectedForMerge, onChunksChanged, chunks])

    // Cancel merge
    const cancelMerge = useCallback(() => {
        setShowMergePreview(false)
        setMergedContent('')
    }, [])

    if (!isOpen) return null

    return (
        <div className="fixed inset-0 z-50 flex bg-black/80">
            <div className="flex-1 flex flex-col max-h-screen overflow-hidden bg-[var(--bg-primary)]">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b shrink-0 border-[var(--border)] bg-[var(--bg-secondary)]">
                    <div className="flex items-center gap-3">
                        <h2 className="text-xl font-semibold text-[var(--text-primary)]">
                            Chunk Editor
                        </h2>
                        {documentName && (
                            <span className="text-sm opacity-60 text-[var(--text-secondary)]">
                                {documentName}
                            </span>
                        )}
                        <span className="px-2 py-1 rounded text-xs bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
                            {chunks.length} chunks
                        </span>
                        {suggestions.length > 0 && (
                            <button
                                onClick={() => setShowSuggestions(!showSuggestions)}
                                className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${showSuggestions
                                    ? 'bg-amber-500/20 text-amber-500'
                                    : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)]'
                                    }`}
                                title={showSuggestions ? 'Hide suggestions' : 'Show suggestions'}
                            >
                                <Lightbulb className="w-3 h-3" />
                                {suggestions.length} suggestions
                            </button>
                        )}
                    </div>
                    <div className="flex items-center gap-3">
                        {/* Global Undo Button */}
                        <button
                            onClick={handleGlobalUndo}
                            disabled={globalUndoStack.length === 0 || isSaving}
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition-colors ${globalUndoStack.length > 0
                                ? 'bg-[var(--bg-tertiary)] hover:bg-[var(--bg-hover)] text-[var(--text-primary)]'
                                : 'opacity-40 cursor-not-allowed text-[var(--text-secondary)]'
                                }`}
                            title={globalUndoStack.length > 0 ? `Undo ${globalUndoStack[globalUndoStack.length - 1].type}` : 'No actions to undo'}
                        >
                            <RotateCcw className="w-4 h-4" />
                            Undo {globalUndoStack.length > 0 && <span className="text-[10px] uppercase ml-0.5 opacity-70">{globalUndoStack[globalUndoStack.length - 1].type}</span>}
                        </button>

                        {selectedForMerge.size >= 2 && !showMergePreview && (
                            <button
                                onClick={initiateMerge}
                                disabled={isSaving}
                                className="flex items-center gap-2 px-3 py-1.5 rounded text-sm font-medium transition-colors bg-[var(--accent-amber)] text-white hover:bg-[var(--accent-amber)]/90"
                            >
                                <Merge className="w-4 h-4" />
                                Merge {selectedForMerge.size} chunks
                            </button>
                        )}
                        {selectedForMerge.size > 0 && !showMergePreview && (
                            <button
                                onClick={() => setSelectedForMerge(new Set())}
                                className="text-sm underline opacity-70 hover:opacity-100 text-[var(--text-secondary)]"
                            >
                                Clear selection
                            </button>
                        )}
                        <button
                            onClick={onClose}
                            className="p-2 rounded-lg transition-colors hover:bg-black/5 dark:hover:bg-white/5 bg-[var(--bg-tertiary)]"
                        >
                            <X className="w-5 h-5 text-[var(--text-primary)]" />
                        </button>
                    </div>
                </div>

                {/* Main content area */}
                <div
                    ref={contentRef}
                    className="flex-1 overflow-y-auto p-6 bg-[var(--bg-primary)]"
                >
                    {isLoading ? (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 className="w-8 h-8 animate-spin text-[var(--accent-amber)]" />
                        </div>
                    ) : chunks.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-12 text-center">
                            <AlertCircle className="w-12 h-12 mb-4 opacity-30 text-[var(--text-secondary)]" />
                            <p className="text-[var(--text-secondary)]">No chunks found for this document</p>
                        </div>
                    ) : showMergePreview ? (
                        // Merge preview
                        <div className="max-w-4xl mx-auto space-y-4">
                            <h3 className="font-semibold text-[var(--text-primary)]">
                                Merge Preview - Edit combined content:
                            </h3>
                            <textarea
                                value={mergedContent}
                                onChange={(e) => setMergedContent(e.target.value)}
                                className="w-full h-[60vh] p-4 rounded-lg resize-none font-mono text-sm bg-[var(--bg-secondary)] text-[var(--text-primary)] border-2 border-[var(--accent-amber)] focus:outline-none"
                            />
                            <div className="flex items-center gap-4">
                                <button
                                    onClick={executeMerge}
                                    disabled={isSaving || !mergedContent.trim()}
                                    className="flex items-center gap-2 px-4 py-2 rounded font-medium transition disabled:opacity-50 bg-[var(--accent-amber)] text-white hover:bg-[var(--accent-amber)]/90"
                                >
                                    {isSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                                    Confirm Merge
                                </button>
                                <button
                                    onClick={cancelMerge}
                                    disabled={isSaving}
                                    className="px-4 py-2 rounded font-medium transition bg-[var(--bg-secondary)] text-[var(--text-primary)] border border-[var(--border)] hover:bg-[var(--bg-tertiary)]"
                                >
                                    Cancel
                                </button>
                                <span className="text-sm ml-auto text-[var(--text-secondary)]">
                                    ~{estimateTokens(mergedContent)} tokens
                                </span>
                            </div>
                        </div>
                    ) : (
                        // Unified document view with chunk highlighting
                        <div className="max-w-4xl mx-auto">
                            <p className="text-xs mb-4 opacity-60" style={{ color: 'var(--text-secondary)' }}>
                                Click to select • Double-click to edit • Use checkbox to select for merge
                            </p>
                            <div
                                className="rounded-lg p-1"
                                style={{ backgroundColor: 'var(--bg-secondary)' }}
                            >
                                {visibleChunks.map((chunk, idx) => {
                                    const color = getChunkColor(idx)
                                    const isHovered = hoveredChunkId === chunk.id
                                    const isSelected = selectedChunkId === chunk.id
                                    const isEditing = editingChunkId === chunk.id
                                    const isSelectedForMerge = selectedForMerge.has(chunk.id)
                                    const suggestion = showSuggestions ? suggestionsByChunk[chunk.id] : null

                                    // Determine background color based on state
                                    const getBgColor = () => {
                                        if (isSelected || isSelectedForMerge) return color.selected
                                        if (isHovered) return color.hover
                                        return 'transparent'
                                    }

                                    // Show border when selected, hovered, or merge-selected
                                    const showBorder = isSelected || isHovered || isSelectedForMerge

                                    return (
                                        <div
                                            key={chunk.id}
                                            className="relative"
                                            onMouseEnter={() => setHoveredChunkId(chunk.id)}
                                            onMouseLeave={() => setHoveredChunkId(null)}
                                        >
                                            {/* Chunk index badge - always visible when selected/merge-selected */}
                                            <div
                                                className="absolute -left-2 top-1 px-1.5 py-0.5 rounded text-[10px] font-mono z-10 transition-opacity"
                                                style={{
                                                    backgroundColor: color.border,
                                                    color: 'white',
                                                    opacity: (isHovered || isSelected || isSelectedForMerge) ? 1 : 0,
                                                }}
                                            >
                                                #{chunk.index}
                                            </div>

                                            {/* Suggestion badge - visible when chunk has suggestion */}
                                            {suggestion && (
                                                <div
                                                    className="absolute right-2 top-2 z-10 group"
                                                    title={`${suggestion.action.toUpperCase()}: ${suggestion.reasoning} (${Math.round(suggestion.confidence * 100)}% confidence)`}
                                                >
                                                    <div
                                                        className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium"
                                                        style={{
                                                            backgroundColor: suggestion.action === 'delete' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(251, 191, 36, 0.2)',
                                                            color: suggestion.action === 'delete' ? '#ef4444' : '#f59e0b',
                                                            border: `1px solid ${suggestion.action === 'delete' ? 'rgba(239, 68, 68, 0.4)' : 'rgba(251, 191, 36, 0.4)'}`,
                                                        }}
                                                    >
                                                        <Lightbulb className="w-3 h-3" />
                                                        {suggestion.action}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Merge checkbox - always visible when selected for merge, or on hover */}
                                            <div
                                                className="absolute -left-8 top-1/2 -translate-y-1/2 z-10 transition-opacity"
                                                style={{ opacity: (isHovered || isSelectedForMerge) ? 1 : 0 }}
                                            >
                                                <input
                                                    type="checkbox"
                                                    checked={isSelectedForMerge}
                                                    onChange={(e) => toggleMergeSelection(chunk.id, e as unknown as React.MouseEvent)}
                                                    onClick={(e) => e.stopPropagation()}
                                                    className="w-4 h-4 rounded cursor-pointer"
                                                    style={{ accentColor: 'var(--accent-amber)' }}
                                                    title="Select for merge"
                                                />
                                            </div>

                                            {/* Chunk content */}
                                            {isEditing ? (
                                                <div
                                                    className="p-3 my-1 rounded"
                                                    style={{ backgroundColor: color.selected, borderLeft: `3px solid ${color.border}` }}
                                                >
                                                    <textarea
                                                        ref={editTextareaRef}
                                                        value={editedContent}
                                                        onChange={(e) => handleTextChange(e.target.value)}
                                                        className="w-full h-48 p-3 rounded resize-y font-mono text-sm"
                                                        style={{
                                                            backgroundColor: 'var(--bg-primary)',
                                                            color: 'var(--text-primary)',
                                                            border: `2px solid ${color.border}`
                                                        }}
                                                    />
                                                    <div className="flex items-center gap-3 mt-3">
                                                        <div className="flex items-center gap-1 mr-2">
                                                            <button
                                                                onClick={handleUndo}
                                                                disabled={undoStack.length === 0}
                                                                className="p-1.5 rounded transition disabled:opacity-30 hover:bg-black/5 dark:hover:bg-white/10"
                                                                title="Undo"
                                                            >
                                                                <RotateCcw className="w-4 h-4" />
                                                            </button>
                                                            <button
                                                                onClick={handleRedo}
                                                                disabled={redoStack.length === 0}
                                                                className="p-1.5 rounded transition disabled:opacity-30 hover:bg-black/5 dark:hover:bg-white/10"
                                                                title="Redo"
                                                            >
                                                                <RotateCw className="w-4 h-4" />
                                                            </button>
                                                        </div>
                                                        <button
                                                            onClick={saveChunk}
                                                            disabled={isSaving || !hasChanges}
                                                            className="flex items-center gap-2 px-4 py-2 rounded text-sm font-medium transition disabled:opacity-50 bg-[var(--accent-amber)] text-white hover:bg-[var(--accent-amber)]/90"
                                                            title={!hasChanges ? 'No changes to save' : 'Save changes'}
                                                        >
                                                            {isSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                                                            Save
                                                        </button>
                                                        <button
                                                            onClick={cancelEditing}
                                                            disabled={isSaving}
                                                            className="flex items-center gap-2 px-4 py-2 rounded text-sm font-medium transition bg-[var(--bg-tertiary)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]/80"
                                                        >
                                                            <XCircle className="w-4 h-4" />
                                                            Cancel
                                                        </button>
                                                        <span className="text-xs ml-auto text-[var(--text-secondary)]">
                                                            ~{estimateTokens(editedContent)} tokens • {editedContent.length} chars
                                                            {hasChanges && <span className="ml-2 text-amber-500">• Modified</span>}
                                                        </span>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div
                                                    onClick={() => handleChunkClick(chunk.id)}
                                                    onDoubleClick={() => handleChunkDoubleClick(chunk)}
                                                    className="px-4 py-2 my-0.5 rounded cursor-pointer transition-all whitespace-pre-wrap font-mono text-sm leading-relaxed"
                                                    style={{
                                                        backgroundColor: getBgColor(),
                                                        borderLeft: showBorder ? `3px solid ${color.border}` : '3px solid transparent',
                                                        color: 'var(--text-primary)',
                                                    }}
                                                >
                                                    {chunk.text}

                                                    {/* Inline action buttons - visible when selected */}
                                                    {isSelected && !isEditing && (
                                                        <div
                                                            className="flex items-center gap-2 mt-3 pt-3 border-t"
                                                            style={{ borderColor: color.border }}
                                                            onClick={(e) => e.stopPropagation()}
                                                        >
                                                            <button
                                                                onClick={() => handleChunkDoubleClick(chunk)}
                                                                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition"
                                                                style={{ backgroundColor: 'var(--accent-amber)', color: 'white' }}
                                                            >
                                                                <Pencil className="w-3.5 h-3.5" />
                                                                Edit
                                                            </button>
                                                            <button
                                                                onClick={(e) => deleteChunk(chunk.id, e)}
                                                                disabled={isSaving}
                                                                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition hover:bg-red-500/20"
                                                                style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--error, #ef4444)' }}
                                                            >
                                                                <Trash2 className="w-3.5 h-3.5" />
                                                                Delete
                                                            </button>
                                                            <span className="text-xs ml-auto" style={{ color: 'var(--text-secondary)' }}>
                                                                ~{estimateTokens(chunk.text)} tokens • {chunk.text.length} chars
                                                            </span>
                                                        </div>
                                                    )}
                                                </div>
                                            )}

                                            {/* Chunk separator line */}
                                            {idx < visibleChunks.length - 1 && !isEditing && (
                                                <div
                                                    className="h-px mx-4 opacity-30"
                                                    style={{ backgroundColor: 'var(--border)' }}
                                                />
                                            )}
                                        </div>
                                    )
                                })}
                            </div>

                            {/* Show More button for large documents */}
                            {hasMoreChunks && (
                                <div className="flex justify-center mt-4 py-4">
                                    <button
                                        onClick={() => setDisplayLimit(chunks.length)}
                                        className="flex items-center gap-2 px-4 py-2 rounded text-sm font-medium transition bg-[var(--bg-tertiary)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]/80"
                                    >
                                        Show all {chunks.length - displayLimit} remaining chunks
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
