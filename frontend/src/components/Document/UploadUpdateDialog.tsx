'use client'

import { useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { XMarkIcon, DocumentTextIcon, ArrowPathIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline'
import { api } from '@/lib/api'
import Loader from '@/components/Utils/Loader'

interface SimilarDocument {
    document_id: string
    filename: string
    is_exact_match: boolean
    is_normalized_match: boolean
    similarity_score: number
    document_type?: string
    created_at?: string
    chunk_count: number
}

interface UploadUpdateDialogProps {
    file: File
    onClose: () => void
    onUploadNew: () => void
    onUpdateExisting: (documentId: string) => void
}

export default function UploadUpdateDialog({
    file,
    onClose,
    onUploadNew,
    onUpdateExisting,
}: UploadUpdateDialogProps) {
    const [matches, setMatches] = useState<SimilarDocument[]>([])
    const [allDocs, setAllDocs] = useState<SimilarDocument[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
    const [isUpdating, setIsUpdating] = useState(false)
    const [searchQuery, setSearchQuery] = useState('')
    const [mounted, setMounted] = useState(false)

    useEffect(() => {
        setMounted(true)
        loadDocuments()
    }, [file.name])

    const loadDocuments = async () => {
        setLoading(true)
        setError(null)
        try {
            // Get similar documents based on filename
            const result = await api.searchSimilarDocuments(file.name)
            setMatches(result.matches)

            // Also get all documents for search
            const stats = await api.getStats()
            const allDocsFormatted: SimilarDocument[] = (stats.documents || []).map((doc: any) => ({
                document_id: doc.document_id,
                filename: doc.original_filename || doc.filename || 'Unknown',
                is_exact_match: false,
                is_normalized_match: false,
                similarity_score: 0,
                document_type: doc.document_type,
                created_at: doc.created_at,
                chunk_count: doc.chunk_count || 0,
            }))
            setAllDocs(allDocsFormatted)

            // Auto-select first exact match
            const exactMatch = result.matches.find(m => m.is_exact_match)
            if (exactMatch) {
                setSelectedDocId(exactMatch.document_id)
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to search')
        } finally {
            setLoading(false)
        }
    }

    // Filter documents based on search query
    const getDisplayedDocs = (): SimilarDocument[] => {
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase()
            return allDocs.filter(doc =>
                doc.filename.toLowerCase().includes(query) ||
                (doc.document_type || '').toLowerCase().includes(query)
            )
        }
        return matches
    }

    const displayedDocs = getDisplayedDocs()

    const handleUpdate = async () => {
        if (!selectedDocId) return
        setIsUpdating(true)
        try {
            await onUpdateExisting(selectedDocId)
        } finally {
            setIsUpdating(false)
        }
    }

    const formatDate = (dateStr?: string) => {
        if (!dateStr) return ''
        try {
            return new Date(dateStr).toLocaleDateString()
        } catch {
            return ''
        }
    }

    if (!mounted) return null

    return createPortal(
        <div
            className="fixed inset-0 z-50 flex items-center justify-center font-sans"
            style={{ backgroundColor: 'rgba(0, 0, 0, 0.6)' }}
            onClick={onClose}
        >
            <div
                className="w-full max-w-lg mx-4 rounded-lg shadow-xl overflow-hidden flex flex-col max-h-[90vh]"
                style={{
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border)',
                }}
                onClick={e => e.stopPropagation()}
            >
                {/* Header */}
                <div
                    className="flex items-center justify-between px-5 py-4 flex-shrink-0"
                    style={{ borderBottom: '1px solid var(--border)' }}
                >
                    <h2
                        className="font-display text-lg font-semibold"
                        style={{ color: 'var(--text-primary)' }}
                    >
                        Upload or Update?
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-1 rounded hover:bg-secondary-700"
                        style={{ color: 'var(--text-secondary)' }}
                    >
                        <XMarkIcon className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="px-5 py-4 overflow-y-auto flex-1">
                    {/* File being uploaded */}
                    <div
                        className="flex items-center gap-3 p-3 rounded-lg mb-4"
                        style={{ backgroundColor: 'var(--bg-tertiary)' }}
                    >
                        <DocumentTextIcon className="w-5 h-5" style={{ color: 'var(--accent-primary)' }} />
                        <div className="flex-1 min-w-0">
                            <p
                                className="text-sm font-medium truncate"
                                style={{ color: 'var(--text-primary)' }}
                            >
                                {file.name}
                            </p>
                            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                                {(file.size / 1024).toFixed(1)} KB
                            </p>
                        </div>
                    </div>

                    {/* Search Input - Always visible */}
                    <div className="relative mb-4">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <MagnifyingGlassIcon className="h-4 w-4 text-secondary-400" />
                        </div>
                        <input
                            type="text"
                            placeholder="Search existing documents..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-9 pr-3 py-2 text-sm rounded-md border bg-transparent focus:outline-none focus:ring-2 focus:ring-primary-500 transition-colors"
                            style={{
                                borderColor: 'var(--border)',
                                color: 'var(--text-primary)'
                            }}
                        />
                    </div>

                    {loading ? (
                        <div className="flex justify-center py-8">
                            <Loader size={24} label="Searching for similar documents..." />
                        </div>
                    ) : error ? (
                        <div className="text-center py-6">
                            <p className="text-sm text-red-500">{error}</p>
                            <button
                                onClick={loadDocuments}
                                className="mt-2 text-sm flex items-center gap-1 mx-auto"
                                style={{ color: 'var(--accent-primary)' }}
                            >
                                <ArrowPathIcon className="w-4 h-4" /> Retry
                            </button>
                        </div>
                    ) : displayedDocs.length === 0 ? (
                        <div className="text-center py-6">
                            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                                {searchQuery ? 'No documents match your search.' : 'No similar documents found.'}
                            </p>
                            <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                                This will be uploaded as a new document.
                            </p>
                        </div>
                    ) : (
                        <>
                            <p className="text-sm mb-3" style={{ color: 'var(--text-secondary)' }}>
                                {searchQuery
                                    ? `Found ${displayedDocs.length} matching document${displayedDocs.length !== 1 ? 's' : ''}`
                                    : `Found ${matches.length} similar document${matches.length !== 1 ? 's' : ''}`
                                }
                                . Select one to update, or upload as new.
                            </p>

                            <div className="space-y-2">
                                {displayedDocs.map(doc => (
                                    <label
                                        key={doc.document_id}
                                        className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all ${selectedDocId === doc.document_id
                                            ? 'ring-2 ring-[var(--accent-primary)]'
                                            : 'hover:bg-opacity-80'
                                            }`}
                                        style={{
                                            backgroundColor: 'var(--bg-tertiary)',
                                        }}
                                    >
                                        <input
                                            type="radio"
                                            name="documentSelect"
                                            checked={selectedDocId === doc.document_id}
                                            onChange={() => setSelectedDocId(doc.document_id)}
                                            className="w-4 h-4"
                                            style={{ accentColor: 'var(--accent-primary)' }}
                                        />
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2">
                                                <p
                                                    className="text-sm font-medium truncate"
                                                    style={{ color: 'var(--text-primary)' }}
                                                >
                                                    {doc.filename}
                                                </p>
                                                {doc.is_exact_match && (
                                                    <span
                                                        className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                                                        style={{
                                                            backgroundColor: 'rgba(50, 215, 75, 0.2)',
                                                            color: '#32D74B'
                                                        }}
                                                    >
                                                        Exact
                                                    </span>
                                                )}
                                                {!doc.is_exact_match && doc.is_normalized_match && (
                                                    <span
                                                        className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                                                        style={{
                                                            backgroundColor: 'rgba(10, 132, 255, 0.2)',
                                                            color: '#0A84FF'
                                                        }}
                                                    >
                                                        Version
                                                    </span>
                                                )}
                                            </div>
                                            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                                                {doc.chunk_count} chunks
                                                {doc.document_type && ` • ${doc.document_type.replace(/_/g, ' ')}`}
                                                {doc.created_at && ` • ${formatDate(doc.created_at)}`}
                                            </p>
                                        </div>
                                        {doc.similarity_score > 0 && (
                                            <span
                                                className="text-xs"
                                                style={{ color: 'var(--text-secondary)' }}
                                            >
                                                {Math.round(doc.similarity_score * 100)}%
                                            </span>
                                        )}
                                    </label>
                                ))}
                            </div>
                        </>
                    )}
                </div>

                {/* Footer */}
                <div
                    className="flex justify-end gap-3 px-5 py-4 flex-shrink-0"
                    style={{ borderTop: '1px solid var(--border)' }}
                >
                    <button
                        onClick={onUploadNew}
                        className="button-secondary px-4 py-2 text-sm"
                        disabled={isUpdating}
                    >
                        Upload as New
                    </button>
                    {/* Show update button if we have matches OR if a document is manually selected via search */}
                    {(matches.length > 0 || selectedDocId) && (
                        <button
                            onClick={handleUpdate}
                            className="button-primary px-4 py-2 text-sm flex items-center gap-2"
                            disabled={!selectedDocId || isUpdating}
                        >
                            {isUpdating ? (
                                <Loader size={14} label="Updating..." />
                            ) : (
                                'Update Selected'
                            )}
                        </button>
                    )}
                </div>
            </div>
        </div>,
        document.body
    )
}
