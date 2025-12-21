'use client'

import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { X, Trash2, AlertTriangle, Database, MessageSquare } from 'lucide-react'

interface ClearDatabaseDialogProps {
    onClose: () => void
    onConfirm: (options: { clearKnowledgeBase: boolean; clearConversations: boolean }) => void
}

export default function ClearDatabaseDialog({
    onClose,
    onConfirm,
}: ClearDatabaseDialogProps) {
    const [mounted, setMounted] = useState(false)
    const [clearKnowledgeBase, setClearKnowledgeBase] = useState(true)
    const [clearConversations, setClearConversations] = useState(false)

    useEffect(() => {
        setMounted(true)
    }, [])

    if (!mounted) return null

    const handleConfirm = () => {
        onConfirm({ clearKnowledgeBase, clearConversations })
    }

    const isSafe = !clearKnowledgeBase && !clearConversations

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
                        <Trash2 className="w-5 h-5 text-red-600" />
                        <h2 className="font-display text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                            Clear Database
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

                <div className="px-5 py-4 space-y-5">
                    <div
                        className="flex items-start gap-3 p-3 rounded-lg"
                        style={{ backgroundColor: 'rgba(220, 38, 38, 0.1 border: 1px solid rgba(220, 38, 38, 0.2)' }}
                    >
                        <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                        <div className="text-sm">
                            <p className="font-medium text-red-600 mb-1">Warning: Unrecoverable Action</p>
                            <p style={{ color: 'var(--text-secondary)' }}>
                                This action cannot be undone. Please ensure you have backups if needed.
                            </p>
                        </div>
                    </div>

                    <div className="space-y-3">
                        <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                            Select data to clear:
                        </p>

                        <label className="flex items-start gap-3 p-3 rounded-lg border cursor-pointer hover:bg-secondary-50 dark:hover:bg-secondary-800 transition-colors"
                            style={{ borderColor: 'var(--border)' }}
                        >
                            <div className="flex items-center h-5 mt-0.5">
                                <input
                                    type="checkbox"
                                    className="w-4 h-4 rounded border-gray-300 text-red-600 focus:ring-red-500"
                                    checked={clearKnowledgeBase}
                                    onChange={(e) => setClearKnowledgeBase(e.target.checked)}
                                />
                            </div>
                            <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                    <Database className="w-4 h-4 text-secondary-500" />
                                    <span className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>Knowledge Base</span>
                                </div>
                                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                                    Includes Documents, text Chunks, extracted Entities, Community Summaries, and Folders.
                                </p>
                            </div>
                        </label>

                        <label className="flex items-start gap-3 p-3 rounded-lg border cursor-pointer hover:bg-secondary-50 dark:hover:bg-secondary-800 transition-colors"
                            style={{ borderColor: 'var(--border)' }}
                        >
                            <div className="flex items-center h-5 mt-0.5">
                                <input
                                    type="checkbox"
                                    className="w-4 h-4 rounded border-gray-300 text-red-600 focus:ring-red-500"
                                    checked={clearConversations}
                                    onChange={(e) => setClearConversations(e.target.checked)}
                                />
                            </div>
                            <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                    <MessageSquare className="w-4 h-4 text-secondary-500" />
                                    <span className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>Conversation History</span>
                                </div>
                                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                                    Includes all Chat sessions and Messages.
                                </p>
                            </div>
                        </label>
                    </div>

                    <div className="flex flex-col gap-2 pt-2">
                        <button
                            type="button"
                            onClick={handleConfirm}
                            disabled={isSafe}
                            className="button-primary text-sm bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed justify-center"
                        >
                            Clear Selected Data
                        </button>
                        <button
                            type="button"
                            onClick={onClose}
                            className="button-ghost text-sm justify-center"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        </div>,
        document.body
    )
}
