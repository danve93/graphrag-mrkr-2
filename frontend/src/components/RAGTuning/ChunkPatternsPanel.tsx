'use client'

import { useState, useEffect, useCallback } from 'react'
import {
    Lightbulb,
    Trash2,
    Merge,
    Pencil,
    Flag,
    ToggleLeft,
    ToggleRight,
    Plus,
    Download,
    Upload,
    Check,
    X,
    Loader2
} from 'lucide-react'
import { api } from '@/lib/api'
import { showToast } from '@/components/Toast/ToastContainer'
import { useUIStore } from '@/store/uiStore'

interface Pattern {
    id: string
    name: string
    description: string
    match_type: string
    match_criteria: Record<string, any>
    action: string
    confidence: number
    is_builtin: boolean
    enabled: boolean
    usage_count: number
    last_used?: string
    created_at?: string
    updated_at?: string
}

const ACTION_ICONS: Record<string, typeof Trash2> = {
    delete: Trash2,
    merge: Merge,
    edit: Pencil,
    flag: Flag,
}

const ACTION_STYLES: Record<string, { bg: string, text: string, border: string }> = {
    delete: { bg: 'bg-red-500/10', text: 'text-red-500', border: 'border-red-500/20' },
    merge: { bg: 'bg-amber-500/10', text: 'text-amber-500', border: 'border-amber-500/20' },
    edit: { bg: 'bg-blue-500/10', text: 'text-blue-500', border: 'border-blue-500/20' },
    flag: { bg: 'bg-purple-500/10', text: 'text-purple-500', border: 'border-purple-500/20' },
}

export default function ChunkPatternsPanel() {
    const [patterns, setPatterns] = useState<Pattern[]>([])
    const [isLoading, setIsLoading] = useState(true)
    const [isUpdating, setIsUpdating] = useState<string | null>(null)
    const [editingPattern, setEditingPattern] = useState<Pattern | null>(null)

    const [showCreateModal, setShowCreateModal] = useState(false)
    const { showSuggestionIndicators, setShowSuggestionIndicators } = useUIStore()

    // Load patterns
    const loadPatterns = useCallback(async () => {
        try {
            const result = await api.getPatterns()
            setPatterns(result.patterns)
        } catch (error) {
            console.error('Failed to load patterns:', error)
            showToast('error', 'Failed to load patterns')
        } finally {
            setIsLoading(false)
        }
    }, [])

    useEffect(() => {
        loadPatterns()
    }, [loadPatterns])

    // Toggle pattern enabled/disabled
    const togglePattern = async (patternId: string, enabled: boolean) => {
        setIsUpdating(patternId)
        try {
            await api.togglePattern(patternId, enabled)
            setPatterns(prev => prev.map(p => p.id === patternId ? { ...p, enabled } : p))
            showToast('success', `Pattern ${enabled ? 'enabled' : 'disabled'}`)
        } catch (error) {
            console.error('Failed to toggle pattern:', error)
            showToast('error', 'Failed to update pattern')
        } finally {
            setIsUpdating(null)
        }
    }

    // Delete pattern
    const deletePattern = async (patternId: string) => {
        if (!confirm('Delete this pattern? This action cannot be undone.')) return

        setIsUpdating(patternId)
        try {
            await api.deletePattern(patternId)
            setPatterns(prev => prev.filter(p => p.id !== patternId))
            showToast('success', 'Pattern deleted')
        } catch (error) {
            console.error('Failed to delete pattern:', error)
            showToast('error', 'Failed to delete pattern')
        } finally {
            setIsUpdating(null)
        }
    }

    // Export patterns
    const exportPatterns = async () => {
        try {
            const data = await api.exportPatterns(false)
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = 'chunk-patterns.json'
            a.click()
            URL.revokeObjectURL(url)
            showToast('success', 'Patterns exported')
        } catch (error) {
            console.error('Failed to export patterns:', error)
            showToast('error', 'Failed to export patterns')
        }
    }

    // Import patterns
    const importPatterns = async () => {
        const input = document.createElement('input')
        input.type = 'file'
        input.accept = '.json'
        input.onchange = async (e) => {
            const file = (e.target as HTMLInputElement).files?.[0]
            if (!file) return

            try {
                const text = await file.text()
                const data = JSON.parse(text)
                const result = await api.importPatterns(data, false)
                showToast('success', `Imported: ${result.results.created} created, ${result.results.updated} updated`)
                loadPatterns()
            } catch (error) {
                console.error('Failed to import patterns:', error)
                showToast('error', 'Failed to import patterns')
            }
        }
        input.click()
    }

    if (isLoading) {
        return (
            <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 animate-spin" style={{ color: 'var(--accent-amber)' }} />
            </div>
        )
    }

    const builtinPatterns = patterns.filter(p => p.is_builtin)
    const customPatterns = patterns.filter(p => !p.is_builtin)

    return (
        <div className="space-y-6">
            {/* Header with actions */}
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="font-semibold text-[var(--text-primary)]">
                        Chunk Patterns
                    </h3>
                    <p className="text-sm text-[var(--text-secondary)]">
                        Patterns for auto-detecting chunk issues
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowSuggestionIndicators(!showSuggestionIndicators)}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-sm transition-colors ${showSuggestionIndicators
                                ? 'bg-amber-500/15 text-amber-500'
                                : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                            }`}
                        title="Show suggestion indicators in document list"
                    >
                        {showSuggestionIndicators ? <ToggleRight className="w-4 h-4" /> : <ToggleLeft className="w-4 h-4" />}
                        Indicators
                    </button>
                    <div className="h-6 w-px bg-gray-200 dark:bg-gray-700 mx-1" />
                    <button
                        onClick={exportPatterns}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded text-sm transition-colors bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]/80"
                        title="Export patterns"
                    >
                        <Download className="w-4 h-4" />
                        Export
                    </button>
                    <button
                        onClick={importPatterns}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded text-sm transition-colors bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]/80"
                        title="Import patterns"
                    >
                        <Upload className="w-4 h-4" />
                        Import
                    </button>
                    <button
                        onClick={() => setShowCreateModal(true)}
                        className="button-primary text-sm py-1.5 px-3"
                    >
                        <Plus className="w-4 h-4" />
                        New Pattern
                    </button>
                </div>
            </div>

            {/* Built-in patterns */}
            {builtinPatterns.length > 0 && (
                <div>
                    <h4 className="text-xs font-semibold uppercase tracking-wide mb-3 px-1 text-[var(--text-tertiary)]">
                        Built-in Patterns ({builtinPatterns.length})
                    </h4>
                    <div className="space-y-2">
                        {builtinPatterns.map(pattern => (
                            <PatternCard
                                key={pattern.id}
                                pattern={pattern}
                                onToggle={togglePattern}
                                isUpdating={isUpdating === pattern.id}
                            />
                        ))}
                    </div>
                </div>
            )}

            {/* Custom patterns */}
            <div>
                <h4 className="text-xs font-semibold uppercase tracking-wide mb-3 px-1 text-[var(--text-tertiary)]">
                    Custom Patterns ({customPatterns.length})
                </h4>
                {customPatterns.length === 0 ? (
                    <div className="text-center py-8 rounded-lg border border-dashed border-[var(--border)] text-[var(--text-secondary)]">
                        <Lightbulb className="w-8 h-8 mx-auto mb-2 opacity-40" />
                        <p>No custom patterns yet</p>
                        <p className="text-sm opacity-60">Create patterns from your chunk editing experience</p>
                    </div>
                ) : (
                    <div className="space-y-2">
                        {customPatterns.map(pattern => (
                            <PatternCard
                                key={pattern.id}
                                pattern={pattern}
                                onToggle={togglePattern}
                                onDelete={deletePattern}
                                isUpdating={isUpdating === pattern.id}
                            />
                        ))}
                    </div>
                )}
            </div>

            {/* Create Pattern Modal */}
            {showCreateModal && (
                <CreatePatternModal
                    onClose={() => setShowCreateModal(false)}
                    onCreated={(pattern) => {
                        setPatterns(prev => [...prev, pattern])
                        setShowCreateModal(false)
                    }}
                />
            )}
        </div>
    )
}

// Pattern Card Component
function PatternCard({
    pattern,
    onToggle,
    onDelete,
    isUpdating
}: {
    pattern: Pattern
    onToggle: (id: string, enabled: boolean) => void
    onDelete?: (id: string) => void
    isUpdating: boolean
}) {
    const ActionIcon = ACTION_ICONS[pattern.action] || Flag
    const styles = ACTION_STYLES[pattern.action] || { bg: 'bg-gray-500/10', text: 'text-gray-500', border: 'border-gray-500/20' }

    return (
        <div
            className={`flex items-center gap-3 p-3 rounded-lg border transition-all duration-200 border-[var(--border)] bg-[var(--bg-secondary)] hover:border-[var(--accent-primary)]/30 ${pattern.enabled ? 'opacity-100' : 'opacity-60'
                }`}
        >
            {/* Action icon */}
            <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${styles.bg}`}>
                <ActionIcon className={`w-4 h-4 ${styles.text}`} />
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                    <span className="font-medium text-sm text-[var(--text-primary)]">
                        {pattern.name}
                    </span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-medium ${styles.bg} ${styles.text}`}>
                        {pattern.action}
                    </span>
                    {pattern.is_builtin && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] uppercase bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
                            Built-in
                        </span>
                    )}
                </div>
                <p className="text-xs truncate text-[var(--text-secondary)]">
                    {pattern.description}
                </p>
                <div className="flex items-center gap-3 mt-1 text-[10px] text-[var(--text-tertiary)]">
                    <span>Confidence: {Math.round(pattern.confidence * 100)}%</span>
                    <span>Used: {pattern.usage_count}x</span>
                    <span>Type: {pattern.match_type}</span>
                </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2 shrink-0">
                {isUpdating ? (
                    <Loader2 className="w-4 h-4 animate-spin text-[var(--accent-primary)]" />
                ) : (
                    <>
                        <button
                            onClick={() => onToggle(pattern.id, !pattern.enabled)}
                            className={`p-1.5 rounded transition hover:bg-black/5 dark:hover:bg-white/5 ${pattern.enabled ? 'text-[var(--accent-primary)]' : 'text-[var(--text-tertiary)]'
                                }`}
                            title={pattern.enabled ? 'Disable' : 'Enable'}
                        >
                            {pattern.enabled ? <ToggleRight className="w-5 h-5" /> : <ToggleLeft className="w-5 h-5" />}
                        </button>
                        {!pattern.is_builtin && onDelete && (
                            <button
                                onClick={() => onDelete(pattern.id)}
                                className="p-1.5 rounded transition hover:bg-red-500/10 text-red-500"
                                title="Delete"
                            >
                                <Trash2 className="w-4 h-4" />
                            </button>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}

// Create Pattern Modal
function CreatePatternModal({
    onClose,
    onCreated
}: {
    onClose: () => void
    onCreated: (pattern: Pattern) => void
}) {
    const [name, setName] = useState('')
    const [description, setDescription] = useState('')
    const [matchType, setMatchType] = useState('regex')
    const [action, setAction] = useState('delete')
    const [confidence, setConfidence] = useState(0.7)
    const [criteria, setCriteria] = useState('')
    const [isCreating, setIsCreating] = useState(false)

    const handleCreate = async () => {
        if (!name.trim()) {
            showToast('warning', 'Name is required')
            return
        }

        setIsCreating(true)
        try {
            let match_criteria: Record<string, any> = {}
            if (criteria.trim()) {
                try {
                    match_criteria = JSON.parse(criteria)
                } catch {
                    showToast('error', 'Invalid JSON for criteria')
                    setIsCreating(false)
                    return
                }
            }

            const result = await api.createPattern({
                name: name.trim(),
                description: description.trim(),
                match_type: matchType,
                match_criteria,
                action,
                confidence,
            })
            onCreated(result.pattern as Pattern)
            showToast('success', 'Pattern created')
        } catch (error) {
            console.error('Failed to create pattern:', error)
            showToast('error', 'Failed to create pattern')
        } finally {
            setIsCreating(false)
        }
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: 'rgba(0, 0, 0, 0.6)' }}>
            <div
                className="w-full max-w-lg rounded-lg shadow-xl p-6"
                style={{ backgroundColor: 'var(--bg-primary)' }}
            >
                <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
                    Create New Pattern
                </h3>

                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Name</label>
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="e.g., Empty Paragraphs"
                            className="w-full px-3 py-2 rounded border text-sm"
                            style={{
                                backgroundColor: 'var(--bg-secondary)',
                                borderColor: 'var(--border)',
                                color: 'var(--text-primary)'
                            }}
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Description</label>
                        <input
                            type="text"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="What this pattern matches"
                            className="w-full px-3 py-2 rounded border text-sm"
                            style={{
                                backgroundColor: 'var(--bg-secondary)',
                                borderColor: 'var(--border)',
                                color: 'var(--text-primary)'
                            }}
                        />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Match Type</label>
                            <select
                                value={matchType}
                                onChange={(e) => setMatchType(e.target.value)}
                                className="w-full px-3 py-2 rounded border text-sm"
                                style={{
                                    backgroundColor: 'var(--bg-secondary)',
                                    borderColor: 'var(--border)',
                                    color: 'var(--text-primary)'
                                }}
                            >
                                <option value="regex">Regex</option>
                                <option value="length">Length</option>
                                <option value="content">Content</option>
                                <option value="similarity">Similarity</option>
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Action</label>
                            <select
                                value={action}
                                onChange={(e) => setAction(e.target.value)}
                                className="w-full px-3 py-2 rounded border text-sm"
                                style={{
                                    backgroundColor: 'var(--bg-secondary)',
                                    borderColor: 'var(--border)',
                                    color: 'var(--text-primary)'
                                }}
                            >
                                <option value="delete">Delete</option>
                                <option value="merge">Merge</option>
                                <option value="edit">Edit</option>
                                <option value="flag">Flag</option>
                            </select>
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>
                            Confidence: {Math.round(confidence * 100)}%
                        </label>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={confidence}
                            onChange={(e) => setConfidence(parseFloat(e.target.value))}
                            className="w-full"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>
                            Match Criteria (JSON)
                        </label>
                        <textarea
                            value={criteria}
                            onChange={(e) => setCriteria(e.target.value)}
                            placeholder='{"pattern": "^\\s*$"}'
                            rows={3}
                            className="w-full px-3 py-2 rounded border text-sm font-mono"
                            style={{
                                backgroundColor: 'var(--bg-secondary)',
                                borderColor: 'var(--border)',
                                color: 'var(--text-primary)'
                            }}
                        />
                    </div>
                </div>

                <div className="flex items-center justify-end gap-3 mt-6">
                    <button
                        onClick={onClose}
                        disabled={isCreating}
                        className="px-4 py-2 rounded text-sm font-medium transition"
                        style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleCreate}
                        disabled={isCreating || !name.trim()}
                        className="flex items-center gap-2 px-4 py-2 rounded text-sm font-medium transition disabled:opacity-50"
                        style={{ backgroundColor: 'var(--accent-amber)', color: 'white' }}
                    >
                        {isCreating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                        Create Pattern
                    </button>
                </div>
            </div>
        </div>
    )
}
