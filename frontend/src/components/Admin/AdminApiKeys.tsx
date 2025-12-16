import React, { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { Trash, Plus, Key, Copy, Check } from 'lucide-react'
import { Button } from '@mui/material'

// Basic types for the API Key objects
interface ApiKey {
    id: string
    name: string
    role: string
    created_at: string
    key_masked?: string
    // Full key only present on creation
    key?: string
}

export default function AdminApiKeys() {
    const [keys, setKeys] = useState<ApiKey[]>([])
    const [loading, setLoading] = useState(true)
    const [creating, setCreating] = useState(false)

    // New key form state
    const [newName, setNewName] = useState('')
    const [newRole, setNewRole] = useState('external')
    const [createdKey, setCreatedKey] = useState<ApiKey | null>(null)

    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        loadKeys()
    }, [])

    async function loadKeys() {
        try {
            setLoading(true)
            const data = await api.getApiKeys()
            setKeys(data)
        } catch (e: any) {
            setError(e.message || 'Failed to load keys')
        } finally {
            setLoading(false)
        }
    }

    async function handleCreate(e: React.FormEvent) {
        e.preventDefault()
        if (!newName) return

        try {
            setCreating(true)
            const result = await api.createApiKey({
                name: newName,
                role: newRole,
                metadata: { source: 'admin_ui' }
            })

            setCreatedKey(result)
            setNewName('')
            loadKeys() // Refresh list
        } catch (e: any) {
            setError(e.message || 'Failed to create key')
        } finally {
            setCreating(false)
        }
    }

    async function handleRevoke(id: string) {
        if (!confirm('Are you sure you want to revoke this API Key? Integrations using it will stop working immediately.')) {
            return
        }

        try {
            await api.revokeApiKey(id)
            loadKeys()
        } catch (e: any) {
            setError(e.message || 'Failed to revoke key')
        }
    }

    return (
        <div className="max-w-4xl w-full">
            {error && (
                <div className="mb-4 p-4 bg-red-900/10 text-red-600 border border-red-900/20 rounded-md flex justify-between items-center">
                    <span>{error}</span>
                    <button className="font-bold hover:text-red-800" onClick={() => setError(null)}>×</button>
                </div>
            )}

            {/* Create Key Section */}
            <div className="bg-[var(--bg-secondary)] p-6 rounded-lg border border-[var(--border)] mb-8">
                <h2 className="text-lg font-semibold mb-4 text-[var(--text-primary)]">Create New API Key</h2>
                <form onSubmit={handleCreate} className="flex gap-4 items-end">
                    <div className="flex-1">
                        <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Name / Description</label>
                        <input
                            type="text"
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            placeholder="e.g. Product Name"
                            className="w-full px-3 py-2 bg-[var(--bg-primary)] border border-[var(--border)] rounded-md text-[var(--text-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
                            required
                        />
                    </div>
                    <div className="w-48">
                        <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Role</label>
                        <select
                            value={newRole}
                            onChange={(e) => setNewRole(e.target.value)}
                            className="block w-full px-3 py-2 bg-[var(--bg-primary)] border border-[var(--border)] rounded-md text-[var(--text-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
                        >
                            <option value="external">External (Standard)</option>
                            <option value="admin">Admin</option>
                        </select>
                    </div>
                    <Button
                        type="submit"
                        variant="contained"
                        disabled={creating}
                        style={{
                            backgroundColor: 'var(--accent)',
                            color: 'white',
                            textTransform: 'none',
                            height: '42px'
                        }}
                        startIcon={<Plus className="w-4 h-4" />}
                    >
                        Generate Key
                    </Button>
                </form>

                {/* Created Key Result */}
                {createdKey && (
                    <div className="mt-6 p-4 bg-green-900/10 border border-green-900/20 rounded-md animate-in fade-in slide-in-from-top-4">
                        <h3 className="text-green-600 font-bold mb-2 flex items-center">
                            <Check className="mr-2 h-5 w-5" /> Key Created Successfully
                        </h3>
                        <p className="text-sm text-green-600/80 mb-3">
                            Copy this key now. It will <strong>never</strong> be shown again.
                        </p>
                        <div className="flex items-center gap-2">
                            <code className="flex-1 bg-[var(--bg-primary)] p-3 rounded border border-green-900/20 font-mono text-sm break-all text-[var(--text-primary)]">
                                {createdKey.key}
                            </code>
                            <button
                                onClick={() => navigator.clipboard.writeText(createdKey.key || '')}
                                className="p-2 text-green-600 hover:bg-green-900/10 rounded transition-colors"
                                title="Copy to clipboard"
                            >
                                <Copy className="h-5 w-5" />
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* List Keys Section */}
            <div className="bg-[var(--bg-secondary)] rounded-lg border border-[var(--border)] overflow-hidden">
                <div className="px-6 py-4 border-b border-[var(--border)]">
                    <h2 className="text-lg font-semibold text-[var(--text-primary)]">Active API Keys</h2>
                </div>

                {loading ? (
                    <div className="p-8 text-center text-[var(--text-secondary)]">Loading keys...</div>
                ) : keys.length === 0 ? (
                    <div className="p-8 text-center text-[var(--text-secondary)]">No API keys found. Create one to get started.</div>
                ) : (
                    <table className="min-w-full divide-y divide-[var(--border)]">
                        <thead className="bg-[var(--bg-primary)]">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">Name</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">Masked Key</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">Role</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">Created</th>
                                <th className="px-6 py-3 text-right text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="bg-[var(--bg-secondary)] divide-y divide-[var(--border)]">
                            {keys.map((key) => (
                                <tr key={key.id} className="hover:bg-[var(--bg-hover)] transition-colors">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-[var(--text-primary)]">{key.name}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-[var(--text-secondary)] font-mono">{key.key_masked || '****'}</td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${key.role === 'admin'
                                            ? 'bg-purple-900/20 text-purple-400'
                                            : 'bg-green-900/20 text-green-400'
                                            }`}>
                                            {key.role}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-[var(--text-secondary)]">
                                        {new Date(key.created_at).toLocaleDateString()}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                        <button
                                            onClick={() => handleRevoke(key.id)}
                                            className="text-red-400 hover:text-red-300 flex items-center ml-auto transition-colors"
                                        >
                                            <Trash className="h-4 w-4 mr-1" /> Revoke
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    )
}

