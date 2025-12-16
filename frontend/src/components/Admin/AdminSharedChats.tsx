import React, { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { MessageSquare, Calendar, Users, ArrowRight } from 'lucide-react'
import { useChatStore } from '@/store/chatStore'

interface SharedSession {
    session_id: string
    title: string
    created_at: string
    updated_at: string
    message_count: number
    preview: string
    user_id?: string
}

export default function AdminSharedChats() {
    const [sessions, setSessions] = useState<SharedSession[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    const loadSession = useChatStore(state => state.loadSession)
    const setActiveView = useChatStore(state => state.setActiveView)

    useEffect(() => {
        loadChats()
    }, [])

    async function loadChats() {
        try {
            setLoading(true)
            const data = await api.getSharedChats()
            setSessions(data)
        } catch (e: any) {
            setError(e.message || 'Failed to load shared chats')
        } finally {
            setLoading(false)
        }
    }

    async function handleViewSession(sessionId: string) {
        try {
            await loadSession(sessionId)
            setActiveView('chat')
        } catch (e: any) {
            alert('Failed to load session: ' + e.message)
        }
    }

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
                        <Users className="text-[#f27a03]" size={24} />
                    </div>
                    <div style={{ flex: 1 }}>
                        <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)' }}>
                            Shared Chats
                        </h1>
                        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
                            Manage conversations shared by external users
                        </p>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
                {error && (
                    <div className="mb-4 p-4 rounded-md border text-sm"
                        style={{ background: '#FECACA30', borderColor: '#FECACA', color: '#DC2626' }}>
                        {error}
                    </div>
                )}

                {loading ? (
                    <div className="flex items-center justify-center h-48">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#f27a03]"></div>
                    </div>
                ) : sessions.length === 0 ? (
                    <div className="flex flex-col items-center justify-center p-12 text-center border rounded-lg border-dashed"
                        style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}>
                        <MessageSquare className="w-12 h-12 mb-4 opacity-50" />
                        <h3 className="text-lg font-medium mb-1">No Shared Conversations</h3>
                        <p className="text-sm">External users haven&apos;t shared any chats yet.</p>
                    </div>
                ) : (
                    <div className="grid gap-4">
                        {sessions.map((session) => (
                            <div
                                key={session.session_id}
                                className="group p-5 rounded-lg border transition-all hover:shadow-sm"
                                style={{
                                    backgroundColor: 'var(--bg-secondary)',
                                    borderColor: 'var(--border)'
                                }}
                            >
                                <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-center">
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2 mb-1">
                                            <h3 className="text-lg font-semibold truncate hover:text-[#f27a03] transition-colors"
                                                style={{ color: 'var(--text-primary)' }}>
                                                {session.title || 'Untitled Conversation'}
                                            </h3>
                                            <span className="px-2 py-0.5 rounded-full text-[10px] font-medium border"
                                                style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}>
                                                {session.session_id.substring(0, 8)}
                                            </span>
                                        </div>

                                        <p className="text-sm line-clamp-2 mb-3" style={{ color: 'var(--text-secondary)' }}>
                                            {session.preview || 'No preview available...'}
                                        </p>

                                        <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--text-tertiary)' }}>
                                            <span className="flex items-center">
                                                <Calendar className="w-3.5 h-3.5 mr-1.5" />
                                                {new Date(session.updated_at || session.created_at).toLocaleDateString()}
                                            </span>
                                            <span className="flex items-center">
                                                <MessageSquare className="w-3.5 h-3.5 mr-1.5" />
                                                {session.message_count} messages
                                            </span>
                                        </div>
                                    </div>

                                    <button
                                        onClick={() => handleViewSession(session.session_id)}
                                        className="shrink-0 px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center border"
                                        style={{
                                            backgroundColor: 'var(--bg-primary)',
                                            borderColor: 'var(--border)',
                                            color: 'var(--text-primary)'
                                        }}
                                    >
                                        View Chat <ArrowRight className="ml-2 h-4 w-4" />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}
