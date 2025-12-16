'use client'

import { useState, useRef, useEffect } from 'react'
import { X, Send, MessageSquare, Loader2 } from 'lucide-react'
import { Button } from '@mui/material'
import type { GraphNode } from '@/types/graph'
import { API_URL } from '@/lib/api'

interface FocusedChatPanelProps {
    node: GraphNode
    onClose: () => void
}

interface ChatMessage {
    role: 'user' | 'assistant'
    content: string
}

export default function FocusedChatPanel({ node, onClose }: FocusedChatPanelProps) {
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const handleSend = async () => {
        if (!input.trim() || isLoading) return

        const userMessage = input.trim()
        setInput('')
        setMessages(prev => [...prev, { role: 'user', content: userMessage }])
        setIsLoading(true)

        try {
            const contextPrompt = `Regarding the entity "${node.label}"${node.description ? ` (${node.description})` : ''}: ${userMessage}`

            const response = await fetch(`${API_URL}/api/chat/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: contextPrompt,
                    stream: false,
                }),
            })

            if (!response.ok) {
                throw new Error('Failed to get response')
            }

            const data = await response.json()
            setMessages(prev => [...prev, { role: 'assistant', content: data.response || 'No response received.' }])
        } catch (error) {
            console.error('Chat error:', error)
            setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }])
        } finally {
            setIsLoading(false)
        }
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="fixed inset-0 z-[120] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <div className="w-full max-w-2xl h-[80vh] bg-[var(--bg-secondary)] rounded-xl shadow-2xl flex flex-col overflow-hidden border border-[var(--border)]">
                {/* Header */}
                <div className="p-4 border-b border-[var(--border)] flex items-center justify-between bg-[var(--bg-secondary)]">
                    <div className="flex items-center gap-3">
                        <div
                            className="w-10 h-10 rounded-lg flex items-center justify-center shadow-sm"
                            style={{ backgroundColor: node.color || 'var(--accent-primary)' }}
                        >
                            <MessageSquare size={20} className="text-white" />
                        </div>
                        <div>
                            <h2 className="font-bold text-[var(--text-primary)]">Chat: {node.label}</h2>
                            <p className="text-xs text-[var(--text-secondary)]">Ask questions about this entity</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] transition-colors"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-[var(--bg-primary)]">
                    {messages.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-center text-[var(--text-secondary)]">
                            <MessageSquare size={48} className="mb-4 opacity-30" />
                            <p className="text-sm">Start a conversation about <strong className="text-[var(--text-primary)]">{node.label}</strong></p>
                            <p className="text-xs mt-2">Your questions will be focused on this entity&apos;s context.</p>
                        </div>
                    )}
                    {messages.map((msg, idx) => (
                        <div
                            key={idx}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`chat-message ${msg.role === 'user' ? 'chat-message-user' : 'chat-message-assistant'}`}
                            >
                                {msg.content}
                            </div>
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="bg-[var(--bg-secondary)] rounded-2xl rounded-bl-none px-4 py-3 border border-[var(--border)]">
                                <Loader2 size={16} className="animate-spin text-accent-primary" />
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="p-4 border-t border-[var(--border)] bg-[var(--bg-secondary)]">
                    <div className="flex items-center gap-2">
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={`Ask about ${node.label}...`}
                            className="input-field max-h-32"
                            rows={1}
                            disabled={isLoading}
                        />
                        <Button
                            variant="contained"
                            onClick={handleSend}
                            disabled={!input.trim() || isLoading}
                            sx={{
                                minWidth: 48,
                                height: 48,
                                borderRadius: '8px',
                                backgroundColor: 'var(--accent-primary)',
                                '&:hover': { backgroundColor: 'var(--accent-hover)' },
                            }}
                        >
                            <Send size={20} />
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    )
}
