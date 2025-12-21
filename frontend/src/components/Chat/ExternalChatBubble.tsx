import { useState, useRef, useEffect } from 'react'
import { useChatStore } from '@/store/chatStore'
import { Send, Share2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '@/lib/api'
import SourcesList from './SourcesList'
import { FeedbackButtons } from './FeedbackButtons'

// Helper to generate UUID that works in non-HTTPS contexts
function generateUUID(): string {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID()
    }
    // Fallback for environments where crypto.randomUUID is not available
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0
        const v = c === 'x' ? r : (r & 0x3) | 0x8
        return v.toString(16)
    })
}

interface MessageData {
    message_id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: string
    sources?: any[]
    quality_score?: any
    session_id?: string
    isStreaming?: boolean
}

export default function ExternalChatBubble() {
    const { messages, addMessage, updateLastMessage, isHistoryLoading, user } = useChatStore()
    const sessionId = useChatStore((state) => state.sessionId)
    const setSessionId = useChatStore((state) => state.setSessionId)
    const [input, setInput] = useState('')
    const [sending, setSending] = useState(false)
    const [sharing, setSharing] = useState(false)
    const [isShared, setIsShared] = useState(false)
    const [chatTuningParams, setChatTuningParams] = useState({
        chunk_weight: 0.6,
        entity_weight: 0.4,
        path_weight: 0.6,
        max_hops: 2,
        beam_size: 8,
        graph_expansion_depth: 2,
        restrict_to_context: true,
        llm_model: undefined as string | undefined,
        embedding_model: undefined as string | undefined,
    })
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    // Fetch chat tuning parameters on mount (M3.5 External View Parity)
    useEffect(() => {
        const fetchChatTuningConfig = async () => {
            try {
                const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
                const response = await fetch(`${API_URL}/api/chat-tuning/config/values`)
                if (response.ok) {
                    const values = await response.json()
                    setChatTuningParams((prev) => ({
                        chunk_weight: values.chunk_weight ?? prev.chunk_weight,
                        entity_weight: values.entity_weight ?? prev.entity_weight,
                        path_weight: values.path_weight ?? prev.path_weight,
                        max_hops: values.max_hops ?? prev.max_hops,
                        beam_size: values.beam_size ?? prev.beam_size,
                        graph_expansion_depth: values.graph_expansion_depth ?? prev.graph_expansion_depth,
                        restrict_to_context: values.restrict_to_context ?? prev.restrict_to_context,
                        llm_model: values.llm_model ?? prev.llm_model,
                        embedding_model: values.embedding_model ?? prev.embedding_model,
                    }))
                }
            } catch (error) {
                console.error('Failed to fetch chat tuning config:', error)
            }
        }
        fetchChatTuningConfig()
    }, [])

    const handleToggleShare = async () => {
        const currentSessionId = useChatStore.getState().sessionId
        if (!currentSessionId || sharing) return

        try {
            setSharing(true)
            if (isShared) {
                await api.unshareSession(currentSessionId)
                setIsShared(false)
            } else {
                await api.shareSession(currentSessionId)
                setIsShared(true)
            }
        } catch (error) {
            console.error('Failed to toggle share:', error)
            alert(isShared ? 'Failed to unshare chat' : 'Failed to share chat with admin')
        } finally {
            setSharing(false)
        }
    }

    const handleSend = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim() || sending) return

        const content = input.trim()
        setInput('')
        setSending(true)

        // Add user message immediately
        const userMsgId = generateUUID()
        const userMsg: MessageData = {
            message_id: userMsgId,
            role: 'user' as const,
            content: content,
            timestamp: new Date().toISOString()
        }
        addMessage(userMsg)

        // Add placeholder assistant message with streaming state
        const assistantMsgId = generateUUID()
        const assistantMsg: MessageData = {
            message_id: assistantMsgId,
            role: 'assistant' as const,
            content: '',
            timestamp: new Date().toISOString(),
            isStreaming: true
        }
        addMessage(assistantMsg)

        try {
            const response = await api.sendMessage({
                message: content,
                session_id: useChatStore.getState().sessionId,
                stream: true,
                chunk_weight: chatTuningParams.chunk_weight,
                entity_weight: chatTuningParams.entity_weight,
                path_weight: chatTuningParams.path_weight,
                max_hops: chatTuningParams.max_hops,
                beam_size: chatTuningParams.beam_size,
                graph_expansion_depth: chatTuningParams.graph_expansion_depth,
                restrict_to_context: chatTuningParams.restrict_to_context,
                llm_model: chatTuningParams.llm_model,
                embedding_model: chatTuningParams.embedding_model,
            })

            if (!response.body) throw new Error('No response body')

            const reader = response.body.getReader()
            const decoder = new TextDecoder()
            let accumulatedContent = ''
            let sources: any[] = []
            let qualityScore: any = null
            let messageId: string | undefined
            let newSessionId: string | undefined

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                const chunk = decoder.decode(value)
                const lines = chunk.split('\n')

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6))
                            if (data.type === 'token') {
                                accumulatedContent += data.content
                                updateLastMessage((prev) => ({
                                    ...prev,
                                    content: accumulatedContent,
                                    isStreaming: true
                                }))
                            } else if (data.type === 'sources') {
                                if (Array.isArray(data.content)) {
                                    sources = [...sources, ...data.content]
                                }
                            } else if (data.type === 'quality_score') {
                                qualityScore = data.content
                            } else if (data.type === 'metadata') {
                                newSessionId = data.content.session_id
                                messageId = data.content.message_id
                                // Set sessionId in store for Share button
                                if (newSessionId && !useChatStore.getState().sessionId) {
                                    setSessionId(newSessionId)
                                }
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e)
                        }
                    }
                }
            }

            // Final update with sources and metadata, streaming complete
            updateLastMessage((prev) => ({
                ...prev,
                content: accumulatedContent,
                sources,
                quality_score: qualityScore,
                message_id: messageId,
                session_id: newSessionId,
                isStreaming: false
            }))

        } catch (err) {
            console.error(err)
            updateLastMessage((prev) => ({
                ...prev,
                content: "Sorry, I encountered an error providing a response.",
                isStreaming: false
            }))
        } finally {
            setSending(false)
        }
    }

    return (
        // Use fixed inset-0 z-50 to ensure we take over the full viewport and sit on top of everything
        <div className="fixed inset-0 z-50 flex flex-col bg-gray-50 dark:bg-neutral-900">
            {/* Header - Fixed */}
            <div className="flex-shrink-0 flex items-center justify-between p-4 border-b border-gray-200 dark:border-neutral-800 bg-white dark:bg-neutral-900">
                <div className="font-semibold text-gray-800 dark:text-gray-200">
                    Amber
                </div>
                <div className="flex items-center gap-3">
                    {user && (
                        <div className="text-xs text-gray-500 hidden sm:block">
                            {user.id}
                        </div>
                    )}
                    <button
                        onClick={handleToggleShare}
                        disabled={!sessionId || sharing}
                        className={`p-1.5 rounded-full transition-all border-2 ${!sessionId
                            ? 'opacity-40 cursor-not-allowed'
                            : isShared
                                ? 'text-white hover:bg-orange-600 hover:border-orange-600'
                                : 'hover:bg-orange-50 dark:hover:bg-orange-900/20'
                            }`}
                        style={{
                            borderColor: `var(--accent-primary)${!sessionId ? '66' : ''}`,
                            color: !sessionId ? 'var(--accent-primary)' : isShared ? 'white' : 'var(--accent-primary)',
                            backgroundColor: isShared ? 'var(--accent-primary)' : undefined,
                        }}
                        title={!sessionId ? "Start chatting to enable sharing" : isShared ? "Click to unshare" : "Share with Admin"}
                    >
                        {isShared ? (
                            <Share2 className={`w-5 h-5 ${sharing ? 'animate-pulse' : ''}`} fill="currentColor" />
                        ) : (
                            <Share2 className={`w-5 h-5 ${sharing ? 'animate-pulse' : ''}`} />
                        )}
                    </button>
                </div>
            </div>

            {/* Messages Area - Scrollable */}
            <div className="flex-1 overflow-y-auto">
                {messages.length === 0 ? (
                    <div className="flex h-full items-center justify-center p-4 text-gray-400 text-sm">
                        Start a conversation...
                    </div>
                ) : (
                    <div className="p-4 space-y-4">
                        {messages.map((msg) => (
                            <div
                                key={msg.message_id || Math.random().toString()}
                                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div
                                    className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm ${msg.role === 'user'
                                        ? 'text-white rounded-br-none'
                                        : 'bg-white dark:bg-neutral-800 border border-gray-200 dark:border-neutral-700 text-gray-800 dark:text-gray-200 rounded-bl-none shadow-sm'
                                        }`}
                                    style={{ backgroundColor: msg.role === 'user' ? 'var(--accent-primary)' : undefined }}
                                >
                                    <ReactMarkdown remarkPlugins={[remarkGfm]} className="prose dark:prose-invert max-w-none text-sm">
                                        {msg.content}
                                    </ReactMarkdown>

                                    {/* Streaming indicator */}
                                    {msg.role === 'assistant' && msg.isStreaming && (
                                        <div className="flex items-center mt-2 text-gray-400">
                                            <div className="spinner" style={{ width: '12px', height: '12px' }} />
                                        </div>
                                    )}

                                    {/* Sources for assistant messages */}
                                    {msg.role === 'assistant' && !msg.isStreaming && msg.sources && msg.sources.length > 0 && (
                                        <div className="mt-3 pt-3 border-t border-gray-200 dark:border-neutral-700">
                                            <SourcesList sources={msg.sources} />
                                        </div>
                                    )}

                                    {/* Feedback buttons for assistant messages */}
                                    {msg.role === 'assistant' && !msg.isStreaming && msg.message_id && msg.session_id && msg.sources && msg.sources.length > 0 && (
                                        <div className="mt-2">
                                            <FeedbackButtons
                                                messageId={msg.message_id}
                                                sessionId={msg.session_id}
                                                query={msg.content}
                                                routingInfo={{}}
                                            />
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input Area - Fixed at bottom */}
            <div className="flex-shrink-0 p-4 bg-white dark:bg-neutral-900 border-t border-gray-200 dark:border-neutral-800">
                <form onSubmit={handleSend} className="relative">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type a message..."
                        disabled={sending}
                        className="w-full pl-4 pr-12 py-3 rounded-full bg-gray-100 dark:bg-neutral-800 border-transparent focus:bg-white dark:focus:bg-neutral-800 focus:ring-0 text-sm transition-all"
                        style={{ outlineColor: 'var(--accent-primary)' }}
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || sending}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 text-white rounded-full disabled:opacity-50 disabled:cursor-not-allowed hover:bg-orange-600 transition-colors"
                        style={{ backgroundColor: 'var(--accent-primary)' }}
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </form>
            </div>
        </div>
    )
}
