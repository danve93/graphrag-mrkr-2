'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { flushSync } from 'react-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Message } from '@/types'
import { api, API_URL } from '@/lib/api'
import { MessageSquare } from 'lucide-react'
import MessageBubble from './MessageBubble'
import ChatInput from './ChatInput'
import FollowUpQuestions from './FollowUpQuestions'
import LoadingIndicator from './LoadingIndicator'
import ConnectionStatus from './ConnectionStatus'
import { useChatStore } from '@/store/chatStore'

export default function ChatInterface() {
  const messages = useChatStore((state) => state.messages)
  const sessionId = useChatStore((state) => state.sessionId)
  const addMessage = useChatStore((state) => state.addMessage)
  const updateLastMessage = useChatStore((state) => state.updateLastMessage)
  const setSessionId = useChatStore((state) => state.setSessionId)
  const clearChat = useChatStore((state) => state.clearChat)
  const notifyHistoryRefresh = useChatStore((state) => state.notifyHistoryRefresh)
  const isHistoryLoading = useChatStore((state) => state.isHistoryLoading)
  const isConnected = useChatStore((state) => state.isConnected)
  const setIsConnected = useChatStore((state) => state.setIsConnected)
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const [currentStage, setCurrentStage] = useState<string>('query_analysis')
  const [completedStages, setCompletedStages] = useState<string[]>([])
  const [stageUpdates, setStageUpdates] = useState<any[]>([])
  const [settings, setSettings] = useState<any>(null)
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
  const healthCheckIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Fetch settings and chat tuning parameters on mount
  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await api.getSettings()
        setSettings(response)
      } catch (error) {
        console.error('Failed to fetch settings:', error)
      }
    }

    const fetchChatTuningConfig = async () => {
      try {
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

    fetchSettings()
    fetchChatTuningConfig()
  }, [])

  // Health check monitoring with adaptive interval and debouncing
  useEffect(() => {
    let isUnmounted = false;
    let consecutiveFailures = 0;
    const MAX_FAILURES_BEFORE_DISCONNECT = 2; // Require 2 consecutive failures

    const HEALTHY_INTERVAL = 60000; // 60 seconds when connected (reduced frequency)
    const UNHEALTHY_INTERVAL = 10000; // 10 seconds when disconnected (less aggressive)

    const scheduleNextCheck = (delay: number) => {
      if (healthCheckIntervalRef.current) {
        clearTimeout(healthCheckIntervalRef.current as unknown as number)
      }
      healthCheckIntervalRef.current = setTimeout(runHealthCheck, delay)
    }

    const runHealthCheck = async () => {
      if (isUnmounted) return;

      // Skip health check if actively loading (query in progress)
      if (isLoading) {
        scheduleNextCheck(HEALTHY_INTERVAL); // Check again later
        return;
      }

      const isHealthy = await api.checkHealth()

      if (isHealthy) {
        consecutiveFailures = 0;
        setIsConnected(true)
        scheduleNextCheck(HEALTHY_INTERVAL)
      } else {
        consecutiveFailures++;
        // Only mark as disconnected after multiple consecutive failures
        if (consecutiveFailures >= MAX_FAILURES_BEFORE_DISCONNECT) {
          setIsConnected(false)
        }
        scheduleNextCheck(UNHEALTHY_INTERVAL)
      }
    }

    // Initial optimistic connection state
    setIsConnected(true)

    // Start health check loop after a delay (let app load first)
    const initialDelay = setTimeout(() => {
      if (!isUnmounted) runHealthCheck()
    }, 3000);

    return () => {
      isUnmounted = true;
      clearTimeout(initialDelay);
      if (healthCheckIntervalRef.current) {
        clearTimeout(healthCheckIntervalRef.current as unknown as number)
      }
    }
  }, [setIsConnected, isLoading])

  // Trigger refresh when connection is restored
  useEffect(() => {
    if (isConnected && typeof window !== 'undefined') {
      // Only emit refresh event if we were previously disconnected
      if (window.dispatchEvent) {
        window.dispatchEvent(new CustomEvent('server:reconnected'))
      }
    }
  }, [isConnected])

  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
  }

  const handleNewChat = useCallback(() => {
    handleStopStreaming()
    // Current session is already saved to history automatically
    // Just clear the UI and reset session ID
    setCurrentStage('query_analysis')
    setCompletedStages([])
    clearChat()
  }, [clearChat])

  // Keyboard shortcut for new chat
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Only trigger if 'n' is pressed and not in an input/textarea
      if (event.key === 'n' &&
        !(document.activeElement instanceof HTMLInputElement) &&
        !(document.activeElement instanceof HTMLTextAreaElement) &&
        !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault()
        handleNewChat()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleNewChat])

  const handleSendMessage = async (
    message: string,
    contextDocuments: string[],
    contextDocumentLabels: string[],
    contextHashtags?: string[],
    categoryFilter?: string[]
  ) => {
    if (!message.trim() || isLoading || isHistoryLoading) return

    // Reset stage indicators for new query
    setCurrentStage('query_analysis')
    setCompletedStages([])
    setStageUpdates([])

    // Add user message
    const userMessage: Message = {
      role: 'user',
      content: message,
      timestamp: new Date().toISOString(),
      context_documents: contextDocuments,
      context_document_labels: contextDocumentLabels,
      context_hashtags: contextHashtags,
    }
    addMessage(userMessage)
    setIsLoading(true)

    let accumulatedContent = ''
    let displayedContent = ''
    let sources: any[] = []
    let qualityScore: any = null
    let followUpQuestions: string[] = []
    let routingInfo: any = null
    let newSessionId = sessionId
    let messageId: string | undefined = undefined
    let inputTokens = 0
    let outputTokens = 0
    let streamCompleted = false
    const streamStartTime = Date.now()
    let animationFrameId: number | null = null
    let contentBuffer: string[] = []
    let effectiveContextDocs = [...contextDocuments]
    const contextDocLabelMap = new Map<string, string>()
    contextDocuments.forEach((id, index) => {
      const label = contextDocumentLabels[index]
      if (label) {
        contextDocLabelMap.set(id, label)
      }
    })
    let effectiveContextDocLabels = [...contextDocumentLabels]

    // Smooth rendering using requestAnimationFrame
    const smoothRender = () => {
      if (contentBuffer.length > 0) {
        // Take a chunk from the buffer
        const chunkSize = Math.ceil(contentBuffer.length / 3) || 1
        const nextChunk = contentBuffer.splice(0, chunkSize).join('')
        displayedContent += nextChunk

        updateLastMessage((prev) => ({
          ...prev,
          content: displayedContent,
        }))
      }

      // Continue rendering if there's still content in the buffer or stream is ongoing
      if (contentBuffer.length > 0 || !streamCompleted) {
        animationFrameId = requestAnimationFrame(smoothRender)
      } else if (streamCompleted && accumulatedContent !== displayedContent) {
        // Ensure all content is displayed
        displayedContent = accumulatedContent
        updateLastMessage((prev) => ({
          ...prev,
          content: displayedContent,
          isStreaming: false,
          sources,
          quality_score: qualityScore,
          follow_up_questions: followUpQuestions,
        }))
      }
    }

    try {
      // Add placeholder for assistant message
      const assistantMessage: Message = {
        role: 'assistant',
        content: '',
        isStreaming: true,
      }
      addMessage(assistantMessage)

      const controller = new AbortController()
      abortControllerRef.current = controller

      const response = await api.sendMessage(
        {
          message,
          session_id: sessionId,
          stream: true,
          context_documents: contextDocuments,
          context_document_labels: contextDocumentLabels,
          context_hashtags: contextHashtags,
          category_filter: categoryFilter,
          chunk_weight: chatTuningParams.chunk_weight,
          entity_weight: chatTuningParams.entity_weight,
          path_weight: chatTuningParams.path_weight,
          max_hops: chatTuningParams.max_hops,
          beam_size: chatTuningParams.beam_size,
          graph_expansion_depth: chatTuningParams.graph_expansion_depth,
          restrict_to_context: chatTuningParams.restrict_to_context,
          llm_model: chatTuningParams.llm_model,
          embedding_model: chatTuningParams.embedding_model,
        },
        { signal: controller.signal }
      )

      if (!response.body) {
        throw new Error('No response body')
      }

      // Start smooth rendering
      animationFrameId = requestAnimationFrame(smoothRender)

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

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
                // Add to buffer instead of updating immediately
                contentBuffer.push(data.content)
              } else if (data.type === 'stage') {
                flushSync(() => {
                  // Handle both legacy string format and new dict format
                  const stageName = typeof data.content === 'string' ? data.content : data.content
                  setCurrentStage(stageName)

                  // Store stage update with timing metadata
                  const stageUpdate: any = {
                    name: stageName,
                    duration_ms: data.duration_ms,
                    timestamp: data.timestamp,
                    metadata: data.metadata || {}
                  }
                  setStageUpdates((prev) => [...prev, stageUpdate])

                  // Track completed stages
                  setCompletedStages((prev) => {
                    if (!prev.includes(stageName)) {
                      return [...prev, stageName]
                    }
                    return prev
                  })
                })
              } else if (data.type === 'sources') {
                // Accumulate sources from batches
                if (Array.isArray(data.content)) {
                  sources = [...sources, ...data.content]
                }
              } else if (data.type === 'quality_score') {
                qualityScore = data.content
              } else if (data.type === 'follow_ups') {
                followUpQuestions = data.content
              } else if (data.type === 'metadata') {
                newSessionId = data.content.session_id
                messageId = data.content.message_id
                inputTokens = data.content.input_tokens || 0
                outputTokens = data.content.output_tokens || 0
                if (Array.isArray(data.content.context_documents)) {
                  effectiveContextDocs = data.content.context_documents
                  effectiveContextDocLabels = data.content.context_documents.map(
                    (docId: string) => contextDocLabelMap.get(docId) || docId
                  )
                }
                // Capture routing info from metadata
                if (data.content.metadata?.routing_info) {
                  routingInfo = data.content.metadata.routing_info
                }
              } else if (data.type === 'done') {
                streamCompleted = true
              } else if (data.type === 'error') {
                throw new Error(data.content)
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }

      // Mark stream as completed and let animation frame finish
      streamCompleted = true

      // Wait for animation to complete
      await new Promise((resolve) => {
        const checkComplete = () => {
          if (contentBuffer.length === 0 && displayedContent === accumulatedContent) {
            resolve(null)
          } else {
            setTimeout(checkComplete, 50)
          }
        }
        checkComplete()
      })

      // Calculate total duration using wall-clock time
      const totalDuration = Date.now() - streamStartTime

      // Final update with all metadata
      updateLastMessage((prev) => ({
        ...prev,
        content: accumulatedContent,
        isStreaming: false,
        sources,
        quality_score: qualityScore,
        follow_up_questions: followUpQuestions,
        context_documents: effectiveContextDocs,
        context_document_labels: effectiveContextDocLabels,
        stages: stageUpdates,
        total_duration_ms: totalDuration,
        message_id: messageId,
        session_id: newSessionId,
        routing_info: routingInfo,
        input_tokens: inputTokens,
        output_tokens: outputTokens,
      }))

      if (newSessionId && !sessionId) {
        setSessionId(newSessionId)
      }

      notifyHistoryRefresh()
    } catch (error) {
      // Cancel animation frame on error
      if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId)
      }

      if (error instanceof DOMException && error.name === 'AbortError') {
        streamCompleted = true
        // Let buffer finish rendering
        await new Promise((resolve) => setTimeout(resolve, 100))

        updateLastMessage((prev) => ({
          ...prev,
          content: accumulatedContent || displayedContent || prev.content || '',
          isStreaming: false,
          sources: sources.length > 0 ? sources : prev.sources,
          quality_score:
            qualityScore !== null && qualityScore !== undefined
              ? qualityScore
              : prev.quality_score,
          follow_up_questions:
            followUpQuestions.length > 0 ? followUpQuestions : prev.follow_up_questions,
          context_documents: effectiveContextDocs,
          context_document_labels: effectiveContextDocLabels,
        }))
      } else {
        console.error('Error sending message:', error)
        updateLastMessage(() => ({
          role: 'assistant',
          content: 'Sorry, I encountered an error processing your request. Please try again.',
          isStreaming: false,
          context_documents: effectiveContextDocs,
          context_document_labels: effectiveContextDocLabels,
        }))
      }
    } finally {
      // Ensure animation frame is cancelled
      if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId)
      }
      abortControllerRef.current = null
      setIsLoading(false)
    }
  }

  const handleFollowUpClick = (question: string) => {
    // Prefer explicit context documents from the last assistant message,
    // fall back to sources if needed.
    const lastAssistantMsg = [...messages].reverse().find((m) => m.role === 'assistant')

    const contextDocIds = (lastAssistantMsg?.context_documents || []).filter(
      (id): id is string => !!id
    )
    const contextLabels = (lastAssistantMsg?.context_document_labels || []).filter(
      (label): label is string => !!label
    )

    const sourceDocIds =
      lastAssistantMsg?.sources
        ?.map((source) => source.document_id)
        .filter((id): id is string => !!id) || []
    const uniqueSourceDocIds = [...new Set(sourceDocIds)]

    const sourceLabels =
      lastAssistantMsg?.sources
        ?.map((source) => source.document_name || source.filename)
        .filter((label): label is string => !!label) || []
    const uniqueSourceLabels = [...new Set(sourceLabels)]

    const effectiveDocIds = contextDocIds.length > 0 ? contextDocIds : uniqueSourceDocIds
    const effectiveLabels =
      contextLabels.length > 0 ? contextLabels : uniqueSourceLabels

    handleSendMessage(question, effectiveDocIds, effectiveLabels)
  }

  const handleRetryWithCategories = (query: string, categories: string[]) => {
    // Send the query again with category filter override
    handleSendMessage(query, [], [], undefined, categories)
  }

  // Get user messages for history navigation
  const userMessages = messages
    .filter((msg) => msg.role === 'user')
    .map((msg) => (typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)))

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      {/* Connection Status Alert */}
      <ConnectionStatus />

      {/* Header */}
      <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '8px',
            backgroundColor: 'var(--accent-subtle)',
            border: '1px solid var(--accent-primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <MessageSquare size={24} style={{ color: 'var(--accent-primary)' }} />
          </div>
          <div style={{ flex: 1 }}>
            <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)' }}>
              {messages.length > 0 && messages[0].role === 'user'
                ? (() => {
                  const c = messages[0].content
                  const s = typeof c === 'string' ? c : JSON.stringify(c)
                  return s.slice(0, 100) + (s.length > 100 ? '...' : '')
                })()
                : 'Chat Interface'
              }
            </h1>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              {messages.length > 0
                ? `${messages.length} message${messages.length !== 1 ? 's' : ''} in this conversation`
                : 'Graph-enhanced RAG with streaming responses and provenance'
              }
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 md:px-6 pt-6 pb-32">
        {messages.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            className="flex flex-col items-center justify-center h-full text-center"
          >
            <motion.div
              animate={{
                scale: [1, 1.05, 1],
                opacity: [0.8, 1, 0.8],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
              className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
              style={{ backgroundColor: 'var(--accent-subtle)' }}
            >
              <MessageSquare className="w-8 h-8" style={{ color: 'var(--accent-primary)' }} />
            </motion.div>
            <h2 className="text-xl font-semibold text-secondary-700 dark:text-secondary-300 mb-2">
              Start a Conversation
            </h2>
            <p className="text-secondary-500 dark:text-secondary-400 max-w-md">
              Upload some documents and start asking questions. I&apos;ll help you find
              relevant information and provide intelligent answers.
            </p>
          </motion.div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-4">
            <AnimatePresence mode="popLayout">
              {messages.map((message, index) => (
                <motion.div
                  key={message.message_id || `msg-${index}`}
                  initial={{ opacity: 0, y: 20, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.98 }}
                  transition={{
                    type: 'spring',
                    stiffness: 400,
                    damping: 30,
                    delay: index < 5 ? index * 0.05 : 0, // Only stagger first 5 messages
                  }}
                  layout={!message.isStreaming}
                >
                  <MessageBubble message={message} onRetryWithCategories={handleRetryWithCategories} currentStage={message.isStreaming ? currentStage : undefined} />
                  {message.role === 'assistant' &&
                    !message.isStreaming &&
                    message.follow_up_questions &&
                    message.follow_up_questions.length > 0 &&
                    index === messages.length - 1 && (
                      <FollowUpQuestions
                        questions={message.follow_up_questions}
                        onQuestionClick={handleFollowUpClick}
                      />
                    )}
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input - Sticky to bottom */}
      <div className="sticky bottom-0 left-0 right-0 border-t px-4 md:px-6 py-4 pb-28 md:pb-28 mt-auto" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
        <div className="max-w-4xl mx-auto">
          <ChatInput
            onSend={handleSendMessage}
            onStop={handleStopStreaming}
            disabled={isHistoryLoading}
            isStreaming={isLoading}
            userMessages={userMessages}
          />
        </div>
      </div>
    </div>
  )
}
