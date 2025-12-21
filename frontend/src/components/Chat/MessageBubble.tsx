'use client'

import { Message } from '@/types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import SourcesList from './SourcesList'
import QualityBadge from './QualityBadge'
import { FeedbackButtons } from './FeedbackButtons'
import RoutingBadge from './RoutingBadge'
import CategoryRetry from './CategoryRetry'

const STAGE_LABELS: Record<string, string> = {
  query_analysis: 'Analyzing',
  routing: 'Routing',
  retrieval: 'Retrieving',
  consolidation: 'Consolidating',
  graph_reasoning: 'Reasoning',
  generation: 'Generating',
  quality_calculation: 'Checking',
  suggestions: 'Finalizing',
}

interface MessageBubbleProps {
  message: Message
  onRetryWithCategories?: (query: string, categories: string[]) => void
  currentStage?: string
}

export default function MessageBubble({ message, onRetryWithCategories, currentStage }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const contextDocDisplay = message.context_document_labels && message.context_document_labels.length > 0
    ? message.context_document_labels
    : message.context_documents
  // Safely stringify message content when it's not a string to avoid rendering objects
  const contentStr = typeof message.content === 'string'
    ? message.content
    : JSON.stringify(message.content, null, 2)

  // Token counts: use actual from backend for assistant, estimate for user
  const estimatedTokens = Math.ceil(contentStr.length / 4)
  const actualOutputTokens = message.output_tokens || 0
  const displayTokens = isUser ? estimatedTokens : (actualOutputTokens || estimatedTokens)
  const totalDurationSec = message.total_duration_ms ? (message.total_duration_ms / 1000).toFixed(1) : null

  return (
    <div className={`flex message-fade-in ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`chat-message ${isUser ? 'chat-message-user' : 'chat-message-assistant relative pr-4 pl-4'
          }`}
      >

        <div className={isUser ? '' : 'prose prose-sm prose-slate dark:prose-invert max-w-none dark:text-secondary-100'}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{contentStr}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
              {contentStr}
            </ReactMarkdown>
          )}
        </div>

        {isUser && Array.isArray(message.context_documents) && message.context_documents.length > 0 && contextDocDisplay && contextDocDisplay.length > 0 && (
          <div className="mt-2">
            <span className="inline-flex flex-wrap items-center" style={{ gap: 'var(--space-1)', borderRadius: 'var(--radius-sm)', background: 'rgba(255,255,255,0.15)', padding: 'var(--space-1) var(--space-3)', fontSize: 'var(--text-xs)', color: 'rgba(255,255,255,0.9)', maxWidth: '100%' }}>
              <span style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'rgba(255,255,255,0.7)', flexShrink: 0 }}>
                {message.context_hashtags && message.context_hashtags.length > 0
                  ? `${message.context_hashtags.map(tag => tag.startsWith('#') ? tag : `#${tag}`).join(', ')}:`
                  : (contextDocDisplay && contextDocDisplay.length > 1 ? 'Documents:' : 'Document:')}
              </span>
              <span
                className="break-words"
                title={contextDocDisplay.join(', ')}
              >
                {contextDocDisplay.join(', ')}
              </span>
            </span>
          </div>
        )}

        {/* Display routing badge for assistant messages */}
        {!isUser && message.routing_info && (
          <div className="mt-2 flex items-center gap-2">
            <RoutingBadge routingInfo={message.routing_info} />
            {onRetryWithCategories && (
              <CategoryRetry
                query={contentStr}
                currentCategories={message.routing_info.categories}
                onRetry={onRetryWithCategories}
              />
            )}
          </div>
        )}

        {message.isStreaming && (
          <div className="flex items-center gap-2 mt-3">
            <div className="flex items-center gap-1">
              <div className="typing-dot" />
              <div className="typing-dot" />
              <div className="typing-dot" />
            </div>
            {currentStage && (
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>
                {STAGE_LABELS[currentStage] || currentStage}
              </span>
            )}
          </div>
        )}

        {/* Feedback buttons for assistant messages */}
        {!isUser && !message.isStreaming && (() => {
          // Don't show feedback buttons for fallback messages (no sources = no context found)
          const hasSources = message.sources && message.sources.length > 0;

          if (!hasSources) return null;

          return (
            <div className="mt-2 flex justify-start">
              {message.message_id && message.session_id && (
                <FeedbackButtons
                  messageId={message.message_id}
                  sessionId={message.session_id}
                  query={contentStr}
                  routingInfo={message.stages?.find(s => s.name === 'routing')?.metadata || {}}
                />
              )}
            </div>
          );
        })()}

        {/* bottom area: sources list (left) and quality badge (anchored) */}
        {!isUser && ((message.sources && message.sources.length > 0) || message.quality_score) && (
          <div className="mt-4 pt-4 relative" style={{ borderTop: '1px solid var(--border)' }}>
            {message.sources && message.sources.length > 0 ? (
              <div className="min-w-0">
                <SourcesList sources={message.sources} />
              </div>
            ) : null}

            {message.quality_score && message.quality_score.total !== undefined && (
              // If sources are present, anchor badge to the top-right of the sources block
              // so it stays on the same horizontal line as the Sources header when expanded.
              // Otherwise, position it at the bubble's bottom-right.
              <div
                className={
                  message.sources && message.sources.length > 0
                    ? 'absolute top-4 right-3 pointer-events-auto'
                    : 'absolute bottom-3 right-3 pointer-events-auto'
                }
              >
                <QualityBadge score={message.quality_score} />
              </div>
            )}
          </div>
        )}

        {/* Duration and token count for assistant messages */}
        {!isUser && !message.isStreaming && (
          <div className="mt-3 flex items-center gap-2" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>
            <div className="flex items-center gap-1">
              <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--accent-primary)' }} />
              <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--accent-primary)' }} />
              <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--accent-primary)' }} />
            </div>
            <span>
              {totalDurationSec && `${totalDurationSec}s`}
              {totalDurationSec && ' · '}
              {actualOutputTokens ? displayTokens : `~${displayTokens}`} tokens
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
