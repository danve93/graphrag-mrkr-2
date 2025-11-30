'use client'

import { Message } from '@/types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import SourcesList from './SourcesList'
import QualityBadge from './QualityBadge'

interface MessageBubbleProps {
  message: Message
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const contextDocDisplay = message.context_document_labels && message.context_document_labels.length > 0
    ? message.context_document_labels
    : message.context_documents

  return (
    <div className={`flex message-fade-in ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`chat-message ${
          isUser ? 'chat-message-user' : 'chat-message-assistant relative pr-4 pl-4'
        }`}
      >

        <div className={isUser ? '' : 'prose prose-sm prose-slate dark:prose-invert max-w-none dark:text-secondary-100'}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
              {message.content}
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

        {message.isStreaming && (
          <div className="flex items-center mt-2" style={{ color: 'var(--text-secondary)' }}>
            <div className="spinner" style={{ width: '12px', height: '12px' }} />
          </div>
        )}

        {/* bottom area: sources list (left) and quality badge (anchored) */}
        {!isUser && ( (message.sources && message.sources.length > 0) || message.quality_score ) && (
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
      </div>
    </div>
  )
}
