# Chat Interface Component

Real-time conversational UI with streaming responses and context management.

## Overview

The Chat Interface is the primary user interaction point for Amber's RAG system. Built with Next.js and React, it provides a responsive chat experience with Server-Sent Events (SSE) streaming, message history, source citations, quality indicators, and follow-up suggestions.

**Location**: `frontend/src/components/chat/`
**Framework**: Next.js 14 (App Router), React 18, TypeScript
**State Management**: Zustand + React Context

## Architecture

```
┌────────────────────────────────────────────────────────┐
│              Chat Interface Architecture                │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          ChatInterface Component                │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │  MessageList                              │  │   │
│  │  │  ┌─────────────────────────────────────┐  │  │   │
│  │  │  │ UserMessage                         │  │  │   │
│  │  │  │ AssistantMessage                    │  │  │   │
│  │  │  │   ├─ StreamingText (tokens)         │  │  │   │
│  │  │  │   ├─ StageIndicator (progress)      │  │  │   │
│  │  │  │   ├─ SourceList (citations)         │  │  │   │
│  │  │  │   ├─ QualityBadge (score)           │  │  │   │
│  │  │  │   └─ FollowUpQuestions              │  │  │   │
│  │  │  └─────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │  ChatInput                                │  │   │
│  │  │  ┌─────────────────────────────────────┐  │  │   │
│  │  │  │ Textarea (auto-resize)              │  │  │   │
│  │  │  │ SendButton (disabled while sending) │  │  │   │
│  │  │  │ CharacterCount (optional)           │  │  │   │
│  │  │  └─────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          State Management (Zustand)             │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ messages: Message[]                       │  │   │
│  │  │ isStreaming: boolean                      │  │   │
│  │  │ currentSessionId: string                  │  │   │
│  │  │ addMessage(), updateMessage()             │  │   │
│  │  │ startStreaming(), stopStreaming()         │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          SSE Stream Handler                     │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Connect to /api/chat/stream               │  │   │
│  │  │ Parse event: stage, token, sources        │  │   │
│  │  │ Update UI in real-time                    │  │   │
│  │  │ Handle errors and reconnection            │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Core Components

### ChatInterface (Main Container)

```typescript
// frontend/src/components/chat/ChatInterface.tsx
'use client';

import { useState, useEffect, useRef } from 'react';
import { useChatStore } from '@/stores/chatStore';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { streamChatResponse } from '@/lib/api-client';
import type { Message, ChatRequest } from '@/types';

export function ChatInterface() {
  const {
    messages,
    currentSessionId,
    isStreaming,
    addMessage,
    updateMessage,
    startStreaming,
    stopStreaming,
  } = useChatStore();

  const [input, setInput] = useState('');
  const messageEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    addMessage(userMessage);
    setInput('');
    startStreaming();

    // Create assistant message placeholder
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      stages: [],
      sources: [],
      quality_score: undefined,
    };
    addMessage(assistantMessage);

    // Start SSE stream
    abortControllerRef.current = new AbortController();

    try {
      const request: ChatRequest = {
        message: input,
        session_id: currentSessionId,
        context_documents: [],
        llm_model: undefined, // Use defaults
        embedding_model: undefined,
      };

      await streamChatResponse(
        request,
        {
          onStage: (stage) => {
            updateMessage(assistantMessageId, {
              stages: [...(assistantMessage.stages || []), stage],
            });
          },
          onToken: (token) => {
            updateMessage(assistantMessageId, {
              content: (assistantMessage.content || '') + token,
            });
          },
          onSources: (sources) => {
            updateMessage(assistantMessageId, {
              sources,
            });
          },
          onQualityScore: (score) => {
            updateMessage(assistantMessageId, {
              quality_score: score,
            });
          },
          onFollowUps: (followUps) => {
            updateMessage(assistantMessageId, {
              follow_up_questions: followUps,
            });
          },
          onComplete: () => {
            stopStreaming();
          },
          onError: (error) => {
            console.error('Stream error:', error);
            updateMessage(assistantMessageId, {
              content: `Error: ${error.message}`,
              error: error.message,
            });
            stopStreaming();
          },
        },
        abortControllerRef.current.signal
      );
    } catch (error) {
      console.error('Failed to send message:', error);
      stopStreaming();
    }
  };

  const handleStop = () => {
    abortControllerRef.current?.abort();
    stopStreaming();
  };

  return (
    <div className="flex h-full flex-col">
      <MessageList messages={messages} />
      <div ref={messageEndRef} />
      
      <ChatInput
        value={input}
        onChange={setInput}
        onSend={handleSend}
        onStop={handleStop}
        disabled={false}
        isStreaming={isStreaming}
      />
    </div>
  );
}
```

### MessageList Component

```typescript
// frontend/src/components/chat/MessageList.tsx
import { UserMessage } from './UserMessage';
import { AssistantMessage } from './AssistantMessage';
import type { Message } from '@/types';

interface MessageListProps {
  messages: Message[];
}

export function MessageList({ messages }: MessageListProps) {
  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-neutral-500">
        <p>Start a conversation by typing a question below</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="mx-auto max-w-3xl space-y-6">
        {messages.map((message) => (
          <div key={message.id}>
            {message.role === 'user' ? (
              <UserMessage message={message} />
            ) : (
              <AssistantMessage message={message} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

### UserMessage Component

```typescript
// frontend/src/components/chat/UserMessage.tsx
import { User } from 'lucide-react';
import type { Message } from '@/types';

interface UserMessageProps {
  message: Message;
}

export function UserMessage({ message }: UserMessageProps) {
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary-500">
        <User className="h-5 w-5 text-white" />
      </div>
      
      <div className="flex-1 pt-1">
        <p className="text-sm text-neutral-900 dark:text-neutral-100">
          {message.content}
        </p>
      </div>
    </div>
  );
}
```

### AssistantMessage Component

```typescript
// frontend/src/components/chat/AssistantMessage.tsx
import { Bot } from 'lucide-react';
import { StageIndicator } from './StageIndicator';
import { SourceList } from './SourceList';
import { QualityBadge } from './QualityBadge';
import { FollowUpQuestions } from './FollowUpQuestions';
import type { Message } from '@/types';

interface AssistantMessageProps {
  message: Message;
}

export function AssistantMessage({ message }: AssistantMessageProps) {
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent-500">
        <Bot className="h-5 w-5 text-white" />
      </div>
      
      <div className="flex-1 space-y-3">
        {/* Stage indicators */}
        {message.stages && message.stages.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {message.stages.map((stage, idx) => (
              <StageIndicator key={idx} stage={stage} />
            ))}
          </div>
        )}
        
        {/* Message content */}
        <div className="prose prose-sm dark:prose-invert max-w-none">
          {message.content || (
            <span className="text-neutral-500">Thinking...</span>
          )}
        </div>
        
        {/* Quality score */}
        {message.quality_score !== undefined && (
          <QualityBadge score={message.quality_score} />
        )}
        
        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <SourceList sources={message.sources} />
        )}
        
        {/* Follow-up questions */}
        {message.follow_up_questions && message.follow_up_questions.length > 0 && (
          <FollowUpQuestions questions={message.follow_up_questions} />
        )}
      </div>
    </div>
  );
}
```

### ChatInput Component

```typescript
// frontend/src/components/chat/ChatInput.tsx
import { useState, useRef, useEffect } from 'react';
import { Send, StopCircle } from 'lucide-react';
import { Button } from '@/components/ui/Button';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  disabled: boolean;
  isStreaming: boolean;
}

export function ChatInput({
  value,
  onChange,
  onSend,
  onStop,
  disabled,
  isStreaming,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = 'auto';
    textarea.style.height = `${textarea.scrollHeight}px`;
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming && value.trim()) {
        onSend();
      }
    }
  };

  return (
    <div className="border-t border-neutral-200 bg-white px-4 py-4 dark:border-neutral-800 dark:bg-neutral-900">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question..."
            disabled={disabled}
            rows={1}
            className="flex-1 resize-none rounded-lg border border-neutral-300 bg-white px-4 py-3 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500 disabled:opacity-50 dark:border-neutral-700 dark:bg-neutral-800 dark:text-white"
            style={{ maxHeight: '200px' }}
          />
          
          {isStreaming ? (
            <Button
              onClick={onStop}
              variant="secondary"
              size="icon"
              className="h-12 w-12 shrink-0"
            >
              <StopCircle className="h-5 w-5" />
            </Button>
          ) : (
            <Button
              onClick={onSend}
              disabled={disabled || !value.trim()}
              variant="primary"
              size="icon"
              className="h-12 w-12 shrink-0"
            >
              <Send className="h-5 w-5" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
```

## Supporting Components

### StageIndicator

```typescript
// frontend/src/components/chat/StageIndicator.tsx
import { CheckCircle2, Loader2 } from 'lucide-react';

interface StageIndicatorProps {
  stage: string;
  isActive?: boolean;
}

export function StageIndicator({ stage, isActive = false }: StageIndicatorProps) {
  const displayName = stage
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="flex items-center gap-1.5 rounded-full bg-neutral-100 px-3 py-1 text-xs dark:bg-neutral-800">
      {isActive ? (
        <Loader2 className="h-3 w-3 animate-spin text-primary-500" />
      ) : (
        <CheckCircle2 className="h-3 w-3 text-success-500" />
      )}
      <span className="text-neutral-700 dark:text-neutral-300">
        {displayName}
      </span>
    </div>
  );
}
```

### QualityBadge

```typescript
// frontend/src/components/chat/QualityBadge.tsx
import { getQualityColor, getQualityLabel } from '@/lib/quality-utils';

interface QualityBadgeProps {
  score: number;
}

export function QualityBadge({ score }: QualityBadgeProps) {
  const color = getQualityColor(score);
  const label = getQualityLabel(score);

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-neutral-500">Quality:</span>
      <div
        className="rounded-full px-2 py-0.5 text-xs font-medium"
        style={{
          backgroundColor: `${color}20`,
          color: color,
        }}
      >
        {label} ({(score * 100).toFixed(0)}%)
      </div>
    </div>
  );
}
```

### SourceList

```typescript
// frontend/src/components/chat/SourceList.tsx
import { FileText, ExternalLink } from 'lucide-react';
import type { Source } from '@/types';

interface SourceListProps {
  sources: Source[];
}

export function SourceList({ sources }: SourceListProps) {
  return (
    <div className="space-y-2">
      <h4 className="text-xs font-semibold text-neutral-700 dark:text-neutral-300">
        Sources ({sources.length})
      </h4>
      
      <div className="space-y-1">
        {sources.map((source, idx) => (
          <button
            key={idx}
            className="flex w-full items-start gap-2 rounded-lg border border-neutral-200 bg-neutral-50 p-2 text-left text-xs hover:bg-neutral-100 dark:border-neutral-700 dark:bg-neutral-800 dark:hover:bg-neutral-750"
          >
            <FileText className="h-4 w-4 shrink-0 text-neutral-500" />
            
            <div className="flex-1 space-y-1">
              <div className="font-medium text-neutral-900 dark:text-neutral-100">
                {source.document_name}
              </div>
              
              {source.text && (
                <p className="line-clamp-2 text-neutral-600 dark:text-neutral-400">
                  {source.text}
                </p>
              )}
              
              <div className="flex items-center gap-2 text-neutral-500">
                <span>Chunk {source.chunk_index}</span>
                <span>•</span>
                <span>Score: {source.score.toFixed(2)}</span>
              </div>
            </div>
            
            <ExternalLink className="h-3 w-3 shrink-0 text-neutral-400" />
          </button>
        ))}
      </div>
    </div>
  );
}
```

### FollowUpQuestions

```typescript
// frontend/src/components/chat/FollowUpQuestions.tsx
import { MessageSquare } from 'lucide-react';
import { useChatStore } from '@/stores/chatStore';

interface FollowUpQuestionsProps {
  questions: string[];
}

export function FollowUpQuestions({ questions }: FollowUpQuestionsProps) {
  const { isStreaming } = useChatStore();

  const handleClick = (question: string) => {
    if (isStreaming) return;
    
    // Trigger send with this question
    // Implementation depends on how you handle sending
    const event = new CustomEvent('sendFollowUp', { detail: question });
    window.dispatchEvent(event);
  };

  return (
    <div className="space-y-2">
      <h4 className="flex items-center gap-1.5 text-xs font-semibold text-neutral-700 dark:text-neutral-300">
        <MessageSquare className="h-3.5 w-3.5" />
        Follow-up Questions
      </h4>
      
      <div className="space-y-1">
        {questions.map((question, idx) => (
          <button
            key={idx}
            onClick={() => handleClick(question)}
            disabled={isStreaming}
            className="w-full rounded-lg border border-neutral-200 bg-white px-3 py-2 text-left text-xs hover:bg-neutral-50 disabled:opacity-50 dark:border-neutral-700 dark:bg-neutral-900 dark:hover:bg-neutral-800"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}
```

## State Management

### Zustand Store

```typescript
// frontend/src/stores/chatStore.ts
import { create } from 'zustand';
import type { Message } from '@/types';

interface ChatState {
  messages: Message[];
  currentSessionId: string;
  isStreaming: boolean;
  
  addMessage: (message: Message) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  clearMessages: () => void;
  startStreaming: () => void;
  stopStreaming: () => void;
  setSessionId: (id: string) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  currentSessionId: '',
  isStreaming: false,
  
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),
  
  updateMessage: (id, updates) =>
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, ...updates } : msg
      ),
    })),
  
  clearMessages: () => set({ messages: [] }),
  
  startStreaming: () => set({ isStreaming: true }),
  
  stopStreaming: () => set({ isStreaming: false }),
  
  setSessionId: (id) => set({ currentSessionId: id }),
}));
```

## Type Definitions

```typescript
// frontend/src/types/index.ts
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  stages?: string[];
  sources?: Source[];
  quality_score?: number;
  follow_up_questions?: string[];
  error?: string;
}

export interface Source {
  chunk_id: string;
  document_id: string;
  document_name: string;
  text: string;
  chunk_index: number;
  score: number;
  metadata?: Record<string, any>;
}

export interface ChatRequest {
  message: string;
  session_id: string;
  context_documents?: string[];
  llm_model?: string;
  embedding_model?: string;
}
```

## Styling & Themes

### Tailwind Classes

```typescript
// Common styling patterns
const styles = {
  container: "flex h-full flex-col",
  messageList: "flex-1 overflow-y-auto px-4 py-6",
  messageContainer: "mx-auto max-w-3xl space-y-6",
  userBubble: "bg-primary-500 text-white rounded-lg px-4 py-2",
  assistantBubble: "bg-neutral-100 dark:bg-neutral-800 rounded-lg px-4 py-2",
  input: "rounded-lg border focus:ring-2 focus:ring-primary-500",
};
```

### Dark Mode Support

```typescript
// Conditional classes for dark mode
<div className="bg-white text-neutral-900 dark:bg-neutral-900 dark:text-neutral-100">
  {/* Content */}
</div>
```

## Testing

### Component Tests

```typescript
// frontend/src/components/chat/__tests__/ChatInterface.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ChatInterface } from '../ChatInterface';
import { useChatStore } from '@/stores/chatStore';

jest.mock('@/lib/api-client');

describe('ChatInterface', () => {
  beforeEach(() => {
    useChatStore.getState().clearMessages();
  });

  it('renders empty state', () => {
    render(<ChatInterface />);
    expect(screen.getByText(/start a conversation/i)).toBeInTheDocument();
  });

  it('sends message on Enter', async () => {
    render(<ChatInterface />);
    
    const input = screen.getByPlaceholderText(/ask a question/i);
    fireEvent.change(input, { target: { value: 'Test question' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    
    await waitFor(() => {
      expect(screen.getByText('Test question')).toBeInTheDocument();
    });
  });

  it('displays streaming response', async () => {
    const { container } = render(<ChatInterface />);
    
    // Simulate streaming
    const store = useChatStore.getState();
    store.addMessage({
      id: '1',
      role: 'assistant',
      content: 'Partial',
      timestamp: new Date().toISOString(),
    });
    
    expect(screen.getByText('Partial')).toBeInTheDocument();
  });

  it('disables input while streaming', () => {
    render(<ChatInterface />);
    
    useChatStore.getState().startStreaming();
    
    const sendButton = screen.getByRole('button');
    expect(sendButton).toBeDisabled();
  });
});
```

## Performance Optimizations

### Memoization

```typescript
import { memo, useMemo } from 'react';

export const AssistantMessage = memo(({ message }: AssistantMessageProps) => {
  const renderedContent = useMemo(
    () => parseMarkdown(message.content),
    [message.content]
  );
  
  return (
    <div>
      {renderedContent}
    </div>
  );
});
```

### Virtual Scrolling (for long conversations)

```typescript
import { useVirtualizer } from '@tanstack/react-virtual';

export function VirtualMessageList({ messages }: MessageListProps) {
  const parentRef = useRef<HTMLDivElement>(null);
  
  const virtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 200,
  });
  
  return (
    <div ref={parentRef} className="h-full overflow-y-auto">
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualItem) => (
          <div
            key={virtualItem.key}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              transform: `translateY(${virtualItem.start}px)`,
            }}
          >
            {/* Render message */}
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Accessibility

### Keyboard Navigation

```typescript
// Support keyboard shortcuts
useEffect(() => {
  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.metaKey && e.key === 'k') {
      e.preventDefault();
      inputRef.current?.focus();
    }
  };
  
  window.addEventListener('keydown', handleKeyPress);
  return () => window.removeEventListener('keydown', handleKeyPress);
}, []);
```

### ARIA Labels

```typescript
<button
  aria-label="Send message"
  aria-disabled={isStreaming}
  onClick={onSend}
>
  <Send />
</button>
```

## Troubleshooting

### Common Issues

**Issue**: Messages not streaming
```typescript
// Check SSE connection
console.log('SSE connection status:', eventSource.readyState);
// 0 = CONNECTING, 1 = OPEN, 2 = CLOSED

// Verify backend is sending events
// Check network tab for event stream
```

**Issue**: Auto-scroll not working
```typescript
// Force scroll after render
useLayoutEffect(() => {
  messageEndRef.current?.scrollIntoView({ behavior: 'auto' });
}, [messages.length]);
```

**Issue**: Input not clearing
```typescript
// Ensure state updates after send
const handleSend = async () => {
  const messageText = input;
  setInput(''); // Clear immediately
  await sendMessage(messageText);
};
```

## Related Documentation

- [SSE Streaming](03-components/frontend/sse-streaming.md)
- [API Client](03-components/frontend/api-client.md)
- [Chat API](06-api-reference/chat.md)
- [Message History](04-features/conversation-history.md)
