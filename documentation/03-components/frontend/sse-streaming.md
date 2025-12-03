# SSE Streaming Component

Server-Sent Events (SSE) implementation for real-time data streaming.

## Overview

The SSE Streaming component handles real-time communication between the frontend and backend using Server-Sent Events. It enables streaming chat responses, progress updates, and live data feeds with automatic reconnection and error handling.

**Protocol**: Server-Sent Events (SSE) over HTTP
**Use Cases**: Chat streaming, progress tracking, live updates
**Features**: Auto-reconnection, event parsing, error recovery

## Architecture

```
┌────────────────────────────────────────────────────────┐
│           SSE Streaming Architecture                    │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Client Side (EventSource)              │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ SSE Connection Manager                    │  │   │
│  │  │  ├─ Connect to /api/chat/stream          │  │   │
│  │  │  ├─ Parse incoming events                 │  │   │
│  │  │  ├─ Route to callbacks                    │  │   │
│  │  │  └─ Handle reconnection                   │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Event Types                               │  │   │
│  │  │  ├─ stage: Pipeline stage update         │  │   │
│  │  │  ├─ token: LLM token (streaming text)    │  │   │
│  │  │  ├─ sources: Retrieved source chunks     │  │   │
│  │  │  ├─ quality_score: Response quality      │  │   │
│  │  │  ├─ follow_up_questions: Suggestions     │  │   │
│  │  │  ├─ metadata: Additional info            │  │   │
│  │  │  └─ error: Error information             │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Server Side (FastAPI)                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Stream Generator                          │  │   │
│  │  │  ├─ Yield SSE formatted events           │  │   │
│  │  │  ├─ Format: "data: {json}\n\n"           │  │   │
│  │  │  ├─ Send keepalive pings                 │  │   │
│  │  │  └─ Signal completion: [DONE]            │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Connection Flow                        │   │
│  │                                                  │   │
│  │  1. Client sends POST with ChatRequest          │   │
│  │  2. Server opens SSE stream (text/event-stream) │   │
│  │  3. Server yields events as processing occurs   │   │
│  │  4. Client parses and routes to UI callbacks    │   │
│  │  5. Stream ends with [DONE] or error            │   │
│  │  6. Auto-reconnect if connection drops          │   │
│  │                                                  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Client Implementation

### SSE Stream Handler

```typescript
// frontend/src/lib/sse-client.ts
export interface SSECallbacks {
  onStage?: (stage: string) => void;
  onToken?: (token: string) => void;
  onSources?: (sources: any[]) => void;
  onQualityScore?: (score: number) => void;
  onFollowUps?: (questions: string[]) => void;
  onMetadata?: (metadata: any) => void;
  onComplete?: () => void;
  onError?: (error: Error) => void;
}

export interface SSEOptions {
  maxRetries?: number;
  retryDelay?: number;
  timeout?: number;
}

export async function streamSSE(
  url: string,
  body: any,
  callbacks: SSECallbacks,
  signal?: AbortSignal,
  options: SSEOptions = {}
): Promise<void> {
  const {
    maxRetries = 3,
    retryDelay = 1000,
    timeout = 60000,
  } = options;

  let retryCount = 0;

  while (retryCount < maxRetries) {
    try {
      await connectSSE(url, body, callbacks, signal, timeout);
      return; // Success
    } catch (error) {
      retryCount++;
      
      // Don't retry on abort or max retries reached
      if (
        signal?.aborted ||
        retryCount >= maxRetries ||
        (error instanceof Error && error.message === 'Stream aborted')
      ) {
        throw error;
      }

      // Wait before retry with exponential backoff
      await new Promise((resolve) =>
        setTimeout(resolve, retryDelay * Math.pow(2, retryCount - 1))
      );
    }
  }
}

async function connectSSE(
  url: string,
  body: any,
  callbacks: SSECallbacks,
  signal?: AbortSignal,
  timeout?: number
): Promise<void> {
  // Create timeout controller
  const timeoutController = new AbortController();
  const timeoutId = timeout
    ? setTimeout(() => timeoutController.abort(), timeout)
    : undefined;

  // Combine abort signals
  const combinedSignal = combineAbortSignals(signal, timeoutController.signal);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(body),
      signal: combinedSignal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          callbacks.onComplete?.();
          break;
        }

        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // Process complete events
        const events = buffer.split('\n\n');
        buffer = events.pop() || ''; // Keep incomplete event in buffer

        for (const event of events) {
          if (!event.trim()) continue;

          parseSSEEvent(event, callbacks);
        }
      }
    } finally {
      reader.releaseLock();
    }
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
}

function parseSSEEvent(event: string, callbacks: SSECallbacks): void {
  const lines = event.split('\n');
  let data = '';
  let eventType = 'message';

  for (const line of lines) {
    if (line.startsWith('event: ')) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      data += line.slice(6);
    }
  }

  if (!data) return;

  // Handle special [DONE] marker
  if (data === '[DONE]') {
    callbacks.onComplete?.();
    return;
  }

  try {
    const parsed = JSON.parse(data);

    // Route to appropriate callback based on content
    if (parsed.stage) {
      callbacks.onStage?.(parsed.stage);
    }

    if (parsed.token) {
      callbacks.onToken?.(parsed.token);
    }

    if (parsed.sources) {
      callbacks.onSources?.(parsed.sources);
    }

    if (parsed.quality_score !== undefined) {
      callbacks.onQualityScore?.(parsed.quality_score);
    }

    if (parsed.follow_up_questions) {
      callbacks.onFollowUps?.(parsed.follow_up_questions);
    }

    if (parsed.metadata) {
      callbacks.onMetadata?.(parsed.metadata);
    }

    if (parsed.error) {
      callbacks.onError?.(new Error(parsed.error));
    }
  } catch (parseError) {
    console.error('Failed to parse SSE data:', data, parseError);
  }
}

// Helper to combine abort signals
function combineAbortSignals(...signals: (AbortSignal | undefined)[]): AbortSignal {
  const controller = new AbortController();

  for (const signal of signals) {
    if (signal?.aborted) {
      controller.abort();
      break;
    }

    signal?.addEventListener('abort', () => controller.abort(), { once: true });
  }

  return controller.signal;
}
```

### React Hook for SSE

```typescript
// frontend/src/lib/hooks/useSSEStream.ts
import { useRef, useCallback } from 'react';
import { streamSSE, SSECallbacks } from '../sse-client';

export function useSSEStream() {
  const abortControllerRef = useRef<AbortController | null>(null);

  const startStream = useCallback(
    async (url: string, body: any, callbacks: SSECallbacks) => {
      // Cancel previous stream if exists
      abortControllerRef.current?.abort();
      
      // Create new abort controller
      abortControllerRef.current = new AbortController();

      try {
        await streamSSE(
          url,
          body,
          callbacks,
          abortControllerRef.current.signal
        );
      } catch (error) {
        if (error instanceof Error && error.name !== 'AbortError') {
          callbacks.onError?.(error);
        }
      }
    },
    []
  );

  const stopStream = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  }, []);

  return { startStream, stopStream };
}
```

### Usage in Chat Component

```typescript
// frontend/src/components/chat/ChatInterface.tsx
import { useSSEStream } from '@/lib/hooks/useSSEStream';

export function ChatInterface() {
  const { startStream, stopStream } = useSSEStream();
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');

  const handleSend = async (input: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);

    // Create placeholder for assistant response
    const assistantId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      stages: [],
      sources: [],
    };

    setMessages((prev) => [...prev, assistantMessage]);

    await startStream(
      '/api/chat/stream',
      {
        message: input,
        session_id: 'current-session',
      },
      {
        onStage: (stage) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, stages: [...(msg.stages || []), stage] }
                : msg
            )
          );
        },
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: msg.content + token }
                : msg
            )
          );
        },
        onSources: (sources) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId ? { ...msg, sources } : msg
            )
          );
        },
        onQualityScore: (score) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId ? { ...msg, quality_score: score } : msg
            )
          );
        },
        onError: (error) => {
          console.error('Stream error:', error);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: `Error: ${error.message}` }
                : msg
            )
          );
        },
      }
    );
  };

  return (
    <div>
      {/* Chat UI */}
      <button onClick={() => stopStream()}>Stop</button>
    </div>
  );
}
```

## Server Implementation

### FastAPI SSE Endpoint

```python
# api/routers/chat.py
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from api.models import ChatRequest
import json
import asyncio

router = APIRouter()

async def generate_sse_response(request: ChatRequest):
    """
    Generate Server-Sent Events for streaming chat response.
    
    Yields events in format:
        data: {json}\n\n
    """
    try:
        # Stage 1: Query analysis
        yield format_sse_event({
            "stage": "query_analysis"
        })
        
        # Stage 2: Retrieval
        yield format_sse_event({
            "stage": "retrieval"
        })
        
        # Simulate retrieval results
        await asyncio.sleep(0.5)
        
        yield format_sse_event({
            "sources": [
                {
                    "chunk_id": "c1",
                    "document_name": "doc.pdf",
                    "text": "Sample text",
                    "score": 0.95
                }
            ]
        })
        
        # Stage 3: Generation with streaming tokens
        yield format_sse_event({
            "stage": "generation"
        })
        
        # Stream tokens
        tokens = ["Hello", " ", "world", "!"]
        for token in tokens:
            yield format_sse_event({
                "token": token
            })
            await asyncio.sleep(0.05)
        
        # Quality score
        yield format_sse_event({
            "quality_score": 0.85
        })
        
        # Follow-up questions
        yield format_sse_event({
            "follow_up_questions": [
                "What is the main topic?",
                "Can you explain more?"
            ]
        })
        
        # Completion marker
        yield format_sse_event("[DONE]")
        
    except Exception as e:
        yield format_sse_event({
            "error": str(e)
        })

def format_sse_event(data) -> str:
    """
    Format data as SSE event.
    
    Args:
        data: Dictionary or string to send
    
    Returns:
        Formatted SSE string
    """
    if isinstance(data, str):
        json_data = data
    else:
        json_data = json.dumps(data, ensure_ascii=False)
    
    return f"data: {json_data}\n\n"

@router.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    Stream chat response using Server-Sent Events.
    
    Returns:
        StreamingResponse with text/event-stream content type
    """
    return StreamingResponse(
        generate_sse_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
```

### LangGraph Integration

```python
# rag/graph_rag.py
async def stream_graph_execution(request: ChatRequest):
    """
    Stream LangGraph pipeline execution with stage updates.
    """
    state = {
        "query": request.message,
        "session_id": request.session_id,
    }
    
    # Stream each node execution
    async for event in graph.astream(state):
        # Extract node name
        node_name = list(event.keys())[0]
        node_data = event[node_name]
        
        # Yield stage update
        yield format_sse_event({
            "stage": node_name
        })
        
        # Yield specific data based on node
        if node_name == "retrieval" and "sources" in node_data:
            yield format_sse_event({
                "sources": node_data["sources"]
            })
        
        elif node_name == "generation" and "content" in node_data:
            # Stream tokens if available
            if isinstance(node_data["content"], str):
                yield format_sse_event({
                    "token": node_data["content"]
                })
    
    yield format_sse_event("[DONE]")
```

## Event Format

### Standard Event Structure

```typescript
// Event types
type SSEEvent =
  | { stage: string }
  | { token: string }
  | { sources: Source[] }
  | { quality_score: number }
  | { follow_up_questions: string[] }
  | { metadata: any }
  | { error: string };

// Wire format
// data: {"stage":"retrieval"}\n\n
// data: {"token":"Hello"}\n\n
// data: {"sources":[...]}\n\n
// data: [DONE]\n\n
```

## Error Handling

### Connection Errors

```typescript
// Handle different error types
try {
  await streamSSE(url, body, callbacks, signal);
} catch (error) {
  if (error instanceof Error) {
    if (error.name === 'AbortError') {
      // User cancelled, not an error
      return;
    }
    
    if (error.message.includes('timeout')) {
      // Show timeout error
      showError('Request timed out. Please try again.');
    } else if (error.message.includes('network')) {
      // Network error
      showError('Network error. Check your connection.');
    } else {
      // Generic error
      showError('An error occurred. Please try again.');
    }
  }
}
```

### Reconnection Logic

```typescript
// Exponential backoff reconnection
let retryCount = 0;
const maxRetries = 5;
const baseDelay = 1000;

async function connectWithRetry() {
  try {
    await streamSSE(url, body, callbacks);
    retryCount = 0; // Reset on success
  } catch (error) {
    if (retryCount < maxRetries) {
      const delay = baseDelay * Math.pow(2, retryCount);
      retryCount++;
      
      console.log(`Reconnecting in ${delay}ms (attempt ${retryCount}/${maxRetries})`);
      
      setTimeout(connectWithRetry, delay);
    } else {
      callbacks.onError?.(new Error('Max reconnection attempts reached'));
    }
  }
}
```

## Testing

### Mock SSE Stream

```typescript
// frontend/src/lib/__tests__/sse-client.test.ts
import { streamSSE } from '../sse-client';

global.fetch = jest.fn();

describe('SSE Client', () => {
  it('parses SSE events', async () => {
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          new TextEncoder().encode('data: {"token":"Hello"}\n\n')
        );
        controller.enqueue(
          new TextEncoder().encode('data: [DONE]\n\n')
        );
        controller.close();
      },
    });

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: mockStream,
    });

    const onToken = jest.fn();
    const onComplete = jest.fn();

    await streamSSE('/test', {}, { onToken, onComplete });

    expect(onToken).toHaveBeenCalledWith('Hello');
    expect(onComplete).toHaveBeenCalled();
  });
});
```

## Performance Optimization

### Buffering Strategy

```typescript
// Buffer tokens for smoother rendering
class TokenBuffer {
  private buffer: string[] = [];
  private flushInterval: number = 50; // ms
  private timerId?: NodeJS.Timeout;

  constructor(private onFlush: (text: string) => void) {}

  add(token: string) {
    this.buffer.push(token);
    
    if (!this.timerId) {
      this.timerId = setTimeout(() => this.flush(), this.flushInterval);
    }
  }

  flush() {
    if (this.buffer.length > 0) {
      this.onFlush(this.buffer.join(''));
      this.buffer = [];
    }
    this.timerId = undefined;
  }
}
```

## Troubleshooting

### Common Issues

**Issue**: Events not received
```typescript
// Check response headers
const response = await fetch(url, options);
console.log('Content-Type:', response.headers.get('content-type'));
// Should be: text/event-stream

// Check nginx configuration (if using)
// X-Accel-Buffering: no must be set
```

**Issue**: Duplicate events
```typescript
// Ensure proper event parsing
// Check for \n\n delimiter between events
// Verify buffer handling doesn't split events incorrectly
```

**Issue**: Connection drops
```typescript
// Implement keepalive on server
async def keepalive_generator():
    while True:
        yield ": keepalive\n\n"
        await asyncio.sleep(15)

// Handle reconnection on client
options.maxRetries = 5;
options.retryDelay = 1000;
```

## Related Documentation

- [Chat Interface](03-components/frontend/chat-interface.md)
- [API Client](03-components/frontend/api-client.md)
- [Chat API](06-api-reference/chat.md)
- [RAG Pipeline](03-components/backend/graph-rag.md)
