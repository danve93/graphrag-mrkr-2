# Streaming SSE Flow

Server-Sent Events token streaming from backend to frontend.

## Overview

This document traces how streaming responses flow from LLM generation through FastAPI SSE endpoints to frontend React components. It shows event formatting, connection management, error handling, and progressive UI updates.

## Flow Diagram

```
LLM Generation starts → Tokens streamed to frontend
│
├─> 1. Backend: LLM Streaming
│   │
│   ├─> LLMManager.stream_generate()
│   │   ├─ OpenAI client: stream=True
│   │   ├─ Async generator yields tokens
│   │   └─ Example: ["VxRail", " backup", " procedures", ...]
│   │
│   └─> RAG Pipeline (LangGraph)
│       ├─ generation node calls stream_generate()
│       ├─ Each token yielded as event
│       └─ Additional events: sources, quality_score, follow_ups
│
├─> 2. FastAPI: SSE Response Formatting
│   │
│   ├─> StreamingResponse
│   │   ├─ media_type: "text/event-stream"
│   │   ├─ headers: Cache-Control, Connection
│   │   └─ Async generator function
│   │
│   ├─> Event Types
│   │   ├─ {"type": "stage", "stage": "...", "message": "..."}
│   │   ├─ {"type": "token", "token": "..."}
│   │   ├─ {"type": "sources", "sources": [...]}
│   │   ├─ {"type": "quality_score", "score": 0.87}
│   │   ├─ {"type": "follow_ups", "questions": [...]}
│   │   └─ {"type": "error", "message": "..."}
│   │
│   └─> SSE Format
│       ├─ Prefix each event: "data: {json}\n\n"
│       ├─ Keep-alive: Empty comment every 15s
│       └─ Error handling: try/catch with error event
│
├─> 3. Network: HTTP Connection
│   │
│   ├─> Request
│   │   ├─ POST /api/chat
│   │   ├─ Content-Type: application/json
│   │   ├─ Accept: text/event-stream
│   │   └─ Body: {message, session_id, ...}
│   │
│   └─> Response Stream
│       ├─ Status: 200 OK
│       ├─ Content-Type: text/event-stream
│       ├─ Transfer-Encoding: chunked
│       └─ Body: Stream of SSE events
│
├─> 4. Frontend: SSE Client
│   │
│   ├─> streamSSE() Function
│   │   ├─ Fetch API with readable stream
│   │   ├─ TextDecoder for UTF-8
│   │   ├─ Line buffering (split by \n)
│   │   └─ Parse "data: " prefix
│   │
│   ├─> Event Parsing
│   │   ├─ Extract JSON from data field
│   │   ├─ Route by event.type
│   │   └─ Call appropriate callback
│   │
│   └─> Callbacks
│       ├─ onToken(token) → Append to message
│       ├─ onSources(sources) → Update message.sources
│       ├─ onQualityScore(score) → Update message.qualityScore
│       ├─ onFollowUps(questions) → Update message.followUpQuestions
│       └─ onError(error) → Display error state
│
├─> 5. React: State Updates
│   │
│   ├─> useChatStore (Zustand)
│   │   ├─ Initial: addMessage({id, role: "assistant", content: ""})
│   │   ├─ Token events: updateMessage(id, {content: prev + token})
│   │   ├─ Sources event: updateMessage(id, {sources})
│   │   └─ Completion: setIsStreaming(false)
│   │
│   └─> Component Re-renders
│       ├─ MessageList observes messages array
│       ├─ AssistantMessage re-renders on content change
│       └─ Progressive markdown rendering
│
└─> 6. UI: Progressive Display
    ├─ Stage indicators show pipeline progress
    ├─ Tokens accumulate into readable text
    ├─ Cursor blinks at end of message (streaming indicator)
    ├─ Sources appear when event received
    ├─ Quality badge appears when score received
    └─ Follow-up questions appear when available
```

## Step-by-Step Trace

### Step 1: Backend LLM Streaming

**Location**: `core/llm.py`

```python
class LLMManager:
    """LLM interaction manager with streaming support."""
    
    async def stream_generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM generation token-by-token.
        
        Args:
            prompt: Generation prompt
            temperature: Sampling temperature
            max_tokens: Max output tokens
        
        Yields:
            Generated tokens
        """
        if self.provider == "openai":
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,  # Enable streaming
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield token
        
        elif self.provider == "ollama":
            # Ollama streaming implementation
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
            )
            
            async for chunk in response:
                yield chunk["response"]
```

### Step 2: RAG Pipeline Integration

**Location**: `rag/nodes/generation.py`

```python
async def generation(state: dict) -> AsyncGenerator[dict, None]:
    """
    Generation node with streaming.
    
    Args:
        state: RAG state
    
    Yields:
        Event dicts for SSE
    """
    llm_manager = get_llm_manager()
    
    # Build prompt from context
    prompt = build_generation_prompt(
        query=state["query"],
        chunks=state["retrieved_chunks"],
        history=state.get("chat_history", []),
    )
    
    # Emit stage start
    yield {"type": "stage", "stage": "generation", "message": "Generating response"}
    
    # Stream tokens
    tokens = []
    async for token in llm_manager.stream_generate(
        prompt=prompt,
        temperature=state["temperature"],
        max_tokens=state["max_tokens"],
    ):
        tokens.append(token)
        
        # Emit token event
        yield {"type": "token", "token": token}
    
    # Collect full response
    response = "".join(tokens)
    state["response"] = response
    
    # Emit sources
    sources = [
        {
            "chunk_id": c["chunk_id"],
            "document_name": c["document_name"],
            "page_number": c["page_number"],
            "relevance_score": c["final_score"],
        }
        for c in state["retrieved_chunks"][:5]
    ]
    
    yield {"type": "sources", "sources": sources}
    
    return state
```

### Step 3: FastAPI SSE Endpoint

**Location**: `api/routers/chat.py`

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Streaming chat endpoint.
    
    Args:
        request: Chat request
    
    Returns:
        StreamingResponse with SSE events
    """
    
    async def event_generator():
        """Generate SSE events from RAG pipeline."""
        try:
            # Initialize RAG state
            state = {
                "query": request.message,
                "session_id": request.session_id,
                "temperature": request.temperature or 0.7,
                # ... other parameters
            }
            
            # Run RAG pipeline (returns async generator)
            async for event in run_rag_pipeline(state):
                # Format as SSE
                sse_event = format_sse_event(event)
                yield sse_event
        
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            
            # Send error event
            error_event = format_sse_event({
                "type": "error",
                "message": str(e),
            })
            yield error_event
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )

def format_sse_event(event: dict) -> str:
    """
    Format event as SSE.
    
    Args:
        event: Event dict
    
    Returns:
        SSE-formatted string
    """
    return f"data: {json.dumps(event)}\n\n"
```

### Step 4: Frontend SSE Client

**Location**: `frontend/src/lib/api-client.ts`

```typescript
export async function* streamSSE(
  url: string,
  body: any,
  callbacks: {
    onStage?: (stage: string, message: string) => void;
    onToken?: (token: string) => void;
    onSources?: (sources: Source[]) => void;
    onQualityScore?: (score: number) => void;
    onFollowUps?: (questions: string[]) => void;
    onError?: (error: string) => void;
  },
): AsyncGenerator<SSEEvent, void, unknown> {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      // Decode chunk
      buffer += decoder.decode(value, { stream: true });

      // Split by newlines
      const lines = buffer.split('\n');
      
      // Keep incomplete line in buffer
      buffer = lines.pop() || '';

      // Process complete lines
      for (const line of lines) {
        if (!line.trim()) continue;

        // Parse SSE event
        if (line.startsWith('data: ')) {
          const eventData = line.slice(6);
          
          try {
            const event = JSON.parse(eventData);
            
            // Route to callbacks
            switch (event.type) {
              case 'stage':
                callbacks.onStage?.(event.stage, event.message);
                break;
              
              case 'token':
                callbacks.onToken?.(event.token);
                break;
              
              case 'sources':
                callbacks.onSources?.(event.sources);
                break;
              
              case 'quality_score':
                callbacks.onQualityScore?.(event.score);
                break;
              
              case 'follow_ups':
                callbacks.onFollowUps?.(event.questions);
                break;
              
              case 'error':
                callbacks.onError?.(event.message);
                break;
            }
            
            yield event;
          } catch (error) {
            console.error('Failed to parse SSE event:', eventData);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
```

### Step 5: React Integration

**Location**: `frontend/src/components/chat/ChatInterface.tsx`

```typescript
'use client';

import { useState } from 'react';
import { useChatStore } from '@/stores/useChatStore';
import { streamChatResponse } from '@/lib/api-client';

export function ChatInterface() {
  const { messages, addMessage, updateMessage, setIsStreaming } = useChatStore();
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = async () => {
    const message = inputValue.trim();
    if (!message) return;

    // Clear input
    setInputValue('');

    // Add user message
    const userMessageId = crypto.randomUUID();
    addMessage({
      id: userMessageId,
      role: 'user',
      content: message,
      timestamp: new Date().toISOString(),
    });

    // Create assistant message (empty initially)
    const assistantMessageId = crypto.randomUUID();
    addMessage({
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
    });

    setIsStreaming(true);

    try {
      // Stream response
      for await (const event of streamChatResponse(message, sessionId)) {
        if (event.type === 'token') {
          // Append token to message content
          updateMessage(assistantMessageId, (prev) => ({
            ...prev,
            content: prev.content + event.token,
          }));
        } else if (event.type === 'sources') {
          updateMessage(assistantMessageId, (prev) => ({
            ...prev,
            sources: event.sources,
          }));
        } else if (event.type === 'quality_score') {
          updateMessage(assistantMessageId, (prev) => ({
            ...prev,
            qualityScore: event.score,
          }));
        } else if (event.type === 'follow_ups') {
          updateMessage(assistantMessageId, (prev) => ({
            ...prev,
            followUpQuestions: event.questions,
          }));
        } else if (event.type === 'error') {
          updateMessage(assistantMessageId, (prev) => ({
            ...prev,
            error: event.message,
          }));
        }
      }
    } catch (error) {
      updateMessage(assistantMessageId, (prev) => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Unknown error',
      }));
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="chat-interface">
      <MessageList messages={messages} />
      <ChatInput
        value={inputValue}
        onChange={setInputValue}
        onSubmit={handleSubmit}
        disabled={isStreaming}
      />
    </div>
  );
}
```

### Step 6: Progressive Rendering

**Location**: `frontend/src/components/chat/AssistantMessage.tsx`

```typescript
'use client';

import ReactMarkdown from 'react-markdown';
import { Message } from '@/types';

export function AssistantMessage({ message }: { message: Message }) {
  const isStreaming = !message.sources && !message.error;

  return (
    <div className="assistant-message">
      {/* Message content with markdown */}
      <div className="prose dark:prose-invert">
        <ReactMarkdown>{message.content}</ReactMarkdown>
        
        {/* Streaming cursor */}
        {isStreaming && (
          <span className="inline-block w-2 h-4 ml-1 bg-primary-500 animate-pulse" />
        )}
      </div>

      {/* Sources */}
      {message.sources && message.sources.length > 0 && (
        <SourceCitations sources={message.sources} />
      )}

      {/* Quality badge */}
      {message.qualityScore !== undefined && (
        <QualityBadge score={message.qualityScore} />
      )}

      {/* Follow-up questions */}
      {message.followUpQuestions && message.followUpQuestions.length > 0 && (
        <FollowUpQuestions questions={message.followUpQuestions} />
      )}

      {/* Error */}
      {message.error && (
        <div className="mt-2 rounded bg-error-50 p-3 text-sm text-error-700">
          {message.error}
        </div>
      )}
    </div>
  );
}
```

## Event Sequence Example

**Complete SSE Stream**:
```
data: {"type":"stage","stage":"query_analysis","message":"Analyzing query"}

data: {"type":"stage","stage":"retrieval","message":"Searching documents"}

data: {"type":"stage","stage":"graph_reasoning","message":"Reasoning over entities"}

data: {"type":"stage","stage":"generation","message":"Generating response"}

data: {"type":"token","token":"VxRail"}

data: {"type":"token","token":" backup"}

data: {"type":"token","token":" procedures"}

data: {"type":"token","token":" involve"}

data: {"type":"token","token":" several"}

data: {"type":"token","token":" key"}

data: {"type":"token","token":" steps"}

data: {"type":"token","token":":\n\n"}

data: {"type":"token","token":"1"}

data: {"type":"token","token":"."}

data: {"type":"token","token":" **"}

data: {"type":"token","token":"Configure"}

... (more tokens)

data: {"type":"sources","sources":[{"chunk_id":"chunk-047","document_name":"VxRail_Admin.pdf","page_number":47,"relevance_score":0.778},{"chunk_id":"chunk-048","document_name":"VxRail_Admin.pdf","page_number":48,"relevance_score":0.756}]}

data: {"type":"quality_score","score":0.87}

data: {"type":"follow_ups","questions":["What retention policies are recommended?","How to restore from backup?","Can backups be automated?"]}
```

## Error Handling

### Connection Errors

```typescript
// Frontend: Retry logic
async function streamWithRetry(
  url: string,
  body: any,
  maxRetries: number = 3,
) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      for await (const event of streamSSE(url, body, callbacks)) {
        yield event;
      }
      break; // Success
    } catch (error) {
      if (attempt === maxRetries - 1) {
        throw error; // Final attempt failed
      }
      
      // Exponential backoff
      await new Promise((resolve) =>
        setTimeout(resolve, Math.pow(2, attempt) * 1000)
      );
    }
  }
}
```

### Backend Errors

```python
# Backend: Error event on exception
async def event_generator():
    try:
        async for event in run_rag_pipeline(state):
            yield format_sse_event(event)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        
        yield format_sse_event({
            "type": "error",
            "message": "An error occurred during generation",
        })
```

## Performance Notes

### Latency Breakdown

**Time to First Token (TTFT)**:
```
- Backend initialization: 50ms
- Retrieval: 300ms
- Graph reasoning: 150ms
- LLM first token: 200-500ms
---
Total TTFT: ~700-1000ms
```

**Token Streaming**:
```
- LLM generation: 20-50 tokens/second
- Network latency: <10ms per token
- Frontend rendering: <5ms per token
---
User sees tokens in real-time
```

### Optimization Strategies

- **Parallel Retrieval**: Start retrieval while user types
- **Chunk Buffering**: Send tokens in small batches (5-10) to reduce overhead
- **Connection Reuse**: Keep-alive for multiple requests
- **Error Recovery**: Graceful degradation with error events

## Related Documentation

- [Chat Interface](03-components/frontend/chat-interface.md)
- [SSE Streaming](03-components/frontend/sse-streaming.md)
- [Chat API](06-api-reference/chat.md)
- [Chat Query Flow](05-data-flows/chat-query-flow.md)
