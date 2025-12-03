# Conversation History Feature

Persistent chat session management with context continuity.

## Overview

Conversation History enables multi-turn conversations by storing message history, maintaining session context, and allowing users to resume previous conversations. It integrates with the RAG pipeline to provide context-aware responses that reference earlier exchanges.

**Key Capabilities**:
- Session-based conversation tracking
- Message persistence and retrieval
- Context window management
- Session restoration and continuation
- History search and filtering

## Architecture

```
┌────────────────────────────────────────────────────────┐
│       Conversation History Architecture                 │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Storage Layer                          │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ In-Memory Store (Development)             │  │   │
│  │  │   ├─ Dict[session_id → List[Message]]     │  │   │
│  │  │   ├─ Fast access                          │  │   │
│  │  │   └─ Lost on restart                      │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Redis Store (Production)                  │  │   │
│  │  │   ├─ Persistent storage                   │  │   │
│  │  │   ├─ TTL-based expiration                 │  │   │
│  │  │   └─ Distributed access                   │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Service Layer                          │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ ChatHistoryService                        │  │   │
│  │  │   ├─ add_message(session, msg)            │  │   │
│  │  │   ├─ get_history(session, limit)          │  │   │
│  │  │   ├─ clear_session(session)               │  │   │
│  │  │   └─ list_sessions(user)                  │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Context Management                     │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Recent messages (last N turns)            │  │   │
│  │  │ Token budget enforcement                  │  │   │
│  │  │ Message summarization (optional)          │  │   │
│  │  │ Context window sliding                    │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          API Endpoints                          │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ GET  /api/history/{session_id}            │  │   │
│  │  │ POST /api/history/{session_id}/message    │  │   │
│  │  │ DELETE /api/history/{session_id}          │  │   │
│  │  │ GET  /api/history/sessions                │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Core Implementation

### History Service

```python
# api/services/chat_history.py
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
from api.models import Message

class ChatHistoryService:
    """Base interface for chat history storage."""
    
    async def add_message(
        self,
        session_id: str,
        message: Message,
    ) -> None:
        """Add message to session history."""
        raise NotImplementedError
    
    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Retrieve session history."""
        raise NotImplementedError
    
    async def clear_session(self, session_id: str) -> None:
        """Clear all messages for session."""
        raise NotImplementedError
    
    async def list_sessions(
        self,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """List all sessions."""
        raise NotImplementedError
```

### In-Memory Implementation

```python
# api/services/chat_history.py (continued)
from collections import defaultdict
from threading import Lock

class InMemoryChatHistory(ChatHistoryService):
    """In-memory chat history storage."""
    
    def __init__(self):
        self._storage: Dict[str, List[Message]] = defaultdict(list)
        self._lock = Lock()
    
    async def add_message(
        self,
        session_id: str,
        message: Message,
    ) -> None:
        """Add message to session."""
        with self._lock:
            self._storage[session_id].append(message)
    
    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get session history."""
        with self._lock:
            messages = self._storage.get(session_id, [])
            
            if limit:
                return messages[-limit:]
            
            return messages.copy()
    
    async def clear_session(self, session_id: str) -> None:
        """Clear session history."""
        with self._lock:
            if session_id in self._storage:
                del self._storage[session_id]
    
    async def list_sessions(
        self,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """List all sessions."""
        with self._lock:
            sessions = []
            for session_id, messages in self._storage.items():
                if not messages:
                    continue
                
                sessions.append({
                    "session_id": session_id,
                    "message_count": len(messages),
                    "last_updated": messages[-1].timestamp if messages else None,
                    "first_message": messages[0].content[:50] if messages else None,
                })
            
            return sorted(
                sessions,
                key=lambda x: x["last_updated"] or "",
                reverse=True,
            )
```

### Redis Implementation

```python
# api/services/chat_history.py (continued)
import redis.asyncio as redis
from config.settings import settings

class RedisChatHistory(ChatHistoryService):
    """Redis-backed chat history storage."""
    
    def __init__(self):
        self.redis = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        self.ttl = settings.history_ttl or 86400 * 7  # 7 days default
    
    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"chat_history:{session_id}"
    
    async def add_message(
        self,
        session_id: str,
        message: Message,
    ) -> None:
        """Add message to session."""
        key = self._key(session_id)
        
        # Serialize message
        data = message.model_dump_json()
        
        # Append to list
        await self.redis.rpush(key, data)
        
        # Set TTL
        await self.redis.expire(key, self.ttl)
    
    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get session history."""
        key = self._key(session_id)
        
        if limit:
            # Get last N messages
            messages = await self.redis.lrange(key, -limit, -1)
        else:
            # Get all messages
            messages = await self.redis.lrange(key, 0, -1)
        
        # Deserialize
        return [Message.model_validate_json(msg) for msg in messages]
    
    async def clear_session(self, session_id: str) -> None:
        """Clear session history."""
        key = self._key(session_id)
        await self.redis.delete(key)
    
    async def list_sessions(
        self,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """List all sessions."""
        pattern = "chat_history:*"
        sessions = []
        
        async for key in self.redis.scan_iter(match=pattern):
            session_id = key.split(":", 1)[1]
            
            # Get message count
            count = await self.redis.llen(key)
            
            # Get last message
            last_msg = await self.redis.lrange(key, -1, -1)
            
            if last_msg:
                msg = Message.model_validate_json(last_msg[0])
                
                sessions.append({
                    "session_id": session_id,
                    "message_count": count,
                    "last_updated": msg.timestamp,
                    "first_message": msg.content[:50],
                })
        
        return sorted(
            sessions,
            key=lambda x: x["last_updated"],
            reverse=True,
        )
```

## Context Management

### Context Window

```python
# api/services/chat_history.py (continued)
from core.token_manager import count_tokens

def get_context_messages(
    messages: List[Message],
    max_tokens: int = 4000,
) -> List[Message]:
    """
    Get recent messages within token budget.
    
    Args:
        messages: Full message history
        max_tokens: Maximum context tokens
    
    Returns:
        Recent messages within budget
    """
    context = []
    total_tokens = 0
    
    # Process in reverse (most recent first)
    for message in reversed(messages):
        msg_tokens = count_tokens(message.content)
        
        if total_tokens + msg_tokens > max_tokens:
            break
        
        context.insert(0, message)
        total_tokens += msg_tokens
    
    return context
```

### Message Formatting

```python
def format_history_for_llm(messages: List[Message]) -> str:
    """
    Format conversation history for LLM prompt.
    
    Args:
        messages: Message history
    
    Returns:
        Formatted conversation string
    """
    formatted = []
    
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    
    return "\n\n".join(formatted)
```

## API Integration

### History Router

```python
# api/routers/history.py
from fastapi import APIRouter, HTTPException
from typing import Optional
from api.services.chat_history import get_history_service
from api.models import Message

router = APIRouter(prefix="/api/history", tags=["history"])

@router.get("/{session_id}")
async def get_session_history(
    session_id: str,
    limit: Optional[int] = None,
):
    """
    Retrieve conversation history for session.
    
    Args:
        session_id: Session identifier
        limit: Max messages to return (most recent)
    
    Returns:
        List of messages
    """
    service = get_history_service()
    messages = await service.get_history(session_id, limit=limit)
    
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": messages,
    }

@router.delete("/{session_id}")
async def clear_session_history(session_id: str):
    """Clear all messages for session."""
    service = get_history_service()
    await service.clear_session(session_id)
    
    return {"status": "cleared", "session_id": session_id}

@router.get("/sessions")
async def list_sessions(user_id: Optional[str] = None):
    """List all conversation sessions."""
    service = get_history_service()
    sessions = await service.list_sessions(user_id=user_id)
    
    return {
        "session_count": len(sessions),
        "sessions": sessions,
    }
```

### Chat Integration

```python
# api/routers/chat.py (extension)
from api.services.chat_history import get_history_service

@router.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with history tracking."""
    service = get_history_service()
    
    # Get conversation history
    history = await service.get_history(request.session_id, limit=10)
    
    # Add context to request
    context_messages = get_context_messages(history, max_tokens=4000)
    
    # Store user message
    user_message = Message(
        role="user",
        content=request.message,
        timestamp=datetime.utcnow().isoformat(),
    )
    await service.add_message(request.session_id, user_message)
    
    # Generate response with context
    response = await generate_response(
        query=request.message,
        history=context_messages,
    )
    
    # Store assistant message
    assistant_message = Message(
        role="assistant",
        content=response.content,
        timestamp=datetime.utcnow().isoformat(),
        sources=response.sources,
        quality_score=response.quality_score,
    )
    await service.add_message(request.session_id, assistant_message)
    
    return response
```

## Frontend Integration

### History Store

```typescript
// frontend/src/stores/historyStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface Session {
  session_id: string;
  message_count: number;
  last_updated: string;
  first_message: string;
}

interface HistoryState {
  sessions: Session[];
  currentSessionId: string | null;
  
  loadSessions: () => Promise<void>;
  createSession: () => string;
  switchSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => Promise<void>;
}

export const useHistoryStore = create<HistoryState>()(
  persist(
    (set, get) => ({
      sessions: [],
      currentSessionId: null,
      
      loadSessions: async () => {
        const response = await fetch('/api/history/sessions');
        const data = await response.json();
        set({ sessions: data.sessions });
      },
      
      createSession: () => {
        const sessionId = `session_${Date.now()}`;
        set({ currentSessionId: sessionId });
        return sessionId;
      },
      
      switchSession: (sessionId: string) => {
        set({ currentSessionId: sessionId });
      },
      
      deleteSession: async (sessionId: string) => {
        await fetch(`/api/history/${sessionId}`, { method: 'DELETE' });
        
        set((state) => ({
          sessions: state.sessions.filter((s) => s.session_id !== sessionId),
          currentSessionId:
            state.currentSessionId === sessionId
              ? null
              : state.currentSessionId,
        }));
      },
    }),
    {
      name: 'chat-history',
      partialize: (state) => ({ currentSessionId: state.currentSessionId }),
    }
  )
);
```

### History Panel

```typescript
// frontend/src/components/history/HistoryPanel.tsx
import { useEffect } from 'react';
import { useHistoryStore } from '@/stores/historyStore';
import { MessageSquare, Trash2 } from 'lucide-react';

export function HistoryPanel() {
  const {
    sessions,
    currentSessionId,
    loadSessions,
    switchSession,
    deleteSession,
  } = useHistoryStore();

  useEffect(() => {
    loadSessions();
  }, []);

  return (
    <div className="w-64 border-r border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
      <h2 className="mb-4 font-semibold">Chat History</h2>
      
      <div className="space-y-2">
        {sessions.map((session) => (
          <button
            key={session.session_id}
            onClick={() => switchSession(session.session_id)}
            className={`group flex w-full items-start gap-2 rounded-lg p-2 text-left hover:bg-neutral-100 dark:hover:bg-neutral-800 ${
              currentSessionId === session.session_id
                ? 'bg-primary-50 dark:bg-primary-900'
                : ''
            }`}
          >
            <MessageSquare className="h-4 w-4 shrink-0 text-neutral-500" />
            
            <div className="flex-1 overflow-hidden">
              <p className="truncate text-sm">{session.first_message}</p>
              <p className="text-xs text-neutral-500">
                {session.message_count} messages
              </p>
            </div>
            
            <button
              onClick={(e) => {
                e.stopPropagation();
                deleteSession(session.session_id);
              }}
              className="opacity-0 group-hover:opacity-100"
            >
              <Trash2 className="h-3.5 w-3.5 text-error-500" />
            </button>
          </button>
        ))}
      </div>
    </div>
  );
}
```

## Configuration

### Settings

```python
# config/settings.py
class Settings(BaseSettings):
    # History storage
    history_backend: str = "memory"  # "memory" or "redis"
    redis_url: str = "redis://localhost:6379"
    history_ttl: int = 86400 * 7  # 7 days
    
    # Context management
    max_history_messages: int = 10
    max_context_tokens: int = 4000
```

## Testing

### Unit Tests

```python
# tests/test_chat_history.py
import pytest
from api.services.chat_history import InMemoryChatHistory
from api.models import Message

@pytest.mark.asyncio
async def test_add_and_retrieve_messages():
    service = InMemoryChatHistory()
    session_id = "test_session"
    
    # Add messages
    msg1 = Message(role="user", content="Hello", timestamp="2024-01-01T00:00:00")
    msg2 = Message(role="assistant", content="Hi", timestamp="2024-01-01T00:00:01")
    
    await service.add_message(session_id, msg1)
    await service.add_message(session_id, msg2)
    
    # Retrieve
    history = await service.get_history(session_id)
    
    assert len(history) == 2
    assert history[0].content == "Hello"
    assert history[1].content == "Hi"

@pytest.mark.asyncio
async def test_history_limit():
    service = InMemoryChatHistory()
    session_id = "test_session"
    
    # Add 5 messages
    for i in range(5):
        msg = Message(role="user", content=f"Msg {i}", timestamp=f"2024-01-01T00:00:0{i}")
        await service.add_message(session_id, msg)
    
    # Get last 3
    history = await service.get_history(session_id, limit=3)
    
    assert len(history) == 3
    assert history[0].content == "Msg 2"
    assert history[-1].content == "Msg 4"

@pytest.mark.asyncio
async def test_clear_session():
    service = InMemoryChatHistory()
    session_id = "test_session"
    
    msg = Message(role="user", content="Test", timestamp="2024-01-01T00:00:00")
    await service.add_message(session_id, msg)
    
    await service.clear_session(session_id)
    
    history = await service.get_history(session_id)
    assert len(history) == 0
```

## Troubleshooting

### Common Issues

**Issue**: History not persisting
```python
# Check storage backend
print(f"History backend: {settings.history_backend}")

# For Redis, verify connection
import redis
r = redis.from_url(settings.redis_url)
print(r.ping())  # Should return True
```

**Issue**: Context too large
```python
# Reduce context window
max_context_tokens = 2000

# Or limit message count
max_history_messages = 5
```

**Issue**: Session ID conflicts
```python
# Use UUID for session IDs
import uuid

def create_session_id() -> str:
    return f"session_{uuid.uuid4().hex}"
```

## Related Documentation

- [Chat Interface](03-components/frontend/chat-interface.md)
- [Chat API](06-api-reference/chat.md)
- [Redis Configuration](08-operations/redis-setup.md)
