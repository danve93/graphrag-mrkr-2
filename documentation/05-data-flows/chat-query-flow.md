# Chat Query Flow

End-to-end trace of a user query through the RAG pipeline.

## Overview

This document traces a complete chat query execution from frontend submission through retrieval, reasoning, generation, and streaming response. It highlights state transformations, component interactions, and data flow between system layers.

## Flow Diagram

```
User Query: "What are the backup procedures for VxRail?"
│
├─> 1. Frontend: ChatInterface
│   ├─ User types message in ChatInput
│   ├─ Press Enter triggers handleSubmit()
│   ├─ useChatStore.addMessage() (optimistic UI)
│   └─ streamChatResponse() starts SSE connection
│
├─> 2. API: POST /api/chat
│   ├─ Parse ChatRequest (with tuning params)
│   ├─ Validate session_id, message, parameters
│   ├─ Retrieve chat history for context
│   └─ Initialize RAG state dict
│
├─> 3. LangGraph: RAG Pipeline
│   │
│   ├─> 3a. query_analysis node
│   │   ├─ Normalize query text
│   │   ├─ Extract context_documents filter
│   │   ├─ Extract hashtag filters
│   │   └─ Update state["processed_query"]
│   │
│   ├─> 3b. retrieval node (hybrid_retrieval)
│   │   ├─ Check retrieval cache (60s TTL)
│   │   ├─ CACHE MISS → Execute retrieval
│   │   │
│   │   ├─> Vector Search
│   │   │   ├─ Generate query embedding
│   │   │   ├─ Check embedding cache
│   │   │   ├─ Neo4j: db.index.vector.queryNodes()
│   │   │   └─ Return top 10 candidates
│   │   │
│   │   ├─> Graph Expansion
│   │   │   ├─ For each candidate chunk:
│   │   │   │   ├─ Traverse SIMILAR_TO (strength >= 0.7)
│   │   │   │   ├─ Traverse MENTIONS relationships
│   │   │   │   └─ Traverse entity RELATED_TO edges
│   │   │   ├─ Deduplicate expanded chunks
│   │   │   └─ Limit to max_expanded_chunks (50)
│   │   │
│   │   ├─> Hybrid Scoring
│   │   │   ├─ chunk_score = vector_score * 0.7 + graph_score * 0.3
│   │   │   ├─ entity_score = Σ(entity_importance * path_strength)
│   │   │   ├─ quality_boost = chunk.quality_score * 0.1
│   │   │   └─ final_score = chunk_score + entity_score + quality_boost
│   │   │
│   │   └─> Reranking (FlashRank)
│   │       ├─ Select top 30 candidates
│   │       ├─ Batch rerank with query
│   │       ├─ rerank_score from FlashRank model
│   │       ├─ blended_score = rerank * 0.5 + hybrid * 0.5
│   │       └─ Return top 10 reranked chunks
│   │
│   ├─> 3c. graph_reasoning node (optional)
│   │   ├─ Extract entities from query (LLM)
│   │   ├─ Match entities in graph (fuzzy search)
│   │   ├─ Discover multi-hop paths (max 2 hops)
│   │   ├─ Score paths by edge strength product
│   │   ├─ Retrieve chunks connected to path entities
│   │   └─ Merge with retrieval results
│   │
│   ├─> 3d. generation node
│   │   ├─ Build context from top chunks
│   │   ├─ Format chat history
│   │   ├─ Construct generation prompt
│   │   ├─ LLM streaming generation
│   │   └─ Collect source citations
│   │
│   └─> 3e. post_generation nodes
│       ├─ quality_scoring: Assess response quality
│       └─ follow_up_generation: Suggest next questions
│
├─> 4. SSE Streaming: Response Events
│   ├─ {"type": "stage", "stage": "query_analysis", "message": "..."}
│   ├─ {"type": "stage", "stage": "retrieval", "message": "..."}
│   ├─ {"type": "stage", "stage": "graph_reasoning", "message": "..."}
│   ├─ {"type": "stage", "stage": "generation", "message": "..."}
│   ├─ {"type": "token", "token": "VxRail"}
│   ├─ {"type": "token", "token": " backup"}
│   ├─ {"type": "token", "token": " procedures"}
│   ├─ ... (streaming tokens)
│   ├─ {"type": "sources", "sources": [{chunk_id, document_name, ...}]}
│   ├─ {"type": "quality_score", "score": 0.87}
│   └─ {"type": "follow_ups", "questions": ["...", "..."]}
│
├─> 5. Frontend: Response Reconstruction
│   ├─ useSSEStream() hook receives events
│   ├─ onToken: Accumulate tokens into message.content
│   ├─ onSources: Set message.sources
│   ├─ onQualityScore: Set message.qualityScore
│   ├─ onFollowUps: Set message.followUpQuestions
│   └─ useChatStore.updateMessage() triggers re-render
│
└─> 6. UI Update: Display Response
    ├─ AssistantMessage component renders
    ├─ Markdown formatting applied
    ├─ SourceCitations displayed with document links
    ├─ QualityBadge shows score with color
    └─ FollowUpQuestions rendered as clickable chips
```

## Step-by-Step Trace

### Step 1: Frontend Submission

**Location**: `frontend/src/components/chat/ChatInterface.tsx`

```typescript
// User types "What are the backup procedures for VxRail?"
// and presses Enter
const handleSubmit = async () => {
  const message = inputValue.trim();
  
  // Optimistic UI update
  const userMessageId = crypto.randomUUID();
  addMessage({
    id: userMessageId,
    role: 'user',
    content: message,
    timestamp: new Date().toISOString(),
  });
  
  // Start streaming
  const assistantMessageId = crypto.randomUUID();
  
  for await (const event of streamChatResponse(
    message,
    sessionId,
    selectedDocuments,
  )) {
    // Process SSE events...
  }
};
```

**State Snapshot**:
```typescript
{
  messages: [
    {
      id: "msg-user-001",
      role: "user",
      content: "What are the backup procedures for VxRail?",
      timestamp: "2025-12-01T10:30:00Z"
    }
  ],
  isStreaming: true
}
```

### Step 2: API Request

**Location**: `api/routers/chat.py`

```python
@router.post("/chat")
async def chat(request: ChatRequest):
    # Merge tuning parameters
    config = merge_tuning_params(request)
    
    # Retrieve chat history
    history_service = get_chat_history_service()
    history = await history_service.get_history(request.session_id)
    
    # Initialize RAG state
    state = {
        "query": request.message,
        "session_id": request.session_id,
        "context_documents": request.context_documents,
        "chat_history": history,
        "stages": [],
        **config,
    }
    
    # Stream RAG pipeline execution
    async for event in run_rag_pipeline(state):
        yield format_sse_event(event)
```

**State Snapshot**:
```python
{
    "query": "What are the backup procedures for VxRail?",
    "session_id": "session-abc-123",
    "context_documents": [],
    "chat_history": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"}
    ],
    "temperature": 0.7,
    "top_k": 10,
    "hybrid_chunk_weight": 0.7,
    "stages": []
}
```

### Step 3a: Query Analysis

**Location**: `rag/nodes/query_analysis.py`

```python
def query_analysis(state: dict) -> dict:
    query = state["query"]
    
    # Normalize
    processed = query.strip().lower()
    
    # Extract filters
    doc_filter = state.get("context_documents", [])
    
    # Extract hashtags
    hashtags = re.findall(r'#(\w+)', query)
    
    state["processed_query"] = processed
    state["document_filter"] = doc_filter
    state["hashtags"] = hashtags
    state["stages"].append("query_analysis")
    
    return state
```

**State Update**:
```python
{
    "processed_query": "what are the backup procedures for vxrail?",
    "document_filter": [],
    "hashtags": [],
    "stages": ["query_analysis"]
}
```

### Step 3b: Hybrid Retrieval

**Location**: `rag/retriever.py`

```python
async def hybrid_retrieval(query: str, top_k: int, config: dict) -> List[Chunk]:
    # Check cache
    cache_key = f"{query}:{top_k}:{config}"
    if cached := retrieval_cache.get(cache_key):
        return cached
    
    # Vector search
    chunks = await vector_search(query, top_k * 3)
    # Returns 30 candidates with similarity scores
    
    # Graph expansion
    expanded = await expand_via_graph(chunks, config)
    # Returns 45 total chunks (30 original + 15 expanded)
    
    # Hybrid scoring
    scored_chunks = calculate_hybrid_scores(expanded, config)
    
    # Reranking
    if config["flashrank_enabled"]:
        reranked = await apply_reranking(scored_chunks[:30], query, config)
        final_chunks = reranked[:top_k]
    else:
        final_chunks = scored_chunks[:top_k]
    
    # Cache result
    retrieval_cache[cache_key] = final_chunks
    
    return final_chunks
```

**Retrieved Chunks**:
```python
[
    {
        "chunk_id": "chunk-001",
        "document_id": "doc-vxrail-admin",
        "document_name": "VxRail Administration Guide.pdf",
        "content": "VxRail backup procedures involve...",
        "vector_score": 0.92,
        "graph_score": 0.85,
        "hybrid_score": 0.89,
        "rerank_score": 0.94,
        "final_score": 0.915,  # (0.94 * 0.5 + 0.89 * 0.5)
        "page_number": 47,
        "entities": ["VxRail", "Backup", "Data Protection"]
    },
    {
        "chunk_id": "chunk-002",
        "document_id": "doc-vxrail-admin",
        "content": "To configure backup policies...",
        "final_score": 0.88,
        "page_number": 48
    },
    # ... 8 more chunks
]
```

### Step 3c: Graph Reasoning

**Location**: `rag/nodes/graph_reasoning.py`

```python
async def graph_reasoning(state: dict) -> dict:
    if not state.get("enable_entity_reasoning"):
        return state
    
    query = state["processed_query"]
    
    # Extract entities from query
    query_entities = await extract_entities_from_query(query)
    # ["VxRail", "backup", "procedures"]
    
    # Match in graph
    matched = await match_entities_in_graph(query_entities)
    # [Entity(name="VxRail", type="Component"), Entity(name="Backup", type="Procedure")]
    
    # Discover paths
    paths = await discover_entity_paths(matched, max_hops=2)
    # Path 1: VxRail -[RELATED_TO:0.9]-> Data Protection -[RELATED_TO:0.8]-> Backup
    # Path 2: VxRail -[RELATED_TO:0.85]-> Backup
    
    # Retrieve chunks for path entities
    entity_chunks = await retrieve_entity_chunks(paths)
    
    # Merge with retrieval results
    state["entity_paths"] = paths
    state["entity_chunks"] = entity_chunks
    state["stages"].append("graph_reasoning")
    
    return state
```

**Entity Paths**:
```python
[
    {
        "entities": ["VxRail", "Backup"],
        "path_strength": 0.85,
        "relationships": [
            {"type": "RELATED_TO", "strength": 0.85}
        ]
    },
    {
        "entities": ["VxRail", "Data Protection", "Backup"],
        "path_strength": 0.72,  # 0.9 * 0.8
        "relationships": [
            {"type": "RELATED_TO", "strength": 0.9},
            {"type": "RELATED_TO", "strength": 0.8}
        ]
    }
]
```

### Step 3d: Generation

**Location**: `rag/nodes/generation.py`

```python
async def generation(state: dict) -> dict:
    chunks = state["retrieved_chunks"]
    query = state["query"]
    history = state.get("chat_history", [])
    
    # Build context
    context = "\n\n".join([
        f"[Document: {c['document_name']}, Page: {c['page_number']}]\n{c['content']}"
        for c in chunks
    ])
    
    # Format history
    history_text = format_history_for_llm(history)
    
    # Build prompt
    prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Previous conversation:
{history_text}

User question: {query}

Provide a clear, accurate answer citing the source documents."""
    
    # Stream LLM generation
    tokens = []
    async for token in llm_manager.stream_generate(
        prompt,
        temperature=state["temperature"],
        max_tokens=state["max_tokens"],
    ):
        tokens.append(token)
        
        # Emit SSE event
        yield {"type": "token", "token": token}
    
    # Collect response
    response = "".join(tokens)
    
    # Extract sources
    sources = [
        {
            "chunk_id": c["chunk_id"],
            "document_name": c["document_name"],
            "page_number": c["page_number"],
            "relevance_score": c["final_score"],
        }
        for c in chunks[:5]  # Top 5 sources
    ]
    
    state["response"] = response
    state["sources"] = sources
    state["stages"].append("generation")
    
    yield {"type": "sources", "sources": sources}
    
    return state
```

**Generated Response**:
```
VxRail backup procedures involve several key steps:

1. **Configure Backup Policies**: Navigate to Data Protection settings
   and define backup schedules, retention policies, and storage targets.

2. **Enable Data Protection Suite**: Activate the integrated backup
   solution for VxRail infrastructure components.

3. **Verify Backup Jobs**: Monitor backup task execution in the
   administration console to ensure successful completion.

For detailed configuration steps, refer to the VxRail Administration
Guide pages 47-52.
```

### Step 4: SSE Streaming

**Location**: `api/routers/chat.py`

```python
def format_sse_event(event: dict) -> str:
    """Format event as SSE."""
    return f"data: {json.dumps(event)}\n\n"

# Stream events
async for event in run_rag_pipeline(state):
    if event["type"] == "stage":
        yield format_sse_event(event)
    elif event["type"] == "token":
        yield format_sse_event(event)
    elif event["type"] == "sources":
        yield format_sse_event(event)
    # ... other event types
```

**SSE Event Sequence**:
```
data: {"type":"stage","stage":"query_analysis","message":"Analyzing query"}

data: {"type":"stage","stage":"retrieval","message":"Searching documents"}

data: {"type":"stage","stage":"graph_reasoning","message":"Reasoning over entities"}

data: {"type":"stage","stage":"generation","message":"Generating response"}

data: {"type":"token","token":"VxRail"}

data: {"type":"token","token":" backup"}

data: {"type":"token","token":" procedures"}

... (more tokens)

data: {"type":"sources","sources":[{"chunk_id":"chunk-001",...}]}

data: {"type":"quality_score","score":0.87}

data: {"type":"follow_ups","questions":["What retention policies...","How to restore..."]}
```

### Step 5: Frontend Processing

**Location**: `frontend/src/lib/api-client.ts`

```typescript
export async function* streamChatResponse(
  message: string,
  sessionId: string,
) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  
  let buffer = '';
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const event = JSON.parse(line.slice(6));
        yield event;
      }
    }
  }
}
```

**Location**: `frontend/src/components/chat/ChatInterface.tsx`

```typescript
// Process events
for await (const event of streamChatResponse(message, sessionId)) {
  if (event.type === 'token') {
    updateMessage(assistantMessageId, {
      content: (prev) => prev + event.token,
    });
  } else if (event.type === 'sources') {
    updateMessage(assistantMessageId, {
      sources: event.sources,
    });
  } else if (event.type === 'quality_score') {
    updateMessage(assistantMessageId, {
      qualityScore: event.score,
    });
  } else if (event.type === 'follow_ups') {
    updateMessage(assistantMessageId, {
      followUpQuestions: event.questions,
    });
  }
}
```

### Step 6: UI Display

**Location**: `frontend/src/components/chat/AssistantMessage.tsx`

```typescript
<div className="assistant-message">
  {/* Response content with markdown */}
  <ReactMarkdown>{message.content}</ReactMarkdown>
  
  {/* Source citations */}
  {message.sources && (
    <SourceCitations sources={message.sources} />
  )}
  
  {/* Quality badge */}
  {message.qualityScore && (
    <QualityBadge score={message.qualityScore} />
  )}
  
  {/* Follow-up questions */}
  {message.followUpQuestions && (
    <FollowUpQuestions questions={message.followUpQuestions} />
  )}
</div>
```

## Performance Notes

### Caching Layers

1. **Embedding Cache**: Query embedding cached by text+model hash (LRU, no expiration)
2. **Entity Label Cache**: Entity name lookups cached (TTL: 300s, hit rate: 70-80%)
3. **Retrieval Cache**: Full retrieval results cached (TTL: 60s, hit rate: 20-30%)

### Batching Opportunities

- **Embedding Generation**: Batch multiple chunks for parallel embedding
- **FlashRank Reranking**: Batch candidates (up to 30) in single API call
- **Neo4j Queries**: Use UNWIND for batch entity lookups

### Streaming Benefits

- **Time to First Token**: ~500ms (vs 5-10s for full generation)
- **Perceived Latency**: Immediate feedback with stage indicators
- **User Experience**: Progressive rendering allows early reading

## Related Documentation

- [RAG Pipeline](03-components/backend/rag-pipeline.md)
- [Hybrid Retrieval](04-features/hybrid-retrieval.md)
- [Chat Interface](03-components/frontend/chat-interface.md)
- [SSE Streaming](03-components/frontend/sse-streaming.md)
