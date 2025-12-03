# RAG Pipeline Component

LangGraph-based state machine implementing graph-enhanced retrieval-augmented generation.

## Overview

The RAG pipeline orchestrates the complete query-to-answer workflow using LangGraph's StateGraph. It combines vector search, graph reasoning, and LLM generation into a modular, observable pipeline.

**Location**: `rag/graph_rag.py`
**Framework**: LangGraph
**State Management**: Plain Python dict (TypedDict)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   LangGraph StateGraph                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  query_analysis → retrieval → graph_reasoning           │
│                                   ↓                      │
│                            should_rerank?                │
│                         ┌───────┴───────┐               │
│                         ↓               ↓               │
│                    reranking        generation           │
│                         │               ↑               │
│                         └───────────────┘               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## State Definition

**File**: `rag/graph_rag.py`

```python
from typing import TypedDict, List, Dict, Optional

class RAGState(TypedDict):
    # Input
    query: str
    session_id: str
    context_documents: List[str]
    
    # RAG parameters
    llm_model: Optional[str]
    embedding_model: Optional[str]
    temperature: Optional[float]
    top_k: Optional[int]
    max_expansion_depth: Optional[int]
    max_expanded_chunks: Optional[int]
    expansion_similarity_threshold: Optional[float]
    hybrid_chunk_weight: Optional[float]
    hybrid_entity_weight: Optional[float]
    retrieval_mode: Optional[str]
    
    # Pipeline data
    query_embedding: Optional[List[float]]
    retrieved_chunks: List[Dict]
    expanded_chunks: List[Dict]
    reranked_chunks: List[Dict]
    final_chunks: List[Dict]
    
    # Output
    answer: str
    sources: List[Dict]
    quality_score: Optional[float]
    follow_up_questions: List[str]
    
    # Metadata
    stages: List[str]
    stage_timings: Dict[str, float]
    error: Optional[str]
```

## Graph Construction

```python
from langgraph.graph import StateGraph, END

# Create graph
graph = StateGraph(RAGState)

# Add nodes
graph.add_node("query_analysis", query_analysis_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("graph_reasoning", graph_reasoning_node)
graph.add_node("reranking", reranking_node)
graph.add_node("generation", generation_node)

# Set entry point
graph.set_entry_point("query_analysis")

# Add edges
graph.add_edge("query_analysis", "retrieval")
graph.add_edge("retrieval", "graph_reasoning")

# Conditional edge for reranking
graph.add_conditional_edges(
    "graph_reasoning",
    should_rerank,
    {
        True: "reranking",
        False: "generation"
    }
)
graph.add_edge("reranking", "generation")
graph.add_edge("generation", END)

# Compile
rag_chain = graph.compile()
```

## Pipeline Nodes

### Query Analysis Node

**File**: `rag/nodes/query_analysis.py`

**Purpose**: Normalize query and extract filters

```python
import re
from typing import Dict

async def query_analysis_node(state: RAGState) -> RAGState:
    """
    Analyze and normalize user query.
    
    Operations:
    - Extract context document filters
    - Parse hashtag filters
    - Normalize whitespace
    - Generate query embedding
    """
    query = state["query"]
    
    # Extract hashtags
    hashtags = re.findall(r'#(\w+)', query)
    clean_query = re.sub(r'#\w+', '', query).strip()
    
    # Extract document mentions (e.g., "in document X")
    doc_pattern = r'in (?:document|doc) ["\']?([^"\']+)["\']?'
    doc_matches = re.findall(doc_pattern, query, re.IGNORECASE)
    if doc_matches:
        state["context_documents"].extend(doc_matches)
        clean_query = re.sub(doc_pattern, '', clean_query, flags=re.IGNORECASE).strip()
    
    # Normalize whitespace
    clean_query = ' '.join(clean_query.split())
    
    # Generate embedding
    from core.embeddings import embedding_manager
    embedding = await embedding_manager.get_embedding(clean_query)
    
    # Update state
    state["query"] = clean_query
    state["query_embedding"] = embedding
    state["stages"].append("query_analysis")
    
    return state
```

### Retrieval Node

**File**: `rag/nodes/retrieval.py`

**Purpose**: Hybrid vector + entity search

```python
from rag.retriever import hybrid_retrieval

async def retrieval_node(state: RAGState) -> RAGState:
    """
    Retrieve initial candidate chunks via hybrid search.
    
    Combines:
    - Vector similarity search
    - Entity name matching
    """
    query = state["query"]
    top_k = state.get("top_k", 10)
    context_documents = state.get("context_documents", [])
    
    # Hybrid retrieval
    chunks = await hybrid_retrieval(
        query=query,
        query_embedding=state["query_embedding"],
        top_k=top_k,
        context_documents=context_documents,
        chunk_weight=state.get("hybrid_chunk_weight", 0.7),
        entity_weight=state.get("hybrid_entity_weight", 0.3),
        retrieval_mode=state.get("retrieval_mode", "hybrid")
    )
    
    # Update state
    state["retrieved_chunks"] = chunks
    state["stages"].append("retrieval")
    
    return state
```

### Graph Reasoning Node

**File**: `rag/nodes/graph_reasoning.py`

**Purpose**: Multi-hop graph expansion

```python
from core.graph_db import get_db

async def graph_reasoning_node(state: RAGState) -> RAGState:
    """
    Expand context via graph traversal.
    
    Strategies:
    - Chunk similarity expansion
    - Entity relationship expansion
    """
    max_depth = state.get("max_expansion_depth", 2)
    max_chunks = state.get("max_expanded_chunks", 30)
    threshold = state.get("expansion_similarity_threshold", 0.7)
    
    if max_depth == 0:
        # No expansion
        state["expanded_chunks"] = state["retrieved_chunks"]
        state["stages"].append("graph_reasoning")
        return state
    
    db = get_db()
    seed_ids = [c["chunk_id"] for c in state["retrieved_chunks"]]
    
    # Chunk similarity expansion
    similarity_results = await db.expand_via_similarity(
        seed_ids=seed_ids,
        max_depth=max_depth,
        threshold=threshold,
        limit=max_chunks
    )
    
    # Entity relationship expansion
    entity_results = await db.expand_via_entities(
        seed_ids=seed_ids,
        max_depth=max_depth,
        threshold=threshold,
        limit=max_chunks
    )
    
    # Merge and deduplicate
    all_chunks = _merge_results(
        state["retrieved_chunks"],
        similarity_results,
        entity_results
    )
    
    # Rank by combined score
    ranked = _rank_chunks(all_chunks, max_chunks)
    
    state["expanded_chunks"] = ranked
    state["stages"].append("graph_reasoning")
    
    return state

def _merge_results(seed, similarity, entity):
    """Merge and deduplicate chunk results."""
    chunks_by_id = {}
    
    # Add seed chunks
    for chunk in seed:
        chunks_by_id[chunk["chunk_id"]] = chunk
    
    # Add similarity results
    for chunk in similarity:
        if chunk["chunk_id"] not in chunks_by_id:
            chunks_by_id[chunk["chunk_id"]] = chunk
        else:
            # Combine scores
            existing = chunks_by_id[chunk["chunk_id"]]
            existing["score"] = max(existing["score"], chunk["score"])
    
    # Add entity results
    for chunk in entity:
        if chunk["chunk_id"] not in chunks_by_id:
            chunks_by_id[chunk["chunk_id"]] = chunk
        else:
            existing = chunks_by_id[chunk["chunk_id"]]
            existing["score"] = max(existing["score"], chunk["score"])
    
    return list(chunks_by_id.values())

def _rank_chunks(chunks, limit):
    """Sort chunks by score and limit."""
    sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
    return sorted_chunks[:limit]
```

### Reranking Node

**File**: `rag/nodes/reranking.py`

**Purpose**: Cross-encoder reranking

```python
from rag.rerankers.flashrank_reranker import FlashRankReranker

async def reranking_node(state: RAGState) -> RAGState:
    """
    Rerank chunks using FlashRank cross-encoder.
    
    Blends rerank scores with hybrid scores.
    """
    reranker = FlashRankReranker()
    
    max_candidates = state.get("flashrank_max_candidates", 50)
    blend_weight = state.get("flashrank_blend_weight", 0.5)
    
    # Take top candidates
    candidates = state["expanded_chunks"][:max_candidates]
    
    # Rerank
    reranked = await reranker.rerank(
        query=state["query"],
        chunks=candidates
    )
    
    # Blend scores
    for chunk in reranked:
        hybrid_score = chunk.get("score", 0)
        rerank_score = chunk.get("rerank_score", 0)
        chunk["final_score"] = (
            blend_weight * rerank_score +
            (1 - blend_weight) * hybrid_score
        )
    
    # Re-sort by final score
    reranked = sorted(reranked, key=lambda c: c["final_score"], reverse=True)
    
    state["reranked_chunks"] = reranked
    state["final_chunks"] = reranked
    state["stages"].append("reranking")
    
    return state
```

### Generation Node

**File**: `rag/nodes/generation.py`

**Purpose**: LLM response generation

```python
from core.llm import get_llm

async def generation_node(state: RAGState) -> RAGState:
    """
    Generate natural language response with LLM.
    
    Streams tokens and tracks sources.
    """
    # Select chunks for context
    final_chunks = state.get("final_chunks") or state.get("expanded_chunks") or state["retrieved_chunks"]
    context_chunks = final_chunks[:20]  # Limit context size
    
    # Build context
    context = "\n\n".join([
        f"[{i+1}] {chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])
    
    # Build prompt
    prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context}

Question: {state['query']}

Instructions:
- Provide a clear, accurate answer
- Cite sources using [1], [2], etc.
- If information is not in context, say so
- Be concise but complete

Answer:"""
    
    # Get LLM
    llm = get_llm(
        model=state.get("llm_model"),
        temperature=state.get("temperature", 0.1)
    )
    
    # Generate with streaming
    answer_tokens = []
    async for token in llm.astream(prompt):
        answer_tokens.append(token)
        # Stream token to client (handled by API layer)
    
    answer = "".join(answer_tokens)
    
    # Extract cited sources
    import re
    cited_indices = set(int(m) - 1 for m in re.findall(r'\[(\d+)\]', answer))
    cited_sources = [context_chunks[i] for i in cited_indices if i < len(context_chunks)]
    
    # Update state
    state["answer"] = answer
    state["sources"] = cited_sources
    state["stages"].append("generation")
    
    return state
```

### Conditional Edge Function

```python
def should_rerank(state: RAGState) -> bool:
    """Determine if reranking should be applied."""
    from config.settings import settings
    
    # Check if FlashRank is enabled
    if not settings.flashrank_enabled:
        return False
    
    # Check if we have enough candidates
    min_candidates = 10
    if len(state.get("expanded_chunks", [])) < min_candidates:
        return False
    
    return True
```

## Pipeline Execution

### Synchronous Execution

```python
from rag.graph_rag import rag_chain

state = {
    "query": "What is the backup procedure?",
    "session_id": "test-session",
    "context_documents": [],
    "stages": [],
    "stage_timings": {}
}

result = rag_chain.invoke(state)

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
print(f"Stages: {result['stages']}")
```

### Streaming Execution

```python
async def execute_rag_stream(query: str):
    state = {
        "query": query,
        "session_id": str(uuid.uuid4()),
        "context_documents": [],
        "stages": [],
        "stage_timings": {}
    }
    
    async for chunk in rag_chain.astream(state):
        # Chunk contains state updates
        if "stages" in chunk and len(chunk["stages"]) > len(state["stages"]):
            # New stage started
            new_stage = chunk["stages"][-1]
            yield {"type": "stage", "stage": new_stage}
        
        if "answer" in chunk and chunk["answer"] != state.get("answer", ""):
            # New tokens generated
            new_content = chunk["answer"][len(state.get("answer", "")):]
            yield {"type": "token", "content": new_content}
        
        state.update(chunk)
    
    # Final outputs
    yield {"type": "sources", "sources": state["sources"]}
```

## Background Tasks

### Quality Scoring

**Execution**: After generation, background task

```python
import asyncio

async def score_response_background(state: RAGState):
    """Score response quality asynchronously."""
    from core.quality_scorer import score_response
    
    quality_score = await score_response(
        query=state["query"],
        answer=state["answer"],
        sources=state["sources"]
    )
    
    state["quality_score"] = quality_score
    return state

# In generation node
asyncio.create_task(score_response_background(state))
```

### Follow-up Generation

```python
async def generate_follow_ups_background(state: RAGState):
    """Generate follow-up questions asynchronously."""
    from api.services.follow_up_generator import generate_follow_ups
    
    follow_ups = await generate_follow_ups(
        query=state["query"],
        answer=state["answer"]
    )
    
    state["follow_up_questions"] = follow_ups
    return state

# In generation node
asyncio.create_task(generate_follow_ups_background(state))
```

## Configuration

### Runtime Parameters

Parameters can be overridden per request:

```python
state = {
    "query": "backup procedure",
    "llm_model": "gpt-4o",              # Override default
    "temperature": 0.3,                  # Override default
    "top_k": 15,                         # Override default
    "max_expansion_depth": 3,            # Override default
    "flashrank_enabled": True,           # Override default
    # ...
}
```

### Default Configuration

**File**: `config/rag_tuning_config.json`

```json
{
  "llm_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "temperature": 0.1,
  "top_k": 10,
  "max_expansion_depth": 2,
  "max_expanded_chunks": 30,
  "expansion_similarity_threshold": 0.7,
  "hybrid_chunk_weight": 0.7,
  "hybrid_entity_weight": 0.3,
  "flashrank_enabled": false,
  "flashrank_max_candidates": 50,
  "flashrank_blend_weight": 0.5
}
```

## Error Handling

### Node-Level Error Handling

```python
async def retrieval_node(state: RAGState) -> RAGState:
    try:
        chunks = await hybrid_retrieval(...)
        state["retrieved_chunks"] = chunks
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state["error"] = f"Retrieval error: {str(e)}"
        state["retrieved_chunks"] = []
    
    state["stages"].append("retrieval")
    return state
```

### Pipeline-Level Error Handling

```python
try:
    result = rag_chain.invoke(state)
except Exception as e:
    logger.error(f"RAG pipeline error: {e}", exc_info=True)
    return {
        "answer": "I apologize, but I encountered an error processing your request.",
        "error": str(e),
        "sources": []
    }
```

## Testing

### Unit Tests

```python
def test_query_analysis_node():
    state = {
        "query": "What is HA in document VMware Guide? #vmware",
        "context_documents": [],
        "stages": []
    }
    
    result = asyncio.run(query_analysis_node(state))
    
    assert "VMware Guide" in result["context_documents"]
    assert "#vmware" not in result["query"]
    assert "query_embedding" in result
    assert "query_analysis" in result["stages"]

def test_should_rerank():
    state = {"expanded_chunks": [{"id": i} for i in range(20)]}
    
    with patch("config.settings.settings.flashrank_enabled", True):
        assert should_rerank(state) == True
    
    with patch("config.settings.settings.flashrank_enabled", False):
        assert should_rerank(state) == False
```

### Integration Tests

```bash
pytest tests/integration/test_rag_pipeline.py -v
```

## Performance

### Latency Breakdown

**Typical Query** (depth 2, no reranking):
```
Query Analysis:     10-20ms
Retrieval:          100-200ms
Graph Reasoning:    200-500ms
Generation:         2-3 seconds (streaming)
─────────────────────────────
Total (first token): 310-720ms
Total (complete):    2.5-4 seconds
```

### Optimization

**Parallel Execution**: Node operations run sequentially but internal operations can be parallelized:
```python
# In retrieval node
vector_task = asyncio.create_task(vector_search(...))
entity_task = asyncio.create_task(entity_search(...))
vector_results, entity_results = await asyncio.gather(vector_task, entity_task)
```

## Related Documentation

- [Graph RAG Pipeline Concepts](02-core-concepts/graph-rag-pipeline.md)
- [Retriever Component](03-components/backend/retriever.md)
- [Reranking Feature](04-features/reranking.md)
- [Chat Query Flow](05-data-flows/chat-query-flow.md)
