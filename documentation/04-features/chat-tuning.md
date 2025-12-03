# Chat Tuning Feature

Runtime RAG parameter tuning and prompt optimization.

## Overview

Chat Tuning provides a UI-driven interface for adjusting RAG pipeline parameters, selecting LLM/embedding models, and optimizing prompts at request time without restarting the backend. It enables experimentation with retrieval strategies, generation settings, and model selection to achieve optimal response quality for specific use cases.

**Tunable Parameters**:
- LLM model selection (OpenAI, Ollama)
- Embedding model selection
- Retrieval parameters (top-k, hybrid weights, expansion depth)
- Reranking configuration (FlashRank blend weight, max candidates)
- Generation settings (temperature, max tokens, system prompt)

**Key Features**:
- Live parameter updates without restart
- Per-request model selection
- Preset configurations for common scenarios
- Parameter validation and constraints
- Reset to defaults

## Architecture

```
┌────────────────────────────────────────────────────────┐
│           Chat Tuning Architecture                      │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Frontend Tuning Flow                   │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ 1. User opens Tuning Panel               │  │   │
│  │  │ 2. Displays current settings + defaults   │  │   │
│  │  │ 3. User adjusts sliders/dropdowns         │  │   │
│  │  │ 4. Validates parameter constraints        │  │   │
│  │  │ 5. Updates TuningStore (Zustand)          │  │   │
│  │  │ 6. ChatRequest includes tuning params     │  │   │
│  │  │ 7. Backend overrides defaults             │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Backend Parameter Override             │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ ChatRequest Model                         │  │   │
│  │  │   ├─ llm_model: Optional[str]             │  │   │
│  │  │   ├─ embedding_model: Optional[str]       │  │   │
│  │  │   ├─ temperature: Optional[float]         │  │   │
│  │  │   ├─ max_tokens: Optional[int]            │  │   │
│  │  │   ├─ retrieval_top_k: Optional[int]       │  │   │
│  │  │   ├─ hybrid_chunk_weight: Optional[float] │  │   │
│  │  │   ├─ hybrid_entity_weight: Optional[float]│  │   │
│  │  │   ├─ expansion_depth: Optional[int]       │  │   │
│  │  │   └─ flashrank_blend_weight: Optional[f]  │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │    ↓                                              │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ RAG Pipeline                              │  │   │
│  │  │   ├─ Use request params if provided       │  │   │
│  │  │   ├─ Fall back to settings defaults       │  │   │
│  │  │   └─ Execute with effective configuration │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Preset Management                      │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Presets:                                  │  │   │
│  │  │   • "Balanced" (default)                  │  │   │
│  │  │   • "Precise" (low temp, high rerank)     │  │   │
│  │  │   • "Creative" (high temp, low rerank)    │  │   │
│  │  │   • "Fast" (low top-k, no expansion)      │  │   │
│  │  │   • "Comprehensive" (high top-k, deep)    │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Backend Implementation

### ChatRequest Model

```python
# api/models.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class ChatRequest(BaseModel):
    """
    Chat request with optional tuning parameters.
    """
    # Required
    message: str = Field(..., min_length=1)
    session_id: str
    
    # Optional context
    context_documents: List[str] = Field(default_factory=list)
    
    # Model selection
    llm_model: Optional[str] = None  # e.g., "gpt-4", "ollama/llama2"
    embedding_model: Optional[str] = None  # e.g., "text-embedding-3-small"
    
    # Generation parameters
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=100, le=8000)
    system_prompt: Optional[str] = None
    
    # Retrieval parameters
    retrieval_mode: str = Field("hybrid", pattern="^(vector|hybrid|entity)$")
    retrieval_top_k: Optional[int] = Field(None, ge=1, le=100)
    
    # Hybrid scoring weights
    hybrid_chunk_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    hybrid_entity_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Graph expansion
    expansion_depth: Optional[int] = Field(None, ge=0, le=3)
    expansion_similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_expanded_chunks: Optional[int] = Field(None, ge=0, le=200)
    
    # Reranking
    flashrank_blend_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    flashrank_max_candidates: Optional[int] = Field(None, ge=5, le=100)
    
    @validator("hybrid_chunk_weight", "hybrid_entity_weight")
    def validate_weights(cls, v, values):
        """Ensure weights sum to reasonable range."""
        if v is not None:
            other_key = (
                "hybrid_entity_weight"
                if "hybrid_chunk_weight" in values
                else "hybrid_chunk_weight"
            )
            other = values.get(other_key)
            
            if other is not None and v + other > 1.0:
                raise ValueError("Weights should sum to <= 1.0")
        
        return v
```

### Parameter Merging

```python
# api/routers/chat.py
from config.settings import settings
from api.models import ChatRequest

def merge_tuning_params(request: ChatRequest) -> dict:
    """
    Merge request tuning parameters with defaults.
    
    Args:
        request: Chat request with optional overrides
    
    Returns:
        Effective configuration dict
    """
    config = {
        # Models
        "llm_model": request.llm_model or settings.llm_model,
        "embedding_model": request.embedding_model or settings.embedding_model,
        
        # Generation
        "temperature": (
            request.temperature
            if request.temperature is not None
            else settings.llm_temperature
        ),
        "max_tokens": (
            request.max_tokens
            if request.max_tokens is not None
            else settings.llm_max_tokens
        ),
        "system_prompt": request.system_prompt or settings.system_prompt,
        
        # Retrieval
        "retrieval_mode": request.retrieval_mode,
        "top_k": (
            request.retrieval_top_k
            if request.retrieval_top_k is not None
            else settings.retrieval_top_k
        ),
        
        # Hybrid weights
        "hybrid_chunk_weight": (
            request.hybrid_chunk_weight
            if request.hybrid_chunk_weight is not None
            else settings.hybrid_chunk_weight
        ),
        "hybrid_entity_weight": (
            request.hybrid_entity_weight
            if request.hybrid_entity_weight is not None
            else settings.hybrid_entity_weight
        ),
        
        # Expansion
        "expansion_depth": (
            request.expansion_depth
            if request.expansion_depth is not None
            else settings.max_expansion_depth
        ),
        "expansion_similarity_threshold": (
            request.expansion_similarity_threshold
            if request.expansion_similarity_threshold is not None
            else settings.expansion_similarity_threshold
        ),
        "max_expanded_chunks": (
            request.max_expanded_chunks
            if request.max_expanded_chunks is not None
            else settings.max_expanded_chunks
        ),
        
        # Reranking
        "flashrank_blend_weight": (
            request.flashrank_blend_weight
            if request.flashrank_blend_weight is not None
            else settings.flashrank_blend_weight
        ),
        "flashrank_max_candidates": (
            request.flashrank_max_candidates
            if request.flashrank_max_candidates is not None
            else settings.flashrank_max_candidates
        ),
    }
    
    return config
```

### Chat Endpoint Integration

```python
# api/routers/chat.py
@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with tuning parameters.
    
    Args:
        request: Chat request with optional tuning overrides
    """
    # Merge parameters
    config = merge_tuning_params(request)
    
    # Initialize RAG state with effective config
    state = {
        "query": request.message,
        "session_id": request.session_id,
        "context_documents": request.context_documents,
        **config,
    }
    
    # Execute RAG pipeline with tuned config
    async for event in run_rag_pipeline(state):
        yield event
```

## Frontend Implementation

### Tuning Store

```typescript
// frontend/src/stores/useTuningStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface TuningConfig {
  // Models
  llmModel?: string;
  embeddingModel?: string;
  
  // Generation
  temperature?: number;
  maxTokens?: number;
  systemPrompt?: string;
  
  // Retrieval
  retrievalMode: 'vector' | 'hybrid' | 'entity';
  retrievalTopK?: number;
  
  // Hybrid weights
  hybridChunkWeight?: number;
  hybridEntityWeight?: number;
  
  // Expansion
  expansionDepth?: number;
  expansionSimilarityThreshold?: number;
  maxExpandedChunks?: number;
  
  // Reranking
  flashrankBlendWeight?: number;
  flashrankMaxCandidates?: number;
}

interface TuningStore {
  config: TuningConfig;
  updateConfig: (updates: Partial<TuningConfig>) => void;
  resetConfig: () => void;
  loadPreset: (preset: string) => void;
}

const DEFAULT_CONFIG: TuningConfig = {
  retrievalMode: 'hybrid',
  temperature: 0.7,
  maxTokens: 2000,
  retrievalTopK: 10,
  hybridChunkWeight: 0.7,
  hybridEntityWeight: 0.3,
  expansionDepth: 1,
  expansionSimilarityThreshold: 0.7,
  maxExpandedChunks: 50,
  flashrankBlendWeight: 0.5,
  flashrankMaxCandidates: 30,
};

export const useTuningStore = create<TuningStore>()(
  persist(
    (set) => ({
      config: DEFAULT_CONFIG,
      
      updateConfig: (updates) =>
        set((state) => ({
          config: { ...state.config, ...updates },
        })),
      
      resetConfig: () =>
        set({ config: DEFAULT_CONFIG }),
      
      loadPreset: (preset) => {
        const presets: Record<string, Partial<TuningConfig>> = {
          balanced: DEFAULT_CONFIG,
          
          precise: {
            temperature: 0.3,
            retrievalTopK: 15,
            flashrankBlendWeight: 0.7,
            expansionDepth: 2,
          },
          
          creative: {
            temperature: 0.9,
            flashrankBlendWeight: 0.3,
            hybridChunkWeight: 0.5,
          },
          
          fast: {
            retrievalTopK: 5,
            expansionDepth: 0,
            maxExpandedChunks: 20,
          },
          
          comprehensive: {
            retrievalTopK: 20,
            expansionDepth: 2,
            maxExpandedChunks: 100,
            flashrankMaxCandidates: 50,
          },
        };
        
        const presetConfig = presets[preset] || DEFAULT_CONFIG;
        
        set({ config: { ...DEFAULT_CONFIG, ...presetConfig } });
      },
    }),
    {
      name: 'amber-tuning',
    }
  )
);
```

### Tuning Panel Component

```typescript
// frontend/src/components/chat/TuningPanel.tsx
'use client';

import { useTuningStore } from '@/stores/useTuningStore';
import { Sliders, RotateCcw } from 'lucide-react';

export function TuningPanel() {
  const { config, updateConfig, resetConfig, loadPreset } = useTuningStore();

  return (
    <div className="space-y-6 rounded-lg border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sliders className="h-5 w-5 text-neutral-600" />
          <h2 className="text-lg font-semibold">RAG Tuning</h2>
        </div>
        
        <button
          onClick={resetConfig}
          className="flex items-center gap-1 rounded px-2 py-1 text-sm text-neutral-600 hover:bg-neutral-100 dark:hover:bg-neutral-800"
        >
          <RotateCcw className="h-4 w-4" />
          Reset
        </button>
      </div>

      {/* Presets */}
      <div>
        <label className="mb-2 block text-sm font-medium">Presets</label>
        <div className="flex flex-wrap gap-2">
          {['balanced', 'precise', 'creative', 'fast', 'comprehensive'].map((preset) => (
            <button
              key={preset}
              onClick={() => loadPreset(preset)}
              className="rounded-full bg-neutral-100 px-3 py-1 text-sm capitalize hover:bg-neutral-200 dark:bg-neutral-800 dark:hover:bg-neutral-700"
            >
              {preset}
            </button>
          ))}
        </div>
      </div>

      {/* Model Selection */}
      <div>
        <label className="mb-2 block text-sm font-medium">LLM Model</label>
        <select
          value={config.llmModel || ''}
          onChange={(e) => updateConfig({ llmModel: e.target.value || undefined })}
          className="w-full rounded-lg border border-neutral-300 px-3 py-2 dark:border-neutral-700 dark:bg-neutral-800"
        >
          <option value="">Default</option>
          <option value="gpt-4">GPT-4</option>
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          <option value="ollama/llama2">Ollama: Llama 2</option>
          <option value="ollama/mistral">Ollama: Mistral</option>
        </select>
      </div>

      {/* Temperature */}
      <div>
        <label className="mb-2 flex items-center justify-between text-sm font-medium">
          <span>Temperature</span>
          <span className="text-neutral-500">{config.temperature?.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={config.temperature || 0.7}
          onChange={(e) => updateConfig({ temperature: parseFloat(e.target.value) })}
          className="w-full"
        />
        <div className="mt-1 flex justify-between text-xs text-neutral-500">
          <span>Precise</span>
          <span>Creative</span>
        </div>
      </div>

      {/* Retrieval Top-K */}
      <div>
        <label className="mb-2 flex items-center justify-between text-sm font-medium">
          <span>Retrieval Top-K</span>
          <span className="text-neutral-500">{config.retrievalTopK}</span>
        </label>
        <input
          type="range"
          min="1"
          max="50"
          step="1"
          value={config.retrievalTopK || 10}
          onChange={(e) => updateConfig({ retrievalTopK: parseInt(e.target.value) })}
          className="w-full"
        />
      </div>

      {/* Hybrid Chunk Weight */}
      <div>
        <label className="mb-2 flex items-center justify-between text-sm font-medium">
          <span>Chunk Weight</span>
          <span className="text-neutral-500">{config.hybridChunkWeight?.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={config.hybridChunkWeight || 0.7}
          onChange={(e) => updateConfig({ hybridChunkWeight: parseFloat(e.target.value) })}
          className="w-full"
        />
        <div className="mt-1 flex justify-between text-xs text-neutral-500">
          <span>Entity Focus</span>
          <span>Chunk Focus</span>
        </div>
      </div>

      {/* Expansion Depth */}
      <div>
        <label className="mb-2 flex items-center justify-between text-sm font-medium">
          <span>Expansion Depth</span>
          <span className="text-neutral-500">{config.expansionDepth}</span>
        </label>
        <input
          type="range"
          min="0"
          max="3"
          step="1"
          value={config.expansionDepth || 1}
          onChange={(e) => updateConfig({ expansionDepth: parseInt(e.target.value) })}
          className="w-full"
        />
      </div>

      {/* FlashRank Blend Weight */}
      <div>
        <label className="mb-2 flex items-center justify-between text-sm font-medium">
          <span>Rerank Blend</span>
          <span className="text-neutral-500">{config.flashrankBlendWeight?.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={config.flashrankBlendWeight || 0.5}
          onChange={(e) => updateConfig({ flashrankBlendWeight: parseFloat(e.target.value) })}
          className="w-full"
        />
        <div className="mt-1 flex justify-between text-xs text-neutral-500">
          <span>Original Score</span>
          <span>Rerank Score</span>
        </div>
      </div>
    </div>
  );
}
```

### API Integration

```typescript
// frontend/src/lib/api-client.ts (excerpt)
import { useTuningStore } from '@/stores/useTuningStore';

export async function* streamChatResponse(
  message: string,
  sessionId: string,
  contextDocuments: string[] = [],
) {
  const tuningConfig = useTuningStore.getState().config;
  
  const requestBody = {
    message,
    session_id: sessionId,
    context_documents: contextDocuments,
    
    // Include tuning parameters
    llm_model: tuningConfig.llmModel,
    embedding_model: tuningConfig.embeddingModel,
    temperature: tuningConfig.temperature,
    max_tokens: tuningConfig.maxTokens,
    retrieval_mode: tuningConfig.retrievalMode,
    retrieval_top_k: tuningConfig.retrievalTopK,
    hybrid_chunk_weight: tuningConfig.hybridChunkWeight,
    hybrid_entity_weight: tuningConfig.hybridEntityWeight,
    expansion_depth: tuningConfig.expansionDepth,
    expansion_similarity_threshold: tuningConfig.expansionSimilarityThreshold,
    max_expanded_chunks: tuningConfig.maxExpandedChunks,
    flashrank_blend_weight: tuningConfig.flashrankBlendWeight,
    flashrank_max_candidates: tuningConfig.flashrankMaxCandidates,
  };
  
  // Stream response
  yield* streamSSE('/api/chat', requestBody, {
    onToken: (token) => console.log(token),
    // ...other callbacks
  });
}
```

## Configuration

### Tuning Settings

```python
# config/settings.py
class Settings(BaseSettings):
    # Default RAG parameters (overridable via ChatRequest)
    llm_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    retrieval_top_k: int = 10
    hybrid_chunk_weight: float = 0.7
    hybrid_entity_weight: float = 0.3
    
    max_expansion_depth: int = 1
    expansion_similarity_threshold: float = 0.7
    max_expanded_chunks: int = 50
    
    flashrank_blend_weight: float = 0.5
    flashrank_max_candidates: int = 30
```

## Testing

### Tuning Tests

```python
# tests/test_tuning.py
import pytest
from api.models import ChatRequest
from api.routers.chat import merge_tuning_params

def test_merge_with_overrides():
    """Test parameter merging with overrides."""
    request = ChatRequest(
        message="test",
        session_id="test",
        temperature=0.5,
        retrieval_top_k=20,
    )
    
    config = merge_tuning_params(request)
    
    assert config["temperature"] == 0.5  # Override
    assert config["top_k"] == 20  # Override
    assert config["max_tokens"] == 2000  # Default

def test_merge_defaults():
    """Test parameter merging without overrides."""
    request = ChatRequest(
        message="test",
        session_id="test",
        retrieval_mode="hybrid",
    )
    
    config = merge_tuning_params(request)
    
    # All should be defaults
    assert config["temperature"] == 0.7
    assert config["top_k"] == 10
    assert config["hybrid_chunk_weight"] == 0.7
```

## Troubleshooting

### Common Issues

**Issue**: Tuning parameters not applied
```typescript
// Solution: Verify store state
const config = useTuningStore.getState().config;
console.log('Effective config:', config);

// Check request payload
console.log('Request body:', requestBody);
```

**Issue**: Weight validation errors
```python
# Solution: Ensure weights sum <= 1.0
if chunk_weight + entity_weight > 1.0:
    raise ValueError("Weights must sum to <= 1.0")
```

**Issue**: Invalid model names
```python
# Solution: Validate model availability
available_models = ["gpt-4", "gpt-3.5-turbo", "ollama/llama2"]

if request.llm_model and request.llm_model not in available_models:
    raise HTTPException(
        status_code=400,
        detail=f"Unknown model: {request.llm_model}",
    )
```

**Issue**: Preset not loading
```typescript
// Solution: Check preset key
const presets = {
  balanced: { /* ... */ },
  precise: { /* ... */ },
};

if (!(preset in presets)) {
  console.warn(`Unknown preset: ${preset}`);
  return;
}
```

## Related Documentation

- [Hybrid Retrieval](04-features/hybrid-retrieval.md)
- [Entity Reasoning](04-features/entity-reasoning.md)
- [Chat Interface](03-components/frontend/chat-interface.md)
- [Chat API](06-api-reference/chat.md)
