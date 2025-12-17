# RAG Tuning

Runtime configuration for retrieval, ingestion, and generation parameters.

## Overview

RAG tuning allows runtime adjustment of parameters **without restarting the server**. Changes take effect immediately on the next operation:
- **Ingestion**: Settings applied at the start of each document upload
- **Retrieval**: Settings applied at the start of each chat query

Configuration can be set via:

1. **RAG Tuning UI** - Frontend panel for ingestion parameters (Content Filtering, Entity Extraction, PDF Processing)
2. **Chat Tuning UI** - Frontend panel for retrieval parameters (Reranking, Weights, Temperature)
3. **JSON Configuration** - `config/rag_tuning_config.json` (ingestion) and `config/chat_tuning_config.json` (retrieval)
4. **Per-Request Parameters** - ChatRequest model fields
5. **Environment Variables** - Initial defaults (overridden by JSON config at runtime)

## Configuration File

### Location

```
config/rag_tuning_config.json
```

### Structure

```json
{
  "default_llm_model": "gpt-4o-mini",
  "sections": [
    {
      "key": "retrieval",
      "label": "Retrieval",
      "llm_override_enabled": false,
      "llm_override_value": null,
      "parameters": [
        {
          "key": "retrieval_mode",
          "label": "Retrieval Mode",
          "type": "select",
          "value": "hybrid",
          "options": ["vector", "hybrid", "entity"]
        },
        {
          "key": "retrieval_top_k",
          "label": "Top K Results",
          "type": "number",
          "value": 10,
          "min": 1,
          "max": 100,
          "step": 1
        }
      ]
    }
  ]
}
```

---

## Parameters Reference

### Ingestion Parameters

These settings affect document processing and are applied at the start of each upload:

#### enable_content_filtering
**Type**: `toggle` | **Default**: `true`

Enable heuristic-based filtering of low-quality chunks before embedding.

#### content_filter_min_length
**Type**: `slider` | **Range**: `10-500` | **Default**: `50`

Minimum chunk length in characters.

#### enable_entity_extraction
**Type**: `toggle` | **Default**: `true`

Enable LLM-based entity and relationship extraction.

#### enable_gleaning
**Type**: `toggle` | **Default**: `true`

Enable multi-pass entity extraction for improved quality.

#### use_marker_for_pdf
**Type**: `toggle` | **Default**: `true`

Use Marker for advanced PDF conversion (better table/equation extraction).

> [!TIP]
> See all 90+ configurable parameters in `config/rag_tuning_config.json`

---

### Retrieval Section

#### retrieval_mode

**Type**: `select`  
**Options**: `vector`, `hybrid`, `entity`  
**Default**: `hybrid`

Controls retrieval strategy:
- `vector` - Pure vector similarity search
- `hybrid` - Combines vector + entity scoring
- `entity` - Entity-first retrieval with expansion

```json
{
  "key": "retrieval_mode",
  "value": "hybrid"
}
```

#### retrieval_top_k

**Type**: `number`  
**Range**: `1-100`  
**Default**: `10`

Number of initial candidates to retrieve before expansion.

```json
{
  "key": "retrieval_top_k",
  "value": 15
}
```

#### hybrid_chunk_weight

**Type**: `slider`  
**Range**: `0.0-1.0`  
**Default**: `0.6`

Weight for chunk-based vector results in hybrid mode. Entity weight is `1 - chunk_weight`.

```json
{
  "key": "hybrid_chunk_weight",
  "value": 0.7
}
```

#### hybrid_entity_weight

**Type**: `slider`  
**Range**: `0.0-1.0`  
**Default**: `0.4`

Weight for entity-filtered results. Should sum with `hybrid_chunk_weight` to 1.0.

```json
{
  "key": "hybrid_entity_weight",
  "value": 0.3
}
```

---

### Graph Expansion Section

#### expansion_depth

**Type**: `number`  
**Range**: `0-3`  
**Default**: `1`

Maximum graph traversal depth for expansion.
- `0` - No expansion
- `1` - One hop from initial results
- `2` - Two hops (recommended)
- `3` - Three hops (high recall, slower)

```json
{
  "key": "expansion_depth",
  "value": 2
}
```

#### expansion_similarity_threshold

**Type**: `slider`  
**Range**: `0.0-1.0`  
**Default**: `0.7`

Minimum similarity for SIMILAR_TO edges during expansion.

```json
{
  "key": "expansion_similarity_threshold",
  "value": 0.75
}
```

#### max_expanded_chunks

**Type**: `number`  
**Range**: `0-200`  
**Default**: `50`

Maximum total chunks after expansion (limits context size).

```json
{
  "key": "max_expanded_chunks",
  "value": 100
}
```

---

### Reranking Section

#### flashrank_blend_weight

**Type**: `slider`  
**Range**: `0.0-1.0`  
**Default**: `0.5`

Blend between hybrid score and rerank score:
- `0.0` - Use reranker ordering exclusively
- `0.5` - Equal blend
- `1.0` - Ignore reranker, use hybrid scores

Formula: `final_score = rerank * blend + hybrid * (1 - blend)`

```json
{
  "key": "flashrank_blend_weight",
  "value": 0.0
}
```

#### flashrank_max_candidates

**Type**: `number`  
**Range**: `5-100`  
**Default**: `30`

Number of top candidates to send to reranker. Higher values improve quality but increase latency.

```json
{
  "key": "flashrank_max_candidates",
  "value": 50
}
```

---

### Generation Section

#### llm_model

**Type**: `select`  
**Options**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`  
**Default**: `gpt-4o-mini`

LLM model for response generation.

```json
{
  "key": "llm_model",
  "value": "gpt-4o"
}
```

#### temperature

**Type**: `slider`  
**Range**: `0.0-2.0`  
**Default**: `0.7`

Controls generation randomness:
- `0.0` - Deterministic
- `0.7` - Balanced (default)
- `1.5+` - Creative

```json
{
  "key": "temperature",
  "value": 0.3
}
```

#### max_tokens

**Type**: `number`  
**Range**: `100-8000`  
**Default**: `2000`

Maximum tokens in generated response.

```json
{
  "key": "max_tokens",
  "value": 3000
}
```

---

## Per-Request Overrides

Chat requests can override tuning settings:

```python
from api.models import ChatRequest

request = ChatRequest(
    message="How do I configure backups?",
    session_id="session-123",
    
    # Override retrieval
    retrieval_mode="hybrid",
    retrieval_top_k=15,
    hybrid_chunk_weight=0.7,
    
    # Override expansion
    expansion_depth=2,
    max_expanded_chunks=100,
    
    # Override reranking
    flashrank_blend_weight=0.0,
    flashrank_max_candidates=50,
    
    # Override generation
    llm_model="gpt-4o",
    temperature=0.5,
    max_tokens=3000
)
```

**Precedence**: Request parameters > RAG tuning config > Environment variables > Defaults

---

## Common Configurations

### High Quality (Slow)

Maximum quality, higher cost and latency.

```json
{
  "retrieval_top_k": 20,
  "hybrid_chunk_weight": 0.6,
  "expansion_depth": 2,
  "expansion_similarity_threshold": 0.65,
  "max_expanded_chunks": 150,
  "flashrank_enabled": true,
  "flashrank_blend_weight": 0.0,
  "flashrank_max_candidates": 50,
  "llm_model": "gpt-4o",
  "temperature": 0.3
}
```

### Balanced (Default)

Good quality/speed tradeoff.

```json
{
  "retrieval_top_k": 10,
  "hybrid_chunk_weight": 0.6,
  "expansion_depth": 1,
  "expansion_similarity_threshold": 0.7,
  "max_expanded_chunks": 50,
  "flashrank_enabled": true,
  "flashrank_blend_weight": 0.5,
  "flashrank_max_candidates": 30,
  "llm_model": "gpt-4o-mini",
  "temperature": 0.7
}
```

### Fast (Lower Quality)

Minimal latency, lower cost.

```json
{
  "retrieval_top_k": 5,
  "hybrid_chunk_weight": 0.7,
  "expansion_depth": 0,
  "max_expanded_chunks": 20,
  "flashrank_enabled": false,
  "llm_model": "gpt-3.5-turbo",
  "temperature": 0.7
}
```

---

## Tuning Guidelines

### Retrieval Top K

- **Small knowledge base** (< 100 docs): `top_k = 5-10`
- **Medium** (100-1000 docs): `top_k = 10-20`
- **Large** (1000+ docs): `top_k = 20-50`

### Expansion Depth

- **Broad queries**: `depth = 2` (follow entity connections)
- **Specific queries**: `depth = 1` (stay close to matches)
- **Performance critical**: `depth = 0` (no expansion)

### Hybrid Weights

- **Technical docs**: `chunk_weight = 0.7` (favor content)
- **Entity-rich**: `chunk_weight = 0.5` (balanced)
- **Relationship queries**: `chunk_weight = 0.4` (favor entities)

### FlashRank Blend

- **Trust reranker**: `blend_weight = 0.0`
- **Combine scores**: `blend_weight = 0.5`
- **Ignore reranker**: `blend_weight = 1.0`

### Temperature

- **Factual answers**: `temperature = 0.3`
- **General Q&A**: `temperature = 0.7`
- **Creative tasks**: `temperature = 1.2+`

---

## Loading Configuration

### Automatic Runtime Sync

Configuration is automatically synced to backend settings at the start of each operation:

```python
# Called automatically at start of:
# - document_processor.process_file() (ingestion)
# - retriever.retrieve() (queries)
from config.settings import apply_rag_tuning_overrides, settings
apply_rag_tuning_overrides(settings)
```

This reads `config/rag_tuning_config.json` and applies values to the `settings` object. **No restart required.**

### Python

```python
from config.settings import load_rag_tuning_config

config = load_rag_tuning_config()
print(config)
# {
#   "default_llm_model": "gpt-4o-mini",
#   "retrieval_top_k": 10,
#   "expansion_depth": 1,
#   ...
# }
```

### API Endpoint

```bash
# Get current tuning config
curl http://localhost:8000/api/chat-tuning/config
```

---

## Monitoring Impact

### Check Quality Score

Quality scores reflect tuning effectiveness:

```python
{
  "quality_score": 0.92,  # High = good tuning
  "components": {
    "context_relevance": 0.95,
    "answer_completeness": 0.90,
    "factual_grounding": 0.93,
    "coherence": 0.88,
    "citation_quality": 0.94
  }
}
```

### Check Retrieval Stats

```bash
# View retrieval cache hit rates
curl http://localhost:8000/api/database/cache-stats
```

### Check Latency

Monitor stage durations in SSE events:

```json
{
  "type": "stage",
  "stage": "retrieval",
  "duration_ms": 234
}
```

---

## Experimentation

### A/B Testing

Compare configurations on same queries:

```python
configs = [
    {"expansion_depth": 0, "flashrank_enabled": False},
    {"expansion_depth": 1, "flashrank_enabled": True},
    {"expansion_depth": 2, "flashrank_enabled": True}
]

for config in configs:
    response = chat(query, **config)
    print(f"Config: {config}")
    print(f"Quality: {response.quality_score}")
    print(f"Sources: {len(response.sources)}")
```

### Parameter Sweep

Test parameter ranges systematically:

```python
import numpy as np

for top_k in [5, 10, 20, 50]:
    for blend in np.linspace(0, 1, 5):
        response = chat(
            query,
            retrieval_top_k=top_k,
            flashrank_blend_weight=blend
        )
        log_metrics(top_k, blend, response.quality_score)
```

---

## Chat Tuning UI

### Access

Frontend → Settings → Chat Tuning

### Features

- Real-time parameter adjustment
- Section-specific LLM model overrides
- Preset configurations (High Quality, Balanced, Fast)
- Reset to defaults

### Workflow

1. Adjust parameters in UI
2. Submit chat query
3. Observe quality score and sources
4. Iterate on parameters
5. Save configuration when satisfied

---

## Related Documentation

- [Environment Variables](07-configuration/environment-variables.md)
- [Optimal Defaults](07-configuration/optimal-defaults.md)
- [Chat Endpoints](06-api-reference/chat-endpoints.md)
- [Retriever](03-components/backend/retriever.md)
- [Chat Tuning Panel](03-components/frontend/chat-tuning.md)
