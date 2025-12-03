# Embeddings Component

Embedding generation, caching, and batch processing for semantic search.

## Overview

The embeddings component manages text-to-vector transformations using OpenAI or Ollama embedding models. It provides async batch processing, rate limiting, caching, and concurrent request management to optimize API usage and performance.

**Location**: `core/embeddings.py`
**Models**: OpenAI (text-embedding-3-small/large), Ollama (nomic-embed-text)
**Cache**: LRU cache keyed by text+model hash

## Architecture

```
┌──────────────────────────────────────────────────┐
│           EmbeddingManager                        │
├──────────────────────────────────────────────────┤
│                                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  Provider  │  │   Cache    │  │    Rate    │ │
│  │  Routing   │  │  Manager   │  │  Limiter   │ │
│  └────────────┘  └────────────┘  └────────────┘ │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │         Batch Processing Engine            │  │
│  ├────────────────────────────────────────────┤  │
│  │  • Semaphore concurrency control           │  │
│  │  • Automatic batching and chunking         │  │
│  │  • Retry logic with exponential backoff    │  │
│  │  • Progress tracking and error handling    │  │
│  └────────────────────────────────────────────┘  │
│                                                   │
└───────────────────┬──────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   ┌──────────┐          ┌──────────┐
   │  OpenAI  │          │  Ollama  │
   │    API   │          │    API   │
   └──────────┘          └──────────┘
```

## Embedding Manager

### Initialization

```python
import asyncio
import hashlib
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import httpx
from config.settings import settings
from core.singletons import cache_manager

class EmbeddingManager:
    def __init__(self):
        self.provider = settings.embedding_provider
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(settings.embedding_concurrency)
        self.min_delay = settings.embedding_delay_min
        self.max_delay = settings.embedding_delay_max
        
        # Initialize clients
        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        elif self.provider == "ollama":
            self.base_url = settings.ollama_base_url
            self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        """Close HTTP clients."""
        if self.provider == "ollama" and self.client:
            await self.client.aclose()
```

### Configuration

```bash
# Provider selection
EMBEDDING_PROVIDER=openai  # or "ollama"
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# OpenAI settings
OPENAI_API_KEY=sk-...

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Concurrency and rate limiting
EMBEDDING_CONCURRENCY=10
EMBEDDING_DELAY_MIN=0.05
EMBEDDING_DELAY_MAX=0.1

# Caching
ENABLE_CACHING=true
EMBEDDING_CACHE_SIZE=10000
```

## Single Embedding Generation

### Core Function with Caching

```python
async def get_embedding(self, text: str, cache_key: Optional[str] = None) -> List[float]:
    """
    Generate embedding for a single text with caching.
    
    Args:
        text: Input text to embed
        cache_key: Optional cache key (defaults to hash of text+model)
    
    Returns:
        List of floats representing the embedding vector
    """
    if not text or not text.strip():
        return [0.0] * self.dimension
    
    # Generate cache key
    if cache_key is None:
        cache_key = self._generate_cache_key(text)
    
    # Check cache
    if settings.enable_caching and cache_key in cache_manager.embedding_cache:
        return cache_manager.embedding_cache[cache_key]
    
    # Generate embedding
    async with self.semaphore:
        if self.provider == "openai":
            embedding = await self._generate_openai_embedding(text)
        elif self.provider == "ollama":
            embedding = await self._generate_ollama_embedding(text)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Cache result
        if settings.enable_caching:
            cache_manager.embedding_cache[cache_key] = embedding
        
        # Rate limiting delay
        await asyncio.sleep(
            self.min_delay + (self.max_delay - self.min_delay) * asyncio.random.random()
        )
        
        return embedding

def _generate_cache_key(self, text: str) -> str:
    """Generate deterministic cache key."""
    content = f"{text}::{self.model}"
    return hashlib.sha256(content.encode()).hexdigest()
```

### OpenAI Provider

```python
async def _generate_openai_embedding(self, text: str) -> List[float]:
    """Generate embedding using OpenAI API."""
    try:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimension
        )
        return response.data[0].embedding
    
    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        raise
```

### Ollama Provider

```python
async def _generate_ollama_embedding(self, text: str) -> List[float]:
    """Generate embedding using Ollama API."""
    try:
        response = await self.client.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    
    except Exception as e:
        logger.error(f"Ollama embedding error: {e}")
        raise
```

## Batch Embedding Generation

### Core Batch Function

```python
async def get_embeddings_batch(
    self,
    texts: List[str],
    show_progress: bool = False
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts with optimal batching.
    
    Args:
        texts: List of texts to embed
        show_progress: Show progress bar (requires tqdm)
    
    Returns:
        List of embedding vectors in same order as input texts
    """
    if not texts:
        return []
    
    # Generate cache keys
    cache_keys = [self._generate_cache_key(text) for text in texts]
    
    # Separate cached and uncached
    embeddings = [None] * len(texts)
    tasks = []
    
    for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
        if settings.enable_caching and cache_key in cache_manager.embedding_cache:
            embeddings[i] = cache_manager.embedding_cache[cache_key]
        else:
            tasks.append((i, text, cache_key))
    
    # Process uncached texts
    if tasks:
        if show_progress:
            from tqdm.asyncio import tqdm
            task_results = await tqdm.gather(
                *[self.get_embedding(text, cache_key) for _, text, cache_key in tasks],
                desc="Generating embeddings"
            )
        else:
            task_results = await asyncio.gather(
                *[self.get_embedding(text, cache_key) for _, text, cache_key in tasks],
                return_exceptions=True
            )
        
        # Assign results
        for (i, _, _), result in zip(tasks, task_results):
            if isinstance(result, Exception):
                logger.error(f"Embedding generation failed for index {i}: {result}")
                embeddings[i] = [0.0] * self.dimension
            else:
                embeddings[i] = result
    
    return embeddings
```

### Provider-Specific Batching

```python
async def get_embeddings_batch_openai(
    self,
    texts: List[str],
    batch_size: int = 100
) -> List[List[float]]:
    """
    OpenAI-optimized batch processing (API supports up to 2048 texts).
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size for API calls
    
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        async with self.semaphore:
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimension
                )
                
                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Cache results
                if settings.enable_caching:
                    for text, embedding in zip(batch, batch_embeddings):
                        cache_key = self._generate_cache_key(text)
                        cache_manager.embedding_cache[cache_key] = embedding
                
                # Rate limiting
                await asyncio.sleep(self.min_delay)
            
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                # Fallback to zero vectors
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))
    
    return all_embeddings
```

## Concurrent Processing

### Semaphore-Based Concurrency

```python
class EmbeddingManager:
    async def process_with_concurrency(
        self,
        items: List[Dict],
        text_extractor: callable,
        max_concurrent: Optional[int] = None
    ) -> List[Dict]:
        """
        Process items with controlled concurrency.
        
        Args:
            items: List of items to process
            text_extractor: Function to extract text from item
            max_concurrent: Override default concurrency limit
        
        Returns:
            Items with 'embedding' field added
        """
        semaphore = asyncio.Semaphore(
            max_concurrent or settings.embedding_concurrency
        )
        
        async def process_item(item: Dict) -> Dict:
            async with semaphore:
                text = text_extractor(item)
                embedding = await self.get_embedding(text)
                item["embedding"] = embedding
                return item
        
        results = await asyncio.gather(
            *[process_item(item) for item in items],
            return_exceptions=True
        )
        
        # Filter exceptions
        return [r for r in results if not isinstance(r, Exception)]
```

### Chunk Processing Example

```python
async def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """Add embeddings to chunk objects."""
    manager = EmbeddingManager()
    
    try:
        embeddings = await manager.get_embeddings_batch(
            texts=[chunk["text"] for chunk in chunks],
            show_progress=True
        )
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        return chunks
    
    finally:
        await manager.close()
```

## Retry Logic and Error Handling

### Retry with Exponential Backoff

```python
import asyncio
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator

class EmbeddingManager:
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def get_embedding_with_retry(self, text: str) -> List[float]:
        """Get embedding with automatic retry."""
        return await self.get_embedding(text)
```

### Graceful Degradation

```python
async def get_embedding_safe(self, text: str) -> List[float]:
    """Get embedding with fallback to zero vector."""
    try:
        return await self.get_embedding(text)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return [0.0] * self.dimension
```

## Cache Management

### Cache Integration

```python
from core.singletons import cache_manager, CacheMetrics

class EmbeddingManager:
    def get_cache_stats(self) -> Dict:
        """Get embedding cache statistics."""
        if not settings.enable_caching:
            return {"enabled": False}
        
        cache = cache_manager.embedding_cache
        metrics = CacheMetrics.get_instance()
        
        return {
            "enabled": True,
            "size": len(cache),
            "max_size": cache.maxsize,
            "hit_rate": metrics.get_hit_rate("embedding"),
            "total_hits": metrics.cache_hits.get("embedding", 0),
            "total_misses": metrics.cache_misses.get("embedding", 0)
        }
    
    def clear_cache(self):
        """Clear embedding cache."""
        if settings.enable_caching:
            cache_manager.embedding_cache.clear()
            logger.info("Embedding cache cleared")
```

### Cache Warmup

```python
async def warmup_cache(texts: List[str]):
    """Pre-populate cache with common texts."""
    manager = EmbeddingManager()
    
    try:
        logger.info(f"Warming up embedding cache with {len(texts)} texts")
        await manager.get_embeddings_batch(texts, show_progress=True)
        
        stats = manager.get_cache_stats()
        logger.info(f"Cache warmup complete: {stats['size']} embeddings cached")
    
    finally:
        await manager.close()
```

## Usage Examples

### Document Ingestion

```python
from core.embeddings import EmbeddingManager

async def ingest_document(chunks: List[Dict]):
    """Ingest document with embeddings."""
    manager = EmbeddingManager()
    
    try:
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await manager.get_embeddings_batch(
            texts=texts,
            show_progress=True
        )
        
        # Attach to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        # Persist to database
        from core.graph_db import get_db
        db = get_db()
        db.create_chunks_batch(document_id, chunks)
        
        logger.info(f"Ingested {len(chunks)} chunks with embeddings")
    
    finally:
        await manager.close()
```

### Query Embedding

```python
async def embed_query(query: str) -> List[float]:
    """Generate embedding for search query."""
    manager = EmbeddingManager()
    
    try:
        return await manager.get_embedding(query)
    finally:
        await manager.close()
```

### Entity Embeddings

```python
async def embed_entities(entities: List[Dict]):
    """Generate embeddings for entity descriptions."""
    manager = EmbeddingManager()
    
    try:
        # Extract entity descriptions
        texts = []
        for entity in entities:
            text = f"{entity['name']}: {entity['description']}"
            texts.append(text)
        
        # Generate embeddings
        embeddings = await manager.get_embeddings_batch(texts)
        
        # Attach to entities
        for entity, embedding in zip(entities, embeddings):
            entity["embedding"] = embedding
        
        return entities
    
    finally:
        await manager.close()
```

## Performance Optimization

### Concurrency Tuning

```python
# High throughput (requires higher API limits)
EMBEDDING_CONCURRENCY=20
EMBEDDING_DELAY_MIN=0.01
EMBEDDING_DELAY_MAX=0.05

# Conservative (for rate-limited accounts)
EMBEDDING_CONCURRENCY=5
EMBEDDING_DELAY_MIN=0.1
EMBEDDING_DELAY_MAX=0.2
```

### Batch Size Optimization

```python
# OpenAI: up to 2048 texts per request
# Recommended batch sizes:
# - Small texts (<100 tokens): 100-500
# - Medium texts (100-500 tokens): 50-100
# - Large texts (>500 tokens): 20-50

async def embed_with_optimal_batching(
    texts: List[str],
    avg_token_count: int
) -> List[List[float]]:
    """Embed with optimal batch size based on token count."""
    if avg_token_count < 100:
        batch_size = 500
    elif avg_token_count < 500:
        batch_size = 100
    else:
        batch_size = 50
    
    manager = EmbeddingManager()
    try:
        return await manager.get_embeddings_batch_openai(
            texts=texts,
            batch_size=batch_size
        )
    finally:
        await manager.close()
```

### Cache Hit Rate Monitoring

```python
from core.cache_metrics import CacheMetrics

def log_cache_performance():
    """Log cache performance metrics."""
    metrics = CacheMetrics.get_instance()
    hit_rate = metrics.get_hit_rate("embedding")
    
    if hit_rate < 0.3:
        logger.warning(f"Low embedding cache hit rate: {hit_rate:.2%}")
    else:
        logger.info(f"Embedding cache hit rate: {hit_rate:.2%}")
```

## Testing

### Unit Tests

```python
import pytest
from core.embeddings import EmbeddingManager

@pytest.fixture
async def manager():
    mgr = EmbeddingManager()
    yield mgr
    await mgr.close()

@pytest.mark.asyncio
async def test_single_embedding(manager):
    text = "This is a test document."
    embedding = await manager.get_embedding(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) == manager.dimension
    assert all(isinstance(x, float) for x in embedding)

@pytest.mark.asyncio
async def test_batch_embeddings(manager):
    texts = ["First text", "Second text", "Third text"]
    embeddings = await manager.get_embeddings_batch(texts)
    
    assert len(embeddings) == len(texts)
    assert all(len(emb) == manager.dimension for emb in embeddings)

@pytest.mark.asyncio
async def test_caching(manager):
    text = "Cached text"
    
    # First call - cache miss
    emb1 = await manager.get_embedding(text)
    
    # Second call - cache hit
    emb2 = await manager.get_embedding(text)
    
    assert emb1 == emb2

@pytest.mark.asyncio
async def test_empty_text(manager):
    embedding = await manager.get_embedding("")
    assert embedding == [0.0] * manager.dimension
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_document_embedding_pipeline():
    """Test full document embedding pipeline."""
    chunks = [
        {"id": "c1", "text": "First chunk"},
        {"id": "c2", "text": "Second chunk"},
        {"id": "c3", "text": "Third chunk"}
    ]
    
    manager = EmbeddingManager()
    try:
        embedded_chunks = await manager.process_with_concurrency(
            items=chunks,
            text_extractor=lambda x: x["text"]
        )
        
        assert len(embedded_chunks) == len(chunks)
        assert all("embedding" in chunk for chunk in embedded_chunks)
    
    finally:
        await manager.close()
```

## Troubleshooting

### Common Issues

**Issue**: Rate limit errors from OpenAI
```python
# Solution: Increase delays
EMBEDDING_DELAY_MIN=0.2
EMBEDDING_DELAY_MAX=0.5
EMBEDDING_CONCURRENCY=5
```

**Issue**: Out of memory with large batches
```python
# Solution: Reduce batch size and concurrency
EMBEDDING_CONCURRENCY=5
# Use smaller batches in get_embeddings_batch_openai
batch_size = 50
```

**Issue**: Timeout errors with Ollama
```python
# Solution: Increase timeout in httpx client
self.client = httpx.AsyncClient(timeout=120.0)
```

**Issue**: Cache not effective
```python
# Solution: Increase cache size
EMBEDDING_CACHE_SIZE=50000

# Or check cache key generation
def _generate_cache_key(self, text: str) -> str:
    # Ensure consistent normalization
    text = text.strip().lower()
    content = f"{text}::{self.model}"
    return hashlib.sha256(content.encode()).hexdigest()
```

## Related Documentation

- [Caching System](02-core-concepts/caching-system.md)
- [Document Processor](03-components/ingestion/document-processor.md)
- [Retriever](03-components/backend/retriever.md)
- [Configuration Reference](07-configuration/environment-variables.md)
