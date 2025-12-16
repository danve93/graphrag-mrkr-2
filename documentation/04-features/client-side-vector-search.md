# Client-Side Vector Search for Static Entities

**Status:** ✅ Implemented
**Version:** 2.0.0
**Last Updated:** 2025-12-12

## Overview

Client-Side Vector Search for Static Entities is a performance optimization that enables sub-10ms category classification by performing in-memory cosine similarity search against precomputed embeddings, eliminating the need for database queries or LLM calls for static taxonomies.

### Key Features

- **Precomputed Embeddings**: Static entities (categories, entity types) are embedded once and stored in compressed JSON files
- **In-Memory Vector Search**: NumPy-based cosine similarity computation without database overhead
- **<10ms Latency**: Typical classification takes <10ms vs 50-200ms for database queries or 200-500ms for LLM routing
- **Automatic Fallback**: Gracefully falls back to LLM-based routing when static matching confidence is low
- **Memory Efficient**: Gzip compression reduces storage by ~70%, typical memory usage 0.5-2MB
- **Cache Integration**: Static routing results are cached for even faster subsequent queries

### When to Use

Client-side vector search is beneficial for:
- **Frequent category classification**: High query volume where routing latency matters
- **Static taxonomies**: Categories that change infrequently (weekly/monthly updates)
- **Cost optimization**: Reduce LLM API calls for routing by 60-80%
- **Latency-sensitive applications**: Real-time query routing requirements
- **Resource-constrained environments**: Minimize database load during peak traffic

---

## How It Works

### 1. Precomputation Phase

Before deployment, run the precomputation script to generate embeddings:

```bash
uv run python scripts/precompute_static_embeddings.py
```

**Process:**
1. Load categories from [config/document_categories.json](../../config/document_categories.json)
2. Build searchable text for each category (title + description + keywords)
3. Generate embeddings using configured embedding model
4. Normalize embeddings for cosine similarity
5. Save to compressed file: [config/static_embeddings.json.gz](../../config/static_embeddings.json.gz)

**Output Format:**
```json
{
  "version": "1.0",
  "model": "text-embedding-3-small",
  "dimension": 1536,
  "categories": [
    {
      "id": "install",
      "title": "Installation",
      "description": "Setup and installation guides",
      "keywords": ["install", "setup", "prerequisites"],
      "text": "Installation\\nSetup and installation guides\\nKeywords: install, setup, prerequisites",
      "embedding": [0.0123, -0.0456, ...]
    }
  ]
}
```

### 2. Runtime Loading

At application startup, the static matcher loads embeddings into memory:

```python
from core.static_entity_matcher import get_static_matcher

# Global singleton instance
matcher = get_static_matcher()

# Automatically loads from config/static_embeddings.json.gz
# Normalizes embeddings for fast cosine similarity
# Memory: ~0.5-2MB for typical category sets
```

### 3. Query Routing with Static Matching

During query routing ([rag/nodes/query_router.py:58-114](../../rag/nodes/query_router.py#L58-L114)), the system:

1. **Check cache first** (lines 33-56): Return cached routing if available
2. **Try static matcher** (lines 58-114):
   - Generate query embedding
   - Compute cosine similarity against all category embeddings (NumPy dot product)
   - Return top-3 matches sorted by similarity
   - If top match ≥ confidence threshold (0.7), use it immediately
   - If below threshold, fall back to LLM routing
3. **Fall back to LLM** (lines 116-141): Use LLM-based routing if static match insufficient
4. **Cache result** (lines 83-97, 119-134): Store routing decision for future queries

**Flow Diagram:**
```
Query → Cache? → Static Match → Confidence ≥ 0.7? → Return category
         ↓            ↓                ↓
        Miss        Miss            No → LLM Routing → Return category
```

### 4. Cosine Similarity Computation

The matcher uses normalized embeddings for efficient cosine similarity:

```python
# Embeddings are normalized at load time
embeddings_matrix = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)

# Query embedding is normalized at match time
query_vector = query_vector / np.linalg.norm(query_vector)

# Cosine similarity = dot product (since vectors are normalized)
similarities = np.dot(embeddings_matrix, query_vector)

# Get top-k indices
top_indices = np.argsort(similarities)[::-1][:top_k]
```

**Performance:** <10ms for 100 categories, O(n) complexity where n = number of categories

---

## Configuration

### Settings ([config/settings.py:237-243](../../config/settings.py#L237-L243))

```python
# Client-Side Vector Search for Static Entities Configuration
enable_static_entity_matching: bool = True      # Master toggle for static matching
static_matching_min_similarity: float = 0.6     # Minimum similarity for matches (lower = more permissive)
```

### Environment Variables

```bash
# Disable static entity matching (fall back to LLM routing)
ENABLE_STATIC_ENTITY_MATCHING=false

# Adjust minimum similarity threshold (default: 0.6)
STATIC_MATCHING_MIN_SIMILARITY=0.7  # Higher = more strict, lower = more permissive
```

### Configuration Tuning

**`enable_static_entity_matching`**:
- **true** (default): Use fast static matching before LLM fallback
- **false**: Disable static matching, always use LLM routing
- **When to disable**: If embeddings file is missing or categories change frequently

**`static_matching_min_similarity`**:
- **0.4-0.5**: Very permissive - matches even loosely related categories
- **0.6** (default): Balanced - requires moderate semantic similarity
- **0.7-0.8**: Strict - only matches closely related queries
- **0.9+**: Very strict - almost exact semantic match required
- **Recommendation**: Start with 0.6, increase if seeing too many false positives

**Note:** Even if a static match is found, it must also exceed the routing `confidence_threshold` (default 0.7) to be used. The final threshold is `max(static_matching_min_similarity, confidence_threshold)`.

---

## Precomputation Script

### Usage

```bash
# Basic usage (output to config/static_embeddings.json.gz)
uv run python scripts/precompute_static_embeddings.py

# Custom output path
uv run python scripts/precompute_static_embeddings.py --output /path/to/embeddings.json.gz

# Disable compression (larger file, faster loading)
uv run python scripts/precompute_static_embeddings.py --no-compress

# Verify existing embeddings file
uv run python scripts/precompute_static_embeddings.py --verify-only
```

### Output Example

```bash
=== Static Embeddings Precomputation ===
Embedding model: text-embedding-3-small
Output: /home/user/amber/config/static_embeddings.json.gz
Compression: enabled

Loaded 5 categories from /home/user/amber/config/document_categories.json
Processing category: install
✓ Generated embedding for 'install' (1536 dimensions)
Processing category: configure
✓ Generated embedding for 'configure' (1536 dimensions)
...

Saved compressed embeddings: config/static_embeddings.json.gz (42.3KB, 68.2% compression)
✓ Verification passed: 5 categories, 1536 dimensions, model: text-embedding-3-small

=== Precomputation Complete ===
Generated embeddings for 5 categories
Embedding dimension: 1536
Output file: config/static_embeddings.json.gz
```

### When to Recompute

Rerun the precomputation script when:
- **Categories are added/removed**: New category in [config/document_categories.json](../../config/document_categories.json)
- **Category metadata changes**: Updated descriptions, titles, or keywords
- **Embedding model changes**: Switched from text-embedding-3-small to ada-002, etc.
- **Quality issues**: Static matching accuracy degrades over time

**Important:** After recomputing, restart the application to reload the new embeddings.

---

## API Reference

### StaticEntityMatcher Class

**File:** [core/static_entity_matcher.py](../../core/static_entity_matcher.py)

#### `__init__(embeddings_path: Optional[Path] = None)`

Initialize matcher and optionally auto-load embeddings.

```python
from core.static_entity_matcher import StaticEntityMatcher
from pathlib import Path

# Auto-load from default path (config/static_embeddings.json.gz)
matcher = StaticEntityMatcher()

# Load from custom path
matcher = StaticEntityMatcher(embeddings_path=Path("/custom/embeddings.json.gz"))
```

#### `load(embeddings_path: Optional[Path] = None) -> bool`

Load precomputed embeddings from file.

```python
from pathlib import Path

matcher = StaticEntityMatcher()
success = matcher.load(Path("config/static_embeddings.json.gz"))

if success:
    print(f"Loaded {len(matcher.entities)} entities")
    print(f"Dimension: {matcher.dimension}")
    print(f"Model: {matcher.model}")
```

**Raises:**
- `FileNotFoundError`: If embeddings file doesn't exist
- `ValueError`: If embeddings file has invalid format

#### `async match_async(query: str, top_k: int = 3, min_similarity: float = 0.0) -> List[Dict]`

Match query against static entities using cosine similarity (async).

```python
import asyncio
from core.static_entity_matcher import get_static_matcher

async def classify_query(query: str):
    matcher = get_static_matcher()
    matches = await matcher.match_async(query, top_k=3, min_similarity=0.6)

    for match in matches:
        print(f"{match['id']}: {match['title']} (similarity: {match['similarity']:.2f})")

asyncio.run(classify_query("how to install the application"))
```

**Output:**
```
install: Installation (similarity: 0.87)
configure: Configuration (similarity: 0.65)
troubleshoot: Troubleshooting (similarity: 0.42)
```

**Returns:** List of matches sorted by similarity (highest first):
```python
[
    {
        "id": "install",
        "title": "Installation",
        "description": "Setup and installation guides",
        "keywords": ["install", "setup", "prerequisites"],
        "similarity": 0.87
    }
]
```

**Raises:** `RuntimeError` if embeddings not loaded

#### `match(query: str, top_k: int = 3, min_similarity: float = 0.0) -> List[Dict]`

Synchronous wrapper for `match_async()`.

```python
from core.static_entity_matcher import get_static_matcher

matcher = get_static_matcher()
matches = matcher.match("how to configure settings", top_k=3)
```

#### `get_entity(entity_id: str) -> Optional[Dict]`

Get entity by ID.

```python
matcher = get_static_matcher()
entity = matcher.get_entity("install")

if entity:
    print(f"{entity['title']}: {entity['description']}")
```

**Returns:** Entity dict or `None` if not found

#### `get_all_entities() -> List[Dict]`

Get all loaded entities (without embeddings).

```python
matcher = get_static_matcher()
entities = matcher.get_all_entities()

for entity in entities:
    print(f"{entity['id']}: {entity['title']}")
```

#### `explain_match(query: str, entity_id: str) -> Dict`

Explain why a query matched an entity.

```python
matcher = get_static_matcher()
explanation = matcher.explain_match("how to setup", "install")

print(f"Similarity: {explanation['similarity']:.2f}")
print(f"Matched keywords: {explanation['matched_keywords']}")
```

**Returns:**
```python
{
    "entity_id": "install",
    "entity_title": "Installation",
    "similarity": 0.85,
    "query": "how to setup",
    "matched_keywords": ["setup"],
    "description": "Setup and installation guides"
}
```

### Global Singleton Function

#### `get_static_matcher() -> StaticEntityMatcher`

Get or create global static entity matcher instance.

```python
from core.static_entity_matcher import get_static_matcher

# Always returns the same instance (singleton pattern)
matcher1 = get_static_matcher()
matcher2 = get_static_matcher()
assert matcher1 is matcher2  # True
```

**Warning:** If embeddings file is not found, the matcher will still be created but `is_loaded` will be `False`. Check `matcher.is_loaded` before using.

---

## Performance Metrics

### Latency Comparison

| Operation | Database Query | LLM Routing | Static Matching |
|-----------|----------------|-------------|-----------------|
| **Typical** | 50-200ms | 200-500ms | **5-10ms** |
| **Best case** | 30ms | 150ms | **3ms** |
| **Worst case** | 500ms | 1000ms | **20ms** |

**Speedup:** 10-50x faster than database queries, 20-100x faster than LLM routing

### Memory Usage

| Category Count | Embedding Dimension | Uncompressed | Compressed (.gz) | In-Memory |
|----------------|---------------------|--------------|------------------|-----------|
| 5 | 1536 | 150KB | 45KB | 0.5MB |
| 20 | 1536 | 600KB | 180KB | 1.2MB |
| 100 | 1536 | 3MB | 900KB | 6MB |

**Compression Ratio:** ~70% reduction with gzip

### Cost Savings

Assuming 10,000 queries/day with static matching enabled:

- **Without static matching**: 10,000 LLM routing calls/day
- **With static matching (70% hit rate)**: 3,000 LLM routing calls/day
- **LLM calls saved**: 7,000/day = 210,000/month
- **Cost savings**: ~$20-50/month (at $0.0001/call for routing)

**Additional benefit:** Reduced database load for category filtering

---

## Monitoring

### Startup Logs

Check that embeddings loaded successfully:

```bash
grep "Loading static embeddings" logs/backend.log

# Example output:
[INFO] Loading static embeddings from config/static_embeddings.json.gz
[INFO] ✓ Loaded 5 entities (1536 dimensions, model: text-embedding-3-small)
[INFO] Embeddings memory usage: 0.52 MB
```

### Routing Logs

Monitor static matching effectiveness:

```bash
# Check static routing hits
grep "Static routing:" logs/backend.log | head -5

# Example output:
[INFO] Static routing: ['install'] (confidence 0.87, filter=True)
[INFO] Static routing: ['configure'] (confidence 0.79, filter=True)
[INFO] Static routing: ['troubleshoot'] (confidence 0.92, filter=True)

# Check fallbacks to LLM
grep "Static match below threshold" logs/backend.log

# Example output:
[DEBUG] Static match below threshold: install (sim=0.62 < 0.70), falling back to LLM
```

### Performance Metrics

Track latency improvements:

```bash
# Calculate average routing time (requires custom logging)
grep "Routing latency" logs/backend.log | awk '{sum+=$NF; count++} END {print "Average:", sum/count "ms"}'
```

### Hit Rate Analysis

Measure static matching effectiveness:

```bash
# Count static hits vs LLM fallbacks
STATIC_HITS=$(grep -c "Static routing:" logs/backend.log)
LLM_FALLBACKS=$(grep -c "Query routed to categories:" logs/backend.log)
TOTAL=$((STATIC_HITS + LLM_FALLBACKS))
HIT_RATE=$((100 * STATIC_HITS / TOTAL))

echo "Static matching hit rate: ${HIT_RATE}%"
```

**Target:** 60-80% hit rate for well-configured static matching

---

## Troubleshooting

### Problem: Embeddings File Not Found

**Symptoms:** Warning at startup: "Static embeddings not found"

**Causes:**
- Precomputation script never run
- Embeddings file deleted or moved
- Incorrect path configuration

**Solutions:**

1. Run precomputation script:
   ```bash
   uv run python scripts/precompute_static_embeddings.py
   ```

2. Verify file exists:
   ```bash
   ls -lh config/static_embeddings.json.gz
   ```

3. Check file permissions:
   ```bash
   chmod 644 config/static_embeddings.json.gz
   ```

### Problem: Low Hit Rate (<40%)

**Symptoms:** Most queries fall back to LLM routing

**Causes:**
- Categories too broad or generic
- `static_matching_min_similarity` too high
- Embedding model mismatch (precomputed with different model)
- Poor category metadata quality

**Solutions:**

1. Lower minimum similarity threshold:
   ```bash
   export STATIC_MATCHING_MIN_SIMILARITY=0.5  # More permissive
   ```

2. Improve category metadata in [config/document_categories.json](../../config/document_categories.json):
   - Add more descriptive keywords
   - Enhance category descriptions
   - Use natural language that matches query patterns

3. Recompute embeddings with better metadata:
   ```bash
   uv run python scripts/precompute_static_embeddings.py
   ```

4. Verify embedding model matches runtime:
   ```bash
   # Check precomputed model
   python -c "import gzip, json; print(json.load(gzip.open('config/static_embeddings.json.gz'))['model'])"

   # Check runtime model
   grep "embedding_model" config/settings.py
   ```

### Problem: High False Positive Rate

**Symptoms:** Static matcher returns wrong categories

**Causes:**
- `static_matching_min_similarity` too low
- Similar categories with overlapping keywords
- Embedding model not distinguishing categories well

**Solutions:**

1. Increase minimum similarity threshold:
   ```bash
   export STATIC_MATCHING_MIN_SIMILARITY=0.75  # More strict
   ```

2. Improve category differentiation:
   - Make category descriptions more distinct
   - Reduce keyword overlap between categories
   - Add negative keywords (what the category is NOT about)

3. Review false positives in logs:
   ```bash
   # Find queries where static match had low confidence
   grep "Static routing:.*confidence 0\.[5-6]" logs/backend.log
   ```

### Problem: High Memory Usage

**Symptoms:** Application memory usage increases after loading embeddings

**Causes:**
- Large number of categories (100+)
- High embedding dimensions (3072 for text-embedding-3-large)
- Multiple matcher instances created

**Solutions:**

1. Use singleton pattern (recommended):
   ```python
   from core.static_entity_matcher import get_static_matcher
   matcher = get_static_matcher()  # Always reuses same instance
   ```

2. Reduce embedding dimensions (if using OpenAI):
   ```python
   # In config/settings.py or environment
   OPENAI_EMBEDDING_DIMENSIONS=768  # Reduce from 1536
   ```

3. Monitor memory usage:
   ```python
   matcher = get_static_matcher()
   memory_mb = matcher.embeddings_matrix.nbytes / (1024 * 1024)
   print(f"Embeddings memory: {memory_mb:.2f} MB")
   ```

### Problem: Outdated Embeddings

**Symptoms:** Static matching accuracy degrades over time

**Causes:**
- Categories updated but embeddings not recomputed
- Embedding model changed in config
- Query patterns shift without category updates

**Solutions:**

1. Recompute embeddings monthly (or when categories change):
   ```bash
   uv run python scripts/precompute_static_embeddings.py
   ```

2. Set up automated recomputation:
   ```bash
   # Cron job (weekly on Sundays at 2 AM)
   0 2 * * 0 cd /path/to/amber && uv run python scripts/precompute_static_embeddings.py
   ```

3. Verify embeddings version:
   ```bash
   python -c "import gzip, json; data=json.load(gzip.open('config/static_embeddings.json.gz')); print(f\"Version: {data['version']}, Model: {data['model']}, Categories: {len(data['categories'])}\")"
   ```

---

## Best Practices

### 1. Recompute Embeddings Regularly

Set up a monthly recomputation schedule to keep embeddings fresh:

```bash
# Add to crontab
0 2 1 * * cd /path/to/amber && uv run python scripts/precompute_static_embeddings.py
```

### 2. Monitor Hit Rate

Track static matching effectiveness and tune thresholds accordingly:

```bash
# Weekly hit rate check
STATIC_HITS=$(grep -c "Static routing:" logs/backend.log)
TOTAL=$(grep -c "routing:" logs/backend.log)
echo "Hit rate: $((100 * STATIC_HITS / TOTAL))%"
```

**Target:** 60-80% hit rate

### 3. Optimize Category Metadata

Ensure high-quality category definitions for better matching:

**Good Example:**
```json
{
  "id": "install",
  "title": "Installation",
  "description": "Complete guides for installing, setting up, and configuring the application for the first time",
  "keywords": ["install", "setup", "installation", "first time setup", "prerequisites", "requirements"]
}
```

**Bad Example:**
```json
{
  "id": "install",
  "title": "Install",
  "description": "Install stuff",
  "keywords": ["install"]
}
```

### 4. Version Embeddings File

Track embeddings file versions for rollback capability:

```bash
# Copy embeddings with version suffix
cp config/static_embeddings.json.gz config/static_embeddings-v1.0.json.gz

# Recompute new version
uv run python scripts/precompute_static_embeddings.py

# If issues, rollback
cp config/static_embeddings-v1.0.json.gz config/static_embeddings.json.gz
```

### 5. Test Before Deployment

Verify embeddings quality before deploying to production:

```bash
# Verify file format
uv run python scripts/precompute_static_embeddings.py --verify-only

# Run unit tests
uv run pytest tests/unit/test_static_entity_matcher.py -v

# Check memory usage
python -c "from core.static_entity_matcher import get_static_matcher; m=get_static_matcher(); print(f'Memory: {m.embeddings_matrix.nbytes/(1024*1024):.2f}MB')"
```

### 6. Use Appropriate Similarity Thresholds

Tune thresholds based on your taxonomy:

- **Small taxonomy (5-10 categories)**: 0.6-0.7 (balanced)
- **Medium taxonomy (20-50 categories)**: 0.65-0.75 (slightly strict)
- **Large taxonomy (100+ categories)**: 0.7-0.8 (strict to avoid false positives)

### 7. Combine with Routing Cache

For best performance, enable both static matching and routing cache:

```python
# config/settings.py
enable_static_entity_matching = True    # <10ms static matching
enable_routing_cache = True             # ~1ms cache hits
```

**Combined Performance:**
- Cache hit (~60%): ~1ms
- Static match (~30%): ~5-10ms
- LLM fallback (~10%): ~200-500ms
- **Average latency:** ~40ms (vs 200ms LLM-only)

---

## Related Documentation

- [Query Routing](./query-routing.md) - LLM-based routing fallback
- [Semantic Caching](./semantic-caching.md) - Cache routing decisions
- [Multi-Stage Retrieval](./multi-stage-retrieval.md) - Use categories for filtering

---

**Last Updated:** 2025-12-12
**Feature Status:** ✅ Production Ready
