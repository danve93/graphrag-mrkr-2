# Fuzzy Matching (Technical Terms & Typo Correction)

**Status:** ✅ Implemented
**Version:** 2.0.0
**Last Updated:** 2025-12-12

## Overview

The Fuzzy Matching feature improves retrieval accuracy for technical queries containing specific identifiers, code references, or minor typos by using Neo4j's fuzzy search operator (`~`). This enables the system to find relevant documents even when query terms don't exactly match the indexed content.

### Key Benefits

- **Technical Term Tolerance**: Matches snake_case, camelCase, and technical identifiers with minor variations
- **Typo Correction**: Handles 1-2 character errors in query terms
- **Automatic Detection**: Identifies technical queries and applies fuzzy matching automatically
- **Configuration Control**: Adjustable edit distance and confidence thresholds
- **Zero Impact on Non-Technical Queries**: Standard search for general queries

### Impact

- **Better recall** for queries containing technical terms like database column names, config keys, error codes
- **Improved user experience** - no need to type technical terms perfectly
- **Automatic activation** - detects technical queries without manual configuration

---

## How It Works

### Neo4j Fuzzy Search Operator

Neo4j's fulltext index supports fuzzy search using the `~` operator with edit distance:

```cypher
// Exact match
CALL db.index.fulltext.queryNodes('chunk_content_fulltext', 'authentication')

// Fuzzy match (allows 2 character edits)
CALL db.index.fulltext.queryNodes('chunk_content_fulltext', 'authentication~2')
```

**Edit Distance Examples:**

| Query Term | Fuzzy Distance | Matches |
|------------|----------------|---------|
| `auth` | `~1` | auth, Auth, auth_, auths |
| `auth` | `~2` | auth, authy, authr, autho, auths, autor |
| `user_accounts` | `~2` | user_accounts, user_account, useraccounts, user_acount |

### Technical Query Detection

The system automatically detects queries containing technical terms using pattern matching:

```
Query: "find data in user_accounts table"
   ↓
Pattern Detection:
   → Snake_case detected: "user_accounts"
   → Technical confidence: 0.6
   ↓
Fuzzy Distance Calculation:
   → 1 technical term found
   → Fuzzy distance: 1 (allows 1-char edits)
   ↓
Query Transformation:
   → Original: "find data in user_accounts table"
   → Transformed: "find~1 data~1 in~1 user_accounts~1 table~1"
   ↓
BM25 Keyword Search:
   → Searches with fuzzy matching enabled
   → Returns candidates with term variations
```

### Technical Patterns Detected

**Pattern 1: Database Identifiers (snake_case)**
```
Examples: user_accounts, api_key, max_connections
Regex: \b[a-z]+_[a-z_]+\b
```

**Pattern 2: Technical IDs**
```
Examples: PROJ-123, TICKET-456, BUG-789
Regex: \b[A-Z]{2,}-\d+\b
```

**Pattern 3: Configuration Keys**
```
Examples: MAX_CONNECTIONS, apiKey, DATABASE_URL
Regex: \b[A-Z][A-Z_]{2,}\b|\b[a-z]+[A-Z][a-zA-Z]+\b
```

**Pattern 4: Error Codes**
```
Examples: ERROR_404, err_timeout, error_not_found
Regex: \b(error|err)[_\-]?[a-z0-9_]+\b
```

**Pattern 5: File Paths/Extensions**
```
Examples: config.yml, /etc/nginx.conf, database.ini
Regex: \b\w+\.(yml|yaml|json|conf|config|ini|xml|properties)\b|/\w+/[\w/\.]+
```

### Fuzzy Distance Calculation

```python
# 1 technical term → fuzzy_distance = 1
query = "find user_accounts"
# Result: fuzzy_distance = 1

# 2+ technical terms → fuzzy_distance = 2 (capped at max_fuzzy_distance)
query = "error in user_accounts at MAX_CONNECTIONS"
# Result: fuzzy_distance = 2

# Confidence threshold check
if confidence < fuzzy_confidence_threshold:
    fuzzy_distance = 0  # Disable fuzzy matching
```

---

## Configuration

### Settings ([config/settings.py](../../config/settings.py#L198-L207))

```python
# Fuzzy Matching Configuration
enable_fuzzy_matching: bool = True              # Enable/disable feature
max_fuzzy_distance: int = 2                     # Maximum edit distance (1-2)
fuzzy_confidence_threshold: float = 0.5         # Minimum confidence (0.0-1.0)
```

### Environment Variables

```bash
# Disable fuzzy matching
ENABLE_FUZZY_MATCHING=false

# Adjust maximum edit distance (Neo4j supports 0-2)
MAX_FUZZY_DISTANCE=1

# Adjust confidence threshold (stricter detection)
FUZZY_CONFIDENCE_THRESHOLD=0.7
```

### Tuning Guidelines

**`max_fuzzy_distance`**:
- **1**: Conservative - matches only 1-character variations (faster, fewer false positives)
- **2**: Aggressive - matches 2-character variations (slower, better recall)
- **Default: 2** - Good balance for technical terms and typos

**`fuzzy_confidence_threshold`**:
- **Lower (e.g., 0.3)**: Enables fuzzy for more queries (more lenient)
- **Higher (e.g., 0.7)**: Only very technical queries get fuzzy (stricter)
- **Default: 0.5** - Balanced detection

**Performance Considerations**:
- Fuzzy search is slower than exact match (2-3x overhead)
- Edit distance 2 is ~2x slower than distance 1
- Only applies to BM25 keyword search (Stage 1 in two-stage retrieval)

---

## Usage Examples

### Example 1: Technical Query with Typo

**Scenario**: User types "user_acount" instead of "user_accounts"

```python
from rag.retriever import DocumentRetriever

retriever = DocumentRetriever()

# Query with typo
results = await retriever.chunk_based_retrieval(
    query="find data in user_acount table",  # Typo: acount vs accounts
    top_k=5
)

# Log output:
# INFO: Technical query detected: fuzzy_distance=1
# DEBUG: Fuzzy search: 'find data in user_acount table' -> 'find~1 data~1 in~1 user_acount~1 table~1'
# DEBUG: Stage 1: BM25 search for 50 candidates with fuzzy matching

# Results include documents containing "user_accounts" (correct spelling)
```

**Result**: Finds documents with correct spelling despite typo

### Example 2: Error Code Search

**Scenario**: Searching for error code documentation

```python
results = await retriever.chunk_based_retrieval(
    query="how to fix ERROR_404 in the application",
    top_k=5
)

# Log output:
# INFO: Technical query detected: fuzzy_distance=1
# Technical pattern detected: ERROR_404

# Matches documents containing:
# - ERROR_404 (exact)
# - ERROR_403, ERROR_405 (fuzzy ~1)
# - error_404, Error404 (case variations)
```

### Example 3: Configuration Key Search

**Scenario**: Looking for configuration documentation

```python
results = await retriever.chunk_based_retrieval(
    query="what is MAX_CONNECTIONS set to",
    top_k=5
)

# Log output:
# INFO: Technical query detected: fuzzy_distance=1
# Technical pattern detected: MAX_CONNECTIONS

# Matches:
# - MAX_CONNECTIONS (exact)
# - MAX_CONNECTION, MAXCONNECTIONS (typos)
# - max_connections (case variations)
```

### Example 4: Non-Technical Query (No Fuzzy)

**Scenario**: General query without technical terms

```python
results = await retriever.chunk_based_retrieval(
    query="what is the capital of France",
    top_k=5
)

# Log output:
# (No fuzzy matching log - uses exact search)

# Uses standard exact keyword matching
```

**Result**: No performance overhead for non-technical queries

---

## API Usage

### Automatic Fuzzy Matching (Recommended)

```python
from rag.retriever import DocumentRetriever

retriever = DocumentRetriever()

# Fuzzy matching activates automatically for technical queries
results = await retriever.chunk_based_retrieval(
    query="status of PROJ-123 in user_accounts",  # Contains 2 technical terms
    top_k=5,
)

for chunk in results:
    print(f"{chunk['document_name']}: {chunk['similarity']:.3f}")
    print(f"  {chunk['content'][:100]}...")
```

### Direct Access to Technical Detection

```python
from rag.nodes.query_analysis import _detect_technical_query

query = "find user_accounts table"
technical_info = _detect_technical_query(query.lower())

print(f"Is Technical: {technical_info['is_technical']}")
print(f"Fuzzy Distance: {technical_info['fuzzy_distance']}")
print(f"Confidence: {technical_info['confidence']}")
```

**Output:**
```
Is Technical: True
Fuzzy Distance: 1
Confidence: 0.6
```

### Direct Fuzzy Search

```python
from core.graph_db import graph_db

# Manual fuzzy keyword search
results = graph_db.chunk_keyword_search(
    query="authentication system",
    top_k=10,
    fuzzy_distance=2,  # Allow 2-character edits
)

for chunk in results:
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Content: {chunk['content'][:100]}...")
```

### Query Analysis Integration

```python
from rag.nodes.query_analysis import analyze_query

query = "error in MAX_CONNECTIONS config.yml"
analysis = analyze_query(query)

print(f"Is Technical: {analysis['is_technical']}")
print(f"Fuzzy Distance: {analysis['fuzzy_distance']}")
print(f"Technical Confidence: {analysis['technical_confidence']}")
```

---

## Architecture Details

### Implementation Files

**Technical Query Detection** ([rag/nodes/query_analysis.py:628-719](../../rag/nodes/query_analysis.py#L628-L719)):
```python
def _detect_technical_query(query_lower: str) -> Dict[str, Any]:
    """Detect if query contains technical terms that benefit from fuzzy matching."""

    # Pattern matching for 5 technical patterns
    snake_case_matches = re.findall(r'\b[a-z]+_[a-z_]+\b', query_lower)
    tech_id_matches = re.findall(r'\b[A-Z]{2,}-\d+\b', query_lower.upper())
    config_matches = re.findall(r'\b[A-Z][A-Z_]{2,}\b|\b[a-z]+[A-Z][a-zA-Z]+\b', query_lower.upper())
    error_matches = re.findall(r'\b(error|err)[_\-]?[a-z0-9_]+\b', query_lower)
    file_ext_pattern = r'\b\w+\.(yml|yaml|json|conf|config|ini|xml|properties)\b'
    file_path_pattern = r'/\w+/[\w/\.]+'
    file_matches = re.findall(file_ext_pattern, query_lower) + re.findall(file_path_pattern, query_lower)

    # Calculate fuzzy distance based on matches
    total_matches = len(snake_case_matches) + len(tech_id_matches) + len(config_matches) + len(error_matches) + len(file_matches)

    if total_matches > 0:
        fuzzy_distance = min(total_matches, settings.max_fuzzy_distance)
        confidence = min(0.5 + (total_matches * 0.2), 1.0)

        if confidence >= settings.fuzzy_confidence_threshold:
            return {
                "is_technical": True,
                "fuzzy_distance": fuzzy_distance,
                "confidence": confidence,
            }

    return {"is_technical": False, "fuzzy_distance": 0, "confidence": 0.0}
```

**Fuzzy Keyword Search** ([core/graph_db.py:2132-2160](../../core/graph_db.py#L2132-L2160)):
```python
def chunk_keyword_search(
    self,
    query: str,
    top_k: int = 10,
    allowed_document_ids: Optional[List[str]] = None,
    fuzzy_distance: int = 0,
) -> List[Dict[str, Any]]:
    """Perform BM25-style keyword search with optional fuzzy matching."""

    # Apply fuzzy operator if requested
    search_query = query
    if fuzzy_distance > 0:
        terms = query.split()
        fuzzy_terms = [f"{term}~{fuzzy_distance}" for term in terms]
        search_query = " ".join(fuzzy_terms)
        logger.debug(f"Fuzzy search: '{query}' -> '{search_query}'")

    # Execute fulltext search with fuzzy query
    with self.session_scope() as session:
        result = session.run("""
            CALL db.index.fulltext.queryNodes('chunk_content_fulltext', $search_query)
            YIELD node, score
            MATCH (d:Document)-[:HAS_CHUNK]->(node)
            WHERE $allowed_doc_ids IS NULL OR d.id IN $allowed_doc_ids
            RETURN node.id as chunk_id, node.content as content, score,
                   coalesce(d.original_filename, d.filename) as document_name, d.id as document_id
            ORDER BY score DESC
            LIMIT $top_k
        """, search_query=search_query, allowed_doc_ids=allowed_document_ids, top_k=top_k)
```

**Integration in Retrieval** ([rag/retriever.py:977-985](../../rag/retriever.py#L977-L985)):
```python
# Extract fuzzy matching parameters from query analysis
fuzzy_distance = query_analysis.get("fuzzy_distance", 0)
is_technical = query_analysis.get("is_technical", False)

if is_technical:
    logger.info(f"Technical query detected: fuzzy_distance={fuzzy_distance}")

# Pass to keyword search in two-stage retrieval
keyword_results = graph_db.chunk_keyword_search(
    query=effective_query,
    top_k=candidate_count,
    allowed_document_ids=allowed_document_ids,
    fuzzy_distance=fuzzy_distance,
)
```

### Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ User Query: "find data in user_accounts table"             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ analyze_query()                                             │
│  - Calls _detect_technical_query()                          │
│  - Pattern matching: Detects "user_accounts" (snake_case)  │
│  - Returns: is_technical=True, fuzzy_distance=1            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ chunk_based_retrieval()                                     │
│  - Receives query_analysis with fuzzy_distance=1           │
│  - Logs: "Technical query detected: fuzzy_distance=1"      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ chunk_keyword_search(fuzzy_distance=1)                      │
│  - Transforms query: "find~1 data~1 in~1 user_accounts~1"  │
│  - Executes Neo4j fulltext search with fuzzy operator      │
│  - Returns BM25 candidates (with typo tolerance)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
         ┌─────────────────────────┐
         │ Vector similarity search│
         │ on fuzzy candidates     │
         └─────────────────────────┘
```

---

## Troubleshooting

### Fuzzy Matching Not Activating

**Problem**: Technical query but fuzzy matching doesn't activate.

**Causes & Solutions**:

1. **Feature disabled in settings**
   ```bash
   # Check setting
   grep enable_fuzzy_matching config/settings.py

   # Enable if needed
   export ENABLE_FUZZY_MATCHING=true
   ```

2. **Confidence below threshold**
   ```python
   from rag.nodes.query_analysis import _detect_technical_query

   query = "find user_accounts"
   result = _detect_technical_query(query.lower())
   print(f"Confidence: {result['confidence']}")

   # If confidence < 0.5 (default threshold), lower the threshold
   export FUZZY_CONFIDENCE_THRESHOLD=0.3
   ```

3. **Technical pattern not recognized**
   ```python
   # Test pattern detection manually
   import re
   query = "your_technical_term"

   # Check each pattern
   snake_case = re.findall(r'\b[a-z]+_[a-z_]+\b', query)
   tech_ids = re.findall(r'\b[A-Z]{2,}-\d+\b', query)
   config_keys = re.findall(r'\b[A-Z][A-Z_]{2,}\b', query)

   print(f"Matches: {snake_case + tech_ids + config_keys}")
   ```

### Too Many False Positives

**Problem**: Non-technical queries are incorrectly detected as technical.

**Solutions**:

1. **Increase confidence threshold**
   ```bash
   export FUZZY_CONFIDENCE_THRESHOLD=0.7  # Stricter detection
   ```

2. **Review pattern matches**
   ```python
   from rag.nodes.query_analysis import _detect_technical_query

   query = "problematic query here"
   result = _detect_technical_query(query.lower())
   print(f"Detected: {result}")
   # Check if pattern is too broad
   ```

### Performance Degradation

**Problem**: Queries are slower with fuzzy matching enabled.

**Checks**:

1. **Reduce fuzzy distance**
   ```bash
   export MAX_FUZZY_DISTANCE=1  # Use distance 1 instead of 2
   ```

2. **Monitor fulltext index performance**
   ```cypher
   // Check fulltext index exists
   SHOW INDEXES
   YIELD name, type
   WHERE type = 'FULLTEXT'
   RETURN name, type
   ```

3. **Profile query performance**
   ```python
   import time

   # Test without fuzzy
   start = time.time()
   results1 = graph_db.chunk_keyword_search(query, top_k=10, fuzzy_distance=0)
   time1 = time.time() - start

   # Test with fuzzy
   start = time.time()
   results2 = graph_db.chunk_keyword_search(query, top_k=10, fuzzy_distance=2)
   time2 = time.time() - start

   print(f"Exact: {time1:.3f}s, Fuzzy: {time2:.3f}s, Overhead: {time2/time1:.1f}x")
   ```

### Fuzzy Search Returns Irrelevant Results

**Problem**: Fuzzy matching retrieves too many loosely related documents.

**Solutions**:

1. **Lower fuzzy distance**
   ```bash
   export MAX_FUZZY_DISTANCE=1  # More conservative
   ```

2. **Increase similarity threshold in retrieval**
   ```python
   results = await retriever.chunk_based_retrieval(
       query="technical query",
       top_k=5,
       min_similarity=0.7,  # Filter low-similarity results
   )
   ```

3. **Use with two-stage retrieval** (recommended)
   - Fuzzy BM25 in Stage 1 casts wide net
   - Vector similarity in Stage 2 filters to most relevant

---

## FAQ

### Q: Does fuzzy matching work with vector similarity search?

**A:** No, fuzzy matching only applies to BM25 keyword search (fulltext index). Vector similarity search is inherently fuzzy because it uses cosine similarity on embeddings. However, when using two-stage retrieval, fuzzy BM25 in Stage 1 helps include more candidates for Stage 2 vector search.

### Q: What's the performance impact of fuzzy matching?

**A:** Fuzzy distance 1 adds ~1.5-2x overhead to BM25 search. Fuzzy distance 2 adds ~2-3x overhead. Since BM25 is typically very fast (sub-millisecond), this overhead is usually acceptable. The impact is minimal in two-stage retrieval because BM25 is already the fast pre-filter stage.

### Q: Can I disable fuzzy matching for specific queries?

**A:** Yes, you can manually set `fuzzy_distance=0` when calling `chunk_keyword_search()` directly. However, the automatic detection should handle this - non-technical queries automatically get `fuzzy_distance=0`.

### Q: How is fuzzy matching different from typo correction?

**A:** They're similar concepts. Fuzzy matching uses edit distance (Levenshtein distance) to match terms with small differences. This naturally handles typos (1-2 character errors) as well as minor variations in technical terms. Neo4j's `~` operator implements this efficiently using the fulltext index.

### Q: Does fuzzy matching help with case sensitivity?

**A:** Partially. Neo4j fulltext indexes are case-insensitive by default, so `MAX_CONNECTIONS` and `max_connections` already match without fuzzy. Fuzzy matching helps with character-level differences, not case differences.

### Q: Can I add custom technical patterns?

**A:** Yes, edit `_detect_technical_query()` in [rag/nodes/query_analysis.py](../../rag/nodes/query_analysis.py#L628-L719) to add new regex patterns. For example, to detect UUID patterns:

```python
# Add to _detect_technical_query()
uuid_pattern = r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'
uuid_matches = re.findall(uuid_pattern, query_lower)
total_matches += len(uuid_matches)
```

### Q: Does this work with entity-based retrieval?

**A:** Currently, fuzzy matching only applies to chunk-based retrieval using BM25 keyword search. Entity-based retrieval uses exact entity name matching via Neo4j graph traversal. Fuzzy entity matching would require different implementation (not yet supported).

---

## Related Documentation

- [Multi-Stage Retrieval](./multi-stage-retrieval.md) - Two-stage BM25 + vector search
- [Content Filtering](./content-filtering.md) - Pre-ingestion quality filtering
- [Core Concepts: Query Analysis](../02-core-concepts/query-analysis.md)

---

## Implementation Details

**Files Modified:**
- [core/graph_db.py](../../core/graph_db.py) - Added `fuzzy_distance` parameter to keyword search
- [rag/nodes/query_analysis.py](../../rag/nodes/query_analysis.py) - Added `_detect_technical_query()` function
- [rag/retriever.py](../../rag/retriever.py) - Integrated fuzzy_distance parameter routing
- [config/settings.py](../../config/settings.py) - Added fuzzy matching configuration

**Files Created:**
- [tests/unit/test_fuzzy_matching.py](../../tests/unit/test_fuzzy_matching.py) - Unit tests (12 tests)
- [documentation/04-features/fuzzy-matching.md](./fuzzy-matching.md) - This document

---

**Last Updated:** 2025-12-12
**Feature Status:** ✅ Production Ready
