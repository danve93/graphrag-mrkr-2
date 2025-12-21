# Chunking Strategy Component

Semantic text chunking with configurable size, overlap, and provenance tracking.
Includes token-aware chunkers for structured HTML and Docling outputs.

## Overview

The chunking component segments document text into overlapping chunks suitable for embedding and retrieval. It uses:
- **HTML heading chunker** for Zendesk/Confluence-style pages (token-aware, section-preserving)
- **Docling Hybrid chunker** for structured Docling output (token-aware)
- **RecursiveCharacterTextSplitter** for legacy character-based chunking

**Location**: `core/chunking.py`
**Strategy**: HTML heading, Docling Hybrid, or RecursiveCharacterTextSplitter
**Default Char Size**: 1200 characters
**Default Char Overlap**: 150 characters
**Default Token Budget**: target 800, min 180, max 1000, overlap 100

## Architecture

```
┌──────────────────────────────────────────────────┐
│           Chunking Pipeline                       │
├──────────────────────────────────────────────────┤
│                                                   │
│  Input: Document Text                            │
│  │                                                │
│  ├─→ Separator Hierarchy                         │
│  │   ┌──────────────────────────────────────┐    │
│  │   │ 1. Double newline (paragraphs)      │    │
│  │   │ 2. Single newline                    │    │
│  │   │ 3. Spaces                            │    │
│  │   │ 4. Character-level fallback          │    │
│  │   └──────────────────────────────────────┘    │
│  │                                                │
│  ├─→ Size & Overlap                              │
│  │   ┌──────────────────────────────────────┐    │
│  │   │ chunk_size: Target characters        │    │
│  │   │ chunk_overlap: Context continuity    │    │
│  │   │ length_function: len() by default    │    │
│  │   └──────────────────────────────────────┘    │
│  │                                                │
│  └─→ Output: List of Chunks                      │
│      ┌──────────────────────────────────────┐    │
│      │ Each chunk:                          │    │
│      │   • Text content                     │    │
│      │   • Position metadata                │    │
│      │   • Overlap with neighbors           │    │
│      └──────────────────────────────────────┘    │
│                                                   │
└──────────────────────────────────────────────────┘
```

## Strategy Selection

Chunking strategy is selected by file type and settings:

- HTML (`.html`, `.htm`, `.xhtml`): `chunker_strategy_html` (default `html_heading`)
- PDF (`.pdf`): `chunker_strategy_pdf` (default `docling_hybrid`)
- All other types: legacy character-based chunker

Note: Docling Hybrid chunking requires Docling conversion (`DOCUMENT_CONVERSION_PROVIDER=docling`)
so the `docling_document` payload is available. Otherwise the pipeline falls back to legacy chunking.

## Core Implementation

### RecursiveCharacterTextSplitter

```python
# core/chunking.py
from core.chunking import document_chunker

chunks = document_chunker.chunk_text(
    text=content,
    document_id=doc_id,
    source_metadata={"file_extension": ".html"},
    docling_document=docling_document,
)
```

### Configuration

```bash
# Environment variables
CHUNK_SIZE=1200
CHUNK_OVERLAP=150
CHUNK_TARGET_TOKENS=800
CHUNK_MIN_TOKENS=180
CHUNK_MAX_TOKENS=1000
CHUNK_OVERLAP_TOKENS=100

# Alternative configurations:

# Small chunks (precise retrieval, more chunks)
CHUNK_SIZE=400
CHUNK_OVERLAP=100

# Large chunks (more context, fewer chunks)
CHUNK_SIZE=1200
CHUNK_OVERLAP=300

# No overlap (faster, less context continuity)
CHUNK_SIZE=800
CHUNK_OVERLAP=0
```

## Chunking Strategies

### Paragraph-Aware Chunking

```python
def chunk_by_paragraphs(
    text: str,
    min_chunk_size: int = 400,
    max_chunk_size: int = 1200
) -> List[str]:
    """
    Chunk text preserving paragraph boundaries.
    
    Groups paragraphs until max_chunk_size is reached.
    Ensures chunks are at least min_chunk_size.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        
        if current_size + para_size > max_chunk_size and current_chunk:
            # Flush current chunk
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(para)
        current_size += para_size
    
    # Flush remaining
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    # Merge small chunks
    merged = []
    for chunk in chunks:
        if len(chunk) < min_chunk_size and merged:
            merged[-1] += "\n\n" + chunk
        else:
            merged.append(chunk)
    
    return merged
```

### Sentence-Aware Chunking

```python
import re

def chunk_by_sentences(
    text: str,
    target_chunk_size: int = 800,
    overlap_sentences: int = 2
) -> List[str]:
    """
    Chunk text by sentences with sentence-level overlap.
    
    Args:
        text: Input text
        target_chunk_size: Target size in characters
        overlap_sentences: Number of sentences to overlap
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > target_chunk_size and current_chunk:
            # Create chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Keep last N sentences for overlap
            current_chunk = current_chunk[-overlap_sentences:]
            current_size = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Flush remaining
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

### Token-Aware Chunking

```python
import tiktoken

def chunk_by_tokens(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Chunk text by token count (useful for LLM context limits).
    
    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap in tokens
        encoding_name: Tokenizer encoding (cl100k_base for GPT-4)
    """
    encoder = tiktoken.get_encoding(encoding_name)
    tokens = encoder.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        
        chunks.append(chunk_text)
        
        # Move start forward with overlap
        start = end - overlap_tokens
    
    return chunks
```

## Metadata Enrichment

### Add Position Metadata

```python
from core.chunking import document_chunker

def chunk_with_metadata(
    text: str,
    document_id: str
) -> List[Dict]:
    """
    Chunk text and add position metadata.
    
    Returns:
        List of chunk dicts with metadata:
            - id: Chunk identifier
            - text: Chunk content
            - document_id: Parent document
            - chunk_index: Position in sequence
            - start_char: Start position in document
            - end_char: End position in document
            - word_count: Number of words
    """
    chunks = document_chunker.chunk_text(text, document_id)
    chunk_objects = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}
        chunk_text = chunk.get("content", "")
        chunk_obj = {
            "id": chunk.get("chunk_id"),
            "text": chunk_text,
            "document_id": document_id,
            "chunk_index": chunk.get("chunk_index"),
            "start_char": metadata.get("start_offset"),
            "end_char": metadata.get("end_offset"),
            "word_count": len(chunk_text.split())
        }
        
        chunk_objects.append(chunk_obj)
    
    return chunk_objects
```

### Add Page Numbers (PDF)

```python
def add_page_numbers(
    chunks: List[Dict],
    page_breaks: List[int]
) -> List[Dict]:
    """
    Add page numbers to chunks based on character positions.
    
    Args:
        chunks: List of chunk dicts with start_char/end_char
        page_breaks: List of character positions where pages break
    
    Returns:
        Chunks with page_number field added
    """
    for chunk in chunks:
        start_char = chunk["start_char"]
        
        # Find which page this chunk starts on
        page_number = 1
        for page_break in sorted(page_breaks):
            if start_char >= page_break:
                page_number += 1
            else:
                break
        
        chunk["page_number"] = page_number
    
    return chunks
```

## Chunking Quality Analysis

### Analyze Chunk Distribution

```python
def analyze_chunks(chunks: List[str]) -> Dict:
    """
    Analyze chunk statistics.
    
    Returns:
        Statistics dict with:
            - count: Number of chunks
            - avg_length: Average chunk length
            - min_length: Shortest chunk
            - max_length: Longest chunk
            - total_chars: Total characters
    """
    if not chunks:
        return {
            "count": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "total_chars": 0
        }
    
    lengths = [len(chunk) for chunk in chunks]
    
    return {
        "count": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_chars": sum(lengths),
        "length_distribution": {
            "p25": sorted(lengths)[len(lengths) // 4],
            "p50": sorted(lengths)[len(lengths) // 2],
            "p75": sorted(lengths)[3 * len(lengths) // 4]
        }
    }
```

### Validate Chunk Quality

```python
def validate_chunks(
    chunks: List[str],
    min_length: int = 100,
    max_length: int = 2000
) -> tuple[List[str], List[str]]:
    """
    Validate chunks and separate good from problematic.
    
    Returns:
        Tuple of (valid_chunks, invalid_chunks)
    """
    valid = []
    invalid = []
    
    for chunk in chunks:
        length = len(chunk)
        
        if length < min_length:
            invalid.append(chunk)
        elif length > max_length:
            invalid.append(chunk)
        elif not chunk.strip():
            invalid.append(chunk)
        else:
            valid.append(chunk)
    
    return valid, invalid
```

## Overlap Visualization

### Compute Overlap Regions

```python
def compute_overlap(
    chunks: List[Dict]
) -> List[Dict]:
    """
    Identify overlapping regions between consecutive chunks.
    
    Args:
        chunks: List of chunk dicts with start_char/end_char
    
    Returns:
        List of overlap dicts:
            - chunk1_idx: First chunk index
            - chunk2_idx: Second chunk index
            - overlap_chars: Number of overlapping characters
            - overlap_text: Overlapping text
    """
    overlaps = []
    
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]
        
        # Calculate overlap
        overlap_start = max(chunk1["start_char"], chunk2["start_char"])
        overlap_end = min(chunk1["end_char"], chunk2["end_char"])
        
        if overlap_end > overlap_start:
            overlap_size = overlap_end - overlap_start
            
            # Extract overlap text
            overlap_text = chunk1["text"][-(overlap_size):]
            
            overlaps.append({
                "chunk1_idx": i,
                "chunk2_idx": i + 1,
                "overlap_chars": overlap_size,
                "overlap_text": overlap_text
            })
    
    return overlaps
```

## Usage Examples

### Basic Chunking

```python
from core.chunking import document_chunker

text = """
This is the first paragraph with important information.

This is the second paragraph with more details.

This is the third paragraph concluding the document.
"""

chunks = document_chunker.chunk_text(text, "demo_doc")

for i, chunk in enumerate(chunks):
    content = chunk["content"]
    print(f"Chunk {i}: {len(content)} chars")
    print(content[:50] + "...")
    print()
```

### Document Processing

```python
from core.chunking import document_chunker

def process_document(document_text: str, document_id: str):
    """Process document into chunks with metadata."""
    chunks = document_chunker.chunk_text(document_text, document_id)
    
    print(f"Created {len(chunks)} chunks")
    
    # Analyze
    stats = analyze_chunks([c["content"] for c in chunks])
    print(f"Average chunk size: {stats['avg_length']:.0f} chars")
    
    return chunks
```

## Performance Considerations

### Chunking Speed

```python
import time
from core.chunking import document_chunker

def benchmark_chunking(text: str, iterations: int = 100):
    """Benchmark chunking performance."""
    start = time.time()
    
    for _ in range(iterations):
        chunks = document_chunker.chunk_text(text, "bench_doc")
    
    elapsed = time.time() - start
    avg_time = elapsed / iterations
    
    print(f"Average chunking time: {avg_time * 1000:.2f}ms")
    print(f"Throughput: {len(text) / avg_time / 1000:.0f}K chars/sec")
```

**Typical Performance**:
- Small documents (<10KB): <5ms
- Medium documents (10-100KB): 5-50ms
- Large documents (>100KB): 50-500ms

### Memory Usage

```python
def estimate_memory_usage(num_chunks: int, avg_chunk_size: int) -> int:
    """
    Estimate memory usage for chunks.
    
    Returns:
        Estimated bytes
    """
    # String overhead + chunk text + metadata
    per_chunk_overhead = 100  # Dict overhead
    per_chunk_text = avg_chunk_size * 4  # Unicode (4 bytes/char worst case)
    
    total = num_chunks * (per_chunk_overhead + per_chunk_text)
    
    return total
```

## Testing

### Unit Tests

```python
import pytest
from config.settings import settings
from core.chunking import document_chunker

def test_basic_chunking():
    text = "A" * 1000
    settings.chunk_size = 200
    settings.chunk_overlap = 50
    chunks = document_chunker.chunk_text(text, "doc-basic")
    
    assert len(chunks) > 1
    assert all(len(chunk["content"]) <= 250 for chunk in chunks)  # Allow some flexibility

def test_chunk_overlap():
    text = "Sentence one. Sentence two. Sentence three."
    settings.chunk_size = 20
    settings.chunk_overlap = 10
    chunks = document_chunker.chunk_text(text, "doc-overlap")
    
    # Verify overlap exists
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        # Check if end of chunk N overlaps with start of chunk N+1
        overlap_exists = chunks[i]["content"][-10:] in chunks[i + 1]["content"][:20]
        assert overlap_exists

def test_chunk_metadata():
    text = "Test document content."
    settings.chunk_size = 10
    settings.chunk_overlap = 2
    chunks = document_chunker.chunk_text(text, "doc123")
    
    assert all("chunk_id" in chunk for chunk in chunks)
    assert all("chunk_index" in chunk for chunk in chunks)
    assert all(chunk["document_id"] == "doc123" for chunk in chunks)
```

## Related Documentation

- [Document Processor](03-components/ingestion/document-processor.md)
- [Embeddings](03-components/backend/embeddings.md)
- [Retrieval Strategies](02-core-concepts/retrieval-strategies.md)
- [Configuration Reference](07-configuration/rag-tuning.md)
