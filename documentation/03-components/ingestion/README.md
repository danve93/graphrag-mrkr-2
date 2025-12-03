# Ingestion Components

Document processing pipeline for multi-format ingestion and enrichment.

## Contents

- [README](03-components/ingestion/README.md) - Ingestion overview
- [Document Processor](03-components/ingestion/document-processor.md) - Main processing coordinator
- [Loaders](03-components/ingestion/loaders.md) - Format-specific document loaders
- [Chunking](03-components/ingestion/chunking.md) - Text segmentation strategy
- [Marker Integration](03-components/ingestion/marker-integration.md) - High-accuracy PDF conversion
- [Quality Scoring](03-components/ingestion/quality-scoring.md) - Chunk quality assessment

## Overview

The ingestion pipeline transforms uploaded documents into searchable knowledge:

```
Upload → Convert → Chunk → Embed → Extract Entities → Persist to Neo4j
```

## Architecture

### Document Processor
**File**: `ingestion/document_processor.py`

Central coordinator managing:
- Document format detection
- Loader selection and invocation
- Chunking with configurable overlap
- Embedding generation (async with concurrency control)
- Optional entity extraction
- Quality scoring and filtering
- Batch persistence to Neo4j
- Progress tracking for UI

### Loaders
**Directory**: `ingestion/loaders/`

Format-specific loaders:
- `pdf_loader.py` - PyPDF extraction with optional Marker integration
- `docx_loader.py` - Microsoft Word documents
- `txt_loader.py` - Plain text files
- `md_loader.py` - Markdown files
- `pptx_loader.py` - PowerPoint presentations
- `xlsx_loader.py` - Excel spreadsheets
- `csv_loader.py` - CSV files
- `image_loader.py` - OCR for images

Each loader returns normalized text with metadata.

### Chunking
**File**: `core/chunking.py`

Recursive text splitter:
- Configurable chunk size (default 1200 tokens)
- Configurable overlap (default 150 tokens)
- Preserves paragraph boundaries
- Tracks chunk index and offset
- Metadata includes source document and position

### Marker Integration
**File**: `ingestion/converters.py`

High-accuracy PDF conversion:
- LLM-assisted extraction mode
- Optional forced OCR
- Table merging
- Math formula recognition
- Configurable device (CPU/CUDA) and dtype
- In-process, CLI, or server invocation modes

### Quality Scoring
**File**: `core/quality_scorer.py`

Heuristic quality assessment:
- Text length analysis
- Special character ratio
- Language detection
- Whitespace patterns
- Scoring threshold for filtering

## Configuration

Key settings:

```bash
CHUNK_SIZE=1200
CHUNK_OVERLAP=150
ENABLE_ENTITY_EXTRACTION=true
USE_MARKER_FOR_PDF=true
MARKER_USE_LLM=true
MARKER_FORCE_OCR=false
ENABLE_QUALITY_SCORING=true
```

## Usage

### Programmatic Ingestion

```python
from ingestion.document_processor import DocumentProcessor

processor = DocumentProcessor()
result = await processor.process_document_async(
    file_path="/path/to/document.pdf",
    filename="document.pdf"
)

print(f"Chunks created: {result['chunks_created']}")
print(f"Entities extracted: {result['entities_created']}")
```

### CLI Ingestion

```bash
python scripts/ingest_documents.py --file /path/to/document.pdf
python scripts/ingest_documents.py --input-dir /path/to/docs --recursive
```

## Processing Stages

### 1. Format Detection and Loading
- Detect MIME type from file extension
- Select appropriate loader
- Extract text and metadata
- Handle conversion errors

### 2. Text Normalization
- Whitespace normalization
- Encoding fixes
- Optional OCR for images/scanned PDFs

### 3. Chunking
- Split text into overlapping segments
- Preserve semantic boundaries
- Track chunk positions

### 4. Embedding Generation
- Async batch processing
- Cache by text + model
- Retry on API errors
- Rate limiting

### 5. Entity Extraction (Optional)
- LLM-based structured extraction
- Optional gleaning (multi-pass)
- NetworkX in-memory deduplication
- Batch persistence

### 6. Quality Scoring (Optional)
- Heuristic quality assessment
- Filter low-quality chunks
- Mark borderline chunks

### 7. Graph Persistence
- Create Document node
- Create Chunk nodes with embeddings
- Create Entity nodes
- Create relationships (CONTAINS, MENTIONS)
- Batch UNWIND for performance

## Performance

### Optimization Strategies

1. **Async embedding**: Concurrent API calls with semaphore
2. **Batch persistence**: Single UNWIND transaction per document
3. **Caching**: Embedding cache reduces redundant API calls
4. **NetworkX deduplication**: In-memory entity merging before DB writes
5. **Marker integration**: Optional for PDFs needing high accuracy

### Benchmarks

- **Small document** (10 pages): 10-20 seconds
- **Medium document** (100 pages): 1-3 minutes
- **Large document** (1000 pages): 10-20 minutes

Times vary based on:
- Entity extraction enabled/disabled
- Gleaning passes
- Marker LLM mode
- API rate limits

## Error Handling

The processor handles:
- Unsupported formats (returns error)
- Extraction failures (falls back to basic extraction)
- API errors (retries with exponential backoff)
- Quality threshold violations (filters or marks chunks)
- Neo4j connection errors (raises exception)

## Testing

```bash
pytest tests/integration/test_full_ingestion_pipeline.py -v
```

## Related Documentation

- [Document Processor API](03-components/ingestion/document-processor.md)
- [Marker Integration](03-components/ingestion/marker-integration.md)
- [Entity Extraction](03-components/backend/entity-extraction.md)
- [Ingestion Flow](05-data-flows/document-ingestion-flow.md)
- [Configuration](07-configuration/environment-variables.md)
