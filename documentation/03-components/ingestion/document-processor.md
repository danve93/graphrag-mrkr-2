# Document Processor Component

Multi-format document ingestion pipeline with chunking, embedding, and entity extraction.

## Overview

The document processor orchestrates the complete ingestion workflow: loading documents from multiple formats, converting to text/Markdown, chunking with overlap, generating embeddings, optionally extracting entities, and persisting to Neo4j. It supports synchronous and asynchronous execution with progress tracking.

**Location**: `ingestion/document_processor.py`
**Formats**: PDF, DOCX, TXT, MD, PPTX, XLSX, CSV, images
**Modes**: Full ingestion, chunks-only, entity extraction (sync/async)

## Architecture

```
┌──────────────────────────────────────────────────┐
│         Document Processing Pipeline              │
├──────────────────────────────────────────────────┤
│                                                   │
│  Stage 1: Load & Convert                         │
│  ┌─────────────────────────────────────────────┐ │
│  │  File → Format Detection → Loader Selection│ │
│  │  → Text/Markdown Extraction                 │ │
│  │  → Metadata Extraction (pages, words)       │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Stage 2: Preprocessing                          │
│  ┌─────────────────────────────────────────────┐ │
│  │  Normalization → OCR (if needed)            │ │
│  │  → Quality Assessment                        │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Stage 3: Chunking                               │
│  ┌─────────────────────────────────────────────┐ │
│  │  Text → Semantic Chunks (RecursiveCharacter)│ │
│  │  → Overlap → Provenance (page, position)    │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Stage 4: Embedding Generation                   │
│  ┌─────────────────────────────────────────────┐ │
│  │  Chunks → Batch Embedding API               │ │
│  │  → Concurrency Control → Caching            │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Stage 5: Entity Extraction (Optional)           │
│  ┌─────────────────────────────────────────────┐ │
│  │  Chunks → LLM Extraction → Deduplication    │ │
│  │  → Entity Embeddings → Relationships        │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Stage 6: Persistence                            │
│  ┌─────────────────────────────────────────────┐ │
│  │  Document Node → Chunk Nodes                │ │
│  │  → Entity Nodes → Relationships → Stats     │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
└──────────────────────────────────────────────────┘
```

## DocumentProcessor Class

### Initialization

```python
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from config.settings import settings
from core.graph_db import get_db
from core.embeddings import EmbeddingManager
from core.chunking import chunk_text
from ingestion.loaders import get_loader

class DocumentProcessor:
    """
    Orchestrates document ingestion pipeline.
    
    Responsibilities:
        - Format detection and loading
        - Text preprocessing and chunking
        - Embedding generation
        - Entity extraction (optional)
        - Neo4j persistence
        - Progress tracking
    """
    
    def __init__(self):
        self.db = get_db()
        self.embedding_manager = EmbeddingManager()
        self.extraction_tasks = {}  # Track background entity extraction
    
    async def close(self):
        """Cleanup resources."""
        await self.embedding_manager.close()
```

### Configuration

```bash
# Chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=200

# Entity extraction
ENABLE_ENTITY_EXTRACTION=true
SYNC_ENTITY_EMBEDDINGS=false  # true = block, false = background

# Quality filtering
ENABLE_QUALITY_SCORING=true
MIN_QUALITY_SCORE=0.3

# OCR
ENABLE_OCR=true
OCR_QUALITY_THRESHOLD=0.5
```

## Document Loading

### Load Document

```python
def load_document(self, file_path: str) -> Dict:
    """
    Load document and extract metadata.
    
    Args:
        file_path: Path to document file
    
    Returns:
        Document dict with metadata and text
    """
    # Verify file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get loader for file type
    loader = get_loader(file_path)
    
    # Load content
    text, metadata = loader.load(file_path)
    
    # Generate document ID
    document_id = self._generate_document_id(file_path)
    
    # Build document object
    document = {
        "id": document_id,
        "filename": os.path.basename(file_path),
        "file_path": file_path,
        "file_type": metadata.get("file_type", "unknown"),
        "file_size": os.path.getsize(file_path),
        "title": metadata.get("title", os.path.basename(file_path)),
        "text": text,
        "page_count": metadata.get("page_count", 0),
        "word_count": len(text.split()),
        "created_at": datetime.utcnow()
    }
    
    return document

def _generate_document_id(self, file_path: str) -> str:
    """Generate deterministic document ID from file path."""
    content = f"{file_path}::{os.path.getmtime(file_path)}"
    return hashlib.md5(content.encode()).hexdigest()
```

### Format Detection

```python
# ingestion/loaders/__init__.py
from pathlib import Path
from .pdf_loader import PDFLoader
from .docx_loader import DOCXLoader
from .text_loader import TextLoader
from .markdown_loader import MarkdownLoader
from .pptx_loader import PPTXLoader
from .excel_loader import ExcelLoader
from .csv_loader import CSVLoader
from .image_loader import ImageLoader

LOADERS = {
    ".pdf": PDFLoader,
    ".docx": DOCXLoader,
    ".doc": DOCXLoader,
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".pptx": PPTXLoader,
    ".ppt": PPTXLoader,
    ".xlsx": ExcelLoader,
    ".xls": ExcelLoader,
    ".csv": CSVLoader,
    ".png": ImageLoader,
    ".jpg": ImageLoader,
    ".jpeg": ImageLoader,
}

def get_loader(file_path: str):
    """Get appropriate loader for file type."""
    extension = Path(file_path).suffix.lower()
    
    loader_class = LOADERS.get(extension)
    if not loader_class:
        raise ValueError(f"Unsupported file type: {extension}")
    
    return loader_class()
```

## Text Preprocessing

### Normalization

```python
import re

def preprocess_text(self, text: str) -> str:
    """
    Normalize text content.
    
    Operations:
        - Remove excessive whitespace
        - Normalize line breaks
        - Remove control characters
        - Preserve paragraph structure
    """
    # Remove control characters (except newlines, tabs)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs → single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines → double newline
    
    # Strip whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()
```

### OCR Integration (Optional)

```python
from core.ocr import apply_ocr_if_needed

def apply_ocr(self, text: str, metadata: Dict) -> str:
    """Apply OCR to image-based documents if needed."""
    if not settings.enable_ocr:
        return text
    
    # Check if OCR is needed (low quality extraction)
    quality_score = metadata.get("extraction_quality", 1.0)
    
    if quality_score < settings.ocr_quality_threshold:
        logger.info(f"Applying OCR (quality: {quality_score})")
        text = apply_ocr_if_needed(text, metadata)
    
    return text
```

## Chunking

### Create Chunks

```python
def create_chunks(self, document: Dict) -> List[Dict]:
    """
    Chunk document text with provenance.
    
    Args:
        document: Document dict with 'text' and 'id'
    
    Returns:
        List of chunk dicts with metadata
    """
    from core.chunking import chunk_text
    
    text = document["text"]
    
    # Chunk text
    chunks = chunk_text(
        text=text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    # Add metadata to each chunk
    chunk_objects = []
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{document['id']}_chunk_{i}"
        
        chunk_obj = {
            "id": chunk_id,
            "text": chunk_text,
            "document_id": document["id"],
            "chunk_index": i,
            "start_char": 0,  # Calculate from chunker
            "end_char": len(chunk_text),
            "page_number": None,  # Extract from metadata if available
            "word_count": len(chunk_text.split()),
            "quality_score": 1.0  # Will be updated by quality scorer
        }
        
        chunk_objects.append(chunk_obj)
    
    logger.info(f"Created {len(chunk_objects)} chunks for {document['filename']}")
    
    return chunk_objects
```

### Quality Scoring (Optional)

```python
from core.quality_scorer import score_chunk_quality

def score_chunks(self, chunks: List[Dict]) -> List[Dict]:
    """Add quality scores to chunks."""
    if not settings.enable_quality_scoring:
        return chunks
    
    for chunk in chunks:
        chunk["quality_score"] = score_chunk_quality(chunk["text"])
    
    return chunks

def filter_low_quality_chunks(
    self,
    chunks: List[Dict],
    min_score: float = 0.3
) -> List[Dict]:
    """Filter out low-quality chunks."""
    filtered = [
        chunk for chunk in chunks
        if chunk["quality_score"] >= min_score
    ]
    
    removed = len(chunks) - len(filtered)
    if removed > 0:
        logger.info(f"Filtered {removed} low-quality chunks")
    
    return filtered
```

## Embedding Generation

### Generate Embeddings

```python
async def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
    """
    Generate embeddings for chunks.
    
    Args:
        chunks: List of chunk dicts with 'text'
    
    Returns:
        Chunks with 'embedding' field added
    """
    texts = [chunk["text"] for chunk in chunks]
    
    embeddings = await self.embedding_manager.get_embeddings_batch(
        texts=texts,
        show_progress=True
    )
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    
    return chunks
```

## Entity Extraction

### Extract Entities (Synchronous)

```python
from core.entity_extraction import extract_entities_batch, add_entity_embeddings

async def extract_entities_sync(
    self,
    chunks: List[Dict]
) -> tuple[List[Dict], List[Dict]]:
    """
    Extract entities synchronously (blocks until complete).
    
    Returns:
        Tuple of (entities, relationships)
    """
    if not settings.enable_entity_extraction:
        return [], []
    
    # Extract entities
    entities, relationships = await extract_entities_batch(
        chunks=chunks,
        show_progress=True
    )
    
    # Generate entity embeddings
    if settings.sync_entity_embeddings:
        entities = await add_entity_embeddings(entities)
    
    return entities, relationships
```

### Extract Entities (Asynchronous)

```python
import asyncio
from api.job_manager import get_job_manager, JobStatus

async def extract_entities_async(
    self,
    document_id: str,
    chunks: List[Dict]
) -> str:
    """
    Extract entities in background task.
    
    Returns:
        Job ID for tracking progress
    """
    if not settings.enable_entity_extraction:
        return None
    
    job_manager = get_job_manager()
    
    # Create job
    job = job_manager.create_job(
        job_type="extraction",
        metadata={"document_id": document_id}
    )
    
    # Start background task
    async def extraction_task():
        try:
            job_manager.update_job(
                job.job_id,
                status=JobStatus.RUNNING,
                message="Extracting entities"
            )
            
            entities, relationships = await extract_entities_batch(chunks)
            
            job_manager.update_job(job.job_id, progress=0.6)
            
            entities = await add_entity_embeddings(entities)
            
            job_manager.update_job(job.job_id, progress=0.9)
            
            # Persist
            from core.entity_extraction import persist_entities_batch, persist_relationships_batch
            for chunk in chunks:
                chunk_entities = [e for e in entities if chunk["id"] in e["chunk_ids"]]
                persist_entities_batch(chunk_entities, chunk["id"])
            
            persist_relationships_batch(relationships)
            
            job_manager.update_job(
                job.job_id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                result={
                    "entity_count": len(entities),
                    "relationship_count": len(relationships)
                }
            )
        
        except Exception as e:
            job_manager.update_job(
                job.job_id,
                status=JobStatus.FAILED,
                error=str(e)
            )
    
    asyncio.create_task(extraction_task())
    
    return job.job_id
```

## Persistence

### Persist Document

```python
def persist_document(self, document: Dict) -> str:
    """
    Create Document node in Neo4j.
    
    Returns:
        Document ID
    """
    doc_data = {
        "id": document["id"],
        "filename": document["filename"],
        "file_path": document["file_path"],
        "file_type": document["file_type"],
        "file_size": document["file_size"],
        "title": document["title"],
        "page_count": document["page_count"],
        "word_count": document["word_count"]
    }
    
    return self.db.create_document(doc_data)
```

### Persist Chunks

```python
def persist_chunks(self, document_id: str, chunks: List[Dict]) -> int:
    """
    Create Chunk nodes and HAS_CHUNK relationships.
    
    Returns:
        Number of chunks created
    """
    return self.db.create_chunks_batch(document_id, chunks)
```

## Complete Pipeline

### Process Document (Synchronous)

```python
async def process_document(
    self,
    file_path: str,
    extract_entities: bool = True
) -> Dict:
    """
    Complete document ingestion pipeline (synchronous).
    
    Args:
        file_path: Path to document
        extract_entities: Whether to extract entities
    
    Returns:
        Processing result with document_id and statistics
    """
    logger.info(f"Processing document: {file_path}")
    
    # Stage 1: Load
    document = self.load_document(file_path)
    document["text"] = self.preprocess_text(document["text"])
    
    # Stage 2: Chunk
    chunks = self.create_chunks(document)
    chunks = self.score_chunks(chunks)
    
    # Stage 3: Embed
    chunks = await self.generate_embeddings(chunks)
    
    # Stage 4: Persist document and chunks
    self.persist_document(document)
    self.persist_chunks(document["id"], chunks)
    
    # Stage 5: Entity extraction
    entity_count = 0
    relationship_count = 0
    
    if extract_entities and settings.enable_entity_extraction:
        if settings.sync_entity_embeddings:
            # Synchronous extraction
            entities, relationships = await self.extract_entities_sync(chunks)
            
            # Persist entities
            from core.entity_extraction import persist_entities_batch, persist_relationships_batch
            for chunk in chunks:
                chunk_entities = [e for e in entities if chunk["id"] in e["chunk_ids"]]
                persist_entities_batch(chunk_entities, chunk["id"])
            
            persist_relationships_batch(relationships)
            
            entity_count = len(entities)
            relationship_count = len(relationships)
        else:
            # Asynchronous extraction
            job_id = await self.extract_entities_async(document["id"], chunks)
    
    # Update document stats
    self.db.update_document_stats(document["id"])
    
    result = {
        "document_id": document["id"],
        "filename": document["filename"],
        "chunk_count": len(chunks),
        "entity_count": entity_count,
        "relationship_count": relationship_count
    }
    
    logger.info(f"Document processed: {result}")
    
    return result
```

### Process Document (Asynchronous)

```python
async def process_document_async(self, file_path: str) -> str:
    """
    Process document with job tracking.
    
    Returns:
        Job ID for monitoring progress
    """
    from api.job_manager import get_job_manager, JobStatus
    
    manager = get_job_manager()
    
    # Create job
    job = manager.create_job(
        job_type="ingestion",
        metadata={
            "file_path": file_path,
            "filename": os.path.basename(file_path)
        }
    )
    
    # Start background task
    async def ingestion_task():
        try:
            manager.update_job(
                job.job_id,
                status=JobStatus.RUNNING,
                progress=0.0,
                message="Loading document"
            )
            
            result = await self.process_document(file_path)
            
            manager.update_job(
                job.job_id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                message="Ingestion complete",
                result=result
            )
        
        except Exception as e:
            manager.update_job(
                job.job_id,
                status=JobStatus.FAILED,
                error=str(e)
            )
    
    asyncio.create_task(ingestion_task())
    
    return job.job_id
```

## Chunk-Only Mode

### Process Without Full Ingestion

```python
async def process_chunks_only(
    self,
    document_id: str,
    chunks: List[Dict]
) -> Dict:
    """
    Process pre-chunked text (for testing or custom workflows).
    
    Args:
        document_id: Document identifier
        chunks: List of chunk dicts with 'text'
    
    Returns:
        Processing result
    """
    # Generate embeddings
    chunks = await self.generate_embeddings(chunks)
    
    # Persist
    self.persist_chunks(document_id, chunks)
    
    # Optional entity extraction
    if settings.enable_entity_extraction:
        await self.extract_entities_async(document_id, chunks)
    
    return {
        "document_id": document_id,
        "chunk_count": len(chunks)
    }
```

## Usage Examples

### Single Document

```python
from ingestion.document_processor import DocumentProcessor

async def ingest_document(file_path: str):
    """Ingest a single document."""
    processor = DocumentProcessor()
    
    try:
        result = await processor.process_document(file_path)
        print(f"Ingested: {result['filename']}")
        print(f"  Chunks: {result['chunk_count']}")
        print(f"  Entities: {result['entity_count']}")
    
    finally:
        await processor.close()
```

### Batch Ingestion

```python
import asyncio
from pathlib import Path

async def ingest_directory(directory: str):
    """Ingest all documents in a directory."""
    processor = DocumentProcessor()
    
    try:
        files = list(Path(directory).glob("**/*"))
        files = [f for f in files if f.is_file()]
        
        for file_path in files:
            try:
                result = await processor.process_document(str(file_path))
                print(f"{result['filename']}")
            except Exception as e:
                print(f"Error - {file_path}: {e}")
    
    finally:
        await processor.close()
```

### With Progress Tracking

```python
async def ingest_with_progress(file_path: str):
    """Ingest document with job tracking."""
    processor = DocumentProcessor()
    
    try:
        job_id = await processor.process_document_async(file_path)
        
        # Poll job status
        from api.job_manager import get_job_manager
        manager = get_job_manager()
        
        while True:
            job = manager.get_job(job_id)
            print(f"Progress: {job.progress:.0%} - {job.message}")
            
            if job.status in ["completed", "failed"]:
                break
            
            await asyncio.sleep(1)
        
        if job.status == "completed":
            print(f"Result: {job.result}")
        else:
            print(f"Error: {job.error}")
    
    finally:
        await processor.close()
```

## Testing

### Unit Tests

```python
import pytest
from ingestion.document_processor import DocumentProcessor

@pytest.fixture
async def processor():
    proc = DocumentProcessor()
    yield proc
    await proc.close()

@pytest.mark.asyncio
async def test_load_document(processor):
    doc = processor.load_document("test.pdf")
    
    assert doc["id"]
    assert doc["filename"] == "test.pdf"
    assert doc["text"]
    assert doc["word_count"] > 0

@pytest.mark.asyncio
async def test_create_chunks(processor):
    doc = {"id": "test123", "text": "Test content " * 100}
    chunks = processor.create_chunks(doc)
    
    assert len(chunks) > 0
    assert all(chunk["document_id"] == "test123" for chunk in chunks)

@pytest.mark.asyncio
async def test_full_pipeline(processor):
    result = await processor.process_document("test.pdf")
    
    assert result["document_id"]
    assert result["chunk_count"] > 0
```

## CLI Script

```python
# scripts/ingest_documents.py
import asyncio
import argparse
from pathlib import Path
from ingestion.document_processor import DocumentProcessor

async def main():
    parser = argparse.ArgumentParser(description="Ingest documents")
    parser.add_argument("path", help="File or directory path")
    parser.add_argument("--no-entities", action="store_true", help="Skip entity extraction")
    args = parser.parse_args()
    
    processor = DocumentProcessor()
    
    try:
        path = Path(args.path)
        
        if path.is_file():
            result = await processor.process_document(
                str(path),
                extract_entities=not args.no_entities
            )
            print(f"Ingested: {result}")
        
        elif path.is_dir():
            files = list(path.glob("**/*"))
            files = [f for f in files if f.is_file()]
            
            for file_path in files:
                try:
                    result = await processor.process_document(str(file_path))
                    print(f"{result['filename']}")
                except Exception as e:
                    print(f"Error - {file_path}: {e}")
    
    finally:
        await processor.close()

if __name__ == "__main__":
    asyncio.run(main())
```

**Usage**:
```bash
python scripts/ingest_documents.py /path/to/document.pdf
python scripts/ingest_documents.py /path/to/directory/
python scripts/ingest_documents.py file.pdf --no-entities
```

## Related Documentation

- [Document Loaders](03-components/ingestion/loaders.md)
- [Chunking Strategy](03-components/ingestion/chunking.md)
- [Entity Extraction](03-components/backend/entity-extraction.md)
- [Embeddings](03-components/backend/embeddings.md)
- [Job Management](03-components/backend/job-management.md)
