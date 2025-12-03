# Document Ingestion Flow

End-to-end trace of document upload through processing and persistence.

## Overview

This document traces a complete document ingestion from file upload through format detection, text extraction, chunking, embedding generation, entity extraction, and Neo4j persistence. It highlights background job management, progress tracking, and status updates.

## Flow Diagram

```
User Upload: "VxRail_Admin_Guide.pdf" (2.3 MB)
│
├─> 1. Frontend: DocumentUpload Component
│   ├─ User drags file into dropzone
│   ├─ Validate format (.pdf) and size (< 50MB)
│   ├─ Generate upload ID
│   ├─ POST /api/upload (multipart/form-data)
│   └─ Track XHR upload progress (0-50%)
│
├─> 2. API: Upload Endpoint
│   ├─ Receive file bytes
│   ├─ Validate extension and size
│   ├─ Generate MD5 hash for deduplication
│   ├─ Save to staging: data/staged_uploads/{hash}_{filename}
│   ├─ Create background job (job_id)
│   └─ Return {"job_id": "...", "status": "processing"}
│
├─> 3. Background Worker: Document Processing
│   │
│   ├─> 3a. Load Document
│   │   ├─ Detect format from extension
│   │   ├─ Select loader: PDFLoader
│   │   ├─ Extract text with pypdf
│   │   ├─ Extract metadata (title, author, page_count)
│   │   └─ Update job: progress=10%, message="Extracted text"
│   │
│   ├─> 3b. Document Normalization
│   │   ├─ Clean whitespace and special characters
│   │   ├─ Normalize line breaks
│   │   ├─ Remove headers/footers (heuristic)
│   │   └─ Update job: progress=20%
│   │
│   ├─> 3c. Chunking
│   │   ├─ RecursiveCharacterTextSplitter
│   │   ├─ chunk_size=1000, overlap=200
│   │   ├─ Split by paragraphs, then sentences
│   │   ├─ Generate 147 chunks
│   │   ├─ Add metadata: page_number, position
│   │   └─ Update job: progress=30%, message="Created 147 chunks"
│   │
│   ├─> 3d. Embedding Generation
│   │   ├─ EmbeddingManager.generate_embeddings_batch()
│   │   ├─ Batch size: 20 chunks per request
│   │   ├─ Concurrent requests: 5 (configurable)
│   │   ├─ Check embedding cache (hit: 0%, cold start)
│   │   ├─ OpenAI text-embedding-3-small API calls
│   │   ├─ Rate limiting: 100ms delay between batches
│   │   ├─ Generate 147 vectors (1536 dimensions each)
│   │   └─ Update job: progress=30-70% (incremental)
│   │
│   ├─> 3e. Entity Extraction (Optional, Async)
│   │   ├─ Check enable_entity_extraction setting
│   │   ├─ EntityExtractor.extract_from_chunks()
│   │   ├─ Sample 50 representative chunks
│   │   ├─ LLM extraction prompt per chunk
│   │   ├─ Parse entity tuples: (name, type, description)
│   │   ├─ Extract relationships: (entity1, relation, entity2)
│   │   ├─ EntityGraph accumulator:
│   │   │   ├─ Deduplicate entities by name
│   │   │   ├─ Merge descriptions
│   │   │   ├─ Sum relationship strengths
│   │   │   └─ Track provenance (source chunks)
│   │   ├─ Extract 83 unique entities
│   │   ├─ Generate entity embeddings (batch)
│   │   └─ Update job: progress=70-85%
│   │
│   ├─> 3f. Neo4j Persistence
│   │   ├─ Create Document node
│   │   │   ├─ Properties: id, filename, title, page_count, upload_date
│   │   │   └─ Label: Document
│   │   │
│   │   ├─> Batch Insert Chunks (UNWIND)
│   │   │   ├─ Create 147 Chunk nodes
│   │   │   ├─ Properties: id, content, page_number, position, embedding
│   │   │   ├─ Create CONTAINS relationships: Document → Chunk
│   │   │   └─ Create vector index entries
│   │   │
│   │   ├─> Batch Insert Entities (UNWIND)
│   │   │   ├─ MERGE 83 Entity nodes (by name)
│   │   │   ├─ Properties: name, type, description, importance, embedding
│   │   │   ├─ Create MENTIONS relationships: Chunk → Entity
│   │   │   └─ Create RELATED_TO relationships: Entity ↔ Entity
│   │   │
│   │   ├─> Compute Similarity Edges (Background)
│   │   │   ├─ Query chunks with embeddings
│   │   │   ├─ Compute cosine similarity matrix (pairwise)
│   │   │   ├─ Filter by threshold (>= 0.7)
│   │   │   ├─ Create SIMILAR_TO relationships
│   │   │   └─ Properties: similarity score
│   │   │
│   │   └─ Update job: progress=85-95%
│   │
│   └─> 3g. Finalization
│       ├─ Cleanup staged file
│       ├─ Update job: status="completed", progress=100%
│       ├─ Result: {document_id, chunk_count: 147, entity_count: 83}
│       └─ Emit completion event
│
├─> 4. Frontend: Status Polling
│   ├─ useQuery: /api/jobs/{job_id}
│   ├─ Poll interval: 1000ms
│   ├─ Update UploadFile state:
│   │   ├─ progress: 0 → 10 → 30 → 70 → 100
│   │   ├─ message: "Extracted text" → "Created chunks" → ...
│   │   └─ status: "uploading" → "processing" → "completed"
│   └─ Stop polling on completion/error
│
└─> 5. UI Update: Display Completion
    ├─ UploadFileItem shows green checkmark
    ├─ Message: "Successfully processed 147 chunks, 83 entities"
    ├─ Document appears in DocumentList
    └─ Refetch documents query
```

## Step-by-Step Trace

### Step 1: Frontend Upload

**Location**: `frontend/src/components/upload/DocumentUpload.tsx`

```typescript
const handleUpload = async (uploadFile: UploadFile) => {
  try {
    // Update UI: uploading
    setFiles((prev) =>
      prev.map((f) =>
        f.id === uploadFile.id
          ? { ...f, status: 'uploading', progress: 0 }
          : f
      )
    );

    // Upload with progress tracking
    const result = await uploadDocument(
      uploadFile.file,
      (progress) => {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === uploadFile.id
              ? { ...f, progress: Math.round(progress / 2) }  // 0-50% for upload
              : f
          )
        );
      }
    );

    // Update UI: processing
    setFiles((prev) =>
      prev.map((f) =>
        f.id === uploadFile.id
          ? {
              ...f,
              status: 'processing',
              jobId: result.job_id,
              progress: 50,
            }
          : f
      )
    );

    // Poll job status
    await pollJobStatus(result.job_id, (status) => {
      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id
            ? {
                ...f,
                progress: 50 + Math.round(status.progress / 2),  // 50-100%
                message: status.message,
              }
            : f
        )
      );
    });

  } catch (error) {
    setFiles((prev) =>
      prev.map((f) =>
        f.id === uploadFile.id
          ? { ...f, status: 'error', error: error.message }
          : f
      )
    );
  }
};
```

### Step 2: Upload Endpoint

**Location**: `api/routers/documents.py`

```python
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Validate
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported: {file_ext}")
    
    # Read content
    content = await file.read()
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    
    # Generate unique filename
    file_hash = hashlib.md5(content).hexdigest()
    staged_filename = f"{file_hash}_{file.filename}"
    
    # Save to staging
    staging_dir = Path("data/staged_uploads")
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staging_dir / staged_filename
    staged_path.write_bytes(content)
    
    # Create job
    job_manager = get_job_manager()
    job_id = await job_manager.create_job(
        job_type="document_ingestion",
        params={
            "file_path": str(staged_path),
            "filename": file.filename,
        },
    )
    
    # Start background processing
    asyncio.create_task(
        process_document_background(job_id, str(staged_path), file.filename)
    )
    
    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "processing",
    }
```

### Step 3a: Document Loading

**Location**: `ingestion/loaders/pdf_loader.py`

```python
class PDFLoader:
    def load(self, file_path: str) -> Document:
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            
            # Extract text from all pages
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                pages.append({
                    "page_number": i + 1,
                    "content": text,
                })
            
            # Extract metadata
            metadata = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "page_count": len(reader.pages),
                "source": file_path,
            }
            
            # Combine text
            full_text = "\n\n".join([p["content"] for p in pages])
            
            return Document(
                content=full_text,
                metadata=metadata,
                pages=pages,
            )
```

**Extracted Document**:
```python
{
    "content": "VxRail Administration Guide\n\nChapter 1: Introduction...",
    "metadata": {
        "title": "VxRail Administration Guide",
        "author": "Dell EMC",
        "page_count": 350,
        "source": "data/staged_uploads/abc123_VxRail_Admin_Guide.pdf"
    },
    "pages": [
        {"page_number": 1, "content": "VxRail Administration Guide..."},
        {"page_number": 2, "content": "Chapter 1: Introduction..."},
        # ... 348 more pages
    ]
}
```

### Step 3c: Chunking

**Location**: `core/chunking.py`

```python
def chunk_document(
    document: Document,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    # Split by page first
    chunks = []
    for page in document.pages:
        page_chunks = splitter.split_text(page["content"])
        
        for i, chunk_text in enumerate(page_chunks):
            chunk = Chunk(
                content=chunk_text,
                metadata={
                    "document_id": document.id,
                    "page_number": page["page_number"],
                    "position": i,
                    "chunk_size": len(chunk_text),
                },
            )
            chunks.append(chunk)
    
    return chunks
```

**Generated Chunks** (sample):
```python
[
    {
        "chunk_id": "chunk-001",
        "content": "VxRail is a hyper-converged infrastructure appliance...",
        "metadata": {
            "document_id": "doc-vxrail-001",
            "page_number": 15,
            "position": 0,
            "chunk_size": 987
        }
    },
    {
        "chunk_id": "chunk-002",
        "content": "infrastructure appliance that integrates compute...",  # Overlap
        "metadata": {
            "document_id": "doc-vxrail-001",
            "page_number": 15,
            "position": 1,
            "chunk_size": 1024
        }
    },
    # ... 145 more chunks
]
```

### Step 3d: Embedding Generation

**Location**: `core/embeddings.py`

```python
async def generate_embeddings_batch(
    self,
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 20,
) -> List[List[float]]:
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Check cache
        batch_embeddings = []
        uncached_indices = []
        
        for j, text in enumerate(batch):
            cache_key = f"{text}:{model}"
            if cached := self.embedding_cache.get(cache_key):
                batch_embeddings.append(cached)
            else:
                uncached_indices.append(j)
        
        # Generate uncached embeddings
        if uncached_indices:
            uncached_texts = [batch[j] for j in uncached_indices]
            
            response = await self.client.embeddings.create(
                model=model,
                input=uncached_texts,
            )
            
            # Cache and collect
            for j, embedding_obj in enumerate(response.data):
                embedding = embedding_obj.embedding
                text = uncached_texts[j]
                
                cache_key = f"{text}:{model}"
                self.embedding_cache[cache_key] = embedding
                
                batch_embeddings.insert(uncached_indices[j], embedding)
        
        embeddings.extend(batch_embeddings)
        
        # Rate limiting
        await asyncio.sleep(0.1)
        
        # Progress callback
        if self.progress_callback:
            progress = int((i + len(batch)) / len(texts) * 40) + 30  # 30-70%
            self.progress_callback(progress, f"Embedded {i + len(batch)}/{len(texts)} chunks")
    
    return embeddings
```

### Step 3e: Entity Extraction

**Location**: `core/entity_extraction.py`

```python
async def extract_from_chunks(
    self,
    chunks: List[Chunk],
    sample_size: int = 50,
) -> EntityGraph:
    # Sample representative chunks
    sampled = self._sample_chunks(chunks, sample_size)
    
    entity_graph = EntityGraph()
    
    for chunk in sampled:
        # Build extraction prompt
        prompt = f"""Extract entities and relationships from this text.

Text:
{chunk.content}

Return a JSON array of entities and relationships:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}},
    ...
  ],
  "relationships": [
    {{"entity1": "...", "relation": "...", "entity2": "...", "strength": 0.0-1.0}},
    ...
  ]
}}

Entity types: {", ".join(CANONICAL_ENTITY_TYPES)}
"""
        
        # LLM extraction
        response = await self.llm_manager.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000,
        )
        
        # Parse response
        try:
            data = json.loads(response)
            
            # Add entities to graph
            for entity_data in data.get("entities", []):
                entity_graph.add_entity(
                    name=entity_data["name"],
                    entity_type=entity_data["type"],
                    description=entity_data["description"],
                    source_chunk_id=chunk.chunk_id,
                )
            
            # Add relationships
            for rel_data in data.get("relationships", []):
                entity_graph.add_relationship(
                    entity1=rel_data["entity1"],
                    entity2=rel_data["entity2"],
                    relation_type=rel_data["relation"],
                    strength=rel_data["strength"],
                    source_chunk_id=chunk.chunk_id,
                )
        
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse entity extraction: {response}")
    
    # Deduplicate and merge
    entity_graph.deduplicate()
    
    # Generate entity embeddings
    entity_texts = [
        f"{e.name}: {e.description}" for e in entity_graph.entities
    ]
    entity_embeddings = await self.embedding_manager.generate_embeddings_batch(
        entity_texts
    )
    
    for entity, embedding in zip(entity_graph.entities, entity_embeddings):
        entity.embedding = embedding
    
    return entity_graph
```

**EntityGraph Accumulation**:
```python
{
    "entities": [
        {
            "name": "VxRail",
            "type": "Component",
            "description": "Hyper-converged infrastructure appliance; integrated compute and storage",
            "importance": 0.95,
            "provenance": ["chunk-001", "chunk-015", "chunk-023"],  # 3 mentions
            "embedding": [0.012, -0.034, ...]
        },
        {
            "name": "Backup",
            "type": "Procedure",
            "description": "Data protection procedures; scheduled backup jobs",
            "importance": 0.82,
            "provenance": ["chunk-047", "chunk-048"],
            "embedding": [0.021, 0.005, ...]
        },
        # ... 81 more entities
    ],
    "relationships": [
        {
            "entity1": "VxRail",
            "relation": "RELATED_TO",
            "entity2": "Backup",
            "strength": 0.85,  # Summed from multiple mentions
            "provenance": ["chunk-047"]
        },
        # ... more relationships
    ]
}
```

### Step 3f: Neo4j Persistence

**Location**: `core/graph_db.py`

```python
def persist_document_with_chunks(
    self,
    document: Document,
    chunks: List[Chunk],
    entity_graph: EntityGraph = None,
):
    with self.driver.session() as session:
        # Create Document node
        session.run("""
            CREATE (d:Document {
                id: $id,
                filename: $filename,
                title: $title,
                page_count: $page_count,
                upload_date: datetime()
            })
        """, id=document.id, filename=document.filename, ...)
        
        # Batch insert chunks with UNWIND
        chunk_data = [
            {
                "id": c.chunk_id,
                "content": c.content,
                "page_number": c.metadata["page_number"],
                "position": c.metadata["position"],
                "embedding": c.embedding,
                "document_id": document.id,
            }
            for c in chunks
        ]
        
        session.run("""
            UNWIND $chunks AS chunk
            CREATE (c:Chunk {
                id: chunk.id,
                content: chunk.content,
                page_number: chunk.page_number,
                position: chunk.position,
                embedding: chunk.embedding
            })
            WITH c, chunk
            MATCH (d:Document {id: chunk.document_id})
            CREATE (d)-[:CONTAINS]->(c)
        """, chunks=chunk_data)
        
        # Batch insert entities
        if entity_graph:
            entity_data = [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "description": e.description,
                    "importance": e.importance,
                    "embedding": e.embedding,
                }
                for e in entity_graph.entities
            ]
            
            session.run("""
                UNWIND $entities AS entity
                MERGE (e:Entity {name: entity.name})
                ON CREATE SET
                    e.entity_type = entity.type,
                    e.description = entity.description,
                    e.importance = entity.importance,
                    e.embedding = entity.embedding
                ON MATCH SET
                    e.description = e.description + '; ' + entity.description,
                    e.importance = (e.importance + entity.importance) / 2
            """, entities=entity_data)
            
            # Batch create MENTIONS relationships
            mentions_data = []
            for entity in entity_graph.entities:
                for chunk_id in entity.provenance:
                    mentions_data.append({
                        "chunk_id": chunk_id,
                        "entity_name": entity.name,
                    })
            
            session.run("""
                UNWIND $mentions AS mention
                MATCH (c:Chunk {id: mention.chunk_id})
                MATCH (e:Entity {name: mention.entity_name})
                CREATE (c)-[:MENTIONS]->(e)
            """, mentions=mentions_data)
            
            # Batch create RELATED_TO relationships
            rel_data = [
                {
                    "entity1": r.entity1,
                    "entity2": r.entity2,
                    "strength": r.strength,
                }
                for r in entity_graph.relationships
            ]
            
            session.run("""
                UNWIND $relationships AS rel
                MATCH (e1:Entity {name: rel.entity1})
                MATCH (e2:Entity {name: rel.entity2})
                MERGE (e1)-[r:RELATED_TO]-(e2)
                ON CREATE SET r.strength = rel.strength
                ON MATCH SET r.strength = r.strength + rel.strength
            """, relationships=rel_data)
```

### Step 4: Status Polling

**Location**: `frontend/src/lib/api-client.ts`

```typescript
export async function pollJobStatus(
  jobId: string,
  onUpdate: (status: JobStatus) => void,
): Promise<void> {
  const pollInterval = 1000; // 1 second
  const maxAttempts = 600; // 10 minutes
  
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const status = await response.json();
    
    onUpdate(status);
    
    if (status.status === 'completed' || status.status === 'failed') {
      break;
    }
    
    await new Promise((resolve) => setTimeout(resolve, pollInterval));
  }
}
```

**Job Status Updates**:
```typescript
// Initial
{ status: "running", progress: 10, message: "Extracted text" }

// Chunking
{ status: "running", progress: 30, message: "Created 147 chunks" }

// Embedding
{ status: "running", progress: 50, message: "Embedded 100/147 chunks" }

// Entity extraction
{ status: "running", progress: 75, message: "Extracted 83 entities" }

// Persistence
{ status: "running", progress: 90, message: "Persisting to database" }

// Completion
{
  status: "completed",
  progress: 100,
  message: "Processing complete",
  result: {
    document_id: "doc-vxrail-001",
    chunk_count: 147,
    entity_count: 83
  }
}
```

## Performance Notes

### Bottlenecks

1. **Embedding Generation**: ~15-20s for 147 chunks (batched, rate-limited)
2. **Entity Extraction**: ~30-40s for 50 chunks (LLM calls)
3. **Neo4j Persistence**: ~2-3s for batch inserts

### Optimization Strategies

- **Parallel Embedding**: 5 concurrent batches (configurable)
- **Selective Entity Extraction**: Sample 50 chunks instead of all 147
- **UNWIND Queries**: Batch 100+ nodes/relationships per query
- **Background Processing**: Non-blocking upload response
- **Incremental Progress**: Fine-grained job updates for UX

### Total Time Estimate

- Small document (20 pages, 30 chunks): ~10-15 seconds
- Medium document (100 pages, 150 chunks): ~40-60 seconds
- Large document (500 pages, 700 chunks): ~3-5 minutes

## Related Documentation

- [Document Processor](03-components/ingestion/document-processor.md)
- [Loaders](03-components/ingestion/loaders.md)
- [Entity Extraction](03-components/ingestion/entity-extraction.md)
- [Document Upload](04-features/document-upload.md)
