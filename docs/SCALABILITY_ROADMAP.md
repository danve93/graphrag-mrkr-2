# Amber Scalability Roadmap
**Inspired by TrustGraph Architecture**

Date: November 27, 2025

---

## Executive Summary

This document provides actionable recommendations to improve Amber's scalability by selectively adopting proven patterns from TrustGraph, while maintaining Amber's simplicity and unified architecture.

**Priority Levels:**
- 🟢 **Quick Win** — Low cost, high impact (1-2 weeks)
- 🟡 **Medium Effort** — Moderate investment (1-2 months)
- 🔴 **Major Refactor** — Significant work (3-6 months)

---

## 1. Observability Stack 🟢 **Quick Win**

### What TrustGraph Does
- Prometheus metrics for every service
- Grafana dashboards tracking:
  - Query latency (p50, p95, p99)
  - LLM cost per session/document
  - Entity extraction success rate
  - Vector search performance
  - Cache hit rates

### Implementation for Amber

**Step 1: Add Prometheus Client**

```python
# requirements.txt
prometheus-client==0.19.0
```

**Step 2: Create Metrics Module**

```python
# core/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
import time

# Chat metrics
chat_requests = Counter('amber_chat_requests_total', 'Total chat requests', ['session_id', 'retrieval_mode'])
chat_latency = Histogram('amber_chat_latency_seconds', 'Chat request duration', ['stage'])
chat_tokens = Counter('amber_llm_tokens_total', 'LLM tokens used', ['model', 'type'])  # type: prompt/completion
chat_cost = Counter('amber_llm_cost_dollars', 'LLM cost in USD', ['model'])

# Ingestion metrics
ingestion_documents = Counter('amber_ingestion_documents_total', 'Documents ingested', ['status'])
ingestion_chunks = Counter('amber_ingestion_chunks_total', 'Chunks created')
ingestion_entities = Counter('amber_ingestion_entities_total', 'Entities extracted')
ingestion_duration = Histogram('amber_ingestion_duration_seconds', 'Document processing time', ['stage'])

# Retrieval metrics
retrieval_vectors = Histogram('amber_vector_search_duration_seconds', 'Vector search latency')
retrieval_graph = Histogram('amber_graph_expansion_duration_seconds', 'Graph traversal latency')
retrieval_results = Histogram('amber_retrieval_results_count', 'Number of chunks retrieved', ['mode'])

# Cache metrics
cache_hits = Counter('amber_cache_hits_total', 'Cache hits', ['cache_type'])
cache_misses = Counter('amber_cache_misses_total', 'Cache misses', ['cache_type'])

# Database metrics
neo4j_queries = Counter('amber_neo4j_queries_total', 'Neo4j queries executed', ['operation'])
neo4j_duration = Histogram('amber_neo4j_query_duration_seconds', 'Neo4j query time', ['operation'])

# System info
system_info = Info('amber_system', 'System information')
system_info.info({
    'version': '2.0.0',
    'llm_provider': 'openai',  # from settings
    'embedding_model': 'text-embedding-3-small',
})

# Quality metrics
quality_score_avg = Gauge('amber_response_quality_avg', 'Average quality score (last 100)')
entity_extraction_success = Gauge('amber_entity_extraction_success_rate', 'Entity extraction success %')
```

**Step 3: Instrument Core Functions**

```python
# rag/graph_rag.py
from core.metrics import chat_requests, chat_latency, retrieval_vectors

async def run_pipeline(state: RAGState):
    chat_requests.labels(
        session_id=state.get("session_id", "unknown"),
        retrieval_mode=state.get("retrieval_mode", "hybrid")
    ).inc()
    
    with chat_latency.labels(stage="query_analysis").time():
        state = await query_analysis(state)
    
    with chat_latency.labels(stage="retrieval").time():
        with retrieval_vectors.time():
            state = await retrieval(state)
    
    # ... continue for each stage
    return state

# core/embeddings.py
from core.metrics import chat_tokens, chat_cost

async def generate_embeddings(texts):
    start = time.time()
    embeddings = await openai_client.embed(texts)
    duration = time.time() - start
    
    tokens = sum(len(t.split()) for t in texts) * 1.3  # rough estimate
    chat_tokens.labels(model="text-embedding-3-small", type="prompt").inc(tokens)
    chat_cost.labels(model="text-embedding-3-small").inc(tokens * 0.00002 / 1000)
    
    return embeddings
```

**Step 4: Add Prometheus Endpoint**

```python
# api/main.py
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

**Step 5: Docker Compose + Grafana**

```yaml
# docker-compose.yml additions
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml
      - grafana_data:/var/lib/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false

volumes:
  prometheus_data:
  grafana_data:
```

**Step 6: Prometheus Config**

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'amber-backend'
    static_configs:
      - targets: ['backend:8000']
```

**Step 7: Grafana Dashboard JSON**

Create `monitoring/grafana-dashboards/amber-main.json` with panels for:
- Chat request rate (requests/min)
- p95 latency per stage
- LLM cost per hour/day
- Entity extraction success rate
- Cache hit ratio

**Estimated Effort:** 3-5 days  
**Impact:** Immediate visibility into performance bottlenecks and costs  
**ROI:** 10x — catch issues before users complain, optimize LLM spend

---

## 2. Connection Pooling & Caching 🟢 **Quick Win**

### What TrustGraph Does
- Long-lived service instances with persistent caches
- LRU cache with TTL for label lookups (5-minute freshness)
- Reuses embedding/LLM clients across requests
- Per-service connection pools

### Current Amber Problem

```python
# rag/graph_rag.py - INEFFICIENT
async def run_pipeline(state):
    # New Neo4j driver connection EVERY request
    graph_db = GraphDatabase(settings.neo4j_uri, ...)
    
    # New LLM client EVERY request
    llm_manager = LLMManager()
    
    # No caching across requests
```

### Implementation for Amber

**Step 1: Application-Level Singletons**

```python
# core/singletons.py
from functools import lru_cache
from neo4j import GraphDatabase
from core.llm import LLMManager
from core.embeddings import EmbeddingManager

_graph_db = None
_llm_manager = None
_embedding_manager = None

def get_graph_db():
    """Singleton Neo4j connection pool"""
    global _graph_db
    if _graph_db is None:
        _graph_db = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_pool_size=50,  # connection pool
            connection_timeout=30,
            max_transaction_retry_time=15,
        )
    return _graph_db

def get_llm_manager():
    """Singleton LLM manager with client reuse"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager

def get_embedding_manager():
    """Singleton embedding manager"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager

# Cleanup on shutdown
def cleanup_singletons():
    global _graph_db, _llm_manager, _embedding_manager
    if _graph_db:
        _graph_db.close()
    _graph_db = None
    _llm_manager = None
    _embedding_manager = None
```

**Step 2: Add to FastAPI Lifespan**

```python
# api/main.py
from contextlib import asynccontextmanager
from core.singletons import get_graph_db, cleanup_singletons

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize singletons
    get_graph_db()  # Prewarm connection pool
    logger.info("Connection pools initialized")
    
    yield
    
    # Shutdown: cleanup
    cleanup_singletons()
    logger.info("Connections closed")

app = FastAPI(lifespan=lifespan)
```

**Step 3: LRU Cache for Entity Labels**

```python
# core/graph_db.py
from functools import lru_cache
from cachetools import TTLCache
import threading

class GraphDatabase:
    def __init__(self, ...):
        # Thread-safe TTL cache (5 min expiry, 5000 items max)
        self._label_cache = TTLCache(maxsize=5000, ttl=300)
        self._cache_lock = threading.Lock()
    
    def get_entity_label(self, entity_uri: str) -> str:
        """Get entity label with caching"""
        with self._cache_lock:
            if entity_uri in self._label_cache:
                cache_hits.labels(cache_type="entity_label").inc()
                return self._label_cache[entity_uri]
        
        cache_misses.labels(cache_type="entity_label").inc()
        
        # Query Neo4j
        query = "MATCH (e:Entity {uri: $uri}) RETURN e.name as label"
        result = self.query(query, {"uri": entity_uri})
        label = result[0]["label"] if result else entity_uri
        
        with self._cache_lock:
            self._label_cache[entity_uri] = label
        
        return label
```

**Step 4: Query Result Caching (Conservative)**

```python
# rag/retriever.py
from cachetools import LRUCache
import hashlib

class Retriever:
    def __init__(self):
        # Cache vector search results (100 queries, no TTL for session)
        self._vector_cache = LRUCache(maxsize=100)
    
    async def vector_search(self, query_embedding, top_k=10):
        cache_key = hashlib.md5(
            f"{query_embedding[:5]}-{top_k}".encode()
        ).hexdigest()
        
        if cache_key in self._vector_cache:
            cache_hits.labels(cache_type="vector_search").inc()
            return self._vector_cache[cache_key]
        
        cache_misses.labels(cache_type="vector_search").inc()
        results = await self._do_vector_search(query_embedding, top_k)
        self._vector_cache[cache_key] = results
        return results
```

**Estimated Effort:** 2-3 days  
**Impact:** 30-50% latency reduction, 10x fewer DB connections  
**ROI:** 20x — minimal code changes, huge performance gain

---

## 3. Separate Query Services 🟡 **Medium Effort**

### What TrustGraph Does
Separate services for different query types:
- `graph-rag` — Entity-centric reasoning (what Amber does)
- `document-rag` — Pure document chunk retrieval
- `ontology-query` — SPARQL/Cypher generation from natural language

### Why This Matters
Users often know if they want "find similar documents" vs "reason over entities." Forcing hybrid retrieval on all queries wastes compute.

### Implementation for Amber

**Step 1: Split Retrieval Modes into Services**

```python
# api/routers/query.py (NEW)
from enum import Enum

class QueryService(str, Enum):
    HYBRID = "hybrid"          # Current default (vector + graph)
    VECTOR_ONLY = "vector"     # Fast document similarity
    GRAPH_ONLY = "graph"       # Entity reasoning only
    CYPHER = "cypher"          # Natural language → Cypher

@router.post("/api/query")
async def unified_query(request: QueryRequest):
    """Route query to appropriate service"""
    
    if request.service == QueryService.VECTOR_ONLY:
        return await vector_only_query(request)
    
    elif request.service == QueryService.GRAPH_ONLY:
        return await graph_only_query(request)
    
    elif request.service == QueryService.CYPHER:
        return await cypher_query(request)
    
    else:  # HYBRID (default)
        return await hybrid_query(request)

async def vector_only_query(request):
    """Fast document retrieval without graph expansion"""
    # 1. Embed query
    # 2. Vector search in Neo4j
    # 3. Generate response from chunks
    # Skip: entity extraction, graph traversal, reranking
    pass

async def graph_only_query(request):
    """Entity reasoning without vector search"""
    # 1. Extract entities from query (NER)
    # 2. Expand via graph relationships
    # 3. Generate from subgraph
    # Skip: vector embeddings, chunk retrieval
    pass

async def cypher_query(request):
    """Convert natural language to Cypher (TrustGraph's ontology-query)"""
    # 1. LLM generates Cypher from query
    # 2. Execute Cypher against Neo4j
    # 3. Format results as natural language
    pass
```

**Step 2: Add Service Selection to UI**

```typescript
// frontend/src/components/ChatInterface.tsx
const [queryService, setQueryService] = useState<QueryService>("hybrid");

<Select value={queryService} onChange={setQueryService}>
  <option value="hybrid">Smart Search (Hybrid)</option>
  <option value="vector">Find Similar Documents</option>
  <option value="graph">Entity Reasoning</option>
  <option value="cypher">Advanced Query (Cypher)</option>
</Select>
```

**Step 3: Implement Cypher Generation**

```python
# rag/cypher_generator.py
CYPHER_PROMPT = """You are a Neo4j Cypher expert. Convert the natural language query to Cypher.

Schema:
- Nodes: Document, Chunk, Entity
- Relationships: HAS_CHUNK, MENTIONS_ENTITY, SIMILAR_TO, RELATED_TO

Query: {query}

Return ONLY valid Cypher (no explanations):"""

async def generate_cypher(query: str) -> str:
    prompt = CYPHER_PROMPT.format(query=query)
    cypher = await llm_manager.generate(prompt, temperature=0.0)
    
    # Validate Cypher syntax
    cypher = cypher.strip().strip("```").strip("cypher").strip()
    
    return cypher

async def execute_cypher_query(query: str):
    cypher = await generate_cypher(query)
    
    # Safety: read-only queries only
    if any(kw in cypher.upper() for kw in ["CREATE", "DELETE", "SET", "MERGE"]):
        raise ValueError("Only read queries allowed")
    
    results = graph_db.query(cypher)
    
    # Format results as natural language
    return await format_cypher_results(query, cypher, results)
```

**Estimated Effort:** 2-3 weeks  
**Impact:** 2-5x faster for simple queries, advanced users love Cypher mode  
**ROI:** 5x — power users unlock value, reduced compute waste

---

## 4. Knowledge Cores (Reusable Datasets) 🟡 **Medium Effort**

### What TrustGraph Does
"Knowledge Cores" are self-contained, reusable knowledge packages:
- Graph triples (RDF/N-Triples format)
- Vector embeddings mapped to graph nodes
- Metadata (schema, version, provenance)
- Load/unload at runtime (multi-tenancy)

### Why This Matters for Amber
- **Multi-client deployments:** Client A shouldn't see Client B's data
- **Versioning:** Roll back to previous knowledge state
- **Reuse:** Pre-built knowledge cores (industry standards, regulations)
- **Isolation:** Separate staging/production knowledge

### Implementation for Amber

**Step 1: Knowledge Core Schema**

```python
# core/knowledge_core.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class KnowledgeCore:
    id: str                      # e.g., "client-acme-v1"
    name: str                    # Human-readable
    description: str
    version: str                 # Semantic version
    created_at: datetime
    updated_at: datetime
    document_count: int
    entity_count: int
    chunk_count: int
    status: str                  # "active", "inactive", "archived"
    tags: List[str]              # ["industry:healthcare", "region:us"]
    
    # Neo4j label for isolation
    collection_label: str        # e.g., "KnowledgeCore_client_acme_v1"
    
    # Access control
    owner: str
    allowed_users: List[str]
```

**Step 2: Neo4j Collection Labels**

Instead of separate databases, use **labels** for isolation:

```cypher
-- Current (single collection)
(:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS_ENTITY]->(:Entity)

-- With Knowledge Cores (multi-collection)
(:Document:KnowledgeCore_acme)-[:HAS_CHUNK]->(:Chunk:KnowledgeCore_acme)-[:MENTIONS_ENTITY]->(:Entity:KnowledgeCore_acme)
(:Document:KnowledgeCore_beta)-[:HAS_CHUNK]->(:Chunk:KnowledgeCore_beta)-[:MENTIONS_ENTITY]->(:Entity:KnowledgeCore_beta)
```

**Step 3: Core Manager**

```python
# core/knowledge_core_manager.py
class KnowledgeCoreManager:
    def __init__(self, graph_db: GraphDatabase):
        self.graph_db = graph_db
    
    def create_core(self, name: str, description: str, owner: str) -> KnowledgeCore:
        """Create new knowledge core"""
        core_id = f"kc_{uuid.uuid4().hex[:8]}"
        collection_label = f"KnowledgeCore_{core_id}"
        
        core = KnowledgeCore(
            id=core_id,
            name=name,
            description=description,
            version="1.0.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            document_count=0,
            entity_count=0,
            chunk_count=0,
            status="active",
            tags=[],
            collection_label=collection_label,
            owner=owner,
            allowed_users=[owner],
        )
        
        # Persist metadata in Neo4j
        self._save_core_metadata(core)
        
        return core
    
    def ingest_to_core(self, core_id: str, document_path: str):
        """Ingest document into specific knowledge core"""
        core = self.get_core(core_id)
        
        # Process document with core label
        processor = DocumentProcessor(
            collection_label=core.collection_label
        )
        processor.process_document(document_path)
        
        # Update counts
        core.document_count += 1
        self._save_core_metadata(core)
    
    def deactivate_core(self, core_id: str):
        """Deactivate knowledge core (soft delete)"""
        core = self.get_core(core_id)
        core.status = "inactive"
        self._save_core_metadata(core)
    
    def delete_core(self, core_id: str):
        """Hard delete knowledge core"""
        core = self.get_core(core_id)
        
        # Delete all nodes with core label
        query = f"""
        MATCH (n:{core.collection_label})
        DETACH DELETE n
        """
        self.graph_db.query(query)
        
        # Delete metadata
        self.graph_db.query(
            "MATCH (m:KnowledgeCoreMetadata {id: $id}) DELETE m",
            {"id": core_id}
        )
    
    def list_cores(self, owner: Optional[str] = None) -> List[KnowledgeCore]:
        """List all knowledge cores"""
        query = """
        MATCH (m:KnowledgeCoreMetadata)
        WHERE $owner IS NULL OR m.owner = $owner
        RETURN m
        """
        results = self.graph_db.query(query, {"owner": owner})
        return [self._core_from_node(r["m"]) for r in results]
```

**Step 4: Update Retrieval to Respect Cores**

```python
# rag/retriever.py
async def retrieve(self, query: str, knowledge_core_id: Optional[str] = None):
    """Retrieve with optional knowledge core filter"""
    
    # Build label filter
    if knowledge_core_id:
        core = core_manager.get_core(knowledge_core_id)
        label_filter = f":{core.collection_label}"
    else:
        label_filter = ""  # Query all
    
    # Vector search with label filter
    cypher = f"""
    CALL db.index.vector.queryNodes('chunk_embeddings', {top_k}, $embedding)
    YIELD node, score
    WHERE node{label_filter} IS NOT NULL
    RETURN node, score
    """
    
    # Graph expansion with label filter
    expansion_cypher = f"""
    MATCH (c:Chunk{label_filter})-[:SIMILAR_TO]->(c2:Chunk{label_filter})
    WHERE c.id IN $seed_chunks
    RETURN c2
    """
```

**Step 5: API Endpoints**

```python
# api/routers/knowledge_cores.py
@router.post("/api/knowledge-cores")
async def create_knowledge_core(request: CreateCoreRequest):
    """Create new knowledge core"""
    core = core_manager.create_core(
        name=request.name,
        description=request.description,
        owner=current_user.id,
    )
    return core

@router.get("/api/knowledge-cores")
async def list_knowledge_cores():
    """List all knowledge cores"""
    return core_manager.list_cores(owner=current_user.id)

@router.post("/api/knowledge-cores/{core_id}/ingest")
async def ingest_to_core(core_id: str, file: UploadFile):
    """Ingest document into knowledge core"""
    await core_manager.ingest_to_core(core_id, file)
    return {"status": "processing"}

@router.delete("/api/knowledge-cores/{core_id}")
async def delete_knowledge_core(core_id: str):
    """Delete knowledge core"""
    core_manager.delete_core(core_id)
    return {"status": "deleted"}
```

**Step 6: Frontend UI**

```typescript
// frontend/src/pages/KnowledgeCores.tsx
export default function KnowledgeCoresPage() {
  const [cores, setCores] = useState<KnowledgeCore[]>([]);
  
  return (
    <div>
      <h1>Knowledge Cores</h1>
      <Button onClick={createCore}>+ New Core</Button>
      
      <Grid>
        {cores.map(core => (
          <CoreCard key={core.id}>
            <h3>{core.name}</h3>
            <p>{core.description}</p>
            <Stats>
              {core.document_count} docs, {core.entity_count} entities
            </Stats>
            <Actions>
              <Button onClick={() => ingestDoc(core.id)}>Upload</Button>
              <Button onClick={() => queryCore(core.id)}>Query</Button>
              <Button onClick={() => deleteCore(core.id)}>Delete</Button>
            </Actions>
          </CoreCard>
        ))}
      </Grid>
    </div>
  );
}
```

**Estimated Effort:** 1-2 months  
**Impact:** Multi-tenancy, versioning, enterprise-ready isolation  
**ROI:** 10x for SaaS deployments, required for B2B sales

---

## 5. Async Background Workers 🔴 **Major Refactor**

### What TrustGraph Does
- Apache Pulsar message queue for inter-service communication
- Background workers consume jobs from queues
- Non-blocking ingestion and processing

### Amber's Current Limitation
```python
# api/routers/documents.py - BLOCKING
@router.post("/api/documents/upload")
async def upload_document(file: UploadFile):
    # This blocks the API thread for 30-60 seconds!
    document_id = await document_processor.process_document(file)
    return {"document_id": document_id, "status": "complete"}
```

### Implementation for Amber (Celery Alternative to Pulsar)

**Why Celery, not Pulsar?**
- Celery: Python-native, Redis-backed, easier to deploy
- Pulsar: Java-based, adds complexity, overkill for monolith

**Step 1: Add Celery + Redis**

```python
# requirements.txt
celery[redis]==5.3.4
redis==5.0.1
```

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  celery-worker:
    build: .
    command: celery -A api.celery_app worker --loglevel=info --concurrency=4
    depends_on:
      - redis
      - neo4j
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0

volumes:
  redis_data:
```

**Step 2: Celery App**

```python
# api/celery_app.py
from celery import Celery
from config.settings import settings

celery_app = Celery(
    "amber",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
)
```

**Step 3: Background Tasks**

```python
# api/tasks.py
from api.celery_app import celery_app
from ingestion.document_processor import DocumentProcessor

@celery_app.task(bind=True)
def process_document_task(self, document_id: str, file_path: str, knowledge_core_id: str):
    """Background task for document processing"""
    try:
        # Update status to "processing"
        update_document_status(document_id, "processing")
        
        processor = DocumentProcessor(
            collection_label=f"KnowledgeCore_{knowledge_core_id}"
        )
        
        # Process document (can take minutes)
        result = processor.process_document(file_path)
        
        # Update status to "complete"
        update_document_status(document_id, "complete", metadata=result)
        
        return {"status": "complete", "chunks": result.chunk_count}
    
    except Exception as e:
        # Update status to "failed"
        update_document_status(document_id, "failed", error=str(e))
        raise

@celery_app.task
def generate_document_summary_task(document_id: str):
    """Background task for summary generation"""
    # ... existing summary logic
    pass

@celery_app.task
def run_clustering_task(knowledge_core_id: str):
    """Background task for Leiden clustering"""
    # ... existing clustering logic
    pass
```

**Step 4: Non-Blocking API**

```python
# api/routers/documents.py
from api.tasks import process_document_task

@router.post("/api/documents/upload")
async def upload_document(
    file: UploadFile,
    knowledge_core_id: str = "default"
):
    """Upload document for async processing"""
    
    # Save file to staging
    document_id = str(uuid.uuid4())
    file_path = f"data/staged_uploads/{document_id}/{file.filename}"
    await save_upload(file, file_path)
    
    # Create document record (status="queued")
    create_document_record(document_id, file.filename, "queued")
    
    # Queue background task
    task = process_document_task.delay(document_id, file_path, knowledge_core_id)
    
    # Return immediately
    return {
        "document_id": document_id,
        "task_id": task.id,
        "status": "queued",
        "message": "Processing in background"
    }

@router.get("/api/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Poll document processing status"""
    doc = get_document_record(document_id)
    return {
        "status": doc.status,  # queued, processing, complete, failed
        "progress": doc.progress,  # 0-100
        "metadata": doc.metadata,
    }
```

**Step 5: Frontend Polling**

```typescript
// frontend/src/hooks/useDocumentStatus.ts
export function useDocumentStatus(documentId: string) {
  const [status, setStatus] = useState<DocumentStatus>("queued");
  
  useEffect(() => {
    const poll = setInterval(async () => {
      const response = await fetch(`/api/documents/${documentId}/status`);
      const data = await response.json();
      
      setStatus(data.status);
      
      if (data.status === "complete" || data.status === "failed") {
        clearInterval(poll);
      }
    }, 2000);  // Poll every 2 seconds
    
    return () => clearInterval(poll);
  }, [documentId]);
  
  return status;
}
```

**Estimated Effort:** 1-2 months  
**Impact:** Non-blocking API, horizontal scalability, better UX  
**ROI:** 15x — required for production SaaS, enables worker auto-scaling

---

## 6. Cost-Benefit Summary

| Feature | Effort | Impact | ROI | Priority |
|---------|--------|--------|-----|----------|
| **Observability (Prometheus + Grafana)** | 1 week | Immediate visibility into bottlenecks | 10x | 🟢 Do first |
| **Connection Pooling & Caching** | 3 days | 30-50% latency reduction | 20x | 🟢 Do first |
| **Separate Query Services** | 3 weeks | 2-5x faster for simple queries | 5x | 🟡 Month 2 |
| **Knowledge Cores** | 2 months | Multi-tenancy, versioning | 10x | 🟡 If B2B |
| **Async Workers (Celery)** | 2 months | Non-blocking API, scalability | 15x | 🔴 If SaaS |

---

## 7. Implementation Sequence

### Phase 1: Quick Wins (Week 1-2)
1. Add Prometheus metrics to all endpoints
2. Deploy Grafana dashboard via Docker Compose
3. Implement singleton connection pools
4. Add TTL cache for entity labels
5. Measure baseline performance

### Phase 2: Query Optimization (Week 3-6)
1. Implement vector-only query service
2. Implement graph-only query service
3. Implement Cypher generation service
4. Add service selector to UI
5. A/B test query performance

### Phase 3: Multi-Tenancy (Month 2-3)
1. Design knowledge core schema
2. Implement core manager
3. Update ingestion pipeline for labels
4. Update retrieval to respect cores
5. Build knowledge cores UI

### Phase 4: Async Processing (Month 4-5)
1. Deploy Redis + Celery workers
2. Migrate ingestion to background tasks
3. Implement status polling API
4. Update frontend for async UX
5. Load test with 100+ concurrent uploads

---

## 8. TrustGraph Features NOT Worth Copying

### ❌ Apache Pulsar
**Why TrustGraph uses it:** Microservices need message queues  
**Why Amber doesn't:** Monolith doesn't need inter-service messaging  
**Alternative:** Celery + Redis (much simpler)

### ❌ RDF/SPARQL
**Why TrustGraph uses it:** Semantic web standards, ontology reasoning  
**Why Amber doesn't:** Property graph (Neo4j Cypher) is more practical  
**Alternative:** Keep Cypher, add Cypher generation from NL

### ❌ 20+ Microservices
**Why TrustGraph uses it:** Enterprise scale, pluggable components  
**Why Amber doesn't:** Monolith is simpler to deploy/debug  
**Alternative:** Keep monolith, scale vertically first

### ❌ Configuration Builder UI
**Why TrustGraph uses it:** Deploy customized stacks per client  
**Why Amber doesn't:** Single deployment config is simpler  
**Alternative:** Environment variables + Chat Tuning panel

---

## 9. Metrics to Track Success

After implementing these improvements, track:

**Performance:**
- p95 latency per query type (target: <2s for vector-only, <5s for hybrid)
- Cache hit rate (target: >70% for entity labels)
- Neo4j connection pool utilization (target: <50%)

**Cost:**
- LLM cost per query (target: <$0.02 for vector-only, <$0.05 for hybrid)
- Embedding cost per document (target: <$0.01 per document)

**Scalability:**
- Concurrent users supported (target: 100+ with <10s latency)
- Document ingestion throughput (target: 10+ docs/min with workers)

**User Experience:**
- Query response time (target: <3s perceived latency with streaming)
- Upload-to-queryable time (target: <5min with async workers)

---

## Conclusion

**Start with observability and caching (Phase 1)** — these are low-hanging fruit with massive ROI.

**Add query services if users complain about slowness (Phase 2)** — but measure first with Grafana.

**Implement knowledge cores if selling to enterprises (Phase 3)** — required for multi-tenancy.

**Move to async workers only when uploads block the API (Phase 4)** — this is the biggest refactor.

**Avoid blindly copying TrustGraph's microservices architecture** — Amber's monolith is a strength, not a weakness. Scale vertically before horizontally.
