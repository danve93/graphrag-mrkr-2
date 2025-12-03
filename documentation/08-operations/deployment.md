# Deployment Checklist

## Overview

This checklist ensures successful deployment of the Amber RAG system with conversation context, enhanced caching, and stage timing features (Milestones 1-5).

## Pre-Deployment Validation

### 1. Environment Variables

Verify all required environment variables are set:

**Required:**
```bash
# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<secure-password>

# OpenAI (if using OpenAI)
OPENAI_API_KEY=<your-api-key>

# Redis (for job management)
REDIS_URL=redis://redis:6379
```

**Optional but Recommended:**
```bash
# Caching (M1-M2)
ENABLE_CACHING=true
ENTITY_LABEL_CACHE_SIZE=5000
ENTITY_LABEL_CACHE_TTL=300
EMBEDDING_CACHE_SIZE=10000
RETRIEVAL_CACHE_SIZE=1000
RETRIEVAL_CACHE_TTL=60
NEO4J_MAX_CONNECTION_POOL_SIZE=50

# LLM/Embeddings
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small

# Retrieval
HYBRID_CHUNK_WEIGHT=1.0
HYBRID_ENTITY_WEIGHT=1.0
HYBRID_KEYWORD_WEIGHT=0.5
MAX_EXPANDED_CHUNKS=15
MAX_EXPANSION_DEPTH=2
EXPANSION_SIMILARITY_THRESHOLD=0.75

# RRF (M2)
ENABLE_RRF=true
RRF_K=60

# Reranking
FLASHRANK_ENABLED=true
FLASHRANK_MODEL_NAME=ms-marco-MiniLM-L-12-v2
FLASHRANK_MAX_CANDIDATES=50
FLASHRANK_BLEND_WEIGHT=0.5

# Entity Extraction
ENABLE_ENTITY_EXTRACTION=true
SYNC_ENTITY_EMBEDDINGS=false

# Query Analysis (M2)
ENABLE_QUERY_EXPANSION=true
QUERY_EXPANSION_THRESHOLD=3
```

**Validation Command:**
```bash
# Check .env file
cat .env | grep -E "(NEO4J|OPENAI|REDIS|ENABLE_CACHING)"

# Verify no secrets in git
git ls-files | xargs grep -l "OPENAI_API_KEY" || echo "No secrets found ✓"
```

### 2. Dependency Check

**Backend:**
```bash
# Verify Python version (3.10-3.13)
python3 --version

# Install and check dependencies
pip install -r requirements.txt
pip check

# Run tests
pytest tests/ -v
```

**Frontend:**
```bash
cd frontend

# Verify Node version (18+)
node --version

# Install dependencies
npm install

# Check for vulnerabilities
npm audit

# Build production bundle
npm run build
```

### 3. Database Validation

**Neo4j Connection:**
```bash
# Test connection
curl http://localhost:8000/api/health

# Expected: {"status":"healthy","neo4j":"connected"}
```

**Database Statistics:**
```bash
curl http://localhost:8000/api/database/stats

# Verify documents, chunks, entities counts are reasonable
```

**Neo4j Indexes:**
```cypher
// Connect to Neo4j browser: http://localhost:7474
// Run index check
SHOW INDEXES;

// Expected indexes:
// - chunk_embedding (Chunk.embedding)
// - entity_label (Entity.label)
// - doc_id (Document.id)
// - rel_strength (RELATED_TO.strength)
```

### 4. Integration Testing

Run comprehensive integration tests (M5):

```bash
# Ensure backend and Neo4j are running
docker compose up -d

# Wait for services
sleep 10

# Run integration tests
python3 scripts/test_integration_m5.py
```

**Expected Results:**
- ✅ Test 1: Initial query with context establishment
- ✅ Test 2: Follow-up question (conversation context preserved)
- ✅ Test 3: Cache parameter isolation
- ✅ Test 4: Stage timing metadata (0ms accuracy)
- ✅ Test 5: SSE streaming with timing

### 5. Unit Test Coverage

```bash
# Run all backend tests
pytest tests/ -v --cov=. --cov-report=term-missing

# Expected: 40+ tests passing, >80% coverage
```

**Critical Test Suites:**
- `tests/test_conversation_context.py` (15 tests) - M1
- `tests/test_cache_keys.py` (16 tests) - M2
- `tests/test_stage_timing_unit.py` (9 tests) - M3
- `scripts/test_integration_m5.py` (5 tests) - M5

### 6. Cache Functionality

Verify caching is enabled and working:

```bash
# Check cache stats endpoint
curl http://localhost:8000/api/database/cache-stats

# Expected response with hit/miss counts:
# {
#   "entity_label_cache": {"size": ..., "hits": ..., "misses": ...},
#   "embedding_cache": {"size": ..., "hits": ..., "misses": ...},
#   "retrieval_cache": {"size": ..., "hits": ..., "misses": ...}
# }
```

**Cache Validation:**
```bash
# Run same query twice and verify hit rate increases
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message":"test query","session_id":"test"}' > /dev/null

curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message":"test query","session_id":"test"}' > /dev/null

# Check cache stats again - hit rate should increase
curl http://localhost:8000/api/database/cache-stats
```

### 7. Stage Timing Validation

Verify stage timing is instrumented:

```bash
# Make a query and check stages in response
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What is Carbonio?","session_id":"test"}' \
  | jq '.stages[] | {name, duration_ms}'

# Expected: 4 stages with duration_ms values
# - Query Analysis
# - Retrieval
# - Graph Reasoning
# - Generation
```

## Deployment Steps

### Docker Compose Deployment

**Step 1: Build Images**
```bash
# Build all services
docker compose build

# Or build individual services
docker compose build backend
docker compose build frontend
```

**Step 2: Start Services**
```bash
# Start all services in detached mode
docker compose up -d

# Verify all containers are running
docker compose ps
```

**Expected Output:**
```
NAME                    STATUS              PORTS
graphrag-backend        Up X minutes        0.0.0.0:8000->8000/tcp
graphrag-frontend       Up X minutes        0.0.0.0:3000->3000/tcp
graphrag-neo4j          Up X minutes        0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
flashrank-worker        Up X minutes        
redis                   Up X minutes        0.0.0.0:6379->6379/tcp
```

**Step 3: Verify Service Health**
```bash
# Wait for services to be ready
bash scripts/wait_for_services.sh

# Check backend health
curl http://localhost:8000/api/health

# Check frontend
curl http://localhost:3000

# Check Neo4j
curl http://localhost:7474
```

### Production Deployment (Non-Docker)

**Step 1: Setup Backend**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (use production .env)
source .env.production

# Run database migrations (if any)
# python scripts/migrate_db.py

# Start backend with gunicorn
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --log-level info
```

**Step 2: Setup Frontend**
```bash
cd frontend

# Install dependencies
npm ci

# Build production bundle
npm run build

# Start production server
npm start
```

**Step 3: Setup Reverse Proxy (Nginx)**
```nginx
# /etc/nginx/sites-available/amber

upstream backend {
    server localhost:8000;
}

upstream frontend {
    server localhost:3000;
}

server {
    listen 80;
    server_name amber.example.com;

    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
    }

    # SSE requires special handling
    location /api/chat/stream {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

## Post-Deployment Validation

### 1. Smoke Tests

Run basic functionality tests:

```bash
# Test chat query (non-streaming)
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Carbonio?",
    "session_id": "smoke-test-1"
  }' | jq '.'

# Expected: response with text, sources, stages, total_duration_ms

# Test follow-up (conversation context)
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I install it?",
    "session_id": "smoke-test-1"
  }' | jq '.response'

# Expected: response that references "Carbonio" from context
```

### 2. Performance Validation

**Initial Query (Cold Cache):**
```bash
time curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What backup options are available?",
    "session_id": "perf-test-1"
  }' > /dev/null

# Expected: ~5-15 seconds
```

**Repeated Query (Warm Cache):**
```bash
time curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What backup options are available?",
    "session_id": "perf-test-2"
  }' > /dev/null

# Expected: ~3-10 seconds (30-50% faster)
```

### 3. Frontend Validation

**Manual Testing:**
1. Open http://localhost:3000
2. Send a message: "What is Carbonio?"
3. Verify:
   - ✅ Loading indicator shows 4 stages
   - ✅ Stage tooltips show timing (e.g., "Retrieving Documents (456ms)")
   - ✅ Response appears with sources
   - ✅ Total duration shown at completion
   - ✅ Follow-up questions displayed
4. Send follow-up: "How do I install it?"
5. Verify:
   - ✅ Context preserved (response references "Carbonio")
   - ✅ Same session_id used
   - ✅ History shows both messages

**Automated E2E Tests:**
```bash
cd frontend
npm run test:e2e  # If Playwright/Cypress tests exist
```

### 4. Conversation Context Validation

Test conversation flow:

```bash
# Message 1
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What is Carbonio?","session_id":"ctx-test"}' \
  | jq '.response' | grep -i "carbonio"

# Message 2 (follow-up)
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I install it?","session_id":"ctx-test"}' \
  | jq '.response' | grep -i "carbonio\|install"

# Expected: Response mentions Carbonio even though not in query
```

### 5. Cache Performance Validation

```bash
# Get initial cache stats
curl http://localhost:8000/api/database/cache-stats | jq '.retrieval_cache'

# Run 5 identical queries
for i in {1..5}; do
  curl -X POST http://localhost:8000/api/chat/query \
    -H "Content-Type: application/json" \
    -d "{\"message\":\"test\",\"session_id\":\"cache-test-$i\"}" \
    > /dev/null
done

# Check cache stats again
curl http://localhost:8000/api/database/cache-stats | jq '.retrieval_cache'

# Expected: hit_rate should increase after repeated queries
```

### 6. Stage Timing Validation

```bash
# Query and extract timing
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message":"test","session_id":"timing-test"}' \
  | jq '{
      stages: [.stages[] | {name, duration_ms}],
      total_duration_ms,
      calculated_total: ([.stages[].duration_ms] | add)
    }'

# Expected:
# - All 4 stages present with duration_ms
# - total_duration_ms equals sum of stage durations (±5ms tolerance)
# - All durations > 0
```

## Monitoring Setup

### 1. Health Check Endpoint

Add to monitoring system:

```bash
# Check every 60 seconds
curl http://localhost:8000/api/health

# Alert if non-200 status or neo4j: "disconnected"
```

### 2. Log Aggregation

**Docker Logs:**
```bash
# Tail all service logs
docker compose logs -f

# Filter by service
docker compose logs -f backend
docker compose logs -f frontend
```

**Production Logs:**
```bash
# Backend logs (gunicorn)
tail -f /var/log/amber/backend.log

# Frontend logs (next.js)
tail -f /var/log/amber/frontend.log

# Neo4j logs
tail -f /var/lib/neo4j/logs/neo4j.log
```

### 3. Metrics Collection

**Cache Metrics:**
```bash
# Poll cache stats every 5 minutes
*/5 * * * * curl -s http://localhost:8000/api/database/cache-stats \
  >> /var/log/amber/cache-metrics.log
```

**Stage Timing Metrics:**
```bash
# Extract stage timing from logs
docker logs graphrag-backend 2>&1 | grep "duration_ms" \
  | awk '{print $NF}' \
  | sort -n \
  | tail -10
```

### 4. Alerting

Configure alerts for:

1. **Critical:**
   - Health check failure
   - Neo4j disconnection
   - Total query duration >30s
   - Any stage duration >15s

2. **Warning:**
   - Cache hit rate <50% (entity label)
   - Cache hit rate <30% (embedding)
   - Cache hit rate <10% (retrieval)
   - Query duration >15s
   - Memory usage >80%

## Rollback Procedures

If issues occur, see [Rollback Procedures](rollback.md) for detailed steps.

**Quick Rollback:**
```bash
# Stop current deployment
docker compose down

# Switch to previous version
git checkout <previous-tag>

# Rebuild and start
docker compose up -d --build

# Verify health
curl http://localhost:8000/api/health
```

## Deployment Verification Checklist

Before marking deployment as complete, verify:

- [ ] All Docker containers running (docker compose ps)
- [ ] Health endpoint returns healthy status
- [ ] Cache stats endpoint returns data
- [ ] Database stats show expected counts
- [ ] Non-streaming query returns response with sources
- [ ] Streaming query emits SSE events with stages
- [ ] Follow-up query preserves conversation context
- [ ] Stage timing displayed in UI tooltips
- [ ] Total duration shown after completion
- [ ] Cache hit rates improving on repeated queries
- [ ] All 5 integration tests passing
- [ ] Frontend loads without errors
- [ ] Chat interface functional
- [ ] History panel shows conversations
- [ ] Upload UI accepts documents
- [ ] Database explorer displays data
- [ ] No secrets in logs or git history
- [ ] Monitoring alerts configured
- [ ] Log aggregation working
- [ ] Rollback procedure tested

## Related Documentation

- [Monitoring & Troubleshooting](monitoring.md)
- [Rollback Procedures](rollback.md)
- [Chat API Reference](06-api-reference/chat-api.md)
- [Configuration Reference](07-configuration/settings.md)
