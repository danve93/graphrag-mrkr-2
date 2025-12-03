# Rollback Procedures

## Overview

This document provides step-by-step procedures for rolling back the Amber RAG system in case of deployment issues or regressions after implementing Milestones 1-5 (conversation context, enhanced caching, stage timing).

## When to Rollback

### Critical Issues (Immediate Rollback)

- System health check failures persisting >5 minutes
- Neo4j connection failures affecting all queries
- Complete cache failure causing 10x+ performance degradation
- Stage timing causing system crashes or memory leaks
- Data corruption or loss
- Security vulnerabilities discovered

### Warning Issues (Evaluate Rollback)

- Query latency >30s consistently
- Cache hit rates <10% across all caches
- Stage timing accuracy issues (>1000ms discrepancy)
- Conversation context not preserved
- UI timing display errors

## Pre-Rollback Checklist

Before rolling back, gather diagnostic information:

```bash
# 1. Capture current health status
curl http://localhost:8000/api/health > rollback_health.json

# 2. Capture database stats
curl http://localhost:8000/api/database/stats > rollback_db_stats.json

# 3. Capture cache stats
curl http://localhost:8000/api/database/cache-stats > rollback_cache_stats.json

# 4. Capture recent logs
docker logs graphrag-backend --tail 1000 > rollback_backend.log
docker logs graphrag-frontend --tail 1000 > rollback_frontend.log
docker logs graphrag-neo4j --tail 1000 > rollback_neo4j.log

# 5. Export conversations (optional, for data preservation)
# curl http://localhost:8000/api/history/export > rollback_history.json

# 6. Note current git commit
git rev-parse HEAD > rollback_commit.txt
```

## Rollback Methods

### Method 1: Docker Compose Rollback (Fastest)

**Use Case:** Running via Docker Compose, need immediate rollback

**Steps:**

1. **Stop current deployment:**
```bash
docker compose down
```

2. **Identify previous stable version:**
```bash
# List recent tags
git tag --sort=-v:refname | head -10

# Or identify commit before M1-M5
git log --oneline --decorate | head -20
```

3. **Checkout previous version:**
```bash
# Option A: Use git tag
git checkout v1.0.0  # Replace with your stable tag

# Option B: Use specific commit
git checkout abc123def  # Commit before M1-M5 changes
```

4. **Rebuild and restart:**
```bash
docker compose up -d --build
```

5. **Verify rollback:**
```bash
# Wait for services
sleep 15

# Check health
curl http://localhost:8000/api/health

# Run basic query
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message":"test","session_id":"rollback-test"}' \
  | jq '.response'
```

**Expected Time:** 3-5 minutes

### Method 2: Feature Flag Rollback (Partial)

**Use Case:** Issues with specific features, want to disable without full rollback

**Steps:**

1. **Disable caching (M1-M2 features):**
```bash
# Edit .env or set environment variable
export ENABLE_CACHING=false

# Restart backend
docker compose up -d --build backend
```

2. **Disable stage timing (M3-M4 features):**
```bash
# Edit config/settings.py (requires code change)
# Set ENABLE_STAGE_TIMING=false (add if needed)

# Or modify API to skip timing emission
# Restart backend
docker compose up -d --build backend
```

3. **Verify partial rollback:**
```bash
# Check cache stats (should show disabled)
curl http://localhost:8000/api/database/cache-stats

# Run query and check response
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message":"test","session_id":"partial-test"}' \
  | jq '{response, stages}'
```

**Expected Time:** 1-2 minutes

### Method 3: Database Rollback (Data Recovery)

**Use Case:** Database corruption or schema issues

**Steps:**

1. **Stop services:**
```bash
docker compose down
```

2. **Restore Neo4j backup:**
```bash
# Locate backup
ls -lh /var/lib/neo4j/data/backups/

# Restore from backup
docker run --rm \
  -v neo4j_data:/data \
  -v /path/to/backup:/backups \
  neo4j:5.21.2 \
  neo4j-admin database restore \
  --from=/backups/neo4j-backup-2024-12-01.dump \
  neo4j
```

3. **Restart services:**
```bash
docker compose up -d
```

4. **Verify database:**
```bash
curl http://localhost:8000/api/database/stats
```

**Expected Time:** 5-15 minutes (depends on database size)

### Method 4: Production Rollback (Non-Docker)

**Use Case:** Production deployment using gunicorn/systemd

**Steps:**

1. **Stop services:**
```bash
sudo systemctl stop amber-backend
sudo systemctl stop amber-frontend
```

2. **Switch to previous version:**
```bash
cd /opt/amber
git fetch --tags
git checkout v1.0.0  # Previous stable version
```

3. **Reinstall dependencies:**
```bash
# Backend
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm ci
npm run build
```

4. **Restart services:**
```bash
sudo systemctl start amber-backend
sudo systemctl start amber-frontend
```

5. **Verify health:**
```bash
curl http://localhost:8000/api/health
curl http://localhost:3000
```

**Expected Time:** 5-10 minutes

## Post-Rollback Validation

After rollback, verify system functionality:

### 1. Health Checks

```bash
# Backend health
curl http://localhost:8000/api/health
# Expected: {"status":"healthy","neo4j":"connected"}

# Database stats
curl http://localhost:8000/api/database/stats
# Expected: Valid counts for documents, chunks, entities

# Frontend health
curl http://localhost:3000
# Expected: 200 OK
```

### 2. Functional Testing

```bash
# Basic query test
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What is Carbonio?","session_id":"validation-1"}' \
  | jq '{
      response: .response[:100],
      sources: (.sources | length),
      session_id
    }'

# Expected: Valid response with sources
```

### 3. Performance Validation

```bash
# Measure query time
time curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message":"test query","session_id":"perf-1"}' \
  > /dev/null

# Expected: <15 seconds (without M1-M5 optimizations)
```

### 4. UI Validation

Manual checks:
1. Open http://localhost:3000
2. Send a test message
3. Verify:
   - ✅ Response appears
   - ✅ Sources displayed (if available)
   - ✅ No JavaScript errors in console
   - ✅ Chat history works
   - ✅ Upload functionality intact

## Feature-Specific Rollback

### Conversation Context (M1)

If only conversation context is problematic:

**Option 1: Clear History**
```bash
# Clear all conversation history
docker exec -it graphrag-neo4j cypher-shell -u neo4j -p <password> \
  "MATCH (m:Message) DELETE m"

# Or via API (if endpoint exists)
curl -X DELETE http://localhost:8000/api/history/all
```

**Option 2: Disable Context Preservation**
```python
# In rag/graph_rag.py, modify query_analysis node
# Comment out conversation history retrieval:

# def query_analysis(state: dict) -> dict:
#     # ... existing code ...
#     # conversation_history = get_conversation_history(session_id)  # Disable
#     state["conversation_history"] = []  # Force empty
#     return state
```

### Enhanced Cache Keys (M2)

If cache key generation causes issues:

**Option 1: Simplify Cache Key**
```python
# In rag/retriever.py, modify cache_key generation
# Reduce from 14 parameters to essential only:

cache_key = f"{query_embedding_str}:{retrieval_mode}:{top_k}"
# Remove expansion params, weights, RRF, etc.
```

**Option 2: Disable Retrieval Cache**
```bash
# Keep entity/embedding caches, disable retrieval cache only
export RETRIEVAL_CACHE_SIZE=0

# Or in config/settings.py
RETRIEVAL_CACHE_SIZE = 0
```

### Stage Timing (M3-M4)

If timing instrumentation causes issues:

**Option 1: Disable Timing Emission**
```python
# In rag/nodes/*.py, comment out timing tracking
# def query_analysis(state: dict) -> dict:
#     start_time = time.time()  # Keep for logging
#     # ... logic ...
#     # duration_ms = int((time.time() - start_time) * 1000)  # Disable
#     # state["stages"].append({"name": "Query Analysis", "duration_ms": duration_ms})  # Disable
#     return state
```

**Option 2: Frontend Timing Fallback**
```typescript
// In frontend/src/components/Chat/LoadingIndicator.tsx
// Remove timing display:

// Old: "Retrieving Documents (456ms) - 5 chunks"
// New: "Retrieving Documents"
const label = stage.name;  // Remove duration display
```

## Data Integrity Validation

After rollback, verify data integrity:

### 1. Database Consistency

```cypher
// Connect to Neo4j browser
// Check for orphaned nodes

// Chunks without documents
MATCH (c:Chunk)
WHERE NOT EXISTS((c)-[:PART_OF]->(:Document))
RETURN count(c) AS orphaned_chunks;
// Expected: 0

// Entities without relationships
MATCH (e:Entity)
WHERE NOT EXISTS((e)--())
RETURN count(e) AS isolated_entities;
// Expected: Small number or 0

// Duplicate entities
MATCH (e1:Entity), (e2:Entity)
WHERE e1.label = e2.label AND id(e1) < id(e2)
RETURN count(*) AS duplicates;
// Expected: 0 (if deduplication worked)
```

### 2. Cache Consistency

```bash
# Clear all caches after rollback
curl -X POST http://localhost:8000/api/database/clear-caches

# Or restart backend to reset in-memory caches
docker compose restart backend
```

### 3. Conversation History

```bash
# Check history for specific session
curl http://localhost:8000/api/history/test-session

# Verify structure matches expected format
# If corrupted, clear problematic sessions
```

## Monitoring After Rollback

### 1. Error Rate Monitoring

```bash
# Monitor logs for errors
docker logs -f graphrag-backend | grep -i "error"

# Count errors per minute
docker logs graphrag-backend --since 10m 2>&1 | grep -c "ERROR"
```

### 2. Performance Baseline

Establish new baseline after rollback:

```bash
# Run 10 test queries and measure timing
for i in {1..10}; do
  time curl -X POST http://localhost:8000/api/chat/query \
    -H "Content-Type: application/json" \
    -d "{\"message\":\"test $i\",\"session_id\":\"baseline-$i\"}" \
    > /dev/null
done

# Average response time should be <15s
```

### 3. Cache Performance (If Still Enabled)

```bash
# Monitor cache hit rates
watch -n 30 'curl -s http://localhost:8000/api/database/cache-stats | jq'

# Expected hit rates after rollback:
# - Entity: 60-70% (may be lower than M1-M5)
# - Embedding: 30-50%
# - Retrieval: 10-20%
```

## Communication Template

Use this template to communicate rollback to team:

```
Subject: Amber RAG System Rollback Notification

Date: [Date/Time]
Performed By: [Name]
Reason: [Brief description of issue]

ROLLBACK DETAILS:
- Previous Version: [Git commit/tag]
- Rolled Back To: [Git commit/tag]
- Method Used: [Docker Compose / Feature Flag / Database / Production]
- Duration: [X minutes]

IMPACT:
- Services Down: [Duration]
- Data Loss: [None / Minimal / Description]
- User Impact: [Description]

VALIDATION:
✅ Health checks passing
✅ Basic queries working
✅ Database integrity verified
✅ UI functional
⚠️ [Any known limitations after rollback]

NEXT STEPS:
1. [Action item 1]
2. [Action item 2]
3. [Root cause analysis scheduled]

MONITORING:
- Logs: [Location]
- Metrics: [Dashboard URL]
- On-call: [Contact]
```

## Root Cause Analysis

After successful rollback, conduct root cause analysis:

1. **Review diagnostic data** collected pre-rollback
2. **Analyze logs** for error patterns
3. **Identify trigger** (code change, config, data, load)
4. **Document issue** in GitHub/Jira
5. **Plan fix** with testing strategy
6. **Schedule re-deployment** with additional safeguards

## Prevention Strategies

To minimize future rollbacks:

### 1. Staging Environment

Always deploy to staging first:
```bash
# Deploy to staging
docker compose -f docker-compose.staging.yml up -d --build

# Run full test suite
pytest tests/ -v
python scripts/test_integration_m5.py

# Manual validation
# ... test key workflows ...

# Deploy to production only after staging validation
```

### 2. Gradual Rollout

Use feature flags for gradual rollout:
```python
# config/settings.py
ENABLE_M1_CONVERSATION_CONTEXT = os.getenv("ENABLE_M1", "true").lower() == "true"
ENABLE_M2_ENHANCED_CACHING = os.getenv("ENABLE_M2", "true").lower() == "true"
ENABLE_M3_STAGE_TIMING = os.getenv("ENABLE_M3", "true").lower() == "true"

# Start with 10% of users, monitor, increase gradually
```

### 3. Automated Rollback Triggers

Set up automated monitoring that triggers rollback:
```bash
#!/bin/bash
# scripts/auto_rollback.sh

# Check health every 60s
while true; do
  health=$(curl -s http://localhost:8000/api/health | jq -r '.status')
  
  if [ "$health" != "healthy" ]; then
    echo "Health check failed, initiating rollback"
    docker compose down
    git checkout <stable-version>
    docker compose up -d --build
    exit 1
  fi
  
  sleep 60
done
```

### 4. Database Backups

Automate regular backups:
```bash
#!/bin/bash
# scripts/backup_neo4j.sh

timestamp=$(date +%Y%m%d_%H%M%S)
docker exec graphrag-neo4j neo4j-admin database dump neo4j \
  --to=/backups/neo4j-backup-${timestamp}.dump

# Retention: keep last 7 days
find /var/backups/neo4j -name "*.dump" -mtime +7 -delete
```

## Related Documentation

- [Deployment Checklist](deployment.md)
- [Monitoring & Troubleshooting](monitoring.md)
- [Chat API Reference](06-api-reference/chat-api.md)
- [Configuration Reference](07-configuration/settings.md)

## Emergency Contacts

- **On-Call Engineer:** [Contact Info]
- **Database Admin:** [Contact Info]
- **DevOps Lead:** [Contact Info]
- **Product Owner:** [Contact Info]

## Rollback History

| Date | Version | Reason | Duration | Outcome |
|------|---------|--------|----------|---------|
| - | - | - | - | - |
