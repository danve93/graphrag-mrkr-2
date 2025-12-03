# Docker Setup

Production-ready deployment using Docker Compose.

## Prerequisites

- Docker 20.10+ installed
- Docker Compose 2.0+ installed
- 8GB RAM minimum (16GB recommended)
- 20GB disk space

**Verify installation**:
```bash
docker --version
docker compose version
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/danve93/graphrag-mrkr-2.git
cd graphrag-mrkr-2
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
# OpenAI (required)
OPENAI_API_KEY=sk-proj-...

# Neo4j (use strong password in production)
NEO4J_PASSWORD=strongpassword123

# Optional: Ollama for local inference
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### 3. Start Services

```bash
docker compose up -d --build
```

This will:
- Build backend and frontend images
- Start Neo4j graph database
- Start backend API server
- Start frontend web server

### 4. Verify Deployment

**Check container status**:
```bash
docker compose ps
```

Expected output:
```
NAME                STATUS              PORTS
neo4j               Up 30 seconds       0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
backend             Up 20 seconds       0.0.0.0:8000->8000/tcp
frontend            Up 10 seconds       0.0.0.0:3000->3000/tcp
```

**Check health**:
```bash
curl http://localhost:8000/api/health
```

Expected: `{"status": "healthy"}`

**Access UI**:
Open browser to http://localhost:3000

## Docker Compose Configuration

### Default Configuration

File: `docker-compose.yml`

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15.0
    container_name: neo4j
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_server_memory_heap_initial__size: 2G
      NEO4J_server_memory_heap_max__size: 4G
      NEO4J_server_memory_pagecache_size: 2G
      NEO4J_dbms_security_procedures_unrestricted: apoc.*
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: backend
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USERNAME: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      neo4j:
        condition: service_healthy

  frontend:
    build:
      context: ./frontend
      dockerfile: ../Dockerfile.frontend
    container_name: frontend
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  neo4j_data:
  neo4j_logs:
```

### Override Configuration

File: `docker-compose.override.yml` (optional)

Use for local customizations without modifying main config:

```yaml
version: '3.8'

services:
  backend:
    environment:
      LOG_LEVEL: DEBUG
      ENABLE_CACHING: "true"
    volumes:
      - ./custom_data:/app/custom_data

  neo4j:
    environment:
      NEO4J_server_memory_heap_max__size: 8G
```

## Common Operations

### Start Services

```bash
docker compose up -d
```

### Stop Services

```bash
docker compose down
```

### Restart Single Service

```bash
docker compose restart backend
```

### Rebuild After Code Changes

```bash
docker compose up -d --build backend
```

### View Logs

**All services**:
```bash
docker compose logs -f
```

**Single service**:
```bash
docker compose logs -f backend
```

**Last 100 lines**:
```bash
docker compose logs --tail=100 backend
```

### Execute Commands in Container

```bash
docker compose exec backend bash
docker compose exec neo4j cypher-shell
```

### Scale Services

```bash
docker compose up -d --scale backend=3
```

## Data Persistence

### Volume Mounts

**Neo4j Data**: Persisted in named volume `neo4j_data`
**Document Files**: Mounted from `./data` directory
**Logs**: Mounted from `./logs` directory

### Backup Data

```bash
docker compose stop neo4j

docker run --rm \
  -v graphrag-mrkr-2_neo4j_data:/data \
  -v $(pwd)/backups:/backups \
  busybox tar -czf /backups/neo4j-backup-$(date +%Y%m%d).tar.gz -C /data .

docker compose start neo4j
```

### Restore Data

```bash
docker compose down

docker run --rm \
  -v graphrag-mrkr-2_neo4j_data:/data \
  -v $(pwd)/backups:/backups \
  busybox tar -xzf /backups/neo4j-backup-20250101.tar.gz -C /data

docker compose up -d
```

## Environment Variables

### Required

```bash
OPENAI_API_KEY          # OpenAI API key for embeddings and generation
NEO4J_PASSWORD          # Neo4j database password
```

### Optional

```bash
# LLM Configuration
OLLAMA_BASE_URL         # Ollama endpoint for local inference
LLM_PROVIDER            # "openai" or "ollama" (default: openai)
OPENAI_MODEL            # GPT model name (default: gpt-4o-mini)
OLLAMA_MODEL            # Ollama model name (default: llama3.1)

# Embedding Configuration
EMBEDDING_MODEL         # Embedding model (default: text-embedding-3-small)

# Neo4j Configuration
NEO4J_URI               # Neo4j connection string (default: bolt://neo4j:7687)
NEO4J_USERNAME          # Neo4j username (default: neo4j)

# Feature Flags
ENABLE_CACHING          # Enable multi-layer caching (default: true)
ENABLE_ENTITY_EXTRACTION # Enable entity extraction (default: true)
ENABLE_CLUSTERING       # Enable Leiden clustering (default: false)
FLASHRANK_ENABLED       # Enable FlashRank reranking (default: false)

# Performance Tuning
EMBEDDING_CONCURRENCY   # Concurrent embedding requests (default: 5)
NEO4J_MAX_CONNECTION_POOL_SIZE # Connection pool size (default: 50)
```

## Networking

### Default Network

Docker Compose creates a default network where services can communicate:

```
backend     → neo4j      (bolt://neo4j:7687)
frontend    → backend    (http://backend:8000)
host        → frontend   (http://localhost:3000)
host        → backend    (http://localhost:8000)
host        → neo4j      (http://localhost:7474, bolt://localhost:7687)
```

### Custom Network

```yaml
networks:
  amber-network:
    driver: bridge

services:
  neo4j:
    networks:
      - amber-network
  backend:
    networks:
      - amber-network
  frontend:
    networks:
      - amber-network
```

## Troubleshooting

### Container Won't Start

**Check logs**:
```bash
docker compose logs backend
```

**Common issues**:
- Missing environment variables
- Port conflicts (3000, 8000, 7474, 7687)
- Insufficient memory
- Docker daemon not running

### Neo4j Connection Errors

**Verify Neo4j is healthy**:
```bash
docker compose exec neo4j cypher-shell -u neo4j -p yourpassword "RETURN 1;"
```

**Check environment variables**:
```bash
docker compose exec backend env | grep NEO4J
```

**Verify network connectivity**:
```bash
docker compose exec backend ping neo4j
```

### Out of Memory

**Increase Docker memory limit**: Docker Desktop > Settings > Resources > Memory

**Reduce Neo4j heap**:
```yaml
environment:
  NEO4J_server_memory_heap_max__size: 2G
```

### Slow Performance

**Check resource usage**:
```bash
docker stats
```

**Enable caching**:
```bash
ENABLE_CACHING=true docker compose up -d backend
```

**Optimize Neo4j memory**:
```yaml
environment:
  NEO4J_server_memory_heap_max__size: 4G
  NEO4J_server_memory_pagecache_size: 4G
```

### Permission Errors

**Fix volume permissions**:
```bash
sudo chown -R $(id -u):$(id -g) data/ logs/
```

## Production Considerations

### Security

**Use secrets management**:
```yaml
services:
  backend:
    secrets:
      - openai_key
    environment:
      OPENAI_API_KEY_FILE: /run/secrets/openai_key

secrets:
  openai_key:
    external: true
```

**Restrict Neo4j ports**:
```yaml
ports:
  - "127.0.0.1:7474:7474"
  - "127.0.0.1:7687:7687"
```

**Use HTTPS**: Place reverse proxy (Nginx, Traefik) in front of services

### High Availability

**Neo4j Cluster**:
```yaml
services:
  neo4j-core-1:
    image: neo4j:5.15.0-enterprise
    environment:
      NEO4J_dbms_mode: CORE
      NEO4J_causal__clustering_initial__discovery__members: neo4j-core-1:5000,neo4j-core-2:5000,neo4j-core-3:5000

  neo4j-core-2:
    # Similar configuration

  neo4j-core-3:
    # Similar configuration
```

**Backend Replicas**:
```bash
docker compose up -d --scale backend=3
```

### Monitoring

**Add Prometheus exporter**:
```yaml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
```

## Next Steps

- [Local Development Setup](01-getting-started/local-development.md) - Development workflow without Docker
- [Configuration Guide](01-getting-started/configuration.md) - Detailed configuration options
- [Operations Guide](08-operations) - Production deployment strategies

## Related Documentation

- [Architecture Overview](01-getting-started/architecture-overview.md)
- [Configuration](01-getting-started/configuration.md)
- [Deployment](08-operations/deployment.md)
