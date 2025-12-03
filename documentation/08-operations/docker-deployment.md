# Docker Deployment

Guide to deploy Amber using Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- `.env` configured (see Configuration section)

## Compose Files

- `docker-compose.yml` - Base services (backend, frontend, neo4j)
- `docker-compose.override.yml` - Local overrides
- `docker-compose.e2e.yml` - End-to-end testing stack

## Commands

```bash
# Build and start all services
docker compose up -d --build

# Build specific services
docker compose up -d --build backend frontend

# View logs
docker compose logs -f backend

# Restart a service
docker compose restart frontend

# Stop and remove containers
docker compose down

# Remove volumes (CAUTION)
docker compose down -v
```

## Environment Variables

Compose loads `.env` automatically. Key vars:
- `OPENAI_API_KEY`
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- `FLASHRANK_CACHE_DIR`

## Ports

- Backend API: `http://localhost:8000`
- Frontend: `http://localhost:3000`
- Neo4j Browser: `http://localhost:7474`

## Health Checks

```bash
# Backend health
curl http://localhost:8000/api/database/health

# Cache stats
curl http://localhost:8000/api/database/cache-stats

# Documents
curl http://localhost:8000/api/documents
```

## Common Issues

- "OpenAI API key missing": set `OPENAI_API_KEY` in `.env`
- Neo4j connection refused: check `NEO4J_URI`, ensure container running
- Frontend blank page: ensure backend reachable at `localhost:8000`
- FlashRank slow start: prewarm with `scripts/flashrank_prewarm_worker.py`

## Rebuild Images

```bash
# After code or Dockerfile changes
docker compose build backend frontend

docker compose up -d
```

## Data Persistence

Neo4j data volume persists between runs. Use `down -v` to clear.

```bash
# CAUTION: wipes database
docker compose down -v
```
