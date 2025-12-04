# Deployment Guide

This guide explains how to deploy Amber in different environments.

## Quick Start

### Local Development (Default)

```bash
# Uses localhost:8000 for backend, localhost:3000 for frontend
docker compose up -d
```

The default `docker-compose.yml` is configured for local development with `localhost` URLs.

### Production Deployment

```bash
# Uses production URLs from docker-compose.prod.yml
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

The `docker-compose.prod.yml` file overrides the localhost defaults with production URLs.

## Environment Files

### docker-compose.yml (Default)
- **Purpose**: Local development
- **Frontend API URL**: `http://localhost:8000`
- **CORS Origins**: `["http://localhost:3000","http://localhost:3001"]`

### docker-compose.prod.yml (Override)
- **Purpose**: Production deployment
- **Frontend API URL**: `http://cph-01.demo.zextras.io:8000` (or your domain)
- **CORS Origins**: Includes production frontend URL

## Switching Between Environments

### From Local to Production

```bash
# Stop local containers
docker compose down

# Start with production overrides
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

**Note**: The `--build` flag is important when switching environments because it rebuilds the frontend with the correct `NEXT_PUBLIC_API_URL` embedded at build time.

### From Production to Local

```bash
# Stop production containers
docker compose -f docker-compose.yml -f docker-compose.prod.yml down

# Rebuild and start with local config
docker compose up -d --build
```

## Custom Production URLs

### Option 1: Modify docker-compose.prod.yml

Edit `docker-compose.prod.yml` to set your own domain:

```yaml
services:
  frontend:
    build:
      args:
        - NEXT_PUBLIC_API_URL=http://your-domain.com:8000
  
  backend:
    environment:
      - CORS_ORIGINS=["http://localhost:3000","http://your-domain.com:3000"]
```

### Option 2: Use Environment Variables

Set environment variables before running docker compose:

```bash
export NEXT_PUBLIC_API_URL=http://your-domain.com:8000
export CORS_ORIGINS='["http://localhost:3000","http://your-domain.com:3000"]'

docker compose up -d --build
```

### Option 3: Create .env File

Add to your `.env` file:

```bash
NEXT_PUBLIC_API_URL=http://your-domain.com:8000
CORS_ORIGINS=["http://localhost:3000","http://your-domain.com:3000"]
```

Then run:

```bash
docker compose up -d --build
```

## Remote Deployment (SSH)

For deploying to a remote server via SSH (like `cph-01.demo.zextras.io`):

```bash
# Sync files to remote
rsync -avz --delete \
  --exclude 'node_modules' \
  --exclude '.next' \
  --exclude 'data' \
  --exclude '.git' \
  ./ root@your-server.com:/root/amber/

# SSH into remote and deploy with production config
ssh root@your-server.com "cd /root/amber && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build"
```

## Important Notes

### Build-Time vs Runtime Configuration

- **`NEXT_PUBLIC_API_URL`**: This is a Next.js build-time variable. It gets embedded into the JavaScript bundles during `npm run build`. Changes require rebuilding with `--build` flag.

- **`CORS_ORIGINS`**: This is a runtime environment variable for the backend. Changes take effect after container restart (no rebuild needed).

### Cache Busting

When switching environments, you may need to clear Docker build cache:

```bash
docker builder prune -af
```

### Health Checks

After deployment, verify all services are healthy:

```bash
# Check container status
docker compose ps

# Check frontend
curl http://localhost:3000  # or your production URL

# Check backend health
curl http://localhost:8000/api/health  # or your production URL

# Check Neo4j
docker exec neo4j bin/cypher-shell -u neo4j -p your_password "RETURN 1"
```

## Troubleshooting

### Frontend shows "Failed to fetch" errors

**Problem**: Frontend making API calls to wrong URL (e.g., localhost when deployed remotely).

**Solution**: Rebuild frontend with correct environment:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build frontend
```

### Backend CORS errors

**Problem**: Backend rejecting requests from frontend origin.

**Solution**: Update `CORS_ORIGINS` in `docker-compose.prod.yml` or `.env` to include your frontend URL, then restart:
```bash
docker compose restart backend
```

### Database connection refused

**Problem**: Backend can't connect to Neo4j.

**Solution**: Check Neo4j is running and healthy:
```bash
docker compose ps neo4j
docker logs neo4j
```

## Production Checklist

Before deploying to production:

- [ ] Update `docker-compose.prod.yml` with your production domain
- [ ] Set strong `NEO4J_PASSWORD` in `.env`
- [ ] Set valid `OPENAI_API_KEY` in `.env`
- [ ] Review `CORS_ORIGINS` includes all necessary frontend origins
- [ ] Test locally first: `docker compose up -d`
- [ ] Deploy to production: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build`
- [ ] Verify all health checks pass
- [ ] Test frontend → backend API connectivity
- [ ] Test document upload and processing
- [ ] Monitor logs for errors: `docker compose logs -f`

## Makefile Shortcuts (Optional)

Create a `Makefile` for convenience:

```makefile
.PHONY: dev prod restart-dev restart-prod

dev:
	docker compose up -d --build

prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

restart-dev:
	docker compose down && docker compose up -d --build

restart-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml down
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Usage:
```bash
make dev   # Start local development
make prod  # Start production deployment
```
