# Operations

Production deployment, monitoring, and maintenance guides.

## Contents

- [README](08-operations/README.md) - Operations overview
- [Deployment](08-operations/deployment.md) - Production deployment strategies
- [Monitoring](08-operations/monitoring.md) - Logging, metrics, and observability
- [Performance Tuning](08-operations/monitoring.md) - Optimization techniques
- [Troubleshooting](08-operations/troubleshooting.md) - Common issues and solutions
- [Backup and Restore](08-operations/maintenance-reindexing.md) - Data persistence strategies
- [Scaling](08-operations/deployment.md) - Horizontal and vertical scaling

## Operations Overview

This section covers operational aspects of running Amber in production environments.

## Quick Operations Guide

### Health Checks

**Backend Health**:
```bash
curl http://localhost:8000/api/health
```

**Neo4j Health**:
```bash
curl http://localhost:7474
```

**Full System Check**:
```bash
./scripts/test-startup.sh quick
```

### Monitoring Endpoints

- **Health**: `GET /api/health`
- **Database Stats**: `GET /api/database/stats`
- **Cache Metrics**: `GET /api/database/cache-stats`
- **Job Status**: `GET /api/jobs`

### Log Access

**Docker Compose**:
```bash
docker compose logs backend -f
docker compose logs frontend -f
docker compose logs neo4j -f
```

**Local Development**:
```bash
tail -f backend.log
tail -f frontend-dev.log
```

## Deployment Strategies

### Docker Compose (Recommended)

Simple single-host deployment:
```bash
docker compose up -d --build
```

Advantages:
- Easy setup and teardown
- Consistent environments
- Built-in networking
- Volume persistence

### Kubernetes

For production scale:
- StatefulSet for Neo4j
- Deployment for backend (replicas)
- Deployment for frontend
- Ingress for routing
- PersistentVolumes for data

See [Deployment](08-operations/deployment.md) for manifests.

### Bare Metal

Direct installation:
```bash
python api/main.py &
cd frontend && npm start &
```

Requires:
- Process manager (systemd, supervisor)
- Reverse proxy (Nginx, Traefik)
- Manual Neo4j installation

## Monitoring

### Key Metrics

**System Health**:
- Backend response time
- Neo4j query latency
- Memory usage
- CPU utilization

**Application Metrics**:
- Cache hit rates (entity, embedding, retrieval, response)
- Ingestion throughput (documents/minute)
- Query latency (p50, p95, p99)
- Error rates

**Business Metrics**:
- Active sessions
- Documents ingested
- Queries per day
- Average response quality

### Monitoring Tools

**Built-in**:
- `/api/health` - System health
- `/api/database/stats` - Database statistics
- `/api/database/cache-stats` - Cache performance

**External**:
- Prometheus for metrics
- Grafana for dashboards
- ELK stack for log aggregation
- Sentry for error tracking

See [Monitoring](08-operations/monitoring.md) for setup instructions.

## Performance Tuning

### Database Optimization

**Neo4j Configuration** (`neo4j.conf`):
```properties
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
dbms.memory.pagecache.size=4G
db.index.fulltext.default_analyzer=standard
```

**Connection Pooling**:
```bash
NEO4J_MAX_CONNECTION_POOL_SIZE=50
```

### Cache Tuning

Adjust cache sizes based on workload:
```bash
ENTITY_LABEL_CACHE_SIZE=5000
EMBEDDING_CACHE_SIZE=10000
RETRIEVAL_CACHE_SIZE=1000
RESPONSE_CACHE_SIZE=2000
```

Monitor hit rates and increase sizes if <50%.

### Ingestion Optimization

**Concurrency**:
```bash
EMBEDDING_CONCURRENCY=5
```

**Batching**:
```bash
NEO4J_UNWIND_BATCH_SIZE=500
```

**Feature Flags**:
```bash
ENABLE_PHASE2_NETWORKX=true
SYNC_ENTITY_EMBEDDINGS=false
```

See [Performance Tuning](08-operations/monitoring.md) for detailed analysis.

## Troubleshooting

### Common Issues

**Neo4j Connection Failed**:
1. Check Neo4j is running: `docker compose ps neo4j`
2. Verify credentials in `.env`
3. Check network connectivity
4. Review Neo4j logs: `docker compose logs neo4j`

**High Memory Usage**:
1. Check cache sizes (reduce if needed)
2. Review Neo4j heap settings
3. Monitor ingestion batch sizes
4. Consider horizontal scaling

**Slow Queries**:
1. Check cache hit rates
2. Review graph expansion depth
3. Enable reranking
4. Optimize Neo4j indexes

**API Errors**:
1. Check API key validity
2. Review rate limits
3. Monitor API quotas
4. Check error logs

See [Troubleshooting](08-operations/troubleshooting.md) for comprehensive guide.

## Backup and Restore

### Neo4j Backup

**Online Backup** (Enterprise):
```bash
neo4j-admin backup --from=localhost:6362 --backup-dir=/backups
```

**Offline Backup** (Community):
```bash
docker compose stop neo4j
tar -czf neo4j-backup.tar.gz neo4j/data/
docker compose start neo4j
```

### Document Files Backup

```bash
tar -czf documents-backup.tar.gz data/documents/ data/extracted/
```

### Configuration Backup

```bash
cp .env .env.backup
tar -czf config-backup.tar.gz config/
```

See [Backup and Restore](08-operations/maintenance-reindexing.md) for detailed procedures.

## Scaling

### Vertical Scaling

Increase resources:
- Neo4j heap: 4GB → 8GB
- Backend workers: 4 → 8
- Connection pool: 50 → 100

### Horizontal Scaling

**Backend Replicas**:
```bash
docker compose up -d --scale backend=3
```

**Load Balancing**:
- Nginx upstream
- Kubernetes Service
- AWS Application Load Balancer

**Neo4j Clustering**:
- Core servers (consensus)
- Read replicas (queries)
- Causal clustering

See [Scaling](08-operations/deployment.md) for architecture patterns.

## Maintenance

### Regular Tasks

**Daily**:
- Monitor health endpoints
- Review error logs
- Check disk space

**Weekly**:
- Review cache performance
- Analyze query latency
- Update API keys if needed

**Monthly**:
- Database backups
- Dependency updates
- Security patches
- Performance review

### Maintenance Windows

For major updates:
1. Announce downtime
2. Backup data
3. Deploy updates
4. Run smoke tests
5. Monitor for issues
6. Rollback if needed

## Security Operations

### Access Control

- Restrict Neo4j ports (7474, 7687)
- Enable HTTPS for API
- Use strong passwords
- Rotate API keys regularly

### Monitoring

- Failed login attempts
- Unusual query patterns
- API key usage
- Database access logs

### Updates

- Regular dependency updates
- Security patch monitoring
- CVE tracking
- Automated scanning

## Related Documentation

- [Deployment Guide](08-operations/deployment.md)
- [Performance Tuning](08-operations/monitoring.md)
- [Troubleshooting](08-operations/troubleshooting.md)
- [Configuration](07-configuration)
