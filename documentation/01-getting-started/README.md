# Getting Started

Initial setup and onboarding documentation for Amber.

## Contents

- [README](01-getting-started/README.md) - Quick start overview
- [Architecture Overview](01-getting-started/architecture-overview.md) - High-level system design and component interaction
- [Docker Setup](01-getting-started/docker-setup.md) - Docker Compose deployment instructions
- [Local Development](01-getting-started/local-development.md) - Setting up a local development environment
- [Configuration](01-getting-started/configuration.md) - Initial configuration and environment variables
 - [Scripts: Neo4j Setup](10-scripts/setup-neo4j.md) - Indexes, constraints, casefold, dedupe
 - [Scripts: Reindex Classification](10-scripts/reindex-classification.md) - Ingestion-time document classification commands

## Prerequisites

- Python 3.10 or 3.11
- Node.js 20+
- Docker and Docker Compose (for container deployment)
- Neo4j 5.x (included in Docker Compose or install separately)
- OpenAI API key or local LLM provider (Ollama)

## Quick Start

Fastest way to run Amber locally:

```bash
git clone https://github.com/danve93/graphrag-mrkr-2.git
cd graphrag-mrkr-2
cp .env.example .env
# Edit .env with your OpenAI API key and Neo4j credentials
docker compose up -d
```

Access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474

## Learning Path

1. **Understand the Architecture** - Read [Architecture Overview](01-getting-started/architecture-overview.md)
2. **Deploy with Docker** - Follow [Docker Setup](01-getting-started/docker-setup.md)
3. **Configure Environment** - Review [Configuration](01-getting-started/configuration.md)
4. **Explore Core Concepts** - Study [../02-core-concepts/](02-core-concepts)
5. **Try Local Development** - Set up using [Local Development](01-getting-started/local-development.md)

## Next Steps

After completing setup:
- Upload a document via the frontend at http://localhost:3000
- Try a chat query to test the RAG pipeline
- Explore the database view to see ingested chunks and entities
- Review [Chat Query Flow](05-data-flows/chat-query-flow.md) to understand the request lifecycle
- Consult [API Reference](06-api-reference) for programmatic access
 - Initialize Neo4j indexes and constraints: see [Scripts: Neo4j Setup](10-scripts/setup-neo4j.md)
 - Classify existing documents and enrich metadata: see [Scripts: Reindex Classification](10-scripts/reindex-classification.md)

## Common Setup Issues

**Neo4j Connection Failed**
- Verify `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` in `.env`
- Ensure Neo4j container is running: `docker compose ps`
- Check Neo4j logs: `docker compose logs neo4j`

**OpenAI API Errors**
- Confirm `OPENAI_API_KEY` is set in `.env`
- Verify API key has sufficient credits
- Check model availability: `OPENAI_MODEL` default is `gpt-4o-mini`

**Frontend Not Connecting to Backend**
- Ensure backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in frontend `.env.local`
- For Docker Compose, remove `frontend/.env.local` to use compose networking

See [Troubleshooting](08-operations/troubleshooting.md) for detailed solutions.
