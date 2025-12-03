# Local Development Setup

Development environment without Docker for active development.

## Prerequisites

### Required Software

**Python**:
- Version: 3.10 or 3.11
- Verify: `python3 --version`

**Node.js**:
- Version: 20.x or later
- Verify: `node --version`

**Neo4j**:
- Option A: Docker container (recommended)
- Option B: Local installation (download from neo4j.com)

**Git**:
- Latest stable version
- Verify: `git --version`

### Development Tools (Recommended)

**IDE**: Visual Studio Code with extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- ESLint (dbaeumer.vscode-eslint)
- Prettier (esbenp.prettier-vscode)

**CLI Tools**:
- curl or HTTPie for API testing
- jq for JSON parsing
- Neo4j Desktop (optional GUI)

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/danve93/graphrag-mrkr-2.git
cd graphrag-mrkr-2
```

### 2. Backend Setup

**Create virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

**Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Install development dependencies** (optional):
```bash
pip install pytest pytest-asyncio pytest-cov black isort ruff mypy
```

### 3. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### 4. Database Setup

**Option A: Docker Neo4j** (recommended):
```bash
docker run -d \
  --name neo4j-dev \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/devpassword \
  -e NEO4J_server_memory_heap_max__size=2G \
  -v neo4j-dev-data:/data \
  neo4j:5.15.0
```

**Option B: Neo4j Desktop**:
1. Download from https://neo4j.com/download/
2. Create new database
3. Set password
4. Start database

**Verify connection**:
```bash
curl http://localhost:7474
```

### 5. Environment Configuration

**Copy example file**:
```bash
cp .env.example .env
```

**Edit `.env`**:
```bash
# OpenAI (required for embeddings and generation)
OPENAI_API_KEY=sk-proj-your-key-here

# Neo4j (match your setup)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=devpassword

# LLM Provider (openai or ollama)
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini

# Embedding Model
EMBEDDING_MODEL=text-embedding-3-small

# Development Settings
LOG_LEVEL=DEBUG
ENABLE_CACHING=true
ENABLE_ENTITY_EXTRACTION=true

# Optional: Local Ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.1
```

**Frontend environment** (create `frontend/.env.local`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Application

### Start Services

**Terminal 1 - Backend**:
```bash
source .venv/bin/activate
python api/main.py
```

Expected output:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Frontend**:
```bash
cd frontend
npm run dev
```

Expected output:
```
▲ Next.js 14.x
- Local:        http://localhost:3000
- Network:      http://192.168.1.x:3000

Ready in 2.5s
```

### Verify Installation

**Backend health**:
```bash
curl http://localhost:8000/api/health
```

**Database connection**:
```bash
curl http://localhost:8000/api/database/stats
```

**Frontend**: Open http://localhost:3000 in browser

## Development Workflow

### Code Changes

**Backend**: Changes auto-reload (Uvicorn `--reload` flag)
```bash
# Edit core/embeddings.py
# Save file
# Server automatically restarts
```

**Frontend**: Changes auto-reload (Next.js Fast Refresh)
```bash
# Edit src/components/Chat.tsx
# Save file
# Browser automatically updates
```

### Testing Backend

**Run all tests**:
```bash
pytest tests/
```

**Run specific test file**:
```bash
pytest tests/unit/test_chunking.py -v
```

**Run with coverage**:
```bash
pytest --cov=core --cov=rag --cov=ingestion tests/
```

**Run specific test**:
```bash
pytest tests/unit/test_chunking.py::test_chunk_overlap -v
```

### Testing Frontend

**Run tests**:
```bash
cd frontend
npm test
```

**Run with coverage**:
```bash
npm run test:coverage
```

**Watch mode**:
```bash
npm run test:watch
```

### Linting and Formatting

**Backend**:
```bash
# Format with Black
black .

# Sort imports
isort .

# Lint with Ruff
ruff check .

# Type check
mypy .
```

**Frontend**:
```bash
cd frontend

# Lint
npm run lint

# Format
npm run format

# Type check
npm run type-check
```

### Database Operations

**Access Neo4j Browser**: http://localhost:7474

**Run Cypher queries**:
```bash
docker exec -it neo4j-dev cypher-shell -u neo4j -p devpassword

# Or use Neo4j Browser UI
```

**Clear database**:
```cypher
MATCH (n) DETACH DELETE n;
```

**View document count**:
```cypher
MATCH (d:Document) RETURN count(d);
```

## Development Tools

### API Testing

**HTTPie examples**:
```bash
# Health check
http GET localhost:8000/api/health

# List documents
http GET localhost:8000/api/documents

# Upload document
http --form POST localhost:8000/api/documents/upload file@sample.pdf

# Chat query
http POST localhost:8000/api/chat \
  message="What is the main topic?" \
  session_id="test-session"
```

**cURL examples**:
```bash
# Health check
curl http://localhost:8000/api/health

# List documents
curl http://localhost:8000/api/documents

# Upload document
curl -F "file=@sample.pdf" http://localhost:8000/api/documents/upload
```

### Interactive Python

```bash
source .venv/bin/activate
python

>>> from core.graph_db import get_db
>>> db = get_db()
>>> stats = db.get_database_stats()
>>> print(stats)
```

### Frontend Development Tools

**React DevTools**: Install browser extension

**Redux DevTools**: For inspecting Zustand state

**Network inspection**: Browser DevTools > Network tab

## Common Development Tasks

### Add New API Endpoint

1. Define Pydantic model in `api/models.py`
2. Create router function in `api/routers/`
3. Register router in `api/main.py`
4. Add tests in `tests/`
5. Update API documentation

### Add New Frontend Component

1. Create component in `src/components/`
2. Add TypeScript interface
3. Import and use in pages
4. Add tests
5. Update Storybook (if available)

### Modify RAG Pipeline

1. Edit node function in `rag/nodes/`
2. Update state type if needed
3. Register node in `rag/graph_rag.py`
4. Add tests
5. Update configuration in `config/rag_tuning_config.json`

### Add New Entity Type

1. Add type to `CANONICAL_ENTITY_TYPES` in `core/entity_extraction.py`
2. Update extraction prompt
3. Test extraction
4. Update frontend entity type filters
5. Document in taxonomy

## Debugging

### Backend Debugging

**Print debugging**:
```python
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Variable value: {value}")
```

**Interactive debugger**:
```python
import pdb; pdb.set_trace()
```

**VSCode debugging** (`.vscode/launch.json`):
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "api.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
      ],
      "jinja": true,
      "justMyCode": false
    }
  ]
}
```

### Frontend Debugging

**Browser console**:
```typescript
console.log('Debug value:', value)
console.error('Error occurred:', error)
```

**React DevTools**: Inspect component props and state

**VSCode debugging**:
```json
{
  "name": "Next.js: debug",
  "type": "node",
  "request": "launch",
  "runtimeExecutable": "npm",
  "runtimeArgs": ["run", "dev"],
  "port": 9229,
  "console": "integratedTerminal"
}
```

### Database Debugging

**Query execution plan**:
```cypher
EXPLAIN MATCH (c:Chunk) WHERE c.text CONTAINS 'keyword' RETURN c;
```

**Profile query performance**:
```cypher
PROFILE MATCH (c:Chunk) WHERE c.text CONTAINS 'keyword' RETURN c;
```

**Connection debugging**:
```bash
docker logs neo4j-dev
```

## Performance Optimization

### Development Mode

**Disable caching** (for testing cache-bypass):
```bash
ENABLE_CACHING=false python api/main.py
```

**Reduce concurrency** (for debugging):
```bash
EMBEDDING_CONCURRENCY=1
LLM_CONCURRENCY=1
```

**Skip entity extraction** (for faster ingestion):
```bash
ENABLE_ENTITY_EXTRACTION=false
```

### Hot Reload Performance

**Backend**: Uvicorn auto-reload can be slow for large codebases. Use `--reload-dir` to limit:
```bash
uvicorn api.main:app --reload --reload-dir api --reload-dir core
```

**Frontend**: Next.js Fast Refresh is generally fast. If slow, check for circular imports.

## Troubleshooting

### Backend Won't Start

**Check Python version**:
```bash
python --version  # Should be 3.10 or 3.11
```

**Check dependencies**:
```bash
pip list | grep -E "fastapi|langgraph|neo4j"
```

**Check port availability**:
```bash
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows
```

### Frontend Won't Start

**Clear cache**:
```bash
cd frontend
rm -rf .next node_modules
npm install
npm run dev
```

**Check Node version**:
```bash
node --version  # Should be 20.x+
```

### Neo4j Connection Failed

**Check Neo4j is running**:
```bash
docker ps | grep neo4j
# or
curl http://localhost:7474
```

**Verify credentials**:
```bash
docker exec neo4j-dev cypher-shell -u neo4j -p devpassword "RETURN 1;"
```

**Check environment variables**:
```bash
grep NEO4J .env
```

### Import Errors

**Check PYTHONPATH**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Verify virtual environment**:
```bash
which python  # Should point to .venv/bin/python
```

## Next Steps

- [Configuration Guide](01-getting-started/configuration.md) - Detailed configuration options
- [Testing Guide](09-development/testing.md) - Comprehensive testing strategies
- [Contributing](09-development/contributing.md) - Contribution guidelines

## Related Documentation

- [Architecture Overview](01-getting-started/architecture-overview.md)
- [Docker Setup](01-getting-started/docker-setup.md)
- [Development Guide](09-development)
