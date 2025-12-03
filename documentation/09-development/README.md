# Development

Developer workflows, testing, and contribution guidelines.

## Contents

- [README](09-development/README.md) - Development overview
- [Coding Standards](09-development/coding-standards.md) - Coding standards and patterns
- [Testing Backend](09-development/testing-backend.md) - Backend test organization and procedures
- [Testing Frontend](09-development/testing-frontend.md) - Frontend test organization and procedures
- [Feature Flag Wiring](09-development/feature-flag-wiring.md) - Feature flag verification and testing
- [Dev Scripts](09-development/dev-scripts.md) - Development scripts and utilities
- [Contributing](09-development/contributing.md) - Contribution guidelines

## Development Setup

### Prerequisites

- Python 3.10 or 3.11
- Node.js 20+
- Docker and Docker Compose
- Git
- IDE with Python and TypeScript support (VSCode recommended)

### Initial Setup

**Clone repository**:
```bash
git clone https://github.com/danve93/graphrag-mrkr-2.git
cd graphrag-mrkr-2
```

**Backend setup**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**Frontend setup**:
```bash
cd frontend
npm install
cp .env.local.example .env.local
```

**Start dependencies**:
```bash
docker compose up neo4j -d
```

**Start services**:
```bash
python api/main.py &
cd frontend && npm run dev &
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

Follow [Code Conventions](09-development/coding-standards.md):
- Use type hints in Python
- Write TypeScript with strict mode
- Add docstrings to public functions
- Keep functions focused and testable

### 3. Write Tests

```bash
pytest tests/unit/test_my_feature.py -v
pytest tests/integration/test_my_feature.py -v
```

### 4. Run Linters

**Backend**:
```bash
ruff check .
black .
isort .
mypy .
```

**Frontend**:
```bash
cd frontend
npm run lint
npm run type-check
```

### 5. Test Locally

```bash
pytest tests/
cd frontend && npm test
```

### 6. Commit Changes

```bash
git add .
git commit -m "feat: add my feature"
```

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Test changes
- `refactor:` Code refactoring
- `perf:` Performance improvement
- `chore:` Maintenance

### 7. Push and Create PR

```bash
git push origin feature/my-feature
```

Open pull request on GitHub.

## Project Structure

```
graphrag-mrkr-2/
├── api/                    # FastAPI backend
│   ├── main.py            # Application entry
│   ├── models.py          # Pydantic models
│   ├── routers/           # API endpoints
│   └── services/          # Business logic
├── core/                  # Core services
│   ├── graph_db.py        # Neo4j operations
│   ├── embeddings.py      # Embedding manager
│   ├── entity_extraction.py
│   └── graph_clustering.py
├── rag/                   # RAG pipeline
│   ├── graph_rag.py       # LangGraph pipeline
│   ├── retriever.py       # Hybrid retrieval
│   └── rerankers/         # Reranking implementations
├── ingestion/             # Document processing
│   ├── document_processor.py
│   └── loaders/           # Format loaders
├── frontend/              # Next.js frontend
│   └── src/
│       ├── app/           # Pages
│       ├── components/    # React components
│       ├── lib/           # Utilities
│       └── store/         # State management
├── tests/                 # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/               # CLI tools
├── config/                # Configuration
└── docs/                  # Documentation
```

## Code Conventions

### Python

**Style Guide**: PEP 8 with Black formatting

```python
from typing import List, Dict, Optional

async def process_document(
    file_path: str,
    extract_entities: bool = True
) -> Dict[str, int]:
    """
    Process a document and extract entities.
    
    Args:
        file_path: Path to document file
        extract_entities: Whether to extract entities
        
    Returns:
        Dictionary with processing stats
        
    Raises:
        ValueError: If file_path is invalid
    """
    result = await processor.process_async(file_path)
    return result
```

**Key Patterns**:
- Use async/await for I/O operations
- Type hints on all functions
- Docstrings on public APIs
- Parameterized Neo4j queries
- Dependency injection via FastAPI

### TypeScript/React

**Style Guide**: ESLint with TypeScript strict mode

```typescript
interface DocumentViewProps {
  documentId: string
  onClose: () => void
}

export function DocumentView({ documentId, onClose }: DocumentViewProps) {
  const [loading, setLoading] = useState(false)
  
  useEffect(() => {
    const fetchDocument = async () => {
      setLoading(true)
      try {
        const doc = await api.getDocument(documentId)
        setDocument(doc)
      } catch (error) {
        console.error('Failed to fetch document', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchDocument()
  }, [documentId])
  
  return (
    <div className="document-view">
      {loading ? <Loader /> : <DocumentContent document={document} />}
    </div>
  )
}
```

**Key Patterns**:
- Props interfaces for all components
- Hooks for state and effects
- Error boundaries for error handling
- Suspense for async data
- Memoization for expensive computations

## Testing

### Test Organization

```
tests/
├── unit/              # Fast, isolated tests
├── integration/       # Tests with Neo4j, APIs
└── e2e/              # Full pipeline tests
```

### Running Tests

**All tests**:
```bash
pytest tests/
```

**By category**:
```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

**With coverage**:
```bash
pytest --cov=api --cov=core --cov=rag --cov=ingestion tests/
```

**Frontend tests**:
```bash
cd frontend
npm test
npm run test:coverage
```

See [Testing](09-development/testing-backend.md) for detailed guidelines.

## Debugging

### Backend Debugging

**Print debugging**:
```python
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Processing document: {doc_id}")
```

**Interactive debugging**:
```python
import pdb; pdb.set_trace()
```

**VSCode debugging**: Use `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["api.main:app", "--reload"],
      "jinja": true
    }
  ]
}
```

### Frontend Debugging

**Browser DevTools**: 
- Network tab for API calls
- Console for errors
- React DevTools for component state

**VSCode debugging**:
```json
{
  "name": "Next.js: debug",
  "type": "node",
  "request": "launch",
  "runtimeExecutable": "npm",
  "runtimeArgs": ["run", "dev"],
  "port": 9229
}
```

See [Debugging](09-development/dev-scripts.md) for advanced techniques.

## Adding Features

### Feature Development Checklist

1. Design
   - [ ] Define requirements
   - [ ] Design API interface
   - [ ] Plan data model changes
   - [ ] Document architecture

2. Implementation
   - [ ] Create feature branch
   - [ ] Implement core logic
   - [ ] Add API endpoints
   - [ ] Update frontend
   - [ ] Add configuration

3. Testing
   - [ ] Write unit tests
   - [ ] Write integration tests
   - [ ] Test edge cases
   - [ ] Verify error handling

4. Documentation
   - [ ] Update API docs
   - [ ] Add usage examples
   - [ ] Update configuration docs
   - [ ] Add to changelog

5. Review
   - [ ] Code review
   - [ ] Test coverage check
   - [ ] Performance testing
   - [ ] Security review

See [Adding Features](09-development/feature-flag-wiring.md) for detailed guide.

## Code Review Guidelines

**For Authors**:
- Keep PRs focused and small
- Write clear commit messages
- Add tests for new code
- Update documentation
- Respond to feedback promptly

**For Reviewers**:
- Review within 24 hours
- Be constructive and specific
- Test changes locally if needed
- Approve when requirements met

## CI/CD

**GitHub Actions Workflows**:
- `.github/workflows/ci.yml` - Run tests on PR
- `.github/workflows/lint.yml` - Linting checks
- `.github/workflows/build.yml` - Docker builds

**Pre-commit Hooks**:
```bash
pip install pre-commit
pre-commit install
```

## Related Documentation

- [Code Conventions](09-development/coding-standards.md)
- [Testing Guide](09-development/testing-backend.md)
- [Adding Features](09-development/feature-flag-wiring.md)
- [Contributing](09-development/contributing.md)
