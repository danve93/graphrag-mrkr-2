# Test Suite Organization

This directory contains all automated tests for the GraphRAG project, organized by test type.

## Directory Structure

```
tests/
├── unit/          # Unit tests (fast, isolated, no external dependencies)
├── integration/   # Integration tests (requires Neo4j, may call APIs)
└── e2e/           # End-to-end tests (full pipeline workflows)
```

## Running Tests

### All Tests
```bash
pytest tests/
```

### By Category
```bash
pytest tests/unit/          # Fast unit tests only
pytest tests/integration/   # Integration tests (requires Neo4j)
pytest tests/e2e/          # End-to-end pipeline tests
```

### Specific Test Files
```bash
pytest tests/integration/test_chat_pipeline.py -v -s
pytest tests/integration/test_full_ingestion_pipeline.py -v -s
pytest tests/e2e/test_full_pipeline.py -v -s
```

### Parallel Execution
```bash
pytest tests/ -n auto  # Run tests in parallel using pytest-xdist
```

### With Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### Unit Tests (`tests/unit/`)
Fast, isolated tests with no external dependencies:
- Test individual functions and classes
- Use mocks for external dependencies
- Should complete in milliseconds
- Can run without Neo4j, APIs, or external services

**Current unit tests:**
- `test_caching.py` - Singleton manager unit tests (14 tests)
- `test_graph_db_caching.py` - GraphDB cache layer tests (4 tests)
- `test_embedding_caching.py` - Embedding cache tests (6 tests)
- `test_retrieval_caching.py` - Retrieval cache tests (7 tests)

**When to add unit tests:**
- Testing utility functions
- Testing data transformations
- Testing business logic without I/O

### Integration Tests (`tests/integration/`)
Tests that validate component interactions:
- Require Neo4j database connection
- May call external APIs (OpenAI, etc.)
- Test multiple components working together
- Validate database operations

**Current integration tests:**
- `test_chat_pipeline.py` - Complete RAG pipeline with chat (includes cache effectiveness test)
- `test_full_ingestion_pipeline.py` - Document ingestion validation
- `test_marker_integration.py` - PDF conversion with Marker
- `test_entity_extraction.py` - Entity extraction workflows
- `test_clustering_and_communities.py` - Graph clustering
- `test_caching_integration.py` - Real service caching integration
- And more...

**When to add integration tests:**
- Testing API endpoints
- Testing database operations
- Testing service integrations
- Testing multi-component workflows

### End-to-End Tests (`tests/e2e/`)
Full pipeline tests simulating real-world usage:
- Test complete user workflows
- Validate entire system behavior
- May take minutes to complete
- Exercise all layers of the application

**Current e2e tests:**
- `test_full_pipeline.py` - Ingestion → clustering → summarization → visualization

**When to add e2e tests:**
- Testing complete user journeys
- Validating critical business workflows
- System-level smoke tests
- Release validation tests

## Test Fixtures

Pytest fixtures provide reusable test data and setup:

```python
@pytest.fixture(scope="module")
def test_document():
    """Create a test document for the pipeline."""
    # Setup
    doc = create_document()
    yield doc
    # Teardown
    cleanup(doc)
```

Common scopes:
- `function` - Run for each test function (default)
- `module` - Run once per test file
- `session` - Run once per test session

## Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.integration  # Requires database
@pytest.mark.slow         # Long-running test
@pytest.mark.e2e          # End-to-end test
```

Run specific markers:
```bash
pytest -m integration  # Run only integration tests
pytest -m "not slow"   # Skip slow tests
```

## Test Configuration

See `pytest.ini` for test configuration:
- Test paths
- Logging settings
- Async configuration
- Marker definitions

## Best Practices

1. **Keep tests isolated** - Each test should be independent
2. **Use descriptive names** - `test_document_ingestion_creates_chunks`
3. **Clean up resources** - Use fixtures with teardown
4. **Mock external services** - Mock APIs in unit tests
5. **Test one thing** - Each test should validate one behavior
6. **Use assertions** - Clear, specific assertions with messages
7. **Document complex tests** - Add docstrings explaining what's tested

## Continuous Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch
- Nightly builds (e2e tests)

CI configuration: `.github/workflows/` (if present)

## Troubleshooting

**Neo4j connection errors:**
```bash
# Ensure Neo4j is running
docker compose up -d neo4j
export NEO4J_URI="bolt://localhost:7687"
```

**Docker port conflicts (tests won't start / compose fails):**
```bash
# Stop lingering stacks and free common ports used by tests
bash scripts/cleanup_docker.sh

# Re-run the test suite (cleanup runs automatically before service startup;
# disable with TEST_DOCKER_CLEANUP=0)
pytest tests/integration
```

**Import errors:**
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Async test issues:**
```bash
# Check pytest-asyncio is installed
pip install pytest-asyncio
```

## Adding New Tests

1. Choose the right category (unit/integration/e2e)
2. Create test file: `test_<feature>.py`
3. Add fixtures if needed
4. Write test functions starting with `test_`
5. Add markers for categorization
6. Document complex test behavior
7. Ensure tests pass: `pytest tests/<category>/test_<feature>.py`

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/mark.html)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
