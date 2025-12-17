# Golden Ingestion Test

The **Golden Ingestion Test** is a comprehensive end-to-end test that validates the complete document ingestion pipeline using real services. It serves as the health check for the ingestion codebase.

> **Rule**: If this test passes, the ingestion pipeline is healthy.

## Overview

| Aspect | Description |
|--------|-------------|
| **Location** | `tests/e2e/golden_ingestion_test.py` |
| **CI Workflow** | `.github/workflows/golden-test.yml` |
| **Duration** | ~5 minutes |
| **Services Used** | Neo4j, OpenAI API |

## What It Tests

| Test | Description |
|------|-------------|
| `test_01_neo4j_connection` | Verifies database connectivity |
| `test_02_embedding_service` | Validates OpenAI embedding generation (1536 dimensions) |
| `test_03_text_file_ingestion` | Full text → chunks → embeddings → entities flow |
| `test_04_pdf_file_ingestion` | PDF parsing via smart OCR, chunking, embedding |
| `test_05_entity_extraction_llm` | Real GPT-4o-mini entity extraction with gleaning |
| `test_06_progress_tracking` | UI progress updates persisted to database |
| `test_07_data_integrity` | Required fields and constraints on all nodes |

## Requirements

### Local Execution
- Neo4j database running (`docker compose up neo4j`)
- `OPENAI_API_KEY` in `.env` or environment
- System dependencies: `poppler-utils`, `tesseract-ocr`

### GitHub CI
- `OPENAI_API_KEY` as repository secret (Settings → Secrets → Actions)

## Running Locally

```bash
# Start Neo4j
docker compose up -d neo4j

# Run the test
TEST_ENABLE_E2E=1 TEST_SKIP_DOCKER=1 NEO4J_PASSWORD=<your-password> \
    uv run pytest tests/e2e/golden_ingestion_test.py -v -s
```

## Expected Output

```
PASSED test_01_neo4j_connection
PASSED test_02_embedding_service
PASSED test_03_text_file_ingestion
PASSED test_04_pdf_file_ingestion
PASSED test_05_entity_extraction_llm
PASSED test_06_progress_tracking
PASSED test_07_data_integrity

======== 7 passed in 318.14s (0:05:18) ========
```

## Difference from Unit Tests

| Aspect | Unit Tests (`tests/`) | Golden Test (`tests/e2e/`) |
|--------|----------------------|---------------------------|
| External Services | Mocked | Real (Neo4j, OpenAI) |
| Duration | Seconds | ~5 minutes |
| Purpose | Test logic in isolation | Validate full pipeline |
| When to Run | Every commit | PRs/Main branch |

## Troubleshooting

### Neo4j Connection Failed
```bash
docker compose restart neo4j
# Wait 30 seconds for startup
```

### Authentication Rate Limit
If you see `AuthenticationRateLimit`, restart Neo4j:
```bash
docker compose restart neo4j
```

### Missing OPENAI_API_KEY
Ensure the key is set in `.env` or passed via environment:
```bash
OPENAI_API_KEY=sk-... uv run pytest ...
```

## See Also
- [Testing Backend](testing-backend.md)
- [Document Ingestion Flow](../05-data-flows/document-ingestion-flow.md)
