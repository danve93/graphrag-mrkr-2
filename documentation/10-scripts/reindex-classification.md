# Reindex: Document Classification (M2.4)

Classify documents into categories and propagate metadata to chunks to improve retrieval.

## Commands

```bash
# Enable classification via env vars (backend)
export ENABLE_DOCUMENT_CLASSIFICATION=1
export CLASSIFICATION_MODEL=gpt-4o-mini
export CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7

# Reindex a single document by id
docker compose exec backend python scripts/reindex_classification.py --doc-id <doc_id>

# Reindex a batch (first N documents)
docker compose exec backend python scripts/reindex_classification.py --limit 100
```

## Behavior
- Document nodes receive: `category`, `categories`, `classification_confidence`, `keywords`, `difficulty`.
- Chunk nodes inherit `category` and include positional metadata like `chunk_number` and optional `semantic_group`.
- Classification falls back to the default `general` category when confidence is below threshold.

## Notes
- Safe to run incrementally; script updates metadata only.
- Model and thresholds are configurable via environment variables.
