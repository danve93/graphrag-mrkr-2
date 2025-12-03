# Ingest Documents Script

Script: `scripts/ingest_documents.py`

## Purpose

Ingest documents from a directory, chunk content, generate embeddings, extract entities (optional), and persist to Neo4j.

## Usage

```bash
python scripts/ingest_documents.py --path data/documents \
  --pattern "*.pdf" \
  --chunk-size 1200 \
  --chunk-overlap 150 \
  --entities true \
  --concurrency 3
```

## Arguments

- `--path` (string, required): Directory containing files
- `--pattern` (string, optional): Glob pattern (default: `*.*`)
- `--chunk-size` (int, optional): Chunk size (default: from settings)
- `--chunk-overlap` (int, optional): Chunk overlap (default: from settings)
- `--entities` (bool, optional): Enable entity extraction (default: settings)
- `--concurrency` (int, optional): Embedding concurrency (default: settings)
- `--max-files` (int, optional): Limit number of files processed
- `--dry-run` (bool, optional): Print plan without executing

## Environment

- `.env` required: `OPENAI_API_KEY`, `NEO4J_*`
- Optional: `USE_MARKER_FOR_PDF`, `ENABLE_GLEANING`

## Output

- Neo4j: Document, Chunk, Entity nodes + relationships
- Files: `data/chunks/`, `data/extracted/`
- Logs: progress, errors, timings

## Examples

```bash
# Ingest PDFs only
python scripts/ingest_documents.py --path data/documents --pattern "*.pdf"

# Dry run
python scripts/ingest_documents.py --path data/documents --dry-run

# Limit files
python scripts/ingest_documents.py --path data/documents --max-files 10
```

## Tips

- Use Marker for high-accuracy PDF conversion
- Set `SYNC_ENTITY_EMBEDDINGS=1` for deterministic tests
- Monitor job via `/api/jobs` if script submits as background task
