# Dev Scripts

Helper scripts and Makefile tasks for development.

## Makefile

```bash
# View tasks
make help

# Common targets
make dev           # run local dev (backend + frontend)
make test          # run backend tests
make clean         # cleanup build artifacts
```

## Scripts (scripts/)

### Ingestion

```bash
python scripts/ingest_documents.py --path data/documents
```

### Similarities

```bash
python scripts/create_similarities.py
```

### Clustering

```bash
python scripts/build_leiden_projection.py
python scripts/run_clustering.py
```

### FlashRank Prewarm

```bash
python scripts/flashrank_prewarm_worker.py
```

### Entity Inspection

```bash
python scripts/inspect_entities.py --limit 50
```

## Tips

- Run scripts from repo root to ensure paths resolve
- Use virtualenv (ensure deps installed)
- Pass `--help` to any script for options
