# Scripts

Command-line tools for ingestion, maintenance, and utilities.

## Contents

- [README](10-scripts/README.md) - Scripts overview
- [Ingest Documents](10-scripts/ingest-documents.md) - Document ingestion CLI
- [Run Clustering](10-scripts/run-clustering.md) - Leiden clustering execution
- [Create Similarities](10-scripts/create-similarities.md) - Chunk similarity calculation
- [Maintenance Scripts](10-scripts/setup-neo4j.md) - Additional utilities
- [Neo4j Setup](10-scripts/setup-neo4j.md) - Indexes, constraints, casefold, dedupe
- [Reindex Classification](10-scripts/reindex-classification.md) - Classify documents and update metadata

## Available Scripts

### Document Ingestion

**Script**: `scripts/ingest_documents.py`

Ingest documents from files or directories into the knowledge graph.

**Basic Usage**:
```bash
python scripts/ingest_documents.py --file /path/to/document.pdf
```

**Directory Ingestion**:
```bash
python scripts/ingest_documents.py --input-dir /path/to/docs --recursive
```

**Options**:
```bash
--file FILE               Single file to ingest
--input-dir DIR           Directory to scan for documents
--recursive              Recursively scan subdirectories
--show-supported         Display supported file formats
--force-reprocess        Reprocess existing documents
--skip-entities          Skip entity extraction
--batch-size N           Process N documents at a time
```

See [Ingest Documents](10-scripts/ingest-documents.md) for detailed usage.

### Graph Clustering

**Script**: `scripts/run_clustering.py`

Run Leiden community detection on entity graph.

**Basic Usage**:
```bash
python scripts/run_clustering.py
```

**With Options**:
```bash
python scripts/run_clustering.py \
  --resolution 1.0 \
  --min-edge-weight 0.3 \
  --relationship-types RELATED_TO,SIMILAR_TO \
  --level 0
```

**Options**:
```bash
--resolution FLOAT       Clustering granularity (default 1.0)
--min-edge-weight FLOAT  Minimum edge weight threshold (default 0.3)
--relationship-types STR Comma-separated relationship types
--level INT             Hierarchy level (default 0)
--dry-run               Preview without persisting
```

See [Run Clustering](10-scripts/run-clustering.md) for algorithm details.

### Chunk Similarities

**Script**: `scripts/create_similarities.py`

Calculate and store chunk-to-chunk similarities.

**Basic Usage**:
```bash
python scripts/create_similarities.py
```

**With Options**:
```bash
python scripts/create_similarities.py \
  --batch-size 100 \
  --similarity-threshold 0.7 \
  --document-id abc123
```

**Options**:
```bash
--batch-size N           Process N chunks at a time (default 100)
--similarity-threshold F Minimum similarity score (default 0.7)
--document-id ID        Process specific document only
--overwrite             Replace existing similarities
```

### Entity Inspection

**Script**: `scripts/inspect_entities.py`

View and analyze extracted entities.

**Basic Usage**:
```bash
python scripts/inspect_entities.py
```

**Options**:
```bash
--document-id ID        Show entities for specific document
--entity-type TYPE      Filter by entity type
--community-id N        Filter by community
--limit N               Limit results (default 100)
--export FILE           Export to JSON file
```

### Database Maintenance

**Script**: `scripts/maintenance.py`

Perform database cleanup and optimization.

**Operations**:
```bash
python scripts/maintenance.py --operation vacuum
python scripts/maintenance.py --operation reindex
python scripts/maintenance.py --operation cleanup-orphans
```

## Script Development

### Creating New Scripts

**Template**:
```python
#!/usr/bin/env python3
"""
Script description here.

Usage:
    python scripts/my_script.py --option value
"""

import argparse
import asyncio
from core.graph_db import get_db

async def main(args):
    """Main script logic."""
    db = get_db()
    
    # Your logic here
    print(f"Processing with option: {args.option}")
    
    await db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My script")
    parser.add_argument("--option", required=True, help="Option description")
    args = parser.parse_args()
    
    asyncio.run(main(args))
```

### Best Practices

1. **Use argparse**: Consistent CLI interface
2. **Add help text**: Document all options
3. **Handle errors**: Graceful failure with messages
4. **Log progress**: Use logging for status updates
5. **Async operations**: Use async/await for I/O
6. **Close connections**: Clean up resources
7. **Dry-run mode**: Preview before modifications

### Testing Scripts

```bash
python scripts/my_script.py --help
python scripts/my_script.py --dry-run
pytest tests/scripts/test_my_script.py
```

## Common Workflows

### Initial Database Setup

```bash
docker compose up -d neo4j

python scripts/ingest_documents.py \
  --input-dir /path/to/docs \
  --recursive

python scripts/create_similarities.py

python scripts/run_clustering.py
```

### Reindexing

```bash
python scripts/maintenance.py --operation cleanup-orphans

python scripts/ingest_documents.py \
  --input-dir /path/to/docs \
  --force-reprocess

python scripts/create_similarities.py --overwrite

python scripts/run_clustering.py
```

### Analyzing Results

```bash
python scripts/inspect_entities.py --limit 100

python scripts/inspect_entities.py \
  --entity-type COMPONENT \
  --export components.json
```

## Script Configuration

Scripts respect environment variables from `.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=sk-...
```

Override at runtime:
```bash
NEO4J_URI=bolt://remote:7687 python scripts/ingest_documents.py --file doc.pdf
```

## Scheduling Scripts

### Cron

```cron
0 2 * * * cd /app && python scripts/maintenance.py --operation vacuum
0 3 * * 0 cd /app && python scripts/run_clustering.py
```

### systemd Timer

```ini
[Unit]
Description=Run clustering weekly

[Timer]
OnCalendar=weekly
Persistent=true

[Install]
WantedBy=timers.target
```

### Docker Compose

```yaml
services:
  clustering-job:
    image: graphrag-backend
    command: python scripts/run_clustering.py
    environment:
      - NEO4J_URI=bolt://neo4j:7687
    depends_on:
      - neo4j
```

## Troubleshooting

**Import errors**:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/graphrag-mrkr-2"
```

**Connection errors**:
```bash
NEO4J_URI=bolt://localhost:7687 python scripts/my_script.py
```

**Permission errors**:
```bash
chmod +x scripts/my_script.py
```

**Memory errors**:
```bash
python scripts/ingest_documents.py --batch-size 10
```

## Related Documentation

- [Ingest Documents Details](10-scripts/ingest-documents.md)
- [Clustering Details](10-scripts/run-clustering.md)
- [Ingestion Pipeline](03-components/ingestion)
- [Graph Clustering](04-features/community-detection.md)
