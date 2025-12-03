# Create Similarities Script

Script: `scripts/create_similarities.py`

## Purpose

Compute `SIMILAR_TO` relationships between chunk nodes based on embedding similarity.

## Usage

```bash
python scripts/create_similarities.py \
  --threshold 0.7 \
  --max-connections 5 \
  --batch-size 500 \
  --limit-docs 100
```

## Arguments

- `--threshold` (float, optional): Similarity threshold (default: `SIMILARITY_THRESHOLD`)
- `--max-connections` (int, optional): Max edges per chunk (default: `MAX_SIMILARITY_CONNECTIONS`)
- `--batch-size` (int, optional): UNWIND batch size (default: `NEO4J_UNWIND_BATCH_SIZE`)
- `--limit-docs` (int, optional): Limit number of documents processed
- `--dry-run` (bool, optional): Print plan without executing

## Environment

- `.env` required: Neo4j credentials

## Output

- Neo4j: `SIMILAR_TO` edges with `similarity` property
- Logs: processed chunks, edges created, timings

## Examples

```bash
# Default thresholds
python scripts/create_similarities.py

# High precision
python scripts/create_similarities.py --threshold 0.8 --max-connections 3
```

## Monitoring

```cypher
MATCH (:Chunk)-[r:SIMILAR_TO]-(:Chunk)
RETURN count(r) AS edge_count, avg(r.similarity) AS avg_similarity
```

## Tips

- Higher thresholds = fewer, stronger edges
- Keep max connections low to avoid dense graphs
- Run after ingestion and before clustering
