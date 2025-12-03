# Build Leiden Projection Script

Script: `scripts/build_leiden_projection.py`

## Purpose

Create an in-memory graph projection for community detection using Neo4j Graph Data Science (GDS).

## Usage

```bash
python scripts/build_leiden_projection.py \
  --name entity-clustering \
  --relationships '["SIMILAR_TO","RELATED_TO"]' \
  --min-edge-weight 0.0
```

## Arguments

- `--name` (string, optional): Projection name (default: `entity-clustering`)
- `--relationships` (json array, optional): Relationship types to include
- `--min-edge-weight` (float, optional): Minimum edge strength
- `--drop-existing` (bool, optional): Drop existing projection

## Environment

- `.env` required: Neo4j credentials

## Output

- GDS projection created (does not write to disk)
- Logs: node/edge counts, filtered edges

## Examples

```bash
# Default build
python scripts/build_leiden_projection.py

# Drop and rebuild
python scripts/build_leiden_projection.py --drop-existing true
```

## Verification (Cypher)

```cypher
CALL gds.graph.list('entity-clustering') YIELD graphName, nodeProjected, relationshipProjected;
```

## Tips

- Ensure similarities are computed before building projection
- Use filtered `RELATED_TO` for precise communities
- Keep projection name consistent across scripts
