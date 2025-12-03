# Run Clustering Script

Script: `scripts/run_clustering.py`

## Purpose

Run the Leiden community detection algorithm on the graph projection and write `community_id` to Entity nodes.

## Usage

```bash
python scripts/run_clustering.py \
  --name entity-clustering \
  --resolution 1.0 \
  --weight-property strength \
  --include-intermediate false
```

## Arguments

- `--name` (string, optional): Projection name (default: `entity-clustering`)
- `--resolution` (float, optional): Leiden resolution (default: `1.0`)
- `--weight-property` (string, optional): Edge weight property (default: `strength`)
- `--include-intermediate` (bool, optional): Include intermediate communities
- `--dry-run` (bool, optional): Preview without writing

## Environment

- `.env` required: Neo4j credentials

## Output

- Entity nodes: `community_id` property assigned
- Logs: community counts and size distribution

## Examples

```bash
# Default run
python scripts/run_clustering.py

# Fine-grained communities
python scripts/run_clustering.py --resolution 1.5
```

## Verification (Cypher)

```cypher
MATCH (e:Entity)
WHERE e.community_id IS NOT NULL
RETURN count(e) AS assigned, avg(e.community_id) AS avg_id;
```

## Tips

- Rebuild projection if parameters change significantly
- Adjust resolution based on corpus size
- Optionally generate community summaries after clustering
