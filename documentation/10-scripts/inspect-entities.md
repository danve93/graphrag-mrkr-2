# Inspect Entities Script

Script: `scripts/inspect_entities.py`

## Purpose

Inspect entity nodes and their relationships for debugging and analysis.

## Usage

```bash
python scripts/inspect_entities.py \
  --limit 50 \
  --type Component \
  --community 5 \
  --min-importance 0.5
```

## Arguments

- `--limit` (int, optional): Number of entities to display
- `--type` (string, optional): Filter by entity type
- `--community` (int, optional): Filter by `community_id`
- `--min-importance` (float, optional): Minimum importance score
- `--name` (string, optional): Search by name substring

## Output

- Table/list of entities with fields: `id`, `name`, `type`, `importance`, `mention_count`, `relationship_count`, `community_id`
- Optional relationship summaries

## Examples

```bash
# Top important entities
python scripts/inspect_entities.py --limit 20 --min-importance 0.7

# Entities in a community
python scripts/inspect_entities.py --community 23
```

## Tips

- Use filters to focus on specific areas of the graph
- Combine with clustering scripts to analyze community composition
- Export results to CSV/JSON if supported
