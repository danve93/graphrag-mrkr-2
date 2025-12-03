# Neo4j Setup

Initialize indexes, constraints, case-insensitive uniqueness, and perform category deduplication.

## Commands

```bash
# Full setup (test → indexes → constraints → casefold → stats)
docker compose exec backend python scripts/setup_neo4j.py

# Indexes (alias supported)
docker compose exec backend python scripts/setup_neo4j.py --setup
docker compose exec backend python scripts/setup_neo4j.py --setup-indexes

# Constraints (unique Category.name)
docker compose exec backend python scripts/setup_neo4j.py --setup-constraints

# Case-insensitive uniqueness (name_lower)
docker compose exec backend python scripts/setup_neo4j.py --setup-casefold

# Dedupe categories (dry-run then apply)
docker compose exec backend python scripts/dedupe_categories.py --list
docker compose exec backend python scripts/dedupe_categories.py --apply --strategy merge-keep-first
```

## Behavior

- **Indexes**: Ensures graph-related and fulltext indexes required by the app.
- **Constraints**: Ensures case-sensitive unique constraint on `Category.name`.
- **Casefold**: Ensures unique constraint on `Category.name_lower` and populates it from `name` for existing nodes.
- **Dedupe**: Safely consolidates pre-existing duplicates before enforcing constraints.

## Notes

- All commands are idempotent; re-running is safe.
- Run dedupe before constraints if you suspect duplicates.
- Use `.env` for credentials; do not commit secrets.
