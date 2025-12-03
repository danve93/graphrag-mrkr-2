# Coding Standards

Consistent coding practices for Python and TypeScript.

## Python

- Style: PEP 8
- Type hints everywhere (mypy-friendly)
- Avoid one-letter variable names
- No inline comments unless requested
- Minimal, focused changes per PR

### Lint & Format

```bash
ruff check .
black .
```

### Patterns

- Use pydantic models for request/response types
- Prefer parameterized Cypher queries (`$param`)
- Isolate business logic in `api/services/` and `core/`
- SSE streams must be JSON-serializable

### Error Handling

- Wrap async functions in try/except
- Return errors in state/response dicts (don’t throw to client)

## TypeScript

- Style: ESLint + Prettier
- Use TypeScript models in `src/types/`
- Favor functional React components and hooks
- Zustand for state; Context for shared settings

### Lint & Format

```bash
cd frontend
npm run lint
npm run format
```

### Patterns

- Strong typing for API consumption
- Progressive rendering for SSE streams
- Tokenized design system (spacing, typography, colors)

## Git & PRs

- Small, focused commits
- Descriptive PR titles and summaries
- Link related docs/issues
- Update `README.md` and `AGENTS.md` when features change

## Performance

- Use caches where applicable (entity label, embedding, retrieval)
- Batch Neo4j operations (UNWIND)
- Concurrency controls for embeddings and LLM calls

## Security

- No secrets in code; use `.env`
- Validate inputs via Pydantic
- Avoid string concatenation in queries
