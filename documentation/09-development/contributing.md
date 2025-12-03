# Contributing

Guidelines for contributing code and documentation.

## Workflow

1. Create a branch from `main`
   - `feat/<topic>` for features
   - `fix/<issue>` for bug fixes
2. Implement changes with focused commits
3. Run tests locally (backend + frontend)
4. Open a Pull Request (PR)
5. Address review comments and CI failures

## Commit Style

- Use clear, concise messages
- Prefix with type: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- Example: `feat: add entity label TTL cache metrics endpoint`

## Code Review

- Keep PRs small and focused
- Include rationale in description
- Link related issues or docs
- Provide test coverage for new logic

## Tests & Coverage

```bash
pytest -q --disable-warnings --maxfail=1
cd frontend && npm test -- --coverage
```

## Lint & Format

```bash
# Python (ruff, black)
ruff check .
black .

# TypeScript (eslint, prettier)
cd frontend
npm run lint
npm run format
```

## Secrets & Configuration

- Never commit secrets or `.env`
- Use `.env.example` for placeholders
- Validate configuration via `/api/database/health`

## PR Checklist

- [ ] Unit tests added/updated
- [ ] Docs updated (`README.md`, `AGENTS.md`, relevant section)
- [ ] No secrets committed
- [ ] Lint and format pass
- [ ] Local tests pass

## Issue Reporting

- Include steps to reproduce
- Provide logs or screenshots
- Add environment info (OS, Python/Node versions)

## Code of Conduct

Be respectful and collaborative. Review constructively, focus on outcomes.
