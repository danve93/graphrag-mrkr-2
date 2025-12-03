# Frontend Testing

Guide to testing the Next.js frontend.

## Prerequisites

- Node.js 18+
- Install dependencies: `npm install`

## Running Tests

```bash
npm test
```

## Tools

- Jest - Test runner
- React Testing Library (RTL) - Component testing
- MSW (Mock Service Worker) - API mocking (recommended)

## Patterns

- Test components with RTL queries (avoid snapshots)
- Mock SSE streams using MSW or custom readable streams
- Isolate state with Zustand test helpers

## Examples

### Component Test (ChatInterface)

```tsx
import { render, screen, fireEvent } from '@testing-library/react'
import ChatInterface from '@/src/components/ChatInterface'

test('sends a message', async () => {
  render(<ChatInterface />)
  fireEvent.change(screen.getByPlaceholderText(/Type your message/i), { target: { value: 'hello' } })
  fireEvent.click(screen.getByText(/Send/i))

  expect(await screen.findByText(/hello/)).toBeInTheDocument()
})
```

### API Mock (MSW)

```ts
import { rest } from 'msw'

export const handlers = [
  rest.get('/api/documents', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({ items: [], total: 0, limit: 50, offset: 0, has_more: false })
    )
  })
]
```

## Coverage

```bash
npm test -- --coverage
```

## Tips

- Prefer role-based selectors in RTL
- Avoid implementation details; test user interactions
- Mock network calls; keep tests fast (< 2s per file)
