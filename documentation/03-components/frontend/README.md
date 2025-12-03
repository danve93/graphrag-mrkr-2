# Frontend Components

Next.js application providing the user interface for Amber.

## Contents

- [README](03-components/frontend/README.md) - Frontend overview
- [Architecture](03-components/frontend/architecture.md) - Next.js structure and state management
- [Chat Interface](03-components/frontend/chat-interface.md) - Chat UI with SSE streaming
- [Document View](03-components/frontend/document-view.md) - Document explorer with pagination
- [Graph Visualization](03-components/frontend/graph-visualization.md) - 3D force-directed graph
- [API Client](03-components/frontend/api-client.md) - Backend API integration

## Overview

The frontend is built with:
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Zustand** - Lightweight state management
- **Force-Graph** - 3D graph visualization
- **React Markdown** - Markdown rendering

## Architecture

```
frontend/
├── src/
│   ├── app/              # Next.js App Router pages
│   ├── components/       # React components
│   │   ├── Chat/        # Chat interface
│   │   ├── Document/    # Document explorer
│   │   ├── Graph/       # Graph visualization
│   │   ├── Sidebar/     # Navigation sidebar
│   │   └── Utils/       # Shared utilities
│   ├── lib/             # Client libraries
│   │   └── api.ts       # API wrapper
│   ├── store/           # Zustand stores
│   ├── types/           # TypeScript types
│   └── styles/          # Global styles
├── public/              # Static assets
└── package.json         # Dependencies
```

## Key Components

### Chat Interface
**File**: `frontend/src/components/Chat/ChatInterface.tsx`

Features:
- Real-time streaming with SSE
- Message history with sources
- Quality score display
- Follow-up question suggestions
- Context document selection
- Model and parameter controls

### Document View
**File**: `frontend/src/components/Document/DocumentView.tsx`

Features:
- Summary-first loading pattern
- Lazy-loaded chunks with pagination
- Entity display grouped by type
- Similarity visualization
- Metadata inspection
- Document preview

### Graph Visualization
**File**: `frontend/src/components/Graph/GraphView.tsx`

Features:
- 3D force-directed graph
- Community-based coloring (10 distinct colors)
- Node hover information
- Community filtering
- Interactive camera controls
- Relationship highlighting

### API Client
**File**: `frontend/src/lib/api.ts`

Provides typed wrappers for:
- Chat query with streaming
- Document operations (list, get, delete)
- Chunk and entity pagination
- History management
- Upload and processing
- Database stats

## State Management

### Zustand Stores

**Chat Store** (`store/chatStore.ts`):
- Messages and conversation state
- Selected document and chunk
- Model selection
- Retrieval parameters

**Theme Store** (context provider):
- Light/dark mode
- Color preferences
- UI customization

## Data Flow

```
User Action
    ↓
React Component
    ↓
API Client (api.ts)
    ↓ HTTP/SSE
Backend API
    ↓
Component State Update
    ↓
UI Re-render
```

### SSE Streaming

Chat responses use Server-Sent Events for real-time updates:

```typescript
const response = await api.sendMessage({
  message: query,
  session_id: sessionId,
  stream: true
})

const reader = response.body.getReader()
const decoder = new TextDecoder()

while (true) {
  const { done, value } = await reader.read()
  if (done) break
  
  const chunk = decoder.decode(value)
  const lines = chunk.split('\n')
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6))
      handleStreamEvent(data)
    }
  }
}
```

Event types:
- `stage` - Pipeline progress (query_analysis, retrieval, generation)
- `token` - Generated text tokens
- `sources` - Retrieved documents
- `quality_score` - Response quality assessment
- `follow_ups` - Suggested follow-up questions
- `metadata` - Additional context

## Styling

### Tailwind Configuration

Custom design tokens:
- Color palette with primary/secondary/accent
- Typography scale
- Spacing system
- Dark mode support

### Component Patterns

Consistent patterns for:
- Loading states (Loader component)
- Error boundaries
- Empty states
- Pagination controls
- Expandable sections

## Configuration

### Environment Variables

**Development** (`.env.local`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Docker Compose** (set by compose):
```bash
NEXT_PUBLIC_API_URL_SERVER=http://backend:8000
```

### Build Configuration

**File**: `next.config.js`

Settings:
- API proxy for local dev
- Environment variable handling
- Output configuration
- ESLint rules

## Development

### Local Development

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Access: http://localhost:3000

### Production Build

```bash
npm run build
npm start
```

### Testing

```bash
npm run test
npm run lint
```

## Performance Optimizations

1. **Pagination**: Lazy-load chunks and entities
2. **SSE Streaming**: Progressive rendering during generation
3. **Caching**: API responses cached client-side
4. **Code Splitting**: Dynamic imports for heavy components
5. **Image Optimization**: Next.js Image component

## Accessibility

- Semantic HTML
- ARIA labels
- Keyboard navigation
- Screen reader support
- Color contrast compliance

## Related Documentation

- [Chat Interface Details](03-components/frontend/chat-interface.md)
- [Document View Details](03-components/frontend/document-view.md)
- [API Client](03-components/frontend/api-client.md)
- [Chat Query Flow](05-data-flows/chat-query-flow.md)
- [API Reference](06-api-reference)
