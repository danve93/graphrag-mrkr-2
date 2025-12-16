# Amber UI Kit

This document serves as the definitive UI Kit reference for the Amber GraphRAG frontend. All new development should adhere to these design tokens, patterns, and components.

---

## Technology Stack

| Technology | Purpose |
|------------|---------|
| **React 18+** | Component framework |
| **Next.js 14+** | App Router, SSR |
| **TailwindCSS 3.x** | Utility-first CSS |
| **@tailwindcss/typography** | Prose/markdown styling |
| **Framer Motion** | Animations |
| **clsx + tailwind-merge** | Class name utilities (`cn`) |
| **Lucide React** | Primary icon library (open source, ISC license) |
| **MUI Components** | Tooltip, Button (supplementary) |

---

## Design Tokens

All tokens are defined as CSS variables in `frontend/src/app/globals.css`.

### Spacing (8pt Grid System)

| Token | Value | Usage |
|-------|-------|-------|
| `--space-1` | 4px | Tight spacing, icon gaps |
| `--space-2` | 8px | Default gap |
| `--space-3` | 12px | Input padding |
| `--space-4` | 16px | Card padding, section gaps |
| `--space-6` | 24px | Large section padding |
| `--space-8` | 32px | Page margins |
| `--space-12` | 48px | Hero sections |
| `--space-16` | 64px | Major layout divisions |
| `--space-24` | 96px | Extra large spacing |

### Typography

| Token | Value | Usage |
|-------|-------|-------|
| `--font-display` | 'Noto Sans Display', sans-serif | Headings, branding |
| `--font-body` | 'Noto Sans', sans-serif | Body text, UI elements |
| `--text-xs` | 12px | Badges, hints |
| `--text-sm` | 14px | Secondary text, labels |
| `--text-base` | 16px | Default body |
| `--text-lg` | 18px | Emphasized text |
| `--text-xl` | 20px | Small headings |
| `--text-2xl` | 24px | Section headings |
| `--text-3xl` | 28px | Panel titles |
| `--text-4xl` | 36px | Page headings |
| `--text-5xl` | 48px | Hero text |

### Colors

#### Brand Colors

| Token | Value | Usage |
|-------|-------|-------|
| `--accent-primary` | `#f27a03` | Primary brand color (orange) |
| `--accent-hover` | `#d96a02` | Hover state |
| `--accent-active` | `#b85902` | Active/pressed state |
| `--accent-subtle` | `rgba(242, 122, 3, 0.12)` | Subtle backgrounds, selections |

#### Apple System Colors (Dark Mode)

For semantic UI states:

| Token | Value | Usage |
|-------|-------|-------|
| `--systemBlue` | `#0A84FF` | Links, info |
| `--systemGreen` | `#32D74B` | Success, connected |
| `--systemRed` | `#FF453A` | Error, destructive |
| `--systemOrange` | `#FF9F0A` | Warning |
| `--systemYellow` | `#FFD60A` | Caution |
| `--systemPurple` | `#BF5AF2` | Special states |
| `--systemPink` | `#FF375F` | Highlights |
| `--systemTeal` | `#64D2FF` | Info alternative |
| `--systemIndigo` | `#5E5CE6` | Special highlights |

#### Gray Scale

| Token | Value | Usage |
|-------|-------|-------|
| `--gray-50` | `#f9fafb` | Light backgrounds |
| `--gray-100` | `#f3f4f6` | Subtle backgrounds |
| `--gray-200` | `#e5e7eb` | Borders (light mode) |
| `--gray-300` | `#d1d5db` | Disabled text (light) |
| `--gray-400` | `#9ca3af` | Placeholder text |
| `--gray-500` | `#6b7280` | Secondary text |
| `--gray-600` | `#4b5563` | Body text (light mode) |
| `--gray-700` | `#374151` | Headings (light mode) |
| `--gray-800` | `#121212` | App background |
| `--gray-900` | `#121212` | Darkest background |
| `--gray-950` | `#030712` | Pure dark |

#### Semantic Colors

| Token | Value | Usage |
|-------|-------|-------|
| `--bg-primary` | `#121212` | Main background |
| `--bg-secondary` | `#1A1A1A` | Elevated surfaces (cards, sidebar) |
| `--bg-tertiary` | `#262626` | Highest elevation, inputs |
| `--text-primary` | `#E5E7EB` | Primary text |
| `--text-secondary` | `#A3A3A3` | Muted text |
| `--border` | `#3D3D3D` | Border color |
| `--heading-text` | `#FFFFFF` | Headings |

### Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| `--radius-sm` | 4px | Buttons, inputs |
| `--radius-md` | 8px | Cards, panels |
| `--radius-lg` | 12px | Modals, large cards |
| `--radius-full` | 9999px | Pills, avatars |

### Shadows

| Token | Value | Usage |
|-------|-------|-------|
| `--shadow-xs` | `0 1px 2px rgba(0, 0, 0, 0.05)` | Subtle lift |
| `--shadow-sm` | `0 1px 3px rgba(0, 0, 0, 0.1)` | Cards, buttons |
| `--shadow-md` | `0 4px 6px rgba(0, 0, 0, 0.07)` | Dropdowns |
| `--shadow-lg` | `0 10px 15px rgba(0, 0, 0, 0.1)` | Modals |
| `--shadow-xl` | `0 20px 25px rgba(0, 0, 0, 0.1)` | Floating elements |

### Animation Timing

| Token | Value | Usage |
|-------|-------|-------|
| `--timing-instant` | 100ms | Immediate feedback |
| `--timing-fast` | 150ms | Hover states |
| `--timing-normal` | 200ms | Standard transitions |
| `--timing-slow` | 300ms | Complex animations |
| `--easing-standard` | `cubic-bezier(0.4, 0, 0.2, 1)` | Material-style easing |

### TailwindCSS Extended Colors

The `tailwind.config.js` extends the default palette with additional semantic colors accessible via Tailwind utility classes:

| Palette | Example Usage | Notes |
|---------|---------------|-------|
| `accent-*` | `bg-accent-primary`, `text-accent-hover` | Brand orange (#f27a03) |
| `primary-*` | `bg-primary-500` | Alias for accent |
| `secondary-*` | `bg-secondary-800`, `text-secondary-400` | Gray scale for UI chrome |
| `systemBlue`, `systemGreen`, etc. | `text-systemGreen` | Apple system colors |
| `surface` | `bg-surface` | Elevated card background (#1C1C1E) |
| `elevated` | `bg-elevated` | Highest elevation (#2C2C2E) |
| `background` | `bg-background` | App background (#0a0a0a) |

---

## Utility Functions

### `cn` - Class Name Merger

Located at `frontend/src/lib/utils.ts`:

```tsx
import { cn } from '@/lib/utils';

// Combines clsx and tailwind-merge for conditional class names
<div className={cn('base-class', isActive && 'active-class', className)} />
```

---

## Component Classes

### Buttons

#### Primary Button

```html
<button class="button-primary">
  Action
</button>
```

**Properties:**
- Background: `--accent-primary`
- Text: white
- Border radius: `--radius-sm`
- Hover: `--accent-hover`
- Active: `--accent-active` + scale(0.98)
- Disabled: 50% opacity

#### Secondary Button

```html
<button class="button-secondary">
  Cancel
</button>
```

**Properties:**
- Background: transparent
- Border: 1px solid `--gray-300`
- Text: `--gray-700` (light) / `--gray-300` (dark)
- Hover: `--gray-50` background

#### Small Button

```html
<button class="small-button">
  Small Action
</button>
```

**Properties:**
- Font size: `--text-xs`
- Padding: 6px 12px
- Used for inline actions

### Inputs

```html
<input class="input-field" placeholder="Enter text..." />
<textarea class="input-field"></textarea>
```

**Properties:**
- Background: white (light) / `--bg-tertiary` (dark)
- Border: 1px solid `--gray-300` / `--border` (dark)
- Border radius: `--radius-sm`
- Padding: `--space-3` `--space-4`
- Focus: 2px outline `--accent-subtle`, border `--accent-primary`

### Cards

```html
<div class="card">
  Card content
</div>
```

**Properties:**
- Background: white (light) / `--bg-secondary` (dark)
- Border: 1px solid `--gray-200` / `--border` (dark)
- Border radius: `--radius-md`
- Padding: `--space-6`
- Shadow: `--shadow-sm`

### Chat Messages

```html
<!-- User message -->
<div class="chat-message chat-message-user">
  User text
</div>

<!-- Assistant message -->
<div class="chat-message chat-message-assistant">
  Assistant response
</div>
```

**User Message Properties:**
- Background: `--accent-primary`
- Text: white
- Aligned right
- Max width: 80%

**Assistant Message Properties:**
- Background: `--gray-900` / `--bg-secondary` (dark)
- Border: 1px solid
- Max width: 85%

### Spinner (Loading)

```html
<div class="spinner"></div>
```

**Properties:**
- 16px × 16px circular
- 2px border
- Rotating animation
- Top border: `--accent-primary`

### Slider

```html
<input type="range" class="slider" />
```

**Properties:**
- Track: `--gray-700`, 8px height
- Thumb: `--accent-primary`, 18px circle
- Hover: thumb scales 1.1×

### Toggles

```html
<div class="toggle-on"></div>  <!-- Active state -->
<div class="toggle-off"></div> <!-- Inactive state -->
```

**Properties:**
- On: `--accent-primary`
- Off: `--gray-700`

---

## Utility Classes

### Selection States

```css
.accent-selected   /* Selected item background */
.accent-hover      /* Hover state with accent border */
.focus-primary     /* Keyboard focus ring */
.is-dragging-ring  /* Drag feedback */
```

### Animations

```css
.message-fade-in   /* Message entry animation */
.tab-content       /* Tab switch transition */
.no-transition     /* Disable transitions */
```

### Markdown Content

Apply `.markdown-content` to containers rendering markdown/prose:

```html
<div class="markdown-content">
  <ReactMarkdown>...</ReactMarkdown>
</div>
```

---

## Layout System

### 12-Column Grid

```html
<div class="grid-12 center">
  <div class="col-span-6">Half width</div>
  <div class="col-span-6">Half width</div>
</div>
```

**Properties:**
- 12 columns, 24px gutter
- Fixed 72px columns on large screens
- Fluid on medium screens (< 1024px)
- Centered with max-width constraint

**Column Span Classes:** `.col-span-1` through `.col-span-12`

---

## React Components

### Location: `frontend/src/components/`

### Utility Components (`/Utils/`)

| Component | Purpose | Props |
|-----------|---------|-------|
| `Tooltip` | Hover tooltips via portal | `content: string`, `children` |
| `Loader` | Spinner with optional label | `size?: number` (default: 6), `label?: string` |
| `ExpandablePanel` | Collapsible section | `title`, `expanded`, `onToggle`, `children` |
| `KeyboardShortcutHint` | Keyboard shortcut display | - |
| `ViewTransition` | Page transitions | - |

### Toast Notifications (`/Toast/`)

```tsx
import { showToast } from '@/components/Toast/ToastContainer';

showToast('success', 'Title', 'Description');
showToast('error', 'Title', 'Description');
showToast('warning', 'Title', 'Description');

// Optional parameters:
showToast('success', 'Title', 'Description', 3000); // duration in ms (default: 5000)
showToast('warning', 'Title', 'Description', 8000, 'top-center'); // position (default: 'bottom-right')
```

**Types:** `success` (green), `error` (red), `warning` (yellow)

**Positions:** `'bottom-right'` (default), `'top-center'`

### Theme Components (`/Theme/`)

| Component | Purpose |
|-----------|---------|
| `StatusIndicator` | Connection status dot + prewarm indicator |
| `ThemeProvider` | Theme context wrapper |

### Branding (`/Branding/`)

```tsx
import { useBranding } from '@/components/Branding/BrandingProvider';

const branding = useBranding();
// branding.title, branding.heading, branding.tagline, etc.
```

### Navigation (`/Navigation/`)

| Component | Purpose |
|-----------|---------|
| `BottomDock` | macOS-style navigation dock |
| `ConditionalBottomDock` | Conditional dock wrapper |

**Dock Features:**
- Magnification effect on hover
- Keyboard shortcuts (⌘1-9)
- Mobile: bottom sheet grid
- Active state: orange dot indicator

### Sidebar (`/Sidebar/`)

| Component | Purpose |
|-----------|---------|
| `Sidebar` | Main resizable sidebar container |
| `SidebarHeader` | Branding/logo area |
| `SidebarUser` | User identity section |
| Various `*SidebarContent` | View-specific sidebar content |

**Sidebar Features:**
- Resizable (240px–400px)
- Collapsible (72px collapsed width)
- Dynamic content based on active view
- Mobile overlay mode

### Chat Components (`/Chat/`)

| Component | Purpose | Key Props |
|-----------|---------|-----------|
| `ChatInterface` | Main chat container | - |
| `ChatInput` | Input with @mention, #hashtag, drag-drop | `onSend`, `disabled`, `isStreaming`, `onStop` |
| `MessageBubble` | Individual message display | `message`, `onRetryWithCategories` |
| `SourcesList` | Grouped document sources | `sources` |
| `LoadingIndicator` | Multi-stage progress | `currentStage`, `completedStages`, `stageUpdates` |
| `FeedbackButtons` | Thumbs up/down feedback | `messageId`, `sessionId`, `query` |
| `QualityBadge` | Color-coded quality score | `score` |
| `RoutingBadge` | Category routing indicator | `routingInfo` |
| `FollowUpQuestions` | Suggested questions | - |
| `ConnectionStatus` | WebSocket status | - |

**Quality Badge Colors:**
- ≥80%: Green (`text-green-600`, `bg-green-50`)
- ≥60%: Yellow (`text-yellow-600`, `bg-yellow-50`)
- <60%: Red (`text-red-600`, `bg-red-50`)

**Routing Badge Confidence Colors:**
- ≥80%: Green (`rgb(34, 197, 94)`)
- ≥60%: Yellow (`rgb(234, 179, 8)`)
- <60%: Orange (`rgb(249, 115, 22)`)

### Metrics Components (`/Metrics/`)

| Component | Purpose |
|-----------|---------|
| `MetricsPanel` | Main metrics container with subpanels |
| `TruLensSubPanel` | TruLens monitoring dashboard |
| `RagasSubPanel` | RAGAS evaluation metrics |
| `OpenTelemetrySubPanel` | OpenTelemetry traces |

**Shared Metrics Components (`/Metrics/shared/`):**

| Component | Purpose | Key Props |
|-----------|---------|-----------|
| `MetricsCard` | Stats display with thresholds | `title`, `value`, `format`, `trend`, `threshold` |
| `HealthIndicator` | Service health status dot | `status` |
| `BenchmarkRunner` | Run evaluation benchmarks | - |
| `ComparisonChart` | Compare metrics over time | - |
| `ResultsTable` | Tabular results display | - |

### Document Components (`/Document/`)

| Component | Purpose |
|-----------|---------|
| `DocumentView` | Main document viewer |
| `DocumentPreview` | PDF/file preview |
| `DocumentGraph` | Document entity graph |
| `CommunitiesSection` | Community clusters |
| `ChunkSimilaritiesSection` | Chunk similarity matrix |

### Admin Components (`/Admin/`)

| Component | Purpose |
|-----------|---------|
| `AdminApiKeys` | API key management |
| `AdminSharedChats` | Shared conversations viewer |

### Motion Primitives (`/components/motion-primitives/`)

Custom animation components for enhanced UI:

```tsx
import { Dock, DockIcon, DockItem, DockLabel } from '@/../components/motion-primitives/dock';

<Dock magnification={60} distance={120} panelHeight={56}>
  <DockItem onClick={handleClick}>
    <DockLabel>Tooltip Label</DockLabel>
    <DockIcon>
      <IconComponent />
    </DockIcon>
  </DockItem>
</Dock>
```

**Dock Props:**
- `magnification`: Max icon size on hover (default: 80)
- `distance`: Mouse distance for magnification (default: 150)
- `panelHeight`: Dock height (default: 64)
- `spring`: Framer Motion spring config

---

## Iconography

### Primary: Heroicons

```tsx
import { PaperAirplaneIcon } from '@heroicons/react/24/solid';
import { XMarkIcon } from '@heroicons/react/24/outline';
```

- Use `/24/outline` for UI chrome
- Use `/24/solid` for action buttons
- Icon size: typically `w-5 h-5` or `w-6 h-6`

### Navigation: Lucide React

```tsx
import { MessageSquare, Database, Network } from 'lucide-react';
```

Used in BottomDock navigation items.

### Supplementary: MUI Icons

```tsx
import { ExpandMore, ExpandLess } from '@mui/icons-material';
```

Used in expandable panels.

### MUI Components

Some components use MUI for complex interactions:

```tsx
import { IconButton, Tooltip } from '@mui/material';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';

<Tooltip title="Helpful" placement="top">
  <IconButton size="small" sx={{ color: '#f27a03' }}>
    <ThumbUpIcon fontSize="small" />
  </IconButton>
</Tooltip>
```

**Usage Guidelines:**
- Prefer custom Tooltip from `/Utils/Tooltip.tsx` for consistency
- Use MUI IconButton when needing MUI Tooltip integration
- Style MUI components with `sx` prop using design tokens

---

## Accessibility

### Focus Visibility

```css
*:focus-visible {
  outline: 2px solid var(--accent-primary);
  outline-offset: 2px;
  border-radius: 4px;
}
```

### Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

### Touch Targets (Mobile)

- Minimum 44px × 44px for buttons and links
- `overscroll-behavior-y: contain` on body
- `-webkit-tap-highlight-color: transparent`

---

## Responsive Breakpoints

| Breakpoint | Width | Behavior |
|------------|-------|----------|
| Mobile | < 768px | Full-width layouts, bottom sheet navigation |
| Tablet | 768px–1024px | Adaptive grid, adjusted padding |
| Desktop | > 1024px | Full UI, dock navigation, sidebar visible |

---

## Visual Hierarchy

### Background Elevation

```
Lowest:  --bg-primary   (#121212)  ← App background
         --bg-secondary (#1A1A1A)  ← Sidebar, cards, panels
Highest: --bg-tertiary  (#262626)  ← Inputs, elevated sections within panels
```

### Text Hierarchy

```
Headings:  --heading-text (#FFFFFF)  ← H1–H6, titles
Primary:   --text-primary (#E5E7EB)  ← Body text
Secondary: --text-secondary (#A3A3A3) ← Labels, hints, muted
```

---

## Best Practices

1. **Use CSS Variables:** Always reference tokens instead of hard-coded values
2. **Semantic Colors:** Use system colors for state feedback (green=success, red=error)
3. **Consistent Spacing:** Follow the 8pt grid system
4. **Animation Restraint:** Use `--timing-fast` for micro-interactions, `--timing-slow` for major transitions
5. **Dark Mode Default:** The app is dark-mode first; light mode shares tokens via `:root[data-theme="dark"]`
6. **Component Reuse:** Check existing components before creating new ones
7. **Accessibility:** Ensure focus states and ARIA attributes on interactive elements

---

## State Management (Zustand)

Global state is managed via Zustand in `frontend/src/store/chatStore.ts`.

### Using the Store

```tsx
import { useChatStore } from '@/store/chatStore';

// Select specific state
const activeView = useChatStore((state) => state.activeView);
const isConnected = useChatStore((state) => state.isConnected);

// Get actions
const setActiveView = useChatStore((state) => state.setActiveView);
```

### Store State

| State | Type | Purpose |
|-------|------|---------|
| `messages` | `Message[]` | Current conversation messages |
| `sessionId` | `string` | Active session ID |
| `activeView` | `ActiveView` | Current main view (see below) |
| `selectedDocumentId` | `string \| null` | Selected document for detail view |
| `selectedChunkId` | `string \| number \| null` | Selected chunk within document |
| `isConnected` | `boolean` | WebSocket connection status |
| `user` | `{ id, token, role } \| null` | Authenticated user |
| `historyRefreshKey` | `number` | Increment to refresh history UI |

### Active Views

```typescript
type ActiveView = 
  | 'chat' 
  | 'document' 
  | 'graph' 
  | 'chatTuning' 
  | 'ragTuning' 
  | 'categories' 
  | 'routing' 
  | 'structuredKg' 
  | 'documentation' 
  | 'metrics' 
  | 'adminApiKeys' 
  | 'adminSharedChats';
```

### Store Actions

| Action | Purpose |
|--------|---------|
| `setActiveView(view)` | Navigate to a view |
| `selectDocument(id)` | Select document and switch to document view |
| `selectDocumentChunk(docId, chunkId)` | Deep-link to specific chunk |
| `clearChat()` | Reset conversation |
| `addMessage(msg)` | Add message to conversation |
| `updateLastMessage(updater)` | Update streaming message |
| `loadSession(sessionId)` | Load conversation from history |
| `identifyUser(username?)` | Authenticate user |
| `logout()` | Clear user session |

---

## TypeScript Interfaces

Core types are defined in `frontend/src/types/index.ts`.

### Message

```typescript
interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
  sources?: Source[];
  quality_score?: QualityScore;
  follow_up_questions?: string[];
  isStreaming?: boolean;
  context_documents?: string[];
  context_document_labels?: string[];
  context_hashtags?: string[];
  stages?: StageUpdate[];
  routing_info?: RoutingInfo;
  message_id?: string;
  session_id?: string;
}
```

### Source

```typescript
interface Source {
  chunk_id?: string;
  entity_id?: string;
  entity_name?: string;
  content: string;
  similarity: number;
  relevance_score?: number;
  document_name: string;
  original_filename?: string;
  document_id?: string;
  filename: string;
  chunk_index?: number;
  contained_entities?: string[];
}
```

### QualityScore

```typescript
interface QualityScore {
  total: number;
  breakdown: {
    context_relevance: number;
    answer_completeness: number;
    factual_grounding: number;
    coherence: number;
    citation_quality: number;
  };
  confidence: 'low' | 'medium' | 'high';
}
```

### RoutingInfo

```typescript
interface RoutingInfo {
  categories: string[];
  confidence: number;
  category_id?: string | null;
}
```

### DocumentSummary

```typescript
interface DocumentSummary {
  document_id: string;
  filename: string;
  original_filename?: string;
  created_at: string;
  chunk_count: number;
  processing_status?: string;
  hashtags?: string[];
}
```

---

## API Client

The API client is defined in `frontend/src/lib/api.ts`.

### Usage

```tsx
import { api, API_URL } from '@/lib/api';

// All methods return Promises
const documents = await api.getDocuments();
const session = await api.getConversation(sessionId);
```

### Authentication

```tsx
// Set auth token (persisted to localStorage)
api.setAuthToken(token);

// Identify user (creates or retrieves user)
const { user_id, token, role } = await api.identifyUser(username);
```

### Key Methods

| Category | Method | Purpose |
|----------|--------|---------|
| **Chat** | `sendMessage(data, options?)` | Send chat message |
| **History** | `getHistory()` | Get all sessions |
| | `getConversation(sessionId)` | Load specific session |
| | `deleteConversation(sessionId)` | Delete session |
| **Documents** | `getDocuments()` | List all documents |
| | `getDocument(id)` | Get document details |
| | `stageFile(file)` | Stage file for processing |
| | `processDocuments(fileIds)` | Start processing |
| | `deleteDocument(id)` | Remove document |
| **Graph** | `getCategories()` | Get routing categories |
| | `getFullGraph()` | Get entity graph data |
| **Settings** | `getSettings()` | Get RAG settings |
| | `updateSettings(settings)` | Update settings |

### Fetch with Auth

All methods use `fetchWithAuth()` which automatically adds the auth token:

```tsx
const fetchWithAuth = async (url: string, options: RequestInit = {}) => {
  const headers = new Headers(options.headers);
  if (authToken) {
    headers.set('Authorization', `Bearer ${authToken}`);
  }
  return fetch(url, { ...options, headers, credentials: 'include' });
};
```

---

## Keyboard Shortcuts

### Navigation (BottomDock)

| Shortcut | Action |
|----------|--------|
| `⌘1` / `Ctrl+1` | Chat |
| `⌘2` / `Ctrl+2` | Database |
| `⌘3` / `Ctrl+3` | Graph |
| `⌘4` / `Ctrl+4` | Categories |
| `⌘5` / `Ctrl+5` | Structured KG |
| `⌘6` / `Ctrl+6` | Chat Tuning |
| `⌘7` / `Ctrl+7` | RAG Tuning |
| `⌘8` / `Ctrl+8` | Metrics |
| `⌘9` / `Ctrl+9` | Documentation |

### Chat Input

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `@` | Trigger document mention |
| `#` | Trigger hashtag mention |
| `↑` / `↓` | Navigate mention list |
| `Tab` / `Enter` | Select mention item |
| `Escape` | Close mention list |
| `↑` (empty input) | Navigate message history |

---

## Dark Mode

The application is **dark mode by default**.

### Tailwind Configuration

```javascript
// tailwind.config.js
module.exports = {
  darkMode: 'class',  // Uses 'dark' class on <html>
  // ...
};
```

### CSS Variable Overrides

```css
/* Default (dark theme) */
:root {
  --bg-primary: #121212;
  --bg-secondary: #1A1A1A;
  --text-primary: #E5E7EB;
}

/* Light theme override (if implemented) */
:root[data-theme="dark"] {
  --bg-primary: #0a0a0a;
  --bg-secondary: #171717;
  --text-primary: #fafafa;
}
```

### Styling Patterns

```css
/* CSS component with dark mode */
.input-field {
  background: white;
  border-color: var(--gray-300);
}

html.dark .input-field {
  background: var(--bg-tertiary);
  border-color: var(--border);
}
```

```tsx
// Tailwind with dark mode
<div className="bg-white dark:bg-neutral-800">
<span className="text-gray-600 dark:text-gray-400">
```

### Theme Application

The `<html>` element has `class="dark"` applied by default:

```tsx
// frontend/src/app/layout.tsx
<html lang="en" className="dark">
```

---

## Layout Patterns

### Main Application Structure

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  ┌──────────┐  ┌─────────────────────────────┐  │
│  │          │  │                             │  │
│  │ Sidebar  │  │      Main Content Area      │  │
│  │ (fixed)  │  │      (flex-1, scrollable)   │  │
│  │          │  │                             │  │
│  │          │  │                             │  │
│  └──────────┘  └─────────────────────────────┘  │
│                                                 │
│  ┌─────────────────────────────────────────────┐│
│  │            BottomDock (fixed)               ││
│  └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```

### Sidebar + Content Pattern

```tsx
<div className="flex h-screen">
  {/* Sidebar - fixed width, full height */}
  <Sidebar 
    open={sidebarOpen} 
    width={sidebarWidth} 
    collapsed={sidebarCollapsed}
  />
  
  {/* Main content - fills remaining space */}
  <main 
    className="flex-1 overflow-auto"
    style={{ marginLeft: sidebarCollapsed ? 72 : sidebarWidth }}
  >
    {activeView === 'chat' && <ChatInterface />}
    {activeView === 'document' && <DocumentView />}
    {/* ... other views */}
  </main>
  
  {/* Bottom dock - fixed at bottom center */}
  <BottomDock />
</div>
```

### Panel Pattern (Views)

Most views follow this structure:

```tsx
<div 
  className="h-full flex flex-col"
  style={{ 
    padding: 'var(--space-6)',
    background: 'var(--bg-primary)' 
  }}
>
  {/* Header */}
  <div className="flex items-center justify-between mb-6">
    <h1 style={{ fontSize: 'var(--text-2xl)', fontWeight: 600 }}>
      Panel Title
    </h1>
    <div className="flex gap-2">
      {/* Action buttons */}
    </div>
  </div>
  
  {/* Content - scrollable */}
  <div className="flex-1 overflow-auto space-y-4">
    {/* Expandable sections */}
    <ExpandablePanel title="Section 1" expanded={...} onToggle={...}>
      {/* Section content */}
    </ExpandablePanel>
  </div>
</div>
```

### Expandable Section Pattern

```tsx
const [expandedSections, setExpandedSections] = useState({
  section1: true,
  section2: false,
  section3: false,
});

const toggleSection = (key: string) => {
  setExpandedSections(prev => ({
    ...prev,
    [key]: !prev[key]
  }));
};

<ExpandablePanel
  title="Section Title"
  expanded={expandedSections.section1}
  onToggle={() => toggleSection('section1')}
>
  {/* Content only rendered when expanded */}
</ExpandablePanel>
```

### Chat Layout Pattern

```tsx
<div className="flex flex-col h-full">
  {/* Messages - scrollable, grows to fill */}
  <div className="flex-1 overflow-y-auto p-4 space-y-4">
    {messages.map(msg => <MessageBubble key={...} message={msg} />)}
    {isLoading && <LoadingIndicator />}
  </div>
  
  {/* Input - fixed at bottom */}
  <div className="border-t p-4" style={{ borderColor: 'var(--border)' }}>
    <ChatInput onSend={handleSend} isStreaming={isStreaming} />
  </div>
</div>
```

