# UI Layout Pattern

## Overview

This document describes the layout architecture for Amber's frontend UI. The layout follows a **sidebar + content** pattern where the sidebar is fixed on the left and the main content area adjusts dynamically based on sidebar state.

## Core Layout Structure

### Root Container (`page.tsx`)


<div className="h-screen overflow-hidden">
  <Sidebar /> {/* Fixed position */}
  <main style={{ marginLeft: sidebarWidth }}>
    {/* Panel components */}
  </main>
  <ThemeToggle />
</div>


**Key CSS Classes:**
- Root div: `h-screen overflow-hidden` - Locks viewport height, prevents page scrolling
- Main: `h-full overflow-hidden` - Fills parent height, no scroll at this level
- Main uses dynamic `marginLeft` style to offset for sidebar width

### Sidebar Component

**Position:** Fixed to left edge, spans full viewport height

**Features:**
- Resizable width (260px - 480px, default 320px)
- Collapsible to icon-only mode (72px)
- Two main tabs: Chat (History) and Database
- Tools section at bottom for additional panels
- Mobile: Overlay mode with backdrop

**Structure:**

<aside className="fixed left-0 top-0 z-40 h-screen">
  <div className="flex flex-col h-full min-h-0">
    {/* Logo/Brand */}
    {/* Tab Navigation */}
    {/* Tab Content (scrollable) */}
    {/* Tools Section */}
  </div>
</aside>


**Navigation Flow:**
- Chat tab → Shows conversation history → Clicking history loads chat view
- Database tab → Shows documents → Clicking document loads document view
- Tools section → Direct buttons to switch between Graph, ChatTuning, Classification, Comblocks

## Panel Components

All panel components should follow this pattern to ensure proper scrolling behavior:

### Standard Panel Pattern


export default function PanelName() {
  return (
    <div className="h-full flex flex-col">
      {/* Optional: Fixed Header */}
      <div className="flex-shrink-0">
        {/* Header content */}
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {/* Main content */}
      </div>

      {/* Optional: Fixed Footer */}
      <div className="flex-shrink-0">
        {/* Footer content */}
      </div>
    </div>
  )
}


**Critical CSS Classes Explained:**

1. **Root container:** `h-full flex flex-col`
   - `h-full` - Fills parent (main) height
   - `flex flex-col` - Vertical stacking

2. **Fixed sections:** `flex-shrink-0`
   - Prevents collapse when content overflows
   - Used for headers, footers, fixed toolbars

3. **Scrollable content:** `flex-1 min-h-0 overflow-y-auto`
   - `flex-1` - Grows to fill available space
   - `min-h-0` - **CRITICAL** - Allows flex child to shrink below content size, enabling scroll
   - `overflow-y-auto` - Adds vertical scrollbar when needed

### Panel Examples

#### Chat Interface

<div className="h-full flex flex-col">
  <ConnectionStatus /> {/* flex-shrink-0 */}
  <div className="flex-1 min-h-0 overflow-y-auto">
    {/* Messages */}
  </div>
  <div className="flex-shrink-0">
    <ChatInput /> {/* Sticky bottom */}
  </div>
</div>


#### Graph View

<div className="flex-1 h-full flex flex-col gap-6 p-6">
  <h2>Graph Explorer</h2>
  {/* Filters */}
  <div className="flex-1 relative">
    <ForceGraph3D /> {/* Fills available space */}
  </div>
</div>


#### ChatTuning / Classification Panels

<div className="h-full flex flex-col">
  <div className="flex-shrink-0 border-b p-4">
    {/* Header */}
  </div>
  <div className="flex-1 min-h-0 overflow-y-auto p-6">
    {/* Form content */}
  </div>
  <div className="flex-shrink-0 border-t p-4">
    {/* Save/Reset buttons */}
  </div>
</div>


## Common Pitfalls to Avoid

### ❌ DON'T: Add padding to root panel container with flex-1

// BAD - outer padding causes height overflow
<div className="flex-1 h-full flex flex-col gap-4 p-6">


### ✅ DO: Move padding inside or use nested container

// GOOD - padding inside scrollable area
<div className="flex-1 h-full flex flex-col gap-6">
  <div className="px-6 pt-6">Header</div>
  <div className="flex-1 min-h-0 overflow-y-auto px-6">Content</div>
</div>


### ❌ DON'T: Forget min-h-0 on flex children

// BAD - will expand beyond parent
<div className="flex-1 overflow-y-auto">


### ✅ DO: Always include min-h-0 with flex-1

// GOOD - enables proper scrolling
<div className="flex-1 min-h-0 overflow-y-auto">


### ❌ DON'T: Use h-screen on inner elements

// BAD - creates multiple viewport-sized elements
<main className="h-screen">
  <div className="h-screen">...</div>
</main>


### ✅ DO: Use h-screen only on root, h-full elsewhere

// GOOD - single viewport lock at root
<div className="h-screen overflow-hidden">
  <main className="h-full">
    <div className="h-full">...</div>
  </main>
</div>


## Responsive Behavior

### Desktop (lg and up)
- Sidebar is always visible
- Sidebar can be collapsed to icon-only (72px)
- Sidebar width is resizable via drag handle
- Main content adjusts marginLeft dynamically

### Mobile (below lg)
- Sidebar becomes overlay (z-50)
- Toggle button in top-left to open/close
- Dark backdrop when open
- Sidebar slides in from left
- Main content stays at marginLeft: 0

## Z-Index Layers


z-60  - Fullscreen modal close button
z-50  - Sidebar (mobile), Mobile toggle button
z-40  - Sidebar (desktop)
z-30  - Modal overlays, loading states
z-20  - Hover panels (Graph tooltips)
z-10  - Graph canvas, modal content


## Active Views

Available views controlled by `chatStore.activeView`:
- `chat` - Chat interface with messages and input
- `document` - Document detail view with chunks/entities
- `graph` - 3D force graph visualization
- `chatTuning` - RAG parameter configuration
- `classification` - Entity types and relationships config
- `comblocks` - Visual tree builder for communities

## Branding & Favicon Configuration

### Branding Setup

Branding is configured via `/frontend/public/branding.json`:

```json
{
  "title": "Amber - Chat with the Docs",
  "description": "Intelligent document Q&A powered by graph-based RAG",
  "heading": "Amber",
  "tagline": "Document Intelligence",
  "use_image": true,
  "image_path": "/amber.svg",
  "short_name": "Amber"
}
```

**Key Points:**
- `title` - Browser tab title
- `heading` - Sidebar header text
- `tagline` - Subtitle below heading in sidebar
- `use_image` - Enable/disable logo image
- `image_path` - Path to logo SVG (relative to `/public`)
- `short_name` - Short name used with logo

### Favicon Configuration

Favicon is set in `layout.tsx`:

```tsx
export const metadata: Metadata = {
  title: branding.title,
  description: branding.description,
  icons: {
    icon: [
      { url: '/favicon.svg', type: 'image/svg+xml' },
      { url: branding.use_image && branding.image_path ? branding.image_path : '/favicon.svg' }
    ],
  },
}
```

**Required Files:**
- `/frontend/public/favicon.svg` - Primary favicon (SVG format preferred)
- `/frontend/public/amber.svg` - Branding logo used in sidebar
- `/frontend/public/branding.json` - Branding configuration

**Branding Path Fix:**
The `brandingPath` in `layout.tsx` must be:
```tsx
const brandingPath = path.join(process.cwd(), 'public', 'branding.json')
```
NOT `path.join(process.cwd(), 'frontend', 'public', 'branding.json')` because `process.cwd()` is already in the frontend directory when Next.js runs.

### Sidebar Branding Display

Branding appears in sidebar header (when not collapsed):

```tsx
<div className="p-6 border-b border-secondary-200 dark:border-secondary-700">
  <h1 className="text-lg branding-heading flex items-center">
    {branding?.use_image && branding.image_path ? (
      <img src={branding.image_path} alt={branding.short_name || branding.heading} className="w-6 h-6 mr-2" />
    ) : null}
    <span>{branding?.use_image ? (branding.short_name || branding.heading) : branding?.heading}</span>
  </h1>
  <p className="text-sm text-secondary-500 mt-1">{branding?.tagline}</p>
</div>
```

## Theme Integration

- All components support dark mode via `dark:` Tailwind variants
- Theme toggle fixed in bottom-right (outside main flow)
- Color variables: `--primary-500`, `--secondary-*`, `--neon-glow`
- Use semantic colors: `bg-white dark:bg-secondary-900`

## Testing Checklist

When implementing or modifying panels:

- [ ] No page-level scrolling occurs
- [ ] Panel content scrolls independently
- [ ] Fixed headers/footers stay visible
- [ ] Layout works with sidebar open/collapsed
- [ ] Mobile overlay mode functions correctly
- [ ] Dark mode styles applied
- [ ] Resize handle doesn't interfere with content
- [ ] Height fills viewport on all screen sizes
- [ ] No horizontal overflow or unwanted scrollbars

## Maintenance

When adding new panels:
1. Follow the standard panel pattern above
2. Add navigation button to Sidebar Tools section
3. Add case to activeView switch in page.tsx
4. Ensure h-full on root, flex-1 min-h-0 on scrollable areas
5. Test with sidebar open, collapsed, and on mobile

When modifying layout:
1. Preserve h-screen overflow-hidden on root
2. Keep sidebar fixed positioning
3. Maintain dynamic marginLeft on main
4. Don't introduce new z-index layers without documenting
5. Test all active views after changes
