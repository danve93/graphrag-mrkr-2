# Document View Component

Document management interface for browsing, viewing, and managing ingested documents.

## Overview

The Document View provides a comprehensive interface for exploring uploaded documents, their metadata, chunk breakdowns, and relationships in the knowledge graph. It integrates with the backend document API to display document lists, preview content, and manage document lifecycle operations.

**Location**: `frontend/src/components/documents/`
**Features**: Document listing, chunk pagination, metadata display, delete operations
**State Management**: React Query for server state

## Architecture

```
┌────────────────────────────────────────────────────────┐
│           Document View Architecture                    │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │         DocumentList Component                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Search/Filter Bar                         │  │   │
│  │  │  ├─ Search input (by name)                │  │   │
│  │  │  ├─ Type filter dropdown                  │  │   │
│  │  │  └─ Sort controls (name, date, size)      │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ DocumentCard (repeated)                   │  │   │
│  │  │  ├─ Document icon (by type)               │  │   │
│  │  │  ├─ Name + file size                      │  │   │
│  │  │  ├─ Chunk count badge                     │  │   │
│  │  │  ├─ Upload date                           │  │   │
│  │  │  └─ Actions (view, delete)                │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Pagination Controls                       │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │      DocumentDetail Component (Modal/Panel)     │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Document Metadata                         │  │   │
│  │  │  ├─ Title, author, creation date          │  │   │
│  │  │  ├─ File type, size, hash                 │  │   │
│  │  │  └─ Processing status                     │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ ChunkList                                 │  │   │
│  │  │  ┌─────────────────────────────────────┐  │  │   │
│  │  │  │ ChunkCard                           │  │  │   │
│  │  │  │   ├─ Chunk index/position           │  │  │   │
│  │  │  │   ├─ Text preview (expandable)      │  │  │   │
│  │  │  │   ├─ Token count                    │  │  │   │
│  │  │  │   ├─ Quality score badge            │  │  │   │
│  │  │  │   └─ Entity tags                    │  │  │   │
│  │  │  └─────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Related Entities/Graph Preview            │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Core Components

### DocumentList (Main Container)

```typescript
// frontend/src/components/documents/DocumentList.tsx
'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Search, Filter, Loader2 } from 'lucide-react';
import { DocumentCard } from './DocumentCard';
import { DocumentDetail } from './DocumentDetail';
import { deleteDocument, getDocuments } from '@/lib/api-client';
import type { Document } from '@/types';

export function DocumentList() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const queryClient = useQueryClient();

  // Fetch documents
  const { data: documents, isLoading, error } = useQuery({
    queryKey: ['documents'],
    queryFn: getDocuments,
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });

  // Filter documents
  const filteredDocs = documents?.filter((doc) =>
    doc.name.toLowerCase().includes(searchQuery.toLowerCase())
  ) || [];

  const handleDelete = async (docId: string) => {
    if (confirm('Are you sure you want to delete this document?')) {
      await deleteMutation.mutateAsync(docId);
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center text-error-500">
        Failed to load documents
      </div>
    );
  }

  return (
    <>
      <div className="flex h-full flex-col">
        {/* Header */}
        <div className="border-b border-neutral-200 bg-white px-6 py-4 dark:border-neutral-800 dark:bg-neutral-900">
          <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
            Documents
          </h1>
          <p className="text-sm text-neutral-500">
            {documents?.length || 0} documents indexed
          </p>
        </div>

        {/* Search bar */}
        <div className="border-b border-neutral-200 bg-neutral-50 px-6 py-3 dark:border-neutral-800 dark:bg-neutral-850">
          <div className="flex items-center gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" />
              <input
                type="text"
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full rounded-lg border border-neutral-300 bg-white py-2 pl-10 pr-4 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500 dark:border-neutral-700 dark:bg-neutral-800"
              />
            </div>
            
            <button className="flex items-center gap-2 rounded-lg border border-neutral-300 bg-white px-4 py-2 text-sm hover:bg-neutral-50 dark:border-neutral-700 dark:bg-neutral-800 dark:hover:bg-neutral-750">
              <Filter className="h-4 w-4" />
              Filter
            </button>
          </div>
        </div>

        {/* Document grid */}
        <div className="flex-1 overflow-y-auto p-6">
          {filteredDocs.length === 0 ? (
            <div className="flex h-full items-center justify-center text-neutral-500">
              <p>No documents found</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {filteredDocs.map((doc) => (
                <DocumentCard
                  key={doc.id}
                  document={doc}
                  onView={() => setSelectedDoc(doc)}
                  onDelete={() => handleDelete(doc.id)}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Detail modal */}
      {selectedDoc && (
        <DocumentDetail
          document={selectedDoc}
          onClose={() => setSelectedDoc(null)}
        />
      )}
    </>
  );
}
```

### DocumentCard Component

```typescript
// frontend/src/components/documents/DocumentCard.tsx
import { FileText, File, Image, Table, MoreVertical, Trash2, Eye } from 'lucide-react';
import { formatFileSize, formatDate } from '@/lib/utils';
import type { Document } from '@/types';

interface DocumentCardProps {
  document: Document;
  onView: () => void;
  onDelete: () => void;
}

export function DocumentCard({ document, onView, onDelete }: DocumentCardProps) {
  const getIcon = () => {
    const type = document.file_type?.toLowerCase();
    
    switch (type) {
      case 'pdf':
      case 'txt':
      case 'md':
        return <FileText className="h-6 w-6" />;
      case 'xlsx':
      case 'csv':
        return <Table className="h-6 w-6" />;
      case 'jpg':
      case 'jpeg':
      case 'png':
        return <Image className="h-6 w-6" />;
      default:
        return <File className="h-6 w-6" />;
    }
  };

  return (
    <div className="group relative overflow-hidden rounded-lg border border-neutral-200 bg-white transition-shadow hover:shadow-md dark:border-neutral-800 dark:bg-neutral-900">
      {/* Icon header */}
      <div className="flex items-center justify-between border-b border-neutral-200 bg-neutral-50 px-4 py-3 dark:border-neutral-800 dark:bg-neutral-850">
        <div className="text-primary-500">{getIcon()}</div>
        
        {/* Actions dropdown */}
        <div className="relative">
          <button className="rounded p-1 hover:bg-neutral-200 dark:hover:bg-neutral-700">
            <MoreVertical className="h-4 w-4 text-neutral-500" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        <h3 className="mb-1 truncate font-medium text-neutral-900 dark:text-neutral-100">
          {document.name}
        </h3>
        
        <div className="mb-3 flex items-center gap-2 text-xs text-neutral-500">
          <span>{formatFileSize(document.file_size || 0)}</span>
          <span>•</span>
          <span>{document.chunk_count || 0} chunks</span>
        </div>
        
        <p className="mb-3 text-xs text-neutral-500">
          Uploaded {formatDate(document.created_at)}
        </p>
        
        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={onView}
            className="flex flex-1 items-center justify-center gap-1 rounded border border-neutral-300 px-3 py-1.5 text-xs font-medium hover:bg-neutral-50 dark:border-neutral-700 dark:hover:bg-neutral-800"
          >
            <Eye className="h-3.5 w-3.5" />
            View
          </button>
          
          <button
            onClick={onDelete}
            className="flex items-center justify-center gap-1 rounded border border-error-300 px-3 py-1.5 text-xs font-medium text-error-600 hover:bg-error-50 dark:border-error-700 dark:text-error-400"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}
```

### DocumentDetail Component

```typescript
// frontend/src/components/documents/DocumentDetail.tsx
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { X, FileText, Calendar, Hash, Layers } from 'lucide-react';
import { Modal } from '@/components/ui/Modal';
import { ChunkList } from './ChunkList';
import { getDocumentChunks } from '@/lib/api-client';
import { formatDate, formatFileSize } from '@/lib/utils';
import type { Document } from '@/types';

interface DocumentDetailProps {
  document: Document;
  onClose: () => void;
}

export function DocumentDetail({ document, onClose }: DocumentDetailProps) {
  const [page, setPage] = useState(0);
  const pageSize = 10;

  // Fetch chunks
  const { data: chunksData, isLoading } = useQuery({
    queryKey: ['document-chunks', document.id, page],
    queryFn: () => getDocumentChunks(document.id, pageSize, page * pageSize),
  });

  return (
    <Modal onClose={onClose} size="xl">
      <div className="flex h-full flex-col">
        {/* Header */}
        <div className="flex items-start justify-between border-b border-neutral-200 p-6 dark:border-neutral-800">
          <div className="flex items-start gap-4">
            <div className="rounded-lg bg-primary-100 p-3 dark:bg-primary-900">
              <FileText className="h-6 w-6 text-primary-600 dark:text-primary-400" />
            </div>
            
            <div>
              <h2 className="text-xl font-bold text-neutral-900 dark:text-neutral-100">
                {document.name}
              </h2>
              <p className="text-sm text-neutral-500">
                {document.file_type?.toUpperCase()} • {formatFileSize(document.file_size || 0)}
              </p>
            </div>
          </div>
          
          <button
            onClick={onClose}
            className="rounded-lg p-2 hover:bg-neutral-100 dark:hover:bg-neutral-800"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Metadata */}
        <div className="border-b border-neutral-200 bg-neutral-50 px-6 py-4 dark:border-neutral-800 dark:bg-neutral-850">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-neutral-500" />
              <div>
                <p className="text-xs text-neutral-500">Uploaded</p>
                <p className="text-sm font-medium">
                  {formatDate(document.created_at)}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-neutral-500" />
              <div>
                <p className="text-xs text-neutral-500">Chunks</p>
                <p className="text-sm font-medium">{document.chunk_count || 0}</p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Hash className="h-4 w-4 text-neutral-500" />
              <div>
                <p className="text-xs text-neutral-500">Document ID</p>
                <p className="truncate text-sm font-medium font-mono">
                  {document.id.substring(0, 8)}...
                </p>
              </div>
            </div>
            
            {document.metadata?.author && (
              <div>
                <p className="text-xs text-neutral-500">Author</p>
                <p className="text-sm font-medium">{document.metadata.author}</p>
              </div>
            )}
          </div>
        </div>

        {/* Chunks */}
        <div className="flex-1 overflow-y-auto p-6">
          <h3 className="mb-4 text-lg font-semibold">Document Chunks</h3>
          
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-neutral-500">Loading chunks...</div>
            </div>
          ) : (
            <>
              <ChunkList chunks={chunksData?.chunks || []} />
              
              {/* Pagination */}
              {chunksData && chunksData.total > pageSize && (
                <div className="mt-6 flex items-center justify-between">
                  <button
                    onClick={() => setPage(Math.max(0, page - 1))}
                    disabled={page === 0}
                    className="rounded border border-neutral-300 px-4 py-2 text-sm disabled:opacity-50 dark:border-neutral-700"
                  >
                    Previous
                  </button>
                  
                  <span className="text-sm text-neutral-500">
                    Page {page + 1} of {Math.ceil(chunksData.total / pageSize)}
                  </span>
                  
                  <button
                    onClick={() => setPage(page + 1)}
                    disabled={!chunksData.has_more}
                    className="rounded border border-neutral-300 px-4 py-2 text-sm disabled:opacity-50 dark:border-neutral-700"
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </Modal>
  );
}
```

### ChunkList Component

```typescript
// frontend/src/components/documents/ChunkList.tsx
import { useState } from 'react';
import { ChevronDown, ChevronUp, Hash } from 'lucide-react';
import { QualityBadge } from '@/components/chat/QualityBadge';
import type { Chunk } from '@/types';

interface ChunkListProps {
  chunks: Chunk[];
}

export function ChunkList({ chunks }: ChunkListProps) {
  return (
    <div className="space-y-3">
      {chunks.map((chunk) => (
        <ChunkCard key={chunk.id} chunk={chunk} />
      ))}
    </div>
  );
}

interface ChunkCardProps {
  chunk: Chunk;
}

function ChunkCard({ chunk }: ChunkCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="rounded-lg border border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex w-full items-center justify-between p-4 text-left"
      >
        <div className="flex items-center gap-3">
          <div className="rounded bg-neutral-100 px-2 py-1 text-xs font-medium dark:bg-neutral-800">
            #{chunk.chunk_index}
          </div>
          
          <div className="flex-1">
            <p className="line-clamp-1 text-sm text-neutral-700 dark:text-neutral-300">
              {chunk.text}
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {chunk.quality_score !== undefined && (
            <QualityBadge score={chunk.quality_score} />
          )}
          
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-neutral-500" />
          ) : (
            <ChevronDown className="h-4 w-4 text-neutral-500" />
          )}
        </div>
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="border-t border-neutral-200 p-4 dark:border-neutral-800">
          <div className="mb-3 text-sm text-neutral-700 dark:text-neutral-300">
            {chunk.text}
          </div>
          
          {/* Metadata */}
          <div className="flex flex-wrap gap-4 text-xs text-neutral-500">
            <div className="flex items-center gap-1">
              <Hash className="h-3 w-3" />
              <span>Chunk ID: {chunk.id.substring(0, 8)}...</span>
            </div>
            
            {chunk.token_count && (
              <span>{chunk.token_count} tokens</span>
            )}
            
            {chunk.position && (
              <span>Position: {chunk.position.start} - {chunk.position.end}</span>
            )}
          </div>
          
          {/* Entity tags */}
          {chunk.entities && chunk.entities.length > 0 && (
            <div className="mt-3">
              <p className="mb-2 text-xs font-semibold text-neutral-700 dark:text-neutral-300">
                Related Entities
              </p>
              <div className="flex flex-wrap gap-2">
                {chunk.entities.map((entity, idx) => (
                  <span
                    key={idx}
                    className="rounded-full bg-primary-100 px-2 py-1 text-xs text-primary-700 dark:bg-primary-900 dark:text-primary-300"
                  >
                    {entity}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

## Data Fetching

### API Client Functions

```typescript
// frontend/src/lib/api-client.ts
import type { Document, Chunk, PaginatedResponse } from '@/types';

export async function getDocuments(): Promise<Document[]> {
  const response = await fetch('/api/documents');
  if (!response.ok) throw new Error('Failed to fetch documents');
  return response.json();
}

export async function getDocumentChunks(
  documentId: string,
  limit: number = 10,
  offset: number = 0
): Promise<PaginatedResponse<Chunk>> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });
  
  const response = await fetch(
    `/api/documents/${documentId}/chunks?${params}`
  );
  
  if (!response.ok) throw new Error('Failed to fetch chunks');
  return response.json();
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`/api/documents/${documentId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) throw new Error('Failed to delete document');
}
```

## Type Definitions

```typescript
// frontend/src/types/index.ts
export interface Document {
  id: string;
  name: string;
  file_type: string;
  file_size: number;
  chunk_count: number;
  created_at: string;
  metadata?: {
    author?: string;
    title?: string;
    created_date?: string;
    [key: string]: any;
  };
}

export interface Chunk {
  id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  token_count?: number;
  quality_score?: number;
  position?: {
    start: number;
    end: number;
  };
  entities?: string[];
  metadata?: Record<string, any>;
}

export interface PaginatedResponse<T> {
  document_id: string;
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  chunks: T[];
}
```

## Utility Functions

```typescript
// frontend/src/lib/utils.ts
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  
  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
  });
}
```

## Testing

### Component Tests

```typescript
// frontend/src/components/documents/__tests__/DocumentList.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DocumentList } from '../DocumentList';
import * as apiClient from '@/lib/api-client';

jest.mock('@/lib/api-client');

const mockDocuments = [
  {
    id: '1',
    name: 'Test.pdf',
    file_type: 'pdf',
    file_size: 1024000,
    chunk_count: 10,
    created_at: new Date().toISOString(),
  },
];

describe('DocumentList', () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  beforeEach(() => {
    jest.mocked(apiClient.getDocuments).mockResolvedValue(mockDocuments);
  });

  it('renders document list', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <DocumentList />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('Test.pdf')).toBeInTheDocument();
    });
  });

  it('filters documents by search', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <DocumentList />
      </QueryClientProvider>
    );

    const searchInput = screen.getByPlaceholderText(/search documents/i);
    fireEvent.change(searchInput, { target: { value: 'Test' } });

    await waitFor(() => {
      expect(screen.getByText('Test.pdf')).toBeInTheDocument();
    });
  });
});
```

## Troubleshooting

### Common Issues

**Issue**: Documents not loading
```typescript
// Check API endpoint
console.log('Fetching from:', '/api/documents');

// Verify response format
const response = await fetch('/api/documents');
const data = await response.json();
console.log('Response:', data);
```

**Issue**: Pagination not working
```typescript
// Verify offset calculation
const offset = page * pageSize;
console.log('Page:', page, 'Offset:', offset);

// Check has_more flag
console.log('Has more:', chunksData?.has_more);
```

**Issue**: Delete confirmation not showing
```typescript
// Ensure confirm dialog is not blocked
const result = window.confirm('Delete this document?');
if (result) {
  // Proceed with deletion
}
```

## Related Documentation

- [Chat Interface](03-components/frontend/chat-interface.md)
- [Upload System](04-features/document-upload.md)
- [Document API](06-api-reference/documents.md)
- [Ingestion Pipeline](03-components/ingestion/document-processor.md)
