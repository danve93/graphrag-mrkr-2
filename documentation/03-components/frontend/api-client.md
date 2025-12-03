# API Client Component

Type-safe HTTP client for backend API communication.

## Overview

The API Client provides a centralized, type-safe interface for all frontend-backend communication. It handles request/response serialization, error handling, authentication, and provides React Query integration for efficient data fetching and caching.

**Location**: `frontend/src/lib/api-client.ts`
**Features**: Type-safe requests, error handling, authentication, retry logic
**Integration**: React Query for caching and state management

## Architecture

```
┌────────────────────────────────────────────────────────┐
│              API Client Architecture                    │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Base HTTP Client                       │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Request Interceptor                       │  │   │
│  │  │  ├─ Add auth headers                      │  │   │
│  │  │  ├─ Add content-type                      │  │   │
│  │  │  └─ Add request ID                        │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Response Interceptor                      │  │   │
│  │  │  ├─ Parse JSON                            │  │   │
│  │  │  ├─ Handle errors                         │  │   │
│  │  │  └─ Extract data                          │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Resource Endpoints                     │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Chat API                                  │  │   │
│  │  │  ├─ sendMessage() → POST /api/chat       │  │   │
│  │  │  ├─ streamChat() → SSE /api/chat/stream  │  │   │
│  │  │  └─ getHistory() → GET /api/history      │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Document API                              │  │   │
│  │  │  ├─ getDocuments() → GET /api/documents  │  │   │
│  │  │  ├─ uploadDocument() → POST /api/upload  │  │   │
│  │  │  ├─ deleteDocument() → DELETE /api/docs  │  │   │
│  │  │  └─ getChunks() → GET /api/docs/:id/chks │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │                                                  │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Database API                              │  │   │
│  │  │  ├─ getStats() → GET /api/database/stats │  │   │
│  │  │  ├─ getGraph() → GET /api/database/graph │  │   │
│  │  │  └─ clearAll() → POST /api/database/clear│  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Error Handling                         │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ APIError class (extends Error)            │  │   │
│  │  │  ├─ statusCode: number                    │  │   │
│  │  │  ├─ message: string                       │  │   │
│  │  │  └─ details?: any                         │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Core Implementation

### Base HTTP Client

```typescript
// frontend/src/lib/api-client.ts
import type {
  ChatRequest,
  ChatResponse,
  Document,
  Source,
  DatabaseStats,
  GraphData,
} from '@/types';

// Custom error class
export class APIError extends Error {
  statusCode: number;
  details?: any;

  constructor(message: string, statusCode: number, details?: any) {
    super(message);
    this.name = 'APIError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

// Base request function
async function request<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const defaultHeaders = {
    'Content-Type': 'application/json',
  };

  const config: RequestInit = {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  };

  try {
    const response = await fetch(url, config);

    // Handle non-OK responses
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        errorData.message || `HTTP ${response.status}`,
        response.status,
        errorData
      );
    }

    // Parse JSON response
    const data = await response.json();
    return data;
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    
    // Network or parsing error
    throw new APIError(
      error instanceof Error ? error.message : 'Request failed',
      0
    );
  }
}

// HTTP method helpers
export const api = {
  get: <T>(url: string, options?: RequestInit) =>
    request<T>(url, { ...options, method: 'GET' }),

  post: <T>(url: string, body?: any, options?: RequestInit) =>
    request<T>(url, {
      ...options,
      method: 'POST',
      body: JSON.stringify(body),
    }),

  put: <T>(url: string, body?: any, options?: RequestInit) =>
    request<T>(url, {
      ...options,
      method: 'PUT',
      body: JSON.stringify(body),
    }),

  delete: <T>(url: string, options?: RequestInit) =>
    request<T>(url, { ...options, method: 'DELETE' }),
};
```

## Chat API

### Send Message

```typescript
// frontend/src/lib/api-client.ts (continued)

export async function sendMessage(
  request: ChatRequest
): Promise<ChatResponse> {
  return api.post<ChatResponse>('/api/chat', request);
}
```

### Stream Chat Response

```typescript
export interface StreamCallbacks {
  onStage?: (stage: string) => void;
  onToken?: (token: string) => void;
  onSources?: (sources: Source[]) => void;
  onQualityScore?: (score: number) => void;
  onFollowUps?: (questions: string[]) => void;
  onComplete?: () => void;
  onError?: (error: Error) => void;
}

export async function streamChatResponse(
  request: ChatRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new APIError(
      error.message || 'Stream failed',
      response.status
    );
  }

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error('Response body is not readable');
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        callbacks.onComplete?.();
        break;
      }

      // Decode chunk
      const chunk = decoder.decode(value, { stream: true });
      
      // Parse SSE events
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        
        const dataStr = line.slice(6); // Remove "data: " prefix
        
        if (dataStr === '[DONE]') {
          callbacks.onComplete?.();
          continue;
        }

        try {
          const data = JSON.parse(dataStr);
          
          // Route to appropriate callback
          if (data.stage) {
            callbacks.onStage?.(data.stage);
          }
          
          if (data.token) {
            callbacks.onToken?.(data.token);
          }
          
          if (data.sources) {
            callbacks.onSources?.(data.sources);
          }
          
          if (data.quality_score !== undefined) {
            callbacks.onQualityScore?.(data.quality_score);
          }
          
          if (data.follow_up_questions) {
            callbacks.onFollowUps?.(data.follow_up_questions);
          }
          
          if (data.error) {
            callbacks.onError?.(new Error(data.error));
          }
        } catch (parseError) {
          console.error('Failed to parse SSE data:', dataStr);
        }
      }
    }
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      // Request was aborted, not an error
      return;
    }
    callbacks.onError?.(error as Error);
  } finally {
    reader.releaseLock();
  }
}
```

### Get Chat History

```typescript
export async function getChatHistory(
  sessionId: string
): Promise<ChatResponse[]> {
  return api.get<ChatResponse[]>(`/api/history/${sessionId}`);
}

export async function clearChatHistory(sessionId: string): Promise<void> {
  return api.delete(`/api/history/${sessionId}`);
}
```

## Document API

### Get Documents

```typescript
export async function getDocuments(): Promise<Document[]> {
  return api.get<Document[]>('/api/documents');
}

export async function getDocument(documentId: string): Promise<Document> {
  return api.get<Document>(`/api/documents/${documentId}`);
}
```

### Upload Document

```typescript
export async function uploadDocument(
  file: File,
  onProgress?: (progress: number) => void
): Promise<{ document_id: string; filename: string }> {
  const formData = new FormData();
  formData.append('file', file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    // Progress tracking
    if (onProgress) {
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const progress = (e.loaded / e.total) * 100;
          onProgress(progress);
        }
      });
    }

    // Success handler
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const response = JSON.parse(xhr.responseText);
          resolve(response);
        } catch (error) {
          reject(new APIError('Invalid response', xhr.status));
        }
      } else {
        try {
          const error = JSON.parse(xhr.responseText);
          reject(new APIError(error.message, xhr.status, error));
        } catch {
          reject(new APIError(`Upload failed: ${xhr.status}`, xhr.status));
        }
      }
    });

    // Error handler
    xhr.addEventListener('error', () => {
      reject(new APIError('Network error', 0));
    });

    // Abort handler
    xhr.addEventListener('abort', () => {
      reject(new APIError('Upload aborted', 0));
    });

    // Send request
    xhr.open('POST', '/api/upload');
    xhr.send(formData);
  });
}
```

### Delete Document

```typescript
export async function deleteDocument(documentId: string): Promise<void> {
  return api.delete(`/api/documents/${documentId}`);
}
```

### Get Document Chunks

```typescript
export interface PaginatedResponse<T> {
  document_id: string;
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  chunks: T[];
}

export async function getDocumentChunks(
  documentId: string,
  limit: number = 10,
  offset: number = 0
): Promise<PaginatedResponse<any>> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });

  return api.get(`/api/documents/${documentId}/chunks?${params}`);
}
```

## Database API

### Get Statistics

```typescript
export async function getDatabaseStats(): Promise<DatabaseStats> {
  return api.get<DatabaseStats>('/api/database/stats');
}

export async function getCacheStats(): Promise<any> {
  return api.get('/api/database/cache-stats');
}
```

### Get Graph Data

```typescript
export async function getGraphData(): Promise<GraphData> {
  return api.get<GraphData>('/api/database/graph');
}

export async function getNodeNeighbors(nodeId: string): Promise<GraphData> {
  return api.get<GraphData>(`/api/database/graph/neighbors/${nodeId}`);
}
```

### Clear Database

```typescript
export async function clearDatabase(): Promise<void> {
  return api.post('/api/database/clear');
}
```

## Jobs API

### Get Job Status

```typescript
export interface JobStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  result?: any;
  error?: string;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  return api.get<JobStatus>(`/api/jobs/${jobId}`);
}

export async function pollJobStatus(
  jobId: string,
  onProgress: (status: JobStatus) => void,
  interval: number = 1000
): Promise<JobStatus> {
  return new Promise((resolve, reject) => {
    const poll = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId);
        onProgress(status);

        if (status.status === 'completed') {
          clearInterval(poll);
          resolve(status);
        } else if (status.status === 'failed') {
          clearInterval(poll);
          reject(new Error(status.error || 'Job failed'));
        }
      } catch (error) {
        clearInterval(poll);
        reject(error);
      }
    }, interval);
  });
}
```

## React Query Integration

### Query Hooks

```typescript
// frontend/src/lib/hooks/useDocuments.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getDocuments, deleteDocument, uploadDocument } from '@/lib/api-client';

export function useDocuments() {
  return useQuery({
    queryKey: ['documents'],
    queryFn: getDocuments,
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => {
      // Invalidate documents cache
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });
}

export function useUploadDocument() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (file: File) => uploadDocument(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });
}
```

### Database Stats Hook

```typescript
// frontend/src/lib/hooks/useDatabaseStats.ts
import { useQuery } from '@tanstack/react-query';
import { getDatabaseStats } from '@/lib/api-client';

export function useDatabaseStats(refetchInterval?: number) {
  return useQuery({
    queryKey: ['database-stats'],
    queryFn: getDatabaseStats,
    refetchInterval, // Auto-refresh if provided
  });
}
```

## Error Handling

### Error Boundary Integration

```typescript
// frontend/src/lib/error-handler.ts
export function handleAPIError(error: unknown): string {
  if (error instanceof APIError) {
    switch (error.statusCode) {
      case 400:
        return 'Invalid request. Please check your input.';
      case 401:
        return 'Authentication required. Please log in.';
      case 403:
        return 'Access denied.';
      case 404:
        return 'Resource not found.';
      case 500:
        return 'Server error. Please try again later.';
      default:
        return error.message || 'An error occurred';
    }
  }
  
  if (error instanceof Error) {
    return error.message;
  }
  
  return 'An unexpected error occurred';
}
```

### Retry Logic

```typescript
// frontend/src/lib/retry.ts
export async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> {
  let lastError: Error | undefined;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      
      // Don't retry client errors (4xx)
      if (error instanceof APIError && error.statusCode >= 400 && error.statusCode < 500) {
        throw error;
      }
      
      // Wait before retry (exponential backoff)
      if (attempt < maxRetries - 1) {
        await new Promise((resolve) => setTimeout(resolve, delay * Math.pow(2, attempt)));
      }
    }
  }
  
  throw lastError;
}
```

## Testing

### Mock API Client

```typescript
// frontend/src/lib/__mocks__/api-client.ts
export const mockDocuments = [
  {
    id: '1',
    name: 'test.pdf',
    file_type: 'pdf',
    file_size: 1024,
    chunk_count: 5,
    created_at: new Date().toISOString(),
  },
];

export const getDocuments = jest.fn().mockResolvedValue(mockDocuments);
export const deleteDocument = jest.fn().mockResolvedValue(undefined);
export const uploadDocument = jest.fn().mockResolvedValue({
  document_id: '2',
  filename: 'new.pdf',
});
```

### Unit Tests

```typescript
// frontend/src/lib/__tests__/api-client.test.ts
import { api, APIError } from '../api-client';

global.fetch = jest.fn();

describe('API Client', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('makes GET request', async () => {
    const mockData = { id: '1', name: 'Test' };
    
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockData,
    });

    const result = await api.get('/test');
    
    expect(global.fetch).toHaveBeenCalledWith('/test', {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    expect(result).toEqual(mockData);
  });

  it('handles error responses', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: async () => ({ message: 'Not found' }),
    });

    await expect(api.get('/test')).rejects.toThrow(APIError);
  });
});
```

## Troubleshooting

### Common Issues

**Issue**: CORS errors
```typescript
// Check API base URL
console.log('API URL:', process.env.NEXT_PUBLIC_API_URL);

// Verify CORS headers on backend
// backend: FastAPI CORS middleware should allow frontend origin
```

**Issue**: Request timeout
```typescript
// Add timeout to fetch
const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 10000);

try {
  const response = await fetch(url, { signal: controller.signal });
} finally {
  clearTimeout(timeout);
}
```

**Issue**: Stream disconnects
```typescript
// Implement reconnection logic
let retryCount = 0;
const maxRetries = 3;

async function connectWithRetry() {
  try {
    await streamChatResponse(request, callbacks);
  } catch (error) {
    if (retryCount < maxRetries) {
      retryCount++;
      setTimeout(connectWithRetry, 1000 * retryCount);
    }
  }
}
```

## Related Documentation

- [Chat Interface](03-components/frontend/chat-interface.md)
- [SSE Streaming](03-components/frontend/sse-streaming.md)
- [Chat API](06-api-reference/chat.md)
- [Document API](06-api-reference/documents.md)
