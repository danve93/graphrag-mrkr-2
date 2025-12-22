import type { DocumentDetails, DocumentChunk, DocumentTextPayload } from '@/types'
import type { ProcessingProgressResponse } from '@/types/upload'
import type { GraphResponse } from '@/types/graph'

// Use an empty default so client-side code issues relative requests
// (e.g. `/api/...`) which are proxied by Next.js. If you need to override
// (e.g. remote API), set `NEXT_PUBLIC_API_URL` at build/runtime.
export const API_URL = process.env.NEXT_PUBLIC_API_URL || ''

let authToken: string | null = null;
if (typeof window !== 'undefined') {
  authToken = localStorage.getItem('authToken');
}

export const setAuthToken = (token: string | null) => {
  authToken = token;
  if (token) localStorage.setItem('authToken', token);
  else localStorage.removeItem('authToken');
}

async function fetchWithAuth(url: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers);
  if (authToken) {
    headers.set('Authorization', `Bearer ${authToken}`);
  }
  // Ensure cookies (admin_session) are sent for same-origin admin endpoints.
  // Default to `include` so the backend can validate `admin_session` cookie.
  const fetchOptions: RequestInit = { ...options, headers };
  if (!('credentials' in fetchOptions)) {
    fetchOptions.credentials = 'include';
  }
  return fetch(url, fetchOptions);
}

export const api = {
  setAuthToken,
  // Identity
  async identifyUser(username?: string): Promise<{ user_id: string; token: string; role: string; is_new: boolean }> {
    const response = await fetchWithAuth(`${API_URL}/api/users/identify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username })
    });

    if (!response.ok) {
      // If identify fails, we might want to clear token if it was invalid?
      // But identify usually generates new one.
      throw new Error(`Identity error: ${response.statusText}`)
    }
    return response.json();
  },

  async adminLogout() {
    try {
      await fetch(`${API_URL}/api/admin/user-management/logout`, {
        method: 'POST',
        credentials: 'include'
      })
    } catch (e) {
      console.error("Admin logout failed", e)
    }
  },

  // Chat endpoints
  async sendMessage(
    data: {
      message: string
      session_id?: string
      model?: string
      retrieval_mode?: string
      top_k?: number
      temperature?: number
      top_p?: number
      use_multi_hop?: boolean
      stream?: boolean
      context_documents?: string[]
      context_document_labels?: string[]
      context_hashtags?: string[]
      chunk_weight?: number
      entity_weight?: number
      path_weight?: number
      max_hops?: number
      beam_size?: number
      graph_expansion_depth?: number
      restrict_to_context?: boolean
      llm_model?: string
      embedding_model?: string
      llm_overrides?: {
        model?: string
        temperature?: number
        top_k?: number
      }
      category_filter?: string[]
    },
    options?: { signal?: AbortSignal }
  ) {
    const response = await fetchWithAuth(`${API_URL}/api/chat/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
      signal: options?.signal,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response
  },

  // History endpoints
  async getHistory() {
    const response = await fetchWithAuth(`${API_URL}/api/history/sessions`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getConversation(sessionId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/history/${sessionId}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteConversation(sessionId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/history/${sessionId}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async clearHistory() {
    const response = await fetchWithAuth(`${API_URL}/api/history/clear`, {
      method: 'POST',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // Database endpoints
  async getStats(includeSuggestionStatus: boolean = false) {
    const params = new URLSearchParams()
    if (includeSuggestionStatus) {
      params.append('include_suggestion_status', 'true')
    }
    const queryString = params.toString() ? `?${params.toString()}` : ''
    const response = await fetchWithAuth(`${API_URL}/api/database/stats${queryString}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async uploadFile(file: File) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetchWithAuth(`${API_URL}/api/database/upload`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async stageFile(file: File) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetchWithAuth(`${API_URL}/api/database/stage`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getFolders() {
    const response = await fetchWithAuth(`${API_URL}/api/database/folders`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async createFolder(name: string) {
    const response = await fetchWithAuth(`${API_URL}/api/database/folders`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name }),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async renameFolder(folderId: string, name: string) {
    const response = await fetchWithAuth(`${API_URL}/api/database/folders/${folderId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name }),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteFolder(folderId: string, mode: 'move_to_root' | 'delete_documents') {
    const response = await fetchWithAuth(`${API_URL}/api/database/folders/${folderId}?mode=${mode}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async moveDocumentToFolder(documentId: string, folderId: string | null) {
    const response = await fetchWithAuth(`${API_URL}/api/database/documents/${documentId}/folder`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ folder_id: folderId }),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async reorderDocuments(folderId: string | null, documentIds: string[]) {
    const response = await fetchWithAuth(`${API_URL}/api/database/documents/order`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ folder_id: folderId, document_ids: documentIds }),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getStagedDocuments() {
    const response = await fetchWithAuth(`${API_URL}/api/database/staged`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteStagedDocument(fileId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/database/staged/${fileId}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async processDocuments(fileIds: string[]) {
    const response = await fetchWithAuth(`${API_URL}/api/database/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file_ids: fileIds }),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getProcessingProgress(fileId?: string): Promise<ProcessingProgressResponse> {
    const url = fileId
      ? `${API_URL}/api/database/progress/${fileId}`
      : `${API_URL}/api/database/progress`
    const response = await fetchWithAuth(url)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteDocument(documentId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/database/documents/${documentId}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async clearDatabase(options?: { clearKnowledgeBase: boolean; clearConversations: boolean }) {
    const response = await fetchWithAuth(`${API_URL}/api/database/clear`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options || { clearKnowledgeBase: true, clearConversations: true }),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async cleanupOrphans(): Promise<{ status: string; chunks_deleted: number; entities_deleted: number }> {
    const response = await fetchWithAuth(`${API_URL}/api/database/cleanup-orphans`, {
      method: 'POST',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async reprocessDocumentChunks(documentId: string) {
    const response = await fetchWithAuth(
      `${API_URL}/api/database/documents/${documentId}/process/chunks`,
      {
        method: 'POST',
      }
    )
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async reprocessDocumentEntities(documentId: string) {
    const response = await fetchWithAuth(
      `${API_URL}/api/database/documents/${documentId}/process/entities`,
      {
        method: 'POST',
      }
    )
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocuments(includeSuggestionStatus: boolean = false) {
    const params = new URLSearchParams()
    if (includeSuggestionStatus) {
      params.append('include_suggestion_status', 'true')
    }
    const queryString = params.toString() ? `?${params.toString()}` : ''
    const response = await fetchWithAuth(`${API_URL}/api/database/documents${queryString}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getHashtags() {
    const response = await fetchWithAuth(`${API_URL}/api/database/hashtags`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocument(documentId: string): Promise<DocumentDetails> {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // New: lightweight summary fetch
  async getDocumentSummary(documentId: string): Promise<{
    id: string
    title?: string
    filename: string
    original_filename?: string
    mime_type?: string
    size_bytes?: number
    created_at?: number | string
    link?: string
    uploader?: string | null
    document_type?: string
    stats: { chunks: number; entities: number; communities: number; similarities: number }
    previews?: {
      top_communities?: Array<{ community_id: number; count: number } | null> | null
      top_similarities?: Array<{ chunk1_id: string; chunk2_id: string; score: number } | null> | null
    }
  }> {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/summary`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // New: paginated entities
  async getDocumentEntitiesPaginated(documentId: string, options?: {
    communityId?: number
    entityType?: string
    limit?: number
    offset?: number
  }): Promise<{
    document_id: string
    total: number
    limit: number
    offset: number
    has_more: boolean
    entities: Array<{
      type: string
      text: string
      community_id?: number | null
      level?: number | null
      count: number
      positions: number[]
    }>
  }> {
    const query = new URLSearchParams()
    if (options?.communityId !== undefined) query.append('community_id', String(options.communityId))
    if (options?.entityType) query.append('entity_type', options.entityType)
    if (options?.limit) query.append('limit', String(options.limit))
    if (options?.offset) query.append('offset', String(options.offset))
    const qs = query.toString()
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/entities${qs ? `?${qs}` : ''}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // New: entity summary (counts grouped by type)
  async getDocumentEntitySummary(documentId: string): Promise<{
    document_id: string
    total: number
    groups: Array<{ type: string; count: number }>
  }> {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/entity-summary`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // New: paginated similarities
  async getDocumentSimilaritiesPaginated(documentId: string, options?: {
    limit?: number
    offset?: number
    minScore?: number
    exactCount?: boolean
  }): Promise<{
    document_id: string
    total: number
    estimated: boolean
    limit: number
    offset: number
    has_more: boolean
    similarities: Array<{ chunk1_id: string; chunk2_id: string; score: number }>
  }> {
    const query = new URLSearchParams()
    if (options?.limit) query.append('limit', String(options.limit))
    if (options?.offset) query.append('offset', String(options.offset))
    if (options?.minScore !== undefined) query.append('min_score', String(options.minScore))
    if (options?.exactCount) query.append('exact_count', String(options.exactCount))
    const qs = query.toString()
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/similarities${qs ? `?${qs}` : ''}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // New: paginated chunks
  async getDocumentChunksPaginated(documentId: string, options?: { limit?: number; offset?: number }): Promise<{
    document_id: string
    total: number
    limit: number
    offset: number
    has_more: boolean
    chunks: DocumentChunk[]
  }> {
    const query = new URLSearchParams()
    if (options?.limit) query.append('limit', String(options.limit))
    if (options?.offset) query.append('offset', String(options.offset))
    const qs = query.toString()
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/chunks${qs ? `?${qs}` : ''}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // New: chunk details on-demand
  async getChunkDetails(chunkId: string): Promise<{ id: string; content: string; index: number; offset: number; document_id: string; document_name?: string | null }> {
    const response = await fetchWithAuth(`${API_URL}/api/documents/chunks/${chunkId}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // Chunk CRUD operations
  async updateChunkContent(chunkId: string, content: string, regenerateEmbedding: boolean = true): Promise<{
    status: string
    chunk: {
      id: string
      content: string
      chunk_index: number
      document_id: string
      embedding_updated: boolean
    }
  }> {
    const response = await fetchWithAuth(`${API_URL}/api/documents/chunks/${chunkId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, regenerate_embedding: regenerateEmbedding }),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteChunk(chunkId: string, cleanupOrphans: boolean = true): Promise<{
    status: string
    deleted_chunk_id: string
    document_id: string
  }> {
    const params = new URLSearchParams({ cleanup_orphans: String(cleanupOrphans) })
    const response = await fetchWithAuth(`${API_URL}/api/documents/chunks/${chunkId}?${params}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async mergeChunks(documentId: string, chunkIds: string[], mergedContent: string): Promise<{
    status: string
    merged_chunk: {
      id: string
      content: string
      chunk_index: number
      document_id: string
      merged_chunk_ids: string[]
      deleted_count: number
    }
  }> {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/chunks/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunk_ids: chunkIds, merged_content: mergedContent }),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async restoreChunk(data: {
    chunk_id: string
    document_id: string
    content: string
    chunk_index: number
  }): Promise<{ status: string; message: string }> {
    const response = await fetchWithAuth(`${API_URL}/api/documents/chunks/restore`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async unmergeChunks(chunkId: string, documentId: string): Promise<{ status: string; message: string }> {
    const response = await fetchWithAuth(`${API_URL}/api/documents/chunks/unmerge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunk_id: chunkId, document_id: documentId }),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // Chunk change history
  async getChunkChanges(documentId: string, options?: {
    chunkId?: string
    action?: string
    limit?: number
    offset?: number
  }): Promise<{
    document_id: string
    total: number
    limit: number
    offset: number
    has_more: boolean
    summary: Record<string, number>
    changes: Array<{
      id: number
      document_id: string
      chunk_id: string
      action: string
      before_content?: string
      after_content?: string
      reasoning?: string
      metadata?: Record<string, any>
      created_at: string
      error?: string
    }>
  }> {
    const query = new URLSearchParams()
    if (options?.chunkId) query.append('chunk_id', options.chunkId)
    if (options?.action) query.append('action', options.action)
    if (options?.limit) query.append('limit', String(options.limit))
    if (options?.offset) query.append('offset', String(options.offset))
    const qs = query.toString()
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/chunk-changes${qs ? `?${qs}` : ''}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async exportChunkChanges(documentId: string, includeContent: boolean = true): Promise<{
    export_date: string
    document_id: string
    total_changes: number
    changes: Array<Record<string, any>>
  }> {
    const response = await fetchWithAuth(
      `${API_URL}/api/documents/${documentId}/chunk-changes/export?include_content=${includeContent}`
    )
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // Chunk suggestions
  async getChunkSuggestions(documentId: string, options?: {
    maxSuggestions?: number
    minConfidence?: number
  }): Promise<{
    document_id: string
    total_suggestions: number
    suggestions: Array<{
      chunk_id: string
      chunk_index: number
      action: 'delete' | 'merge' | 'edit' | 'split'
      confidence: number
      reasoning: string
      pattern_name: string
      related_chunk_ids?: string[]
      suggested_content?: string
    }>
    history_analysis: {
      total_changes: number
      action_counts?: Record<string, number>
      patterns_found?: string[]
      last_change?: string
    }
  }> {
    const query = new URLSearchParams()
    if (options?.maxSuggestions) query.append('max_suggestions', String(options.maxSuggestions))
    if (options?.minConfidence) query.append('min_confidence', String(options.minConfidence))
    const qs = query.toString()
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/chunk-suggestions${qs ? `?${qs}` : ''}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // Chunk Patterns CRUD
  async getPatterns(options?: {
    enabledOnly?: boolean
    action?: string
    includeBuiltin?: boolean
  }): Promise<{
    total: number
    patterns: Array<{
      id: string
      name: string
      description: string
      match_type: string
      match_criteria: Record<string, any>
      action: string
      confidence: number
      is_builtin: boolean
      enabled: boolean
      usage_count: number
      last_used?: string
      created_at?: string
      updated_at?: string
    }>
  }> {
    const query = new URLSearchParams()
    if (options?.enabledOnly) query.append('enabled_only', 'true')
    if (options?.action) query.append('action', options.action)
    if (options?.includeBuiltin !== undefined) query.append('include_builtin', String(options.includeBuiltin))
    const qs = query.toString()
    const response = await fetchWithAuth(`${API_URL}/api/patterns${qs ? `?${qs}` : ''}`)
    if (!response.ok) throw new Error(`API error: ${response.statusText}`)
    return response.json()
  },

  async getPattern(patternId: string): Promise<Record<string, any>> {
    const response = await fetchWithAuth(`${API_URL}/api/patterns/${patternId}`)
    if (!response.ok) throw new Error(`API error: ${response.statusText}`)
    return response.json()
  },

  async createPattern(data: {
    name: string
    description?: string
    match_type: string
    match_criteria?: Record<string, any>
    action: string
    confidence?: number
  }): Promise<{ status: string; pattern: Record<string, any> }> {
    const response = await fetchWithAuth(`${API_URL}/api/patterns`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) throw new Error(`API error: ${response.statusText}`)
    return response.json()
  },

  async updatePattern(patternId: string, data: Record<string, any>): Promise<{ status: string; pattern: Record<string, any> }> {
    const response = await fetchWithAuth(`${API_URL}/api/patterns/${patternId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) throw new Error(`API error: ${response.statusText}`)
    return response.json()
  },

  async deletePattern(patternId: string): Promise<{ status: string; deleted_id: string }> {
    const response = await fetchWithAuth(`${API_URL}/api/patterns/${patternId}`, { method: 'DELETE' })
    if (!response.ok) throw new Error(`API error: ${response.statusText}`)
    return response.json()
  },

  async togglePattern(patternId: string, enabled: boolean): Promise<{ status: string; pattern: Record<string, any> }> {
    const response = await fetchWithAuth(`${API_URL}/api/patterns/${patternId}/toggle?enabled=${enabled}`, { method: 'POST' })
    if (!response.ok) throw new Error(`API error: ${response.statusText}`)
    return response.json()
  },

  async exportPatterns(includeBuiltin: boolean = false): Promise<Record<string, any>> {
    const response = await fetchWithAuth(`${API_URL}/api/patterns/export/json?include_builtin=${includeBuiltin}`)
    if (!response.ok) throw new Error(`API error: ${response.statusText}`)
    return response.json()
  },

  async importPatterns(data: Record<string, any>, overwrite: boolean = false): Promise<{ status: string; results: Record<string, number> }> {
    const response = await fetchWithAuth(`${API_URL}/api/patterns/import/json?overwrite=${overwrite}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) throw new Error(`API error: ${response.statusText}`)
    return response.json()
  },

  async getGraph(params?: {
    community_id?: number
    node_type?: string
    level?: number
    limit?: number
    document_id?: string
  }): Promise<GraphResponse> {
    const query = new URLSearchParams()
    if (params?.community_id !== undefined) {
      query.append('community_id', String(params.community_id))
    }
    if (params?.node_type) {
      query.append('node_type', params.node_type)
    }
    if (params?.level !== undefined) {
      query.append('level', String(params.level))
    }
    if (params?.limit !== undefined) {
      query.append('limit', String(params.limit))
    }
    if (params?.document_id) {
      query.append('document_id', params.document_id)
    }

    const queryString = query.toString()
    const response = await fetchWithAuth(
      `${API_URL}/api/graph/clustered${queryString ? `?${queryString}` : ''}`
    )
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocumentChunks(documentId: string): Promise<DocumentChunk[]> {
    // DEPRECATED: This method fetches ALL chunks which can be very slow for large documents.
    // Use getDocumentChunksPaginated() instead for better performance.
    console.warn(
      `[api.getDocumentChunks] Fetching all chunks for document ${documentId}. ` +
      'This may be slow for large documents. Use getDocumentChunksPaginated() instead.'
    )
    const pageSize = 200
    let offset = 0
    let all: DocumentChunk[] = []
    while (true) {
      const page = await this.getDocumentChunksPaginated(documentId, { limit: pageSize, offset })
      if (Array.isArray(page.chunks)) all = all.concat(page.chunks)
      if (!page.has_more) break
      offset += pageSize
    }
    return all
  },

  async getDocumentText(documentId: string): Promise<DocumentTextPayload> {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/text`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocumentChunkSimilarities(documentId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/similarities`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocumentPreview(
    documentId: string
  ): Promise<{ preview_url: string } | Response> {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/preview`, {
      redirect: 'follow',
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    // headers may be unavailable for opaque cross-origin responses; guard access
    const contentType = (response.headers && typeof response.headers.get === 'function') ? response.headers.get('Content-Type') || '' : ''
    if (contentType.includes('application/json')) {
      return response.json()
    }

    return response
  },

  async generateDocumentSummary(documentId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/generate-summary`, {
      method: 'POST',
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  },

  async updateDocumentHashtags(documentId: string, hashtags: string[]) {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/hashtags`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ hashtags }),
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  },

  async updateDocumentMetadata(documentId: string, metadata: Record<string, any>) {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/metadata`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ metadata }),
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  },

  async updateDocumentDetails(documentId: string, details: { title?: string; document_type?: string }) {
    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}/details`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(details),
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  },

  // Incremental document update - updates only changed chunks
  async updateDocument(documentId: string, file: File): Promise<{
    document_id: string
    status: string
    changes?: {
      unchanged_chunks: number
      added_chunks: number
      removed_chunks: number
      entities_removed: number
      relationships_cleaned: number
    }
    processing_time?: number
    error?: string
    details?: string
  }> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetchWithAuth(`${API_URL}/api/documents/${documentId}`, {
      method: 'PUT',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `API error: ${response.statusText}`)
    }

    return response.json()
  },

  // Search for documents with similar filenames (for update dialog)
  async searchSimilarDocuments(filename: string, limit: number = 10): Promise<{
    query_filename: string
    matches: Array<{
      document_id: string
      filename: string
      is_exact_match: boolean
      is_normalized_match: boolean
      similarity_score: number
      document_type?: string
      created_at?: string
      chunk_count: number
    }>
    total_found: number
  }> {
    const params = new URLSearchParams({ filename, limit: String(limit) })
    const response = await fetchWithAuth(`${API_URL}/api/documents/search-similar?${params}`)

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  },

  async hasDocumentPreview(documentId: string): Promise<boolean> {
    // Try a HEAD request first to avoid downloading the full file.
    try {
      const headResp = await fetch(`${API_URL}/api/documents/${documentId}/preview`, {
        method: 'HEAD',
        redirect: 'follow',
      })

      if (headResp.ok) return true

      // Some servers may not accept HEAD. Try a lightweight GET requesting only the first byte.
      if (headResp.status === 405) {
        const getResp = await fetch(`${API_URL}/api/documents/${documentId}/preview`, {
          method: 'GET',
          redirect: 'follow',
          headers: {
            Range: 'bytes=0-0',
          },
        })
        return getResp.ok || getResp.status === 206
      }

      return false
    } catch (err) {
      return false
    }
  },

  async getSettings() {
    const response = await fetchWithAuth(`${API_URL}/api/health`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async checkHealth(): Promise<boolean> {
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout (more forgiving)

      const response = await fetchWithAuth(`${API_URL}/api/health`, {
        method: 'GET',
        signal: controller.signal,
      })

      clearTimeout(timeoutId)
      return response.ok
    } catch (error) {
      return false
    }
  },

  // Category Management
  async getCategories(approvedOnly: boolean = false) {
    const response = await fetchWithAuth(`${API_URL}/api/classification/categories?approved_only=${approvedOnly}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async generateCategories(maxCategories: number = 10, sampleSize: number = 100) {
    const response = await fetchWithAuth(`${API_URL}/api/classification/categories/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_categories: maxCategories, sample_size: sampleSize })
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async createCategory(data: {
    name: string
    description: string
    keywords: string[]
    patterns?: string[]
    parent_id?: string | null
  }) {
    const response = await fetchWithAuth(`${API_URL}/api/classification/categories`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async approveCategory(categoryId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/classification/categories/${categoryId}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteCategory(categoryId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/classification/categories/${categoryId}`, {
      method: 'DELETE'
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async updateCategory(categoryId: string, data: { name?: string; description?: string; keywords?: string[]; patterns?: string[]; parent_id?: string | null }) {
    const response = await fetchWithAuth(`${API_URL}/api/classification/categories/${categoryId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async autoCategorizeDocuments(batchSize: number = 10) {
    const response = await fetchWithAuth(`${API_URL}/api/classification/categories/auto-assign?batch_size=${batchSize}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // Category Prompts Management
  async getPrompts() {
    const response = await fetchWithAuth(`${API_URL}/api/prompts/`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getPrompt(category: string) {
    const response = await fetchWithAuth(`${API_URL}/api/prompts/${category}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async createPrompt(data: {
    category: string
    retrieval_strategy: string
    generation_template: string
    format_instructions: string
    specificity_level: string
  }) {
    const response = await fetchWithAuth(`${API_URL}/api/prompts/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async updatePrompt(category: string, data: {
    retrieval_strategy: string
    generation_template: string
    format_instructions: string
    specificity_level: string
  }) {
    const response = await fetchWithAuth(`${API_URL}/api/prompts/${category}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deletePrompt(category: string) {
    const response = await fetchWithAuth(`${API_URL}/api/prompts/${category}`, {
      method: 'DELETE'
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async reloadPrompts() {
    const response = await fetchWithAuth(`${API_URL}/api/prompts/reload`, {
      method: 'POST'
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // Structured KG endpoints
  async executeStructuredQuery(query: string, context?: Record<string, any>) {
    const response = await fetchWithAuth(`${API_URL}/api/structured-kg/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, context })
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getStructuredKGConfig() {
    const response = await fetch(`${API_URL}/api/structured-kg/config`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getStructuredKGSchema() {
    const response = await fetch(`${API_URL}/api/structured-kg/schema`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async validateCypher(cypher: string) {
    const response = await fetch(`${API_URL}/api/structured-kg/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cypher })
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // Admin / API Keys
  async getApiKeys() {
    const response = await fetchWithAuth(`${API_URL}/api/admin/api-keys`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async createApiKey(data: { name: string; role?: string; metadata?: any }) {
    const response = await fetchWithAuth(`${API_URL}/api/admin/api-keys`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async revokeApiKey(keyId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/admin/api-keys/${keyId}`, {
      method: 'DELETE'
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },



  // Admin / Shared Chats
  async getSharedChats(limit: number = 50, offset: number = 0) {
    const response = await fetchWithAuth(`${API_URL}/api/history/admin/shared?limit=${limit}&offset=${offset}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async shareSession(sessionId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/history/${sessionId}/share`, {
      method: 'POST'
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async unshareSession(sessionId: string) {
    const response = await fetchWithAuth(`${API_URL}/api/history/${sessionId}/share`, {
      method: 'DELETE'
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

}
