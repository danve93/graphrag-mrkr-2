import type { DocumentDetails, DocumentChunk, DocumentTextPayload } from '@/types'
import type { ProcessingProgressResponse } from '@/types/upload'
import type { GraphResponse } from '@/types/graph'

// Use an empty default so client-side code issues relative requests
// (e.g. `/api/...`) which are proxied by Next.js. If you need to override
// (e.g. remote API), set `NEXT_PUBLIC_API_URL` at build/runtime.
const API_URL = process.env.NEXT_PUBLIC_API_URL || ''

export const api = {
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
    },
    options?: { signal?: AbortSignal }
  ) {
    const response = await fetch(`${API_URL}/api/chat/query`, {
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
    const response = await fetch(`${API_URL}/api/history/sessions`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getConversation(sessionId: string) {
    const response = await fetch(`${API_URL}/api/history/${sessionId}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteConversation(sessionId: string) {
    const response = await fetch(`${API_URL}/api/history/${sessionId}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async clearHistory() {
    const response = await fetch(`${API_URL}/api/history/clear`, {
      method: 'POST',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // Database endpoints
  async getStats() {
    const response = await fetch(`${API_URL}/api/database/stats`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async uploadFile(file: File) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_URL}/api/database/upload`, {
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

    const response = await fetch(`${API_URL}/api/database/stage`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getStagedDocuments() {
    const response = await fetch(`${API_URL}/api/database/staged`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteStagedDocument(fileId: string) {
    const response = await fetch(`${API_URL}/api/database/staged/${fileId}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async processDocuments(fileIds: string[]) {
    const response = await fetch(`${API_URL}/api/database/process`, {
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
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteDocument(documentId: string) {
    const response = await fetch(`${API_URL}/api/database/documents/${documentId}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async clearDatabase() {
    const response = await fetch(`${API_URL}/api/database/clear`, {
      method: 'POST',
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async reprocessDocumentChunks(documentId: string) {
    const response = await fetch(
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
    const response = await fetch(
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

  async getDocuments() {
    const response = await fetch(`${API_URL}/api/database/documents`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getHashtags() {
    const response = await fetch(`${API_URL}/api/database/hashtags`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocument(documentId: string): Promise<DocumentDetails> {
    const response = await fetch(`${API_URL}/api/documents/${documentId}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
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
    const response = await fetch(
      `${API_URL}/api/graph/clustered${queryString ? `?${queryString}` : ''}`
    )
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocumentChunks(documentId: string): Promise<DocumentChunk[]> {
    const response = await fetch(`${API_URL}/api/documents/${documentId}/chunks`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    const payload = await response.json()
    return Array.isArray(payload?.chunks) ? payload.chunks : []
  },

  async getDocumentText(documentId: string): Promise<DocumentTextPayload> {
    const response = await fetch(`${API_URL}/api/documents/${documentId}/text`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocumentChunkSimilarities(documentId: string) {
    const response = await fetch(`${API_URL}/api/documents/${documentId}/similarities`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocumentPreview(
    documentId: string
  ): Promise<{ preview_url: string } | Response> {
    const response = await fetch(`${API_URL}/api/documents/${documentId}/preview`, {
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
    const response = await fetch(`${API_URL}/api/documents/${documentId}/generate-summary`, {
      method: 'POST',
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  },

  async updateDocumentHashtags(documentId: string, hashtags: string[]) {
    const response = await fetch(`${API_URL}/api/documents/${documentId}/hashtags`, {
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
    const response = await fetch(`${API_URL}/api/health`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async checkHealth(): Promise<boolean> {
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout (more forgiving)
      
      const response = await fetch(`${API_URL}/api/health`, {
        method: 'GET',
        signal: controller.signal,
      })
      
      clearTimeout(timeoutId)
      return response.ok
    } catch (error) {
      return false
    }
  },

}
