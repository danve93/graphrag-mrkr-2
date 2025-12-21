import type { ProcessProgress } from './upload'

export interface StageUpdate {
  name: string
  duration_ms?: number
  timestamp?: number
  metadata?: {
    chunks_retrieved?: number
    context_items?: number
    response_length?: number
    routing_category_id?: string
    routing_confidence?: number
    routing_categories?: string[]
    document_count?: number
    [key: string]: any
  }
}

export interface RoutingInfo {
  categories: string[]
  confidence: number
  category_id?: string | null
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
  sources?: Source[]
  quality_score?: QualityScore
  follow_up_questions?: string[]
  isStreaming?: boolean
  context_documents?: string[]
  context_document_labels?: string[]
  context_hashtags?: string[]
  stages?: StageUpdate[]
  total_duration_ms?: number
  message_id?: string
  session_id?: string
  routing_info?: RoutingInfo
  input_tokens?: number
  output_tokens?: number
}

export interface Source {
  chunk_id?: string
  entity_id?: string
  entity_name?: string
  content: string
  similarity: number
  relevance_score?: number
  document_name: string
  original_filename?: string
  document_id?: string
  filename: string
  chunk_index?: number
  contained_entities?: string[]
  metadata?: Record<string, any>
  citation?: string
}

export interface QualityScore {
  total: number
  breakdown: {
    context_relevance: number
    answer_completeness: number
    factual_grounding: number
    coherence: number
    citation_quality: number
  }
  confidence: 'low' | 'medium' | 'high'
}

export interface ChatSession {
  session_id: string
  created_at: string
  updated_at: string
  message_count: number
  preview?: string
}

export interface DatabaseStats {
  total_documents: number
  total_chunks: number
  total_entities: number
  total_relationships: number
  documents: DocumentSummary[]
  processing?: ProcessingSummary
}

export interface FolderSummary {
  id: string
  name: string
  created_at?: number
  document_count: number
}

export interface DocumentSummary {
  document_id: string
  title?: string
  filename: string
  original_filename?: string
  created_at: string | number
  chunk_count: number
  processing_status?: string
  processing_stage?: string
  processing_progress?: number
  queue_position?: number | null
  hashtags?: string[]
  document_type?: string
  folder_id?: string | null
  folder_name?: string | null
  folder_order?: number | null
}

export interface ProcessingSummary {
  is_processing: boolean
  current_file_id?: string | null
  current_document_id?: string | null
  current_filename?: string | null
  current_stage?: string | null
  progress_percentage?: number | null
  queue_length: number
  pending_documents: ProcessProgress[]
}

export interface UploaderInfo {
  id?: string
  name?: string
}

export interface DocumentChunk {
  id: string | number
  text: string
  index?: number
  offset?: number
  score?: number | null
}

export interface DocumentEntity {
  type: string
  text: string
  count?: number
  positions?: Array<number>
}

export interface RelatedDocument {
  id: string
  title?: string
  link?: string
}

export interface DocumentDetails {
  id: string
  title?: string
  file_name?: string
  original_filename?: string
  mime_type?: string
  preview_url?: string
  uploaded_at?: string
  uploader?: UploaderInfo | null
  summary?: string | null
  document_type?: string | null
  hashtags?: string[]
  folder_id?: string | null
  folder_name?: string | null
  folder_order?: number | null
  chunks: DocumentChunk[]
  entities: DocumentEntity[]
  quality_scores?: Record<string, any> | null
  related_documents?: RelatedDocument[]
  metadata?: Record<string, any>
}

export interface UploadResponse {
  filename: string
  status: string
  chunks_created: number
  document_id?: string
  error?: string
}

export interface DocumentTextPayload {
  document_id: string
  text: string
}

export interface ChatRequest {
  message: string
  session_id?: string
  retrieval_mode?: 'hybrid' | 'simple' | 'graph_enhanced'
  top_k?: number
  temperature?: number
  use_multi_hop?: boolean
  stream?: boolean
  context_documents?: string[]
  chunk_weight?: number
  entity_weight?: number
  path_weight?: number
  max_hops?: number
  beam_size?: number
  graph_expansion_depth?: number
  restrict_to_context?: boolean
}
