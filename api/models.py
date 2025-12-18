"""
Pydantic models for API requests and responses.
"""

from typing import Any, Dict, List, Optional
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from config.settings import settings


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[str] = None
    message_id: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    quality_score: Optional[Dict[str, Any]] = None
    follow_up_questions: Optional[List[str]] = None
    context_documents: Optional[List[str]] = None
    context_document_labels: Optional[List[str]] = None
    context_hashtags: Optional[List[str]] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for chat history")
    retrieval_mode: str = Field("hybrid", description="Retrieval mode")
    top_k: int = Field(5, description="Number of chunks to retrieve")
    temperature: float = Field(0.7, description="LLM temperature")
    use_multi_hop: bool = Field(False, description="Enable multi-hop reasoning")
    chunk_weight: float = Field(
        default=settings.hybrid_chunk_weight,
        description="Weight to allocate to vector chunks",
    )
    entity_weight: Optional[float] = Field(
        default=settings.hybrid_entity_weight,
        description="Weight to allocate to entity-filtered chunks",
    )
    path_weight: Optional[float] = Field(
        default=settings.hybrid_path_weight,
        description="Weight to allocate to multi-hop path chunks",
    )
    max_hops: Optional[int] = Field(
        default=settings.multi_hop_max_hops,
        description="Depth limit for multi-hop traversal",
    )
    beam_size: Optional[int] = Field(
        default=settings.multi_hop_beam_size,
        description="Beam width for multi-hop traversal",
    )
    graph_expansion_depth: Optional[int] = Field(
        default=settings.max_expansion_depth,
        description="Depth limit for graph expansion",
    )
    restrict_to_context: bool = Field(
        default=settings.default_context_restriction,
        description="Restrict retrieval to provided context documents",
    )
    llm_model: Optional[str] = Field(
        default=getattr(settings, 'openai_model', None),
        description="Override LLM model to use for generation",
    )
    embedding_model: Optional[str] = Field(
        default=getattr(settings, 'embedding_model', None),
        description="Override embedding model to use for query embedding",
    )
    stream: bool = Field(True, description="Enable streaming response")
    context_documents: List[str] = Field(
        default_factory=list,
        description="List of document IDs to restrict retrieval context",
    )
    context_document_labels: List[str] = Field(
        default_factory=list,
        description="List of document labels/names for UI display",
    )
    category_filter: Optional[List[str]] = Field(
        default=None,
        description="Override routing to specific categories (bypasses automatic routing)",
    )
    context_hashtags: List[str] = Field(
        default_factory=list,
        description="List of hashtags used to filter documents",
    )

    # Evaluation-specific feature flag overrides
    # These are optional per-request overrides for A/B testing variants
    # When None, global settings are used (normal production behavior)
    eval_enable_query_routing: Optional[bool] = Field(
        None,
        description="[Evaluation] Override enable_query_routing setting for this request only"
    )
    eval_enable_structured_kg: Optional[bool] = Field(
        None,
        description="[Evaluation] Override enable_structured_kg setting for this request only"
    )
    eval_enable_rrf: Optional[bool] = Field(
        None,
        description="[Evaluation] Override enable_rrf (Reciprocal Rank Fusion) for this request only"
    )
    eval_enable_routing_cache: Optional[bool] = Field(
        None,
        description="[Evaluation] Override enable_routing_cache setting for this request only"
    )
    eval_flashrank_enabled: Optional[bool] = Field(
        None,
        description="[Evaluation] Override flashrank_enabled (reranking) for this request only"
    )
    eval_enable_graph_clustering: Optional[bool] = Field(
        None,
        description="[Evaluation] Override enable_graph_clustering for this request only"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: str = Field(..., description="Assistant response")
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    quality_score: Optional[Dict[str, Any]] = None
    follow_up_questions: List[str] = Field(default_factory=list)
    session_id: str = Field(..., description="Session ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    context_documents: List[str] = Field(
        default_factory=list,
        description="Document IDs used to constrain retrieval",
    )
    stages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pipeline stages with timing metadata",
    )
    total_duration_ms: Optional[int] = Field(
        None,
        description="Total pipeline duration in milliseconds",
    )


class FollowUpRequest(BaseModel):
    """Request model for follow-up question generation."""

    query: str = Field(..., description="User query")
    response: str = Field(..., description="Assistant response")
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)


class FollowUpResponse(BaseModel):
    """Response model for follow-up questions."""

    questions: List[str] = Field(..., description="Generated follow-up questions")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    filename: str
    status: str
    chunks_created: int
    document_id: Optional[str] = None
    processing_status: Optional[str] = None
    processing_stage: Optional[str] = None
    error: Optional[str] = None


class StagedDocument(BaseModel):
    """Model for a staged document waiting to be processed."""

    file_id: str
    filename: str
    file_size: int
    file_path: str
    timestamp: float
    document_id: Optional[str] = None
    mode: Literal["full", "chunks_only", "entities_only"] = Field(
        "full", description="Processing mode"
    )


class StageDocumentResponse(BaseModel):
    """Response model for staging a document."""

    file_id: str
    filename: str
    document_id: Optional[str] = None
    status: str


class ChunkSimilarity(BaseModel):
    """Model for a chunk similarity relationship."""

    chunk1_id: str
    chunk2_id: str
    score: float


class PaginatedSimilaritiesResponse(BaseModel):
    """Response model for paginated chunk similarities."""

    document_id: str
    total: int
    estimated: bool = Field(default=False, description="Whether total count is estimated (fast) or exact (slow)")
    limit: int
    offset: int
    has_more: bool
    similarities: List[ChunkSimilarity]


class ChunkDetails(BaseModel):
    """Model for detailed chunk information."""

    id: str
    content: str
    index: int
    offset: int
    document_id: str
    document_name: Optional[str] = None


class DocumentStats(BaseModel):
    """Model for document statistics."""

    chunks: int
    entities: int
    communities: int
    similarities: int


class DocumentSummary(BaseModel):
    """Model for lightweight document summary."""

    id: str
    filename: str
    original_filename: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    link: Optional[str] = None
    uploader: Optional[str] = None
    stats: DocumentStats


class DocumentEntity(BaseModel):
    """Model for a document entity."""

    type: str
    text: str
    community_id: Optional[int] = None
    level: Optional[int] = None
    count: int
    positions: List[int]


class PaginatedEntitiesResponse(BaseModel):
    """Response model for paginated entities."""

    document_id: str
    total: int
    limit: int
    offset: int
    has_more: bool
    entities: List[DocumentEntity]
    processing_status: Optional[str] = None
    processing_stage: Optional[str] = None
    error: Optional[str] = None


class ProcessProgress(BaseModel):
    """Model for document processing progress."""

    file_id: str
    document_id: Optional[str] = None
    filename: str
    status: str  # 'queued', 'processing', 'completed', 'error'
    stage: Optional[str] = Field(None, description="Current processing stage")
    mode: Optional[str] = Field(None, description="Processing mode")
    queue_position: Optional[int] = Field(None, description="Position in processing queue")
    chunks_processed: int
    total_chunks: int
    chunk_progress: Optional[float] = Field(None, description="Chunk processing progress 0-1")
    entity_progress: Optional[float] = Field(None, description="Entity processing progress 0-1")
    progress_percentage: float
    entity_state: Optional[str] = Field(None, description="Entity extraction state")
    error: Optional[str] = None
    cancelled: bool = Field(False, description="Whether this job has been cancelled")
    message: Optional[str] = Field(None, description="Detailed status message")


class ProcessDocumentsRequest(BaseModel):
    """Request model for processing staged documents."""

    file_ids: List[str] = Field(..., description="List of file IDs to process")


class ProcessingSummary(BaseModel):
    """Global processing summary for UI indicators."""

    is_processing: bool = Field(False, description="Whether any processing job is active")
    current_file_id: Optional[str] = Field(None, description="Currently processed staged file id")
    current_document_id: Optional[str] = Field(None, description="Currently processed document id")
    current_filename: Optional[str] = Field(None, description="Human friendly filename")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    progress_percentage: Optional[float] = Field(None, description="Overall progress percentage")
    queue_length: int = Field(0, description="Number of pending jobs including current")
    pending_documents: List[ProcessProgress] = Field(default_factory=list)


class DatabaseStats(BaseModel):
    """Database statistics model."""

    total_documents: int
    total_chunks: int
    total_entities: int
    total_relationships: int
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    processing: Optional[ProcessingSummary] = None
    enable_delete_operations: bool = Field(True, description="Whether delete operations are enabled")


class DocumentChunk(BaseModel):
    """Chunk information associated with a document."""

    id: str | int
    text: str
    index: int | None = None
    offset: int | None = None
    score: float | None = None


class DocumentEntity(BaseModel):
    """Entity extracted from a document."""

    type: str
    text: str
    community_id: int | None = None
    level: int | None = None
    count: int | None = None
    positions: List[int] | None = None


class RelatedDocument(BaseModel):
    """Related document link."""

    id: str
    title: str | None = None
    link: str | None = None


class UploaderInfo(BaseModel):
    """Information about the document uploader."""

    id: str | None = None
    name: str | None = None


class DocumentMetadataResponse(BaseModel):
    """Response model for document metadata."""

    id: str
    title: str | None = None
    file_name: str | None = None
    original_filename: str | None = None
    mime_type: str | None = None
    preview_url: str | None = None
    uploaded_at: str | None = None
    uploader: UploaderInfo | None = None
    summary: str | None = None
    document_type: str | None = None
    hashtags: List[str] = Field(default_factory=list)
    chunks: List[DocumentChunk] = Field(default_factory=list)
    entities: List[DocumentEntity] = Field(default_factory=list)
    quality_scores: Dict[str, Any] | None = None
    related_documents: List[RelatedDocument] | None = None
    metadata: Dict[str, Any] | None = None


class ConversationSession(BaseModel):
    """Conversation session model."""

    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    preview: Optional[str] = None
    deleted_at: Optional[str] = None

class UpdateHashtagsRequest(BaseModel):
    """Request model for updating document hashtags."""

    hashtags: List[str] = Field(
        ...,
        description="List of hashtags to set for the document",
        max_length=20,  # Issue #10: Maximum 20 hashtags per document
    )
    
    @field_validator('hashtags')
    @classmethod
    def validate_hashtags(cls, v: List[str]) -> List[str]:
        """Validate hashtag format and length (Issue #10)."""
        import re
        validated = []
        for tag in v:
            # Strip leading # if present
            tag = tag.lstrip('#').strip()
            if not tag:
                continue
            # Max length 50 characters
            if len(tag) > 50:
                raise ValueError(f"Hashtag '{tag[:20]}...' exceeds maximum length of 50 characters")
            # Only alphanumeric, hyphens, underscores
            if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                raise ValueError(f"Hashtag '{tag}' contains invalid characters. Use only letters, numbers, hyphens, underscores.")
            validated.append(tag)
        return validated


class UpdateMetadataRequest(BaseModel):
    """Request model for updating document metadata."""

    metadata: Dict[str, Any] = Field(
        ...,
        description="Dictionary of metadata key-value pairs to set"
    )


class ConversationHistory(BaseModel):
    """Conversation history model."""

    session_id: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str
    deleted_at: Optional[str] = None


class SessionCreateRequest(BaseModel):
    """Request model for creating a session explicitly."""

    session_id: Optional[str] = Field(
        default=None, description="Optional session id to use for the new conversation"
    )
    title: Optional[str] = Field(
        default=None, description="Optional human friendly title for the session"
    )


class MessageCreateRequest(BaseModel):
    """Request model for adding a message to a session."""

    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Raw message content")
    sources: Optional[List[Dict[str, Any]]] = None
    quality_score: Optional[Dict[str, Any]] = None
    follow_up_questions: Optional[List[str]] = None
    context_documents: Optional[List[str]] = None
    context_document_labels: Optional[List[str]] = None
    context_hashtags: Optional[List[str]] = None


class MessageSearchResult(BaseModel):
    """Result model for message search operations."""

    session_id: str
    message_id: str
    role: str
    content: str
    timestamp: Optional[str] = None
    quality_score: Optional[Dict[str, Any]] = None
    context_documents: List[str] = Field(default_factory=list)


class MessageSearchResponse(BaseModel):
    """Envelope for message search responses."""

    query: str
    results: List[MessageSearchResult]


class GraphDocumentRef(BaseModel):
    """Reference to a document connected to an entity or text unit."""

    document_id: Optional[str] = None
    document_name: Optional[str] = None


class GraphTextUnit(BaseModel):
    """TextUnit or chunk provenance for relationships."""

    id: str
    document_id: Optional[str] = None
    document_name: Optional[str] = None


class GraphNode(BaseModel):
    """Graph node enriched with clustering metadata."""

    id: str
    label: str
    type: Optional[str] = None
    community_id: Optional[int] = None
    level: Optional[int] = None
    degree: int = 0
    documents: List[GraphDocumentRef] = Field(default_factory=list)


class GraphEdge(BaseModel):
    """Graph edge with strength and provenance."""

    source: str
    target: str
    type: Optional[str] = None
    weight: float = Field(0.5, description="Relationship strength or weight")
    description: Optional[str] = None
    text_units: List[GraphTextUnit] = Field(default_factory=list)


class GraphCommunity(BaseModel):
    """Community metadata."""

    community_id: int
    level: Optional[int] = None


class GraphResponse(BaseModel):
    """Clustered graph payload with nodes, edges, and filter metadata."""

    nodes: List[GraphNode]
    edges: List[GraphEdge]
    communities: List[GraphCommunity]
    node_types: List[str]


class ReindexResult(BaseModel):
    """Response model for the reindex operation."""

    status: Literal["success", "partial", "failed"] = Field(
        ..., description="Outcome status of the reindex operation"
    )
    message: str = Field(..., description="Human readable summary of the operation")
    documents_processed: int = Field(0, description="Number of documents processed during extraction")
    entities_cleared: int = Field(0, description="Number of entities removed during the clearing step")
    extraction_result: Optional[Dict[str, Any]] = Field(
        None, description="Optional raw extraction result payload"
    )
    clustering_result: Optional[Dict[str, Any]] = Field(
        None, description="Optional clustering result payload"
    )


class ReindexJobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[ReindexResult] = None


class ReindexJobResponse(BaseModel):
    job_id: str
    status_url: str
    status: str
