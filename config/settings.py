"""
Configuration management for the GraphRAG pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="Allowed origins for CORS"
    )

    # LLM Provider Configuration
    llm_provider: str = Field(default="openai", description="LLM provider: openai, anthropic, mistral, gemini, ollama, lmstudio")

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI base URL")
    openai_model: Optional[str] = Field(
        default="gpt-4o-mini", description="OpenAI model name"
    )
    openai_proxy: Optional[str] = Field(default=None, description="OpenAI proxy URL")

    # Anthropic/Claude Configuration
    anthropic_api_key: Optional[str] = Field(
        default=None, 
        description="Anthropic API key",
        validation_alias="CLAUDE_API_KEY"  # Also accept CLAUDE_API_KEY
    )
    anthropic_base_url: Optional[str] = Field(default=None, description="Anthropic base URL (optional)")
    anthropic_model: str = Field(
        default="claude-sonnet-4-5-20250929", description="Anthropic model name"
    )

    # Mistral Configuration
    mistral_api_key: Optional[str] = Field(default=None, description="Mistral API key")
    mistral_base_url: Optional[str] = Field(default=None, description="Mistral base URL (optional)")
    mistral_model: str = Field(
        default="mistral-large-latest", description="Mistral model name"
    )

    # Gemini Configuration
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key",
        validation_alias="GOOGLE_API_KEY"
    )
    gemini_model: str = Field(
        default="gemini-3-flash-preview", description="Gemini model name (e.g., gemini-3-flash-preview, gemini-3-pro-preview)"
    )
    gemini_embedding_model: str = Field(
        default="models/text-embedding-004", description="Gemini embedding model"
    )

    # Ollama Configuration
    ollama_base_url: Optional[str] = Field(
        default="http://localhost:11434", description="Ollama base URL"
    )
    ollama_model: Optional[str] = Field(
        default="llama2", description="Ollama model name"
    )
    ollama_embedding_model: Optional[str] = Field(
        default="nomic-embed-text", description="Ollama embedding model (e.g., nomic-embed-text)"
    )

    # LM Studio Configuration (OpenAI-compatible local server)
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1", description="LM Studio base URL"
    )
    lmstudio_model: str = Field(
        default="local-model", description="LM Studio model name (as loaded in LM Studio)"
    )


    # Neo4j Configuration
    neo4j_uri: str = Field(
        default="bolt://localhost:7687", description="Neo4j connection URI"
    )
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="neo4j", description="Neo4j password")

    # Embedding Configuration
    embedding_model: str = Field(
        default="text-embedding-ada-002", description="Embedding model (e.g. text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)"
    )

    # Streaming Configuration
    enable_llm_streaming: bool = Field(
        default=True, description="Enable true LLM token streaming for responses"
    )

    # Keyword Search Configuration
    enable_chunk_fulltext: bool = Field(
        default=True, description="Enable BM25-style fulltext search on chunk content"
    )
    keyword_search_weight: float = Field(
        default=0.3, description="Weight for keyword search results in hybrid retrieval"
    )
    # Reciprocal Rank Fusion (RRF)
    enable_rrf: bool = Field(
        default=False,
        description="Enable Reciprocal Rank Fusion to combine ranked lists"
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant k controlling rank discount (higher = flatter)"
    )
    # Number of concurrent embedding requests
    embedding_concurrency: int = Field(default=3, description="Embedding concurrency")
    llm_concurrency: int = Field(default=2, description="LLM concurrency")
    # Rate limiting delays (in seconds)
    embedding_delay_min: float = Field(default=0.5, description="Minimum delay between embedding requests")
    embedding_delay_max: float = Field(default=1.0, description="Maximum delay between embedding requests")
    llm_delay_min: float = Field(default=0.5, description="Minimum delay between LLM requests")
    llm_delay_max: float = Field(default=1.0, description="Maximum delay between LLM requests")
    sync_entity_embeddings: bool = Field(
        default=False,
        description="Force synchronous entity embedding & persistence (set SYNC_ENTITY_EMBEDDINGS=1 for deterministic test runs)",
    )
    skip_entity_embeddings: bool = Field(
        default=False,
        description="Skip entity embedding generation entirely (set SKIP_ENTITY_EMBEDDINGS=1 to bypass during tests or bulk ingestion)",
    )

    # Document Processing Configuration
    chunk_size: int = Field(default=1200, description="Document chunk size")
    chunk_overlap: int = Field(default=150, description="Document chunk overlap")
    chunk_target_tokens: int = Field(
        default=800, description="Target tokens per chunk for token-aware chunkers"
    )
    chunk_min_tokens: int = Field(
        default=180, description="Minimum tokens per chunk for token-aware chunkers"
    )
    chunk_max_tokens: int = Field(
        default=1000, description="Maximum tokens per chunk for token-aware chunkers"
    )
    chunk_overlap_tokens: int = Field(
        default=100, description="Token overlap for token-aware chunkers"
    )
    chunk_tokenizer: str = Field(
        default="cl100k_base", description="Tokenizer name for token-aware chunking"
    )
    chunk_include_heading_path: bool = Field(
        default=True, description="Prefix HTML heading path into chunk text"
    )
    chunker_strategy_pdf: str = Field(
        default="docling_hybrid",
        description="Chunking strategy for PDF files: docling_hybrid|legacy",
    )
    chunker_strategy_html: str = Field(
        default="html_heading",
        description="Chunking strategy for HTML files: html_heading|docling_hybrid|legacy",
    )

    # Similarity Configuration
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    max_similarity_connections: int = Field(
        default=5, description="Max similarity connections"
    )

    # Entity Extraction Configuration
    enable_entity_extraction: bool = Field(
        default=True, description="Enable entity extraction"
    )

    # Gleaning Configuration (Phase 1: Multi-pass Entity Extraction)
    enable_gleaning: bool = Field(
        default=True,
        description="Enable multi-pass entity extraction with gleaning (enabled for quality)"
    )
    max_gleanings: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Number of additional extraction passes after initial pass (1=2 total passes for optimal quality)"
    )
    gleaning_by_doc_type: dict = Field(
        default={
            "admin": 2,
            "user": 1,
            "support": 0,
        },
        description="Document-type-specific gleaning configuration (overrides max_gleanings)"
    )

    # OCR Configuration
    enable_ocr: bool = Field(
        default=True, description="Enable OCR processing for scanned documents"
    )
    enable_quality_filtering: bool = Field(
        default=True, description="Enable chunk quality filtering (always on for production)"
    )
    ocr_quality_threshold: float = Field(
        default=0.6, description="Quality threshold for OCR processing"
    )

    # Content Filtering Configuration (Heuristic-based)
    enable_content_filtering: bool = Field(
        default=False, description="Enable heuristic content filtering before embedding"
    )
    content_filter_min_length: int = Field(
        default=50, description="Minimum chunk length in characters"
    )
    content_filter_max_length: int = Field(
        default=100000, description="Maximum chunk length in characters"
    )
    content_filter_unique_ratio: float = Field(
        default=0.3, description="Minimum ratio of unique words (0.0-1.0)"
    )
    content_filter_max_special_char_ratio: float = Field(
        default=0.5, description="Maximum ratio of special characters"
    )
    content_filter_min_alphanumeric_ratio: float = Field(
        default=0.3, description="Minimum ratio of alphanumeric characters"
    )
    content_filter_enable_conversation: bool = Field(
        default=True, description="Enable conversation thread quality filtering"
    )
    content_filter_enable_structured: bool = Field(
        default=True, description="Enable structured data quality filtering"
    )
    content_filter_enable_code: bool = Field(
        default=True, description="Enable code quality filtering"
    )
    content_filtering_llm_model: str = Field(
        default="gpt-4o-mini", description="LLM model for content filtering analysis"
    )

    # Temporal Graph Configuration
    enable_temporal_filtering: bool = Field(
        default=True, description="Enable temporal node creation and time-based retrieval"
    )
    default_time_decay_weight: float = Field(
        default=0.2, description="Default weight for time-decay scoring (0.0-1.0)"
    )
    temporal_window_days: int = Field(
        default=30, description="Default time window in days for temporal correlation queries"
    )

    # Multi-Stage Retrieval Configuration
    enable_two_stage_retrieval: bool = Field(
        default=True, description="Enable two-stage retrieval (BM25 pre-filter + vector search on candidates)"
    )
    two_stage_threshold_docs: int = Field(
        default=5000, description="Minimum corpus size to activate two-stage retrieval"
    )
    two_stage_multiplier: int = Field(
        default=10, description="Multiplier for BM25 candidate count (top_k * multiplier)"
    )

    # Sentence-Window Retrieval Configuration
    enable_sentence_window_retrieval: bool = Field(
        default=False, description="Enable sentence-level embedding and window-based retrieval"
    )
    sentence_window_size: int = Field(
        default=5, description="Number of sentences to include on each side of matched sentence"
    )
    sentence_min_length: int = Field(
        default=10, description="Minimum character length for a valid sentence"
    )

    # Fuzzy Matching Configuration
    enable_fuzzy_matching: bool = Field(
        default=True, description="Enable fuzzy matching for technical terms and typo correction"
    )
    max_fuzzy_distance: int = Field(
        default=2, description="Maximum edit distance for fuzzy matching (1-2 recommended)"
    )
    fuzzy_confidence_threshold: float = Field(
        default=0.5, description="Minimum confidence (0.0-1.0) to enable fuzzy matching for a query"
    )

    # Quality Monitoring Configuration
    enable_quality_monitoring: bool = Field(
        default=True, description="Enable continuous retrieval quality monitoring and alerting"
    )
    quality_monitor_window_size: int = Field(
        default=1000, description="Number of queries to track in the sliding window"
    )
    quality_alert_threshold: float = Field(
        default=0.7, description="Quality drop threshold (0.0-1.0) for triggering alerts"
    )

    # Query Expansion Configuration
    enable_query_expansion: bool = Field(
        default=True, description="Enable query expansion for abbreviations and synonyms to improve recall"
    )
    query_expansion_threshold: int = Field(
        default=3, description="Trigger expansion when initial retrieval returns fewer than this many results"
    )
    max_expansions: int = Field(
        default=5, description="Maximum number of expansion terms to generate per query"
    )
    expansion_penalty: float = Field(
        default=0.7, description="Score penalty (0.0-1.0) applied to chunks retrieved from expanded terms"
    )
    use_llm_expansion: bool = Field(
        default=False, description="Use LLM for synonym generation (slower but broader coverage)"
    )

    # Client-Side Vector Search for Static Entities Configuration
    enable_static_entity_matching: bool = Field(
        default=True, description="Enable client-side vector matching for static entities (categories) using precomputed embeddings"
    )
    static_matching_min_similarity: float = Field(
        default=0.6, description="Minimum cosine similarity (0.0-1.0) for static entity matches (lower = more permissive)"
    )

    # Layered Memory System Configuration
    enable_memory_system: bool = Field(
        default=False, description="Enable layered memory system for user context and conversation history"
    )
    memory_max_facts: int = Field(
        default=20, description="Maximum number of user facts to load per session"
    )
    memory_max_conversations: int = Field(
        default=5, description="Maximum number of past conversation summaries to load"
    )
    memory_min_fact_importance: float = Field(
        default=0.3, description="Minimum importance threshold (0.0-1.0) for facts to be loaded"
    )

    # Document Classification During Ingestion
    enable_document_classification: bool = Field(
        default=False,
        description="Enable LLM-assisted document category classification during ingestion"
    )
    classification_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for document classification"
    )
    classification_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence (0.0-1.0) to apply category metadata"
    )
    classification_default_category: str = Field(
        default="general",
        description="Fallback category when classification is disabled or low confidence"
    )

    # Retrieval Configuration
    min_retrieval_similarity: float = Field(
        default=0.1, description="Minimum similarity for chunk retrieval"
    )
    hybrid_chunk_weight: float = Field(
        default=0.6, description="Weight for chunk-based results"
    )
    hybrid_entity_weight: float = Field(
        default=0.4, description="Weight for entity-filtered results"
    )
    
    # Path Scoring Weights (Issue #9: Extracted from magic numbers)
    # Used in multi-hop reasoning to compute final path scores
    path_score_alpha: float = Field(
        default=0.6, description="Weight for path score from graph edges"
    )
    path_score_beta: float = Field(
        default=0.3, description="Weight for query similarity to path content"
    )
    path_score_gamma: float = Field(
        default=0.1, description="Weight for max chunk similarity in path"
    )
    
    enable_graph_expansion: bool = Field(
        default=True, description="Enable graph expansion"
    )
    default_context_restriction: bool = Field(
        default=True,
        description="Restrict retrieval to the provided context document set when available"
    )

    # Graph Expansion Limits
    max_expanded_chunks: int = Field(
        default=500, description="Maximum number of chunks after expansion"
    )
    max_entity_connections: int = Field(
        default=20, description="Maximum entity connections to follow"
    )
    max_chunk_connections: int = Field(
        default=10, description="Maximum chunk similarity connections to follow"
    )
    expansion_similarity_threshold: float = Field(
        default=0.1, description="Minimum similarity for expansion"
    )
    max_expansion_depth: int = Field(
        default=2, description="Maximum depth for graph traversal"
    )
    
    # Entity Healing Configuration (Issue #15)
    heal_description_max_chars: int = Field(
        default=500,
        description="Maximum characters for entity description in heal prompts (default raised from 300)"
    )
    
    # Centralized Retry Configuration (Issue #17)
    # Used by embeddings, entity_extraction, llm, and singletons
    retry_max_attempts: int = Field(
        default=5, description="Maximum retry attempts for API calls"
    )
    retry_base_delay: float = Field(
        default=2.0, description="Base delay in seconds for exponential backoff"
    )
    retry_max_delay: float = Field(
        default=120.0, description="Maximum delay between retries"
    )

    # Graph Clustering Configuration
    enable_clustering: bool = Field(
        default=True,
        description="High-level toggle for graph clustering and downstream summaries",
    )
    enable_graph_clustering: bool = Field(
        default=True, description="Enable Leiden clustering jobs"
    )
    clustering_relationship_types: List[str] = Field(
        default_factory=lambda: ["SIMILAR_TO", "RELATED_TO"],
        description="Relationship labels to include in clustering projections",
    )
    clustering_resolution: float = Field(
        default=1.0, description="Leiden resolution parameter"
    )
    clustering_min_edge_weight: float = Field(
        default=0.0, description="Minimum edge weight to keep in clustering"
    )
    clustering_level: int = Field(default=0, description="Hierarchy level to tag on nodes")
    default_edge_weight: float = Field(
        default=1.0,
        description="Fallback edge weight when no unit-level weight is present",
    )

    # Summarization Configuration
    summarization_batch_size: int = Field(
        default=20,
        description="Number of chunks to include when building document summaries",
    )

    # Multi-hop Reasoning Configuration
    multi_hop_max_hops: int = Field(
        default=2, description="Maximum number of hops for multi-hop reasoning"
    )
    multi_hop_beam_size: int = Field(
        default=8, description="Beam size for multi-hop path search"
    )
    multi_hop_min_edge_strength: float = Field(
        default=0.0, description="Minimum edge strength for multi-hop traversal"
    )
    hybrid_path_weight: float = Field(
        default=0.6, description="Weight for path-based results in hybrid mode"
    )

    # FlashRank reranker configuration (optional)
    flashrank_enabled: bool = Field(
        default=True,
        description="Enable FlashRank reranker for post-retrieval re-ranking (enabled for quality)",
    )
    # Control whether the FlashRank prewarm runs inside the web process.
    # In production it's preferable to run prewarm in a separate worker to avoid
    # impacting web request latency. Set `FLASHRANK_PREWARM_IN_PROCESS=0` to
    # run the prewarm externally via the provided worker script.
    flashrank_prewarm_in_process: bool = Field(
        default=True,
        description="Run FlashRank prewarm inside the web process (True) or in an external worker (False)",
    )
    flashrank_model_name: str = Field(
        default="ms-marco-TinyBERT-L-2-v2",
        description="Default FlashRank model name (tiny by default)",
    )
    flashrank_cache_dir: Optional[str] = Field(
        default=None,
        description="Optional cache directory for FlashRank / HF model downloads",
    )
    flashrank_max_candidates: int = Field(
        default=100,
        description="Maximum number of top candidates to send to the reranker",
    )
    flashrank_blend_weight: float = Field(
        default=0.0,
        description="Blend weight between hybrid_score and rerank_score (0.0 = use reranker ordering)",
    )
    flashrank_max_length: int = Field(
        default=128,
        description="Max token length passed to the reranker for (query+passage) pairs",
    )
    flashrank_batch_size: int = Field(
        default=32,
        description="Batch size for reranker calls (where applicable)",
    )

    # Caching Configuration
    cache_type: str = Field(
        default="disk",
        description="Cache backend type: 'disk' (persistent) or 'memory' (ephemeral)"
    )
    cache_dir: str = Field(
        default="data/cache",
        description="Directory for disk-based cache storage"
    )
    
    # Embedding Cache (7 days default)
    embedding_cache_size: int = Field(
        default=10000,
        description="Maximum entries in embedding cache (for memory backend)"
    )
    embedding_cache_ttl: int = Field(
        default=604800,  # 7 days
        description="TTL for embedding cache in seconds"
    )

    # Retrieval Cache (1 minute default)
    retrieval_cache_size: int = Field(
        default=1000,
        description="Maximum entries in retrieval cache"
    )
    retrieval_cache_ttl: int = Field(
        default=60,
        description="TTL for retrieval cache (seconds)"
    )
    
    # Response/Conversation Cache (2 hours default)
    response_cache_size: int = Field(
        default=2000,
        description="Maximum entries in response cache"
    )
    response_cache_ttl: int = Field(
        default=7200,  # 2 hours
        description="TTL for response cache in seconds"
    )
    
    # Entity Label Cache (5 minutes default)
    entity_label_cache_size: int = Field(
        default=5000,
        description="Maximum entries in entity label cache"
    )
    entity_label_cache_ttl: int = Field(
        default=300,
        description="TTL for entity label cache (seconds)"
    )
    # Provider-level LLM streaming (experimental)
    enable_llm_streaming: bool = Field(
        default=False,
        description="Enable provider-level LLM streaming (use with care)"
    )
    neo4j_max_connection_pool_size: int = Field(
        default=50,
        description="Maximum connections in Neo4j pool"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable caching system (set to false for rollback)"
    )

    # === Query Routing ===
    enable_query_routing: bool = Field(
        default=False,
        description="Enable automatic query routing to document categories",
    )
    query_routing_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence (0.0-1.0) to apply category filtering",
    )
    query_routing_strict: bool = Field(
        default=False,
        description="If True, ONLY search routed categories (no fallback to all docs)",
    )

    # === Semantic Routing Cache ===
    enable_routing_cache: bool = Field(
        default=True, description="Enable semantic caching for routing decisions"
    )
    routing_cache_similarity_threshold: float = Field(
        default=0.92, description="Minimum cosine similarity (0.0-1.0) for cache hit"
    )
    routing_cache_size: int = Field(default=1000)
    routing_cache_ttl: int = Field(default=3600)

    # === Fallback Validation ===
    fallback_enabled: bool = Field(
        default=True,
        description="Enable fallback strategies when routing returns insufficient results",
    )
    fallback_min_results: int = Field(
        default=3, description="Minimum chunks required before triggering fallback"
    )
    fallback_expand_to_related: bool = Field(default=True)
    fallback_all_documents: bool = Field(default=True)

    # === Smart Consolidation ===
    consolidation_strategy: str = Field(
        default="category_aware",
        description="Consolidation mode: 'category_aware', 'semantic_dedup', 'balanced', 'none'",
    )
    consolidation_max_context_tokens: int = Field(
        default=8000,
        description="Maximum tokens for LLM context (8K optimal per research)",
    )
    consolidation_ensure_representation: bool = Field(
        default=True,
        description="Ensure at least one chunk from each category appears in top-k",
    )
    consolidation_semantic_threshold: float = Field(
        default=0.95,
        description="Threshold for semantic deduplication (cosine similarity)",
    )

    # Category-Specific Prompts (M3.2)
    enable_category_prompts: bool = Field(
        default=True,
        description="Use category-specific prompt templates for generation",
    )
    enable_category_prompt_instructions: bool = Field(
        default=True,
        description="Include format instructions from category prompts",
    )
    category_prompts_file: str = Field(
        default="config/category_prompts.json",
        description="Path to category prompts configuration file",
    )

    # Structured KG Query Path (M3.3)
    enable_structured_kg: bool = Field(
        default=True,
        description="Enable structured knowledge graph query path (Text-to-Cypher)",
    )
    structured_kg_entity_threshold: float = Field(
        default=0.85,
        description="Confidence threshold for entity linking in structured queries",
    )
    structured_kg_max_corrections: int = Field(
        default=2,
        description="Maximum Cypher correction attempts on query errors",
    )
    structured_kg_timeout: int = Field(
        default=5000,
        description="Query execution timeout in milliseconds",
    )
    structured_kg_query_types: List[str] = Field(
        default=['aggregation', 'path', 'comparison', 'hierarchical', 'relationship'],
        description="Query types suitable for structured path",
    )

    # Adaptive routing with user feedback (M3.4)
    enable_adaptive_routing: bool = Field(
        default=True,
        description="Enable adaptive weight adjustment based on user feedback",
    )
    adaptive_learning_rate: float = Field(
        default=0.1,
        description="Learning rate for weight adjustments (0.0-1.0)",
    )
    adaptive_min_samples: int = Field(
        default=5,
        description="Minimum feedback samples before adjusting weights",
    )
    adaptive_weight_min: float = Field(
        default=0.1,
        description="Minimum allowed weight value",
    )
    adaptive_weight_max: float = Field(
        default=0.9,
        description="Maximum allowed weight value",
    )
    adaptive_decay_factor: float = Field(
        default=0.95,
        description="Exponential moving average decay factor",
    )

    # Document detail precomputed summaries
    enable_document_summaries: bool = Field(
        default=True,
        description="Enable precomputed per-document summary fields to speed up document detail pages",
    )
    document_summary_ttl: int = Field(
        default=300,
        description="TTL (seconds) for short-lived document summary caches",
    )
    document_detail_cache_ttl: int = Field(
        default=60,
        description="TTL (seconds) for cached document detail responses (frontend-level)",
    )
    document_summary_top_n_communities: int = Field(
        default=10,
        description="Number of top communities to store as a preview on the document node",
    )
    document_summary_top_n_similarities: int = Field(
        default=20,
        description="Number of top chunk similarities to store as a preview on the document node",
    )

    # Application Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    max_upload_size: int = Field(
        default=100 * 1024 * 1024, description="Max upload size"
    )  # 100MB

    # Quality Scoring Configuration
    enable_quality_scoring: bool = Field(
        default=True, description="Enable quality scoring for LLM answers (always on for production)"
    )
    quality_score_weights: dict = Field(
        default={
            "context_relevance": 0.30,
            "answer_completeness": 0.25,
            "factual_grounding": 0.25,
            "coherence": 0.10,
            "citation_quality": 0.10,
        },
        description="Weights for different quality score components",
    )

    # Database Operations Configuration
    enable_delete_operations: bool = Field(
        default=True, description="Enable ability to delete documents and clear database"
    )

    # Phase 2: NetworkX Intermediate Layer Configuration
    enable_phase2_networkx: bool = Field(
        default=True,
        description="Enable NetworkX intermediate graph layer for batch persistence (Phase 2) - reduces duplicates by 22%"
    )
    neo4j_unwind_batch_size: int = Field(
        default=500,
        description="Maximum entities per UNWIND batch query"
    )
    max_nodes_per_doc: int = Field(
        default=2000,
        description="Maximum entity nodes per document (memory limit)"
    )
    max_edges_per_doc: int = Field(
        default=5000,
        description="Maximum relationship edges per document (memory limit)"
    )
    importance_score_threshold: float = Field(
        default=0.3,
        description="Minimum importance score to include entity (early filtering)"
    )
    strength_threshold: float = Field(
        default=0.4,
        description="Minimum relationship strength to persist"
    )
    phase_version: str = Field(
        default="phase2_v1",
        description="Phase version tag for node metadata"
    )

    enable_stale_job_cleanup: bool = Field(
        default=True,
        description="Whether to mark documents in 'processing' status as 'failed' on startup. Disable in distributed setups."
    )

    # Orphan Cleanup Configuration
    enable_orphan_cleanup_on_startup: bool = Field(
        default=True,
        description="Enable automatic cleanup of orphaned chunks (not connected to any Document) on startup"
    )
    orphan_cleanup_grace_period_minutes: int = Field(
        default=5,
        description="Grace period in minutes - only delete orphans created more than this many minutes ago"
    )

    # Phase 3: Tuple-Delimited Output Format Configuration
    entity_extraction_format: str = Field(
        default="tuple_v1",
        description="Entity extraction output format (tuple_v1=tuple-delimited)"
    )
    entity_extraction_llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for entity extraction"
    )
    tuple_format_validation: bool = Field(
        default=True,
        description="Enable strict validation of tuple format output"
    )
    # Removed legacy fallback to pipe parser
    tuple_delimiter: str = Field(
        default="<|>",
        description="Delimiter used in tuple format (default: <|>)"
    )
    tuple_max_description_length: int = Field(
        default=500,
        description="Maximum description length in tuple format (truncate longer descriptions)"
    )

    # Phase 4: Description Summarization Configuration
    enable_description_summarization: bool = Field(
        default=True,
        description="Enable LLM-based description summarization (reduces verbosity by 50-70% per Microsoft GraphRAG)"
    )
    description_summarization_llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for description summarization"
    )
    summarization_min_mentions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Minimum entity/relationship mentions to trigger summarization (default: 3)"
    )
    summarization_min_length: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Minimum description length (characters) to trigger summarization (default: 200)"
    )
    summarization_batch_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of entities/relationships to summarize per LLM call (default: 5, balance efficiency vs token limits)"
    )
    summarization_cache_enabled: bool = Field(
        default=True,
        description="Enable summarization caching to avoid re-summarizing identical descriptions (default: True)"
    )

    # Document Conversion Provider
    document_conversion_provider: str = Field(
        default="auto",
        description="Document conversion engine: auto|native|marker|docling"
    )

    # Marker Conversion (PDF/Image) Configuration
    use_marker_for_pdf: bool = Field(
        default=True,
        description="Use Marker for PDF conversion to Markdown/JSON (enabled for quality)"
    )
    marker_output_format: str = Field(
        default="markdown",
        description="Marker output format: markdown|json|html|chunks"
    )
    marker_use_llm: bool = Field(
        default=True,
        description="Enable Marker LLM hybrid processors (tables/complex regions) - highest accuracy"
    )
    marker_paginate_output: bool = Field(
        default=True,
        description="Include pagination markers for provenance in Marker output"
    )
    marker_force_ocr: bool = Field(
        default=True,
        description="Force OCR on all PDF pages in Marker for inline math and highest quality"
    )
    marker_strip_existing_ocr: bool = Field(
        default=False,
        description="Strip embedded OCR text and re-OCR in Marker"
    )
    marker_pdftext_workers: int = Field(
        default=4,
        description="Number of pdftext workers used by Marker PdfProvider"
    )
    marker_llm_service: Optional[str] = Field(
        default=None,
        description="Optional Marker LLM service class path (e.g., marker.services.openai.OpenAIService)"
    )
    marker_llm_model: Optional[str] = Field(
        default="gpt-4o-mini",
        description="LLM model to use for Marker's LLM-enhanced processing (table extraction, complex layouts)"
    )
    marker_llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for Marker LLM service from environment (MARKER_LLM_API_KEY or OPENAI_API_KEY). Never store in config files."
    )

    # Issue #35: Configurable regex patterns for technical query detection
    technical_term_patterns: Dict[str, str] = Field(
        default_factory=lambda: {
            "snake_case": r'\b[a-z]+_[a-z_]+\b',
            "tech_id": r'\b[A-Z]{2,}-\d+\b',
            "config_key": r'\b[A-Z][A-Z_]{2,}\b|\b[a-z]+[A-Z][a-zA-Z]+\b',
            "error_code": r'\b(ERROR|WARN|INFO|DEBUG|FATAL|Exception|Error)\b|\b0x[0-9a-fA-F]+\b|\b[A-Z]+_[A-Z0-9]+\b',
            "file_ext": r'\b\w+\.(conf|json|yaml|yml|xml|py|js|ts|java|go|rs|c|cpp|h|hpp|sql|sh|bat|ps1|ini|env|properties)\b',
            "file_path": r'/\w+/[\w/\.]+'
        },
        description="Regex patterns for detecting technical terms in queries"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def load_rag_tuning_config() -> Dict[str, Any]:
    """
    Load RAG tuning configuration from JSON file.
    
    Returns a flat dictionary of all parameter values including section overrides.
    Falls back to empty dict if file doesn't exist or has errors.
    """
    config_path = Path(__file__).parent / "rag_tuning_config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        values = {"default_llm_model": config.get("default_llm_model")}
        
        for section in config.get("sections", []):
            # Add section-level LLM overrides
            if section.get("llm_override_enabled") and section.get("llm_override_value"):
                values[f"{section['key']}_llm_model"] = section["llm_override_value"]
            
            # Add all parameters
            for param in section.get("parameters", []):
                values[param["key"]] = param["value"]
        
        return values
    except FileNotFoundError:
        logger.warning(f"RAG tuning config not found at {config_path}, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in RAG tuning config: {e}, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Failed to load RAG tuning config: {e}, using defaults")
        return {}


# Issue S6: Consolidated chat tuning config loader
# Use this instead of duplicating logic in chat.py and other files
def load_chat_tuning_config() -> Dict[str, Any]:
    """
    Load chat tuning configuration from JSON file.
    
    This is the central utility for loading chat_tuning_config.json.
    DO NOT duplicate this logic in other files - import from here.
    
    Returns a flat dictionary of parameter values.
    Falls back to empty dict if file doesn't exist or has errors.
    """
    config_path = Path(__file__).parent / "chat_tuning_config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Flatten parameters for easy access
        values = {}
        for param in config.get("parameters", []):
            values[param["key"]] = param["value"]
        
        return values
    except FileNotFoundError:
        logger.debug(f"Chat tuning config not found at {config_path}")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load chat tuning config: {e}")
        return {}

def apply_rag_tuning_overrides(settings_instance: "Settings") -> None:
    """
    Apply RAG tuning configuration overrides to settings instance.
    
    This allows runtime configuration of ingestion parameters without
    restarting the server or changing environment variables.
    """
    rag_config = load_rag_tuning_config()
    if not rag_config:
        return
    
    # Apply direct parameter mappings
    param_mappings = {
        # === Content Filtering (Ingestion) ===
        "enable_content_filtering": "enable_content_filtering",
        "content_filter_min_length": "content_filter_min_length",
        "content_filter_unique_ratio": "content_filter_unique_ratio",
        "content_filter_max_special_char_ratio": "content_filter_max_special_char_ratio",
        "content_filter_min_alphanumeric_ratio": "content_filter_min_alphanumeric_ratio",
        "content_filter_enable_conversation": "content_filter_enable_conversation",
        "content_filter_enable_structured": "content_filter_enable_structured",
        "content_filter_enable_code": "content_filter_enable_code",
        # === Temporal Retrieval ===
        "enable_temporal_filtering": "enable_temporal_filtering",
        "default_time_decay_weight": "default_time_decay_weight",
        "temporal_window_days": "temporal_window_days",
        # === Multi-Stage Retrieval ===
        "enable_two_stage_retrieval": "enable_two_stage_retrieval",
        "two_stage_threshold_docs": "two_stage_threshold_docs",
        "two_stage_multiplier": "two_stage_multiplier",
        # === Fuzzy Matching ===
        "enable_fuzzy_matching": "enable_fuzzy_matching",
        "max_fuzzy_distance": "max_fuzzy_distance",
        "fuzzy_confidence_threshold": "fuzzy_confidence_threshold",
        # === Quality Monitoring ===
        "enable_quality_monitoring": "enable_quality_monitoring",
        "quality_monitor_window_size": "quality_monitor_window_size",
        "quality_alert_threshold": "quality_alert_threshold",
        # === Entity Extraction ===
        "enable_entity_extraction": "enable_entity_extraction",
        "enable_gleaning": "enable_gleaning",
        "max_gleanings": "max_gleanings",
        "entity_extraction_format": "entity_extraction_format",
        "tuple_format_validation": "tuple_format_validation",
        "tuple_max_description_length": "tuple_max_description_length",
        # === Description Enhancement ===
        "enable_description_summarization": "enable_description_summarization",
        "summarization_min_mentions": "summarization_min_mentions",
        "summarization_min_length": "summarization_min_length",
        "summarization_batch_size": "summarization_batch_size",
        "summarization_cache_enabled": "summarization_cache_enabled",
        # === Graph Persistence ===
        "enable_phase2_networkx": "enable_phase2_networkx",
        "neo4j_unwind_batch_size": "neo4j_unwind_batch_size",
        "max_nodes_per_doc": "max_nodes_per_doc",
        "max_edges_per_doc": "max_edges_per_doc",
        "importance_score_threshold": "importance_score_threshold",
        "strength_threshold": "strength_threshold",
        # === OCR & Image Processing ===
        "enable_ocr": "enable_ocr",
        "ocr_quality_threshold": "ocr_quality_threshold",
        # === Performance & Limits ===
        "llm_concurrency": "llm_concurrency",
        "embedding_concurrency": "embedding_concurrency",
        "llm_delay_min": "llm_delay_min",
        "llm_delay_max": "llm_delay_max",
        "embedding_delay_min": "embedding_delay_min",
        "embedding_delay_max": "embedding_delay_max",
        # === PDF Processing ===
        "document_conversion_provider": "document_conversion_provider",
        "use_marker_for_pdf": "use_marker_for_pdf",
        "marker_output_format": "marker_output_format",
        "marker_use_llm": "marker_use_llm",
        "marker_paginate_output": "marker_paginate_output",
        "marker_force_ocr": "marker_force_ocr",
        "marker_strip_existing_ocr": "marker_strip_existing_ocr",
        "marker_pdftext_workers": "marker_pdftext_workers",
        "marker_llm_model": "marker_llm_model",
        # === Retrieval Fusion & Ranking ===
        "enable_rrf": "enable_rrf",
        "rrf_k": "rrf_k",
        "enable_chunk_fulltext": "enable_chunk_fulltext",
        "keyword_search_weight": "keyword_search_weight",
        "hybrid_chunk_weight": "hybrid_chunk_weight",
        "hybrid_entity_weight": "hybrid_entity_weight",
        # === Chunking ===
        "chunk_target_tokens": "chunk_target_tokens",
        "chunk_min_tokens": "chunk_min_tokens",
        "chunk_max_tokens": "chunk_max_tokens",
        "chunk_overlap_tokens": "chunk_overlap_tokens",
        "chunk_tokenizer": "chunk_tokenizer",
        "chunk_include_heading_path": "chunk_include_heading_path",
        "chunker_strategy_pdf": "chunker_strategy_pdf",
        "chunker_strategy_html": "chunker_strategy_html",
        # === Sentence-Window Retrieval ===
        "enable_sentence_window_retrieval": "enable_sentence_window_retrieval",
        "sentence_window_size": "sentence_window_size",
        "sentence_min_length": "sentence_min_length",
        "enable_stale_job_cleanup": "enable_stale_job_cleanup",
        "flashrank_enabled": "flashrank_enabled",
        "flashrank_model_name": "flashrank_model_name",
        "flashrank_blend_weight": "flashrank_blend_weight",
        "flashrank_max_candidates": "flashrank_max_candidates",
        "flashrank_batch_size": "flashrank_batch_size",
        # === Query Expansion ===
        "enable_query_expansion": "enable_query_expansion",
        "query_expansion_threshold": "query_expansion_threshold",
        "max_expansions": "max_expansions",
        "expansion_penalty": "expansion_penalty",
        "use_llm_expansion": "use_llm_expansion",
        # === Client-Side Vector Search ===
        "enable_static_entity_matching": "enable_static_entity_matching",
        "static_matching_min_similarity": "static_matching_min_similarity",
        # === Layered Memory System ===
        "enable_memory_system": "enable_memory_system",
        "memory_max_facts": "memory_max_facts",
        "memory_max_conversations": "memory_max_conversations",
        "memory_min_fact_importance": "memory_min_fact_importance",
        # === LLM Overrides (Issue S2) ===
        "content_filtering_llm_model": "content_filtering_llm_model",
        "entity_extraction_llm_model": "entity_extraction_llm_model",
        "description_enhancement_llm_model": "description_summarization_llm_model",
        "pdf_processing_llm_model": "marker_llm_model",
        # === Technical Query Detection (Issue #35) ===
        "technical_term_patterns": "technical_term_patterns",
    }
    
    for config_key, settings_attr in param_mappings.items():
        if config_key in rag_config:
            try:
                setattr(settings_instance, settings_attr, rag_config[config_key])
            except Exception as e:
                logger.warning(f"Failed to apply RAG config override for {settings_attr}: {e}")

    provider_override = rag_config.get("document_conversion_provider")
    if provider_override:
        provider = str(provider_override).strip().lower()
        if provider == "marker":
            settings_instance.use_marker_for_pdf = True
        elif provider in {"docling", "native"}:
            settings_instance.use_marker_for_pdf = False

    # Issue S7: Propagate default_llm_model to provider-specific settings
    # This ensures the UI model selection overrides all provider defaults
    default_model = rag_config.get("default_llm_model")
    if default_model:
        provider_model_fields = [
            "openai_model", "anthropic_model", "gemini_model", 
            "ollama_model", "azure_openai_model"
        ]
        for field in provider_model_fields:
            if hasattr(settings_instance, field):
                try:
                    setattr(settings_instance, field, default_model)
                except Exception as e:
                    logger.debug(f"Could not set {field} to {default_model}: {e}")
        logger.info(f"Applied default LLM model override: {default_model}")
    
    logger.info("Applied RAG tuning configuration overrides")


# Global settings instance - will read from environment or use defaults
settings = Settings()

# If Docker Compose provided NEO4J_AUTH (format: user/password), prefer it
# when explicit NEO4J_USERNAME/NEO4J_PASSWORD were not set in the environment.
try:
    import os

    if os.environ.get("NEO4J_AUTH"):
        # Only override when explicit username/password env vars are not provided
        if (not os.environ.get("NEO4J_USERNAME")) and (not os.environ.get("NEO4J_PASSWORD")):
            auth = os.environ.get("NEO4J_AUTH", "")
            if "/" in auth:
                u, p = auth.split("/", 1)
                # Apply to settings instance so downstream modules pick them up
                try:
                    settings.neo4j_username = u
                    settings.neo4j_password = p
                    logger.info("Applied NEO4J_AUTH to settings (username from NEO4J_AUTH)")
                except Exception:
                    pass
except Exception:
    pass

# Normalize and validate LLM provider selection to avoid unexpected defaults.
# Priority: explicit `LLM_PROVIDER` env -> use if valid; otherwise prefer OpenAI when an API key is present.
try:
    _prov = (settings.llm_provider or "").lower()
except Exception:
    _prov = ""

if _prov not in ("openai", "ollama"):
    # If an OpenAI API key is configured, prefer OpenAI
    if getattr(settings, "openai_api_key", None):
        settings.llm_provider = "openai"
    else:
        # Fall back to ollama only if no OpenAI key is available
        settings.llm_provider = "ollama"
else:
    settings.llm_provider = _prov

# Apply RAG tuning configuration overrides (must be after settings initialization)
apply_rag_tuning_overrides(settings)

# Emit a debug-friendly representation for other modules
try:
    logger.info(f"Resolved LLM provider: {settings.llm_provider}")
    if getattr(settings, "sync_entity_embeddings", False):
        logger.info("SYNC_ENTITY_EMBEDDINGS enabled: using synchronous entity persistence path")
    if getattr(settings, "skip_entity_embeddings", False):
        logger.info("SKIP_ENTITY_EMBEDDINGS enabled: entity embeddings will be omitted")
    
    # Phase 3: Validate and log entity extraction format
    format_type = getattr(settings, "entity_extraction_format", "tuple_v1")
    if format_type != "tuple_v1":
        logger.warning(f"Invalid entity_extraction_format '{format_type}', using 'tuple_v1'")
        settings.entity_extraction_format = "tuple_v1"
    else:
        logger.info(f"Entity extraction format: {settings.entity_extraction_format}")

    # Validate document conversion provider
    provider = getattr(settings, "document_conversion_provider", "auto")
    if provider not in {"auto", "native", "marker", "docling"}:
        logger.warning(f"Invalid document_conversion_provider '{provider}', defaulting to 'auto'")
        settings.document_conversion_provider = "auto"
        provider = "auto"
    if provider == "docling":
        logger.info("Docling conversion enabled")
    elif provider == "native":
        logger.info("Native conversion enabled (built-in loaders)")

    # Validate Marker output format
    marker_fmt = getattr(settings, "marker_output_format", "markdown")
    if marker_fmt not in ["markdown", "json", "html", "chunks"]:
        logger.warning(f"Invalid marker_output_format '{marker_fmt}', defaulting to 'markdown'")
        settings.marker_output_format = "markdown"
    if provider in {"marker", "auto"} and getattr(settings, "use_marker_for_pdf", False):
        logger.info("Marker PDF conversion enabled")
except Exception:
    pass

# Environment variables documentation (summary):
#   SYNC_ENTITY_EMBEDDINGS=1 -> run entity extraction & embedding fully synchronously (deterministic tests)
#   SKIP_ENTITY_EMBEDDINGS=1 -> bypass embedding generation for entities (faster ingestion, empty embeddings stored)
