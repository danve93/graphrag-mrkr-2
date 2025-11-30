"""
Document retrieval node for LangGraph RAG pipeline.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from config.settings import settings
from rag.retriever import DocumentRetriever, RetrievalMode
from core.singletons import get_blocking_executor, SHUTTING_DOWN

logger = logging.getLogger(__name__)

# Initialize enhanced retriever
document_retriever = DocumentRetriever()


async def retrieve_documents_async(
    query: str,
    query_analysis: Dict[str, Any],
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    chunk_weight: float = 0.5,
    entity_weight: Optional[float] = None,
    path_weight: Optional[float] = None,
    graph_expansion: bool = True,
    use_multi_hop: bool = False,
    max_hops: Optional[int] = None,
    beam_size: Optional[int] = None,
    restrict_to_context: bool = True,
    expansion_depth: Optional[int] = None,
    context_documents: Optional[List[str]] = None,
    embedding_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents based on query and analysis using enhanced retriever.

    Args:
        query: User query string (will use contextualized_query if available in analysis)
        query_analysis: Query analysis results
        retrieval_mode: Retrieval strategy ("chunk_only", "entity_only", "hybrid", "auto")
        top_k: Number of chunks to retrieve
        chunk_weight: Weight for chunk-based results in hybrid mode
        entity_weight: Weight for entity-filtered results in hybrid mode
        path_weight: Weight for path-based results in hybrid mode
        graph_expansion: Whether to use graph expansion
        use_multi_hop: Whether to use multi-hop reasoning
        max_hops: Depth limit for graph traversal
        beam_size: Beam width for multi-hop search
        restrict_to_context: Whether to enforce context_documents filtering
        expansion_depth: Optional override for graph expansion depth
        context_documents: Optional list of document IDs to restrict retrieval scope

    Returns:
        List of relevant document chunks
    """
    try:
        allowed_docs = context_documents or []

        # Use contextualized query if this is a follow-up question
        search_query = query_analysis.get("contextualized_query", query)
        if search_query != query:
            logger.info(f"Using contextualized query for retrieval: {search_query}")

        # Determine retrieval strategy based on query analysis and mode
        complexity = query_analysis.get("complexity", "simple")
        requires_multiple = query_analysis.get("requires_multiple_sources", False)
        query_type = query_analysis.get("query_type", "factual")

        # Adjust top_k based on query complexity
        adjusted_top_k = top_k
        adjustment_reason = None
        if complexity == "complex" or requires_multiple:
            adjusted_top_k = min(top_k + 3, 10)
            adjustment_reason = "complexity or multiple sources required"
        elif query_type == "comparative":
            adjusted_top_k = min(top_k + 5, 12)
            adjustment_reason = "comparative query"

        # Log why top_k was adjusted (if it changed)
        if adjusted_top_k != top_k:
            logger.info(
                "Adjusted top_k from %d to %d (%s) — query_type=%s, complexity=%s, requires_multiple=%s",
                top_k,
                adjusted_top_k,
                adjustment_reason or "adjusted",
                query_type,
                complexity,
                requires_multiple,
            )

        # Map retrieval modes to enhanced retriever modes
        mode_mapping = {
            "simple": RetrievalMode.CHUNK_ONLY,
            "chunk_only": RetrievalMode.CHUNK_ONLY,
            "entity_only": RetrievalMode.ENTITY_ONLY,
            "hybrid": RetrievalMode.HYBRID,
            "graph_enhanced": RetrievalMode.HYBRID,  # Legacy compatibility
            "auto": (
                RetrievalMode.HYBRID
                if settings.enable_entity_extraction
                else RetrievalMode.CHUNK_ONLY
            ),
        }

        # Get the appropriate retrieval mode
        enhanced_mode = mode_mapping.get(retrieval_mode, RetrievalMode.HYBRID)

        allowed_ids = allowed_docs if allowed_docs else None

        # Use enhanced retriever. Prefer graph expansion when configured
        if (complexity == "complex" or query_type == "comparative") and graph_expansion:
            chunks = await document_retriever.retrieve_with_graph_expansion(
                query=search_query,
                mode=enhanced_mode,
                top_k=adjusted_top_k,
                use_multi_hop=use_multi_hop,
                chunk_weight=chunk_weight,
                entity_weight=entity_weight,
                path_weight=path_weight,
                max_hops=max_hops,
                beam_size=beam_size,
                restrict_to_context=restrict_to_context,
                expand_depth=expansion_depth or settings.max_expansion_depth,
                allowed_document_ids=allowed_ids,
                embedding_model=embedding_model,
            )
        else:
            # Pass chunk_weight and multi_hop through to hybrid retriever if present
            chunks = await document_retriever.retrieve(
                query=search_query,
                mode=enhanced_mode,
                top_k=adjusted_top_k,
                chunk_weight=chunk_weight,
                use_multi_hop=use_multi_hop,
                entity_weight=entity_weight,
                path_weight=path_weight,
                max_hops=max_hops,
                beam_size=beam_size,
                restrict_to_context=restrict_to_context,
                allowed_document_ids=allowed_ids,
                embedding_model=embedding_model,
            )

        logger.info(
            "Retrieved %d chunks using %s mode (enhanced_mode: %s) with top_k=%d, multi_hop=%s, restricted_docs=%s",
            len(chunks),
            retrieval_mode,
            enhanced_mode.value,
            adjusted_top_k,
            use_multi_hop,
            bool(allowed_docs),
        )
        return chunks

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


def retrieve_documents(
    query: str,
    query_analysis: Dict[str, Any],
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    chunk_weight: float = 0.5,
    entity_weight: Optional[float] = None,
    path_weight: Optional[float] = None,
    graph_expansion: bool = True,
    use_multi_hop: bool = False,
    max_hops: Optional[int] = None,
    beam_size: Optional[int] = None,
    restrict_to_context: bool = True,
    expansion_depth: Optional[int] = None,
    context_documents: Optional[List[str]] = None,
    embedding_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for document retrieval.

    Args:
        query: User query string
        query_analysis: Query analysis results
        retrieval_mode: Retrieval strategy
        top_k: Number of chunks to retrieve
        chunk_weight: Weight for chunk-based results
        entity_weight: Weight for entity-filtered results
        path_weight: Weight for path-based results
        graph_expansion: Whether to use graph expansion
        use_multi_hop: Whether to use multi-hop reasoning
        max_hops: Depth limit for graph traversal
        beam_size: Beam width for multi-hop search
        restrict_to_context: Whether to enforce context_documents filtering
        expansion_depth: Optional override for graph expansion depth
        context_documents: Optional list of document IDs to restrict retrieval scope

    Returns:
        List of relevant document chunks
    """
    try:
        allowed_docs = context_documents or []

        # Build the coroutine to run when needed
        coro = retrieve_documents_async(
            query=query,
            query_analysis=query_analysis,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            chunk_weight=chunk_weight,
            entity_weight=entity_weight,
            path_weight=path_weight,
            graph_expansion=graph_expansion,
            use_multi_hop=use_multi_hop,
            max_hops=max_hops,
            beam_size=beam_size,
            restrict_to_context=restrict_to_context,
            expansion_depth=expansion_depth,
            context_documents=allowed_docs,
            embedding_model=embedding_model,
        )

        # If we're running inside an event loop (e.g., FastAPI), submit the
        # coroutine to the shared blocking executor to avoid creating a short
        # lived ThreadPoolExecutor each call.
        try:
            asyncio.get_running_loop()
            executor = get_blocking_executor()
            try:
                fut = executor.submit(asyncio.run, coro)
            except RuntimeError:
                if SHUTTING_DOWN:
                    return []
                executor = get_blocking_executor()
                fut = executor.submit(asyncio.run, coro)

            return fut.result()
        except RuntimeError:
            # No running loop — safe to run directly
            return asyncio.run(coro)

    except Exception as e:
        logger.error(f"Error in synchronous retrieval wrapper: {e}")
        return []
