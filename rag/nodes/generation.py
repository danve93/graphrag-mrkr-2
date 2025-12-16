"""
Response generation node for LangGraph RAG pipeline.
"""

import logging
from typing import Any, Dict, List

from core.llm import llm_manager
from core.quality_scorer import quality_scorer

logger = logging.getLogger(__name__)


def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    query_analysis: Dict[str, Any],
    temperature: float = 0.7,
    chat_history: List[Dict[str, str]] = None,
    llm_model: str | None = None,
    custom_prompt: str | None = None,
    memory_context: Any = None,
) -> Dict[str, Any]:
    """
    Generate response using retrieved context and query analysis.

    Args:
        query: User query string
        context_chunks: Retrieved document chunks
        query_analysis: Query analysis results
        temperature: LLM temperature for response generation
        chat_history: Optional conversation history for follow-up questions

    Returns:
        Dictionary containing response and metadata
    """
    try:
        # Debug: log incoming context chunk summary
        try:
            sample_ctx = [
                {"chunk_id": c.get("chunk_id") or c.get("id"), "similarity": c.get("similarity", c.get("hybrid_score", 0.0))}
                for c in (context_chunks or [])[:5]
            ]
            logger.info("Generation node received %d context chunks. Sample: %s", len(context_chunks or []), sample_ctx)
        except Exception:
            logger.debug("Failed to log context chunk sample in generation node")

        if not context_chunks:
            return {
                "response": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "metadata": {
                    "chunks_used": 0,
                    "query_type": query_analysis.get("query_type", "unknown"),
                },
                "quality_score": None,
            }

        # Deduplicate chunks by chunk_id while keeping the best scoring version
        deduped_chunks = {}
        for chunk in context_chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            score = chunk.get("similarity", chunk.get("hybrid_score", 0.0))
            if not chunk_id:
                chunk_id = f"anon_{hash(chunk.get('content', ''))}"
            current = deduped_chunks.get(chunk_id)
            if not current or score > current.get("similarity", current.get("hybrid_score", 0.0)):
                deduped_chunks[chunk_id] = chunk

        # Filter out chunks with 0.000 similarity before processing sources
        relevant_chunks = [
            chunk
            for chunk in deduped_chunks.values()
            if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
        ]

        # Build memory context prompt if available
        memory_prompt = None
        if memory_context:
            try:
                from core.conversation_memory import memory_manager
                session_id = getattr(memory_context, 'session_id', None)
                if session_id:
                    memory_prompt = memory_manager.build_context_prompt(
                        session_id=session_id,
                        include_facts=True,
                        include_conversation_summaries=True,
                        max_message_history=5,
                    )
                    if memory_prompt:
                        logger.info(f"Including memory context in generation (length: {len(memory_prompt)})")
            except Exception as e:
                logger.warning(f"Failed to build memory context prompt: {e}")

        # Prepend memory context to custom_prompt if both exist
        if memory_prompt and custom_prompt:
            custom_prompt = f"{memory_prompt}\n\n---\n\n{custom_prompt}"
        elif memory_prompt and not custom_prompt:
            # If we have memory but no custom prompt, we'll need to inject it differently
            # Add a special chunk at the beginning with memory context
            memory_chunk = {
                "chunk_id": "memory_context",
                "content": f"**User Context & History:**\n{memory_prompt}",
                "similarity": 1.0,
                "document_name": "User Memory",
            }
            relevant_chunks = [memory_chunk] + relevant_chunks

        # Generate response using LLM with only relevant chunks
        # Include chat history if this is a follow-up question
        response_data = llm_manager.generate_rag_response(
            query=query,
            context_chunks=relevant_chunks,
            include_sources=True,
            temperature=temperature,
            chat_history=chat_history if query_analysis.get("is_follow_up") else None,
            model_override=llm_model,
            custom_prompt=custom_prompt,
        )

        # Prepare sources information with entity support
        sources = []
        seen_citations = set()
        for i, chunk in enumerate(relevant_chunks):
            source_info = {
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "content": chunk.get("content", ""),
                "similarity": chunk.get("similarity", chunk.get("hybrid_score", 0.0)),
                "document_name": chunk.get("document_name", "Unknown Document"),
                "document_id": chunk.get("document_id", ""),
                "filename": chunk.get(
                    "filename", chunk.get("document_name", "Unknown Document")
                ),
                "metadata": chunk.get("metadata", {}),
                "chunk_index": chunk.get("chunk_index"),
                "citation": f"{chunk.get('document_name', 'Unknown Document')}#"
                f"{chunk.get('chunk_index', i)}",
            }

            # Build a preview URL that the frontend can deep-link to.
            # Prefer chunk_index when available, otherwise fall back to chunk_id.
            preview_url = None
            try:
                doc_id = source_info.get("document_id")
                cidx = source_info.get("chunk_index")
                cid = source_info.get("chunk_id")
                if doc_id:
                    if cidx is not None:
                        preview_url = f"/api/documents/{doc_id}/preview?chunk_index={cidx}"
                    elif cid:
                        preview_url = f"/api/documents/{doc_id}/preview?chunk_id={cid}"
            except Exception:
                preview_url = None

            if preview_url:
                source_info["preview_url"] = preview_url

            # Add entity information if available
            retrieval_mode = chunk.get("retrieval_mode", "")
            retrieval_source = chunk.get("retrieval_source", "")

            # Check if chunk has entity information regardless of retrieval mode
            contained_entities = chunk.get("contained_entities", [])
            relevant_entities = chunk.get("relevant_entities", [])

            # Use the most relevant entities or contained entities
            entities = relevant_entities or contained_entities

            # For entity-based retrieval, create entity sources
            if retrieval_mode == "entity_based" or retrieval_source == "entity_based":
                if entities:
                    # Create separate entity sources for entity-based retrieval
                    for entity_name in entities[:3]:  # Limit to top 3 entities
                        entity_source = {
                            "entity_name": entity_name,
                            "entity_type": "Entity",  # Default type
                            "entity_id": f"entity_{hash(entity_name) % 10000}",
                            "relevance_score": source_info["similarity"],
                            "content": chunk.get("content", ""),
                            "related_chunks": [
                                {
                                    "chunk_id": chunk.get("chunk_id"),
                                    "content": chunk.get("content", "")[:200] + "...",
                                }
                            ],
                            "document_name": source_info["document_name"],
                            "filename": source_info["filename"],
                            "citation": source_info["citation"],
                            "similarity": source_info["similarity"],
                        }

                        # Propagate preview_url to entity-level sources as well
                        if source_info.get("preview_url"):
                            entity_source["preview_url"] = source_info.get("preview_url")

                        if entity_source["citation"] not in seen_citations:
                            seen_citations.add(entity_source["citation"])
                            sources.append(entity_source)
                else:
                    # No entities, add as regular chunk
                    if source_info["citation"] not in seen_citations:
                        seen_citations.add(source_info["citation"])
                        sources.append(source_info)
            else:
                # For chunk-based or hybrid mode, add entity info to chunk source
                if entities:
                    source_info["contained_entities"] = entities
                    source_info["entity_enhanced"] = True

                if source_info["citation"] not in seen_citations:
                    seen_citations.add(source_info["citation"])
                    sources.append(source_info)

        # Enhance response with analysis insights
        query_type = query_analysis.get("query_type", "factual")
        complexity = query_analysis.get("complexity", "simple")

        metadata = {
            "chunks_used": len(relevant_chunks),
            "chunks_filtered": len(context_chunks) - len(relevant_chunks),
            "query_type": query_type,
            "complexity": complexity,
            "requires_reasoning": query_analysis.get("requires_reasoning", False),
            "key_concepts": query_analysis.get("key_concepts", []),
        }

        if len(relevant_chunks) < len(context_chunks):
            logger.info(
                f"Filtered out {len(context_chunks) - len(relevant_chunks)} chunks with 0.000 similarity"
            )
        logger.info(f"Generated response using {len(relevant_chunks)} relevant chunks")

        quality_score = quality_scorer.calculate_quality_score(
            answer=response_data.get("answer", ""),
            query=query,
            context_chunks=relevant_chunks,
            sources=sources,
        )

        return {
            "response": response_data.get("answer", ""),
            "sources": sources,
            "metadata": metadata,
            "quality_score": quality_score,
        }

    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return {
            "response": f"I apologize, but I encountered an error generating the response: {str(e)}",
            "sources": [],
            "metadata": {
                "error": str(e),
                "chunks_used": len(context_chunks) if context_chunks else 0,
            },
            "quality_score": None,
        }


def stream_generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    query_analysis: Dict[str, Any],
    temperature: float = 0.7,
    chat_history: List[Dict[str, str]] = None,
    llm_model: str | None = None,
):
    """
    Stream generation tokens for a RAG response. This is a synchronous
    generator that yields token fragments (strings) as produced by the LLM.
    After the stream completes, callers can reconstruct the full response
    from the concatenated tokens.
    """
    try:
        if not context_chunks:
            # Immediately yield a short message and return
            yield "I couldn't find any relevant information to answer your question."
            return

        # Deduplicate and filter chunks same as non-streaming path
        deduped_chunks = {}
        for chunk in context_chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            score = chunk.get("similarity", chunk.get("hybrid_score", 0.0))
            if not chunk_id:
                chunk_id = f"anon_{hash(chunk.get('content', ''))}"
            current = deduped_chunks.get(chunk_id)
            if not current or score > current.get("similarity", current.get("hybrid_score", 0.0)):
                deduped_chunks[chunk_id] = chunk

        relevant_chunks = [
            chunk
            for chunk in deduped_chunks.values()
            if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
        ]

        # Use LLM manager streaming interface
        from core.llm import llm_manager

        system_message = "You are a helpful assistant that answers questions based on the provided context."

        # Stream tokens from provider
        buffer = []
        try:
            for token in llm_manager.stream_generate_rag_response(
                query=query,
                context_chunks=relevant_chunks,
                system_message=system_message,
                include_sources=True,
                temperature=temperature,
                chat_history=chat_history if query_analysis.get("is_follow_up") else None,
                model_override=llm_model,
            ):
                # Yield each token fragment as-is
                buffer.append(token)
                yield token
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            # On error, yield an error message
            yield f"\n\n[Error streaming response: {e}]"
            return

        # Finalize: after stream ends we could compute sources/metadata/quality externally
        return
    except Exception as e:
        logger.error(f"Streamed response generator failed: {e}")
        yield f"I apologize, but I encountered an error generating the streamed response: {e}"
        return
