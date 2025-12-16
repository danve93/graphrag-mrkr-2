"""
Chat router for handling chat requests and responses.
"""

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse

from api.models import ChatRequest, ChatResponse, FollowUpRequest, FollowUpResponse
from api.auth import get_current_user
from api.routers.chat_tuning import load_config as load_chat_tuning_config
from config.settings import settings
from api.services.chat_history_service import chat_history_service
from api.services.follow_up_service import follow_up_service
from core.quality_scorer import quality_scorer
from rag.graph_rag import graph_rag

logger = logging.getLogger(__name__)

router = APIRouter()


async def stream_response_generator(
    result: dict,
    session_id: str,
    message_id: str,
    user_query: str,
    context_documents: List[str],
    context_document_labels: List[str],
    context_hashtags: List[str],
    chat_history: List[dict],
    stage_updates: Optional[List[str]] = None,
) -> AsyncGenerator[str, None]:
    """Generate streaming response with SSE format."""
    try:
        # Emit pipeline stages progressively with timing metadata
        if stage_updates:
            logger.info(f"Emitting {len(stage_updates)} pipeline stages with timing")
            
            for stage_info in stage_updates:
                # Handle both old format (strings) and new format (dicts with timing)
                if isinstance(stage_info, str):
                    # Legacy format: just stage name
                    stage_data = {
                        "type": "stage",
                        "content": stage_info,
                    }
                else:
                    # New format: stage dict with name, duration_ms, timestamp, metadata
                    stage_data = {
                        "type": "stage",
                        "content": stage_info.get("name"),
                        "duration_ms": stage_info.get("duration_ms"),
                        "timestamp": stage_info.get("timestamp"),
                        "metadata": stage_info.get("metadata", {}),
                    }
                
                logger.info(f"Emitting stage: {stage_data['content']} ({stage_data.get('duration_ms', 0)}ms)")
                yield f"data: {json.dumps(stage_data)}\n\n"
                await asyncio.sleep(0.1)  # Small pause for smoother UI updates

        response_text = result.get("response", "")

        # Stream response with word-based buffering for smoother rendering
        if response_text:
            # Split into words while preserving whitespace and newlines
            words = []
            current_word = ""
            
            for char in response_text:
                current_word += char
                # Break on space or newline to create natural word boundaries
                if char in {" ", "\n", "\t"}:
                    if current_word:
                        words.append(current_word)
                        current_word = ""
            
            # Add any remaining content
            if current_word:
                words.append(current_word)

            # Stream words with small delay for natural typing effect
            for word in words:
                chunk_data = {
                    "type": "token",
                    "content": word,
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.015)  # Slightly faster for smoother feel

        # NOW emit quality calculation stage (AFTER response is done)
        quality_score = result.get("quality_score")
        try:
            # Emit quality calculation stage
            stage_data = {
                "type": "stage",
                "content": "quality_calculation",
            }
            yield f"data: {json.dumps(stage_data)}\n\n"
            await asyncio.sleep(0.05)

            if not quality_score:
                context_chunks = result.get("graph_context", [])
                if not context_chunks:
                    context_chunks = result.get("retrieved_chunks", [])

                relevant_chunks = [
                    chunk
                    for chunk in context_chunks
                    if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
                ]

                quality_score = quality_scorer.calculate_quality_score(
                    answer=response_text,
                    query=user_query,
                    context_chunks=relevant_chunks,
                    sources=result.get("sources", []),
                )
        except Exception as e:
            logger.warning(f"Quality scoring failed: {e}")

        # Generate follow-up questions and emit suggestions stage
        follow_up_questions = []
        try:
            # Emit suggestions stage LAST
            stage_data = {
                "type": "stage",
                "content": "suggestions",
            }
            yield f"data: {json.dumps(stage_data)}\n\n"
            await asyncio.sleep(0.05)

            alternative_chunks = result.get("alternative_chunks", [])
            logger.info(f"[Streaming] Passing {len(alternative_chunks)} alternative chunks to follow-up service")
            follow_up_questions = await follow_up_service.generate_follow_ups(
                query=user_query,
                response=response_text,
                sources=result.get("sources", []),
                chat_history=chat_history,
                alternative_chunks=alternative_chunks,
            )
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")

        # Save to chat history
        try:
            await chat_history_service.save_message(
                session_id=session_id,
                role="user",
                content=user_query,
                context_documents=context_documents,
                context_document_labels=context_document_labels,
            )
            await chat_history_service.save_message(
                session_id=session_id,
                role="assistant",
                content=response_text,
                sources=result.get("sources", []),
                quality_score=quality_score or result.get("quality_score"),
                follow_up_questions=follow_up_questions,
                context_documents=context_documents,
                context_document_labels=context_document_labels,
                context_hashtags=context_hashtags,
            )
            logger.info(f"Saved chat to history for session: {session_id}")
        except Exception as e:
            logger.warning(f"Could not save to chat history: {e}")

        # Send sources in batches to avoid SSE size limits
        sources = result.get("sources", [])
        MAX_CONTENT_LENGTH = 300  # Limit content to 300 chars for SSE transmission
        BATCH_SIZE = 5  # Send sources in batches of 5
        
        total_sources = len(sources)
        for batch_start in range(0, total_sources, BATCH_SIZE):
            batch_sources = sources[batch_start:batch_start + BATCH_SIZE]
            truncated_batch = []
            
            for source in batch_sources:
                # Create a minimal source object with only essential fields
                truncated_source = {
                    "chunk_id": source.get("chunk_id", ""),
                    "content": source.get("content", "")[:MAX_CONTENT_LENGTH] + ("..." if len(source.get("content", "")) > MAX_CONTENT_LENGTH else ""),
                    "similarity": source.get("similarity", 0.0),
                    "document_name": source.get("document_name", ""),
                    "citation": source.get("citation", ""),
                    "preview_url": source.get("preview_url", ""),
                }
                # Add entity-specific fields if present
                if "entity_name" in source:
                    truncated_source["entity_name"] = source.get("entity_name", "")
                    truncated_source["entity_type"] = source.get("entity_type", "")
                if len(source.get("content", "")) > MAX_CONTENT_LENGTH:
                    truncated_source["content_truncated"] = True
                truncated_batch.append(truncated_source)
            
            sources_data = {
                "type": "sources",
                "content": truncated_batch,
                "batch_info": {
                    "batch_start": batch_start,
                    "batch_size": len(truncated_batch),
                    "total_sources": total_sources,
                }
            }
            try:
                # Use ensure_ascii=False to properly handle unicode, separators to minimize size
                sources_json = json.dumps(sources_data, ensure_ascii=False, separators=(',', ':'))
                logger.info(f"Sending batch {batch_start//BATCH_SIZE + 1}/{(total_sources + BATCH_SIZE - 1)//BATCH_SIZE}: {len(truncated_batch)} sources, {len(sources_json)} bytes")
                yield f"data: {sources_json}\n\n"
            except Exception as e:
                logger.error(f"Failed to serialize sources batch: {e}")
                # Continue with next batch

        # Send quality score
        quality_payload = quality_score or result.get("quality_score")
        if quality_payload:
            quality_data = {
                "type": "quality_score",
                "content": quality_payload,
            }
            yield f"data: {json.dumps(quality_data)}\n\n"

        # Send follow-up questions
        if follow_up_questions:
            followup_data = {
                "type": "follow_ups",
                "content": follow_up_questions,
            }
            yield f"data: {json.dumps(followup_data)}\n\n"

        # Send metadata (with safe JSON serialization)
        metadata_data = {
            "type": "metadata",
            "content": {
                "message_id": message_id,
                "session_id": session_id,
                "metadata": result.get("metadata", {}),
                "context_documents": result.get("context_documents", []),
            },
        }
        try:
            metadata_json = json.dumps(metadata_data, ensure_ascii=False, separators=(',', ':'))
            yield f"data: {metadata_json}\n\n"
        except Exception as e:
            logger.error(f"Failed to serialize metadata: {e}")
            # Send minimal metadata on error
            minimal_metadata = {
                "type": "metadata",
                "content": {
                    "message_id": message_id,
                    "session_id": session_id,
                    "metadata": {},
                    "context_documents": [],
                },
            }
            yield f"data: {json.dumps(minimal_metadata)}\n\n"

        # Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Error in stream generator: {e}")
        error_data = {
            "type": "error",
            "content": str(e),
        }
        yield f"data: {json.dumps(error_data)}\n\n"


async def _prepare_chat_context(
    request: ChatRequest,
    user_id: Optional[str] = None,
) -> Tuple[str, List[dict], dict, List[str], List[str], List[str]]:
    """Load chat history, run RAG, and enrich metadata for downstream handlers."""

    session_id = request.session_id or str(uuid.uuid4())
    
    # Ensure session exists and is linked to user if provided
    await chat_history_service.create_session(
        session_id=session_id,
        user_id=user_id
    )

    # Determine user_type for TruLens tracking (external, admin, user, anonymous)
    user_type = "anonymous"
    if user_id:
        if user_id == "admin":
            user_type = "admin"
        else:
            try:
                from api.services.user_service import user_service
                user = user_service.get_user(user_id)
                if user:
                    user_type = user.get("role", "user")
            except Exception:
                user_type = "user"

    chat_history: List[dict] = []
    if request.session_id:
        try:
            # Enforce user ownership during history load? 
            # Currently get_conversation verifies ownership if user_id passed
            history = await chat_history_service.get_conversation(session_id, user_id=user_id)
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in history.messages
            ]
        except Exception as exc:
            logger.warning(f"Could not load chat history: {exc}")

    context_documents = request.context_documents or []
    context_document_labels = request.context_document_labels or []
    context_hashtags = request.context_hashtags or []

    # Load chat tuning config values and use them as runtime defaults when the
    # incoming request fields equal the original application defaults. This
    # lets the UI change behavior immediately without restarting the server.
    try:
        cfg = load_chat_tuning_config()
        param_values = {p["key"]: p["value"] for p in cfg.get("parameters", [])}
    except Exception:
        param_values = {}

    # Helper to prefer request value unless it equals the server default
    def _prefer_request_or_cfg(key: str, req_val, default_val):
        if req_val is None or req_val == default_val:
            return param_values.get(key, req_val)
        return req_val

    # Compute effective retrieval parameters
    effective_top_k = _prefer_request_or_cfg("top_k", request.top_k, 5)
    effective_chunk_weight = _prefer_request_or_cfg(
        "chunk_weight", request.chunk_weight, settings.hybrid_chunk_weight
    )
    effective_entity_weight = _prefer_request_or_cfg(
        "entity_weight", request.entity_weight, settings.hybrid_entity_weight
    )
    effective_path_weight = _prefer_request_or_cfg(
        "path_weight", request.path_weight, settings.hybrid_path_weight
    )
    effective_max_hops = _prefer_request_or_cfg(
        "max_hops", request.max_hops, settings.multi_hop_max_hops
    )
    effective_beam_size = _prefer_request_or_cfg(
        "beam_size", request.beam_size, settings.multi_hop_beam_size
    )
    effective_graph_expansion_depth = _prefer_request_or_cfg(
        "graph_expansion_depth", request.graph_expansion_depth, settings.max_expansion_depth
    )
    effective_restrict_to_context = _prefer_request_or_cfg(
        "restrict_to_context", request.restrict_to_context, settings.default_context_restriction
    )
    
    # Prefer chat tuning config for temperature and model selections if not explicitly provided
    effective_temperature = request.temperature
    if effective_temperature == 0.7:  # default value from ChatRequest
        effective_temperature = param_values.get("temperature", 0.7)
    
    effective_llm_model = request.llm_model
    if not effective_llm_model or effective_llm_model == getattr(settings, 'openai_model', None):
        effective_llm_model = param_values.get("llm_model", request.llm_model)
    
    effective_embedding_model = request.embedding_model
    if not effective_embedding_model or effective_embedding_model == getattr(settings, 'embedding_model', None):
        effective_embedding_model = param_values.get("embedding_model", request.embedding_model)

    # Apply other tuning that the retriever reads from `settings` directly
    # so that DocumentRetriever and expansion logic pick them up immediately.
    try:
        if "min_retrieval_similarity" in param_values:
            settings.min_retrieval_similarity = float(param_values["min_retrieval_similarity"])
        if "max_expanded_chunks" in param_values:
            settings.max_expanded_chunks = int(param_values["max_expanded_chunks"])
        if "expansion_similarity_threshold" in param_values:
            settings.expansion_similarity_threshold = float(param_values["expansion_similarity_threshold"])
        if "flashrank_enabled" in param_values:
            settings.flashrank_enabled = bool(param_values["flashrank_enabled"])
        if "flashrank_blend_weight" in param_values:
            settings.flashrank_blend_weight = float(param_values["flashrank_blend_weight"])
        if "flashrank_model_name" in param_values:
            try:
                settings.flashrank_model_name = str(param_values["flashrank_model_name"])
            except Exception:
                logger.warning("Invalid flashrank_model_name provided in chat tuning config; ignoring")
    except Exception as exc:
        logger.warning(f"Failed to apply chat-tuning retriever overrides to settings: {exc}")

    # Apply evaluation-specific feature flag overrides
    # These are temporary per-request overrides for A/B testing variants
    # Settings are restored immediately after graph_rag.query() completes
    original_settings: Dict[str, Any] = {}

    def _apply_eval_override(attr: str, value: Optional[bool]) -> None:
        """Apply per-request feature flag override if value is not None."""
        if value is not None:
            # Save original value for restoration
            original_settings[attr] = getattr(settings, attr)
            # Apply override
            setattr(settings, attr, value)
            logger.info(
                f"[Eval Override] {attr} = {value} (original: {original_settings[attr]})"
            )

    try:
        # Apply all evaluation overrides
        _apply_eval_override("enable_query_routing", request.eval_enable_query_routing)
        _apply_eval_override("enable_structured_kg", request.eval_enable_structured_kg)
        _apply_eval_override("enable_rrf", request.eval_enable_rrf)
        _apply_eval_override("enable_routing_cache", request.eval_enable_routing_cache)
        _apply_eval_override("flashrank_enabled", request.eval_flashrank_enabled)
        _apply_eval_override("enable_graph_clustering", request.eval_enable_graph_clustering)

        # Run RAG query with potentially overridden settings
        result = graph_rag.query(
            user_query=request.message,
            session_id=session_id,
            user_id=user_id,
            user_type=user_type,
            retrieval_mode=request.retrieval_mode,
            top_k=effective_top_k,
            temperature=effective_temperature,
            use_multi_hop=request.use_multi_hop,
            chunk_weight=effective_chunk_weight,
            entity_weight=effective_entity_weight,
            path_weight=effective_path_weight,
            max_hops=effective_max_hops,
            beam_size=effective_beam_size,
            restrict_to_context=effective_restrict_to_context,
            graph_expansion_depth=effective_graph_expansion_depth,
            llm_model=effective_llm_model,
            embedding_model=effective_embedding_model,
            chat_history=chat_history,
            context_documents=context_documents,
            category_filter=request.category_filter,
        )

    finally:
        # CRITICAL: Restore original settings even if query raises exception
        for attr, orig_value in original_settings.items():
            setattr(settings, attr, orig_value)
            logger.debug(f"[Eval Restore] {attr} = {orig_value}")

    metadata = result.get("metadata", {}) or {}
    metadata["chat_history_turns"] = len(chat_history)
    metadata.setdefault("context_documents", context_documents)
    metadata["retrieval_tuning"] = {
        "chunk_weight": effective_chunk_weight,
        "entity_weight": effective_entity_weight,
        "path_weight": effective_path_weight,
        "max_hops": effective_max_hops,
        "beam_size": effective_beam_size,
        "graph_expansion_depth": effective_graph_expansion_depth,
        "restrict_to_context": effective_restrict_to_context,
        "temperature": effective_temperature,
        "llm_model": effective_llm_model,
        "embedding_model": effective_embedding_model,
    }

    quality_score = result.get("quality_score")
    if not quality_score:
        try:
            context_chunks = result.get("graph_context", []) or result.get("retrieved_chunks", [])
            relevant_chunks = [
                chunk
                for chunk in context_chunks
                if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
            ]
            quality_score = quality_scorer.calculate_quality_score(
                answer=result.get("response", ""),
                query=request.message,
                context_chunks=relevant_chunks,
                sources=result.get("sources", []),
            )
        except Exception as exc:
            logger.warning(f"Quality scoring failed in chat preparation: {exc}")

    if quality_score:
        metadata["quality_score"] = quality_score
        result["quality_score"] = quality_score

    result["metadata"] = metadata

    return (
        session_id,
        chat_history,
        result,
        context_documents,
        context_document_labels,
        context_hashtags,
    )


@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest, user_id: Optional[str] = Depends(get_current_user)):
    """
    Handle chat query request.

    Args:
        request: Chat request with message and parameters
        user_id: Authenticated user ID (optional)

    Returns:
        Chat response with answer, sources, and metadata
    """
    try:
        (
            session_id,
            chat_history,
            result,
            context_documents,
            context_document_labels,
            context_hashtags,
        ) = await _prepare_chat_context(request, user_id=user_id)

        # Log the stages for debugging
        stages = result.get("stages", [])
        logger.info(f"RAG pipeline completed with stages: {stages}")

        # Generate unique message_id for feedback
        message_id = str(uuid.uuid4())

        # If streaming is requested, return SSE stream
        if request.stream:
            return StreamingResponse(
                stream_response_generator(
                    result,
                    session_id,
                    message_id,
                    request.message,
                    context_documents,
                    context_document_labels,
                    context_hashtags,
                    chat_history,
                    stages,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        quality_score = result.get("quality_score")

        # Generate follow-up questions
        follow_up_questions = []
        try:
            alternative_chunks = result.get("alternative_chunks", [])
            logger.info(f"Passing {len(alternative_chunks)} alternative chunks to follow-up service")
            follow_up_questions = await follow_up_service.generate_follow_ups(
                query=request.message,
                response=result.get("response", ""),
                sources=result.get("sources", []),
                chat_history=chat_history,
                alternative_chunks=alternative_chunks,
            )
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")

        # Save to chat history
        try:
            await chat_history_service.save_message(
                session_id=session_id,
                role="user",
                content=request.message,
                context_documents=context_documents,
                context_document_labels=context_document_labels,
                context_hashtags=context_hashtags,
            )
            await chat_history_service.save_message(
                session_id=session_id,
                role="assistant",
                content=result.get("response", ""),
                sources=result.get("sources", []),
                quality_score=quality_score,
                follow_up_questions=follow_up_questions,
                context_documents=context_documents,
                context_document_labels=context_document_labels,
                context_hashtags=context_hashtags,
            )
        except Exception as e:
            logger.warning(f"Could not save to chat history: {e}")

        # Calculate total duration from stages
        stages = result.get("stages", [])
        total_duration_ms = sum(
            stage.get("duration_ms", 0) for stage in stages if isinstance(stage, dict)
        )
        
        return ChatResponse(
            message=result.get("response", ""),
            sources=result.get("sources", []),
            quality_score=quality_score,
            follow_up_questions=follow_up_questions,
            session_id=session_id,
            metadata=result.get("metadata", {}),
            context_documents=result.get("context_documents", context_documents),
            stages=stages,
            total_duration_ms=total_duration_ms if total_duration_ms > 0 else None,
        )

    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest, http_request: Request, user_id: Optional[str] = Depends(get_current_user)):
    """Dedicated SSE endpoint that always streams responses."""

    try:
        # Prepare session and chat history (but do NOT run full RAG.query here)
        session_id = request.session_id or str(uuid.uuid4())

        chat_history: List[dict] = []
        if request.session_id:
            try:
                # Enforce user ownership
                history = await chat_history_service.get_conversation(session_id, user_id=user_id)
                chat_history = [{"role": msg.role, "content": msg.content} for msg in history.messages]
            except Exception as exc:
                logger.warning(f"Could not load chat history for streaming: {exc}")

        # Load chat tuning config values (same approach as _prepare_chat_context)
        try:
            cfg = load_chat_tuning_config()
            param_values = {p["key"]: p["value"] for p in cfg.get("parameters", [])}
        except Exception:
            param_values = {}

        def _prefer_request_or_cfg(key: str, req_val, default_val):
            if req_val is None or req_val == default_val:
                return param_values.get(key, req_val)
            return req_val

        effective_top_k = _prefer_request_or_cfg("top_k", request.top_k, 5)
        effective_chunk_weight = _prefer_request_or_cfg("chunk_weight", request.chunk_weight, settings.hybrid_chunk_weight)
        effective_entity_weight = _prefer_request_or_cfg("entity_weight", request.entity_weight, settings.hybrid_entity_weight)
        effective_path_weight = _prefer_request_or_cfg("path_weight", request.path_weight, settings.hybrid_path_weight)
        effective_max_hops = _prefer_request_or_cfg("max_hops", request.max_hops, settings.multi_hop_max_hops)
        effective_beam_size = _prefer_request_or_cfg("beam_size", request.beam_size, settings.multi_hop_beam_size)
        effective_graph_expansion_depth = _prefer_request_or_cfg("graph_expansion_depth", request.graph_expansion_depth, settings.max_expansion_depth)
        effective_restrict_to_context = _prefer_request_or_cfg("restrict_to_context", request.restrict_to_context, settings.default_context_restriction)

        effective_temperature = request.temperature
        if effective_temperature == 0.7:
            effective_temperature = param_values.get("temperature", 0.7)

        effective_llm_model = request.llm_model
        if not effective_llm_model or effective_llm_model == getattr(settings, 'openai_model', None):
            effective_llm_model = param_values.get("llm_model", request.llm_model)

        effective_embedding_model = request.embedding_model
        if not effective_embedding_model or effective_embedding_model == getattr(settings, 'embedding_model', None):
            effective_embedding_model = param_values.get("embedding_model", request.embedding_model)

        # If streaming at provider level is enabled, use GraphRAG.stream_query to stream tokens
        if getattr(settings, "enable_llm_streaming", False):
            # Monitor client disconnects and cancel the stream when detected
            cancel_event = asyncio.Event()

            async def _monitor_disconnect(req: Request, ev: asyncio.Event):
                try:
                    await req.is_disconnected()
                    ev.set()
                except Exception:
                    ev.set()

            # create monitor task using the explicit ASGI Request passed to this endpoint
            try:
                asyncio.create_task(_monitor_disconnect(http_request, cancel_event))
            except Exception:
                pass

            gen = graph_rag.stream_query(
                user_query=request.message,
                session_id=session_id,
                retrieval_mode=request.retrieval_mode,
                top_k=effective_top_k,
                temperature=effective_temperature,
                chunk_weight=effective_chunk_weight,
                entity_weight=effective_entity_weight,
                path_weight=effective_path_weight,
                graph_expansion=True,
                use_multi_hop=request.use_multi_hop,
                max_hops=effective_max_hops,
                beam_size=effective_beam_size,
                restrict_to_context=effective_restrict_to_context,
                graph_expansion_depth=effective_graph_expansion_depth,
                chat_history=chat_history,
                context_documents=request.context_documents or [],
                llm_model=effective_llm_model,
                embedding_model=effective_embedding_model,
                cancel_event=cancel_event,
            )

            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Fallback: run full RAG query and stream the pre-generated response (existing behavior)
        (
            session_id,
            chat_history,
            result,
            context_documents,
            context_document_labels,
            context_hashtags,
        ) = await _prepare_chat_context(request, user_id=user_id)

        stages = result.get("stages", [])
        logger.info(f"RAG pipeline completed with stages: {stages}")

        # Generate unique message_id for feedback
        message_id = str(uuid.uuid4())

        return StreamingResponse(
            stream_response_generator(
                result,
                session_id,
                message_id,
                request.message,
                context_documents,
                context_document_labels,
                context_hashtags,
                chat_history,
                stages,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as exc:
        logger.error(f"Streamed chat failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/follow-ups", response_model=FollowUpResponse)
async def generate_follow_ups(request: FollowUpRequest):
    """
    Generate follow-up questions based on conversation context.

    Args:
        request: Follow-up request with query, response, and context

    Returns:
        List of follow-up questions
    """
    try:
        questions = await follow_up_service.generate_follow_ups(
            query=request.query,
            response=request.response,
            sources=request.sources,
            chat_history=request.chat_history,
            alternative_chunks=getattr(request, 'alternative_chunks', []),
        )

        return FollowUpResponse(questions=questions)

    except Exception as e:
        logger.error(f"Follow-up generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
