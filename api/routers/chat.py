"""
Chat router for handling chat requests and responses.
"""

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.models import ChatRequest, ChatResponse, FollowUpRequest, FollowUpResponse
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
    user_query: str,
    context_documents: List[str],
    context_document_labels: List[str],
    context_hashtags: List[str],
    chat_history: List[dict],
    stage_updates: Optional[List[str]] = None,
) -> AsyncGenerator[str, None]:
    """Generate streaming response with SSE format."""
    try:
        # Emit pipeline stages progressively with timing
        # Map actual backend stages to what was executed
        if stage_updates:
            logger.info(f"Emitting pipeline stages: {stage_updates}")
            
            # Core pipeline stages that always execute
            core_stages = ['query_analysis', 'retrieval', 'graph_reasoning', 'generation']
            
            for stage in core_stages:
                if stage in stage_updates:
                    logger.info(f"Emitting stage: {stage}")
                    stage_data = {
                        "type": "stage",
                        "content": stage,
                    }
                    yield f"data: {json.dumps(stage_data)}\n\n"
                    await asyncio.sleep(1.0)  # Pause to show each stage

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

            follow_up_questions = await follow_up_service.generate_follow_ups(
                query=user_query,
                response=response_text,
                sources=result.get("sources", []),
                chat_history=chat_history,
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

        # Send sources
        sources_data = {
            "type": "sources",
            "content": result.get("sources", []),
        }
        yield f"data: {json.dumps(sources_data)}\n\n"

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

        # Send metadata
        metadata_data = {
            "type": "metadata",
            "content": {
                "session_id": session_id,
                "metadata": result.get("metadata", {}),
                "context_documents": result.get("context_documents", []),
            },
        }
        yield f"data: {json.dumps(metadata_data)}\n\n"

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
) -> Tuple[str, List[dict], dict, List[str], List[str], List[str]]:
    """Load chat history, run RAG, and enrich metadata for downstream handlers."""

    session_id = request.session_id or str(uuid.uuid4())

    chat_history: List[dict] = []
    if request.session_id:
        try:
            history = await chat_history_service.get_conversation(session_id)
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

    result = graph_rag.query(
        user_query=request.message,
        retrieval_mode=request.retrieval_mode,
        top_k=effective_top_k,
        temperature=request.temperature,
        use_multi_hop=request.use_multi_hop,
        chunk_weight=effective_chunk_weight,
        entity_weight=effective_entity_weight,
        path_weight=effective_path_weight,
        max_hops=effective_max_hops,
        beam_size=effective_beam_size,
        restrict_to_context=effective_restrict_to_context,
        graph_expansion_depth=effective_graph_expansion_depth,
        llm_model=request.llm_model,
        embedding_model=request.embedding_model,
        chat_history=chat_history,
        context_documents=context_documents,
    )

    metadata = result.get("metadata", {}) or {}
    metadata["chat_history_turns"] = len(chat_history)
    metadata.setdefault("context_documents", context_documents)
    metadata["retrieval_tuning"] = {
        "chunk_weight": request.chunk_weight,
        "entity_weight": request.entity_weight,
        "path_weight": request.path_weight,
        "max_hops": request.max_hops,
        "beam_size": request.beam_size,
        "graph_expansion_depth": request.graph_expansion_depth,
        "restrict_to_context": request.restrict_to_context,
        "llm_model": request.llm_model,
        "embedding_model": request.embedding_model,
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
async def chat_query(request: ChatRequest):
    """
    Handle chat query request.

    Args:
        request: Chat request with message and parameters

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
        ) = await _prepare_chat_context(request)

        # Log the stages for debugging
        stages = result.get("stages", [])
        logger.info(f"RAG pipeline completed with stages: {stages}")

        # If streaming is requested, return SSE stream
        if request.stream:
            return StreamingResponse(
                stream_response_generator(
                    result,
                    session_id,
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
            follow_up_questions = await follow_up_service.generate_follow_ups(
                query=request.message,
                response=result.get("response", ""),
                sources=result.get("sources", []),
                chat_history=chat_history,
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

        return ChatResponse(
            message=result.get("response", ""),
            sources=result.get("sources", []),
            quality_score=quality_score,
            follow_up_questions=follow_up_questions,
            session_id=session_id,
            metadata=result.get("metadata", {}),
            context_documents=result.get("context_documents", context_documents),
        )

    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Dedicated SSE endpoint that always streams responses."""

    try:
        (
            session_id,
            chat_history,
            result,
            context_documents,
            context_document_labels,
            context_hashtags,
        ) = await _prepare_chat_context(request)

        stages = result.get("stages", [])
        logger.info(f"RAG pipeline completed with stages: {stages}")

        return StreamingResponse(
            stream_response_generator(
                result,
                session_id,
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
        )

        return FollowUpResponse(questions=questions)

    except Exception as e:
        logger.error(f"Follow-up generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
