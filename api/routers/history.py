"""
History router for managing chat conversation history.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.models import (
    ConversationHistory,
    ConversationSession,
    MessageCreateRequest,
    MessageSearchResponse,
    SessionCreateRequest,
)
from api.services.chat_history_service import chat_history_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/sessions", response_model=List[ConversationSession])
async def list_sessions():
    """
    List all conversation sessions.

    Returns:
        List of conversation sessions with metadata
    """
    try:
        sessions = await chat_history_service.list_sessions()
        return sessions

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions", response_model=ConversationSession)
async def create_session(request: SessionCreateRequest):
    """Create a new conversation session explicitly."""

    try:
        session_id = await chat_history_service.create_session(
            session_id=request.session_id, title=request.title
        )

        sessions = await chat_history_service.list_sessions()
        created = next((s for s in sessions if s.session_id == session_id), None)
        if created:
            return created

        return ConversationSession(
            session_id=session_id,
            created_at="",
            updated_at="",
            message_count=0,
            preview=None,
        )

    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=ConversationHistory)
async def get_conversation(session_id: str):
    """
    Get conversation history for a specific session.

    Args:
        session_id: Session ID

    Returns:
        Conversation history with all messages
    """
    try:
        history = await chat_history_service.get_conversation(session_id)
        return history

    except Exception as e:
        logger.error(f"Failed to get conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/messages", response_model=ConversationHistory)
async def add_message(session_id: str, request: MessageCreateRequest):
    """Add a message to a session (useful for manual CRUD operations)."""

    try:
        await chat_history_service.save_message(
            session_id=session_id,
            role=request.role,
            content=request.content,
            sources=request.sources,
            quality_score=request.quality_score,
            follow_up_questions=request.follow_up_questions,
            context_documents=request.context_documents,
            context_document_labels=request.context_document_labels,
            context_hashtags=request.context_hashtags,
        )
        return await chat_history_service.get_conversation(session_id)

    except Exception as e:
        logger.error(f"Failed to add message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_conversation(session_id: str):
    """
    Delete a conversation session.

    Args:
        session_id: Session ID to delete

    Returns:
        Deletion result
    """
    try:
        await chat_history_service.delete_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted"}

    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{session_id}/restore")
async def restore_conversation(session_id: str):
    """Restore a previously soft-deleted session."""

    try:
        await chat_history_service.restore_session(session_id)
        return await chat_history_service.get_conversation(session_id)

    except Exception as e:
        logger.error(f"Failed to restore session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_history():
    """
    Clear all conversation history.

    Returns:
        Clear operation result
    """
    try:
        await chat_history_service.clear_all()
        return {"status": "success", "message": "All history cleared"}

    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}/messages/{message_id}")
async def delete_message(session_id: str, message_id: str):
    """Soft delete a single message within a session."""

    try:
        await chat_history_service.soft_delete_message(session_id, message_id)
        return {"status": "success", "message": f"Message {message_id} deleted"}

    except Exception as e:
        logger.error(f"Failed to delete message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=MessageSearchResponse)
async def search_messages(query: str, include_deleted: Optional[bool] = Query(False)):
    """Search messages across sessions by substring content."""

    try:
        return await chat_history_service.search_messages(query, include_deleted)

    except Exception as e:
        logger.error(f"Failed to search messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))
