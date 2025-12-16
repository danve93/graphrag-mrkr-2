"""
History router for managing chat conversation history.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends

from api.models import (
    ConversationHistory,
    ConversationSession,
    MessageCreateRequest,
    MessageSearchResponse,
    SessionCreateRequest,
)
from api.services.chat_history_service import chat_history_service
from api.auth import get_current_user, require_admin

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/sessions", response_model=List[ConversationSession])
async def list_sessions(user_id: Optional[str] = Depends(get_current_user)):
    """
    List conversation sessions for the current user.
    If anonymous, returns empty list (or could return ephemeral sessions if tracked).
    """
    try:
        if user_id:
            sessions = await chat_history_service.list_user_sessions(user_id)
            return sessions
        
        # Security: Do not list all sessions for anonymous users
        return []

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions", response_model=ConversationSession)
async def create_session(request: SessionCreateRequest, user_id: Optional[str] = Depends(get_current_user)):
    """Create a new conversation session explicitly."""

    try:
        session_id = await chat_history_service.create_session(
            session_id=request.session_id, title=request.title, user_id=user_id
        )

        # Return the created session object
        # We construct it manually since it's new
        return ConversationSession(
            session_id=session_id,
            created_at="", # Service generates timestamp, we could fetch it but this is faster
            updated_at="",
            message_count=0,
            preview=None,
            title=request.title,
        )

    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=ConversationHistory)
async def get_conversation(session_id: str, user_id: Optional[str] = Depends(get_current_user)):
    """
    Get conversation history for a specific session.
    Verifies ownership if user is logged in.
    """
    try:
        viewer_role = "user"
        if user_id:
            from api.services.user_service import user_service
            user = user_service.get_user(user_id)
            if user:
                viewer_role = user.get("role", "user")

        history = await chat_history_service.get_conversation(session_id, user_id=user_id, viewer_role=viewer_role)
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


@router.post("/{session_id}/share")
async def share_session(session_id: str, user_id: Optional[str] = Depends(get_current_user)):
    """
    Share a session with the admin (or other target roles).
    Requires authenticated user (either internal or external).
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required to share session")

    # Verify ownership before sharing? 
    # Current get_conversation enforces ownership, so we assume user_id is checked there
    # But share_session in service just takes ID. Ideally we check ownership here.
    
    try:
        # Check ownership logic could be here, but for MVP:
        # If user can read it, they can share it.
        # Calling get_conversation enforces ownership check if applicable.
        try:
            await chat_history_service.get_conversation(session_id, user_id=user_id)
        except ValueError:
            raise HTTPException(status_code=403, detail="Session not found or not owned by user")
            
        success = await chat_history_service.share_session(session_id, target_role="admin")
        return {"status": "success", "message": f"Session {session_id} shared with admin"}

    except Exception as e:
        logger.error(f"Failed to share session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}/share")
async def unshare_session(session_id: str, user_id: Optional[str] = Depends(get_current_user)):
    """
    Unshare a previously shared session.
    Requires authenticated user (owner of the session).
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required to unshare session")
    
    try:
        # Check ownership
        try:
            await chat_history_service.get_conversation(session_id, user_id=user_id)
        except ValueError:
            raise HTTPException(status_code=403, detail="Session not found or not owned by user")
        
        success = await chat_history_service.unshare_session(session_id)
        return {"status": "success", "message": f"Session {session_id} unshared"}

    except Exception as e:
        logger.error(f"Failed to unshare session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/shared", response_model=List[ConversationSession])
async def list_shared_sessions(
    limit: int = 50, 
    offset: int = 0, 
    _token: str = Depends(require_admin)
):
    """
    List shared sessions for admin review.
    Requires Admin privileges.
    """
    try:
        return await chat_history_service.list_shared_sessions(limit=limit, offset=offset)
    except Exception as e:
        logger.error(f"Failed to list shared sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
