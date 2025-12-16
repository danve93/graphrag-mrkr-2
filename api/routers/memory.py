"""
Memory management API endpoints for layered memory system.

Provides endpoints for:
- User fact management (preferences, important information)
- Conversation history (summaries, not full transcripts)
- Session management

SECURITY NOTE:
These endpoints require proper authentication to prevent users from accessing
each other's data. The current implementation uses X-User-ID header validation.

For production use, implement proper authentication (OAuth2, JWT) and replace
the get_authenticated_user_id dependency with your auth system.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Header, Depends
from pydantic import BaseModel, Field

from core.conversation_memory import memory_manager
from core.graph_db import graph_db
from config.settings import settings
from api import auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])


# ============================================================
# Authentication & Authorization
# ============================================================

def get_authenticated_user_id(
    x_user_id: Optional[str] = Header(None, description="Authenticated user ID"),
    authorization: Optional[str] = Header(None),
) -> str:
    """Extract and validate authenticated user ID.

    SECURITY: This is a placeholder implementation. In production, replace this
    with proper OAuth2/JWT authentication that extracts user_id from validated tokens.

    Current implementation:
    - Requires valid authorization token (user or admin)
    - Reads user_id from X-User-ID header
    - In production, user_id should come from decoded JWT token

    Args:
        x_user_id: User ID from header (temporary - should come from token)
        authorization: Authorization token

    Returns:
        Authenticated user ID

    Raises:
        HTTPException: If authentication fails or user_id is missing
    """
    # Verify authorization token (validates user has valid credentials)
    try:
        auth.verify_token(authorization, require_admin=False)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide valid Authorization header."
        )

    # Extract user_id (in production, this should come from JWT token payload)
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-ID header required. In production, user ID should be extracted from JWT token."
        )

    return x_user_id


def verify_user_access(requested_user_id: str, authenticated_user_id: str) -> None:
    """Verify that authenticated user can access requested user's data.

    Args:
        requested_user_id: User ID from API path parameter
        authenticated_user_id: User ID from authentication

    Raises:
        HTTPException: If user tries to access another user's data
    """
    if requested_user_id != authenticated_user_id:
        logger.warning(
            f"Authorization denied: User {authenticated_user_id} attempted to access "
            f"data for user {requested_user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You can only access your own data."
        )


# ============================================================
# Request/Response Models
# ============================================================

class UserFactCreate(BaseModel):
    """Request model for creating a user fact."""
    content: str = Field(..., description="Fact content")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score (0.0-1.0)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class UserFactUpdate(BaseModel):
    """Request model for updating a user fact."""
    content: Optional[str] = Field(None, description="Updated content")
    importance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Updated importance")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class UserFactResponse(BaseModel):
    """Response model for user fact."""
    fact_id: str
    content: str
    importance: float
    created_at: Any
    metadata: Dict[str, Any]


class ConversationResponse(BaseModel):
    """Response model for conversation summary."""
    conversation_id: str
    user_id: str
    title: str
    summary: str
    created_at: Any
    updated_at: Any
    metadata: Dict[str, Any]


class SessionEndRequest(BaseModel):
    """Request model for ending a session."""
    save_summary: bool = Field(True, description="Whether to save conversation summary")
    title: Optional[str] = Field(None, description="Optional conversation title")
    summary: Optional[str] = Field(None, description="Optional conversation summary")


class SessionContextResponse(BaseModel):
    """Response model for session context."""
    user_id: str
    session_id: str
    facts_count: int
    conversations_count: int
    messages_count: int
    memory_prompt_length: int
    token_savings: Optional[Dict[str, int]] = None


# ============================================================
# User Facts Endpoints
# ============================================================

@router.post("/users/{user_id}/facts", response_model=UserFactResponse, status_code=status.HTTP_201_CREATED)
def create_user_fact(
    user_id: str,
    fact: UserFactCreate,
    authenticated_user: str = Depends(get_authenticated_user_id)
):
    """Create a new fact for a user.

    Args:
        user_id: User identifier
        fact: Fact creation data
        authenticated_user: Authenticated user ID from token

    Returns:
        Created fact data
    """
    if not settings.enable_memory_system:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Memory system is not enabled. Set ENABLE_MEMORY_SYSTEM=true in configuration."
        )

    # Verify user can only access their own data
    verify_user_access(user_id, authenticated_user)

    try:
        fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        result = memory_manager.add_user_fact(
            user_id=user_id,
            fact_id=fact_id,
            content=fact.content,
            importance=fact.importance,
            metadata=fact.metadata or {},
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create fact"
            )

        return UserFactResponse(**result)

    except Exception as e:
        logger.error(f"Failed to create user fact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create fact: {str(e)}"
        )


@router.get("/users/{user_id}/facts", response_model=List[UserFactResponse])
def list_user_facts(
    user_id: str,
    min_importance: float = 0.0,
    limit: int = 100,
    authenticated_user: str = Depends(get_authenticated_user_id)
):
    """List all facts for a user.

    Args:
        user_id: User identifier
        min_importance: Minimum importance threshold
        limit: Maximum number of facts to return
        authenticated_user: Authenticated user ID from token

    Returns:
        List of user facts
    """
    if not settings.enable_memory_system:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Memory system is not enabled"
        )

    # Verify user can only access their own data
    verify_user_access(user_id, authenticated_user)

    try:
        facts = graph_db.get_user_facts(
            user_id=user_id,
            min_importance=min_importance,
            limit=limit
        )
        return [UserFactResponse(**fact) for fact in facts]

    except Exception as e:
        logger.error(f"Failed to list user facts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list facts: {str(e)}"
        )


@router.put("/users/{user_id}/facts/{fact_id}", response_model=UserFactResponse)
def update_user_fact(
    user_id: str,
    fact_id: str,
    fact: UserFactUpdate,
    authenticated_user: str = Depends(get_authenticated_user_id)
):
    """Update a user fact.

    Args:
        user_id: User identifier
        fact_id: Fact identifier
        fact: Fact update data
        authenticated_user: Authenticated user ID from token

    Returns:
        Updated fact data
    """
    if not settings.enable_memory_system:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Memory system is not enabled"
        )

    # Verify user can only access their own data
    verify_user_access(user_id, authenticated_user)

    try:
        result = memory_manager.update_user_fact(
            user_id=user_id,
            fact_id=fact_id,
            content=fact.content,
            importance=fact.importance,
            metadata=fact.metadata,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Fact not found"
            )

        return UserFactResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user fact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update fact: {str(e)}"
        )


@router.delete("/users/{user_id}/facts/{fact_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_fact(
    user_id: str,
    fact_id: str,
    authenticated_user: str = Depends(get_authenticated_user_id)
):
    """Delete a user fact.

    Args:
        user_id: User identifier
        fact_id: Fact identifier
        authenticated_user: Authenticated user ID from token
    """
    if not settings.enable_memory_system:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Memory system is not enabled"
        )

    # Verify user can only access their own data
    verify_user_access(user_id, authenticated_user)

    try:
        deleted = memory_manager.delete_user_fact(user_id=user_id, fact_id=fact_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Fact not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user fact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete fact: {str(e)}"
        )


# ============================================================
# Conversation History Endpoints
# ============================================================

@router.get("/users/{user_id}/conversations", response_model=List[ConversationResponse])
def list_user_conversations(
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    authenticated_user: str = Depends(get_authenticated_user_id)
):
    """List recent conversations for a user.

    Args:
        user_id: User identifier
        limit: Maximum number of conversations to return
        offset: Number of conversations to skip
        authenticated_user: Authenticated user ID from token

    Returns:
        List of conversation summaries
    """
    if not settings.enable_memory_system:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Memory system is not enabled"
        )

    # Verify user can only access their own data
    verify_user_access(user_id, authenticated_user)

    try:
        conversations = graph_db.get_user_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        return [ConversationResponse(**conv) for conv in conversations]

    except Exception as e:
        logger.error(f"Failed to list user conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(conversation_id: str):
    """Get a specific conversation by ID.

    Args:
        conversation_id: Conversation identifier

    Returns:
        Conversation summary
    """
    if not settings.enable_memory_system:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Memory system is not enabled"
        )

    try:
        conversation = graph_db.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        return ConversationResponse(**conversation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}"
        )


# ============================================================
# Session Management Endpoints
# ============================================================

@router.get("/sessions/{session_id}/context", response_model=SessionContextResponse)
def get_session_context(session_id: str):
    """Get the current context for a session.

    Args:
        session_id: Session identifier

    Returns:
        Session context summary
    """
    if not settings.enable_memory_system:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Memory system is not enabled"
        )

    try:
        context = memory_manager.get_session_context(session_id)

        if not context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        # Build memory prompt to measure size
        memory_prompt = memory_manager.build_context_prompt(session_id)

        # Get token savings estimate
        token_savings = memory_manager.estimate_token_savings(session_id)

        return SessionContextResponse(
            user_id=context.user_id,
            session_id=context.session_id,
            facts_count=len(context.user_facts),
            conversations_count=len(context.conversation_summaries),
            messages_count=len(context.current_messages),
            memory_prompt_length=len(memory_prompt),
            token_savings=token_savings if "error" not in token_savings else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session context: {str(e)}"
        )


@router.post("/sessions/{session_id}/end", status_code=status.HTTP_204_NO_CONTENT)
def end_session(session_id: str, request: SessionEndRequest):
    """End a session and optionally save conversation summary.

    Args:
        session_id: Session identifier
        request: Session end request data
    """
    if not settings.enable_memory_system:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Memory system is not enabled"
        )

    try:
        memory_manager.end_session(
            session_id=session_id,
            save_summary=request.save_summary,
            title=request.title,
            summary=request.summary,
        )

    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end session: {str(e)}"
        )


# ============================================================
# Health & Status Endpoints
# ============================================================

@router.get("/health")
def memory_system_health():
    """Check memory system health and configuration.

    Returns:
        System status and configuration
    """
    return {
        "enabled": settings.enable_memory_system,
        "active_sessions": len(memory_manager.active_sessions),
        "configuration": {
            "max_facts": settings.memory_max_facts,
            "max_conversations": settings.memory_max_conversations,
            "min_fact_importance": settings.memory_min_fact_importance,
        }
    }
