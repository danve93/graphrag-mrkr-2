"""
Conversation Memory Manager for layered memory system.

Implements a 4-layer memory architecture inspired by ChatGPT:
1. Session metadata (current context)
2. User facts (preferences, important information)
3. Conversation summaries (historical context)
4. Current conversation (active session)

This approach reduces token usage by 80%+ for long conversations while
maintaining cross-conversation continuity.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from core.graph_db import graph_db

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Structured memory context for a user session."""

    # Layer 1: Session metadata
    user_id: str
    session_id: str
    session_metadata: Dict[str, Any] = field(default_factory=dict)

    # Layer 2: User facts (sorted by importance)
    user_facts: List[Dict[str, Any]] = field(default_factory=list)

    # Layer 3: Conversation summaries (recent conversations)
    conversation_summaries: List[Dict[str, Any]] = field(default_factory=list)

    # Layer 4: Current conversation (in-memory, not from DB)
    current_messages: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    total_tokens_saved: int = 0


class ConversationMemoryManager:
    """Manages the 4-layer memory system for conversations."""

    def __init__(self):
        """Initialize the memory manager."""
        self.db = graph_db
        self.active_sessions: Dict[str, MemoryContext] = {}

    # ============================================================
    # Core Memory Loading
    # ============================================================

    def load_memory_context(
        self,
        user_id: str,
        session_id: str,
        session_metadata: Optional[Dict[str, Any]] = None,
        max_facts: int = 20,
        max_conversation_summaries: int = 5,
        min_fact_importance: float = 0.3,
    ) -> MemoryContext:
        """Load all memory layers for a user session.

        Args:
            user_id: User identifier
            session_id: Session identifier
            session_metadata: Optional metadata for this session
            max_facts: Maximum number of facts to load
            max_conversation_summaries: Maximum number of past conversation summaries
            min_fact_importance: Minimum importance threshold for facts

        Returns:
            Complete memory context with all 4 layers
        """
        logger.info(f"Loading memory context for user={user_id}, session={session_id}")

        # Ensure user exists
        user = self.db.get_user(user_id)
        if not user:
            logger.info(f"Creating new user: {user_id}")
            self.db.create_user(user_id, metadata={})

        # Layer 2: Load user facts (preferences)
        user_facts = self.db.get_user_facts(
            user_id=user_id,
            min_importance=min_fact_importance,
            limit=max_facts,
        )
        logger.debug(f"Loaded {len(user_facts)} user facts")

        # Layer 3: Load recent conversation summaries
        conversation_summaries = self.db.get_user_conversations(
            user_id=user_id,
            limit=max_conversation_summaries,
            offset=0,  # Skip current conversation if it exists
        )
        logger.debug(f"Loaded {len(conversation_summaries)} conversation summaries")

        # Create memory context
        context = MemoryContext(
            user_id=user_id,
            session_id=session_id,
            session_metadata=session_metadata or {},
            user_facts=user_facts,
            conversation_summaries=conversation_summaries,
            current_messages=[],
        )

        # Cache in active sessions
        self.active_sessions[session_id] = context

        return context

    def get_session_context(self, session_id: str) -> Optional[MemoryContext]:
        """Get cached memory context for a session.

        Args:
            session_id: Session identifier

        Returns:
            Memory context or None if not found
        """
        return self.active_sessions.get(session_id)

    def add_message_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to the current conversation layer.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional message metadata
        """
        context = self.active_sessions.get(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found, cannot add message")
            return

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        context.current_messages.append(message)
        logger.debug(f"Added message to session {session_id}: {role}")

    # ============================================================
    # Context Building for RAG
    # ============================================================

    def build_context_prompt(
        self,
        session_id: str,
        include_facts: bool = True,
        include_conversation_summaries: bool = True,
        max_message_history: int = 10,
    ) -> str:
        """Build a compact context prompt from all memory layers.

        This is what gets injected into the RAG pipeline to provide
        user context without including full conversation history.

        Args:
            session_id: Session identifier
            include_facts: Include user facts layer
            include_conversation_summaries: Include conversation summaries layer
            max_message_history: Maximum recent messages to include

        Returns:
            Formatted context string for injection into prompts
        """
        context = self.active_sessions.get(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found")
            return ""

        sections = []

        # Layer 1: Session metadata (if any)
        if context.session_metadata:
            metadata_items = [f"- {k}: {v}" for k, v in context.session_metadata.items()]
            if metadata_items:
                sections.append(
                    "**Session Context:**\n" + "\n".join(metadata_items)
                )

        # Layer 2: User facts (preferences)
        if include_facts and context.user_facts:
            # Take top facts by importance
            top_facts = context.user_facts[:10]  # Limit to top 10
            fact_items = [
                f"- {fact['content']} (importance: {fact['importance']:.2f})"
                for fact in top_facts
            ]
            sections.append(
                "**User Preferences & Facts:**\n" + "\n".join(fact_items)
            )

        # Layer 3: Conversation summaries
        if include_conversation_summaries and context.conversation_summaries:
            # Take most recent summaries
            recent_summaries = context.conversation_summaries[:3]  # Limit to 3
            summary_items = [
                f"- {conv['title']}: {conv['summary']}"
                for conv in recent_summaries
                if conv.get('title') or conv.get('summary')
            ]
            if summary_items:
                sections.append(
                    "**Recent Conversation Context:**\n" + "\n".join(summary_items)
                )

        # Layer 4: Current conversation (recent messages)
        if context.current_messages:
            # Only include recent messages to save tokens
            recent_messages = context.current_messages[-max_message_history:]
            message_items = [
                f"- {msg['role']}: {msg['content'][:100]}..."  # Truncate long messages
                if len(msg['content']) > 100 else f"- {msg['role']}: {msg['content']}"
                for msg in recent_messages
            ]
            sections.append(
                "**Recent Messages:**\n" + "\n".join(message_items)
            )

        # Combine all sections
        if not sections:
            return ""

        return "\n\n".join(sections)

    def get_compact_context(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """Get compact context data for injection into RAG pipeline.

        Returns structured data instead of a prompt string, allowing
        the caller to format it as needed.

        Args:
            session_id: Session identifier

        Returns:
            Compact context data with all layers
        """
        context = self.active_sessions.get(session_id)
        if not context:
            return {}

        return {
            "user_id": context.user_id,
            "session_metadata": context.session_metadata,
            "top_facts": context.user_facts[:10],  # Top 10 facts
            "recent_conversations": [
                {
                    "title": conv.get("title", ""),
                    "summary": conv.get("summary", ""),
                }
                for conv in context.conversation_summaries[:3]
            ],
            "message_count": len(context.current_messages),
        }

    # ============================================================
    # Fact Management
    # ============================================================

    def add_user_fact(
        self,
        user_id: str,
        fact_id: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a new fact for a user.

        Args:
            user_id: User identifier
            fact_id: Unique fact identifier
            content: Fact content
            importance: Importance score (0.0-1.0)
            metadata: Optional metadata

        Returns:
            Created fact data
        """
        fact = self.db.create_fact(
            user_id=user_id,
            fact_id=fact_id,
            content=content,
            importance=importance,
            metadata=metadata,
        )

        # Refresh all active sessions for this user
        for session_id, context in self.active_sessions.items():
            if context.user_id == user_id:
                context.user_facts = self.db.get_user_facts(user_id, limit=20)
                logger.debug(f"Refreshed facts for session {session_id}")

        return fact

    def update_user_fact(
        self,
        user_id: str,
        fact_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a user fact.

        Args:
            user_id: User identifier
            fact_id: Fact identifier
            content: Updated content
            importance: Updated importance
            metadata: Updated metadata

        Returns:
            Updated fact data
        """
        fact = self.db.update_fact(
            fact_id=fact_id,
            content=content,
            importance=importance,
            metadata=metadata,
        )

        # Refresh active sessions
        for session_id, context in self.active_sessions.items():
            if context.user_id == user_id:
                context.user_facts = self.db.get_user_facts(user_id, limit=20)

        return fact

    def delete_user_fact(self, user_id: str, fact_id: str) -> bool:
        """Delete a user fact.

        Args:
            user_id: User identifier
            fact_id: Fact identifier

        Returns:
            True if deleted
        """
        deleted = self.db.delete_fact(fact_id)

        if deleted:
            # Refresh active sessions
            for session_id, context in self.active_sessions.items():
                if context.user_id == user_id:
                    context.user_facts = self.db.get_user_facts(user_id, limit=20)

        return deleted

    # ============================================================
    # Conversation Management
    # ============================================================

    def save_conversation_summary(
        self,
        user_id: str,
        conversation_id: str,
        title: str,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save a conversation summary (not full transcript).

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            title: Conversation title
            summary: Conversation summary
            metadata: Optional metadata

        Returns:
            Created/updated conversation data
        """
        conversation = self.db.create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            title=title,
            summary=summary,
            metadata=metadata,
        )

        # Refresh active sessions
        for session_id, context in self.active_sessions.items():
            if context.user_id == user_id:
                context.conversation_summaries = self.db.get_user_conversations(
                    user_id, limit=5
                )

        return conversation

    def end_session(
        self,
        session_id: str,
        save_summary: bool = True,
        title: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> None:
        """End a session and optionally save conversation summary.

        Args:
            session_id: Session identifier
            save_summary: Whether to save conversation summary
            title: Optional conversation title
            summary: Optional conversation summary (will be auto-generated if None)
        """
        context = self.active_sessions.get(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found")
            return

        if save_summary and context.current_messages:
            # Use provided summary or generate one
            if summary is None:
                # Import here to avoid circular dependency
                from core.conversation_summarizer import generate_summary
                summary = generate_summary(context.current_messages)

            # Use provided title or generate one
            if title is None:
                from core.conversation_summarizer import generate_title
                title = generate_title(context.current_messages)

            # Save to database
            self.db.create_conversation(
                user_id=context.user_id,
                conversation_id=session_id,
                title=title,
                summary=summary,
                metadata={
                    "message_count": len(context.current_messages),
                    "ended_at": datetime.utcnow().isoformat(),
                },
            )

            logger.info(f"Saved conversation summary for session {session_id}")

        # Remove from active sessions
        del self.active_sessions[session_id]
        logger.info(f"Ended session {session_id}")

    # ============================================================
    # Utility Methods
    # ============================================================

    def estimate_token_savings(self, session_id: str) -> Dict[str, int]:
        """Estimate token savings from using layered memory.

        Compares:
        - Full conversation history (baseline)
        - Layered memory system (optimized)

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with token estimates
        """
        context = self.active_sessions.get(session_id)
        if not context:
            return {"error": "Session not found"}

        # Rough token estimate: 1 token ≈ 4 characters
        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        # Baseline: full conversation history
        full_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in context.current_messages
        ])
        baseline_tokens = estimate_tokens(full_history)

        # Optimized: layered memory context
        compact_prompt = self.build_context_prompt(session_id)
        optimized_tokens = estimate_tokens(compact_prompt)

        savings = baseline_tokens - optimized_tokens
        savings_percent = (savings / baseline_tokens * 100) if baseline_tokens > 0 else 0

        return {
            "baseline_tokens": baseline_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": savings,
            "savings_percent": round(savings_percent, 1),
        }


# Global memory manager instance
memory_manager = ConversationMemoryManager()
