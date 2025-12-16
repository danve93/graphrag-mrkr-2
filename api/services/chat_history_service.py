"""Chat history service for managing conversation persistence."""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from api.models import (
    ChatMessage,
    ConversationHistory,
    ConversationSession,
    MessageSearchResponse,
    MessageSearchResult,
)
from core.graph_db import graph_db

logger = logging.getLogger(__name__)


def strip_markdown(text: str) -> str:
    """
    Remove markdown formatting from text.
    
    Args:
        text: Text with markdown formatting
        
    Returns:
        Plain text without markdown tags
    """
    if not text:
        return text
    
    # Remove headers (# ## ### etc)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic (**text** or __text__ or *text* or _text_)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)  # bold+italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)  # italic
    text = re.sub(r'___(.+?)___', r'\1', text)  # bold+italic
    text = re.sub(r'__(.+?)__', r'\1', text)  # bold
    text = re.sub(r'_(.+?)_', r'\1', text)  # italic
    
    # Remove strikethrough (~~text~~)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    
    # Remove inline code (`text`)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # Remove links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove images ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    
    # Remove blockquotes (> text)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Remove horizontal rules (---, ***, ___)
    text = re.sub(r'^[\-\*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Remove list markers (-, *, +, 1.)
    text = re.sub(r'^\s*[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


class ChatHistoryService:
    """Service for managing chat conversation history."""

    @staticmethod
    def _generate_timestamp() -> str:
        """Return a timezone aware ISO timestamp."""

        return datetime.now().astimezone().isoformat()

    async def create_session(self, session_id: Optional[str] = None, title: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """Create a session node if it doesn't exist and return the id."""

        session_identifier = session_id or str(uuid.uuid4())

        driver = graph_db.driver
        if driver is None:
            raise RuntimeError("Neo4j driver is not initialized")

        timestamp = self._generate_timestamp()
        
        # Dual-write schema: Support both ChatHistory (:ConversationSession) and Memory (:Conversation)
        # 1. :ConversationSession used by legacy Chat
        # 2. :Conversation used by Memory Manager
        # Relationships:
        # - :HAS_CONVERSATION -> :Conversation (Memory System)
        if user_id:
            query = """
            MERGE (u:User {id: $user_id})
            MERGE (s:ConversationSession {session_id: $session_id})
            ON CREATE SET s.created_at = $timestamp,
                          s.updated_at = $timestamp,
                          s.deleted_at = null,
                          s.title = $title
            ON MATCH SET s.updated_at = $timestamp,
                          s.deleted_at = null,
                          s.title = coalesce($title, s.title)
            SET s:Conversation,
                s.id = $session_id,
                s.user_id = $user_id
            
            MERGE (u)-[:HAS_CONVERSATION]->(s)
            MERGE (u)-[:HAS_SESSION]->(s)
            """
            driver.execute_query(
                query,
                session_id=session_identifier,
                user_id=user_id,
                timestamp=timestamp,
                title=title,
            )
        else:
            # Anonymous session (legacy behavior)
            driver.execute_query(
                """
                MERGE (s:ConversationSession {session_id: $session_id})
                ON CREATE SET s.created_at = $timestamp,
                              s.updated_at = $timestamp,
                              s.deleted_at = null,
                              s.title = $title
                ON MATCH SET s.updated_at = $timestamp,
                              s.deleted_at = null,
                              s.title = coalesce($title, s.title)
                SET s:Conversation,
                    s.id = $session_id
                """,
                session_id=session_identifier,
                timestamp=timestamp,
                title=title,
            )

        return session_identifier

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        quality_score: Optional[Dict[str, Any]] = None,
        follow_up_questions: Optional[List[str]] = None,
        context_documents: Optional[List[str]] = None,
        context_document_labels: Optional[List[str]] = None,
        context_hashtags: Optional[List[str]] = None,
    ) -> None:
        """
        Save a message to chat history.

        Args:
            session_id: Conversation session ID
            role: Message role (user/assistant)
            content: Message content
            sources: Optional sources for assistant messages
            quality_score: Optional quality score for assistant messages
            follow_up_questions: Optional follow-up questions
            context_documents: Optional list of context document IDs
            context_document_labels: Optional list of context document labels/names
            context_hashtags: Optional list of context hashtags
        """
        try:
            # Use the machine's local timezone so frontend shows relative times in local context
            timestamp = self._generate_timestamp()
            message_id = str(uuid.uuid4())

            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")

            # Create message node in Neo4j
            query = """
            MERGE (s:ConversationSession {session_id: $session_id})
            ON CREATE SET s.created_at = $timestamp,
                          s.updated_at = $timestamp,
                          s.deleted_at = null
            ON MATCH SET s.updated_at = $timestamp,
                          s.deleted_at = null
            CREATE (m:Message {
                message_id: $message_id,
                role: $role,
                content: $content,
                timestamp: $timestamp,
                sources: $sources,
                quality_score: $quality_score,
                follow_up_questions: $follow_up_questions,
                context_documents: $context_documents,
                context_document_labels: $context_document_labels,
                context_hashtags: $context_hashtags,
                deleted_at: null
            })
            CREATE (s)-[:HAS_MESSAGE]->(m)
            """

            driver.execute_query(
                query,
                session_id=session_id,
                message_id=message_id,
                role=role,
                content=content,
                timestamp=timestamp,
                sources=json.dumps(sources or []),
                quality_score=json.dumps(quality_score) if quality_score else None,
                follow_up_questions=follow_up_questions or [],
                context_documents=context_documents or [],
                context_document_labels=context_document_labels or [],
                context_hashtags=context_hashtags or [],
            )

            logger.info(f"Saved message to session {session_id} with id {message_id}")

        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            raise

    async def get_conversation(self, session_id: str, user_id: Optional[str] = None, viewer_role: str = "user") -> ConversationHistory:
        """
        Retrieve conversation history for a session.

        Args:
            session_id: Conversation session ID
            user_id: Optional User ID to enforce ownership
            viewer_role: Role of the viewer (affects access control)

        Returns:
            Conversation history with all messages
        """
        try:
            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")

            query = """
            MATCH (s:ConversationSession {session_id: $session_id})
            """
            
            if user_id:
                if viewer_role == "admin":
                    # Admin can see:
                    # 1. Their own sessions (s.user_id = $user_id)
                    # 2. Sessions shared with admin (s.is_shared = true AND s.shared_with_role = 'admin')
                    # 3. Orphaned/Anonymous sessions? (maybe not needed here, stick to strict)
                    query += """
                    WHERE s.user_id = $user_id 
                       OR (s.is_shared = true AND s.shared_with_role = 'admin')
                       OR (s.user_id IS NULL)
                    """
                else:
                    query += "WHERE s.user_id = $user_id OR (s.user_id IS NULL) "
            
            query += """
            MATCH (s)-[:HAS_MESSAGE]->(m:Message)
            WHERE (s.deleted_at IS NULL OR s.deleted_at = "")
              AND (m.deleted_at IS NULL OR m.deleted_at = "")
            RETURN s, m
            ORDER BY m.timestamp
            """

            result = driver.execute_query(query, session_id=session_id, user_id=user_id)

            if not result or not result.records:
                raise ValueError(f"Session {session_id} not found")

            messages = []
            session_data: Dict[str, Any] = {}

            for record in result.records:
                if not session_data:
                    session_node = record["s"]
                    session_data = {
                        "created_at": session_node.get("created_at", ""),
                        "updated_at": session_node.get("updated_at", ""),
                    }

                msg_node = record["m"]
                sources_data = msg_node.get("sources")
                if isinstance(sources_data, str):
                    try:
                        sources_data = json.loads(sources_data)
                    except json.JSONDecodeError:
                        sources_data = []

                quality_data = msg_node.get("quality_score")
                if isinstance(quality_data, str):
                    try:
                        quality_data = json.loads(quality_data)
                    except json.JSONDecodeError:
                        quality_data = None

                messages.append(
                    ChatMessage(
                        role=msg_node.get("role", ""),
                        message_id=msg_node.get("message_id"),
                        content=msg_node.get("content", ""),
                        timestamp=msg_node.get("timestamp"),
                        sources=sources_data,
                        quality_score=quality_data,
                        follow_up_questions=msg_node.get("follow_up_questions"),
                        context_documents=msg_node.get("context_documents"),
                        context_document_labels=msg_node.get("context_document_labels"),
                        context_hashtags=msg_node.get("context_hashtags"),
                    )
                )

            # Normalize session timestamps if they are numeric epoch ms
            def _normalize_ts(value):
                try:
                    # Numeric epoch (ms) from Neo4j -> convert to local tz ISO
                    if isinstance(value, (int, float)):
                        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).astimezone().isoformat()
                    # If it's already a string, return as-is (assume it includes timezone or local time)
                    if isinstance(value, str) and value:
                        return value
                except Exception:
                    pass
                return ""

            return ConversationHistory(
                session_id=session_id,
                messages=messages,
                created_at=_normalize_ts(session_data.get("created_at", "")),
                updated_at=_normalize_ts(session_data.get("updated_at", "")),
                deleted_at=_normalize_ts(session_data.get("deleted_at", "")),
            )

        except Exception as e:
            logger.error(f"Failed to get conversation: {e}")
            raise

    async def list_sessions(self) -> List[ConversationSession]:
        """
        List all conversation sessions.

        Returns:
            List of conversation sessions
        """
        try:
            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")

            query = (
                "MATCH (s:ConversationSession)\n"
                "WHERE s.deleted_at IS NULL OR s.deleted_at = ''\n"
                "OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)\n"
                "WHERE m.deleted_at IS NULL OR m.deleted_at = ''\n"
                "WITH s, [msg IN collect(m) WHERE msg IS NOT NULL] as messages\n"
                "RETURN s.session_id as session_id,\n"
                "       s.created_at as created_at,\n"
                "       s.updated_at as updated_at,\n"
                "       s.deleted_at as deleted_at,\n"
                "       size(messages) as message_count,\n"
                "       CASE WHEN size(messages) > 0 THEN messages[0].content ELSE '' END as preview\n"
                "ORDER BY coalesce(s.updated_at, s.created_at) DESC"
            )

            result = driver.execute_query(query)

            sessions = []
            if result and result.records:
                for record in result.records:
                    preview = record["preview"]
                    if preview:
                        # Strip markdown formatting from preview
                        preview = strip_markdown(preview)
                        if len(preview) > 100:
                            preview = preview[:100] + "..."

                    # Normalize created_at / updated_at to ISO8601 strings.
                    def _normalize_ts(value):
                        # If Neo4j stored an epoch milliseconds integer, convert it to local tz ISO
                        try:
                            if isinstance(value, (int, float)):
                                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).astimezone().isoformat()
                            if isinstance(value, str) and value:
                                return value
                        except Exception:
                            pass
                        return ""

                    sessions.append(
                        ConversationSession(
                            session_id=record["session_id"],
                            created_at=_normalize_ts(record.get("created_at")),
                            updated_at=_normalize_ts(record.get("updated_at")),
                            message_count=record["message_count"],
                            preview=preview,
                            deleted_at=_normalize_ts(record.get("deleted_at")),
                        )
                    )

            return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            raise

    async def list_user_sessions(self, user_id: str, limit: int = 20, offset: int = 0) -> List[ConversationSession]:
        """
        List conversations for a specific user.
        """
        try:
            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")
            
            # Using the schema compatible with Memory Manager (:User)-[:HAS_CONVERSATION]->(:Conversation)
            # but returning ConversationSession objects
            query = (
                "MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(s:Conversation)\n"
                "WHERE (s.deleted_at IS NULL OR s.deleted_at = '')\n"
                "OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)\n"
                "WHERE m.deleted_at IS NULL OR m.deleted_at = ''\n"
                "WITH s, count(m) as msg_count, collect(m)[0] as first_msg\n"
                "RETURN s.session_id as session_id,\n"
                "       s.created_at as created_at,\n"
                "       s.updated_at as updated_at,\n"
                "       s.deleted_at as deleted_at,\n"
                "       s.title as title,\n"
                "       msg_count as message_count,\n"
                "       CASE WHEN s.summary <> '' AND s.summary IS NOT NULL THEN s.summary ELSE first_msg.content END as preview\n"
                "ORDER BY s.updated_at DESC\n"
                "SKIP $offset LIMIT $limit"
            )
            
            result = driver.execute_query(query, user_id=user_id, limit=limit, offset=offset)

            sessions = []
            if result and result.records:
                for record in result.records:
                    preview = record["preview"] or ""
                    if preview:
                        preview = strip_markdown(preview)
                        if len(preview) > 100:
                            preview = preview[:100] + "..."

                    def _normalize_ts(value):
                        try:
                            if isinstance(value, (int, float)):
                                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).astimezone().isoformat()
                            if isinstance(value, str) and value:
                                return value
                        except Exception:
                            pass
                        return ""

                    sessions.append(
                        ConversationSession(
                            session_id=record["session_id"],
                            created_at=_normalize_ts(record.get("created_at")),
                            updated_at=_normalize_ts(record.get("updated_at")),
                            message_count=record["message_count"],
                            preview=preview,
                            deleted_at=_normalize_ts(record.get("deleted_at")),
                            title=record.get("title")
                        )
                    )
            return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            raise

    async def delete_session(self, session_id: str) -> None:
        """
        Delete a conversation session.

        Args:
            session_id: Session ID to delete
        """
        try:
            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")

            timestamp = self._generate_timestamp()

            query = """
            MATCH (s:ConversationSession {session_id: $session_id})
            SET s.deleted_at = $timestamp,
                s.updated_at = $timestamp
            WITH s
            OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
            SET m.deleted_at = $timestamp
            """

            driver.execute_query(query, session_id=session_id, timestamp=timestamp)
            logger.info(f"Soft deleted session {session_id}")

        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            raise

    async def clear_all(self) -> None:
        """Clear all conversation history."""
        try:
            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")

            query = """
            MATCH (s:ConversationSession)
            SET s.deleted_at = $timestamp,
                s.updated_at = $timestamp
            WITH s
            OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
            SET m.deleted_at = $timestamp
            """

            timestamp = self._generate_timestamp()
            driver.execute_query(query, timestamp=timestamp)
            logger.info("Soft cleared all conversation history")

        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            raise

    async def restore_session(self, session_id: str) -> None:
        """Restore a previously soft-deleted session and its messages."""

        driver = graph_db.driver
        if driver is None:
            raise RuntimeError("Neo4j driver is not initialized")

        timestamp = self._generate_timestamp()

        driver.execute_query(
            """
            MATCH (s:ConversationSession {session_id: $session_id})
            SET s.deleted_at = null,
                s.updated_at = $timestamp
            WITH s
            OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
            SET m.deleted_at = null
            """,
            session_id=session_id,
            timestamp=timestamp,
        )

    async def soft_delete_message(self, session_id: str, message_id: str) -> None:
        """Mark a single message as deleted without removing its session."""

        driver = graph_db.driver
        if driver is None:
            raise RuntimeError("Neo4j driver is not initialized")

        timestamp = self._generate_timestamp()
        driver.execute_query(
            """
            MATCH (s:ConversationSession {session_id: $session_id})-[:HAS_MESSAGE]->(m:Message {message_id: $message_id})
            SET m.deleted_at = $timestamp
            """,
            session_id=session_id,
            message_id=message_id,
            timestamp=timestamp,
        )

    async def search_messages(self, query: str, include_deleted: bool = False) -> MessageSearchResponse:
        """Search messages by a case-insensitive substring query."""

        driver = graph_db.driver
        if driver is None:
            raise RuntimeError("Neo4j driver is not initialized")

        filters = ""
        if not include_deleted:
            filters = (
                "WHERE (s.deleted_at IS NULL OR s.deleted_at = '') "
                "AND (m.deleted_at IS NULL OR m.deleted_at = '')"
            )

        cypher = f"""
        MATCH (s:ConversationSession)-[:HAS_MESSAGE]->(m:Message)
        {filters}
        WITH s, m
        WHERE toLower(m.content) CONTAINS toLower($query)
        RETURN s.session_id as session_id,
               m.message_id as message_id,
               m.role as role,
               m.content as content,
               m.timestamp as timestamp,
               m.quality_score as quality_score,
               m.context_documents as context_documents
        ORDER BY timestamp DESC
        LIMIT 50
        """

        result = driver.execute_query(cypher, query=query)

        results: List[MessageSearchResult] = []
        if result and result.records:
            for record in result.records:
                quality_data = record.get("quality_score")
                if isinstance(quality_data, str):
                    try:
                        quality_data = json.loads(quality_data)
                    except json.JSONDecodeError:
                        quality_data = None

                results.append(
                    MessageSearchResult(
                        session_id=record.get("session_id"),
                        message_id=record.get("message_id"),
                        role=record.get("role", ""),
                        content=record.get("content", ""),
                        timestamp=record.get("timestamp"),
                        quality_score=quality_data,
                        context_documents=record.get("context_documents") or [],
                    )
                )

        return MessageSearchResponse(query=query, results=results)


    async def share_session(self, session_id: str, target_role: str = "admin") -> bool:
        """
        Mark a session as shared with a specific role.
        """
        try:
            timestamp = self._generate_timestamp()
            query = """
            MATCH (s:ConversationSession {session_id: $session_id})
            SET s.is_shared = true,
                s.shared_with_role = $target_role,
                s.updated_at = $timestamp
            RETURN s
            """
            result = graph_db.driver.execute_query(
                query, 
                session_id=session_id, 
                target_role=target_role,
                timestamp=timestamp
            )
            return bool(result and result.records)
        except Exception as e:
            logger.error(f"Failed to share session: {e}")
            raise

    async def unshare_session(self, session_id: str) -> bool:
        """
        Remove sharing from a session.
        """
        try:
            timestamp = self._generate_timestamp()
            query = """
            MATCH (s:ConversationSession {session_id: $session_id})
            SET s.is_shared = false,
                s.shared_with_role = null,
                s.updated_at = $timestamp
            RETURN s
            """
            result = graph_db.driver.execute_query(
                query, 
                session_id=session_id, 
                timestamp=timestamp
            )
            return bool(result and result.records)
        except Exception as e:
            logger.error(f"Failed to unshare session: {e}")
            raise

    async def list_shared_sessions(self, limit: int = 50, offset: int = 0) -> List[ConversationSession]:
        """
        List all sessions that have been shared (e.g. for admin review).
        """
        try:
            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")
            
            # Find sessions marked as shared
            # Ideally filter by shared_with_role = 'admin' if multiple roles exist
            query = (
                "MATCH (s:ConversationSession)\n"
                "WHERE s.is_shared = true\n"
                "  AND (s.deleted_at IS NULL OR s.deleted_at = '')\n"
                "OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)\n"
                "WHERE m.deleted_at IS NULL OR m.deleted_at = ''\n"
                "WITH s, count(m) as msg_count, collect(m)[0] as first_msg\n"
                "RETURN s.session_id as session_id,\n"
                "       s.created_at as created_at,\n"
                "       s.updated_at as updated_at,\n"
                "       s.deleted_at as deleted_at,\n"
                "       s.title as title,\n"
                "       msg_count as message_count,\n"
                "       CASE WHEN s.summary <> '' AND s.summary IS NOT NULL THEN s.summary ELSE first_msg.content END as preview,\n"
                "       s.user_id as user_id\n"
                "ORDER BY s.updated_at DESC\n"
                "SKIP $offset LIMIT $limit"
            )
            
            result = driver.execute_query(query, limit=limit, offset=offset)

            sessions = []
            if result and result.records:
                for record in result.records:
                    preview = record["preview"] or ""
                    if preview:
                        preview = strip_markdown(preview)
                        if len(preview) > 100:
                            preview = preview[:100] + "..."

                    def _normalize_ts(value):
                        try:
                            if isinstance(value, (int, float)):
                                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).astimezone().isoformat()
                            if isinstance(value, str) and value:
                                return value
                        except Exception:
                            pass
                        return ""

                    # We could extend ConversationSession to include user_id info 
                    # but for now we re-use the base model
                    sessions.append(
                        ConversationSession(
                            session_id=record["session_id"],
                            created_at=_normalize_ts(record.get("created_at")),
                            updated_at=_normalize_ts(record.get("updated_at")),
                            message_count=record["message_count"],
                            preview=preview,
                            deleted_at=_normalize_ts(record.get("deleted_at")),
                            title=record.get("title") or f"Shared by {record.get('user_id', 'Unknown')}"
                        )
                    )
            return sessions

        except Exception as e:
            logger.error(f"Failed to list shared sessions: {e}")
            raise

# Global service instance
chat_history_service = ChatHistoryService()

