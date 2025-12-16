"""
Unit tests for layered memory system.

Tests cover:
- User node creation and management
- Fact creation and management
- Conversation summary creation
- Memory context loading
- Token savings estimation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from core.conversation_memory import ConversationMemoryManager, MemoryContext


@pytest.fixture
def memory_manager():
    """Create a fresh memory manager for testing."""
    return ConversationMemoryManager()


@pytest.fixture
def mock_graph_db():
    """Mock graph database."""
    with patch('core.conversation_memory.graph_db') as mock_db:
        # Mock user creation
        mock_db.get_user.return_value = None
        mock_db.create_user.return_value = {
            "user_id": "test_user",
            "created_at": datetime.utcnow(),
            "metadata": {},
        }

        # Mock facts
        mock_db.get_user_facts.return_value = [
            {
                "fact_id": "fact1",
                "content": "User prefers Python",
                "importance": 0.8,
                "created_at": datetime.utcnow(),
                "metadata": {},
            },
            {
                "fact_id": "fact2",
                "content": "User works on GraphRAG",
                "importance": 0.9,
                "created_at": datetime.utcnow(),
                "metadata": {},
            },
        ]

        # Mock conversations
        mock_db.get_user_conversations.return_value = [
            {
                "conversation_id": "conv1",
                "user_id": "test_user",
                "title": "Discussion about RAG",
                "summary": "Discussed RAG implementation details",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "metadata": {},
            }
        ]

        yield mock_db


class TestMemoryContext:
    """Tests for MemoryContext dataclass."""

    def test_memory_context_creation(self):
        """Test creating a MemoryContext."""
        context = MemoryContext(
            user_id="test_user",
            session_id="test_session",
        )

        assert context.user_id == "test_user"
        assert context.session_id == "test_session"
        assert context.user_facts == []
        assert context.conversation_summaries == []
        assert context.current_messages == []

    def test_memory_context_with_data(self):
        """Test MemoryContext with pre-populated data."""
        facts = [{"content": "Test fact", "importance": 0.5}]
        conversations = [{"title": "Test conv", "summary": "Summary"}]

        context = MemoryContext(
            user_id="test_user",
            session_id="test_session",
            user_facts=facts,
            conversation_summaries=conversations,
        )

        assert len(context.user_facts) == 1
        assert len(context.conversation_summaries) == 1


class TestMemoryManager:
    """Tests for ConversationMemoryManager."""

    def test_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert memory_manager.db is not None
        assert memory_manager.active_sessions == {}

    @patch('core.conversation_memory.graph_db')
    def test_load_memory_context_new_user(self, mock_db, memory_manager):
        """Test loading memory context for a new user."""
        # Mock database responses
        mock_db.get_user.return_value = None
        mock_db.create_user.return_value = {"user_id": "new_user"}
        mock_db.get_user_facts.return_value = []
        mock_db.get_user_conversations.return_value = []

        context = memory_manager.load_memory_context(
            user_id="new_user",
            session_id="session1",
        )

        assert context.user_id == "new_user"
        assert context.session_id == "session1"
        assert "session1" in memory_manager.active_sessions
        mock_db.create_user.assert_called_once()

    def test_load_memory_context_with_data(self, memory_manager, mock_graph_db):
        """Test loading memory context with existing data."""
        # Set up existing user
        mock_graph_db.get_user.return_value = {"user_id": "test_user"}

        context = memory_manager.load_memory_context(
            user_id="test_user",
            session_id="session1",
        )

        assert context.user_id == "test_user"
        assert len(context.user_facts) == 2
        assert len(context.conversation_summaries) == 1
        assert context.user_facts[0]["importance"] == 0.8

    def test_get_session_context(self, memory_manager):
        """Test retrieving cached session context."""
        # Create a context
        context = MemoryContext(
            user_id="test_user",
            session_id="session1",
        )
        memory_manager.active_sessions["session1"] = context

        # Retrieve it
        retrieved = memory_manager.get_session_context("session1")
        assert retrieved is not None
        assert retrieved.user_id == "test_user"

        # Try non-existent session
        none_context = memory_manager.get_session_context("nonexistent")
        assert none_context is None

    def test_add_message_to_session(self, memory_manager):
        """Test adding messages to a session."""
        # Create a session
        context = MemoryContext(
            user_id="test_user",
            session_id="session1",
        )
        memory_manager.active_sessions["session1"] = context

        # Add messages
        memory_manager.add_message_to_session(
            session_id="session1",
            role="user",
            content="What is GraphRAG?",
        )

        memory_manager.add_message_to_session(
            session_id="session1",
            role="assistant",
            content="GraphRAG combines graph databases with RAG...",
        )

        assert len(context.current_messages) == 2
        assert context.current_messages[0]["role"] == "user"
        assert context.current_messages[1]["role"] == "assistant"

    def test_build_context_prompt(self, memory_manager):
        """Test building context prompt from memory layers."""
        # Create a session with data
        context = MemoryContext(
            user_id="test_user",
            session_id="session1",
            user_facts=[
                {"content": "User prefers Python", "importance": 0.8},
                {"content": "User works on RAG", "importance": 0.9},
            ],
            conversation_summaries=[
                {"title": "Previous chat", "summary": "Discussed databases"}
            ],
            current_messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        )
        memory_manager.active_sessions["session1"] = context

        # Build prompt
        prompt = memory_manager.build_context_prompt("session1")

        assert prompt != ""
        assert "User Preferences & Facts" in prompt
        assert "Python" in prompt
        assert "Recent Conversation Context" in prompt
        assert "Recent Messages" in prompt

    def test_build_context_prompt_empty_session(self, memory_manager):
        """Test building context prompt for empty session."""
        context = MemoryContext(
            user_id="test_user",
            session_id="session1",
        )
        memory_manager.active_sessions["session1"] = context

        prompt = memory_manager.build_context_prompt("session1")
        assert prompt == ""

    def test_get_compact_context(self, memory_manager):
        """Test getting compact context data."""
        context = MemoryContext(
            user_id="test_user",
            session_id="session1",
            user_facts=[{"content": "Fact", "importance": 0.5}] * 15,  # 15 facts
            conversation_summaries=[{"title": "Conv", "summary": "Sum"}] * 5,
            current_messages=[{"role": "user", "content": "msg"}] * 10,
        )
        memory_manager.active_sessions["session1"] = context

        compact = memory_manager.get_compact_context("session1")

        assert compact["user_id"] == "test_user"
        assert len(compact["top_facts"]) == 10  # Limited to 10
        assert len(compact["recent_conversations"]) == 3  # Limited to 3
        assert compact["message_count"] == 10

    @patch('core.conversation_memory.graph_db')
    def test_add_user_fact(self, mock_db, memory_manager):
        """Test adding a user fact."""
        mock_db.create_fact.return_value = {
            "fact_id": "fact1",
            "content": "User likes testing",
            "importance": 0.7,
            "created_at": datetime.utcnow(),
            "metadata": {},
        }
        mock_db.get_user_facts.return_value = [mock_db.create_fact.return_value]

        # Create a session first
        context = MemoryContext(user_id="test_user", session_id="session1")
        memory_manager.active_sessions["session1"] = context

        fact = memory_manager.add_user_fact(
            user_id="test_user",
            fact_id="fact1",
            content="User likes testing",
            importance=0.7,
        )

        assert fact["content"] == "User likes testing"
        assert fact["importance"] == 0.7
        # Session should be refreshed
        assert len(context.user_facts) == 1

    @patch('core.conversation_memory.graph_db')
    def test_update_user_fact(self, mock_db, memory_manager):
        """Test updating a user fact."""
        mock_db.update_fact.return_value = {
            "fact_id": "fact1",
            "content": "Updated content",
            "importance": 0.9,
            "created_at": datetime.utcnow(),
            "metadata": {},
        }
        mock_db.get_user_facts.return_value = [mock_db.update_fact.return_value]

        context = MemoryContext(user_id="test_user", session_id="session1")
        memory_manager.active_sessions["session1"] = context

        fact = memory_manager.update_user_fact(
            user_id="test_user",
            fact_id="fact1",
            content="Updated content",
            importance=0.9,
        )

        assert fact["importance"] == 0.9
        assert len(context.user_facts) == 1

    @patch('core.conversation_memory.graph_db')
    def test_delete_user_fact(self, mock_db, memory_manager):
        """Test deleting a user fact."""
        mock_db.delete_fact.return_value = True
        mock_db.get_user_facts.return_value = []

        context = MemoryContext(user_id="test_user", session_id="session1")
        memory_manager.active_sessions["session1"] = context

        deleted = memory_manager.delete_user_fact(
            user_id="test_user",
            fact_id="fact1",
        )

        assert deleted is True
        assert len(context.user_facts) == 0

    @patch('core.conversation_memory.graph_db')
    def test_save_conversation_summary(self, mock_db, memory_manager):
        """Test saving a conversation summary."""
        mock_db.create_conversation.return_value = {
            "conversation_id": "conv1",
            "user_id": "test_user",
            "title": "Test Conversation",
            "summary": "This was a test",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "metadata": {},
        }
        mock_db.get_user_conversations.return_value = [mock_db.create_conversation.return_value]

        context = MemoryContext(user_id="test_user", session_id="session1")
        memory_manager.active_sessions["session1"] = context

        conversation = memory_manager.save_conversation_summary(
            user_id="test_user",
            conversation_id="conv1",
            title="Test Conversation",
            summary="This was a test",
        )

        assert conversation["title"] == "Test Conversation"
        assert len(context.conversation_summaries) == 1

    @patch('core.conversation_memory.graph_db')
    @patch('core.conversation_summarizer.generate_summary')
    @patch('core.conversation_summarizer.generate_title')
    def test_end_session_with_save(self, mock_title, mock_summary, mock_db, memory_manager):
        """Test ending a session and saving summary."""
        mock_title.return_value = "Auto-generated Title"
        mock_summary.return_value = "Auto-generated summary"
        mock_db.create_conversation.return_value = {}

        context = MemoryContext(user_id="test_user", session_id="session1")
        context.current_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        memory_manager.active_sessions["session1"] = context

        memory_manager.end_session(
            session_id="session1",
            save_summary=True,
        )

        # Session should be removed
        assert "session1" not in memory_manager.active_sessions
        # Summary should have been generated and saved
        mock_summary.assert_called_once()
        mock_title.assert_called_once()
        mock_db.create_conversation.assert_called_once()

    def test_end_session_without_save(self, memory_manager):
        """Test ending a session without saving."""
        context = MemoryContext(user_id="test_user", session_id="session1")
        memory_manager.active_sessions["session1"] = context

        memory_manager.end_session(
            session_id="session1",
            save_summary=False,
        )

        assert "session1" not in memory_manager.active_sessions

    def test_estimate_token_savings(self, memory_manager):
        """Test estimating token savings."""
        # Create a session with messages
        context = MemoryContext(user_id="test_user", session_id="session1")
        context.current_messages = [
            {"role": "user", "content": "This is a long message " * 50},
            {"role": "assistant", "content": "This is another long message " * 50},
            {"role": "user", "content": "And another one " * 30},
        ]
        memory_manager.active_sessions["session1"] = context

        savings = memory_manager.estimate_token_savings("session1")

        assert "baseline_tokens" in savings
        assert "optimized_tokens" in savings
        assert "tokens_saved" in savings
        assert "savings_percent" in savings
        assert savings["baseline_tokens"] > 0

    def test_estimate_token_savings_nonexistent_session(self, memory_manager):
        """Test token savings estimation for non-existent session."""
        savings = memory_manager.estimate_token_savings("nonexistent")
        assert "error" in savings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
