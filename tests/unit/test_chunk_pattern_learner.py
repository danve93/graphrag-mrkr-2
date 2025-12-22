import sys
from unittest.mock import Mock, MagicMock, patch

# Mock neo4j before checking imports
sys.modules["neo4j"] = MagicMock()
sys.modules["neo4j.exceptions"] = MagicMock()

import pytest
from core.chunk_pattern_learner import ChunkPatternLearner
from core.chunk_pattern_store import ChunkPattern, ChunkPatternStore

@pytest.fixture
def mock_store():
    store = MagicMock(spec=ChunkPatternStore)
    return store

@pytest.fixture
def learner(mock_store):
    with patch('core.chunk_pattern_learner.get_pattern_store', return_value=mock_store):
        learner = ChunkPatternLearner()
        # Mocking _load_chunks to avoid DB dependency
        learner._load_chunks = Mock()
        yield learner

def test_has_suggestions_returns_true_on_match(learner, mock_store):
    """Test that has_suggestions returns True immediately when a pattern matches."""
    # Setup pattern
    pattern = ChunkPattern(
        id="p1", 
        name="Test Pattern", 
        description="Test Pattern Description",
        match_type="regex", 
        match_criteria={"pattern": "error"}, 
        action="flag",
        confidence=0.8,
        enabled=True
    )
    mock_store.get_patterns.return_value = [pattern]
    
    # Setup chunks
    learner._load_chunks.return_value = [
        {"id": "c1", "content": "This text contains an error.", "token_count": 5}
    ]
    
    # Act
    result = learner.has_suggestions("doc1")
    
    # Assert
    assert result is True
    # Should stop after finding first match, so get_patterns called once
    mock_store.get_patterns.assert_called_with(enabled_only=True)

def test_has_suggestions_returns_false_when_no_match(learner, mock_store):
    """Test that has_suggestions returns False when no patterns match."""
    # Setup pattern
    pattern = ChunkPattern(
        id="p1", 
        name="Test Pattern", 
        description="Test Pattern Description",
        match_type="regex", 
        match_criteria={"pattern": "error"}, 
        action="flag",
        confidence=0.8,
        enabled=True
    )
    mock_store.get_patterns.return_value = [pattern]
    
    # Setup chunks (no match)
    learner._load_chunks.return_value = [
        {"id": "c1", "content": "This text is clean.", "token_count": 5}
    ]
    
    # Act
    result = learner.has_suggestions("doc1")
    
    # Assert
    assert result is False

def test_has_suggestions_returns_false_empty_chunks(learner, mock_store):
    """Test that has_suggestions returns False if no chunks loaded."""
    learner._load_chunks.return_value = []
    
    # Act
    result = learner.has_suggestions("doc1")
    
    # Assert
    assert result is False
