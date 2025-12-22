import sys
from unittest.mock import MagicMock, patch

# Mock neo4j
sys.modules["neo4j"] = MagicMock()
sys.modules["neo4j.exceptions"] = MagicMock()
sys.modules["config"] = MagicMock()
sys.modules["config.settings"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["httpx"] = MagicMock()
# Mock core.embeddings to avoid importing it (and its deps)
mock_embeddings_module = MagicMock()
sys.modules["core.embeddings"] = mock_embeddings_module
sys.modules["cachetools"] = MagicMock()
sys.modules["core.singletons"] = MagicMock()


import pytest
from core.graph_db import GraphDB

@pytest.fixture
def mock_graph_db():
    with patch("core.graph_db.get_graph_db_driver") as mock_driver_factory:
        mock_driver = MagicMock()
        mock_driver_factory.return_value = mock_driver
        
        # Create instance
        # We need to bypass __init__ logic or mock it because it tries to connect
        with patch("core.graph_db.GraphDB.__init__", return_value=None):
            db = GraphDB()
            db.driver = mock_driver
            db._entity_label_cache = {} # Mock cache
            
            # Mock session_scope
            @patch("contextlib.contextmanager")
            def session_scope_mock(max_attempts=3, initial_backoff=0.5):
                session = MagicMock()
                yield session
            
            db.session_scope = MagicMock()
            # We need session_scope to be a context manager that yields a mock session
            mock_session = MagicMock()
            db.session_scope.return_value.__enter__.return_value = mock_session
            
            yield db, mock_session

@pytest.fixture
def mock_embedding():
    with patch("core.graph_db.embedding_manager") as mock_em:
        mock_em.get_embedding.return_value = [0.1, 0.2, 0.3]
        yield mock_em

def test_update_chunk_content(mock_graph_db, mock_embedding):
    db, session = mock_graph_db
    
    # Setup mock return
    session.run.return_value.single.return_value = {
        "chunk_id": "c1", "content": "new text", "embedding": [0.1, 0.2, 0.3],
        "document_id": "doc1",
        "chunk_index": 0,
        "id": "c1"
    }
    
    # Call
    result = db.update_chunk_content("c1", "new text")
    
    # Verify
    assert result["id"] == "c1"
    assert result["content"] == "new text"
    
    # Check embedding generation called
    mock_embedding.get_embedding.assert_called_with("new text")
    
    # Check cypher execution (find the update query)
    found_update = False
    for call in session.run.call_args_list:
        query = call[0][0]
        params = call[1]
        
        if "MATCH (c:Chunk {id: $chunk_id})" in query and "SET c.content = $content" in query:
             found_update = True
             assert params["chunk_id"] == "c1"
             assert params["content"] == "new text"
             break
             
    assert found_update, "Update query not found in session.run calls"

def test_delete_chunk(mock_graph_db):
    db, session = mock_graph_db
    
    # Setup return
    # First call (exists check): returns non-None
    # Last call (orphan count): returns dict with orphan_count
    
    # We use side_effect to return different values for different calls if needed, 
    # or just a mock that behaves like a result object
    mock_result = MagicMock()
    mock_result.single.return_value = {"orphan_count": 0}
    # For the exists check (which calls single()), it should return non-None (truthy)
    
    # Simpler approach: configure session.run to return a mock that handles both
    session.run.return_value.single.return_value = {"orphan_count": 1, "id": "c1"} 
    
    # Call
    result = db.delete_chunk("c1")
    
    # Verify
    assert result is True
    
    # Check cypher was called (any of the calls)
    found_delete = False
    for call in session.run.call_args_list:
        query = call[0][0]
        if "MATCH (c:Chunk {id: $chunk_id})" in query and "DETACH DELETE c" in query:
            found_delete = True
            break
            
    assert found_delete, "Delete query not found in session.run calls"

def test_restore_chunk(mock_graph_db):
    db, session = mock_graph_db
    
    # Call
    result = db.restore_chunk("c1", "doc1", "original content", 0)
    
    # Verify
    assert result is True
    
    # Check MERGE query
    found_restore = False
    for call in session.run.call_args_list:
        query = call[0][0]
        params = call[1]
        
        if "MERGE (c:Chunk {id: $chunk_id})" in query:
            found_restore = True
            assert params["chunk_id"] == "c1"
            assert params["content"] == "original content"
            assert params["chunk_index"] == 0
            break
            
    assert found_restore, "Restore query not found"

def test_unmerge_chunks(mock_graph_db):
    # This largely wraps delete_chunk, but we test the interface
    db, session = mock_graph_db
    
    # Setup mock return for deleted/orphan check
    session.run.return_value.single.return_value = {"orphan_count": 0, "id": "merged_c1"}
    
    # Call
    result = db.unmerge_chunks("merged_c1")
    
    # Verify
    assert result is True
    
    # Check delete call
    found_delete = False
    for call in session.run.call_args_list:
        query = call[0][0]
        if "MATCH (c:Chunk {id: $chunk_id})" in query and "DETACH DELETE c" in query:
             found_delete = True
             break
    assert found_delete, "Unmerge (delete) query not found"
