"""Unit tests for BM25/fulltext keyword search functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.graph_db import GraphDB


def test_chunk_keyword_search_basic():
    """Test basic keyword search returns chunks with scores."""
    # Mock session and result
    mock_record_1 = {
        "chunk_id": "chunk1",
        "content": "This is about machine learning",
        "chunk_index": 0,
        "document_id": "doc1",
        "document_name": "ML Guide",
        "filename": "ml_guide.pdf",
        "keyword_score": 2.5,
    }
    mock_record_2 = {
        "chunk_id": "chunk2",
        "content": "Deep learning fundamentals",
        "chunk_index": 1,
        "document_id": "doc1",
        "document_name": "ML Guide",
        "filename": "ml_guide.pdf",
        "keyword_score": 1.8,
    }
    
    # Use MagicMock with __getitem__ support
    mock_rec1 = MagicMock()
    mock_rec1.__getitem__.side_effect = lambda k: mock_record_1[k]
    mock_rec2 = MagicMock()
    mock_rec2.__getitem__.side_effect = lambda k: mock_record_2[k]
    
    mock_result = [mock_rec1, mock_rec2]
    mock_session = Mock()
    mock_session.run.return_value = mock_result
    
    graph_db = GraphDB()
    
    with patch.object(graph_db, 'session_scope') as mock_scope:
        mock_scope.return_value.__enter__.return_value = mock_session
        
        chunks = graph_db.chunk_keyword_search("machine learning", top_k=10)
        
        assert len(chunks) == 2
        assert chunks[0]["chunk_id"] == "chunk1"
        assert chunks[0]["keyword_score"] == 2.5
        assert chunks[1]["chunk_id"] == "chunk2"
        assert chunks[1]["keyword_score"] == 1.8


def test_chunk_keyword_search_with_document_filter():
    """Test keyword search respects document ID filtering."""
    mock_record = {
        "chunk_id": "chunk1",
        "content": "Filtered content",
        "chunk_index": 0,
        "document_id": "doc1",
        "document_name": "Doc 1",
        "filename": "doc1.pdf",
        "keyword_score": 3.0,
    }
    
    mock_rec = MagicMock()
    mock_rec.__getitem__.side_effect = lambda k: mock_record[k]
    mock_result = [mock_rec]
    mock_session = Mock()
    mock_session.run.return_value = mock_result
    
    graph_db = GraphDB()
    
    with patch.object(graph_db, 'session_scope') as mock_scope:
        mock_scope.return_value.__enter__.return_value = mock_session
        
        chunks = graph_db.chunk_keyword_search(
            "content",
            top_k=10,
            allowed_document_ids=["doc1", "doc2"]
        )
        
        # Verify the query was called with allowed_doc_ids parameter
        call_args = mock_session.run.call_args
        assert "allowed_doc_ids" in call_args[1]
        assert call_args[1]["allowed_doc_ids"] == ["doc1", "doc2"]
        
        assert len(chunks) == 1
        assert chunks[0]["document_id"] == "doc1"


def test_chunk_keyword_search_error_handling():
    """Test keyword search handles errors gracefully."""
    graph_db = GraphDB()
    
    with patch.object(graph_db, 'session_scope') as mock_scope:
        mock_scope.side_effect = Exception("Database error")
        
        # Should return empty list on error, not raise
        chunks = graph_db.chunk_keyword_search("query", top_k=10)
        
        assert chunks == []


def test_setup_indexes_creates_fulltext_index():
    """Test that setup_indexes creates the fulltext index on chunks."""
    mock_session = Mock()
    graph_db = GraphDB()
    
    with patch.object(graph_db, 'session_scope') as mock_scope:
        mock_scope.return_value.__enter__.return_value = mock_session
        
        graph_db.setup_indexes()
        
        # Check that fulltext index creation was attempted
        calls = [str(call) for call in mock_session.run.call_args_list]
        fulltext_calls = [c for c in calls if 'FULLTEXT INDEX chunk_content_fulltext' in c]
        
        assert len(fulltext_calls) > 0, "Fulltext index creation should be called"


@pytest.mark.asyncio
async def test_keyword_search_integration_with_retriever():
    """Test that keyword search integrates into hybrid retrieval."""
    from rag.retriever import DocumentRetriever
    from config.settings import settings
    
    # Enable keyword search
    original_flag = settings.enable_chunk_fulltext
    settings.enable_chunk_fulltext = True
    
    try:
        retriever = DocumentRetriever()
        
        # Mock keyword search results
        mock_keyword_results = [
            {
                "chunk_id": "kw_chunk1",
                "content": "Keyword match content",
                "document_id": "doc1",
                "document_name": "Doc 1",
                "filename": "doc1.pdf",
                "keyword_score": 2.0,
                "chunk_index": 0,
            }
        ]
        
        # Mock vector search to return empty
        with patch.object(retriever, 'chunk_based_retrieval', return_value=[]):
            with patch.object(retriever, 'entity_based_retrieval', return_value=[]):
                with patch('core.graph_db.graph_db.chunk_keyword_search', return_value=mock_keyword_results):
                    results = await retriever._hybrid_retrieval_direct(
                        query="test query",
                        top_k=5,
                        chunk_weight=0.4,
                        entity_weight=0.3,
                    )
                    
                    # Should have keyword results in the output
                    assert len(results) > 0
                    # Check that keyword source is present
                    keyword_sources = [r for r in results if "keyword" in r.get("retrieval_source", "")]
                    assert len(keyword_sources) > 0
    
    finally:
        settings.enable_chunk_fulltext = original_flag
