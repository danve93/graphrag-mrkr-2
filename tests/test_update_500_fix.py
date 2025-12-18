import os
import sys
import asyncio
import pytest
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.mark.asyncio
async def test_update_document_async_safety():
    """
    Verify that update_document can be called in an async environment
    and initiates background work correctly.
    """
    # Mock dependencies BEFORE instantiating DocumentProcessor
    with patch("ingestion.document_processor.graph_db") as mock_db, \
         patch("ingestion.document_processor.document_chunker") as mock_chunker, \
         patch("ingestion.document_processor.settings") as mock_settings, \
         patch("ingestion.document_processor.EntityExtractor"):
        
        # Setup mocks
        mock_settings.chunk_size = 500
        mock_settings.chunk_overlap = 100
        mock_settings.enable_entity_extraction = True
        mock_settings.cache_type = "memory"
        
        mock_db.get_document_chunking_params.return_value = {
            "chunk_size_used": 500,
            "chunk_overlap_used": 100
        }
        mock_db.get_chunk_hashes_for_document.return_value = {"old_hash": "chunk1"}
        
        from ingestion.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        
        # Mocking converter
        processor.converter = MagicMock()
        processor.converter.convert.return_value = {
            "content": "new content",
            "metadata": {}
        }
        
        # Mocking chunker
        mock_chunker.chunk_text.return_value = [
            {"content": "new content", "metadata": {"content_hash": "new_hash"}}
        ]
        
        # Mocking background workers to avoid actual async.run/threading
        processor._background_full_update_worker_impl = MagicMock()
        
        # Test document info
        doc_id = "test_doc_123"
        file_path = Path("dummy.txt")
        # Create dummy file
        file_path.write_text("dummy")
        
        try:
            # Call the update method
            result = processor.update_document(
                doc_id=doc_id,
                file_path=file_path
            )
            
            # Verify immediate response
            assert result["status"] == "processing"
            
            # Verify background worker was launched
            assert processor._background_full_update_worker_impl.called
            
            # Check arguments passed to background worker
            args, kwargs = processor._background_full_update_worker_impl.call_args
            assert args[0] == doc_id
            bg_safe_path_str = str(args[1])
            assert "bg_full_upd_test_doc_123" in bg_safe_path_str
            assert bg_safe_path_str.endswith(".txt") # Original extension preserved
            
        finally:
            if file_path.exists():
                file_path.unlink()

if __name__ == "__main__":
    asyncio.run(test_update_document_async_safety())
