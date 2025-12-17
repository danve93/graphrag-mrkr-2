import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ingestion.document_processor import DocumentProcessor
from core.entity_extraction import EntityExtractor

@pytest.mark.asyncio
async def test_process_file_async_cancellation():
    """Test that process_file_async stops when cancelled."""
    processor = DocumentProcessor()
    
    # Create dummy chunks
    chunks = [{"chunk_id": f"id_{i}", "content": f"text {i}", "metadata": {"chunk_index": i}} for i in range(20)]
    
    # Mock dependencies
    with patch("ingestion.document_processor.embedding_manager") as mock_embed, \
         patch("ingestion.document_processor.graph_db") as mock_db:
        
        # Make embedding slow enough to trigger cancellation check
        async def slow_embed(*args, **kwargs):
            await asyncio.sleep(0.05)
            return [0.1] * 1536
        
        mock_embed.aget_embedding = AsyncMock(side_effect=slow_embed)
        mock_db.create_chunk_node = MagicMock()
        
        # Setup progress callback that cancels after a few chunks
        chunks_processed = 0
        should_cancel = False
        
        def progress_callback(processed):
            nonlocal chunks_processed, should_cancel
            chunks_processed = processed
            if processed >= 3:
                should_cancel = True
        
        progress_callback.is_cancelled = lambda: should_cancel
        
        # Run and expect cancellation
        with pytest.raises(asyncio.CancelledError):
            await processor.process_file_async(
                chunks,
                "test_doc_id",
                progress_callback
            )
            
        # Should have stopped early
        assert chunks_processed < 20
        assert chunks_processed >= 3

@pytest.mark.asyncio
async def test_extract_from_chunks_cancellation():
    """Test that entity extraction stops when cancelled."""
    extractor = EntityExtractor()
    chunks = [{"chunk_id": f"id_{i}", "content": f"text {i}"} for i in range(20)]
    
    # Mock extract_from_chunk to be slow
    extractor.extract_from_chunk = AsyncMock()
    async def slow_extract(*args):
        await asyncio.sleep(0.05)
        return [], []
    extractor.extract_from_chunk.side_effect = slow_extract
    
    # Setup cancellation
    should_cancel = False
    
    def is_cancelled():
        return should_cancel
        
    # Start the task
    task = asyncio.create_task(extractor.extract_from_chunks(chunks, is_cancelled=is_cancelled))
    
    # Let it run a bit then cancel
    await asyncio.sleep(0.1)
    should_cancel = True
    
    # Expect CancelledError
    with pytest.raises(asyncio.CancelledError):
        await task
        
@pytest.mark.asyncio
async def test_extract_from_chunks_with_gleaning_cancellation():
    """Test that gleaning extraction stops when cancelled."""
    extractor = EntityExtractor()
    chunks = [{"chunk_id": f"id_{i}", "content": f"text {i}"} for i in range(5)]
    
    # Mock inner gleaning method
    extractor.extract_from_chunk_with_gleaning = AsyncMock()
    async def slow_extract(*args, **kwargs):
        await asyncio.sleep(0.05)
        return [], []
    extractor.extract_from_chunk_with_gleaning.side_effect = slow_extract
    
    # Setup cancellation
    should_cancel = False
    
    def is_cancelled():
        return should_cancel
        
    task = asyncio.create_task(
        extractor.extract_from_chunks_with_gleaning(
            chunks, 
            max_gleanings=1, 
            is_cancelled=is_cancelled
        )
    )
    
    await asyncio.sleep(0.1)
    should_cancel = True
    
    with pytest.raises(asyncio.CancelledError):
        await task
