import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from functools import partial
from ingestion.document_processor import DocumentProcessor, EntityExtractionState

@pytest.mark.asyncio
async def test_processing_status_flow():
    """Test that all processing steps report their status correctly."""
    processor = DocumentProcessor()
    
    # Mock mocks on instance
    processor.converter = MagicMock()
    processor.converter.convert.return_value = {"content": "text content"}
    
    # Mock chunks return
    chunks = [{"chunk_id": "1", "content": "text", "metadata": {}}]
    
    # Track status updates
    status_updates = []
    
    def progress_callback(count, message=None):
        print(f"CALLBACK: count={count}, message={message}")
        if message:
            status_updates.append(f"MESSAGE: {message}")
            
    # Mock dependencies
    with patch("ingestion.document_processor.embedding_manager") as mock_embed, \
         patch("ingestion.document_processor.graph_db") as mock_db, \
         patch("ingestion.document_processor.document_summarizer") as mock_sum, \
         patch("ingestion.document_processor.document_chunker") as mock_chunker, \
         patch("ingestion.document_processor.EntityExtractor") as MockExtractor, \
         patch("core.graph_clustering.run_auto_clustering") as mock_clustering, \
         patch("ingestion.document_processor.settings") as mock_settings, \
         patch("pathlib.Path.exists", return_value=True):
         
        # Setup settings
        mock_settings.embedding_concurrency = 1
        
        # Setup mocks
        mock_embed.aget_embedding = AsyncMock(return_value=[0.1]*1536)
        mock_db.create_chunk_node = MagicMock()
        mock_sum.extract_summary = MagicMock(return_value={"summary": "sum", "document_type": "type"})
        mock_chunker.chunk_text.return_value = chunks
        
        
        # Mock file path object
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.name = "dummy.txt"
        mock_path.suffix = ".txt"
        mock_path.__str__.return_value = "dummy.txt"
        stat_mock = MagicMock()
        stat_mock.st_mtime = 123456.0
        stat_mock.st_size = 100
        stat_mock.st_ctime = 123456.0
        mock_path.stat.return_value = stat_mock
        
        # 1. Test Summarization Reporting in process_file_chunks_only
        # We need to run it in executor because it calls asyncio.run internally
        loop = asyncio.get_running_loop()
        
        await loop.run_in_executor(
            None, 
            partial(
                processor.process_file_chunks_only, 
                mock_path, 
                "dummy.txt", 
                progress_callback
            )
        )
        
        # Verify Summarization message
        assert "MESSAGE: Generating abstract" in status_updates
        
        # 2. Test Clustering Reporting
        entity_updates = []
        def entity_callback(state, fraction, info=None):
            entity_updates.append((state, info))
            
        mock_extractor_instance = MockExtractor.return_value
        mock_extractor_instance.extract_from_chunks = AsyncMock(return_value=({}, []))
        mock_clustering.return_value = {"status": "success"}

        await loop.run_in_executor(
            None,
            partial(
                processor.extract_entities_for_document,
                "doc_id",
                "file.txt",
                entity_callback
            )
        )
             
        # Verify CLUSTERING state
        clustering_calls = [u for u in entity_updates if u[0] == EntityExtractionState.CLUSTERING]
        assert len(clustering_calls) > 0
        assert clustering_calls[0][1] == "Detecting communities"
