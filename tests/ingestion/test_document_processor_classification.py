
import pytest
from unittest.mock import MagicMock, patch, ANY
import time
from pathlib import Path

# Import the class to test
from ingestion.document_processor import DocumentProcessor
from config.settings import settings

class TestDocumentProcessorClassification:

    @pytest.fixture
    def mock_graph_db(self):
        with patch("ingestion.document_processor.graph_db") as mock_db:
            yield mock_db

    @pytest.fixture
    def mock_chunker(self):
        with patch("ingestion.document_processor.document_chunker") as mock_chunk:
            # Setup chunk_text to return dummy chunks
            mock_chunk.chunk_text.return_value = [{"content": "foo", "metadata": {}}]
            yield mock_chunk

    @pytest.fixture
    def mock_summarizer(self):
        with patch("ingestion.document_processor.document_summarizer") as mock_sum:
             mock_sum.extract_summary.return_value = {"summary": "foo", "document_type": "text", "hashtags": []}
             yield mock_sum

    @pytest.fixture
    def processor(self):
        return DocumentProcessor()

    def test_process_file_calls_classification(self, processor, mock_graph_db, mock_chunker, mock_summarizer, tmp_path):
        """
        Verify that the classification stage is executed and status is reported
        when enable_document_classification is True.
        """
        # Create a dummy file
        dummy_file = tmp_path / "test.txt"
        dummy_file.write_text("dummy content")

        # Mock the classification method
        processor.classify_document_categories = MagicMock(return_value={
            "categories": ["tech"],
            "confidence": 0.9,
            "keywords": ["testing"],
            "difficulty": "easy"
        })

        # Mock internal helpers to avoid actual IO/Loading
        processor._generate_document_id = MagicMock(return_value="doc_123")
        processor._extract_metadata = MagicMock(return_value={"filename": "test.txt", "file_extension": ".txt"})
        processor._derive_content_primary_type = MagicMock(return_value="text")
        
        # Mock loaders to bypass actual file reading complexity
        mock_loader = MagicMock()
        mock_loader.load.return_value = "dummy content"
        processor.loaders = {".txt": mock_loader}

        # Mock async process part
        processor.process_file_async = MagicMock()
        async def async_return():
             return []
        processor.process_file_async.return_value = async_return()
        
        # Force setting to True
        with patch.object(settings, "enable_document_classification", True):
            # Run
            processor.process_file(dummy_file, "test.txt")

        # Assertions
        
        # Check if classification status was reported (10%)
        mock_graph_db.create_document_node.assert_any_call(
            "doc_123",
            {"processing_stage": "classification", "processing_progress": 10.0}
        )
        
        # Check if classification method was called
        processor.classify_document_categories.assert_called_once()
        
        # Check if result metadata was saved
        # The easiest way is to check the final call or specific call with metadata
        # We look for a call that includes "categories": ["tech"]
        found_classification_save = False
        for call in mock_graph_db.create_document_node.call_args_list:
            args, _ = call
            if args[0] == "doc_123" and isinstance(args[1], dict):
                if args[1].get("categories") == ["tech"]:
                    found_classification_save = True
                    break
        
        assert found_classification_save, "Did not find graph_db call saving classification results"
        
        # Check conflict with secondary classification (62%)
        # It should NOT be called if we renamed it or if we logic handles it, 
        # but in current code it might be called if we enabled it same way.
        # But wait, we edited the code to use "metadata_enrichment" for the second one?
        # Let's check if "metadata_enrichment" is called at 62%
        
        # Note: The test uses the LOCAL code, so if I haven't applied the rename LOCALLY yet (I reverted previously?), 
        # then it might fail or show "classification" appearing twice.
        # In step 227 I re-applied the logging locally, but did I apply the RENAME locally?
        # I did multi_replace in step 132 on the local file. So the local file SHOULD have the rename.
        
        mock_graph_db.create_document_node.assert_any_call(
             "doc_123",
             {"processing_stage": "metadata_enrichment", "processing_progress": 62.0}
        )

