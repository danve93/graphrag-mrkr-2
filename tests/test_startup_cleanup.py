
import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock dependencies
mock_graph_db = MagicMock()
mock_settings = MagicMock()
mock_settings.enable_entity_extraction = True

with patch.dict(sys.modules, {
    "core.graph_db": MagicMock(graph_db=mock_graph_db),
    "config.settings": MagicMock(settings=mock_settings),
    "core.chunking": MagicMock(),
    "core.document_summarizer": MagicMock(),
    "core.embeddings": MagicMock(),
    "core.entity_extraction": MagicMock(),
    "core.singletons": MagicMock(SHUTTING_DOWN=False),
    "ingestion.converters": MagicMock(),
    "ingestion.content_filters": MagicMock(),
    "core.entity_graph": MagicMock(),
}):
    from ingestion.document_processor import DocumentProcessor

class TestStartupCleanup(unittest.TestCase):
    def test_cleanup_stale_jobs(self):
        # Setup
        mock_graph_db.get_documents_by_status.return_value = [
            {"id": "doc1", "processing_status": "processing"},
            {"id": "doc2", "processing_status": "processing"}
        ]
        
        # Instantiate processor (should trigger cleanup in __init__)
        processor = DocumentProcessor()
        
        # Verify
        mock_graph_db.get_documents_by_status.assert_called_with("processing")
        self.assertEqual(mock_graph_db.create_document_node.call_count, 2)
        
        # Check first call
        args, kwargs = mock_graph_db.create_document_node.call_args_list[0]
        self.assertEqual(args[0], "doc1")
        self.assertEqual(args[1]["processing_status"], "failed")
        self.assertIn("interrupted", args[1]["error"])
        
        print("SUCCESS: Startup cleanup correctly marked stale jobs as failed.")

if __name__ == "__main__":
    unittest.main()
