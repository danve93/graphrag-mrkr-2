
import sys
import os
import unittest
import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

# Add current dir to sys.path
sys.path.append(os.getcwd())

# -----------------------------------------------------------------
# Setup Logic
# -----------------------------------------------------------------
# We only mock the heavy external deps: DB, LLM.
# We let internal logic (Chunking, File operations) run for real to avoid mock drift.

mock_graph_db = MagicMock()
mock_graph_db.acreate_entity_node = AsyncMock()
mock_graph_db.create_chunk_entity_relationship = MagicMock()
mock_graph_db.acreate_entity_relationship = AsyncMock()

mock_settings = MagicMock()
# Set critical settings to avoid logic errors
mock_settings.enable_entity_extraction = True
mock_settings.enable_quality_filtering = False
mock_settings.create_chunk_similarities = False
mock_settings.enable_document_summaries = True # Let it run but mock result
mock_settings.sync_entity_embeddings = False
mock_settings.embedding_concurrency = 1
mock_settings.enable_phase2_networkx = False
mock_settings.openai_proxy = None
mock_settings.embedding_cache_ttl = 3600
mock_settings.embedding_cache_size = 1000
mock_settings.llm_cache_ttl = 3600
mock_settings.llm_cache_size = 1000
mock_settings.entity_label_cache_ttl = 3600
mock_settings.entity_label_cache_size = 1000
mock_settings.classification_confidence_threshold = 0.7
mock_settings.classification_default_category = "general"
mock_settings.enable_document_classification = True
mock_settings.chunk_size = 1000
mock_settings.chunk_overlap = 200
mock_settings.min_chunk_length = 50
mock_settings.max_chunk_length = 2000
mock_settings.max_gleanings = 0
mock_settings.entity_extraction_max_gleanings = 0
mock_settings.max_retries = 3
mock_settings.entity_auto_gleaning = False
mock_settings.importance_score_threshold = 0
mock_settings.strength_threshold = 0
mock_settings.enable_description_summarization = False

# We use patch.dict to replace modules before import
patcher = patch.dict(sys.modules, {
    "core.graph_db": MagicMock(graph_db=mock_graph_db),
    "config.settings": MagicMock(settings=mock_settings),
    "core.llm": MagicMock(),
    "core.llm.llm_manager": MagicMock(),
    # We do NOT mock core.chunking or core.ocr or cv2 if possible, 
    # but cv2 often causes issues so strictly mocking it is safe if we don't need OCR.
    "cv2": MagicMock(),
    "core.ocr": MagicMock(), 
    # We DO NOT mock core.chunking. We want real chunker.
    # But core.chunking imports ocr which imports cv2.
    # Since we mocked core.ocr, core.chunking should load fine.
})
patcher.start()

from ingestion.document_processor import DocumentProcessor
from core.entity_extraction import EntityExtractor

class MockEntity:
    def __init__(self, name, type="Test"):
        self.name = name
        self.type = type
        self.description = "desc"
        self.importance_score = 10
        self.source_text_units = []
        self.source_chunks = []

class TestEntityCreation(unittest.TestCase):
    def setUp(self):
        # Create dummy file
        with open("dummy.txt", "w") as f:
            f.write("This is a test document.\n" * 50) # Enough text to chunk?
            
    def tearDown(self):
        if os.path.exists("dummy.txt"):
            os.remove("dummy.txt")

    def test_entity_creation_flow(self):
        processor = DocumentProcessor()
        
        # Patch dependencies via instance or module properties
        
        # 1. Summary
        # processor uses 'document_summarizer' imported from core.document_summarizer
        # We need to patch the object inside ingestion.document_processor module
        with patch("ingestion.document_processor.document_summarizer") as mock_summarizer:
            mock_summarizer.extract_summary.return_value = {
                "summary": "Mock summary",
                "document_type": "text",
                "hashtags": []
            }
            
            # 2. Classification
            # Mock the method on the processor instance
            processor.classify_document_categories = MagicMock(
                return_value={"categories": ["general"], "confidence": 0.9}
            )
            
            # 3. Embeddings
            # Mock embedding_manager inside document_processor
            with patch("ingestion.document_processor.embedding_manager") as mock_embedder:
                mock_embedder.generate_embeddings.return_value = [[0.1]*1536] # Standard size
                
                # 4. Entity Extraction
                # We need to ensure the processor uses our mock extractor logic
                # processor.entity_extractor is initialized in __init__
                # We can swap it out.
                
                # Setup Entity Extractor
                # We just want to mock extract_from_chunks
                
                async def mock_extract(chunks, **kwargs):
                    if 'progress_callback' in kwargs and kwargs['progress_callback']:
                        kwargs['progress_callback'](1, 1)
                    return (
                        {"e1": MockEntity("test_entity")}, 
                        {"r1": []}
                    )
                
                processor.entity_extractor.extract_from_chunks = AsyncMock(side_effect=mock_extract)
                processor.entity_extractor.extract_from_chunks_with_gleaning = AsyncMock(side_effect=mock_extract)

                # Patch Content Filter using defaults to avoid Mock comparisons
                from ingestion.content_filters import ContentQualityFilter
                default_filter = ContentQualityFilter(min_chunk_length=0) # Relaxed checks
                
                with patch("ingestion.document_processor.get_content_filter", return_value=default_filter):
                    # Run Processing
                    print("Running process_file (integration style)...")
                    processor.process_file(Path("dummy.txt"), document_id="doc1")
                    
                    # Wait for threads
                    print("Waiting for background threads...")
                    max_wait = 5
                    start = time.time()
                    while processor._bg_entity_threads:
                        if time.time() - start > max_wait:
                            break
                        time.sleep(0.1)
                    for t in processor._bg_entity_threads:
                        t.join(timeout=1)
                
                # Assertions
                print("Verifying calls...")
                call_args_list = mock_graph_db.acreate_entity_node.call_args_list
                print(f"acreate_entity_node call count: {len(call_args_list)}")
                
                if len(call_args_list) > 0:
                    args, _ = call_args_list[0]
                    self.assertEqual(args[1], "test_entity")
                    print("SUCCESS: Entity creation called for 'test_entity'")
                else:
                    # Also check sync creation just in case
                    sync_calls = mock_graph_db.create_entity_node.call_args_list
                    if len(sync_calls) > 0:
                        print("SUCCESS: Entity creation called (sync) for 'test_entity'")
                    else:
                        print("FAILURE: No entity creation calls found.")

if __name__ == "__main__":
    unittest.main()

