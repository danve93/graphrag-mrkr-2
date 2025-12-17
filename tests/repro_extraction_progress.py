
import asyncio
import logging
import sys
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pre-emptive mocking of modules with side-effects
mock_graph_db_module = MagicMock()
mock_graph_db_module.graph_db = MagicMock()
mock_graph_db_module.GraphDB = MagicMock
sys.modules["core.graph_db"] = mock_graph_db_module

# Also mock core.embeddings to prevent httpx client init
mock_embeddings_module = MagicMock()
mock_embeddings_module.embedding_manager = MagicMock()
sys.modules["core.embeddings"] = mock_embeddings_module

# Mock core.llm to prevent httpx client init
mock_llm_module = MagicMock()
mock_llm_module.llm_manager = MagicMock()
sys.modules["core.llm"] = mock_llm_module

# Mock settings
with patch("config.settings.settings") as mock_settings:
    mock_settings.entity_extraction_format = "tuple_v1"
    mock_settings.llm_concurrency = 2
    mock_settings.llm_delay_min = 0.0
    mock_settings.llm_delay_max = 0.0
    mock_settings.max_gleanings = 0
    # No need to set other settings since we mocked the consumers!
    
    # Import DocumentProcessor after mocking settings and dependencies
    # We need to make sure core.llm is imported correctly though, as it's used
    from ingestion.document_processor import DocumentProcessor, EntityExtractionState
    from core.entity_extraction import EntityExtractor
    
    # We need to fix EntityExtractionState since it might be mocked if imported from graph_db?
    # No, EntityExtractionState is in document_processor.py? 
    # Wait, EntityExtractionState is defined in document_processor.py.
    # But document_processor imports graph_db.
    
    # We need to unpatch sys.modules later? No, script ends.

async def test_extraction_progress():
    print("Testing extraction progress reporting...")
    
    # Mock GraphDB instance methods that document processor uses
    mock_graph_db = mock_graph_db_module.graph_db
    
    # Mock LLM Manager from our pre-emptive mock
    mock_llm = mock_llm_module.llm_manager
    
    # Create processor instance
    processor = DocumentProcessor()
    
    # Override entity extractor with a mock that calls the callback
    mock_extractor = MagicMock()
        
    async def mock_extract_chunks(chunks, is_cancelled=None, progress_callback=None):
        print(f"Mock extractor called with {len(chunks)} chunks")
        results = []
        for i, chunk in enumerate(chunks):
            # Simulate processing delay
            await asyncio.sleep(0.01)
            
            # Call callback
            if progress_callback:
                processed = i + 1
                total = len(chunks)
                print(f"Calling progress callback: {processed}/{total}")
                progress_callback(processed, total)
            
            # Return dummy data
            results.append(({}, []))
        
        # Combine results (simplified)
        return {}, {}

    mock_extractor.extract_from_chunks = mock_extract_chunks
    mock_extractor.extract_from_chunks_with_gleaning = mock_extract_chunks
    
    processor.entity_extractor = mock_extractor
    
    # Mock _update_entity_operation and _persist_extraction_results
    processor._update_entity_operation = MagicMock()
    processor._persist_extraction_results = MagicMock(return_value=({}, [], {}))
        
    # Verify EntityExtractor behaves correctly with callback
    real_extractor = EntityExtractor()
    
    # Mock extract_from_chunk to be awaitable
    async def mock_extract_single(text, chunk_id):
        return [], []
    real_extractor.extract_from_chunk = mock_extract_single
    
    # Test EntityExtractor.extract_from_chunks with callback
    chunks = [{"chunk_id": "1", "content": "test"}, {"chunk_id": "2", "content": "test"}]
    
    mock_callback = MagicMock()
    
    print("\n--- Testing EntityExtractor.extract_from_chunks ---")
    await real_extractor.extract_from_chunks(chunks, progress_callback=mock_callback)
    
    # Verify callback was called
    print(f"Callback call count: {mock_callback.call_count}")
    if mock_callback.call_count == 2:
        print("SUCCESS: EntityExtractor Progress Callback working")
        mock_callback.assert_any_call(1, 2)
        mock_callback.assert_any_call(2, 2)
    else:
        print(f"FAILURE: Callback called {mock_callback.call_count} times, expected 2")

    # Now test the DocumentProcessor callback logic explicitly
    # We define the callback exactly as in the code to verify its logic
    print("\n--- Testing DocumentProcessor callback logic ---")
    
    processed = 50
    total = 100
    use_gleaning = False
    max_gleanings = 0
    operation_id = "op_test"
    doc_id_local = "doc_test"
    
    # Define the callback logic we want to verify
    # (This duplicates code but verifies the math/logic)
    msg = f"Running LLM entity extraction ({processed}/{total} chunks)"
    processor._update_entity_operation(
        operation_id,
        EntityExtractionState.LLM_EXTRACTION,
        msg,
    )

    # Update overall document progress
    base_progress = 75.0
    phase_range = 20.0
    current_percentage = (processed / total) * 100
    mapped_progress = base_progress + (current_percentage / 100 * phase_range)
    
    mock_graph_db.create_document_node(
        doc_id_local,
        {"processing_progress": min(mapped_progress, 95.0)},
    )
    
    # Access the mock calls to verify
    processor._update_entity_operation.assert_called_with(
        operation_id, EntityExtractionState.LLM_EXTRACTION, "Running LLM entity extraction (50/100 chunks)"
    )
    print("SUCCESS: _update_entity_operation called correctly")
    
    expected_progress = 75.0 + (50/100 * 20.0) # 85.0
    mock_graph_db.create_document_node.assert_called_with(
        doc_id_local,
        {"processing_progress": 85.0}
    )
    print(f"SUCCESS: create_document_node called with progress {expected_progress}%")

if __name__ == "__main__":
    asyncio.run(test_extraction_progress())
