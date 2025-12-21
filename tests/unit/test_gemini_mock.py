
import sys
from unittest.mock import MagicMock, patch
import pytest

# Define the mock module globally so we can configure it
mock_genai_module = MagicMock(name="mock.generativeai")

@pytest.fixture(scope="function")
def setup_genai_mock():
    # Setup the mock in sys.modules to allow Initial import
    with patch.dict(sys.modules, {"google.generativeai": mock_genai_module, "google": MagicMock()}):
        # Import core.llm (if not already imported)
        if "core.llm" not in sys.modules:
            import core.llm
        else:
            # If already imported, we don't need to reload, we will patch directly
            pass
            
        if "core.embeddings" not in sys.modules:
            import core.embeddings
            
        yield mock_genai_module

def test_gemini_llm_generation(setup_genai_mock):
    mock_genai_sys = setup_genai_mock
    import core.llm
    
    # Force the module's genai attribute to be OUR mock
    # This bypasses any import/reload confusion
    core.llm.genai = mock_genai_sys
    
    # Confirm they are the same
    assert mock_genai_sys is core.llm.genai
    
    # Reset mock
    mock_genai_sys.reset_mock()
    
    # Prepare the mock chain
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    # Configure the 'text' attribute
    mock_response.text = "I am a Gemini model."
    
    # Configure usage metadata
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 10
    mock_usage.candidates_token_count = 20
    mock_response.usage_metadata = mock_usage
    
    mock_model_instance.generate_content.return_value = mock_response
    mock_genai_sys.GenerativeModel.return_value = mock_model_instance
    
    from core.llm import LLMManager
    from config.settings import settings
    
    original_provider = settings.llm_provider
    settings.llm_provider = "gemini"
    settings.gemini_api_key = "fake_key"
    settings.gemini_model = "gemini-3.0-flash"
    
    try:
        manager = LLMManager()
        response = manager.generate_response("Who are you?", include_usage=True)
        
        # Verify content
        assert response["content"] == "I am a Gemini model."
        assert response["usage"]["input"] == 10
        assert response["usage"]["output"] == 20
        
        # Verify calls
        mock_genai_sys.configure.assert_called_with(api_key="fake_key")
        mock_genai_sys.GenerativeModel.assert_called_with("gemini-3.0-flash")
        
    finally:
        settings.llm_provider = original_provider

def test_gemini_embedding_generation(setup_genai_mock):
    mock_genai_sys = setup_genai_mock
    import core.embeddings
    
    # Force patching
    core.embeddings.genai = mock_genai_sys
    
    mock_genai_sys.reset_mock()
    
    # Mock embedding return
    mock_genai_sys.embed_content.return_value = {'embedding': [0.1, 0.2, 0.3]}
    
    from core.embeddings import EmbeddingManager
    from config.settings import settings
    
    # Force settings change
    with patch("config.settings.settings.llm_provider", "gemini"), \
         patch("config.settings.settings.gemini_embedding_model", "models/text-embedding-004"):
         
         settings.llm_provider = "gemini"
         settings.gemini_api_key = "fake_key"
         
         manager = EmbeddingManager()
         embedding = manager._generate_gemini_embedding("test text")
         
         assert embedding == [0.1, 0.2, 0.3]
         
         mock_genai_sys.embed_content.assert_called_with(
             model="models/text-embedding-004",
             content="test text",
             task_type="retrieval_document",
             title=None
         )
