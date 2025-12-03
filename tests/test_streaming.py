"""Unit tests for LLM streaming functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.llm import LLMManager


def test_openai_streaming_basic(monkeypatch):
    """Test OpenAI streaming yields tokens correctly."""
    # Mock OpenAI streaming response
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=" world"))]),
        Mock(choices=[Mock(delta=Mock(content="!"))]),
    ]
    
    mock_stream = iter(mock_chunks)
    
    with patch("openai.chat.completions.create", return_value=mock_stream):
        llm = LLMManager()
        llm.provider = "openai"
        llm.model = "gpt-4o-mini"
        
        tokens = list(llm.stream_generate_openai("test prompt"))
        
        assert tokens == ["Hello", " world", "!"]


def test_ollama_streaming_basic(monkeypatch):
    """Test Ollama streaming yields tokens correctly."""
    import json
    
    # Mock Ollama streaming response
    mock_lines = [
        json.dumps({"message": {"content": "Hello"}, "done": False}).encode(),
        json.dumps({"message": {"content": " world"}, "done": False}).encode(),
        json.dumps({"message": {"content": "!"}, "done": True}).encode(),
    ]
    
    mock_response = Mock()
    mock_response.iter_lines.return_value = iter(mock_lines)
    mock_response.raise_for_status = Mock()
    
    with patch("requests.post", return_value=mock_response):
        llm = LLMManager()
        llm.provider = "ollama"
        llm.model = "llama2"
        llm.ollama_base_url = "http://localhost:11434"
        
        tokens = list(llm.stream_generate_ollama("test prompt"))
        
        assert tokens == ["Hello", " world", "!"]


def test_stream_rag_response_openai(monkeypatch):
    """Test stream_generate_rag_response delegates to OpenAI streaming."""
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="The"))]),
        Mock(choices=[Mock(delta=Mock(content=" answer"))]),
        Mock(choices=[Mock(delta=Mock(content=" is"))]),
    ]
    
    mock_stream = iter(mock_chunks)
    
    with patch("openai.chat.completions.create", return_value=mock_stream):
        llm = LLMManager()
        llm.provider = "openai"
        llm.model = "gpt-4o-mini"
        
        context_chunks = [
            {"content": "Sample context", "similarity": 0.9}
        ]
        
        tokens = list(llm.stream_generate_rag_response(
            query="test query",
            context_chunks=context_chunks,
            system_message="You are a helpful assistant",
            temperature=0.7
        ))
        
        assert tokens == ["The", " answer", " is"]


def test_stream_rag_response_ollama(monkeypatch):
    """Test stream_generate_rag_response delegates to Ollama streaming."""
    import json
    
    mock_lines = [
        json.dumps({"message": {"content": "The"}, "done": False}).encode(),
        json.dumps({"message": {"content": " answer"}, "done": False}).encode(),
        json.dumps({"message": {"content": " is"}, "done": True}).encode(),
    ]
    
    mock_response = Mock()
    mock_response.iter_lines.return_value = iter(mock_lines)
    mock_response.raise_for_status = Mock()
    
    with patch("requests.post", return_value=mock_response):
        llm = LLMManager()
        llm.provider = "ollama"
        llm.model = "llama2"
        llm.ollama_base_url = "http://localhost:11434"
        
        context_chunks = [
            {"content": "Sample context", "similarity": 0.9}
        ]
        
        tokens = list(llm.stream_generate_rag_response(
            query="test query",
            context_chunks=context_chunks,
            system_message="You are a helpful assistant",
            temperature=0.7
        ))
        
        assert tokens == ["The", " answer", " is"]


def test_streaming_error_handling(monkeypatch):
    """Test streaming handles errors gracefully."""
    with patch("openai.chat.completions.create", side_effect=Exception("API Error")):
        llm = LLMManager()
        llm.provider = "openai"
        llm.model = "gpt-4o-mini"
        
        # Should not raise, should return empty generator
        tokens = list(llm.stream_generate_openai("test prompt"))
        
        assert tokens == []
