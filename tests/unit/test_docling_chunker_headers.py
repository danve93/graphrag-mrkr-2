
from unittest.mock import MagicMock
import unittest.mock
from core.docling_chunker import DoclingHybridChunker

class MockDoclingChunk:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata

class MockInternalChunker:
    def __init__(self, chunks):
        self.chunks = chunks

    def chunk(self, document):
        return self.chunks

def test_chunk_document_include_heading_path():
    # Setup - Realistic Sample Data
    # Simulating a technical manual structure
    chunks_data = [
        # Chapter 1
        {
            "text": "This chapter covers basic installation steps.",
            "metadata": {
                "heading_path": "Chapter 1: Installation",
                "page": 5
            }
        },
        # Chapter 1 > Section 1.1
        {
            "text": "Ensure you have Python 3.10 installed.",
            "metadata": {
                "heading_path": "Chapter 1: Installation > 1.1 Prerequisites",
                "page": 5
            }
        },
        # Chapter 2 (No hierarchy)
        {
            "text": "Overview of the architecture.",
            "metadata": {
                "heading_path": "Chapter 2: Architecture",
                "page": 8
            }
        },
        # Chunk with NO heading path
        {
            "text": "orphan text",
            "metadata": {}
        }
    ]
    
    mock_chunks = [MockDoclingChunk(c["text"], c["metadata"]) for c in chunks_data]
    
    # Initialize DoclingHybridChunker with include_heading_path=True
    # We patch _build_chunker to avoid the "unavailable" warning since we don't have docling installed
    with unittest.mock.patch('core.docling_chunker.DoclingHybridChunker._build_chunker', return_value=None):
        chunker = DoclingHybridChunker(
            target_tokens=100,
            min_tokens=10,
            max_tokens=200,
            overlap_tokens=0,
            include_heading_path=True
        )
    
    # Inject our mock internal chunker
    chunker._chunker = MockInternalChunker(mock_chunks)

    # Execute
    results = chunker.chunk_document("dummy_doc")

    # Verify
    assert len(results) == 4
    
    # Chunk 0: Should have header
    assert results[0]["text"] == "Chapter 1: Installation\n\nThis chapter covers basic installation steps."
    
    # Chunk 1: Should have nested header
    assert results[1]["text"] == "Chapter 1: Installation > 1.1 Prerequisites\n\nEnsure you have Python 3.10 installed."
    
    # Chunk 2: Should have header
    assert results[2]["text"] == "Chapter 2: Architecture\n\nOverview of the architecture."
    
    # Chunk 3: Should NOT have header (none provided)
    assert results[3]["text"] == "orphan text"


def test_chunk_document_exclude_heading_path():
    # Setup - Same data
    mock_chunks = [
        MockDoclingChunk(
            text="Content",
            metadata={"heading_path": "Chapter 1"}
        )
    ]
    
    # Initialize with include_heading_path=False
    with unittest.mock.patch('core.docling_chunker.DoclingHybridChunker._build_chunker', return_value=None):
        chunker = DoclingHybridChunker(
            target_tokens=100,
            min_tokens=10,
            max_tokens=200,
            overlap_tokens=0,
            include_heading_path=False
        )
    chunker._chunker = MockInternalChunker(mock_chunks)

    # Execute
    results = chunker.chunk_document("dummy_doc")

    # Verify: Text remains unchanged
    assert results[0]["text"] == "Content"
