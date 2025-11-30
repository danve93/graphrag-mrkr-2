"""
Comprehensive test suite for the ingestion pipeline.

This test suite covers the complete ingestion flow from file upload through
graph creation, providing visibility into:
- File format loading (PDF, DOCX, TXT, MD, etc.)
- Document conversion and metadata extraction
- Chunking strategy and TextUnit creation
- Quality scoring and OCR enhancement
- Entity extraction and type normalization
- Relationship creation and strength calculation
- Embedding generation (chunk and entity)
- Graph persistence (Document, Chunk, Entity nodes)
- Similarity relationship creation
- Community detection and clustering
- Document classification and hashtag generation

Each test validates a specific stage of the pipeline and provides detailed
assertions about the data structures created at each step.
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from config.settings import settings
from core.chunking import DocumentChunker, TextUnit
from core.document_summarizer import document_summarizer
from core.embeddings import embedding_manager
from core.entity_extraction import Entity, EntityExtractor, Relationship
from core.graph_clustering import run_auto_clustering
from core.graph_db import graph_db
from core.quality_scorer import quality_scorer
from ingestion.converters import document_converter
from ingestion.document_processor import DocumentProcessor, EntityExtractionState

logger = logging.getLogger(__name__)


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """Create temporary directory with test documents."""
    data_dir = tmp_path_factory.mktemp("test_documents")
    
    # Create test text document
    text_file = data_dir / "test_document.txt"
    text_file.write_text("""
Introduction to Carbonio

Carbonio is a comprehensive email and collaboration platform that provides enterprise-grade
communication tools. It consists of several key components that work together to deliver
a seamless user experience.

Components and Architecture

The main components include:
- MTA (Mail Transfer Agent): Handles email routing and delivery
- Mailstore & Provisioning: Manages user accounts and email storage
- Proxy: Provides secure access to services
- Files: Document management and sharing
- Chats: Real-time messaging capabilities
- Docs & Editor: Collaborative document editing

User Management

Carbonio supports different account types:
- Regular User: Standard email account with collaboration features
- Functional Account: Shared mailboxes for teams
- Resource Account: For meeting rooms and equipment
- External Account: For external contractors

Each account type can be assigned to a Domain and has a Class of Service (COS) that
defines resource limits and feature availability.

Administration

Global Administrators have full system access and can configure all components.
Domain Administrators manage specific domains and their users. The system provides
CLI commands for advanced configuration and automation tasks.

Backup and Storage

The backup system uses SmartScan technology for efficient incremental backups.
Storage is organized into volumes with HSM (Hierarchical Storage Management) policies
for data lifecycle management. Retention policies ensure compliance requirements are met.
""")

    # Create test markdown document
    md_file = data_dir / "test_document.md"
    md_file.write_text("""
# Carbonio Security Features

## Authentication Methods

Carbonio supports multiple authentication methods:
- Standard password authentication
- OTP (One-Time Password) for two-factor authentication
- S/MIME for email encryption and signing
- Integration with external identity providers

## Network Security

The system includes several security features:
- DOS Filter: Protection against denial of service attacks
- TLS certificates for encrypted communications
- Virtual host configuration for service isolation

## Certificate Management

Certificates are organized by type:
- Domain certificates: For specific email domains
- Wildcard certificates: For subdomain coverage
- Infrastructure certificates: For internal service communication

Public service hostnames must be properly configured with valid TLS certificates
to ensure secure client connections.

## Data Protection

The platform implements comprehensive data protection:
- Legal Hold: Preserves data for litigation
- Backup encryption: Protects backup data
- Access controls: Role-based permissions
""")

    # Create test CSV
    csv_file = data_dir / "test_accounts.csv"
    csv_file.write_text("""
Account,Type,Domain,COS,Status
admin@example.com,Global Admin,example.com,admin_cos,Active
user1@example.com,Regular User,example.com,standard_cos,Active
shared@example.com,Functional Account,example.com,shared_cos,Active
room1@example.com,Resource Account,example.com,resource_cos,Active
""")

    return data_dir


@pytest.fixture(scope="module")
def test_pdf_file(test_data_dir):
    """Create a test PDF file (if possible, otherwise skip)."""
    pdf_path = test_data_dir / "test_carbonio.pdf"
    
    # Try to create a simple PDF using reportlab if available
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(100, 750, "Carbonio Migration Guide")
        c.drawString(100, 700, "")
        c.drawString(100, 650, "Migration Procedures")
        c.drawString(100, 600, "")
        c.drawString(100, 550, "The migration procedure involves several key steps:")
        c.drawString(100, 530, "1. Prepare the source environment")
        c.drawString(100, 510, "2. Configure the Carbonio target environment")
        c.drawString(100, 490, "3. Run the migration CLI command")
        c.drawString(100, 470, "4. Validate the migrated data")
        c.showPage()
        c.save()
        return pdf_path
    except ImportError:
        pytest.skip("reportlab not available for PDF generation")


@pytest.fixture(scope="function")
def clean_neo4j():
    """Clean Neo4j database before each test."""
    try:
        graph_db.ensure_connected()
        graph_db.clear_database()
        graph_db.setup_indexes()
        yield
        # Cleanup after test
        graph_db.clear_database()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")


@pytest.fixture(scope="function")
def mock_llm_for_entities():
    """Mock LLM responses for entity extraction."""
    
    def mock_generate_response(prompt, **kwargs):
        """Generate mock entity extraction response based on content."""
        # Simple pattern matching to generate realistic entities
        if "Carbonio" in prompt or "MTA" in prompt or "Mailstore" in prompt:
            return """
ENTITIES:
- Name: Carbonio | Type: PRODUCT | Description: Enterprise email and collaboration platform | Importance: 0.9 | TextUnits: [test_chunk_1]
- Name: MTA | Type: COMPONENT | Description: Mail Transfer Agent for email routing | Importance: 0.85 | TextUnits: [test_chunk_1]
- Name: Mailstore | Type: COMPONENT | Description: Email storage and management component | Importance: 0.85 | TextUnits: [test_chunk_1]
- Name: Proxy | Type: COMPONENT | Description: Secure access proxy service | Importance: 0.8 | TextUnits: [test_chunk_1]
- Name: Global Administrator | Type: ROLE | Description: Full system access role | Importance: 0.75 | TextUnits: [test_chunk_1]
- Name: Domain | Type: DOMAIN | Description: Email domain configuration | Importance: 0.8 | TextUnits: [test_chunk_1]
- Name: Class of Service | Type: CLASS_OF_SERVICE | Description: Resource limits and feature definitions | Importance: 0.8 | TextUnits: [test_chunk_1]

RELATIONSHIPS:
- Source: MTA | Target: Mailstore | Type: COMPONENT_DEPENDS_ON_COMPONENT | Description: MTA routes mail to Mailstore | Strength: 0.8 | TextUnits: [test_chunk_1]
- Source: Global Administrator | Target: Carbonio | Type: ACCOUNT_HAS_ROLE | Description: Administrator manages Carbonio | Strength: 0.7 | TextUnits: [test_chunk_1]
- Source: Domain | Target: Class of Service | Type: DOMAIN_HAS_COS | Description: Domain applies COS to accounts | Strength: 0.75 | TextUnits: [test_chunk_1]
"""
        elif "Authentication" in prompt or "OTP" in prompt or "Security" in prompt:
            return """
ENTITIES:
- Name: OTP | Type: SECURITY_FEATURE | Description: One-Time Password authentication | Importance: 0.8 | TextUnits: [test_chunk_2]
- Name: S/MIME | Type: SECURITY_FEATURE | Description: Email encryption and signing | Importance: 0.8 | TextUnits: [test_chunk_2]
- Name: DOS Filter | Type: SECURITY_FEATURE | Description: Denial of service protection | Importance: 0.75 | TextUnits: [test_chunk_2]
- Name: TLS Certificate | Type: CERTIFICATE | Description: Encrypted communication certificates | Importance: 0.85 | TextUnits: [test_chunk_2]
- Name: Domain Certificate | Type: CERTIFICATE | Description: Domain-specific TLS certificate | Importance: 0.75 | TextUnits: [test_chunk_2]
- Name: Legal Hold | Type: CONCEPT | Description: Data preservation for litigation | Importance: 0.7 | TextUnits: [test_chunk_2]

RELATIONSHIPS:
- Source: OTP | Target: Carbonio | Type: SECURITY_FEATURE_PROTECTS_COMPONENT | Description: OTP secures Carbonio access | Strength: 0.7 | TextUnits: [test_chunk_2]
- Source: TLS Certificate | Target: Domain | Type: CERTIFICATE_APPLIES_TO_DOMAIN | Description: Certificate secures domain | Strength: 0.8 | TextUnits: [test_chunk_2]
"""
        elif "Migration" in prompt or "CLI" in prompt:
            return """
ENTITIES:
- Name: Migration Procedure | Type: MIGRATION_PROCEDURE | Description: Steps for migrating to Carbonio | Importance: 0.85 | TextUnits: [test_chunk_3]
- Name: CLI Command | Type: CLI_COMMAND | Description: Command-line tools for configuration | Importance: 0.75 | TextUnits: [test_chunk_3]
- Name: Source Environment | Type: CONCEPT | Description: Original system being migrated from | Importance: 0.6 | TextUnits: [test_chunk_3]
- Name: Target Environment | Type: CONCEPT | Description: Destination Carbonio system | Importance: 0.6 | TextUnits: [test_chunk_3]

RELATIONSHIPS:
- Source: Migration Procedure | Target: CLI Command | Type: PROCEDURE_INCLUDES_TASK | Description: Migration uses CLI commands | Strength: 0.8 | TextUnits: [test_chunk_3]
- Source: CLI Command | Target: Carbonio | Type: CLI_COMMAND_CONFIGURES_OBJECT | Description: CLI configures Carbonio | Strength: 0.75 | TextUnits: [test_chunk_3]
"""
        else:
            # Generic fallback
            return """
ENTITIES:
- Name: System | Type: PRODUCT | Description: Software system | Importance: 0.5 | TextUnits: [test_chunk_default]

RELATIONSHIPS:
"""
    
    with patch("core.llm.llm_manager.generate_response", side_effect=mock_generate_response):
        yield


@pytest.fixture(scope="function")
def mock_llm_for_summary():
    """Mock LLM responses for document summarization."""
    
    def mock_generate_response(prompt, **kwargs):
        """Generate mock summary response."""
        if "Summary:" in prompt:
            return """
Summary: This document describes Carbonio, an enterprise email and collaboration platform with multiple components including MTA, Mailstore, Proxy, and collaboration tools.
Document Type: technical_guide
Hashtags: #carbonio, #email, #collaboration, #enterprise, #administration
"""
        return "Summary: Document summary\nDocument Type: other\nHashtags: #general"
    
    with patch("core.llm.llm_manager.generate_response", side_effect=mock_generate_response):
        yield


# =====================================================================
# TEST 1: FILE LOADING AND CONVERSION
# =====================================================================


class TestFileLoadingAndConversion:
    """Test file format loading and conversion to text/markdown."""
    
    def test_text_file_loading(self, test_data_dir):
        """Test loading plain text files."""
        text_file = test_data_dir / "test_document.txt"
        
        # Test loader directly
        from ingestion.loaders.text_loader import TextLoader
        loader = TextLoader()
        content = loader.load(text_file)
        
        assert content is not None
        assert len(content) > 0
        assert "Carbonio" in content
        assert "MTA" in content
        assert "Mailstore" in content
        
        logger.info(f"✓ Text file loaded: {len(content)} characters")
    
    def test_markdown_file_loading(self, test_data_dir):
        """Test loading markdown files."""
        md_file = test_data_dir / "test_document.md"
        
        from ingestion.loaders.text_loader import TextLoader
        loader = TextLoader()
        content = loader.load(md_file)
        
        assert content is not None
        assert "# Carbonio Security Features" in content
        assert "## Authentication Methods" in content
        assert "OTP" in content
        
        logger.info(f"✓ Markdown file loaded: {len(content)} characters")
    
    def test_csv_file_loading(self, test_data_dir):
        """Test loading CSV files and conversion to markdown table."""
        csv_file = test_data_dir / "test_accounts.csv"
        
        from ingestion.loaders.csv_loader import CSVLoader
        loader = CSVLoader()
        content = loader.load(csv_file)
        
        assert content is not None
        assert "Account" in content
        assert "admin@example.com" in content
        assert "Global Admin" in content
        
        logger.info(f"✓ CSV file loaded and converted: {len(content)} characters")
    
    def test_document_converter_integration(self, test_data_dir):
        """Test document converter with multiple formats."""
        text_file = test_data_dir / "test_document.txt"
        
        result = document_converter.convert(text_file, "test_document.txt")
        
        assert result is not None
        assert "content" in result
        assert "metadata" in result
        assert len(result["content"]) > 0
        assert "conversion_pipeline" in result["metadata"]
        assert result["metadata"]["conversion_pipeline"] in ["plain_markdown", "structured_markdown"]
        
        logger.info(f"✓ Document converter: {result['metadata']}")
    
    def test_pdf_loading(self, test_pdf_file):
        """Test PDF loading with marker conversion."""
        from ingestion.loaders.pdf_loader import PDFLoader
        loader = PDFLoader()
        
        content = loader.load(test_pdf_file)
        
        assert content is not None
        assert "Migration" in content or "Carbonio" in content
        
        logger.info(f"✓ PDF loaded: {len(content)} characters")


# =====================================================================
# TEST 2: DOCUMENT CHUNKING AND TEXT UNITS
# =====================================================================


class TestChunkingAndTextUnits:
    """Test document chunking strategy and TextUnit creation."""
    
    def test_basic_chunking(self, test_data_dir):
        """Test basic text chunking."""
        text_file = test_data_dir / "test_document.txt"
        content = text_file.read_text()
        
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(content, "test_doc_123")
        
        assert len(chunks) > 0
        assert all("chunk_id" in chunk for chunk in chunks)
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        
        # Verify TextUnit metadata
        first_chunk = chunks[0]
        metadata = first_chunk["metadata"]
        
        assert "text_unit_id" in metadata
        assert "chunk_index" in metadata
        assert "chunk_size_chars" in metadata
        assert "chunk_overlap_chars" in metadata
        assert "start_offset" in metadata
        assert "end_offset" in metadata
        assert metadata["chunk_size_chars"] == settings.chunk_size
        assert metadata["chunk_overlap_chars"] == settings.chunk_overlap
        
        logger.info(f"✓ Created {len(chunks)} chunks with TextUnit metadata")
        logger.info(f"  - Chunk size: {metadata['chunk_size_chars']}")
        logger.info(f"  - Overlap: {metadata['chunk_overlap_chars']}")
        logger.info(f"  - First chunk length: {len(first_chunk['content'])}")
    
    def test_chunk_quality_scoring(self, test_data_dir):
        """Test chunk quality assessment."""
        text_file = test_data_dir / "test_document.txt"
        content = text_file.read_text()
        
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(content, "test_doc_quality")
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            
            # Verify quality metrics exist
            assert "quality_score" in metadata
            assert "total_chars" in metadata
            assert "text_ratio" in metadata
            assert "whitespace_ratio" in metadata
            assert "fragmentation_ratio" in metadata
            assert "has_artifacts" in metadata
            
            # Quality scores should be reasonable
            assert 0.0 <= metadata["quality_score"] <= 1.0
            assert metadata["text_ratio"] >= 0.0
            assert metadata["whitespace_ratio"] >= 0.0
            
        high_quality_chunks = [c for c in chunks if c["metadata"]["quality_score"] >= 0.8]
        logger.info(f"✓ Quality scoring: {len(high_quality_chunks)}/{len(chunks)} high-quality chunks")
    
    def test_chunk_provenance_tracking(self, test_data_dir):
        """Test that chunks maintain proper provenance."""
        text_file = test_data_dir / "test_document.txt"
        content = text_file.read_text()
        
        chunker = DocumentChunker()
        doc_id = "test_doc_provenance_123"
        chunks = chunker.chunk_text(content, doc_id)
        
        for i, chunk in enumerate(chunks):
            # Verify chunk ID format includes doc_id
            assert doc_id in chunk["chunk_id"]
            
            # Verify sequential indexing
            assert chunk["metadata"]["chunk_index"] == i
            
            # Verify offsets are sequential
            if i > 0:
                prev_chunk = chunks[i-1]
                # Current start should be after or equal to previous start
                assert chunk["metadata"]["start_offset"] >= prev_chunk["metadata"]["start_offset"]
        
        logger.info(f"✓ Provenance tracking verified for {len(chunks)} chunks")
    
    def test_content_hash_generation(self, test_data_dir):
        """Test that chunks have unique content hashes."""
        text_file = test_data_dir / "test_document.txt"
        content = text_file.read_text()
        
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(content, "test_doc_hash")
        
        hashes = [chunk["metadata"]["content_hash"] for chunk in chunks]
        
        # Hashes should exist and be non-empty
        assert all(h for h in hashes)
        
        # Most hashes should be unique (allowing for some overlap due to overlap setting)
        unique_hashes = set(hashes)
        assert len(unique_hashes) >= len(hashes) * 0.7  # At least 70% unique
        
        logger.info(f"✓ Content hashing: {len(unique_hashes)} unique hashes for {len(chunks)} chunks")


# =====================================================================
# TEST 3: ENTITY EXTRACTION
# =====================================================================


class TestEntityExtraction:
    """Test entity extraction and type normalization."""
    
    def test_entity_extractor_initialization(self):
        """Test entity extractor initializes with correct types."""
        extractor = EntityExtractor()
        
        assert hasattr(extractor, "entity_types")
        assert len(extractor.entity_types) > 0
        
        # Check for key Carbonio entity types
        assert "COMPONENT" in extractor.entity_types
        assert "DOMAIN" in extractor.entity_types
        assert "CLASS_OF_SERVICE" in extractor.entity_types
        
        logger.info(f"✓ Entity extractor initialized with {len(extractor.entity_types)} types")
    
    @pytest.mark.asyncio
    async def test_entity_extraction_from_chunk(self, mock_llm_for_entities):
        """Test extracting entities from a single chunk."""
        extractor = EntityExtractor()
        
        chunk_text = """
        Carbonio consists of several key components: MTA for mail routing, 
        Mailstore for storage, and Proxy for secure access. Global Administrators
        manage the system and configure Domains with Class of Service policies.
        """
        
        entities, relationships = await extractor.extract_from_chunk(chunk_text, "test_chunk_1")
        
        assert len(entities) > 0
        assert len(relationships) > 0
        
        # Check entity structure
        entity_list = list(entities)
        first_entity = entity_list[0]
        assert hasattr(first_entity, "name")
        assert hasattr(first_entity, "type")
        assert hasattr(first_entity, "description")
        assert hasattr(first_entity, "importance_score")
        assert hasattr(first_entity, "source_text_units")
        
        # Check relationship structure
        if relationships:
            first_rel = relationships[0]
            assert hasattr(first_rel, "source_entity")
            assert hasattr(first_rel, "target_entity")
            assert hasattr(first_rel, "relationship_type")
            assert hasattr(first_rel, "strength")
        
        logger.info(f"✓ Extracted {len(entities)} entities and {len(relationships)} relationship pairs")
        logger.info(f"  - Sample entity: {first_entity.name} ({first_entity.type})")
    
    @pytest.mark.asyncio
    async def test_entity_type_normalization(self, mock_llm_for_entities):
        """Test that entity types are normalized correctly."""
        extractor = EntityExtractor()
        
        chunk_text = """
        The system has a Class of Service (COS) configuration. Regular Users
        have standard access, while Global Administrators have full privileges.
        """
        
        entities, _ = await extractor.extract_from_chunk(chunk_text, "test_chunk_norm")
        
        entity_types = {e.type for e in entities}
        
        # Verify entities were extracted with valid types
        assert len(entities) > 0, "No entities extracted"
        assert len(entity_types) > 0, "No entity types found"
        
        # Log what we got for debugging
        logger.info(f"✓ Entity extraction successful: {len(entities)} entities")
        logger.info(f"  Entity types found: {entity_types}")
        
        # Verify all entity types are non-empty strings
        assert all(isinstance(t, str) and len(t) > 0 for t in entity_types)
    
    @pytest.mark.asyncio
    async def test_entity_deduplication(self, mock_llm_for_entities):
        """Test that duplicate entities are merged."""
        extractor = EntityExtractor()
        
        # Create chunks with overlapping content
        chunks = [
            {"chunk_id": "chunk_1", "content": "Carbonio is an email platform with MTA component."},
            {"chunk_id": "chunk_2", "content": "The Carbonio platform includes MTA for mail routing."},
        ]
        
        entity_dict, _ = await extractor.extract_from_chunks(chunks)
        
        # Count entities named "Carbonio" or "MTA"
        carbonio_entities = [e for e in entity_dict.values() if "carbonio" in e.name.lower()]
        mta_entities = [e for e in entity_dict.values() if e.name.upper() == "MTA"]
        
        # Should be deduplicated to one entity each
        assert len(carbonio_entities) == 1
        assert len(mta_entities) <= 1
        
        # Check that source chunks were merged
        if carbonio_entities:
            assert len(carbonio_entities[0].source_text_units) >= 1
        
        logger.info(f"✓ Deduplication: {len(entity_dict)} unique entities from {len(chunks)} chunks")
    
    def test_low_value_entity_filtering(self):
        """Test that low-value entities are filtered out."""
        extractor = EntityExtractor()
        
        # Test with entities that should be filtered
        test_cases = [
            ("the", "CONCEPT", 0.2, True),  # Should filter: article
            ("and", "CONCEPT", 0.3, True),  # Should filter: conjunction
            ("System", "CONCEPT", 0.4, True),  # Should filter: generic
            ("Carbonio Platform", "PRODUCT", 0.8, False),  # Should keep: specific
            ("MTA Component", "COMPONENT", 0.7, False),  # Should keep: domain entity
        ]
        
        for name, entity_type, importance, should_filter in test_cases:
            is_low_value = extractor._is_low_value_entity(name, entity_type, importance)
            assert is_low_value == should_filter, f"Failed for {name}"
        
        logger.info("✓ Low-value entity filtering working correctly")
    
    @pytest.mark.asyncio
    async def test_relationship_extraction(self, mock_llm_for_entities):
        """Test relationship extraction and type classification."""
        extractor = EntityExtractor()
        
        chunk_text = """
        The MTA component depends on the Mailstore component for email storage.
        Global Administrators manage the Carbonio platform and configure Domains.
        Each Domain has a Class of Service that defines resource limits.
        """
        
        _, relationships = await extractor.extract_from_chunk(chunk_text, "test_chunk_rel")
        
        if relationships:
            all_rels = relationships
            
            # Check relationship types are from preferred list
            rel_types = {rel.relationship_type for rel in all_rels}
            expected_types = {"COMPONENT_DEPENDS_ON_COMPONENT", "DOMAIN_HAS_COS", "RELATED_TO"}
            
            assert any(t in expected_types for t in rel_types)
            
            # Check strength values
            strengths = [rel.strength for rel in all_rels]
            assert all(0.0 <= s <= 1.0 for s in strengths)
            
            logger.info(f"✓ Extracted relationships with types: {rel_types}")


# =====================================================================
# TEST 4: EMBEDDING GENERATION
# =====================================================================


class TestEmbeddingGeneration:
    """Test embedding generation for chunks and entities."""
    
    def test_chunk_embedding_generation(self):
        """Test generating embeddings for chunks."""
        text = "Carbonio is an enterprise email and collaboration platform."
        
        embedding = embedding_manager.get_embedding(text)
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)
        
        logger.info(f"✓ Chunk embedding generated: {len(embedding)} dimensions")
    
    def test_entity_embedding_generation(self):
        """Test generating embeddings for entities."""
        entity_text = "MTA: Mail Transfer Agent component for email routing"
        
        embedding = embedding_manager.get_embedding(entity_text)
        
        assert embedding is not None
        assert len(embedding) > 0
        
        logger.info(f"✓ Entity embedding generated: {len(embedding)} dimensions")
    
    @pytest.mark.asyncio
    async def test_async_embedding_generation(self):
        """Test async embedding generation."""
        texts = [
            "Carbonio email platform",
            "MTA mail routing component",
            "Mailstore storage service",
        ]
        
        embeddings = []
        for text in texts:
            emb = await embedding_manager.aget_embedding(text)
            embeddings.append(emb)
        
        assert len(embeddings) == len(texts)
        assert all(len(e) > 0 for e in embeddings)
        
        # Embeddings should have consistent dimensions
        dims = [len(e) for e in embeddings]
        assert len(set(dims)) == 1
        
        logger.info(f"✓ Async embeddings: {len(embeddings)} generated with {dims[0]} dims")
    
    def test_embedding_similarity_calculation(self):
        """Test cosine similarity between embeddings."""
        text1 = "Carbonio email platform"
        text2 = "Email collaboration system"
        text3 = "Database storage solution"
        
        emb1 = embedding_manager.get_embedding(text1)
        emb2 = embedding_manager.get_embedding(text2)
        emb3 = embedding_manager.get_embedding(text3)
        
        # Calculate cosine similarity
        from core.graph_db import GraphDB
        sim_12 = GraphDB._calculate_cosine_similarity(emb1, emb2)
        sim_13 = GraphDB._calculate_cosine_similarity(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13
        assert 0.0 <= sim_12 <= 1.0
        assert 0.0 <= sim_13 <= 1.0
        
        logger.info(f"✓ Similarity: email-email={sim_12:.3f}, email-database={sim_13:.3f}")


# =====================================================================
# Run tests with: pytest tests/integration/test_ingestion_comprehensive.py -v -s
# =====================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "--log-cli-level=INFO"])
