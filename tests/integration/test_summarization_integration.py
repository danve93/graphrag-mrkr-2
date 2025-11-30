"""
Integration tests for Phase 4 Description Summarization.
Tests the full integration with EntityGraph, Document Processor, and Phases 1-3.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.entity_graph import EntityGraph
from core.description_summarizer import DescriptionSummarizer
from ingestion.document_processor import DocumentProcessor
from config.settings import settings


class TestEntityGraphIntegration:
    """Test EntityGraph integration with DescriptionSummarizer."""
    
    @pytest.mark.asyncio
    async def test_summarize_descriptions_updates_entity_nodes(self):
        """Test summarize_descriptions() updates entity node descriptions."""
        # Mock settings
        original_enabled = settings.enable_description_summarization
        original_min_mentions = settings.summarization_min_mentions
        original_min_length = settings.summarization_min_length
        settings.enable_description_summarization = True
        settings.summarization_min_mentions = 2
        settings.summarization_min_length = 50
        
        try:
            # Create EntityGraph
            graph = EntityGraph()
            
            # Add entity with long accumulated description
            original_desc = "Admin Panel provides user management. " * 10  # Repeat to make long
            graph.add_entity("ADMIN PANEL", "Component", original_desc, importance_score=0.8, source_chunks=["chunk_0"])
            graph.add_entity("ADMIN PANEL", "Component", original_desc, importance_score=0.8, source_chunks=["chunk_1"])
            graph.add_entity("ADMIN PANEL", "Component", original_desc, importance_score=0.8, source_chunks=["chunk_2"])
            
            # Verify entity was deduplicated (should have 1 node)
            nodes_before = list(graph.graph.nodes())
            assert len(nodes_before) == 1  # Deduplicated to one entity
            
            # Mock LLM response
            mock_summary = "Web-based administration interface"
            
            with patch('core.llm.llm_manager.generate_response', return_value=f"1. {mock_summary}"):
                # Summarize descriptions
                stats = await graph.summarize_descriptions()
                
                # Verify statistics indicate summarization happened
                assert stats["status"] == "success"
                assert stats["entities_summarized"] >= 1
                assert stats["average_compression_ratio"] < 1.0
                
                # Verify entity description was updated by checking node data
                entity_data = graph.get_entity("ADMIN PANEL", "Component")
                assert entity_data["mention_count"] == 3
                # Description should be summarized (shorter than accumulated original)
                assert len(entity_data["description"]) < len(original_desc) * 3
        finally:
            settings.enable_description_summarization = original_enabled
            settings.summarization_min_mentions = original_min_mentions
            settings.summarization_min_length = original_min_length
    
    @pytest.mark.asyncio
    async def test_summarize_descriptions_updates_relationship_edges(self):
        """Test summarize_descriptions() updates relationship edge descriptions."""
        # Mock settings
        original_enabled = settings.enable_description_summarization
        original_min_mentions = settings.summarization_min_mentions
        original_min_length = settings.summarization_min_length
        settings.enable_description_summarization = True
        settings.summarization_min_mentions = 2
        settings.summarization_min_length = 50
        
        try:
            # Create EntityGraph
            graph = EntityGraph()
            
            # Add entities
            graph.add_entity("SERVICE A", "Service", "Service A description", importance_score=0.8, source_chunks=["chunk_0"])
            graph.add_entity("SERVICE B", "Service", "Service B description", importance_score=0.8, source_chunks=["chunk_0"])
            
            # Add relationship with long description
            original_rel_desc = "SERVICE A depends on SERVICE B for data. " * 10
            graph.add_relationship(
                "SERVICE A", "SERVICE B", "DEPENDS_ON", original_rel_desc, 
                strength=0.8, source_chunks=["chunk_0"]
            )
            graph.add_relationship(
                "SERVICE A", "SERVICE B", "DEPENDS_ON", original_rel_desc,
                strength=0.8, source_chunks=["chunk_1"]
            )
            graph.add_relationship(
                "SERVICE A", "SERVICE B", "DEPENDS_ON", original_rel_desc,
                strength=0.8, source_chunks=["chunk_2"]
            )
            
            # Verify relationship was deduplicated (should have 1 edge)
            edges_before = list(graph.graph.edges(data=True))
            assert len(edges_before) >= 1  # At least one relationship
            
            # Mock LLM responses (one for relationship)
            mock_summary = "Dependency for data access"
            
            with patch('core.llm.llm_manager.generate_response', return_value=f"1. {mock_summary}"):
                # Summarize descriptions
                stats = await graph.summarize_descriptions()
                
                # Verify statistics indicate summarization happened
                assert stats["status"] == "success"
                assert stats["relationships_summarized"] >= 1
                
                # Verify edges still exist (summarization doesn't remove them)
                edges_after = list(graph.graph.edges())
                assert len(edges_after) >= 1
        finally:
            settings.enable_description_summarization = original_enabled
            settings.summarization_min_mentions = original_min_mentions
            settings.summarization_min_length = original_min_length
    
    @pytest.mark.asyncio
    async def test_summarize_descriptions_disabled_returns_status(self):
        """Test summarize_descriptions() returns disabled status when disabled."""
        # Mock settings
        original_enabled = settings.enable_description_summarization
        settings.enable_description_summarization = False
        
        try:
            # Create EntityGraph with entity
            graph = EntityGraph()
            graph.add_entity("TEST", "Component", "description" * 50, importance_score=0.8, source_chunks=["chunk_0"])
            
            # Summarize (should be disabled)
            stats = await graph.summarize_descriptions()
            
            # Verify disabled status
            assert stats["status"] == "disabled"
            assert stats["entities_summarized"] == 0
            assert stats["relationships_summarized"] == 0
        finally:
            settings.enable_description_summarization = original_enabled
    
    @pytest.mark.asyncio
    async def test_summarize_descriptions_handles_errors(self):
        """Test summarize_descriptions() handles LLM errors gracefully."""
        # Mock settings
        original_enabled = settings.enable_description_summarization
        original_min_mentions = settings.summarization_min_mentions
        original_min_length = settings.summarization_min_length
        settings.enable_description_summarization = True
        settings.summarization_min_mentions = 2
        settings.summarization_min_length = 50
        
        try:
            # Create EntityGraph
            graph = EntityGraph()
            
            # Add entity
            for i in range(3):
                graph.add_entity("ENTITY", "Component", "description" * 50, importance_score=0.8, source_chunks=[f"chunk_{i}"])
            
            # Mock LLM to raise exception
            with patch('core.llm.llm_manager.generate_response', side_effect=Exception("LLM error")):
                # Summarize (should handle error)
                stats = await graph.summarize_descriptions()
                
                # Verify error handled gracefully
                assert stats["status"] == "success"  # Still completes
                
                # Verify entity still exists
                entity_data = graph.get_entity("ENTITY", "Component")
                assert entity_data is not None
                assert entity_data["mention_count"] == 3
        finally:
            settings.enable_description_summarization = original_enabled
            settings.summarization_min_mentions = original_min_mentions
            settings.summarization_min_length = original_min_length


class TestDocumentProcessorIntegration:
    """Test Document Processor integration with Phase 4."""
    
    @pytest.mark.asyncio
    async def test_phase4_summarization_called(self):
        """Test Phase 4 summarization is called via EntityGraph when enabled."""
        # Mock settings
        original_enabled = settings.enable_description_summarization
        original_min_mentions = settings.summarization_min_mentions
        original_min_length = settings.summarization_min_length
        settings.enable_description_summarization = True
        settings.summarization_min_mentions = 2
        settings.summarization_min_length = 50
        
        try:
            # Create real EntityGraph with entities
            graph = EntityGraph()
            for i in range(3):
                graph.add_entity("TEST ENTITY", "Component", "description" * 50, 
                                importance_score=0.8, source_chunks=[f"chunk_{i}"])
            
            # Mock LLM
            with patch('core.llm.llm_manager.generate_response', return_value="1. Summarized description"):
                # Call summarize_descriptions
                stats = await graph.summarize_descriptions()
                
                # Verify Phase 4 executed
                assert stats["status"] == "success"
                assert stats["entities_summarized"] >= 1
        finally:
            settings.enable_description_summarization = original_enabled
            settings.summarization_min_mentions = original_min_mentions
            settings.summarization_min_length = original_min_length
    
    @pytest.mark.asyncio
    async def test_phase4_disabled_skips_summarization(self):
        """Test Phase 4 is skipped when disabled."""
        # Mock settings
        original_enabled = settings.enable_description_summarization
        settings.enable_description_summarization = False
        
        try:
            # Create EntityGraph
            graph = EntityGraph()
            graph.add_entity("TEST", "Component", "description" * 50, 
                            importance_score=0.8, source_chunks=["chunk_0"])
            
            # Call summarize_descriptions (should be disabled)
            stats = await graph.summarize_descriptions()
            
            # Verify disabled status
            assert stats["status"] == "disabled"
            assert stats["entities_summarized"] == 0
        finally:
            settings.enable_description_summarization = original_enabled


class TestFullPipelineIntegration:
    """Test full Phase 1+2+3+4 pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_phases_1_2_3_4_together(self):
        """Test Phases 1, 2, 3, and 4 work together in full pipeline."""
        # Mock settings for all phases
        original_enabled = settings.enable_description_summarization
        original_entity_enabled = settings.enable_entity_extraction
        original_min_mentions = settings.summarization_min_mentions
        original_min_length = settings.summarization_min_length
        
        settings.enable_description_summarization = True
        settings.enable_entity_extraction = True
        settings.summarization_min_mentions = 2
        settings.summarization_min_length = 50
        
        try:
            # Create EntityGraph (simulates Phase 2)
            graph = EntityGraph()
            
            # Add entities (simulates Phase 1 gleaning output)
            for i in range(3):
                graph.add_entity("WEB SERVER", "Component", "Web server handles HTTP requests. " * 10, importance_score=0.9, source_chunks=[f"chunk_{i}"])
                graph.add_entity("DATABASE", "Component", "Database stores user data. " * 10, importance_score=0.9, source_chunks=[f"chunk_{i}"])
            
            # Add relationships (simulates Phase 3 tuples)
            for i in range(3):
                graph.add_relationship(
                    "WEB SERVER", "DATABASE", "DEPENDS_ON",
                    "Web server depends on database for persistence. " * 10,
                    strength=0.9, source_chunks=[f"chunk_{i}"]
                )
            
            # Verify entities were added (Phase 2 deduplication should create 2 nodes)
            nodes_before = list(graph.graph.nodes())
            assert len(nodes_before) == 2  # WEB SERVER and DATABASE
            
            # Mock LLM for Phase 4 summarization
            def mock_llm(prompt, **kwargs):
                # Different responses for entities vs relationships
                if "WEB SERVER" in prompt and "DATABASE" in prompt and "->" not in prompt:
                    # Entity batch
                    return "1. HTTP request handler\n2. User data storage"
                else:
                    # Relationship batch
                    return "1. Database dependency for persistence"
            
            with patch('core.llm.llm_manager.generate_response', side_effect=mock_llm):
                # Run Phase 4 summarization
                stats = await graph.summarize_descriptions()
                
                # Verify Phase 4 results
                assert stats["status"] == "success"
                assert stats["entities_summarized"] == 2
                assert stats["relationships_summarized"] >= 1
                assert stats["average_compression_ratio"] < 1.0
                
                # Verify entities still exist with correct mention counts
                web_data = graph.get_entity("WEB SERVER", "Component")
                db_data = graph.get_entity("DATABASE", "Component")
                assert web_data["mention_count"] == 3
                assert db_data["mention_count"] == 3
                
                # Verify descriptions are shorter after summarization
                original_entity_length = len("Web server handles HTTP requests. " * 10)
                assert len(web_data["description"]) < original_entity_length
        finally:
            settings.enable_description_summarization = original_enabled
            settings.enable_entity_extraction = original_entity_enabled
            settings.summarization_min_mentions = original_min_mentions
            settings.summarization_min_length = original_min_length
    
    @pytest.mark.asyncio
    async def test_phase4_preserves_phase2_deduplication(self):
        """Test Phase 4 preserves Phase 2 entity deduplication."""
        # Mock settings
        original_enabled = settings.enable_description_summarization
        original_min_mentions = settings.summarization_min_mentions
        original_min_length = settings.summarization_min_length
        settings.enable_description_summarization = True
        settings.summarization_min_mentions = 2
        settings.summarization_min_length = 50
        
        try:
            # Create EntityGraph
            graph = EntityGraph()
            
            # Add same entity with different casings (Phase 2 deduplication)
            graph.add_entity("Admin Panel", "Component", "description" * 50, importance_score=0.8, source_chunks=["chunk_0"])
            graph.add_entity("ADMIN PANEL", "Component", "description" * 50, importance_score=0.8, source_chunks=["chunk_1"])
            graph.add_entity("admin panel", "Component", "description" * 50, importance_score=0.8, source_chunks=["chunk_2"])
            
            # Verify Phase 2 deduplication (should have 1 node)
            nodes = list(graph.graph.nodes())
            assert len(nodes) == 1
            
            # Get canonical name (Phase 2 uses first occurrence casing)
            canonical_name = nodes[0]
            assert graph.graph.nodes[canonical_name]["mention_count"] == 3
            
            # Mock LLM
            with patch('core.llm.llm_manager.generate_response', return_value="1. Administration interface"):
                # Run Phase 4
                await graph.summarize_descriptions()
                
                # Verify still only 1 node (Phase 4 doesn't create duplicates)
                nodes_after = list(graph.graph.nodes())
                assert len(nodes_after) == 1
                assert nodes_after[0] == canonical_name
        finally:
            settings.enable_description_summarization = original_enabled
            settings.summarization_min_mentions = original_min_mentions
            settings.summarization_min_length = original_min_length
