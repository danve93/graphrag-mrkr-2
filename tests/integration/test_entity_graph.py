"""
Unit tests for EntityGraph (Phase 2 NetworkX implementation).

Tests cover:
- Entity deduplication by canonical key
- Description accumulation
- Relationship strength summation
- Orphan entity creation
- UNWIND query generation
- Batch size handling
- Canonical key matching
- Error handling
"""

import pytest
from core.entity_graph import EntityGraph


class TestEntityDeduplication:
    """Tests for entity deduplication logic."""
    
    def test_add_entity_deduplication_case_insensitive(self):
        """Test that entities with same canonical key are merged (case-insensitive)."""
        graph = EntityGraph()
        
        # Add same entity with different casing
        graph.add_entity("Admin Panel", "COMPONENT", "Web interface", 0.8, ["chunk1"])
        graph.add_entity("ADMIN PANEL", "Component", "User management", 0.9, ["chunk2"])
        graph.add_entity("admin panel", "component", "Settings", 0.7, ["chunk3"])
        
        # Should be merged into single entity
        assert graph.graph.number_of_nodes() == 1
        
        # Check merged data
        entity = graph.get_entity("Admin Panel", "COMPONENT")
        assert entity is not None
        assert entity["mention_count"] == 3
        assert "Web interface" in entity["description"]
        assert "User management" in entity["description"]
        assert "Settings" in entity["description"]
    
    def test_add_entity_different_types_not_merged(self):
        """Test that entities with same name but different types are NOT merged."""
        graph = EntityGraph()
        
        graph.add_entity("System", "COMPONENT", "Hardware system", 0.8, ["chunk1"])
        graph.add_entity("System", "CONCEPT", "Abstract concept", 0.9, ["chunk2"])
        
        # Should be two separate entities
        assert graph.graph.number_of_nodes() == 2
        
        # Check both exist
        assert graph.has_entity("System", "COMPONENT")
        assert graph.has_entity("System", "CONCEPT")
    
    def test_canonical_key_whitespace_normalization(self):
        """Test that canonical keys normalize whitespace."""
        graph = EntityGraph()
        
        graph.add_entity("  Admin  Panel  ", "COMPONENT", "Desc 1", 0.8, ["chunk1"])
        graph.add_entity("Admin Panel", "COMPONENT", "Desc 2", 0.9, ["chunk2"])
        
        # Should be merged via canonical key
        # Note: node IDs may differ, but canonical map treats them as same entity
        entity = graph.get_entity("Admin Panel", "COMPONENT")
        assert entity is not None
        assert entity["mention_count"] == 2
        assert "Desc 1" in entity["description"]
        assert "Desc 2" in entity["description"]
    
    def test_empty_type_handling(self):
        """Test that entities with empty types are handled correctly."""
        graph = EntityGraph()
        
        graph.add_entity("EntityA", "", "Description", 0.8, ["chunk1"])
        graph.add_entity("EntityB", None, "Description", 0.8, ["chunk1"])
        
        # Should create two entities (both have empty type)
        assert graph.graph.number_of_nodes() == 2


class TestDescriptionAccumulation:
    """Tests for description accumulation logic."""
    
    def test_description_accumulation_deduplicates(self):
        """Test that duplicate descriptions are not accumulated."""
        graph = EntityGraph()
        
        graph.add_entity("Service A", "SERVICE", "Handles authentication", 0.7, ["chunk1"])
        graph.add_entity("Service A", "SERVICE", "Handles authentication", 0.8, ["chunk2"])
        graph.add_entity("Service A", "SERVICE", "Provides OAuth", 0.9, ["chunk3"])
        
        entity = graph.get_entity("Service A", "SERVICE")
        
        # "Handles authentication" should appear only once
        assert entity["description"].count("Handles authentication") == 1
        # Should have both unique descriptions
        assert "Handles authentication" in entity["description"]
        assert "Provides OAuth" in entity["description"]
    
    def test_description_newline_join(self):
        """Test that descriptions are joined with newlines."""
        graph = EntityGraph()
        
        graph.add_entity("DB", "SERVICE", "PostgreSQL database", 0.8, ["chunk1"])
        graph.add_entity("DB", "SERVICE", "Stores user data", 0.9, ["chunk2"])
        
        entity = graph.get_entity("DB", "SERVICE")
        
        # Check newline separator
        assert "\n" in entity["description"]
        assert entity["description"].count("\n") >= 1
    
    def test_empty_descriptions_filtered(self):
        """Test that empty descriptions are not added."""
        graph = EntityGraph()
        
        graph.add_entity("Entity", "TYPE", "Real description", 0.8, ["chunk1"])
        graph.add_entity("Entity", "TYPE", "", 0.9, ["chunk2"])
        graph.add_entity("Entity", "TYPE", None, 0.7, ["chunk3"])
        
        entity = graph.get_entity("Entity", "TYPE")
        
        # Should only have the real description
        assert entity["description"] == "Real description"
    
    def test_description_sorted(self):
        """Test that descriptions are sorted alphabetically."""
        graph = EntityGraph()
        
        graph.add_entity("X", "T", "Zebra", 0.8, ["chunk1"])
        graph.add_entity("X", "T", "Apple", 0.9, ["chunk2"])
        graph.add_entity("X", "T", "Mango", 0.7, ["chunk3"])
        
        entity = graph.get_entity("X", "T")
        descriptions = entity["description"].split("\n")
        
        # Should be sorted
        assert descriptions == ["Apple", "Mango", "Zebra"]


class TestImportanceScoreAveraging:
    """Tests for importance score averaging."""
    
    def test_importance_score_averaged(self):
        """Test that importance scores are averaged across mentions."""
        graph = EntityGraph()
        
        graph.add_entity("Entity", "TYPE", "Desc", 0.8, ["chunk1"])
        graph.add_entity("Entity", "TYPE", "Desc", 0.6, ["chunk2"])
        
        entity = graph.get_entity("Entity", "TYPE")
        
        # Average of 0.8 and 0.6 = 0.7
        assert entity["importance_score"] == pytest.approx(0.7, abs=0.01)
    
    def test_importance_score_three_mentions(self):
        """Test averaging with three mentions."""
        graph = EntityGraph()
        
        graph.add_entity("Entity", "TYPE", "Desc", 0.9, ["chunk1"])
        graph.add_entity("Entity", "TYPE", "Desc", 0.6, ["chunk2"])
        graph.add_entity("Entity", "TYPE", "Desc", 0.9, ["chunk3"])
        
        entity = graph.get_entity("Entity", "TYPE")
        
        # Average of 0.9, 0.6, 0.9 = 0.8
        assert entity["importance_score"] == pytest.approx(0.8, abs=0.01)


class TestSourceChunkMerging:
    """Tests for source chunk merging."""
    
    def test_source_chunks_deduplicated(self):
        """Test that source chunks are deduplicated."""
        graph = EntityGraph()
        
        graph.add_entity("Entity", "TYPE", "Desc", 0.8, ["chunk1", "chunk2"])
        graph.add_entity("Entity", "TYPE", "Desc", 0.9, ["chunk2", "chunk3"])
        
        entity = graph.get_entity("Entity", "TYPE")
        
        # Should have 3 unique chunks
        assert len(entity["source_chunks"]) == 3
        assert set(entity["source_chunks"]) == {"chunk1", "chunk2", "chunk3"}
    
    def test_source_chunks_sorted(self):
        """Test that source chunks are sorted."""
        graph = EntityGraph()
        
        graph.add_entity("Entity", "TYPE", "Desc", 0.8, ["chunk3", "chunk1"])
        graph.add_entity("Entity", "TYPE", "Desc", 0.9, ["chunk2"])
        
        entity = graph.get_entity("Entity", "TYPE")
        
        # Should be sorted
        assert entity["source_chunks"] == ["chunk1", "chunk2", "chunk3"]


class TestMentionCount:
    """Tests for mention count tracking."""
    
    def test_mention_count_increments(self):
        """Test that mention count increments correctly."""
        graph = EntityGraph()
        
        graph.add_entity("Entity", "TYPE", "Desc1", 0.8, ["chunk1"])
        entity = graph.get_entity("Entity", "TYPE")
        assert entity["mention_count"] == 1
        
        graph.add_entity("Entity", "TYPE", "Desc2", 0.9, ["chunk2"])
        entity = graph.get_entity("Entity", "TYPE")
        assert entity["mention_count"] == 2
        
        graph.add_entity("Entity", "TYPE", "Desc3", 0.7, ["chunk3"])
        entity = graph.get_entity("Entity", "TYPE")
        assert entity["mention_count"] == 3


class TestRelationshipStrengthSummation:
    """Tests for relationship strength summation."""
    
    def test_relationship_strength_summed(self):
        """Test that relationship strengths are summed across mentions."""
        graph = EntityGraph()
        
        # Add entities first
        graph.add_entity("A", "COMPONENT", "Entity A", 0.8, ["chunk1"])
        graph.add_entity("B", "COMPONENT", "Entity B", 0.8, ["chunk1"])
        
        # Add same relationship twice
        graph.add_relationship("A", "B", "DEPENDS_ON", "First mention", 0.5, ["chunk1"])
        graph.add_relationship("A", "B", "DEPENDS_ON", "Second mention", 0.7, ["chunk2"])
        
        # Check edge data (node IDs include type: A_COMPONENT, B_COMPONENT)
        # Find the actual node IDs
        node_ids = list(graph.graph.nodes())
        assert len(node_ids) == 2
        source_id = [n for n in node_ids if "A" in n][0]
        target_id = [n for n in node_ids if "B" in n][0]
        
        assert graph.graph.has_edge(source_id, target_id)
        edge_data = list(graph.graph[source_id][target_id].values())[0]
        
        # Strength should be summed: 0.5 + 0.7 = 1.2
        assert edge_data["strength"] == pytest.approx(1.2, abs=0.01)
        assert edge_data["mention_count"] == 2
    
    def test_relationship_strength_can_exceed_one(self):
        """Test that relationship strength can exceed 1.0."""
        graph = EntityGraph()
        
        graph.add_entity("A", "COMPONENT", "Entity A", 0.8, ["chunk1"])
        graph.add_entity("B", "COMPONENT", "Entity B", 0.8, ["chunk1"])
        
        # Add relationship 5 times with strength 0.6 each
        for i in range(5):
            graph.add_relationship("A", "B", "CALLS", f"Mention {i}", 0.6, [f"chunk{i}"])
        
        # Find actual node IDs
        node_ids = list(graph.graph.nodes())
        source_id = [n for n in node_ids if "A" in n][0]
        target_id = [n for n in node_ids if "B" in n][0]
        edge_data = list(graph.graph[source_id][target_id].values())[0]
        
        # Strength should be 3.0 (5 × 0.6)
        assert edge_data["strength"] == pytest.approx(3.0, abs=0.01)
    
    def test_different_relationship_types_not_merged(self):
        """Test that different relationship types are not merged."""
        graph = EntityGraph()
        
        graph.add_entity("A", "COMPONENT", "Entity A", 0.8, ["chunk1"])
        graph.add_entity("B", "COMPONENT", "Entity B", 0.8, ["chunk1"])
        
        graph.add_relationship("A", "B", "DEPENDS_ON", "Dependency", 0.5, ["chunk1"])
        graph.add_relationship("A", "B", "CALLS", "API call", 0.7, ["chunk2"])
        
        # Should have 2 edges between A and B
        assert graph.graph.number_of_edges() == 2


class TestRelationshipDescriptionAccumulation:
    """Tests for relationship description accumulation."""
    
    def test_relationship_descriptions_accumulated(self):
        """Test that relationship descriptions are accumulated."""
        graph = EntityGraph()
        
        graph.add_entity("A", "COMPONENT", "Entity A", 0.8, ["chunk1"])
        graph.add_entity("B", "COMPONENT", "Entity B", 0.8, ["chunk1"])
        
        graph.add_relationship("A", "B", "DEPENDS_ON", "For authentication", 0.5, ["chunk1"])
        graph.add_relationship("A", "B", "DEPENDS_ON", "For data access", 0.7, ["chunk2"])
        
        # Find actual node IDs
        node_ids = list(graph.graph.nodes())
        source_id = [n for n in node_ids if "A" in n][0]
        target_id = [n for n in node_ids if "B" in n][0]
        edge_data = list(graph.graph[source_id][target_id].values())[0]
        
        assert "For authentication" in edge_data["description"]
        assert "For data access" in edge_data["description"]
        assert "\n" in edge_data["description"]


class TestOrphanEntityCreation:
    """Tests for orphan entity creation."""
    
    def test_orphan_entity_created_for_missing_source(self):
        """Test that orphan entities are created for missing relationship sources."""
        graph = EntityGraph()
        
        # Add target entity but not source
        graph.add_entity("B", "COMPONENT", "Entity B", 0.8, ["chunk1"])
        
        # Add relationship with missing source
        graph.add_relationship("A", "B", "CALLS", "API call", 0.6, ["chunk1"])
        
        # A should be created as orphan
        assert graph.has_entity("A", "")
        orphan = graph.get_entity("A", "")
        assert orphan["is_orphan"] is True
        assert orphan["description"] == ""
        assert orphan["importance_score"] == 0.0
        assert orphan["mention_count"] == 0
    
    def test_orphan_entity_created_for_missing_target(self):
        """Test that orphan entities are created for missing relationship targets."""
        graph = EntityGraph()
        
        # Add source entity but not target
        graph.add_entity("A", "COMPONENT", "Entity A", 0.8, ["chunk1"])
        
        # Add relationship with missing target
        graph.add_relationship("A", "B", "DEPENDS_ON", "Dependency", 0.7, ["chunk1"])
        
        # B should be created as orphan
        assert graph.has_entity("B", "")
        orphan = graph.get_entity("B", "")
        assert orphan["is_orphan"] is True
    
    def test_orphan_upgraded_when_real_entity_added(self):
        """Test that orphans can be upgraded to real entities."""
        graph = EntityGraph()
        
        # Add relationship that creates orphan
        graph.add_relationship("A", "B", "CALLS", "API call", 0.6, ["chunk1"])
        
        # B is orphan
        assert graph.has_entity("B", "")
        orphan = graph.get_entity("B", "")
        assert orphan["is_orphan"] is True
        
        # Now add B as real entity with type
        graph.add_entity("B", "SERVICE", "Real service", 0.9, ["chunk2"])
        
        # B should now be a real entity (note: type is now "SERVICE", not "")
        # The orphan with type="" still exists, but we also have real entity
        real_entity = graph.get_entity("B", "SERVICE")
        assert real_entity is not None
        assert real_entity["is_orphan"] is False
        assert real_entity["description"] == "Real service"
    
    def test_orphan_count_in_stats(self):
        """Test that orphan count is tracked in stats."""
        graph = EntityGraph()
        
        # Add explicit entities
        graph.add_entity("EntityA", "COMPONENT", "Entity A", 0.8, ["chunk1"])
        graph.add_entity("EntityB", "SERVICE", "Entity B", 0.8, ["chunk1"])
        
        # Add relationships that will create orphans (EntityC and EntityD)
        # Note: add_relationship uses empty type for orphans since relationships
        # only specify entity names, not types
        graph.add_relationship("EntityA", "EntityC", "CALLS", "Call", 0.6, ["chunk1"])
        graph.add_relationship("EntityB", "EntityD", "USES", "Usage", 0.7, ["chunk1"])
        
        stats = graph.get_stats()
        
        # Should have 2 orphans (EntityC and EntityD)
        # EntityA and EntityB are real entities (not orphans)
        assert stats["orphan_count"] == 2


class TestUNWINDQueryGeneration:
    """Tests for UNWIND query generation."""
    
    def test_unwind_query_format(self):
        """Test that UNWIND queries are correctly formatted."""
        graph = EntityGraph()
        
        graph.add_entity("Entity1", "TYPE1", "Description", 0.8, ["chunk1"])
        graph.add_entity("Entity2", "TYPE2", "Description", 0.9, ["chunk2"])
        
        entity_query, entity_params, rel_query, rel_params = \
            graph.to_neo4j_batch_queries("doc123")
        
        # Check entity query format
        assert "UNWIND $entities AS entity" in entity_query
        assert "MERGE (e:Entity {name: entity.name})" in entity_query
        assert "SET e.type = entity.type" in entity_query
        
        # Check parameters
        assert "entities" in entity_params
        assert len(entity_params["entities"]) == 2
        assert entity_params["entities"][0]["name"] == "Entity1"
        assert entity_params["entities"][0]["doc_id"] == "doc123"
    
    def test_unwind_relationship_query_format(self):
        """Test that relationship UNWIND queries are correctly formatted."""
        graph = EntityGraph()
        
        graph.add_entity("A", "TYPE", "Desc", 0.8, ["chunk1"])
        graph.add_entity("B", "TYPE", "Desc", 0.9, ["chunk2"])
        graph.add_relationship("A", "B", "DEPENDS_ON", "Dependency", 0.7, ["chunk1"])
        
        _, _, rel_query, rel_params = graph.to_neo4j_batch_queries("doc123")
        
        # Check relationship query format
        assert "UNWIND $relationships AS rel" in rel_query
        assert "MATCH (source:Entity {name: rel.source_name})" in rel_query
        assert "MATCH (target:Entity {name: rel.target_name})" in rel_query
        assert "MERGE (source)-[r:RELATED_TO" in rel_query
        
        # Check parameters
        assert "relationships" in rel_params
        assert len(rel_params["relationships"]) == 1
        assert rel_params["relationships"][0]["source_name"] == "A"
        assert rel_params["relationships"][0]["target_name"] == "B"
    
    def test_phase_version_included(self):
        """Test that phase_version is included in all nodes."""
        from config.settings import settings
        
        graph = EntityGraph()
        graph.add_entity("Entity", "TYPE", "Desc", 0.8, ["chunk1"])
        
        _, entity_params, _, _ = graph.to_neo4j_batch_queries("doc123")
        
        assert entity_params["entities"][0]["phase_version"] == settings.phase_version


class TestGraphStats:
    """Tests for graph statistics."""
    
    def test_get_stats_node_count(self):
        """Test that stats correctly reports node count."""
        graph = EntityGraph()
        
        graph.add_entity("A", "TYPE", "Desc", 0.8, ["chunk1"])
        graph.add_entity("B", "TYPE", "Desc", 0.9, ["chunk2"])
        graph.add_entity("C", "TYPE", "Desc", 0.7, ["chunk3"])
        
        stats = graph.get_stats()
        assert stats["node_count"] == 3
    
    def test_get_stats_edge_count(self):
        """Test that stats correctly reports edge count."""
        graph = EntityGraph()
        
        graph.add_entity("A", "TYPE", "Desc", 0.8, ["chunk1"])
        graph.add_entity("B", "TYPE", "Desc", 0.9, ["chunk2"])
        
        graph.add_relationship("A", "B", "CALLS", "Call", 0.6, ["chunk1"])
        graph.add_relationship("A", "B", "DEPENDS_ON", "Dep", 0.7, ["chunk1"])
        
        stats = graph.get_stats()
        assert stats["edge_count"] == 2
    
    def test_get_stats_total_mentions(self):
        """Test that stats correctly reports total mentions."""
        graph = EntityGraph()
        
        graph.add_entity("A", "TYPE", "Desc", 0.8, ["chunk1"])
        graph.add_entity("A", "TYPE", "Desc", 0.9, ["chunk2"])
        graph.add_entity("B", "TYPE", "Desc", 0.7, ["chunk3"])
        
        stats = graph.get_stats()
        
        # A has 2 mentions, B has 1 mention = 3 total
        assert stats["total_mentions"] == 3
    
    def test_get_stats_avg_mentions(self):
        """Test that stats correctly calculates average mentions per entity."""
        graph = EntityGraph()
        
        graph.add_entity("A", "TYPE", "Desc", 0.8, ["chunk1"])
        graph.add_entity("A", "TYPE", "Desc", 0.9, ["chunk2"])
        graph.add_entity("A", "TYPE", "Desc", 0.7, ["chunk3"])
        graph.add_entity("B", "TYPE", "Desc", 0.8, ["chunk4"])
        
        stats = graph.get_stats()
        
        # A has 3 mentions, B has 1 mention, avg = 4/2 = 2.0
        assert stats["avg_mentions_per_entity"] == pytest.approx(2.0, abs=0.01)
    
    def test_get_stats_max_strength(self):
        """Test that stats correctly reports max relationship strength."""
        graph = EntityGraph()
        
        graph.add_entity("A", "TYPE", "Desc", 0.8, ["chunk1"])
        graph.add_entity("B", "TYPE", "Desc", 0.9, ["chunk2"])
        graph.add_entity("C", "TYPE", "Desc", 0.7, ["chunk3"])
        
        graph.add_relationship("A", "B", "CALLS", "Call", 0.6, ["chunk1"])
        graph.add_relationship("A", "B", "CALLS", "Call", 0.5, ["chunk2"])  # Summed to 1.1
        graph.add_relationship("B", "C", "USES", "Usage", 0.8, ["chunk3"])
        
        stats = graph.get_stats()
        
        # Max strength is 1.1 (A->B)
        assert stats["max_strength"] == pytest.approx(1.1, abs=0.01)


class TestHasEntity:
    """Tests for has_entity method."""
    
    def test_has_entity_returns_true(self):
        """Test that has_entity returns True for existing entities."""
        graph = EntityGraph()
        
        graph.add_entity("Entity", "TYPE", "Desc", 0.8, ["chunk1"])
        
        assert graph.has_entity("Entity", "TYPE") is True
        assert graph.has_entity("ENTITY", "type") is True  # Case-insensitive
    
    def test_has_entity_returns_false(self):
        """Test that has_entity returns False for non-existing entities."""
        graph = EntityGraph()
        
        assert graph.has_entity("NonExistent", "TYPE") is False


class TestGetEntity:
    """Tests for get_entity method."""
    
    def test_get_entity_returns_data(self):
        """Test that get_entity returns entity data."""
        graph = EntityGraph()
        
        graph.add_entity("Entity", "TYPE", "Description", 0.8, ["chunk1"])
        
        entity = graph.get_entity("Entity", "TYPE")
        assert entity is not None
        assert entity["name"] == "Entity"
        assert entity["type"] == "TYPE"
        assert entity["description"] == "Description"
    
    def test_get_entity_returns_none_for_missing(self):
        """Test that get_entity returns None for non-existing entities."""
        graph = EntityGraph()
        
        entity = graph.get_entity("NonExistent", "TYPE")
        assert entity is None


class TestEmptyGraph:
    """Tests for empty graph handling."""
    
    def test_empty_graph_queries(self):
        """Test that empty graph generates valid queries."""
        graph = EntityGraph()
        
        entity_query, entity_params, rel_query, rel_params = \
            graph.to_neo4j_batch_queries("doc123")
        
        assert entity_params["entities"] == []
        assert rel_params["relationships"] == []
    
    def test_empty_graph_stats(self):
        """Test that empty graph returns zero stats."""
        graph = EntityGraph()
        
        stats = graph.get_stats()
        
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["orphan_count"] == 0
        assert stats["total_mentions"] == 0
        assert stats["avg_mentions_per_entity"] == 0
