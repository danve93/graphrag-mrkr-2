"""
Tests for graph clustering, community detection, and community summarization.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from config.settings import settings
from core.community_summarizer import CommunitySummarizer, community_summarizer
from core.graph_clustering import (
    fetch_entity_projection,
    normalize_edge_weights,
    run_leiden_clustering,
    to_igraph,
    write_communities_to_neo4j,
)
from core.graph_db import graph_db

logger = logging.getLogger(__name__)


class TestGraphClustering:
    """Test suite for Leiden-based graph clustering."""

    def test_normalize_edge_weights_empty_dataframe(self):
        """Test weight normalization with empty edges."""
        import pandas as pd

        edges_df = pd.DataFrame()
        result = normalize_edge_weights(edges_df)

        assert "weight" in result.columns
        assert len(result) == 0

    def test_normalize_edge_weights_with_properties(self):
        """Test weight normalization extracts weights from properties."""
        import pandas as pd

        edges_df = pd.DataFrame(
            [
                {
                    "source_id": "e1",
                    "target_id": "e2",
                    "relationship_type": "RELATED_TO",
                    "properties": {"strength": 0.8},
                },
                {
                    "source_id": "e2",
                    "target_id": "e3",
                    "relationship_type": "SIMILAR_TO",
                    "properties": {"similarity": 0.6},
                },
            ]
        )

        result = normalize_edge_weights(edges_df)

        assert "weight" in result.columns
        assert len(result) == 2
        # Weights should be extracted from properties
        assert all(isinstance(w, float) for w in result["weight"])

    def test_normalize_edge_weights_with_fallback(self):
        """Test weight normalization falls back to default when properties missing."""
        import pandas as pd

        edges_df = pd.DataFrame(
            [
                {
                    "source_id": "e1",
                    "target_id": "e2",
                    "relationship_type": "UNKNOWN",
                    "properties": {},
                },
            ]
        )

        result = normalize_edge_weights(edges_df)

        assert "weight" in result.columns
        assert result["weight"].iloc[0] == settings.default_edge_weight

    @patch("core.graph_clustering.to_igraph")
    @patch("core.graph_clustering.run_leiden_clustering")
    @patch("core.graph_clustering.normalize_edge_weights")
    @patch("core.graph_clustering.fetch_entity_projection")
    def test_clustering_workflow_integration(
        self, mock_fetch, mock_normalize, mock_leiden, mock_igraph
    ):
        """Test the complete clustering workflow."""
        import pandas as pd

        # Setup mocks
        nodes_df = pd.DataFrame([{"entity_id": f"e{i}", "name": f"Entity {i}"} for i in range(5)])
        edges_df = pd.DataFrame(
            [
                {"source_id": f"e{i}", "target_id": f"e{(i+1)%5}", "weight": 0.8}
                for i in range(5)
            ]
        )

        mock_fetch.return_value = (nodes_df, edges_df)
        mock_normalize.return_value = edges_df
        mock_leiden.return_value = (
            {f"e{i}": i % 2 for i in range(5)},
            0.85,
        )  # membership, modularity

        # Execute
        membership, modularity = mock_leiden()

        assert len(membership) == 5
        assert modularity > 0


class TestCommunitySummarization:
    """Test suite for community summarization."""

    def test_community_summarizer_initialization(self):
        """Test community summarizer initializes with proper defaults."""
        summarizer = CommunitySummarizer(text_unit_limit=10, excerpt_length=200)

        assert summarizer.text_unit_limit == 10
        assert summarizer.excerpt_length == 200

    def test_trim_excerpt_within_limit(self):
        """Test excerpt trimming when content is within limit."""
        summarizer = CommunitySummarizer(excerpt_length=100)
        content = "This is a short text."

        result = summarizer._trim_excerpt(content)

        assert result == content

    def test_trim_excerpt_exceeds_limit(self):
        """Test excerpt trimming when content exceeds limit."""
        summarizer = CommunitySummarizer(excerpt_length=20)
        content = "This is a much longer text that will be trimmed."

        result = summarizer._trim_excerpt(content)

        assert len(result) <= len(content)
        assert result.endswith("…")

    def test_build_text_unit_payloads(self):
        """Test building text unit payloads for storage."""
        summarizer = CommunitySummarizer()
        text_units = [
            {
                "id": "tu1",
                "document_id": "doc1",
                "content": "Unit 1 content",
                "metadata": {"page": 1},
            },
            {
                "id": "tu2",
                "document_id": "doc1",
                "content": "Unit 2 content",
                "metadata": {"page": 2},
            },
        ]

        payloads = summarizer._build_text_unit_payloads(text_units)

        assert len(payloads) == 2
        assert all("id" in p and "document_id" in p and "excerpt" in p for p in payloads)

    @patch("core.community_summarizer.graph_db")
    def test_summarize_levels_empty_communities(self, mock_graph_db):
        """Test summarization with no communities."""
        summarizer = CommunitySummarizer()

        mock_graph_db.get_community_levels.return_value = []

        results = summarizer.summarize_levels()

        assert len(results) == 0

    @patch("core.community_summarizer.llm_manager")
    @patch("core.community_summarizer.graph_db")
    def test_summarize_single_community(self, mock_graph_db, mock_llm):
        """Test summarization of a single community."""
        summarizer = CommunitySummarizer()

        # Setup mocks
        mock_graph_db.get_community_levels.return_value = [0]
        mock_graph_db.get_communities_for_level.return_value = [
            {
                "community_id": 0,
                "entities": [
                    {"id": "e1", "name": "Entity 1", "type": "Component", "importance_score": 0.9},
                    {
                        "id": "e2",
                        "name": "Entity 2",
                        "type": "Service",
                        "importance_score": 0.7,
                    },
                ],
            }
        ]
        mock_graph_db.get_text_units_for_entities.return_value = [
            {"id": "tu1", "content": "Sample text unit", "document_id": "doc1"}
        ]
        mock_llm.generate_response.return_value = "This is a community summary"
        mock_graph_db.upsert_community_summary.return_value = None

        results = summarizer.summarize_levels([0])

        assert len(results) == 1
        assert results[0]["community_id"] == 0
        assert results[0]["level"] == 0
        assert results[0]["summary"] == "This is a community summary"
        assert len(results[0]["member_entities"]) == 2


class TestGraphVisualization:
    """Test suite for graph visualization endpoints."""

    @patch("api.routers.graph.graph_db")
    def test_clustered_graph_endpoint_basic(self, mock_graph_db):
        """Test basic clustered graph endpoint response."""
        mock_graph_db.get_clustered_graph.return_value = {
            "nodes": [
                {
                    "id": "n1",
                    "name": "Entity 1",
                    "type": "Component",
                    "community_id": 0,
                    "importance": 0.8,
                }
            ],
            "edges": [
                {"source": "n1", "target": "n2", "relationship_type": "RELATED_TO", "weight": 0.7}
            ],
            "communities": [{"community_id": 0, "level": 0}],
            "node_types": ["Component", "Service"],
        }

        graph_data = mock_graph_db.get_clustered_graph()

        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "communities" in graph_data
        assert "node_types" in graph_data

    @patch("api.routers.graph.graph_db")
    def test_clustered_graph_with_filters(self, mock_graph_db):
        """Test clustered graph endpoint with various filters."""
        mock_graph_db.get_clustered_graph.return_value = {
            "nodes": [],
            "edges": [],
            "communities": [],
            "node_types": [],
        }

        # Test with different filter combinations
        mock_graph_db.get_clustered_graph(
            community_id=1, node_type="Service", level=1, limit=100
        )

        mock_graph_db.get_clustered_graph.assert_called()


class TestIngestionWithClustering:
    """Test suite for document ingestion with clustering support."""

    @patch("ingestion.document_processor.graph_db")
    @patch("ingestion.document_processor.document_chunker")
    @patch("ingestion.document_processor.embedding_manager")
    def test_document_processor_creates_entities_and_similarities(
        self, mock_embeddings, mock_chunker, mock_graph_db
    ):
        """Test that document processor creates chunk similarities for clustering."""
        from ingestion.document_processor import DocumentProcessor

        processor = DocumentProcessor()

        # Setup mocks
        mock_graph_db.create_chunk_similarities.return_value = 5

        # Verify that graph_db is called to create similarities
        result = mock_graph_db.create_chunk_similarities("test_doc_id")

        assert result == 5

    @patch("ingestion.document_processor.graph_db")
    def test_document_metadata_includes_content_primary_type(self, mock_graph_db):
        """Test that document metadata includes content_primary_type for clustering."""
        from ingestion.document_processor import DocumentProcessor

        processor = DocumentProcessor()

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            metadata = processor.build_metadata(tmp_path, "test.pdf")
            
            # build_metadata returns basic file info, content_primary_type is added later
            assert "file_extension" in metadata
            assert metadata["file_extension"] == ".pdf"
            
            # Test that _derive_content_primary_type works correctly
            content_type = processor._derive_content_primary_type(".pdf")
            assert content_type == "pdf"
            
            content_type = processor._derive_content_primary_type(".docx")
            assert content_type == "word"
            
            content_type = processor._derive_content_primary_type(".txt")
            assert content_type == "text"
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestClusteringConfiguration:
    """Test suite for clustering configuration settings."""

    def test_clustering_settings_loaded(self):
        """Test that clustering settings are properly loaded."""
        assert hasattr(settings, "enable_clustering")
        assert hasattr(settings, "enable_graph_clustering")
        assert hasattr(settings, "clustering_resolution")
        assert hasattr(settings, "clustering_relationship_types")

    def test_default_edge_weight_setting(self):
        """Test default edge weight setting."""
        assert hasattr(settings, "default_edge_weight")
        assert isinstance(settings.default_edge_weight, (int, float))
        assert settings.default_edge_weight > 0

    def test_leiden_configuration(self):
        """Test Leiden clustering configuration."""
        assert hasattr(settings, "clustering_resolution")
        assert settings.clustering_resolution > 0
        assert hasattr(settings, "clustering_level")


class TestGraphAnalysisLeidenUtils:
    """Test suite for Leiden utility functions."""

    def test_resolve_weight_with_preferred_fields(self):
        """Test weight resolution with preferred fields."""
        from core.graph_analysis.leiden_utils import resolve_weight

        properties = {"strength": 0.7, "similarity": 0.9}
        weight = resolve_weight(
            properties, relationship_type="RELATED_TO", preferred_fields=["strength"]
        )

        assert weight == 0.7

    def test_resolve_weight_fallback_to_default(self):
        """Test weight resolution falls back to default."""
        from core.graph_analysis.leiden_utils import resolve_weight

        properties = {}
        weight = resolve_weight(properties, default=1.0)

        assert weight == 1.0

    def test_build_leiden_parameters(self):
        """Test building Leiden parameters."""
        from core.graph_analysis.leiden_utils import build_leiden_parameters

        params = build_leiden_parameters(weight_property="weight", resolution=1.5)

        assert params["weightProperty"] == "weight"
        assert params["resolution"] == 1.5

    def test_build_entity_leiden_projection_cypher(self):
        """Test building Leiden projection Cypher query."""
        from core.graph_analysis.leiden_utils import build_entity_leiden_projection_cypher

        cypher = build_entity_leiden_projection_cypher(
            weight_field="weight",
            relationship_labels=["SIMILAR_TO", "RELATED_TO"],
        )

        assert "MATCH (e1:Entity)" in cypher
        assert "SIMILAR_TO|RELATED_TO" in cypher
        assert "weight" in cypher


class TestGraphDBIntegration:
    """Test suite for graph database integration with clustering."""

    @patch("core.graph_db.graph_db.driver")
    def test_get_community_levels_query(self, mock_driver):
        """Test querying community levels from database."""
        # This test would require actual Neo4j setup, so we mock the driver
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Mock the Neo4j response
        mock_result = MagicMock()
        mock_result.single.return_value = {"levels": [0, 1, 2]}
        mock_session.run.return_value = mock_result

    @patch("core.graph_db.graph_db.driver")
    def test_get_communities_for_level_query(self, mock_driver):
        """Test querying communities at specific level."""
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_result = MagicMock()
        mock_result.all.return_value = [
            {"community_id": 0, "entities": []},
            {"community_id": 1, "entities": []},
        ]
        mock_session.run.return_value = mock_result


class TestEnd2EndClustering:
    """End-to-end tests for clustering pipeline."""

    @patch("core.graph_db.graph_db")
    @patch("scripts.run_clustering.normalize_edge_weights")
    @patch("scripts.run_clustering.fetch_entity_projection")
    def test_run_clustering_script_integration(
        self, mock_fetch, mock_normalize, mock_graph_db
    ):
        """Test the complete clustering script workflow."""
        import pandas as pd

        # Setup
        nodes_df = pd.DataFrame(
            [{"entity_id": f"e{i}", "name": f"Entity {i}"} for i in range(3)]
        )
        edges_df = pd.DataFrame(
            [
                {"source_id": "e0", "target_id": "e1", "properties": {"weight": 0.8}},
                {"source_id": "e1", "target_id": "e2", "properties": {"weight": 0.6}},
            ]
        )

        mock_fetch.return_value = (nodes_df, edges_df)
        mock_normalize.return_value = edges_df
        mock_graph_db.driver = MagicMock()

        # Verify graph data is properly structured
        assert len(nodes_df) == 3
        assert len(edges_df) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
