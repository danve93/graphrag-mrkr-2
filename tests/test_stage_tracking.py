"""
Tests for Milestone 3: Stage Tracking Enhancement.

Verifies that stage tracking includes timing metadata and that
stages are emitted correctly in the streaming response.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from rag.graph_rag import GraphRAG


class TestStageTimingMetadata:
    """Test that stages include timing metadata."""

    @pytest.mark.asyncio
    async def test_stages_include_timing_metadata(self):
        """Verify stages include duration_ms, timestamp, and metadata."""
        graph_rag = GraphRAG()
        
        with patch("rag.nodes.query_analysis.analyze_query") as mock_analyze, \
             patch("rag.nodes.retrieval.retrieve_documents") as mock_retrieve, \
             patch("rag.nodes.graph_reasoning.reason_with_graph") as mock_reason, \
             patch("rag.nodes.generation.generate_response") as mock_generate:
            
            # Setup mocks
            mock_analyze.return_value = {"query_type": "factual"}
            mock_retrieve.return_value = [{"chunk_id": "1", "content": "test"}]
            mock_reason.return_value = [{"chunk_id": "1", "content": "test"}]
            mock_generate.return_value = {
                "response": "Test response",
                "sources": [],
                "metadata": {}
            }
            
            # Execute query
            result = graph_rag.query(
                user_query="What is VxRail?",
                retrieval_mode="hybrid",
                top_k=5,
                temperature=0.7,
                use_multi_hop=False,
            )
            
            # Verify stages were tracked
            assert "stages" in result, "Result should include stages"
            stages = result["stages"]
            assert len(stages) > 0, "Should have at least one stage"
            
            # Check each stage has required fields
            for stage in stages:
                assert isinstance(stage, dict), "Stage should be a dict"
                assert "name" in stage, "Stage should have name"
                assert "duration_ms" in stage, "Stage should have duration_ms"
                assert "timestamp" in stage, "Stage should have timestamp"
                
                # Verify types
                assert isinstance(stage["name"], str), "Stage name should be string"
                assert isinstance(stage["duration_ms"], int), "Duration should be integer"
                assert isinstance(stage["timestamp"], float), "Timestamp should be float"
                
                # Verify values are reasonable
                assert stage["duration_ms"] >= 0, "Duration should be non-negative"
                assert stage["timestamp"] > 0, "Timestamp should be positive"

    @pytest.mark.asyncio
    async def test_stages_tracked_in_order(self):
        """Verify stages are tracked in execution order."""
        graph_rag = GraphRAG()
        
        with patch("rag.nodes.query_analysis.analyze_query") as mock_analyze, \
             patch("rag.nodes.retrieval.retrieve_documents") as mock_retrieve, \
             patch("rag.nodes.graph_reasoning.reason_with_graph") as mock_reason, \
             patch("rag.nodes.generation.generate_response") as mock_generate:
            
            mock_analyze.return_value = {"query_type": "factual"}
            mock_retrieve.return_value = []
            mock_reason.return_value = []
            mock_generate.return_value = {
                "response": "Test",
                "sources": [],
                "metadata": {}
            }
            
            result = graph_rag.query(
                user_query="Test query",
                retrieval_mode="hybrid",
                top_k=5,
            )
            
            stages = result.get("stages", [])
            assert len(stages) == 4, "Should have 4 stages"
            
            # Verify stage order
            expected_order = ["query_analysis", "retrieval", "graph_reasoning", "generation"]
            actual_order = [stage["name"] for stage in stages]
            assert actual_order == expected_order, f"Stage order mismatch: {actual_order}"
            
            # Verify timestamps are increasing
            timestamps = [stage["timestamp"] for stage in stages]
            for i in range(len(timestamps) - 1):
                assert timestamps[i] <= timestamps[i + 1], "Timestamps should be in order"

    @pytest.mark.asyncio
    async def test_stage_metadata_includes_relevant_info(self):
        """Verify stage metadata includes relevant information."""
        graph_rag = GraphRAG()
        
        with patch("rag.nodes.query_analysis.analyze_query") as mock_analyze, \
             patch("rag.nodes.retrieval.retrieve_documents") as mock_retrieve, \
             patch("rag.nodes.graph_reasoning.reason_with_graph") as mock_reason, \
             patch("rag.nodes.generation.generate_response") as mock_generate:
            
            mock_analyze.return_value = {"query_type": "factual"}
            mock_retrieve.return_value = [{"chunk_id": str(i)} for i in range(5)]
            mock_reason.return_value = [{"chunk_id": str(i)} for i in range(3)]
            mock_generate.return_value = {
                "response": "This is a test response with some length",
                "sources": [],
                "metadata": {}
            }
            
            result = graph_rag.query(
                user_query="Test query",
                retrieval_mode="hybrid",
                top_k=5,
            )
            
            stages = result.get("stages", [])
            
            # Find retrieval stage and check metadata
            retrieval_stage = next((s for s in stages if s["name"] == "retrieval"), None)
            assert retrieval_stage is not None, "Should have retrieval stage"
            assert "metadata" in retrieval_stage, "Retrieval stage should have metadata"
            assert "chunks_retrieved" in retrieval_stage["metadata"], "Should track chunks retrieved"
            assert retrieval_stage["metadata"]["chunks_retrieved"] == 5, "Should retrieve 5 chunks"
            
            # Find graph_reasoning stage and check metadata
            reasoning_stage = next((s for s in stages if s["name"] == "graph_reasoning"), None)
            assert reasoning_stage is not None, "Should have graph_reasoning stage"
            assert "metadata" in reasoning_stage, "Graph reasoning stage should have metadata"
            assert "context_items" in reasoning_stage["metadata"], "Should track context items"
            assert reasoning_stage["metadata"]["context_items"] == 3, "Should have 3 context items"
            
            # Find generation stage and check metadata
            generation_stage = next((s for s in stages if s["name"] == "generation"), None)
            assert generation_stage is not None, "Should have generation stage"
            assert "metadata" in generation_stage, "Generation stage should have metadata"
            assert "response_length" in generation_stage["metadata"], "Should track response length"
            assert generation_stage["metadata"]["response_length"] > 0, "Response length should be positive"


class TestStageStreamingFormat:
    """Test that stages are properly formatted for streaming."""

    def test_legacy_string_format_still_supported(self):
        """Verify backwards compatibility with old string-based stage format."""
        from api.routers.chat import stream_response_generator
        import asyncio
        
        # Create a result with old-style string stages
        result = {
            "response": "Test response",
            "stages": ["query_analysis", "retrieval"],
            "sources": [],
        }
        
        async def collect_stages():
            stages_emitted = []
            async for event in stream_response_generator(
                result,
                session_id="test123",
                user_query="test",
                context_documents=[],
                context_document_labels=[],
                context_hashtags=[],
                chat_history=[],
                stage_updates=result["stages"],
            ):
                if '"type": "stage"' in event:
                    stages_emitted.append(event)
            return stages_emitted
        
        stages = asyncio.run(collect_stages())
        assert len(stages) == 2, "Should emit 2 stages"

    def test_new_dict_format_includes_timing(self):
        """Verify new dict format includes timing information in stream."""
        from api.routers.chat import stream_response_generator
        import asyncio
        import json
        
        # Create a result with new-style dict stages
        result = {
            "response": "Test response",
            "stages": [
                {
                    "name": "query_analysis",
                    "duration_ms": 150,
                    "timestamp": time.time(),
                    "metadata": {},
                },
                {
                    "name": "retrieval",
                    "duration_ms": 350,
                    "timestamp": time.time(),
                    "metadata": {"chunks_retrieved": 5},
                },
            ],
            "sources": [],
        }
        
        async def collect_stage_data():
            stage_data = []
            async for event in stream_response_generator(
                result,
                session_id="test123",
                user_query="test",
                context_documents=[],
                context_document_labels=[],
                context_hashtags=[],
                chat_history=[],
                stage_updates=result["stages"],
            ):
                if '"type": "stage"' in event:
                    # Parse SSE format: "data: {...}\n\n"
                    json_str = event.replace("data: ", "").strip()
                    data = json.loads(json_str)
                    stage_data.append(data)
            return stage_data
        
        stages = asyncio.run(collect_stage_data())
        assert len(stages) == 2, "Should emit 2 stages"
        
        # Check first stage
        assert stages[0]["content"] == "query_analysis"
        assert stages[0]["duration_ms"] == 150
        assert "timestamp" in stages[0]
        
        # Check second stage
        assert stages[1]["content"] == "retrieval"
        assert stages[1]["duration_ms"] == 350
        assert stages[1]["metadata"]["chunks_retrieved"] == 5


class TestStageTimingAccuracy:
    """Test that timing measurements are accurate."""

    @pytest.mark.asyncio
    async def test_stage_durations_are_positive(self):
        """Verify all stage durations are positive integers."""
        graph_rag = GraphRAG()
        
        with patch("rag.nodes.query_analysis.analyze_query") as mock_analyze, \
             patch("rag.nodes.retrieval.retrieve_documents") as mock_retrieve, \
             patch("rag.nodes.graph_reasoning.reason_with_graph") as mock_reason, \
             patch("rag.nodes.generation.generate_response") as mock_generate:
            
            mock_analyze.return_value = {"query_type": "factual"}
            mock_retrieve.return_value = []
            mock_reason.return_value = []
            mock_generate.return_value = {
                "response": "Test",
                "sources": [],
                "metadata": {}
            }
            
            result = graph_rag.query(
                user_query="Test",
                retrieval_mode="hybrid",
                top_k=5,
            )
            
            stages = result.get("stages", [])
            for stage in stages:
                duration = stage["duration_ms"]
                assert isinstance(duration, int), f"{stage['name']} duration should be int"
                assert duration >= 0, f"{stage['name']} duration should be non-negative"

    @pytest.mark.asyncio
    async def test_total_pipeline_duration_is_sum_of_stages(self):
        """Verify total duration roughly equals sum of stage durations."""
        graph_rag = GraphRAG()
        
        with patch("rag.nodes.query_analysis.analyze_query") as mock_analyze, \
             patch("rag.nodes.retrieval.retrieve_documents") as mock_retrieve, \
             patch("rag.nodes.graph_reasoning.reason_with_graph") as mock_reason, \
             patch("rag.nodes.generation.generate_response") as mock_generate:
            
            # Add small delays to simulate work
            def analyze_with_delay(*args, **kwargs):
                time.sleep(0.01)
                return {"query_type": "factual"}
            
            def retrieve_with_delay(*args, **kwargs):
                time.sleep(0.01)
                return []
            
            def reason_with_delay(*args, **kwargs):
                time.sleep(0.01)
                return []
            
            def generate_with_delay(*args, **kwargs):
                time.sleep(0.01)
                return {"response": "Test", "sources": [], "metadata": {}}
            
            mock_analyze.side_effect = analyze_with_delay
            mock_retrieve.side_effect = retrieve_with_delay
            mock_reason.side_effect = reason_with_delay
            mock_generate.side_effect = generate_with_delay
            
            start = time.time()
            result = graph_rag.query(
                user_query="Test",
                retrieval_mode="hybrid",
                top_k=5,
            )
            total_elapsed = time.time() - start
            
            stages = result.get("stages", [])
            stage_durations_sum = sum(s["duration_ms"] for s in stages)
            
            # Total elapsed should be close to sum of stages (within 50ms tolerance for overhead)
            assert abs((total_elapsed * 1000) - stage_durations_sum) < 50, \
                f"Total elapsed {total_elapsed*1000:.0f}ms should be close to sum of stages {stage_durations_sum}ms"
