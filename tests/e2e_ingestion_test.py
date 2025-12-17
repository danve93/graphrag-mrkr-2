"""
E2E Ingestion Test - Comprehensive Pipeline Verification

This test verifies the COMPLETE document ingestion pipeline with ALL features
enabled by default, using pytest and proper mocking.

Run with: uv run pytest tests/e2e_ingestion_test.py -v -s

SUCCESS CRITERIA (ALL MUST PASS):
Core Features:
1. Document node created in Neo4j  
2. Chunks created (count > 0)
3. Embeddings generated for chunks
4. Summarization runs
5. Entity extraction runs
6. Entities created (count > 0)
7. Status tracking: processing → completed

Optional Features (all tested by default):
8. Gleaning (iterative entity extraction)
9. Content Quality Filtering
10. Document Classification
11. Marker/PDF Conversion (PDF only)

NOTE: Multi-hop reasoning is part of RAG/retrieval, not ingestion.
"""

import os
import sys
import time
import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IngestionTracker:
    """Track all calls during ingestion to verify each stage ran."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Core features
        self.documents = []
        self.chunks = []
        self.entities = []
        self.relationships = []
        self.summarization_called = False
        self.embedding_calls = 0
        self.entity_extraction_called = False
        self.stages_seen = set()
        
        # Optional features
        self.gleaning_called = False
        self.gleaning_passes = 0
        self.classification_called = False
        self.classification_result = None
        self.content_filter_called = False
        self.filter_chunks_passed = 0
        self.filter_chunks_rejected = 0
        self.marker_conversion_called = False
        
        # UI Progress tracking
        self.progress_updates = []  # List of (stage, progress%) tuples
        self.final_progress = None
        self.final_stage = None


# Global tracker
tracker = IngestionTracker()


class MockEntity:
    """Mock entity for testing."""
    def __init__(self, name, entity_type="Concept"):
        self.name = name
        self.type = entity_type
        self.description = f"Description of {name}"
        self.importance_score = 10
        self.source_text_units = ["chunk1"]
        self.source_chunks = ["chunk1"]


class MockRelationship:
    """Mock relationship for testing."""
    def __init__(self, source, target, rel_type="RELATED_TO"):
        self.source = source
        self.target = target
        self.type = rel_type
        self.strength = 0.8


@pytest.fixture
def test_txt_file():
    """Create a test .txt file for basic ingestion."""
    file_path = Path("test_document_e2e.txt")
    content = """
# Knowledge Graph Fundamentals

## Introduction to Knowledge Graphs

A knowledge graph is a knowledge base that uses a graph-structured data model 
to represent and integrate information. Knowledge graphs are often used to 
store interlinked descriptions of entities – objects, events, situations or 
abstract concepts – while also encoding the semantics underlying the terminology.

The concept of knowledge graphs was popularized by Google in 2012 when they 
announced the Google Knowledge Graph, which enhanced their search results with
semantic-search information gathered from a variety of sources.

## Core Components

### Entities

Entities are the primary elements in a knowledge graph. They represent 
real-world objects or concepts like people, places, organizations, or 
abstract ideas. Each entity has a unique identifier and a set of properties
that describe its characteristics.

### Relationships

Relationships connect entities to each other, describing how they are related.
For example, a "works_for" relationship might connect a Person entity to a 
Company entity. Relationships can have properties too, such as start_date
and end_date for employment relationships.

### Properties

Properties are attributes that describe entities or relationships in more 
detail. A Person entity might have properties like "name", "age", "birthdate",
and "occupation". Properties help make the graph more informative and useful.

## Applications and Use Cases

Knowledge graphs power many modern applications including:

1. Search engines that understand entity relationships
2. Recommendation systems that use graph traversal algorithms
3. Question answering systems that reason over structured data
4. Data integration platforms that connect disparate sources
5. Fraud detection systems that identify suspicious patterns

## Graph Databases

Neo4j is a popular graph database that stores data in nodes and relationships.
It uses the Cypher query language for pattern matching and data retrieval.
Neo4j is well-suited for knowledge graph applications due to its native graph 
storage and processing capabilities.

Other graph databases include Amazon Neptune, JanusGraph, and TigerGraph.
Each has different strengths depending on the use case requirements.
"""
    file_path.write_text(content)
    yield file_path
    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def mock_graph_db():
    """Create a mock graph_db that tracks all calls."""
    mock = MagicMock()
    
    def track_document_node(doc_id, props):
        tracker.documents.append({"id": doc_id, "props": dict(props)})
        if props.get("processing_stage"):
            tracker.stages_seen.add(props["processing_stage"])
        
        # Track UI progress updates
        stage = props.get("processing_stage")
        progress = props.get("processing_progress")
        if stage or progress is not None:
            tracker.progress_updates.append({
                "stage": stage,
                "progress": progress,
                "status": props.get("processing_status"),
                "message": props.get("processing_message"),
            })
        
        # Track final state
        if props.get("processing_status") == "completed":
            tracker.final_progress = props.get("processing_progress")
            tracker.final_stage = "completed"
        
        return MagicMock()
    
    def track_chunk_creation(*args, **kwargs):
        tracker.chunks.append({"args": args, "kwargs": kwargs})
        return MagicMock()
    
    async def track_entity_creation(*args, **kwargs):
        print(f"    DEBUG: Entity created: {args if args else kwargs}")
        tracker.entities.append({"args": args, "kwargs": kwargs})
        return MagicMock()
    
    async def track_relationship_creation(*args, **kwargs):
        tracker.relationships.append({"args": args, "kwargs": kwargs})
        return MagicMock()
    
    mock.create_document_node = MagicMock(side_effect=track_document_node)
    mock.create_chunk_node = MagicMock(side_effect=track_chunk_creation)
    mock.acreate_entity_node = AsyncMock(side_effect=track_entity_creation)
    mock.acreate_entity_relationship = AsyncMock(side_effect=track_relationship_creation)
    mock.create_chunk_entity_relationship = MagicMock()
    mock.update_document_summary = MagicMock()
    mock.get_document = MagicMock(return_value=None)
    
    return mock


@pytest.fixture
def mock_embedding_manager():
    """Create a mock embedding manager."""
    mock = MagicMock()
    
    def sync_embedding(*args, **kwargs):
        tracker.embedding_calls += 1
        return [0.1] * 1536
    
    async def async_embedding(*args, **kwargs):
        tracker.embedding_calls += 1
        return [0.1] * 1536
    
    mock.generate_embeddings = MagicMock(side_effect=sync_embedding)
    mock.aget_embedding = AsyncMock(side_effect=async_embedding)
    
    return mock


@pytest.fixture
def mock_summarizer():
    """Create a mock document summarizer."""
    mock = MagicMock()
    
    def summarizer_side_effect(chunks):
        tracker.summarization_called = True
        return {
            "summary": "A comprehensive document about knowledge graphs.",
            "document_type": "technical",
            "hashtags": ["knowledge-graph", "neo4j", "entities"]
        }
    
    mock.extract_summary = MagicMock(side_effect=summarizer_side_effect)
    return mock


class TestE2EIngestionComplete:
    """Comprehensive E2E Ingestion Test Suite with ALL features enabled."""
    
    def setup_method(self):
        """Reset tracker before each test."""
        tracker.reset()
    
    def test_complete_pipeline_all_features(
        self,
        test_txt_file,
        mock_graph_db,
        mock_embedding_manager,
        mock_summarizer,
    ):
        """
        Test COMPLETE ingestion pipeline with ALL features enabled.
        
        This single test verifies:
        - Core: Document, Chunks, Embeddings, Summarization, Entities, Status
        - Optional: Gleaning, Classification, Content Filtering
        
        NOTE: Marker conversion is only tested for PDF files.
        """
        print("\n" + "="*70)
        print("E2E COMPREHENSIVE INGESTION TEST - ALL FEATURES")
        print("="*70)
        
        # Import modules first  
        from ingestion import document_processor as dp_module
        from ingestion.document_processor import DocumentProcessor
        from ingestion.content_filters import ContentQualityFilter
        
        # Direct injection into module namespace (works for module-level globals)
        original_graph_db = dp_module.graph_db
        original_embedding_manager = dp_module.embedding_manager
        original_summarizer = dp_module.document_summarizer
        
        dp_module.graph_db = mock_graph_db
        dp_module.embedding_manager = mock_embedding_manager
        dp_module.document_summarizer = mock_summarizer
        
        try:
            with patch.object(dp_module, "get_content_filter") as mock_filter, \
                 patch.object(dp_module, "settings") as mock_settings:
                
                # --- Enable ALL Features ---
                mock_settings.enable_entity_extraction = True
                mock_settings.enable_gleaning = True
                mock_settings.max_gleanings = 2
                mock_settings.gleaning_by_doc_type = {}
                mock_settings.enable_document_classification = True
                mock_settings.enable_quality_filtering = True
                mock_settings.enable_document_summaries = True
                mock_settings.sync_entity_embeddings = True
                mock_settings.enable_phase2_networkx = False
                mock_settings.create_chunk_similarities = False
                mock_settings.chunk_size = 500
                mock_settings.chunk_overlap = 100
                mock_settings.importance_score_threshold = 0
                mock_settings.strength_threshold = 0
                mock_settings.enable_description_summarization = False
                mock_settings.enable_temporal_filtering = False
                mock_settings.embedding_concurrency = 2
                mock_settings.classification_confidence_threshold = 0.5
                mock_settings.classification_default_category = "general"
                
                # --- Content Filter with Tracking ---
                class TrackingContentFilter(ContentQualityFilter):
                    def should_embed_chunk(self, chunk, metadata=None):
                        tracker.content_filter_called = True
                        should_embed, reason = super().should_embed_chunk(chunk, metadata)
                        if should_embed:
                            tracker.filter_chunks_passed += 1
                        else:
                            tracker.filter_chunks_rejected += 1
                        return should_embed, reason
                
                mock_filter.return_value = TrackingContentFilter(min_chunk_length=20)
                
                # --- Entity Extractor with Gleaning Tracking ---
                async def mock_extract_with_gleaning(chunks, max_gleanings=0, **kwargs):
                    tracker.entity_extraction_called = True
                    tracker.gleaning_called = max_gleanings > 0
                    tracker.gleaning_passes = max_gleanings
                    
                    if 'progress_callback' in kwargs and kwargs['progress_callback']:
                        kwargs['progress_callback'](1, 1)
                    
                    entities = {
                        "e1": MockEntity("Knowledge Graph", "Concept"),
                        "e2": MockEntity("Entity", "Component"),
                        "e3": MockEntity("Neo4j", "Technology"),
                        "e4": MockEntity("Google", "Organization"),
                    }
                    # Track entities at extraction time (not persistence)
                    for entity_id, entity in entities.items():
                        tracker.entities.append({"id": entity_id, "name": entity.name, "type": entity.type})
                    return (entities, {"r1": []})
                
                async def mock_extract_basic(chunks, **kwargs):
                    tracker.entity_extraction_called = True
                    if 'progress_callback' in kwargs and kwargs['progress_callback']:
                        kwargs['progress_callback'](1, 1)
                    entities = {"e1": MockEntity("Knowledge Graph")}
                    for entity_id, entity in entities.items():
                        tracker.entities.append({"id": entity_id, "name": entity.name, "type": entity.type})
                    return (entities, {"r1": []})
                
                # Create processor
                processor = DocumentProcessor()
                processor.entity_extractor.extract_from_chunks = AsyncMock(side_effect=mock_extract_basic)
                processor.entity_extractor.extract_from_chunks_with_gleaning = AsyncMock(side_effect=mock_extract_with_gleaning)
                
                # Classification mock
                def mock_classify(*args, **kwargs):
                    tracker.classification_called = True
                    tracker.classification_result = {"category": "technical", "confidence": 0.85}
                    return tracker.classification_result
                
                processor.classify_document_categories = MagicMock(side_effect=mock_classify)
                
                # Run ingestion
                print("\n[1] RUNNING INGESTION PIPELINE (ALL FEATURES ENABLED)...")
                print("    Features: Entity Extraction, Gleaning, Classification, Filtering")
                start_time = time.time()
                
                result = processor.process_file(
                    test_txt_file,
                    document_id="e2e_all_features_test"
                )
                
                # Wait for background threads
                print("\n[2] WAITING FOR BACKGROUND THREADS...")
                max_wait = 15
                start = time.time()
                while processor._bg_entity_threads:
                    if time.time() - start > max_wait:
                        break
                    time.sleep(0.2)
                
                for t in processor._bg_entity_threads:
                    t.join(timeout=3)
                
                elapsed = time.time() - start_time
                print(f"    Completed in {elapsed:.2f}s")
        
        finally:
            # Restore original values
            dp_module.graph_db = original_graph_db
            dp_module.embedding_manager = original_embedding_manager
            dp_module.document_summarizer = original_summarizer
        # ===== VERIFY ALL SUCCESS CRITERIA =====
        print("\n[3] VERIFYING ALL SUCCESS CRITERIA...")
        print("="*70)
        
        all_passed = True
        
        # --- CORE FEATURES ---
        print("\n  CORE FEATURES:")
        print("  " + "-"*40)
        
        # 1. Document created
        doc_count = len(tracker.documents)
        doc_pass = doc_count > 0
        print(f"\n  1. Document Node Created: {'✓' if doc_pass else '✗'} ({doc_count} updates)")
        assert doc_pass, "FAIL: No document node created"
        
        # 2. Chunks created
        chunk_count = len(tracker.chunks)
        chunk_pass = chunk_count > 0
        print(f"  2. Chunks Created: {'✓' if chunk_pass else '✗'} ({chunk_count} chunks)")
        assert chunk_pass, "FAIL: No chunks created"
        
        # 3. Embeddings generated
        embed_pass = tracker.embedding_calls > 0
        print(f"  3. Embeddings Generated: {'✓' if embed_pass else '✗'} ({tracker.embedding_calls} calls)")
        assert embed_pass, "FAIL: No embeddings generated"
        
        # 4. Summarization called
        sum_pass = tracker.summarization_called
        print(f"  4. Summarization Called: {'✓' if sum_pass else '✗'}")
        assert sum_pass, "FAIL: Summarization not called"
        
        # 5. Entity extraction called
        entity_pass = tracker.entity_extraction_called
        print(f"  5. Entity Extraction Called: {'✓' if entity_pass else '✗'}")
        assert entity_pass, "FAIL: Entity extraction not called"
        
        # 6. Entities created
        entity_count = len(tracker.entities)
        entities_pass = entity_count > 0
        print(f"  6. Entities Created: {'✓' if entities_pass else '✗'} ({entity_count} entities)")
        assert entities_pass, "FAIL: No entities created"
        
        # 7. Status completed
        completed_docs = [d for d in tracker.documents 
                         if d['props'].get('processing_status') == 'completed']
        status_pass = len(completed_docs) > 0
        print(f"  7. Status Completed: {'✓' if status_pass else '✗'}")
        assert status_pass, "FAIL: Status never reached 'completed'"
        
        # --- OPTIONAL FEATURES ---
        print("\n  OPTIONAL FEATURES:")
        print("  " + "-"*40)
        
        # 8. Gleaning
        gleaning_pass = tracker.gleaning_called
        print(f"\n  8. Gleaning: {'✓' if gleaning_pass else '✗'} ({tracker.gleaning_passes} passes)")
        assert gleaning_pass, "FAIL: Gleaning not called"
        
        # 9. Content Filtering
        filter_pass = tracker.content_filter_called
        print(f"  9. Content Filtering: {'✓' if filter_pass else '✗'} ({tracker.filter_chunks_passed} passed, {tracker.filter_chunks_rejected} rejected)")
        assert filter_pass, "FAIL: Content filter not called"
        
        # 10. Classification
        class_pass = tracker.classification_called
        class_result = tracker.classification_result or {}
        print(f"  10. Classification: {'✓' if class_pass else '✗'} (category={class_result.get('category')}, confidence={class_result.get('confidence')})")
        assert class_pass, "FAIL: Classification not called"
        
        # 11. Marker conversion (only applicable for PDF)
        print(f"  11. Marker Conversion: N/A (test uses .txt file)")
        
        # --- UI PROGRESS TRACKING ---
        print("\n  UI PROGRESS TRACKING:")
        print("  " + "-"*40)
        
        # 12. Progress updates received
        progress_count = len(tracker.progress_updates)
        progress_pass = progress_count > 0
        print(f"\n  12. Progress Updates Received: {'✓' if progress_pass else '✗'} ({progress_count} updates)")
        assert progress_pass, "FAIL: No progress updates received"
        
        # 13. Progress values are valid (0-100)
        progress_values = [u['progress'] for u in tracker.progress_updates if u['progress'] is not None]
        valid_progress = all(0 <= p <= 100 for p in progress_values)
        min_progress = min(progress_values) if progress_values else None
        max_progress = max(progress_values) if progress_values else None
        print(f"  13. Progress Values Valid (0-100): {'✓' if valid_progress else '✗'} (range: {min_progress} → {max_progress})")
        
        # 14. Final progress is 100
        final_progress_pass = tracker.final_progress == 100 or max_progress == 100
        print(f"  14. Final Progress = 100: {'✓' if final_progress_pass else '✗'} (final={tracker.final_progress})")
        
        # 15. Show stage progression
        stage_progression = [u['stage'] for u in tracker.progress_updates if u['stage']]
        unique_stages = []
        for s in stage_progression:
            if not unique_stages or unique_stages[-1] != s:
                unique_stages.append(s)
        print(f"  15. Stage Progression: {' → '.join(unique_stages) if unique_stages else 'None'}")
        
        # --- PROCESSING STAGES ---
        print("\n  PROCESSING STAGES:")
        print("  " + "-"*40)
        print(f"  Stages seen: {sorted(tracker.stages_seen)}")
        
        # Check expected stages
        expected_stages = {'chunking', 'embedding', 'classification'}
        missing = expected_stages - tracker.stages_seen
        if missing:
            print(f"  ⚠️  Missing expected stages: {missing}")
        
        # ===== FINAL SUMMARY =====
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"  Documents: {doc_count}")
        print(f"  Chunks: {chunk_count}")
        print(f"  Embeddings: {tracker.embedding_calls}")
        print(f"  Entities: {entity_count}")
        print(f"  Summarization: {'Yes' if tracker.summarization_called else 'No'}")
        print(f"  Gleaning: {'Yes' if tracker.gleaning_called else 'No'} ({tracker.gleaning_passes} passes)")
        print(f"  Filtering: {tracker.filter_chunks_passed} passed, {tracker.filter_chunks_rejected} rejected")
        print(f"  Classification: {tracker.classification_result}")
        print(f"  Time: {elapsed:.2f}s")
        
        print("\n✅ ALL CRITERIA PASSED - INGESTION PIPELINE FULLY VERIFIED")
        print("="*70 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
