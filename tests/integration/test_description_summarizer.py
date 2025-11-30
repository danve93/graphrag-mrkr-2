"""
Unit tests for description_summarizer.py (Phase 4)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.description_summarizer import DescriptionSummarizer, SummarizationResult
from config.settings import settings


class TestDescriptionSummarizerInitialization:
    """Test DescriptionSummarizer initialization."""
    
    def test_summarizer_creates_empty_cache(self):
        """Test DescriptionSummarizer creates empty cache on init."""
        summarizer = DescriptionSummarizer()
        
        assert summarizer._summary_cache == {}
        assert isinstance(summarizer._stats, dict)
    
    def test_summarizer_initializes_stats(self):
        """Test statistics are initialized to zero."""
        summarizer = DescriptionSummarizer()
        
        assert summarizer._stats["entities_summarized"] == 0
        assert summarizer._stats["relationships_summarized"] == 0
        assert summarizer._stats["cache_hits"] == 0
        assert summarizer._stats["cache_misses"] == 0
        assert summarizer._stats["summarization_errors"] == 0


class TestShouldSummarize:
    """Test _should_summarize criteria."""
    
    def test_should_summarize_with_sufficient_criteria(self):
        """Test _should_summarize returns True when criteria met."""
        summarizer = DescriptionSummarizer()
        
        # ≥3 mentions, ≥200 chars → should summarize
        description = "a" * 250
        assert summarizer._should_summarize(description, mention_count=3) == True
    
    def test_should_not_summarize_too_few_mentions(self):
        """Test _should_summarize returns False with <3 mentions."""
        summarizer = DescriptionSummarizer()
        
        description = "a" * 250
        assert summarizer._should_summarize(description, mention_count=2) == False
    
    def test_should_not_summarize_too_short(self):
        """Test _should_summarize returns False with short description."""
        summarizer = DescriptionSummarizer()
        
        description = "a" * 150  # < 200
        assert summarizer._should_summarize(description, mention_count=5) == False
    
    def test_should_not_summarize_empty_description(self):
        """Test _should_summarize returns False with empty description."""
        summarizer = DescriptionSummarizer()
        
        assert summarizer._should_summarize("", mention_count=5) == False
        assert summarizer._should_summarize("   ", mention_count=5) == False
        assert summarizer._should_summarize(None, mention_count=5) == False


class TestCacheBehavior:
    """Test cache hit and miss behavior."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_no_llm_call(self):
        """Test cache returns cached summary without LLM call."""
        # Mock settings BEFORE creating summarizer
        original_enabled = settings.enable_description_summarization
        original_cache = settings.summarization_cache_enabled
        original_min_mentions = settings.summarization_min_mentions
        original_min_length = settings.summarization_min_length
        settings.enable_description_summarization = True
        settings.summarization_cache_enabled = True
        settings.summarization_min_mentions = 2  # Lower threshold for test
        settings.summarization_min_length = 50  # Lower threshold for test
        
        try:
            summarizer = DescriptionSummarizer()
            
            # Manually populate cache
            description = "Admin Panel provides user management. Admin Panel has RBAC."
            desc_hash = summarizer._compute_description_hash(description)
            summarizer._summary_cache[desc_hash] = "Cached summary"
            
            # Summarize (should hit cache)
            results = await summarizer.summarize_entity_descriptions([
                ("ADMIN PANEL", description, 3)
            ])
            
            assert len(results) == 1
            assert results[0].summarized_description == "Cached summary"
            assert summarizer._stats["cache_hits"] == 1
            assert summarizer._stats["cache_misses"] == 0
        finally:
            settings.enable_description_summarization = original_enabled
            settings.summarization_cache_enabled = original_cache
            settings.summarization_min_mentions = original_min_mentions
            settings.summarization_min_length = original_min_length
    
    @pytest.mark.asyncio
    async def test_cache_miss_triggers_llm_call(self):
        """Test cache miss triggers LLM call and caches result."""
        # Mock settings BEFORE creating summarizer
        original_enabled = settings.enable_description_summarization
        original_cache = settings.summarization_cache_enabled
        original_min_mentions = settings.summarization_min_mentions
        original_min_length = settings.summarization_min_length
        settings.enable_description_summarization = True
        settings.summarization_cache_enabled = True
        settings.summarization_min_mentions = 2
        settings.summarization_min_length = 50
        
        try:
            # Mock LLM response
            mock_response = "1. Web-based administration interface"
            
            with patch('core.llm.llm_manager.generate_response', return_value=mock_response):
                summarizer = DescriptionSummarizer()
                description = "Admin Panel provides user management. Admin Panel has RBAC."
                
                # First call (cache miss)
                results = await summarizer.summarize_entity_descriptions([
                    ("ADMIN PANEL", description, 3)
                ])
                
                assert len(results) == 1
                assert summarizer._stats["cache_misses"] == 1
                
                # Verify result cached
                desc_hash = summarizer._compute_description_hash(description)
                assert desc_hash in summarizer._summary_cache
        finally:
            settings.enable_description_summarization = original_enabled
            settings.summarization_cache_enabled = original_cache
            settings.summarization_min_mentions = original_min_mentions
            settings.summarization_min_length = original_min_length


class TestResponseParsing:
    """Test _parse_summarization_response."""
    
    def test_parse_valid_numbered_format(self):
        """Test parsing valid numbered summaries."""
        summarizer = DescriptionSummarizer()
        
        response = """
1. Web-based administration interface
2. Database for user credentials
3. Caching service for sessions
"""
        
        items = [
            ("ADMIN PANEL", "desc1", 3),
            ("USER DATABASE", "desc2", 4),
            ("REDIS CACHE", "desc3", 3),
        ]
        
        summaries = summarizer._parse_summarization_response(response, items)
        
        assert len(summaries) == 3
        assert summaries[0] == "Web-based administration interface"
        assert summaries[1] == "Database for user credentials"
        assert summaries[2] == "Caching service for sessions"
    
    def test_parse_with_count_mismatch_pads(self):
        """Test parser pads with original descriptions on count mismatch."""
        summarizer = DescriptionSummarizer()
        
        # Response has only 2 summaries, but 3 items provided
        response = """
1. First summary
2. Second summary
"""
        
        items = [
            ("E1", "original_desc_1", 3),
            ("E2", "original_desc_2", 3),
            ("E3", "original_desc_3", 3),
        ]
        
        summaries = summarizer._parse_summarization_response(response, items)
        
        # Should pad third summary with original description
        assert len(summaries) == 3
        assert summaries[0] == "First summary"
        assert summaries[1] == "Second summary"
        assert summaries[2] == "original_desc_3"  # Padded
    
    def test_parse_empty_response(self):
        """Test parser handles empty response."""
        summarizer = DescriptionSummarizer()
        
        response = ""
        items = [
            ("E1", "desc1", 3),
            ("E2", "desc2", 3),
        ]
        
        summaries = summarizer._parse_summarization_response(response, items)
        
        # Should pad all with original descriptions
        assert len(summaries) == 2
        assert summaries[0] == "desc1"
        assert summaries[1] == "desc2"
    
    def test_parse_with_extra_whitespace(self):
        """Test parser handles extra whitespace."""
        summarizer = DescriptionSummarizer()
        
        response = """
  1.   First summary with spaces   
  
  2.   Second summary   
"""
        
        items = [
            ("E1", "desc1", 3),
            ("E2", "desc2", 3),
        ]
        
        summaries = summarizer._parse_summarization_response(response, items)
        
        assert len(summaries) == 2
        assert summaries[0] == "First summary with spaces"
        assert summaries[1] == "Second summary"


class TestCompressionRatio:
    """Test compression ratio calculation."""
    
    def test_compression_ratio_calculation(self):
        """Test SummarizationResult compression ratio."""
        result = SummarizationResult(
            entity_name="TEST",
            original_description="a" * 400,
            summarized_description="b" * 200,
            original_length=400,
            summarized_length=200,
            compression_ratio=0.5
        )
        
        assert result.compression_ratio == 0.5  # 50% compression
    
    def test_compression_ratio_no_compression(self):
        """Test compression ratio when no compression occurred."""
        result = SummarizationResult(
            entity_name="TEST",
            original_description="a" * 100,
            summarized_description="a" * 100,
            original_length=100,
            summarized_length=100,
            compression_ratio=1.0
        )
        
        assert result.compression_ratio == 1.0  # No compression


class TestErrorHandling:
    """Test error handling and fallback."""
    
    @pytest.mark.asyncio
    async def test_error_fallback_to_original(self):
        """Test summarizer falls back to original on error."""
        summarizer = DescriptionSummarizer()
        
        # Mock settings
        original_enabled = settings.enable_description_summarization
        settings.enable_description_summarization = True
        
        try:
            # Mock LLM to raise exception
            with patch('core.llm.llm_manager.generate_response', side_effect=Exception("LLM error")):
                original_desc = "Original description" * 20  # Make it long enough
                
                # Summarize (should fallback to original)
                results = await summarizer.summarize_entity_descriptions([
                    ("ENTITY", original_desc, 3)
                ])
                
                assert len(results) == 1
                assert results[0].summarized_description == original_desc  # Unchanged
                assert results[0].error is not None
                assert "LLM error" in results[0].error
                assert summarizer._stats["summarization_errors"] == 1
        finally:
            settings.enable_description_summarization = original_enabled


class TestBatchProcessing:
    """Test batch processing."""
    
    @pytest.mark.asyncio
    async def test_single_batch(self):
        """Test processing single batch."""
        summarizer = DescriptionSummarizer()
        
        # Mock settings
        original_enabled = settings.enable_description_summarization
        original_batch_size = settings.summarization_batch_size
        settings.enable_description_summarization = True
        settings.summarization_batch_size = 5
        
        try:
            # Mock LLM response
            mock_response = """
1. First entity
2. Second entity
3. Third entity
"""
            
            with patch('core.llm.llm_manager.generate_response', return_value=mock_response):
                # Provide 3 entities (< batch size)
                entities = [
                    ("E1", "desc1" * 50, 3),
                    ("E2", "desc2" * 50, 3),
                    ("E3", "desc3" * 50, 3),
                ]
                
                results = await summarizer.summarize_entity_descriptions(entities)
                
                assert len(results) == 3
                assert results[0].summarized_description == "First entity"
                assert results[1].summarized_description == "Second entity"
                assert results[2].summarized_description == "Third entity"
        finally:
            settings.enable_description_summarization = original_enabled
            settings.summarization_batch_size = original_batch_size
    
    @pytest.mark.asyncio
    async def test_multiple_batches(self):
        """Test processing multiple batches."""
        # Mock settings BEFORE creating summarizer
        original_enabled = settings.enable_description_summarization
        original_batch_size = settings.summarization_batch_size
        original_min_mentions = settings.summarization_min_mentions
        original_min_length = settings.summarization_min_length
        original_cache = settings.summarization_cache_enabled
        settings.enable_description_summarization = True
        settings.summarization_batch_size = 2  # Small batch size for testing
        settings.summarization_min_mentions = 2
        settings.summarization_min_length = 50
        settings.summarization_cache_enabled = False  # Disable cache to force LLM calls
        
        try:
            # Mock LLM to return different responses per batch
            call_count = 0
            def mock_llm(prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "1. First\n2. Second"
                elif call_count == 2:
                    return "1. Third\n2. Fourth"
                else:
                    return "1. Fifth"
            
            with patch('core.llm.llm_manager.generate_response', side_effect=mock_llm):
                summarizer = DescriptionSummarizer()
                
                # Provide 5 entities (3 batches: 2, 2, 1)
                entities = [
                    ("E1", "d" * 250, 3),
                    ("E2", "d" * 250, 3),
                    ("E3", "d" * 250, 3),
                    ("E4", "d" * 250, 3),
                    ("E5", "d" * 250, 3),
                ]
                
                results = await summarizer.summarize_entity_descriptions(entities)
                
                # Should have called LLM 3 times (3 batches)
                assert call_count == 3
                assert len(results) == 5
        finally:
            settings.enable_description_summarization = original_enabled
            settings.summarization_batch_size = original_batch_size
            settings.summarization_min_mentions = original_min_mentions
            settings.summarization_min_length = original_min_length
            settings.summarization_cache_enabled = original_cache


class TestRelationshipSummarization:
    """Test relationship-specific summarization."""
    
    @pytest.mark.asyncio
    async def test_relationship_identifier_format(self):
        """Test relationship identifier format."""
        summarizer = DescriptionSummarizer()
        
        # Mock settings
        original_enabled = settings.enable_description_summarization
        settings.enable_description_summarization = True
        
        try:
            mock_response = "1. Relationship description"
            
            with patch('core.llm.llm_manager.generate_response', return_value=mock_response):
                relationships = [
                    ("SOURCE", "TARGET", "DEPENDS_ON", "desc" * 50, 3),
                ]
                
                results = await summarizer.summarize_relationship_descriptions(relationships)
                
                assert len(results) == 1
                # Identifier should be formatted as "SOURCE -> TARGET (TYPE)"
                assert "SOURCE" in results[0].entity_name
                assert "TARGET" in results[0].entity_name
                assert "DEPENDS_ON" in results[0].entity_name
        finally:
            settings.enable_description_summarization = original_enabled


class TestStatistics:
    """Test statistics tracking."""
    
    def test_get_statistics_initial(self):
        """Test get_statistics returns correct initial values."""
        summarizer = DescriptionSummarizer()
        
        stats = summarizer.get_statistics()
        
        assert stats["entities_summarized"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["average_compression_ratio"] == 1.0
        assert stats["estimated_tokens_saved"] == 0
    
    @pytest.mark.asyncio
    async def test_statistics_after_summarization(self):
        """Test statistics updated after summarization."""
        summarizer = DescriptionSummarizer()
        
        # Mock settings
        original_enabled = settings.enable_description_summarization
        settings.enable_description_summarization = True
        
        try:
            mock_response = "1. Summary (50 chars)"
            
            with patch('core.llm.llm_manager.generate_response', return_value=mock_response):
                # Original: 200 chars, Summary: 50 chars (from mock)
                entities = [
                    ("E1", "a" * 200, 3),
                ]
                
                await summarizer.summarize_entity_descriptions(entities)
                
                stats = summarizer.get_statistics()
                
                assert stats["entities_summarized"] == 1
                assert stats["cache_misses"] == 1
                assert stats["total_original_length"] == 200
                # Note: actual summarized length will be from parsed response
                assert stats["average_compression_ratio"] < 1.0
        finally:
            settings.enable_description_summarization = original_enabled


class TestDisabledSummarization:
    """Test behavior when summarization is disabled."""
    
    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        """Test returns empty results when disabled."""
        summarizer = DescriptionSummarizer()
        
        # Mock settings
        original_enabled = settings.enable_description_summarization
        settings.enable_description_summarization = False
        
        try:
            entities = [
                ("E1", "a" * 250, 3),
            ]
            
            results = await summarizer.summarize_entity_descriptions(entities)
            
            assert len(results) == 0
        finally:
            settings.enable_description_summarization = original_enabled


class TestDescriptionHash:
    """Test description hash computation."""
    
    def test_hash_consistency(self):
        """Test same description produces same hash."""
        summarizer = DescriptionSummarizer()
        
        desc1 = "Test description"
        desc2 = "Test description"
        
        hash1 = summarizer._compute_description_hash(desc1)
        hash2 = summarizer._compute_description_hash(desc2)
        
        assert hash1 == hash2
    
    def test_hash_uniqueness(self):
        """Test different descriptions produce different hashes."""
        summarizer = DescriptionSummarizer()
        
        desc1 = "Test description 1"
        desc2 = "Test description 2"
        
        hash1 = summarizer._compute_description_hash(desc1)
        hash2 = summarizer._compute_description_hash(desc2)
        
        assert hash1 != hash2


class TestPromptBuilding:
    """Test prompt building."""
    
    def test_entity_prompt_contains_context(self):
        """Test entity prompt contains necessary context."""
        summarizer = DescriptionSummarizer()
        
        items = [
            ("ADMIN PANEL", "desc1\ndesc2", 3),
        ]
        
        prompt = summarizer._build_summarization_prompt(items, entity_type=True)
        
        # Should contain key elements
        assert "entity" in prompt.lower()
        assert "ADMIN PANEL" in prompt
        assert "mentioned 3 times" in prompt
        assert "deduplicate" in prompt.lower()
        assert "synthesize" in prompt.lower()
    
    def test_relationship_prompt_format(self):
        """Test relationship prompt format."""
        summarizer = DescriptionSummarizer()
        
        items = [
            ("SRC -> TGT (TYPE)", "desc1\ndesc2", 2),
        ]
        
        prompt = summarizer._build_summarization_prompt(items, entity_type=False)
        
        assert "relationship" in prompt.lower()
        assert "SRC -> TGT (TYPE)" in prompt
        assert "mentioned 2 times" in prompt
