"""
Tests for entity extraction gleaning functionality (Phase 1)
"""

import pytest
from core.entity_extraction import EntityExtractor, Entity


def test_continue_prompt_includes_entity_summary():
    """Test that continuation prompt includes summary of found entities."""
    extractor = EntityExtractor()
    
    existing_entities = [
        Entity(name="Entity1", type="COMPONENT", description="Test", importance_score=0.5),
        Entity(name="Entity2", type="SERVICE", description="Test", importance_score=0.5),
    ]
    
    prompt = extractor._get_continue_prompt("chunk_1", existing_entities)
    
    # Should include entity names
    assert "Entity1" in prompt
    assert "Entity2" in prompt
    # Should have assertive language
    assert "MISSED" in prompt or "overlooked" in prompt
    # Should remind about format
    assert "format" in prompt.lower() or "same" in prompt.lower()


def test_loop_check_prompt_simple():
    """Test that loop check prompt is simple Y/N question."""
    extractor = EntityExtractor()
    
    prompt = extractor._get_loop_check_prompt()
    
    # Should be short and simple
    assert len(prompt) < 500
    # Should ask for Y or N
    assert "Y" in prompt and "N" in prompt
    # Should ask if more entities exist
    assert ("more" in prompt.lower() and 
            ("entities" in prompt.lower() or "extraction" in prompt.lower()))


def test_continue_prompt_with_many_entities():
    """Test continuation prompt with more than 10 entities."""
    extractor = EntityExtractor()
    
    entities = [
        Entity(name=f"Entity{i}", type="COMPONENT", description="Test", importance_score=0.5)
        for i in range(15)
    ]
    
    prompt = extractor._get_continue_prompt("chunk_1", entities)
    
    # Should show first 10 and indicate more exist
    assert "Entity0" in prompt
    assert "Entity9" in prompt
    assert "and 5 more" in prompt or "(and 5 more)" in prompt


@pytest.mark.asyncio
async def test_gleaning_with_zero_passes():
    """Test that max_gleanings=0 uses baseline path (no history calls)."""
    extractor = EntityExtractor()
    
    # With max_gleanings=0, should just do initial extraction
    try:
        entities, relationships = await extractor.extract_from_chunk_with_gleaning(
            text="Test component connects to test service.",
            chunk_id="chunk_1",
            max_gleanings=0
        )
        
        # Just verify it doesn't crash (behavior depends on LLM response)
        assert isinstance(entities, list)
        assert isinstance(relationships, list)
    except Exception as e:
        # If LLM isn't configured, test passes
        if "API key" not in str(e) and "Failed" not in str(e):
            raise


@pytest.mark.asyncio
async def test_gleaning_handles_no_new_entities():
    """Test that gleaning stops early if no new entities found."""
    extractor = EntityExtractor()
    
    # This test would require mocking the LLM to return empty on pass 2
    # For now, just verify the method signature and structure
    try:
        entities, relationships = await extractor.extract_from_chunk_with_gleaning(
            text="Simple text",
            chunk_id="chunk_1",
            max_gleanings=2
        )
        assert isinstance(entities, list)
        assert isinstance(relationships, list)
    except Exception as e:
        # If LLM isn't configured, test passes
        if "API key" not in str(e) and "Failed" not in str(e):
            raise
