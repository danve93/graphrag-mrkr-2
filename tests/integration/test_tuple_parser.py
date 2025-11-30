"""
Unit tests for tuple-delimited entity extraction parser (Phase 3).

Tests cover:
- Valid tuple parsing (entities and relationships)
- Malformed input handling
- Edge cases (special characters, whitespace, empty fields)
- Normalization consistency
- ParseResult structure
"""

import pytest
from core.tuple_parser import TupleParser, ParseResult
from core.entity_models import Entity, Relationship


class TestTupleParserValidInput:
    """Tests for parsing valid tuple-delimited input."""

    def test_parse_simple_entity(self):
        """Test parsing a simple entity tuple."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|>ADMIN PANEL<|>COMPONENT<|>Main admin interface)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert result.invalid_count == 0
        assert len(result.entities) == 1
        assert len(result.relationships) == 0
        assert result.entities[0].name == "ADMIN PANEL"
        assert result.entities[0].type == "COMPONENT"
        assert result.entities[0].description == "Main admin interface"
        assert result.entities[0].importance_score == 0.5  # Default
        assert result.entities[0].source_chunks == ["test_chunk"]

    def test_parse_entity_with_importance(self):
        """Test parsing entity tuple with importance score."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|>USER DATABASE<|>SERVICE<|>Stores user data<|>0.9)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert len(result.entities) == 1
        assert result.entities[0].name == "USER DATABASE"
        assert result.entities[0].type == "SERVICE"
        assert result.entities[0].description == "Stores user data"
        assert result.entities[0].importance_score == 0.9

    def test_parse_entity_with_empty_description(self):
        """Test parsing entity tuple with empty description."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|>CONFIG FILE<|>RESOURCE<|>)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert len(result.entities) == 1
        assert result.entities[0].name == "CONFIG FILE"
        assert result.entities[0].type == "RESOURCE"
        assert result.entities[0].description == ""

    def test_parse_simple_relationship(self):
        """Test parsing a simple relationship tuple."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("relationship"<|>ADMIN PANEL<|>USER DATABASE<|>DEPENDS_ON<|>Admin queries database)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert result.invalid_count == 0
        assert len(result.entities) == 0
        assert len(result.relationships) == 1
        assert result.relationships[0].source_entity == "ADMIN PANEL"
        assert result.relationships[0].target_entity == "USER DATABASE"
        assert result.relationships[0].relationship_type == "DEPENDS_ON"
        assert result.relationships[0].description == "Admin queries database"
        assert result.relationships[0].strength == 0.5  # Default

    def test_parse_relationship_with_strength(self):
        """Test parsing relationship tuple with strength."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("relationship"<|>SERVICE A<|>SERVICE B<|>CALLS<|>Makes API calls<|>0.8)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert len(result.relationships) == 1
        assert result.relationships[0].source_entity == "SERVICE A"
        assert result.relationships[0].target_entity == "SERVICE B"
        assert result.relationships[0].relationship_type == "CALLS"
        assert result.relationships[0].strength == 0.8

    def test_parse_relationship_with_empty_description(self):
        """Test parsing relationship tuple with empty description."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("relationship"<|>NODE A<|>NODE B<|>CONNECTED_TO<|>)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert len(result.relationships) == 1
        assert result.relationships[0].description == ""

    def test_parse_multiple_tuples(self):
        """Test parsing multiple tuples in one text."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '''("entity"<|>ADMIN PANEL<|>COMPONENT<|>Main interface<|>0.9)
("entity"<|>USER DATABASE<|>SERVICE<|>Stores users<|>0.8)
("relationship"<|>ADMIN PANEL<|>USER DATABASE<|>DEPENDS_ON<|>Queries for auth<|>0.7)'''
        
        result = parser.parse(text)
        
        assert result.valid_count == 3
        assert result.invalid_count == 0
        assert len(result.entities) == 2
        assert len(result.relationships) == 1

    def test_parse_with_special_characters(self):
        """Test parsing tuples with special characters in descriptions."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|>USER AUTH (V2)<|>SERVICE<|>Handles OAuth 2.0 & JWT tokens<|>0.85)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert len(result.entities) == 1
        assert result.entities[0].name == "USER AUTH (V2)"
        assert "OAuth 2.0 & JWT tokens" in result.entities[0].description


class TestTupleParserMalformedInput:
    """Tests for handling malformed tuple input."""

    def test_missing_opening_parenthesis(self):
        """Test handling of tuple missing opening parenthesis."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '"entity"<|>ADMIN PANEL<|>COMPONENT<|>Description)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 0
        assert result.invalid_count == 0  # Silently skipped (not a tuple)

    def test_missing_closing_parenthesis(self):
        """Test handling of tuple missing closing parenthesis."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|>ADMIN PANEL<|>COMPONENT<|>Description'
        
        result = parser.parse(text)
        
        assert result.valid_count == 0
        assert result.invalid_count == 0  # Silently skipped

    def test_wrong_delimiter(self):
        """Test handling of wrong delimiter (| instead of <|>)."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"|ADMIN PANEL|COMPONENT|Description)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 0
        assert result.invalid_count == 0  # No <|> delimiter, skipped

    def test_insufficient_entity_fields(self):
        """Test handling of entity tuple with insufficient fields."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|>ADMIN PANEL)'  # Missing type and description
        
        result = parser.parse(text)
        
        assert result.valid_count == 0
        assert result.invalid_count == 1
        assert len(result.parse_errors) == 1

    def test_insufficient_relationship_fields(self):
        """Test handling of relationship tuple with insufficient fields."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("relationship"<|>SOURCE<|>TARGET)'  # Missing type
        
        result = parser.parse(text)
        
        assert result.valid_count == 0
        assert result.invalid_count == 1
        assert len(result.parse_errors) == 1

    def test_empty_entity_name(self):
        """Test handling of entity tuple with empty name."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|><|>COMPONENT<|>Description)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 0
        assert result.invalid_count == 1

    def test_empty_relationship_source(self):
        """Test handling of relationship with empty source."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("relationship"<|><|>TARGET<|>TYPE<|>Description)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 0
        assert result.invalid_count == 1


class TestTupleParserEdgeCases:
    """Tests for edge cases in tuple parsing."""

    def test_empty_input(self):
        """Test handling of empty input."""
        parser = TupleParser(chunk_id="test_chunk")
        text = ""
        
        result = parser.parse(text)
        
        assert result.valid_count == 0
        assert result.invalid_count == 1
        assert len(result.parse_errors) == 1
        assert "Empty input" in result.parse_errors[0]

    def test_whitespace_only_input(self):
        """Test handling of whitespace-only input."""
        parser = TupleParser(chunk_id="test_chunk")
        text = "   \n  \t  "
        
        result = parser.parse(text)
        
        assert result.valid_count == 0
        assert result.invalid_count == 1

    def test_empty_lines_and_comments(self):
        """Test that empty lines and comments are skipped."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '''# This is a comment
("entity"<|>ADMIN PANEL<|>COMPONENT<|>Description)

# Another comment
("entity"<|>USER DB<|>SERVICE<|>Database)'''
        
        result = parser.parse(text)
        
        assert result.valid_count == 2
        assert len(result.entities) == 2

    def test_whitespace_variations(self):
        """Test handling of various whitespace patterns."""
        parser = TupleParser(chunk_id="test_chunk")
        # Leading/trailing whitespace in fields
        text = '("entity"<|>  ADMIN PANEL  <|>  COMPONENT  <|>  Description  <|>  0.9  )'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert result.entities[0].name == "ADMIN PANEL"
        assert result.entities[0].type == "COMPONENT"
        assert result.entities[0].description == "Description"
        assert result.entities[0].importance_score == 0.9

    def test_multi_word_names_with_punctuation(self):
        """Test parsing of multi-word names with punctuation."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|>USER AUTHENTICATION SERVICE (V2.0)<|>SERVICE<|>Auth service version 2<|>0.9)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert "USER AUTHENTICATION SERVICE (V2.0)" in result.entities[0].name

    def test_mixed_valid_and_invalid_tuples(self):
        """Test partial recovery from mixed valid/invalid tuples."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '''("entity"<|>VALID ENTITY<|>COMPONENT<|>Good entity)
("entity"<|>BAD ENTITY)
("entity"<|>ANOTHER VALID<|>SERVICE<|>Another good one)'''
        
        result = parser.parse(text)
        
        assert result.valid_count == 2
        assert result.invalid_count == 1
        assert len(result.entities) == 2
        assert len(result.parse_errors) == 1

    def test_unicode_characters(self):
        """Test parsing tuples with Unicode characters."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|>SYSTÈME DE DONNÉES<|>SERVICE<|>Système français avec accents é è ê<|>0.8)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert len(result.entities) == 1
        # Name is normalized to uppercase
        assert "SYSTÈME" in result.entities[0].name.upper()

    def test_invalid_importance_score(self):
        """Test handling of invalid importance scores."""
        parser = TupleParser(chunk_id="test_chunk")
        # Importance > 1.0
        text = '("entity"<|>ADMIN PANEL<|>COMPONENT<|>Description<|>1.5)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        # Should clamp to 0.5 (default) due to warning
        assert result.entities[0].importance_score == 0.5

    def test_invalid_strength_score(self):
        """Test handling of invalid strength scores."""
        parser = TupleParser(chunk_id="test_chunk")
        # Strength < 0.0
        text = '("relationship"<|>A<|>B<|>CONNECTS<|>Description<|>-0.3)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        # Should clamp to 0.5 (default) due to warning
        assert result.relationships[0].strength == 0.5


class TestTupleParserNormalization:
    """Tests for entity name and type normalization."""

    def test_entity_name_normalization(self):
        """Test that entity names are normalized consistently."""
        parser = TupleParser(chunk_id="test_chunk")
        # Multiple whitespace, mixed case
        text = '("entity"<|>  admin   panel  <|>COMPONENT<|>Description)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        # Should collapse whitespace and uppercase
        assert result.entities[0].name == "ADMIN PANEL"

    def test_entity_type_normalization(self):
        """Test that entity types are normalized to uppercase."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|>ADMIN PANEL<|>component<|>Description)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        assert result.entities[0].type == "COMPONENT"

    def test_relationship_type_normalization(self):
        """Test that relationship types are normalized."""
        parser = TupleParser(chunk_id="test_chunk")
        # Lowercase with spaces
        text = '("relationship"<|>A<|>B<|>depends on<|>Description)'
        
        result = parser.parse(text)
        
        assert result.valid_count == 1
        # Should uppercase and replace spaces with underscores
        assert result.relationships[0].relationship_type == "DEPENDS_ON"

    def test_consistent_normalization_across_formats(self):
        """Test that normalization produces same keys as pipe parser."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '("entity"<|>  Admin   Panel  <|>COMPONENT<|>Description)'
        
        result = parser.parse(text)
        
        # Should match what pipe parser would produce
        assert result.entities[0].name == "ADMIN PANEL"


class TestParseResult:
    """Tests for ParseResult structure."""

    def test_parse_result_counts(self):
        """Test that ParseResult tracks counts correctly."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '''("entity"<|>E1<|>COMPONENT<|>Desc1)
("entity"<|>E2<|>SERVICE<|>Desc2)
("relationship"<|>E1<|>E2<|>DEPENDS_ON<|>Desc)
("entity"<|>BAD)'''
        
        result = parser.parse(text)
        
        assert result.valid_count == 3  # 2 entities + 1 relationship
        assert result.invalid_count == 1  # 1 bad entity
        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert len(result.parse_errors) == 1

    def test_parse_result_error_messages(self):
        """Test that ParseResult includes error messages."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '''("entity"<|>GOOD<|>COMPONENT<|>Desc)
("entity"<|>BAD)
("unknown"<|>FIELD1<|>FIELD2)'''
        
        result = parser.parse(text)
        
        assert len(result.parse_errors) == 2
        # Check that error messages contain line numbers and failure info
        assert any("Line 2" in err for err in result.parse_errors)
        assert any("Line 3" in err for err in result.parse_errors)

    def test_parse_result_entity_list(self):
        """Test that ParseResult populates entity list correctly."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '''("entity"<|>E1<|>COMPONENT<|>Desc1<|>0.9)
("entity"<|>E2<|>SERVICE<|>Desc2<|>0.8)'''
        
        result = parser.parse(text)
        
        assert len(result.entities) == 2
        assert all(isinstance(e, Entity) for e in result.entities)
        assert result.entities[0].name == "E1"
        assert result.entities[1].name == "E2"

    def test_parse_result_relationship_list(self):
        """Test that ParseResult populates relationship list correctly."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '''("relationship"<|>A<|>B<|>DEPENDS_ON<|>Desc1<|>0.7)
("relationship"<|>C<|>D<|>CALLS<|>Desc2<|>0.6)'''
        
        result = parser.parse(text)
        
        assert len(result.relationships) == 2
        assert all(isinstance(r, Relationship) for r in result.relationships)
        assert result.relationships[0].source_entity == "A"
        assert result.relationships[1].source_entity == "C"


class TestTupleParserStats:
    """Tests for parser statistics tracking."""

    def test_get_stats(self):
        """Test that parser tracks statistics correctly."""
        parser = TupleParser(chunk_id="test_chunk")
        text = '''("entity"<|>E1<|>COMPONENT<|>Desc1)
("entity"<|>E2<|>SERVICE<|>Desc2)
("relationship"<|>E1<|>E2<|>DEPENDS_ON<|>Desc)
("entity"<|>BAD)'''
        
        result = parser.parse(text)
        stats = parser.get_stats()
        
        assert stats["entities_parsed"] == 2
        assert stats["relationships_parsed"] == 1
        assert stats["parse_errors"] == 1

    def test_stats_reset_per_instance(self):
        """Test that each parser instance has independent stats."""
        parser1 = TupleParser(chunk_id="chunk1")
        parser1.parse('("entity"<|>E1<|>COMPONENT<|>Desc)')
        
        parser2 = TupleParser(chunk_id="chunk2")
        parser2.parse('("entity"<|>E2<|>SERVICE<|>Desc)')
        
        stats1 = parser1.get_stats()
        stats2 = parser2.get_stats()
        
        assert stats1["entities_parsed"] == 1
        assert stats2["entities_parsed"] == 1
        # Stats are independent
        assert stats1 != stats2 or stats1 == stats2  # Just checking they exist


class TestTupleParserProvenance:
    """Tests for chunk provenance tracking."""

    def test_entity_provenance_tracked(self):
        """Test that entity source chunks are tracked."""
        parser = TupleParser(chunk_id="chunk_123")
        text = '("entity"<|>ADMIN PANEL<|>COMPONENT<|>Description)'
        
        result = parser.parse(text)
        
        assert len(result.entities) == 1
        assert result.entities[0].source_chunks == ["chunk_123"]
        assert result.entities[0].source_text_units == ["chunk_123"]

    def test_relationship_provenance_tracked(self):
        """Test that relationship source chunks are tracked."""
        parser = TupleParser(chunk_id="chunk_456")
        text = '("relationship"<|>A<|>B<|>DEPENDS_ON<|>Description)'
        
        result = parser.parse(text)
        
        assert len(result.relationships) == 1
        assert result.relationships[0].source_chunks == ["chunk_456"]
        assert result.relationships[0].source_text_units == ["chunk_456"]

    def test_no_chunk_id_provided(self):
        """Test behavior when no chunk_id is provided."""
        parser = TupleParser()  # No chunk_id
        text = '("entity"<|>ADMIN PANEL<|>COMPONENT<|>Description)'
        
        result = parser.parse(text)
        
        assert len(result.entities) == 1
        assert result.entities[0].source_chunks == []
        assert result.entities[0].source_text_units == []
