"""
Unit tests for content quality filtering.

Tests heuristic-based filtering logic to ensure:
1. Low-quality chunks are filtered correctly
2. High-quality chunks pass through
3. Metrics tracking is accurate
4. Configuration options work as expected
"""

import pytest
from ingestion.content_filters import ContentQualityFilter, FilterMetrics


class TestContentQualityFilter:
    """Test suite for ContentQualityFilter class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.filter = ContentQualityFilter(
            min_chunk_length=50,
            min_unique_word_ratio=0.3,
            max_special_char_ratio=0.5,
            min_alphanumeric_ratio=0.3,
        )

    def test_initialization(self):
        """Test filter initialization with default settings."""
        assert self.filter.min_chunk_length == 50
        assert self.filter.min_unique_word_ratio == 0.3
        assert self.filter.metrics.total_chunks == 0

    def test_minimum_length_filter(self):
        """Test that chunks below minimum length are filtered."""
        # Too short
        short_chunk = "Too short"
        should_embed, reason = self.filter.should_embed_chunk(short_chunk)

        assert should_embed is False
        assert "too_short" in reason
        assert self.filter.metrics.filtered_chunks == 1

    def test_maximum_length_filter(self):
        """Test that chunks above maximum length are filtered."""
        # Create extremely long chunk (> 100000 chars)
        long_chunk = "A" * 100001
        should_embed, reason = self.filter.should_embed_chunk(long_chunk)

        assert should_embed is False
        assert "too_long" in reason

    def test_empty_chunk_filter(self):
        """Test that empty or whitespace-only chunks are filtered."""
        # Empty string (caught by length filter)
        should_embed, reason = self.filter.should_embed_chunk("")
        assert should_embed is False
        assert "too_short" in reason or "empty" in reason

        # Whitespace only (caught by length filter)
        should_embed, reason = self.filter.should_embed_chunk("    \n\n\t  ")
        assert should_embed is False
        assert "too_short" in reason or "empty" in reason

    def test_repetitive_content_filter(self):
        """Test that highly repetitive content is filtered."""
        # Chunk with low unique word ratio
        repetitive_chunk = " ".join(["test"] * 100)  # Only one unique word
        should_embed, reason = self.filter.should_embed_chunk(repetitive_chunk)

        assert should_embed is False
        assert "repetitive" in reason

    def test_single_word_repetition_filter(self):
        """Test filtering of single word repeated many times."""
        repetitive_chunk = " ".join(["error"] * 50 + ["different", "words", "here"])
        should_embed, reason = self.filter.should_embed_chunk(repetitive_chunk)

        assert should_embed is False
        # Caught by repetition check (unique ratio or single word repetition)
        assert "repetitive" in reason or "single_word_repetition" in reason

    def test_character_distribution_filter(self):
        """Test filtering based on character distribution."""
        # Too many special characters (garbage data)
        garbage_chunk = "!@#$%^&*()_+{}[]|\\:;<>?,./" * 10
        should_embed, reason = self.filter.should_embed_chunk(garbage_chunk)

        assert should_embed is False
        assert "alphanumeric" in reason or "special_chars" in reason

    def test_high_quality_chunk_passes(self):
        """Test that high-quality chunks pass filtering."""
        good_chunk = """
        This is a well-formed document with meaningful content.
        It contains multiple sentences with diverse vocabulary.
        The text has proper structure and conveys information clearly.
        """
        should_embed, reason = self.filter.should_embed_chunk(good_chunk)

        assert should_embed is True
        assert reason is None
        assert self.filter.metrics.passed_chunks == 1

    def test_code_chunk_allows_special_characters(self):
        """Test that code chunks allow higher special character ratio."""
        code_chunk = """
        def process_data(items):
            result = [
                {
                    'key': item['value'],
                    'count': len(item),
                }
                for item in items if item['active']
            ]
            return result
        """

        metadata = {"file_type": ".py", "is_code": True}
        should_embed, reason = self.filter.should_embed_chunk(code_chunk, metadata)

        # Code should pass even with special characters
        assert should_embed is True

    def test_conversation_quality_filter(self):
        """Test conversation thread quality filtering."""
        # Short thread without resolution (but enough text to pass length filter)
        short_thread = "Hi there, how are you doing today?\nHello! I'm doing well, thanks for asking."
        metadata = {
            "is_conversation": True,
            "message_count": 2,
            "participant_count": 1,
        }

        should_embed, reason = self.filter.should_embed_chunk(short_thread, metadata)
        assert should_embed is False
        assert "thread_too_short" in reason

    def test_conversation_with_resolution_passes(self):
        """Test that conversation with resolution indicators passes."""
        resolved_thread = """
        User: My application is crashing on startup
        Support: Can you check the logs?
        User: Found the issue, it was a missing dependency
        Support: Great!
        User: Thanks, it's working now and resolved
        """
        metadata = {
            "is_conversation": True,
            "message_count": 5,
            "participant_count": 2,
        }

        should_embed, reason = self.filter.should_embed_chunk(
            resolved_thread, metadata
        )
        assert should_embed is True

    def test_spam_detection(self):
        """Test spam pattern detection in conversations."""
        spam_content = "Congratulations! You've won a prize! Click here to claim now!"
        metadata = {"is_conversation": True, "message_count": 5, "participant_count": 2}

        should_embed, reason = self.filter.should_embed_chunk(spam_content, metadata)
        assert should_embed is False
        assert "spam" in reason

    def test_structured_data_empty_table_filter(self):
        """Test filtering of empty tables."""
        metadata = {"is_structured_data": True, "row_count": 0, "column_count": 5}

        # Empty table will be caught by length or empty filter
        should_embed, reason = self.filter.should_embed_chunk("", metadata)
        assert should_embed is False
        assert "empty" in reason or "too_short" in reason

    def test_structured_data_single_column_filter(self):
        """Test filtering of single-column tables."""
        metadata = {"is_structured_data": True, "row_count": 10, "column_count": 1}

        # Need enough content to pass length filter
        long_enough_content = "Table data with single column: " + ", ".join([f"row{i}" for i in range(10)])
        should_embed, reason = self.filter.should_embed_chunk(
            long_enough_content, metadata
        )
        assert should_embed is False
        assert "single_column" in reason

    def test_structured_data_header_only_filter(self):
        """Test filtering of header-only tables."""
        metadata = {"is_structured_data": True, "row_count": 1, "column_count": 5}

        # Need enough content to pass length filter
        long_enough_content = "Header row with five columns: col1, col2, col3, col4, col5"
        should_embed, reason = self.filter.should_embed_chunk(
            long_enough_content, metadata
        )
        assert should_embed is False
        assert "header_only" in reason

    def test_code_comment_only_filter(self):
        """Test filtering of comment-only code blocks."""
        comment_only = """
        # This is a comment
        # Another comment
        # Yet another comment
        # More comments
        """
        metadata = {"is_code": True, "file_type": ".py"}

        should_embed, reason = self.filter.should_embed_chunk(comment_only, metadata)
        assert should_embed is False
        assert "comment_only" in reason

    def test_code_mostly_comments_filter(self):
        """Test filtering of code with >80% comments."""
        mostly_comments = """
        # Comment 1
        # Comment 2
        # Comment 3
        # Comment 4
        # Comment 5
        # Comment 6
        # Comment 7
        # Comment 8
        x = 1  # Only one line of code
        # Comment 10
        """
        metadata = {"is_code": True, "file_type": ".py"}

        should_embed, reason = self.filter.should_embed_chunk(
            mostly_comments, metadata
        )
        assert should_embed is False
        assert "mostly_comments" in reason

    def test_auto_generated_code_filter(self):
        """Test filtering of auto-generated code."""
        auto_generated = """
        # AUTO-GENERATED CODE - DO NOT EDIT
        # Generated by code generator v1.0

        x = 1
        y = 2
        """
        metadata = {"is_code": True, "file_type": ".py"}

        should_embed, reason = self.filter.should_embed_chunk(
            auto_generated, metadata
        )
        assert should_embed is False
        assert "auto_generated" in reason

    def test_substantial_auto_generated_code_passes(self):
        """Test that auto-generated code with substantial content passes."""
        substantial_generated = """
        # Auto-generated code

        """ + "\n".join(
            [f"def function_{i}():\n    return {i}" for i in range(20)]
        )

        metadata = {"is_code": True, "file_type": ".py"}

        should_embed, reason = self.filter.should_embed_chunk(
            substantial_generated, metadata
        )
        assert should_embed is True

    def test_filter_chunks_batch(self):
        """Test batch filtering of multiple chunks."""
        chunks = [
            {"content": "Good chunk with meaningful content that passes all filters."},
            {"content": "Too short"},
            {"content": "test " * 50},  # Repetitive
            {
                "content": "Another good chunk with diverse vocabulary and proper length."
            },
            {"content": ""},  # Empty
        ]

        filtered_chunks, metrics = self.filter.filter_chunks(chunks)

        assert len(filtered_chunks) == 2  # Only 2 good chunks
        assert metrics.total_chunks == 5
        assert metrics.filtered_chunks == 3
        assert metrics.passed_chunks == 2
        assert metrics.filter_rate == 60.0
        assert metrics.pass_rate == 40.0

    def test_metrics_tracking(self):
        """Test that metrics are tracked correctly."""
        chunks = [
            {"content": "This is a good chunk with enough length to pass the minimum length filter."},
            {"content": "x"},  # Too short
            {"content": "y"},  # Too short
        ]

        self.filter.filter_chunks(chunks)

        metrics = self.filter.get_metrics()
        assert "too_short" in metrics.filter_reasons
        assert metrics.filter_reasons["too_short"] == 2

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        self.filter.should_embed_chunk("Good chunk that passes")
        assert self.filter.metrics.total_chunks == 1

        self.filter.reset_metrics()
        assert self.filter.metrics.total_chunks == 0
        assert self.filter.metrics.passed_chunks == 0
        assert len(self.filter.metrics.filter_reasons) == 0

    def test_get_config(self):
        """Test configuration retrieval."""
        config = self.filter.get_config()

        assert config["min_chunk_length"] == 50
        assert config["min_unique_word_ratio"] == 0.3
        assert "enable_conversation_filters" in config

    def test_disable_conversation_filters(self):
        """Test disabling conversation filters."""
        filter_no_conv = ContentQualityFilter(enable_conversation_filters=False)

        # Short conversation that would normally be filtered
        metadata = {
            "is_conversation": True,
            "message_count": 2,
            "participant_count": 1,
        }

        should_embed, reason = filter_no_conv.should_embed_chunk(
            "Short thread with enough text to pass basic filters", metadata
        )

        # Should pass because conversation filters are disabled
        assert should_embed is True

    def test_disable_code_filters(self):
        """Test disabling code filters."""
        filter_no_code = ContentQualityFilter(enable_code_filters=False)

        # Comment-only code that would normally be filtered
        comment_only = "# Comment\n# Another comment\n# More comments\n# Even more"
        metadata = {"is_code": True, "file_type": ".py"}

        should_embed, reason = filter_no_code.should_embed_chunk(
            comment_only, metadata
        )

        # Should pass because code filters are disabled
        assert should_embed is True

    def test_filter_metrics_summary(self):
        """Test metrics summary generation."""
        metrics = FilterMetrics()
        metrics.add_passed()
        metrics.add_filtered("too_short")
        metrics.add_filtered("too_short")
        metrics.add_passed()
        metrics.add_filtered("repetitive")

        summary = metrics.get_summary()

        assert summary["total_chunks"] == 5
        assert summary["passed_chunks"] == 2
        assert summary["filtered_chunks"] == 3
        assert summary["pass_rate"] == 40.0
        assert summary["filter_rate"] == 60.0
        assert summary["filter_reasons"]["too_short"] == 2
        assert summary["filter_reasons"]["repetitive"] == 1

    def test_real_world_documentation_chunk(self):
        """Test with real-world documentation content."""
        doc_chunk = """
        ## Installation Guide

        To install this package, follow these steps:

        1. Install Python 3.8 or higher
        2. Run `pip install package-name`
        3. Configure your environment variables

        For more information, see the configuration guide.
        """

        should_embed, reason = self.filter.should_embed_chunk(doc_chunk)
        assert should_embed is True

    def test_real_world_csv_summary(self):
        """Test with real-world CSV summary content."""
        csv_summary = """
        === CSV File Analysis: sales_data.csv ===
        Dimensions: 10000 rows × 12 columns

        === Column Structure ===
        Column 1: order_id (int64) - 10000 unique values
        Column 2: customer_name (object) - 5000 unique values
        Column 3: amount (float64) - Range: $10.50 to $9999.99
        """

        metadata = {"is_structured_data": True, "row_count": 10000, "column_count": 12}

        should_embed, reason = self.filter.should_embed_chunk(csv_summary, metadata)
        assert should_embed is True

    def test_edge_case_exact_length_threshold(self):
        """Test chunks at exact length threshold."""
        # Exactly at minimum length
        chunk = "x" * 50
        should_embed, reason = self.filter.should_embed_chunk(chunk)

        # Should pass length check but may fail other checks
        assert "too_short" not in (reason or "")


class TestFilterMetrics:
    """Test suite for FilterMetrics class."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = FilterMetrics()

        assert metrics.total_chunks == 0
        assert metrics.filtered_chunks == 0
        assert metrics.passed_chunks == 0
        assert metrics.filter_rate == 0.0
        assert metrics.pass_rate == 0.0

    def test_add_filtered(self):
        """Test adding filtered chunk."""
        metrics = FilterMetrics()
        metrics.add_filtered("test_reason")

        assert metrics.total_chunks == 1
        assert metrics.filtered_chunks == 1
        assert metrics.filter_reasons["test_reason"] == 1

    def test_add_passed(self):
        """Test adding passed chunk."""
        metrics = FilterMetrics()
        metrics.add_passed()

        assert metrics.total_chunks == 1
        assert metrics.passed_chunks == 1

    def test_multiple_filter_reasons(self):
        """Test tracking multiple filter reasons."""
        metrics = FilterMetrics()
        metrics.add_filtered("too_short")
        metrics.add_filtered("too_short")
        metrics.add_filtered("repetitive")

        assert metrics.filter_reasons["too_short"] == 2
        assert metrics.filter_reasons["repetitive"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
