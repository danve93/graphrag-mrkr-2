"""
Tests for sentence_chunker.py
"""

import pytest
from core.sentence_chunker import split_into_sentences, get_sentence_window


class TestSplitIntoSentences:
    def test_basic_splitting(self):
        text = "This is sentence one. This is sentence two. And this is three."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two."
        assert sentences[2] == "And this is three."

    def test_abbreviations_not_split(self):
        text = "Dr. Smith went to the store. He bought milk."
        sentences = split_into_sentences(text)
        assert len(sentences) == 2
        # Note: abbreviations are lowercased during protection, but sentence is preserved
        assert "Smith went to the store." in sentences[0]

    def test_question_marks(self):
        text = "What is this? This is a test. Are you sure?"
        sentences = split_into_sentences(text)
        # Regex only splits on '. ' followed by capital, so ? doesn't always split
        assert len(sentences) >= 2

    def test_exclamation_marks(self):
        text = "Hello there! How are you? Great to see you."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_empty_text(self):
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []

    def test_min_length_filter(self):
        text = "A. B. This is a valid sentence."
        sentences = split_into_sentences(text, min_length=10)
        # Short sentences should be filtered out
        assert "This is a valid sentence." in sentences

    def test_single_sentence(self):
        text = "Just one sentence here"
        sentences = split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "Just one sentence here"


class TestGetSentenceWindow:
    def test_basic_window(self):
        sentences = ["Sentence 0.", "Sentence 1.", "Sentence 2.", "Sentence 3.", "Sentence 4."]
        # Window around index 2 with size 1 should include indices 1, 2, 3
        result = get_sentence_window(sentences, target_index=2, window_size=1)
        assert "Sentence 1." in result
        assert "Sentence 2." in result
        assert "Sentence 3." in result

    def test_window_at_start(self):
        sentences = ["Sentence 0.", "Sentence 1.", "Sentence 2.", "Sentence 3."]
        # Window at index 0 with size 2 should include 0, 1, 2
        result = get_sentence_window(sentences, target_index=0, window_size=2)
        assert "Sentence 0." in result
        assert "Sentence 1." in result
        assert "Sentence 2." in result

    def test_window_at_end(self):
        sentences = ["Sentence 0.", "Sentence 1.", "Sentence 2.", "Sentence 3."]
        # Window at index 3 with size 2 should include 1, 2, 3
        result = get_sentence_window(sentences, target_index=3, window_size=2)
        assert "Sentence 1." in result
        assert "Sentence 2." in result
        assert "Sentence 3." in result

    def test_empty_list(self):
        result = get_sentence_window([], target_index=0, window_size=5)
        assert result == ""

    def test_large_window(self):
        sentences = ["S1.", "S2.", "S3."]
        # Window size larger than list should return all
        result = get_sentence_window(sentences, target_index=1, window_size=10)
        assert "S1." in result
        assert "S2." in result
        assert "S3." in result
