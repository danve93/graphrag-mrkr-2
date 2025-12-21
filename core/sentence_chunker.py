"""
Sentence splitting utility for Sentence-Window Retrieval.

Splits text into individual sentences for fine-grained embedding and retrieval.
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Sentence boundary pattern: period, exclamation, or question mark followed by space and capital
# Also handles common abbreviations to avoid false splits
ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "i.e.", "e.g.",
    "fig.", "vol.", "no.", "pp.", "p.", "ed.", "eds.", "rev.", "st.", "inc.", "corp.",
    "ltd.", "co.", "dept.", "univ.", "approx.", "est.", "min.", "max.", "avg.",
}


def split_into_sentences(text: str, min_length: int = 10) -> List[str]:
    """
    Split text into sentences using regex-based approach.
    
    Args:
        text: Input text to split
        min_length: Minimum character length for a valid sentence
        
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Protect abbreviations by temporarily replacing periods
    protected_text = text
    for abbrev in ABBREVIATIONS:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(abbrev), re.IGNORECASE)
        protected_text = pattern.sub(abbrev.replace(".", "<PERIOD>"), protected_text)
    
    # Split on sentence boundaries
    # Match: sentence-ending punctuation followed by space and capital letter OR end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    raw_sentences = re.split(sentence_pattern, protected_text)
    
    # Restore periods in abbreviations and clean up
    sentences = []
    for sentence in raw_sentences:
        sentence = sentence.replace("<PERIOD>", ".").strip()
        
        # Skip sentences that are too short
        if len(sentence) >= min_length:
            sentences.append(sentence)
    
    # If no valid sentences found, return the whole text as one sentence
    if not sentences and text.strip():
        return [text.strip()]
    
    logger.debug(f"Split text into {len(sentences)} sentences")
    return sentences


def get_sentence_window(
    sentences: List[str],
    target_index: int,
    window_size: int = 5,
) -> str:
    """
    Get a window of sentences around a target sentence.
    
    Args:
        sentences: List of all sentences
        target_index: Index of the target sentence
        window_size: Number of sentences to include on each side
        
    Returns:
        Combined text of the sentence window
    """
    if not sentences:
        return ""
    
    # Clamp target_index to valid range
    target_index = max(0, min(target_index, len(sentences) - 1))
    
    # Calculate window bounds
    start_index = max(0, target_index - window_size)
    end_index = min(len(sentences), target_index + window_size + 1)
    
    # Join sentences in the window
    window_sentences = sentences[start_index:end_index]
    return " ".join(window_sentences)
