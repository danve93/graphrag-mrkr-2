# Quality Scoring Component

Content quality assessment for chunk filtering and ranking.

## Overview

The quality scoring component evaluates text chunks to identify low-quality content that may degrade retrieval performance. It uses heuristic-based scoring to detect garbled text, formatting artifacts, excessive special characters, and other quality issues. Scores range from 0.0 (poor) to 1.0 (excellent).

**Location**: `core/quality_scorer.py`
**Scoring**: Heuristic-based (fast, no LLM required)
**Range**: 0.0 - 1.0 (configurable threshold for filtering)

## Architecture

```
┌──────────────────────────────────────────────────┐
│         Quality Scoring Pipeline                  │
├──────────────────────────────────────────────────┤
│                                                   │
│  Input: Text Chunk                               │
│  │                                                │
│  ├─→ Metric 1: Character Distribution            │
│  │   ┌──────────────────────────────────────┐    │
│  │   │ Alphanumeric ratio                   │    │
│  │   │ Whitespace ratio                     │    │
│  │   │ Special character ratio              │    │
│  │   └──────────────────────────────────────┘    │
│  │                                                │
│  ├─→ Metric 2: Text Structure                    │
│  │   ┌──────────────────────────────────────┐    │
│  │   │ Average word length                  │    │
│  │   │ Average sentence length              │    │
│  │   │ Line length consistency              │    │
│  │   └──────────────────────────────────────┘    │
│  │                                                │
│  ├─→ Metric 3: Content Indicators                │
│  │   ┌──────────────────────────────────────┐    │
│  │   │ Repeated patterns (>3 times)         │    │
│  │   │ Gibberish detection                  │    │
│  │   │ URL/email density                    │    │
│  │   └──────────────────────────────────────┘    │
│  │                                                │
│  ├─→ Metric 4: Language Features                 │
│  │   ┌──────────────────────────────────────┐    │
│  │   │ Vowel ratio (expected: 35-45%)       │    │
│  │   │ Common word presence                 │    │
│  │   │ Capitalization patterns              │    │
│  │   └──────────────────────────────────────┘    │
│  │                                                │
│  └─→ Output: Quality Score (0.0 - 1.0)           │
│      ┌──────────────────────────────────────┐    │
│      │ Weighted average of all metrics      │    │
│      │ Penalties for anomalies              │    │
│      └──────────────────────────────────────┘    │
│                                                   │
└──────────────────────────────────────────────────┘
```

## Core Implementation

### Quality Scoring Function

```python
# core/quality_scorer.py
import re
from typing import Dict

def score_chunk_quality(text: str) -> float:
    """
    Score text quality on a scale of 0.0 (poor) to 1.0 (excellent).
    
    Args:
        text: Text content to score
    
    Returns:
        Quality score (0.0 - 1.0)
    
    Example:
        >>> score_chunk_quality("This is normal English text.")
        0.95
        >>> score_chunk_quality("asdf jkl; qwer zxcv")
        0.25
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    
    # Initialize score
    score = 1.0
    
    # Calculate individual metrics
    metrics = {
        "char_distribution": _score_character_distribution(text),
        "text_structure": _score_text_structure(text),
        "content_quality": _score_content_quality(text),
        "language_features": _score_language_features(text)
    }
    
    # Weighted average
    weights = {
        "char_distribution": 0.3,
        "text_structure": 0.25,
        "content_quality": 0.25,
        "language_features": 0.2
    }
    
    score = sum(
        metrics[key] * weights[key]
        for key in metrics
    )
    
    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, score))
```

## Scoring Metrics

### Character Distribution

```python
def _score_character_distribution(text: str) -> float:
    """
    Score based on character type distribution.
    
    Checks:
        - Alphanumeric ratio (should be high)
        - Whitespace ratio (should be reasonable)
        - Special character ratio (should be low)
    """
    if not text:
        return 0.0
    
    total = len(text)
    alphanum = sum(c.isalnum() for c in text)
    whitespace = sum(c.isspace() for c in text)
    special = total - alphanum - whitespace
    
    alphanum_ratio = alphanum / total
    whitespace_ratio = whitespace / total
    special_ratio = special / total
    
    score = 1.0
    
    # Penalize low alphanumeric content
    if alphanum_ratio < 0.5:
        score *= alphanum_ratio / 0.5
    
    # Penalize excessive special characters
    if special_ratio > 0.2:
        score *= (1.0 - (special_ratio - 0.2) * 2)
    
    # Penalize excessive whitespace
    if whitespace_ratio > 0.4:
        score *= (1.0 - (whitespace_ratio - 0.4) * 2)
    
    return max(0.0, score)
```

### Text Structure

```python
import statistics

def _score_text_structure(text: str) -> float:
    """
    Score based on text structure patterns.
    
    Checks:
        - Average word length (3-10 chars is normal)
        - Average sentence length (10-40 words is normal)
        - Line length consistency
    """
    score = 1.0
    
    # Word length analysis
    words = text.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        
        # Normal range: 3-10 characters
        if avg_word_len < 2 or avg_word_len > 15:
            score *= 0.5
        elif avg_word_len < 3 or avg_word_len > 10:
            score *= 0.8
    
    # Sentence length analysis
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if sentences:
        sentence_word_counts = [len(s.split()) for s in sentences]
        avg_sentence_len = statistics.mean(sentence_word_counts)
        
        # Normal range: 10-40 words
        if avg_sentence_len < 3 or avg_sentence_len > 100:
            score *= 0.5
        elif avg_sentence_len < 5 or avg_sentence_len > 50:
            score *= 0.8
    
    # Line length consistency
    lines = text.split('\n')
    lines = [line for line in lines if line.strip()]
    
    if len(lines) > 3:
        line_lengths = [len(line) for line in lines]
        try:
            stdev = statistics.stdev(line_lengths)
            mean = statistics.mean(line_lengths)
            
            # High variance suggests inconsistent formatting
            if stdev / mean > 2.0:
                score *= 0.7
        except:
            pass
    
    return max(0.0, score)
```

### Content Quality

```python
def _score_content_quality(text: str) -> float:
    """
    Score based on content indicators.
    
    Checks:
        - Repeated patterns (gibberish)
        - Excessive URLs/emails
        - Meaningless character sequences
    """
    score = 1.0
    
    # Check for repeated short patterns (e.g., "aaaa", "1111")
    for length in [2, 3, 4]:
        pattern = r'(.{' + str(length) + r'})\1{3,}'
        matches = re.findall(pattern, text)
        if matches:
            score *= 0.6
            break
    
    # Check for excessive URLs
    url_pattern = r'https?://\S+'
    url_count = len(re.findall(url_pattern, text))
    if url_count > len(text) / 100:  # More than 1 URL per 100 chars
        score *= 0.7
    
    # Check for excessive email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_count = len(re.findall(email_pattern, text))
    if email_count > len(text) / 200:
        score *= 0.8
    
    # Check for gibberish (sequences of consonants/vowels)
    consonant_seq = re.findall(r'[bcdfghjklmnpqrstvwxyz]{6,}', text.lower())
    if len(consonant_seq) > 2:
        score *= 0.5
    
    # Check for numeric-heavy content
    digits = sum(c.isdigit() for c in text)
    if digits > len(text) * 0.5:
        score *= 0.7
    
    return max(0.0, score)
```

### Language Features

```python
def _score_language_features(text: str) -> float:
    """
    Score based on natural language features.
    
    Checks:
        - Vowel ratio (English: ~40%)
        - Common word presence
        - Capitalization patterns
    """
    score = 1.0
    
    # Vowel ratio
    vowels = 'aeiouAEIOU'
    letters = [c for c in text if c.isalpha()]
    
    if letters:
        vowel_ratio = sum(c in vowels for c in letters) / len(letters)
        
        # Expected range: 35-45%
        if vowel_ratio < 0.2 or vowel_ratio > 0.6:
            score *= 0.5
        elif vowel_ratio < 0.3 or vowel_ratio > 0.5:
            score *= 0.8
    
    # Common English words presence
    common_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
        'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
        'do', 'at', 'this', 'but', 'his', 'by', 'from'
    }
    
    words_lower = [w.lower() for w in text.split()]
    common_count = sum(w in common_words for w in words_lower)
    
    if words_lower and common_count / len(words_lower) < 0.05:
        score *= 0.7
    
    # Excessive capitalization
    if letters:
        caps_ratio = sum(c.isupper() for c in letters) / len(letters)
        
        if caps_ratio > 0.5:  # More than 50% caps
            score *= 0.6
    
    return max(0.0, score)
```

## Quality Analysis

### Detailed Quality Report

```python
def analyze_chunk_quality(text: str) -> Dict:
    """
    Generate detailed quality analysis report.
    
    Returns:
        Dict with:
            - overall_score: Combined score
            - metrics: Individual metric scores
            - issues: List of detected issues
            - recommendations: Suggested actions
    """
    metrics = {
        "char_distribution": _score_character_distribution(text),
        "text_structure": _score_text_structure(text),
        "content_quality": _score_content_quality(text),
        "language_features": _score_language_features(text)
    }
    
    overall_score = score_chunk_quality(text)
    
    # Identify issues
    issues = []
    if metrics["char_distribution"] < 0.5:
        issues.append("Poor character distribution")
    if metrics["text_structure"] < 0.5:
        issues.append("Unusual text structure")
    if metrics["content_quality"] < 0.5:
        issues.append("Low content quality (gibberish/spam)")
    if metrics["language_features"] < 0.5:
        issues.append("Unnatural language features")
    
    # Recommendations
    recommendations = []
    if overall_score < 0.3:
        recommendations.append("Consider filtering this chunk")
    elif overall_score < 0.5:
        recommendations.append("Review manually before including")
    else:
        recommendations.append("Good quality chunk")
    
    return {
        "overall_score": overall_score,
        "metrics": metrics,
        "issues": issues,
        "recommendations": recommendations
    }
```

## Filtering Integration

### Filter Low-Quality Chunks

```python
from config.settings import settings

def filter_chunks_by_quality(
    chunks: List[Dict],
    min_score: float = None
) -> tuple[List[Dict], List[Dict]]:
    """
    Filter chunks based on quality score.
    
    Args:
        chunks: List of chunk dicts with 'text'
        min_score: Minimum quality score (default from settings)
    
    Returns:
        Tuple of (good_chunks, filtered_chunks)
    """
    if min_score is None:
        min_score = settings.min_quality_score or 0.3
    
    good_chunks = []
    filtered_chunks = []
    
    for chunk in chunks:
        score = score_chunk_quality(chunk["text"])
        chunk["quality_score"] = score
        
        if score >= min_score:
            good_chunks.append(chunk)
        else:
            filtered_chunks.append(chunk)
    
    logger.info(
        f"Quality filtering: {len(good_chunks)} passed, "
        f"{len(filtered_chunks)} filtered (threshold: {min_score})"
    )
    
    return good_chunks, filtered_chunks
```

### Mark Instead of Filter

```python
def mark_low_quality_chunks(
    chunks: List[Dict],
    threshold: float = 0.5
) -> List[Dict]:
    """
    Mark low-quality chunks instead of filtering.
    
    Adds 'is_low_quality' flag for downstream processing.
    """
    for chunk in chunks:
        score = score_chunk_quality(chunk["text"])
        chunk["quality_score"] = score
        chunk["is_low_quality"] = score < threshold
    
    return chunks
```

## Configuration

### Environment Variables

```bash
# Quality scoring
ENABLE_QUALITY_SCORING=true
MIN_QUALITY_SCORE=0.3       # Threshold for filtering (0.0-1.0)
QUALITY_FILTER_MODE=mark    # "filter" or "mark"
```

### Settings

```python
# config/settings.py
class Settings(BaseSettings):
    enable_quality_scoring: bool = True
    min_quality_score: float = 0.3
    quality_filter_mode: str = "mark"  # "filter" or "mark"
```

## Usage Examples

### Score Single Chunk

```python
from core.quality_scorer import score_chunk_quality

text = "This is a well-formed English sentence with proper structure."
score = score_chunk_quality(text)

print(f"Quality score: {score:.2f}")
# Output: Quality score: 0.95
```

### Analyze Quality

```python
from core.quality_scorer import analyze_chunk_quality

text = "asdf jkl qwer zxcv 1234 @#$%"
report = analyze_chunk_quality(text)

print(f"Score: {report['overall_score']:.2f}")
print(f"Issues: {', '.join(report['issues'])}")
print(f"Recommendation: {report['recommendations'][0]}")
```

### Filter Document Chunks

```python
from core.quality_scorer import filter_chunks_by_quality

chunks = [
    {"id": "c1", "text": "Normal English text."},
    {"id": "c2", "text": "asdf jkl qwer zxcv"},
    {"id": "c3", "text": "Another good chunk."}
]

good, filtered = filter_chunks_by_quality(chunks, min_score=0.5)

print(f"Accepted: {len(good)}")
print(f"Filtered: {len(filtered)}")
```

### Integration with Ingestion

```python
from ingestion.document_processor import DocumentProcessor

class DocumentProcessor:
    async def process_document(self, file_path: str):
        # ... load and chunk ...
        
        # Score chunks
        if settings.enable_quality_scoring:
            if settings.quality_filter_mode == "filter":
                chunks, filtered = filter_chunks_by_quality(chunks)
                logger.info(f"Filtered {len(filtered)} low-quality chunks")
            else:
                chunks = mark_low_quality_chunks(chunks)
        
        # ... continue processing ...
```

## Batch Analysis

### Analyze Document Quality Distribution

```python
def analyze_document_quality(chunks: List[Dict]) -> Dict:
    """Analyze quality distribution across document."""
    scores = [score_chunk_quality(c["text"]) for c in chunks]
    
    return {
        "total_chunks": len(chunks),
        "avg_score": statistics.mean(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "median_score": statistics.median(scores),
        "low_quality_count": sum(s < 0.3 for s in scores),
        "medium_quality_count": sum(0.3 <= s < 0.7 for s in scores),
        "high_quality_count": sum(s >= 0.7 for s in scores)
    }
```

## Performance

### Scoring Speed

```python
import time

def benchmark_quality_scoring(chunks: List[str], iterations: int = 100):
    """Benchmark quality scoring performance."""
    start = time.time()
    
    for _ in range(iterations):
        for chunk in chunks:
            score_chunk_quality(chunk)
    
    elapsed = time.time() - start
    total_ops = len(chunks) * iterations
    ops_per_sec = total_ops / elapsed
    
    print(f"Scoring rate: {ops_per_sec:.0f} chunks/sec")
    print(f"Avg time per chunk: {elapsed / total_ops * 1000:.2f}ms")
```

**Typical Performance**:
- ~1000-5000 chunks/second on modern CPU
- <1ms per chunk for typical sizes

## Testing

### Unit Tests

```python
import pytest
from core.quality_scorer import score_chunk_quality, analyze_chunk_quality

def test_good_quality_text():
    text = "This is a well-written English paragraph with proper grammar."
    score = score_chunk_quality(text)
    assert score > 0.8

def test_poor_quality_text():
    text = "asdf jkl qwer zxcv 1234 !@#$"
    score = score_chunk_quality(text)
    assert score < 0.5

def test_empty_text():
    score = score_chunk_quality("")
    assert score == 0.0

def test_repeated_pattern():
    text = "aaaa aaaa aaaa aaaa"
    score = score_chunk_quality(text)
    assert score < 0.7

def test_quality_analysis():
    text = "Normal text content."
    report = analyze_chunk_quality(text)
    
    assert "overall_score" in report
    assert "metrics" in report
    assert "issues" in report
    assert "recommendations" in report

@pytest.mark.parametrize("text,expected_min", [
    ("The quick brown fox jumps.", 0.8),
    ("Random: asdf qwer zxcv", 0.3),
    ("http://url.com http://url2.com", 0.5)
])
def test_quality_scoring_cases(text, expected_min):
    score = score_chunk_quality(text)
    assert score >= expected_min or score < expected_min
```

## Troubleshooting

### Common Issues

**Issue**: Too many chunks filtered
```python
# Solution: Lower threshold
MIN_QUALITY_SCORE=0.2

# Or use marking instead of filtering
QUALITY_FILTER_MODE=mark
```

**Issue**: Low-quality chunks passing through
```python
# Solution: Raise threshold
MIN_QUALITY_SCORE=0.5

# Or add custom quality checks
def custom_quality_check(text: str) -> bool:
    # Add domain-specific checks
    return "specific_keyword" in text.lower()
```

**Issue**: False positives on technical content
```python
# Solution: Adjust scoring weights
weights = {
    "char_distribution": 0.4,  # Increase
    "content_quality": 0.1,    # Decrease for code/technical
}
```

## Related Documentation

- [Document Processor](03-components/ingestion/document-processor.md)
- [Chunking Strategy](03-components/ingestion/chunking.md)
- [Retrieval Strategies](02-core-concepts/retrieval-strategies.md)
- [Configuration Reference](07-configuration/rag-tuning.md)
