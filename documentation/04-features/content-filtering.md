# Content Quality Filtering

## Quick Start
Heuristic-based content filtering reduces embedding costs by 50-90% by filtering out low-quality chunks before expensive embedding operations. Uses programmatic rules (no LLM calls) to identify repetitive, malformed, or low-value content.

## Use Cases
1. **Cost Optimization** - Reduce embedding costs on noisy data sources (chat logs, support tickets, auto-generated content)
2. **Quality Improvement** - Improve retrieval quality by reducing noise in the vector index
3. **Performance** - Faster indexing by generating fewer embeddings

## How It Works
The filter applies a series of heuristic checks to each chunk before embedding:

1. **Length validation** - Filters chunks that are too short (<50 chars) or too long (>100K chars)
2. **Repetition detection** - Identifies highly repetitive content (low unique word ratio)
3. **Character distribution** - Detects garbage data (excessive special characters, low alphanumeric content)
4. **Conversation quality** - Filters low-value chat threads (too short, no resolution, spam)
5. **Structured data quality** - Filters empty tables, single-column data, header-only tables
6. **Code quality** - Filters comment-only blocks, auto-generated code

All filtering is based on simple programmatic rules - **no LLM calls required**.

## Configuration

### Enabling Content Filtering

```python
# config/settings.py or .env
enable_content_filtering = True  # Default: False
```

### Basic Settings

```python
# Minimum and maximum chunk length
content_filter_min_length = 50  # Default: 50 characters
content_filter_max_length = 100000  # Default: 100,000 characters

# Content quality thresholds
content_filter_unique_ratio = 0.3  # Min ratio of unique words (0.0-1.0)
content_filter_max_special_char_ratio = 0.5  # Max special char ratio
content_filter_min_alphanumeric_ratio = 0.3  # Min alphanumeric ratio
```

### Feature Toggles

```python
# Enable/disable specific filter types
content_filter_enable_conversation = True  # Conversation thread filtering
content_filter_enable_structured = True  # Structured data filtering
content_filter_enable_code = True  # Code quality filtering
```

### Environment Variables

```bash
# .env
ENABLE_CONTENT_FILTERING=true
CONTENT_FILTER_MIN_LENGTH=50
CONTENT_FILTER_UNIQUE_RATIO=0.3
```

## Usage Examples

### Example 1: Default Configuration
```python
# In your .env or config
ENABLE_CONTENT_FILTERING=true

# Process a document - filtering happens automatically
from ingestion.document_processor import DocumentProcessor

processor = DocumentProcessor()
result = processor.process_file("documents/noisy_chat_log.txt")

# Check filtering metrics in logs:
# INFO: Content filtering complete for doc123:
#       150/500 chunks passed (30.0% pass rate, 70.0% filtered)
# INFO: Filter reasons: {'too_short': 200, 'repetitive': 100, 'spam_detected': 50}
```

**Expected Output:**
- 70% of low-quality chunks filtered out
- Significant cost savings on embedding generation
- Better retrieval quality (less noise)

### Example 2: Strict Filtering for Noisy Data
```python
# config/settings.py
content_filter_min_length = 100  # Raise minimum length
content_filter_unique_ratio = 0.5  # Require more diverse vocabulary
content_filter_max_special_char_ratio = 0.3  # Lower tolerance for special chars
```

Use this for very noisy sources like raw chat logs or social media data.

### Example 3: Lenient Filtering for High-Quality Sources
```python
# config/settings.py
content_filter_min_length = 30  # Allow shorter chunks
content_filter_unique_ratio = 0.2  # Allow more repetition
content_filter_enable_conversation = False  # Disable conversation filters
```

Use this for curated documentation or well-structured content.

### Example 4: Code-Specific Filtering
```python
# Only enable code filtering, disable others
content_filter_enable_conversation = False
content_filter_enable_structured = False
content_filter_enable_code = True
```

Useful when processing code repositories to filter out comment-only files and auto-generated code.

## Filtering Logic Details

### Length Filters
- **Too Short**: Chunks < `content_filter_min_length` characters
- **Too Long**: Chunks > `content_filter_max_length` characters (likely unprocessed data)

### Repetition Filters
- **Low Unique Ratio**: `unique_words / total_words < content_filter_unique_ratio`
- **Single Word Repetition**: One word appears >70% of the time

### Character Distribution Filters
- **Low Alphanumeric**: `alphanumeric_chars / total_chars < content_filter_min_alphanumeric_ratio`
- **High Special Chars**: `special_chars / total_chars > content_filter_max_special_char_ratio`
- **Exception**: Code files (.py, .js, .html) allow higher special character ratios

### Conversation Filters
Enabled when chunk metadata indicates conversation content:
- **Thread Too Short**: `message_count < 3`
- **No Resolution**: Single participant AND no resolution keywords (solved, fixed, thanks, working)
- **Spam Detection**: Pattern matching for common spam (viagra, prizes, click here)

### Structured Data Filters
Enabled when chunk metadata indicates CSV/table content:
- **Empty Table**: `row_count == 0`
- **Single Column**: `column_count == 1` (usually not useful)
- **Header Only**: `row_count == 1` (no data rows)

### Code Filters
Enabled when chunk metadata indicates code content:
- **Comment Only**: 100% comment lines, 0% code lines
- **Mostly Comments**: >80% comment lines
- **Auto-Generated**: Contains markers like "auto-generated", "do not edit"
  - Exception: Allowed if >10 lines of actual code

## Performance Considerations

**Latency**: <1ms per chunk (simple heuristics, no LLM calls)

**Memory Usage**: Minimal (operates on individual chunks)

**Scalability**: Processes 10,000+ chunks per second

**Cost Impact**:
- 50-70% embedding cost reduction on typical noisy data
- 70-90% embedding cost reduction on very noisy data (chat logs, social media)
- 10-30% embedding cost reduction on well-curated content

## Monitoring & Metrics

### Log Messages
```
INFO: Applying content quality filtering to 500 chunks
INFO: Content filtering complete for doc123:
      200/500 chunks passed (40.0% pass rate, 60.0% filtered)
INFO: Filter reasons: {'too_short': 150, 'repetitive': 100, 'spam_detected': 50}
```

### Metrics Tracked
- **total_chunks**: Total chunks evaluated
- **passed_chunks**: Chunks that passed filtering
- **filtered_chunks**: Chunks that were filtered out
- **pass_rate**: Percentage that passed
- **filter_rate**: Percentage filtered out
- **filter_reasons**: Breakdown by filter type

### Interpreting Results

**High Filter Rate (>70%)**:
- Good! You're filtering out lots of noise
- Verify a few filtered chunks to ensure no false positives
- Consider if source data quality can be improved upstream

**Low Filter Rate (<20%)**:
- Data is already high quality, or
- Filters may be too lenient for your use case
- Consider adjusting thresholds

**Specific Filter Reasons**:
- High `too_short`: Chunking strategy may need adjustment
- High `repetitive`: Common in logs, transcripts (expected)
- High `spam_detected`: Social media or public forum data

## Troubleshooting

### Issue: Too many chunks filtered (>90%)
**Symptoms:** Very high filter rate, loss of useful content

**Cause:** Thresholds too strict for your content type

**Solution:**
```python
# Lower the minimum length for short-form content
content_filter_min_length = 30

# Allow more repetition for logs/transcripts
content_filter_unique_ratio = 0.2

# Check which filter is responsible
# Look at filter_reasons in logs
```

### Issue: Not enough chunks filtered (<10%)
**Symptoms:** Still seeing low-quality content in search results

**Cause:** Thresholds too lenient, or wrong filter types disabled

**Solution:**
```python
# Raise quality bar
content_filter_min_length = 100
content_filter_unique_ratio = 0.5

# Enable all filter types
content_filter_enable_conversation = True
content_filter_enable_structured = True
content_filter_enable_code = True
```

### Issue: Important content being filtered
**Symptoms:** Search misses known content, gaps in knowledge base

**Cause:** Over-aggressive filtering, or content type mismatch

**Solution:**
1. Check logs for specific filter reasons
2. Disable the problematic filter:
   ```python
   # If filtering good code
   content_filter_enable_code = False

   # If filtering good short content
   content_filter_min_length = 20
   ```
3. Add metadata to mark important content as exempt

### Issue: Performance degradation
**Symptoms:** Slower document processing

**Cause:** Unlikely - filtering adds <1ms per chunk

**Solution:**
- Verify filtering is the cause (disable and compare)
- Check if other factors (network, database) are involved

## FAQ

**Q: When should I enable content filtering?**
A: Enable when:
- Processing noisy data sources (chat logs, support tickets, social media)
- Embedding costs are significant
- Search quality is degraded by low-value content

**Q: Will this filter out important content?**
A: Unlikely with default settings. The filters use conservative thresholds. Monitor the `filter_reasons` logs to verify. You can always adjust thresholds or disable specific filters.

**Q: How much cost savings can I expect?**
A: Depends on data quality:
- Well-curated content: 10-30% reduction
- Typical business content: 50-70% reduction
- Very noisy sources: 70-90% reduction

**Q: Can I customize filtering logic?**
A: Yes! The filter is modular:
1. Extend `ContentQualityFilter` class
2. Override specific filter methods
3. Add custom filter logic
4. See `ingestion/content_filters.py` for implementation

**Q: What's the performance impact?**
A: Minimal - filtering adds <1ms per chunk. The benefit (fewer embeddings to generate) far outweighs the cost.

**Q: Does this work with all file types?**
A: Yes! Filtering is format-agnostic but has special logic for:
- Code files (allow higher special character ratio)
- Structured data (check row/column counts)
- Conversations (check engagement and resolution)

**Q: Can I see what was filtered?**
A: Yes! Check the logs:
```
INFO: Filter reasons: {'too_short': 150, 'repetitive': 100}
```

You can also enable debug logging to see individual filtered chunks.

## Related Features
- [Document Processing](../03-components/backend/document-processing.md)
- [Chunking Strategy](../02-core-concepts/chunking.md)
- [Embedding Generation](../03-components/backend/embeddings.md)

## Changelog
- **v2.0.0**: Content filtering feature added
  - Heuristic-based filtering before embedding
  - 8 configurable filter types
  - 50-90% cost reduction on noisy data
