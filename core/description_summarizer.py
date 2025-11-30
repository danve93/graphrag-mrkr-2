"""
Description summarization for entity graph.

This module provides LLM-based summarization of accumulated entity and
relationship descriptions. Used after NetworkX deduplication to create
concise, cohesive descriptions from multiple mentions.

Key features:
- Batch summarization (multiple entities per LLM call)
- Deduplication-aware prompts
- Caching (avoid re-summarizing same descriptions)
- Fallback to original descriptions on error
- Token budget management
"""

import logging
import hashlib
import re
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.llm import llm_manager
from core.singletons import get_blocking_executor, SHUTTING_DOWN
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SummarizationResult:
    """Result of description summarization."""
    entity_name: str
    original_description: str
    summarized_description: str
    original_length: int
    summarized_length: int
    compression_ratio: float
    error: Optional[str] = None


class DescriptionSummarizer:
    """
    LLM-based description summarizer for entity graphs.
    
    This class handles:
    - Identifying entities/relationships needing summarization
    - Batching summarization requests (efficiency)
    - Prompt engineering for deduplication + synthesis
    - Caching summaries (same description hash → cached result)
    - Error handling (fallback to original description)
    
    Usage:
        summarizer = DescriptionSummarizer()
        
        # Summarize entity descriptions
        results = await summarizer.summarize_entity_descriptions(
            entities=[
                ("ADMIN PANEL", "desc1\\ndesc2\\ndesc3", 3),
                ("USER DATABASE", "desc1\\ndesc2", 2),
            ]
        )
        
        # Apply summaries to graph
        for result in results:
            graph.update_entity_description(result.entity_name, result.summarized_description)
    
    Configuration:
        settings.enable_description_summarization: Enable/disable summarization
        settings.summarization_min_mentions: Min mentions to trigger (default: 3)
        settings.summarization_min_length: Min description length (default: 200)
        settings.summarization_batch_size: Entities per LLM call (default: 5)
        settings.summarization_cache_enabled: Enable caching (default: True)
    """
    
    def __init__(self):
        """
        Initialize summarizer.
        """
        # Summarization cache: description_hash → summary
        self._summary_cache: Dict[str, str] = {}
        
        # Statistics
        self._stats = {
            "entities_summarized": 0,
            "relationships_summarized": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "summarization_errors": 0,
            "total_original_length": 0,
            "total_summarized_length": 0,
        }
        
        logger.info(
            f"DescriptionSummarizer initialized "
            f"(enabled: {settings.enable_description_summarization})"
        )
    
    async def summarize_entity_descriptions(
        self,
        entities: List[Tuple[str, str, int]]
    ) -> List[SummarizationResult]:
        """
        Summarize entity descriptions.
        
        Args:
            entities: List of (entity_name, accumulated_description, mention_count) tuples
        
        Returns:
            List of SummarizationResult objects
        
        Algorithm:
        1. Filter entities needing summarization (mention_count ≥ threshold)
        2. Check cache for each description (hash-based lookup)
        3. Batch uncached entities (batch_size per LLM call)
        4. For each batch:
           a. Build summarization prompt
           b. Call LLM
           c. Parse response
           d. Cache results
        5. Return all results (cached + new)
        """
        if not settings.enable_description_summarization:
            logger.debug("Description summarization disabled")
            return []
        
        # Filter entities needing summarization
        entities_to_summarize = []
        
        for entity_name, description, mention_count in entities:
            # Check if summarization needed
            if not self._should_summarize(description, mention_count):
                logger.debug(
                    f"Skipping summarization for '{entity_name}' "
                    f"(mentions: {mention_count}, length: {len(description)})"
                )
                continue
            
            entities_to_summarize.append((entity_name, description, mention_count))
        
        if not entities_to_summarize:
            logger.info("No entities require summarization")
            return []
        
        logger.info(f"Summarizing {len(entities_to_summarize)} entity descriptions")
        
        # Process in batches
        results = []
        batch_size = settings.summarization_batch_size
        
        for i in range(0, len(entities_to_summarize), batch_size):
            batch = entities_to_summarize[i:i + batch_size]
            
            logger.debug(f"Processing batch {i // batch_size + 1} ({len(batch)} entities)")
            
            batch_results = await self._summarize_batch(batch, entity_type=True)
            results.extend(batch_results)
        
        # Update statistics
        self._update_statistics(results, entity_type=True)
        
        logger.info(
            f"Entity summarization complete: {len(results)} entities, "
            f"{self._stats['cache_hits']} cache hits, "
            f"{self._stats['summarization_errors']} errors"
        )
        
        return results
    
    async def summarize_relationship_descriptions(
        self,
        relationships: List[Tuple[str, str, str, str, int]]
    ) -> List[SummarizationResult]:
        """
        Summarize relationship descriptions.
        
        Args:
            relationships: List of (source, target, rel_type, description, mention_count) tuples
        
        Returns:
            List of SummarizationResult objects
        
        Note: Similar to entity summarization but formats relationship identifier as:
              "SOURCE -> TARGET (REL_TYPE)"
        """
        if not settings.enable_description_summarization:
            return []
        
        # Filter relationships needing summarization
        relationships_to_summarize = []
        
        for source, target, rel_type, description, mention_count in relationships:
            if not self._should_summarize(description, mention_count):
                continue
            
            # Create relationship identifier
            rel_id = f"{source} -> {target} ({rel_type})"
            relationships_to_summarize.append((rel_id, description, mention_count))
        
        if not relationships_to_summarize:
            logger.info("No relationships require summarization")
            return []
        
        logger.info(f"Summarizing {len(relationships_to_summarize)} relationship descriptions")
        
        # Process in batches
        results = []
        batch_size = settings.summarization_batch_size
        
        for i in range(0, len(relationships_to_summarize), batch_size):
            batch = relationships_to_summarize[i:i + batch_size]
            batch_results = await self._summarize_batch(batch, entity_type=False)
            results.extend(batch_results)
        
        self._update_statistics(results, entity_type=False)
        
        logger.info(
            f"Relationship summarization complete: {len(results)} relationships"
        )
        
        return results
    
    async def _summarize_batch(
        self,
        batch: List[Tuple[str, str, int]],
        entity_type: bool = True
    ) -> List[SummarizationResult]:
        """
        Summarize a batch of descriptions.
        
        Args:
            batch: List of (identifier, description, mention_count) tuples
            entity_type: True for entities, False for relationships
        
        Returns:
            List of SummarizationResult objects
        """
        results = []
        
        # Check cache first
        uncached_items = []
        
        for identifier, description, mention_count in batch:
            # Compute description hash for cache lookup
            desc_hash = self._compute_description_hash(description)
            
            if settings.summarization_cache_enabled and desc_hash in self._summary_cache:
                # Cache hit
                cached_summary = self._summary_cache[desc_hash]
                
                logger.debug(f"Cache hit for '{identifier}'")
                self._stats["cache_hits"] += 1
                
                results.append(SummarizationResult(
                    entity_name=identifier,
                    original_description=description,
                    summarized_description=cached_summary,
                    original_length=len(description),
                    summarized_length=len(cached_summary),
                    compression_ratio=len(cached_summary) / len(description) if description else 1.0
                ))
            else:
                # Cache miss, needs summarization
                uncached_items.append((identifier, description, mention_count))
                self._stats["cache_misses"] += 1
        
        if not uncached_items:
            # All items were cached
            return results
        
        # Build prompt for uncached items
        prompt = self._build_summarization_prompt(uncached_items, entity_type)
        
        # Call LLM
        try:
            logger.debug(f"Calling LLM for {len(uncached_items)} descriptions")
            
            # Use thread executor to run synchronous LLM call
            loop = asyncio.get_running_loop()
            try:
                executor = get_blocking_executor()
                response = await loop.run_in_executor(
                    executor,
                    lambda: llm_manager.generate_response(
                        prompt=prompt,
                        temperature=0.3,  # Low temperature for consistency
                        max_tokens=1000,  # Batch response can be longer
                    ),
                )
            except RuntimeError as e:
                logger.debug(f"Blocking executor unavailable for summarization: {e}.")
                if SHUTTING_DOWN:
                    logger.info("Process shutting down; aborting summarization batch")
                    raise
                try:
                    executor = get_blocking_executor()
                    response = await loop.run_in_executor(
                        executor,
                        lambda: llm_manager.generate_response(
                            prompt=prompt,
                            temperature=0.3,
                            max_tokens=1000,
                        ),
                    )
                except Exception as e2:
                    logger.error(f"Summarization scheduling failed: {e2}")
                    raise
            
            # Parse response
            summaries = self._parse_summarization_response(response, uncached_items)
            
            # Create results and cache
            for (identifier, description, mention_count), summary in zip(uncached_items, summaries):
                # Cache summary
                if settings.summarization_cache_enabled:
                    desc_hash = self._compute_description_hash(description)
                    self._summary_cache[desc_hash] = summary
                
                # Create result
                results.append(SummarizationResult(
                    entity_name=identifier,
                    original_description=description,
                    summarized_description=summary,
                    original_length=len(description),
                    summarized_length=len(summary),
                    compression_ratio=len(summary) / len(description) if description else 1.0
                ))
        
        except Exception as e:
            # Summarization failed, fallback to original descriptions
            logger.error(f"Summarization failed: {str(e)}")
            self._stats["summarization_errors"] += 1
            
            for identifier, description, mention_count in uncached_items:
                results.append(SummarizationResult(
                    entity_name=identifier,
                    original_description=description,
                    summarized_description=description,  # Fallback to original
                    original_length=len(description),
                    summarized_length=len(description),
                    compression_ratio=1.0,
                    error=str(e)
                ))
        
        return results
    
    def _should_summarize(self, description: str, mention_count: int) -> bool:
        """
        Check if description should be summarized.
        
        Criteria:
        - mention_count ≥ summarization_min_mentions (default: 3)
        - description length ≥ summarization_min_length (default: 200)
        - description is not empty
        
        Args:
            description: Accumulated description
            mention_count: Number of times entity was mentioned
        
        Returns:
            True if summarization should be performed
        """
        if not description or not description.strip():
            return False
        
        if mention_count < settings.summarization_min_mentions:
            return False
        
        if len(description) < settings.summarization_min_length:
            return False
        
        return True
    
    def _build_summarization_prompt(
        self,
        items: List[Tuple[str, str, int]],
        entity_type: bool
    ) -> str:
        """
        Build LLM prompt for summarization.
        
        Prompt engineering principles:
        - Explicit instructions (deduplicate, synthesize, preserve details)
        - Output format (one line per entity, numbered)
        - Examples (show desired output style)
        - Context (explain why descriptions are fragmented)
        
        Args:
            items: List of (identifier, description, mention_count) tuples
            entity_type: True for entities, False for relationships
        
        Returns:
            Formatted prompt string
        """
        item_type = "entity" if entity_type else "relationship"
        
        prompt = f"""# Description Summarization

You are summarizing accumulated {item_type} descriptions from a knowledge graph extraction process.

## Context

These descriptions were extracted from multiple document chunks and concatenated. They contain:
- Redundant information (same facts repeated)
- Fragmented sentences (each from different chunk)
- Repetitive entity names

## Task

For each {item_type} below, create a single concise summary that:
1. **Deduplicates**: Remove redundant/repeated information
2. **Synthesizes**: Combine fragments into cohesive description
3. **Preserves**: Keep all unique technical details, capabilities, relationships
4. **Concise**: Aim for 2-3 sentences maximum
5. **Complete**: Make summary self-contained (include entity name if needed for clarity)

## Output Format

Provide summaries in this format:
```
1. [First summary]
2. [Second summary]
3. [Third summary]
...
```

One summary per line, numbered. Do NOT include the {item_type} name/identifier in the summary (it's already tracked).

## Examples

### Bad Summary (redundant, fragmented)
"Admin Panel is a component. Admin Panel provides user management. Admin Panel is used for administration."

### Good Summary (concise, synthesized)
"Web-based administration interface providing user management, role-based access control, and system configuration capabilities."

## {item_type.capitalize()}s to Summarize

"""
        
        # Add each item with its accumulated description
        for idx, (identifier, description, mention_count) in enumerate(items, start=1):
            prompt += f"""
### {idx}. {identifier} (mentioned {mention_count} times)

Accumulated descriptions:
```
{description}
```

"""
        
        prompt += """
## Your Output

Provide numbered summaries (one per line):
"""
        
        return prompt
    
    def _parse_summarization_response(
        self,
        response: str,
        items: List[Tuple[str, str, int]]
    ) -> List[str]:
        """
        Parse LLM summarization response.
        
        Expected format:
        1. First summary
        2. Second summary
        3. Third summary
        
        Args:
            response: LLM response text
            items: Original items (for count validation)
        
        Returns:
            List of summary strings (same order as items)
        
        Error handling:
        - If parsing fails, return original descriptions
        - If count mismatch, pad with original descriptions
        """
        summaries = []
        
        # Split by lines
        lines = response.strip().split('\n')
        
        # Extract numbered summaries
        number_pattern = re.compile(r'^\s*(\d+)\.\s*(.+)$')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = number_pattern.match(line)
            if match:
                summary_text = match.group(2).strip()
                summaries.append(summary_text)
        
        # Validate count
        if len(summaries) != len(items):
            logger.warning(
                f"Summary count mismatch: expected {len(items)}, got {len(summaries)}"
            )
            
            # Pad with original descriptions if needed
            while len(summaries) < len(items):
                idx = len(summaries)
                original_description = items[idx][1]
                summaries.append(original_description)
                logger.debug(f"Padded summary {idx + 1} with original description")
        
        return summaries[:len(items)]  # Truncate if too many
    
    def _compute_description_hash(self, description: str) -> str:
        """
        Compute hash of description for cache lookup.
        
        Uses SHA256 for collision resistance.
        
        Args:
            description: Description text
        
        Returns:
            Hex digest hash string
        """
        return hashlib.sha256(description.encode('utf-8')).hexdigest()
    
    def _update_statistics(self, results: List[SummarizationResult], entity_type: bool = True) -> None:
        """Update summarization statistics."""
        for result in results:
            if result.error:
                continue
            
            if entity_type:
                self._stats["entities_summarized"] += 1
            else:
                self._stats["relationships_summarized"] += 1
            
            self._stats["total_original_length"] += result.original_length
            self._stats["total_summarized_length"] += result.summarized_length
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get summarization statistics.
        
        Returns:
            Dict with metrics:
            - entities_summarized
            - relationships_summarized
            - cache_hits / cache_misses
            - average_compression_ratio
            - total_tokens_saved (estimated)
        """
        stats = self._stats.copy()
        
        # Calculate average compression ratio
        if stats["total_original_length"] > 0:
            stats["average_compression_ratio"] = (
                stats["total_summarized_length"] / stats["total_original_length"]
            )
        else:
            stats["average_compression_ratio"] = 1.0
        
        # Estimate tokens saved (rough: 1 token ≈ 4 characters)
        chars_saved = stats["total_original_length"] - stats["total_summarized_length"]
        stats["estimated_tokens_saved"] = chars_saved // 4
        
        return stats
