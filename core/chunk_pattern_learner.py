"""
Chunk Pattern Learner - AI-based pattern detection for chunk improvement suggestions.

Provides intelligent suggestions for chunk management based on:
- Patterns defined in ChunkPatternStore (regex, length, content, similarity)
- Statistical analysis of chunk characteristics
- Learning from user edit history
"""

import logging
import re
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from datetime import datetime

from core.graph_db import graph_db
from core.chunk_change_log import get_change_log
from core.chunk_pattern_store import get_pattern_store, ChunkPattern

logger = logging.getLogger(__name__)


class SuggestionAction(str, Enum):
    """Types of chunk improvement actions."""
    DELETE = "delete"
    MERGE = "merge"
    EDIT = "edit"
    SPLIT = "split"
    FLAG = "flag"


@dataclass
class ChunkSuggestion:
    """A suggestion for improving a chunk."""
    chunk_id: str
    chunk_index: int
    action: SuggestionAction
    confidence: float  # 0.0 - 1.0
    reasoning: str
    pattern_name: str
    related_chunk_ids: list[str] = field(default_factory=list)
    suggested_content: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "action": self.action.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "pattern_name": self.pattern_name,
            "related_chunk_ids": self.related_chunk_ids,
            "suggested_content": self.suggested_content,
        }


class ChunkPatternLearner:
    """Analyzes chunks and provides improvement suggestions using stored patterns."""

    def __init__(self):
        self.store = get_pattern_store()

    def has_suggestions(self, document_id: str, min_confidence: float = 0.5) -> bool:
        """Check if a document has any improvement suggestions (fast)."""
        try:
            # Load chunks
            chunks = self._load_chunks(document_id)
            if not chunks:
                return False

            # Get enabled patterns from store
            patterns = self.store.get_patterns(enabled_only=True)

            # Apply each pattern - stop at first match
            for pattern in patterns:
                matches = self._apply_pattern(pattern, chunks)
                if matches and any(m.confidence >= min_confidence for m in matches):
                    return True
            
            return False

        except Exception as e:
            logger.error("Failed to check suggestions for %s: %s", document_id, e)
            return False

    def get_suggestions(
        self,
        document_id: str,
        max_suggestions: int = 10,
        min_confidence: float = 0.5,
    ) -> list[ChunkSuggestion]:
        """
        Analyze document chunks and return improvement suggestions.
        """
        suggestions: list[ChunkSuggestion] = []
        
        try:
            # Load chunks
            chunks = self._load_chunks(document_id)
            if not chunks:
                return []

            # Get enabled patterns from store
            patterns = self.store.get_patterns(enabled_only=True)

            # Apply each pattern
            for pattern in patterns:
                matches = self._apply_pattern(pattern, chunks)
                suggestions.extend(matches)
                
                # Update usage count if pattern found matches
                if matches:
                    self.store.increment_usage(pattern.id)

            # Filter by confidence
            suggestions = [s for s in suggestions if s.confidence >= min_confidence]

            # Sort by confidence (highest first)
            suggestions.sort(key=lambda s: s.confidence, reverse=True)

            # Limit results
            return suggestions[:max_suggestions]

        except Exception as e:
            logger.error("Failed to generate suggestions for %s: %s", document_id, e)
            return []

    def _load_chunks(self, document_id: str) -> list[dict[str, Any]]:
        """Load chunks for a document."""
        try:
            with graph_db.session_scope() as session:
                result = session.run(
                    """
                    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                    RETURN c.id as id, c.content as content, c.chunk_index as chunk_index
                    ORDER BY c.chunk_index ASC
                    """,
                    doc_id=document_id,
                )
                return [dict(r) for r in result]
        except Exception as e:
            logger.error("Failed to load chunks: %s", e)
            return []

    def _apply_pattern(self, pattern: ChunkPattern, chunks: list[dict]) -> list[ChunkSuggestion]:
        """Apply a single pattern to all chunks."""
        matches = []
        criteria = pattern.match_criteria

        try:
            if pattern.match_type == "regex":
                matches = self._match_regex(pattern, chunks, criteria)
            elif pattern.match_type == "length":
                matches = self._match_length(pattern, chunks, criteria)
            elif pattern.match_type == "content":
                matches = self._match_content(pattern, chunks, criteria)
            elif pattern.match_type == "similarity":
                if criteria.get("method") == "exact_normalized":
                    matches = self._detect_duplicates(pattern, chunks)
            
        except Exception as e:
            logger.warning("Failed to apply pattern %s: %s", pattern.name, e)
            
        return matches

    def _match_regex(self, pattern: ChunkPattern, chunks: list[dict], criteria: dict) -> list[ChunkSuggestion]:
        """Match content against regex pattern."""
        matches = []
        regex_str = criteria.get("pattern")
        if not regex_str:
            return []
            
        flags = 0
        if criteria.get("flags", "").lower().count("i"):
            flags |= re.IGNORECASE
            
        try:
            regex = re.compile(regex_str, flags)
        except re.error:
            logger.warning("Invalid regex for pattern %s: %s", pattern.name, regex_str)
            return []

        for chunk in chunks:
            content = chunk.get("content", "")
            if regex.search(content):
                matches.append(ChunkSuggestion(
                    chunk_id=chunk["id"],
                    chunk_index=chunk.get("chunk_index", 0),
                    action=SuggestionAction(pattern.action),
                    confidence=pattern.confidence,
                    reasoning=f"Matched pattern: {pattern.name}",
                    pattern_name=pattern.name,
                ))
        return matches

    def _match_length(self, pattern: ChunkPattern, chunks: list[dict], criteria: dict) -> list[ChunkSuggestion]:
        """Match based on content length."""
        matches = []
        max_len = criteria.get("max_length")
        min_len = criteria.get("min_length")
        min_consecutive = criteria.get("min_consecutive", 1)

        # Simple length check
        if min_consecutive == 1:
            for chunk in chunks:
                length = len(chunk.get("content", "").strip())
                is_match = False
                
                if max_len is not None and length < max_len:
                    is_match = True
                if min_len is not None and length > min_len:
                    is_match = True
                    
                if is_match:
                    matches.append(ChunkSuggestion(
                        chunk_id=chunk["id"],
                        chunk_index=chunk.get("chunk_index", 0),
                        action=SuggestionAction(pattern.action),
                        confidence=pattern.confidence,
                        reasoning=f"Length {length} matches criteria for {pattern.name}",
                        pattern_name=pattern.name,
                    ))
            return matches

        # Consecutive chunks check (e.g. for merging)
        i = 0
        while i < len(chunks):
            run = []
            j = i
            while j < len(chunks):
                length = len(chunks[j].get("content", "").strip())
                is_match = False
                if max_len is not None and length < max_len:
                    is_match = True
                
                if not is_match:
                    break
                run.append(chunks[j])
                j += 1
            
            if len(run) >= min_consecutive:
                # Suggest merging/action on the first chunk
                matches.append(ChunkSuggestion(
                    chunk_id=run[0]["id"],
                    chunk_index=run[0].get("chunk_index", 0),
                    action=SuggestionAction(pattern.action),
                    confidence=pattern.confidence,
                    reasoning=f"Found {len(run)} consecutive chunks matching {pattern.name}",
                    pattern_name=pattern.name,
                    related_chunk_ids=[c["id"] for c in run[1:]],
                ))
                i = j  # Skip processed run
            else:
                i += 1
                
        return matches

    def _match_content(self, pattern: ChunkPattern, chunks: list[dict], criteria: dict) -> list[ChunkSuggestion]:
        """Match based on content characteristics."""
        matches = []
        min_alpha_ratio = criteria.get("min_alpha_ratio")

        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                continue

            if min_alpha_ratio is not None:
                alnum_count = sum(1 for c in content if c.isalnum())
                total = len(content)
                ratio = alnum_count / total if total > 0 else 0
                
                if ratio < min_alpha_ratio:
                    matches.append(ChunkSuggestion(
                        chunk_id=chunk["id"],
                        chunk_index=chunk.get("chunk_index", 0),
                        action=SuggestionAction(pattern.action),
                        confidence=pattern.confidence,
                        reasoning=f"Low alphanumeric ratio ({int(ratio*100)}%) matches {pattern.name}",
                        pattern_name=pattern.name,
                    ))
        return matches

    def _detect_duplicates(self, pattern: ChunkPattern, chunks: list[dict]) -> list[ChunkSuggestion]:
        """Detect duplicate or near-duplicate chunks."""
        suggestions = []
        seen: dict[str, str] = {}  # normalized content -> chunk_id
        
        for chunk in chunks:
            content = chunk.get("content", "").strip().lower()
            # Normalize whitespace
            normalized = re.sub(r'\s+', ' ', content)
            
            if len(normalized) < 20: # Skip very short duplicates
                continue

            if normalized in seen:
                suggestions.append(ChunkSuggestion(
                    chunk_id=chunk["id"],
                    chunk_index=chunk.get("chunk_index", 0),
                    action=SuggestionAction(pattern.action),
                    confidence=pattern.confidence,
                    reasoning="Duplicate content of another chunk",
                    pattern_name=pattern.name,
                    related_chunk_ids=[seen[normalized]],
                ))
            else:
                seen[normalized] = chunk["id"]
        
        return suggestions

    def analyze_from_history(self, document_id: str) -> dict[str, Any]:
        """
        Analyze change history to find patterns in user edits.
        """
        try:
            change_log = get_change_log()
            changes = change_log.get_changes(document_id=document_id, limit=100)
            
            if not changes:
                return {"total_changes": 0, "patterns": []}
            
            # Count actions
            action_counts = {}
            for change in changes:
                action = change.get("action", "unknown")
                action_counts[action] = action_counts.get(action, 0) + 1
            
            return {
                "total_changes": len(changes),
                "action_counts": action_counts,
                "patterns_found": [], # Future: Use LLM to generalize specific edits into regex patterns
                "last_change": changes[0]["created_at"] if changes else None,
            }
        except Exception as e:
            logger.error("Failed to analyze history: %s", e)
            return {"error": str(e)}


# Global instance
_pattern_learner: Optional[ChunkPatternLearner] = None


def get_pattern_learner() -> ChunkPatternLearner:
    """Get or create global ChunkPatternLearner instance."""
    global _pattern_learner
    if _pattern_learner is None:
        _pattern_learner = ChunkPatternLearner()
    return _pattern_learner
