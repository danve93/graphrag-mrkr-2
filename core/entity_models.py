"""
Entity and Relationship data models for entity extraction.

This module defines the core data structures used throughout the entity extraction
and graph RAG pipeline. Extracted separately to avoid circular imports.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Entity:
    """Represents an extracted entity."""

    name: str
    type: str
    description: str
    importance_score: float = 0.5
    source_text_units: Optional[List[str]] = None
    source_chunks: Optional[List[str]] = None

    def __post_init__(self):
        resolved_text_units = self.source_text_units or self.source_chunks or []
        self.source_text_units = list(resolved_text_units)
        self.source_chunks = list(self.source_chunks or self.source_text_units or [])


@dataclass
class Relationship:
    """Represents a relationship between two entities."""

    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    strength: float = 0.5
    source_text_units: Optional[List[str]] = None
    source_chunks: Optional[List[str]] = None

    def __post_init__(self):
        resolved_text_units = self.source_text_units or self.source_chunks or []
        self.source_text_units = list(resolved_text_units)
        self.source_chunks = list(self.source_chunks or self.source_text_units or [])
