"""
Tuple-delimited entity extraction parser.

This module parses Microsoft GraphRAG-style tuple-delimited output
from LLMs into Entity and Relationship objects.

Format:
    ("entity"<|>NAME<|>TYPE<|>DESCRIPTION<|>IMPORTANCE)
    ("relationship"<|>SOURCE<|>TARGET<|>TYPE<|>DESCRIPTION<|>STRENGTH)

Delimiter: <|> (chosen for low collision with natural text)
"""

import logging
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from core.entity_models import Entity, Relationship

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of tuple parsing."""
    entities: List[Entity]
    relationships: List[Relationship]
    parse_errors: List[str]
    valid_count: int
    invalid_count: int


class TupleParser:
    """
    Parser for tuple-delimited entity extraction output.
    
    This parser is designed to be robust:
    - Handles malformed tuples gracefully (logs error, continues)
    - Recovers partial results from mixed valid/invalid output
    - Provides detailed error reporting for debugging
    - Normalizes entity names/types using same logic as current parser
    
    Usage:
        parser = TupleParser(chunk_id="chunk_123")
        result = parser.parse(llm_output_text)
        
        for entity in result.entities:
            # Process entities
            
        if result.parse_errors:
            logger.warning(f"Had {len(result.parse_errors)} parse errors")
    """
    
    def __init__(self, chunk_id: Optional[str] = None):
        """
        Initialize parser.
        
        Args:
            chunk_id: Optional chunk ID for provenance tracking
        """
        self.chunk_id = chunk_id
        self._stats = {
            "entities_parsed": 0,
            "relationships_parsed": 0,
            "parse_errors": 0,
        }
    
    def parse(self, text: str) -> ParseResult:
        """
        Parse tuple-delimited text into entities and relationships.
        
        Algorithm:
        1. Split text into lines
        2. For each line:
           a. Check if line looks like a tuple
           b. Extract type field ("entity" or "relationship")
           c. Split remaining content by <|> delimiter
           d. Validate field count
           e. Create Entity or Relationship object
           f. Handle errors gracefully (log and continue)
        3. Return ParseResult with all parsed objects + errors
        
        Args:
            text: LLM output text with tuple-delimited extractions
        
        Returns:
            ParseResult with entities, relationships, and parse errors
        """
        entities = []
        relationships = []
        parse_errors = []
        
        if not text or not text.strip():
            logger.warning("Empty input text provided to tuple parser")
            return ParseResult(
                entities=[],
                relationships=[],
                parse_errors=["Empty input text"],
                valid_count=0,
                invalid_count=1
            )
        
        # Find all tuples in text
        lines = text.strip().split('\n')
        
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Try to parse as tuple
            try:
                result = self._parse_tuple_line(line)
                
                if result is None:
                    # Not a valid tuple, skip silently (might be explanation text)
                    continue
                
                tuple_type, fields = result
                
                if tuple_type == "entity":
                    entity = self._parse_entity_tuple(fields, line_num)
                    if entity:
                        entities.append(entity)
                        self._stats["entities_parsed"] += 1
                    else:
                        parse_errors.append(f"Line {line_num}: Failed to parse entity tuple")
                        self._stats["parse_errors"] += 1
                
                elif tuple_type == "relationship":
                    relationship = self._parse_relationship_tuple(fields, line_num)
                    if relationship:
                        relationships.append(relationship)
                        self._stats["relationships_parsed"] += 1
                    else:
                        parse_errors.append(f"Line {line_num}: Failed to parse relationship tuple")
                        self._stats["parse_errors"] += 1
                
                else:
                    logger.debug(f"Line {line_num}: Unknown tuple type '{tuple_type}'")
                    parse_errors.append(f"Line {line_num}: Unknown tuple type '{tuple_type}'")
                    self._stats["parse_errors"] += 1
            
            except Exception as e:
                logger.warning(f"Line {line_num}: Parse error - {str(e)}")
                parse_errors.append(f"Line {line_num}: {str(e)}")
                self._stats["parse_errors"] += 1
                continue
        
        valid_count = len(entities) + len(relationships)
        invalid_count = len(parse_errors)
        
        logger.info(
            f"Tuple parsing complete: {len(entities)} entities, "
            f"{len(relationships)} relationships, {invalid_count} errors"
        )
        
        return ParseResult(
            entities=entities,
            relationships=relationships,
            parse_errors=parse_errors,
            valid_count=valid_count,
            invalid_count=invalid_count
        )
    
    def _parse_tuple_line(self, line: str) -> Optional[Tuple[str, List[str]]]:
        """
        Parse a single tuple line.
        
        Returns:
            (tuple_type, fields) or None if not a valid tuple
        
        Example:
            Input: '("entity"<|>ADMIN PANEL<|>COMPONENT<|>Description)'
            Output: ("entity", ["ADMIN PANEL", "COMPONENT", "Description"])
        """
        # Check if line looks like a tuple
        if not (line.startswith('("') and line.endswith(')')):
            return None
        
        # Remove outer parentheses
        inner = line[1:-1]  # Remove leading ( and trailing )
        
        # Check for opening quote
        if not inner.startswith('"'):
            return None
        
        # Find closing quote for type field
        type_end = inner.find('"', 1)
        if type_end == -1:
            return None
        
        # Extract type
        tuple_type = inner[1:type_end].strip().lower()
        
        # Get remaining content after type
        remaining = inner[type_end + 1:]
        
        # Check for delimiter after type
        if not remaining.startswith('<|>'):
            return None
        
        # Remove leading delimiter
        remaining = remaining[3:]  # Remove <|>
        
        # Split by delimiter to get fields
        fields = remaining.split('<|>')
        
        # Trim whitespace from all fields
        fields = [f.strip() for f in fields]
        
        return (tuple_type, fields)
    
    def _parse_entity_tuple(self, fields: List[str], line_num: int) -> Optional[Entity]:
        """
        Parse entity tuple fields into Entity object.
        
        Expected fields: [name, type, description] or [name, type, description, importance]
        Minimum fields: 2 (name, type; description optional)
        
        Args:
            fields: List of field values from tuple
            line_num: Line number for error reporting
        
        Returns:
            Entity object or None if parsing fails
        """
        if len(fields) < 2:
            logger.warning(
                f"Line {line_num}: Entity tuple has insufficient fields "
                f"(expected 3-4, got {len(fields)})"
            )
            return None
        
        # Extract fields
        name = fields[0].strip()
        entity_type = fields[1].strip() if len(fields) > 1 else ""
        description = fields[2].strip() if len(fields) > 2 else ""
        importance = float(fields[3]) if len(fields) > 3 and fields[3].strip() else 0.5
        
        # Validate name (required)
        if not name:
            logger.warning(f"Line {line_num}: Entity tuple has empty name")
            return None
        
        # Normalize entity name (same as current parser)
        name = self._normalize_entity_name(name)
        
        # Normalize type (uppercase)
        entity_type = entity_type.upper()
        
        # Validate importance range
        if importance < 0.0 or importance > 1.0:
            logger.warning(
                f"Line {line_num}: Invalid importance {importance}, using 0.5"
            )
            importance = 0.5
        
        # Create Entity object
        entity = Entity(
            name=name,
            type=entity_type,
            description=description,
            importance_score=importance,
            source_text_units=[self.chunk_id] if self.chunk_id else [],
            source_chunks=[self.chunk_id] if self.chunk_id else [],
        )
        
        logger.debug(f"Parsed entity: {name} ({entity_type})")
        
        return entity
    
    def _parse_relationship_tuple(
        self,
        fields: List[str],
        line_num: int
    ) -> Optional[Relationship]:
        """
        Parse relationship tuple fields into Relationship object.
        
        Expected fields: [source, target, type, description] or [source, target, type, description, strength]
        Minimum fields: 3 (source, target, type; description optional)
        
        Args:
            fields: List of field values from tuple
            line_num: Line number for error reporting
        
        Returns:
            Relationship object or None if parsing fails
        """
        if len(fields) < 3:
            logger.warning(
                f"Line {line_num}: Relationship tuple has insufficient fields "
                f"(expected 4-5, got {len(fields)})"
            )
            return None
        
        # Extract fields
        source = fields[0].strip()
        target = fields[1].strip()
        rel_type = fields[2].strip()
        description = fields[3].strip() if len(fields) > 3 else ""
        strength = float(fields[4]) if len(fields) > 4 and fields[4].strip() else 0.5
        
        # Validate required fields
        if not source or not target:
            logger.warning(
                f"Line {line_num}: Relationship tuple has empty source or target"
            )
            return None
        
        # Normalize entity names (same as current parser)
        source = self._normalize_entity_name(source)
        target = self._normalize_entity_name(target)
        
        # Normalize relationship type (uppercase, underscores)
        rel_type = rel_type.upper().replace(' ', '_')
        
        # Validate strength range
        if strength < 0.0 or strength > 1.0:
            logger.warning(
                f"Line {line_num}: Invalid strength {strength}, using 0.5"
            )
            strength = 0.5
        
        # Create Relationship object
        relationship = Relationship(
            source_entity=source,
            target_entity=target,
            relationship_type=rel_type,
            description=description,
            strength=strength,
            source_text_units=[self.chunk_id] if self.chunk_id else [],
            source_chunks=[self.chunk_id] if self.chunk_id else [],
        )
        
        logger.debug(
            f"Parsed relationship: {source} -> {target} ({rel_type})"
        )
        
        return relationship
    
    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name using same logic as current parser.
        
        Rules:
        - Convert to uppercase
        - Trim leading/trailing whitespace
        - Collapse internal whitespace to single spaces
        
        Args:
            name: Raw entity name
        
        Returns:
            Normalized entity name
        """
        # Trim whitespace
        name = name.strip()
        
        # Collapse internal whitespace
        name = re.sub(r'\s+', ' ', name)
        
        # Convert to uppercase
        name = name.upper()
        
        return name
    
    def get_stats(self) -> dict:
        """
        Get parsing statistics.
        
        Returns:
            Dictionary with parsing stats
        """
        return self._stats.copy()
