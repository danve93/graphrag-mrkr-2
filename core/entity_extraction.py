"""
Entity and relationship extraction using LLM for GraphRAG pipeline.
"""

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from core.llm import llm_manager

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0):
    """
    Decorator for retrying LLM calls with exponential backoff on rate limiting errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (increased from 2.0 to 3.0)
        max_delay: Maximum delay in seconds (increased from 120.0 to 180.0)
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check for rate limiting error (429) or connection errors
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    # Check if this is a retryable error
                    is_retryable = False
                    if (
                        hasattr(e, "status_code")
                        and getattr(e, "status_code", None) == 429
                    ):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit hit in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Too Many Requests" in str(e) or "429" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit detected in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Connection" in str(e) or "Timeout" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Connection error in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )

                    if not is_retryable:
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0.2, 0.5) * delay  # Add 20-50% jitter (increased)
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)

            return None  # Should never reach here

        return wrapper

    return decorator


@dataclass
class Entity:
    """Represents an extracted entity."""

    name: str
    type: str
    description: str
    importance_score: float = 0.5
    source_chunks: Optional[List[str]] = None

    def __post_init__(self):
        if self.source_chunks is None:
            self.source_chunks = []


@dataclass
class Relationship:
    """Represents a relationship between two entities."""

    source_entity: str
    target_entity: str
    description: str
    strength: float = 0.5
    source_chunks: Optional[List[str]] = None

    def __post_init__(self):
        if self.source_chunks is None:
            self.source_chunks = []


class EntityExtractor:
    """Extracts entities and relationships from text using LLM."""

    # Default entity types tailored for Carbonio use cases
    DEFAULT_ENTITY_TYPES = [
        # Infrastruttura e architettura
        "COMPONENT",
        "SERVICE",
        "NODE",

        # Provisioning e oggetti utente
        "DOMAIN",
        "CLASS_OF_SERVICE",
        "ACCOUNT",
        "ACCOUNT_TYPE",
        "ROLE",

        "RESOURCE",

        # Quota e gestione utilizzo
        "QUOTA_OBJECT",

        # Backup, storage, migrazione
        "BACKUP_OBJECT",
        "ITEM",
        "STORAGE_OBJECT",
        "MIGRATION_PROCEDURE",

        # Configurazione, sicurezza, certificati
        "CERTIFICATE",
        "CONFIG_OPTION",
        "SECURITY_FEATURE",

        # CLI, API, procedure operative
        "CLI_COMMAND",
        "API_OBJECT",
        "TASK",
        "PROCEDURE",

        # Articoli Zendesk
        "ARTICLE",
    ]

    CARBONIO_ENTITY_TYPE_OVERRIDES = {
        # Provisioning e ruoli
        "CLASS OF SERVICE": "CLASS_OF_SERVICE",
        "CLASS OF SERVICES": "CLASS_OF_SERVICE",
        "CLASS OF SERVICES (COS)": "CLASS_OF_SERVICE",
        "COS": "CLASS_OF_SERVICE",

        "DOMAIN": "DOMAIN",
        "E-MAIL DOMAIN": "DOMAIN",
        "MAIL DOMAIN": "DOMAIN",

        "REGULAR USER": "ACCOUNT_TYPE",
        "USER ACCOUNT": "ACCOUNT_TYPE",
        "END USER": "ACCOUNT_TYPE",

        "FUNCTIONAL ACCOUNT": "ACCOUNT_TYPE",
        "SHARED ACCOUNT": "ACCOUNT_TYPE",
        "RESOURCE ACCOUNT": "RESOURCE",
        "SYSTEM ACCOUNT": "ACCOUNT_TYPE",
        "EXTERNAL ACCOUNT": "ACCOUNT_TYPE",

        "RESOURCE": "RESOURCE",

        "GLOBAL ADMIN": "ROLE",
        "GLOBAL ADMINISTRATOR": "ROLE",
        "DELEGATED ADMIN": "ROLE",
        "DELEGATED ADMINISTRATOR": "ROLE",
        "DOMAIN ADMIN": "ROLE",

        # Componenti e servizi
        "MTA": "COMPONENT",
        "MTA AV/AS": "COMPONENT",
        "MAILSTORE": "COMPONENT",
        "MAILSTORE & PROVISIONING": "COMPONENT",
        "PROXY": "COMPONENT",
        "FILES": "COMPONENT",
        "CHATS": "COMPONENT",
        "DOCS": "COMPONENT",
        "DOCS & EDITOR": "COMPONENT",
        "TASKS": "COMPONENT",
        "VIDEO SERVER": "COMPONENT",
        "MONITORING": "COMPONENT",
        "BACKUP": "COMPONENT",

        "MESH & DIRECTORY": "SERVICE",
        "DIRECTORY": "SERVICE",
        "DIRECTORY REPLICA": "SERVICE",
        "EVENT STREAMING": "SERVICE",

        "NODE": "NODE",
        "SERVER NODE": "NODE",
        "CARBONIO NODE": "NODE",

        # Backup e storage
        "ITEM": "ITEM",
        "BACKUP ITEM": "ITEM",

        "SMARTSCAN": "BACKUP_OBJECT",
        "SMART SCAN": "BACKUP_OBJECT",
        "REALTIME SCANNER": "BACKUP_OBJECT",
        "REAL TIME SCANNER": "BACKUP_OBJECT",
        "BACKUP PATH": "BACKUP_OBJECT",
        "RETENTION POLICY": "BACKUP_OBJECT",
        "LEGAL HOLD": "BACKUP_OBJECT",

        "VOLUME": "STORAGE_OBJECT",
        "PRIMARY VOLUME": "STORAGE_OBJECT",
        "SECONDARY VOLUME": "STORAGE_OBJECT",
        "HSM VOLUME": "STORAGE_OBJECT",
        "OBJECT STORAGE": "STORAGE_OBJECT",
        "STORAGE TIER": "STORAGE_OBJECT",

        # Certificati e configurazione
        "DOMAIN CERTIFICATE": "CERTIFICATE",
        "WILDCARD CERTIFICATE": "CERTIFICATE",
        "INFRASTRUCTURE CERTIFICATE": "CERTIFICATE",
        "TLS CERTIFICATE": "CERTIFICATE",
        "CERTIFICATE": "CERTIFICATE",

        "PUBLIC SERVICE HOSTNAME": "CONFIG_OPTION",
        "VIRTUAL HOST NAME": "CONFIG_OPTION",
        "PUBLIC HOSTNAME": "CONFIG_OPTION",

        "HSM POLICY": "CONFIG_OPTION",
        "HSM SETTINGS": "CONFIG_OPTION",

        # Sicurezza
        "DOS FILTER": "SECURITY_FEATURE",
        "DENIAL OF SERVICE FILTER": "SECURITY_FEATURE",
        "OTP": "SECURITY_FEATURE",
        "ONE-TIME PASSWORD": "SECURITY_FEATURE",
        "S/MIME": "SECURITY_FEATURE",
        "SMIME": "SECURITY_FEATURE",
        "AUTHENTICATION METHOD": "SECURITY_FEATURE",

        # Migrazione, CLI, API, procedure
        "MIGRATION PROCEDURE": "MIGRATION_PROCEDURE",
        "MIGRATION FLOW": "MIGRATION_PROCEDURE",
        "MIGRATION PATH": "MIGRATION_PROCEDURE",

        "CLI COMMAND": "CLI_COMMAND",
        "CARBONIO CLI COMMAND": "CLI_COMMAND",

        "API OBJECT": "API_OBJECT",
        "FILES API OBJECT": "API_OBJECT",

        "TASK": "TASK",
        "ADMIN TASK": "TASK",
        "MAINTENANCE TASK": "TASK",

        "PROCEDURE": "PROCEDURE",
        "ADMIN PROCEDURE": "PROCEDURE",
        "MAINTENANCE PROCEDURE": "PROCEDURE",

        # Concetti chiave
        "RPO": "CONCEPT",
        "RTO": "CONCEPT",
        "BACKUP STRATEGY": "CONCEPT",
        "USER MANAGEMENT": "CONCEPT",
    }

    # Patterns for low-value entities that should be filtered out
    LOW_VALUE_PATTERNS = [
        # Common articles, prepositions, conjunctions
        r"^(?:the|and|or|but|with|from|for|at|by|on|in|to|of|a|an)$",
        # Common pronouns and demonstratives
        r"^(?:this|that|these|those|here|there|where|when|what|who|how|why)$",
        # Generic business terms
        r"^(?:company|organization|group|team|department|division|system|process|method|approach|way|means)$",
        # Generic data terms
        r"^(?:data|information|content|text|document|report|file|item|thing|stuff)$",
        # Common adjectives
        r"^(?:new|old|first|last|next|previous|current|recent|good|bad|big|small|high|low|major|minor)$",
        # Just numbers or single characters
        r"^\d{1,3}$",
        r"^[a-zA-Z]$",
        # Very short meaningless strings
        r"^.{1,2}$",
        # Common file extensions or codes
        r"^\.[a-z]{2,4}$",
        r"^[A-Z]{1,3}\d*$",
    ]

    # Entity type normalization mapping
    ENTITY_TYPE_MAPPING = {
        **{
            # Preserve a few generic cleanups to keep compatibility with legacy data
            "SECTION": "CONCEPT",
            "SERVICE": "PRODUCT",
            "CONTACT": "TECHNOLOGY",
        },
        **CARBONIO_ENTITY_TYPE_OVERRIDES,
    }

    RELATION_TYPE_SUGGESTIONS = [
        # Architettura e componenti
        "COMPONENT_RUNS_ON_NODE",
        "COMPONENT_DEPENDS_ON_COMPONENT",
        "SERVICE_DEPENDS_ON_COMPONENT",
        "COMPONENT_PROVIDES_FEATURE",

        # Provisioning e account
        "DOMAIN_HAS_COS",
        "COS_APPLIES_TO_ACCOUNT_TYPE",
        "ACCOUNT_BELONGS_TO_DOMAIN",
        "ACCOUNT_HAS_ROLE",
        "ACCOUNT_HAS_QUOTA",

        # Backup e storage
        "BACKUP_COVERS_ITEM",
        "ITEM_STORED_ON_STORAGE_OBJECT",
        "HSM_POLICY_APPLIES_TO_STORAGE_OBJECT",

        # Configurazione, certificati, sicurezza
        "CERTIFICATE_APPLIES_TO_DOMAIN",
        "CONFIG_OPTION_AFFECTS_COMPONENT",
        "SECURITY_FEATURE_PROTECTS_COMPONENT",

        # Migrazione e operazioni
        "MIGRATION_PROCEDURE_TARGETS_COMPONENT",
        "MIGRATION_PROCEDURE_TARGETS_DOMAIN",
        "CLI_COMMAND_CONFIGURES_OBJECT",
        "TASK_OPERATES_ON_OBJECT",
        "PROCEDURE_INCLUDES_TASK",

        # Fallback generico
        "RELATED_TO",
    ]

    def __init__(self, entity_types: Optional[List[str]] = None):
        """Initialize entity extractor."""
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name to reduce duplicates."""
        # Remove extra whitespace and normalize case
        normalized = re.sub(r"\s+", " ", name.strip())

        # Remove unnecessary punctuation but keep meaningful ones
        normalized = re.sub(r"[^\w\s\-\.\(\)\/]", "", normalized)

        # Normalize common variations
        normalized = re.sub(
            r"\b(?:sub[\-\s]?floor)\b", "subfloor", normalized, flags=re.IGNORECASE
        )
        normalized = re.sub(
            r"\b(?:sub[\-\s]?structure)\b",
            "substructure",
            normalized,
            flags=re.IGNORECASE,
        )

        # Remove redundant parenthetical content
        normalized = re.sub(r"\s*\([^)]*\)\s*", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type using mapping."""
        # Clean up the type string
        cleaned_type = entity_type.strip().upper()

        # Apply mapping if available
        if cleaned_type in self.ENTITY_TYPE_MAPPING:
            return self.ENTITY_TYPE_MAPPING[cleaned_type]

        # Default normalization for unmapped types
        # Remove excessive descriptors
        if "(" in cleaned_type and ")" in cleaned_type:
            base_type = cleaned_type.split("(")[0].strip()
            if base_type in self.DEFAULT_ENTITY_TYPES:
                return base_type

        # Remove prefixed modifiers like "**TYPE**"
        cleaned_type = re.sub(r"^\*+([A-Z]+)\*+$", r"\1", cleaned_type)

        # Default to CONCEPT for unclear types
        if cleaned_type not in self.DEFAULT_ENTITY_TYPES:
            return "CONCEPT"

        return cleaned_type

    def _is_low_value_entity(
        self, name: str, entity_type: str, importance: float
    ) -> bool:
        """Check if entity is low value and should be filtered out."""
        # Filter by importance threshold
        if importance < 0.3:
            return True

        # Filter by name patterns
        name_lower = name.lower().strip()
        for pattern in self.LOW_VALUE_PATTERNS:
            if re.match(pattern, name_lower, re.IGNORECASE):
                return True

        # Filter very generic concepts
        if entity_type == "CONCEPT" and importance < 0.6:
            generic_patterns = [
                r"^(?:management|system|program|process|method|approach|solution)$",
                r"^(?:inspection|treatment|damage|condition|presence|lack)$",
                r"^(?:area|areas|location|locations|structure|structures)$",
            ]
            for pattern in generic_patterns:
                if re.match(pattern, name_lower, re.IGNORECASE):
                    return True

        return False

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on normalized names."""
        seen_entities = {}
        deduplicated = []

        for entity in entities:
            normalized_name = self._normalize_entity_name(entity.name)
            normalized_type = self._normalize_entity_type(entity.type)

            # Create a key for deduplication
            key = (normalized_name.lower(), normalized_type)

            if key not in seen_entities:
                # Create new entity with normalized values
                normalized_entity = Entity(
                    name=normalized_name,
                    type=normalized_type,
                    description=entity.description,
                    importance_score=entity.importance_score,
                    source_chunks=(
                        entity.source_chunks.copy() if entity.source_chunks else []
                    ),
                )
                seen_entities[key] = normalized_entity
                deduplicated.append(normalized_entity)
            else:
                # Merge with existing entity
                existing = seen_entities[key]
                if entity.source_chunks:
                    existing.source_chunks.extend(entity.source_chunks)
                # Use better description if available
                if len(entity.description) > len(existing.description):
                    existing.description = entity.description
                # Average importance scores
                existing.importance_score = (
                    existing.importance_score + entity.importance_score
                ) / 2

        return deduplicated

    def _get_extraction_prompt(self, text: str) -> str:
        """Generate prompt for entity and relationship extraction."""
        entity_types_str = ", ".join(self.entity_types)
        relation_types_str = ", ".join(self.RELATION_TYPE_SUGGESTIONS)

        return f"""You are an expert at extracting entities and relationships from text.

**Task**: Extract all relevant entities and relationships from the given text.

**Entity Types**: Focus on these types: {entity_types_str}
**Relationship Types**: Prefer these relationship patterns when applicable: {relation_types_str}

**Instructions**:
1. Extract entities with: name, type, description, importance (0.0-1.0)
2. Extract relationships with: source entity, target entity, description, strength (0.0-1.0)
3. Use exact entity names from the text
4. Provide detailed descriptions
5. Rate importance/strength based on context significance

**Output Format**:
ENTITIES:
- Name: [entity_name] | Type: [entity_type] | Description: [description] | Importance: [0.0-1.0]

RELATIONSHIPS:
- Source: [source_entity] | Target: [target_entity] | Description: [description] | Strength: [0.0-1.0]

**Text to analyze**:
{text}

**Output**:"""

    def _parse_extraction_response(
        self, response: str, chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM response to extract entities and relationships."""
        entities = []
        relationships = []

        try:
            # Split response into entities and relationships sections
            # Handle different formats: "RELATIONSHIPS:" or "**RELATIONSHIPS**"
            if "**RELATIONSHIPS**" in response:
                sections = response.split("**RELATIONSHIPS**")
                entities_section = sections[0].replace("**ENTITIES**", "").strip()
                relationships_section = sections[1].strip() if len(sections) > 1 else ""
            else:
                sections = response.split("RELATIONSHIPS:")
                entities_section = sections[0].replace("ENTITIES:", "").strip()
                relationships_section = sections[1].strip() if len(sections) > 1 else ""

            # Parse entities
            entity_pattern = r"- Name: ([^|]+) \| Type: ([^|]+) \| Description: ([^|]+) \| Importance: ([\d.]+)"
            for match in re.finditer(entity_pattern, entities_section):
                name = match.group(1).strip()
                entity_type = match.group(2).strip().upper()
                description = match.group(3).strip()
                importance = float(match.group(4))

                # Apply normalization
                normalized_name = self._normalize_entity_name(name)
                normalized_type = self._normalize_entity_type(entity_type)

                # Filter low-value entities
                if self._is_low_value_entity(
                    normalized_name, normalized_type, importance
                ):
                    continue

                entity = Entity(
                    name=normalized_name,
                    type=normalized_type,
                    description=description,
                    importance_score=min(max(importance, 0.0), 1.0),
                    source_chunks=[chunk_id],
                )
                entities.append(entity)

            # Parse relationships
            relationship_pattern = r"- Source: ([^|]+) \| Target: ([^|]+) \| Description: ([^|]+) \| Strength: ([\d.]+)"
            for match in re.finditer(relationship_pattern, relationships_section):
                source = match.group(1).strip()
                target = match.group(2).strip()
                description = match.group(3).strip()
                strength = float(match.group(4))

                # Normalize entity names in relationships
                normalized_source = self._normalize_entity_name(source)
                normalized_target = self._normalize_entity_name(target)

                relationship = Relationship(
                    source_entity=normalized_source,
                    target_entity=normalized_target,
                    description=description,
                    strength=min(max(strength, 0.0), 1.0),
                    source_chunks=[chunk_id],
                )
                relationships.append(relationship)

        except Exception as e:
            logger.error(f"Error parsing extraction response for chunk {chunk_id}: {e}")
            logger.debug(f"Response was: {response}")

        # Apply deduplication
        entities = self._deduplicate_entities(entities)

        logger.info(
            f"Extracted {len(entities)} entities and {len(relationships)} relationships from chunk {chunk_id}"
        )
        return entities, relationships

    async def extract_from_chunk(
        self, text: str, chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a single text chunk with retry logic."""

        @retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0)
        def _generate_response_with_retry(prompt):
            return llm_manager.generate_response(
                prompt=prompt, max_tokens=4000, temperature=0.1
            )

        try:
            prompt = self._get_extraction_prompt(text)

            # Offload synchronous/blocking LLM call to a thread executor with retry
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, lambda: _generate_response_with_retry(prompt)
            )

            return self._parse_extraction_response(response, chunk_id)

        except Exception as e:
            logger.error(f"Entity extraction failed for chunk {chunk_id}: {e}")
            return [], []

    async def extract_from_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Entity], Dict[str, List[Relationship]]]:
        """
        Extract entities and relationships from multiple chunks.

        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'content' keys

        Returns:
            Tuple of (consolidated_entities, relationships_by_entity_pair)
        """
        logger.info(f"Starting entity extraction from {len(chunks)} chunks")

        # Concurrency limit from settings (use embedding_concurrency to match pattern)
        concurrency = getattr(settings, "llm_concurrency")
        sem = asyncio.Semaphore(concurrency)
        
        # Track last request time for rate limiting (use dict to make it mutable)
        request_tracker = {"last_time": 0.0}

        async def _sem_extract(chunk):
            async with sem:
                try:
                    # Enforce minimum delay between LLM requests
                    current_time = time.time()
                    time_since_last = current_time - request_tracker["last_time"]
                    min_delay = random.uniform(settings.llm_delay_min, settings.llm_delay_max)
                    
                    if time_since_last < min_delay:
                        sleep_time = min_delay - time_since_last
                        logger.debug(f"Rate limiting LLM: sleeping {sleep_time:.2f}s")
                        await asyncio.sleep(sleep_time)
                    
                    request_tracker["last_time"] = time.time()
                    
                    return await self.extract_from_chunk(
                        chunk["content"], chunk["chunk_id"]
                    )
                except Exception as e:
                    logger.error(
                        f"Extraction failed for chunk {chunk.get('chunk_id')}: {e}"
                    )
                    return [], []

        # Schedule tasks with semaphore control
        extraction_tasks = [
            asyncio.create_task(_sem_extract(chunk)) for chunk in chunks
        ]

        results = []
        for coro in asyncio.as_completed(extraction_tasks):
            try:
                res = await coro
                results.append(res)
            except Exception as e:
                logger.error(f"Error in extraction task: {e}")

        # Consolidate entities and relationships
        all_entities_list = []
        all_relationships = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Extraction failed for chunk {i}: {result}")
                continue

            if isinstance(result, tuple) and len(result) == 2:
                entities, relationships = result
            else:
                logger.error(f"Unexpected result format for chunk {i}: {result}")
                continue

            # Collect all entities for global deduplication
            all_entities_list.extend(entities)
            all_relationships.extend(relationships)

        # Apply global deduplication across all chunks
        deduplicated_entities = self._deduplicate_entities(all_entities_list)

        # Convert to dictionary for compatibility
        all_entities = {
            entity.name.upper().strip(): entity for entity in deduplicated_entities
        }

        # Group relationships by entity pair
        relationships_by_pair = {}
        for rel in all_relationships:
            source_key = rel.source_entity.upper().strip()
            target_key = rel.target_entity.upper().strip()

            # Only keep relationships where both entities exist
            if source_key in all_entities and target_key in all_entities:
                # Create consistent key regardless of direction
                pair_key = tuple(sorted([source_key, target_key]))
                if pair_key not in relationships_by_pair:
                    relationships_by_pair[pair_key] = []
                relationships_by_pair[pair_key].append(rel)

        logger.info(
            f"Consolidated to {len(all_entities)} entities and {len(relationships_by_pair)} relationship pairs"
        )
        return all_entities, relationships_by_pair


# Global instance
entity_extractor = EntityExtractor()
