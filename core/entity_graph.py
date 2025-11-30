"""
NetworkX-based entity graph for in-memory deduplication and batch persistence.

This module implements Phase 2 of the Microsoft GraphRAG approach:
- Build intermediate graph in memory during extraction
- Deduplicate entities by canonical (name, type) key
- Accumulate descriptions (newline-join) instead of overwriting
- Sum relationship strengths across mentions
- Create orphan entities for missing relationship targets
- Convert to batch UNWIND queries for Neo4j

Key benefits:
- 5-10x faster ingestion (batch vs individual transactions)
- Better deduplication (pre-persistence merging)
- Provenance tracking (mention counts, source chunks)
- Memory efficient (~9MB per 1000-chunk document)
"""

import logging
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntityGraphStats:
    """Statistics about the entity graph."""
    node_count: int
    edge_count: int
    orphan_count: int
    total_mentions: int
    avg_mentions_per_entity: float
    max_strength: float
    description_lengths: Dict[str, int]


class EntityGraph:
    """
    NetworkX-based intermediate graph for entity deduplication.
    
    This class wraps NetworkX MultiDiGraph to provide:
    - Entity deduplication by canonical (name, type) key
    - Description accumulation (newline-separated, deduplicated)
    - Relationship strength summation
    - Orphan entity creation
    - Batch UNWIND query generation
    
    Usage:
        graph = EntityGraph()
        
        # Add entities
        graph.add_entity("Admin Panel", "COMPONENT", "Web interface", 0.8, ["chunk1"])
        graph.add_entity("ADMIN PANEL", "Component", "User management", 0.9, ["chunk2"])
        # ^ These merge because canonical keys match
        
        # Add relationships
        graph.add_relationship("Admin Panel", "Database", "DEPENDS_ON", "Queries DB", 0.7, ["chunk1"])
        
        # Convert to Neo4j batch queries
        entity_query, entity_params, rel_query, rel_params = graph.to_neo4j_batch_queries("doc123")
        
        # Execute with Neo4j driver
        with driver.session() as session:
            session.run(entity_query, entity_params)
            session.run(rel_query, rel_params)
    """
    
    def __init__(self):
        """Initialize empty entity graph."""
        self.graph = nx.MultiDiGraph()
        self._canonical_map: Dict[Tuple[str, str], str] = {}
        self._orphan_entities: Set[str] = set()
    
    def _canonical_key(self, name: str, type: str) -> Tuple[str, str]:
        """
        Generate canonical key for entity deduplication.
        
        Canonical key is case-insensitive and whitespace-normalized:
        - "Admin Panel" + "COMPONENT" → ("ADMIN PANEL", "COMPONENT")
        - "admin panel" + "component" → ("ADMIN PANEL", "COMPONENT")
        - "  Admin  Panel  " → ("ADMIN PANEL", "COMPONENT")
        
        Args:
            name: Entity name
            type: Entity type
            
        Returns:
            Tuple of (normalized_name, normalized_type)
        """
        import re
        # Normalize: strip, collapse multiple whitespace, uppercase
        normalized_name = re.sub(r'\s+', ' ', (name or "").strip()).upper()
        normalized_type = re.sub(r'\s+', ' ', (type or "").strip()).upper()
        return (normalized_name, normalized_type)
    
    def _get_node_id(self, name: str, type: str) -> str:
        """
        Get or create node ID for canonical key.
        
        Returns the node ID (original casing) for the canonical key.
        If entity doesn't exist, returns a unique node ID combining name and type.
        
        Args:
            name: Entity name
            type: Entity type
            
        Returns:
            Node ID (str)
        """
        canonical = self._canonical_key(name, type)
        if canonical in self._canonical_map:
            return self._canonical_map[canonical]
        # Use name + type as unique node ID (preserving original casing)
        return f"{name}_{type}" if type else name
    
    def has_entity(self, name: str, type: str) -> bool:
        """
        Check if entity exists in graph.
        
        Args:
            name: Entity name
            type: Entity type
            
        Returns:
            True if entity exists
        """
        canonical = self._canonical_key(name, type)
        return canonical in self._canonical_map
    
    def add_entity(
        self,
        name: str,
        type: str,
        description: str,
        importance_score: float,
        source_chunks: List[str]
    ) -> None:
        """
        Add entity to graph or merge with existing entity.
        
        If entity with same canonical key exists:
        - Accumulate descriptions (newline-join, deduplicated)
        - Average importance scores
        - Merge source chunks (deduplicated)
        - Increment mention count
        
        Args:
            name: Entity name
            type: Entity type
            description: Entity description
            importance_score: Importance score (0-1)
            source_chunks: List of source chunk IDs
        """
        canonical = self._canonical_key(name, type)
        
        if canonical in self._canonical_map:
            # Entity exists - merge
            node_id = self._canonical_map[canonical]
            node_data = self.graph.nodes[node_id]
            
            # Accumulate descriptions (deduplicated)
            existing_descriptions = set(node_data["description"].split("\n")) if node_data["description"] else set()
            if description and description.strip():
                existing_descriptions.add(description.strip())
            node_data["description"] = "\n".join(sorted(filter(None, existing_descriptions)))
            
            # Average importance scores
            current_count = node_data["mention_count"]
            current_score = node_data["importance_score"]
            node_data["importance_score"] = (current_score * current_count + importance_score) / (current_count + 1)
            
            # Merge source chunks (deduplicated)
            existing_sources = set(node_data["source_chunks"])
            existing_sources.update(source_chunks or [])
            node_data["source_chunks"] = sorted(existing_sources)
            
            # Increment mention count
            node_data["mention_count"] += 1
            
            logger.debug(f"Merged entity: {name} (canonical: {canonical[0]}, mentions: {node_data['mention_count']})")
        else:
            # New entity - add to graph
            # Use unique node ID combining name and type
            node_id = f"{name.strip()}_{type.strip()}" if (type and type.strip()) else name.strip()
            self.graph.add_node(
                node_id,
                name=name.strip(),
                type=type.strip() if type else "",
                description=description or "",
                importance_score=importance_score,
                source_chunks=list(source_chunks or []),
                mention_count=1,
                is_orphan=False
            )
            self._canonical_map[canonical] = node_id
            logger.debug(f"Added entity: {name} (type: {type})")
    
    def get_entity(self, name: str, type: str) -> Optional[Dict[str, Any]]:
        """
        Get entity data by name and type.
        
        Args:
            name: Entity name
            type: Entity type
            
        Returns:
            Entity node data dict or None if not found
        """
        canonical = self._canonical_key(name, type)
        if canonical not in self._canonical_map:
            return None
        node_id = self._canonical_map[canonical]
        return dict(self.graph.nodes[node_id])
    
    def add_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        description: str,
        strength: float,
        source_chunks: List[str]
    ) -> None:
        """
        Add relationship to graph or merge with existing relationship.
        
        If source or target entity doesn't exist, creates orphan entity.
        If relationship exists between same entities:
        - Sum strengths
        - Accumulate descriptions (deduplicated)
        - Merge source chunks
        - Increment mention count
        
        Args:
            source: Source entity name
            target: Target entity name
            rel_type: Relationship type
            description: Relationship description
            strength: Relationship strength (0-1, can sum to >1)
            source_chunks: List of source chunk IDs
        """
        # Ensure both entities exist (create orphans if needed)
        source_id = self._ensure_entity_exists(source, "")
        target_id = self._ensure_entity_exists(target, "")
        
        # Check if relationship already exists
        existing_edge = None
        if self.graph.has_edge(source_id, target_id):
            # Find matching relationship by type
            for key, edge_data in self.graph[source_id][target_id].items():
                if edge_data.get("relationship_type") == rel_type:
                    existing_edge = (key, edge_data)
                    break
        
        if existing_edge:
            # Relationship exists - merge
            key, edge_data = existing_edge
            
            # Sum strengths
            edge_data["strength"] += strength
            
            # Accumulate descriptions (deduplicated)
            existing_descriptions = set(edge_data["description"].split("\n")) if edge_data["description"] else set()
            if description and description.strip():
                existing_descriptions.add(description.strip())
            edge_data["description"] = "\n".join(sorted(filter(None, existing_descriptions)))
            
            # Merge source chunks (deduplicated)
            existing_sources = set(edge_data["source_chunks"])
            existing_sources.update(source_chunks or [])
            edge_data["source_chunks"] = sorted(existing_sources)
            
            # Increment mention count
            edge_data["mention_count"] += 1
            
            logger.debug(f"Merged relationship: {source} -> {target} (type: {rel_type}, strength: {edge_data['strength']:.2f})")
        else:
            # New relationship - add to graph
            self.graph.add_edge(
                source_id,
                target_id,
                relationship_type=rel_type,
                description=description or "",
                strength=strength,
                source_chunks=list(source_chunks or []),
                mention_count=1
            )
            logger.debug(f"Added relationship: {source} -> {target} (type: {rel_type})")
    
    def _ensure_entity_exists(self, name: str, type: str) -> str:
        """
        Ensure entity exists in graph, creating orphan if needed.
        
        Orphan entities are created when a relationship references
        an entity that wasn't explicitly extracted.
        
        Note: Relationships only know entity names, not types. So we check
        if an entity with this name exists with ANY type. If so, use it.
        Otherwise, create an orphan with empty type.
        
        Args:
            name: Entity name
            type: Entity type (may be empty for orphans)
            
        Returns:
            Node ID
        """
        canonical = self._canonical_key(name, type)
        
        # Check if exact match exists
        if canonical in self._canonical_map:
            return self._canonical_map[canonical]
        
        # If type is empty (from relationship), check if entity exists with ANY type
        if not type or not type.strip():
            normalized_name = canonical[0]  # Already normalized by _canonical_key
            for (cname, ctype), node_id in self._canonical_map.items():
                if cname == normalized_name:
                    # Found entity with this name, use it
                    logger.debug(f"Relationship references entity {name}, found as {node_id}")
                    return node_id
        
        # Create orphan entity
        # Use unique node ID combining name and type
        node_id = f"{name.strip()}_{type.strip()}" if (type and type.strip()) else name.strip()
        self.graph.add_node(
            node_id,
            name=name.strip(),
            type=type.strip() if type else "",
            description="",  # Empty description for orphans
            importance_score=0.0,  # Zero importance for orphans
            source_chunks=[],
            mention_count=0,  # Orphans have 0 direct mentions
            is_orphan=True
        )
        self._canonical_map[canonical] = node_id
        self._orphan_entities.add(node_id)
        logger.debug(f"Created orphan entity: {name}")
        return node_id
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the entity graph.
        
        Returns:
            Dict with node_count, edge_count, orphan_count, etc.
        """
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        orphan_count = len(self._orphan_entities)
        
        # Calculate mention statistics
        total_mentions = sum(
            data.get("mention_count", 0)
            for _, data in self.graph.nodes(data=True)
        )
        avg_mentions = total_mentions / node_count if node_count > 0 else 0
        
        # Calculate max strength
        max_strength = max(
            (data.get("strength", 0) for _, _, data in self.graph.edges(data=True)),
            default=0
        )
        
        # Description length statistics
        description_lengths = {
            node_id: len(data.get("description", ""))
            for node_id, data in self.graph.nodes(data=True)
        }
        
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "orphan_count": orphan_count,
            "total_mentions": total_mentions,
            "avg_mentions_per_entity": avg_mentions,
            "max_strength": max_strength,
            "description_lengths": description_lengths
        }
    
    def to_neo4j_batch_queries(self, doc_id: str) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
        """
        Convert entity graph to Neo4j batch UNWIND queries.
        
        Generates two queries:
        1. Entity UNWIND: Creates/merges all entities
        2. Relationship UNWIND: Creates all relationships
        
        Args:
            doc_id: Document ID for metadata tagging
            
        Returns:
            Tuple of (entity_query, entity_params, rel_query, rel_params)
        """
        from config.settings import settings
        
        # Build entity parameters
        import hashlib

        def _compute_entity_id(name: str) -> str:
            # Deterministic short id matching other codepaths (md5, 16 chars)
            return hashlib.md5((name or "").lower().encode()).hexdigest()[:16]

        entity_params = {
            "entities": [
                {
                    "id": _compute_entity_id(node_data["name"]),
                    "name": node_data["name"],
                    "type": node_data["type"],
                    "description": node_data["description"],
                    "importance_score": node_data["importance_score"],
                    "mention_count": node_data["mention_count"],
                    "source_chunks": node_data["source_chunks"],
                    "is_orphan": node_data.get("is_orphan", False),
                    "phase_version": settings.phase_version,
                    "doc_id": doc_id,
                }
                for node_id, node_data in self.graph.nodes(data=True)
            ]
        }
        
        # Build relationship parameters
        rel_params = {
            "relationships": [
                {
                    "source_id": _compute_entity_id(self.graph.nodes[source]["name"]),
                    "target_id": _compute_entity_id(self.graph.nodes[target]["name"]),
                    "source_name": self.graph.nodes[source]["name"],
                    "target_name": self.graph.nodes[target]["name"],
                    "relationship_type": edge_data["relationship_type"],
                    "description": edge_data["description"],
                    "strength": edge_data["strength"],
                    "mention_count": edge_data["mention_count"],
                    "source_chunks": edge_data["source_chunks"],
                    "phase_version": settings.phase_version,
                    "doc_id": doc_id,
                }
                for source, target, edge_data in self.graph.edges(data=True)
            ]
        }
        
        # Entity UNWIND query
        entity_query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {name: entity.name})
        SET e.type = entity.type,
            e.id = entity.id,
            e.description = entity.description,
            e.importance_score = entity.importance_score,
            e.mention_count = entity.mention_count,
            e.source_chunks = entity.source_chunks,
            e.is_orphan = entity.is_orphan,
            e.phase_version = entity.phase_version,
            e.doc_id = entity.doc_id,
            e.updated_at = datetime()
        """
        
        # Relationship UNWIND query
        rel_query = """
        UNWIND $relationships AS rel
        MATCH (source:Entity {name: rel.source_name})
        MATCH (target:Entity {name: rel.target_name})
        MERGE (source)-[r:RELATED_TO {type: rel.relationship_type}]->(target)
        SET r.description = rel.description,
            r.strength = rel.strength,
            r.mention_count = rel.mention_count,
            r.source_chunks = rel.source_chunks,
            r.phase_version = rel.phase_version,
            r.doc_id = rel.doc_id,
            r.updated_at = datetime()
        """
        
        logger.info(
            f"Generated batch queries: {len(entity_params['entities'])} entities, "
            f"{len(rel_params['relationships'])} relationships"
        )
        
        return (entity_query, entity_params, rel_query, rel_params)
    
    async def summarize_descriptions(self) -> Dict[str, any]:
        """
        Summarize entity and relationship descriptions in graph.
        
        This method:
        1. Identifies entities/relationships with high mention counts
        2. Calls DescriptionSummarizer to generate summaries
        3. Updates node/edge descriptions in NetworkX graph
        4. Returns statistics
        
        Returns:
            Dict with summarization statistics
        
        Usage:
            # After building graph
            entity_graph.add_entities_batch(entities)
            entity_graph.add_relationships_batch(relationships)
            
            # Before Neo4j persistence
            await entity_graph.summarize_descriptions()
            
            # Then persist
            entity_graph.to_neo4j_batch_queries(doc_id)
        """
        # Import here to avoid circular dependency
        from config.settings import settings
        from core.description_summarizer import DescriptionSummarizer
        
        if not settings.enable_description_summarization:
            logger.debug("Description summarization disabled")
            return {"status": "disabled", "entities_summarized": 0, "relationships_summarized": 0}
        
        summarizer = DescriptionSummarizer()
        
        logger.info("Starting description summarization for entity graph")
        
        # === Summarize Entity Descriptions ===
        
        # Collect entities for summarization
        entities_for_summarization = []
        
        for node_name, node_data in self.graph.nodes(data=True):
            description = node_data.get("description", "")
            mention_count = node_data.get("mention_count", 1)
            
            entities_for_summarization.append((node_name, description, mention_count))
        
        # Summarize entities
        entity_results = await summarizer.summarize_entity_descriptions(
            entities_for_summarization
        )
        
        # Apply entity summaries to graph
        for result in entity_results:
            if result.error:
                logger.warning(f"Skipping failed summary for '{result.entity_name}'")
                continue
            
            # Update node description
            if self.graph.has_node(result.entity_name):
                self.graph.nodes[result.entity_name]["description"] = result.summarized_description
                
                logger.debug(
                    f"Updated '{result.entity_name}' description "
                    f"({result.original_length} → {result.summarized_length} chars, "
                    f"{result.compression_ratio:.1%} compression)"
                )
        
        # === Summarize Relationship Descriptions ===
        
        # Collect relationships for summarization
        relationships_for_summarization = []
        
        for source, target, edge_data in self.graph.edges(data=True):
            description = edge_data.get("description", "")
            mention_count = edge_data.get("mention_count", 1)
            rel_type = edge_data.get("relationship_type", "RELATED_TO")
            
            relationships_for_summarization.append(
                (source, target, rel_type, description, mention_count)
            )
        
        # Summarize relationships
        relationship_results = await summarizer.summarize_relationship_descriptions(
            relationships_for_summarization
        )
        
        # Apply relationship summaries to graph
        for result in relationship_results:
            if result.error:
                continue
            
            # Parse relationship identifier (format: "SOURCE -> TARGET (TYPE)")
            # Note: We stored identifier as entity_name in result
            parts = result.entity_name.split(" -> ")
            if len(parts) == 2:
                source = parts[0].strip()
                target_and_type = parts[1].strip()
                target = target_and_type.split(" (")[0].strip()
                
                # Update edge description (handle MultiGraph - multiple edges between same nodes)
                if self.graph.has_edge(source, target):
                    # For MultiGraph, iterate through all edges and update each
                    for edge_key in self.graph[source][target]:
                        self.graph[source][target][edge_key]["description"] = result.summarized_description
                    
                    logger.debug(
                        f"Updated relationship {source} -> {target} description "
                        f"({result.compression_ratio:.1%} compression)"
                    )
        
        # Get summarizer statistics
        summarizer_stats = summarizer.get_statistics()
        
        logger.info(
            f"Description summarization complete: "
            f"{len(entity_results)} entities, {len(relationship_results)} relationships "
            f"(avg compression: {summarizer_stats['average_compression_ratio']:.1%})"
        )
        
        return {
            "status": "success",
            "entities_summarized": len(entity_results),
            "relationships_summarized": len(relationship_results),
            "cache_hits": summarizer_stats["cache_hits"],
            "cache_misses": summarizer_stats["cache_misses"],
            "average_compression_ratio": summarizer_stats["average_compression_ratio"],
            "estimated_tokens_saved": summarizer_stats["estimated_tokens_saved"],
        }
    
    def to_graphml(self, filepath: str) -> None:
        """
        Export graph to GraphML format for debugging/visualization.
        
        Args:
            filepath: Path to output GraphML file
        """
        nx.write_graphml(self.graph, filepath)
        logger.info(f"Exported entity graph to {filepath}")
