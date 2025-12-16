"""
Enhanced retrieval logic with support for chunk-based, entity-based, hybrid modes, and multi-hop reasoning.
"""

import asyncio
import hashlib
import logging
import threading
from enum import Enum
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.embeddings import embedding_manager
from core.graph_db import graph_db
from core.singletons import get_retrieval_cache, hash_retrieval_params
from rag.nodes.query_analysis import analyze_query

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Different retrieval modes supported by the system."""

    CHUNK_ONLY = "chunk_only"
    ENTITY_ONLY = "entity_only"
    HYBRID = "hybrid"


class DocumentRetriever:
    """Document retriever with multiple retrieval strategies."""

    def __init__(self):
        """Initialize the enhanced document retriever."""
        # Initialize retrieval cache for hybrid_retrieval results
        self._cache = get_retrieval_cache()
        self._cache_lock = threading.Lock()
        # Store metadata from last retrieval (for alternative chunks)
        self.last_retrieval_metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def _apply_rrf(ranked_lists: List[List[Dict[str, Any]]], k: int = 60) -> Dict[str, float]:
        """Apply Reciprocal Rank Fusion over multiple ranked lists.

        Args:
            ranked_lists: List of ranked result lists (best first) each item must have a unique 'chunk_id'.
            k: RRF constant controlling discount (typical default ~60).

        Returns:
            Mapping of chunk_id -> fused RRF score.
        """
        scores: Dict[str, float] = {}
        for lst in ranked_lists:
            # ensure we only operate on lists that have items
            if not lst:
                continue
            for rank, item in enumerate(lst, start=1):
                cid = item.get("chunk_id")
                if not cid:
                    continue
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        return scores

    @staticmethod
    def _extract_hashtags_from_query(query: str) -> List[str]:
        """Extract hashtags from query (words starting with #)."""
        import re
        hashtags = re.findall(r'#\w+', query)
        return hashtags

    @staticmethod
    def _filter_chunks_by_documents(
        chunks: List[Dict[str, Any]],
        allowed_document_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Restrict chunks to a set of allowed document IDs."""
        if not allowed_document_ids:
            return chunks

        allowed_set = set(allowed_document_ids)
        return [chunk for chunk in chunks if chunk.get("document_id") in allowed_set]

    def _generate_entity_id(self, entity_name: str) -> str:
        """Generate a consistent entity ID from entity name."""
        return hashlib.md5(entity_name.upper().strip().encode()).hexdigest()

    def _get_entity_ids_from_names(self, entity_names: List[str]) -> List[str]:
        """Get actual entity IDs from entity names by querying the database.

        Args:
            entity_names: List of entity names to look up

        Returns:
            List of actual entity IDs found in database
        """
        if not entity_names:
            return []

        with graph_db.session_scope() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $entity_names
                RETURN e.id as entity_id, e.name as name
                """,
                entity_names=entity_names,
            )
            found_entities = [record.data() for record in result]

        entity_ids = [entity["entity_id"] for entity in found_entities]

        if len(entity_ids) < len(entity_names):
            missing_count = len(entity_names) - len(entity_ids)
            logger.debug(
                f"Found {len(entity_ids)}/{len(entity_names)} entities in database. {missing_count} not found."
            )

        return entity_ids

    async def chunk_based_retrieval(
        self,
        query: str,
        search_query: Optional[str] = None,
        top_k: int = 5,
        allowed_document_ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
        time_decay_weight: float = 0.0,
        temporal_window: Optional[int] = None,
        fuzzy_distance: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Traditional chunk-based retrieval using vector similarity with optional temporal filtering.

        Args:
            query: Original user query (for logging)
            search_query: Contextualized query to use for embeddings (if None, uses query)
            top_k: Number of similar chunks to retrieve
            allowed_document_ids: Optional list of document IDs to restrict retrieval
            query_embedding: Pre-computed query embedding (to avoid recomputation)
            time_decay_weight: Weight for time-decay scoring (0.0-1.0)
            temporal_window: Time window in days for temporal filtering
            fuzzy_distance: Edit distance for fuzzy matching (0=exact, 1-2=fuzzy)

        Returns:
            List of similar chunks with metadata
        """
        try:
            # Use search_query if provided, otherwise fall back to query
            effective_query = search_query if search_query is not None else query

            # Generate query embedding if not provided
            if query_embedding is None:
                query_embedding = embedding_manager.get_embedding(effective_query)

            # Determine if two-stage retrieval should be used
            use_two_stage = False
            if settings.enable_two_stage_retrieval:
                corpus_size = graph_db.estimate_total_chunks()
                use_two_stage = corpus_size >= settings.two_stage_threshold_docs
                if use_two_stage:
                    logger.info(
                        f"Using two-stage retrieval: corpus_size={corpus_size}, "
                        f"threshold={settings.two_stage_threshold_docs}"
                    )

            # Perform retrieval using two-stage or standard approach
            if use_two_stage:
                # Stage 1: BM25 keyword search to get candidates
                candidate_count = top_k * settings.two_stage_multiplier
                logger.debug(f"Stage 1: BM25 search for {candidate_count} candidates")

                keyword_results = graph_db.chunk_keyword_search(
                    query=effective_query,
                    top_k=candidate_count,
                    allowed_document_ids=allowed_document_ids,
                    fuzzy_distance=fuzzy_distance,
                )

                candidate_chunk_ids = [chunk["chunk_id"] for chunk in keyword_results]
                logger.debug(f"Stage 1 complete: {len(candidate_chunk_ids)} candidates from BM25")

                # Stage 2: Vector search only on BM25 candidates
                if candidate_chunk_ids:
                    logger.debug(f"Stage 2: Vector search on {len(candidate_chunk_ids)} candidates")
                    search_limit = top_k * 3  # Allow some buffer for filtering

                    similar_chunks = graph_db.retrieve_chunks_by_ids_with_similarity(
                        query_embedding=query_embedding,
                        candidate_chunk_ids=candidate_chunk_ids,
                        top_k=search_limit,
                    )
                    logger.debug(f"Stage 2 complete: {len(similar_chunks)} chunks with similarity scores")
                else:
                    logger.warning("Stage 1 returned no candidates, falling back to full vector search")
                    similar_chunks = graph_db.vector_similarity_search(
                        query_embedding, top_k * 3
                    )
            else:
                # Standard retrieval path
                search_limit = top_k * 3
                if allowed_document_ids:
                    search_limit = max(search_limit, top_k * 5)

                # Use temporal filtering if enabled and temporal parameters provided
                use_temporal = (
                    settings.enable_temporal_filtering
                    and (time_decay_weight > 0 or temporal_window is not None)
                )

                if use_temporal:
                    # Calculate date range if temporal window is specified
                    from datetime import datetime, timedelta, timezone
                    after_date = None
                    if temporal_window:
                        after_date = (datetime.now(timezone.utc) - timedelta(days=temporal_window)).strftime("%Y-%m-%d")
                        logger.info(f"Using temporal filtering: after_date={after_date}, time_decay_weight={time_decay_weight}")

                    similar_chunks = graph_db.retrieve_chunks_with_temporal_filter(
                        query_embedding,
                        top_k=search_limit,
                        after_date=after_date,
                        time_decay_weight=time_decay_weight,
                        allowed_document_ids=allowed_document_ids,
                    )
                else:
                    similar_chunks = graph_db.vector_similarity_search(
                        query_embedding, search_limit
                    )

            # Enforce document restriction if provided
            similar_chunks = self._filter_chunks_by_documents(
                similar_chunks, allowed_document_ids
            )

            # Filter chunks by minimum similarity threshold
            filtered_chunks = [
                chunk
                for chunk in similar_chunks
                if chunk.get("similarity", 0.0) >= settings.min_retrieval_similarity
            ]

            # Return only top_k after filtering
            final_chunks = filtered_chunks[:top_k]

            logger.info(
                "Retrieved %d chunks, filtered to %d, returning %d chunks using chunk-based retrieval (restricted=%s)",
                len(similar_chunks),
                len(filtered_chunks),
                len(final_chunks),
                bool(allowed_document_ids),
            )
            return final_chunks

        except Exception as e:
            logger.error(f"Chunk-based retrieval failed: {e}")
            return []

    async def entity_based_retrieval(
        self,
        query: str,
        search_query: Optional[str] = None,
        top_k: int = 5,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Entity-based retrieval using entity similarity and relationships.

        Args:
            query: Original user query (for logging)
            search_query: Contextualized query to use for entity search (if None, uses query)
            top_k: Number of relevant chunks to retrieve
            allowed_document_ids: Optional list of document IDs to restrict retrieval

        Returns:
            List of chunks related to relevant entities
        """
        try:
            # Use search_query if provided, otherwise fall back to query
            effective_query = search_query if search_query is not None else query
            
            # First, find relevant entities using full-text search
            relevant_entities = graph_db.entity_similarity_search(effective_query, top_k)

            if not relevant_entities:
                logger.info("No relevant entities found for entity-based retrieval")
                return []

            # Get entity IDs
            entity_ids = [entity["entity_id"] for entity in relevant_entities]

            # Get chunks that contain these entities
            relevant_chunks = graph_db.get_chunks_for_entities(entity_ids)

            # Respect document restriction if provided
            relevant_chunks = self._filter_chunks_by_documents(
                relevant_chunks, allowed_document_ids
            )

            if not relevant_chunks:
                logger.info(
                    "Entity-based retrieval filtered out all chunks due to document restriction"
                )
                return []

            # Calculate similarity scores for chunks based on contextualized query
            query_embedding = embedding_manager.get_embedding(effective_query)

            # Enhance chunks with entity information and similarity scores
            for chunk in relevant_chunks:
                chunk["retrieval_mode"] = "entity_based"
                chunk["relevant_entities"] = chunk.get("contained_entities", [])

                # Calculate similarity score if chunk has content
                if chunk.get("content"):
                    # Get or calculate chunk embedding
                    chunk_embedding = None
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id:
                        # Get embedding from database
                        with graph_db.session_scope() as session:
                            result = session.run(
                                "MATCH (c:Chunk {id: $chunk_id}) RETURN c.embedding as embedding",
                                chunk_id=chunk_id,
                            )
                            record = result.single()
                            if record and record["embedding"]:
                                chunk_embedding = record["embedding"]

                    if chunk_embedding:
                        # Calculate cosine similarity
                        similarity = graph_db._calculate_cosine_similarity(
                            query_embedding, chunk_embedding
                        )
                        chunk["similarity"] = similarity
                    else:
                        # No embedding available - this chunk should be filtered out
                        chunk["similarity"] = 0.0
                else:
                    # No content - this chunk should be filtered out
                    chunk["similarity"] = 0.0

            # Filter chunks by minimum similarity threshold
            filtered_chunks = [
                chunk
                for chunk in relevant_chunks
                if chunk.get("similarity", 0.0) >= settings.min_retrieval_similarity
            ]

            # Sort by similarity score
            filtered_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

            # Return top_k chunks
            final_chunks = filtered_chunks[:top_k]
            logger.info(
                "Entity-based retrieval: found %d chunks, filtered to %d, returning %d with scores (restricted=%s)",
                len(relevant_chunks),
                len(filtered_chunks),
                len(final_chunks),
                bool(allowed_document_ids),
            )

            return final_chunks

        except Exception as e:
            logger.error(f"Entity-based retrieval failed: {e}")
            return []

    async def entity_expansion_retrieval(
        self,
        initial_entities: List[str],
        expansion_depth: int = 1,
        max_chunks: Optional[int] = None,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Expand retrieval by following entity relationships.

        Args:
            initial_entities: List of initial entity IDs
            expansion_depth: How many relationship hops to follow
            max_chunks: Maximum chunks to retrieve (defaults to settings.max_expanded_chunks)

        Returns:
            List of chunks from expanded entity network
        """
        if max_chunks is None:
            max_chunks = settings.max_expanded_chunks

        try:
            expanded_entities = set(initial_entities)
            total_relationships = 0

            # Track entity relationship strengths for scoring
            entity_scores = {}
            for entity_id in initial_entities:
                entity_scores[entity_id] = 1.0  # Initial entities get full score

            # Expand entity network by following relationships (with limits)
            entities_to_process = list(initial_entities)
            current_depth = 0

            while entities_to_process and current_depth < min(
                expansion_depth, settings.max_expansion_depth
            ):
                next_entities = []
                entities_processed_this_depth = 0

                for entity_id in entities_to_process:
                    if entities_processed_this_depth >= settings.max_entity_connections:
                        break  # Limit entities processed per depth

                    relationships = graph_db.get_entity_relationships(entity_id)
                    total_relationships += len(relationships)

                    # Limit and prioritize relationships by strength
                    relationships.sort(
                        key=lambda x: x.get("strength", 0.0), reverse=True
                    )
                    relationships = relationships[: settings.max_entity_connections]

                    for rel in relationships:
                        related_id = rel["related_entity_id"]
                        strength = rel.get("strength", 0.5)

                        # Only follow high-quality relationships
                        if strength >= settings.expansion_similarity_threshold:
                            if related_id not in expanded_entities:
                                expanded_entities.add(related_id)
                                next_entities.append(related_id)

                            # Score related entities based on relationship strength with depth decay
                            decay_factor = 0.7 ** (current_depth + 1)
                            entity_scores[related_id] = max(
                                entity_scores.get(related_id, 0.0),
                                strength * decay_factor,
                            )

                    entities_processed_this_depth += 1

                entities_to_process = next_entities
                current_depth += 1

                # Stop if we have too many entities
                if len(expanded_entities) > settings.max_entity_connections * 3:
                    break

            # Get chunks for expanded entity set (limit the entity set if too large)
            if len(expanded_entities) > settings.max_entity_connections * 2:
                # Keep only the highest scoring entities
                sorted_entities = sorted(
                    expanded_entities,
                    key=lambda eid: entity_scores.get(eid, 0.0),
                    reverse=True,
                )
                expanded_entities = set(
                    sorted_entities[: settings.max_entity_connections * 2]
                )

            expanded_chunks = graph_db.get_chunks_for_entities(list(expanded_entities))

            expanded_chunks = self._filter_chunks_by_documents(
                expanded_chunks, allowed_document_ids
            )

            if not expanded_chunks:
                return []

            # Add expansion metadata and similarity scores
            for chunk in expanded_chunks:
                chunk["retrieval_mode"] = "entity_expansion"
                chunk["expansion_depth"] = current_depth

                # Calculate similarity based on contained entities' scores
                contained_entities = chunk.get("contained_entities", [])
                if contained_entities:
                    # Get entity IDs for contained entities
                    contained_entity_ids = self._get_entity_ids_from_names(
                        contained_entities
                    )
                    if contained_entity_ids:
                        # Use the highest score among contained entities
                        max_entity_score = (
                            max(
                                entity_scores.get(entity_id, 0.0)
                                for entity_id in contained_entity_ids
                                if entity_id in entity_scores
                            )
                            if any(eid in entity_scores for eid in contained_entity_ids)
                            else 0.0
                        )
                        chunk["similarity"] = max_entity_score
                    else:
                        chunk["similarity"] = 0.0  # No entity match
                else:
                    chunk["similarity"] = 0.0  # No entities

            # Filter by enhanced similarity threshold
            filtered_chunks = [
                chunk
                for chunk in expanded_chunks
                if chunk.get("similarity", 0.0)
                >= settings.expansion_similarity_threshold
            ]

            # Sort by similarity score
            filtered_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

            # Apply final chunk limit
            final_chunks = filtered_chunks[:max_chunks]
            logger.info(
                f"Entity expansion: {len(initial_entities)} entities → {len(expanded_entities)} expanded entities "
                f"({total_relationships} relationships, depth {current_depth}) → {len(expanded_chunks)} chunks → {len(filtered_chunks)} filtered → {len(final_chunks)} returned"
            )

            return final_chunks

        except Exception as e:
            logger.error(f"Entity expansion retrieval failed: {e}")
            return []

    async def multi_hop_reasoning_retrieval(
        self,
        query: str,
        search_query: Optional[str] = None,
        seed_top_k: int = 5,
        max_hops: int = 2,
        beam_size: int = 8,
        use_hybrid_seeding: bool = True,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop reasoning retrieval using path traversal.

        Args:
            query: Original user query (for logging)
            search_query: Contextualized query to use for seed selection (if None, uses query)
            seed_top_k: Number of seed entities to start from
            max_hops: Maximum number of hops to traverse
            beam_size: Beam size for path search
            use_hybrid_seeding: Whether to use hybrid seeding (chunks + entities)
            allowed_document_ids: Optional list of document IDs to restrict retrieval

        Returns:
            List of chunks with path-based scoring and provenance
        """
        try:
            if allowed_document_ids:
                logger.info(
                    "Skipping multi-hop reasoning because retrieval is restricted to specific documents"
                )
                return []
            
            # Use search_query if provided, otherwise fall back to query
            effective_query = search_query if search_query is not None else query
            
            # Step 1: Seed entity selection
            seed_entity_ids = []

            if use_hybrid_seeding:
                # Hybrid seeding: get chunks first, then extract entities
                query_embedding = embedding_manager.get_embedding(effective_query)
                similar_chunks = graph_db.vector_similarity_search(
                    query_embedding, seed_top_k * 2
                )

                # Extract entities from these chunks
                chunk_ids = [chunk["chunk_id"] for chunk in similar_chunks]
                if chunk_ids:
                    entities_in_chunks = graph_db.get_entities_for_chunks(chunk_ids)
                    # Sort by importance and take top seed_top_k
                    entities_in_chunks.sort(
                        key=lambda e: e.get("importance_score", 0.0), reverse=True
                    )
                    seed_entity_ids = [
                        e["entity_id"] for e in entities_in_chunks[:seed_top_k]
                    ]
            else:
                # Entity-only seeding: find entities by similarity
                relevant_entities = graph_db.entity_similarity_search(effective_query, seed_top_k)
                seed_entity_ids = [e["entity_id"] for e in relevant_entities]

            if not seed_entity_ids:
                logger.info("No seed entities found for multi-hop reasoning")
                return []

            logger.info(
                f"Multi-hop reasoning: starting with {len(seed_entity_ids)} seed entities"
            )

            # Step 2: Find scored paths using beam search
            paths = graph_db.find_scored_paths(
                seed_entity_ids=seed_entity_ids,
                max_hops=max_hops,
                beam_size=beam_size,
                min_edge_strength=settings.multi_hop_min_edge_strength,
            )

            if not paths:
                logger.info("No paths found in multi-hop reasoning")
                return []

            # Step 3: Compose path contexts and score
            query_embedding = embedding_manager.get_embedding(query)
            path_results = []

            for path in paths:
                # Gather supporting chunks for all hops
                all_chunk_ids = []
                for hop_chunks in path.supporting_chunk_ids:
                    all_chunk_ids.extend(hop_chunks)

                # Remove duplicates while preserving order
                unique_chunk_ids = list(dict.fromkeys(all_chunk_ids))

                if not unique_chunk_ids:
                    # Path has no supporting chunks, skip
                    continue

                # Get chunk data
                with graph_db.session_scope() as session:
                    chunks_data = session.run(
                        """
                        MATCH (c:Chunk)
                        WHERE c.id IN $chunk_ids
                        OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
                        RETURN c.id as chunk_id, c.content as content,
                               c.embedding as embedding,
                               d.filename as document_name, d.id as document_id
                        """,
                        chunk_ids=unique_chunk_ids,
                    ).data()

                # Calculate path embedding (average of entity embeddings)
                entity_embeddings = [
                    e.embedding for e in path.entities if e.embedding is not None
                ]

                if entity_embeddings:
                    # Average entity embeddings for path representation
                    path_embedding = [
                        sum(emb[i] for emb in entity_embeddings)
                        / len(entity_embeddings)
                        for i in range(len(entity_embeddings[0]))
                    ]

                    # Calculate query similarity to path
                    path_query_similarity = graph_db._calculate_cosine_similarity(
                        query_embedding, path_embedding
                    )
                else:
                    path_query_similarity = 0.0

                # Calculate max chunk similarity
                max_chunk_similarity = 0.0
                for chunk_data in chunks_data:
                    if chunk_data.get("embedding"):
                        chunk_sim = graph_db._calculate_cosine_similarity(
                            query_embedding, chunk_data["embedding"]
                        )
                        max_chunk_similarity = max(max_chunk_similarity, chunk_sim)

                # Compute final path score with configurable weights
                # alpha * path_score_from_edges + beta * query_similarity + gamma * max_chunk_sim
                alpha, beta, gamma = 0.6, 0.3, 0.1
                final_score = (
                    alpha * path.score
                    + beta * path_query_similarity
                    + gamma * max_chunk_similarity
                )

                # Create result entries for each chunk in the path
                for chunk_data in chunks_data:
                    path_result = {
                        "chunk_id": chunk_data["chunk_id"],
                        "content": chunk_data["content"],
                        "document_name": chunk_data.get("document_name", "Unknown"),
                        "document_id": chunk_data.get("document_id", ""),
                        "similarity": final_score,
                        "retrieval_mode": "multi_hop_reasoning",
                        "path_score": path.score,
                        "path_query_similarity": path_query_similarity,
                        "max_chunk_similarity": max_chunk_similarity,
                        "path_entities": [e.name for e in path.entities],
                        "path_relationships": [
                            {
                                "type": r.type,
                                "description": r.description,
                                "strength": r.strength,
                            }
                            for r in path.relationships
                        ],
                        "path_length": len(path.entities),
                    }
                    path_results.append(path_result)

            # Sort by similarity and deduplicate by chunk_id
            seen_chunks = {}
            for result in path_results:
                chunk_id = result["chunk_id"]
                if (
                    chunk_id not in seen_chunks
                    or result["similarity"] > seen_chunks[chunk_id]["similarity"]
                ):
                    seen_chunks[chunk_id] = result

            final_results = list(seen_chunks.values())
            final_results.sort(key=lambda x: x["similarity"], reverse=True)

            logger.info(
                f"Multi-hop reasoning: found {len(paths)} paths, "
                f"generated {len(final_results)} unique chunks"
            )
            return final_results

        except Exception as e:
            logger.error(f"Multi-hop reasoning retrieval failed: {e}")
            return []

    async def hybrid_retrieval(
        self,
        query: str,
        contextualized_query: Optional[str] = None,
        top_k: int = 5,
        chunk_weight: float = 0.5,
        entity_weight: Optional[float] = None,
        path_weight: Optional[float] = None,
        use_multi_hop: bool = False,
        max_hops: Optional[int] = None,
        beam_size: Optional[int] = None,
        restrict_to_context: bool = True,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval with result caching and contextualized query support.
        
        Args:
            query: Original user query (for logging/debugging)
            contextualized_query: Enriched query with conversation context (used for actual search)
            top_k: Number of results to return
            chunk_weight: Weight for chunk-based results
            entity_weight: Weight for entity-based results
            path_weight: Weight for path-based results
            use_multi_hop: Whether to use multi-hop reasoning
            max_hops: Maximum hops for multi-hop
            beam_size: Beam size for multi-hop
            restrict_to_context: Whether to restrict to context documents
            allowed_document_ids: List of allowed document IDs
        
        Cache Strategy:
            - Hit: Return cached results (TTL: 60s)
            - Miss: Perform retrieval, cache, return
        """
        # Use contextualized query if provided, otherwise use original
        search_query = contextualized_query or query
        
        if contextualized_query and contextualized_query != query:
            logger.info(f"Using contextualized query: '{search_query[:80]}...' (original: '{query[:50]}...')")
        
        # Check if caching is enabled
        if not settings.enable_caching:
            return await self._hybrid_retrieval_direct(
                query, search_query, top_k, chunk_weight, entity_weight, path_weight,
                use_multi_hop, max_hops, beam_size, restrict_to_context,
                allowed_document_ids
            )
        
        # Normalize parameters for cache key
        normalized_entity_weight = (
            settings.hybrid_entity_weight if entity_weight is None else entity_weight
        )
        normalized_path_weight = (
            settings.hybrid_path_weight if path_weight is None else path_weight
        )
        
        # Generate cache key using contextualized query (important for follow-ups)
        # Include ALL parameters that affect retrieval to ensure proper cache isolation
        cache_key = hash_retrieval_params(
            query=search_query,  # Use contextualized query for cache key
            mode="hybrid",
            top_k=top_k,
            chunk_weight=chunk_weight,
            entity_weight=normalized_entity_weight,
            path_weight=normalized_path_weight,
            use_multi_hop=use_multi_hop,
            max_hops=max_hops,
            beam_size=beam_size,
            restrict_to_context=restrict_to_context,
            allowed_document_ids=allowed_document_ids,
            embedding_model=settings.embedding_model,
            enable_rrf=settings.enable_rrf,
            flashrank_enabled=settings.flashrank_enabled,
            routing_categories=allowed_document_ids,  # Include routing context in cache key
        )
        
        # Check cache (thread-safe read)
        with self._cache_lock:
            if cache_key in self._cache:
                logger.info(f"Retrieval cache HIT for query: {search_query[:50]}...")
                return self._cache[cache_key]
        
        # Cache miss - perform retrieval
        logger.info(f"Retrieval cache MISS for query: {search_query[:50]}...")
        results = await self._hybrid_retrieval_direct(
            query, search_query, top_k, chunk_weight, entity_weight, path_weight,
            use_multi_hop, max_hops, beam_size, restrict_to_context,
            allowed_document_ids
        )
        
        # Store in cache (thread-safe write)
        with self._cache_lock:
            self._cache[cache_key] = results
        
        return results

    async def _hybrid_retrieval_direct(
        self,
        query: str,
        search_query: str,
        top_k: int = 5,
        chunk_weight: float = 0.5,
        entity_weight: Optional[float] = None,
        path_weight: Optional[float] = None,
        use_multi_hop: bool = False,
        max_hops: Optional[int] = None,
        beam_size: Optional[int] = None,
        restrict_to_context: bool = True,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Direct hybrid retrieval without caching.
        
        Combines chunk-based, entity-based, and optionally multi-hop approaches.

        Args:
            query: Original user query (for logging)
            search_query: Contextualized query to use for actual retrieval
            top_k: Total number of chunks to retrieve
            chunk_weight: Weight for chunk-based results (0.0-1.0)
            entity_weight: Weight for entity-based results (0.0-1.0)
            path_weight: Weight for multi-hop path results (0.0-1.0)
            use_multi_hop: Whether to include multi-hop reasoning
            max_hops: Depth limit for multi-hop traversal
            beam_size: Beam width for path search
            restrict_to_context: Whether to enforce provided context document boundaries

        Returns:
            List of chunks from all approaches, de-duplicated and ranked
        """
        try:
            normalized_chunk_weight = max(0.0, min(chunk_weight, 1.0))
            normalized_entity_weight = (
                settings.hybrid_entity_weight
                if entity_weight is None
                else max(0.0, min(entity_weight, 1.0))
            )
            normalized_path_weight = (
                settings.hybrid_path_weight
                if path_weight is None
                else max(0.0, min(path_weight, 1.0))
            )

            # Analyze query to determine if multi-hop would be beneficial
            # Use search_query (contextualized) for analysis
            query_analysis = analyze_query(search_query)
            multi_hop_recommended = query_analysis.get("multi_hop_recommended", True)
            query_type = query_analysis.get("query_type", "factual")
            suggested_strategy = query_analysis.get("suggested_strategy", "balanced")
            
            # Initialize keyword_weight
            keyword_weight = settings.keyword_search_weight if settings.enable_chunk_fulltext else 0.0
            
            # Apply strategy-based weight adjustments
            if suggested_strategy == "entity_focused":
                # Boost entity weight for relationship queries
                normalized_entity_weight = min(1.0, normalized_entity_weight * 1.3)
                normalized_chunk_weight = max(0.3, normalized_chunk_weight * 0.8)
            elif suggested_strategy == "keyword_focused":
                # Boost keyword weight for exact-term queries
                if settings.enable_chunk_fulltext:
                    keyword_weight = min(0.5, keyword_weight * 1.4)
                normalized_chunk_weight = min(1.0, normalized_chunk_weight * 1.1)
                normalized_entity_weight = max(0.2, normalized_entity_weight * 0.7)

            # Only apply restriction if both enabled AND there are actual documents to restrict to
            # Empty list means "no context documents selected" -> should search all docs
            allowed_set = (
                set(allowed_document_ids)
                if (allowed_document_ids and len(allowed_document_ids) > 0 and restrict_to_context)
                else None
            )
            if not restrict_to_context or not allowed_document_ids:
                allowed_document_ids = None

            # Override multi-hop decision based on query analysis
            effective_use_multi_hop = (
                use_multi_hop
                and multi_hop_recommended
                and not (allowed_document_ids and len(allowed_document_ids) > 0)
            )

            # Adjust path weight based on query type
            base_path_weight = normalized_path_weight
            if query_type == "comparative":
                # Comparative queries benefit more from multi-hop
                path_weight = min(0.8, base_path_weight * 1.3)
            elif query_type == "analytical":
                # Analytical queries benefit from multi-hop
                path_weight = min(0.7, base_path_weight * 1.1)
            else:
                # Factual queries - reduce multi-hop influence
                path_weight = max(0.2, base_path_weight * 0.7)

            logger.info(
                "Multi-hop decision: requested=%s, recommended=%s, effective=%s, query_type=%s, strategy=%s, path_weight=%.2f, restricted=%s",
                use_multi_hop,
                multi_hop_recommended,
                effective_use_multi_hop,
                query_type,
                suggested_strategy,
                path_weight,
                bool(allowed_document_ids),
            )
            
            # Store metadata for potential expansion
            retrieval_metadata = {
                "query_analysis": query_analysis,
                "initial_strategy": suggested_strategy,
            }

            # Calculate split for each approach including optional keyword search
            path_weight = path_weight if effective_use_multi_hop else 0.0
            # keyword_weight already initialized above based on strategy
            
            combined_weight = max(
                1e-5,
                normalized_chunk_weight + normalized_entity_weight + path_weight + keyword_weight,
            )
            chunk_fraction = normalized_chunk_weight / combined_weight
            entity_fraction = normalized_entity_weight / combined_weight
            path_fraction = path_weight / combined_weight
            keyword_fraction = keyword_weight / combined_weight

            # Fetch extra results for follow-up suggestions (top_k + 3 for alternatives)
            effective_top_k = top_k + 3
            
            logger.info(f"Weight fractions: chunk={chunk_fraction:.3f}, entity={entity_fraction:.3f}, path={path_fraction:.3f}, keyword={keyword_fraction:.3f}")
            
            # Use ceiling for counts to ensure we get enough total results for alternatives
            import math
            chunk_count = max(1, math.ceil(effective_top_k * chunk_fraction))
            entity_count = max(1, math.ceil(effective_top_k * entity_fraction))
            path_count = max(0, math.ceil(effective_top_k * path_fraction))
            keyword_count = max(0, math.ceil(effective_top_k * keyword_fraction)) if keyword_weight > 0 else 0
            
            logger.info(f"Retrieval counts: top_k={top_k}, effective_top_k={effective_top_k}, chunk={chunk_count}, entity={entity_count}, path={path_count}, keyword={keyword_count}")
            
            total_count = chunk_count + entity_count + path_count + keyword_count
            logger.debug(f"Before trim: chunk={chunk_count}, entity={entity_count}, path={path_count}, keyword={keyword_count}, total={total_count}, effective_top_k={effective_top_k}")
            
            if total_count > effective_top_k:
                # Trim the smallest bucket to respect the limit (use effective_top_k which includes alternatives)
                overage = total_count - effective_top_k
                logger.debug(f"Trimming overage: {overage}")
                if keyword_count > 0:
                    keyword_count = max(0, keyword_count - overage)
                elif path_count > 0:
                    path_count = max(0, path_count - overage)
                else:
                    entity_count = max(1, entity_count - overage)
            
            logger.info(f"Final counts: chunk={chunk_count}, entity={entity_count}, path={path_count}, keyword={keyword_count}")

            # Extract temporal parameters from query analysis
            time_decay_weight = query_analysis.get("time_decay_weight", 0.0)
            temporal_window = query_analysis.get("temporal_window")
            is_temporal = query_analysis.get("is_temporal", False)

            # Extract fuzzy matching parameters from query analysis
            fuzzy_distance = query_analysis.get("fuzzy_distance", 0)
            is_technical = query_analysis.get("is_technical", False)

            if is_temporal:
                logger.info(f"Temporal query detected: time_decay_weight={time_decay_weight}, window={temporal_window} days")

            if is_technical:
                logger.info(f"Technical query detected: fuzzy_distance={fuzzy_distance}")

            # Get results from different approaches using search_query
            chunk_results = await self.chunk_based_retrieval(
                query,
                search_query,
                chunk_count,
                allowed_document_ids=allowed_document_ids,
                time_decay_weight=time_decay_weight,
                temporal_window=temporal_window,
                fuzzy_distance=fuzzy_distance,
            )
            entity_results = await self.entity_based_retrieval(
                query, search_query, entity_count, allowed_document_ids=allowed_document_ids
            )
            
            # Get keyword search results if enabled
            keyword_results = []
            if settings.enable_chunk_fulltext and keyword_count > 0:
                try:
                    import asyncio as aio
                    loop = aio.get_event_loop()
                    keyword_results = await loop.run_in_executor(
                        None,
                        graph_db.chunk_keyword_search,
                        query,
                        keyword_count,
                        allowed_document_ids,
                    )
                    logger.info(f"Keyword search returned {len(keyword_results)} chunks")
                except Exception as e:
                    logger.warning(f"Keyword search failed, continuing without it: {e}")
                    keyword_results = []

            path_results = []
            if effective_use_multi_hop:
                path_results = await self.multi_hop_reasoning_retrieval(
                    query,
                    search_query,
                    seed_top_k=5,
                    max_hops=max_hops or settings.multi_hop_max_hops,
                    beam_size=beam_size or settings.multi_hop_beam_size,
                    use_hybrid_seeding=True,
                    allowed_document_ids=allowed_document_ids,
                )
                # Limit path results, but be more generous if we have high-quality results
                if len(path_results) > path_count:
                    # Sort by similarity to keep the best ones
                    path_results.sort(
                        key=lambda x: x.get("similarity", 0.0), reverse=True
                    )
                    # Keep top path_count, but allow up to 2x if quality is high
                    high_quality_count = sum(
                        1 for r in path_results if r.get("similarity", 0.0) > 0.5
                    )
                    keep_count = min(
                        len(path_results),
                        max(path_count, min(high_quality_count, path_count * 2)),
                    )
                    path_results = path_results[:keep_count]
                    logger.info(
                        f"Multi-hop: keeping {keep_count}/{len(path_results)} path results (path_count={path_count}, high_quality={high_quality_count})"
                    )

            # Optional: Use Reciprocal Rank Fusion to combine lists
            if getattr(settings, "enable_rrf", False):
                try:
                    # Prepare ranked lists by modality-specific score
                    ranked_chunk = sorted(
                        chunk_results, key=lambda x: x.get("similarity", 0.0), reverse=True
                    ) if chunk_results else []
                    ranked_entity = sorted(
                        entity_results, key=lambda x: x.get("similarity", 0.0), reverse=True
                    ) if entity_results else []
                    ranked_keyword = sorted(
                        keyword_results, key=lambda x: x.get("keyword_score", 0.0), reverse=True
                    ) if keyword_results else []
                    ranked_path = sorted(
                        path_results, key=lambda x: x.get("similarity", 0.0), reverse=True
                    ) if path_results else []

                    lists_to_fuse = [lst for lst in [ranked_chunk, ranked_entity, ranked_keyword, ranked_path] if lst]
                    rrf_scores = self._apply_rrf(lists_to_fuse, getattr(settings, "rrf_k", 60)) if lists_to_fuse else {}

                    # Choose a representative item per chunk_id, prioritizing chunk->entity->keyword->path
                    by_chunk: Dict[str, Dict[str, Any]] = {}
                    def ingest(lst: List[Dict[str, Any]]):
                        for it in lst:
                            cid = it.get("chunk_id")
                            if cid and cid not in by_chunk:
                                by_chunk[cid] = dict(it)
                    ingest(ranked_chunk)
                    ingest(ranked_entity)
                    ingest(ranked_keyword)
                    ingest(ranked_path)

                    # Attach scores and mark source
                    for cid, score in rrf_scores.items():
                        if cid in by_chunk:
                            by_chunk[cid]["rrf_score"] = score
                            # For downstream compatibility, store under hybrid_score as well
                            by_chunk[cid]["hybrid_score"] = score
                            prev = by_chunk[cid].get("retrieval_source", "")
                            if prev:
                                by_chunk[cid]["retrieval_source"] = f"{prev}+rrf" if "rrf" not in prev else prev
                            else:
                                by_chunk[cid]["retrieval_source"] = "hybrid_rrf"

                    final_results = [v for k, v in by_chunk.items() if k in rrf_scores]
                    final_results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

                    # Optionally rerank top candidates using FlashRank (runs in threadpool)
                    if getattr(settings, "flashrank_enabled", False) and final_results:
                        try:
                            import asyncio

                            from rag.rerankers.flashrank_reranker import rerank_with_flashrank
                            from core.singletons import get_blocking_executor, SHUTTING_DOWN

                            dynamic_base = max(top_k * 4, 32)
                            cap = min(
                                len(final_results),
                                getattr(settings, "flashrank_max_candidates", 100),
                                dynamic_base,
                            )
                            loop = asyncio.get_running_loop()
                            try:
                                executor = get_blocking_executor()
                                reranked = await loop.run_in_executor(executor, rerank_with_flashrank, query, final_results, cap)
                            except RuntimeError as e:
                                logger.debug(f"Blocking executor unavailable for FlashRank rerank: {e}.")
                                if SHUTTING_DOWN:
                                    logger.info("Process shutting down; skipping FlashRank rerank")
                                    reranked = []
                                else:
                                    executor = get_blocking_executor()
                                    reranked = await loop.run_in_executor(executor, rerank_with_flashrank, query, final_results, cap)

                            if isinstance(reranked, list) and reranked:
                                final_results = reranked
                                logger.info("Applied FlashRank reranker to top %d candidates (RRF path)", cap)
                        except Exception as e:
                            logger.warning("FlashRank reranker failed on RRF path (continuing without it): %s", e)

                    # Store alternative chunks for follow-up suggestions (RRF path)
                    alternatives = final_results[top_k:top_k+3]
                    retrieval_metadata["alternative_chunks"] = alternatives
                    self.last_retrieval_metadata = retrieval_metadata
                    logger.info(f"Storing {len(alternatives)} alternative chunks for follow-ups (RRF path, total results: {len(final_results)}, top_k: {top_k}, slice [{top_k}:{top_k+3}])")
                    
                    return final_results[:top_k]
                except Exception as e:
                    logger.warning(f"RRF fusion failed, falling back to weighted hybrid: {e}")

            # Combine and deduplicate results by chunk_id
            combined_results = {}

            # Add chunk-based results
            for result in chunk_results:
                chunk_id = result.get("chunk_id")
                if chunk_id:
                    result["retrieval_source"] = "chunk_based"
                    result["chunk_score"] = result.get("similarity", 0.0)
                    result["hybrid_score"] = result.get("chunk_score", 0.0)
                    combined_results[chunk_id] = result

            # Add entity-based results (merge if duplicate)
            for result in entity_results:
                chunk_id = result.get("chunk_id")
                if chunk_id:
                    if chunk_id in combined_results:
                        # Merge information from both sources
                        existing = combined_results[chunk_id]
                        existing["retrieval_source"] = "hybrid"
                        existing["relevant_entities"] = result.get(
                            "contained_entities", []
                        )
                        existing["contained_entities"] = result.get(
                            "contained_entities", []
                        )
                        # Boost score for chunks found by both methods
                        chunk_score = existing.get("chunk_score", 0.0)
                        entity_score = result.get("similarity", 0.3)
                        weighted_score = (
                            chunk_fraction * chunk_score
                            + entity_fraction * entity_score
                        )
                        existing["hybrid_score"] = min(1.0, weighted_score)
                    else:
                        result["retrieval_source"] = "entity_based"
                        # Use actual similarity score, with better fallback
                        if allowed_set and result.get("document_id") not in allowed_set:
                            continue
                        result["hybrid_score"] = min(
                            1.0, entity_fraction * result.get("similarity", 0.3)
                        )
                        combined_results[chunk_id] = result

            # Add path-based results (merge if duplicate)
            for result in path_results:
                chunk_id = result.get("chunk_id")
                if chunk_id:
                    if chunk_id in combined_results:
                        # Merge with existing
                        existing = combined_results[chunk_id]
                        existing["retrieval_source"] = "hybrid_with_paths"
                        existing["path_entities"] = result.get("path_entities", [])
                        existing["path_relationships"] = result.get(
                            "path_relationships", []
                        )
                        existing["path_length"] = result.get("path_length", 0)
                        # Boost score for chunks found by multiple methods
                        current_score = existing.get("hybrid_score", 0.0)
                        path_score = result.get("similarity", 0.3)
                        weighted_score = (
                            current_score * (chunk_fraction + entity_fraction)
                            + path_fraction * path_score
                        )
                        existing["hybrid_score"] = min(1.0, weighted_score)
                    else:
                        result["retrieval_source"] = "path_based"
                        result["hybrid_score"] = min(
                            1.0, path_fraction * result.get("similarity", 0.3)
                        )
                        combined_results[chunk_id] = result

            # Add keyword-based results (merge if duplicate)
            for result in keyword_results:
                chunk_id = result.get("chunk_id")
                if chunk_id:
                    if chunk_id in combined_results:
                        # Merge with existing
                        existing = combined_results[chunk_id]
                        existing["retrieval_source"] = "hybrid_with_keywords"
                        existing["keyword_score"] = result.get("keyword_score", 0.0)
                        # Boost score for chunks found by multiple methods
                        current_score = existing.get("hybrid_score", 0.0)
                        # Normalize keyword score to 0-1 range (fulltext scores can be > 1)
                        normalized_keyword_score = min(1.0, result.get("keyword_score", 0.0) / 5.0)
                        weighted_score = (
                            current_score * (chunk_fraction + entity_fraction + path_fraction)
                            + keyword_fraction * normalized_keyword_score
                        )
                        existing["hybrid_score"] = min(1.0, weighted_score)
                    else:
                        result["retrieval_source"] = "keyword_based"
                        # Normalize keyword score for hybrid scoring
                        normalized_keyword_score = min(1.0, result.get("keyword_score", 0.0) / 5.0)
                        result["similarity"] = normalized_keyword_score
                        result["hybrid_score"] = min(
                            1.0, keyword_fraction * normalized_keyword_score
                        )
                        combined_results[chunk_id] = result

            # Sort by hybrid score and return top_k
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

            # Count overlaps for better reporting
            chunk_only_count = sum(
                1 for r in final_results if r.get("retrieval_source") == "chunk_based"
            )
            entity_only_count = sum(
                1 for r in final_results if r.get("retrieval_source") == "entity_based"
            )
            path_only_count = sum(
                1 for r in final_results if r.get("retrieval_source") == "path_based"
            )
            hybrid_count = sum(
                1
                for r in final_results
                if r.get("retrieval_source") in ["hybrid", "hybrid_with_paths"]
            )

            logger.info(
                f"Hybrid retrieval: {len(chunk_results)} chunk + {len(entity_results)} entity"
                + (f" + {len(path_results)} path" if effective_use_multi_hop else "")
                + f" → {len(final_results)} total "
                f"({chunk_only_count} chunk-only, {entity_only_count} entity-only"
                + (f", {path_only_count} path-only" if effective_use_multi_hop else "")
                + f", {hybrid_count} overlapping)"
            )
            
            # Optional query expansion if results are sparse
            if getattr(settings, "enable_query_expansion", False) and len(final_results) < getattr(settings, "query_expansion_threshold", 3):
                try:
                    from rag.query_expansion import expand_query_terms
                    expanded_terms = expand_query_terms(query, len(final_results))
                    if expanded_terms:
                        retrieval_metadata["expanded_terms"] = expanded_terms
                        logger.info(f"Query expansion generated {len(expanded_terms)} terms for sparse results")
                        # Could optionally re-retrieve with expanded query here
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}")

            # Optionally rerank top candidates using FlashRank (runs in threadpool)
            if getattr(settings, "flashrank_enabled", False) and final_results:
                try:
                    import asyncio

                    from rag.rerankers.flashrank_reranker import rerank_with_flashrank
                    from core.singletons import get_blocking_executor, SHUTTING_DOWN

                    dynamic_base = max(top_k * 4, 32)
                    cap = min(
                        len(final_results),
                        getattr(settings, "flashrank_max_candidates", 100),
                        dynamic_base,
                    )
                    loop = asyncio.get_running_loop()
                    # execute the synchronous reranker in a thread to avoid blocking
                    try:
                        executor = get_blocking_executor()
                        reranked = await loop.run_in_executor(executor, rerank_with_flashrank, query, final_results, cap)
                    except RuntimeError as e:
                        logger.debug(f"Blocking executor unavailable for FlashRank rerank: {e}.")
                        if SHUTTING_DOWN:
                            logger.info("Process shutting down; skipping FlashRank rerank")
                            reranked = []
                        else:
                            executor = get_blocking_executor()
                            reranked = await loop.run_in_executor(executor, rerank_with_flashrank, query, final_results, cap)

                    if isinstance(reranked, list) and reranked:
                        final_results = reranked
                        logger.info("Applied FlashRank reranker to top %d candidates", cap)
                except Exception as e:
                    logger.warning("FlashRank reranker failed (continuing without it): %s", e)

            # Store alternative chunks for follow-up suggestions (next 3 after top_k)
            retrieval_metadata["alternative_chunks"] = final_results[top_k:top_k+3]
            
            # Store metadata for access by retrieval node
            self.last_retrieval_metadata = retrieval_metadata
            
            logger.info(f"Storing {len(final_results[top_k:top_k+3])} alternative chunks for follow-ups (total results: {len(final_results)})")
            
            return final_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []

    async def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 5,
        use_multi_hop: bool = False,
        chunk_weight: float = 0.5,
        entity_weight: Optional[float] = None,
        path_weight: Optional[float] = None,
        max_hops: Optional[int] = None,
        beam_size: Optional[int] = None,
        restrict_to_context: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method that dispatches to the appropriate strategy.

        Args:
            query: User query
            mode: Retrieval mode to use
            top_k: Number of results to return
            use_multi_hop: Whether to use multi-hop reasoning (for hybrid mode)
            **kwargs: Additional parameters for specific retrieval modes

        Returns:
            List of relevant chunks with metadata
        """
        logger.info(
            f"Starting retrieval with mode: {mode.value}, top_k: {top_k}, multi_hop: {use_multi_hop}"
        )

        allowed_document_ids = kwargs.pop("allowed_document_ids", None)

        if mode == RetrievalMode.CHUNK_ONLY:
            # Note: query here is already contextualized if it came from retrieval.py
            return await self.chunk_based_retrieval(
                query=query,
                search_query=query,  # Pass same value - already contextualized
                top_k=top_k,
                allowed_document_ids=allowed_document_ids
            )
        elif mode == RetrievalMode.ENTITY_ONLY:
            # Note: query here is already contextualized if it came from retrieval.py
            return await self.entity_based_retrieval(
                query=query,
                search_query=query,  # Pass same value - already contextualized
                top_k=top_k,
                allowed_document_ids=allowed_document_ids
            )
        elif mode == RetrievalMode.HYBRID:
            # Note: query here is already contextualized if it came from retrieval.py
            # We pass it as both query and contextualized_query since it's the effective search query
            return await self.hybrid_retrieval(
                query=query,
                contextualized_query=query,  # Pass same value - already contextualized
                top_k=top_k,
                chunk_weight=chunk_weight,
                entity_weight=entity_weight,
                path_weight=path_weight,
                use_multi_hop=use_multi_hop,
                max_hops=max_hops,
                beam_size=beam_size,
                restrict_to_context=restrict_to_context,
                allowed_document_ids=allowed_document_ids,
            )
        else:
            logger.error(f"Unknown retrieval mode: {mode}")
            return []

    async def retrieve_with_graph_expansion(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 3,
        expand_depth: int = 2,
        use_multi_hop: bool = False,
        chunk_weight: float = 0.5,
        entity_weight: Optional[float] = None,
        path_weight: Optional[float] = None,
        max_hops: Optional[int] = None,
        beam_size: Optional[int] = None,
        restrict_to_context: bool = True,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks and expand using graph relationships.

        Args:
            query: User query
            mode: Initial retrieval mode
            top_k: Number of initial chunks to retrieve
            expand_depth: Depth of graph expansion
            use_multi_hop: Whether to use multi-hop reasoning

        Returns:
            List of chunks including expanded context
        """
        try:
            # Get initial results
            initial_chunks = await self.retrieve(
                query,
                mode,
                top_k,
                use_multi_hop=use_multi_hop,
                chunk_weight=chunk_weight,
                entity_weight=entity_weight,
                path_weight=path_weight,
                max_hops=max_hops,
                beam_size=beam_size,
                restrict_to_context=restrict_to_context,
                allowed_document_ids=allowed_document_ids,
            )

            if not initial_chunks:
                return []

            allowed_set = set(allowed_document_ids) if allowed_document_ids else None

            if allowed_set:
                initial_chunks = [
                    chunk
                    for chunk in initial_chunks
                    if chunk.get("document_id") in allowed_set
                ]

                if not initial_chunks:
                    logger.info(
                        "Initial retrieval yielded no chunks after applying document restriction"
                    )
                    return []

            expanded_chunks = []
            seen_chunk_ids = set()

            # Add initial chunks
            for chunk in initial_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    if allowed_set and chunk.get("document_id") not in allowed_set:
                        continue
                    expanded_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)

            # Expand based on mode
            if mode in [RetrievalMode.ENTITY_ONLY, RetrievalMode.HYBRID]:
                # Entity-based expansion (with limits)
                entity_chunks_added = 0
                for chunk in initial_chunks:
                    if entity_chunks_added >= settings.max_expanded_chunks // 2:
                        break  # Reserve half the limit for entity expansion

                    entities = chunk.get("contained_entities", [])
                    if entities:
                        entity_ids = self._get_entity_ids_from_names(entities)
                        if entity_ids:  # Only expand if we found valid entity IDs
                            # Limit entities to process
                            entity_ids = entity_ids[: settings.max_entity_connections]

                            expanded = await self.entity_expansion_retrieval(
                                entity_ids,
                                expansion_depth=expand_depth,
                                max_chunks=settings.max_expanded_chunks
                                // len(initial_chunks)
                                + 1,
                                allowed_document_ids=allowed_document_ids,
                            )

                            source_similarity = chunk.get(
                                "similarity", chunk.get("hybrid_score", 0.3)
                            )

                            for exp_chunk in expanded:
                                exp_chunk_id = exp_chunk.get("chunk_id")
                                if exp_chunk_id and exp_chunk_id not in seen_chunk_ids:
                                    if allowed_set and exp_chunk.get("document_id") not in allowed_set:
                                        continue
                                    chunk_similarity = exp_chunk.get("similarity", 0.0)

                                    # Only add high-quality expansions
                                    if (
                                        chunk_similarity
                                        >= settings.expansion_similarity_threshold
                                    ):
                                        exp_chunk["expansion_context"] = {
                                            "source_chunk": chunk.get("chunk_id"),
                                            "expansion_type": "entity_relationship",
                                        }
                                        # Use the calculated similarity from entity expansion
                                        expanded_chunks.append(exp_chunk)
                                        seen_chunk_ids.add(exp_chunk_id)
                                        entity_chunks_added += 1

                                        if (
                                            entity_chunks_added
                                            >= settings.max_expanded_chunks // 2
                                        ):
                                            break

                        if entity_chunks_added >= settings.max_expanded_chunks // 2:
                            break

            if mode in [RetrievalMode.CHUNK_ONLY, RetrievalMode.HYBRID]:
                # Chunk similarity expansion (with limits)
                chunks_added = 0
                for chunk in initial_chunks:
                    if chunks_added >= settings.max_chunk_connections * len(
                        initial_chunks
                    ):
                        break  # Limit total chunk expansion

                    chunk_id = chunk.get("chunk_id")
                    if chunk_id:
                        # Limit depth and get related chunks
                        effective_depth = min(
                            expand_depth, settings.max_expansion_depth
                        )
                        related_chunks = graph_db.get_related_chunks(
                            chunk_id, max_depth=effective_depth
                        )

                        # Limit and prioritize by similarity
                        related_chunks = related_chunks[
                            : settings.max_chunk_connections
                        ]

                        source_similarity = chunk.get(
                            "similarity", chunk.get("hybrid_score", 0.3)
                        )

                        for rel_chunk in related_chunks:
                            rel_chunk_id = rel_chunk.get("chunk_id")
                            if rel_chunk_id and rel_chunk_id not in seen_chunk_ids:
                                if allowed_set and rel_chunk.get("document_id") not in allowed_set:
                                    continue
                                # Calculate similarity based on distance and source similarity
                                distance = rel_chunk.get("distance", 1)
                                # Decay factor based on graph distance
                                decay_factor = 1.0 / (distance + 1)
                                calculated_similarity = source_similarity * decay_factor

                                # Only add if similarity meets threshold
                                if (
                                    calculated_similarity
                                    >= settings.expansion_similarity_threshold
                                ):
                                    rel_chunk["expansion_context"] = {
                                        "source_chunk": chunk_id,
                                        "expansion_type": "chunk_similarity",
                                    }
                                    rel_chunk["similarity"] = calculated_similarity
                                    expanded_chunks.append(rel_chunk)
                                    seen_chunk_ids.add(rel_chunk_id)
                                    chunks_added += 1

                                    # Stop if we've added too many
                                    if chunks_added >= settings.max_expanded_chunks:
                                        break

                        if chunks_added >= settings.max_expanded_chunks:
                            break

            # Filter out chunks with similarity below threshold
            filtered_chunks = [
                chunk
                for chunk in expanded_chunks
                if chunk.get("similarity", 0.0)
                >= settings.expansion_similarity_threshold
            ]

            # Sort by similarity and apply final limit
            filtered_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

            # Apply absolute limit to prevent resource overload
            if len(filtered_chunks) > settings.max_expanded_chunks:
                filtered_chunks = filtered_chunks[: settings.max_expanded_chunks]
                logger.warning(
                    f"Truncated expansion results to {settings.max_expanded_chunks} chunks"
                )

            # Count expansion types
            original_count = sum(
                1 for chunk in filtered_chunks if not chunk.get("expansion_context")
            )
            entity_expansion_count = sum(
                1
                for chunk in filtered_chunks
                if chunk.get("expansion_context", {}).get("expansion_type")
                == "entity_relationship"
            )
            chunk_expansion_count = sum(
                1
                for chunk in filtered_chunks
                if chunk.get("expansion_context", {}).get("expansion_type")
                == "chunk_similarity"
            )

            logger.info(
                f"Graph expansion ({mode.value}): {len(initial_chunks)} initial → {len(expanded_chunks)} total → {len(filtered_chunks)} final "
                f"({original_count} original + {entity_expansion_count} entity-expanded + {chunk_expansion_count} similarity-expanded)"
            )
            return filtered_chunks

        except Exception as e:
            logger.error(f"Graph expansion retrieval failed: {e}")
            # Return empty list if we don't have initial_chunks
            return []


# Global instance of the document retriever
document_retriever = DocumentRetriever()
