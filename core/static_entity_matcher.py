"""
Client-side vector matcher for static entities using precomputed embeddings.

Performs in-memory cosine similarity search for fast classification of queries
against static taxonomies (categories, entity types) without database queries.

Performance: <10ms latency vs 50-200ms for database-based vector search.
"""

import gzip
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.embeddings import embedding_manager

logger = logging.getLogger(__name__)


class StaticEntityMatcher:
    """
    In-memory vector matcher for static entities.

    Pre-loads embeddings from compressed JSON file and performs cosine similarity
    search without hitting the database.

    Attributes:
        entities: List of entity dicts with id, title, description, embedding
        embeddings_matrix: NumPy array of embeddings for fast computation
        dimension: Embedding vector dimension
        model: Embedding model used
        is_loaded: Whether embeddings are loaded
    """

    def __init__(self, embeddings_path: Optional[Path] = None):
        """
        Initialize matcher.

        Args:
            embeddings_path: Path to precomputed embeddings file (.json or .json.gz)
                            Defaults to config/static_embeddings.json.gz
        """
        self.embeddings_path = embeddings_path
        self.entities: List[Dict[str, Any]] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.dimension: Optional[int] = None
        self.model: Optional[str] = None
        self.is_loaded = False

        # Auto-load if path provided or default exists
        if self.embeddings_path:
            self.load()
        else:
            # Try default path
            default_path = Path(__file__).parent.parent / "config" / "static_embeddings.json.gz"
            if default_path.exists():
                self.embeddings_path = default_path
                self.load()

    def load(self, embeddings_path: Optional[Path] = None) -> bool:
        """
        Load precomputed embeddings from file.

        Args:
            embeddings_path: Optional path to embeddings file (overrides init path)

        Returns:
            True if successfully loaded

        Raises:
            FileNotFoundError: If embeddings file doesn't exist
            ValueError: If embeddings file has invalid format
        """
        if embeddings_path:
            self.embeddings_path = embeddings_path

        if not self.embeddings_path:
            raise ValueError("No embeddings path provided")

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")

        logger.info(f"Loading static embeddings from {self.embeddings_path}")

        try:
            # Load JSON (handle both compressed and uncompressed)
            if str(self.embeddings_path).endswith('.gz'):
                with gzip.open(self.embeddings_path, 'rb') as f:
                    data = json.loads(f.read().decode('utf-8'))
            else:
                with open(self.embeddings_path, 'r') as f:
                    data = json.load(f)

            # Validate format
            if 'categories' not in data:
                raise ValueError("Invalid embeddings format: missing 'categories' field")

            self.model = data.get('model', 'unknown')
            self.dimension = data.get('dimension')
            self.entities = data['categories']

            # Convert embeddings to NumPy matrix for fast computation
            embeddings_list = [entity['embedding'] for entity in self.entities]
            self.embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

            # Normalize embeddings for cosine similarity (dot product = cosine sim)
            norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
            self.embeddings_matrix = self.embeddings_matrix / norms

            self.is_loaded = True

            logger.info(
                f"✓ Loaded {len(self.entities)} entities "
                f"({self.dimension} dimensions, model: {self.model})"
            )

            # Log memory usage
            memory_mb = self.embeddings_matrix.nbytes / (1024 * 1024)
            logger.info(f"Embeddings memory usage: {memory_mb:.2f} MB")

            return True

        except Exception as e:
            logger.error(f"Failed to load static embeddings: {e}")
            self.is_loaded = False
            raise

    async def match_async(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Match query against static entities using cosine similarity.

        Args:
            query: Search query text
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of matches sorted by similarity (highest first):
            [
                {
                    "id": "category_id",
                    "title": "Category Title",
                    "description": "Description",
                    "keywords": ["keyword1", ...],
                    "similarity": 0.85
                }
            ]

        Raises:
            RuntimeError: If embeddings not loaded
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Static embeddings not loaded. Call load() first or initialize with embeddings_path."
            )

        if not query or not query.strip():
            logger.warning("Empty query provided to static matcher")
            return []

        # Generate query embedding
        query_embedding = await embedding_manager.aget_embedding(query)
        query_vector = np.array(query_embedding, dtype=np.float32)

        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        else:
            logger.warning("Query embedding has zero norm")
            return []

        # Compute cosine similarities (dot product since vectors are normalized)
        similarities = np.dot(self.embeddings_matrix, query_vector)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])

            # Skip if below threshold
            if similarity < min_similarity:
                continue

            entity = self.entities[idx]
            results.append({
                "id": entity['id'],
                "title": entity.get('title', entity['id']),
                "description": entity.get('description', ''),
                "keywords": entity.get('keywords', []),
                "similarity": round(similarity, 3)
            })

        # Format results for logging
        results_str = ", ".join([f"{r['id']}({r['similarity']:.2f})" for r in results])
        logger.debug(
            f"Static matcher: '{query[:50]}...' -> [{results_str}]"
        )

        return results

    def match(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for match_async.

        Args:
            query: Search query text
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of matches sorted by similarity
        """
        import asyncio
        try:
            # Try to use existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.match_async(query, top_k, min_similarity))
                    return future.result()
            else:
                return loop.run_until_complete(self.match_async(query, top_k, min_similarity))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.match_async(query, top_k, min_similarity))

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity dict or None if not found
        """
        if not self.is_loaded:
            raise RuntimeError("Static embeddings not loaded")

        for entity in self.entities:
            if entity['id'] == entity_id:
                return {
                    "id": entity['id'],
                    "title": entity.get('title', entity_id),
                    "description": entity.get('description', ''),
                    "keywords": entity.get('keywords', [])
                }

        return None

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """
        Get all loaded entities (without embeddings).

        Returns:
            List of entity dicts
        """
        if not self.is_loaded:
            raise RuntimeError("Static embeddings not loaded")

        return [
            {
                "id": entity['id'],
                "title": entity.get('title', entity['id']),
                "description": entity.get('description', ''),
                "keywords": entity.get('keywords', [])
            }
            for entity in self.entities
        ]

    def explain_match(self, query: str, entity_id: str) -> Dict[str, Any]:
        """
        Explain why a query matched an entity.

        Args:
            query: Search query
            entity_id: Entity to explain

        Returns:
            Dictionary with match explanation:
            {
                "entity_id": "install",
                "similarity": 0.85,
                "query": "how to setup",
                "entity_title": "Installation",
                "matched_keywords": ["install", "setup"]
            }
        """
        if not self.is_loaded:
            raise RuntimeError("Static embeddings not loaded")

        # Find entity
        entity = self.get_entity(entity_id)
        if not entity:
            raise ValueError(f"Entity not found: {entity_id}")

        # Get similarity score
        matches = self.match(query, top_k=len(self.entities))
        similarity = next(
            (m['similarity'] for m in matches if m['id'] == entity_id),
            0.0
        )

        # Find keyword matches
        query_lower = query.lower()
        matched_keywords = [
            kw for kw in entity.get('keywords', [])
            if kw.lower() in query_lower
        ]

        return {
            "entity_id": entity_id,
            "entity_title": entity.get('title', entity_id),
            "similarity": similarity,
            "query": query,
            "matched_keywords": matched_keywords,
            "description": entity.get('description', '')
        }


# Global instance (lazy-loaded)
_static_matcher: Optional[StaticEntityMatcher] = None


def get_static_matcher() -> StaticEntityMatcher:
    """
    Get or create global static entity matcher instance.

    Returns:
        Singleton StaticEntityMatcher instance

    Raises:
        RuntimeError: If embeddings can't be loaded
    """
    global _static_matcher

    if _static_matcher is None:
        _static_matcher = StaticEntityMatcher()

        if not _static_matcher.is_loaded:
            logger.warning(
                "Static embeddings not found. Run: "
                "python scripts/precompute_static_embeddings.py"
            )

    return _static_matcher
