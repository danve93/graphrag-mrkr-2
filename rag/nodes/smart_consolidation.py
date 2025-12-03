"""
Smart consolidation for multi-category retrieval results.

Ensures diverse category representation while respecting token budgets
and removing semantic duplicates.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np

from core.embeddings import embedding_manager
from config.settings import settings

logger = logging.getLogger(__name__)


class SmartConsolidator:
    """
    Smart consolidation with category-aware ranking and semantic deduplication.
    
    Features:
    - Ensures at least 1 chunk per category in top-k
    - Removes semantic duplicates (0.95 similarity threshold)
    - Enforces token budget (8K max context)
    - Preserves diversity while maximizing relevance
    """
    
    def __init__(
        self,
        max_tokens: int = 8000,
        semantic_threshold: float = 0.95,
        ensure_category_representation: bool = True,
        min_chunks_per_category: int = 1,
    ):
        self.max_tokens = max_tokens
        self.semantic_threshold = semantic_threshold
        self.ensure_category_representation = ensure_category_representation
        self.min_chunks_per_category = min_chunks_per_category
    
    async def consolidate(
        self,
        chunks: List[Dict[str, Any]],
        categories: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Consolidate chunks with category awareness and deduplication.
        
        Args:
            chunks: Retrieved chunks with scores and metadata
            categories: Target categories for representation (from routing)
            top_k: Maximum number of chunks to return
            
        Returns:
            Consolidated list of diverse, deduplicated chunks within token budget
        """
        if not chunks:
            return []
        
        try:
            # Step 1: Group by category
            category_groups = self._group_by_category(chunks)
            
            # Step 2: Ensure category representation
            if self.ensure_category_representation and categories:
                selected = self._ensure_representation(
                    category_groups, 
                    categories, 
                    top_k
                )
            else:
                # Simple top-k by score
                selected = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
            
            # Step 3: Semantic deduplication
            deduplicated = await self._deduplicate_semantic(selected)
            
            # Step 4: Enforce token budget
            final = self._enforce_token_budget(deduplicated)
            
            logger.info(
                f"Smart consolidation: {len(chunks)} → {len(selected)} (repr) → "
                f"{len(deduplicated)} (dedup) → {len(final)} (budget)"
            )
            
            return final
            
        except Exception as e:
            logger.error(f"Smart consolidation failed: {e}")
            # Fallback: return top-k by score
            return sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
    
    def _group_by_category(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by their category metadata."""
        groups = defaultdict(list)
        
        for chunk in chunks:
            # Try multiple possible category field names
            category = (
                chunk.get('category') or 
                chunk.get('document_category') or
                chunk.get('routing_category') or
                'uncategorized'
            )
            groups[category].append(chunk)
        
        # Sort each group by score
        for cat in groups:
            groups[cat].sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return dict(groups)
    
    def _ensure_representation(
        self,
        category_groups: Dict[str, List[Dict[str, Any]]],
        target_categories: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Ensure at least min_chunks_per_category from each target category.
        
        Strategy:
        1. Allocate min_chunks_per_category slots per target category
        2. Fill remaining slots with highest-scoring chunks overall
        """
        selected = []
        remaining_slots = top_k
        
        # Phase 1: Ensure minimum representation
        for category in target_categories:
            if category not in category_groups:
                logger.warning(f"Target category '{category}' has no chunks")
                continue
            
            cat_chunks = category_groups[category]
            allocation = min(self.min_chunks_per_category, len(cat_chunks), remaining_slots)
            
            selected.extend(cat_chunks[:allocation])
            remaining_slots -= allocation
            
            if remaining_slots <= 0:
                break
        
        # Phase 2: Fill remaining slots with best chunks
        if remaining_slots > 0:
            # Collect all chunks not yet selected
            selected_ids = {id(c) for c in selected}
            candidates = [
                c for group in category_groups.values() 
                for c in group 
                if id(c) not in selected_ids
            ]
            
            # Sort by score and take top remaining_slots
            candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
            selected.extend(candidates[:remaining_slots])
        
        # Sort final selection by score for consistency
        selected.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return selected
    
    async def _deduplicate_semantic(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove semantic duplicates using embedding similarity.
        
        Keeps the highest-scoring chunk from each duplicate cluster.
        """
        if len(chunks) <= 1:
            return chunks
        
        try:
            # Get embeddings for all chunks
            texts = [c.get('text', '') or c.get('content', '') for c in chunks]
            embeddings = []
            
            for text in texts:
                emb = await embedding_manager.get_embedding(text)
                if emb is not None:
                    embeddings.append(np.array(emb, dtype=float))
                else:
                    # Use zero vector if embedding fails
                    embeddings.append(np.zeros(1536, dtype=float))
            
            # Build similarity matrix
            n = len(embeddings)
            keep = [True] * n
            
            for i in range(n):
                if not keep[i]:
                    continue
                
                for j in range(i + 1, n):
                    if not keep[j]:
                        continue
                    
                    # Compute cosine similarity
                    emb_i = embeddings[i]
                    emb_j = embeddings[j]
                    
                    norm_i = np.linalg.norm(emb_i)
                    norm_j = np.linalg.norm(emb_j)
                    
                    if norm_i == 0 or norm_j == 0:
                        continue
                    
                    similarity = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                    
                    if similarity >= self.semantic_threshold:
                        # Mark lower-scoring chunk for removal
                        score_i = chunks[i].get('score', 0)
                        score_j = chunks[j].get('score', 0)
                        
                        if score_i >= score_j:
                            keep[j] = False
                        else:
                            keep[i] = False
                            break  # i is removed, skip remaining comparisons
            
            # Return only chunks marked to keep
            result = [chunks[i] for i in range(n) if keep[i]]
            
            duplicates_removed = n - len(result)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} semantic duplicates (threshold: {self.semantic_threshold})")
            
            return result
            
        except Exception as e:
            logger.error(f"Semantic deduplication failed: {e}")
            return chunks  # Return original on error
    
    def _enforce_token_budget(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enforce token budget by removing lowest-scoring chunks until under limit.
        
        Uses rough estimate: 1 token ≈ 4 characters.
        """
        if not chunks:
            return []
        
        # Estimate tokens for each chunk
        def estimate_tokens(chunk: Dict[str, Any]) -> int:
            text = chunk.get('text', '') or chunk.get('content', '')
            return len(text) // 4  # Rough estimate
        
        # Calculate cumulative tokens
        total_tokens = 0
        result = []
        
        for chunk in chunks:
            chunk_tokens = estimate_tokens(chunk)
            
            if total_tokens + chunk_tokens <= self.max_tokens:
                result.append(chunk)
                total_tokens += chunk_tokens
            else:
                # Budget exceeded, stop adding chunks
                logger.info(
                    f"Token budget reached: {total_tokens}/{self.max_tokens} tokens, "
                    f"included {len(result)}/{len(chunks)} chunks"
                )
                break
        
        return result


# Singleton instance
_consolidator: Optional[SmartConsolidator] = None


def get_consolidator() -> SmartConsolidator:
    """Get or create singleton SmartConsolidator instance."""
    global _consolidator
    
    if _consolidator is None:
        _consolidator = SmartConsolidator(
            max_tokens=settings.consolidation_max_context_tokens,
            semantic_threshold=settings.consolidation_semantic_threshold,
            ensure_category_representation=settings.consolidation_ensure_representation,
        )
    
    return _consolidator


async def consolidate_chunks(
    chunks: List[Dict[str, Any]],
    categories: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Convenience function for smart consolidation.
    
    Args:
        chunks: Retrieved chunks with scores
        categories: Target categories from routing
        top_k: Maximum chunks to return
        
    Returns:
        Consolidated, deduplicated chunks within token budget
    """
    consolidator = get_consolidator()
    return await consolidator.consolidate(chunks, categories, top_k)
