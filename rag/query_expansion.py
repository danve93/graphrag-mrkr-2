"""
Optional query expansion utility for sparse retrieval scenarios.
"""

import logging
from typing import List, Dict, Any, Optional

from core.llm import llm_manager
from config.settings import settings

logger = logging.getLogger(__name__)


def expand_query_terms(
    query: str,
    initial_results_count: int,
    min_threshold: int = 3,
    session_id: Optional[str] = None,
) -> List[str]:
    """
    Generate expanded terms (synonyms, related concepts) when initial retrieval is sparse.
    
    Args:
        query: Original user query
        initial_results_count: Number of results from initial retrieval
        min_threshold: Minimum results before triggering expansion
        
    Returns:
        List of expanded terms/phrases to add to retrieval
    """
    # Check if expansion is enabled and needed
    if not getattr(settings, "enable_query_expansion", False):
        return []
    
    expansion_threshold = getattr(settings, "query_expansion_threshold", min_threshold)
    
    if initial_results_count >= expansion_threshold:
        # Enough results, no expansion needed
        return []
    
    try:
        logger.info(f"Query expansion triggered: {initial_results_count} results < threshold {expansion_threshold}")
        
        expansion_prompt = f"""Given this search query, suggest 3-5 alternative terms, synonyms, or related concepts that would help find relevant information.

Query: {query}

Provide terms that are:
1. Synonyms or alternative phrasings
2. Related technical terms or concepts
3. Broader or narrower terms
4. Common variations or abbreviations

Return ONLY a JSON list of strings, no explanation:
["term1", "term2", "term3"]"""

        system_message = "You are a query expansion assistant. Return only valid JSON."
        
        result = llm_manager.generate_response(
            prompt=expansion_prompt,
            system_message=system_message,
            temperature=0.3,  # Lower temperature for consistent expansions
            max_tokens=150,
            include_usage=True,
        )

        # Track token usage
        if isinstance(result, dict) and "usage" in result:
            try:
                from core.llm_usage_tracker import usage_tracker
                from config.settings import settings
                usage_tracker.record(
                    operation="rag.query_expansion",
                    provider=getattr(settings, "llm_provider", "openai"),
                    model=settings.openai_model,
                    input_tokens=result["usage"].get("input", 0),
                    output_tokens=result["usage"].get("output", 0),
                    conversation_id=session_id,
                )
            except Exception as track_err:
                logger.debug(f"Token tracking failed: {track_err}")
            result = (result.get("content") or "").strip()
        else:
            result = (result or "").strip()
        
        # Parse JSON response
        import json
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        expanded_terms = json.loads(result)
        
        if isinstance(expanded_terms, list) and expanded_terms:
            logger.info(f"Query expansion generated {len(expanded_terms)} terms: {expanded_terms}")
            return expanded_terms[:5]  # Limit to 5 terms
        else:
            return []
            
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return []
