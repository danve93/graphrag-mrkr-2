"""
Query routing node - classifies queries to document categories with optional semantic cache.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from config.settings import settings
from core.llm import llm_manager
from core.static_entity_matcher import get_static_matcher
from rag.nodes.routing_cache import routing_cache

logger = logging.getLogger(__name__)


def route_query_to_categories(
    query: str,
    query_analysis: Dict[str, Any],
    confidence_threshold: float = 0.7,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Route user query to relevant document categories.

    Returns a dict: {
        "categories": [ids],
        "confidence": float,
        "reasoning": str,
        "should_filter": bool
    }
    """
    try:
        # Try semantic cache first
        cached: Optional[Dict[str, Any]] = None
        if settings.enable_routing_cache:
            try:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                cached = loop.run_until_complete(routing_cache.get(query)) if not loop.is_running() else None
            except Exception:
                cached = None

        if cached:
            categories = cached.get("categories", [])
            confidence = float(cached.get("confidence", 0.0))
            should_filter = confidence >= confidence_threshold and categories != ["general"]
            logger.info(
                f"Routing cache hit: {categories} (confidence {confidence:.2f}, filter={should_filter})"
            )
            return {
                "categories": categories,
                "confidence": confidence,
                "reasoning": cached.get("reasoning", "cached"),
                "should_filter": should_filter,
            }

        # Try static matcher first (fast path <10ms vs 200ms LLM call)
        if settings.enable_static_entity_matching:
            try:
                static_matcher = get_static_matcher()
                if static_matcher.is_loaded:
                    matches = static_matcher.match(
                        query,
                        top_k=3,
                        min_similarity=settings.static_matching_min_similarity
                    )

                    if matches and matches[0]['similarity'] >= confidence_threshold:
                        # Use top static match
                        categories = [matches[0]['id']]
                        confidence = matches[0]['similarity']
                        should_filter = confidence >= confidence_threshold and categories != ["general"]

                        reasoning = f"static_match: {matches[0]['title']} (sim={confidence:.2f})"
                        if len(matches) > 1:
                            reasoning += f", alternatives: {[m['id'] for m in matches[1:]]}"

                        logger.info(
                            f"Static routing: {categories} (confidence {confidence:.2f}, filter={should_filter})"
                        )

                        # Cache decision asynchronously
                        try:
                            import asyncio
                            payload = {
                                "categories": categories,
                                "confidence": confidence,
                                "reasoning": reasoning,
                            }
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)

                            if loop.is_running():
                                asyncio.create_task(routing_cache.set(query, payload))
                            else:
                                loop.run_until_complete(routing_cache.set(query, payload))
                        except Exception:
                            pass

                        return {
                            "categories": categories,
                            "confidence": confidence,
                            "reasoning": reasoning,
                            "should_filter": should_filter,
                        }
                    else:
                        # Static match exists but below threshold, log for debugging
                        if matches:
                            logger.debug(
                                f"Static match below threshold: {matches[0]['id']} "
                                f"(sim={matches[0]['similarity']:.2f} < {confidence_threshold:.2f}), "
                                f"falling back to LLM"
                            )
            except Exception as e:
                logger.warning(f"Static matcher failed: {e}, falling back to LLM routing")

        # Disabled routing → no filtering
        if not settings.enable_query_routing:
            return {
                "categories": ["general"],
                "confidence": 0.0,
                "reasoning": "routing_disabled",
                "should_filter": False,
            }

        config = _load_category_config()
        categories_desc = _format_categories_for_prompt(config)

        routing_prompt = f"""Analyze this user query and determine which documentation categories are most relevant.

Query: {query}

Query Type: {query_analysis.get('query_type', 'unknown')}
Key Concepts: {', '.join(query_analysis.get('key_concepts', []))}

Available Categories:
{categories_desc}

Instructions:
1. Select 1-3 most relevant categories
2. If the query spans multiple areas, list all relevant categories
3. If the query is ambiguous or could apply to many areas, select ["general"]
4. Provide your confidence level (0.0-1.0)

Respond with JSON:
{{
  "categories": ["category_id1", "category_id2"],
  "confidence": 0.85,
  "reasoning": "Brief explanation of why these categories"
}}"""

        system_message = "You are a query routing assistant. Respond only with valid JSON."

        result = llm_manager.generate_response(
            prompt=routing_prompt,
            system_message=system_message,
            temperature=0.0,
            max_tokens=200,
            include_usage=True,
        )

        # Track token usage
        if isinstance(result, dict) and "usage" in result:
            try:
                from core.llm_usage_tracker import usage_tracker
                usage_tracker.record(
                    operation="rag.query_routing",
                    provider=getattr(settings, "llm_provider", "openai"),
                    model=settings.openai_model,
                    input_tokens=result["usage"].get("input", 0),
                    output_tokens=result["usage"].get("output", 0),
                    conversation_id=session_id,
                )
            except Exception as track_err:
                logger.debug(f"Token tracking failed: {track_err}")
            text = (result.get("content") or "").strip()
        else:
            text = (result or "").strip()

        # Parse response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        routing_result = json.loads(text)

        confidence = float(routing_result.get("confidence", 0.0))
        categories = routing_result.get("categories", []) or ["general"]
        should_filter = confidence >= confidence_threshold and categories != ["general"]

        logger.info(
            f"Query routed to categories: {categories} (confidence: {confidence:.2f}, filtering: {should_filter})"
        )

        # Cache decision asynchronously (best effort)
        try:
            import asyncio

            payload = {
                "categories": list(categories) if categories else [],
                "confidence": float(confidence) if confidence else 0.0,
                "reasoning": str(routing_result.get("reasoning", "")),
            }
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(routing_cache.set(query, payload))
            else:
                loop.run_until_complete(routing_cache.set(query, payload))
        except Exception:
            pass

        return {
            "categories": categories,
            "confidence": confidence,
            "reasoning": routing_result.get("reasoning", ""),
            "should_filter": should_filter,
        }

    except Exception as e:
        logger.warning(f"Query routing failed: {e}")
        return {
            "categories": ["general"],
            "confidence": 0.0,
            "reasoning": "error",
            "should_filter": False,
        }


def _load_category_config() -> Dict[str, Any]:
    """Load category configuration from file."""
    import os

    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "config",
        "document_categories.json",
    )
    with open(config_path, "r") as f:
        return json.load(f)


def _format_categories_for_prompt(config: Dict[str, Any]) -> str:
    """Format category definitions for LLM prompt."""
    lines: List[str] = []
    cats: Dict[str, Any] = config.get("categories", {})
    for cat_id, cat in cats.items():
        name = cat.get("name", cat_id)
        desc = cat.get("description", "")
        keywords = ", ".join(cat.get("keywords", [])[:8])
        lines.append(f"- {cat_id} ({name}): {desc} | keywords: {keywords}")
    return "\n".join(lines)


def get_documents_by_categories(categories: List[str]) -> List[str]:
    """
    Get document IDs that belong to specified categories using BELONGS_TO.
    """
    from core.graph_db import graph_db

    try:
        if not categories:
            return []

        query = """
        MATCH (d:Document)-[:BELONGS_TO]->(c:Category)
        WHERE c.id IN $category_ids
        RETURN DISTINCT d.document_id AS document_id
        """
        params = {"category_ids": categories}
        rows = graph_db.run_query(query, params)
        document_ids = [row.get("document_id") for row in rows if row.get("document_id")]
        return document_ids
    except Exception as e:
        logger.error(f"Failed to get documents by category: {e}")
        return []
