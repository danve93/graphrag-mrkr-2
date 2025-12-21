"""
Query analysis node for LangGraph RAG pipeline.
"""

import logging
from typing import Any, Dict, List, Optional

from core.llm import llm_manager
from rag.nodes.query_expansion import expand_query, should_expand_query

logger = logging.getLogger(__name__)


def analyze_query(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze user query to extract intent and key concepts.

    Args:
        query: User query string
        chat_history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]
        session_id: Optional session ID for token tracking

    Returns:
        Dictionary containing query analysis
    """
    try:
        # Check if this is a follow-up question
        is_follow_up = False
        needs_context = False
        context_query = query  # Always use original query for retrieval
        
        # DISABLED: Contextualization was making retrieval worse by rephrasing queries
        # The chat history is already passed to the LLM during generation, so it has context
        # Let retrieval work with the user's exact words for better matching
        # if chat_history and len(chat_history) >= 2:
        #     # Detect follow-up questions using LLM
        #     follow_up_detection = _detect_follow_up_question(query, chat_history)
        #     is_follow_up = follow_up_detection.get("is_follow_up", False)
        #     needs_context = follow_up_detection.get("needs_context", False)
        #
        #     if is_follow_up and needs_context:
        #         # Create a contextualized version of the query
        #         context_query = _create_contextualized_query(query, chat_history)
        #         logger.info(
        #             f"Follow-up question detected. Original: '{query}' -> Contextualized: '{context_query}'"
        #         )

        # Use LLM to analyze the query (using contextualized version if needed)
        analysis_result = llm_manager.analyze_query(context_query)

        # Extract key information (simplified version)
        analysis = {
            "original_query": query,
            "contextualized_query": context_query,
            "is_follow_up": is_follow_up,
            "needs_context": needs_context,
            "query_type": "factual",  # Default type
            "key_concepts": [],
            "intent": "information_seeking",
            "complexity": "simple",
            "analysis_text": analysis_result.get("analysis", ""),
            "requires_reasoning": False,
            "requires_multiple_sources": False,
            "suggested_strategy": "balanced",  # balanced, entity_focused, keyword_focused
            "confidence": 0.7,  # confidence in query type classification
            "expanded_terms": [],  # for optional query expansion
        }

        # Simple heuristics to enhance analysis (use contextualized query for better analysis)
        query_lower = context_query.lower()

        # Detect question types
        if any(
            word in query_lower
            for word in ["compare", "difference", "vs", "versus", "contrast"]
        ):
            analysis["query_type"] = "comparative"
            analysis["requires_multiple_sources"] = True
            analysis["requires_reasoning"] = True
        elif any(
            word in query_lower
            for word in [
                "why",
                "how",
                "explain",
                "reason",
                "analyze",
                "relationship",
                "connection",
            ]
        ):
            analysis["query_type"] = "analytical"
            analysis["requires_reasoning"] = True
        elif any(word in query_lower for word in ["what", "who", "when", "where"]):
            analysis["query_type"] = "factual"

        # Detect complexity
        if len(query.split()) > 10 or "and" in query_lower or "or" in query_lower:
            analysis["complexity"] = "complex"
            analysis["requires_multiple_sources"] = True

        # Detect temporal queries
        temporal_info = _detect_temporal_query(query_lower)
        analysis["is_temporal"] = temporal_info["is_temporal"]
        analysis["temporal_intent"] = temporal_info["intent"]
        analysis["temporal_window"] = temporal_info.get("window")
        analysis["time_decay_weight"] = temporal_info.get("decay_weight", 0.0)

        # Detect technical queries (for fuzzy matching)
        technical_info = _detect_technical_query(query_lower)
        analysis["is_technical"] = technical_info["is_technical"]
        analysis["fuzzy_distance"] = technical_info["fuzzy_distance"]
        analysis["technical_confidence"] = technical_info["confidence"]

        # Extract potential key concepts (simple keyword extraction)
        # Skip common words
        stop_words = {
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "that",
            "this",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
        }

        words = query_lower.replace("?", "").replace("!", "").replace(",", "").split()
        key_concepts = [
            word for word in words if len(word) > 2 and word not in stop_words
        ]
        analysis["key_concepts"] = key_concepts[:5]  # Limit to top 5 concepts

        # Determine if multi-hop reasoning would be beneficial
        multi_hop_beneficial = False

        # Multi-hop is beneficial for:
        # 1. Comparative queries (need to connect multiple entities)
        if analysis["query_type"] == "comparative":
            multi_hop_beneficial = True

        # 2. Analytical queries that need reasoning (relationships, explanations)
        elif analysis["query_type"] == "analytical" and analysis["requires_reasoning"]:
            multi_hop_beneficial = True

        # 3. Complex queries with multiple concepts
        elif analysis["complexity"] == "complex" and len(key_concepts) >= 3:
            multi_hop_beneficial = True

        # 4. Queries explicitly asking for relationships or connections
        elif any(
            word in query_lower
            for word in [
                "relationship",
                "connection",
                "related",
                "link",
                "connect",
                "between",
            ]
        ):
            multi_hop_beneficial = True

        # 5. Queries asking about trends, patterns, or implications
        elif any(
            word in query_lower
            for word in [
                "trend",
                "pattern",
                "impact",
                "effect",
                "influence",
                "implication",
            ]
        ):
            multi_hop_beneficial = True

        # Multi-hop is NOT beneficial for:
        # 1. Simple factual lookups (addresses, names, single facts)
        # 2. Direct "what is X" questions about specific entities
        # 3. Simple definition requests
        if (
            analysis["query_type"] == "factual"
            and analysis["complexity"] == "simple"
            and len(key_concepts) <= 2
            and not analysis["requires_multiple_sources"]
        ):
            multi_hop_beneficial = False

        analysis["multi_hop_recommended"] = multi_hop_beneficial

        # Determine suggested retrieval strategy
        strategy, confidence = _determine_retrieval_strategy(analysis, query_lower)
        analysis["suggested_strategy"] = strategy
        analysis["confidence"] = confidence

        # Query expansion (populate expanded_terms if beneficial)
        from config.settings import settings

        if should_expand_query(analysis):
            max_expansions = getattr(settings, "max_expansions", 5)
            use_llm_expansion = getattr(settings, "use_llm_expansion", False)
            expanded_terms = expand_query(
                query=context_query,
                query_analysis=analysis,
                max_expansions=max_expansions,
                use_llm=use_llm_expansion,
            )
            analysis["expanded_terms"] = expanded_terms
        else:
            analysis["expanded_terms"] = []

        logger.info(
            f"Query analysis completed: {analysis['query_type']}, {len(key_concepts)} concepts, "
            f"multi-hop recommended: {multi_hop_beneficial}, strategy: {strategy}, is_follow_up: {is_follow_up}, "
            f"expanded_terms: {len(analysis['expanded_terms'])}"
        )
        return analysis

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        return {
            "original_query": query,
            "contextualized_query": query,
            "is_follow_up": False,
            "needs_context": False,
            "query_type": "factual",
            "key_concepts": [],
            "intent": "information_seeking",
            "complexity": "simple",
            "analysis_text": "",
            "requires_reasoning": False,
            "requires_multiple_sources": False,
            "suggested_strategy": "balanced",
            "confidence": 0.5,
            "expanded_terms": [],
            "error": str(e),
        }


def _determine_retrieval_strategy(analysis: Dict[str, Any], query_lower: str) -> tuple[str, float]:
    """
    Determine the optimal retrieval strategy based on query characteristics.
    
    Returns:
        Tuple of (strategy, confidence)
        - strategy: 'entity_focused', 'keyword_focused', or 'balanced'
        - confidence: float between 0 and 1
    """
    query_type = analysis["query_type"]
    complexity = analysis["complexity"]
    key_concepts = analysis["key_concepts"]
    
    # Entity-focused strategy for relationship and analytical queries
    if query_type in ["comparative", "analytical"]:
        return ("entity_focused", 0.8)
    
    # Keyword-focused for exact-term lookups (version numbers, codes, specific names)
    if any(word in query_lower for word in ["version", "code", "id", "number", "exact"]):
        return ("keyword_focused", 0.75)
    
    # Keyword-focused for procedural how-to queries
    if query_type == "factual" and any(word in query_lower for word in ["how to", "steps", "procedure", "install", "configure"]):
        return ("keyword_focused", 0.7)
    
    # Entity-focused for relationship questions
    if any(word in query_lower for word in ["relationship", "connection", "related to", "between"]):
        return ("entity_focused", 0.85)
    
    # Balanced for simple factual queries
    if query_type == "factual" and complexity == "simple":
        return ("balanced", 0.65)
    
    # Default to balanced with moderate confidence
    return ("balanced", 0.6)


def _detect_follow_up_question(
    query: str, chat_history: List[Dict[str, str]], session_id: Optional[str] = None
) -> Dict[str, bool]:
    """
    Detect if the current query is a follow-up question that requires previous context.

    Args:
        query: Current user query
        chat_history: List of previous messages
        session_id: Optional session ID for token tracking

    Returns:
        Dictionary with is_follow_up and needs_context flags
    """
    try:
        # Quick heuristic checks first (for efficiency)
        query_lower = query.lower().strip()

        # Strong follow-up indicators
        follow_up_indicators = [
            "tell me more",
            "what about",
            "and",
            "also",
            "additionally",
            "his ",
            "her ",
            "their ",
            "its ",
            "this ",
            "that ",
            "these ",
            "those ",
            "he ",
            "she ",
            "they ",
            "it ",
            "more about",
            "explain",
            "clarify",
            "elaborate",
            "same",
            "similar",
            "different",
            "compared to",
        ]

        # Pronouns and references that likely need context
        context_references = [
            "he",
            "she",
            "they",
            "it",
            "this",
            "that",
            "these",
            "those",
            "him",
            "her",
            "them",
            "his",
            "her",
            "their",
            "its",
        ]

        # Check if query starts with these indicators (strong signal)
        starts_with_indicator = any(
            query_lower.startswith(indicator) for indicator in follow_up_indicators
        )

        # Check if query contains context references
        contains_reference = any(
            f" {ref} " in f" {query_lower} " or query_lower.startswith(f"{ref} ")
            for ref in context_references
        )

        # If quick checks suggest it's NOT a follow-up, return early
        if not (starts_with_indicator or contains_reference):
            # Check for questions without specific entities (likely need context)
            if len(query.split()) < 4 and "?" in query:
                # Short question, might be follow-up
                pass
            else:
                return {"is_follow_up": False, "needs_context": False}

        # Use LLM for more sophisticated detection
        # Get last 2-4 exchanges for context (limit to avoid token overflow)
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

        history_text = "\n".join(
            [f"{msg['role'].title()}: {msg['content']}" for msg in recent_history]
        )

        detection_prompt = f"""Analyze if the current question is a follow-up question that requires context from the previous conversation.

Previous conversation:
{history_text}

Current question: {query}

A follow-up question is one that:
1. Uses pronouns (he, she, it, they, this, that) referring to previous entities
2. Asks for more details about something just discussed
3. Makes implicit references to previous topics
4. Would be ambiguous or unclear without the previous context

Answer with JSON format:
{{"is_follow_up": true/false, "needs_context": true/false, "reason": "brief explanation"}}"""

        system_message = "You are a query analyzer. Respond only with valid JSON."

        result = llm_manager.generate_response(
            prompt=detection_prompt,
            system_message=system_message,
            temperature=0.3,  # Issue #23: Lower temperature for deterministic JSON parsing
            max_tokens=150,
            include_usage=True,
        )

        # Track token usage
        if isinstance(result, dict) and "usage" in result:
            try:
                from core.llm_usage_tracker import usage_tracker
                from config.settings import settings
                usage_tracker.record(
                    operation="rag.query_analysis",
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

        # Parse the response
        import json

        try:
            # Try to extract JSON from the response
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            analysis = json.loads(result)
            logger.info(f"Follow-up detection: {analysis}")
            return {
                "is_follow_up": analysis.get("is_follow_up", False),
                "needs_context": analysis.get("needs_context", False),
            }
        except json.JSONDecodeError:
            # Fallback to heuristic if JSON parsing fails
            logger.warning(f"Failed to parse follow-up detection result: {result}")
            return {
                "is_follow_up": starts_with_indicator or contains_reference,
                "needs_context": starts_with_indicator or contains_reference,
            }

    except Exception as e:
        logger.error(f"Follow-up detection failed: {e}")
        # Default to safe fallback
        return {"is_follow_up": False, "needs_context": False}


def _create_contextualized_query(query: str, chat_history: List[Dict[str, str]], session_id: Optional[str] = None) -> str:
    """
    Create a contextualized version of the query by incorporating relevant previous context.

    Args:
        query: Current user query
        chat_history: List of previous messages
        session_id: Optional session ID for token tracking

    Returns:
        Contextualized query string
    """
    try:
        # Get last 2-4 exchanges for context
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

        history_text = "\n".join(
            [
                f"{msg['role'].title()}: {msg['content'][:500]}"  # Limit length
                for msg in recent_history
            ]
        )

        contextualization_prompt = f"""Given the conversation history and the current follow-up question, rewrite the question to be self-contained and clear without the previous context.

Previous conversation:
{history_text}

Current follow-up question: {query}

Rewrite this question to include necessary context from the conversation. The rewritten question should:
1. Be clear and specific without needing to read the previous messages
2. Replace pronouns (he, she, it, they, this, that) with specific entities or concepts
3. Include relevant context that makes the question unambiguous
4. Maintain the original intent and scope of the question

Rewritten question:"""

        system_message = "You are a query rewriter. Provide only the rewritten question without explanation."

        contextualized = llm_manager.generate_response(
            prompt=contextualization_prompt,
            system_message=system_message,
            temperature=1.0,
            max_tokens=200,
            include_usage=True,
        )

        # Track token usage
        if isinstance(contextualized, dict) and "usage" in contextualized:
            try:
                from core.llm_usage_tracker import usage_tracker
                from config.settings import settings
                usage_tracker.record(
                    operation="rag.query_contextualization",
                    provider=getattr(settings, "llm_provider", "openai"),
                    model=settings.openai_model,
                    input_tokens=contextualized["usage"].get("input", 0),
                    output_tokens=contextualized["usage"].get("output", 0),
                    conversation_id=session_id,
                )
            except Exception as track_err:
                logger.debug(f"Token tracking failed: {track_err}")
            contextualized = (contextualized.get("content") or "").strip()
        else:
            contextualized = (contextualized or "").strip()

        # Clean up the response
        # Remove quotes if present
        if contextualized.startswith('"') and contextualized.endswith('"'):
            contextualized = contextualized[1:-1]
        if contextualized.startswith("'") and contextualized.endswith("'"):
            contextualized = contextualized[1:-1]

        logger.info(f"Contextualized query: {query} -> {contextualized}")
        return contextualized

    except Exception as e:
        logger.error(f"Query contextualization failed: {e}")
        # Return original query as fallback
        return query


def _detect_temporal_query(query_lower: str) -> Dict[str, Any]:
    """Detect if query has temporal intent and extract temporal parameters.

    Args:
        query_lower: Lowercase query string

    Returns:
        Dictionary with temporal information:
        - is_temporal: bool
        - intent: str (recent, specific_period, trending, etc.)
        - window: Optional[int] (time window in days)
        - decay_weight: float (0.0-1.0)
    """
    import re
    from datetime import datetime, timedelta

    temporal_info = {
        "is_temporal": False,
        "intent": "none",
        "window": None,
        "decay_weight": 0.0,
    }

    # Recent/latest indicators (high time decay weight)
    recent_patterns = [
        "recent", "latest", "new", "current", "up to date",
        "today", "yesterday", "past", "since"
    ]

    # Specific time periods
    time_period_patterns = {
        "last week": 7,
        "last month": 30,
        "last year": 365,
        "past week": 7,
        "past month": 30,
        "past year": 365,
        "this week": 7,
        "this month": 30,
        "this year": 365,
    }

    # Check for recent/latest queries
    for pattern in recent_patterns:
        if pattern in query_lower:
            temporal_info["is_temporal"] = True
            temporal_info["intent"] = "recent"
            temporal_info["decay_weight"] = 0.3
            break

    # Check for specific time periods (only if not already matched as recent)
    if temporal_info["intent"] == "none":
        for pattern, days in time_period_patterns.items():
            if pattern in query_lower:
                temporal_info["is_temporal"] = True
                temporal_info["intent"] = "specific_period"
                temporal_info["window"] = days
                temporal_info["decay_weight"] = 0.2
                break

    # Check for "in the last N days/weeks/months"
    days_match = re.search(r"(?:in the |last |past )(\d+) day", query_lower)
    weeks_match = re.search(r"(?:in the |last |past )(\d+) week", query_lower)
    months_match = re.search(r"(?:in the |last |past )(\d+) month", query_lower)

    if days_match:
        temporal_info["is_temporal"] = True
        temporal_info["intent"] = "specific_period"
        temporal_info["window"] = int(days_match.group(1))
        temporal_info["decay_weight"] = 0.2
    elif weeks_match:
        temporal_info["is_temporal"] = True
        temporal_info["intent"] = "specific_period"
        temporal_info["window"] = int(weeks_match.group(1)) * 7
        temporal_info["decay_weight"] = 0.2
    elif months_match:
        temporal_info["is_temporal"] = True
        temporal_info["intent"] = "specific_period"
        temporal_info["window"] = int(months_match.group(1)) * 30
        temporal_info["decay_weight"] = 0.2

    # Check for trending/evolution queries (moderate decay)
    # Only set to trending if no earlier temporal intent was detected
    if temporal_info["intent"] == "none" and any(word in query_lower for word in ["trend", "trending", "evolve", "evolution", "over time", "historically"]):
        temporal_info["is_temporal"] = True
        temporal_info["intent"] = "trending"
        temporal_info["decay_weight"] = 0.1  # Lower weight, we want broader time range

    # Check for "when" questions - these need temporal context but not necessarily recent
    if query_lower.startswith("when "):
        temporal_info["is_temporal"] = True
        if temporal_info["intent"] == "none":
            temporal_info["intent"] = "when_question"
        # Don't apply decay for "when" questions unless other temporal indicators present

    logger.debug(f"Temporal detection: is_temporal={temporal_info['is_temporal']}, "
                 f"intent={temporal_info['intent']}, window={temporal_info['window']}, "
                 f"decay_weight={temporal_info['decay_weight']}")

    return temporal_info


def _detect_technical_query(query_lower: str) -> Dict[str, Any]:
    """Detect if query contains technical terms that benefit from fuzzy matching.

    Technical queries include:
    - Database table/column names (snake_case, camelCase)
    - Technical IDs (PROJ-123, TICKET-456)
    - Configuration keys (MAX_CONNECTIONS, api_key)
    - Error codes (ERROR_404, ECONNREFUSED)
    - File paths/names (config.yml, /etc/nginx.conf)

    Args:
        query_lower: Lowercased query string

    Returns:
        Dict with technical query detection info
    """
    import re
    from config.settings import settings

    technical_info = {
        "is_technical": False,
        "fuzzy_distance": 0,
        "confidence": 0.0,
    }

    # Check if fuzzy matching is enabled
    if not settings.enable_fuzzy_matching:
        return technical_info

    # Pattern 1: Database-style identifiers (snake_case with underscores)
    if "snake_case" in settings.technical_term_patterns:
        snake_case_matches = re.findall(settings.technical_term_patterns["snake_case"], query_lower)
    else:
        snake_case_matches = []

    # Pattern 2: Technical IDs
    # Note: query_lower is converted to upper to match ID patterns commonly defined in uppercase
    if "tech_id" in settings.technical_term_patterns:
        tech_id_matches = re.findall(settings.technical_term_patterns["tech_id"], query_lower.upper())
    else:
        tech_id_matches = []

    # Pattern 3: Configuration keys
    if "config_key" in settings.technical_term_patterns:
        config_matches = re.findall(settings.technical_term_patterns["config_key"], query_lower)
    else:
        config_matches = []

    # Pattern 4: Error codes
    if "error_code" in settings.technical_term_patterns:
        error_matches = re.findall(settings.technical_term_patterns["error_code"], query_lower, re.IGNORECASE)
    else:
        error_matches = []
    
    # Pattern 5: File extensions
    # Pattern 5: File extensions and paths
    file_matches = []
    if "file_ext" in settings.technical_term_patterns:
        file_matches.extend(re.findall(settings.technical_term_patterns["file_ext"], query_lower))
    
    if "file_path" in settings.technical_term_patterns:
        file_matches.extend(re.findall(settings.technical_term_patterns["file_path"], query_lower))

    # Calculate confidence based on number of technical patterns found
    total_matches = (
        len(snake_case_matches) +
        len(tech_id_matches) +
        len(config_matches) +
        len(error_matches) +
        len(file_matches)
    )

    if total_matches > 0:
        # Calculate confidence based on number of matches
        if total_matches >= 3:
            confidence = 1.0
            fuzzy_distance = 2
        elif total_matches >= 2:
            confidence = 0.8
            fuzzy_distance = 2
        else:
            confidence = 0.5
            fuzzy_distance = 1

        # Only enable fuzzy matching if confidence meets threshold
        if confidence >= settings.fuzzy_confidence_threshold:
            technical_info["is_technical"] = True
            # Respect max_fuzzy_distance setting
            technical_info["fuzzy_distance"] = min(fuzzy_distance, settings.max_fuzzy_distance)
            technical_info["confidence"] = confidence

            logger.debug(
                f"Technical query detected: matches={total_matches}, "
                f"fuzzy_distance={technical_info['fuzzy_distance']}, "
                f"confidence={confidence:.2f}, "
                f"patterns=(snake_case:{len(snake_case_matches)}, "
                f"tech_id:{len(tech_id_matches)}, config:{len(config_matches)}, "
                f"error:{len(error_matches)}, file:{len(file_matches)})"
            )

    return technical_info
