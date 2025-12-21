"""
Query expansion module for handling abbreviations, synonyms, and vocabulary mismatches.

Improves recall on large document sets by expanding queries with related terms.
"""

import logging
import re
from typing import Dict, List, Set

from config.settings import settings
from core.llm import llm_manager

logger = logging.getLogger(__name__)


# Common technical abbreviations and their expansions
ABBREVIATION_MAP = {
    # Database terms
    "db": ["database"],
    "dbs": ["databases"],
    "rdbms": ["relational database management system", "relational database"],
    "nosql": ["no sql", "non-relational database"],
    "sql": ["structured query language"],
    "orm": ["object relational mapping", "object relational mapper"],

    # Programming
    "api": ["application programming interface"],
    "rest": ["representational state transfer", "restful"],
    "sdk": ["software development kit"],
    "cli": ["command line interface"],
    "gui": ["graphical user interface"],
    "ui": ["user interface"],
    "ux": ["user experience"],
    "cicd": ["continuous integration continuous deployment", "ci cd"],
    "ci": ["continuous integration"],
    "cd": ["continuous deployment", "continuous delivery"],

    # Infrastructure
    "k8s": ["kubernetes"],
    "vm": ["virtual machine"],
    "vms": ["virtual machines"],
    "cdn": ["content delivery network"],
    "dns": ["domain name system"],
    "ssl": ["secure sockets layer"],
    "tls": ["transport layer security"],
    "vpn": ["virtual private network"],
    "ssh": ["secure shell"],

    # Data & Analytics
    "ml": ["machine learning"],
    "ai": ["artificial intelligence"],
    "nlp": ["natural language processing"],
    "etl": ["extract transform load"],
    "bi": ["business intelligence"],
    "kpi": ["key performance indicator"],
    "roi": ["return on investment"],

    # Web Technologies
    "http": ["hypertext transfer protocol"],
    "https": ["hypertext transfer protocol secure"],
    "html": ["hypertext markup language"],
    "css": ["cascading style sheets"],
    "js": ["javascript"],
    "ts": ["typescript"],
    "json": ["javascript object notation"],
    "xml": ["extensible markup language"],
    "yaml": ["yaml ain't markup language"],

    # Architecture
    "soa": ["service oriented architecture"],
    "microservices": ["micro services"],
    "saas": ["software as a service"],
    "paas": ["platform as a service"],
    "iaas": ["infrastructure as a service"],
    "serverless": ["server less"],

    # Testing
    "qa": ["quality assurance"],
    "tdd": ["test driven development"],
    "bdd": ["behavior driven development"],
    "e2e": ["end to end"],

    # Version Control
    "git": ["version control"],
    "vcs": ["version control system"],
    "pr": ["pull request"],
    "mr": ["merge request"],

    # Operations
    "ops": ["operations"],
    "devops": ["development operations", "dev ops"],
    "sre": ["site reliability engineering"],
    "mfa": ["multi factor authentication", "multifactor authentication"],
    "sso": ["single sign on"],

    # Storage
    "s3": ["simple storage service", "object storage"],
    "blob": ["binary large object"],
    "cdn": ["content delivery network"],

    # Networking
    "ip": ["internet protocol"],
    "tcp": ["transmission control protocol"],
    "udp": ["user datagram protocol"],
    "nat": ["network address translation"],
    "dhcp": ["dynamic host configuration protocol"],

    # Authentication
    "oauth": ["open authorization"],
    "jwt": ["json web token"],
    "saml": ["security assertion markup language"],

    # Messaging
    "mq": ["message queue"],
    "amqp": ["advanced message queuing protocol"],
    "mqtt": ["message queuing telemetry transport"],

    # GraphRAG specific
    "rag": ["retrieval augmented generation"],
    "kg": ["knowledge graph"],
    "llm": ["large language model"],
    "embedding": ["vector embedding", "embeddings"],
}


def expand_query(
    query: str,
    query_analysis: Dict,
    max_expansions: int = 5,
    use_llm: bool = False,
) -> List[str]:
    """
    Generate query expansions to improve recall.

    Uses a combination of:
    1. Rule-based abbreviation expansion (fast, reliable)
    2. Optional LLM-based synonym expansion (slower, broader)

    Args:
        query: User query string
        query_analysis: Query analysis dict from analyze_query()
        max_expansions: Maximum number of expansion terms to return
        use_llm: Whether to use LLM for synonym expansion

    Returns:
        List of expansion terms (without the original query)
    """
    if not settings.enable_query_expansion:
        return []

    try:
        expanded_terms: Set[str] = set()

        # 1. Rule-based abbreviation expansion
        abbrev_expansions = _expand_abbreviations(query)
        expanded_terms.update(abbrev_expansions)

        # 2. LLM-based expansion (optional, for complex queries)
        if use_llm and len(expanded_terms) < max_expansions:
            llm_expansions = _expand_with_llm(query, query_analysis)
            expanded_terms.update(llm_expansions)

        # Limit to max_expansions
        result = list(expanded_terms)[:max_expansions]

        if result:
            logger.info(f"Query expansion: '{query}' -> {result}")

        return result

    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return []


def _expand_abbreviations(query: str) -> List[str]:
    """
    Expand known abbreviations in the query using rule-based mapping.

    Args:
        query: User query string

    Returns:
        List of expansion terms
    """
    expansions: Set[str] = set()
    query_lower = query.lower()

    # Extract words from query
    # Use word boundaries to match whole words only
    words = re.findall(r'\b\w+\b', query_lower)

    for word in words:
        if word in ABBREVIATION_MAP:
            # Add all expansions for this abbreviation
            expansions.update(ABBREVIATION_MAP[word])

    return list(expansions)


def _expand_with_llm(query: str, query_analysis: Dict) -> List[str]:
    """
    Use LLM to generate synonym expansions for complex queries.

    Only use this for queries where recall is critical and rule-based
    expansion isn't sufficient.

    Args:
        query: User query string
        query_analysis: Query analysis dict

    Returns:
        List of LLM-generated expansion terms
    """
    try:
        query_type = query_analysis.get("query_type", "factual")
        key_concepts = query_analysis.get("key_concepts", [])

        # Build expansion prompt
        expansion_prompt = f"""Generate 3-5 related search terms or synonyms for this query to improve search recall.

Query: {query}
Query Type: {query_type}
Key Concepts: {', '.join(key_concepts)}

Rules:
1. Generate terms that are semantically related but phrased differently
2. Include common synonyms or alternative phrasings
3. Focus on technical domain-specific vocabulary
4. Keep terms concise (1-3 words each)
5. Avoid terms that would introduce noise or false positives

Respond with a comma-separated list of expansion terms only (no explanations):"""

        system_message = "You are a query expansion assistant. Provide only the comma-separated list of terms."

        result = llm_manager.generate_response(
            prompt=expansion_prompt,
            system_message=system_message,
            temperature=0.3,  # Low temperature for consistency
            max_tokens=100,
            include_usage=True,
        )

        # Track token usage
        if isinstance(result, dict) and "usage" in result:
            try:
                from core.llm_usage_tracker import usage_tracker
                usage_tracker.record(
                    operation="rag.query_expansion",
                    provider=getattr(settings, "llm_provider", "openai"),
                    model=settings.openai_model,
                    input_tokens=result["usage"].get("input", 0),
                    output_tokens=result["usage"].get("output", 0),
                )
            except Exception as track_err:
                logger.debug(f"Token tracking failed: {track_err}")
            result = (result.get("content") or "").strip()
        else:
            result = (result or "").strip()

        # Parse the response
        # Split on commas and clean up
        terms = [term.strip().lower() for term in result.split(",")]

        # Filter out:
        # - Empty terms
        # - Terms that are just the original query
        # - Terms that are too short (likely noise)
        # - Terms that are too long (likely explanations)
        filtered_terms = [
            term for term in terms
            if term
            and term != query.lower()
            and 2 < len(term) < 50
            and not term.startswith(("note:", "example:", "e.g."))
        ]

        logger.info(f"LLM expansion: {query} -> {filtered_terms}")
        return filtered_terms[:5]  # Limit to 5 terms

    except Exception as e:
        logger.error(f"LLM-based expansion failed: {e}")
        return []


def should_expand_query(
    query_analysis: Dict,
    initial_results_count: int = 0,
) -> bool:
    """
    Determine if query expansion should be triggered.

    Expansion is beneficial when:
    1. Initial retrieval returns few results (sparse results)
    2. Query contains abbreviations
    3. Query is technical and may have vocabulary mismatches

    Args:
        query_analysis: Query analysis dict
        initial_results_count: Number of results from initial retrieval (0 if not done yet)

    Returns:
        True if expansion should be applied
    """
    if not settings.enable_query_expansion:
        return False

    # Check sparse results threshold
    if initial_results_count > 0:
        threshold = getattr(settings, "query_expansion_threshold", 3)
        if initial_results_count < threshold:
            logger.info(
                f"Query expansion triggered: sparse results ({initial_results_count} < {threshold})"
            )
            return True

    # Check if query is technical (likely to have abbreviations)
    if query_analysis.get("is_technical", False):
        logger.info("Query expansion triggered: technical query detected")
        return True

    # Check for complex analytical queries that might benefit from synonyms
    if (
        query_analysis.get("query_type") in ["analytical", "comparative"]
        and query_analysis.get("complexity") == "complex"
    ):
        logger.info("Query expansion triggered: complex analytical query")
        return True

    # Default: don't expand
    return False
