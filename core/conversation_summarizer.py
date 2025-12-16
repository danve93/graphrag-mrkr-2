"""
Conversation Summarizer for generating titles and summaries.

Generates lightweight summaries from conversation history instead of
storing full transcripts, reducing token usage by 80%+.
"""

import logging
from typing import Any, Dict, List

from core.llm import llm_manager
from config.settings import settings

logger = logging.getLogger(__name__)


def generate_title(messages: List[Dict[str, Any]], max_length: int = 60) -> str:
    """Generate a concise title for a conversation.

    Args:
        messages: List of conversation messages
        max_length: Maximum title length

    Returns:
        Generated title (e.g., "Discussion about GraphRAG implementation")
    """
    if not messages:
        return "New Conversation"

    # Extract first few user messages for context
    user_messages = [
        msg["content"]
        for msg in messages
        if msg.get("role") == "user"
    ][:3]  # First 3 user messages

    if not user_messages:
        return "New Conversation"

    # Combine messages for title generation
    context = "\n".join(user_messages)

    prompt = f"""Generate a concise, descriptive title (max {max_length} characters) for this conversation.
The title should capture the main topic or purpose.

Conversation excerpt:
{context[:500]}

Respond with ONLY the title, no quotes or extra text."""

    try:
        title = llm_manager.generate_response(
            prompt=prompt,
            max_tokens=50,
            temperature=0.3,
        ).strip()

        # Clean up quotes if LLM added them
        title = title.strip('"\'')

        # Truncate if needed
        if len(title) > max_length:
            title = title[:max_length-3] + "..."

        return title if title else "New Conversation"

    except Exception as e:
        logger.error(f"Failed to generate title: {e}")
        # Fallback: use first user message
        first_msg = user_messages[0][:50]
        return f"{first_msg}..." if len(user_messages[0]) > 50 else first_msg


def generate_summary(
    messages: List[Dict[str, Any]],
    max_summary_length: int = 500,
    include_key_points: bool = True,
) -> str:
    """Generate a structured summary of a conversation.

    Args:
        messages: List of conversation messages
        max_summary_length: Maximum summary length in characters
        include_key_points: Whether to include bullet points

    Returns:
        Generated summary with key points
    """
    if not messages:
        return ""

    # Build conversation transcript
    transcript_parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # Truncate very long messages in transcript
        if len(content) > 300:
            content = content[:300] + "..."
        transcript_parts.append(f"{role.capitalize()}: {content}")

    transcript = "\n".join(transcript_parts)

    # Limit transcript length for LLM
    if len(transcript) > 4000:
        transcript = transcript[:4000] + "\n[...conversation continues...]"

    prompt = f"""Summarize this conversation in {max_summary_length} characters or less.

{'Include 2-4 key bullet points of important topics or decisions.' if include_key_points else 'Focus on the main topics and outcome.'}

Conversation:
{transcript}

Respond with a structured summary:"""

    try:
        summary = llm_manager.generate_response(
            prompt=prompt,
            max_tokens=200,
            temperature=0.3,
        ).strip()

        # Truncate if needed
        if len(summary) > max_summary_length:
            summary = summary[:max_summary_length-3] + "..."

        return summary

    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        # Fallback: extract user message snippets
        user_snippets = [
            msg["content"][:100]
            for msg in messages
            if msg.get("role") == "user"
        ][:3]

        if user_snippets:
            return "Discussed: " + "; ".join(user_snippets)
        return "Conversation summary unavailable"


def extract_user_preferences(
    messages: List[Dict[str, Any]],
    existing_facts: List[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Extract user preferences and facts from conversation.

    Analyzes the conversation to identify statements about user preferences,
    interests, or important information worth remembering.

    Args:
        messages: List of conversation messages
        existing_facts: Optional list of existing facts to avoid duplicates

    Returns:
        List of extracted facts with content and suggested importance
    """
    if not messages:
        return []

    # Build conversation context
    user_statements = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if len(content) > 20:  # Skip very short messages
                user_statements.append(content)

    if not user_statements:
        return []

    # Combine statements
    combined = "\n".join(user_statements[:10])  # Limit to 10 messages

    # List existing facts for deduplication
    existing_content = [
        fact.get("content", "")
        for fact in (existing_facts or [])
    ]
    existing_str = "\n".join(existing_content) if existing_content else "None"

    prompt = f"""Analyze this conversation and extract user preferences, interests, or important facts worth remembering.

Existing facts (avoid duplicates):
{existing_str}

User statements:
{combined}

Extract 0-5 new facts (only if genuinely important). For each fact:
- Content: A clear, concise statement
- Importance: Score from 0.0-1.0 (0.8-1.0 = very important, 0.5-0.7 = moderately important, 0.3-0.4 = minor)

Respond ONLY with JSON array:
[
  {{"content": "User prefers Python for data analysis", "importance": 0.7}},
  {{"content": "User is working on GraphRAG project", "importance": 0.8}}
]

If no important facts, respond with: []"""

    try:
        import json

        response = llm_manager.generate_response(
            prompt=prompt,
            max_tokens=300,
            temperature=0.2,
        ).strip()

        # Parse JSON response
        # Remove markdown code blocks if present
        if response.startswith("```"):
            # Extract JSON from code block
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        facts = json.loads(response)

        # Validate structure
        valid_facts = []
        for fact in facts:
            if isinstance(fact, dict) and "content" in fact and "importance" in fact:
                # Ensure importance is in valid range
                importance = float(fact["importance"])
                if 0.0 <= importance <= 1.0:
                    valid_facts.append({
                        "content": str(fact["content"]),
                        "importance": importance,
                    })

        logger.info(f"Extracted {len(valid_facts)} user preferences")
        return valid_facts

    except Exception as e:
        logger.error(f"Failed to extract preferences: {e}")
        return []


def generate_message_snippet(message: str, max_length: int = 100) -> str:
    """Generate a short snippet from a message.

    Args:
        message: Full message content
        max_length: Maximum snippet length

    Returns:
        Truncated message with ellipsis if needed
    """
    if len(message) <= max_length:
        return message

    # Try to cut at sentence boundary
    truncated = message[:max_length]
    last_period = truncated.rfind(".")
    last_question = truncated.rfind("?")
    last_exclamation = truncated.rfind("!")

    sentence_end = max(last_period, last_question, last_exclamation)

    if sentence_end > max_length // 2:  # At least halfway through
        return truncated[:sentence_end + 1]

    # Otherwise just truncate and add ellipsis
    return truncated.rstrip() + "..."
