"""
Follow-up question generation service.
"""

import logging
import re
from typing import Any, Dict, List, Set

from core.llm import llm_manager

logger = logging.getLogger(__name__)


class FollowUpService:
    """Service for generating follow-up questions."""

    _STOPWORDS: Set[str] = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
        "what", "when", "where", "which", "who", "why", "with", "your",
    }

    def _is_low_signal_chunk(self, content: str) -> bool:
        stripped = (content or "").strip()
        if len(stripped) < 80:
            return True
        if re.match(r"^##\\s*Page\\s*\\d+\\b", stripped, re.IGNORECASE):
            return True
        if re.match(r"^Page\\s*\\d+\\b", stripped, re.IGNORECASE):
            return True
        alpha = sum(1 for c in stripped if c.isalpha())
        return alpha / max(len(stripped), 1) < 0.2

    def _question_grounded_in_chunk(self, question: str, content: str) -> bool:
        text = (question or "").lower()
        keywords = [
            token
            for token in re.findall(r"[a-z0-9]{4,}", text)
            if token not in self._STOPWORDS
        ]
        if not keywords:
            return False
        content_lower = (content or "").lower()
        return any(keyword in content_lower for keyword in keywords)

    async def generate_follow_ups(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]],
        max_questions: int = 3,
        alternative_chunks: List[Dict[str, Any]] = None,
        session_id: str = None,
    ) -> List[str]:
        """
        Generate follow-up questions from alternative retrieval chunks.
        
        Uses the top-ranked chunks that didn't make it into the final context
        (from the reranking process) to suggest alternative questions the user
        might want to ask. This is much faster than LLM generation and ensures
        the questions are actually answerable.

        Args:
            query: User's original query
            response: Assistant's response
            sources: Sources used in the response
            chat_history: Previous conversation messages
            max_questions: Maximum number of questions to generate
            alternative_chunks: Alternative chunks from retrieval (not used in response)

        Returns:
            List of follow-up questions derived from alternative chunks
        """
        try:
            # If no alternative chunks provided, return empty list
            if not alternative_chunks:
                logger.info("No alternative chunks provided for follow-up generation")
                return []
            
            logger.info(f"Generating follow-ups from {len(alternative_chunks)} alternative chunks")
            
            # Generate follow-up questions from alternative chunks
            # These chunks were high-quality but didn't make the final cut
            follow_ups = []
            
            for chunk in alternative_chunks[:max_questions]:
                # Extract key information from the chunk
                content = chunk.get("content", "")
                document_name = chunk.get("document_name", chunk.get("filename", ""))

                if self._is_low_signal_chunk(content):
                    logger.info("Skipping low-signal chunk for follow-up generation")
                    continue
                
                # Extract entities if available
                entities = chunk.get("contained_entities", []) or chunk.get("relevant_entities", [])
                
                # Generate a question based on the chunk content
                question = await self._generate_question_from_chunk(
                    content, entities, document_name, query, session_id
                )
                
                if question and self._question_grounded_in_chunk(question, content):
                    follow_ups.append(question)
                    logger.debug(f"Generated follow-up from alternative chunk: {question}")
                elif question:
                    logger.info("Skipping ungrounded follow-up question")
            
            logger.info(f"Generated {len(follow_ups)} follow-up questions from alternative chunks")
            return follow_ups
        
        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            return []
    
    async def _generate_question_from_chunk(
        self, content: str, entities: List[str], document_name: str, original_query: str, session_id: str = None
    ) -> str:
        """
        Generate a follow-up question from a chunk using LLM.
        
        Args:
            content: Chunk content
            entities: Entities mentioned in the chunk
            document_name: Name of the document
            original_query: The user's original query
            
        Returns:
            A follow-up question or empty string if generation fails
        """
        try:
            # Create a concise prompt to generate a question from the chunk
            entity_context = ""
            if entities:
                entity_list = ", ".join(entities[:3])  # Limit to top 3 entities
                entity_context = f"Key entities: {entity_list}\n"
            
            prompt = f"""Based on this content, generate ONE specific follow-up question that a user might ask after reading their previous answer.

Original user question: {original_query}

Relevant content not yet discussed:
{entity_context}
{content[:300]}...

Generate a single, simple question (What is X? How does Y work? What are Z's features?) that:
1. Asks about something specific mentioned in the content
2. Is different from the original question
3. Can be answered using this content

Return ONLY the question, nothing else."""

            # Generate the question
            result = llm_manager.generate_response(
                prompt=prompt,
                temperature=0.7,
                max_tokens=50,
                include_usage=True,
            )

            # Track token usage
            if isinstance(result, dict) and "usage" in result:
                try:
                    from core.llm_usage_tracker import usage_tracker
                    from config.settings import settings
                    usage_tracker.record(
                        operation="rag.follow_up",
                        provider=getattr(settings, "llm_provider", "openai"),
                        model=settings.openai_model,
                        input_tokens=result["usage"].get("input", 0),
                        output_tokens=result["usage"].get("output", 0),
                        conversation_id=session_id,
                    )
                except Exception as track_err:
                    logger.debug(f"Token tracking failed: {track_err}")
                question = (result.get("content") or "").strip()
            else:
                question = (result or "").strip()
            
            # Remove numbering or prefixes
            for prefix in ["1.", "2.", "3.", "- ", "• ", "* ", "Question:", "Follow-up:"]:
                if question.startswith(prefix):
                    question = question[len(prefix):].strip()
            
            # Ensure it ends with ?
            if question and not question.endswith("?"):
                question += "?"
            
            # Basic validation: must be > 10 chars and contain question words
            if len(question) > 10 and any(word in question.lower() for word in ["what", "how", "why", "when", "where", "which"]):
                return question
            
            return ""
            
        except Exception as e:
            logger.warning(f"Failed to generate question from chunk: {e}")
            return ""

    async def _get_related_topics_from_graph(
        self, chunk_ids: List[str], max_topics: int = 10
    ) -> List[str]:
        """Query the knowledge graph for related entities from source chunks."""
        try:
            from core.graph_db import graph_db
            # Cypher query for entities connected to chunks and their 1-hop neighbors
            cypher_query = (
                "MATCH (c:Chunk) WHERE c.id IN $chunk_ids "
                "MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity) "
                "OPTIONAL MATCH (e)-[r:RELATED_TO|SIMILAR_TO]-(related:Entity) "
                "WHERE r.strength >= 0.5 OR r.similarity >= 0.7 "
                "WITH COLLECT(DISTINCT e.name) + COLLECT(DISTINCT related.name) as all_entities "
                "UNWIND all_entities as entity "
                "WITH DISTINCT entity "
                "WHERE entity IS NOT NULL "
                "RETURN entity "
                "LIMIT $max_topics"
            )

            topics = []
            with graph_db.session_scope() as session:
                result = session.run(
                    cypher_query, chunk_ids=chunk_ids, max_topics=max_topics
                )
                topics = [record["entity"] for record in result if record.get("entity")]
            
            if topics:
                logger.info(f"Found {len(topics)} related topics from graph")
            
            return topics
        
        except Exception as e:
            logger.warning(f"Failed to get related topics from graph: {e}")
            return []


# Global service instance
follow_up_service = FollowUpService()
